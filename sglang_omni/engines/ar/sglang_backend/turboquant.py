"""TurboQuant: Near-optimal online KV cache quantization at 3.5 bits/channel.

MSE-optimal variant (no QJL).  At >=3 bits the MSE-only quantizer achieves
*lower* inner-product error than TurboQuant_prod (Figure 3a of the paper)
because all bits go to MSE precision rather than reserving 1 bit for QJL.
The tiny multiplicative bias at b>=3 is absorbed by softmax normalization
and subsequent LayerNorm, so attention quality is unaffected.

Algorithm per 128-dim head vector:
  1. Store L2 norm as FP16, normalize to unit sphere
  2. Split into two 64-dim halves
  3. Apply precomputed random orthogonal rotation (via QR on CPU, seed=42)
  4. Quantize with Lloyd-Max codebooks for N(0, 1/64):
     - High half (dims 0-63):  4-bit MSE (16 centroids) -> 32 bytes packed
     - Low half  (dims 64-127): 3-bit MSE (8 centroids)  -> 24 bytes packed
  5. Store: 32 + 24 + 2 (norm) = 58 bytes/head  vs 256 bytes BF16  (4.4x)

Per-token across 36 layers, 8 KV heads, K+V:  32.6 KB  vs  144 KB BF16.

Key improvements over previous implementation:
  - No QJL:  eliminates 2 dense 64x64 S^T matmuls per dequant (50% faster)
  - Per-head L2 norm:  paper requires unit-sphere input (Theorem 1)
  - torch.bucketize:  O(log k) binary search instead of O(k) argmin
  - torch.compile on pack/unpack:  fuses ~15 intermediate tensors per call
  - Page deduplication:  torch.unique on page_table avoids dequantizing
    shared prefix pages multiple times
  - No pre-allocated temp buffers:  dequantize_pages returns fresh views,
    eliminating the 2 GB temp_k/temp_v pre-allocation
"""

from __future__ import annotations

import gc
import logging
import math
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ===================================================================
# Lloyd-Max codebook centroids for standard normal N(0, 1)
# From Max (1960) / Lloyd (1982).  Scaled by sigma=1/sqrt(d) at init.
# ===================================================================

_STD_CENTROIDS_4BIT = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9424, -0.6568, -0.3882, -0.1284,
     0.1284,  0.3882,  0.6568,  0.9424,  1.2562,  1.6180,  2.0690,  2.7326,
]
_STD_CENTROIDS_3BIT = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
]


# ===================================================================
# Bit packing — torch.compile fuses the element-wise ops
# ===================================================================

@torch.compile(fullgraph=True, dynamic=True)
def _pack_4bit(idx: Tensor) -> Tensor:
    """[..., 64] int -> [..., 32] uint8.  Two nibbles per byte."""
    return (idx[..., 0::2] | (idx[..., 1::2] << 4)).to(torch.uint8)


@torch.compile(fullgraph=True, dynamic=True)
def _unpack_4bit(packed: Tensor) -> Tensor:
    """[..., 32] uint8 -> [..., 64] long."""
    low = (packed & 0xF).long()
    high = ((packed >> 4) & 0xF).long()
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], 64)


@torch.compile(fullgraph=True, dynamic=True)
def _pack_3bit(idx: Tensor) -> Tensor:
    """[..., 64] int -> [..., 24] uint8.  Eight 3-bit values into 3 bytes."""
    g = idx.view(*idx.shape[:-1], 8, 8).int()
    b0 = g[..., 0] | (g[..., 1] << 3) | ((g[..., 2] & 0x3) << 6)
    b1 = ((g[..., 2] >> 2) | (g[..., 3] << 1)
           | (g[..., 4] << 4) | ((g[..., 5] & 0x1) << 7))
    b2 = (g[..., 5] >> 1) | (g[..., 6] << 2) | (g[..., 7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).reshape(
        *idx.shape[:-1], 24
    ).to(torch.uint8)


@torch.compile(fullgraph=True, dynamic=True)
def _unpack_3bit(packed: Tensor) -> Tensor:
    """[..., 24] uint8 -> [..., 64] long."""
    g = packed.view(*packed.shape[:-1], 8, 3).int()
    b0, b1, b2 = g[..., 0], g[..., 1], g[..., 2]
    v0 = b0 & 0x7
    v1 = (b0 >> 3) & 0x7
    v2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
    v3 = (b1 >> 1) & 0x7
    v4 = (b1 >> 4) & 0x7
    v5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
    v6 = (b2 >> 2) & 0x7
    v7 = (b2 >> 5) & 0x7
    return torch.stack(
        [v0, v1, v2, v3, v4, v5, v6, v7], dim=-1
    ).reshape(*packed.shape[:-1], 64).long()


# ===================================================================
# Core TurboQuant quantizer
# ===================================================================

class TurboQuant:
    """MSE-optimal vector quantizer at 3.5 bits per dimension.

    Splits 128-dim vectors into two 64-dim halves:
      High (dims 0-63):  4-bit (16 centroids)
      Low  (dims 64-127): 3-bit (8 centroids)

    Each half is independently rotated with a fixed random orthogonal matrix,
    quantized via binary-search over Lloyd-Max boundaries, and packed.
    Per-vector L2 norms stored as FP16 for unit-sphere rescaling.
    """

    def __init__(self, device: torch.device, seed: int = 42):
        self.device = device
        self.d = 64

        # Deterministic orthogonal rotation matrices (CPU seed, then move)
        rng = torch.Generator(device="cpu").manual_seed(seed)
        Q1, _ = torch.linalg.qr(torch.randn(64, 64, generator=rng))
        Q2, _ = torch.linalg.qr(torch.randn(64, 64, generator=rng))
        self.rot_hi = Q1.to(device=device, dtype=torch.float32)
        self.rot_lo = Q2.to(device=device, dtype=torch.float32)

        # Codebooks scaled for marginal distribution N(0, 1/d), d=64
        sigma = 1.0 / math.sqrt(64)
        c4 = torch.tensor(_STD_CENTROIDS_4BIT, dtype=torch.float32, device=device) * sigma
        c3 = torch.tensor(_STD_CENTROIDS_3BIT, dtype=torch.float32, device=device) * sigma
        self.centroids_hi = c4   # [16]
        self.centroids_lo = c3   # [8]
        self.boundaries_hi = (c4[:-1] + c4[1:]) / 2   # [15]
        self.boundaries_lo = (c3[:-1] + c3[1:]) / 2   # [7]

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize to compressed representation.

        Args:
            x: [..., 128] tensor (any float dtype, typically bf16 from attention)
        Returns:
            packed_hi: [..., 32] uint8   (4-bit indices, dims 0-63)
            packed_lo: [..., 24] uint8   (3-bit indices, dims 64-127)
            norms:     [...]    float16  (per-vector L2 norms)
        """
        x = x.float()
        norms = x.norm(dim=-1)                             # [...]
        x_normed = x / norms.unsqueeze(-1).clamp(min=1e-8)

        # Rotate each half: coords become ~ N(0, 1/d) by Lemma 1
        y_hi = x_normed[..., :64] @ self.rot_hi.T         # [..., 64]
        y_lo = x_normed[..., 64:] @ self.rot_lo.T         # [..., 64]

        # Binary-search quantization: O(log k) per element via bucketize
        idx_hi = torch.bucketize(y_hi, self.boundaries_hi)  # [..., 64] in [0,15]
        idx_lo = torch.bucketize(y_lo, self.boundaries_lo)  # [..., 64] in [0,7]

        return _pack_4bit(idx_hi), _pack_3bit(idx_lo), norms.half()

    def dequantize(self, packed_hi: Tensor, packed_lo: Tensor, norms: Tensor) -> Tensor:
        """Dequantize compressed representation to BF16.

        Args:
            packed_hi: [..., 32] uint8
            packed_lo: [..., 24] uint8
            norms:     [...]    float16
        Returns:
            x_hat: [..., 128] bfloat16
        """
        idx_hi = _unpack_4bit(packed_hi)                   # [..., 64] long
        idx_lo = _unpack_3bit(packed_lo)                   # [..., 64] long

        y_hi = self.centroids_hi[idx_hi]                   # [..., 64] f32
        y_lo = self.centroids_lo[idx_lo]                   # [..., 64] f32

        x_hi = y_hi @ self.rot_hi                          # inverse rotation
        x_lo = y_lo @ self.rot_lo

        x = torch.cat([x_hi, x_lo], dim=-1)               # [..., 128]
        x = x * norms.float().unsqueeze(-1)
        return x.bfloat16()


# ===================================================================
# TurboQuantKVPool — compressed KV cache with SGLang-compatible API
# ===================================================================
# Duck-typed to match MHATokenToKVPool interface.  Does NOT inherit to
# avoid the parent __init__ allocating BF16 buffers + CUDA mem pools.

_ACTIVE_TQ_POOL: Optional["TurboQuantKVPool"] = None


class TurboQuantKVPool:
    """Compressed KV cache pool using TurboQuant 3.5-bit quantization.

    Storage: 58 bytes/head/token  vs  256 bytes BF16  (4.4x compression).

    get_kv_buffer() returns empty dummy tensors.  The actual dequantization
    is done by the monkey-patched flash_attn_with_kvcache, which calls
    dequantize_pages() with only the unique pages from the current batch's
    page_table.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        tq: TurboQuant,
        start_layer: int = 0,
    ):
        self.size = size
        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        self.dtype = torch.bfloat16
        self.store_dtype = torch.bfloat16
        self.start_layer = start_layer
        self.end_layer = start_layer + layer_num - 1
        self.tq = tq
        self._current_layer_id = 0

        # SGLang compat attributes
        self.row_dim = head_num * head_dim
        self.same_kv_dim = True
        self.layer_transfer_counter = None
        self.alt_stream = None
        self._kv_copy_config = None
        self.mem_usage = 0.0
        self.cpu_offloading_chunk_size = 8192

        total = size + page_size

        # --- Compressed storage: per-layer lists of tensors ---
        # 4-bit packed:  [total, heads, 32] uint8   (dims 0-63)
        # 3-bit packed:  [total, heads, 24] uint8   (dims 64-127)
        # L2 norms:      [total, heads]     float16
        self.k_packed_hi = [torch.zeros(total, head_num, 32, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.k_packed_lo = [torch.zeros(total, head_num, 24, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.k_norms     = [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]
        self.v_packed_hi = [torch.zeros(total, head_num, 32, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.v_packed_lo = [torch.zeros(total, head_num, 24, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.v_norms     = [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]

        # Empty dummy for get_kv_buffer: view(-1, ps, h, d) -> (0, ps, h, d)
        self._dummy = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.k_buffer = [self._dummy for _ in range(layer_num)]
        self.v_buffer = [self._dummy for _ in range(layer_num)]

        # Zero pointers/strides (move_kv_cache is overridden, no Triton copy)
        self.k_data_ptrs = torch.zeros(layer_num, dtype=torch.uint64, device=device)
        self.v_data_ptrs = torch.zeros(layer_num, dtype=torch.uint64, device=device)
        self.data_ptrs   = torch.zeros(2 * layer_num, dtype=torch.uint64, device=device)
        self.data_strides = torch.zeros(2 * layer_num, dtype=torch.int64, device=device)

        # Log
        k_bytes, v_bytes = self.get_kv_size_bytes()
        total_gb = (k_bytes + v_bytes) / (1024**3)
        bf16_gb = total * head_num * head_dim * 2 * 2 * layer_num / (1024**3)
        self.mem_usage = total_gb
        logger.info(
            "TurboQuantKVPool: %d tokens, %d layers, %.2f GB compressed "
            "(vs %.2f GB BF16, %.1fx savings)",
            size, layer_num, total_gb, bf16_gb,
            bf16_gb / max(total_gb, 1e-9),
        )

    # ---- set / get KV buffer (called by FA3 backend) ----

    def set_kv_buffer(
        self, layer, loc, cache_k, cache_v,
        k_scale=None, v_scale=None, layer_id_override=None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        li = layer_id - self.start_layer
        self._current_layer_id = layer_id

        k_hi, k_lo, k_n = self.tq.quantize(cache_k)
        v_hi, v_lo, v_n = self.tq.quantize(cache_v)

        self.k_packed_hi[li][loc] = k_hi
        self.k_packed_lo[li][loc] = k_lo
        self.k_norms[li][loc]     = k_n
        self.v_packed_hi[li][loc] = v_hi
        self.v_packed_lo[li][loc] = v_lo
        self.v_norms[li][loc]     = v_n

    def _get_key_buffer(self, layer_id: int):
        return self._dummy

    def get_key_buffer(self, layer_id: int):
        self._current_layer_id = layer_id
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._dummy

    def _get_value_buffer(self, layer_id: int):
        return self._dummy

    def get_value_buffer(self, layer_id: int):
        self._current_layer_id = layer_id
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._dummy

    def get_kv_buffer(self, layer_id: int):
        self._current_layer_id = layer_id
        return self._dummy, self._dummy

    # ---- page dequantization (called from monkey-patched flash_attn) ----

    def dequantize_pages(self, layer_id: int, page_indices: Tensor):
        """Dequantize KV for specific pages from compressed store.

        Args:
            layer_id:     transformer layer index
            page_indices: [N] int tensor of page numbers
        Returns:
            k: [N, page_size, heads, head_dim] bfloat16
            v: same shape
        """
        li = layer_id - self.start_layer
        N = page_indices.shape[0]

        # Page -> flat token indices: page p = tokens [p*ps .. (p+1)*ps)
        offsets = torch.arange(self.page_size, device=self.device)
        locs = (page_indices.unsqueeze(-1) * self.page_size + offsets).reshape(-1)

        k = self.tq.dequantize(
            self.k_packed_hi[li][locs],
            self.k_packed_lo[li][locs],
            self.k_norms[li][locs],
        )
        v = self.tq.dequantize(
            self.v_packed_hi[li][locs],
            self.v_packed_lo[li][locs],
            self.v_norms[li][locs],
        )

        return (
            k.view(N, self.page_size, self.head_num, self.head_dim),
            v.view(N, self.page_size, self.head_num, self.v_head_dim),
        )

    # ---- page move/copy (SGLang memory compaction / retraction) ----

    def move_kv_cache(self, tgt_loc: Tensor, src_loc: Tensor):
        if tgt_loc.numel() == 0:
            return
        for li in range(self.layer_num):
            self.k_packed_hi[li][tgt_loc] = self.k_packed_hi[li][src_loc]
            self.k_packed_lo[li][tgt_loc] = self.k_packed_lo[li][src_loc]
            self.k_norms[li][tgt_loc]     = self.k_norms[li][src_loc]
            self.v_packed_hi[li][tgt_loc] = self.v_packed_hi[li][src_loc]
            self.v_packed_lo[li][tgt_loc] = self.v_packed_lo[li][src_loc]
            self.v_norms[li][tgt_loc]     = self.v_norms[li][src_loc]

    # ---- utility / compat ----

    def get_kv_size_bytes(self):
        k = sum(t.nbytes for ts in (self.k_packed_hi, self.k_packed_lo, self.k_norms) for t in ts)
        v = sum(t.nbytes for ts in (self.v_packed_hi, self.v_packed_lo, self.v_norms) for t in ts)
        return k, v

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("TurboQuantKVPool: disagg not supported")

    def get_cpu_copy(self, indices):
        raise NotImplementedError("TurboQuantKVPool: CPU offload not supported")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("TurboQuantKVPool: CPU offload not supported")

    def register_layer_transfer_counter(self, counter):
        self.layer_transfer_counter = counter

    def _clear_buffers(self):
        del self.k_packed_hi, self.k_packed_lo, self.k_norms
        del self.v_packed_hi, self.v_packed_lo, self.v_norms

    def maybe_get_custom_mem_pool(self):
        return None


# ===================================================================
# Monkey-patched flash_attn_with_kvcache
# ===================================================================

_orig_flash_attn = None


def _patched_flash_attn(
    q, k_cache, v_cache, page_table=None, cache_seqlens=None, **kwargs
):
    """Drop-in replacement for flash_attn_with_kvcache.

    Pass-through when:
      - No TQ pool active
      - No page_table (Fast AR codebook loop uses page_table=None)
      - Non-empty k_cache (would mean non-TQ pool, shouldn't happen)

    TQ path:
      1. Flatten page_table, deduplicate via torch.unique
      2. Dequantize only unique pages -> [U, page_size, heads, dim] BF16
      3. Remap page_table indices to the compact buffer
      4. Call original flash_attn_with_kvcache with the temp buffer
    """
    pool = _ACTIVE_TQ_POOL

    if pool is None or page_table is None or k_cache.numel() > 0:
        return _orig_flash_attn(
            q, k_cache=k_cache, v_cache=v_cache,
            page_table=page_table, cache_seqlens=cache_seqlens,
            **kwargs,
        )

    # Deduplicate: shared voice prefix pages appear in every request's
    # page_table row but only need one dequantization
    all_pages = page_table.reshape(-1)
    unique_pages, inverse_map = all_pages.unique(return_inverse=True)

    k_deq, v_deq = pool.dequantize_pages(pool._current_layer_id, unique_pages)
    new_page_table = inverse_map.reshape(page_table.shape).to(torch.int32)

    return _orig_flash_attn(
        q, k_cache=k_deq, v_cache=v_deq,
        page_table=new_page_table, cache_seqlens=cache_seqlens,
        **kwargs,
    )


# ===================================================================
# Injection entry point
# ===================================================================

def inject_turboquant(model_runner):
    """Replace BF16 KV cache with TurboQuant 3.5-bit compressed pool.

    Called from model_worker.py after SGLModelRunner is instantiated.
    Frees the ~12 GB BF16 pool, creates ~2.7 GB compressed pool,
    and patches flash_attn_with_kvcache for on-demand page dequantization.
    """
    global _ACTIVE_TQ_POOL, _orig_flash_attn

    old_pool = model_runner.token_to_kv_pool
    params = dict(
        size=old_pool.size,
        page_size=old_pool.page_size,
        head_num=old_pool.head_num,
        head_dim=old_pool.head_dim,
        layer_num=old_pool.layer_num,
        device=old_pool.device,
        start_layer=getattr(old_pool, "start_layer", 0),
    )

    # Free old BF16 pool
    if hasattr(old_pool, "k_buffer") and old_pool.k_buffer is not None:
        for i in range(len(old_pool.k_buffer)):
            old_pool.k_buffer[i] = None
        old_pool.k_buffer = None
    if hasattr(old_pool, "v_buffer") and old_pool.v_buffer is not None:
        for i in range(len(old_pool.v_buffer)):
            old_pool.v_buffer[i] = None
        old_pool.v_buffer = None
    for attr in ("k_data_ptrs", "v_data_ptrs", "data_ptrs", "data_strides"):
        if hasattr(old_pool, attr):
            setattr(old_pool, attr, None)
    del old_pool
    model_runner.token_to_kv_pool = None
    gc.collect()
    torch.cuda.empty_cache()

    free_before = torch.cuda.mem_get_info()[0]
    logger.info("TurboQuant: freed BF16 KV pool. VRAM free: %.2f GB", free_before / 1e9)

    tq = TurboQuant(params["device"])
    tq_pool = TurboQuantKVPool(tq=tq, **params)
    model_runner.token_to_kv_pool = tq_pool
    _ACTIVE_TQ_POOL = tq_pool

    free_after = torch.cuda.mem_get_info()[0]
    logger.info(
        "TurboQuant: pool created. VRAM free: %.2f GB (net freed: %.2f GB)",
        free_after / 1e9, (free_after - free_before) / 1e9,
    )

    # Monkey-patch flash_attn at both the source module and the backend's
    # already-imported name binding
    import sgl_kernel.flash_attn as _sgl_fa

    _orig_flash_attn = _sgl_fa.flash_attn_with_kvcache
    _sgl_fa.flash_attn_with_kvcache = _patched_flash_attn

    try:
        import sglang.srt.layers.attention.flashattention_backend as _fa_backend
        _fa_backend.flash_attn_with_kvcache = _patched_flash_attn
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not patch FA backend module: %s", exc)

    logger.info(
        "TurboQuant: active. %d slots x %d layers, 3.5 bits/ch, %.1fx compression",
        params["size"], params["layer_num"], 256.0 / 58.0,
    )
