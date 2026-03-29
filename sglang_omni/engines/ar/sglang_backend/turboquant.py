"""TurboQuant KV cache quantization.

This module still contains the paper-faithful `TurboQuant_prod` path
(3-bit MSE + 1-bit QJL residual), but the active KV-cache integration uses
the deterministic MSE-only reconstruction.

Why: for this TTS model, the residual QJL term adds too much variance.
Empirically, pure MSE reconstruction sounds better end-to-end even though it
gives up the unbiasedness guarantee. The active path now uses grouped 2x64
4-bit MSE-only quantization plus a small recent-token BF16 cache.

Storage layout is unchanged:
  48 + 16 bytes are reused as a full 64-byte 4-bit payload
  + 2 + 2 bytes for the two half norms = 68 bytes/head.
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
# Exact Lloyd-Max codebooks for Beta(d=128) marginal distribution
# Precomputed via Lloyd-Max iteration on f(x) = C(1-x^2)^{62.5}
# NOT Gaussian-approximation scaled centroids.
# ===================================================================

_CENTROIDS_3BIT_D128 = [
    -0.1883914408, -0.1181338373, -0.0665812261, -0.0216027005,
     0.0216027005,  0.0665812261,  0.1181338373,  0.1883914408,
]
_BOUNDARIES_3BIT_D128 = [
    -0.1532626391, -0.0923575317, -0.0440919633,
     0.0,
     0.0440919633,  0.0923575317,  0.1532626391,
]
_CENTROIDS_4BIT_D128 = [
    -0.2378074305, -0.1809952734, -0.1419647482, -0.1104392440,
    -0.0829625272, -0.0578798312, -0.0342211064, -0.0113236335,
     0.0113286603,  0.0342211064,  0.0578798312,  0.0829625272,
     0.1104392440,  0.1419647482,  0.1809952734,  0.2378074305,
]
_BOUNDARIES_4BIT_D128 = [
    -0.2094013520, -0.1614800108, -0.1262019961, -0.0967008856,
    -0.0704211792, -0.0460504688, -0.0227723700,  0.0000025134,
     0.0227748834,  0.0460504688,  0.0704211792,  0.0967008856,
     0.1262019961,  0.1614800108,  0.2094013520,
]
_CENTROIDS_4BIT_D64 = [
    -0.3309249278, -0.2530586197, -0.1990016977, -0.1550604365,
    -0.1166026352, -0.0814005930, -0.0481465008, -0.0159365436,
     0.0159464914,  0.0481564461,  0.0814058480,  0.1166026352,
     0.1550604365,  0.1990016977,  0.2530586197,  0.3309249278,
]
_BOUNDARIES_4BIT_D64 = [
    -0.2919917737, -0.2260301587, -0.1770310671, -0.1358315359,
    -0.0990016141, -0.0647735469, -0.0320415222,  0.0000049739,
     0.0320514688,  0.0647811471,  0.0990042416,  0.1358315359,
     0.1770310671,  0.2260301587,  0.2919917737,
]


# ===================================================================
# Hadamard matrix (Sylvester construction)
# ===================================================================

def _build_hadamard_matrix(d: int) -> Tensor:
    """Build d x d Hadamard matrix with entries +/-1. H @ H = d * I."""
    assert d > 0 and (d & (d - 1)) == 0, "d must be a power of 2"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H


# ===================================================================
# Bit packing
# ===================================================================

def _pack_3bit(idx: Tensor) -> Tensor:
    """[..., N*8] int -> [..., N*3] uint8.  8 x 3-bit values into 3 bytes.
    Last dim must be divisible by 8."""
    d = idx.shape[-1]
    g = idx.view(*idx.shape[:-1], d // 8, 8).int()
    b0 = g[..., 0] | (g[..., 1] << 3) | ((g[..., 2] & 0x3) << 6)
    b1 = ((g[..., 2] >> 2) | (g[..., 3] << 1)
           | (g[..., 4] << 4) | ((g[..., 5] & 0x1) << 7))
    b2 = (g[..., 5] >> 1) | (g[..., 6] << 2) | (g[..., 7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).reshape(
        *idx.shape[:-1], (d // 8) * 3).to(torch.uint8)


def _unpack_3bit(packed: Tensor, out_dim: int) -> Tensor:
    """[..., N*3] uint8 -> [..., N*8] long."""
    n_groups = out_dim // 8
    g = packed.view(*packed.shape[:-1], n_groups, 3).int()
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
    ).reshape(*packed.shape[:-1], out_dim).long()


def _pack_4bit(idx: Tensor) -> Tensor:
    """[..., N*2] int -> [..., N] uint8. Two 4-bit values per byte."""
    return (idx[..., 0::2] | (idx[..., 1::2] << 4)).to(torch.uint8)


def _unpack_4bit(packed: Tensor, out_dim: int) -> Tensor:
    """[..., N] uint8 -> [..., N*2] long."""
    low = (packed & 0xF).long()
    high = ((packed >> 4) & 0xF).long()
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], out_dim)


def _pack_1bit(bits: Tensor) -> Tensor:
    """[..., 128] bool/int -> [..., 16] uint8.  8 bits per byte."""
    g = bits.view(*bits.shape[:-1], 16, 8).int()
    packed = g[..., 0]
    for i in range(1, 8):
        packed = packed | (g[..., i] << i)
    return packed.to(torch.uint8)


def _unpack_1bit(packed: Tensor) -> Tensor:
    """[..., 16] uint8 -> [..., 128] float.  Values are -1.0 or +1.0."""
    p = packed.int()
    bits = torch.stack([(p >> i) & 1 for i in range(8)], dim=-1)
    return bits.reshape(*packed.shape[:-1], 128).float() * 2.0 - 1.0


# ===================================================================
# Core TurboQuant_prod quantizer
# ===================================================================

class TurboQuant:
    """TurboQuant_prod: unbiased inner-product-optimal quantizer at 4 bits/dim.

    Full 128-d vector processing:
      1. Normalize to unit sphere, store L2 norm (FP16)
      2. Randomized Hadamard rotation (per-layer D signs + shared H matrix)
      3. Uniform 3-bit MSE quantize on all 128 channels
      4. Compute residual in original space
      5. QJL 1-bit: sign(S @ r), store ||r|| as FP16
      => Total: 3 + 1 = 4 bits/channel

    Per-layer rotation diversity decorrelates quantization errors across layers.
    Dense Gaussian S matrix preserves exact QJL unbiasedness (Lemma 4).
    """

    def __init__(self, device: torch.device, num_layers: int = 36, seed: int = 42):
        self.device = device
        self.d = 128
        self.num_layers = num_layers

        # Shared Hadamard matrix (normalized: H_norm @ H_norm = I)
        H = _build_hadamard_matrix(128)
        self.H_norm = (H / math.sqrt(128)).to(device=device, dtype=torch.float32)
        H64 = _build_hadamard_matrix(64)
        self.H64_norm = (H64 / math.sqrt(64)).to(device=device, dtype=torch.float32)

        # Per-layer random sign vectors for rotation diversity
        self.D_signs = []
        rng = torch.Generator(device="cpu")
        for layer_id in range(num_layers):
            rng.manual_seed(seed + layer_id)
            signs = torch.where(
                torch.rand(128, generator=rng) > 0.5,
                torch.ones(128), -torch.ones(128),
            )
            self.D_signs.append(signs.to(device=device, dtype=torch.float32))

        self.D64_signs_hi = []
        self.D64_signs_lo = []
        for layer_id in range(num_layers):
            rng.manual_seed(seed + 10_000 + 2 * layer_id)
            signs_hi = torch.where(
                torch.rand(64, generator=rng) > 0.5,
                torch.ones(64), -torch.ones(64),
            )
            rng.manual_seed(seed + 10_000 + 2 * layer_id + 1)
            signs_lo = torch.where(
                torch.rand(64, generator=rng) > 0.5,
                torch.ones(64), -torch.ones(64),
            )
            self.D64_signs_hi.append(signs_hi.to(device=device, dtype=torch.float32))
            self.D64_signs_lo.append(signs_lo.to(device=device, dtype=torch.float32))

        # Shared QJL projection matrix: S with i.i.d. N(0,1) entries
        rng.manual_seed(seed + num_layers + 1000)
        self.S = torch.randn(128, 128, generator=rng).to(
            device=device, dtype=torch.float32
        )

        # Exact Beta(d=128) Lloyd-Max codebook — uniform 3-bit for all channels
        self.centroids_3bit = torch.tensor(
            _CENTROIDS_3BIT_D128, dtype=torch.float32, device=device)
        self.boundaries_3bit = torch.tensor(
            _BOUNDARIES_3BIT_D128, dtype=torch.float32, device=device)
        self.centroids_4bit = torch.tensor(
            _CENTROIDS_4BIT_D128, dtype=torch.float32, device=device)
        self.boundaries_4bit = torch.tensor(
            _BOUNDARIES_4BIT_D128, dtype=torch.float32, device=device)
        self.centroids_4bit_64 = torch.tensor(
            _CENTROIDS_4BIT_D64, dtype=torch.float32, device=device)
        self.boundaries_4bit_64 = torch.tensor(
            _BOUNDARIES_4BIT_D64, dtype=torch.float32, device=device)

        self._qjl_scale = math.sqrt(math.pi / 2.0) / 128.0

    def _rotate_forward(self, x: Tensor, layer_id: int) -> Tensor:
        """y = (x * D_l) @ H_norm  (randomized Hadamard rotation)."""
        flat = x.reshape(-1, 128)
        y = (flat * self.D_signs[layer_id]) @ self.H_norm
        return y.view_as(x)

    def _rotate_inverse(self, y: Tensor, layer_id: int) -> Tensor:
        """x = (y @ H_norm) * D_l  (inverse rotation)."""
        flat = y.reshape(-1, 128)
        x = (flat @ self.H_norm) * self.D_signs[layer_id]
        return x.view_as(y)

    def _rotate_half_forward(self, x: Tensor, layer_id: int, half: int) -> Tensor:
        flat = x.reshape(-1, 64)
        signs = self.D64_signs_hi[layer_id] if half == 0 else self.D64_signs_lo[layer_id]
        y = (flat * signs) @ self.H64_norm
        return y.view_as(x)

    def _rotate_half_inverse(self, y: Tensor, layer_id: int, half: int) -> Tensor:
        flat = y.reshape(-1, 64)
        signs = self.D64_signs_hi[layer_id] if half == 0 else self.D64_signs_lo[layer_id]
        x = (flat @ self.H64_norm) * signs
        return x.view_as(y)

    def quantize(
        self,
        x: Tensor,
        layer_id: int,
        use_qjl: bool = True,
        mse_bits: int = 3,
        grouped_64: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantize to TurboQuant_prod compressed representation.

        Args:
            x: [..., 128] tensor
            layer_id: transformer layer index (for per-layer rotation)
        Returns:
            packed_mse: [..., 48] uint8
            packed_qjl: [..., 16] uint8
            norms:      [...]    fp16
            r_norms:    [...]    fp16
        """
        x = x.float()
        if use_qjl and mse_bits != 3:
            raise ValueError("QJL residual path only supports 3-bit MSE base quantization")
        if grouped_64 and (use_qjl or mse_bits != 4):
            raise ValueError("Grouped 2x64 mode only supports 4-bit MSE-only quantization")

        if grouped_64:
            x_hi = x[..., :64]
            x_lo = x[..., 64:]
            norms_hi = x_hi.norm(dim=-1)
            norms_lo = x_lo.norm(dim=-1)
            x_hi_unit = x_hi / norms_hi.unsqueeze(-1).clamp(min=1e-8)
            x_lo_unit = x_lo / norms_lo.unsqueeze(-1).clamp(min=1e-8)

            y_hi = self._rotate_half_forward(x_hi_unit, layer_id, 0)
            y_lo = self._rotate_half_forward(x_lo_unit, layer_id, 1)
            idx_hi = torch.bucketize(y_hi.contiguous(), self.boundaries_4bit_64)
            idx_lo = torch.bucketize(y_lo.contiguous(), self.boundaries_4bit_64)

            packed_full = torch.cat([_pack_4bit(idx_hi), _pack_4bit(idx_lo)], dim=-1)
            return (
                packed_full[..., :48],
                packed_full[..., 48:],
                norms_hi.half(),
                norms_lo.half(),
            )

        # 1. Full-vector normalization to unit sphere
        norms = x.norm(dim=-1)
        x_unit = x / norms.unsqueeze(-1).clamp(min=1e-8)

        # 2. Full 128-d randomized Hadamard rotation
        y = self._rotate_forward(x_unit, layer_id)

        # 3. MSE quantization on all 128 channels
        if mse_bits == 3:
            idx = torch.bucketize(y.contiguous(), self.boundaries_3bit)
            y_hat = self.centroids_3bit[idx]
            packed_main = _pack_3bit(idx)
            packed_aux = None
        elif mse_bits == 4:
            idx = torch.bucketize(y.contiguous(), self.boundaries_4bit)
            y_hat = self.centroids_4bit[idx]
            packed_full = _pack_4bit(idx)
            packed_main = packed_full[..., :48]
            packed_aux = packed_full[..., 48:]
        else:
            raise ValueError(f"Unsupported mse_bits={mse_bits}")

        # 4. MSE reconstruction (needed for QJL residual)
        x_mse = self._rotate_inverse(y_hat, layer_id)
        x_mse = x_mse * norms.unsqueeze(-1)

        if use_qjl:
            r = x - x_mse
            r_norms = r.norm(dim=-1)
            qjl_proj = r.reshape(-1, 128) @ self.S.T
            qjl_signs = (qjl_proj > 0).view(*r.shape[:-1], 128)
            packed_qjl = _pack_1bit(qjl_signs)
            packed_rnorm = r_norms.half()
        else:
            if mse_bits == 4:
                packed_qjl = packed_aux
            else:
                packed_qjl = torch.zeros(
                    *x.shape[:-1], 16, dtype=torch.uint8, device=x.device
                )
            packed_rnorm = torch.zeros(
                *x.shape[:-1], dtype=torch.float16, device=x.device
            )

        # 6. Pack
        return (
            packed_main,           # [..., 48] uint8
            packed_qjl,            # [..., 16] uint8
            norms.half(),
            packed_rnorm,
        )

    def dequantize(
        self,
        packed_mse: Tensor, packed_qjl: Tensor,
        norms: Tensor, r_norms: Tensor,
        layer_id: int,
        use_qjl: bool = True,
        mse_bits: int = 3,
        grouped_64: bool = False,
    ) -> Tensor:
        """Dequantize to BF16 (Algorithm 2, DeQuant_prod).

        Args:
            packed_mse: [..., 48] uint8
            packed_qjl: [..., 16] uint8
            norms:      [...]    fp16
            r_norms:    [...]    fp16
            layer_id:   int
        Returns:
            x_hat: [..., 128] bfloat16
        """
        if grouped_64:
            packed_full = torch.cat([packed_mse, packed_qjl], dim=-1)
            packed_hi = packed_full[..., :32]
            packed_lo = packed_full[..., 32:]
            idx_hi = _unpack_4bit(packed_hi, 64)
            idx_lo = _unpack_4bit(packed_lo, 64)
            y_hi_hat = self.centroids_4bit_64[idx_hi]
            y_lo_hat = self.centroids_4bit_64[idx_lo]
            x_hi = self._rotate_half_inverse(y_hi_hat, layer_id, 0)
            x_lo = self._rotate_half_inverse(y_lo_hat, layer_id, 1)
            x_hi = x_hi * norms.float().unsqueeze(-1)
            x_lo = x_lo * r_norms.float().unsqueeze(-1)
            return torch.cat([x_hi, x_lo], dim=-1).bfloat16()

        # 1. MSE reconstruction
        if mse_bits == 3:
            idx = _unpack_3bit(packed_mse, 128)
            y_hat = self.centroids_3bit[idx]
        elif mse_bits == 4:
            packed_full = torch.cat([packed_mse, packed_qjl], dim=-1)
            idx = _unpack_4bit(packed_full, 128)
            y_hat = self.centroids_4bit[idx]
        else:
            raise ValueError(f"Unsupported mse_bits={mse_bits}")
        x_mse = self._rotate_inverse(y_hat, layer_id)
        x_mse = x_mse * norms.float().unsqueeze(-1)

        if not use_qjl:
            return x_mse.bfloat16()

        # 2. QJL correction
        qjl_pm = _unpack_1bit(packed_qjl)                    # [..., 128] ±1
        x_qjl = qjl_pm.reshape(-1, 128) @ self.S             # S^T @ signs
        scale = (r_norms.float() * self._qjl_scale).unsqueeze(-1)
        x_qjl = x_qjl.view_as(x_mse) * scale

        return (x_mse + x_qjl).bfloat16()


# ===================================================================
# TurboQuantKVPool
# ===================================================================

_ACTIVE_TQ_POOL: Optional["TurboQuantKVPool"] = None


class TurboQuantKVPool:
    """Compressed KV cache: 68 bytes/head vs 256 bytes BF16 (3.76x).

    Buffers per layer per K/V:
      packed_mse: [total, heads, 48] uint8   (3-bit MSE, 128 channels)
      packed_qjl: [total, heads, 16] uint8   (1-bit QJL, 128 channels)
      norms:      [total, heads]     float16 (input L2 norm)
      r_norms:    [total, heads]     float16 (residual L2 norm)
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
        use_qjl: bool = False,
        mse_bits: int = 4,
        grouped_64: bool = False,
        recent_raw_capacity: int = 0,
        recent_raw_max_write: int = 64,
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
        self.use_qjl = use_qjl
        self.mse_bits = mse_bits
        self.grouped_64 = grouped_64
        self.recent_raw_capacity = recent_raw_capacity
        self.recent_raw_max_write = recent_raw_max_write
        self._current_layer_id = 0

        # SGLang compat
        self.row_dim = head_num * head_dim
        self.same_kv_dim = True
        self.layer_transfer_counter = None
        self.alt_stream = None
        self._kv_copy_config = None
        self.mem_usage = 0.0
        self.cpu_offloading_chunk_size = 8192

        total = size + page_size

        # 4 buffers per K, 4 per V (down from 5+5 with the 2-bit channel)
        self.k_mse  = [torch.zeros(total, head_num, 48, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.k_qjl  = [torch.zeros(total, head_num, 16, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.k_norm = [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]
        self.k_rnorm= [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]

        self.v_mse  = [torch.zeros(total, head_num, 48, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.v_qjl  = [torch.zeros(total, head_num, 16, dtype=torch.uint8, device=device) for _ in range(layer_num)]
        self.v_norm = [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]
        self.v_rnorm= [torch.zeros(total, head_num, dtype=torch.float16, device=device) for _ in range(layer_num)]

        if recent_raw_capacity > 0:
            self.recent_k = [
                torch.zeros(recent_raw_capacity, head_num, head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(layer_num)
            ]
            self.recent_v = [
                torch.zeros(recent_raw_capacity, head_num, self.v_head_dim, dtype=torch.bfloat16, device=device)
                for _ in range(layer_num)
            ]
            self.recent_slot_of_loc = torch.full(
                (total,), -1, dtype=torch.int32, device=device
            )
            self.recent_slot_loc = torch.full(
                (recent_raw_capacity,), -1, dtype=torch.int64, device=device
            )
            self._recent_next_slot = 0
        else:
            self.recent_k = None
            self.recent_v = None
            self.recent_slot_of_loc = None
            self.recent_slot_loc = None
            self._recent_next_slot = 0

        # Dummy empty for get_kv_buffer
        self._dummy = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.k_buffer = [self._dummy for _ in range(layer_num)]
        self.v_buffer = [self._dummy for _ in range(layer_num)]

        self.k_data_ptrs = torch.zeros(layer_num, dtype=torch.uint64, device=device)
        self.v_data_ptrs = torch.zeros(layer_num, dtype=torch.uint64, device=device)
        self.data_ptrs   = torch.zeros(2 * layer_num, dtype=torch.uint64, device=device)
        self.data_strides = torch.zeros(2 * layer_num, dtype=torch.int64, device=device)

        k_bytes, v_bytes = self.get_kv_size_bytes()
        total_gb = (k_bytes + v_bytes) / (1024**3)
        bf16_gb = total * head_num * head_dim * 2 * 2 * layer_num / (1024**3)
        self.mem_usage = total_gb
        logger.info(
            "TurboQuantKVPool: %d tokens, %d layers, %.2f GB (vs %.2f GB BF16, %.1fx)",
            size, layer_num, total_gb, bf16_gb, bf16_gb / max(total_gb, 1e-9),
        )

    # ---- set / get ----

    def _assign_recent_slots(self, loc: Tensor) -> Tensor:
        loc_flat = loc.reshape(-1).long()
        slots = torch.empty_like(loc_flat, dtype=torch.int32)
        if self.recent_raw_capacity <= 0:
            return slots.fill_(-1).view_as(loc)

        loc_list = loc_flat.tolist()
        for idx, cur_loc in enumerate(loc_list):
            slot = self._recent_next_slot
            old_loc = int(self.recent_slot_loc[slot].item())
            if old_loc >= 0:
                if int(self.recent_slot_of_loc[old_loc].item()) == slot:
                    self.recent_slot_of_loc[old_loc] = -1
            self.recent_slot_loc[slot] = cur_loc
            self.recent_slot_of_loc[cur_loc] = slot
            slots[idx] = slot
            self._recent_next_slot = (slot + 1) % self.recent_raw_capacity
        return slots.view_as(loc)

    def _lookup_recent_slots(self, loc: Tensor) -> Tensor:
        if self.recent_raw_capacity <= 0:
            return torch.full_like(loc, -1, dtype=torch.int32)
        return self.recent_slot_of_loc[loc.long()]

    def set_kv_buffer(
        self, layer, loc, cache_k, cache_v,
        k_scale=None, v_scale=None, layer_id_override=None,
    ):
        layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
        li = layer_id - self.start_layer
        self._current_layer_id = layer_id

        if self.recent_raw_capacity > 0:
            if layer_id == self.start_layer:
                if loc.numel() <= self.recent_raw_max_write:
                    slots = self._assign_recent_slots(loc)
                else:
                    slots = torch.full_like(loc, -1, dtype=torch.int32)
            else:
                slots = self._lookup_recent_slots(loc)
            valid_slots = slots.reshape(-1) >= 0
            if valid_slots.any():
                slot_flat = slots.reshape(-1)[valid_slots].long()
                cache_k_flat = cache_k.reshape(-1, self.head_num, self.head_dim)[valid_slots]
                cache_v_flat = cache_v.reshape(-1, self.head_num, self.v_head_dim)[valid_slots]
                self.recent_k[li][slot_flat] = cache_k_flat.to(torch.bfloat16)
                self.recent_v[li][slot_flat] = cache_v_flat.to(torch.bfloat16)

        km, kq, kn, krn = self.tq.quantize(
            cache_k, layer_id, use_qjl=self.use_qjl, mse_bits=self.mse_bits,
            grouped_64=self.grouped_64,
        )
        vm, vq, vn, vrn = self.tq.quantize(
            cache_v, layer_id, use_qjl=self.use_qjl, mse_bits=self.mse_bits,
            grouped_64=self.grouped_64,
        )

        self.k_mse[li][loc]   = km
        self.k_qjl[li][loc]   = kq
        self.k_norm[li][loc]  = kn
        self.k_rnorm[li][loc] = krn
        self.v_mse[li][loc]   = vm
        self.v_qjl[li][loc]   = vq
        self.v_norm[li][loc]  = vn
        self.v_rnorm[li][loc] = vrn

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

    # ---- page dequantization ----

    def dequantize_pages(self, layer_id: int, page_indices: Tensor):
        li = layer_id - self.start_layer
        N = page_indices.shape[0]
        offsets = torch.arange(self.page_size, device=self.device)
        locs = (page_indices.unsqueeze(-1) * self.page_size + offsets).reshape(-1)

        if self.recent_raw_capacity > 0:
            slots = self._lookup_recent_slots(locs)
            recent_mask = slots >= 0
        else:
            slots = None
            recent_mask = None

        if recent_mask is not None and recent_mask.any():
            k = torch.empty(
                locs.shape[0], self.head_num, self.head_dim,
                dtype=torch.bfloat16, device=self.device,
            )
            v = torch.empty(
                locs.shape[0], self.head_num, self.v_head_dim,
                dtype=torch.bfloat16, device=self.device,
            )
            recent_slots = slots[recent_mask].long()
            k[recent_mask] = self.recent_k[li][recent_slots]
            v[recent_mask] = self.recent_v[li][recent_slots]

            old_mask = ~recent_mask
            if old_mask.any():
                old_locs = locs[old_mask]
                k[old_mask] = self.tq.dequantize(
                    self.k_mse[li][old_locs], self.k_qjl[li][old_locs],
                    self.k_norm[li][old_locs], self.k_rnorm[li][old_locs], layer_id,
                    use_qjl=self.use_qjl, mse_bits=self.mse_bits,
                    grouped_64=self.grouped_64,
                )
                v[old_mask] = self.tq.dequantize(
                    self.v_mse[li][old_locs], self.v_qjl[li][old_locs],
                    self.v_norm[li][old_locs], self.v_rnorm[li][old_locs], layer_id,
                    use_qjl=self.use_qjl, mse_bits=self.mse_bits,
                    grouped_64=self.grouped_64,
                )
        else:
            k = self.tq.dequantize(
                self.k_mse[li][locs], self.k_qjl[li][locs],
                self.k_norm[li][locs], self.k_rnorm[li][locs], layer_id,
                use_qjl=self.use_qjl, mse_bits=self.mse_bits,
                grouped_64=self.grouped_64,
            )
            v = self.tq.dequantize(
                self.v_mse[li][locs], self.v_qjl[li][locs],
                self.v_norm[li][locs], self.v_rnorm[li][locs], layer_id,
                use_qjl=self.use_qjl, mse_bits=self.mse_bits,
                grouped_64=self.grouped_64,
            )
        return (
            k.view(N, self.page_size, self.head_num, self.head_dim),
            v.view(N, self.page_size, self.head_num, self.v_head_dim),
        )

    # ---- page move ----

    def move_kv_cache(self, tgt_loc: Tensor, src_loc: Tensor):
        if tgt_loc.numel() == 0:
            return
        if self.recent_raw_capacity > 0:
            slots = self.recent_slot_of_loc[src_loc.long()]
            valid = slots >= 0
            if valid.any():
                valid_slots = slots[valid].long()
                self.recent_slot_of_loc[tgt_loc[valid].long()] = slots[valid]
                self.recent_slot_of_loc[src_loc[valid].long()] = -1
                self.recent_slot_loc[valid_slots] = tgt_loc[valid].long()
        for li in range(self.layer_num):
            for buf_list in (self.k_mse, self.k_qjl, self.k_norm, self.k_rnorm,
                             self.v_mse, self.v_qjl, self.v_norm, self.v_rnorm):
                buf_list[li][tgt_loc] = buf_list[li][src_loc]

    # ---- compat ----

    def get_kv_size_bytes(self):
        k = sum(t.nbytes for bufs in (self.k_mse, self.k_qjl, self.k_norm, self.k_rnorm) for t in bufs)
        v = sum(t.nbytes for bufs in (self.v_mse, self.v_qjl, self.v_norm, self.v_rnorm) for t in bufs)
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
        for attr in ("k_mse", "k_qjl", "k_norm", "k_rnorm",
                      "v_mse", "v_qjl", "v_norm", "v_rnorm"):
            delattr(self, attr)
        for attr in ("recent_k", "recent_v", "recent_slot_of_loc", "recent_slot_loc"):
            if hasattr(self, attr):
                delattr(self, attr)

    def maybe_get_custom_mem_pool(self):
        return None


# ===================================================================
# Monkey-patched flash_attn_with_kvcache
# ===================================================================

_orig_flash_attn = None


def _patched_flash_attn(
    q, k_cache, v_cache, page_table=None, cache_seqlens=None, **kwargs
):
    pool = _ACTIVE_TQ_POOL

    if pool is None or page_table is None or k_cache.numel() > 0:
        return _orig_flash_attn(
            q, k_cache=k_cache, v_cache=v_cache,
            page_table=page_table, cache_seqlens=cache_seqlens,
            **kwargs,
        )

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
# Injection
# ===================================================================

def inject_turboquant(model_runner):
    """Replace BF16 KV cache with TurboQuant_prod 4-bit compressed pool."""
    global _ACTIVE_TQ_POOL, _orig_flash_attn

    if _ACTIVE_TQ_POOL is not None:
        logger.warning("TurboQuant: already injected, skipping")
        return

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
    logger.info("TurboQuant: freed BF16 pool. VRAM free: %.2f GB", free_before / 1e9)

    tq = TurboQuant(params["device"], num_layers=params["layer_num"])
    tq_pool = TurboQuantKVPool(
        tq=tq, use_qjl=False, mse_bits=4, grouped_64=True,
        recent_raw_capacity=min(params["size"] + params["page_size"], 1024),
        recent_raw_max_write=64,
        **params
    )
    model_runner.token_to_kv_pool = tq_pool
    _ACTIVE_TQ_POOL = tq_pool

    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
    if allocator is not None and hasattr(allocator, "_kvcache"):
        allocator._kvcache = tq_pool
        logger.info("TurboQuant: updated allocator._kvcache")

    free_after = torch.cuda.mem_get_info()[0]
    logger.info(
        "TurboQuant: pool created. VRAM free: %.2f GB (freed: %.2f GB)",
        free_after / 1e9, (free_after - free_before) / 1e9,
    )

    import sgl_kernel.flash_attn as _sgl_fa
    if _orig_flash_attn is None:
        _orig_flash_attn = _sgl_fa.flash_attn_with_kvcache
    _sgl_fa.flash_attn_with_kvcache = _patched_flash_attn

    try:
        import sglang.srt.layers.attention.flashattention_backend as _fa
        _fa.flash_attn_with_kvcache = _patched_flash_attn
    except (ImportError, AttributeError) as exc:
        logger.warning("Could not patch FA backend: %s", exc)

    logger.info(
        "TurboQuant: active. %d slots x %d layers, MSE-only reconstruction "
        "(QJL disabled, grouped 2x64 4-bit) + recent BF16 tail "
        "(cap=%d, max_write=%d)",
        params["size"], params["layer_num"], tq_pool.recent_raw_capacity,
        tq_pool.recent_raw_max_write,
    )
