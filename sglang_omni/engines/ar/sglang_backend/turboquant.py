import torch
import math
import torch.nn.functional as F
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
import sgl_kernel.flash_attn

# Centroids for d=64
CENTROIDS_3BIT = [-0.264483, -0.166657, -0.094167, -0.030803, 0.030023, 0.093285, 0.165646, 0.263617]
CENTROIDS_2BIT = [-0.187196, -0.056323, 0.056377, 0.187178]

def pack_3bit(idx: torch.Tensor) -> torch.Tensor:
    idx = idx.view(*idx.shape[:-1], 8, 8)
    b0 = (idx[..., 0]) | (idx[..., 1] << 3) | ((idx[..., 2] & 0x3) << 6)
    b1 = (idx[..., 2] >> 2) | (idx[..., 3] << 1) | (idx[..., 4] << 4) | ((idx[..., 5] & 0x1) << 7)
    b2 = (idx[..., 5] >> 1) | (idx[..., 6] << 2) | (idx[..., 7] << 5)
    return torch.stack([b0, b1, b2], dim=-1).view(*idx.shape[:-2], 24).to(torch.uint8)

def unpack_3bit(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(*packed.shape[:-1], 8, 3).int()
    b0, b1, b2 = packed[..., 0], packed[..., 1], packed[..., 2]
    v0 = b0 & 0x7
    v1 = (b0 >> 3) & 0x7
    v2 = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
    v3 = (b1 >> 1) & 0x7
    v4 = (b1 >> 4) & 0x7
    v5 = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
    v6 = (b2 >> 2) & 0x7
    v7 = (b2 >> 5) & 0x7
    return torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).view(*packed.shape[:-2], 64).long()

def pack_2bit(idx: torch.Tensor) -> torch.Tensor:
    idx = idx.view(*idx.shape[:-1], 16, 4)
    b0 = idx[..., 0] | (idx[..., 1] << 2) | (idx[..., 2] << 4) | (idx[..., 3] << 6)
    return b0.view(*idx.shape[:-2], 16).to(torch.uint8)

def unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(*packed.shape[:-1], 16, 1).int()
    b0 = packed[..., 0]
    v0 = b0 & 0x3
    v1 = (b0 >> 2) & 0x3
    v2 = (b0 >> 4) & 0x3
    v3 = (b0 >> 6) & 0x3
    return torch.stack([v0, v1, v2, v3], dim=-1).view(*packed.shape[:-2], 64).long()

def pack_1bit(idx: torch.Tensor) -> torch.Tensor:
    idx = idx.view(*idx.shape[:-1], 8, 8)
    b0 = idx[..., 0] | (idx[..., 1] << 1) | (idx[..., 2] << 2) | (idx[..., 3] << 3) | \
         (idx[..., 4] << 4) | (idx[..., 5] << 5) | (idx[..., 6] << 6) | (idx[..., 7] << 7)
    return b0.view(*idx.shape[:-2], 8).to(torch.uint8)

def unpack_1bit(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(*packed.shape[:-1], 8, 1).int()
    b0 = packed[..., 0]
    return torch.stack([(b0 >> i) & 1 for i in range(8)], dim=-1).view(*packed.shape[:-2], 64).long()

def quantize_mse(x: torch.Tensor, centroids: torch.Tensor):
    dists = (x.unsqueeze(-1) - centroids.unsqueeze(0)).abs()
    idx = dists.argmin(dim=-1)
    quantized = centroids[idx]
    return idx, x - quantized

class TurboQuant:
    def __init__(self, device):
        self.device = device
        self.d = 64
        torch.manual_seed(42)
        q1, _ = torch.linalg.qr(torch.randn(64, 64, device=device))
        self.pi_outlier = q1
        q2, _ = torch.linalg.qr(torch.randn(64, 64, device=device))
        self.pi_inlier = q2
        self.s_outlier = torch.randn(64, 64, device=device)
        self.s_inlier = torch.randn(64, 64, device=device)
        
        self.c3 = torch.tensor(CENTROIDS_3BIT, device=device, dtype=torch.float32)
        self.c2 = torch.tensor(CENTROIDS_2BIT, device=device, dtype=torch.float32)
    
    def quantize(self, x: torch.Tensor):
        x_outlier = x[..., :64].float()
        x_inlier = x[..., 64:].float()
        
        y_outlier = x_outlier @ self.pi_outlier
        idx_out, r_out = quantize_mse(y_outlier, self.c3)
        r_out_orig = r_out @ self.pi_outlier.T
        r_out_norm = r_out_orig.norm(dim=-1, keepdim=True)
        qjl_out = (self.s_outlier @ r_out_orig.unsqueeze(-1)).squeeze(-1) > 0
        
        y_inlier = x_inlier @ self.pi_inlier
        idx_in, r_in = quantize_mse(y_inlier, self.c2)
        r_in_orig = r_in @ self.pi_inlier.T
        r_in_norm = r_in_orig.norm(dim=-1, keepdim=True)
        qjl_in = (self.s_inlier @ r_in_orig.unsqueeze(-1)).squeeze(-1) > 0
        
        pack_idx_out = pack_3bit(idx_out)
        pack_idx_in = pack_2bit(idx_in)
        pack_qjl_out = pack_1bit(qjl_out.long())
        pack_qjl_in = pack_1bit(qjl_in.long())
        
        return pack_idx_out, pack_idx_in, pack_qjl_out, pack_qjl_in, r_out_norm.half(), r_in_norm.half()

    def dequantize(self, p_idx_out, p_idx_in, p_qjl_out, p_qjl_in, r_out_norm, r_in_norm):
        idx_out = unpack_3bit(p_idx_out)
        idx_in = unpack_2bit(p_idx_in)
        qjl_out = unpack_1bit(p_qjl_out).float() * 2 - 1.0
        qjl_in = unpack_1bit(p_qjl_in).float() * 2 - 1.0
        
        y_out = self.c3[idx_out]
        x_mse_out = y_out @ self.pi_outlier.T
        x_qjl_out = (r_out_norm.float() * math.sqrt(math.pi / 2) / self.d) * (qjl_out @ self.s_outlier.T)
        x_out = x_mse_out + x_qjl_out
        
        y_in = self.c2[idx_in]
        x_mse_in = y_in @ self.pi_inlier.T
        x_qjl_in = (r_in_norm.float() * math.sqrt(math.pi / 2) / self.d) * (qjl_in @ self.s_inlier.T)
        x_in = x_mse_in + x_qjl_in
        
        return torch.cat([x_out, x_in], dim=-1).to(torch.bfloat16)

class TurboQuantKVPool(MHATokenToKVPool):
    def __init__(self, size: int, page_size: int, dtype: torch.dtype, head_num: int, head_dim: int, layer_num: int, device: str, enable_memory_saver: bool, **kwargs):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.num_kv_heads = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.device = device
        
        # Calculate flat padded size similar to SGLang
        pad_size = (size + page_size - 1) // page_size * page_size
        self.flat_size = pad_size
        
        self.tq = TurboQuant(device)
        
        # Allocate buffers. SGLang uses a list of tensors for layers.
        # We will keep a flat tensor and index it by layer_id.
        shape_prefix = (self.layer_num, self.flat_size, self.num_kv_heads)
        
        self.k_idx_out = torch.zeros(*shape_prefix, 24, dtype=torch.uint8, device=device)
        self.k_idx_in = torch.zeros(*shape_prefix, 16, dtype=torch.uint8, device=device)
        self.k_qjl_out = torch.zeros(*shape_prefix, 8, dtype=torch.uint8, device=device)
        self.k_qjl_in = torch.zeros(*shape_prefix, 8, dtype=torch.uint8, device=device)
        self.k_r_out = torch.zeros(*shape_prefix, 1, dtype=torch.float16, device=device)
        self.k_r_in = torch.zeros(*shape_prefix, 1, dtype=torch.float16, device=device)
        
        self.v_idx_out = torch.zeros(*shape_prefix, 24, dtype=torch.uint8, device=device)
        self.v_idx_in = torch.zeros(*shape_prefix, 16, dtype=torch.uint8, device=device)
        self.v_qjl_out = torch.zeros(*shape_prefix, 8, dtype=torch.uint8, device=device)
        self.v_qjl_in = torch.zeros(*shape_prefix, 8, dtype=torch.uint8, device=device)
        self.v_r_out = torch.zeros(*shape_prefix, 1, dtype=torch.float16, device=device)
        self.v_r_in = torch.zeros(*shape_prefix, 1, dtype=torch.float16, device=device)
        
        self.current_layer_id = 0
        
        self.max_temp_pages = 4096  # enough for ~48 concurrent × ~85 pages each
        self.temp_k = torch.zeros(self.max_temp_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)
        self.temp_v = torch.zeros(self.max_temp_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=device)

    def set_kv_buffer(self, layer, cache_loc, k, v, k_scale=None, v_scale=None):
        layer_id = layer.layer_id

        try:
            k_pack = self.tq.quantize(k)
            v_pack = self.tq.quantize(v)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                f"TQ quantize failed: k={k.shape} v={v.shape} cache_loc={cache_loc.shape} err={e}"
            )
            raise
        
        self.k_idx_out[layer_id, cache_loc] = k_pack[0]
        self.k_idx_in[layer_id, cache_loc] = k_pack[1]
        self.k_qjl_out[layer_id, cache_loc] = k_pack[2]
        self.k_qjl_in[layer_id, cache_loc] = k_pack[3]
        self.k_r_out[layer_id, cache_loc] = k_pack[4]
        self.k_r_in[layer_id, cache_loc] = k_pack[5]

        self.v_idx_out[layer_id, cache_loc] = v_pack[0]
        self.v_idx_in[layer_id, cache_loc] = v_pack[1]
        self.v_qjl_out[layer_id, cache_loc] = v_pack[2]
        self.v_qjl_in[layer_id, cache_loc] = v_pack[3]
        self.v_r_out[layer_id, cache_loc] = v_pack[4]
        self.v_r_in[layer_id, cache_loc] = v_pack[5]

    def get_kv_buffer(self, layer_id):
        self.current_layer_id = layer_id
        return torch.empty(0, dtype=torch.bfloat16, device=self.device), torch.empty(0, dtype=torch.bfloat16, device=self.device)

    def dequantize_pages(self, pages: torch.Tensor):
        layer_id = self.current_layer_id
        # pages: [N]
        # In our flat cache_loc structure, page i corresponds to indices [i * page_size : (i+1) * page_size].
        # We need to gather the actual cache_locs for these pages!
        N = pages.numel()
        offsets = torch.arange(self.page_size, device=self.device)
        cache_locs = (pages.unsqueeze(-1) * self.page_size + offsets).view(-1)
        
        if N > self.max_temp_pages:
            self.max_temp_pages = N * 2
            self.temp_k = torch.zeros(self.max_temp_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=self.device)
            self.temp_v = torch.zeros(self.max_temp_pages, self.page_size, self.num_kv_heads, self.head_dim, dtype=torch.bfloat16, device=self.device)
            
        k_pack = (
            self.k_idx_out[layer_id, cache_locs], self.k_idx_in[layer_id, cache_locs],
            self.k_qjl_out[layer_id, cache_locs], self.k_qjl_in[layer_id, cache_locs],
            self.k_r_out[layer_id, cache_locs], self.k_r_in[layer_id, cache_locs]
        )
        v_pack = (
            self.v_idx_out[layer_id, cache_locs], self.v_idx_in[layer_id, cache_locs],
            self.v_qjl_out[layer_id, cache_locs], self.v_qjl_in[layer_id, cache_locs],
            self.v_r_out[layer_id, cache_locs], self.v_r_in[layer_id, cache_locs]
        )
        
        k_deq = self.tq.dequantize(*k_pack).view(N, self.page_size, self.num_kv_heads, self.head_dim)
        v_deq = self.tq.dequantize(*v_pack).view(N, self.page_size, self.num_kv_heads, self.head_dim)
        
        self.temp_k[:N].copy_(k_deq)
        self.temp_v[:N].copy_(v_deq)
        return self.temp_k[:N], self.temp_v[:N]

# Global reference to the active pool for the monkey patch
GLOBAL_TQ_POOL = None

def inject_turboquant(model_runner):
    global GLOBAL_TQ_POOL
    import logging
    logger = logging.getLogger(__name__)

    pool = model_runner.token_to_kv_pool
    size = pool.size
    page_size = pool.page_size
    head_num = pool.head_num
    head_dim = pool.head_dim
    layer_num = pool.layer_num
    device = pool.device

    # Force-free every tensor in the old KV pool
    # 1. Zero the buffer lists (36 tensors each, ~200MB per tensor)
    if hasattr(pool, 'k_buffer') and pool.k_buffer is not None:
        for i in range(len(pool.k_buffer)):
            pool.k_buffer[i] = None
        pool.k_buffer = None
    if hasattr(pool, 'v_buffer') and pool.v_buffer is not None:
        for i in range(len(pool.v_buffer)):
            pool.v_buffer[i] = None
        pool.v_buffer = None
    # 2. Clear derived pointer tensors
    for attr in ('k_data_ptrs', 'v_data_ptrs', 'data_ptrs', 'data_strides'):
        if hasattr(pool, attr):
            setattr(pool, attr, None)
    # 3. Drop the pool reference itself
    del pool
    model_runner.token_to_kv_pool = None
    # 4. Force Python GC + CUDA cache clear
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    free_before = torch.cuda.mem_get_info()[0]
    logger.info(f"TurboQuant: freed old KV pool. VRAM free: {free_before/1e9:.1f} GB")

    tq_pool = TurboQuantKVPool(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = tq_pool
    GLOBAL_TQ_POOL = tq_pool

    free_after = torch.cuda.mem_get_info()[0]
    logger.info(f"TurboQuant: KV pool replaced. VRAM free: {free_after/1e9:.1f} GB (saved {(free_after-free_before+1.81e9)/1e9:.1f} GB)")

# Monkey patch flash_attn_with_kvcache
orig_flash_attn = sgl_kernel.flash_attn.flash_attn_with_kvcache

def my_flash_attn_with_kvcache(q, k_cache, v_cache, page_table=None, cache_seqlens=None, **kwargs):
    if GLOBAL_TQ_POOL is None or page_table is None:
        return orig_flash_attn(q, k_cache=k_cache, v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens, **kwargs)
    
    # Check if k_cache is dummy
    if k_cache.numel() == 0:
        bs, max_pages = page_table.shape
        flat_pages = page_table.flatten()
        k_deq, v_deq = GLOBAL_TQ_POOL.dequantize_pages(flat_pages)
        temp_page_table = torch.arange(flat_pages.numel(), dtype=torch.int32, device=q.device).view(bs, max_pages)
        return orig_flash_attn(q, k_cache=k_deq, v_cache=v_deq, page_table=temp_page_table, cache_seqlens=cache_seqlens, **kwargs)
    else:
        return orig_flash_attn(q, k_cache=k_cache, v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens, **kwargs)

sgl_kernel.flash_attn.flash_attn_with_kvcache = my_flash_attn_with_kvcache

# ALSO patch SGLang's internal wrapper if it was already imported
import sglang.srt.layers.attention.flashattention_backend
sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache = my_flash_attn_with_kvcache
