# SPDX-License-Identifier: Apache-2.0
"""Flow-matching acoustic transformer for Voxtral TTS.

Generates audio codebook tokens from LLM hidden states via iterative
Euler ODE integration with classifier-free guidance (CFG).

Architecture:
  - 3-layer bidirectional transformer (dim=3072, 32 heads, 8 KV heads)
  - Input: [acoustic_embedding(1), time_embedding(1), llm_projection(1)] = 3 tokens
  - 8 Euler integration steps with CFG alpha=1.2
  - Output: 1 semantic code (argmax) + 36 acoustic codes (quantized flow output)

All operations are CUDA-graph-safe: uses pre-allocated buffers and
Tensor.normal_() for noise generation (captured as cuRAND kernel).
"""

from __future__ import annotations

import math
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow matching timestep."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class BidirectionalAttention(nn.Module):
    """Multi-head attention without RoPE (bidirectional, non-causal)."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
            x = x.unsqueeze(0)
        else:
            bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # GQA expansion
        if self.repeats > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)

        # [B, H, S, D] format for SDPA
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y).squeeze(0) if bsz == 1 and x.dim() == 2 else self.wo(y)


class AcousticTransformerBlock(nn.Module):
    """Pre-norm transformer block for acoustic transformer."""

    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, norm_eps: float, bias: bool = False) -> None:
        super().__init__()
        self.attention = BidirectionalAttention(dim, n_heads, n_kv_heads, head_dim, bias)
        self.feed_forward_w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.feed_forward_w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))
        r = self.feed_forward_w2(F.silu(self.feed_forward_w1(self.ffn_norm(h))) * self.feed_forward_w3(self.ffn_norm(h)))
        return h + r


class FlowMatchingAcousticTransformer(nn.Module):
    """Flow-matching acoustic transformer head for Voxtral TTS.

    Runs on each decode step after the LLM backbone produces hidden states.
    Generates 1 semantic codebook token + 36 acoustic codebook tokens.

    The forward pass is fully CUDA-graph-compatible when using pre-allocated
    buffers (see setup_buffers).
    """

    # Flow matching constants
    N_EULER_STEPS = 8
    CFG_ALPHA = 1.2
    NOISE_SCALE = 1.0

    # Audio special token IDs (offsets within acoustic output space)
    EMPTY_AUDIO_ID = 0
    END_AUDIO_ID = 1
    N_SPECIAL_TOKENS = 2

    def __init__(
        self,
        audio_model_args: dict[str, Any],
        *,
        dim: int = 3072,
        n_layers: int = 3,
        head_dim: int = 128,
        hidden_dim: int = 9216,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        # Parse audio model args
        self.semantic_codebook_size = audio_model_args.get("semantic_codebook_size", 8192)
        self.acoustic_codebook_size = audio_model_args.get("acoustic_codebook_size", 21)
        self.n_acoustic_codebook = audio_model_args.get("n_acoustic_codebook", 36)

        # Override from acoustic_transformer_args if present
        at_args = audio_model_args.get("acoustic_transformer_args", {})
        dim = at_args.get("dim", dim)
        n_layers = at_args.get("n_layers", n_layers)
        head_dim = at_args.get("head_dim", head_dim)
        hidden_dim = at_args.get("hidden_dim", hidden_dim)
        n_heads = at_args.get("n_heads", n_heads)
        n_kv_heads = at_args.get("n_kv_heads", n_kv_heads)
        norm_eps = at_args.get("sigma", norm_eps)
        input_dim = at_args.get("input_dim", dim)

        self.dim = dim
        self.n_layers = n_layers
        self.acoustic_levels = self.acoustic_codebook_size  # FSQ levels per acoustic codebook

        # Padded semantic codebook output size (including special tokens, padded to 128)
        padded_semantic = self._pad_to_multiple(
            self.semantic_codebook_size + self.N_SPECIAL_TOKENS, 128
        )

        # Projections
        self.input_projection = nn.Linear(self.n_acoustic_codebook, dim, bias=False)
        self.time_projection = nn.Linear(dim, dim, bias=False)
        self.llm_projection = nn.Linear(input_dim, dim, bias=False)

        # Time embedding
        self.time_embedding = TimeEmbedding(dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            AcousticTransformerBlock(
                dim=dim, hidden_dim=hidden_dim, n_heads=n_heads,
                n_kv_heads=n_kv_heads, head_dim=head_dim, norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim, eps=norm_eps)

        # Output heads
        self.semantic_codebook_output = nn.Linear(dim, padded_semantic, bias=False)
        self.acoustic_codebook_output = nn.Linear(dim, self.n_acoustic_codebook, bias=False)

        # Pre-computed timesteps (registered as buffer for device/dtype tracking)
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, self.N_EULER_STEPS),
            persistent=False,
        )

        # Semantic mask start: everything after (special_tokens + codebook_size) is masked
        self._semantic_mask_start = self.N_SPECIAL_TOKENS + self.semantic_codebook_size

    @staticmethod
    def _pad_to_multiple(n: int, multiple: int) -> int:
        return multiple * ((n + multiple - 1) // multiple)

    def setup_buffers(self, max_batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-allocate persistent buffers for CUDA graph compatibility."""
        # Noise buffer (filled with normal_() before each use — graph safe)
        self.register_buffer(
            "_noise",
            torch.zeros(max_batch_size, self.n_acoustic_codebook, device=device, dtype=dtype),
            persistent=False,
        )
        # Zero hidden states for unconditional CFG branch
        # Will be resized dynamically via slicing
        self.register_buffer(
            "_zero_hidden",
            torch.zeros(max_batch_size, self.dim, device=device, dtype=dtype),
            persistent=False,
        )
        logger.info(
            "Acoustic transformer buffers allocated: max_bs=%d, device=%s",
            max_batch_size, device,
        )

    def _predict_velocity(
        self,
        x_t: torch.Tensor,      # [2B, n_acoustic]
        llm_output: torch.Tensor,  # [2B, dim]
        t_emb: torch.Tensor,    # [2B, dim]
    ) -> torch.Tensor:
        """Single velocity prediction step through the transformer."""
        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        # Build input: [acoustic(1), time(1), llm(1)] = 3 tokens
        inputs = torch.cat([
            self.input_projection(x_t.unsqueeze(1)),
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ], dim=1)  # [2B, 3, dim]

        # Forward through transformer
        h = inputs
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Extract velocity from acoustic token position (index 0)
        return self.acoustic_codebook_output(h[:, 0, :])

    @torch.no_grad()
    def decode_one_frame(
        self,
        llm_hidden: torch.Tensor,  # [B, dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one frame of audio codes from LLM hidden states.

        Returns:
            semantic_code: [B, 1] — semantic codebook index (with special token offset)
            acoustic_codes: [B, n_acoustic] — acoustic codebook indices (with special token offset)
        """
        B = llm_hidden.shape[0]
        dtype = llm_hidden.dtype
        device = llm_hidden.device

        # --- Semantic code via constrained argmax ---
        semantic_logits = self.semantic_codebook_output(llm_hidden).float()
        # Mask: block EMPTY_AUDIO, allow END_AUDIO, allow codebook range, block padding
        semantic_logits[:, self.EMPTY_AUDIO_ID] = -float("inf")
        semantic_logits[:, self._semantic_mask_start:] = -float("inf")
        semantic_code = semantic_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

        # --- Flow matching: Euler ODE for acoustic codes ---
        should_decode = semantic_code.squeeze(1) != self.END_AUDIO_ID

        # Initial noise (normal_() is CUDA graph safe)
        self._noise[:B].normal_()
        x = self._noise[:B] * self.NOISE_SCALE

        # Zero hidden for unconditional branch
        zero_hidden = self._zero_hidden[:B]

        timesteps = self._timesteps.to(dtype=dtype)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - t

            t_emb = self.time_embedding(t.view(-1, 1).expand(B, 1)).to(dtype)

            # Batch cond + uncond in single forward pass (2B)
            x_batched = torch.cat([x, x], dim=0)
            llm_batched = torch.cat([llm_hidden, zero_hidden], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(x_batched.to(dtype), llm_batched, t_emb_batched)
            v_cond, v_uncond = v_all[:B], v_all[B:]

            # CFG combination
            v_t = self.CFG_ALPHA * v_cond + (1 - self.CFG_ALPHA) * v_uncond
            x = x + v_t * dt

        # --- Quantize acoustic codes ---
        x_clamped = torch.clamp(x, -1, 1)
        scaled = ((x_clamped + 1) / 2) * (self.acoustic_levels - 1)
        acoustic_codes = scaled.round().long()

        # Mask end-of-audio frames
        acoustic_codes[~should_decode] = self.EMPTY_AUDIO_ID

        # Offset by special tokens
        acoustic_codes = acoustic_codes + self.N_SPECIAL_TOKENS

        return semantic_code, acoustic_codes

    def load_weights(self, weights: dict[str, torch.Tensor]) -> set[str]:
        """Load weights with prefix stripping (acoustic_transformer.*)."""
        loaded = set()
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        all_names = {**params, **buffers}

        for name, tensor in weights.items():
            # Remap layer dict keys: layers.{0,1,2}.* → layers.{0,1,2}.*
            # The weight file uses layers.N.feed_forward.w1 etc. which maps to our
            # feed_forward_w1 (flattened naming). Handle the remapping.
            mapped = self._remap_weight_name(name)
            if mapped in all_names:
                target = all_names[mapped]
                if isinstance(target, nn.Parameter):
                    target.data.copy_(tensor)
                else:
                    target.copy_(tensor)
                loaded.add(name)
            else:
                logger.debug("Acoustic transformer: skipping weight %s (mapped to %s)", name, mapped)
        return loaded

    @staticmethod
    def _remap_weight_name(name: str) -> str:
        """Remap checkpoint weight names to module parameter names."""
        # layers.N.feed_forward.w1.weight → layers.N.feed_forward_w1.weight
        # layers.N.feed_forward.w2.weight → layers.N.feed_forward_w2.weight
        # layers.N.feed_forward.w3.weight → layers.N.feed_forward_w3.weight
        for suffix in ("w1", "w2", "w3"):
            old = f"feed_forward.{suffix}"
            new = f"feed_forward_{suffix}"
            if old in name:
                return name.replace(old, new)
        return name
