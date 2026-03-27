# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS audio tokenizer (codec decoder): codebook tokens → waveform.

Architecture (from params.json):
  - Causal conv+transformer codec at 24kHz
  - Encoder: Conv → downsample blocks with sliding-window attention (not needed for TTS decode)
  - Decoder: Conv → upsample blocks with sliding-window attention → waveform
  - Quantizer: SemanticCodebook (8192, dim=256, EMA) + AcousticCodebook (FSQ, 21 levels, 36 dims)
  - Frame rate: 12.5 Hz (24000 / (240 × 8) = 12.5)
  - Each frame = 80ms of audio = 1920 samples

The decoder pipeline:
  1. Decode semantic+acoustic codes → continuous embeddings
  2. Concat → [B, 292, T] continuous representation
  3. Run through decoder blocks (conv transpose + transformer)
  4. Output projection → waveform [B, 1, T × upsample_factor]
"""

from __future__ import annotations

import math
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

weight_norm = torch.nn.utils.parametrizations.weight_norm


# ---------------------------------------------------------------------------
# Codec primitives
# ---------------------------------------------------------------------------


class SemanticCodebook(nn.Module):
    """EMA-based semantic codebook (Euclidean distance lookup)."""

    def __init__(self, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            emb = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
            self.register_buffer("_embedding", emb, persistent=False)
            return emb
        return self._embedding

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] integer codes → [B, semantic_dim, T] continuous."""
        codes = codes.squeeze(1)  # [B, T]
        quantized = F.embedding(codes, self.embedding.to(codes.device))
        return quantized.permute(0, 2, 1)  # [B, D, T]


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization (FSQ) for acoustic codebooks."""

    def __init__(self, n_levels: int, codebook_dim: int) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.dim = codebook_dim

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[B, C, T] integer codes → [B, C, T] continuous in [-1, 1]."""
        return ((codes.to(dtype) * 2 / (self.n_levels - 1)) - 1)


class MistralAudioCodebook(nn.Module):
    """Combined semantic + acoustic codebook for Voxtral codec."""

    def __init__(self, semantic_codebook_size: int, semantic_dim: int,
                 acoustic_codebook_size: int, acoustic_dim: int) -> None:
        super().__init__()
        self.semantic_codebook = SemanticCodebook(semantic_codebook_size, semantic_dim)
        self.acoustic_codebook = AcousticCodebook(acoustic_codebook_size, acoustic_dim)
        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

    @property
    def num_codebooks(self) -> int:
        return 1 + self.acoustic_dim  # 1 semantic + N acoustic

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """[B, K, T] codes → [B, D, T] continuous (D = semantic_dim + acoustic_dim)."""
        semantic_codes = codes[:, :1, :]
        acoustic_codes = codes[:, 1:, :]
        semantic_emb = self.semantic_codebook.decode(semantic_codes).to(dtype)
        acoustic_emb = self.acoustic_codebook.decode(acoustic_codes, dtype)
        return torch.cat([semantic_emb, acoustic_emb], dim=1)


# ---------------------------------------------------------------------------
# Causal convolutions (matching Voxtral's decoder architecture)
# ---------------------------------------------------------------------------


def pad1d(x: torch.Tensor, paddings: tuple[int, int], mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 dilation: int = 1, use_weight_norm: bool = True) -> None:
        super().__init__()
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=0, dilation=dilation)
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self._stride = stride
        self._effective_ks = (kernel_size - 1) * dilation + 1
        self._padding_total = self._effective_ks - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (x.shape[-1] - self._effective_ks + self._padding_total) / self._stride + 1
        target_len = (math.ceil(n_frames) - 1) * self._stride + (self._effective_ks - self._padding_total)
        extra = target_len - x.shape[-1]
        x = pad1d(x, (self._padding_total, extra), mode="reflect")
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 use_weight_norm: bool = True) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride)
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_pad = kernel_size - stride
        out = self.conv(x)
        right_pad = math.ceil(total_pad)
        left_pad = total_pad - right_pad
        return out[..., left_pad: out.shape[-1] - right_pad]


# ---------------------------------------------------------------------------
# Decoder transformer blocks (sliding-window causal attention)
# ---------------------------------------------------------------------------


class CodecFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecAttention(nn.Module):
    """Sliding-window causal self-attention for codec transformer blocks."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 window_size: int, qk_norm: bool = True, qk_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.repeats = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.repeats > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Sliding window causal mask
        if self.window_size > 0 and T > self.window_size:
            # Build mask: causal + sliding window
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            row_idx = torch.arange(T, device=x.device).unsqueeze(1)
            col_idx = torch.arange(T, device=x.device).unsqueeze(0)
            window_mask = (row_idx - col_idx) < self.window_size
            mask = mask & window_mask
            y = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask.unsqueeze(0).unsqueeze(0))
        else:
            y = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class CodecTransformerBlock(nn.Module):
    """Pre-norm transformer block with layer scale for codec decoder."""

    def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, window_size: int, norm_eps: float,
                 qk_norm: bool = True, qk_norm_eps: float = 1e-6,
                 layer_scale: bool = True, layer_scale_init: float | None = None) -> None:
        super().__init__()
        self.attention = CodecAttention(dim, n_heads, n_kv_heads, head_dim, window_size, qk_norm, qk_norm_eps)
        self.feed_forward = CodecFeedForward(dim, hidden_dim)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

        if layer_scale:
            init_val = layer_scale_init if layer_scale_init is not None else 0.01
            self.attention_scale = nn.Parameter(init_val * torch.ones(dim))
            self.ffn_scale = nn.Parameter(init_val * torch.ones(dim))
        else:
            self.register_buffer("attention_scale", None)
            self.register_buffer("ffn_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] (time-second format for transformer)
        r = self.attention(self.attention_norm(x))
        if self.attention_scale is not None:
            r = r * self.attention_scale
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
        if self.ffn_scale is not None:
            r = r * self.ffn_scale
        return h + r


# ---------------------------------------------------------------------------
# Full codec decoder
# ---------------------------------------------------------------------------


class VoxtralAudioTokenizer(nn.Module):
    """Voxtral audio codec decoder: codebook tokens → waveform.

    Decoder structure (from weights):
      decoder_blocks.{0,2,4,6} = CausalConv(Transpose)1d  (upsampling)
      decoder_blocks.{1,3,5,7} = 2-layer transformer block
      output_proj = CausalConv1d → waveform

    Strides: [1, 2, 2, 2] → total upsample = 8
    Patch size: 240 → total samples per frame = 240 × 8 = 1920
    Frame rate: 24000 / 1920 = 12.5 Hz
    """

    def __init__(
        self,
        *,
        semantic_codebook_size: int = 8192,
        semantic_dim: int = 256,
        acoustic_codebook_size: int = 21,
        acoustic_dim: int = 36,
        dim: int = 1024,
        hidden_dim: int = 4096,
        head_dim: int = 128,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        norm_eps: float = 0.01,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        layer_scale: bool = True,
        layer_scale_init: float | None = 0.01,
        pretransform_patch_size: int = 240,
        patch_proj_kernel_size: int = 7,
        decoder_convs_kernels: tuple[int, ...] = (3, 4, 4, 4),
        decoder_convs_strides: tuple[int, ...] = (1, 2, 2, 2),
        decoder_transformer_lengths: tuple[int, ...] = (2, 2, 2, 2),
        attn_sliding_window_size: int = 16,
        half_attn_window_upon_downsampling: bool = True,
        channels: int = 1,
        sampling_rate: int = 24000,
        conv_weight_norm: bool = True,
    ) -> None:
        super().__init__()

        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim
        self.total_input_dim = semantic_dim + acoustic_dim  # 292
        self.dim = dim
        self.sampling_rate = sampling_rate
        self.pretransform_patch_size = pretransform_patch_size

        # Quantizer
        self.quantizer = MistralAudioCodebook(
            semantic_codebook_size, semantic_dim,
            acoustic_codebook_size, acoustic_dim,
        )

        # Decoder blocks: alternating conv + transformer
        self.decoder_blocks = nn.ModuleList()
        current_window = attn_sliding_window_size
        # Scale window upward for decoder (opposite of encoder downsampling)
        if half_attn_window_upon_downsampling:
            # Start with smallest window, double at each upsample stage
            n_upsamples = sum(1 for s in decoder_convs_strides if s > 1)
            current_window = attn_sliding_window_size * (2 ** n_upsamples)

        for i, (kernel, stride, n_transformer_layers) in enumerate(
            zip(decoder_convs_kernels, decoder_convs_strides, decoder_transformer_lengths)
        ):
            # Conv (transpose for upsampling, or regular for stride=1)
            in_ch = self.total_input_dim if i == 0 else dim
            if stride > 1:
                self.decoder_blocks.append(
                    CausalConvTranspose1d(in_ch, dim, kernel, stride, use_weight_norm=conv_weight_norm)
                )
            else:
                self.decoder_blocks.append(
                    CausalConv1d(in_ch, dim, kernel, stride, use_weight_norm=conv_weight_norm)
                )

            # Transformer layers
            if n_transformer_layers > 0:
                layers = nn.ModuleList([
                    CodecTransformerBlock(
                        dim=dim, hidden_dim=hidden_dim, n_heads=n_heads,
                        n_kv_heads=n_kv_heads, head_dim=head_dim,
                        window_size=current_window, norm_eps=norm_eps,
                        qk_norm=qk_norm, qk_norm_eps=qk_norm_eps,
                        layer_scale=layer_scale, layer_scale_init=layer_scale_init,
                    )
                    for _ in range(n_transformer_layers)
                ])
                self.decoder_blocks.append(layers)

            if stride > 1 and half_attn_window_upon_downsampling:
                current_window = max(current_window // 2, 1)

        # Output projection: dim → patch_size × channels
        self.output_proj = CausalConv1d(
            dim, pretransform_patch_size * channels, patch_proj_kernel_size,
            use_weight_norm=conv_weight_norm,
        )

        # Compute downsample factor
        self._strides = decoder_convs_strides
        self.downsample_factor = pretransform_patch_size * math.prod(decoder_convs_strides)

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_factor

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codebook tokens to audio waveform.

        Args:
            codes: [B, K, T] integer codes (K = num_codebooks = 37)

        Returns:
            audio: [B, 1, T_audio] waveform at sampling_rate
        """
        # Strip special token offset (EMPTY_AUDIO=0, END_AUDIO=1 prepended by
        # the acoustic transformer). The raw codebook indices start at offset 2.
        N_SPECIAL = 2
        codes = codes.clone()
        codes[:, :1, :] = (codes[:, :1, :] - N_SPECIAL).clamp(min=0)   # semantic
        codes[:, 1:, :] = (codes[:, 1:, :] - N_SPECIAL).clamp(min=0)   # acoustic

        # Codes → continuous representation
        # Use the model's own dtype for consistency with weight-normed convolutions
        model_dtype = next(self.parameters()).dtype
        x = self.quantizer.decode(codes, dtype=model_dtype)  # [B, D=292, T]

        # Run decoder blocks
        for block in self.decoder_blocks:
            if isinstance(block, nn.ModuleList):
                # Transformer block: [B, D, T] → [B, T, D] → transformer → [B, D, T]
                x = x.transpose(1, 2)  # [B, T, D]
                for layer in block:
                    x = layer(x)
                x = x.transpose(1, 2)  # [B, D, T]
            else:
                # Conv block
                x = block(x)

        # Output projection: [B, dim, T] → [B, patch_size, T] → [B, 1, T*patch_size]
        x = self.output_proj(x)
        B, C, T = x.shape
        # Reshape: treat each position as a patch of patch_size samples
        x = x.view(B, 1, -1)  # [B, 1, T_audio]
        return x

    @torch.no_grad()
    def decode_frames(self, frame_codes_list: list[torch.Tensor]) -> torch.Tensor:
        """Decode a list of per-frame code tensors to audio.

        Args:
            frame_codes_list: list of [K] or [K, 1] tensors, one per frame

        Returns:
            audio: [T_audio] 1D waveform tensor
        """
        # Stack frames into [1, K, T]
        frames = []
        for codes in frame_codes_list:
            if codes.dim() == 1:
                codes = codes.unsqueeze(-1)  # [K] → [K, 1]
            frames.append(codes)
        codes = torch.cat(frames, dim=-1).unsqueeze(0)  # [1, K, T]
        audio = self.decode(codes)
        return audio[0, 0]  # [T_audio]

    def load_weight(self, name_tensor: tuple[str, torch.Tensor]) -> str:
        """Load a single weight tensor by name."""
        name, tensor = name_tensor
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        all_names = {**params, **buffers}

        if name in all_names:
            target = all_names[name]
            if isinstance(target, nn.Parameter):
                target.data.copy_(tensor)
            else:
                target.copy_(tensor)
            return name

        logger.debug("Audio tokenizer: skipping weight %s", name)
        return name

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "VoxtralAudioTokenizer":
        """Build from params.json audio_tokenizer_args dict."""
        tok_args = params.get("multimodal", {}).get("audio_tokenizer_args", {})

        def parse_str_tuple(s: str) -> tuple[int, ...]:
            return tuple(int(x) for x in s.split(","))

        return cls(
            semantic_codebook_size=tok_args.get("semantic_codebook_size", 8192),
            semantic_dim=tok_args.get("semantic_dim", 256),
            acoustic_codebook_size=tok_args.get("acoustic_codebook_size", 21),
            acoustic_dim=tok_args.get("acoustic_dim", 36),
            dim=tok_args.get("dim", 1024),
            hidden_dim=tok_args.get("hidden_dim", 4096),
            head_dim=tok_args.get("head_dim", 128),
            n_heads=tok_args.get("n_heads", 8),
            n_kv_heads=tok_args.get("n_kv_heads", 8),
            norm_eps=tok_args.get("norm_eps", 0.01),
            qk_norm=tok_args.get("qk_norm", True),
            qk_norm_eps=tok_args.get("qk_norm_eps", 1e-6),
            layer_scale=tok_args.get("layer_scale", True),
            layer_scale_init=tok_args.get("layer_scale_init", 0.01),
            pretransform_patch_size=tok_args.get("pretransform_patch_size", 240),
            patch_proj_kernel_size=tok_args.get("patch_proj_kernel_size", 7),
            decoder_convs_kernels=parse_str_tuple(tok_args.get("decoder_convs_kernels_str", "3,4,4,4")),
            decoder_convs_strides=parse_str_tuple(tok_args.get("decoder_convs_strides_str", "1,2,2,2")),
            decoder_transformer_lengths=parse_str_tuple(tok_args.get("decoder_transformer_lengths_str", "2,2,2,2")),
            attn_sliding_window_size=tok_args.get("attn_sliding_window_size", 16),
            half_attn_window_upon_downsampling=tok_args.get("half_attn_window_upon_downsampling", True),
            channels=tok_args.get("channels", 1),
            sampling_rate=tok_args.get("sampling_rate", 24000),
            conv_weight_norm=tok_args.get("conv_weight_norm", True),
        )
