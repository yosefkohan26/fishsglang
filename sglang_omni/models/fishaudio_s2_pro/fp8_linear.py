"""
FP8 Linear layer replacement for Slow AR weight quantization.

Keeps weights in FP8 E4M3 format on GPU and uses torch._scaled_mm for
the GEMM. Activations are dynamically quantized to FP8 per-tensor before
the multiply. Output is BF16.

This halves the memory bandwidth per decode step (reads 3.79 GB instead
of 7.58 GB), directly speeding up the memory-bound decode loop.
"""

import torch
import torch.nn as nn


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using FP8 tensor core GEMM."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight stored in FP8 E4M3 — (out_features, in_features) layout
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(1, dtype=torch.float32),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, weight_fp8: torch.Tensor = None,
                    weight_scale: torch.Tensor = None) -> "FP8Linear":
        """Convert an nn.Linear to FP8Linear."""
        has_bias = linear.bias is not None
        fp8 = cls(linear.in_features, linear.out_features, bias=has_bias)

        if weight_fp8 is not None and weight_scale is not None:
            # Pre-quantized FP8 weights
            fp8.weight_fp8.copy_(weight_fp8)
            fp8.weight_scale.copy_(weight_scale)
        else:
            # Quantize BF16 weights to FP8 on the fly
            w = linear.weight.data.float()
            amax = w.abs().max().clamp(min=1e-12)
            scale = amax / 448.0
            fp8.weight_fp8.copy_((w / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
            fp8.weight_scale.copy_(scale)

        if has_bias:
            fp8.bias.data.copy_(linear.bias.data.bfloat16())

        return fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic per-tensor activation quantization to FP8
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        x_float = x_2d.float()
        x_amax = x_float.abs().max().clamp(min=1e-12)
        x_scale = x_amax / 448.0
        x_fp8 = (x_float / x_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

        # FP8 GEMM: (M, K) @ (N, K).T → (M, N) in BF16
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=x_scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )

        out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out
