import importlib.util
import math
from pathlib import Path

import torch


def _load_turboquant_module():
    path = Path(__file__).resolve().parents[1] / "sglang_omni" / "engines" / "ar" / "sglang_backend" / "turboquant.py"
    spec = importlib.util.spec_from_file_location("turboquant_local", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_randomized_hadamard_applies_signs_before_transform():
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    x = torch.randn(32, 128)
    expected = (x * tq.D_signs[0]) @ tq.H_norm

    y = tq._rotate_forward(x, 0)
    x_roundtrip = tq._rotate_inverse(y, 0)

    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_roundtrip, x, atol=1e-5, rtol=1e-5)


def test_quantization_stays_stable_on_hadamard_basis_vectors():
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    hadamard = mod._build_hadamard_matrix(128).float() / math.sqrt(128)
    packed_mse, packed_qjl, norms, r_norms = tq.quantize(hadamard, 0)
    restored = tq.dequantize(packed_mse, packed_qjl, norms, r_norms, 0).float()

    mse = ((hadamard - restored) ** 2).sum(dim=-1).mean().item()

    assert mse < 0.2
