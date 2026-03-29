import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_turboquant_module():
    path = Path(__file__).resolve().parents[1] / "sglang_omni" / "engines" / "ar" / "sglang_backend" / "turboquant.py"
    spec = importlib.util.spec_from_file_location("turboquant_local", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_randomized_hadamard_applies_signs_before_transform():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    x = torch.randn(32, 128)
    expected = (x * tq.D_signs[0]) @ tq.H_norm

    y = tq._rotate_forward(x, 0)
    x_roundtrip = tq._rotate_inverse(y, 0)

    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(x_roundtrip, x, atol=1e-5, rtol=1e-5)


def test_quantization_stays_stable_on_hadamard_basis_vectors():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    hadamard = mod._build_hadamard_matrix(128).float() / math.sqrt(128)
    packed_mse, packed_qjl, norms, r_norms = tq.quantize(
        hadamard, 0, use_qjl=False, mse_bits=4, grouped_64=True
    )
    restored = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0,
        use_qjl=False, mse_bits=4, grouped_64=True
    ).float()

    mse = ((hadamard - restored) ** 2).sum(dim=-1).mean().item()

    assert mse < 0.05


def test_mse_only_mode_has_lower_reconstruction_error_than_qjl_mode():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    x = torch.randn(2048, 128)
    x = x / x.norm(dim=-1, keepdim=True)

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(x, 0, use_qjl=True)
    restored_qjl = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0, use_qjl=True
    ).float()

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(x, 0, use_qjl=False)
    restored_mse = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0, use_qjl=False
    ).float()

    mse_qjl = ((x - restored_qjl) ** 2).sum(dim=-1).mean().item()
    mse_mse = ((x - restored_mse) ** 2).sum(dim=-1).mean().item()

    assert mse_mse < mse_qjl


def test_4bit_mse_mode_beats_3bit_mse_mode():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    x = torch.randn(2048, 128)
    x = x / x.norm(dim=-1, keepdim=True)

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(
        x, 0, use_qjl=False, mse_bits=3
    )
    restored_3bit = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0, use_qjl=False, mse_bits=3
    ).float()

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(
        x, 0, use_qjl=False, mse_bits=4, grouped_64=True
    )
    restored_4bit = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0,
        use_qjl=False, mse_bits=4, grouped_64=True
    ).float()

    mse_3bit = ((x - restored_3bit) ** 2).sum(dim=-1).mean().item()
    mse_4bit = ((x - restored_4bit) ** 2).sum(dim=-1).mean().item()

    assert mse_4bit < mse_3bit


def test_grouped_4bit_mode_beats_full_vector_4bit_mode():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)

    x = torch.randn(2048, 128)
    x = x / x.norm(dim=-1, keepdim=True)

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(
        x, 0, use_qjl=False, mse_bits=4, grouped_64=False
    )
    restored_full = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0,
        use_qjl=False, mse_bits=4, grouped_64=False
    ).float()

    packed_mse, packed_qjl, norms, r_norms = tq.quantize(
        x, 0, use_qjl=False, mse_bits=4, grouped_64=True
    )
    restored_grouped = tq.dequantize(
        packed_mse, packed_qjl, norms, r_norms, 0,
        use_qjl=False, mse_bits=4, grouped_64=True
    ).float()

    mse_full = ((x - restored_full) ** 2).sum(dim=-1).mean().item()
    mse_grouped = ((x - restored_grouped) ** 2).sum(dim=-1).mean().item()

    assert mse_grouped < mse_full


def test_recent_bf16_tail_cache_returns_exact_recent_pages_and_evicts_old_ones():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)
    pool = mod.TurboQuantKVPool(
        size=8,
        page_size=1,
        head_num=1,
        head_dim=128,
        layer_num=1,
        device="cpu",
        tq=tq,
        use_qjl=False,
        mse_bits=4,
        grouped_64=True,
        recent_raw_capacity=2,
    )
    layer = SimpleNamespace(layer_id=0)

    x0 = torch.randn(1, 1, 128)
    x1 = torch.randn(1, 1, 128)
    x2 = torch.randn(1, 1, 128)
    v0 = torch.randn(1, 1, 128)
    v1 = torch.randn(1, 1, 128)
    v2 = torch.randn(1, 1, 128)

    pool.set_kv_buffer(layer, torch.tensor([0], dtype=torch.int64), x0, v0)
    pool.set_kv_buffer(layer, torch.tensor([1], dtype=torch.int64), x1, v1)

    k, v = pool.dequantize_pages(0, torch.tensor([0, 1], dtype=torch.int64))
    assert torch.equal(k[0, 0, 0], x0[0, 0].bfloat16())
    assert torch.equal(v[1, 0, 0], v1[0, 0].bfloat16())

    pool.set_kv_buffer(layer, torch.tensor([2], dtype=torch.int64), x2, v2)
    k_old, _ = pool.dequantize_pages(0, torch.tensor([0], dtype=torch.int64))
    assert not torch.equal(k_old[0, 0, 0], x0[0, 0].bfloat16())


def test_recent_bf16_tail_cache_survives_location_moves():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=1, seed=42)
    pool = mod.TurboQuantKVPool(
        size=8,
        page_size=1,
        head_num=1,
        head_dim=128,
        layer_num=1,
        device="cpu",
        tq=tq,
        use_qjl=False,
        mse_bits=4,
        grouped_64=True,
        recent_raw_capacity=2,
    )
    layer = SimpleNamespace(layer_id=0)

    x = torch.randn(1, 1, 128)
    v = torch.randn(1, 1, 128)
    pool.set_kv_buffer(layer, torch.tensor([0], dtype=torch.int64), x, v)
    pool.move_kv_cache(
        torch.tensor([3], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )

    k, vv = pool.dequantize_pages(0, torch.tensor([3], dtype=torch.int64))
    assert torch.equal(k[0, 0, 0], x[0, 0].bfloat16())
    assert torch.equal(vv[0, 0, 0], v[0, 0].bfloat16())


def test_recent_bf16_tail_cache_skips_large_writes():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=2, seed=42)
    pool = mod.TurboQuantKVPool(
        size=16,
        page_size=1,
        head_num=1,
        head_dim=128,
        layer_num=2,
        device="cpu",
        tq=tq,
        use_qjl=False,
        mse_bits=4,
        grouped_64=True,
        recent_raw_capacity=4,
        recent_raw_max_write=2,
    )
    layer0 = SimpleNamespace(layer_id=0)
    layer1 = SimpleNamespace(layer_id=1)

    x = torch.randn(3, 1, 128)
    v = torch.randn(3, 1, 128)
    loc = torch.tensor([0, 1, 2], dtype=torch.int64)

    pool.set_kv_buffer(layer0, loc, x, v)
    assert torch.all(pool.recent_slot_of_loc[loc] == -1)
    pool.set_kv_buffer(layer1, loc, x, v)
    assert torch.all(pool.recent_slot_of_loc[loc] == -1)


def test_recent_bf16_tail_cache_does_not_reassign_slots_on_later_layers():
    torch.manual_seed(0)
    mod = _load_turboquant_module()
    tq = mod.TurboQuant(torch.device("cpu"), num_layers=2, seed=42)
    pool = mod.TurboQuantKVPool(
        size=8,
        page_size=1,
        head_num=1,
        head_dim=128,
        layer_num=2,
        device="cpu",
        tq=tq,
        use_qjl=False,
        mse_bits=4,
        grouped_64=True,
        recent_raw_capacity=2,
        recent_raw_max_write=2,
    )
    layer0 = SimpleNamespace(layer_id=0)
    layer1 = SimpleNamespace(layer_id=1)

    x = torch.randn(2, 1, 128)
    v = torch.randn(2, 1, 128)
    loc = torch.tensor([0, 1], dtype=torch.int64)
    pool.set_kv_buffer(layer0, loc, x, v)
    slots_after_l0 = pool.recent_slot_of_loc.clone()
    pool.set_kv_buffer(layer1, loc, x, v)

    assert torch.equal(pool.recent_slot_of_loc, slots_after_l0)
