# SPDX-License-Identifier: Apache-2.0
"""Direct weight loading utilities for split model components."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers.utils.hub import cached_file


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        # Default to BF16 to avoid unintentionally loading FP32 weights.
        return torch.bfloat16
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[key]


@lru_cache(maxsize=4)
def resolve_model_path(model_path: str, *, local_files_only: bool = False) -> Path:
    """Resolve a model_path to a local path, downloading if needed."""
    path = Path(model_path)
    if path.exists():
        return path
    if local_files_only:
        config_path = cached_file(model_path, "config.json", local_files_only=True)
        return Path(config_path).parent
    return Path(snapshot_download(model_path, local_files_only=False))


def _load_bin_shard(path: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def _read_safetensors_keys(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    if not keys:
        return state_dict
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in keys:
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _dequantize_int4(qweight: torch.Tensor, scales: torch.Tensor,
                     qzeros: torch.Tensor, group_size: int = 128,
                     awq_scales: torch.Tensor | None = None) -> torch.Tensor:
    """Dequantize INT4 packed weights back to FP16."""
    out_features = qweight.shape[0]
    packed_in = qweight.shape[1]
    in_features = packed_in * 8

    w_unpacked = torch.zeros(out_features, in_features, dtype=torch.int32)
    for i in range(8):
        w_unpacked[:, i::8] = (qweight >> (i * 4)) & 0xF

    n_groups = qzeros.shape[0]
    packed_out = qzeros.shape[1]
    zp_unpacked = torch.zeros(n_groups, packed_out * 8, dtype=torch.int32)
    for i in range(8):
        zp_unpacked[:, i::8] = (qzeros >> (i * 4)) & 0xF
    zp_unpacked = zp_unpacked[:, :out_features]

    w_grouped = w_unpacked.reshape(out_features, n_groups, group_size).float()
    zp = zp_unpacked.transpose(0, 1).unsqueeze(-1).float()
    sc = scales.transpose(0, 1).unsqueeze(-1).float()

    w_dequant = (w_grouped - zp) * sc
    w_dequant = w_dequant.reshape(out_features, in_features)

    if awq_scales is not None:
        w_dequant = w_dequant / awq_scales.float().unsqueeze(0)

    return w_dequant.bfloat16()


def _load_safetensors_sharded(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return {}

    with index_file.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]

    # Detect quantization from metadata or key patterns
    metadata = index_data.get("metadata", {})
    is_quantized = metadata.get("quantization") in ("gptq-int4", "awq-int4")
    if not is_quantized:
        is_quantized = any(k.endswith(".qweight") for k in weight_map)
    group_size = int(metadata.get("group_size", 128))

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}

    if not is_quantized:
        for shard, keys in shards.items():
            shard_path = model_path / shard
            shard_weights = _read_safetensors_keys(shard_path, keys)
            for key, tensor in shard_weights.items():
                new_key = key[len(prefix):]
                state_dict[new_key] = tensor
        return state_dict

    # Quantized path: dequantize .qweight/.scales/.qzeros → .weight
    import logging
    logger = logging.getLogger(__name__)
    quant_type = metadata.get("quantization", "int4")
    bits = int(metadata.get("bits", 4))
    logger.info("Detected %s quantized weights (bits=%d, group_size=%d), dequantizing…",
                quant_type, bits, group_size)

    for shard, keys in shards.items():
        shard_path = model_path / shard
        shard_weights = _read_safetensors_keys(shard_path, keys)

        # Group quantized weight sets
        qweight_bases = {
            k[len(prefix):-len(".qweight")]
            for k in keys if k.endswith(".qweight")
        }
        loaded_keys: set[str] = set()

        for base in qweight_bases:
            full_base = prefix + base
            qw = shard_weights.get(full_base + ".qweight")
            sc = shard_weights.get(full_base + ".scales")
            if qw is None or sc is None:
                continue

            qz = shard_weights.get(full_base + ".qzeros")

            if qz is not None:
                # INT4 group quantization (asymmetric, packed)
                awq_sc = shard_weights.get(full_base + ".awq_scales")
                w = _dequantize_int4(qw, sc, qz, group_size, awq_sc)
                loaded_keys.add(full_base + ".qzeros")
                if awq_sc is not None:
                    loaded_keys.add(full_base + ".awq_scales")
            else:
                # INT8 per-channel symmetric quantization
                w = (qw.float() * sc.float().unsqueeze(1)).bfloat16()

            # base already ends with ".weight" (e.g. "layers.0.attention.wqkv.weight")
            state_dict[base] = w

            loaded_keys.add(full_base + ".qweight")
            loaded_keys.add(full_base + ".scales")

        # Load non-quantized weights
        for key in keys:
            if key not in loaded_keys:
                new_key = key[len(prefix):]
                state_dict[new_key] = shard_weights[key]

    logger.info("Dequantized %d weight matrices, %d total tensors",
                len([k for k in state_dict if '.weight' in k]), len(state_dict))
    return state_dict


def _load_safetensors_single(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    single = model_path / "model.safetensors"
    if not single.exists():
        return {}

    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(single), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = f.get_tensor(key)
    return state_dict


def _load_bin_sharded(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    index_file = model_path / "pytorch_model.bin.index.json"
    if not index_file.exists():
        return {}

    with index_file.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        shard_weights = _load_bin_shard(str(model_path / shard))
        for key in keys:
            new_key = key[len(prefix) :]
            state_dict[new_key] = shard_weights[key]
    return state_dict


def _load_bin_single(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    single = model_path / "pytorch_model.bin"
    if not single.exists():
        return {}

    all_weights = _load_bin_shard(str(single))
    return {k[len(prefix) :]: v for k, v in all_weights.items() if k.startswith(prefix)}


def _normalize_prefixes(prefixes: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefixes)


def _load_weights_from_resolved_path(
    model_path: Path, prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Return the first matching state_dict, or {} if no prefix matches.

    An empty dict here means "no weights found for the requested prefixes" for
    this resolved path. Exceptions from missing/corrupted shard reads are
    allowed to propagate so the caller can decide whether to retry with a
    refreshed remote snapshot.
    """
    for prefix_item in prefixes:
        state_dict = _load_safetensors_sharded(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_safetensors_single(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_bin_sharded(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_bin_single(model_path, prefix_item)
        if state_dict:
            return state_dict
    return {}


def _should_retry_remote_weight_load(
    *,
    model_path: str,
    local_files_only: bool,
) -> bool:
    return not local_files_only and not Path(model_path).exists()


def load_weights_by_prefix(
    model_path: str,
    *,
    prefix: str | tuple[str, ...] | list[str],
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load weights matching one of the prefixes, stripping the matched prefix."""
    resolved_model_path = resolve_model_path(
        model_path, local_files_only=local_files_only
    )
    prefixes = _normalize_prefixes(prefix)
    should_retry_remote_load = _should_retry_remote_weight_load(
        model_path=model_path,
        local_files_only=local_files_only,
    )

    try:
        state_dict = _load_weights_from_resolved_path(resolved_model_path, prefixes)
    except Exception:
        if not should_retry_remote_load:
            raise
        state_dict = {}

    if state_dict:
        return state_dict

    # A poisoned/partial HF cache can still yield a snapshot path that is missing
    # weight index files or shards. Refresh once before failing for remote models.
    if should_retry_remote_load:
        resolve_model_path.cache_clear()
        resolved_model_path = Path(
            snapshot_download(model_path, local_files_only=False, force_download=True)
        )
        state_dict = _load_weights_from_resolved_path(resolved_model_path, prefixes)
        if state_dict:
            return state_dict

    raise FileNotFoundError(
        f"No weights found for prefixes {list(prefixes)!r} under {resolved_model_path}"
    )


def load_module(
    module: nn.Module,
    model_path: str,
    *,
    prefix: str | tuple[str, ...] | list[str],
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
    local_files_only: bool = False,
) -> nn.Module:
    """Load weights into module by prefix, optionally move to device."""
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=prefix,
        local_files_only=local_files_only,
    )
    # Prefer assign=True to avoid expensive in-place tensor copies during load.
    try:
        module.load_state_dict(state_dict, strict=strict, assign=True)
    except (TypeError, RuntimeError):
        module.load_state_dict(state_dict, strict=strict)
    module.eval()
    if device is not None or dtype is not None:
        if device is not None and dtype is not None:
            module = module.to(device=device, dtype=dtype)
        elif device is not None:
            module = module.to(device=device)
        else:
            module = module.to(dtype=dtype)
    return module


# picked from sglang.srt.model_loader.weight_utils.py
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
