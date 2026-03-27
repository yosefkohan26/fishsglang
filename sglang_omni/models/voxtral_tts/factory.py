# SPDX-License-Identifier: Apache-2.0
"""Factory function for creating Voxtral TTS engine on SGLang backend.

Creates an OmniEngine with:
  - SGLang ModelWorker running VoxtralTTSSGLangModel (26-layer Mistral + acoustic head)
  - Custom iteration controller for TTS EOS detection
  - Custom model runner with voice embedding injection + acoustic output read
  - CUDA graph support for both LLM decode and acoustic transformer
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import PretrainedConfig

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _create_hf_config(model_path: str) -> PretrainedConfig:
    """Create an HF-compatible config from Voxtral's params.json.

    SGLang's ModelConfig.from_server_args() needs a config that provides:
    - architectures, vocab_size, hidden_size, num_hidden_layers, etc.

    We write a temporary config.json in the model directory so AutoConfig
    can discover it. This is only done once.
    """
    model_dir = Path(model_path)
    config_json_path = model_dir / "config.json"

    if config_json_path.exists():
        return PretrainedConfig.from_pretrained(str(model_dir))

    params_path = model_dir / "params.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Build HF-format config. Use model_type "mistral" so AutoConfig
    # recognizes it without custom registration. Our custom architecture
    # class name is set in "architectures" and registered in SGLang's
    # ModelRegistry separately.
    hf_config = {
        "architectures": ["VoxtralTTSSGLangModel"],
        "model_type": "mistral",
        "vocab_size": params["vocab_size"],
        "hidden_size": params["dim"],
        "intermediate_size": params["hidden_dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "head_dim": params["head_dim"],
        "max_position_embeddings": params.get("max_position_embeddings", 128000),
        "rms_norm_eps": params["norm_eps"],
        "rope_theta": params["rope_theta"],
        "tie_word_embeddings": params.get("tied_embeddings", True),
        "torch_dtype": "bfloat16",
        "max_seq_len": params.get("max_seq_len", 65536),
        # Audio config (stored for later use by pipeline)
        "audio_config": params.get("multimodal", {}),
    }

    with open(config_json_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    logger.info("Generated config.json at %s", config_json_path)
    return PretrainedConfig.from_pretrained(str(model_dir))


def _load_acoustic_transformer(
    model_path: str, audio_model_args: dict, device: str
) -> torch.nn.Module:
    """Load the acoustic transformer weights from checkpoint."""
    from safetensors import safe_open

    from sglang_omni.models.voxtral_tts.acoustic_transformer import (
        FlowMatchingAcousticTransformer,
    )

    t0 = time.perf_counter()
    logger.info("Loading acoustic transformer from %s …", model_path)

    acoustic = FlowMatchingAcousticTransformer(audio_model_args)

    # Load weights
    safetensors_path = os.path.join(model_path, "consolidated.safetensors")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        acoustic_weights = {}
        for key in f.keys():
            if key.startswith("acoustic_transformer."):
                stripped = key[len("acoustic_transformer."):]
                acoustic_weights[stripped] = f.get_tensor(key)

    loaded = acoustic.load_weights(acoustic_weights)
    logger.info(
        "Acoustic transformer loaded: %d weights in %.2fs",
        len(loaded), time.perf_counter() - t0,
    )

    acoustic = acoustic.to(device=device, dtype=torch.bfloat16).eval()
    return acoustic


def _load_audio_tokenizer(model_path: str, params: dict, device: str) -> torch.nn.Module:
    """Load the audio tokenizer (codec decoder) weights."""
    from safetensors import safe_open

    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralAudioTokenizer

    t0 = time.perf_counter()
    logger.info("Loading audio tokenizer from %s …", model_path)

    tokenizer = VoxtralAudioTokenizer.from_params(params)

    safetensors_path = os.path.join(model_path, "consolidated.safetensors")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        loaded_count = 0
        for key in f.keys():
            if key.startswith("audio_tokenizer."):
                stripped = key[len("audio_tokenizer."):]
                tokenizer.load_weight((stripped, f.get_tensor(key)))
                loaded_count += 1

    logger.info(
        "Audio tokenizer loaded: %d weights in %.2fs",
        loaded_count, time.perf_counter() - t0,
    )

    tokenizer = tokenizer.to(device=device, dtype=torch.bfloat16).eval()
    return tokenizer


def _warmup_audio_tokenizer(codec: torch.nn.Module, device: str) -> int:
    """Warm up codec and measure samples-per-frame."""
    logger.info("Warming up audio tokenizer on %s …", device)
    t0 = time.perf_counter()

    n_frames = 4
    num_codebooks = codec.num_codebooks
    dummy_codes = torch.randint(0, 10, (1, num_codebooks, n_frames), device=device)
    with torch.no_grad():
        audio = codec.decode(dummy_codes)
    samples_per_frame = audio.shape[-1] // n_frames

    logger.info(
        "Audio tokenizer warmup done in %.2fs (upsample=%d samples/frame, sr=%d)",
        time.perf_counter() - t0, samples_per_frame, codec.sampling_rate,
    )
    return samples_per_frame


def create_voxtral_sglang_engine(
    server_args: Any,
    acoustic_transformer: torch.nn.Module,
    tokenizer: Any = None,
    *,
    gpu_id: int = 0,
    max_new_tokens: int = 2048,
    batch_window_ms: float = 1.0,
    audio_token_id: int = 24,
    eos_token_id: int = 2,
) -> OmniEngine:
    """Create a Voxtral TTS engine backed by SGLang.

    This is the core engine factory analogous to create_s2pro_sglang_engine.
    """
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )
    from sglang_omni.engines.ar.sglang_backend.scheduler.cache import create_tree_cache
    from sglang_omni.engines.ar.sglang_backend.scheduler.decode import DecodeManager
    from sglang_omni.engines.ar.sglang_backend.scheduler.prefill import PrefillManager
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangBatchPlanner

    from .runtime.voxtral_sglang_ar import (
        VoxtralSGLangIterationController,
        VoxtralSGLangModelRunner,
        VoxtralSGLangResourceManager,
    )

    # Register our model class in SGLang's registry
    from sglang.srt.models.registry import ModelRegistry
    from .sglang_model import VoxtralTTSSGLangModel
    ModelRegistry.models["VoxtralTTSSGLangModel"] = VoxtralTTSSGLangModel

    # Defer CUDA graph: setup_acoustic_decode must run first
    want_cuda_graph = not server_args.disable_cuda_graph
    server_args.disable_cuda_graph = True

    model_worker = ModelWorker(
        config=ModelWorkerConfig(),
        server_args=server_args,
        gpu_id=gpu_id,
    )

    # Restore setting for later
    server_args.disable_cuda_graph = not want_cuda_graph

    # Attach acoustic transformer to the text model
    text_model = model_worker.model_runner.model
    max_bs = server_args.max_running_requests
    text_model.setup_acoustic_decode(
        acoustic_transformer,
        audio_token_id=audio_token_id,
        eos_token_id=eos_token_id,
        max_batch_size=max_bs,
    )

    # Now capture CUDA graphs with acoustic decode in the graph
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    # Build SGLang scheduling components
    req_to_token_pool, token_to_kv_pool_allocator = model_worker.get_memory_pool()

    tree_cache = create_tree_cache(
        server_args,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        server_args.page_size,
    )

    prefill_mgr = PrefillManager(
        page_size=server_args.page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
        max_prefill_tokens=server_args.max_prefill_tokens,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_worker.model_config,
        enable_overlap=False,
    )
    decode_mgr = DecodeManager(
        server_args=server_args,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        on_retract=lambda req: prefill_mgr.add_one_request(req),
    )

    batch_planner = SGLangBatchPlanner(prefill_mgr, decode_mgr, server_args)
    resource_mgr = VoxtralSGLangResourceManager(
        token_to_kv_pool_allocator, req_to_token_pool, tree_cache
    )
    iteration_ctrl = VoxtralSGLangIterationController(
        tree_cache=tree_cache,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
    )

    def _stream_adapter(request, output):
        step_out = output.data
        if step_out is None:
            return None
        return step_out.codes

    scheduler = Scheduler(
        batch_planner=batch_planner,
        resource_manager=resource_mgr,
        iteration_controller=iteration_ctrl,
        stream_adapter=_stream_adapter,
    )

    model_runner = VoxtralSGLangModelRunner(
        model_worker, batch_planner, audio_token_id=audio_token_id,
    )

    return OmniEngine(
        scheduler=scheduler,
        model_runner=model_runner,
        batch_window_ms=batch_window_ms,
    )
