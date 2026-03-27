# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the Voxtral TTS pipeline.

Two fused stages:
  1. Preprocessing: tokenize text, resolve voice preset
  2. TTS Engine: SGLang AR generation + streaming vocoding

Streaming vocoding is done in-band (async thread) — no separate vocoder stage.
The codec decoder runs on a dedicated CUDA stream for GPU parallelism with
the engine's token generation.

Startup optimizations:
  - All 20 voice presets pre-loaded at init
  - Codec decoder eagerly loaded and warmed up
  - Optional: auto-warmup all voices to populate RadixCache
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.voxtral_tts.io import VoxtralState
from sglang_omni.models.voxtral_tts.pipeline.engine_io import (
    apply_tts_result,
    build_voxtral_tts_request,
)
from sglang_omni.models.voxtral_tts.pipeline.state_io import (
    load_state,
    store_state,
    strip_live_state,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_STREAM_CODES_KEY = "_stream_output_codes"
_STREAM_EMITTED_SAMPLES_KEY = "_stream_emitted_samples"
_STREAM_LAST_VOCODE_TOKENS_KEY = "_stream_last_vocode_tokens"
_STREAM_NEXT_VOCODE_TOKENS_KEY = "_stream_next_vocode_tokens"


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download
    return snapshot_download(checkpoint)


# ---------------------------------------------------------------------------
# Voice cache (shared across requests)
# ---------------------------------------------------------------------------

_voice_cache: dict[str, dict[str, Any]] = {}
_voice_cache_lock = threading.Lock()


def get_cached_voice(voice_id: str) -> dict | None:
    with _voice_cache_lock:
        return _voice_cache.get(voice_id)


def put_cached_voice(voice_id: str, embedding: torch.Tensor) -> None:
    with _voice_cache_lock:
        _voice_cache[voice_id] = {"embedding": embedding.cpu()}


def delete_cached_voice(voice_id: str) -> bool:
    with _voice_cache_lock:
        return _voice_cache.pop(voice_id, None) is not None


def list_cached_voices() -> list[str]:
    with _voice_cache_lock:
        return list(_voice_cache.keys())


# ---------------------------------------------------------------------------
# Preprocessing executor
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    """Build the preprocessing executor: tokenize text, resolve voice."""
    checkpoint_dir = _resolve_checkpoint(model_path)

    from sglang_omni.models.voxtral_tts.tokenizer import VoxtralTokenizer
    tok = VoxtralTokenizer(checkpoint_dir)

    # Pre-populate voice cache from tokenizer's presets
    for voice_name in tok.voice_names:
        preset = tok.get_voice(voice_name)
        if preset is not None:
            put_cached_voice(voice_name, preset.embedding)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        voice = inputs.get("voice_id") or inputs.get("voice") or params.get("voice")

        # Build prompt
        prompt_data = tok.build_prompt(text=text, voice=voice)

        state = VoxtralState(
            input_ids=prompt_data["input_ids"],
            voice_embedding=prompt_data["voice_embedding"],
            voice_name=voice,
            max_new_tokens=params.get("max_new_tokens", 2048),
            temperature=params.get("temperature", 0.0),
            top_p=params.get("top_p", 1.0),
            top_k=params.get("top_k", -1),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


# ---------------------------------------------------------------------------
# TTS Engine executor (with streaming vocoding)
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    batch_window_ms: float = 1.0,
    # Streaming vocode params
    stream_stride: int = 3,               # First vocode after 3 frames (~240ms audio)
    stream_followup_stride: int = 10,     # Subsequent vocode every 10 frames
    stream_left_context: int = 5,         # Overlap frames for codec continuity
    stream_vocoder_device: str | None = None,
) -> EngineExecutor:
    """Factory for the Voxtral TTS engine stage."""
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.voxtral_tts.factory import (
        _create_hf_config,
        _load_acoustic_transformer,
        _load_audio_tokenizer,
        _warmup_audio_tokenizer,
        create_voxtral_sglang_engine,
    )

    if stream_vocoder_device is None:
        stream_vocoder_device = device

    checkpoint_dir = _resolve_checkpoint(model_path)

    # Load params.json for audio config
    with open(os.path.join(checkpoint_dir, "params.json"), "r") as f:
        params = json.load(f)

    audio_model_args = params.get("multimodal", {}).get("audio_model_args", {})

    # Generate HF config.json if needed
    _create_hf_config(checkpoint_dir)

    # Load acoustic transformer (flow matching head)
    acoustic_transformer = _load_acoustic_transformer(
        checkpoint_dir, audio_model_args, device
    )

    # Load and warm up codec decoder
    logger.info("Loading audio tokenizer for streaming vocode …")
    _stream_codec = _load_audio_tokenizer(checkpoint_dir, params, stream_vocoder_device)
    _samples_per_frame = _warmup_audio_tokenizer(_stream_codec, stream_vocoder_device)
    _vocode_stream = torch.cuda.Stream(device=stream_vocoder_device) if "cuda" in stream_vocoder_device else None

    def _get_stream_codec():
        return _stream_codec, _samples_per_frame

    # Build tokenizer for request building
    from sglang_omni.models.voxtral_tts.tokenizer import VoxtralTokenizer
    voxtral_tokenizer = VoxtralTokenizer(checkpoint_dir)

    # Build SGLang ServerArgs
    server_args = ServerArgs(
        model_path=checkpoint_dir,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        max_prefill_tokens=32768,
        max_running_requests=64,
        disable_cuda_graph=False,
        context_length=params.get("max_seq_len", 65536),
    )

    # Create engine
    engine = create_voxtral_sglang_engine(
        server_args=server_args,
        acoustic_transformer=acoustic_transformer,
        tokenizer=voxtral_tokenizer,
        gpu_id=int(device.split(":")[-1]) if ":" in device else 0,
        max_new_tokens=max_new_tokens,
        batch_window_ms=batch_window_ms,
    )

    # ---------------------------------------------------------------
    # Request/result builders
    # ---------------------------------------------------------------

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_voxtral_tts_request(
            state, voxtral_tokenizer, request_id=payload.request_id
        )

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_tts_result(state, result)
        # Build a minimal serializable result — don't store_state (has tensors).
        # For streaming TTS the audio was already sent via chunks; the
        # completion message only needs usage info.
        data: dict = {}
        if state.prompt_tokens or state.completion_tokens:
            data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
        data["sample_rate"] = state.sample_rate
        data["modality"] = "audio"
        payload.data = data
        return payload

    # ---------------------------------------------------------------
    # Streaming vocode (async thread + dedicated CUDA stream)
    # ---------------------------------------------------------------

    async def _stream_builder(
        payload: StagePayload | None, item: Any
    ) -> dict[str, Any] | None:
        if payload is None or not payload.request.params.get("stream"):
            return None
        if not isinstance(item, torch.Tensor):
            return None
        if not isinstance(payload.data, dict):
            return None

        # Accumulate codes (clone to avoid aliasing persistent buffer)
        stream_codes: list[torch.Tensor] = payload.data.setdefault(_STREAM_CODES_KEY, [])
        stream_codes.append(item.detach().clone())

        base_offset = int(payload.data.get("_stream_base_offset", 0))
        abs_tokens = base_offset + len(stream_codes)
        next_vocode = int(payload.data.get(_STREAM_NEXT_VOCODE_TOKENS_KEY, stream_stride))
        if abs_tokens < next_vocode:
            return None

        # Prepare vocode inputs
        codec, samples_per_frame = _get_stream_codec()
        prev_end = int(payload.data.get(_STREAM_LAST_VOCODE_TOKENS_KEY, 0))
        ctx = min(stream_left_context, prev_end)
        window_start = prev_end - ctx

        local_start = window_start - base_offset
        # Stack codes: each is [num_codebooks], stack → [num_codebooks, N_frames]
        window_codes = torch.stack(stream_codes[local_start:], dim=-1)
        codec_input = window_codes.unsqueeze(0).to(stream_vocoder_device)  # [1, K, T]

        # Update state
        abs_total = base_offset + len(stream_codes)
        payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] = abs_total
        payload.data[_STREAM_NEXT_VOCODE_TOKENS_KEY] = abs_total + stream_followup_stride

        # Evict old codes to bound memory
        keep_from = max(0, len(stream_codes) - stream_left_context)
        if keep_from > 0:
            del stream_codes[:keep_from]
            payload.data["_stream_base_offset"] = base_offset + keep_from

        sample_rate = codec.sampling_rate
        trim_samples = ctx * samples_per_frame

        # Vocode in thread (with dedicated CUDA stream for GPU parallelism)
        def _vocode() -> dict[str, Any] | None:
            if _vocode_stream is not None:
                with torch.cuda.stream(_vocode_stream):
                    with torch.no_grad():
                        audio = codec.decode(codec_input)
                _vocode_stream.synchronize()
            else:
                with torch.no_grad():
                    audio = codec.decode(codec_input)

            audio_flat = audio[0, 0].float().cpu()
            if trim_samples >= audio_flat.shape[-1]:
                return None
            delta = audio_flat[trim_samples:].contiguous() if trim_samples > 0 else audio_flat.contiguous()
            if delta.numel() == 0:
                return None
            return {
                "audio_waveform": delta.numpy().tobytes(),
                "audio_waveform_shape": list(delta.shape),
                "audio_waveform_dtype": "float32",
                "sample_rate": sample_rate,
                "modality": "audio",
            }

        return await asyncio.to_thread(_vocode)

    def _flush_stream(payload: StagePayload | None) -> dict[str, Any] | None:
        """Flush remaining un-vocoded frames at stream end."""
        if payload is None or not isinstance(payload.data, dict):
            return None
        stream_codes = payload.data.get(_STREAM_CODES_KEY)
        if not stream_codes:
            return None

        base_offset = int(payload.data.get("_stream_base_offset", 0))
        abs_total = base_offset + len(stream_codes)
        prev_end = int(payload.data.get(_STREAM_LAST_VOCODE_TOKENS_KEY, 0))
        if abs_total <= prev_end:
            return None

        codec, samples_per_frame = _get_stream_codec()
        ctx = min(stream_left_context, prev_end)
        window_start = prev_end - ctx
        local_start = max(window_start - base_offset, 0)
        if local_start >= len(stream_codes):
            return None

        window_codes = torch.stack(stream_codes[local_start:], dim=-1)
        codec_input = window_codes.unsqueeze(0).to(stream_vocoder_device)
        if codec_input.shape[-1] == 0:
            return None

        trim_samples = max(ctx, 0) * samples_per_frame
        sample_rate = codec.sampling_rate

        if _vocode_stream is not None:
            with torch.cuda.stream(_vocode_stream):
                with torch.no_grad():
                    audio = codec.decode(codec_input)
            _vocode_stream.synchronize()
        else:
            with torch.no_grad():
                audio = codec.decode(codec_input)

        audio_flat = audio[0, 0].float().cpu()
        if trim_samples >= audio_flat.shape[-1]:
            return None
        delta = audio_flat[trim_samples:].contiguous() if trim_samples > 0 else audio_flat.contiguous()
        if delta.numel() == 0:
            return None
        return {
            "audio_waveform": delta.numpy().tobytes(),
            "audio_waveform_shape": list(delta.shape),
            "audio_waveform_dtype": "float32",
            "sample_rate": sample_rate,
            "modality": "audio",
        }

    _stream_builder.flush = _flush_stream  # type: ignore[attr-defined]

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )
