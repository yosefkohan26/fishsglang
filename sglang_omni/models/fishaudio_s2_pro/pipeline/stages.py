# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the S2-Pro TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.pipeline.engine_io import (
    apply_tts_result,
    build_sglang_tts_request,
)
from sglang_omni.models.fishaudio_s2_pro.pipeline.state_io import (
    load_state,
    store_state,
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


def _load_audio_decoder(checkpoint: str, device: str):
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
        FishQwen3OmniConfig,
    )
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3OmniForCausalLM,
    )

    checkpoint = _resolve_checkpoint(checkpoint)
    logger.info("Loading S2-Pro model from %s …", checkpoint)
    t0 = time.perf_counter()

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)
    model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint, config=config)
    model = model.to(dtype=torch.bfloat16).eval()

    audio_decoder = model.audio_decoder
    audio_decoder.to(device=device)
    num_codebooks = config.audio_decoder_config.num_codebooks
    codebook_size = config.audio_decoder_config.vocab_size

    del model
    torch.cuda.empty_cache()
    logger.info("Audio decoder loaded in %.2fs", time.perf_counter() - t0)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    return audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint


def _load_codec(checkpoint_dir: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading DAC codec from %s …", codec_path)
    t0 = time.perf_counter()

    import sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    codec = instantiate(cfg)

    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    codec.load_state_dict(state_dict, strict=False, assign=True)
    codec.eval()
    codec.to(device)
    logger.info("DAC codec loaded in %.2fs", time.perf_counter() - t0)
    return codec


def _warmup_codec(codec: Any, *, num_codebooks: int, device: str) -> int:
    """Pre-load mmap'd codec weights and measure upsample ratio.

    Returns:
        total_upsample: audio samples produced per codec token.
    """
    logger.info("Warming up stream codec on %s …", device)
    t0 = time.perf_counter()
    # Decode a short sequence to warm up weights and measure upsample ratio.
    n_tokens = 4
    dummy = torch.zeros(
        1, num_codebooks - 1, n_tokens, dtype=torch.long, device=device
    )
    with torch.no_grad():
        audio = codec.from_indices(dummy)
    total_upsample = audio.shape[-1] // n_tokens
    logger.info(
        "Stream codec warmup done in %.2fs (upsample=%d samples/token)",
        time.perf_counter() - t0,
        total_upsample,
    )
    return total_upsample


def _build_incremental_audio_chunk(
    payload: StagePayload,
    *,
    codec: Any,
    device: str,
    left_context: int,
    total_upsample: int,
) -> dict[str, Any] | None:
    """Decode only a sliding window of recent tokens (O(1) per call).

    Instead of re-decoding ALL accumulated tokens each time (O(N²) total),
    decode ``[left_context + new_tokens]`` and trim the context portion.
    The left context provides the causal codec enough history for smooth
    overlap at the boundary.
    """
    if not isinstance(payload.data, dict):
        return None

    stream_codes = payload.data.get(_STREAM_CODES_KEY)
    if not isinstance(stream_codes, list) or not stream_codes:
        return None

    total_tokens = len(stream_codes)
    prev_vocode_end = int(payload.data.get(_STREAM_LAST_VOCODE_TOKENS_KEY, 0))
    new_tokens = total_tokens - prev_vocode_end
    if new_tokens <= 0:
        return None

    # Sliding window: [left_context | new_tokens]
    ctx = min(left_context, prev_vocode_end)
    window_start = prev_vocode_end - ctx

    window_codes = torch.cat(stream_codes[window_start:total_tokens], dim=1)
    codebook_codes = window_codes[1:].to(device)

    with torch.no_grad():
        audio = codec.from_indices(codebook_codes[None])

    audio_np = audio[0, 0].float().cpu()

    # Trim the context portion — it was only included for decoder continuity
    trim = ctx * total_upsample
    if trim >= audio_np.shape[-1]:
        payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] = total_tokens
        return None
    delta_audio = audio_np[trim:] if trim > 0 else audio_np

    if delta_audio.numel() == 0:
        payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] = total_tokens
        return None

    payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] = total_tokens

    return {
        "audio_data": delta_audio.tolist(),
        "sample_rate": codec.sample_rate,
        "modality": "audio",
    }


def _maybe_build_incremental_audio_chunk(
    payload: StagePayload,
    codes: Any,
    *,
    codec: Any,
    device: str,
    stream_stride: int,
    stream_followup_stride: int,
    left_context: int,
    total_upsample: int,
) -> dict[str, Any] | None:
    if not isinstance(codes, torch.Tensor) or codes.ndim != 2:
        return None
    if not isinstance(payload.data, dict):
        return None

    stream_codes: list[torch.Tensor] = payload.data.setdefault(_STREAM_CODES_KEY, [])
    stream_codes.append(codes.detach().cpu())

    total_tokens = len(stream_codes)
    next_vocode_tokens = int(
        payload.data.get(_STREAM_NEXT_VOCODE_TOKENS_KEY, stream_stride)
    )
    if total_tokens < next_vocode_tokens:
        return None

    chunk = _build_incremental_audio_chunk(
        payload,
        codec=codec,
        device=device,
        left_context=left_context,
        total_upsample=total_upsample,
    )
    payload.data[_STREAM_NEXT_VOCODE_TOKENS_KEY] = total_tokens + stream_followup_stride
    return chunk


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)

    codec = _load_codec(checkpoint_dir, "cpu")

    def _encode_reference_audio(audio_path: str, device: str = "cpu") -> torch.Tensor:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        # s2-pro-alpha codec expects [B, T] (adds channel dim internally)
        audios = audio.squeeze(0).unsqueeze(0).to(device)  # [1, T]
        audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # Speech endpoint sends prompt as a plain string
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        # Build voice-cloning references
        references: list[Reference] | None = None
        raw_refs = inputs.get("references")
        if raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)

                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])

                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        prompt_data = adapter.build_prompt(
            text=text,
            references=references,
            num_codebooks=num_codebooks,
        )

        state = S2ProState(
            input_ids=prompt_data["input_ids"],
            vq_mask_tokens=prompt_data["vq_mask_tokens"],
            vq_parts=prompt_data["vq_parts"],
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 30),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
    stream_stride: int = 1,
    stream_followup_stride: int = 50,
    stream_left_context: int = 25,
    stream_vocoder_device: str | None = None,
) -> EngineExecutor:
    """Factory for the S2-Pro TTS engine stage."""
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.fishaudio_s2_pro.factory import (
        _patch_fish_config_for_sglang,
        create_s2pro_sglang_engine,
    )

    # Default stream vocoder to same GPU as the engine for lowest latency.
    # Override to "cpu" if GPU memory is tight.
    if stream_vocoder_device is None:
        stream_vocoder_device = device

    audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint_dir = (
        _load_audio_decoder(model_path, device)
    )

    # Lazy-init: codec + upsample ratio measured on first streaming request.
    _stream_codec: Any = None
    _total_upsample: int = 0

    def _get_stream_codec() -> tuple[Any, int]:
        nonlocal _stream_codec, _total_upsample
        if _stream_codec is None:
            codec = _load_codec(checkpoint_dir, stream_vocoder_device)
            _total_upsample = _warmup_codec(
                codec, num_codebooks=num_codebooks, device=stream_vocoder_device
            )
            _stream_codec = codec
        return _stream_codec, _total_upsample

    _patch_fish_config_for_sglang(checkpoint_dir)
    server_args = ServerArgs(
        model_path=checkpoint_dir,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        max_running_requests=64,
        disable_cuda_graph=False,
    )

    engine = create_s2pro_sglang_engine(
        server_args=server_args,
        audio_decoder=audio_decoder,
        tokenizer=tokenizer,
        gpu_id=int(device.split(":")[-1]) if ":" in device else 0,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_sglang_tts_request(state, tokenizer, request_id=payload.request_id)

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_tts_result(state, result)
        payload = store_state(payload, state)
        # Attach usage so streaming callers (which skip the vocoder) still
        # see token counts in the final completion message.
        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
        return payload

    async def _stream_builder(
        payload: StagePayload | None, item: Any
    ) -> dict[str, Any] | None:
        """Async stream builder: accumulates codes on GPU, vocodes in thread.

        Fast path (no vocode needed): ~microseconds, no thread spawn.
        Slow path (vocode): offloaded to thread so the engine continues
        generating tokens while audio is decoded in parallel.
        """
        if payload is None or not payload.request.params.get("stream"):
            return None
        if not isinstance(item, torch.Tensor) or item.ndim != 2:
            return None
        if not isinstance(payload.data, dict):
            return None

        # ── Fast path: accumulate codes (stays on original device) ──
        # .clone() is critical: item is a VIEW into the model's persistent
        # _output_codes buffer which the engine overwrites every decode step.
        # Without clone, all accumulated entries alias the same storage and
        # the left-context window would vocode corrupted (latest-only) values.
        # GPU clone of [11, 1] ≈ 1 μs — still 5-10× faster than .cpu().
        stream_codes: list[torch.Tensor] = payload.data.setdefault(
            _STREAM_CODES_KEY, []
        )
        stream_codes.append(item.detach().clone())

        base_offset = int(payload.data.get("_stream_base_offset", 0))
        abs_tokens = base_offset + len(stream_codes)
        next_vocode = int(
            payload.data.get(_STREAM_NEXT_VOCODE_TOKENS_KEY, stream_stride)
        )
        if abs_tokens < next_vocode:
            return None

        # ── Prepare vocode inputs (event loop, fast) ──
        codec, upsample = _get_stream_codec()
        base_offset = int(payload.data.get("_stream_base_offset", 0))
        prev_end = int(payload.data.get(_STREAM_LAST_VOCODE_TOKENS_KEY, 0))
        ctx = min(stream_left_context, prev_end)
        window_start = prev_end - ctx

        # torch.cat on GPU is fast (~microseconds for small tensors)
        local_start = window_start - base_offset
        window_codes = torch.cat(stream_codes[local_start:], dim=1)
        codebook_input = window_codes[1:].to(stream_vocoder_device)

        # Update state BEFORE thread spawn — avoids race with _result_builder
        abs_total = base_offset + len(stream_codes)
        payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] = abs_total
        payload.data[_STREAM_NEXT_VOCODE_TOKENS_KEY] = (
            abs_total + stream_followup_stride
        )

        # Evict codes that will never be in a future context window.
        # Keeps list bounded to ~left_context instead of growing to max_new_tokens.
        keep_from = max(0, len(stream_codes) - stream_left_context)
        if keep_from > 0:
            del stream_codes[:keep_from]
            payload.data["_stream_base_offset"] = base_offset + keep_from

        sample_rate = codec.sample_rate
        trim_samples = ctx * upsample

        # ── Slow path: vocode in thread (engine keeps generating) ──
        def _vocode() -> dict[str, Any] | None:
            with torch.no_grad():
                audio = codec.from_indices(codebook_input[None])
            audio_flat = audio[0, 0].float().cpu()
            if trim_samples >= audio_flat.shape[-1]:
                return None
            delta = (
                audio_flat[trim_samples:].contiguous()
                if trim_samples > 0
                else audio_flat.contiguous()
            )
            if delta.numel() == 0:
                return None
            # Raw bytes: ~8 KB vs .tolist() creating thousands of Python floats
            return {
                "audio_waveform": delta.numpy().tobytes(),
                "audio_waveform_shape": list(delta.shape),
                "audio_waveform_dtype": "float32",
                "sample_rate": sample_rate,
                "modality": "audio",
            }

        return await asyncio.to_thread(_vocode)

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
) -> PreprocessingExecutor:
    """Factory for the vocoder stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)
    codec = _load_codec(checkpoint_dir, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        output_codes = state.output_codes

        codebook_codes = output_codes[1:].to(device)

        with torch.no_grad():
            audio = codec.from_indices(codebook_codes[None])

        audio_np = audio[0, 0].float().cpu()
        state.audio_samples = audio_np
        state.sample_rate = codec.sample_rate
        payload = store_state(payload, state)

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = codec.sample_rate
        payload.data["modality"] = "audio"
        if state.prompt_tokens or state.completion_tokens:
            usage = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
            if state.engine_time_s:
                usage["engine_time_s"] = round(state.engine_time_s, 6)
            payload.data["usage"] = usage
        return payload

    return PreprocessingExecutor(_vocode)
