# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the S2-Pro TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
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
    strip_live_state,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

_STREAM_CODES_KEY = "_stream_output_codes"
_STREAM_EMITTED_SAMPLES_KEY = "_stream_emitted_samples"
_STREAM_LAST_VOCODE_TOKENS_KEY = "_stream_last_vocode_tokens"
_STREAM_NEXT_VOCODE_TOKENS_KEY = "_stream_next_vocode_tokens"

# ---------------------------------------------------------------------------
# Voice reference cache — keyed by voice_id, stores encoded VQ codes so
# subsequent requests skip the expensive codec.encode() step entirely.
# Thread-safe: _preprocess runs in a thread pool, API routes run on the
# event loop.
# ---------------------------------------------------------------------------


@dataclass
class _CachedVoice:
    vq_codes: torch.Tensor  # [num_codebooks, T] on CPU
    ref_text: str
    # Pre-built prompt prefix (system + user role header) so build_prompt()
    # only needs to tokenize the short user text, not re-encode ~700 VQ codes.
    prompt_prefix: dict | None = None  # {input_ids, vq_mask_tokens, vq_parts} for prefix


_voice_cache: dict[str, _CachedVoice] = {}
_voice_cache_lock = threading.Lock()


def get_cached_voice(voice_id: str) -> _CachedVoice | None:
    with _voice_cache_lock:
        return _voice_cache.get(voice_id)


def put_cached_voice(voice_id: str, vq_codes: torch.Tensor, ref_text: str) -> None:
    with _voice_cache_lock:
        _voice_cache[voice_id] = _CachedVoice(vq_codes=vq_codes.cpu(), ref_text=ref_text)


def delete_cached_voice(voice_id: str) -> bool:
    with _voice_cache_lock:
        return _voice_cache.pop(voice_id, None) is not None


def list_cached_voices() -> list[str]:
    with _voice_cache_lock:
        return list(_voice_cache.keys())


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


def _build_prompt_from_cached_prefix(
    adapter: Any,
    user_text: str,
    prefix: dict,
    num_codebooks: int,
) -> dict:
    """Build a full prompt by concatenating cached prefix with user text suffix.

    The prefix contains the system message (voice reference VQ codes + text) and
    the user role header. We only need to tokenize the short user text and the
    assistant voice marker, then concatenate with the cached prefix tokens.

    This avoids re-tokenizing ~700 VQ semantic tokens on every request.
    """
    # Build just the user-text suffix (no references)
    suffix_data = adapter.build_prompt(
        text=user_text,
        references=None,
        num_codebooks=num_codebooks,
    )

    # Concatenate prefix + suffix
    prefix_ids = prefix["input_ids"]
    suffix_ids = suffix_data["input_ids"]

    if isinstance(prefix_ids, torch.Tensor) and isinstance(suffix_ids, torch.Tensor):
        combined_ids = torch.cat([prefix_ids, suffix_ids])
    else:
        combined_ids = list(prefix_ids) + list(suffix_ids)

    prefix_mask = prefix["vq_mask_tokens"]
    suffix_mask = suffix_data["vq_mask_tokens"]
    if isinstance(prefix_mask, torch.Tensor) and isinstance(suffix_mask, torch.Tensor):
        combined_mask = torch.cat([prefix_mask, suffix_mask])
    else:
        combined_mask = list(prefix_mask) + list(suffix_mask)

    combined_vq = list(prefix.get("vq_parts") or []) + list(suffix_data.get("vq_parts") or [])

    return {
        "input_ids": combined_ids,
        "vq_mask_tokens": combined_mask,
        "vq_parts": combined_vq,
    }


def _cache_prompt_prefix(
    voice_id: str,
    user_text: str,
    full_prompt_data: dict,
    adapter: Any,
    num_codebooks: int,
) -> None:
    """Extract and cache the voice prefix from a full prompt.

    The prefix = full_prompt - user_text_suffix. We build the suffix separately,
    then subtract its length from the full prompt to get the prefix tokens.
    """
    try:
        suffix_data = adapter.build_prompt(
            text=user_text,
            references=None,
            num_codebooks=num_codebooks,
        )
        suffix_len = len(suffix_data["input_ids"])
        full_ids = full_prompt_data["input_ids"]
        full_mask = full_prompt_data["vq_mask_tokens"]
        prefix_len = len(full_ids) - suffix_len

        if prefix_len <= 0:
            return

        if isinstance(full_ids, torch.Tensor):
            prefix_ids = full_ids[:prefix_len].clone()
            prefix_mask = full_mask[:prefix_len].clone()
        else:
            prefix_ids = list(full_ids)[:prefix_len]
            prefix_mask = list(full_mask)[:prefix_len]

        # VQ parts belong entirely to the prefix (voice reference)
        prefix_vq = list(full_prompt_data.get("vq_parts") or [])

        prefix_dict = {
            "input_ids": prefix_ids,
            "vq_mask_tokens": prefix_mask,
            "vq_parts": prefix_vq,
        }

        with _voice_cache_lock:
            cached = _voice_cache.get(voice_id)
            if cached is not None:
                cached.prompt_prefix = prefix_dict
                logger.info(
                    "Cached prompt prefix for voice_id=%s (%d prefix tokens)",
                    voice_id, prefix_len,
                )
    except Exception as e:
        logger.warning("Failed to cache prompt prefix for %s: %s", voice_id, e)


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
        t0 = time.perf_counter()
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # Speech endpoint sends prompt as a plain string
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)
        voice_id = inputs.get("voice_id")

        # Build voice-cloning references
        references: list[Reference] | None = None
        raw_refs = inputs.get("references")

        # Check voice cache first — skip encoding entirely if cached
        prompt_data = None
        cache_hit = False
        if voice_id and not raw_refs:
            cached = get_cached_voice(voice_id)
            if cached is not None:
                cache_hit = True
                references = [
                    Reference(audio_bytes=b"", text=cached.ref_text, vq_codes=cached.vq_codes)
                ]

        t_cache = time.perf_counter()

        if references is None and raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)

                if vq_codes is None and ref_data.get("audio_path"):
                    t_enc0 = time.perf_counter()
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])
                    t_enc1 = time.perf_counter()
                    logger.info("[PROFILE] ref_audio_encode: %.1fms", (t_enc1 - t_enc0) * 1000)

                ref_text = ref_data.get("text", "")
                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_text,
                        vq_codes=vq_codes,
                    )
                )

                # Populate cache on first use when voice_id is provided
                if voice_id and vq_codes is not None:
                    put_cached_voice(voice_id, vq_codes, ref_text)
                    logger.info("Cached voice_id=%s (%d tokens)", voice_id, vq_codes.shape[-1])

        t_refs = time.perf_counter()

        if prompt_data is None:
            prompt_data = adapter.build_prompt(
                text=text,
                references=references,
                num_codebooks=num_codebooks,
            )
            # Cache prompt prefix for this voice on first full build
            if voice_id and references and not cache_hit:
                _cache_prompt_prefix(voice_id, text, prompt_data, adapter, num_codebooks)

        t_prompt = time.perf_counter()

        state = S2ProState(
            input_ids=prompt_data["input_ids"],
            vq_mask_tokens=prompt_data["vq_mask_tokens"],
            vq_parts=prompt_data["vq_parts"],
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 2048),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 30),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )

        return store_state(payload, state)

    # ------------------------------------------------------------------
    # Pre-load voices from voices/ directory at startup
    #
    # Each voice folder can contain:
    #   audio.wav  + transcript.txt   → encoded on first run, cached as codes.pt
    #   codes.pt   + transcript.txt   → loaded instantly (no codec needed)
    # ------------------------------------------------------------------
    # Look for voices/ relative to the working directory (project root)
    voices_dir = os.path.join(os.getcwd(), "voices")
    if os.path.isdir(voices_dir):
        logger.info("Pre-loading voices from %s …", voices_dir)
        for voice_name in sorted(os.listdir(voices_dir)):
            voice_path = os.path.join(voices_dir, voice_name)
            if not os.path.isdir(voice_path):
                continue

            transcript_file = os.path.join(voice_path, "transcript.txt")
            ref_text = ""
            if os.path.exists(transcript_file):
                with open(transcript_file, "r") as f:
                    ref_text = f.read().strip()

            codes_file = os.path.join(voice_path, "codes.pt")
            t_start = time.perf_counter()

            try:
                if os.path.exists(codes_file):
                    # Fast path: load pre-encoded VQ codes (~1ms)
                    vq_codes = torch.load(codes_file, map_location="cpu", weights_only=True)
                    put_cached_voice(voice_name, vq_codes, ref_text)
                    # Build and cache prompt prefix so first request is fast
                    ref = Reference(audio_bytes=b"", text=ref_text, vq_codes=vq_codes)
                    full_prompt = adapter.build_prompt(
                        text=".", references=[ref], num_codebooks=10,
                    )
                    _cache_prompt_prefix(voice_name, ".", full_prompt, adapter, 10)
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "  Loaded voice '%s' from codes.pt in %.0fms (%d VQ frames, prefix cached)",
                        voice_name, elapsed * 1000, vq_codes.shape[-1],
                    )
                else:
                    # Slow path: encode audio → save codes.pt for next time
                    audio_file = None
                    for ext in ("wav", "mp3", "flac", "ogg"):
                        candidate = os.path.join(voice_path, f"audio.{ext}")
                        if os.path.exists(candidate):
                            audio_file = candidate
                            break
                    if audio_file is None:
                        continue
                    vq_codes = _encode_reference_audio(audio_file)
                    put_cached_voice(voice_name, vq_codes, ref_text)
                    torch.save(vq_codes, codes_file)
                    elapsed = time.perf_counter() - t_start
                    logger.info(
                        "  Encoded voice '%s' in %.1fs (%d VQ frames) → saved codes.pt",
                        voice_name, elapsed, vq_codes.shape[-1],
                    )
            except Exception as e:
                logger.warning("  Failed to load voice '%s': %s", voice_name, e)
    else:
        logger.info("No voices/ directory found at %s, skipping preload", voices_dir)

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
    batch_window_ms: float = 2.0,
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

    # Eagerly load and warm up the stream codec at factory time to avoid
    # 500ms-2s cold-start latency on the first streaming request.
    logger.info("Eagerly loading stream codec for TTFA optimization …")
    _stream_codec = _load_codec(checkpoint_dir, stream_vocoder_device)
    _total_upsample = _warmup_codec(
        _stream_codec, num_codebooks=num_codebooks, device=stream_vocoder_device
    )

    # torch.compile the codec decoder for faster vocoding (~2x speedup).
    # The DAC decoder is purely convolutional + snake activations, which
    # TorchInductor fuses very well.
    if stream_vocoder_device != "cpu":
        try:
            _stream_codec.from_indices = torch.compile(
                _stream_codec.from_indices, mode="reduce-overhead"
            )
            logger.info("torch.compiled stream codec from_indices")
        except Exception as e:
            logger.warning("torch.compile on stream codec failed, using eager: %s", e)

    def _get_stream_codec() -> tuple[Any, int]:
        return _stream_codec, _total_upsample

    _patch_fish_config_for_sglang(checkpoint_dir)
    server_args = ServerArgs(
        model_path=checkpoint_dir,
        tp_size=1,
        dtype="bfloat16",
        # FP8 quantization requires pre-quantized checkpoint — see quantize_fp8.py
        attention_backend="fa3",  # Must match training; flashinfer causes dynamo conflicts
        mem_fraction_static=0.85,
        chunked_prefill_size=8192,
        max_running_requests=48,
        disable_cuda_graph=True,
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
        batch_window_ms=batch_window_ms,
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
        # Strip the live S2ProState before the payload leaves this stage
        # (it can't be serialized by msgpack for ZMQ transport).
        strip_live_state(payload.data)
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
            return {
                "audio_waveform": delta.numpy().tobytes(),
                "audio_waveform_shape": list(delta.shape),
                "audio_waveform_dtype": "float32",
                "sample_rate": sample_rate,
                "modality": "audio",
            }

        return await asyncio.to_thread(_vocode)

    async def _flush_stream(payload: StagePayload | None):
        """Flush remaining un-vocoded tokens when the stream ends.

        Called by EngineExecutor after the last stream item. Without this,
        tokens accumulated after the last periodic vocode are lost — causing
        audio to be cut off at the end.

        Async so the vocode runs in the thread pool where torch.compile's
        CUDA graph TLS is available (same pool as _stream_builder).
        """
        if payload is None or not isinstance(payload.data, dict):
            return None

        stream_codes = payload.data.get(_STREAM_CODES_KEY)
        if not stream_codes:
            return None

        base_offset = int(payload.data.get("_stream_base_offset", 0))
        abs_total = base_offset + len(stream_codes)
        prev_end = int(payload.data.get(_STREAM_LAST_VOCODE_TOKENS_KEY, 0))
        if abs_total <= prev_end:
            return None  # nothing new to vocode

        codec, upsample = _get_stream_codec()
        ctx = min(stream_left_context, prev_end)
        window_start = prev_end - ctx
        local_start = window_start - base_offset
        # Guard against negative index from code eviction
        if local_start < 0:
            ctx = ctx + local_start  # reduce context by the missing amount
            local_start = 0
        if local_start >= len(stream_codes):
            return None

        window_codes = torch.cat(stream_codes[local_start:], dim=1)
        codebook_input = window_codes[1:].to(stream_vocoder_device)
        if codebook_input.shape[1] == 0:
            return None

        trim_samples = max(ctx, 0) * upsample
        sample_rate = codec.sample_rate
        remaining = abs_total - prev_end

        def _vocode_flush():
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
            logger.info("[FLUSH] vocoded %d remaining tokens → %d samples (%.2fs)",
                         remaining, delta.numel(), delta.numel() / sample_rate)
            return {
                "audio_waveform": delta.numpy().tobytes(),
                "audio_waveform_shape": list(delta.shape),
                "audio_waveform_dtype": "float32",
                "sample_rate": sample_rate,
                "modality": "audio",
            }

        return await asyncio.to_thread(_vocode_flush)

    # Attach flush so EngineExecutor calls it after the last stream item
    _stream_builder.flush = _flush_stream  # type: ignore[attr-defined]

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
