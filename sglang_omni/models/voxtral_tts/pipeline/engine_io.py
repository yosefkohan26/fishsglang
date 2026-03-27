# SPDX-License-Identifier: Apache-2.0
"""Engine request/result helpers for the Voxtral TTS stage."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.models.voxtral_tts.io import VoxtralState
from sglang_omni.models.voxtral_tts.runtime.voxtral_sglang_ar import (
    VoxtralSGLangRequestData,
)


def build_voxtral_tts_request(
    state: VoxtralState, tokenizer: Any, *, request_id: str = ""
) -> VoxtralSGLangRequestData:
    """Build a VoxtralSGLangRequestData from VoxtralState."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids_list = list(state.input_ids)
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    # Voice embedding (may be None for no-voice requests)
    voice_embedding = None
    if state.voice_embedding is not None:
        if isinstance(state.voice_embedding, torch.Tensor):
            voice_embedding = state.voice_embedding
        else:
            voice_embedding = torch.tensor(state.voice_embedding)

    # Audio token mask
    audio_token_mask = None
    if voice_embedding is not None:
        audio_token_mask = torch.tensor(
            [t == state.audio_token_id for t in input_ids_list],
            dtype=torch.bool,
        )

    sampling_params = SamplingParams(
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
    )

    vocab_size = getattr(tokenizer, "vocab_size", 131072)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )

    return VoxtralSGLangRequestData(
        input_ids=input_ids,
        req=req,
        voice_embedding=voice_embedding,
        audio_token_mask=audio_token_mask,
        num_codebooks=state.num_codebooks,
        max_new_tokens=state.max_new_tokens,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
    )


def apply_tts_result(state: VoxtralState, result: VoxtralSGLangRequestData) -> None:
    """Copy engine output codes into pipeline state."""
    if result.output_codes:
        state.output_codes = result.output_codes
        state.completion_tokens = len(result.output_codes)
    else:
        state.output_codes = None
    state.prompt_tokens = len(result.input_ids) if result.input_ids is not None else 0
