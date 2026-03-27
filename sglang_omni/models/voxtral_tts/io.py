# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS pipeline state definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VoxtralState:
    """Per-request pipeline state for Voxtral TTS."""

    # -- From preprocessing ------------------------------------------------
    input_ids: list[int] | None = None
    voice_embedding: Any | None = None  # [N, 3072] tensor or None
    voice_name: str | None = None
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    num_codebooks: int = 37  # 1 semantic + 36 acoustic

    # -- Generation params -------------------------------------------------
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.1

    # -- From TTS engine ---------------------------------------------------
    output_codes: Any | None = None  # list of [37] tensors per frame
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # -- From vocoder ------------------------------------------------------
    audio_samples: Any | None = None
    sample_rate: int = 24000

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.input_ids is not None:
            data["input_ids"] = self.input_ids
        if self.voice_name is not None:
            data["voice_name"] = self.voice_name
        data["audio_token_id"] = self.audio_token_id
        data["begin_audio_token_id"] = self.begin_audio_token_id
        data["num_codebooks"] = self.num_codebooks
        data["max_new_tokens"] = self.max_new_tokens
        data["temperature"] = self.temperature
        data["top_p"] = self.top_p
        data["top_k"] = self.top_k
        data["repetition_penalty"] = self.repetition_penalty
        if self.prompt_tokens:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens:
            data["completion_tokens"] = self.completion_tokens
        if self.engine_time_s:
            data["engine_time_s"] = self.engine_time_s
        data["sample_rate"] = self.sample_rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> VoxtralState:
        return cls(
            input_ids=data.get("input_ids"),
            voice_name=data.get("voice_name"),
            audio_token_id=data.get("audio_token_id", 24),
            begin_audio_token_id=data.get("begin_audio_token_id", 25),
            num_codebooks=data.get("num_codebooks", 37),
            max_new_tokens=data.get("max_new_tokens", 2048),
            temperature=data.get("temperature", 0.0),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k", -1),
            repetition_penalty=data.get("repetition_penalty", 1.1),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            engine_time_s=data.get("engine_time_s", 0.0),
            sample_rate=data.get("sample_rate", 24000),
        )
