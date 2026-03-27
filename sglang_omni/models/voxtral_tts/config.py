# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Voxtral TTS.

Provides the PipelineConfig subclass for registry-based discovery.
Uses a 2-stage pipeline (fused preprocessing + engine) with in-band
streaming vocoding — no separate vocoder stage needed.

Usage:
    sgl-omni serve --model-path mistralai/Voxtral-4B-TTS-2603
"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.voxtral_tts.pipeline.next_stage import (
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
)

_PKG = "sglang_omni.models.voxtral_tts.pipeline"


_STAGES: list[StageConfig] = [
    StageConfig(
        name=PREPROCESSING_STAGE,
        executor=ExecutorConfig(
            factory=f"{_PKG}.stages.create_preprocessing_executor",
        ),
        get_next=f"{_PKG}.next_stage.preprocessing_next",
        relay=RelayConfig(device="cpu"),
    ),
    StageConfig(
        name=TTS_ENGINE_STAGE,
        executor=ExecutorConfig(
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            args={
                "device": "cuda:0",
                "max_new_tokens": 2048,
                "batch_window_ms": 1.0,
                "stream_stride": 3,
                "stream_followup_stride": 10,
                "stream_left_context": 5,
            },
        ),
        get_next=f"{_PKG}.next_stage.tts_engine_next",
        relay=RelayConfig(device="cuda"),
    ),
]


class VoxtralPipelineConfig(PipelineConfig):
    """Default Voxtral TTS pipeline config (2 stages, streaming vocode in-band)."""

    architecture: ClassVar[str] = "VoxtralTTSSGLangModel"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = _STAGES


EntryClass = VoxtralPipelineConfig
