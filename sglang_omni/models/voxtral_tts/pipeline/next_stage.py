# SPDX-License-Identifier: Apache-2.0
"""Stage routing callbacks for the Voxtral TTS pipeline."""

from __future__ import annotations

from typing import Any

from sglang_omni.proto import StagePayload

PREPROCESSING_STAGE = "preprocessing"
TTS_ENGINE_STAGE = "tts_engine"


def preprocessing_next(request_id: str, output: Any) -> str | None:
    del request_id, output
    return TTS_ENGINE_STAGE


def tts_engine_next(request_id: str, output: Any) -> str | None:
    """Terminal stage — streaming vocoding is done in-band."""
    del request_id, output
    return None
