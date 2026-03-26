# SPDX-License-Identifier: Apache-2.0
"""High-level preprocessing utilities (model-agnostic)."""

from sglang_omni.preprocessing.audio import (
    AudioMediaIO,
    build_audio_mm_inputs,
    compute_audio_cache_key,
    ensure_audio_list_async,
)
from sglang_omni.preprocessing.base import MediaIO
from sglang_omni.preprocessing.resource_connector import (
    MultiModalResourceConnector,
    get_global_resource_connector,
)
from sglang_omni.preprocessing.text import (
    append_modality_placeholders,
    apply_chat_template,
    ensure_chat_template,
    load_chat_template,
    normalize_messages,
)

__all__ = [
    "append_modality_placeholders",
    "apply_chat_template",
    "AudioMediaIO",
    "build_audio_mm_inputs",
    "compute_audio_cache_key",
    "ensure_audio_list_async",
    "ensure_chat_template",
    "get_global_resource_connector",
    "load_chat_template",
    "MultiModalResourceConnector",
    "MediaIO",
    "normalize_messages",
]
