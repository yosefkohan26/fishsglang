# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and VoxtralState."""

from __future__ import annotations

from sglang_omni.models.voxtral_tts.io import VoxtralState
from sglang_omni.proto import StagePayload

_LIVE_STATE_KEY = "_voxtral_live_state"


def load_state(payload: StagePayload) -> VoxtralState:
    data = payload.data
    if isinstance(data, dict):
        live = data.get(_LIVE_STATE_KEY)
        if isinstance(live, VoxtralState):
            return live
    return VoxtralState.from_dict(data)


def store_state(payload: StagePayload, state: VoxtralState) -> StagePayload:
    d = state.to_dict()
    d[_LIVE_STATE_KEY] = state
    payload.data = d
    return payload


def strip_live_state(data: dict) -> dict:
    """Remove non-serializable objects before ZMQ transport."""
    if not isinstance(data, dict):
        return data
    data.pop(_LIVE_STATE_KEY, None)
    # Remove accumulated stream tensors and other non-serializable keys
    keys_to_remove = [k for k in data if k.startswith("_")]
    for k in keys_to_remove:
        data.pop(k, None)
    return data
