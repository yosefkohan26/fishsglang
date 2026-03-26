# SPDX-License-Identifier: Apache-2.0
"""Helpers to convert between StagePayload.data and S2ProState.

Optimized to avoid tensor-to-list-to-tensor round-trips when stages
run in the same process (the common single-process deployment).
"""

from __future__ import annotations

from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.proto import StagePayload

# Sentinel key indicating the payload carries a live S2ProState object
# (with tensors intact) rather than a serialized dict.
_LIVE_STATE_KEY = "_s2pro_live_state"


def load_state(payload: StagePayload) -> S2ProState:
    """Extract S2ProState from payload, avoiding deserialization when possible."""
    data = payload.data
    if isinstance(data, dict):
        live = data.get(_LIVE_STATE_KEY)
        if isinstance(live, S2ProState):
            return live
    return S2ProState.from_dict(data)


def store_state(payload: StagePayload, state: S2ProState) -> StagePayload:
    """Store S2ProState into payload, keeping tensors alive for in-process use.

    The dict form is still produced (for relay serialization / vocoder stage),
    but the live S2ProState is also attached so the next in-process consumer
    can skip the list->tensor reconstruction.
    """
    d = state.to_dict()
    # Attach live state for in-process consumers (avoids list->tensor round-trip).
    # Uses a non-string key so it's automatically skipped by dict serializers
    # (msgpack, pickle of dict) that only handle string keys, and won't poison
    # the ZMQ completion message.
    d[_LIVE_STATE_KEY] = state
    payload.data = d
    return payload


def strip_live_state(data: dict) -> dict:
    """Remove the live S2ProState before serialization across process boundaries."""
    if isinstance(data, dict) and _LIVE_STATE_KEY in data:
        data.pop(_LIVE_STATE_KEY, None)
    return data
