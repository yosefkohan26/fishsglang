# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS SGLang runtime: iteration controller, model runner, output processor.

The model runner handles:
  - Prefill: inject voice embeddings via input_embeds
  - Decode: read audio codes from model's persistent output buffers
  - EOS detection: acoustic END_AUDIO token → request finished

Architecture mirrors S2-Pro's s2pro_sglang_ar.py but adapted for:
  - Flow matching instead of AR codebook loop
  - Voice embeddings (float tensors) instead of VQ codes
  - 37 codebooks (1 semantic + 36 acoustic) instead of 11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from sglang_omni.engines.omni.runtime.sglang_ar import (
    SGLangARRequestData,
    SGLangBatchPlanner,
    SGLangResourceManager,
)
from sglang_omni.engines.omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:
    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker
    from sglang_omni.models.voxtral_tts.acoustic_transformer import FlowMatchingAcousticTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step output
# ---------------------------------------------------------------------------

@dataclass
class VoxtralStepOutput:
    """Per-step output: 37 codebook tokens (1 semantic + 36 acoustic)."""
    codes: torch.Tensor  # [num_codebooks] or [num_codebooks, 1]
    is_eos: bool = False


# ---------------------------------------------------------------------------
# Request data
# ---------------------------------------------------------------------------

@dataclass
class VoxtralSGLangRequestData(SGLangARRequestData):
    """Voxtral-specific request data extending SGLang AR request."""
    # Voice embedding for prefill injection
    voice_embedding: torch.Tensor | None = None
    audio_token_mask: torch.Tensor | None = None

    # Audio output accumulator
    num_codebooks: int = 37
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None

    # Previous step's codes for autoregressive feedback
    # Shape: [num_codebooks] — fed back as input embedding for next decode step
    _last_codes: torch.Tensor | None = None

    # Sampling params
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.1


# ---------------------------------------------------------------------------
# Iteration Controller
# ---------------------------------------------------------------------------

class VoxtralSGLangIterationController:
    """Handles per-request state updates and EOS detection for Voxtral TTS."""

    def __init__(
        self,
        tree_cache: Any,
        eos_token_id: int,
        max_new_tokens: int = 2048,
    ) -> None:
        self.tree_cache = tree_cache
        self._eos_token_id = eos_token_id
        self._max_new_tokens = max_new_tokens

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: VoxtralSGLangRequestData = request.data
        req = data.req

        # Chunked prefill: decrement counter, don't process output
        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        step_out: VoxtralStepOutput | None = output.data
        if step_out is None:
            return

        # Accumulate output codes
        data.output_codes.append(step_out.codes.clone())

        # Store last codes for autoregressive feedback — next decode step
        # will use these to compute input_embeds via audio_token_embedding
        data._last_codes = step_out.codes.clone()

        # Append the fake "audio token" or EOS to SGLang's output_ids
        if step_out.is_eos:
            req.output_ids.append(self._eos_token_id)
        else:
            req.output_ids.append(24)  # AUDIO_TOKEN_ID placeholder

        if not req.finished() and req.decode_batch_idx == 0:
            self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: VoxtralSGLangRequestData = request.data

        if data.req.is_chunked > 0:
            return False

        step_out = output.data
        if step_out is None:
            return False

        if step_out.is_eos:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False


# ---------------------------------------------------------------------------
# Model Runner
# ---------------------------------------------------------------------------

class VoxtralSGLangModelRunner:
    """Unified model runner: Mistral LLM + flow matching acoustic head.

    Prefill: injects voice embeddings via input_embeds.
    Decode: injects previous step's audio codes as input_embeds via
            audio_token_embedding (MultiVocabEmbeddings), then runs
            LLM + acoustic decode. This is the key AR feedback loop.
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        batch_planner: SGLangBatchPlanner,
        audio_token_id: int = 24,
    ):
        self.model_worker = model_worker
        self.batch_planner = batch_planner
        self._audio_token_id = audio_token_id
        self.device = torch.device(f"cuda:{model_worker.gpu_id}")

    def _inject_voice_embeds_prefill(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compute input_embeds with voice embedding injection for prefill."""
        device = model_worker_batch.input_ids.device
        text_model = self.model_worker.model_runner.model
        embed_tokens = text_model.get_embed_tokens()

        input_ids = model_worker_batch.input_ids
        text_embeds = embed_tokens(input_ids)

        offset = 0
        for sched_req in scheduler_output.requests:
            data: VoxtralSGLangRequestData = sched_req.data
            req_len = data.req.extend_input_len

            if data.voice_embedding is not None and data.audio_token_mask is not None:
                voice_emb = data.voice_embedding.to(device=device, dtype=text_embeds.dtype)
                mask = data.audio_token_mask.to(device)

                # Handle prefix cache: only inject for the extend portion
                prefix_len = len(data.req.prefix_indices)
                mask_slice = mask[prefix_len: prefix_len + req_len]

                if mask_slice.any():
                    # Count how many audio tokens were already in prefix
                    audio_before = mask[:prefix_len].sum().item() if prefix_len > 0 else 0
                    n_audio_in_slice = int(mask_slice.sum().item())
                    voice_slice = voice_emb[audio_before: audio_before + n_audio_in_slice]

                    # Replace audio token positions with voice embeddings
                    mask_indices = mask_slice.nonzero(as_tuple=True)[0] + offset
                    text_embeds[mask_indices] = voice_slice

            offset += req_len

        model_worker_batch.input_embeds = text_embeds

    def _inject_code_embeds_decode(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compute input_embeds from previous step's audio codes for decode.

        This is the AR feedback loop: the acoustic codes from the previous step
        are embedded via audio_token_embedding and summed to produce the input
        embedding for the current decode step. Without this, the LLM has no
        information about what audio it generated previously.
        """
        device = model_worker_batch.input_ids.device
        text_model = self.model_worker.model_runner.model
        embed_tokens = text_model.get_embed_tokens()
        audio_emb = text_model.audio_token_embedding

        input_ids = model_worker_batch.input_ids  # [total_tokens]
        text_embeds = embed_tokens(input_ids)

        has_code_feedback = False
        for i, sched_req in enumerate(scheduler_output.requests):
            data: VoxtralSGLangRequestData = sched_req.data
            if data._last_codes is None:
                continue

            # The audio_token_embedding has shape [9088, 3072] which maps
            # all codebook tokens (37 codebooks with offsets) to embeddings.
            # We need to look up each codebook's code with the proper offset
            # and sum them to get the combined embedding for this position.
            codes = data._last_codes.to(device)  # [37]

            # Compute offsets for MultiVocabEmbeddings lookup
            # codebook_sizes with special tokens: semantic=8194, acoustic=23 each
            # But our codes already include the special token offset (0-8193, 0-22)
            # The audio_token_embedding was trained with cumulative offsets:
            #   codebook 0: codes 0..8193 → embedding indices 0..8193
            #   codebook 1: codes 0..22 → embedding indices 8194..8216
            #   codebook 2: codes 0..22 → embedding indices 8217..8239
            #   etc.
            semantic_size = 8192 + 2  # 8194 padded to 128 = 8192+2 = 8194? Let me use the actual
            acoustic_size = 21 + 2    # 23

            # Build offset indices
            offsets = [0]  # semantic starts at 0
            for cb in range(36):
                offsets.append(offsets[-1] + (semantic_size if cb == 0 else acoustic_size))
            offsets_t = torch.tensor(offsets, device=device, dtype=torch.long)

            # Look up each codebook's embedding and sum
            offset_codes = codes + offsets_t[:len(codes)]
            # Clamp to valid range
            offset_codes = offset_codes.clamp(0, audio_emb.weight.shape[0] - 1)
            all_embs = audio_emb(offset_codes)  # [37, 3072]
            combined_emb = all_embs.sum(dim=0)  # [3072]

            text_embeds[i] = combined_emb.to(text_embeds.dtype)
            has_code_feedback = True

        if has_code_feedback:
            model_worker_batch.input_embeds = text_embeds

    def _build_outputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Read codes from model's persistent output buffers."""
        text_model = self.model_worker.model_runner.model
        outputs = {}

        for i, sched_req in enumerate(scheduler_output.requests):
            data: VoxtralSGLangRequestData = sched_req.data
            if data.req.is_chunked > 0:
                outputs[sched_req.request_id] = RequestOutput(
                    request_id=sched_req.request_id,
                    data=None,
                    finished=False,
                )
                continue

            # Read from persistent buffer and clone
            codes = text_model._output_codes[i].clone()
            is_eos = bool(text_model._output_is_eos[i].item())

            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=VoxtralStepOutput(codes=codes, is_eos=is_eos),
                finished=False,
            )
        return outputs

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        # Ensure correct CUDA device context
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})

        model_worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = schedule_batch.forward_mode.is_extend()

        if is_prefill:
            self._inject_voice_embeds_prefill(model_worker_batch, scheduler_output)
        else:
            self._inject_code_embeds_decode(model_worker_batch, scheduler_output)

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )
        batch_result = self.model_worker.forward_batch_generation(forward_batch)

        # For prefill-only batches, produce dummy output tokens
        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )

        self.batch_planner.record_last_batch(schedule_batch)

        # Read codes from model's persistent output buffers
        outputs = self._build_outputs(scheduler_output)

        # Set output_ids for SGLang's token tracking
        text_model = self.model_worker.model_runner.model
        bs = len(scheduler_output.requests)
        if text_model._acoustic_ready:
            # Use audio_token_id as placeholder; EOS gets actual EOS
            fake_output_ids = torch.full(
                (bs,), self._audio_token_id,
                dtype=torch.long, device=self.device,
            )
            # Set EOS for finished requests
            eos_mask = text_model._output_is_eos[:bs]
            fake_output_ids[eos_mask] = 2  # EOS_TOKEN_ID
            schedule_batch.output_ids = fake_output_ids
        else:
            schedule_batch.output_ids = batch_result.next_token_ids

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )


# ---------------------------------------------------------------------------
# Resource Manager
# ---------------------------------------------------------------------------

class VoxtralSGLangResourceManager(SGLangResourceManager):
    def free(self, request: SchedulerRequest) -> None:
        data: VoxtralSGLangRequestData = request.data
        release_kv_cache(data.req, self.tree_cache)
        data.output_codes.clear()
        data.voice_embedding = None
        data.audio_token_mask = None
