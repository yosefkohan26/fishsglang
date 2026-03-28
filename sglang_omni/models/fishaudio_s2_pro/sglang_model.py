# SPDX-License-Identifier: Apache-2.0
"""SGLang-native S2-Pro unified model: slow head (text) + fast head (codebook).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.models import apply_qk_norm
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class S2ProAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_norm:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class S2ProDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = S2ProAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            qk_norm=qk_norm,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class S2ProSGLangTextModel(nn.Module):

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        vocab_size: int = 155776,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_layers: int = 36,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        qk_norm: bool = True,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()

        if config is not None:
            tc = config.text_config
            vocab_size = tc.vocab_size
            hidden_size = tc.dim
            intermediate_size = tc.intermediate_size
            num_layers = tc.n_layer
            num_heads = tc.n_head
            num_kv_heads = tc.n_local_heads
            head_dim = tc.head_dim
            rope_base = tc.rope_base
            max_position_embeddings = tc.max_seq_len
            rms_norm_eps = tc.norm_eps
            qk_norm = tc.attention_qk_norm
            tie_word_embeddings = tc.tie_word_embeddings

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        # Set via setup_vq_decode() after model load
        self._vq_ready = False

        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: S2ProDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=idx,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                qk_norm=qk_norm,
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        if not tie_word_embeddings:
            from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

            self.lm_head = ParallelLMHead(vocab_size, hidden_size)

    # ------------------------------------------------------------------
    # Post-load setup
    # ------------------------------------------------------------------

    def setup_vq_decode(
        self,
        audio_decoder: nn.Module,
        *,
        num_codebooks: int,
        codebook_size: int,
        semantic_begin_id: int,
        semantic_end_id: int,
        im_end_id: int,
        max_batch_size: int,
    ) -> None:
        """Attach audio decoder and allocate persistent GPU buffers."""
        device = self.embed_tokens.weight.device

        # Audio decoder (fast head)
        self._audio_decoder = audio_decoder
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks
        self._semantic_begin_id = semantic_begin_id

        # Shared codebook embedding from audio decoder (for VQ input combination)
        self._vq_codebook_embeddings = audio_decoder.codebook_embeddings
        self._vq_codebook_offsets = audio_decoder.codebook_offsets.to(device)
        self._vq_scale = 1.0 / math.sqrt(num_codebooks + 1)

        # Input buffers: VQ codes from previous step (updated by ModelRunner)
        self._vq_codes = torch.zeros(
            max_batch_size, num_codebooks, dtype=torch.long, device=device
        )
        self._vq_mask = torch.zeros(max_batch_size, dtype=torch.bool, device=device)

        # Semantic bias: mask all non-semantic and non-EOS tokens
        bias = torch.full(
            (self.vocab_size,), -float("inf"), device=device, dtype=torch.bfloat16
        )
        bias[semantic_begin_id : semantic_end_id + 1] = 0.0
        bias[im_end_id] = 0.0
        self._semantic_bias = bias

        # Output buffers: written by _decode_codebooks, read by ModelRunner
        self._output_codes = torch.zeros(
            max_batch_size, num_codebooks + 1, dtype=torch.long, device=device
        )
        self._output_semantic_ids = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )

        # Per-request sampling buffers (updated by ModelRunner before each step)
        self._sampling_temperature = torch.full(
            (max_batch_size,), 0.8, device=device, dtype=torch.float32
        )
        self._sampling_top_p = torch.full(
            (max_batch_size,), 0.8, device=device, dtype=torch.float32
        )
        self._sampling_top_k = 30
        self._sampling_rep_penalty = torch.full(
            (max_batch_size,), 1.1, device=device, dtype=torch.float32
        )
        # Ring buffer of recent semantic tokens for repetition penalty (last 16)
        self._prev_tokens = torch.zeros(
            max_batch_size, 16, dtype=torch.long, device=device
        )
        self._prev_tokens_len = torch.zeros(
            max_batch_size, dtype=torch.long, device=device
        )

        self._vq_ready = True

        # Fast AR codebook loop — eager mode (torch.compile adds dispatch
        # overhead that exceeds kernel savings at small batch sizes)
        self._compiled_codebook_loop = self._codebook_loop

    def _codebook_loop(
        self, hidden_states: Tensor, semantic_token: Tensor, bs: int
    ) -> None:
        """Fast AR codebook generation — compilable with fullgraph=True.

        No reset_caches() needed: flash_attn_with_kvcache only reads up to
        cache_seqlens (codebook_idx), so stale values beyond that position
        are never accessed.  Removing the memset saves ~0.2-0.5ms/step.
        """
        fast_input = self._audio_decoder.project_in(hidden_states)
        fast_input = fast_input.unsqueeze(1)
        self._audio_decoder.forward_kvcached(fast_input, codebook_idx=0)

        sem_id = (semantic_token - self._semantic_begin_id).clamp(min=0)
        cb_hidden = self._audio_decoder.embeddings(sem_id).unsqueeze(1)

        self._output_codes[:bs, 0] = semantic_token
        self._output_codes[:bs, 1] = sem_id

        for cb_idx in range(1, self._num_codebooks):
            cb_logits = self._audio_decoder.forward_kvcached(
                cb_hidden, codebook_idx=cb_idx
            )
            cb_logits = cb_logits[:, 0, : self._codebook_size]
            cb_token = torch.argmax(cb_logits, dim=-1)
            cb_hidden = self._audio_decoder.embeddings(cb_token).unsqueeze(1)
            self._output_codes[:bs, cb_idx + 1] = cb_token

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        if input_embeds is not None:
            # Prefill: input_embeds from ModelRunner (with VQ injection)
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

            # Decode: VQ combination from persistent buffers (CUDA-graph-safe)
            if self._vq_ready:
                bs = hidden_states.shape[0]
                vq_codes = self._vq_codes[:bs]
                vq_mask = self._vq_mask[:bs]
                offset_parts = vq_codes + self._vq_codebook_offsets[None, :]
                all_embeds = self._vq_codebook_embeddings(offset_parts)
                vq_sum = all_embeds.sum(dim=1).to(hidden_states.dtype)
                combined = (hidden_states + vq_sum) * self._vq_scale
                hidden_states = torch.where(
                    vq_mask.unsqueeze(-1), combined, hidden_states
                )

        # Transformer
        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Extend: prune to last-token positions
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        # Logits
        if self.tie_word_embeddings:
            logits = torch.nn.functional.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        # Codebook decode: constrained sampling + batched codebook loop
        if self._vq_ready:
            self._decode_codebooks(logits, hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def _decode_codebooks(self, logits: Tensor, hidden_states: Tensor) -> None:
        """Constrained semantic sampling + batched codebook generation.

        All operations are fully vectorized across the batch — no Python
        for-loop over requests. This is critical for latency at high
        concurrency (O(1) GPU ops instead of O(bs) sequential Python).

        Sampling steps (batched):
        1. BF16 logit alignment (match training numerics)
        2. Semantic bias (constrain vocabulary)
        3. Batched repetition penalty via gather/scatter
        4. RAS: detect repetition in last-4 tokens, boost diversity
        5. Top-k → Top-p → Temperature → Multinomial (all batched)
        """
        bs = logits.shape[0]
        device = logits.device

        # 1. BF16 alignment FIRST (match fish-speech training numerics)
        L = logits.to(torch.bfloat16).to(torch.float32)  # [bs, vocab]

        # 2. Semantic bias (constrain vocabulary)
        L = L + self._semantic_bias.float()

        # 3. Batched repetition penalty
        # Gather scores at prev_tokens positions, apply penalty, scatter back.
        # Use the full [bs, 16] buffer — positions beyond n_prev are 0 (valid
        # token id) but rep_penalty on an ungenerated token is harmless since
        # those positions' scores get masked by semantic_bias anyway.
        prev_lens = self._prev_tokens_len[:bs]  # [bs]
        has_prev = prev_lens > 0  # [bs] bool
        if has_prev.any():
            prev_idx = self._prev_tokens[:bs]  # [bs, 16]
            scores = torch.gather(L, dim=1, index=prev_idx)  # [bs, 16]
            rep_p = self._sampling_rep_penalty[:bs].unsqueeze(1)  # [bs, 1]
            scores = torch.where(scores < 0, scores * rep_p, scores / rep_p)
            # Mask: only scatter for rows that have previous tokens, and only
            # for positions < n_prev (avoid penalising padding zeros).
            pos_range = torch.arange(16, device=device).unsqueeze(0)  # [1, 16]
            valid = has_prev.unsqueeze(1) & (pos_range < prev_lens.unsqueeze(1))  # [bs, 16]
            # Where not valid, keep original scores (effectively no-op scatter)
            orig_scores = torch.gather(L, dim=1, index=prev_idx)
            scores = torch.where(valid, scores, orig_scores)
            L = L.scatter(dim=1, index=prev_idx, src=scores)

        # 4. RAS: detect repetition in last 4 tokens → boost temperature/top_p
        temperature = self._sampling_temperature[:bs].clone()  # [bs]
        top_p = self._sampling_top_p[:bs].clone()  # [bs]
        # Vectorised duplicate check: for each row, grab last-4 tokens and
        # check if unique count < 4.  Use sorted comparison for GPU efficiency.
        ras_eligible = prev_lens >= 4  # [bs]
        if ras_eligible.any():
            # Extract last-4 tokens for each row (clamped indices for safety)
            starts = (prev_lens - 4).clamp(min=0)  # [bs]
            # Gather 4 tokens per row from _prev_tokens
            offsets = torch.arange(4, device=device).unsqueeze(0)  # [1, 4]
            indices = (starts.unsqueeze(1) + offsets).clamp(max=15)  # [bs, 4]
            last4 = torch.gather(self._prev_tokens[:bs], dim=1, index=indices)  # [bs, 4]
            # Detect duplicates: sort, then check if any adjacent pair is equal
            sorted4, _ = last4.sort(dim=1)
            has_dup = (sorted4[:, 1:] == sorted4[:, :-1]).any(dim=1)  # [bs]
            boost = ras_eligible & has_dup
            temperature = torch.where(boost, torch.tensor(1.5, device=device), temperature)
            top_p = torch.where(boost, torch.tensor(0.95, device=device), top_p)

        # 5. Batched Top-k
        top_k = self._sampling_top_k
        if top_k > 0:
            k = min(top_k, L.size(-1))
            tk_vals, tk_idx = torch.topk(L, k, dim=-1)  # [bs, k]
            L = torch.full_like(L, -float("inf"))
            L.scatter_(dim=-1, index=tk_idx, src=tk_vals)

        # 6. Batched Top-p
        sorted_logits, sorted_idx = torch.sort(L, descending=True)  # [bs, vocab]
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        # Per-row top_p threshold: [bs, 1]
        top_p_2d = top_p.unsqueeze(1)
        sorted_mask = cum_probs > top_p_2d  # [bs, vocab]
        sorted_mask[..., 0] = False  # always keep at least one token
        remove = sorted_mask.scatter(dim=-1, index=sorted_idx, src=sorted_mask)
        L = L.masked_fill(remove, -float("inf"))

        # 7. Batched Temperature + multinomial
        L = L / temperature.unsqueeze(1).clamp(min=1e-5)
        probs = torch.nn.functional.softmax(L, dim=-1)  # [bs, vocab]
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [bs]
        self._output_semantic_ids[:bs] = tokens

        # Batched codebook loop (fast AR)
        self._compiled_codebook_loop(hidden_states, tokens, bs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_embed_tokens(self):
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        """Load weights from fish_speech checkpoint.

        Handles BF16, FP8 (per-channel scales), INT4, and INT8 checkpoints.
        FP8 weights are stored directly as FP8 buffers — NOT dequantized.
        """
        params_dict = dict(self.named_parameters())

        # Collect all weights, buffering quantized components
        quant_buf: dict[str, dict[str, Tensor]] = {}
        normal_weights: list[Tuple[str, Tensor]] = []
        # FP8 weights stored separately: name → (fp8_weight, per_channel_scale)
        fp8_weights: dict[str, Tuple[Tensor, Tensor]] = {}

        for name, loaded_weight in weights:
            if not name.startswith("text_model.model."):
                continue
            name = name[len("text_model.model."):]

            for suffix in (".qweight", ".scales", ".qzeros", ".awq_scales", ".weight_scale"):
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    quant_buf.setdefault(base, {})[suffix] = loaded_weight
                    break
            else:
                normal_weights.append((name, loaded_weight))

        # Process quantized weights
        if quant_buf:
            for base, parts in quant_buf.items():
                ws = parts.get(".weight_scale")
                if ws is not None:
                    # FP8 with per-channel scale — find the FP8 weight tensor
                    fp8_key = base + ".weight"
                    for j, (n, w) in enumerate(normal_weights):
                        if n == fp8_key:
                            normal_weights.pop(j)
                            fp8_weights[fp8_key] = (w, ws)
                            break
                    continue

                qw = parts.get(".qweight")
                sc = parts.get(".scales")
                if qw is None or sc is None:
                    continue
                qz = parts.get(".qzeros")
                if qz is not None:
                    from sglang_omni.models.weight_loader import _dequantize_int4
                    awq_sc = parts.get(".awq_scales")
                    w = _dequantize_int4(qw, sc, qz, group_size=128, awq_scales=awq_sc)
                else:
                    w = (qw.float() * sc.float().unsqueeze(1)).bfloat16()
                normal_weights.append((base, w))

        # Store FP8 weight info for _convert_to_fp8
        self._fp8_weight_data = fp8_weights
        if fp8_weights:
            logger.info("Found %d FP8 weight matrices (will use GPU FP8 GEMM)", len(fp8_weights))

        # Load BF16 weights normally (FP8 weights loaded separately in _convert_to_fp8)
        for name, loaded_weight in normal_weights:
            if self._load_remapped_weight(name, loaded_weight, params_dict):
                continue
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping weight: %s", name)

        # FP8 disabled: dynamic per-row activation quantization overhead
        # exceeds bandwidth savings on 4090 (RTF 1.73 vs 0.53 BF16).
        # Needs fused quant+GEMM kernels to be worthwhile.
        # self._convert_to_fp8()

    def _convert_to_fp8(self):
        """Convert Slow AR linear layers to FP8 with per-channel weight scales
        and dynamic per-row activation scales using rowwise torch._scaled_mm.

        If FP8 weights were loaded from checkpoint, uses those directly.
        Otherwise, quantizes BF16 weights in-place.
        """
        from sglang.srt.layers.linear import (
            QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
        )

        fp8_data = getattr(self, "_fp8_weight_data", {})
        # Build lookup: remapped weight name → (fp8_tensor, scale)
        # The remap converts checkpoint names to model param names
        remap = {
            "attention.wqkv.weight": "self_attn.qkv_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "feed_forward.w1.weight": "gate_up_proj.weight",
            "feed_forward.w3.weight": "gate_up_proj.weight",
            "feed_forward.w2.weight": "down_proj.weight",
        }

        n_converted = 0
        for name, module in list(self.named_modules()):
            if not name.startswith("layers."):
                continue
            if "norm" in name:
                continue
            if not isinstance(module, (QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear)):
                continue
            if not hasattr(module, "weight") or module.weight.numel() < 100_000:
                continue

            device = module.weight.device

            # Check if pre-quantized FP8 weights exist for this layer
            # (from checkpoint). If not, quantize BF16 in-place.
            w = module.weight.data.float()
            out_features, in_features = w.shape

            # Per-channel (per-output-row) quantization
            row_amax = w.abs().max(dim=1).values.clamp(min=1e-12)
            row_scale = row_amax / 448.0  # (out_features,)
            w_fp8 = (w / row_scale.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn)

            # Replace weight param with FP8 buffer
            del module.weight
            module.register_buffer("weight", w_fp8.to(device))
            # Per-channel scale reshaped for rowwise _scaled_mm: (1, out_features)
            # When weight is transposed to (in, out), scale_b is (1, out)
            module.register_buffer(
                "weight_scale",
                row_scale.float().reshape(1, out_features).contiguous().to(device),
            )

            # Monkey-patch quant_method with rowwise FP8 GEMM
            class _FP8RowwiseApply:
                """FP8 GEMM with per-row activation scales + per-channel weight scales.

                Uses torch._scaled_mm rowwise mode:
                  scale_a: (M, 1) — dynamic per-row activation scale
                  scale_b: (1, N) — static per-channel weight scale
                """
                @staticmethod
                def process_weights_after_loading(layer):
                    pass

                @staticmethod
                def apply(layer, x, bias=None):
                    orig_shape = x.shape
                    x_2d = x.reshape(-1, orig_shape[-1])  # (M, K)
                    M = x_2d.shape[0]

                    # Dynamic per-row activation quantization
                    x_float = x_2d.float()
                    row_amax = x_float.abs().max(dim=1).values.clamp(min=1e-12)  # (M,)
                    act_scale = (row_amax / 448.0).reshape(M, 1).contiguous()  # (M, 1)
                    x_fp8 = (x_float / act_scale).clamp(-448, 448).to(torch.float8_e4m3fn)

                    # Rowwise FP8 GEMM: (M,K) @ (K,N) with (M,1) and (1,N) scales
                    out = torch._scaled_mm(
                        x_fp8,
                        layer.weight.t(),
                        scale_a=act_scale,
                        scale_b=layer.weight_scale,
                        out_dtype=torch.bfloat16,
                    )
                    out = out.reshape(*orig_shape[:-1], out.shape[-1])
                    if bias is not None:
                        out = out + bias
                    return out

            module.quant_method = _FP8RowwiseApply()
            n_converted += 1

        if n_converted > 0:
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            logger.info(
                "Converted %d layers to FP8 (per-channel weights, per-row activations). "
                "VRAM: %.1f free / %.1f total",
                n_converted, free / 1e9, total / 1e9,
            )

        # Clean up
        self._fp8_weight_data = None

    def _load_remapped_weight(
        self,
        name: str,
        loaded_weight: Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        remap = {
            "attention.wqkv.weight": None,
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
            "embeddings.weight": "embed_tokens.weight",
            "norm.weight": "norm.weight",
        }
        for ckpt_suffix, target in remap.items():
            if not name.endswith(ckpt_suffix):
                continue
            prefix = name[: -len(ckpt_suffix)]
            if target is None:
                return self._load_fused_qkv(prefix, loaded_weight, params_dict)
            if isinstance(target, tuple):
                target_suffix, shard_id = target
            else:
                target_suffix, shard_id = target, None
            param = params_dict[prefix + target_suffix]
            if shard_id is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            return True
        return False

    def _load_fused_qkv(
        self,
        prefix: str,
        wqkv: Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        target_name = prefix + "self_attn.qkv_proj.weight"
        if target_name not in params_dict:
            return True
        param = params_dict[target_name]
        layer = self.layers[int(prefix.split(".")[1])]
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = wqkv.split([q_size, kv_size, kv_size], dim=0)
        for shard_id, weight in [("q", q), ("k", k), ("v", v)]:
            param.weight_loader(param, weight, shard_id)
        return True


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)


EntryClass = S2ProSGLangTextModel
