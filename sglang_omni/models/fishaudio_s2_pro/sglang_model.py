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

        Matches the sampling behaviour of S2ProSGLangOutputProcessor._two_stage_decode:
        1. BF16 logit alignment (BEFORE bias — matches training numerics)
        2. Semantic bias (constrain to semantic tokens + EOS)
        3. Per-request repetition penalty on last-16 tokens
        4. RAS: if last-4 tokens contain duplicates, boost temperature/top_p
        5. Top-k → Top-p → Temperature → Multinomial sampling
        """
        bs = logits.shape[0]

        # 1. BF16 alignment FIRST (match fish-speech training numerics)
        aligned_logits = logits.to(torch.bfloat16).to(torch.float32)

        # 2. Semantic bias (constrain vocabulary)
        aligned_logits = aligned_logits + self._semantic_bias.float()

        # Per-request sampling
        for i in range(bs):
            row = aligned_logits[i : i + 1]  # [1, vocab]
            n_prev = int(self._prev_tokens_len[i].item())

            # 3. Repetition penalty
            if n_prev > 0:
                prev = self._prev_tokens[i, :n_prev].unsqueeze(0)
                score = torch.gather(row, dim=-1, index=prev)
                rep_p = self._sampling_rep_penalty[i]
                score = torch.where(score < 0, score * rep_p, score / rep_p)
                row = row.clone()
                row.scatter_(dim=-1, index=prev, src=score)

            # 4. RAS: detect repetition in last 4 tokens → boost diversity
            temperature = self._sampling_temperature[i]
            top_p = self._sampling_top_p[i]
            if n_prev >= 4:
                last4 = self._prev_tokens[i, max(0, n_prev - 4) : n_prev].tolist()
                if len(set(last4)) < len(last4):
                    temperature = torch.tensor(1.5, device=row.device)
                    top_p = torch.tensor(0.95, device=row.device)

            # 5. Top-k
            top_k = self._sampling_top_k
            if top_k > 0:
                tk_vals, tk_idx = torch.topk(
                    row, min(top_k, row.size(-1)), dim=-1
                )
                row = torch.full_like(row, -float("inf"))
                row.scatter_(dim=-1, index=tk_idx, src=tk_vals)

            # 6. Top-p
            sorted_logits, sorted_idx = torch.sort(row, descending=True)
            cum_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )
            mask = cum_probs > top_p
            mask[..., 0] = False
            remove = mask.scatter(dim=-1, index=sorted_idx, src=mask)
            row = row.masked_fill(remove, -float("inf"))

            # 7. Temperature + multinomial
            row = row / torch.clamp(temperature, min=1e-5)
            probs = torch.nn.functional.softmax(row, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            self._output_semantic_ids[i] = token.item()



        semantic_token = self._output_semantic_ids[:bs]

        # Batched codebook loop (fast AR)
        self._audio_decoder.reset_caches()
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
    # Helpers
    # ------------------------------------------------------------------

    def get_embed_tokens(self):
        return self.embed_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        """Load weights from fish_speech FishQwen3OmniForCausalLM checkpoint.

        Supports INT4/INT8 quantized checkpoints: collects .qweight/.scales/.qzeros
        triplets, dequantizes to BF16, then loads as normal weights.
        """
        params_dict = dict(self.named_parameters())

        # First pass: collect all weights, buffering quantized components
        quant_buf: dict[str, dict[str, Tensor]] = {}  # base_name → {suffix → tensor}
        normal_weights: list[Tuple[str, Tensor]] = []

        for name, loaded_weight in weights:
            if not name.startswith("text_model.model."):
                continue
            name = name[len("text_model.model."):]

            # Detect quantized weight components
            for suffix in (".qweight", ".scales", ".qzeros", ".awq_scales"):
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    quant_buf.setdefault(base, {})[suffix] = loaded_weight
                    break
            else:
                normal_weights.append((name, loaded_weight))

        # Dequantize buffered quantized weights
        if quant_buf:
            logger.info("Dequantizing %d quantized weight matrices in load_weights…", len(quant_buf))
            for base, parts in quant_buf.items():
                qw = parts.get(".qweight")
                sc = parts.get(".scales")
                if qw is None or sc is None:
                    continue
                qz = parts.get(".qzeros")
                if qz is not None:
                    # INT4 group quantization
                    from sglang_omni.models.weight_loader import _dequantize_int4
                    awq_sc = parts.get(".awq_scales")
                    w = _dequantize_int4(qw, sc, qz, group_size=128, awq_scales=awq_sc)
                else:
                    # INT8 per-channel symmetric
                    w = (qw.float() * sc.float().unsqueeze(1)).bfloat16()
                # base is e.g. "layers.0.attention.wqkv.weight"
                normal_weights.append((base, w))

        # Now load all (dequantized + original) weights
        for name, loaded_weight in normal_weights:
            if self._load_remapped_weight(name, loaded_weight, params_dict):
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping weight: %s", name)

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
