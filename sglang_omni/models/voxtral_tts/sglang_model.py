# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Voxtral TTS unified model: Mistral backbone + flow matching acoustic head.

Weight layout (consolidated.safetensors):
  layers.{0-25}.*           → 26-layer Mistral LLM (dim=3072, 32/8 GQA heads)
  norm.weight               → final RMSNorm
  mm_audio_embeddings.tok_embeddings.weight           → [131072, 3072] text+audio embedding
  mm_audio_embeddings.audio_codebook_embeddings.*     → [9088, 3072] multimodal audio embedding
  acoustic_transformer.*    → 3-layer flow matching head
  audio_tokenizer.*         → Codec decoder (loaded separately for streaming vocode)

Forward flow (decode step):
  1. Embed input token (with VQ code injection if decode has prior acoustic codes)
  2. Run 26-layer Mistral transformer (paged KV via RadixAttention)
  3. Produce logits via tied embeddings
  4. Run acoustic head: semantic argmax + 8-step flow matching ODE → [B, 37] codes
  5. Store codes in persistent output buffer for ModelRunner to read
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Mistral decoder layer (standard GQA + SwiGLU)
# ---------------------------------------------------------------------------


class VoxtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-5,
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
            hidden_size, head_dim, num_heads, num_kv_heads, bias=False,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim, hidden_size, bias=False,
        )
        self.rotary_emb = get_rope(
            head_dim, rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base, is_neox_style=False,
        )
        self.attn = RadixAttention(
            num_heads, head_dim, self.scaling,
            num_kv_heads=num_kv_heads, layer_id=layer_id,
        )

    def forward(self, positions: Tensor, hidden_states: Tensor, forward_batch: ForwardBatch) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class VoxtralDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = VoxtralAttention(
            hidden_size=hidden_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            layer_id=layer_id, rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
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
        hidden_states = F.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class VoxtralTTSSGLangModel(nn.Module):
    """SGLang-native Voxtral TTS: Mistral 3B backbone with flow matching acoustic head."""

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        # Voxtral defaults from params.json
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        intermediate_size: int = 9216,
        num_layers: int = 26,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        # Audio embedding
        audio_codebook_vocab_size: int = 9088,
    ) -> None:
        super().__init__()

        if config is not None:
            tc = config.text_config if hasattr(config, "text_config") else config
            vocab_size = getattr(tc, "vocab_size", vocab_size)
            hidden_size = getattr(tc, "hidden_size", hidden_size)
            intermediate_size = getattr(tc, "intermediate_size", intermediate_size)
            num_layers = getattr(tc, "num_hidden_layers", num_layers)
            num_heads = getattr(tc, "num_attention_heads", num_heads)
            num_kv_heads = getattr(tc, "num_key_value_heads", num_kv_heads)
            head_dim = getattr(tc, "head_dim", head_dim)
            rope_base = getattr(tc, "rope_theta", rope_base)
            max_position_embeddings = getattr(tc, "max_position_embeddings", max_position_embeddings)
            rms_norm_eps = getattr(tc, "rms_norm_eps", rms_norm_eps)
            tie_word_embeddings = getattr(tc, "tie_word_embeddings", tie_word_embeddings)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        # Acoustic head — set via setup_acoustic_decode()
        self._acoustic_ready = False

        # Text + audio token embedding (shared)
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)

        # Audio codebook embedding for multimodal voice injection
        self.audio_token_embedding = nn.Embedding(audio_codebook_vocab_size, hidden_size)

        # Transformer layers
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: VoxtralDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=idx,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    # ------------------------------------------------------------------
    # Acoustic decode setup (called after weight loading)
    # ------------------------------------------------------------------

    def setup_acoustic_decode(
        self,
        acoustic_transformer: nn.Module,
        *,
        audio_token_id: int = 24,
        eos_token_id: int = 2,
        max_batch_size: int = 64,
    ) -> None:
        """Attach acoustic transformer and allocate persistent GPU buffers."""
        device = self.embed_tokens.weight.device
        dtype = torch.bfloat16

        self._acoustic_transformer = acoustic_transformer
        self._audio_token_id = audio_token_id
        self._eos_token_id = eos_token_id

        # Setup acoustic transformer buffers
        acoustic_transformer.setup_buffers(max_batch_size, device, dtype)

        # Persistent output buffer: [max_bs, num_codebooks]
        num_codebooks = 1 + acoustic_transformer.n_acoustic_codebook  # semantic + acoustic
        self._output_codes = torch.zeros(
            max_batch_size, num_codebooks, dtype=torch.long, device=device
        )

        # EOS detection: semantic END_AUDIO token → real EOS for scheduler
        self._output_is_eos = torch.zeros(
            max_batch_size, dtype=torch.bool, device=device
        )

        self._acoustic_ready = True
        logger.info(
            "Voxtral acoustic decode ready: %d codebooks, max_bs=%d",
            num_codebooks, max_batch_size,
        )

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
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        # Transformer
        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Prune to last-token positions for extend mode
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        # Logits (tied embeddings)
        logits = F.linear(hidden_states, self.embed_tokens.weight)

        # Acoustic decode: produce audio codes from hidden states
        if self._acoustic_ready:
            self._run_acoustic_decode(hidden_states)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def _run_acoustic_decode(self, hidden_states: Tensor) -> None:
        """Run flow matching to produce audio codes. Writes to persistent buffers."""
        bs = hidden_states.shape[0]

        semantic_code, acoustic_codes = self._acoustic_transformer.decode_one_frame(
            hidden_states
        )

        # Write to persistent output buffer
        self._output_codes[:bs, 0] = semantic_code.squeeze(-1)
        self._output_codes[:bs, 1:] = acoustic_codes

        # EOS detection: semantic END_AUDIO
        from sglang_omni.models.voxtral_tts.acoustic_transformer import FlowMatchingAcousticTransformer
        self._output_is_eos[:bs] = (
            semantic_code.squeeze(-1) == FlowMatchingAcousticTransformer.END_AUDIO_ID
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_embed_tokens(self):
        return self.embed_tokens

    def embed_input_ids_with_voice(
        self,
        input_ids: Tensor,
        voice_embedding: Tensor,
        audio_token_mask: Tensor,
    ) -> Tensor:
        """Replace audio_token positions with voice embedding.

        Args:
            input_ids: [seq_len] token IDs
            voice_embedding: [N, hidden_size] pre-computed voice embeddings
            audio_token_mask: [seq_len] bool mask where input_ids == audio_token_id
        """
        text_embeds = self.embed_tokens(input_ids)
        if voice_embedding is not None and audio_token_mask.any():
            n_audio = int(audio_token_mask.sum().item())
            voice_emb = voice_embedding[:n_audio].to(
                device=text_embeds.device, dtype=text_embeds.dtype
            )
            text_embeds[audio_token_mask] = voice_emb
        return text_embeds

    # ------------------------------------------------------------------
    # Weight loading: Mistral-native → SGLang format
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
        """Load weights from Voxtral checkpoint (Mistral-native format)."""
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Route: only handle LLM + embedding weights here
            if name.startswith("acoustic_transformer.") or name.startswith("audio_tokenizer."):
                continue

            # Remap embedding names
            if name == "mm_audio_embeddings.tok_embeddings.weight":
                name = "embed_tokens.weight"
            elif name.startswith("mm_audio_embeddings.audio_codebook_embeddings.embeddings."):
                name = name.replace(
                    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.",
                    "audio_token_embedding.",
                )
            elif name.startswith("layers."):
                # Remap Mistral-native layer weight names
                mapped = self._remap_layer_weight(name)
                if mapped is None:
                    continue
                name = mapped
            elif name == "norm.weight":
                name = "norm.weight"
            else:
                logger.debug("Skipping weight: %s", name)
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.debug("Skipping unmapped weight: %s", name)

    def _remap_layer_weight(self, name: str) -> str | None:
        """Remap layers.N.mistral_name → layers.N.sglang_name."""
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "layers":
            return None

        layer_idx = parts[1]
        rest = ".".join(parts[2:])

        remap = {
            "attention.wq.weight": ("self_attn.qkv_proj.weight", "q"),
            "attention.wk.weight": ("self_attn.qkv_proj.weight", "k"),
            "attention.wv.weight": ("self_attn.qkv_proj.weight", "v"),
            "attention.wo.weight": ("self_attn.o_proj.weight", None),
            "attention_norm.weight": ("input_layernorm.weight", None),
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),  # gate
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),  # up
            "feed_forward.w2.weight": ("down_proj.weight", None),
            "ffn_norm.weight": ("post_attention_layernorm.weight", None),
        }

        target = remap.get(rest)
        if target is None:
            logger.debug("Unknown layer weight: %s", rest)
            return None

        target_name, shard_id = target
        full_name = f"layers.{layer_idx}.{target_name}"

        if full_name not in dict(self.named_parameters()):
            return None

        param = dict(self.named_parameters())[full_name]

        if shard_id is not None:
            param.weight_loader(param, loaded_weight=None, shard_id=shard_id)
            # Return None to skip the normal load path — weight_loader is called inline
            self._load_shard(full_name, loaded_weight=None, shard_id=shard_id)
            return None  # handled inline

        return full_name

    def _load_shard(self, name: str, loaded_weight: Any, shard_id: Any) -> None:
        """This is a placeholder — actual shard loading happens in load_weights."""
        pass  # Handled by the fused QKV/gate_up weight loader


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor):
    param.data.copy_(loaded_weight)


# Override load_weights to handle fused QKV properly
_orig_load_weights = VoxtralTTSSGLangModel.load_weights


def _patched_load_weights(self, weights: Iterable[Tuple[str, Tensor]]):
    """Load weights with proper fused QKV and gate_up handling."""
    params_dict = dict(self.named_parameters())

    for name, loaded_weight in weights:
        # Skip non-LLM weights
        if name.startswith("acoustic_transformer.") or name.startswith("audio_tokenizer."):
            continue

        # Remap embedding names
        if name == "mm_audio_embeddings.tok_embeddings.weight":
            target = "embed_tokens.weight"
            if target in params_dict:
                param = params_dict[target]
                wl = getattr(param, "weight_loader", _default_weight_loader)
                wl(param, loaded_weight)
            continue

        if name.startswith("mm_audio_embeddings.audio_codebook_embeddings.embeddings."):
            target = name.replace(
                "mm_audio_embeddings.audio_codebook_embeddings.embeddings.",
                "audio_token_embedding.",
            )
            if target in params_dict:
                params_dict[target].data.copy_(loaded_weight)
            continue

        if name == "norm.weight":
            if "norm.weight" in params_dict:
                params_dict["norm.weight"].data.copy_(loaded_weight)
            continue

        if not name.startswith("layers."):
            continue

        # Parse layer weight
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])

        # Fused QKV
        if rest == "attention.wq.weight":
            target = f"layers.{layer_idx}.self_attn.qkv_proj.weight"
            if target in params_dict:
                params_dict[target].weight_loader(params_dict[target], loaded_weight, "q")
            continue
        if rest == "attention.wk.weight":
            target = f"layers.{layer_idx}.self_attn.qkv_proj.weight"
            if target in params_dict:
                params_dict[target].weight_loader(params_dict[target], loaded_weight, "k")
            continue
        if rest == "attention.wv.weight":
            target = f"layers.{layer_idx}.self_attn.qkv_proj.weight"
            if target in params_dict:
                params_dict[target].weight_loader(params_dict[target], loaded_weight, "v")
            continue

        # Fused gate_up
        if rest == "feed_forward.w1.weight":
            target = f"layers.{layer_idx}.gate_up_proj.weight"
            if target in params_dict:
                params_dict[target].weight_loader(params_dict[target], loaded_weight, 0)
            continue
        if rest == "feed_forward.w3.weight":
            target = f"layers.{layer_idx}.gate_up_proj.weight"
            if target in params_dict:
                params_dict[target].weight_loader(params_dict[target], loaded_weight, 1)
            continue

        # Simple 1:1 mappings
        simple_map = {
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "feed_forward.w2.weight": "down_proj.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }
        mapped = simple_map.get(rest)
        if mapped is not None:
            target = f"layers.{layer_idx}.{mapped}"
            if target in params_dict:
                param = params_dict[target]
                wl = getattr(param, "weight_loader", _default_weight_loader)
                wl(param, loaded_weight)
            continue

        logger.debug("Skipping unknown layer weight: %s", name)


VoxtralTTSSGLangModel.load_weights = _patched_load_weights

EntryClass = VoxtralTTSSGLangModel
