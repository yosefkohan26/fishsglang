# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS tokenizer/prompt builder.

Builds token sequences in the Mistral TTS format:
  [BOS=1] [begin_audio=25] [audio_token=24 × N_voice] [INST=35] text_tokens [/INST=36] [begin_audio=25]

Voice embeddings are pre-loaded .pt files of shape [N, 3072] that replace
audio_token positions during prefill via input_embeds injection.

For RadixCache compatibility:
  - Each voice has a naturally unique prefix length (67-218 tokens)
  - The 2 collision pairs are padded to ensure uniqueness
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Mistral special token IDs
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
AUDIO_TOKEN_ID = 24
BEGIN_AUDIO_TOKEN_ID = 25
INST_START_TOKEN_ID = 35  # [INST]
INST_END_TOKEN_ID = 36    # [/INST]

# Voice preset names (all 20 from params.json)
VOXTRAL_VOICE_PRESETS = [
    "casual_female", "casual_male", "cheerful_female",
    "neutral_female", "neutral_male",
    "pt_male", "pt_female", "nl_male", "nl_female",
    "it_male", "it_female", "fr_male", "fr_female",
    "es_male", "es_female", "de_male", "de_female",
    "ar_male", "hi_male", "hi_female",
]


@dataclass
class VoicePreset:
    """Cached voice preset data."""
    name: str
    embedding: torch.Tensor  # [N, 3072] on CPU
    n_frames: int
    prefix_tokens: list[int]  # [BOS, begin_audio, audio_tok × N]


class VoxtralTokenizer:
    """Builds Voxtral TTS prompts from text + voice preset.

    Pre-loads all 20 voice embeddings and tokenizer at init time
    for zero-latency preprocessing.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)

        # Load Tekken tokenizer
        self._tokenizer = self._load_tekken_tokenizer()

        # Pre-load all voice presets
        self._voices: dict[str, VoicePreset] = {}
        self._load_voice_presets()

        # Ensure unique prefix lengths for RadixCache
        self._deduplicate_prefix_lengths()

    def _load_tekken_tokenizer(self) -> Any:
        """Load the Mistral Tekken tokenizer."""
        tekken_path = self._model_path / "tekken.json"

        # Try mistral_common first (best compatibility)
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            tok = MistralTokenizer.from_file(str(tekken_path))
            logger.info("Loaded Mistral Tekken tokenizer via mistral_common")
            return tok
        except (ImportError, Exception) as e:
            logger.debug("mistral_common not available: %s", e)

        # Fallback: use transformers' PreTrainedTokenizerFast
        try:
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast(tokenizer_file=str(tekken_path))
            logger.info("Loaded Tekken tokenizer via transformers (fallback)")
            return tok
        except Exception as e:
            logger.warning("Failed to load Tekken tokenizer: %s", e)
            return None

    def _load_voice_presets(self) -> None:
        """Pre-load all 20 voice .pt files."""
        voice_dir = self._model_path / "voice_embedding"
        if not voice_dir.is_dir():
            logger.warning("Voice embedding directory not found: %s", voice_dir)
            return

        for name in VOXTRAL_VOICE_PRESETS:
            pt_path = voice_dir / f"{name}.pt"
            if not pt_path.exists():
                logger.warning("Voice embedding not found: %s", pt_path)
                continue

            embedding = torch.load(str(pt_path), map_location="cpu", weights_only=False)
            if not isinstance(embedding, torch.Tensor):
                logger.warning("Unexpected voice embedding type for %s: %s", name, type(embedding))
                continue

            n_frames = embedding.shape[0]
            prefix_tokens = [BOS_TOKEN_ID, BEGIN_AUDIO_TOKEN_ID] + [AUDIO_TOKEN_ID] * n_frames

            self._voices[name] = VoicePreset(
                name=name,
                embedding=embedding,
                n_frames=n_frames,
                prefix_tokens=prefix_tokens,
            )
            logger.debug("Loaded voice preset: %s (%d frames)", name, n_frames)

        logger.info("Loaded %d/%d voice presets", len(self._voices), len(VOXTRAL_VOICE_PRESETS))

    def _deduplicate_prefix_lengths(self) -> None:
        """Pad voice embeddings to ensure unique prefix lengths for RadixCache.

        If two voices have the same frame count, their token prefixes are
        identical ([BOS, begin_audio, audio_tok × N]) and RadixCache would
        serve wrong KV states. We pad one voice by 1 frame to break ties.
        """
        length_map: dict[int, list[str]] = {}
        for name, preset in self._voices.items():
            length_map.setdefault(preset.n_frames, []).append(name)

        for n_frames, names in length_map.items():
            if len(names) <= 1:
                continue
            # Pad all but the first voice in this collision group
            for name in names[1:]:
                preset = self._voices[name]
                logger.info(
                    "Padding voice '%s' from %d to %d frames for RadixCache uniqueness",
                    name, preset.n_frames, preset.n_frames + 1,
                )
                # Duplicate last frame
                pad_frame = preset.embedding[-1:].clone()
                preset.embedding = torch.cat([preset.embedding, pad_frame], dim=0)
                preset.n_frames += 1
                preset.prefix_tokens.append(AUDIO_TOKEN_ID)

    @property
    def voice_names(self) -> list[str]:
        return list(self._voices.keys())

    def get_voice(self, name: str) -> VoicePreset | None:
        return self._voices.get(name)

    @property
    def eos_token_id(self) -> int:
        return EOS_TOKEN_ID

    @property
    def audio_token_id(self) -> int:
        return AUDIO_TOKEN_ID

    @property
    def vocab_size(self) -> int:
        return 131072

    def encode_text(self, text: str) -> list[int]:
        """Tokenize text using the Tekken tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded")

        # mistral_common tokenizer: access inner Tekkenizer via instruct_tokenizer
        if hasattr(self._tokenizer, "instruct_tokenizer"):
            inner = self._tokenizer.instruct_tokenizer.tokenizer
            tokens = inner.encode(text, bos=False, eos=False)
            return list(tokens)

        # HF tokenizer fallback
        if hasattr(self._tokenizer, "encode"):
            return self._tokenizer.encode(text, add_special_tokens=False)

        raise RuntimeError("Tokenizer has no encode method")

    def build_prompt(
        self,
        text: str,
        voice: str | None = None,
        voice_embedding: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Build a Voxtral TTS prompt.

        Format: [BOS] [begin_audio] [audio_tok × N] [INST] text [/INST] [begin_audio]

        Returns:
            dict with keys:
              - input_ids: list[int]
              - voice_embedding: Tensor [N, 3072] or None
              - audio_token_mask: list[bool] (True at audio_token positions)
        """
        # Resolve voice embedding
        voice_emb = voice_embedding
        voice_n_frames = 0
        if voice is not None and voice_emb is None:
            preset = self._voices.get(voice)
            if preset is not None:
                voice_emb = preset.embedding
                voice_n_frames = preset.n_frames
            else:
                logger.warning("Unknown voice preset: %s", voice)

        if voice_emb is not None and voice_n_frames == 0:
            voice_n_frames = voice_emb.shape[0]

        # Encode text
        text_tokens = self.encode_text(text)

        # Build token sequence
        tokens: list[int] = [BOS_TOKEN_ID]

        if voice_n_frames > 0:
            # Voice reference section
            tokens.append(BEGIN_AUDIO_TOKEN_ID)
            tokens.extend([AUDIO_TOKEN_ID] * voice_n_frames)

        # Instruction section
        tokens.append(INST_START_TOKEN_ID)
        tokens.extend(text_tokens)
        tokens.append(INST_END_TOKEN_ID)

        # Generation start marker
        tokens.append(BEGIN_AUDIO_TOKEN_ID)

        # Build audio token mask (True where voice embedding should be injected)
        audio_mask = [t == AUDIO_TOKEN_ID for t in tokens]

        return {
            "input_ids": tokens,
            "voice_embedding": voice_emb,
            "audio_token_mask": audio_mask,
        }
