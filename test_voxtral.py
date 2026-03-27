#!/usr/bin/env python3
"""Standalone test: load Voxtral TTS on GPU and generate audio from a prompt.

This bypasses the SGLang backend and tests the core components directly:
  1. Load all weights (LLM backbone + acoustic transformer + codec decoder)
  2. Build prompt with voice preset
  3. Run autoregressive generation (prefill + decode loop)
  4. Vocode to audio and save WAV
"""

import json
import time
import struct
import io

import torch
from safetensors import safe_open

MODEL_PATH = "voxtral_model"
DEVICE = "cuda:0"
VOICE = "casual_male"
TEXT = "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"
MAX_TOKENS = 200  # ~16 seconds of audio at 12.5 Hz


def encode_wav(audio_np, sample_rate=24000):
    """Encode float32 audio to WAV bytes."""
    import numpy as np
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm = (audio_np * 32767.0).astype(np.int16)
    pcm_bytes = pcm.tobytes()
    num_channels = 1
    bits = 16
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm_bytes)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm_bytes)))
    buf.write(pcm_bytes)
    return buf.getvalue()


def main():
    print(f"=== Voxtral TTS Standalone Test ===")
    print(f"Device: {DEVICE}")
    print(f"Voice: {VOICE}")
    print(f"Text: {TEXT[:80]}...")
    print()

    # ---------------------------------------------------------------
    # 1. Load tokenizer + build prompt
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    from sglang_omni.models.voxtral_tts.tokenizer import VoxtralTokenizer
    tok = VoxtralTokenizer(MODEL_PATH)
    prompt = tok.build_prompt(TEXT, voice=VOICE)
    input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long, device=DEVICE)
    voice_emb = prompt["voice_embedding"].to(DEVICE, dtype=torch.bfloat16)
    audio_mask = torch.tensor(prompt["audio_token_mask"], dtype=torch.bool, device=DEVICE)
    print(f"[1] Prompt: {len(prompt['input_ids'])} tokens, voice_emb={list(voice_emb.shape)} ({time.perf_counter()-t0:.2f}s)")

    # ---------------------------------------------------------------
    # 2. Load LLM backbone (raw transformer, no SGLang)
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    with open(f"{MODEL_PATH}/params.json") as f:
        params = json.load(f)

    # Build a simple transformer for direct inference (no paged attention)
    from sglang_omni.models.voxtral_tts.acoustic_transformer import FlowMatchingAcousticTransformer
    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralAudioTokenizer

    # Use the HF model for the LLM backbone (simpler than SGLang for testing)
    # We'll build a minimal causal LM from the weights
    from transformers import PretrainedConfig, MistralForCausalLM

    from transformers import MistralConfig
    hf_config = MistralConfig(
        vocab_size=params["vocab_size"],
        hidden_size=params["dim"],
        intermediate_size=params["hidden_dim"],
        num_hidden_layers=params["n_layers"],
        num_attention_heads=params["n_heads"],
        num_key_value_heads=params["n_kv_heads"],
        head_dim=params["head_dim"],
        max_position_embeddings=params.get("max_position_embeddings", 128000),
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        hidden_act="silu",
    )

    print(f"  Loading Mistral LLM ({params['n_layers']} layers, dim={params['dim']})...")
    llm = MistralForCausalLM(hf_config).to(dtype=torch.bfloat16, device=DEVICE)

    # Load weights with remapping
    with safe_open(f"{MODEL_PATH}/consolidated.safetensors", framework="pt", device="cpu") as f:
        llm_state = {}
        for key in f.keys():
            if key.startswith("layers."):
                parts = key.split(".")
                layer_idx = parts[1]
                rest = ".".join(parts[2:])
                remap = {
                    "attention.wq.weight": "self_attn.q_proj.weight",
                    "attention.wk.weight": "self_attn.k_proj.weight",
                    "attention.wv.weight": "self_attn.v_proj.weight",
                    "attention.wo.weight": "self_attn.o_proj.weight",
                    "attention_norm.weight": "input_layernorm.weight",
                    "feed_forward.w1.weight": "mlp.gate_proj.weight",
                    "feed_forward.w3.weight": "mlp.up_proj.weight",
                    "feed_forward.w2.weight": "mlp.down_proj.weight",
                    "ffn_norm.weight": "post_attention_layernorm.weight",
                }
                if rest in remap:
                    new_key = f"model.layers.{layer_idx}.{remap[rest]}"
                    llm_state[new_key] = f.get_tensor(key)
            elif key == "norm.weight":
                llm_state["model.norm.weight"] = f.get_tensor(key)
            elif key == "mm_audio_embeddings.tok_embeddings.weight":
                llm_state["model.embed_tokens.weight"] = f.get_tensor(key)
                llm_state["lm_head.weight"] = f.get_tensor(key)  # tied

        # Load audio codebook embedding separately
        audio_emb_key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
        audio_emb_weight = f.get_tensor(audio_emb_key) if audio_emb_key in f.keys() else None

    llm.load_state_dict(llm_state, strict=False)
    llm.eval()
    print(f"  LLM loaded ({time.perf_counter()-t0:.2f}s)")

    # ---------------------------------------------------------------
    # 3. Load acoustic transformer
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    audio_model_args = params["multimodal"]["audio_model_args"]
    acoustic = FlowMatchingAcousticTransformer(audio_model_args)
    with safe_open(f"{MODEL_PATH}/consolidated.safetensors", framework="pt", device="cpu") as f:
        w = {k[len("acoustic_transformer."):]: f.get_tensor(k)
             for k in f.keys() if k.startswith("acoustic_transformer.")}
    acoustic.load_weights(w)
    acoustic = acoustic.to(DEVICE, dtype=torch.bfloat16).eval()
    acoustic.setup_buffers(1, torch.device(DEVICE), torch.bfloat16)
    print(f"[2] Acoustic transformer loaded ({time.perf_counter()-t0:.2f}s)")

    # ---------------------------------------------------------------
    # 4. Load codec decoder
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    codec = VoxtralAudioTokenizer.from_params(params)
    with safe_open(f"{MODEL_PATH}/consolidated.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("audio_tokenizer."):
                codec.load_weight((k[len("audio_tokenizer."):], f.get_tensor(k)))
    codec = codec.to(DEVICE).eval()
    # Warmup
    with torch.no_grad():
        _ = codec.decode(torch.randint(0, 10, (1, 37, 2), device=DEVICE))
    print(f"[3] Codec decoder loaded + warmed up ({time.perf_counter()-t0:.2f}s)")

    # ---------------------------------------------------------------
    # 5. Build input embeddings with voice injection
    # ---------------------------------------------------------------
    with torch.no_grad():
        input_embeds = llm.model.embed_tokens(input_ids.unsqueeze(0))  # [1, seq, dim]
        if voice_emb is not None and audio_mask.any():
            n_audio = int(audio_mask.sum().item())
            input_embeds[0, audio_mask] = voice_emb[:n_audio].to(input_embeds.dtype)

    print(f"[4] Input embeds: {list(input_embeds.shape)}")

    # ---------------------------------------------------------------
    # 6. Autoregressive generation
    # ---------------------------------------------------------------
    print(f"\n--- Generating (max {MAX_TOKENS} tokens) ---")
    END_AUDIO_ID = 1  # acoustic transformer's END_AUDIO special token

    all_codes = []
    past_key_values = None
    gen_input_embeds = input_embeds
    gen_input_ids = None

    t_gen_start = time.perf_counter()
    t_first_token = None

    for step in range(MAX_TOKENS):
        with torch.no_grad():
            # Run transformer backbone only (no lm_head — we need hidden states)
            if gen_input_embeds is not None:
                model_out = llm.model(
                    inputs_embeds=gen_input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                model_out = llm.model(
                    input_ids=gen_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = model_out.past_key_values
            last_hidden = model_out.last_hidden_state[:, -1:, :]  # [1, 1, dim]

            # Run acoustic transformer on last hidden state
            sem, aco = acoustic.decode_one_frame(last_hidden.squeeze(1))
            codes = torch.cat([sem, aco], dim=-1)[0]  # [37]
            all_codes.append(codes.cpu())

            if t_first_token is None:
                t_first_token = time.perf_counter()
                print(f"  First token: {(t_first_token - t_gen_start)*1000:.0f}ms (prefill)")

            # Check EOS
            if sem[0, 0].item() == END_AUDIO_ID:
                print(f"  EOS at step {step+1}")
                break

            # Next input: audio_token_id as placeholder
            gen_input_ids = torch.tensor([[24]], device=DEVICE)
            gen_input_embeds = None

        if (step + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_gen_start
            audio_ms = (step + 1) * 80
            print(f"  Step {step+1}: {audio_ms}ms audio generated in {elapsed:.1f}s (RTF={elapsed/(audio_ms/1000):.3f})")

    t_gen_end = time.perf_counter()
    gen_time = t_gen_end - t_gen_start
    n_frames = len(all_codes)
    audio_duration = n_frames * 80 / 1000
    print(f"\n  Generated {n_frames} frames = {audio_duration:.1f}s audio in {gen_time:.1f}s")
    print(f"  RTF: {gen_time/audio_duration:.3f}")

    # ---------------------------------------------------------------
    # 7. Vocode to audio
    # ---------------------------------------------------------------
    print(f"\n--- Vocoding ---")
    t0 = time.perf_counter()
    code_tensor = torch.stack(all_codes, dim=-1).unsqueeze(0).to(DEVICE)  # [1, 37, T]
    with torch.no_grad():
        audio = codec.decode(code_tensor)
    audio_np = audio[0, 0].float().cpu().numpy()
    print(f"  Vocoded: {len(audio_np)} samples = {len(audio_np)/24000:.2f}s ({time.perf_counter()-t0:.2f}s)")

    # ---------------------------------------------------------------
    # 8. Save WAV
    # ---------------------------------------------------------------
    wav_bytes = encode_wav(audio_np, 24000)
    out_path = "voxtral_test_output.wav"
    with open(out_path, "wb") as f:
        f.write(wav_bytes)
    print(f"\n  Saved: {out_path} ({len(wav_bytes)/1024:.1f} KB)")

    print(f"\n=== Done ===")
    print(f"  Text: {TEXT}")
    print(f"  Voice: {VOICE}")
    print(f"  Audio: {audio_duration:.1f}s at 24kHz")
    print(f"  Generation: {gen_time:.1f}s (RTF={gen_time/audio_duration:.3f})")


if __name__ == "__main__":
    main()
