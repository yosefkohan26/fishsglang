#!/usr/bin/env python3
"""Generate TTS samples and save raw chunks + combined WAV per request.

Creates a timestamped folder under audio_samples/ with:
  - req_N_chunk_M.wav   (individual raw WAV chunks from SSE)
  - req_N_combined.wav  (all chunks concatenated)
  - req_N_meta.json     (token counts, timing, finish_reason)
  - run_info.json       (texts, voice, timestamp)

Usage:
  python test_and_save.py                          # defaults
  python test_and_save.py --voice jackson
  python test_and_save.py --texts "Hello world" "Another sentence"
  python test_and_save.py --repeat 3               # repeat each text 3 times
"""

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from datetime import datetime

import httpx
import numpy as np


DEFAULT_TEXTS = [
    "Hello, this is Jackson calling about the energy efficiency assessment for your home.",
    "I wanted to let you know that we have some exciting new solar panel options available for your property.",
    "Could you give me a call back when you get a chance? I would love to discuss the details with you.",
    "Our team has been working on a custom proposal that I think you will really like. We can schedule a time to review it.",
    "Thank you so much for your time today, and I hope to hear from you soon. Have a great afternoon.",
]

BASE_DIR = "/home/yosef-kohan/tts/audio_samples"


async def generate_one(client, text, voice, run_dir, req_idx):
    t0 = time.perf_counter()
    t_first = None
    chunks_raw = []  # list of (raw_bytes, sample_rate)
    n_tokens = 0
    finish_reason = None
    prompt_tokens = 0
    chunk_idx = 0

    async with client.stream(
        "POST",
        "http://localhost:8000/v1/audio/speech",
        json={"input": text, "voice_id": voice, "stream": True},
        timeout=120.0,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            audio = data.get("audio")
            usage = data.get("usage")
            fr = data.get("finish_reason")
            if fr:
                finish_reason = fr
            if usage:
                n_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
            if audio and audio.get("data"):
                if t_first is None:
                    t_first = time.perf_counter()
                raw = base64.b64decode(audio["data"])
                sr = audio.get("sample_rate", 44100)
                chunks_raw.append((raw, sr))
                # Save individual chunk as-is
                chunk_path = os.path.join(run_dir, f"req_{req_idx}_chunk_{chunk_idx}.wav")
                with open(chunk_path, "wb") as f:
                    f.write(raw)
                chunk_idx += 1

    total_time = time.perf_counter() - t0
    ttfa = (t_first - t0) if t_first else None

    # Combine chunks into one WAV
    pcm_parts = []
    sr_final = 44100
    for raw, sr in chunks_raw:
        sr_final = sr
        # Skip WAV header (44 bytes)
        pcm = np.frombuffer(raw[44:], dtype=np.int16)
        pcm_parts.append(pcm)

    combined_path = os.path.join(run_dir, f"req_{req_idx}_combined.wav")
    duration = 0.0
    if pcm_parts:
        combined = np.concatenate(pcm_parts)
        duration = len(combined) / sr_final
        with wave.open(combined_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr_final)
            w.writeframes(combined.tobytes())

    # Save metadata
    meta = {
        "req_idx": req_idx,
        "text": text,
        "voice": voice,
        "completion_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
        "num_audio_chunks": chunk_idx,
        "duration_s": round(duration, 2),
        "ttfa_ms": round(ttfa * 1000, 1) if ttfa else None,
        "total_time_ms": round(total_time * 1000, 1),
    }
    meta_path = os.path.join(run_dir, f"req_{req_idx}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    status = "OK" if n_tokens > 50 else "SHORT"
    print(
        f"  R{req_idx}: {n_tokens:>4} tokens | {duration:.1f}s | {chunk_idx} chunks | "
        f"TTFA={meta['ttfa_ms']}ms | {finish_reason} | {status} | \"{text[:55]}...\""
    )
    return meta


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="jackson")
    parser.add_argument("--texts", nargs="+", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    texts = args.texts or DEFAULT_TEXTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    run_info = {
        "timestamp": timestamp,
        "voice": args.voice,
        "texts": texts,
        "repeat": args.repeat,
        "url": args.url,
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Run: {run_dir}")
    print(f"Voice: {args.voice} | Texts: {len(texts)} | Repeat: {args.repeat}")
    print()

    all_meta = []
    req_idx = 0
    async with httpx.AsyncClient() as client:
        for rep in range(args.repeat):
            if args.repeat > 1:
                print(f"--- Repeat {rep + 1}/{args.repeat} ---")
            for text in texts:
                meta = await generate_one(client, text, args.voice, run_dir, req_idx)
                all_meta.append(meta)
                req_idx += 1

    # Summary
    print()
    short = [m for m in all_meta if m["completion_tokens"] <= 50]
    if short:
        print(f"WARNING: {len(short)} requests had <= 50 tokens (likely cut short):")
        for m in short:
            print(f"  R{m['req_idx']}: {m['completion_tokens']} tokens - \"{m['text'][:60]}\"")
    else:
        print(f"All {len(all_meta)} requests completed with >50 tokens.")

    # Save summary
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\nFiles saved to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
