"""
Profile TTFA breakdown for Fish Audio S2-Pro.

Measures time spent in each stage of the TTS pipeline:
  1. Preprocessing (CPU): reference audio encoding, tokenization
  2. Prefill: processing prompt tokens through the Slow AR
  3. First decode step: generating first semantic token
  4. Codebook loop: Fast AR generating 9 acoustic tokens
  5. Vocoder decode: converting first codebook frame to audio
  6. Total TTFA: end-to-end from request to first audio chunk

Usage:
    python profile_ttfa.py --runs 5
"""

import argparse
import base64
import json
import os
import time

import requests


def profile_single_request(url: str, text: str, voice_ref: dict | None = None,
                           stream: bool = True) -> dict:
    """Send a request and measure TTFA precisely."""
    payload = {
        "input": text,
        "model": "fishaudio/s2-pro",
        "stream": stream,
    }
    if voice_ref:
        payload.update(voice_ref)

    t_submit = time.perf_counter()

    if stream:
        resp = requests.post(f"{url}/v1/audio/speech", json=payload, stream=True)
        t_headers = time.perf_counter()

        t_first_chunk = None
        t_first_audio = None
        chunk_count = 0
        total_audio_bytes = 0

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="replace")
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk_count += 1
                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter()
                try:
                    data = json.loads(line[6:])
                    audio_data = data.get("audio", {})
                    if isinstance(audio_data, dict) and "data" in audio_data:
                        audio_bytes = base64.b64decode(audio_data["data"])
                        total_audio_bytes += len(audio_bytes)
                        if t_first_audio is None:
                            t_first_audio = time.perf_counter()
                except (json.JSONDecodeError, KeyError):
                    pass

        t_done = time.perf_counter()

        return {
            "total_ms": (t_done - t_submit) * 1000,
            "ttfb_ms": (t_headers - t_submit) * 1000,
            "ttfa_ms": ((t_first_audio or t_done) - t_submit) * 1000,
            "first_chunk_ms": ((t_first_chunk or t_done) - t_submit) * 1000,
            "generation_ms": (t_done - (t_first_audio or t_submit)) * 1000,
            "chunks": chunk_count,
            "audio_bytes": total_audio_bytes,
        }
    else:
        resp = requests.post(f"{url}/v1/audio/speech", json=payload)
        t_done = time.perf_counter()
        return {
            "total_ms": (t_done - t_submit) * 1000,
            "http_status": resp.status_code,
        }


def build_voice_ref(audio_path: str, text_path: str) -> dict:
    """Build voice reference payload from files."""
    with open(text_path, "r") as f:
        ref_text = f.read().strip()
    return {
        "voice_id": "profile_voice",
        "references": [{
            "audio_path": os.path.abspath(audio_path),
            "text": ref_text,
        }],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--ref-audio", default="benchmark_audio/reference.wav")
    parser.add_argument("--ref-text", default="benchmark_audio/reference.txt")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--short", action="store_true", help="Use very short text")
    args = parser.parse_args()

    if args.short:
        args.text = "Hi."

    # Health check
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        health = r.json()
        print(f"Server: {health.get('status', 'unknown')}")
    except Exception as e:
        print(f"Server not reachable at {args.url}: {e}")
        return

    voice_ref = None
    if not args.no_voice and os.path.exists(args.ref_audio):
        voice_ref = build_voice_ref(args.ref_audio, args.ref_text)
        print(f"Using voice reference: {args.ref_audio}")
    else:
        print("No voice reference (--no-voice or file not found)")

    print(f"Text: \"{args.text}\"")
    print(f"Runs: {args.runs}")
    print()

    # Warmup
    print("Warming up...", end=" ", flush=True)
    profile_single_request(args.url, "Warmup.", voice_ref)
    print("done")
    print()

    # Profile runs
    results = []
    print(f"{'Run':>4} | {'TTFA':>8} | {'TTFB':>8} | {'1st Chunk':>9} | {'Gen':>8} | {'Total':>8} | Chunks")
    print("-" * 75)

    for i in range(args.runs):
        r = profile_single_request(args.url, args.text, voice_ref)
        results.append(r)
        print(f"{i+1:4d} | {r['ttfa_ms']:7.1f}ms | {r['ttfb_ms']:7.1f}ms | "
              f"{r['first_chunk_ms']:8.1f}ms | {r['generation_ms']:7.1f}ms | "
              f"{r['total_ms']:7.1f}ms | {r['chunks']}")

    print("-" * 75)

    # Averages (skip first run as it may include compilation)
    skip = 1 if len(results) > 2 else 0
    avg_results = results[skip:]

    avg = {k: sum(r[k] for r in avg_results) / len(avg_results)
           for k in ['ttfa_ms', 'ttfb_ms', 'first_chunk_ms', 'generation_ms', 'total_ms']}

    print(f" AVG | {avg['ttfa_ms']:7.1f}ms | {avg['ttfb_ms']:7.1f}ms | "
          f"{avg['first_chunk_ms']:8.1f}ms | {avg['generation_ms']:7.1f}ms | "
          f"{avg['total_ms']:7.1f}ms |")
    print()
    print("Legend:")
    print("  TTFA      = Time to first audio byte (what matters for perceived latency)")
    print("  TTFB      = Time to first HTTP byte (SSE header)")
    print("  1st Chunk = Time to first SSE data chunk (may not contain audio)")
    print("  Gen       = Time from first audio to last chunk (generation throughput)")
    print("  Total     = End-to-end request time")
    print()

    # Breakdown analysis
    print("TTFA Breakdown (estimated):")
    print(f"  HTTP overhead:          ~{avg['ttfb_ms']:.0f}ms")
    print(f"  Preprocessing + Prefill: ~{avg['ttfa_ms'] - avg['ttfb_ms']:.0f}ms")
    print(f"    (includes: ref audio encode, tokenization, transformer prefill,")
    print(f"     first decode step, codebook loop, vocoder decode)")


if __name__ == "__main__":
    main()
