#!/usr/bin/env python3
"""TTFA benchmark for S2-Pro streaming TTS.

Measures Time to First Audio across concurrency levels with voice cloning.
Each run creates a folder under benchmark_runs/ with:
  - results.csv (appended globally to benchmark_results.csv too)
  - WAV files for every request at every concurrency level
  - run_info.json with full metadata

Usage:
    python benchmark_ttfa.py -d "baseline BF16 no optimizations"
    python benchmark_ttfa.py -d "FP8 KV cache" --no-voice
    python benchmark_ttfa.py -d "quick test" --concurrencies 1 2 4

Requirements:
    - Server must be running on --port (default 8000)
    - For voice tests: benchmark_audio/reference.wav and reference.txt must exist
    - git add and commit your changes BEFORE running (enforced by default)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8000
DEFAULT_CSV = "benchmark_results.csv"
RUNS_DIR = "benchmark_runs"
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
WARMUP_ROUNDS = 3
REF_AUDIO = "benchmark_audio/reference.wav"
REF_TEXT_FILE = "benchmark_audio/reference.txt"

# ~10s of speech — intentionally different from reference audio to test
# novel content generation, not parroting. Keep consistent across runs.
DEFAULT_TEXT = (
    "Good afternoon. I wanted to follow up on our conversation from last week "
    "regarding the solar panel installation for your property. We have a few "
    "different options available that I think would work really well for your "
    "situation, and I would love to walk you through them whenever you have a moment."
)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def encode_wav(samples: np.ndarray, sr: int = 44100) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    buf = BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        commit = result.stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if diff.stdout.strip():
            commit += "-dirty"
        return commit
    except Exception:
        return "unknown"


def get_git_message() -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:80]
    except Exception:
        return ""


def is_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def get_gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().replace("\n", " | ")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


async def generate_one(
    client: httpx.AsyncClient, url: str, text: str, voice_id: str | None,
) -> dict:
    """Send a single streaming TTS request, collect audio chunks and TTFA."""
    payload: dict = {"input": text, "stream": True}
    if voice_id:
        payload["voice_id"] = voice_id

    t0 = time.perf_counter()
    t_first = None
    sr = 44100
    audio_chunks: list[np.ndarray] = []
    error = None

    try:
        async with client.stream(
            "POST", f"{url}/v1/audio/speech", json=payload, timeout=180.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                audio = data.get("audio")
                if audio and audio.get("data"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    raw = base64.b64decode(audio["data"])
                    sr = audio.get("sample_rate", 44100)
                    arr = np.frombuffer(
                        raw[44:], dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    audio_chunks.append(arr)
    except Exception as e:
        error = str(e)[:120]

    elapsed = time.perf_counter() - t0
    ttfa_ms = (t_first - t0) * 1000 if t_first else None
    combined = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
    audio_s = len(combined) / sr if sr > 0 else 0

    return {
        "ttfa_ms": ttfa_ms,
        "total_s": elapsed,
        "audio_s": audio_s,
        "sample_rate": sr,
        "samples": combined,
        "num_chunks": len(audio_chunks),
        "error": error,
    }


async def run_concurrency(
    url: str, text: str, voice_id: str | None, n: int,
) -> tuple[dict, list[dict]]:
    """Run n concurrent requests. Returns (aggregated_stats, per_request_results)."""
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [generate_one(client, url, text, voice_id) for _ in range(n)]
        results = list(await asyncio.gather(*tasks))
        wall = time.perf_counter() - t0

    ttfas = sorted(
        [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
    )
    errors = sum(1 for r in results if r["error"] is not None)
    total_audio = sum(r["audio_s"] for r in results)

    if not ttfas:
        stats = {
            "concurrency": n,
            "ttfa_avg_ms": None, "ttfa_p50_ms": None, "ttfa_p99_ms": None,
            "ttfa_min_ms": None, "ttfa_max_ms": None,
            "wall_s": round(wall, 2), "total_audio_s": round(total_audio, 1),
            "rtfx": 0, "errors": errors, "success": 0,
        }
    else:
        stats = {
            "concurrency": n,
            "ttfa_avg_ms": round(sum(ttfas) / len(ttfas), 1),
            "ttfa_p50_ms": round(ttfas[len(ttfas) // 2], 1),
            "ttfa_p99_ms": round(ttfas[-1], 1),
            "ttfa_min_ms": round(ttfas[0], 1),
            "ttfa_max_ms": round(ttfas[-1], 1),
            "wall_s": round(wall, 2),
            "total_audio_s": round(total_audio, 1),
            "rtfx": round(total_audio / wall, 1) if wall > 0 else 0,
            "errors": errors,
            "success": len(ttfas),
        }

    return stats, results


async def warmup(
    url: str, text: str, voice_id: str | None,
    ref_audio: str | None, ref_text: str | None, rounds: int = 3,
) -> None:
    """Warm up voice cache, semantic LUT, and RadixCache."""
    print("Warming up", end="", flush=True)

    if voice_id and ref_audio and os.path.exists(ref_audio):
        payload: dict = {
            "input": "Hello warmup.", "voice_id": voice_id,
            "ref_audio": ref_audio, "stream": True,
        }
        if ref_text:
            payload["ref_text"] = ref_text
        with httpx.stream(
            "POST", f"{url}/v1/audio/speech", json=payload, timeout=120,
        ) as r:
            for _ in r.iter_lines():
                pass
        print(".", end="", flush=True)

    for _ in range(rounds):
        payload = {"input": text, "stream": True}
        if voice_id:
            payload["voice_id"] = voice_id
        with httpx.stream(
            "POST", f"{url}/v1/audio/speech", json=payload, timeout=60,
        ) as r:
            for _ in r.iter_lines():
                pass
        print(".", end="", flush=True)

    print(" done")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp", "git_commit", "git_message", "description", "gpu",
    "voice_cloning", "run_dir", "concurrency",
    "ttfa_avg_ms", "ttfa_p50_ms", "ttfa_p99_ms", "ttfa_min_ms", "ttfa_max_ms",
    "wall_s", "total_audio_s", "rtfx", "errors", "success",
]


def write_csv_row(csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="TTFA benchmark for S2-Pro TTS")
    parser.add_argument(
        "--description", "-d", required=True,
        help="What changed in this run (required)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Global CSV path")
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=DEFAULT_CONCURRENCIES,
    )
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice-id", default="joseph")
    parser.add_argument("--warmup-rounds", type=int, default=WARMUP_ROUNDS)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    # Enforce clean git state
    if is_git_dirty() and not args.allow_dirty:
        print("ERROR: You have uncommitted changes.")
        print("       Commit first so the benchmark tracks the exact code state.")
        print("       Or use --allow-dirty to override.")
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    voice_id = None if args.no_voice else args.voice_id
    voice_cloning = not args.no_voice

    # Check server health
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        health = resp.json()
        if not health.get("running"):
            print(f"Server not healthy: {health}")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach server at {url}: {e}")
        sys.exit(1)

    # System info
    git_commit = get_git_commit()
    git_message = get_git_message()
    gpu = get_gpu_info()
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat(timespec="seconds")

    # Create run directory: benchmark_runs/<timestamp>_<commit>_<slug>/
    slug = args.description[:40].replace(" ", "_").replace("/", "-")
    run_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{git_commit}_{slug}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"{'=' * 65}")
    print(f"  TTFA Benchmark")
    print(f"{'=' * 65}")
    print(f"  Description:    {args.description}")
    print(f"  Git commit:     {git_commit}")
    print(f"  Git message:    {git_message}")
    print(f"  GPU:            {gpu}")
    print(f"  Voice cloning:  {voice_cloning}")
    print(f"  Concurrencies:  {args.concurrencies}")
    print(f"  Run directory:  {run_dir}")
    print(f"  CSV output:     {args.csv}")
    print(f"{'=' * 65}\n")

    # Save run metadata
    run_info = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "git_message": git_message,
        "description": args.description,
        "gpu": gpu,
        "voice_cloning": voice_cloning,
        "voice_id": voice_id,
        "text": args.text,
        "concurrencies": args.concurrencies,
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    # Load ref text
    ref_text = None
    if os.path.exists(REF_TEXT_FILE):
        ref_text = open(REF_TEXT_FILE).read().strip()
    ref_audio_path = REF_AUDIO if os.path.exists(REF_AUDIO) else None

    # Warmup
    await warmup(
        url, args.text, voice_id, ref_audio_path, ref_text,
        rounds=args.warmup_rounds,
    )

    # Run benchmarks
    header = (
        f"{'Conc':>5} | {'TTFA avg':>9} | {'TTFA p50':>9} | "
        f"{'TTFA p99':>9} | {'Wall':>6} | {'Audio':>7} | "
        f"{'RTFx':>6} | Err"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for n in args.concurrencies:
        stats, results = await run_concurrency(url, args.text, voice_id, n)

        # Print summary
        if stats["ttfa_avg_ms"] is not None:
            print(
                f"{n:5d} | {stats['ttfa_avg_ms']:7.0f}ms | "
                f"{stats['ttfa_p50_ms']:7.0f}ms | "
                f"{stats['ttfa_p99_ms']:7.0f}ms | "
                f"{stats['wall_s']:4.1f}s | "
                f"{stats['total_audio_s']:5.0f}s | "
                f"{stats['rtfx']:5.1f}x | {stats['errors']}"
            )
        else:
            print(
                f"{n:5d} |      FAIL |           |           | "
                f"{stats['wall_s']:4.1f}s |       |       | {stats['errors']}"
            )

        # Save audio files: benchmark_runs/<run>/conc_<N>/req_<i>.wav
        conc_dir = os.path.join(run_dir, f"conc_{n:03d}")
        os.makedirs(conc_dir, exist_ok=True)
        for i, r in enumerate(results):
            if r["samples"] is not None and len(r["samples"]) > 0:
                wav_path = os.path.join(conc_dir, f"req_{i:03d}.wav")
                with open(wav_path, "wb") as f:
                    f.write(encode_wav(r["samples"], r["sample_rate"]))

        # Write per-request details to run CSV
        run_csv = os.path.join(run_dir, "details.csv")
        run_csv_exists = os.path.exists(run_csv)
        with open(run_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "concurrency", "request_idx", "ttfa_ms", "total_s",
                "audio_s", "num_chunks", "error", "wav_file",
            ])
            if not run_csv_exists:
                writer.writeheader()
                run_csv_exists = True
            for i, r in enumerate(results):
                wav_name = f"conc_{n:03d}/req_{i:03d}.wav" if r["audio_s"] > 0 else ""
                writer.writerow({
                    "concurrency": n,
                    "request_idx": i,
                    "ttfa_ms": round(r["ttfa_ms"], 1) if r["ttfa_ms"] else None,
                    "total_s": round(r["total_s"], 2),
                    "audio_s": round(r["audio_s"], 1),
                    "num_chunks": r["num_chunks"],
                    "error": r["error"],
                    "wav_file": wav_name,
                })

        # Write to global CSV
        row = {
            "timestamp": timestamp,
            "git_commit": git_commit,
            "git_message": git_message,
            "description": args.description,
            "gpu": gpu,
            "voice_cloning": voice_cloning,
            "run_dir": run_dir,
            **stats,
        }
        write_csv_row(args.csv, row)

        await asyncio.sleep(0.5)

    print(f"\nAudio files saved to {run_dir}/")
    print(f"Results appended to {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
