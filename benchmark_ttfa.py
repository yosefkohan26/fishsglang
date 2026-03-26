#!/usr/bin/env python3
"""TTFA benchmark for S2-Pro streaming TTS.

Measures Time to First Audio across concurrency levels with voice cloning.
Writes results to a CSV for tracking across code changes.

Usage:
    python benchmark_ttfa.py -d "baseline BF16 no optimizations"
    python benchmark_ttfa.py -d "added codec lock" --csv results.csv
    python benchmark_ttfa.py -d "FP8 KV cache" --no-voice
    python benchmark_ttfa.py -d "test run" --concurrencies 1 2 4

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
import subprocess
import sys
import time
from datetime import datetime, timezone

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8000
DEFAULT_CSV = "benchmark_results.csv"
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
WARMUP_ROUNDS = 3
REF_AUDIO = "benchmark_audio/reference.wav"
REF_TEXT_FILE = "benchmark_audio/reference.txt"

# ~10s of speech
DEFAULT_TEXT = (
    "Hi, my name is Joseph, and I am calling you from the Energy Efficiency Agency. "
    "The reason for my call today is because I wanted to see if I can provide you any "
    "services that will benefit and better your experience working in our industry. "
    "Please call me back at any time."
)


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
    """Send a single streaming TTS request and measure TTFA."""
    payload: dict = {"input": text, "stream": True}
    if voice_id:
        payload["voice_id"] = voice_id

    t0 = time.perf_counter()
    t_first = None
    total_samples = 0
    sr = 44100
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
                    total_samples += (len(raw) - 44) // 2
    except Exception as e:
        error = str(e)[:120]

    elapsed = time.perf_counter() - t0
    ttfa_ms = (t_first - t0) * 1000 if t_first else None
    audio_s = total_samples / sr if sr > 0 else 0

    return {
        "ttfa_ms": ttfa_ms,
        "total_s": elapsed,
        "audio_s": audio_s,
        "error": error,
    }


async def run_concurrency(
    url: str, text: str, voice_id: str | None, n: int,
) -> dict:
    """Run n concurrent requests and return aggregated stats."""
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        tasks = [generate_one(client, url, text, voice_id) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    ttfas = sorted(
        [r["ttfa_ms"] for r in results if r["ttfa_ms"] is not None]
    )
    errors = sum(1 for r in results if r["error"] is not None)
    total_audio = sum(r["audio_s"] for r in results)

    if not ttfas:
        return {
            "concurrency": n,
            "ttfa_avg_ms": None,
            "ttfa_p50_ms": None,
            "ttfa_p99_ms": None,
            "ttfa_min_ms": None,
            "ttfa_max_ms": None,
            "wall_s": round(wall, 2),
            "total_audio_s": round(total_audio, 1),
            "rtfx": 0,
            "errors": errors,
            "success": 0,
        }

    return {
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


async def warmup(
    url: str,
    text: str,
    voice_id: str | None,
    ref_audio: str | None,
    ref_text: str | None,
    rounds: int = 3,
) -> None:
    """Warm up voice cache, semantic LUT, and RadixCache."""
    print("Warming up", end="", flush=True)

    # First request: cache voice if using voice cloning with ref_audio
    if voice_id and ref_audio and os.path.exists(ref_audio):
        payload: dict = {
            "input": "Hello warmup.",
            "voice_id": voice_id,
            "ref_audio": ref_audio,
            "stream": True,
        }
        if ref_text:
            payload["ref_text"] = ref_text
        with httpx.stream(
            "POST", f"{url}/v1/audio/speech", json=payload, timeout=120,
        ) as r:
            for _ in r.iter_lines():
                pass
        print(".", end="", flush=True)

    # Sequential requests to warm RadixCache + LUT
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
    "timestamp",
    "git_commit",
    "git_message",
    "description",
    "gpu",
    "voice_cloning",
    "concurrency",
    "ttfa_avg_ms",
    "ttfa_p50_ms",
    "ttfa_p99_ms",
    "ttfa_min_ms",
    "ttfa_max_ms",
    "wall_s",
    "total_audio_s",
    "rtfx",
    "errors",
    "success",
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
    parser = argparse.ArgumentParser(
        description="TTFA benchmark for S2-Pro TTS",
    )
    parser.add_argument(
        "--description", "-d", required=True,
        help="What changed in this run (required)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV output path")
    parser.add_argument(
        "--concurrencies", type=int, nargs="+",
        default=DEFAULT_CONCURRENCIES,
        help="Concurrency levels to test (default: 1 2 4 8 16 32 64)",
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Benchmark without voice cloning",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice-id", default="joseph")
    parser.add_argument("--warmup-rounds", type=int, default=WARMUP_ROUNDS)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Allow running with uncommitted changes",
    )
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
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"{'=' * 65}")
    print(f"  TTFA Benchmark")
    print(f"{'=' * 65}")
    print(f"  Description:    {args.description}")
    print(f"  Git commit:     {git_commit}")
    print(f"  Git message:    {git_message}")
    print(f"  GPU:            {gpu}")
    print(f"  Voice cloning:  {voice_cloning}")
    print(f"  Concurrencies:  {args.concurrencies}")
    print(f"  CSV output:     {args.csv}")
    print(f"{'=' * 65}\n")

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

    # Run
    header = (
        f"{'Conc':>5} | {'TTFA avg':>9} | {'TTFA p50':>9} | "
        f"{'TTFA p99':>9} | {'Wall':>6} | {'Audio':>7} | "
        f"{'RTFx':>6} | Err"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for n in args.concurrencies:
        stats = await run_concurrency(url, args.text, voice_id, n)

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
                f"{stats['wall_s']:4.1f}s |       |       | "
                f"{stats['errors']}"
            )

        # Write CSV
        row = {
            "timestamp": timestamp,
            "git_commit": git_commit,
            "git_message": git_message,
            "description": args.description,
            "gpu": gpu,
            "voice_cloning": voice_cloning,
            **stats,
        }
        write_csv_row(args.csv, row)

        await asyncio.sleep(0.5)

    print(f"\nResults appended to {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
