"""
Detailed TTFA trace: sends requests and parses server-side logs to show
the full waterfall from request arrival to first audio byte.

Usage:
    # Start server, then:
    python trace_request.py --concurrency 1
    python trace_request.py --concurrency 4
"""

import argparse
import asyncio
import base64
import json
import subprocess
import time

import httpx


async def send_request(client, url, text, voice_id, idx):
    """Send one streaming request, return timing info."""
    t0 = time.perf_counter()
    t_first_audio = None
    n_tokens = 0
    n_chunks = 0

    async with client.stream(
        "POST", f"{url}/v1/audio/speech",
        json={"input": text, "voice_id": voice_id, "stream": True},
        timeout=60.0,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            usage = data.get("usage")
            if usage:
                n_tokens = usage.get("completion_tokens", 0)
            audio = data.get("audio")
            if audio and audio.get("data"):
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                n_chunks += 1

    total = time.perf_counter() - t0
    ttfa = (t_first_audio - t0) if t_first_audio else total
    return {
        "idx": idx,
        "ttfa_ms": ttfa * 1000,
        "total_s": total,
        "tokens": n_tokens,
        "chunks": n_chunks,
        "text": text[:50],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--voice", default="jackson")
    args = parser.parse_args()

    texts = [
        "Hello, this is a test of the text to speech system.",
        "I wanted to follow up on our conversation from last week.",
        "Could you give me a call back when you get a chance?",
        "Our team has been working on a custom proposal for you.",
        "Thank you so much for your time today.",
        "The weather forecast looks great for the weekend ahead.",
        "Please review the attached documents at your earliest convenience.",
        "We are excited to announce our new product line launching next month.",
    ]

    # Mark log position
    marker = f"TRACE_START_{int(time.time())}"
    # Touch the log to mark our start point
    subprocess.run(
        ["bash", "-c", f'echo "{marker}" >> /tmp/sglang_fa3.log'],
        capture_output=True,
    )

    n = args.concurrency
    selected = texts[:n]

    print(f"Sending {n} concurrent request(s) with voice={args.voice}")
    print()

    async with httpx.AsyncClient() as client:
        tasks = [
            send_request(client, args.url, text, args.voice, i)
            for i, text in enumerate(selected)
        ]
        results = await asyncio.gather(*tasks)

    # Print client-side results
    print(f"{'Req':>4} | {'TTFA':>8} | {'Total':>7} | {'Tokens':>6} | {'Chunks':>6} | Text")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["idx"]):
        print(f"  R{r['idx']+1} | {r['ttfa_ms']:7.1f}ms | {r['total_s']:6.2f}s | {r['tokens']:6d} | {r['chunks']:6d} | {r['text']}")

    print()
    print("=" * 80)
    print("  SERVER-SIDE TRACE (all [PROFILE] and [TRACE] logs)")
    print("=" * 80)

    # Extract logs after our marker
    result = subprocess.run(
        ["bash", "-c", f"sed -n '/{marker}/,$p' /tmp/sglang_fa3.log | grep '\\[PROFILE\\]\\|\\[TRACE\\]'"],
        capture_output=True, text=True,
    )
    print(result.stdout)


if __name__ == "__main__":
    asyncio.run(main())
