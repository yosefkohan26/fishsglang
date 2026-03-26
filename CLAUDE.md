# SGLang-Omni

To load all source files into context memory, read `fishsglang/readallfiles/readthistoreadallfiles.md`. That file sits next to a CLAUDE.md with `@` references to every source file, so reading it automatically pulls the full codebase into context.

## Testing Protocol

**Always commit before benchmarking.** The benchmark script tracks git commit hashes so we can trace every result to exact code. Workflow:

1. Make your changes
2. `git add` and `git commit` with a descriptive message
3. Start the server: `python -m sglang_omni.cli.cli serve --model-path fishaudio/s2-pro --port 8000`
4. Run the benchmark:
   ```
   python benchmark_ttfa.py -d "description of what changed"
   ```
5. Results append to `benchmark_results.csv` with git commit, GPU, timestamps

The benchmark enforces a clean git state by default. Use `--allow-dirty` to override during development.

### Benchmark options
```
python benchmark_ttfa.py -d "my change"                    # full suite with voice
python benchmark_ttfa.py -d "my change" --no-voice         # without voice cloning
python benchmark_ttfa.py -d "quick test" --concurrencies 1 4 8  # subset
```

### Key files for performance work
- `sglang_omni/engines/omni/engine.py` — engine loop, batch window, event-driven wakeup
- `sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py` — voice cache, codec warmup, stream vocode, flush
- `sglang_omni/models/fishaudio_s2_pro/pipeline/state_io.py` — tensor serialization bypass
- `sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py` — output codes clone, iteration controller
- `sglang_omni/models/fishaudio_s2_pro/fish_speech/content_sequence.py` — semantic token LUT
- `sglang_omni/serve/openai_api.py` — voice cache API, warmup endpoint
- `sglang_omni/serve/protocol.py` — voice_id field
