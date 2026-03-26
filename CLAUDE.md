# SGLang-Omni

To load all source files into context memory, read `fishsglang/readallfiles/readthistoreadallfiles.md`. That file sits next to a CLAUDE.md with `@` references to every source file, so reading it automatically pulls the full codebase into context.

## Testing Protocol

**Always commit before benchmarking.** The benchmark script tracks git commit hashes so we can trace every result to exact code. Workflow:

1. Make your changes
2. `git add` and `git commit` with a descriptive message
3. Restart the server: `./restart_server.sh`
4. Run the benchmark: `python benchmark_ttfa.py -d "description of what changed"`
5. Review audio quality: listen to files in `benchmark_runs/<latest>/conc_008/*.wav`
6. Commit results: `git add benchmark_results.csv && git commit -m "benchmark: ..."`

The benchmark enforces a clean git state by default. Use `--allow-dirty` to override during development.

### Server commands
```bash
./restart_server.sh          # kill, free GPU, start fresh, wait until ready (port 8000)
./restart_server.sh 9000     # custom port
```

The restart script:
- Kills any running sglang_omni process
- Waits 5s for GPU memory to free
- Starts the server in background with logs to `/tmp/sglang_server_<port>.log`
- Polls health endpoint every 10s until ready (up to 15 min timeout)
- Exits with error if server dies during startup

### Benchmark commands
```bash
python benchmark_ttfa.py -d "my change"                         # full suite with voice
python benchmark_ttfa.py -d "my change" --no-voice              # without voice cloning
python benchmark_ttfa.py -d "quick test" --concurrencies 1 4 8  # subset
```

Each run creates `benchmark_runs/<timestamp>_<commit>_<description>/` with:
- `run_info.json` — git commit, GPU, description, text, params
- `details.csv` — per-request TTFA, audio duration, chunk count, errors
- `conc_001/req_000.wav`, `conc_008/req_003.wav`, etc — actual audio files

### Key files for performance work
- `sglang_omni/engines/omni/engine.py` — engine loop, batch window, event-driven wakeup
- `sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py` — voice cache, codec warmup, stream vocode, flush
- `sglang_omni/models/fishaudio_s2_pro/pipeline/state_io.py` — tensor serialization bypass
- `sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py` — output codes clone, iteration controller
- `sglang_omni/models/fishaudio_s2_pro/fish_speech/content_sequence.py` — semantic token LUT
- `sglang_omni/serve/openai_api.py` — voice cache API, warmup endpoint
- `sglang_omni/serve/protocol.py` — voice_id field
