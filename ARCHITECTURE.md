# Fish Audio S2-Pro SGLang Architecture Guide

## For the next agent: Read this first, then read the key files listed below.

---

## Model Architecture (fishaudio/s2-pro)

**Dual-Autoregressive TTS**: Text → semantic tokens → audio waveform.

### Two-headed transformer:

1. **Slow AR (4.2B params)**: Qwen3-4B backbone, 36 layers, dim=2560, 32 heads, 8 KV heads, head_dim=128, intermediate=9728. Predicts one semantic token per step from a vocab of 4096 audio tokens + 151K text tokens. This is the bottleneck — reads 7.58 GB of weights from HBM every decode step.

2. **Fast AR (530M params)**: 4 layers, same dim. Given the Slow AR's hidden state + semantic token, generates 9 additional codebook tokens (acoustic detail). Runs sequentially: 9 `forward_kvcached` calls per Slow AR step. Uses flash_attn_with_kvcache with cache_seqlens masking.

3. **DAC Codec/Vocoder (695M params, FP32)**: Convolutional encoder + RVQ (10 codebooks, 4096 entries each) + EVA-GAN decoder. 44.1kHz, 21 Hz frame rate (2048 samples per token). The `from_indices` method converts codebook indices → audio waveform.

### Token flow per decode step:
```
Slow AR forward (36 layers) → semantic token (multinomial sampling)
    → Fast AR codebook loop (9 × 4-layer forward) → 9 acoustic tokens
    → [semantic + 9 acoustic] = 10 codebook values for this timestep
    → Multi-Codebook Fusion: sum 10 embeddings → input for next Slow AR step
```

### Prompt structure:
```
[system] "convert the provided text to speech..."
[ref text] "<|speaker:0|>Hi, my name is Joseph..."  (voice reference transcript)
[ref audio] <semantic:1234> <semantic:5678> ...      (1600-1900 VQ tokens from codec encode)
[/system]
[user] "<|speaker:0|>The text to synthesize..."
[assistant]
<|audio_start|>                                       (model generates from here)
```

The reference audio VQ tokens are the voice "prompt" — ~1600-1900 tokens depending on reference length. These are the same across all requests for the same voice, making them perfect for RadixCache prefix caching.

---

## Key Files (read these)

### Model definition
- **`sglang_omni/models/fishaudio_s2_pro/sglang_model.py`** — The SGLang-native model. `S2ProSGLangTextModel` wraps the Slow AR. `forward()` runs the transformer only (no sampling — moved out for CUDA graph compatibility). `_decode_codebooks()` does semantic sampling + Fast AR codebook loop. `setup_vq_decode()` attaches the Fast AR and allocates persistent GPU buffers. `load_weights()` handles BF16/INT4/INT8/FP8 checkpoints.

- **`sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py`** — The raw Dual-AR model (Fast AR lives here). `AudioDecoder.forward_kvcached()` is the Fast AR's KV-cached forward. `reset_caches()` zeros KV caches (we removed this call — flash_attn only reads up to cache_seqlens).

- **`sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/modded_dac.py`** — DAC codec. `encode()` converts audio → VQ indices. `from_indices()` converts VQ indices → audio waveform. This is torch.compiled with `reduce-overhead` mode.

### Runtime / execution
- **`sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py`** — The model runner. `S2ProSGLangModelRunner.execute()` is called every engine step: prepares VQ embeddings (prefill or decode), runs GPU forward, calls `_decode_codebooks`, builds outputs. `_inject_vq_embeds_prefill()` computes combined text+VQ embeddings for prefill. `_update_vq_buffers()` writes per-request sampling params (temperature, top_p, rep_penalty, previous tokens) to persistent GPU buffers. `S2ProSGLangIterationController` checks EOS and manages RadixCache.

- **`sglang_omni/engines/omni/engine.py`** — The engine loop. `_step_normal()` runs schedule → execute → update in a tight loop. No batch window (set to 0).

- **`sglang_omni/engines/ar/sglang_backend/model_worker.py`** — Wraps SGLang's model runner. `forward_batch_generation()` calls the actual GPU forward pass.

### Pipeline / serving
- **`sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py`** — Factory functions for both pipeline stages. `create_preprocessing_executor()` builds the preprocessing stage (tokenization, voice cache, reference audio encoding). `create_sglang_tts_engine_executor()` builds the TTS engine stage (loads model, codec, creates SGLang engine). Contains voice preloading from `voices/` directory, stream vocode logic, and all server config (mem_fraction, attention backend, etc.).

- **`sglang_omni/models/fishaudio_s2_pro/pipeline/engine_io.py`** — Converts `S2ProState` to SGLang `Req` objects. **Critical: `input_ids_list = state.input_ids.tolist()`** — must use `.tolist()` not `list()` or RadixCache breaks (tensor elements hash differently from ints).

- **`sglang_omni/serve/openai_api.py`** — FastAPI server. `_speech_stream()` yields SSE audio chunks. `_build_speech_generate_request()` maps API params to pipeline inputs.

- **`sglang_omni/models/fishaudio_s2_pro/pipeline/next_stage.py`** — Stage routing. `tts_engine_next()` returns `None` for streaming (terminates) or `"vocoder"` for non-streaming.

### Weight loading
- **`sglang_omni/models/weight_loader.py`** — Loads weights by prefix from safetensors shards. Has INT4/INT8/FP8 dequantization support via `_dequantize_int4()` and the sharded loader.

- **`sglang_omni/models/fishaudio_s2_pro/fp8_linear.py`** — FP8Linear drop-in replacement (not currently used — FP8 disabled due to activation quantization overhead exceeding bandwidth savings without fused kernels).

---

## What We Optimized (and why)

### RadixCache fix (204ms → 45ms TTFA)
**File**: `engine_io.py` line 27
**Change**: `list(state.input_ids)` → `state.input_ids.tolist()`
**Why**: `list(tensor)` produces 0-d tensor elements. `hash(tensor(151644)) != hash(151644)`. RadixCache keys by hash, so it never matched prefixes. With `.tolist()`, token IDs are plain ints, RadixCache hits 99.3% of the voice prefix (~1900 tokens cached, only ~20 new text tokens prefilled per request).

### Voice preloading (95s → 0ms first request)
**File**: `stages.py`, bottom of `create_preprocessing_executor()`
**How**: `voices/` directory with `audio.wav` + `transcript.txt` per voice. First run encodes to `codes.pt` (VQ codes saved as torch tensor). Subsequent starts load `codes.pt` in <1ms. Voice cache (`_CachedVoice`) stores VQ codes + ref_text in memory.

### Vectorized semantic sampling
**File**: `sglang_model.py:_decode_codebooks()`
**Change**: Replaced `for i in range(bs)` Python loop with batched tensor ops. Repetition penalty via batched gather/scatter. RAS duplicate detection via sorted comparison. Batched top-k/top-p/multinomial.
**Why**: At bs=16, the Python loop was 16 sequential iterations of small CUDA kernel launches (~15-25ms). Vectorized version does one batched operation per step.

### _decode_codebooks split from forward()
**File**: `sglang_model.py` (removed from `forward()`) + `s2pro_sglang_ar.py` (added to `execute()`)
**Why**: Separates graph-capturable ops (transformer) from non-capturable ops (multinomial, codebook loop). CUDA graphs didn't help in practice (FA3 kernels are already large/well-fused, re-entry overhead > launch savings), but the split is correct architecture.

### Removed reset_caches()
**File**: `sglang_model.py:_codebook_loop()`
**Why**: Fast AR uses `flash_attn_with_kvcache` with `cache_seqlens=codebook_idx`. Only reads KV entries up to that index. Zeroing the cache (~22MB GPU memset) was redundant.

### Batched VQ buffer updates
**File**: `s2pro_sglang_ar.py:_update_vq_buffers()`
**Change**: Collect per-request params in Python lists, write as single batched `torch.as_tensor()` to GPU. Pre-allocate CPU buffer for prev_tokens.

### Bulk clone in _build_outputs
**File**: `s2pro_sglang_ar.py:_build_outputs()`
**Change**: One `_output_codes[:bs].clone()` instead of per-row clones. Non-overlapping views for each request.

### Stream vocoder torch.compile
**File**: `stages.py`
**How**: `torch.compile(_stream_codec.from_indices, mode="reduce-overhead")`. DAC decoder is purely convolutional — TorchInductor fuses snake activations and transposed convolutions well. First call triggers ~6s JIT compilation, subsequent calls ~9ms.

---

## What We Tried That Didn't Work

### FP8 weight quantization
Implemented per-channel weight scales + per-row activation scales + rowwise `torch._scaled_mm`. Model VRAM dropped 7.58→4.46 GB. But **RTF went from 0.53 to 1.73** (3.3x slower). The dynamic activation quantization (`abs().max()` per row per GEMM × 180 GEMMs/step) overhead exceeded HBM bandwidth savings. Needs fused quant+GEMM kernels (CUTLASS/Triton) to be worthwhile. Code is in `sglang_model.py:_convert_to_fp8()` and `fp8_linear.py`, disabled.

### INT4 weight quantization (AWQ/RTN)
Broke EOS detection — model generates forever. The constrained sampling mask allows semantic tokens + im_end token. INT4 perturbation shifts logit distribution enough that im_end probability never exceeds a semantic token. Even INT8 RTN worked for quality but only saves disk (dequants to BF16 at load).

### FP8 KV cache
SGLang's `kv_cache_dtype="fp8_e4m3"` without calibrated per-head scales. Model generated ~10% fewer tokens per sentence (early EOS). Quantization noise in attention scores accumulates over 36 layers.

### CUDA graphs
Captured successfully after the `_decode_codebooks` split, but performed WORSE at all concurrency levels. FA3 attention kernels are already large (2-3 per layer), so kernel launch overhead is minimal. The graph replay → CPU return → re-enter for codebook decode adds a synchronization bubble that costs more than the launch savings.

### Separate CUDA stream for vocoder
Vocode and decode compete for GPU memory bandwidth, not compute. Separate streams don't help with bandwidth saturation. The `synchronize()` blocks anyway since we need the audio result immediately.

### Prompt prefix cache
Cached tokenized prompt prefix and concatenated new text tokens. Produced slightly different token sequences than `build_prompt()`, reducing RadixCache hit quality. Made concurrency 4 worse (106-158ms vs 93-123ms).

### torch.compile on Fast AR codebook loop
`max-autotune-no-cudagraphs` mode. Dispatch overhead exceeded kernel savings at small batch sizes.

---

## Performance Numbers (RTX 4090, BF16, all optimizations)

| Metric | Value |
|---|---|
| TTFA concurrency 1 | 45ms |
| TTFA concurrency 4 | 109ms |
| TTFA concurrency 8 | 175ms |
| RTF | 0.52 (1.9x realtime) |
| Model VRAM | 7.58 GB |
| KV cache per token | 144 KB (36 layers × 2 × 8 heads × 128 dim × 2 bytes) |
| Voice prefix (RadixCache, shared) | ~275 MB (1866 tokens × 144 KB) |
| Per-request incremental KV | ~14 MB (100 output tokens × 144 KB) |
| Max concurrent (4090 24GB) | ~16 before OOM |

---

## Server Configuration (stages.py)

```python
ServerArgs(
    model_path=checkpoint_dir,
    tp_size=1,
    dtype="bfloat16",
    attention_backend="fa3",     # MUST match training. Flashinfer causes dynamo conflicts.
    mem_fraction_static=0.85,
    chunked_prefill_size=8192,
    max_running_requests=48,
    disable_cuda_graph=True,     # Graphs hurt — FA3 kernels already well-fused.
)
```

Stream vocode settings:
- `stream_stride=1` — vocode after 1 token (lowest TTFA, 0.05s first chunk)
- `stream_followup_stride=50` — vocode every 50 tokens after first (~2.4s chunks)
- `stream_left_context=25` — overlap window for crossfade

---

## Voice System

### Directory structure:
```
voices/
├── jackson/
│   ├── audio.wav          # Reference audio (any format)
│   ├── transcript.txt     # What is said in the audio
│   └── codes.pt           # Pre-encoded VQ codes (auto-generated on first run)
├── joseph/
│   ├── audio.wav
│   ├── transcript.txt
│   └── codes.pt
```

### How it works:
1. On startup, `create_preprocessing_executor()` scans `voices/` directory
2. If `codes.pt` exists: loads VQ codes in <1ms (`torch.load`)
3. If not: encodes `audio.wav` through DAC codec (GPU: 0.7s, CPU: 95s), saves `codes.pt`
4. VQ codes stored in `_voice_cache` dict, keyed by folder name
5. API requests use `voice_id` to look up cached VQ codes
6. `adapter.build_prompt()` constructs the full token sequence with VQ codes embedded

### Encoding a new voice:
```bash
mkdir voices/newvoice
cp reference.wav voices/newvoice/audio.wav
echo "Transcript of what is said in reference.wav" > voices/newvoice/transcript.txt
# Restart server — codes.pt auto-generated
```

---

## Critical Invariants (DO NOT BREAK)

1. **`engine_io.py` must use `.tolist()` not `list()`** for `state.input_ids`. Breaking this zeros RadixCache hit rate.

2. **Embeddings (`embed_tokens`) and all norms must stay in BF16**. FP16 or any quantization breaks EOS detection.

3. **Fast AR must stay in BF16**. Only 530M params, quantization directly degrades audio quality (timbre, pitch).

4. **`attention_backend="fa3"`** must be set. Flashinfer on Ada GPUs causes torch.compile/dynamo conflicts that crash requests intermittently.

5. **`_decode_codebooks` must run OUTSIDE `forward()`** (in `execute()`). Inside `forward()` prevents future CUDA graph capture of the transformer.

6. **No `reset_caches()` before codebook loop**. `flash_attn_with_kvcache` masks by `cache_seqlens` — stale values are never read.

---

## Next Steps (for TurboQuant KV cache integration)

The bottleneck for >16 concurrent is KV cache memory. TurboQuant at 3.5 bits would compress KV from 144 KB/token to ~32 KB/token, enabling 40+ concurrent on a single 4090.

Integration points:
1. **KV cache allocation**: `sglang.srt.mem_cache.memory_pool` — where K/V tensors are allocated
2. **KV cache write**: `flash_attn_with_kvcache` writes K/V during attention — need to quantize before storing
3. **KV cache read**: attention reads K/V — need to dequantize before computing attention scores
4. **The quantization must be per-head** — different attention heads have different value ranges
5. **The Slow AR has 8 KV heads × 128 dim** — TurboQuant operates on the 128-dim vectors within each head

The key constraint: TurboQuant must be **online** (no calibration data needed) and **unbiased** (inner product estimation must be unbiased for attention to work correctly). The TurboQuant paper proves both properties at 3.5 bits.
