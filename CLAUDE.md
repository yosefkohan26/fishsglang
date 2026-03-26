# SGLang-Omni Full Codebase

Import all source files for full context.

## Config
- @./pyproject.toml
- @./sglang_omni/__init__.py
- @./sglang_omni/environ.py

## CLI
- @./sglang_omni/cli/__init__.py
- @./sglang_omni/cli/cli.py
- @./sglang_omni/cli/config.py
- @./sglang_omni/cli/serve.py

## Client
- @./sglang_omni/client/__init__.py
- @./sglang_omni/client/audio.py
- @./sglang_omni/client/client.py
- @./sglang_omni/client/types.py

## Config System
- @./sglang_omni/config/__init__.py
- @./sglang_omni/config/compiler.py
- @./sglang_omni/config/manager.py
- @./sglang_omni/config/runner.py
- @./sglang_omni/config/schema.py

## Engines - Base
- @./sglang_omni/engines/__init__.py
- @./sglang_omni/engines/async_module.py
- @./sglang_omni/engines/base.py

## Engines - AR SGLang Backend
- @./sglang_omni/engines/ar/sglang_backend/args.py
- @./sglang_omni/engines/ar/sglang_backend/model_runner.py
- @./sglang_omni/engines/ar/sglang_backend/model_worker.py
- @./sglang_omni/engines/ar/sglang_backend/server_args_builder.py
- @./sglang_omni/engines/ar/sglang_backend/scheduler/__init__.py
- @./sglang_omni/engines/ar/sglang_backend/scheduler/cache.py
- @./sglang_omni/engines/ar/sglang_backend/scheduler/decode.py
- @./sglang_omni/engines/ar/sglang_backend/scheduler/prefill.py
- @./sglang_omni/engines/ar/sglang_backend/scheduler/scheduler.py

## Engines - Omni
- @./sglang_omni/engines/omni/__init__.py
- @./sglang_omni/engines/omni/engine.py
- @./sglang_omni/engines/omni/factory.py
- @./sglang_omni/engines/omni/model_runner.py
- @./sglang_omni/engines/omni/scheduler.py
- @./sglang_omni/engines/omni/types.py

## Engines - Omni Runtime
- @./sglang_omni/engines/omni/runtime/__init__.py
- @./sglang_omni/engines/omni/runtime/_hidden_capture.py
- @./sglang_omni/engines/omni/runtime/ar.py
- @./sglang_omni/engines/omni/runtime/cache.py
- @./sglang_omni/engines/omni/runtime/common.py
- @./sglang_omni/engines/omni/runtime/encoder.py
- @./sglang_omni/engines/omni/runtime/interfaces.py
- @./sglang_omni/engines/omni/runtime/sglang_ar.py
- @./sglang_omni/engines/omni/runtime/tokenizer.py

## Executors
- @./sglang_omni/executors/__init__.py
- @./sglang_omni/executors/direct_model_executor.py
- @./sglang_omni/executors/engine_executor.py
- @./sglang_omni/executors/engine_request_builders.py
- @./sglang_omni/executors/fused_executor.py
- @./sglang_omni/executors/interface.py
- @./sglang_omni/executors/preprocessing_executor.py

## Models - Registry & Loader
- @./sglang_omni/models/__init__.py
- @./sglang_omni/models/registry.py
- @./sglang_omni/models/weight_loader.py
- @./sglang_omni/models/fishaudio_s2_pro_sglang.py

## Models - FishAudio S2Pro
- @./sglang_omni/models/fishaudio_s2_pro/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/config.py
- @./sglang_omni/models/fishaudio_s2_pro/factory.py
- @./sglang_omni/models/fishaudio_s2_pro/io.py
- @./sglang_omni/models/fishaudio_s2_pro/sglang_model.py
- @./sglang_omni/models/fishaudio_s2_pro/tokenizer.py

## Models - FishAudio S2Pro Pipeline
- @./sglang_omni/models/fishaudio_s2_pro/pipeline/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/pipeline/engine_io.py
- @./sglang_omni/models/fishaudio_s2_pro/pipeline/next_stage.py
- @./sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py
- @./sglang_omni/models/fishaudio_s2_pro/pipeline/state_io.py

## Models - FishAudio S2Pro Runtime
- @./sglang_omni/models/fishaudio_s2_pro/runtime/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_ar.py
- @./sglang_omni/models/fishaudio_s2_pro/runtime/s2pro_sglang_ar.py

## Models - FishAudio Fish Speech
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/content_sequence.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/conversation.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/tokenizer.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/utils/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/utils/logger.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/kernels/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/modded_dac.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/dac/rvq.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/__init__.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/configuration.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/modeling.py
- @./sglang_omni/models/fishaudio_s2_pro/fish_speech/models/text2semantic/utils.py

## Pipeline
- @./sglang_omni/pipeline/__init__.py
- @./sglang_omni/pipeline/control_plane.py
- @./sglang_omni/pipeline/coordinator.py
- @./sglang_omni/pipeline/mp_runner.py
- @./sglang_omni/pipeline/stage/__init__.py
- @./sglang_omni/pipeline/stage/input.py
- @./sglang_omni/pipeline/stage/router.py
- @./sglang_omni/pipeline/stage/runtime.py
- @./sglang_omni/pipeline/stage/stream_queue.py
- @./sglang_omni/pipeline/stage/work.py
- @./sglang_omni/pipeline/worker/__init__.py
- @./sglang_omni/pipeline/worker/data_plane.py
- @./sglang_omni/pipeline/worker/runtime.py

## Preprocessing
- @./sglang_omni/preprocessing/__init__.py
- @./sglang_omni/preprocessing/audio.py
- @./sglang_omni/preprocessing/base.py
- @./sglang_omni/preprocessing/cache_key.py
- @./sglang_omni/preprocessing/resource_connector.py
- @./sglang_omni/preprocessing/text.py

## Profiler
- @./sglang_omni/profiler/base_profiler.py
- @./sglang_omni/profiler/profiler_control.py
- @./sglang_omni/profiler/torch_profiler.py

## Proto
- @./sglang_omni/proto/__init__.py
- @./sglang_omni/proto/messages.py
- @./sglang_omni/proto/request.py
- @./sglang_omni/proto/stage.py

## Relay
- @./sglang_omni/relay/__init__.py
- @./sglang_omni/relay/base.py
- @./sglang_omni/relay/mooncake.py
- @./sglang_omni/relay/nccl.py
- @./sglang_omni/relay/nixl.py
- @./sglang_omni/relay/shm.py

## Serve
- @./sglang_omni/serve/__init__.py
- @./sglang_omni/serve/launcher.py
- @./sglang_omni/serve/openai_api.py
- @./sglang_omni/serve/protocol.py

## Utils
- @./sglang_omni/utils/__init__.py
- @./sglang_omni/utils/hf.py
- @./sglang_omni/utils/misc.py

## Vendor
- @./sglang_omni/vendor/sglang/__init__.py
- @./sglang_omni/vendor/sglang/core.py
- @./sglang_omni/vendor/sglang/distributed.py
- @./sglang_omni/vendor/sglang/layers.py
- @./sglang_omni/vendor/sglang/models.py
- @./sglang_omni/vendor/sglang/server_args.py
- @./sglang_omni/vendor/sglang/utils.py
