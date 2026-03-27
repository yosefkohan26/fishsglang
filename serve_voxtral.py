#!/usr/bin/env python3
"""Launch Voxtral TTS server via sglang-omni pipeline."""

import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("serve_voxtral")

MODEL_PATH = os.environ.get("VOXTRAL_MODEL_PATH", "voxtral_model")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))


def main():
    from sglang_omni.models.voxtral_tts.config import VoxtralPipelineConfig
    from sglang_omni.serve.launcher import launch_server

    logger.info("Building Voxtral TTS pipeline config from %s", MODEL_PATH)

    config = VoxtralPipelineConfig(model_path=MODEL_PATH)

    logger.info("Launching server on %s:%d", HOST, PORT)
    launch_server(
        config,
        host=HOST,
        port=PORT,
        model_name="voxtral-tts",
        log_level="info",
    )


if __name__ == "__main__":
    main()
