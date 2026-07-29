"""FastAPI application for Whisper ASR."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.cache import close_redis, init_redis
from app.routers import asr, asr_streaming
from whisper_asr.config import load_config

CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/default.yml")

# Config loading is cheap and decides which routes exist, so it happens at
# import. Model loading does not -- see the lifespan below.
config = load_config(CONFIG_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the transcriber and connect Redis on startup; tear both down after.

    The model is loaded here rather than at import time so that importing this
    module never touches the network -- which is what lets the test suite run
    offline. Transcriber is imported here too, so module import does not pull
    in transformers and optimum-quanto either.
    """
    from whisper_asr.transcriber import Transcriber

    app.state.config = config
    # One model instance, so concurrent generate() calls contend rather than
    # parallelise. The semaphore makes that explicit instead of leaving it to
    # chance.
    app.state.inference_semaphore = asyncio.Semaphore(config["max_concurrent_inferences"])
    app.state.transcriber = Transcriber(config)

    redis_url = os.environ.get("REDIS_URL") or config.get("redis_url")
    await init_redis(redis_url)
    try:
        yield
    finally:
        await close_redis()


app = FastAPI(
    title="Whisper ASR API",
    description="Automatic Speech Recognition using Whisper (pipeline and direct backends)",
    version="0.1.0",
    lifespan=lifespan,
)

# Credentialed requests cannot use a wildcard origin -- browsers reject the
# combination -- so only allow credentials once origins are pinned.
cors_origins = config["cors_allow_origins"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(asr.router, prefix="/api/v1", tags=["asr"])
if config.get("enable_streaming", False):
    app.include_router(asr_streaming.router, prefix="/api/v1", tags=["asr-streaming"])


@app.get("/health")
def health():
    """Liveness probe. Cheap and dependency-free, for orchestrators."""
    return {"status": "ok"}


@app.get("/")
def root(request: Request):
    """Service summary: what this instance is configured to do.

    Reads live state rather than the import-time config, so it always reports
    what is actually running.
    """
    active = request.app.state.config
    return {
        "status": "ok",
        "model_id": active["model_id"],
        "device": active["device"],
        "default_backend": active["default_backend"],
        "streaming_enabled": active.get("enable_streaming", False),
        "long_form_mode": active["long_form_mode"],
        "max_audio_seconds": active["max_audio_seconds"],
    }
