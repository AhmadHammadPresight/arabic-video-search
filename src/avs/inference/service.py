"""Native inference service: the only process that touches the Apple GPU.

It runs on the host (not in a container) because containers on macOS get no Metal access.
It loads RF-DETR and SigLIP 2 once, then serves two calls:

- `POST /v1/embed/text` turns query text into SigLIP 2 vectors.
- `POST /v1/index` indexes one recording and streams NDJSON events while it works.

It binds to 127.0.0.1 only and never calls out: model weights are fetched by
`scripts/setup.sh`, and `scripts/start.sh` runs it with the Hugging Face hub offline.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from avs.inference.detector import RFDETRDetector
from avs.inference.embedder import SigLIP2Embedder
from avs.inference.pipeline import IndexOptions, index_video
from avs.media import make_playback_copy, probe
from avs.settings import get_settings

log = logging.getLogger("avs.inference")
RECORDING_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Models:
    detector: RFDETRDetector | None = None
    embedder: SigLIP2Embedder | None = None


models = Models()
_index_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    models.detector = RFDETRDetector(
        settings.detector, settings.models_dir, settings.device, settings.detection_threshold
    )
    models.embedder = SigLIP2Embedder(settings.embed_model, settings.embed_revision, settings.device)
    log.info("models loaded on %s: %s, %s", settings.device, settings.detector, settings.embed_model)
    yield


app = FastAPI(title="AVS inference", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class IndexRequest(BaseModel):
    recording_id: str
    video_path: str  # relative to the data directory
    sample_fps: float | None = Field(default=None, ge=0.5, le=10)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok" if models.embedder and models.detector else "loading",
        "device": settings.device,
        "detector": settings.detector,
        "embedder": settings.embed_model,
        "dim": models.embedder.dim if models.embedder else None,
        "indexing": _index_lock.locked(),
    }


@app.post("/v1/embed/text")
def embed_text(request: TextRequest) -> dict:
    vectors = models.embedder.embed_texts(request.texts)
    return {"model": get_settings().embed_model, "dim": int(vectors.shape[1]), "embeddings": vectors.tolist()}


@app.post("/v1/index")
def index(request: IndexRequest) -> StreamingResponse:
    settings = get_settings()
    if not RECORDING_ID.match(request.recording_id):
        raise HTTPException(400, "invalid recording_id")
    try:
        video = settings.resolve(request.video_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not video.is_file():
        raise HTTPException(404, f"video not found: {request.video_path}")
    if _index_lock.locked():
        raise HTTPException(409, "another recording is being indexed")

    options = IndexOptions(
        sample_fps=request.sample_fps or settings.sample_fps,
        crops_per_track=settings.crops_per_track,
        min_track_detections=settings.min_track_detections,
    )

    def events() -> Iterator[bytes]:
        # The lock is taken inside the generator so it is only held while a stream is running.
        if not _index_lock.acquire(blocking=False):
            yield _line({"type": "error", "message": "another recording is being indexed"})
            return
        try:
            info = probe(video)
            playback_rel = f"playback/{request.recording_id}.mp4"
            make_playback_copy(video, settings.resolve(playback_rel))
            yield _line({"type": "playback", "path": playback_rel})
            crops_rel = f"crops/{request.recording_id}"
            crops_dir = settings.resolve(crops_rel)
            shutil.rmtree(crops_dir, ignore_errors=True)
            for event in index_video(video, crops_dir, crops_rel, models.detector, models.embedder, options, info):
                yield _line(event)
        except Exception as exc:  # reported to the worker, which marks the recording failed
            log.exception("indexing %s failed", request.recording_id)
            yield _line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            _index_lock.release()

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()
