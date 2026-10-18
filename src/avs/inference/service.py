"""Native inference service: the vision models on the Apple GPU (the query model runs in Ollama).

It runs on the host (not in a container) because containers on macOS get no Metal access.
It loads RF-DETR, SigLIP 2 and the Qwen3-VL vision model once, then serves:

- `POST /v1/embed/text` turns query text into SigLIP 2 vectors.
- `POST /v1/index` indexes one recording and streams NDJSON events while it works.
- `POST /v1/check` asks the vision model whether a result's crops show each detail of a query.
- `POST /v1/describe` describes crops with either attribute method (for benchmarks and debugging).
- `POST /v1/trace/frames` decodes a few frames of a recording and runs the detector on them.

It binds to 127.0.0.1 only and never calls out: model weights are fetched by
`scripts/setup.sh`, and `scripts/start.sh` runs it with the Hugging Face hub offline.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from avs.inference.attributes import Describer, SigLIPAttributes, VLMAttributes
from avs.inference.detector import RFDETRDetector
from avs.inference.embedder import SigLIP2Embedder
from avs.inference.pipeline import IndexOptions, index_video
from avs.inference.vlm import QwenVL
from avs.media import make_playback_copy, probe, read_frame
from avs.settings import get_settings

log = logging.getLogger("avs.inference")
RECORDING_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Models:
    def __init__(self) -> None:
        self.detector: RFDETRDetector | None = None
        self.embedder: SigLIP2Embedder | None = None
        self.vlm: QwenVL | None = None
        self.describers: dict[str, Describer] = {}
        self.attributes: Describer | None = None


models = Models()
_index_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    models.detector = RFDETRDetector(
        settings.detector, settings.models_dir, settings.device, settings.detection_threshold
    )
    models.embedder = SigLIP2Embedder(settings.embed_model, settings.embed_revision, settings.device)
    models.vlm = QwenVL(settings.vlm_model, settings.vlm_revision)
    models.describers = {
        "siglip": SigLIPAttributes(models.embedder, models.embedder.logit_scale),
        "vlm": VLMAttributes(models.vlm),
    }
    if settings.attributes not in (*models.describers, "none"):
        raise ValueError(f"AVS_ATTRIBUTES must be vlm, siglip or none, not {settings.attributes!r}")
    models.attributes = models.describers.get(settings.attributes)
    log.info("models loaded on %s: %s, %s, %s; attributes from %s", settings.device, settings.detector,
             settings.embed_model, settings.vlm_model, settings.attributes)
    yield


app = FastAPI(title="AVS inference", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class IndexRequest(BaseModel):
    recording_id: str
    video_path: str  # relative to the data directory
    sample_fps: float | None = Field(default=None, ge=0.5, le=10)


class CheckRequest(BaseModel):
    crops: list[str] = Field(min_length=1, max_length=4)  # relative to the data directory
    subject: Literal["person", "bag"] = "person"
    english: str = Field(min_length=1, max_length=300)
    details: list[str] = Field(min_length=1, max_length=12)


class DescribeRequest(BaseModel):
    crops: list[str] = Field(min_length=1, max_length=8)  # best first, relative to the data directory
    subject: Literal["person", "bag"] = "person"
    label: str = "person"
    method: Literal["vlm", "siglip"] = "vlm"


class FramesRequest(BaseModel):
    recording_id: str
    video_path: str  # relative to the data directory
    times: list[float] = Field(min_length=1, max_length=12)


def _crop_images(paths: list[str]) -> list[Image.Image]:
    settings = get_settings()
    images = []
    for rel in paths:
        try:
            path = settings.resolve(rel)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.is_file():
            raise HTTPException(404, f"crop not found: {rel}")
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
    return images


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok" if models.embedder and models.detector and models.vlm else "loading",
        "device": settings.device,
        "detector": settings.detector,
        "detection_threshold": settings.detection_threshold,
        "embedder": settings.embed_model,
        "embedder_revision": settings.embed_revision,
        "vlm": settings.vlm_model,
        "vlm_revision": settings.vlm_revision,
        "attributes": models.attributes.source if models.attributes else None,
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
            # Each run writes its crops to a new folder, so the crops of the points still in Qdrant stay
            # in place while a recording is re-indexed; the worker removes the old folders once it has
            # replaced the points.
            crops_rel = f"crops/{request.recording_id}/{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
            crops_dir = settings.resolve(crops_rel)
            yield _line({"type": "crops", "path": crops_rel})
            for event in index_video(video, crops_dir, crops_rel, models.detector, models.embedder, options, info,
                                     describer=models.attributes):
                yield _line(event)
        except Exception as exc:  # reported to the worker, which marks the recording failed
            log.exception("indexing %s failed", request.recording_id)
            yield _line({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            _index_lock.release()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/v1/check")
def check(request: CheckRequest) -> dict:
    images = _crop_images(request.crops)
    result = models.vlm.check(images, request.subject, request.english, request.details)
    return result | {"model": get_settings().vlm_model}


@app.post("/v1/describe")
def describe(request: DescribeRequest) -> dict:
    images = _crop_images(request.crops)
    vectors = models.embedder.embed_images(images)
    describer = models.describers[request.method]
    return describer.describe(images, vectors, request.subject, request.label) | {"source": describer.source}


@app.post("/v1/trace/frames")
def trace_frames(request: FramesRequest) -> dict:
    """Decode frames at the given seconds, run the detector on each, and save them as JPEGs for the trace page."""
    settings = get_settings()
    if not RECORDING_ID.match(request.recording_id):
        raise HTTPException(400, "invalid recording_id")
    try:
        video = settings.resolve(request.video_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not video.is_file():
        raise HTTPException(404, f"video not found: {request.video_path}")
    info = probe(video)
    out_rel = f"trace/{request.recording_id}"
    out_dir = settings.resolve(out_rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in request.times:
        t = round(min(max(0.0, t), max(0.0, info.duration - 0.05)), 3)
        frame = read_frame(video, t, info)
        started = time.monotonic()
        detections = models.detector.detect(frame)
        detect_ms = round((time.monotonic() - started) * 1000)
        name = f"frame-{t:09.3f}.jpg"
        Image.fromarray(frame).save(out_dir / name, quality=85)
        frames.append({
            "t": t, "path": f"{out_rel}/{name}", "detect_ms": detect_ms,
            "detections": [
                {"label": d.label, "confidence": round(d.confidence, 4),
                 "box": [round(d.box[0] / info.width, 4), round(d.box[1] / info.height, 4),
                         round(d.box[2] / info.width, 4), round(d.box[3] / info.height, 4)]}
                for d in sorted(detections, key=lambda d: -d.confidence)
            ],
        })
    return {"width": info.width, "height": info.height, "fps": info.fps, "duration": info.duration,
            "detector": settings.detector, "threshold": settings.detection_threshold, "frames": frames}


def _line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()
