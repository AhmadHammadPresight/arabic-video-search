"""Web API and UI server.

Run it with `uvicorn avs.api.app:create_app --factory`. It serves the Arabic-first web UI,
the JSON API under `/api`, and the playback copies, crops and trace frames under `/media`.
Uploads are stored and queued here; the worker process does the indexing.
Searches run in hybrid mode (phase 2) unless option A is asked for; both store a trace.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from functools import cached_property
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from avs.api import db
from avs.api.checker import CheckRunner
from avs.api.hybrid import HybridOutcome, run_hybrid
from avs.api.inference_client import InferenceClient
from avs.api.query_parser import QueryParser
from avs.api.search import SearchQuery, run_search
from avs.api.times import parse_local_time
from avs.api.trace import STAGES, Trace, replace_stage
from avs.api.vectors import VectorStore
from avs.inference.crops import CANDIDATES_PER_TRACK, EDGE_MARGIN, FULL_DETAIL_HEIGHT, MIN_CROP_HEIGHT
from avs.media import MediaError, probe
from avs.settings import REPO_ROOT, Settings, get_settings

log = logging.getLogger("avs.api")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm", ".ts"}
CAMERA_ID = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
WEB_DIR = REPO_ROOT / "web"


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate (ETag) on every load, so UI updates show at once."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class Services:
    """Collaborators of the API, created lazily so the UI works while models are still loading."""

    def __init__(
        self,
        settings: Settings,
        inference: InferenceClient | None = None,
        store: VectorStore | None = None,
        parser: QueryParser | None = None,
        checker: CheckRunner | None = None,
    ) -> None:
        self.settings = settings
        self.inference = inference or InferenceClient(settings.inference_url)
        self.qdrant = store.client if store is not None else QdrantClient(url=settings.qdrant_url)
        if store is not None:
            self.__dict__["store"] = store
        self.parser = parser or QueryParser(settings.llm_url, settings.query_model)
        self.checker = checker or CheckRunner(self.inference.check, settings.db_path, settings.check_top, settings.vlm_model)

    @cached_property
    def store(self) -> VectorStore:
        # The collection's vector size comes from the embedding model the inference service runs.
        store = VectorStore(self.qdrant, self.settings.qdrant_collection, self.inference.health()["dim"])
        store.ensure_collection()
        return store


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["hybrid", "baseline"] = "hybrid"
    cameras: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    limit: int = Field(default=24, ge=1, le=100)
    actor: str = Field(default="anonymous", min_length=1, max_length=80)
    case_id: str | None = Field(default=None, max_length=80)


class FeedbackRequest(BaseModel):
    search_id: str
    track_key: str
    verdict: Literal["confirm", "reject"]
    actor: str = Field(default="anonymous", min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=500)


def create_app(services: Services | None = None) -> FastAPI:
    services = services or Services(get_settings())
    settings = services.settings
    db.init_db(settings.db_path)
    trace_dir = settings.data_dir / "trace"
    for directory in (settings.uploads_dir, settings.playback_dir, settings.crops_dir, trace_dir):
        directory.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warm_up := getattr(services.parser, "warm_up", None):
            threading.Thread(target=warm_up, name="query-model-warm-up", daemon=True).start()
        yield
        services.checker.close()  # stop without working through queued double-checks

    app = FastAPI(title="Arabic video search", docs_url=None, redoc_url=None, lifespan=lifespan)

    def conn():
        return db.connect(settings.db_path)

    def recording_or_404(c, recording_id: str) -> dict:
        recording = db.get_recording(c, recording_id)
        if recording is None:
            raise HTTPException(404, "recording not found")
        return recording

    @app.get("/api/health")
    def health() -> dict:
        status = {"api": "ok"}
        try:
            status["inference"] = services.inference.health()
        except Exception as exc:
            status["inference"] = {"status": "down", "error": str(exc)}
        try:
            services.qdrant.get_collections()
            status["qdrant"] = "ok"
        except Exception as exc:
            status["qdrant"] = f"down: {exc}"
        status["query_model"] = services.parser.info()
        return status

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        with conn() as c:
            recordings = db.list_recordings(c)
        by_camera: dict[str, dict] = {}
        for r in recordings:
            entry = by_camera.setdefault(r["camera_id"], {"camera_id": r["camera_id"], "places": [], "recordings": 0})
            entry["recordings"] += 1
            if r["place"] not in entry["places"]:
                entry["places"].append(r["place"])
        return sorted(by_camera.values(), key=lambda e: e["camera_id"])

    @app.get("/api/recordings")
    def recordings() -> list[dict]:
        with conn() as c:
            return db.list_recordings(c)

    @app.post("/api/recordings", status_code=201)
    def upload(
        file: UploadFile = File(...),
        camera_id: str = Form(...),
        place: str = Form(...),
        start_time: str = Form(...),
        actor: str = Form("anonymous"),
    ) -> dict:
        camera_id, place = camera_id.strip(), place.strip()
        if not CAMERA_ID.match(camera_id):
            raise HTTPException(422, "camera_id: use 1-40 letters, digits, dot, dash or underscore")
        if not 1 <= len(place) <= 200:
            raise HTTPException(422, "place: 1-200 characters")
        try:
            start = parse_local_time(start_time).isoformat()
        except ValueError as exc:
            raise HTTPException(422, f"start_time: {exc}") from exc
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(422, f"unsupported file type {suffix!r}")

        recording_id = db.new_id()
        video_rel = f"uploads/{recording_id}{suffix}"
        video_path = settings.resolve(video_rel)
        with video_path.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
        try:
            info = probe(video_path)
        except MediaError as exc:
            video_path.unlink(missing_ok=True)
            raise HTTPException(422, f"not a readable video: {exc}") from exc

        with conn() as c:
            recording = db.create_recording(
                c, recording_id, camera_id, place, start, file.filename or video_rel, video_rel,
                info.duration, info.width, info.height,
            )
            db.append_audit(c, actor, "upload", {
                "recording_id": recording_id, "camera_id": camera_id, "place": place,
                "start_time": start, "file": file.filename, "duration_s": info.duration,
            })
        return recording

    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id: str) -> dict:
        with conn() as c:
            return recording_or_404(c, recording_id)

    @app.get("/api/recordings/{recording_id}/tracks")
    def tracks(recording_id: str) -> list[dict]:
        with conn() as c:
            recording_or_404(c, recording_id)
            return db.tracks_for_recording(c, recording_id)

    @app.post("/api/recordings/{recording_id}/reindex")
    def reindex(recording_id: str, actor: str = "anonymous") -> dict:
        with conn() as c:
            recording_or_404(c, recording_id)
            db.requeue_recording(c, recording_id)
            db.append_audit(c, actor, "reindex", {"recording_id": recording_id})
            return db.get_recording(c, recording_id)

    @app.delete("/api/recordings/{recording_id}")
    def delete(recording_id: str, actor: str = "anonymous") -> dict:
        """Delete a recording together with everything derived from it: index, tracks, crops, playback copy."""
        with conn() as c:
            recording = recording_or_404(c, recording_id)
            if recording["status"] == "indexing":
                raise HTTPException(409, "recording is being indexed; try again when it finishes")
            services.store.delete_recording(recording_id)
            db.delete_recording(c, recording_id)
            for rel in (recording["video_path"], recording["playback_path"]):
                if rel:
                    settings.resolve(rel).unlink(missing_ok=True)
            shutil.rmtree(settings.resolve(f"crops/{recording_id}"), ignore_errors=True)
            shutil.rmtree(settings.resolve(f"trace/{recording_id}"), ignore_errors=True)
            db.append_audit(c, actor, "delete", {
                "recording_id": recording_id, "camera_id": recording["camera_id"], "file": recording["original_name"],
            })
        return {"deleted": recording_id}

    @app.post("/api/search")
    def search(request: SearchRequest, trace: bool = False) -> dict:
        """Search in hybrid mode (default) or option A; `?trace=1` also returns every stage's output."""
        started = time.monotonic()
        search_id = db.new_id()
        query = SearchQuery(request.query.strip(), request.cameras or None, request.time_from or None,
                            request.time_to or None, request.limit)
        details = {
            "query": query.text, "cameras": query.cameras, "time_from": query.time_from,
            "time_to": query.time_to, "limit": query.limit, "case_id": request.case_id,
        }
        tracer = Trace(search_id, request.mode, details | {"mode": request.mode})
        embedder = f"{settings.embed_model}@{settings.embed_revision[:10]}"
        tracer.data["models"] = {"embedder": embedder}
        if request.mode == "hybrid":
            tracer.data["models"] |= {"query_model": services.parser.info(), "vision_model": settings.vlm_model}
        with conn() as c:
            recordings = {r["id"]: r for r in db.list_recordings(c)}
            try:
                if request.mode == "baseline":
                    outcome = HybridOutcome(run_search(services.store, services.inference.embed_text, recordings, query,
                                                       trace=tracer, embed_model=embedder), "baseline")
                else:
                    outcome = run_hybrid(services.store, services.inference.embed_texts, services.parser.parse,
                                         recordings, query, tracer, embedder)
            except ValueError as exc:
                db.append_audit(c, request.actor, "search", details | {"mode": request.mode, "error": str(exc)}, search_id)
                raise HTTPException(422, str(exc)) from exc
            results = outcome.results
            took_ms = round((time.monotonic() - started) * 1000)
            trace_data = tracer.finish()
            db.save_trace(c, search_id, outcome.mode, trace_data)
            parsed = outcome.parsed.fields if outcome.parsed else None
            db.append_audit(c, request.actor, "search", details | {
                "mode": outcome.mode, "english": parsed.english if parsed else None,
                "result_count": len(results), "took_ms": took_ms,
                "results": [{"track_key": r["track_key"], "score": r["score"]} for r in results],
            }, search_id)
        response = {"search_id": search_id, "query": query.text, "mode": outcome.mode, "took_ms": took_ms, "results": results}
        if parsed is not None:
            response["parsed"] = parsed.model_dump()
        if parsed is not None and results:
            check = services.checker.submit(search_id, parsed, results, outcome.check_crops)
            response["check"] = {k: check[k] for k in ("status", "total", "details")}
        if trace:
            response["trace"] = trace_data
        return response

    @app.get("/api/search/{search_id}/check")
    def search_check(search_id: str) -> dict:
        """Progress of the vision-model double-check; when done, the results in their new order."""
        status = services.checker.status(search_id)
        if status is None:
            raise HTTPException(404, "no double-check for this search")
        return status

    @app.post("/api/search/{search_id}/check/cancel")
    def search_check_cancel(search_id: str) -> dict:
        """Stop the double-check of a search the page no longer shows; a finished check is left as it is."""
        status = services.checker.cancel(search_id) or services.checker.status(search_id)
        if status is None:
            raise HTTPException(404, "no double-check for this search")
        return status

    @app.get("/api/search/{search_id}/trace")
    def search_trace(search_id: str) -> dict:
        """The stored trace of a search: every stage with its output and timing."""
        with conn() as c:
            stored = db.get_trace(c, search_id)
        if stored is None:
            raise HTTPException(404, "no trace for this search")
        live = services.checker.status(search_id)  # the check's progress is newer in memory than in the stored trace
        if live is not None:
            ar, en = STAGES["check"]
            replace_stage(stored, {"key": "check", "title_ar": ar, "title_en": en, "ms": live.get("took_ms"), "output": live})
        return stored

    @app.get("/api/recordings/{recording_id}/trace")
    def recording_trace(recording_id: str, frames: int = 6, at: str | None = None) -> dict:
        """How a recording was indexed: settings, sampled frames with live detections, tracks, crops and attributes."""
        with conn() as c:
            rec = recording_or_404(c, recording_id)
            stats = db.get_index_stats(c, recording_id)
            tracks = db.tracks_for_recording(c, recording_id)
        fps = (stats or {}).get("sample_fps") or settings.sample_fps
        duration = rec["duration_s"] or 0.0
        if at:
            try:
                times = [float(x) for x in at.split(",") if x.strip()][:12]
            except ValueError as exc:
                raise HTTPException(422, "at: comma-separated seconds") from exc
        else:
            n = max(1, min(frames, 12))
            times = [round(round(duration * (i + 0.5) / n * fps) / fps, 3) for i in range(n)]
        frame_trace, frames_error = None, None
        try:
            frame_trace = services.inference.trace_frames(recording_id, rec["video_path"], times)
        except Exception as exc:
            frames_error = str(exc)
        if frame_trace:
            for frame in frame_trace["frames"]:
                frame["url"] = f"/media/{frame['path']}"
                frame["tracks"] = [
                    {"track_key": t["track_key"], "track_id": t["track_id"], "label": t["label"], "box": box[1:]}
                    for t in tracks for box in t["boxes"] if abs(box[0] - frame["t"]) <= 0.5 / fps
                ]
        return {
            "recording": rec,
            "stats": stats,
            "rules": {
                "sample_fps": fps,
                "detector_threshold": settings.detection_threshold,
                "min_track_detections": settings.min_track_detections,
                "crops_per_track": settings.crops_per_track,
                "pool_capacity": CANDIDATES_PER_TRACK,
                "min_crop_height_px": MIN_CROP_HEIGHT,
                "quality": f"detector confidence x size (box height / {FULL_DETAIL_HEIGHT:.0f} px, at most 1) "
                           f"x edge (0.5 when the box is within {EDGE_MARGIN:.0f} px of the frame edge)",
                "spacing": "picked greedily by quality, at least min_gap seconds apart; min_gap is the track's "
                           "length / (2 x crops per track), at least 0.5 s",
            },
            "frames": frame_trace,
            "frames_error": frames_error,
            "tracks": [
                {"track_key": t["track_key"], "track_id": t["track_id"], "label": t["label"],
                 "t_start": t["t_start"], "t_end": t["t_end"], "detections": len(t["boxes"]),
                 "crops": [c | {"url": f"/media/{c['path']}"} for c in t["crops"]],
                 "selection": t["selection"], "attributes": t["attributes"]}
                for t in tracks
            ],
        }

    @app.post("/api/feedback", status_code=201)
    def feedback(request: FeedbackRequest) -> dict:
        with conn() as c:
            searched = db.find_audit(c, request.search_id)
            if searched is None:
                raise HTTPException(404, "unknown search_id")
            shown = {r["track_key"] for r in searched["details"].get("results", [])}
            if request.track_key not in shown:
                raise HTTPException(422, "track_key was not a result of this search")
            audit_id = db.append_audit(c, request.actor, request.verdict, {
                "track_key": request.track_key, "query": searched["details"].get("query"),
                "case_id": searched["details"].get("case_id"), "note": request.note,
            }, request.search_id)
        return {"audit_id": audit_id}

    @app.get("/api/audit")
    def audit(limit: int = 200) -> list[dict]:
        with conn() as c:
            return db.list_audit(c, max(1, min(limit, 1000)))

    # Only the derived media is served; uploads, models and the database are not reachable over HTTP.
    app.mount("/media/playback", StaticFiles(directory=settings.playback_dir), name="playback")
    app.mount("/media/crops", StaticFiles(directory=settings.crops_dir), name="crops")
    app.mount("/media/trace", StaticFiles(directory=trace_dir), name="trace")
    app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
