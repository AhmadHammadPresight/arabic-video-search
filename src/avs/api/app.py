"""Web API and UI server.

Run it with `uvicorn avs.api.app:create_app --factory`. It serves the Arabic-first web UI,
the JSON API under `/api`, and the playback copies and crops under `/media`.
Uploads are stored and queued here; the worker process does the indexing.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from functools import cached_property
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from avs.api import db
from avs.api.inference_client import InferenceClient
from avs.api.search import SearchQuery, run_search
from avs.api.times import parse_local_time
from avs.api.vectors import VectorStore
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
    ) -> None:
        self.settings = settings
        self.inference = inference or InferenceClient(settings.inference_url)
        self.qdrant = store.client if store is not None else QdrantClient(url=settings.qdrant_url)
        if store is not None:
            self.__dict__["store"] = store

    @cached_property
    def store(self) -> VectorStore:
        # The collection's vector size comes from the embedding model the inference service runs.
        store = VectorStore(self.qdrant, self.settings.qdrant_collection, self.inference.health()["dim"])
        store.ensure_collection()
        return store


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
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
    for directory in (settings.uploads_dir, settings.playback_dir, settings.crops_dir):
        directory.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Arabic video search", docs_url=None, redoc_url=None)

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
            db.append_audit(c, actor, "delete", {
                "recording_id": recording_id, "camera_id": recording["camera_id"], "file": recording["original_name"],
            })
        return {"deleted": recording_id}

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict:
        started = time.monotonic()
        search_id = db.new_id()
        query = SearchQuery(request.query.strip(), request.cameras or None, request.time_from or None,
                            request.time_to or None, request.limit)
        details = {
            "query": query.text, "cameras": query.cameras, "time_from": query.time_from,
            "time_to": query.time_to, "limit": query.limit, "case_id": request.case_id,
        }
        with conn() as c:
            recordings = {r["id"]: r for r in db.list_recordings(c)}
            try:
                results = run_search(services.store, services.inference.embed_text, recordings, query)
            except ValueError as exc:
                db.append_audit(c, request.actor, "search", details | {"error": str(exc)}, search_id)
                raise HTTPException(422, str(exc)) from exc
            took_ms = round((time.monotonic() - started) * 1000)
            db.append_audit(c, request.actor, "search", details | {
                "result_count": len(results), "took_ms": took_ms,
                "results": [{"track_key": r["track_key"], "score": r["score"]} for r in results],
            }, search_id)
        return {"search_id": search_id, "query": query.text, "took_ms": took_ms, "results": results}

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
    app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
