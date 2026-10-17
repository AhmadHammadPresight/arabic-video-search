"""Indexing worker: takes queued jobs from SQLite and runs them through the inference service.

Run it with `python -m avs.api.worker`. One job runs at a time. A job left running by a
stopped worker is queued again on the next start, and re-indexing a recording replaces its
tracks and vectors, so a job can always be retried safely.
"""

from __future__ import annotations

import logging
import signal
import time

from qdrant_client import QdrantClient, models

from avs.api import db
from avs.api.inference_client import InferenceClient
from avs.api.times import offset
from avs.api.vectors import VectorStore, point_id
from avs.settings import Settings, get_settings

log = logging.getLogger("avs.worker")


def build_points(recording: dict, track: dict) -> list[models.PointStruct]:
    track_key = f"{recording['id']}/{track['track_id']}"
    start = recording["start_time"]
    shared = {
        "track_key": track_key,
        "recording_id": recording["id"],
        "track_id": track["track_id"],
        "label": track["label"],
        "camera_id": recording["camera_id"],
        "place": recording["place"],
        "track_t_start": track["t_start"],
        "track_t_end": track["t_end"],
        "abs_start": offset(start, track["t_start"]),
        "abs_end": offset(start, track["t_end"]),
    }
    return [
        models.PointStruct(
            id=point_id(track_key, i),
            vector=crop["embedding"],
            payload=shared | {"t": crop["t"], "abs_t": offset(start, crop["t"]), "crop_path": crop["path"], "box": crop["box"]},
        )
        for i, crop in enumerate(track["crops"])
    ]


class Worker:
    def __init__(self, settings: Settings, inference: InferenceClient, store: VectorStore) -> None:
        self.settings = settings
        self.inference = inference
        self.store = store
        self.stopping = False

    def run_job(self, job: dict) -> None:
        with db.connect(self.settings.db_path) as conn:
            recording = db.get_recording(conn, job["recording_id"])
            if recording is None:
                db.finish_job(conn, job["id"], "failed", "recording no longer exists")
                return
            db.update_recording(conn, recording["id"], status="indexing", progress=0.0, error=None)
            tracks: list[dict] = []
            try:
                for event in self.inference.index(recording["id"], recording["video_path"], self.settings.sample_fps):
                    kind = event["type"]
                    if kind == "playback":
                        db.update_recording(conn, recording["id"], playback_path=event["path"])
                    elif kind == "progress" and event.get("duration"):
                        db.update_recording(conn, recording["id"], progress=min(0.99, event["t"] / event["duration"]))
                    elif kind == "track":
                        tracks.append(event)
                    elif kind == "done":
                        log.info("indexed %s: %d tracks in %.1fs", recording["id"], event["tracks"], event["seconds"])
                points = [p for track in tracks for p in build_points(recording, track)]
                self.store.replace_recording(recording["id"], points)
                db.replace_tracks(conn, recording["id"], tracks)
                db.update_recording(
                    conn, recording["id"], status="indexed", progress=1.0, track_count=len(tracks), indexed_at=db.now()
                )
                db.finish_job(conn, job["id"], "done")
            except Exception as exc:
                log.exception("job %s failed", job["id"])
                db.update_recording(conn, recording["id"], status="failed", error=str(exc))
                db.finish_job(conn, job["id"], "failed", str(exc))

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        with db.connect(self.settings.db_path) as conn:
            if requeued := db.requeue_running_jobs(conn):
                log.info("requeued %d interrupted job(s)", requeued)
        while not self.stopping:
            with db.connect(self.settings.db_path) as conn:
                job = db.claim_job(conn)
            if job is None:
                time.sleep(poll_seconds)
                continue
            self.run_job(job)


def wait_for_inference(inference: InferenceClient, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        try:
            health = inference.health()
            if health.get("status") == "ok":
                return health
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("inference service did not become ready")
        time.sleep(2.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    db.init_db(settings.db_path)
    inference = InferenceClient(settings.inference_url)
    health = wait_for_inference(inference)
    store = VectorStore(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection, health["dim"])
    store.ensure_collection()
    worker = Worker(settings, inference, store)

    def stop(*_):
        worker.stopping = True
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    log.info("worker ready")
    worker.run_forever()


if __name__ == "__main__":
    main()
