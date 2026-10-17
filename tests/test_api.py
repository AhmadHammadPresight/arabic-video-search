"""The web API with a fake inference service and an in-memory Qdrant."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from helpers import unit

from avs.api import db
from avs.api.app import Services, create_app
from avs.api.worker import Worker

RED, BLUE = unit(1, 0, 0, 0), unit(0, 1, 0, 0)


class FakeInference:
    """Indexes every upload as one red-shirt person and one blue bag."""

    def health(self) -> dict:
        return {"status": "ok", "dim": 4}

    def embed_text(self, text: str) -> list[float]:
        return RED if "أحمر" in text else BLUE

    def index(self, recording_id: str, video_path: str, sample_fps: float | None = None):
        yield {"type": "playback", "path": f"playback/{recording_id}.mp4"}
        yield {"type": "progress", "t": 1.0, "duration": 3.0}
        for track_id, label, vector in (("person-1", "person", RED), ("bag-1", "backpack", BLUE)):
            yield {"type": "track", "track_id": track_id, "label": label, "t_start": 0.5, "t_end": 2.5,
                   "boxes": [[0.5, 0.1, 0.1, 0.3, 0.9], [2.5, 0.2, 0.1, 0.4, 0.9]],
                   "crops": [{"t": 1.0, "box": [0.1, 0.1, 0.3, 0.9], "quality": 0.8,
                              "path": f"crops/{recording_id}/{track_id}-0.jpg", "embedding": vector}]}
        yield {"type": "done", "tracks": 2, "frames": 12, "seconds": 0.1}


@pytest.fixture
def env(settings, store, moving_box_video):
    services = Services(settings, inference=FakeInference(), store=store)
    client = TestClient(create_app(services))
    worker = Worker(settings, services.inference, store)
    return client, worker, settings, moving_box_video


def upload(client, video, camera="G421", start="2018-03-15T14:52:30"):
    with video.open("rb") as fh:
        return client.post("/api/recordings", data={"camera_id": camera, "place": "مقهى · cafe", "start_time": start, "actor": "officer-1"},
                           files={"file": ("cafe.mp4", fh, "video/mp4")})


def index_all(worker, settings):
    with db.connect(settings.db_path) as conn:
        while (job := db.claim_job(conn)) is not None:
            worker.run_job(job)


def test_upload_index_search_confirm_and_audit(env):
    client, worker, settings, video = env
    response = upload(client, video)
    assert response.status_code == 201, response.text
    rec = response.json()
    assert rec["status"] == "queued" and rec["duration_s"] == pytest.approx(3, abs=0.2)
    assert (settings.data_dir / rec["video_path"]).is_file()

    index_all(worker, settings)
    rec = client.get(f"/api/recordings/{rec['id']}").json()
    assert rec["status"] == "indexed" and rec["track_count"] == 2 and rec["playback_path"]
    assert len(client.get(f"/api/recordings/{rec['id']}/tracks").json()) == 2
    assert client.get("/api/cameras").json() == [{"camera_id": "G421", "places": ["مقهى · cafe"], "recordings": 1}]

    found = client.post("/api/search", json={"query": "رجل يرتدي قميص أحمر", "actor": "officer-1", "case_id": "C-7"}).json()
    top = found["results"][0]
    assert top["track_key"] == f"{rec['id']}/person-1" and top["camera_id"] == "G421"
    assert top["time"] == "2018-03-15T14:52:31" and top["clip_start"] == 0.0

    ok = client.post("/api/feedback", json={"search_id": found["search_id"], "track_key": top["track_key"], "verdict": "confirm", "actor": "officer-1"})
    assert ok.status_code == 201
    no = client.post("/api/feedback", json={"search_id": found["search_id"], "track_key": found["results"][1]["track_key"], "verdict": "reject"})
    assert no.status_code == 201

    actions = [(e["action"], e["actor"]) for e in client.get("/api/audit").json()]
    assert actions == [("reject", "anonymous"), ("confirm", "officer-1"), ("search", "officer-1"), ("upload", "officer-1")]
    search_entry = client.get("/api/audit").json()[2]
    assert search_entry["details"]["case_id"] == "C-7" and search_entry["details"]["result_count"] == 2


def test_feedback_must_point_at_a_shown_result(env):
    client, worker, settings, video = env
    upload(client, video)
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "أحمر", "limit": 1}).json()
    assert client.post("/api/feedback", json={"search_id": "nope", "track_key": "x", "verdict": "confirm"}).status_code == 404
    hidden = found["results"][0]["track_key"].replace("person-1", "bag-1")
    assert client.post("/api/feedback", json={"search_id": found["search_id"], "track_key": hidden, "verdict": "confirm"}).status_code == 422


def test_upload_validation(env, tmp_path):
    client, _, _, video = env
    assert upload(client, video, camera="bad id!").status_code == 422
    assert upload(client, video, start="2018-03-15T14:52:30+03:00").status_code == 422
    text = tmp_path / "x.mp4"
    text.write_text("hello")
    assert upload(client, text).status_code == 422
    assert client.get("/api/recordings").json() == []


def test_bad_time_window_is_rejected_and_still_logged(env):
    client, *_ = env
    response = client.post("/api/search", json={"query": "x", "time_from": "2018-03-15T15:00:00", "time_to": "2018-03-15T14:00:00"})
    assert response.status_code == 422
    [entry] = client.get("/api/audit").json()
    assert entry["action"] == "search" and "error" in entry["details"]


def test_delete_removes_files_index_and_is_audited(env):
    client, worker, settings, video = env
    rec = upload(client, video).json()
    index_all(worker, settings)
    (settings.data_dir / f"crops/{rec['id']}").mkdir(parents=True)
    assert client.delete(f"/api/recordings/{rec['id']}?actor=supervisor").status_code == 200
    assert not (settings.data_dir / rec["video_path"]).exists()
    assert not (settings.data_dir / f"crops/{rec['id']}").exists()
    assert client.post("/api/search", json={"query": "أحمر"}).json()["results"] == []
    assert client.get("/api/audit").json()[1]["action"] == "delete"


def test_only_derived_media_is_served(env):
    client, _, settings, video = env
    upload(client, video)
    assert client.get("/media/../avs.db").status_code == 404
    assert client.get("/media/uploads/anything.mp4").status_code == 404
    assert client.get("/").status_code == 200 and 'dir="rtl"' in client.get("/").text


def test_failed_indexing_marks_the_recording(env):
    client, worker, settings, video = env

    def broken(*_args, **_kwargs):
        raise RuntimeError("GPU fell over")
        yield  # pragma: no cover

    worker.inference.index = broken
    rec = upload(client, video).json()
    index_all(worker, settings)
    rec = client.get(f"/api/recordings/{rec['id']}").json()
    assert rec["status"] == "failed" and "GPU fell over" in rec["error"]
    assert client.post(f"/api/recordings/{rec['id']}/reindex").json()["status"] == "queued"


def test_health_reports_each_service_on_its_own(settings, store):
    inference = FakeInference()
    client = TestClient(create_app(Services(settings, inference=inference, store=store)))
    assert client.get("/api/health").json() | {"inference": None} == {"api": "ok", "inference": None, "qdrant": "ok"}

    def down():
        raise ConnectionError("inference stopped")

    inference.health = down
    health = client.get("/api/health").json()
    assert health["inference"]["status"] == "down" and health["qdrant"] == "ok"
