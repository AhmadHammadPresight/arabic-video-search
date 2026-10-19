"""The web API with a fake inference service, a fake query model and an in-memory Qdrant."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from helpers import unit

from avs.api import db
from avs.api.app import Services, create_app
from avs.api.checker import FINISHED, CheckRunner
from avs.api.query_parser import ParseError, ParseResult
from avs.api.worker import Worker, prune_crops
from avs.fields import QueryFields

RED, BLUE = unit(1, 0, 0, 0), unit(0, 1, 0, 0)
PERSON_ATTRS = {"subject": "person", "gender": "female", "upper": {"garment": "jacket", "colours": ["red"]}}
BAG_ATTRS = {"subject": "bag", "carried": [{"object": "backpack", "colours": ["blue"]}]}


class FakeInference:
    """Indexes every upload as one woman in a red jacket and one blue backpack."""

    def __init__(self) -> None:
        self.checked: list[list[str]] = []

    def health(self) -> dict:
        return {"status": "ok", "dim": 4}

    def embed_text(self, text: str) -> list[float]:
        return RED if "أحمر" in text or "red" in text else BLUE

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def check(self, crops: list[str], subject: str, english: str, details: list[str]) -> dict:
        self.checked.append(crops)
        yes = "bag" not in crops[0]
        return {"answers": {d: "yes" if yes else "no" for d in details}, "note": "fake", "ms": 1, "raw": "{}"}

    def trace_frames(self, recording_id: str, video_path: str, times: list[float]) -> dict:
        return {"width": 320, "height": 240, "fps": 10, "duration": 3.0, "detector": "fake", "threshold": 0.35, "frames": [
            {"t": t, "path": f"trace/{recording_id}/frame-{t}.jpg", "detect_ms": 1,
             "detections": [{"label": "person", "confidence": 0.9, "box": [0.1, 0.1, 0.3, 0.9]}]} for t in times]}

    def index(self, recording_id: str, video_path: str, sample_fps: float | None = None):
        yield {"type": "playback", "path": f"playback/{recording_id}.mp4"}
        yield {"type": "progress", "t": 1.0, "duration": 3.0}
        for track_id, label, vector, attrs in (("person-1", "person", RED, PERSON_ATTRS), ("bag-1", "backpack", BLUE, BAG_ATTRS)):
            yield {"type": "track", "track_id": track_id, "label": label, "t_start": 0.5, "t_end": 2.5,
                   "boxes": [[0.5, 0.1, 0.1, 0.3, 0.9], [2.5, 0.2, 0.1, 0.4, 0.9]],
                   "crops": [{"t": 1.0, "box": [0.1, 0.1, 0.3, 0.9], "quality": 0.8,
                              "path": f"crops/{recording_id}/{track_id}-0.jpg", "embedding": vector}],
                   "selection": {"min_gap": 0.5, "pool_capacity": 12, "detections": 2, "labels": {label: 2},
                                 "candidates": [{"t": 1.0, "quality": 0.8, "chosen": True, "reason": "picked: best quality"}]},
                   "attributes": {"source": "fake", "attributes": attrs, "ms": 3}}
        yield {"type": "done", "tracks": 2, "frames": 12, "seconds": 0.1, "detections": 20, "dropped_short_tracks": 0}


class FakeParser:
    """A query model that knows two descriptions, and is down for anything with "down" in it."""

    model = "fake-query-model"

    def parse(self, text: str) -> ParseResult:
        if "down" in text:
            raise ParseError("the query model server is unreachable: fake")
        if "أحمر" in text or "red" in text:
            fields = QueryFields(gender="female", upper={"garment": "jacket", "colours": ["red"]}, english="a woman in a red jacket")
        else:
            fields = QueryFields(subject="bag", carried=[{"object": "backpack", "colours": ["blue"]}], english="a blue backpack")
        return ParseResult(fields=fields, raw=fields.model_dump_json(), model=self.model, took_ms=5)

    def info(self) -> dict:
        return {"model": self.model, "status": "ok"}


@pytest.fixture
def env(settings, store, moving_box_video):
    inference = FakeInference()
    checker = CheckRunner(inference.check, settings.db_path, top=8, model="fake-vlm")
    services = Services(settings, inference=inference, store=store, parser=FakeParser(), checker=checker)
    client = TestClient(create_app(services))
    worker = Worker(settings, services.inference, store)
    return client, worker, settings, moving_box_video


def wait_for_check(client, search_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        status = client.get(f"/api/search/{search_id}/check").json()
        if status["status"] in FINISHED or time.monotonic() > deadline:
            return status
        time.sleep(0.05)


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

    found = client.post("/api/search", json={"query": "رجل يرتدي قميص أحمر", "actor": "officer-1", "case_id": "C-7", "mode": "baseline"}).json()
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
    found = client.post("/api/search", json={"query": "أحمر", "limit": 1, "mode": "baseline"}).json()
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
    response = client.post("/api/search", json={"query": "x", "time_from": "2018-03-15T15:00:00", "time_to": "2018-03-15T14:00:00", "mode": "baseline"})
    assert response.status_code == 422
    [entry] = client.get("/api/audit").json()
    assert entry["action"] == "search" and "error" in entry["details"]


def test_delete_removes_files_index_and_is_audited(env):
    client, worker, settings, video = env
    rec = upload(client, video).json()
    index_all(worker, settings)
    (settings.data_dir / f"crops/{rec['id']}").mkdir(parents=True)
    (settings.data_dir / f"trace/{rec['id']}").mkdir(parents=True)
    assert client.delete(f"/api/recordings/{rec['id']}?actor=supervisor").status_code == 200
    assert not (settings.data_dir / rec["video_path"]).exists()
    assert not (settings.data_dir / f"crops/{rec['id']}").exists()
    assert client.post("/api/search", json={"query": "أحمر", "mode": "baseline"}).json()["results"] == []
    assert client.post("/api/search", json={"query": "أحمر"}).json()["results"] == []
    assert not (settings.data_dir / f"trace/{rec['id']}").exists()
    assert client.get("/api/audit").json()[2]["action"] == "delete"


def test_only_derived_media_is_served(env):
    client, _, settings, video = env
    upload(client, video)
    assert client.get("/media/../avs.db").status_code == 404
    assert client.get("/media/uploads/anything.mp4").status_code == 404
    assert client.get("/").status_code == 200 and 'dir="rtl"' in client.get("/").text


def test_the_technical_view_serves_the_same_page(env):
    client = env[0]
    page = client.get("/")
    assert page.status_code == 200
    for path in ("/tech", "/tech/"):
        tech = client.get(path)
        assert tech.status_code == 200, path
        assert tech.text == page.text
        assert tech.headers["content-type"].startswith("text/html")
        assert tech.headers["cache-control"] == "no-cache"
    # The page's script, its translation module and both tables load from either view.
    for path in ("/app.js", "/i18n.js", "/locales/ar.json", "/locales/en.json"):
        assert client.get(path).status_code == 200, path


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
    client = TestClient(create_app(Services(settings, inference=inference, store=store, parser=FakeParser())))
    assert client.get("/api/health").json() | {"inference": None} == {
        "api": "ok", "inference": None, "qdrant": "ok", "query_model": {"model": "fake-query-model", "status": "ok"}}

    def down():
        raise ConnectionError("inference stopped")

    inference.health = down
    health = client.get("/api/health").json()
    assert health["inference"]["status"] == "down" and health["qdrant"] == "ok"


def test_hybrid_search_parses_matches_attributes_and_double_checks(env):
    client, worker, settings, video = env
    rec = upload(client, video).json()
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "مرة لابسة جاكيت أحمر", "actor": "officer-1"}).json()
    assert found["mode"] == "hybrid" and found["parsed"]["english"] == "a woman in a red jacket"
    keys = [r["track_key"] for r in found["results"]]
    assert keys == [f"{rec['id']}/person-1"]  # the subject filter keeps people only
    top = found["results"][0]
    assert top["attribute_score"] == 1.0 and top["described"]
    assert {(w["field"], w["result"]) for w in top["why"]} == {("gender", "match"), ("upper_colour", "match"), ("upper_garment", "match")}
    assert found["check"]["status"] in ("queued", "running", "done") and found["check"]["total"] == 1

    check = wait_for_check(client, found["search_id"])
    assert check["status"] == "done"
    assert check["verdicts"][top["track_key"]]["verdict"] == "confirmed"
    assert check["results"][0]["check"] == "confirmed" and check["results"][0]["score_checked"] == top["score"] + 2.0
    actions = [e["action"] for e in client.get("/api/audit").json()]
    assert actions[:2] == ["check", "search"]
    # Feedback still works on a re-ranked result.
    ok = client.post("/api/feedback", json={"search_id": found["search_id"], "track_key": top["track_key"], "verdict": "confirm"})
    assert ok.status_code == 201


def test_bag_query_only_returns_bags_and_the_check_can_reject(env):
    client, worker, settings, video = env
    rec = upload(client, video).json()
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "شنطة ظهر زرقا"}).json()
    assert [r["label"] for r in found["results"]] == ["backpack"]
    check = wait_for_check(client, found["search_id"])
    assert check["verdicts"][f"{rec['id']}/bag-1"]["verdict"] == "rejected"  # the fake vision model says no to bags


def test_option_a_stays_selectable_with_unchanged_results(env):
    client, worker, settings, video = env
    upload(client, video)
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "مرة لابسة جاكيت أحمر", "mode": "baseline"}).json()
    assert found["mode"] == "baseline" and "parsed" not in found and "check" not in found
    assert [r["label"] for r in found["results"]] == ["person", "backpack"]  # no subject filter in option A
    assert "why" not in found["results"][0] and found["results"][0]["score"] == pytest.approx(1.0, abs=1e-3)


def test_hybrid_falls_back_to_option_a_when_the_query_model_is_down(env):
    client, worker, settings, video = env
    upload(client, video)
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "red down", "trace": True}, params={"trace": 1}).json()
    assert found["mode"] == "baseline" and "check" not in found
    assert [r["label"] for r in found["results"]] == ["person", "backpack"]
    assert found["trace"]["mode"] == "baseline" and "option A" in found["trace"]["notes"][0]
    parse = next(s for s in found["trace"]["stages"] if s["key"] == "parse")
    assert "unreachable" in parse["output"]["error"]
    audit = client.get("/api/audit").json()[0]
    assert audit["details"]["mode"] == "baseline" and audit["details"]["english"] is None


def test_old_crops_are_removed_only_after_a_new_run_replaced_the_points(settings):
    folder = settings.data_dir / "crops" / "rec1"
    for name in ("20261017T100000-aaaaaa/person-1-0.jpg", "20261018T090000-bbbbbb/person-1-0.jpg", "person-1-0.jpg"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"jpg")
    prune_crops(settings, "rec1", "crops/rec1")  # not a run folder (an older inference service): nothing is removed
    assert len(list(folder.rglob("*.jpg"))) == 3
    prune_crops(settings, "rec1", "crops/rec1/20261018T090000-bbbbbb")
    assert [p.relative_to(folder).as_posix() for p in folder.rglob("*.jpg")] == ["20261018T090000-bbbbbb/person-1-0.jpg"]


def test_a_check_can_be_cancelled_and_the_trace_shows_its_state(env):
    client, worker, settings, video = env
    upload(client, video)
    index_all(worker, settings)
    found = client.post("/api/search", json={"query": "مرة لابسة جاكيت أحمر"}).json()
    cancelled = client.post(f"/api/search/{found['search_id']}/check/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] in ("cancelled", "running", "done")
    final = wait_for_check(client, found["search_id"])
    assert final["status"] in ("cancelled", "done")
    stage = next(s for s in client.get(f"/api/search/{found['search_id']}/trace").json()["stages"] if s["key"] == "check")
    assert stage["output"]["status"] == final["status"]
    assert client.post("/api/search/nope/check/cancel").status_code == 404
