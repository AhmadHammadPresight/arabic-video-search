"""Search and recording traces through the API: returned with ?trace=1, stored for every search, reopened by id."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_api import FakeInference, FakeParser, index_all, upload, wait_for_check

from avs.api.app import Services, create_app
from avs.api.checker import CheckRunner
from avs.api.trace import Trace, replace_stage
from avs.api.worker import Worker


@pytest.fixture
def indexed(settings, store, moving_box_video):
    inference = FakeInference()
    services = Services(settings, inference=inference, store=store, parser=FakeParser(),
                        checker=CheckRunner(inference.check, settings.db_path, top=8, model="fake-vlm"))
    client = TestClient(create_app(services))
    rec = upload(client, moving_box_video).json()
    index_all(Worker(settings, inference, store), settings)
    return client, rec, inference


def test_trace_timing_order_and_replacement():
    trace = Trace("s", "hybrid", {"query": "q"})
    with trace.stage("input") as stage:
        stage["output"] = {"text": "q"}
    with pytest.raises(RuntimeError), trace.stage("parse"):
        raise RuntimeError("boom")  # a failing stage is still recorded with its time
    data = trace.finish()
    assert [s["key"] for s in data["stages"]] == ["input", "parse"] and data["took_ms"] >= 0
    assert data["stages"][0]["title_ar"] == "النص المدخل" and data["stages"][0]["ms"] >= 0
    replace_stage(data, {"key": "input", "output": {"text": "new"}})  # replaced in place
    assert [s["key"] for s in data["stages"]] == ["input", "parse"] and data["stages"][0]["output"]["text"] == "new"
    replace_stage(data, {"key": "check", "output": {}})  # a new stage goes at the end
    assert [s["key"] for s in data["stages"]] == ["input", "parse", "check"]


def test_option_a_trace_shows_its_stages_and_models(indexed):
    client, rec, _ = indexed
    found = client.post("/api/search?trace=1", json={"query": "أحمر", "mode": "baseline"}).json()
    trace = found["trace"]
    assert trace["search_id"] == found["search_id"] and trace["mode"] == "baseline"
    assert [s["key"] for s in trace["stages"]] == ["input", "embed", "filter", "candidates", "ranking"]
    assert trace["models"]["embedder"].startswith("google/siglip2-so400m-patch16-384@")
    stages = {s["key"]: s["output"] for s in trace["stages"]}
    assert stages["filter"]["labels"] == "all" and stages["filter"]["qdrant_filter"] is None
    assert [r["track_key"] for r in stages["candidates"]["rows"]] == [r["track_key"] for r in found["results"]]
    assert stages["candidates"]["rows"][0]["similarity"] == found["results"][0]["score"]
    # Without ?trace=1 the response stays as before, but the trace is still stored.
    plain = client.post("/api/search", json={"query": "أحمر", "mode": "baseline"}).json()
    assert "trace" not in plain
    assert client.get(f"/api/search/{plain['search_id']}/trace").json()["stages"][0]["key"] == "input"


def test_hybrid_trace_is_stored_and_gains_the_check_and_final_ranking(indexed):
    client, rec, _ = indexed
    found = client.post("/api/search", params={"trace": 1}, json={"query": "مرة لابسة جاكيت أحمر", "cameras": ["G421"]}).json()
    trace = found["trace"]
    assert [s["key"] for s in trace["stages"]] == ["input", "parse", "embed", "filter", "candidates", "attributes", "fusion", "ranking"]
    assert trace["models"]["query_model"] == {"model": "fake-query-model", "status": "ok"} and trace["models"]["vision_model"]
    assert trace["request"]["cameras"] == ["G421"] and trace["request"]["mode"] == "hybrid"
    wait_for_check(client, found["search_id"])
    stored = client.get(f"/api/search/{found['search_id']}/trace").json()
    keys = [s["key"] for s in stored["stages"]]
    assert keys[-2:] == ["check", "final"]
    check = stored["stages"][-2]["output"]
    assert check["status"] == "done" and check["model"] == "fake-vlm"
    assert check["verdicts"][f"{rec['id']}/person-1"]["answers"] == {"a woman": "yes", "red jacket": "yes"}
    assert stored["stages"][-1]["output"][0]["check"] == "confirmed"


def test_unknown_search_has_no_trace_or_check(indexed):
    client, *_ = indexed
    assert client.get("/api/search/nope/trace").status_code == 404
    assert client.get("/api/search/nope/check").status_code == 404


def test_recording_trace_shows_rules_frames_tracks_crops_and_attributes(indexed):
    client, rec, _ = indexed
    trace = client.get(f"/api/recordings/{rec['id']}/trace?frames=3").json()
    assert trace["recording"]["id"] == rec["id"]
    assert trace["stats"]["frames"] == 12 and trace["stats"]["detections"] == 20 and trace["stats"]["sample_fps"] == 4
    assert trace["stats"]["models"] == {"dim": 4}  # what the fake inference service reports about itself
    assert trace["rules"]["crops_per_track"] == 5 and "confidence" in trace["rules"]["quality"]
    frames = trace["frames"]["frames"]
    assert [f["t"] for f in frames] == [0.5, 1.5, 2.5]  # evenly spread, snapped to the 4 fps sampling grid
    assert frames[0]["url"] == f"/media/trace/{rec['id']}/frame-0.5.jpg" and frames[0]["detections"][0]["label"] == "person"
    assert {t["track_id"] for t in frames[0]["tracks"]} == {"person-1", "bag-1"}  # indexed boxes at that moment
    assert frames[1]["tracks"] == []  # no stored box within half a sampling step of 1.5 s
    person = next(t for t in trace["tracks"] if t["track_id"] == "person-1")
    assert person["crops"][0]["url"].startswith("/media/crops/") and person["crops"][0]["quality"] == 0.8
    assert person["selection"]["candidates"][0]["reason"] == "picked: best quality"
    assert person["attributes"]["attributes"]["gender"] == "female" and person["detections"] == 2
    custom = client.get(f"/api/recordings/{rec['id']}/trace?at=1.25,2").json()
    assert [f["t"] for f in custom["frames"]["frames"]] == [1.25, 2.0]
    assert client.get(f"/api/recordings/{rec['id']}/trace?at=x").status_code == 422


def test_recording_trace_reports_a_frame_failure_without_failing(indexed):
    client, rec, inference = indexed

    def broken(*args):
        raise RuntimeError("decoder busy")

    inference.trace_frames = broken
    trace = client.get(f"/api/recordings/{rec['id']}/trace").json()
    assert trace["frames"] is None and "decoder busy" in trace["frames_error"] and len(trace["tracks"]) == 2
