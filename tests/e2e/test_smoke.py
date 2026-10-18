"""End-to-end smoke test: index one short clip through the running services and find a known person.

The clip is indexed with the attribute step, then searched in option A and in the hybrid mode
(query model, attribute match, vision-model double-check), and both traces are read back.

Needs `scripts/start.sh` running and the starter footage fetched (`scripts/fetch_footage.sh`):

    .venv/bin/pytest -m e2e
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

from avs.api.checker import FINISHED
from avs.api.times import parse_local_time, to_epoch
from avs.media import encode_h264
from avs.tools.evaluate import Anchor, match_anchor
from avs.tools.fetch_footage import clip_path

pytestmark = pytest.mark.e2e

API = f"http://127.0.0.1:{os.environ.get('AVS_WEB_PORT', '8470')}"
# 20 s of the cafe clip; the person in the red hoodie stands near the counter about 5 s in.
SOURCE_START, LENGTH = 35.0, 20.0
START_TIME = "2030-01-01T09:00:00"
# Centre of the red hoodie's box at cafe-clip second 40.0 (from eval/queries.yaml, target school-red-hoodie).
RED_HOODIE = (5.0, 0.262, 0.319)


@pytest.fixture(scope="module")
def api():
    client = httpx.Client(base_url=API, timeout=60.0)
    try:
        health = client.get("/api/health").json()
    except httpx.HTTPError:
        pytest.skip(f"services are not running at {API}; run scripts/start.sh")
    if health.get("inference", {}).get("status") != "ok":
        pytest.skip("inference service is not ready")
    return client


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    source = clip_path("meva-g421-cafe")
    if not source.exists():
        pytest.skip("starter footage not fetched; run scripts/fetch_footage.sh")
    out = tmp_path_factory.mktemp("smoke") / "cafe-20s.mp4"
    encode_h264(source, out, start=SOURCE_START, duration=LENGTH)
    return out


def test_index_one_clip_and_find_the_person_in_the_red_hoodie(api, clip):
    camera = f"SMOKE-{uuid.uuid4().hex[:6]}"
    with clip.open("rb") as fh:
        response = api.post(
            "/api/recordings",
            data={"camera_id": camera, "place": "اختبار · smoke test", "start_time": START_TIME, "actor": "smoke-test"},
            files={"file": (clip.name, fh, "video/mp4")},
        )
    assert response.status_code == 201, response.text
    recording = response.json()
    try:
        deadline = time.monotonic() + 600
        while recording["status"] not in ("indexed", "failed"):
            assert time.monotonic() < deadline, "indexing took too long"
            time.sleep(2)
            recording = api.get(f"/api/recordings/{recording['id']}").json()
        assert recording["status"] == "indexed", recording["error"]
        assert recording["track_count"] > 5

        tracks = {recording["id"]: api.get(f"/api/recordings/{recording['id']}/tracks").json()}
        t, x, y = RED_HOODIE
        anchor = Anchor(camera, to_epoch(parse_local_time(START_TIME)) + t, x, y)
        known = match_anchor(anchor, [recording], tracks)
        assert known, "no track at the red hoodie's position"

        found = api.post("/api/search", json={"query": "شخص يرتدي هودي أحمر", "cameras": [camera], "limit": 5,
                                              "actor": "smoke-test", "mode": "baseline"}).json()
        top = [r["track_key"] for r in found["results"]]
        assert known in top[:3], f"red hoodie {known} not in the top 3: {top}"
        hit = found["results"][top.index(known)]
        assert hit["camera_id"] == camera and hit["time"].startswith("2030-01-01T09:00:")
        assert api.get(hit["video_url"], headers={"Range": "bytes=0-99"}).status_code == 206
        assert api.get(hit["crop_url"]).status_code == 200

        verdict = api.post("/api/feedback", json={"search_id": found["search_id"], "track_key": known, "verdict": "confirm", "actor": "smoke-test"})
        assert verdict.status_code == 201
        recent = api.get("/api/audit?limit=5").json()
        assert [e["action"] for e in recent[:2]] == ["confirm", "search"]

        # The tracks were described by the vision model at index time.
        described = next(t for t in tracks[recording["id"]] if t["track_key"] == known)
        assert described["attributes"] and described["attributes"]["source"].startswith("vlm:")
        assert "red" in described["attributes"]["attributes"]["upper"]["colours"]

        # Hybrid: a Jordanian phrasing, parsed by the query model, matched on attributes, then double-checked.
        hybrid = api.post("/api/search?trace=1", json={"query": "واحد لابس هودي أحمر", "cameras": [camera], "limit": 5,
                                                        "actor": "smoke-test"}).json()
        assert hybrid["mode"] == "hybrid", "the query model is not answering"
        assert "red" in hybrid["parsed"]["upper"]["colours"] and "hoodie" in hybrid["parsed"]["english"].lower()
        assert hybrid["took_ms"] < 5000, f"first results took {hybrid['took_ms']} ms"
        keys = [r["track_key"] for r in hybrid["results"]]
        assert known in keys[:3], f"red hoodie {known} not in the hybrid top 3: {keys}"
        assert any(w["field"] == "upper_colour" and w["result"] == "match" for w in hybrid["results"][keys.index(known)]["why"])
        assert [s["key"] for s in hybrid["trace"]["stages"]] == [
            "input", "parse", "embed", "filter", "candidates", "attributes", "fusion", "ranking"]
        deadline = time.monotonic() + 120
        while (check := api.get(f"/api/search/{hybrid['search_id']}/check").json())["status"] not in FINISHED:
            assert time.monotonic() < deadline, "the double-check took too long"
            time.sleep(1)
        assert check["status"] == "done" and check["verdicts"][known]["verdict"] in ("confirmed", "unsure")
        assert [r["track_key"] for r in check["results"]][:3].count(known) == 1
        stored = api.get(f"/api/search/{hybrid['search_id']}/trace").json()
        assert [s["key"] for s in stored["stages"]][-2:] == ["check", "final"]

        rec_trace = api.get(f"/api/recordings/{recording['id']}/trace?frames=2").json()
        assert len(rec_trace["frames"]["frames"]) == 2 and rec_trace["frames"]["frames"][0]["detections"]
        assert api.get(rec_trace["frames"]["frames"][0]["url"]).status_code == 200
        assert rec_trace["stats"]["attributes_source"].startswith("vlm:")
        assert all(t["selection"]["candidates"] for t in rec_trace["tracks"])
    finally:
        api.delete(f"/api/recordings/{recording['id']}?actor=smoke-test")
