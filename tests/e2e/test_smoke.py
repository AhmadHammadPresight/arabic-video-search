"""End-to-end smoke test: index one short clip through the running services and find a known person.

Needs `scripts/start.sh` running and the starter footage fetched (`scripts/fetch_footage.sh`):

    .venv/bin/pytest -m e2e
"""

from __future__ import annotations

import os
import time
import uuid

import httpx
import pytest

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

        found = api.post("/api/search", json={"query": "شخص يرتدي هودي أحمر", "cameras": [camera], "limit": 5, "actor": "smoke-test"}).json()
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
    finally:
        api.delete(f"/api/recordings/{recording['id']}?actor=smoke-test")
