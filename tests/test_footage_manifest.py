"""Guards on the starter footage list: allowed licences only, excluded sources stay out."""

from __future__ import annotations

import pytest

from avs.api.times import parse_local_time
from avs.tools.fetch_footage import load_manifest

MANIFEST = load_manifest()
EXCLUDED = ("virat", "mixkit", "viratdata", "envato")


@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda c: c["id"])
def test_clip_is_complete_and_licence_clean(clip):
    source = MANIFEST["sources"][clip["source"]]
    assert "NC" not in source["licence"].split()
    assert not any(word in clip["url"].lower() for word in EXCLUDED)
    assert clip["use"] in ("demo", "neutral-only")
    parse_local_time(clip["start_time"])
    assert clip["camera_id"] and clip["place"]


def test_stock_footage_is_never_a_demo_story():
    for clip in MANIFEST["clips"]:
        if clip["source"] == "pexels":
            assert clip["use"] == "neutral-only" and clip["stories"] == []


def test_clip_ids_are_unique_and_camera_times_do_not_collide():
    ids = [c["id"] for c in MANIFEST["clips"]]
    keys = [(c["camera_id"], c["start_time"]) for c in MANIFEST["clips"]]
    assert len(set(ids)) == len(ids) and len(set(keys)) == len(keys)
