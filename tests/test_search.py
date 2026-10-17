from __future__ import annotations

import pytest

from avs.api.search import MAX_CLIP_SECONDS, SearchQuery, clip_window, run_search
from avs.api.times import from_epoch, offset, parse_local_time, to_epoch
from avs.api.worker import build_points
from helpers import unit

RED, BLUE = unit(1, 0, 0, 0), unit(0, 1, 0, 0)


def recording(rid: str, camera: str, start: str) -> dict:
    return {"id": rid, "camera_id": camera, "place": f"place {camera}", "start_time": start,
            "duration_s": 120.0, "playback_path": f"playback/{rid}.mp4"}


def track(track_id: str, vectors: list[list[float]], t0: float) -> dict:
    return {
        "track_id": track_id, "label": "person", "t_start": t0, "t_end": t0 + 10,
        "crops": [{"t": t0 + i, "box": [0.1, 0.1, 0.3, 0.9], "path": f"crops/x/{track_id}-{i}.jpg", "embedding": v}
                  for i, v in enumerate(vectors)],
    }


@pytest.fixture
def indexed(store):
    recs = {
        "a": recording("a", "CAM1", "2018-03-15T14:00:00"),
        "b": recording("b", "CAM2", "2018-03-15T15:00:00"),
    }
    store.replace_recording("a", build_points(recs["a"], track("person-1", [RED, unit(1, 0.2, 0, 0)], 5.0))
                            + build_points(recs["a"], track("person-2", [BLUE], 50.0)))
    store.replace_recording("b", build_points(recs["b"], track("person-1", [unit(1, 0.1, 0, 0)], 20.0)))
    return recs


def fake_embed(text: str) -> list[float]:
    return RED if "أحمر" in text or "red" in text else BLUE


def test_times_round_trip_and_reject_offsets():
    assert from_epoch(to_epoch(parse_local_time("2018-03-15T14:52:30"))) == "2018-03-15T14:52:30"
    assert from_epoch(offset("2018-03-15T14:52:30", 1.5)) == "2018-03-15T14:52:31.500"
    with pytest.raises(ValueError):
        parse_local_time("2018-03-15T14:52:30+03:00")


def test_search_groups_by_track_and_ranks_by_similarity(store, indexed):
    results = run_search(store, fake_embed, indexed, SearchQuery("رجل يرتدي قميص أحمر", limit=10))
    keys = [r["track_key"] for r in results]
    assert keys[:2] in (["a/person-1", "b/person-1"], ["b/person-1", "a/person-1"])
    assert keys[-1] == "a/person-2" and len(keys) == 3  # one result per track, not per crop
    top = results[0]
    assert top["rank"] == 1 and top["score"] == pytest.approx(1.0, abs=1e-3)
    assert top["video_url"].startswith("/media/playback/") and top["crop_url"].startswith("/media/crops/")


def test_camera_and_time_filters(store, indexed):
    only_cam2 = run_search(store, fake_embed, indexed, SearchQuery("red", cameras=["CAM2"]))
    assert [r["track_key"] for r in only_cam2] == ["b/person-1"]
    # Window 14:00:40-14:01:30 overlaps person-2 (14:00:50-14:01:00) only.
    window = run_search(store, fake_embed, indexed, SearchQuery("red", time_from="2018-03-15T14:00:40", time_to="2018-03-15T14:01:30"))
    assert [r["track_key"] for r in window] == ["a/person-2"]
    assert window[0]["seen_from"] == "2018-03-15T14:00:50"


def test_reversed_time_window_is_an_error(store, indexed):
    with pytest.raises(ValueError):
        run_search(store, fake_embed, indexed, SearchQuery("red", time_from="2018-03-15T15:00:00", time_to="2018-03-15T14:00:00"))


def test_replacing_a_recording_removes_its_old_points(store, indexed):
    store.replace_recording("a", [])
    keys = [r["track_key"] for r in run_search(store, fake_embed, indexed, SearchQuery("red"))]
    assert keys == ["b/person-1"]


def test_clip_window_pads_and_caps_around_the_best_crop():
    assert clip_window(5.0, 3.0, 8.0, 60.0) == (2.0, 9.0)
    assert clip_window(1.0, 0.0, 8.0, 5.0) == (0.0, 5.0)
    start, end = clip_window(100.0, 10.0, 200.0, 300.0)
    assert end - start == MAX_CLIP_SECONDS and start <= 100.0 <= end
