from __future__ import annotations

from avs.inference.detector import Detection
from avs.inference.tracking import MultiClassTracker, group_of


def test_group_of():
    assert group_of("person") == "person"
    assert {group_of(x) for x in ("backpack", "handbag", "suitcase")} == {"bag"}


def test_two_walkers_and_a_bag_keep_their_own_tracks():
    tracker = MultiClassTracker(sample_fps=4)
    seen: dict[str, set[str]] = {"left": set(), "right": set(), "bag": set()}
    for i in range(12):
        t = i * 0.25
        dets = [
            Detection((100 + 8 * i, 100, 160 + 8 * i, 300), "person", 0.9),
            Detection((800 - 8 * i, 120, 860 - 8 * i, 320), "person", 0.85),
            # The detector flips the bag's class; it must stay one bag track.
            Detection((500, 400, 540, 440), "handbag" if i % 2 else "backpack", 0.7),
        ]
        for tracked in tracker.update(t, dets):
            x1 = tracked.detection.box[0]
            key = "bag" if tracked.track_id.startswith("bag") else ("left" if x1 < 450 else "right")
            seen[key].add(tracked.track_id)
    assert all(len(ids) == 1 for ids in seen.values()), seen
    assert len({next(iter(ids)) for ids in seen.values()}) == 3


def test_a_track_needs_two_frames_before_it_gets_an_id():
    tracker = MultiClassTracker(sample_fps=4)
    det = [Detection((10, 10, 60, 160), "person", 0.9)]
    assert tracker.update(0.0, det) == []
    assert len(tracker.update(0.25, det)) == 1
