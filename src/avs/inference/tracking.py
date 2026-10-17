"""Per-camera tracking with Roboflow `trackers` ByteTrack.

People and bags are tracked by separate trackers, so a bag box can never take over a
person's track; the three bag classes share one tracker because the detector often
flips between them for the same object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from avs.inference.detector import BAG_LABELS, Detection


def group_of(label: str) -> str:
    return "bag" if label in BAG_LABELS else "person"


@dataclass(frozen=True)
class Tracked:
    track_id: str  # unique within one video, such as "person-3" or "bag-1"
    detection: Detection


class MultiClassTracker:
    def __init__(self, sample_fps: float, lost_seconds: float = 2.0) -> None:
        self._trackers = {
            group: ByteTrackTracker(
                lost_track_buffer=int(round(lost_seconds * 30)),
                frame_rate=sample_fps,
                track_activation_threshold=0.5,
                high_conf_det_threshold=0.5,
                minimum_consecutive_frames=2,
                minimum_iou_threshold=0.1,
            )
            for group in ("person", "bag")
        }

    def update(self, t: float, detections: list[Detection]) -> list[Tracked]:
        out: list[Tracked] = []
        for group, tracker in self._trackers.items():
            members = [d for d in detections if group_of(d.label) == group]
            if members:
                sv_dets = sv.Detections(
                    xyxy=np.array([d.box for d in members], dtype=np.float32),
                    confidence=np.array([d.confidence for d in members], dtype=np.float32),
                    class_id=np.zeros(len(members), dtype=int),
                    data={"index": np.arange(len(members))},
                )
            else:
                sv_dets = sv.Detections.empty()
            result = tracker.update(sv_dets, timestamp=t)
            if len(result) == 0:
                continue
            for index, tracker_id in zip(result.data["index"], result.tracker_id, strict=True):
                if tracker_id >= 0:
                    out.append(Tracked(f"{group}-{int(tracker_id)}", members[int(index)]))
        return out
