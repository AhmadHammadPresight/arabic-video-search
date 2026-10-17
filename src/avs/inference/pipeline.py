"""Index one recording: sample frames, detect, track, keep the best crops, embed them.

`index_video` yields plain dict events so the service can stream them as NDJSON:
`info` once, `progress` every few seconds of video, one `track` per finished track,
and `done` at the end. Tracks are finished as soon as they go stale, so memory stays
bounded by the people on screen rather than by the length of the video.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from avs.inference.crops import (
    MIN_CROP_HEIGHT,
    Candidate,
    TrackCandidates,
    crop_quality,
    padded_box,
    select_crops,
)
from avs.inference.detector import Detection, Detector
from avs.inference.embedder import Embedder
from avs.inference.tracking import MultiClassTracker, group_of
from avs.media import VideoInfo, iter_frames, probe

# Crops are stored no taller than this; SigLIP 2 itself sees 384x384.
CROP_MAX_HEIGHT = 512
PROGRESS_EVERY_SECONDS = 2.0


@dataclass(frozen=True)
class IndexOptions:
    sample_fps: float = 4.0
    crops_per_track: int = 5
    min_track_detections: int = 3
    # A track not seen for this long is finished; ByteTrack drops it after 2 s anyway.
    stale_after: float = 4.0


@dataclass
class _TrackState:
    track_id: str
    group: str
    boxes: list[tuple[float, tuple[float, float, float, float]]] = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)
    pool: TrackCandidates = field(default_factory=TrackCandidates)

    @property
    def last_t(self) -> float:
        return self.boxes[-1][0]

    def add(self, t: float, det: Detection, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self.boxes.append((t, det.box))
        self.labels[det.label] += 1
        if det.box[3] - det.box[1] < MIN_CROP_HEIGHT[self.group]:
            return
        quality = crop_quality(det.box, det.confidence, width, height)
        if not self.pool.would_accept(quality):
            return
        x1, y1, x2, y2 = padded_box(det.box, width, height)
        image = Image.fromarray(frame[y1:y2, x1:x2])
        if image.height > CROP_MAX_HEIGHT:
            image = image.resize((max(1, round(image.width * CROP_MAX_HEIGHT / image.height)), CROP_MAX_HEIGHT))
        self.pool.offer(Candidate(t=t, box=det.box, quality=quality, image=image))


def index_video(
    video: Path,
    crops_dir: Path,
    crops_rel: str,
    detector: Detector,
    embedder: Embedder,
    options: IndexOptions | None = None,
    info: VideoInfo | None = None,
) -> Iterator[dict]:
    started = time.monotonic()
    options = options or IndexOptions()
    info = info or probe(video)
    crops_dir.mkdir(parents=True, exist_ok=True)
    yield {"type": "info", "duration": info.duration, "width": info.width, "height": info.height, "fps": info.fps}

    tracker = MultiClassTracker(options.sample_fps)
    active: dict[str, _TrackState] = {}
    finished = frames = 0
    next_progress = 0.0

    def finish(state: _TrackState) -> dict | None:
        return _finish_track(state, info, crops_dir, crops_rel, embedder, options)

    for t, frame in iter_frames(video, options.sample_fps, info):
        frames += 1
        for tracked in tracker.update(t, detector.detect(frame)):
            state = active.get(tracked.track_id)
            if state is None:
                state = active[tracked.track_id] = _TrackState(tracked.track_id, group_of(tracked.detection.label))
            state.add(t, tracked.detection, frame)
        for track_id in [k for k, s in active.items() if t - s.last_t > options.stale_after]:
            if (event := finish(active.pop(track_id))) is not None:
                finished += 1
                yield event
        if t >= next_progress:
            next_progress = t + PROGRESS_EVERY_SECONDS
            yield {"type": "progress", "t": t, "duration": info.duration}

    for state in active.values():
        if (event := finish(state)) is not None:
            finished += 1
            yield event
    yield {"type": "done", "tracks": finished, "frames": frames, "seconds": round(time.monotonic() - started, 2)}


def _finish_track(
    state: _TrackState,
    info: VideoInfo,
    crops_dir: Path,
    crops_rel: str,
    embedder: Embedder,
    options: IndexOptions,
) -> dict | None:
    candidates = state.pool.candidates()
    if len(state.boxes) < options.min_track_detections or not candidates:
        return None
    t_start, t_end = state.boxes[0][0], state.boxes[-1][0]
    min_gap = max(0.5, (t_end - t_start) / (2 * options.crops_per_track))
    chosen = select_crops(candidates, options.crops_per_track, min_gap)
    vectors = embedder.embed_images([c.image for c in chosen])
    crops = []
    for i, (cand, vector) in enumerate(zip(chosen, vectors, strict=True)):
        name = f"{state.track_id}-{i}.jpg"
        cand.image.save(crops_dir / name, quality=90)
        crops.append({
            "t": round(cand.t, 3),
            "box": _normalise_box(cand.box, info),
            "quality": round(cand.quality, 4),
            "path": f"{crops_rel}/{name}",
            "embedding": [round(float(v), 6) for v in vector],
        })
    label = "person" if state.group == "person" else state.labels.most_common(1)[0][0]
    return {
        "type": "track",
        "track_id": state.track_id,
        "label": label,
        "t_start": round(t_start, 3),
        "t_end": round(t_end, 3),
        "boxes": [[round(t, 3), *_normalise_box(box, info)] for t, box in state.boxes],
        "crops": crops,
    }


def _normalise_box(box: tuple[float, float, float, float], info: VideoInfo) -> list[float]:
    x1, y1, x2, y2 = box
    w, h = info.width, info.height
    return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
