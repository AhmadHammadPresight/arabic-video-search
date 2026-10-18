"""Pick the few best crops of each track.

A crop scores higher when the detector is confident, the box is large (more detail for the
embedding), and the box does not touch the frame edge (a cut-off person describes badly).
The chosen crops are spread over the track so they show the target from different moments.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]

# Box height in pixels at which the size factor saturates.
FULL_DETAIL_HEIGHT = 256.0
# Below this height a crop is too small to describe.
MIN_CROP_HEIGHT = {"person": 48.0, "bag": 20.0}
# How many candidates a track keeps in memory before the final selection.
CANDIDATES_PER_TRACK = 12
EDGE_MARGIN = 2.0


def touches_edge(box: Box, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= width - EDGE_MARGIN or y2 >= height - EDGE_MARGIN


def quality_parts(box: Box, confidence: float, width: int, height: int) -> dict[str, float]:
    """The three factors of a crop's quality, kept so the recording trace can show why a crop was picked."""
    box_height = max(0.0, box[3] - box[1])
    return {
        "confidence": confidence,
        "size": min(1.0, box_height / FULL_DETAIL_HEIGHT),
        "edge": 0.5 if touches_edge(box, width, height) else 1.0,
    }


def crop_quality(box: Box, confidence: float, width: int, height: int) -> float:
    parts = quality_parts(box, confidence, width, height)
    return parts["confidence"] * parts["size"] * parts["edge"]


def padded_box(box: Box, width: int, height: int, pad: float = 0.08) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(width, int(np.ceil(x2 + dx))),
        min(height, int(np.ceil(y2 + dy))),
    )


@dataclass(eq=False)
class Candidate:
    t: float
    box: Box
    quality: float
    image: Image.Image = field(repr=False)
    parts: dict[str, float] = field(default_factory=dict, repr=False)


_tiebreak = itertools.count()


@dataclass
class TrackCandidates:
    """Bounded pool of the highest-quality crops seen so far for one track."""

    capacity: int = CANDIDATES_PER_TRACK
    _heap: list[tuple[float, int, Candidate]] = field(default_factory=list)

    def would_accept(self, quality: float) -> bool:
        return len(self._heap) < self.capacity or quality > self._heap[0][0]

    def offer(self, candidate: Candidate) -> None:
        item = (candidate.quality, next(_tiebreak), candidate)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, item)
        elif candidate.quality > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def candidates(self) -> list[Candidate]:
        return [item[2] for item in self._heap]


def select_crops(candidates: list[Candidate], k: int, min_gap: float) -> list[Candidate]:
    """Greedy pick by quality, skipping crops closer than `min_gap` seconds to one already picked.

    If the spacing rule leaves fewer than `k`, the best remaining crops fill the gap.
    """
    return select_crops_explained(candidates, k, min_gap)[0]


def select_crops_explained(candidates: list[Candidate], k: int, min_gap: float) -> tuple[list[Candidate], list[dict]]:
    """`select_crops`, plus one note per candidate saying why it was or was not picked."""
    ranked = sorted(candidates, key=lambda c: c.quality, reverse=True)
    chosen: list[Candidate] = []
    reasons: dict[int, str] = {}
    for cand in ranked:
        if len(chosen) == k:
            reasons.setdefault(id(cand), f"not needed: {k} crops already picked")
            continue
        near = next((c for c in chosen if abs(cand.t - c.t) < min_gap), None)
        if near is None:
            reasons[id(cand)] = "picked: best quality left" if chosen else "picked: best quality"
            chosen.append(cand)
        else:
            reasons[id(cand)] = f"skipped: within {min_gap:.1f} s of the better crop at {near.t:.2f} s"
    for cand in ranked:
        if len(chosen) == k:
            break
        if cand not in chosen:
            reasons[id(cand)] = "picked to fill up: spacing left too few crops"
            chosen.append(cand)
    picked = {id(c) for c in chosen}
    notes = [
        {"t": round(c.t, 3), "quality": round(c.quality, 4), **{k_: round(v, 4) for k_, v in c.parts.items()},
         "chosen": id(c) in picked, "reason": reasons[id(c)]}
        for c in sorted(candidates, key=lambda c: c.t)
    ]
    return sorted(chosen, key=lambda c: c.t), notes
