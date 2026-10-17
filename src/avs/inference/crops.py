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


def crop_quality(box: Box, confidence: float, width: int, height: int) -> float:
    box_height = max(0.0, box[3] - box[1])
    size = min(1.0, box_height / FULL_DETAIL_HEIGHT)
    edge = 0.5 if touches_edge(box, width, height) else 1.0
    return confidence * size * edge


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
    ranked = sorted(candidates, key=lambda c: c.quality, reverse=True)
    chosen: list[Candidate] = []
    for cand in ranked:
        if len(chosen) == k:
            break
        if all(abs(cand.t - c.t) >= min_gap for c in chosen):
            chosen.append(cand)
    for cand in ranked:
        if len(chosen) == k:
            break
        if cand not in chosen:
            chosen.append(cand)
    return sorted(chosen, key=lambda c: c.t)
