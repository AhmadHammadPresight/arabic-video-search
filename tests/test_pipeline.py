"""The indexing pipeline on a tiny synthetic clip, with a colour-threshold 'detector' and a fake embedder."""

from __future__ import annotations

import numpy as np
from PIL import Image

from avs.inference.detector import Detection
from avs.inference.pipeline import IndexOptions, index_video


class RedBoxDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        red = (frame[..., 0] > 150) & (frame[..., 1] < 80) & (frame[..., 2] < 80)
        ys, xs = np.nonzero(red)
        if len(xs) == 0:
            return []
        return [Detection((float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1), "person", 0.9)]


class MeanColourEmbedder:
    dim = 3

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        v = np.array([np.asarray(im, dtype=float).reshape(-1, 3).mean(axis=0) for im in images])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


def test_index_video_finds_one_track_with_spread_out_crops(moving_box_video, tmp_path):
    crops_dir = tmp_path / "crops" / "rec1"
    options = IndexOptions(sample_fps=4, crops_per_track=3, min_track_detections=3)
    events = list(index_video(moving_box_video, crops_dir, "crops/rec1", RedBoxDetector(), MeanColourEmbedder(), options))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "info" and kinds[-1] == "done" and "progress" in kinds
    tracks = [e for e in events if e["type"] == "track"]
    assert len(tracks) == 1
    track = tracks[0]
    assert track["label"] == "person"
    assert track["t_start"] == 0.25  # a track is confirmed on its second frame
    assert track["t_end"] >= 2.5
    assert len(track["crops"]) == 3
    times = [c["t"] for c in track["crops"]]
    assert times == sorted(times) and times[-1] - times[0] >= 1.0

    first = track["boxes"][0]
    assert len(first) == 5 and all(0.0 <= v <= 1.0 for v in first[1:])
    assert track["boxes"][-1][1] > first[1]  # moved right

    for crop in track["crops"]:
        assert (tmp_path / crop["path"]).is_file()
        assert crop["path"].startswith("crops/rec1/person-")
        vector = np.array(crop["embedding"])
        assert vector.shape == (3,) and abs(np.linalg.norm(vector) - 1) < 1e-4
        assert vector[0] > vector[1]  # crops are mostly red
    assert events[-1]["tracks"] == 1


def test_short_tracks_are_dropped(moving_box_video, tmp_path):
    options = IndexOptions(sample_fps=4, min_track_detections=100)
    events = list(index_video(moving_box_video, tmp_path / "c", "c", RedBoxDetector(), MeanColourEmbedder(), options))
    assert not [e for e in events if e["type"] == "track"]
    assert events[-1]["tracks"] == 0


class CountingDescriber:
    source = "test-describer"

    def __init__(self) -> None:
        self.calls = []

    def describe(self, images, vectors, subject, label):
        self.calls.append((len(images), vectors.shape, subject, label))
        return {"attributes": {"subject": subject, "upper": {"garment": None, "colours": ["red"]}}, "ms": 250}


def test_tracks_carry_their_crop_selection_and_attributes(moving_box_video, tmp_path):
    describer = CountingDescriber()
    options = IndexOptions(sample_fps=4, crops_per_track=3, min_track_detections=3)
    events = list(index_video(moving_box_video, tmp_path / "c", "c", RedBoxDetector(), MeanColourEmbedder(), options,
                              describer=describer))
    [track] = [e for e in events if e["type"] == "track"]
    selection = track["selection"]
    assert selection["pool_capacity"] == 12 and selection["detections"] == len(track["boxes"])
    assert selection["labels"] == {"person": len(track["boxes"])}
    assert sum(c["chosen"] for c in selection["candidates"]) == 3
    assert all({"t", "quality", "confidence", "size", "edge", "reason"} <= set(c) for c in selection["candidates"])
    assert track["attributes"] == {"source": "test-describer", "attributes": {"subject": "person", "upper": {"garment": None, "colours": ["red"]}}, "ms": 250}
    assert describer.calls == [(3, (3, 3), "person", "person")]
    done = events[-1]
    assert done["attributes_source"] == "test-describer" and done["attributes_seconds"] == 0.25
    assert done["detections"] >= len(track["boxes"]) and done["dropped_short_tracks"] == 0


def test_without_a_describer_tracks_have_no_attributes(moving_box_video, tmp_path):
    events = list(index_video(moving_box_video, tmp_path / "c", "c", RedBoxDetector(), MeanColourEmbedder(), IndexOptions(sample_fps=4)))
    [track] = [e for e in events if e["type"] == "track"]
    assert "attributes" not in track and "attributes_source" not in events[-1]
