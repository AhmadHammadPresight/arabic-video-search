"""Person and bag detection with RF-DETR (Apache-2.0 sizes only: nano, small, medium, large)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# COCO category ids used by RF-DETR's COCO checkpoints.
COCO_LABELS = {1: "person", 27: "backpack", 31: "handbag", 33: "suitcase"}
BAG_LABELS = frozenset({"backpack", "handbag", "suitcase"})


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    label: str
    confidence: float


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


# RF-DETR XL and 2XL are under the PML licence (rfdetr_plus) and must not be added here.
_VARIANTS = {
    "rfdetr-nano": ("RFDETRNano", "rf-detr-nano.pth"),
    "rfdetr-small": ("RFDETRSmall", "rf-detr-small.pth"),
    "rfdetr-medium": ("RFDETRMedium", "rf-detr-medium.pth"),
    "rfdetr-large": ("RFDETRLarge", "rf-detr-large.pth"),
}


class RFDETRDetector:
    def __init__(self, variant: str, weights_dir: Path, device: str, threshold: float) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"unknown or non-Apache detector {variant!r}; choose one of {sorted(_VARIANTS)}")
        import rfdetr

        class_name, weights = _VARIANTS[variant]
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.name = variant
        self.threshold = threshold
        self._model = getattr(rfdetr, class_name)(pretrain_weights=str(weights_dir / weights), device=device)
        self._lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with self._lock:
            result = self._model.predict(frame, threshold=self.threshold, include_source_image=False)
        out = []
        for box, class_id, conf in zip(result.xyxy, result.class_id, result.confidence, strict=True):
            label = COCO_LABELS.get(int(class_id))
            if label:
                out.append(Detection(tuple(float(v) for v in box), label, float(conf)))
        return out
