"""Per-track attributes at index time: clothing colours, garments, gender, hair, headwear, carried objects.

Two ways to fill the shared fields of `avs.fields`, chosen with AVS_ATTRIBUTES:

- `vlm` (default): the vision model describes the track's best crops (see docs/decisions.md for why).
- `siglip`: zero-shot SigLIP 2 prompts. Each field has one prompt per allowed value, such as
  "a photo of a person wearing a grey top"; the track's crop embeddings are compared with them,
  and a value is kept only when the softmax over the field's prompts is confident enough.
  It costs almost nothing at index time, because the crop embeddings already exist.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from PIL import Image

from avs.fields import (
    COLOURS,
    HAIR_COLOURS,
    HEADWEAR_TYPES,
    LOWER_GARMENTS,
    OBJECT_TYPES,
    UPPER_GARMENTS,
    Description,
)
from avs.inference.embedder import Embedder

# How many crops the vision model sees per track: the best one and the next best from another moment.
VLM_CROPS = 2


class Describer(Protocol):
    source: str

    def describe(self, images: list[Image.Image], vectors: np.ndarray, subject: str, label: str) -> dict: ...


def _article(noun: str) -> str:
    if noun in ("trousers", "jeans", "shorts"):
        return noun
    return f"an {noun}" if noun[0] in "aeiou" else f"a {noun}"


def _prompt_sets(subject: str, label: str) -> dict[str, dict[str, str]]:
    if subject == "bag":
        kind = {"backpack": "backpack", "handbag": "handbag", "suitcase": "suitcase"}.get(label, "bag")
        return {"bag_colour": {c: f"a photo of a {c} {kind}" for c in COLOURS}}
    return {
        "gender": {"male": "a photo of a man", "female": "a photo of a woman"},
        "age_band": {"child": "a photo of a child", "adult": "a photo of an adult", "elderly": "a photo of an elderly person"},
        "upper_colour": {c: f"a photo of a person wearing a {c} top" for c in COLOURS},
        "upper_garment": {g: f"a photo of a person wearing {_article(g)}" for g in UPPER_GARMENTS},
        "lower_colour": {c: f"a photo of a person wearing {c} trousers" for c in COLOURS},
        "lower_garment": {g: f"a photo of a person wearing {_article(g)}" for g in LOWER_GARMENTS},
        "hair_colour": {c: f"a photo of a person with {c} hair" for c in HAIR_COLOURS},
        "headwear": {"none": "a photo of a person with no hat"} | {h: f"a photo of a person wearing {_article(h)}" for h in HEADWEAR_TYPES},
        "carried": {"none": "a photo of a person with empty hands"} | {
            o: f"a photo of a person carrying {_article(o)}" for o in OBJECT_TYPES if o != "bag"
        },
    }


# Minimum softmax probability for a value to be kept, per field (fewer options need more confidence).
_MIN_PROBABILITY = {"gender": 0.7, "age_band": 0.6, "headwear": 0.5, "carried": 0.35}
_DEFAULT_MIN_PROBABILITY = 0.3
# A second colour is kept when it is at least this share of the first colour's probability.
_SECOND_COLOUR = 0.6


class SigLIPAttributes:
    source = "siglip2-zero-shot"

    def __init__(self, embedder: Embedder, logit_scale: float) -> None:
        self.embedder = embedder
        self.logit_scale = logit_scale
        self._cache: dict[tuple[str, str], dict[str, tuple[list[str], np.ndarray]]] = {}

    def _prompts(self, subject: str, label: str) -> dict[str, tuple[list[str], np.ndarray]]:
        key = (subject, label if subject == "bag" else "")
        if key not in self._cache:
            sets = {}
            for field, prompts in _prompt_sets(subject, label).items():
                values = list(prompts)
                sets[field] = (values, self.embedder.embed_texts([prompts[v] for v in values]))
            self._cache[key] = sets
        return self._cache[key]

    def probabilities(self, vectors: np.ndarray, subject: str, label: str) -> dict[str, dict[str, float]]:
        out = {}
        for field, (values, text) in self._prompts(subject, label).items():
            logits = self.logit_scale * (vectors @ text.T).mean(axis=0)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            out[field] = {v: round(float(x), 4) for v, x in zip(values, p, strict=True)}
        return out

    def describe(self, images: list[Image.Image], vectors: np.ndarray, subject: str, label: str) -> dict:
        started = time.monotonic()
        probs = self.probabilities(vectors, subject, label)

        def top(field: str) -> str | None:
            value, p = max(probs[field].items(), key=lambda kv: kv[1])
            return value if p >= _MIN_PROBABILITY.get(field, _DEFAULT_MIN_PROBABILITY) and value != "none" else None

        def colours(field: str) -> list[str]:
            ranked = sorted(probs[field].items(), key=lambda kv: -kv[1])
            if ranked[0][1] < _DEFAULT_MIN_PROBABILITY:
                return []
            keep = [ranked[0][0]]
            if ranked[1][1] >= _SECOND_COLOUR * ranked[0][1]:
                keep.append(ranked[1][0])
            return keep

        if subject == "bag":
            description = Description(subject="bag", carried=[{"object": label if label in OBJECT_TYPES else "bag",
                                                                "colours": colours("bag_colour")}])
        else:
            carried = top("carried")
            headwear = top("headwear")
            description = Description(
                subject="person",
                gender=top("gender"),
                age_band=top("age_band"),
                upper={"garment": top("upper_garment"), "colours": colours("upper_colour")},
                lower={"garment": top("lower_garment"), "colours": colours("lower_colour")},
                hair={"colour": top("hair_colour")},
                headwear={"type": headwear},
                carried=[{"object": carried}] if carried else [],
            )
        return {"attributes": description.model_dump(), "probabilities": probs,
                "ms": round((time.monotonic() - started) * 1000)}


class VLMAttributes:
    def __init__(self, vlm) -> None:
        self.vlm = vlm
        self.source = f"vlm:{vlm.name}"

    def describe(self, images: list[Image.Image], vectors: np.ndarray, subject: str, label: str) -> dict:
        return self.vlm.describe(images[:VLM_CROPS], subject, bag_label=label if subject == "bag" else None)

