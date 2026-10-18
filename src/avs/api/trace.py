"""A record of every stage of one search, with each stage's real output and time taken.

Every search stores its trace (table `search_traces`), so a past search can be reopened from
the audit log; `POST /api/search?trace=1` also returns it, and the "under the hood" page shows it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

# Stage titles in Arabic and English, in the order a search runs them.
STAGES = {
    "input": ("النص المدخل", "Raw text"),
    "parse": ("فهم الاستعلام بالنموذج اللغوي", "Query model: fields and English rewrite"),
    "embed": ("تمثيل النص بـ SigLIP 2", "SigLIP 2 text embedding"),
    "filter": ("التصفية قبل الترتيب", "Filters applied before ranking"),
    "candidates": ("المرشحون من Qdrant", "Candidate tracks from Qdrant"),
    "attributes": ("مطابقة السمات", "Attribute match per field"),
    "fusion": ("دمج الدرجات", "Score fusion"),
    "ranking": ("الترتيب الأول", "First ranking (shown at once)"),
    "check": ("التحقق البصري", "Vision-model double-check"),
    "final": ("الترتيب النهائي", "Final ranking"),
}


def vector_preview(vector: list[float], n: int = 8) -> dict:
    norm = sum(v * v for v in vector) ** 0.5
    return {"dim": len(vector), "norm": round(norm, 4), "first": [round(v, 4) for v in vector[:n]]}


class Trace:
    def __init__(self, search_id: str, mode: str, request: dict) -> None:
        self._started = time.monotonic()
        self.data: dict = {"search_id": search_id, "mode": mode, "request": request, "models": {},
                           "notes": [], "stages": []}

    @contextmanager
    def stage(self, key: str) -> Iterator[dict]:
        ar, en = STAGES[key]
        stage: dict = {"key": key, "title_ar": ar, "title_en": en}
        started = time.monotonic()
        try:
            yield stage
        finally:
            stage["ms"] = round((time.monotonic() - started) * 1000, 1)
            self.data["stages"].append(stage)

    def note(self, text: str) -> None:
        self.data["notes"].append(text)

    def finish(self) -> dict:
        self.data["took_ms"] = round((time.monotonic() - self._started) * 1000)
        return self.data


def replace_stage(trace: dict, stage: dict) -> dict:
    """Put a stage into a stored trace: in place of an earlier stage with the same key, or at the end."""
    keys = [s["key"] for s in trace["stages"]]
    if stage["key"] in keys:
        trace["stages"][keys.index(stage["key"])] = stage
    else:
        trace["stages"].append(stage)
    return trace
