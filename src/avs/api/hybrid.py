"""Phase 2 search: query fields, filters, SigLIP 2 similarity and attribute matching, fused.

1. The query model turns the text into fields and an English rewrite (`avs.api.query_parser`).
2. SigLIP 2 embeds both the English rewrite and the original text.
3. Qdrant filters by camera and time and by the subject (people, or the bag classes). Gender is
   matched softly with the other attributes rather than filtered, because the vision model gets it
   wrong for some tracks (see docs/decisions.md).
4. The most similar tracks for either text form the candidate pool; every crop of every candidate
   is scored against both texts. A track's similarity blends its best score for the English rewrite
   with its best score for the original text, which may come from different crops; the crop shown
   and checked first is the one with the best blended score.
5. Each candidate's attributes are matched field by field (`avs.api.matching`), and the attribute
   score is fused with the similarity. Results show why they matched, per field.

The vision-model double-check of the top results runs afterwards (`avs.api.checker`).
When the query model is unavailable the search falls back to option A and says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pydantic import ValidationError

from avs.api.matching import (
    ATTRIBUTE_WEIGHT,
    ENGLISH_SHARE,
    VALUES,
    WEIGHTS,
    FieldMatch,
    attribute_score,
    blend_similarity,
    match_fields,
    standardise,
)
from avs.api.query_parser import ParseError, ParseResult
from avs.api.search import SearchQuery, run_search, shape_result
from avs.api.trace import Trace, vector_preview
from avs.api.vectors import Hit, VectorStore, build_filter
from avs.fields import Description, QueryFields, TrackDescription

PERSON_LABELS = ["person"]
BAG_LABELS = ["backpack", "handbag", "suitcase"]
# Tracks fetched per query text before fusion: at least this many, or a few times the limit.
POOL_MIN = 60
POOL_FACTOR = 4
# Drop tracks whose attributes give the other gender. Off: on the phase 1 queries it never helped
# and lowered precision, because 8% of the target tracks were described with the wrong gender.
GENDER_FILTER = False
# Crops per result the double-check looks at.
CHECK_CROPS = 2
# How many candidate rows the trace keeps for the candidates, attributes and fusion stages.
TRACE_ROWS = 40


@dataclass
class Candidate:
    track_key: str
    crops: list[tuple[float, float, dict]] = field(default_factory=list)  # (sim_english, sim_original, payload)
    attributes: Description | None = None
    matches: list[FieldMatch] = field(default_factory=list)
    attr_score: float = 0.0
    score: float = 0.0

    @property
    def sim_english(self) -> float:
        return max(c[0] for c in self.crops)

    @property
    def sim_original(self) -> float:
        return max(c[1] for c in self.crops)

    @property
    def similarity(self) -> float:
        return blend_similarity(self.sim_english, self.sim_original)

    def best_crops(self) -> list[dict]:
        """Payloads of the track's crops, the best match to the blended query first."""
        return [c[2] for c in sorted(self.crops, key=lambda c: -blend_similarity(c[0], c[1]))]


@dataclass
class HybridOutcome:
    results: list[dict]
    mode: str  # "hybrid", or "baseline" when the query model was unavailable
    parsed: ParseResult | None = None
    check_crops: dict[str, list[str]] = field(default_factory=dict)


def _script(text: str) -> str:
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic and latin:
        return "mixed Arabic and Latin"
    return "Arabic" if arabic else "Latin" if latin else "other"


def why(matches: list[FieldMatch]) -> list[dict]:
    return [{"field": m.field, "label": m.label, "result": m.result} for m in matches]


def run_hybrid(
    store: VectorStore,
    embed_texts: Callable[[list[str]], list[list[float]]],
    parse: Callable[[str], ParseResult],
    recordings: dict[str, dict],
    query: SearchQuery,
    trace: Trace,
    embed_model: str = "",
) -> HybridOutcome:
    abs_from, abs_to = query.window()  # an invalid window fails before any model call
    with trace.stage("input") as stage:
        stage["output"] = {"text": query.text, "characters": len(query.text), "script": _script(query.text)}

    with trace.stage("parse") as stage:
        try:
            parsed = parse(query.text)
        except ParseError as exc:
            stage["output"] = {"error": str(exc), "raw": exc.raw, "fallback": "option A (the text goes straight to SigLIP 2)"}
            parsed = None
        else:
            stage["output"] = {
                "fields": parsed.fields.model_dump(exclude={"english"}),
                "english": parsed.fields.english,
                "raw": parsed.raw,
                "model": parsed.model,
                "cached": parsed.cached,
                "prompt_tokens": parsed.prompt_tokens,
                "output_tokens": parsed.output_tokens,
                "load_ms": parsed.load_ms,
                "model_ms": parsed.took_ms,
            }
    if parsed is None:
        trace.note("The query model was unavailable, so this search ran as option A.")
        trace.data["mode"] = "baseline"
        return HybridOutcome(run_search(store, lambda t: embed_texts([t])[0], recordings, query, trace=trace), "baseline")

    fields: QueryFields = parsed.fields
    with trace.stage("embed") as stage:
        texts = [fields.english, query.text]
        vec_english, vec_original = embed_texts(texts)
        stage["output"] = {
            "model": embed_model, "padding": "max_length 64 tokens, Latin text lower-cased",
            "texts": [{"role": role, "text": text, **vector_preview(vector)}
                      for role, text, vector in zip(("English rewrite", "original text"), texts, (vec_english, vec_original), strict=True)],
        }

    with trace.stage("filter") as stage:
        labels = BAG_LABELS if fields.subject == "bag" else PERSON_LABELS
        exclude = None
        if GENDER_FILTER and fields.subject == "person" and fields.gender:
            exclude = "male" if fields.gender == "female" else "female"
        query_filter = build_filter(query.cameras, abs_from, abs_to, labels, exclude)
        stage["output"] = {
            "cameras": query.cameras or "all", "time_from": query.time_from, "time_to": query.time_to,
            "labels": labels, "exclude_gender": exclude,
            "explain": [
                f"subject is {fields.subject}: keep detector labels {', '.join(labels)}",
                f"drop tracks described as {exclude} (tracks with no attributes stay)" if exclude else "no attribute filter",
            ],
            "qdrant_filter": query_filter.model_dump(mode="json", exclude_none=True) if query_filter else None,
        }

    with trace.stage("candidates") as stage:
        pool = max(POOL_MIN, POOL_FACTOR * query.limit)
        by_english = store.search_tracks(vec_english, pool, query_filter)
        by_original = store.search_tracks(vec_original, pool, query_filter)
        keys = list(dict.fromkeys([h.payload["track_key"] for h in by_english + by_original]))
        candidates = _score_crops(store.track_crops(keys), vec_english, vec_original)
        candidates = [c for c in candidates if c.best_crops()[0].get("recording_id") in recordings]
        rows = sorted(candidates, key=lambda c: -c.similarity)
        stage["output"] = {
            "pool_per_text": pool, "found_by_english": len(by_english), "found_by_original": len(by_original),
            "tracks": len(candidates), "crops_scored": sum(len(c.crops) for c in candidates),
            "rows": [_candidate_row(c) for c in rows[:TRACE_ROWS]],
        }

    with trace.stage("attributes") as stage:
        for c in candidates:
            c.matches = match_fields(fields, c.attributes)
            c.attr_score = attribute_score(c.matches)
        stage["output"] = {
            "weights": WEIGHTS, "values": VALUES,
            "query_fields": [m.label for m in match_fields(fields, Description())],
            "described_tracks": sum(1 for c in candidates if c.attributes is not None),
            "rows": [{"track_key": c.track_key, "attribute_score": c.attr_score,
                      "matches": [m.as_dict() for m in c.matches]} for c in rows[:TRACE_ROWS]],
        }

    with trace.stage("fusion") as stage:
        sims = [c.similarity for c in candidates]
        zs = standardise(sims)
        for c, z in zip(candidates, zs, strict=True):
            c.score = round(z + ATTRIBUTE_WEIGHT * c.attr_score, 4)
        ranked = sorted(candidates, key=lambda c: -c.score)
        stage["output"] = {
            "formula": f"score = z(similarity) + {ATTRIBUTE_WEIGHT} x attribute score; "
                       f"similarity = {ENGLISH_SHARE} x cos(crop, English) + {1 - ENGLISH_SHARE} x cos(crop, original)",
            "attribute_weight": ATTRIBUTE_WEIGHT, "english_share": ENGLISH_SHARE,
            "similarity_mean": round(float(np.mean(sims)), 4) if sims else None,
            "similarity_std": round(float(np.std(sims)), 4) if sims else None,
            "rows": [{"track_key": c.track_key, "similarity": round(c.similarity, 4), "z": round(z, 3),
                      "attribute_score": c.attr_score, "score": c.score}
                     for c, z in sorted(zip(candidates, zs, strict=True), key=lambda cz: -cz[0].score)[:TRACE_ROWS]],
        }

    results, check_crops = [], {}
    for c in ranked[: query.limit]:
        payloads = c.best_crops()
        recording = recordings[payloads[0]["recording_id"]]
        result = shape_result(Hit(score=c.score, payload=payloads[0]), recording, rank=len(results) + 1)
        result |= {
            "similarity": round(c.similarity, 4),
            "sim_english": round(c.sim_english, 4),
            "sim_original": round(c.sim_original, 4),
            "attribute_score": c.attr_score,
            "why": why(c.matches),
            "described": c.attributes is not None,
        }
        results.append(result)
        check_crops[c.track_key] = [p["crop_path"] for p in payloads[:CHECK_CROPS]]

    with trace.stage("ranking") as stage:
        stage["output"] = [{"rank": r["rank"], "track_key": r["track_key"], "score": r["score"],
                            "similarity": r["similarity"], "attribute_score": r["attribute_score"],
                            "crop_url": r["crop_url"], "why": r["why"]} for r in results]
    return HybridOutcome(results, "hybrid", parsed, check_crops)


def _score_crops(points: list[tuple[dict, list[float]]], vec_english: list[float], vec_original: list[float]) -> list[Candidate]:
    if not points:
        return []
    matrix = np.asarray([v for _, v in points], dtype=np.float32)
    sims = matrix @ np.asarray([vec_english, vec_original], dtype=np.float32).T
    by_track: dict[str, Candidate] = {}
    for (payload, _), (s_en, s_orig) in zip(points, sims, strict=True):
        cand = by_track.setdefault(payload["track_key"], Candidate(payload["track_key"]))
        cand.crops.append((float(s_en), float(s_orig), payload))
        if cand.attributes is None:
            cand.attributes = read_attributes(payload.get("attributes"))
    return list(by_track.values())


# The parts of a description that attribute matching compares.
_MATCHED_PARTS = {"gender", "age_band", "upper", "lower", "hair", "headwear", "carried"}


def read_attributes(stored: dict | None) -> TrackDescription | None:
    """A track's stored attributes, or None when there is nothing to compare.

    None covers a track that was never described, a description with none of the matched fields
    filled in (such as a vision-model reply that could not be read, stored as an empty description
    by earlier builds), and values outside the current vocabulary (indexed before a schema change).
    Every field of such a track is then "unknown" rather than "missing" or "mismatch".
    """
    if not stored:
        return None
    try:
        description = TrackDescription.model_validate(stored)
    except ValidationError:
        return None
    return description if description.model_dump(exclude_defaults=True, include=_MATCHED_PARTS) else None


def _candidate_row(c: Candidate) -> dict:
    best = c.best_crops()[0]
    return {
        "track_key": c.track_key, "label": best.get("label"), "camera_id": best.get("camera_id"),
        "crop_url": f"/media/{best['crop_path']}", "crops": len(c.crops),
        "sim_english": round(c.sim_english, 4), "sim_original": round(c.sim_original, 4),
        "similarity": round(c.similarity, 4),
    }
