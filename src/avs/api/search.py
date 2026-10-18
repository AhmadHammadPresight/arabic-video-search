"""Search, option A baseline: the query text goes straight to the SigLIP 2 text encoder.

There is no query parsing, translation or attribute filtering in this mode; the phase 2 hybrid
is in `avs.api.hybrid`, and option A stays selectable so the two can be compared on one index.
With a trace, each stage's output and timing is recorded; the results are the same either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from avs.api.times import from_epoch, parse_local_time, to_epoch
from avs.api.trace import Trace, vector_preview
from avs.api.vectors import Hit, VectorStore, build_filter

# Longest clip the results grid plays for one track.
MAX_CLIP_SECONDS = 30.0
CLIP_PAD_SECONDS = 1.0


@dataclass(frozen=True)
class SearchQuery:
    text: str
    cameras: list[str] | None = None
    time_from: str | None = None
    time_to: str | None = None
    limit: int = 24

    def window(self) -> tuple[float | None, float | None]:
        start = to_epoch(parse_local_time(self.time_from)) if self.time_from else None
        end = to_epoch(parse_local_time(self.time_to)) if self.time_to else None
        if start is not None and end is not None and end < start:
            raise ValueError("time_to is before time_from")
        return start, end


def run_search(
    store: VectorStore,
    embed_text: Callable[[str], list[float]],
    recordings: dict[str, dict],
    query: SearchQuery,
    trace: Trace | None = None,
    embed_model: str = "",
) -> list[dict]:
    if trace is None:
        abs_from, abs_to = query.window()
        hits = store.search(embed_text(query.text), query.limit, query.cameras, abs_from, abs_to)
        return _results(hits, recordings)

    abs_from, abs_to = query.window()  # an invalid window fails before any model call, as without a trace
    if not any(s["key"] == "input" for s in trace.data["stages"]):
        with trace.stage("input") as stage:
            stage["output"] = {"text": query.text, "characters": len(query.text)}
    with trace.stage("embed") as stage:
        vector = embed_text(query.text)
        stage["output"] = {"model": embed_model, "padding": "max_length 64 tokens, Latin text lower-cased",
                           "texts": [{"role": "original text", "text": query.text, **vector_preview(vector)}]}
    with trace.stage("filter") as stage:
        query_filter = build_filter(query.cameras, abs_from, abs_to)
        stage["output"] = {"cameras": query.cameras or "all", "time_from": query.time_from, "time_to": query.time_to,
                           "labels": "all", "explain": ["camera and time only; option A has no attribute filters"],
                           "qdrant_filter": query_filter.model_dump(mode="json", exclude_none=True) if query_filter else None}
    with trace.stage("candidates") as stage:
        hits = store.search(vector, query.limit, query.cameras, abs_from, abs_to)
        stage["output"] = {"pool_per_text": query.limit, "tracks": len(hits), "rows": [
            {"track_key": h.payload["track_key"], "label": h.payload.get("label"), "camera_id": h.payload.get("camera_id"),
             "crop_url": f"/media/{h.payload['crop_path']}", "similarity": round(h.score, 4)} for h in hits]}
    results = _results(hits, recordings)
    with trace.stage("ranking") as stage:
        stage["output"] = [{"rank": r["rank"], "track_key": r["track_key"], "score": r["score"],
                            "similarity": r["score"], "crop_url": r["crop_url"]} for r in results]
    return results


def _results(hits: list[Hit], recordings: dict[str, dict]) -> list[dict]:
    results = []
    for hit in hits:
        recording = recordings.get(hit.payload.get("recording_id", ""))
        if recording is None:  # recording deleted while its points were still being removed
            continue
        results.append(shape_result(hit, recording, rank=len(results) + 1))
    return results


def clip_window(t: float, t_start: float, t_end: float, duration: float | None) -> tuple[float, float]:
    """The stretch of video to play for a track: the whole track, padded, capped around the best crop."""
    start = max(0.0, t_start - CLIP_PAD_SECONDS)
    end = t_end + CLIP_PAD_SECONDS
    if duration:
        end = min(end, duration)
    if end - start > MAX_CLIP_SECONDS:
        start = max(start, t - MAX_CLIP_SECONDS / 2)
        end = min(end, start + MAX_CLIP_SECONDS)
    return round(start, 2), round(end, 2)


def shape_result(hit: Hit, recording: dict, rank: int) -> dict:
    p = hit.payload
    clip_start, clip_end = clip_window(p["t"], p["track_t_start"], p["track_t_end"], recording.get("duration_s"))
    return {
        "rank": rank,
        "track_key": p["track_key"],
        "label": p["label"],
        "score": round(hit.score, 4),
        "camera_id": p["camera_id"],
        "place": p["place"],
        "recording_id": p["recording_id"],
        "time": from_epoch(p["abs_t"]),
        "seen_from": from_epoch(p["abs_start"]),
        "seen_to": from_epoch(p["abs_end"]),
        "t": p["t"],
        "clip_start": clip_start,
        "clip_end": clip_end,
        "box": p["box"],
        "crop_url": f"/media/{p['crop_path']}",
        "video_url": f"/media/{recording['playback_path']}" if recording.get("playback_path") else None,
    }
