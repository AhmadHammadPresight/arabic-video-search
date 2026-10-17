"""Search, option A baseline: the query text goes straight to the SigLIP 2 text encoder.

There is no query parsing, translation or attribute filtering yet; those are phase 2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from avs.api.times import from_epoch, parse_local_time, to_epoch
from avs.api.vectors import Hit, VectorStore

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
) -> list[dict]:
    abs_from, abs_to = query.window()
    hits = store.search(embed_text(query.text), query.limit, query.cameras, abs_from, abs_to)
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
