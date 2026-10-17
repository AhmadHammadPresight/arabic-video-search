"""Run the query file against the live index and report top-10 hit rate and precision at 5.

    python -m avs.tools.evaluate [eval/queries.yaml] [--api URL] [--out data/eval]

Ground truth is written as anchors, not track ids, so it survives re-indexing with other
settings: an anchor is a point on the target (camera, local time, x and y as fractions of
the frame). A result track is correct when, within half a second of an anchor's time, one of
its boxes contains the anchor point; among several such tracks the one centred closest wins.

Queries go through the API like any other search (actor "eval"), with each query's camera and
time filter, so they are audit-logged.
Queries marked `expect: no-match` describe nobody in the footage; they are listed with their
top score but left out of the averages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from avs.api.times import parse_local_time, to_epoch
from avs.settings import REPO_ROOT, get_settings

ANCHOR_TOLERANCE_S = 0.5
POINT_MARGIN = 0.01


@dataclass(frozen=True)
class Anchor:
    camera: str
    at: float  # epoch seconds of the camera's local time
    x: float
    y: float


def load_queries(path: Path) -> tuple[dict[str, list[Anchor]], list[dict]]:
    data = yaml.safe_load(path.read_text())
    targets = {
        name: [Anchor(a["camera"], to_epoch(parse_local_time(str(a["at"]))), float(a["x"]), float(a["y"])) for a in t["anchors"]]
        for name, t in (data.get("targets") or {}).items()
    }
    queries = data["queries"]
    for q in queries:
        missing = [t for t in q.get("targets", []) if t not in targets]
        if missing:
            raise ValueError(f"query {q['id']} names unknown targets {missing}")
        if not q.get("targets") and q.get("expect") != "no-match":
            raise ValueError(f"query {q['id']} has no targets; mark it `expect: no-match` if nobody matches")
    return targets, queries


def match_anchor(anchor: Anchor, recordings: list[dict], tracks_by_recording: dict[str, list[dict]]) -> str | None:
    """Track key of the track the anchor points at, or None."""
    best: tuple[float, str] | None = None
    for rec in recordings:
        if rec["camera_id"] != anchor.camera:
            continue
        t_rel = anchor.at - to_epoch(parse_local_time(rec["start_time"]))
        if not -ANCHOR_TOLERANCE_S <= t_rel <= (rec["duration_s"] or 0) + ANCHOR_TOLERANCE_S:
            continue
        for track in tracks_by_recording.get(rec["id"], []):
            for t, x1, y1, x2, y2 in track["boxes"]:
                if abs(t - t_rel) > ANCHOR_TOLERANCE_S:
                    continue
                if x1 - POINT_MARGIN <= anchor.x <= x2 + POINT_MARGIN and y1 - POINT_MARGIN <= anchor.y <= y2 + POINT_MARGIN:
                    distance = ((x1 + x2) / 2 - anchor.x) ** 2 + ((y1 + y2) / 2 - anchor.y) ** 2
                    if best is None or distance < best[0]:
                        best = (distance, track["track_key"])
    return best[1] if best else None


def score_query(result_keys: list[str], relevant: set[str]) -> dict:
    hits = [key in relevant for key in result_keys]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "hit_at_10": float(any(hits[:10])),
        "precision_at_5": sum(hits[:5]) / 5,
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "first_relevant_rank": first,
    }


def summarise(rows: list[dict], key: str | None = None) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["expect"] == "no-match":
            continue
        groups["all"].append(row)
        if key:
            groups[row[key]].append(row)
    return {
        name: {
            "queries": len(items),
            "hit_at_10": round(sum(r["hit_at_10"] for r in items) / len(items), 3),
            "precision_at_5": round(sum(r["precision_at_5"] for r in items) / len(items), 3),
            "mrr": round(sum(r["reciprocal_rank"] for r in items) / len(items), 3),
        }
        for name, items in groups.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", nargs="?", default=str(REPO_ROOT / "eval" / "queries.yaml"))
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    parser.add_argument("--out", default=str(get_settings().data_dir / "eval"))
    args = parser.parse_args(argv)

    targets, queries = load_queries(Path(args.queries))
    client = httpx.Client(base_url=args.api, timeout=60.0)
    recordings = [r for r in client.get("/api/recordings").raise_for_status().json() if r["status"] == "indexed"]
    tracks = {r["id"]: client.get(f"/api/recordings/{r['id']}/tracks").raise_for_status().json() for r in recordings}

    resolved: dict[str, set[str]] = {}
    unresolved: list[str] = []
    for name, anchors in targets.items():
        matched = [match_anchor(a, recordings, tracks) for a in anchors]
        resolved[name] = {k for k in matched if k}
        if misses := matched.count(None):
            unresolved.append(f"{name}: {misses} of {len(anchors)} anchor(s) matched no track")

    rows = []
    for q in queries:
        body = {"query": q["text"], "limit": 10, "actor": "eval", **(q.get("filter") or {})}
        results = client.post("/api/search", json=body).raise_for_status().json()["results"]
        relevant = set().union(*(resolved[t] for t in q.get("targets", []))) if q.get("targets") else set()
        row = {
            "id": q["id"], "text": q["text"], "register": q.get("register", "formal"), "story": q.get("story", "suspect"),
            "language": "english" if q.get("register") == "english" else "arabic",
            "expect": q.get("expect", "match"), "relevant_tracks": len(relevant),
            "top_score": results[0]["score"] if results else None,
            "top": [r["track_key"] for r in results],
        }
        row |= score_query(row["top"], relevant)
        rows.append(row)

    report = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "queries_file": str(args.queries),
        "summary": summarise(rows, "register"),
        "by_language": summarise(rows, "language"),
        "by_story": summarise(rows, "story"),
        "unresolved_targets": unresolved,
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"report-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"{'id':5} {'register':10} {'hit@10':>6} {'P@5':>5} {'rank':>4} {'top':>6}  query")
    for r in rows:
        rank = r["first_relevant_rank"] or "-"
        mark = "  (no-match query)" if r["expect"] == "no-match" else ""
        print(f"{r['id']:5} {r['register']:10} {r['hit_at_10']:6.0f} {r['precision_at_5']:5.2f} {rank:>4} {r['top_score'] or 0:6.3f}  {r['text']}{mark}")
    print("\n| Group | Queries | Top-10 hit rate | Precision at 5 | MRR |\n|---|---|---|---|---|")
    groups = list(report["summary"].items())
    groups += [(f"{k} (all registers)", v) for k, v in report["by_language"].items() if k == "arabic"]
    groups += [(f"story: {k}", v) for k, v in report["by_story"].items() if k != "all"]
    for name, s in groups:
        print(f"| {name} | {s['queries']} | {s['hit_at_10']:.2f} | {s['precision_at_5']:.2f} | {s['mrr']:.2f} |")
    if unresolved:
        print("\nAnchors that matched no track (check the ground truth):", *unresolved, sep="\n  ")
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
