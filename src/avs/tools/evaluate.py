"""Run the query file against the live index and report top-10 hit rate and precision at 5.

    python -m avs.tools.evaluate [eval/queries.yaml] [--api URL] [--mode baseline|hybrid|both] [--out data/eval]

Each query runs in option A (`baseline`) and in the phase 2 hybrid; for the hybrid, the first
ranking and the ranking after the vision-model double-check are scored separately, so the
report shows option A, hybrid, and hybrid plus check side by side on the same index.

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
import time
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
# Scored rankings: option A, the hybrid's first ranking, and the hybrid after the double-check.
RANKINGS = {"baseline": "Option A", "hybrid": "Hybrid", "checked": "Hybrid + check"}
CHECK_TIMEOUT_S = 300


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


def run_query(client: httpx.Client, q: dict, mode: str) -> dict[str, dict]:
    """The scored rankings of one query in one mode, with their timings."""
    body = {"query": q["text"], "limit": 10, "actor": "eval", "mode": mode, **(q.get("filter") or {})}
    found = client.post("/api/search", json=body).raise_for_status().json()
    first = {"top": [r["track_key"] for r in found["results"]],
             "top_score": found["results"][0]["score"] if found["results"] else None,
             "took_ms": found["took_ms"], "ran_as": found.get("mode", mode)}
    if mode == "baseline":
        return {"baseline": first}
    out = {"hybrid": first | {"english": (found.get("parsed") or {}).get("english")}}
    if "check" not in found:  # the hybrid fell back to option A; nothing was checked
        return out | {"checked": dict(first)}
    started, deadline = time.monotonic(), time.monotonic() + CHECK_TIMEOUT_S
    while True:
        status = client.get(f"/api/search/{found['search_id']}/check").raise_for_status().json()
        if status["status"] in ("done", "skipped", "failed", "cancelled", "interrupted") or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    results = status["results"] if status["status"] == "done" else found["results"]
    verdicts = {k: v["verdict"] for k, v in status.get("verdicts", {}).items()}
    out["checked"] = {"top": [r["track_key"] for r in results], "check_status": status["status"],
                      "check_ms": round((time.monotonic() - started) * 1000), "verdicts": verdicts}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", nargs="?", default=str(REPO_ROOT / "eval" / "queries.yaml"))
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    parser.add_argument("--mode", choices=["baseline", "hybrid", "both"], default="both")
    parser.add_argument("--only", nargs="*", help="run only these query ids")
    parser.add_argument("--out", default=str(get_settings().data_dir / "eval"))
    args = parser.parse_args(argv)
    modes = ["baseline", "hybrid"] if args.mode == "both" else [args.mode]

    targets, queries = load_queries(Path(args.queries))
    if args.only:
        queries = [q for q in queries if q["id"] in args.only]
    client = httpx.Client(base_url=args.api, timeout=120.0)
    recordings = [r for r in client.get("/api/recordings").raise_for_status().json() if r["status"] == "indexed"]
    tracks = {r["id"]: client.get(f"/api/recordings/{r['id']}/tracks").raise_for_status().json() for r in recordings}

    resolved: dict[str, set[str]] = {}
    unresolved: list[str] = []
    for name, anchors in targets.items():
        matched = [match_anchor(a, recordings, tracks) for a in anchors]
        resolved[name] = {k for k in matched if k}
        if misses := matched.count(None):
            unresolved.append(f"{name}: {misses} of {len(anchors)} anchor(s) matched no track")

    rows: dict[str, list[dict]] = {k: [] for k in RANKINGS}
    for q in queries:
        relevant = set().union(*(resolved[t] for t in q.get("targets", []))) if q.get("targets") else set()
        base = {
            "id": q["id"], "text": q["text"], "register": q.get("register", "formal"), "story": q.get("story", "suspect"),
            "language": "english" if q.get("register") == "english" else "arabic",
            "set": "added in phase 2" if q.get("added") == "phase2" else "phase 1 set",
            "expect": q.get("expect", "match"), "relevant_tracks": len(relevant),
        }
        for mode in modes:
            for ranking, result in run_query(client, q, mode).items():
                row = base | result
                row |= score_query(row["top"], relevant)
                rows[ranking].append(row)
        print(f"  {q['id']:5} " + "  ".join(
            f"{RANKINGS[k]}: {rows[k][-1]['first_relevant_rank'] or '-'}" for k in RANKINGS if rows[k] and rows[k][-1]["id"] == q["id"]
        ) + f"  {q['text']}", flush=True)

    scored = {k: v for k, v in rows.items() if v}
    report = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "queries_file": str(args.queries),
        "rankings": list(scored),
        "summary": {k: summarise(v, "register") for k, v in scored.items()},
        "by_language": {k: summarise(v, "language") for k, v in scored.items()},
        "by_story": {k: summarise(v, "story") for k, v in scored.items()},
        "by_set": {k: summarise(v, "set") for k, v in scored.items()},
        "timing_ms": timing(scored),
        "unresolved_targets": unresolved,
        "rows": scored,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"report-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + side_by_side(report))
    print(f"\ntiming (ms): {report['timing_ms']}")
    if unresolved:
        print("\nAnchors that matched no track (check the ground truth):", *unresolved, sep="\n  ")
    print(f"\nreport: {path}")
    return 0


def timing(scored: dict[str, list[dict]]) -> dict:
    def stats(values: list[float]) -> dict:
        values = sorted(values)
        return {"median": values[len(values) // 2], "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1]} if values else {}

    out = {k: stats([r["took_ms"] for r in v]) for k, v in scored.items() if k != "checked"}
    if "checked" in scored:
        out["double_check"] = stats([r["check_ms"] for r in scored["checked"] if r.get("check_ms") is not None])
    return out


def side_by_side(report: dict) -> str:
    """Markdown table: one row per group, three numbers per ranking."""
    names = report["rankings"]
    head = "| Group | Queries | " + " | ".join(f"{RANKINGS[n]} hit@10 | P@5 | MRR" for n in names) + " |"
    lines = [head, "|---|---|" + "---|---|---|" * len(names)]
    first = names[0]
    groups = [(k, k) for k in report["summary"][first]]
    groups += [("arabic", "Arabic, all registers")] if "arabic" in report["by_language"][first] else []
    groups += [(f"story:{k}", f"{k} story") for k in report["by_story"][first] if k != "all"]
    groups += [(f"set:{k}", k) for k in report["by_set"][first] if k != "all"]

    def lookup(name: str, key: str) -> dict:
        if key.startswith("story:"):
            return report["by_story"][name][key[6:]]
        if key.startswith("set:"):
            return report["by_set"][name][key[4:]]
        if key == "arabic":
            return report["by_language"][name]["arabic"]
        return report["summary"][name][key]

    for key, label in groups:
        cells = [lookup(n, key) for n in names]
        lines.append(f"| {label} | {cells[0]['queries']} | " + " | ".join(
            f"{c['hit_at_10']:.2f} | {c['precision_at_5']:.2f} | {c['mrr']:.2f}" for c in cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
