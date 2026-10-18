"""Measure the two ways of attaching attributes to tracks against labels marked by eye.

    python -m avs.tools.attribute_benchmark [eval/attributes.yaml] [--api URL] [--inference URL] [--out DIR]

For every labelled track (resolved through its anchor, as in the evaluation), both methods
describe the same crops through the inference service: SigLIP 2 zero-shot prompts and the
Qwen3-VL description. Each field is scored against the label: the gender, the main colour of
the upper and lower garments, the hair colour, whether there is headwear, whether the person
carries a bag, and a bag's main colour. A method that gives no value for a labelled field is
counted wrong, and its coverage is reported alongside. Time per track is measured too, and the
time for the whole starter index is projected from it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from avs.api.inference_client import InferenceClient
from avs.api.times import parse_local_time, to_epoch
from avs.fields import BAG_OBJECTS, Description
from avs.settings import REPO_ROOT, get_settings
from avs.tools.evaluate import Anchor, match_anchor

METHODS = ("siglip", "vlm")
FIELDS = ("gender", "upper", "lower", "hair", "headwear", "bag", "bag_colour")


def predicted(attrs: dict, field: str):
    if field == "gender":
        return attrs.get("gender")
    if field in ("upper", "lower"):
        colours = attrs[field]["colours"]
        return colours[0] if colours else None
    if field == "hair":
        return attrs["hair"]["colour"]
    if field == "headwear":
        return attrs["headwear"]["type"] is not None
    if field == "bag":
        return any(c["object"] in BAG_OBJECTS for c in attrs["carried"])
    if field == "bag_colour":
        colours = attrs["carried"][0]["colours"] if attrs["carried"] else []
        return colours[0] if colours else None
    raise ValueError(field)


def expected(label: dict, field: str):
    """The label's answer for a field (a set of right values, or a bool), or None when not labelled."""
    if field == "bag_colour":
        return set(label["colour"]) if label.get("colour") else None
    if field == "headwear":
        return None if "headwear" not in label else label["headwear"] != "none"
    if field == "bag":
        return label.get("bag")
    if field == "gender":
        return {label["gender"]} if label.get("gender") else None
    value = label.get(field)
    return set(value) if value else None


def score(label: dict, attrs: dict) -> dict[str, bool | None]:
    out = {}
    for field in FIELDS:
        want = expected(label, field)
        if want is None:
            out[field] = None
            continue
        got = predicted(attrs, field)
        out[field] = (got == want) if isinstance(want, bool) else (got in want)
    return out


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="?", default=str(REPO_ROOT / "eval" / "attributes.yaml"))
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    parser.add_argument("--inference", default=settings.inference_url)
    parser.add_argument("--out", default=str(settings.data_dir / "eval"))
    args = parser.parse_args(argv)

    labels = [t for t in yaml.safe_load(Path(args.labels).read_text())["tracks"] if not t.get("skip")]
    api = httpx.Client(base_url=args.api, timeout=60.0)
    inference = InferenceClient(args.inference)
    recordings = [r for r in api.get("/api/recordings").raise_for_status().json() if r["status"] == "indexed"]
    tracks = {r["id"]: api.get(f"/api/recordings/{r['id']}/tracks").raise_for_status().json() for r in recordings}
    by_key = {t["track_key"]: t for ts in tracks.values() for t in ts}
    total_tracks = len(by_key)

    rows = []
    for label in labels:
        a = label["anchor"]
        key = match_anchor(Anchor(a["camera"], to_epoch(parse_local_time(str(a["at"]))), a["x"], a["y"]), recordings, tracks)
        if key is None:
            print(f"  #{label['n']}: anchor matched no track", file=sys.stderr)
            continue
        track = by_key[key]
        crops = [c["path"] for c in sorted(track["crops"], key=lambda c: -(c.get("quality") or 0))]
        subject = "bag" if track["label"] != "person" else "person"
        row = {"n": label["n"], "track_key": key, "label": track["label"]}
        for method in METHODS:
            out = inference.describe(crops, subject, track["label"], method)
            attrs = out["attributes"] or Description(subject=subject).model_dump()  # an unreadable reply sees nothing
            row[method] = {"attributes": attrs, "ms": out["ms"], "score": score(label, attrs), "raw": out.get("raw")}
        rows.append(row)
        print(f"  #{label['n']:3} {key:32} " + "  ".join(
            f"{m}: {sum(1 for v in row[m]['score'].values() if v)}/{sum(1 for v in row[m]['score'].values() if v is not None)}"
            f" {row[m]['ms']} ms" for m in METHODS), flush=True)

    summary = {}
    for method in METHODS:
        per_field = {}
        for field in FIELDS:
            cells = [r[method]["score"][field] for r in rows if r[method]["score"][field] is not None]
            answered = [r for r in rows if r[method]["score"][field] is not None and predicted(r[method]["attributes"], field) is not None]
            per_field[field] = {"n": len(cells), "accuracy": round(100 * sum(cells) / len(cells), 1) if cells else None,
                                "coverage": round(100 * len(answered) / len(cells), 1) if cells else None}
        all_cells = [v for r in rows for v in r[method]["score"].values() if v is not None]
        ms = [r[method]["ms"] for r in rows]
        summary[method] = {
            "accuracy": round(100 * sum(all_cells) / len(all_cells), 1),
            "per_field": per_field,
            "median_ms_per_track": statistics.median(ms),
            "mean_ms_per_track": round(statistics.fmean(ms)),
            "projected_minutes_for_index": round(statistics.fmean(ms) * total_tracks / 60000, 1),
        }

    report = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "labels": args.labels, "tracks_scored": len(rows),
              "index_tracks": total_tracks, "summary": summary, "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"attributes-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n| Field | Labelled | " + " | ".join(f"{m} accuracy (coverage)" for m in METHODS) + " |")
    print("|---|---|" + "---|" * len(METHODS))
    for field in FIELDS:
        cells = [summary[m]["per_field"][field] for m in METHODS]
        print(f"| {field} | {cells[0]['n']} | " + " | ".join(f"{c['accuracy']}% ({c['coverage']}%)" for c in cells) + " |")
    print("| all fields | | " + " | ".join(f"{summary[m]['accuracy']}%" for m in METHODS) + " |")
    print("| time per track | | " + " | ".join(f"{summary[m]['mean_ms_per_track']} ms" for m in METHODS) + " |")
    print(f"| projected for {total_tracks} tracks | | " + " | ".join(f"{summary[m]['projected_minutes_for_index']} min" for m in METHODS) + " |")
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
