"""Helpers for writing ground truth: contact sheets of indexed tracks, and anchors for chosen tracks.

    python -m avs.tools.track_sheet sheet --camera G331 [--start 2018-03-07T17:00:00] [--label person]
    python -m avs.tools.track_sheet anchors RECORDING_ID/person-3 [...]

`sheet` writes numbered JPEG pages of every track's best crops to data/eval/sheets, to pick
targets by eye without looking at search results. `anchors` prints YAML anchors (camera,
local time, box centre) for chosen tracks, ready to paste into eval/queries.yaml.
"""

from __future__ import annotations

import argparse
import sys

import httpx
from PIL import Image, ImageDraw

from avs.api.times import from_epoch, offset
from avs.settings import get_settings

TILE_W, TILE_H, CAPTION = 240, 300, 30


def fetch(api: str) -> tuple[list[dict], dict[str, list[dict]]]:
    client = httpx.Client(base_url=api, timeout=60.0)
    recordings = client.get("/api/recordings").raise_for_status().json()
    tracks = {r["id"]: client.get(f"/api/recordings/{r['id']}/tracks").raise_for_status().json()
              for r in recordings if r["status"] == "indexed"}
    return recordings, tracks


def sheet(args) -> None:
    recordings, tracks = fetch(args.api)
    data = get_settings().data_dir
    out_dir = data / "eval" / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in recordings:
        if rec["camera_id"] != args.camera or (args.start and rec["start_time"] != args.start):
            continue
        chosen = [t for t in tracks.get(rec["id"], [])
                  if (not args.label or (t["label"] == "person") == (args.label == "person"))
                  and t["t_end"] - t["t_start"] >= args.min_duration]
        for page, start in enumerate(range(0, len(chosen), args.page_size)):
            batch = chosen[start : start + args.page_size]
            cols = 6
            rows = (len(batch) + cols - 1) // cols
            img = Image.new("RGB", (cols * TILE_W, rows * (TILE_H + CAPTION)), (25, 25, 25))
            draw = ImageDraw.Draw(img)
            for i, track in enumerate(batch):
                x, y = (i % cols) * TILE_W, (i // cols) * (TILE_H + CAPTION)
                crops = track["crops"]
                # Two crops side by side: early and late in the track.
                for j, crop in enumerate((crops[0], crops[-1])):
                    im = Image.open(data / crop["path"]).convert("RGB")
                    im.thumbnail((TILE_W // 2 - 2, TILE_H))
                    img.paste(im, (x + j * (TILE_W // 2), y))
                t0 = from_epoch(offset(rec["start_time"], track["t_start"]))[11:19]
                t1 = from_epoch(offset(rec["start_time"], track["t_end"]))[11:19]
                draw.text((x + 3, y + TILE_H + 2), f"{track['track_id']}", fill=(255, 230, 0))
                draw.text((x + 3, y + TILE_H + 15), f"{t0}-{t1}", fill=(200, 200, 200))
            path = out_dir / f"{rec['camera_id']}_{rec['start_time'].replace(':', '')}_{args.label or 'all'}_p{page + 1}.jpg"
            img.save(path, quality=88)
            print(f"{path}  ({len(batch)} tracks, recording {rec['id']})")


def anchors(args) -> None:
    recordings, tracks = fetch(args.api)
    by_id = {r["id"]: r for r in recordings}
    for key in args.track_keys:
        recording_id = key.split("/")[0]
        rec = by_id[recording_id]
        track = next(t for t in tracks[recording_id] if t["track_key"] == key)
        print(f"      # {key}")
        crops = track["crops"]
        for crop in (crops[0], crops[len(crops) // 2], crops[-1]) if len(crops) > 2 else crops:
            x1, y1, x2, y2 = crop["box"]
            at = from_epoch(offset(rec["start_time"], crop["t"]))
            print(f'      - {{camera: {rec["camera_id"]}, at: "{at}", x: {(x1 + x2) / 2:.3f}, y: {(y1 + y2) / 2:.3f}}}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--camera", required=True)
    s.add_argument("--start", help="recording start time, when a camera has several recordings")
    s.add_argument("--label", choices=["person", "bag"])
    s.add_argument("--min-duration", type=float, default=1.0)
    s.add_argument("--page-size", type=int, default=40)
    s.set_defaults(func=sheet)
    a = sub.add_parser("anchors")
    a.add_argument("track_keys", nargs="+")
    a.set_defaults(func=anchors)
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
