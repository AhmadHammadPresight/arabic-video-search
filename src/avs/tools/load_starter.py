"""Upload the fetched starter clips through the API, exactly as a user would, and wait for indexing.

    python -m avs.tools.load_starter [--api URL] [--no-wait]

A clip whose camera and start time are already uploaded is skipped, so running it again is safe.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from avs.tools.fetch_footage import clip_path, load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    parser.add_argument("--actor", default="starter-loader")
    parser.add_argument("--no-wait", action="store_true", help="return once the uploads are queued")
    args = parser.parse_args(argv)

    client = httpx.Client(base_url=args.api, timeout=httpx.Timeout(30.0, write=300.0))
    existing = {(r["camera_id"], r["start_time"]) for r in client.get("/api/recordings").raise_for_status().json()}
    for clip in load_manifest()["clips"]:
        path = clip_path(clip["id"])
        if not path.exists():
            print(f"  skip {clip['id']}: not fetched (run scripts/fetch_footage.sh)")
            continue
        if (clip["camera_id"], clip["start_time"]) in existing:
            print(f"  have {clip['id']}")
            continue
        with path.open("rb") as fh:
            response = client.post(
                "/api/recordings",
                data={"camera_id": clip["camera_id"], "place": clip["place"], "start_time": clip["start_time"], "actor": args.actor},
                files={"file": (path.name, fh, "video/mp4")},
            )
        if response.status_code != 201:
            print(f"  FAIL {clip['id']}: {response.status_code} {response.text}", file=sys.stderr)
            return 1
        print(f"  sent {clip['id']} -> recording {response.json()['id']}")

    if args.no_wait:
        return 0
    started = time.monotonic()
    while True:
        recordings = client.get("/api/recordings").raise_for_status().json()
        pending = [r for r in recordings if r["status"] in ("queued", "indexing")]
        failed = [r for r in recordings if r["status"] == "failed"]
        done = len(recordings) - len(pending) - len(failed)
        current = next((r for r in recordings if r["status"] == "indexing"), None)
        note = f", now {current['camera_id']} {current['progress']:.0%}" if current else ""
        print(f"\r  indexed {done}/{len(recordings)}{note}   ", end="", flush=True)
        if not pending:
            break
        time.sleep(3)
    print(f"\n  finished in {time.monotonic() - started:.0f}s")
    for r in failed:
        print(f"  FAILED {r['camera_id']} {r['start_time']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
