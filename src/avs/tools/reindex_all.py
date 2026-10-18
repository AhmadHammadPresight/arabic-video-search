"""Re-index every recording through the API, so existing recordings get the current pipeline (attributes included).

    python -m avs.tools.reindex_all [--api URL] [--actor NAME] [--no-wait]

Each recording is queued again exactly as the re-index button does (and audit-logged the same way);
the worker replaces its tracks, crops and vectors. Recordings already queued or indexing are left alone.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8470")
    parser.add_argument("--actor", default="reindex-all")
    parser.add_argument("--no-wait", action="store_true", help="return once every recording is queued")
    args = parser.parse_args(argv)

    client = httpx.Client(base_url=args.api, timeout=60.0)
    recordings = client.get("/api/recordings").raise_for_status().json()
    for r in recordings:
        if r["status"] in ("queued", "indexing"):
            print(f"  busy {r['camera_id']} {r['start_time']}")
            continue
        client.post(f"/api/recordings/{r['id']}/reindex", params={"actor": args.actor}).raise_for_status()
        print(f"  queued {r['camera_id']} {r['start_time']}")
    if args.no_wait:
        return 0
    started = time.monotonic()
    while True:
        recordings = client.get("/api/recordings").raise_for_status().json()
        pending = [r for r in recordings if r["status"] in ("queued", "indexing")]
        current = next((r for r in recordings if r["status"] == "indexing"), None)
        note = f", now {current['camera_id']} {current['progress']:.0%}" if current else ""
        print(f"\r  indexed {len(recordings) - len(pending)}/{len(recordings)}{note}   ", end="", flush=True)
        if not pending:
            break
        time.sleep(3)
    print(f"\n  finished in {time.monotonic() - started:.0f}s")
    failed = [r for r in recordings if r["status"] == "failed"]
    for r in failed:
        print(f"  FAILED {r['camera_id']} {r['start_time']}: {r['error']}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
