#!/usr/bin/env bash
# Stop the services started by scripts/start.sh. A job interrupted mid-way is queued again on the next start.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${AVS_DATA_DIR:-$(pwd)/data}/run"

for name in worker api inference qdrant; do
  pidfile="$RUN/$name.pid"
  [[ -f "$pidfile" ]] || continue
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    echo "  $name stopped"
  fi
  rm -f "$pidfile"
done
