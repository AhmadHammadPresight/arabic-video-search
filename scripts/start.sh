#!/usr/bin/env bash
# Start every service natively in the background: Qdrant, the inference service (Apple GPU),
# the web/API server and the indexing worker. All of them listen on 127.0.0.1 only.
# Logs go to data/logs/<service>.log; scripts/stop.sh stops them.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

export AVS_DATA_DIR="${AVS_DATA_DIR:-$ROOT/data}"
export AVS_WEB_PORT="${AVS_WEB_PORT:-8470}"
export AVS_INFERENCE_URL="${AVS_INFERENCE_URL:-http://127.0.0.1:8471}"
export AVS_QDRANT_URL="${AVS_QDRANT_URL:-http://127.0.0.1:6433}"
# Nothing leaves the Mac: models come from the local cache, no hub calls, no telemetry.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
# Run the rare PyTorch op that has no Apple GPU kernel on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUN="$AVS_DATA_DIR/run"
LOGS="$AVS_DATA_DIR/logs"
mkdir -p "$RUN" "$LOGS" "$AVS_DATA_DIR/qdrant"
[[ -x data/bin/qdrant && -x .venv/bin/python ]] || { echo "start: run scripts/setup.sh first" >&2; exit 1; }

running() { [[ -f "$RUN/$1.pid" ]] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null; }

launch() {
  local name="$1"; shift
  if running "$name"; then
    echo "  $name already running (pid $(cat "$RUN/$name.pid"))"
    return
  fi
  nohup "$@" >>"$LOGS/$name.log" 2>&1 &
  echo $! >"$RUN/$name.pid"
  echo "  $name started (pid $!, log data/logs/$name.log)"
}

wait_for() {
  local name="$1" url="$2" seconds="$3"
  for _ in $(seq "$seconds"); do
    curl -fsS "$url" 2>/dev/null | grep -qF "$4" && { echo "  $name ready"; return; }
    running "$name" || { echo "start: $name exited; see data/logs/$name.log" >&2; tail -20 "$LOGS/$name.log" >&2; exit 1; }
    sleep 1
  done
  echo "start: $name not ready after ${seconds}s; see data/logs/$name.log" >&2
  exit 1
}

echo "==> Starting services"
launch qdrant bash -c "cd '$AVS_DATA_DIR/qdrant' && exec '$ROOT/data/bin/qdrant' --config-path '$ROOT/config/qdrant.yaml'"
wait_for qdrant "$AVS_QDRANT_URL/readyz" 60 "ready"
launch inference .venv/bin/python -m uvicorn avs.inference.service:app --host 127.0.0.1 --port "${AVS_INFERENCE_URL##*:}"
wait_for inference "$AVS_INFERENCE_URL/health" 300 '"status":"ok"'
launch api .venv/bin/python -m uvicorn avs.api.app:create_app --factory --host 127.0.0.1 --port "$AVS_WEB_PORT"
wait_for api "http://127.0.0.1:$AVS_WEB_PORT/api/recordings" 60 "["
launch worker .venv/bin/python -m avs.api.worker

echo
echo "Open http://127.0.0.1:$AVS_WEB_PORT"
