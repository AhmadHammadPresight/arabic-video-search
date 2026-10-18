#!/usr/bin/env bash
# Start every service natively in the background: Qdrant, the query-model server (Ollama),
# the inference service (Apple GPU), the web/API server and the indexing worker.
# All of them listen on 127.0.0.1 only.
# Logs go to data/logs/<service>.log; scripts/stop.sh stops them.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

export AVS_DATA_DIR="${AVS_DATA_DIR:-$ROOT/data}"
export AVS_WEB_PORT="${AVS_WEB_PORT:-8470}"
export AVS_INFERENCE_URL="${AVS_INFERENCE_URL:-http://127.0.0.1:8471}"
export AVS_LLM_URL="${AVS_LLM_URL:-http://127.0.0.1:8472}"
export AVS_QDRANT_URL="${AVS_QDRANT_URL:-http://127.0.0.1:6433}"
# Nothing leaves the Mac: models come from the local cache, no hub calls, no telemetry.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
# Run the rare PyTorch op that has no Apple GPU kernel on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUN="$AVS_DATA_DIR/run"
LOGS="$AVS_DATA_DIR/logs"
mkdir -p "$RUN" "$LOGS" "$AVS_DATA_DIR/qdrant"
[[ -x data/bin/qdrant && -x data/bin/ollama/ollama && -x .venv/bin/python ]] || { echo "start: run scripts/setup.sh first" >&2; exit 1; }

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
# Qdrant's own ports follow AVS_QDRANT_URL (gRPC on the next port), so a second copy can run beside the first.
QDRANT_PORT="${AVS_QDRANT_URL##*:}"
launch qdrant env QDRANT__SERVICE__HTTP_PORT="$QDRANT_PORT" QDRANT__SERVICE__GRPC_PORT="$((QDRANT_PORT + 1))" \
  bash -c "cd '$AVS_DATA_DIR/qdrant' && exec '$ROOT/data/bin/qdrant' --config-path '$ROOT/config/qdrant.yaml'"
wait_for qdrant "$AVS_QDRANT_URL/readyz" 60 "ready"
# The query model server: cloud features off, models kept loaded, and the shared model store never pruned.
launch ollama env OLLAMA_HOST="127.0.0.1:${AVS_LLM_URL##*:}" OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 OLLAMA_KEEP_ALIVE=-1 \
  OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_MAX_LOADED_MODELS=1 ${AVS_OLLAMA_MODELS:+OLLAMA_MODELS="$AVS_OLLAMA_MODELS"} data/bin/ollama/ollama serve
wait_for ollama "$AVS_LLM_URL/api/version" 60 '"version"'
QUERY_MODEL="$(.venv/bin/python -c 'from avs.settings import get_settings; print(get_settings().query_model)')"
echo "  loading the query model $QUERY_MODEL"
curl -fsS "$AVS_LLM_URL/api/generate" -d "{\"model\": \"$QUERY_MODEL\", \"options\": {\"num_ctx\": 4096}}" >/dev/null \
  || echo "start: could not load $QUERY_MODEL; hybrid search falls back to option A until it loads" >&2
launch inference .venv/bin/python -m uvicorn avs.inference.service:app --host 127.0.0.1 --port "${AVS_INFERENCE_URL##*:}"
wait_for inference "$AVS_INFERENCE_URL/health" 300 '"status":"ok"'
launch api .venv/bin/python -m uvicorn avs.api.app:create_app --factory --host 127.0.0.1 --port "$AVS_WEB_PORT"
wait_for api "http://127.0.0.1:$AVS_WEB_PORT/api/recordings" 60 "["
launch worker .venv/bin/python -m avs.api.worker

echo
echo "Open http://127.0.0.1:$AVS_WEB_PORT"
