#!/usr/bin/env bash
# Re-index every recording with the current pipeline (for example to attach attributes to recordings
# indexed before phase 2). Needs the services running (scripts/start.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m avs.tools.reindex_all --api "http://127.0.0.1:${AVS_WEB_PORT:-8470}" "$@"
