#!/usr/bin/env bash
# Run eval/queries.yaml against the running index; prints the scores and saves a JSON report in data/eval.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m avs.tools.evaluate --api "http://127.0.0.1:${AVS_WEB_PORT:-8470}" "$@"
