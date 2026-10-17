#!/usr/bin/env bash
# Upload the fetched starter clips through the API and wait until they are indexed.
# Needs the services running (scripts/start.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m avs.tools.load_starter --api "http://127.0.0.1:${AVS_WEB_PORT:-8470}" "$@"
