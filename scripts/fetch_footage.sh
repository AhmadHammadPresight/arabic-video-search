#!/usr/bin/env bash
# Download, trim and attribute the public starter footage (footage/starter.json) into data/footage.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m avs.tools.fetch_footage "$@"
