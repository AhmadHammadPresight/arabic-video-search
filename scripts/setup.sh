#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: FFmpeg, the locked Python environment,
# the pinned Qdrant binary and the model weights. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_VERSION="v1.19.2"
QDRANT_SHA256="d24dfd25f4584684b337dd30ee016472cd25f53fc2501f1cd058ab3d7ee0b019"
QDRANT_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-aarch64-apple-darwin.tar.gz"

die() { echo "setup: $*" >&2; exit 1; }

[[ "$(uname -s)/$(uname -m)" == "Darwin/arm64" ]] || die "this PoC needs an Apple Silicon Mac (Darwin/arm64)"

if ! command -v ffmpeg >/dev/null || ! command -v uv >/dev/null; then
  command -v brew >/dev/null || die "install Homebrew first (https://brew.sh), then run this script again"
  command -v ffmpeg >/dev/null || brew install ffmpeg
  command -v uv >/dev/null || brew install uv
fi
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_videotoolbox || die "this ffmpeg has no h264_videotoolbox encoder"

echo "==> Python environment (uv.lock, Python $(cat .python-version))"
uv sync --frozen

echo "==> Qdrant ${QDRANT_VERSION}"
mkdir -p data/bin data/qdrant
if ! data/bin/qdrant --version 2>/dev/null | grep -q "${QDRANT_VERSION#v}"; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fL --retry 3 -o "$tmp/qdrant.tar.gz" "$QDRANT_URL"
  echo "${QDRANT_SHA256}  $tmp/qdrant.tar.gz" | shasum -a 256 -c - >/dev/null || die "Qdrant checksum mismatch"
  tar -xzf "$tmp/qdrant.tar.gz" -C data/bin qdrant
  chmod +x data/bin/qdrant
fi
data/bin/qdrant --version

echo "==> Model weights"
TQDM_DISABLE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m avs.tools.download_models

echo "setup done. Next: scripts/fetch_footage.sh, then scripts/start.sh"
