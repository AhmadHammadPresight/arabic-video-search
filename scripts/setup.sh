#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: FFmpeg, the locked Python environment, the pinned
# Qdrant and Ollama binaries, the vision model weights and the query model. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

QDRANT_VERSION="v1.19.2"
QDRANT_SHA256="d24dfd25f4584684b337dd30ee016472cd25f53fc2501f1cd058ab3d7ee0b019"
QDRANT_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-aarch64-apple-darwin.tar.gz"
OLLAMA_VERSION="v0.40.6"
OLLAMA_SHA256="b2b55434865a668decb494110d7fd970eed3ecc8837be2872f672ccbb95df974"
OLLAMA_URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-darwin.tgz"
LLM_URL="${AVS_LLM_URL:-http://127.0.0.1:8472}"

die() { echo "setup: $*" >&2; exit 1; }

tmp="$(mktemp -d)"
server=""
cleanup() {
  [[ -n "$server" ]] && kill "$server" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

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
mkdir -p data/bin data/qdrant data/logs
if ! data/bin/qdrant --version 2>/dev/null | grep -q "${QDRANT_VERSION#v}"; then
  curl -fL --retry 3 -o "$tmp/qdrant.tar.gz" "$QDRANT_URL"
  echo "${QDRANT_SHA256}  $tmp/qdrant.tar.gz" | shasum -a 256 -c - >/dev/null || die "Qdrant checksum mismatch"
  tar -xzf "$tmp/qdrant.tar.gz" -C data/bin qdrant
  chmod +x data/bin/qdrant
fi
data/bin/qdrant --version

echo "==> Ollama ${OLLAMA_VERSION} (runs the query model)"
if ! data/bin/ollama/ollama --version 2>&1 | grep -q "${OLLAMA_VERSION#v}"; then
  curl -fL --retry 3 -o "$tmp/ollama.tgz" "$OLLAMA_URL"
  echo "${OLLAMA_SHA256}  $tmp/ollama.tgz" | shasum -a 256 -c - >/dev/null || die "Ollama checksum mismatch"
  rm -rf data/bin/ollama
  mkdir -p data/bin/ollama
  tar -xzf "$tmp/ollama.tgz" -C data/bin/ollama
fi
data/bin/ollama/ollama --version 2>&1 | tail -1

echo "==> Model weights"
TQDM_DISABLE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m avs.tools.download_models

QUERY_MODEL="$(.venv/bin/python -c 'from avs.settings import get_settings; print(get_settings().query_model)')"
echo "==> Query model ${QUERY_MODEL}"
# Pull through the running server if there is one, otherwise through a temporary one on the same port.
export OLLAMA_HOST="127.0.0.1:${LLM_URL##*:}"
if ! curl -fsS "$LLM_URL/api/version" >/dev/null 2>&1; then
  env OLLAMA_NO_CLOUD=1 OLLAMA_NOPRUNE=1 ${AVS_OLLAMA_MODELS:+OLLAMA_MODELS="$AVS_OLLAMA_MODELS"} \
    data/bin/ollama/ollama serve >>data/logs/ollama-setup.log 2>&1 &
  server=$!
  for _ in $(seq 30); do curl -fsS "$LLM_URL/api/version" >/dev/null 2>&1 && break; sleep 1; done
fi
echo "  pulling into the Ollama model store (about 18 GB the first time; a model already there is only checked)"
data/bin/ollama/ollama pull "$QUERY_MODEL" || die "could not pull $QUERY_MODEL"
data/bin/ollama/ollama list | grep -F "$QUERY_MODEL" >/dev/null || die "$QUERY_MODEL is not in the Ollama model store"

echo "setup done. Next: scripts/fetch_footage.sh, then scripts/start.sh"
