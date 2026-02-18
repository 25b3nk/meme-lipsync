#!/usr/bin/env bash
# setup.sh — one-time project setup for meme-lipsync
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Creating working directories"
mkdir -p models temp outputs

# ── Clone Wav2Lip ─────────────────────────────────────────────────────────────
if [ -d "wav2lip/.git" ]; then
  echo "==> Wav2Lip repo already present — skipping clone"
else
  echo "==> Cloning Wav2Lip"
  git clone https://github.com/Rudrabha/Wav2Lip wav2lip
fi

# ── Install Wav2Lip Python dependencies ──────────────────────────────────────
echo "==> Installing Wav2Lip requirements"
pip install -r wav2lip/requirements.txt

# ── Install backend requirements ──────────────────────────────────────────────
echo "==> Installing backend requirements"
pip install -r backend/requirements.txt

# ── Download model weights ────────────────────────────────────────────────────
MODEL_PATH="models/wav2lip_gan.pth"
MODEL_URL="https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
EXPECTED_SHA256="ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"

if [ -f "$MODEL_PATH" ]; then
  echo "==> Model weights already present at $MODEL_PATH — skipping download"
else
  echo "==> Downloading wav2lip_gan.pth from Hugging Face (~436 MB)"
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$MODEL_PATH" "$MODEL_URL"
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
  else
    echo "ERROR: Neither wget nor curl found. Install one and re-run." >&2
    exit 1
  fi

  echo "==> Verifying checksum"
  if command -v sha256sum &>/dev/null; then
    ACTUAL=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
  elif command -v shasum &>/dev/null; then
    ACTUAL=$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')
  else
    echo "WARNING: No sha256 tool found — skipping checksum verification"
    ACTUAL="$EXPECTED_SHA256"
  fi

  if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: Checksum mismatch! File may be corrupt." >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  actual:   $ACTUAL" >&2
    rm -f "$MODEL_PATH"
    exit 1
  fi
  echo "==> Checksum OK"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
cat <<'MSG'

==========================================================================
 Setup complete!
==========================================================================

Start the full stack with Docker:

    docker-compose up --build

Or run locally without Docker (three separate terminals):

    redis-server
    celery -A backend.tasks worker --loglevel=info --concurrency=1
    uvicorn backend.main:app --reload

Then open http://localhost:8000 in your browser.

MSG
