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

# ── Model weights notice ──────────────────────────────────────────────────────
cat <<'MSG'

==========================================================================
 MANUAL STEP REQUIRED — Download Wav2Lip model weights
==========================================================================

 The Wav2Lip GAN model weights cannot be downloaded automatically
 because Google Drive requires browser-based authentication.

 Please download  wav2lip_gan.pth  from the link below and save it to:

     models/wav2lip_gan.pth

 Download link:
 https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha_m_research_iiit_ac_in%2FDocuments%2FWav2Lip_Models%2Fwav2lip_gan%2Epth

 Use wav2lip_gan.pth (NOT wav2lip.pth) — the GAN variant produces
 sharper, more visually convincing lip movements.

==========================================================================

Setup complete. After placing the model weights, start the stack with:

    docker-compose up --build

or, for local development without Docker:

    redis-server &
    celery -A backend.tasks worker --loglevel=info --concurrency=1 &
    uvicorn backend.main:app --reload

Then open http://localhost:8000 in your browser.

MSG
