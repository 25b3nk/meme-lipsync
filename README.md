# Meme Lip-Sync Generator

Upload any meme GIF or MP4 containing a face, type new text, and receive a lip-synced GIF in seconds. The pipeline converts the video to MP4, synthesises speech with Edge TTS, runs Wav2Lip inference for realistic lip movement, and re-encodes the result as an optimised GIF.

---

## Prerequisites

**Option A — Docker (recommended)**
- Docker ≥ 24 and Docker Compose v2

**Option B — Local Python**
- Python 3.10+
- FFmpeg (`ffmpeg` and `ffprobe` in `PATH`)
- Gifsicle (`gifsicle` in `PATH`)
- Redis

---

## Setup

### 1. Run the setup script

```bash
bash setup.sh
```

This will:
- Create `models/`, `temp/`, and `outputs/` directories
- Clone the Wav2Lip repository into `wav2lip/`
- Install Wav2Lip's Python dependencies
- Install backend Python dependencies
- Print instructions for the manual model download

### 2. Download the model weights (manual)

The Wav2Lip GAN weights cannot be downloaded automatically (Google Drive requires browser authentication). Download `wav2lip_gan.pth` from the link below and place it at `models/wav2lip_gan.pth`.

**Download link:**
https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha_m_research_iiit_ac_in%2FDocuments%2FWav2Lip_Models%2Fwav2lip_gan%2Epth

> Use `wav2lip_gan.pth`, **not** `wav2lip.pth` — the GAN variant produces sharper, more visually convincing lip movements.

### 3. Start the application

**Docker:**
```bash
docker-compose up --build
```

**Local (without Docker):**
```bash
# In separate terminals:
redis-server
celery -A backend.tasks worker --loglevel=info --concurrency=1
uvicorn backend.main:app --reload
```

Open http://localhost:8000 in your browser.

---

## Usage

1. Open http://localhost:8000 (the frontend is served by the API).
2. Drag and drop or click to upload a GIF or MP4 containing a face.
3. Type the text you want the character to say (max 200 characters).
4. Click **Generate Lip-Sync** and wait for the pipeline to finish.
5. Download the resulting GIF.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload GIF or MP4. Returns `job_id`, `preview_url`. |
| `POST` | `/generate` | Start pipeline. Body: `{ job_id, text }`. Returns `task_id`. |
| `GET`  | `/status/{task_id}` | Poll task status. Returns `{ status, progress, output_url, error }`. |
| `GET`  | `/output/{filename}` | Download completed GIF. |
| `GET`  | `/health` | Liveness check. |

### Status values

| Status | Meaning |
|--------|---------|
| `uploaded` | File received, not yet started |
| `queued` | Waiting for a Celery worker |
| `preprocessing` | Converting input to MP4, face detection |
| `tts` | Synthesising speech |
| `lipsync` | Running Wav2Lip inference |
| `postprocessing` | Converting to optimised GIF |
| `done` | Pipeline complete — `output_url` is populated |
| `error` | Pipeline failed — `error` field contains the message |

---

## Architecture

```
Browser (index.html)
    │  REST
    ▼
FastAPI (main.py)                  Redis
    │  Celery task                   │
    ▼                                │
Celery worker (tasks.py) ───────────┘
    │
    ├─ Stage 1: preprocess.py   (FFmpeg + OpenCV)
    ├─ Stage 2: tts.py          (Edge TTS + FFmpeg)
    ├─ Stage 3: lipsync.py      (Wav2Lip subprocess)
    └─ Stage 4: postprocess.py  (FFmpeg + Gifsicle)
```

---

## Known Limitations

- **Face orientation** — Wav2Lip works best with forward-facing faces. Profile or heavily angled faces may produce poor results or fail face detection.
- **Text length** — Audio longer than 1.5× the video duration is rejected. Keep text concise for short meme clips.
- **CPU vs GPU** — On CPU, Wav2Lip inference takes 30–60 s for a 3-second clip. On a GPU (e.g. RunPod T4) the same clip takes 3–5 s. No code changes are needed to use a CUDA-capable GPU; install the GPU version of PyTorch inside the container.
- **GIF file size** — Complex or high-resolution input GIFs produce large outputs. The pipeline applies Gifsicle `-O3 --lossy=80` compression to mitigate this.
- **Single concurrency** — The Celery worker is configured with `--concurrency=1` to avoid GPU memory conflicts. Increase this only if you have sufficient VRAM.
