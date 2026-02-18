"""Celery async tasks for the meme lip-sync pipeline."""

import json
import logging
import os

from celery import Celery

from backend.config import OUTPUT_DIR, REDIS_URL, TEMP_DIR
from backend.pipeline.lipsync import run_lipsync
from backend.pipeline.postprocess import convert_to_gif
from backend.pipeline.preprocess import preprocess_video
from backend.pipeline.tts import generate_speech

logger = logging.getLogger(__name__)

celery_app = Celery("meme_lipsync", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# ── Redis helpers ─────────────────────────────────────────────────────────────

def _get_redis():
    """Return a Redis client."""
    import redis
    return redis.from_url(REDIS_URL)


def _set_job_state(job_id: str, state: dict) -> None:
    r = _get_redis()
    r.set(f"job:{job_id}", json.dumps(state))


def _get_job_state(job_id: str) -> dict | None:
    r = _get_redis()
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


def _update_state(
    job_id: str,
    status: str,
    progress: int,
    output_url: str | None = None,
    error: str | None = None,
) -> None:
    state = {
        "status": status,
        "progress": progress,
        "output_url": output_url,
        "error": error,
    }
    _set_job_state(job_id, state)
    logger.info("Job %s → %s (%d%%)", job_id, status, progress)


# ── Celery task ───────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="process_meme")
def process_meme(self, job_id: str, text: str) -> dict:
    """
    Run the four pipeline stages for a meme lip-sync job.

    Stages
    ------
    1. Preprocess  — GIF/MP4 → normalised MP4
    2. TTS         — text → 16 kHz mono WAV
    3. Lip sync    — Wav2Lip inference
    4. Postprocess — MP4 → optimised GIF
    """
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Retrieve job state to locate the uploaded file
    job_state = _get_job_state(job_id)
    if not job_state:
        _update_state(job_id, "error", 0, error=f"Job {job_id} not found in Redis.")
        return {"error": "job not found"}

    input_path = job_state.get("input_path")
    if not input_path or not os.path.exists(input_path):
        _update_state(job_id, "error", 0, error="Uploaded file not found.")
        return {"error": "input file missing"}

    # ── Stage 1: Preprocess ───────────────────────────────────────────────────
    try:
        _update_state(job_id, "preprocessing", 5)
        preprocess_result = preprocess_video(input_path, job_dir)
        mp4_path = preprocess_result["mp4_path"]
        fps = preprocess_result["fps"]
        video_duration = preprocess_result["duration_seconds"]
        _update_state(job_id, "preprocessing", 20)
    except Exception as exc:
        logger.exception("Preprocess failed for job %s", job_id)
        _update_state(job_id, "error", 20, error=str(exc))
        return {"error": str(exc)}

    # ── Stage 2: TTS ─────────────────────────────────────────────────────────
    try:
        _update_state(job_id, "tts", 25)
        wav_path = os.path.join(job_dir, "speech.wav")
        audio_duration = generate_speech(text, wav_path)
        _update_state(job_id, "tts", 40)
    except Exception as exc:
        logger.exception("TTS failed for job %s", job_id)
        _update_state(job_id, "error", 25, error=str(exc))
        return {"error": str(exc)}

    # ── Stage 3: Lip sync ─────────────────────────────────────────────────────
    try:
        _update_state(job_id, "lipsync", 45)
        lipsync_output = os.path.join(job_dir, "lipsync_output.mp4")
        run_lipsync(mp4_path, wav_path, lipsync_output, job_dir)
        _update_state(job_id, "lipsync", 75)
    except Exception as exc:
        logger.exception("Lipsync failed for job %s", job_id)
        _update_state(job_id, "error", 45, error=str(exc))
        return {"error": str(exc)}

    # ── Stage 4: Postprocess ──────────────────────────────────────────────────
    try:
        _update_state(job_id, "postprocessing", 80)
        output_filename = f"{job_id}.gif"
        output_gif_path = os.path.join(OUTPUT_DIR, output_filename)
        convert_to_gif(lipsync_output, output_gif_path, fps)
        output_url = f"/output/{output_filename}"
        _update_state(job_id, "done", 100, output_url=output_url)
        logger.info("Job %s complete: %s", job_id, output_url)
        return {"output_url": output_url}
    except Exception as exc:
        logger.exception("Postprocess failed for job %s", job_id)
        _update_state(job_id, "error", 80, error=str(exc))
        return {"error": str(exc)}
