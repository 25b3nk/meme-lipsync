"""FastAPI application — meme lip-sync generator."""

import json
import logging
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import MAX_FILE_SIZE_MB, OUTPUT_DIR, REDIS_URL, TEMP_DIR
from backend.tasks import celery_app, process_meme

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meme Lip-Sync Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis helpers ─────────────────────────────────────────────────────────────

def _get_redis():
    import redis
    return redis.from_url(REDIS_URL)


def _get_job_state(job_id: str) -> dict | None:
    r = _get_redis()
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


def _set_job_state(job_id: str, state: dict) -> None:
    r = _get_redis()
    r.set(f"job:{job_id}", json.dumps(state))


# ── Request/response models ───────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    job_id: str
    text: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a GIF or MP4 upload.

    Returns
    -------
    JSON with job_id and preview_url.
    """
    allowed_types = {"image/gif", "video/mp4", "video/quicktime", "video/webm", "video/avi"}
    allowed_extensions = {".gif", ".mp4", ".mov", ".webm", ".avi"}

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a GIF or MP4.",
        )

    # Read first chunk to check size without loading the whole file
    content = await file.read()
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_filename = f"upload{suffix}"
    input_path = os.path.join(job_dir, input_filename)

    async with aiofiles.open(input_path, "wb") as f:
        await f.write(content)

    # Store initial job state
    _set_job_state(job_id, {
        "status": "uploaded",
        "progress": 0,
        "output_url": None,
        "error": None,
        "input_path": input_path,
    })

    preview_url = f"/output/preview/{job_id}{suffix}"

    logger.info("Uploaded job %s: %s (%d bytes)", job_id, file.filename, len(content))

    return JSONResponse({
        "job_id": job_id,
        "preview_url": preview_url,
        "filename": file.filename,
        "size_bytes": len(content),
    })


@app.get("/output/preview/{job_id}{suffix}")
async def get_preview(job_id: str, suffix: str):
    """Serve the uploaded file for preview in the frontend."""
    # Reconstruct the file path from job_id
    job_state = _get_job_state(job_id)
    if not job_state:
        raise HTTPException(status_code=404, detail="Job not found.")

    input_path = job_state.get("input_path")
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Preview file not found.")

    return FileResponse(input_path)


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Start the lip-sync pipeline for an uploaded job.

    Returns
    -------
    JSON with task_id.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(request.text) > 200:
        raise HTTPException(status_code=400, detail="Text must be 200 characters or fewer.")

    job_state = _get_job_state(request.job_id)
    if not job_state:
        raise HTTPException(status_code=404, detail=f"Job {request.job_id} not found.")

    task = process_meme.apply_async(
        args=[request.job_id, request.text],
        task_id=str(uuid.uuid4()),
    )

    # Update job state to record task_id
    job_state["task_id"] = task.id
    job_state["status"] = "queued"
    _set_job_state(request.job_id, job_state)

    logger.info("Started task %s for job %s", task.id, request.job_id)

    return JSONResponse({"task_id": task.id, "job_id": request.job_id})


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Return the current status of a Celery task.

    The job state is stored in Redis under the job_id, so we look up the
    task → job_id mapping via Celery's result backend.
    """
    result = celery_app.AsyncResult(task_id)

    # Try to find job state from Redis by scanning for task_id
    r = _get_redis()
    job_state = None
    # Celery result backend stores task state independently; we scan job keys
    for key in r.scan_iter("job:*"):
        raw = r.get(key)
        if not raw:
            continue
        state = json.loads(raw)
        if state.get("task_id") == task_id:
            job_state = state
            break

    if job_state is None:
        # Fall back to Celery result status
        celery_status = result.status
        if celery_status == "PENDING":
            return JSONResponse({
                "status": "queued",
                "progress": 0,
                "output_url": None,
                "error": None,
            })
        if celery_status == "FAILURE":
            return JSONResponse({
                "status": "error",
                "progress": 0,
                "output_url": None,
                "error": str(result.result),
            })
        return JSONResponse({
            "status": celery_status.lower(),
            "progress": 0,
            "output_url": None,
            "error": None,
        })

    return JSONResponse({
        "status": job_state.get("status", "unknown"),
        "progress": job_state.get("progress", 0),
        "output_url": job_state.get("output_url"),
        "error": job_state.get("error"),
    })


@app.get("/output/{filename}")
async def serve_output(filename: str):
    """Serve a completed output GIF."""
    # Prevent path traversal
    safe_filename = Path(filename).name
    file_path = os.path.join(OUTPUT_DIR, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(file_path, media_type="image/gif")


@app.get("/health")
async def health():
    return {"status": "ok"}
