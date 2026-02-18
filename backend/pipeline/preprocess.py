"""Stage 1: Preprocess GIF or MP4 input into a standardised MP4 for Wav2Lip."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess command and raise on non-zero exit."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _extract_gif_fps(input_path: str) -> float:
    """Return average FPS derived from GIF frame delays via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        input_path,
    ]
    result = _run(cmd, "ffprobe")
    data = json.loads(result.stdout)

    # Try to pull r_frame_rate from the first video stream
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "")
            if r_frame_rate and "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                num, den = int(num), int(den)
                if den > 0 and num > 0:
                    fps = num / den
                    # GIF r_frame_rate is often reported as 100/1; clamp to sane range
                    if fps > 50:
                        fps = 10.0
                    return fps

    # Fallback: count frames and use avg_frame_rate
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            avg = stream.get("avg_frame_rate", "10/1")
            if avg and "/" in avg:
                num, den = avg.split("/")
                num, den = int(num), int(den)
                if den > 0 and num > 0:
                    return min(num / den, 30.0)

    return 10.0  # safe default for GIFs


def _gif_to_mp4(input_path: str, output_path: str, fps: float) -> None:
    """Convert a GIF to MP4 using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    _run(cmd, "GIF→MP4 conversion")


def _get_video_info(mp4_path: str) -> dict:
    """Return fps, frame_count, and duration_seconds for an MP4."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        mp4_path,
    ]
    result = _run(cmd, "ffprobe (mp4 info)")
    data = json.loads(result.stdout)

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "25/1")
            num, den = r_frame_rate.split("/")
            fps = int(num) / int(den)

            nb_frames = stream.get("nb_frames")
            duration = stream.get("duration")

            if nb_frames:
                frame_count = int(nb_frames)
            elif duration:
                frame_count = int(float(duration) * fps)
            else:
                frame_count = 0

            duration_seconds = float(duration) if duration else (frame_count / fps if fps else 0)
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration_seconds,
            }

    raise ValueError("No video stream found in file")


def _has_face(mp4_path: str, max_frames: int = 10) -> bool:
    """Return True if a face is detected in any of the first max_frames frames."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for face detection: {mp4_path}")

    found = False
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            found = True
            break
        frame_idx += 1

    cap.release()
    return found


def preprocess_video(input_path: str, job_dir: str) -> dict:
    """
    Preprocess a GIF or MP4 input into a standardised MP4.

    Parameters
    ----------
    input_path : str
        Absolute path to the uploaded GIF or MP4.
    job_dir : str
        Working directory for this job.

    Returns
    -------
    dict with keys: mp4_path, fps, frame_count, duration_seconds, has_face
    """
    os.makedirs(job_dir, exist_ok=True)
    input_path = str(input_path)
    suffix = Path(input_path).suffix.lower()

    mp4_path = os.path.join(job_dir, "input.mp4")

    if suffix == ".gif":
        logger.info("Input is GIF — extracting FPS and converting to MP4")
        fps = _extract_gif_fps(input_path)
        logger.info("Detected GIF FPS: %.2f", fps)
        _gif_to_mp4(input_path, mp4_path, fps)
    elif suffix in (".mp4", ".mov", ".webm", ".avi"):
        logger.info("Input is video — copying to job directory")
        shutil.copy2(input_path, mp4_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Please upload a GIF or MP4.")

    info = _get_video_info(mp4_path)

    logger.info("Running face detection on first 10 frames")
    face_found = _has_face(mp4_path)

    if not face_found:
        raise ValueError(
            "No face detected in the uploaded video. "
            "Wav2Lip requires a clearly visible, forward-facing face."
        )

    return {
        "mp4_path": mp4_path,
        "fps": info["fps"],
        "frame_count": info["frame_count"],
        "duration_seconds": info["duration_seconds"],
        "has_face": True,
    }
