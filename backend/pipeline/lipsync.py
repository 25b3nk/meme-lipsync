"""Stage 3: Wav2Lip inference wrapper."""

import logging
import os
import subprocess
import sys

import cv2

from backend.config import MODEL_PATH, WAV2LIP_DIR

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "", log_file: str | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess and optionally write stdout/stderr to a log file."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout or "")
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr or "")

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _get_audio_duration(wav_path: str) -> float:
    """Return audio duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    return float(result.stdout.strip())


def _get_video_duration(video_path: str) -> float:
    """Return video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return 0.0
    return float(result.stdout.strip())


def _trim_video_to_duration(video_path: str, duration: float, output_path: str) -> None:
    """Trim a video to at most `duration` seconds."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path,
    ]
    _run(cmd, "Video trim")


SFD_MODEL_PATH = os.path.join(WAV2LIP_DIR, "face_detection/detection/sfd/s3fd.pth")
SFD_MIN_BYTES = 80 * 1024 * 1024  # ~86 MB when complete


def _sfd_ready() -> bool:
    """Return True if the SFD face-detection model is fully downloaded."""
    return os.path.isfile(SFD_MODEL_PATH) and os.path.getsize(SFD_MODEL_PATH) >= SFD_MIN_BYTES


def _detect_face_box(video_path: str) -> tuple[int, int, int, int] | None:
    """
    Detect the first face in the video using MTCNN (facenet-pytorch).

    Returns (y1, y2, x1, x2) for wav2lip --box argument, or None if SFD is
    available (inference.py will then use SFD natively, which is more accurate).
    """
    if _sfd_ready():
        logger.info("SFD model ready — letting inference.py handle face detection natively")
        return None

    try:
        from facenet_pytorch import MTCNN
        from PIL import Image
    except ImportError:
        logger.warning("facenet-pytorch not installed; falling back to SFD detector")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mtcnn = MTCNN(keep_all=False, device="cpu", post_process=False)
    found = None

    for _ in range(30):  # check up to 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        h_frame, w_frame = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, probs = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0 and probs[0] > 0.9:
            fx1, fy1, fx2, fy2 = [int(v) for v in boxes[0]]
            bh = fy2 - fy1
            # Add padding for chin/forehead so Wav2Lip sees the full face
            pad_v = int(bh * 0.25)
            pad_h = int(bh * 0.1)
            y1 = max(0, fy1 - pad_v)
            y2 = min(h_frame, fy2 + pad_v)
            x1 = max(0, fx1 - pad_h)
            x2 = min(w_frame, fx2 + pad_h)
            found = (y1, y2, x1, x2)
            break

    cap.release()
    if found:
        logger.info("Face detected via MTCNN: box=%s (skipping SFD download)", found)
    else:
        logger.warning("No face detected via MTCNN; inference.py will use SFD detector")
    return found


def run_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    job_dir: str,
) -> str:
    """
    Run Wav2Lip inference to generate a lip-synced video.

    Parameters
    ----------
    video_path : str
        Path to the input MP4 (with face).
    audio_path : str
        Path to the 16 kHz mono WAV.
    output_path : str
        Destination path for the Wav2Lip output MP4.
    job_dir : str
        Job working directory (used for log file and temp files).

    Returns
    -------
    str
        Path to the lip-synced MP4 (same as output_path).
    """
    log_file = os.path.join(job_dir, "lipsync.log")

    audio_duration = _get_audio_duration(audio_path)
    video_duration = _get_video_duration(video_path)

    logger.info(
        "Audio duration: %.2f s | Video duration: %.2f s",
        audio_duration,
        video_duration,
    )

    # Trim video to audio duration (avoids processing unnecessary frames,
    # especially important for image inputs that generate long looping videos)
    effective_video_path = video_path
    if audio_duration > 0 and video_duration > audio_duration:
        logger.info("Trimming video to audio duration (%.2f s)", audio_duration)
        trimmed_mp4 = os.path.join(job_dir, "input_trimmed.mp4")
        _trim_video_to_duration(video_path, audio_duration, trimmed_mp4)
        effective_video_path = trimmed_mp4

    effective_audio_path = audio_path

    # Detect face bounding box with OpenCV so we can pass --box and skip
    # the SFD face-detector download inside inference.py
    face_box = _detect_face_box(video_path)

    # Verify Wav2Lip repo and model exist
    inference_script = os.path.join(WAV2LIP_DIR, "inference.py")
    if not os.path.exists(inference_script):
        raise RuntimeError(
            f"Wav2Lip inference script not found at {inference_script}. "
            "Run setup.sh to clone the Wav2Lip repository."
        )
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Wav2Lip model not found at {MODEL_PATH}. "
            "Download wav2lip_gan.pth and place it in the models/ directory."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        sys.executable, inference_script,
        "--checkpoint_path", MODEL_PATH,
        "--face", effective_video_path,
        "--audio", effective_audio_path,
        "--outfile", output_path,
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]
    if face_box is not None:
        y1, y2, x1, x2 = face_box
        cmd += ["--box", str(y1), str(y2), str(x1), str(x2)]

    logger.info("Running Wav2Lip inference")
    _run(cmd, "Wav2Lip inference", log_file=log_file)

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Wav2Lip did not produce an output file at {output_path}. "
            f"Check the log at {log_file} for details."
        )

    logger.info("Wav2Lip finished: %s", output_path)
    return output_path
