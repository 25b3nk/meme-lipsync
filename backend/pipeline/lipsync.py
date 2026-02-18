"""Stage 3: Wav2Lip inference wrapper."""

import logging
import os
import subprocess

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


def _pad_audio_with_silence(wav_path: str, target_duration: float, output_path: str) -> None:
    """Pad a WAV file with trailing silence to match target_duration."""
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-af", f"apad",
        "-t", str(target_duration),
        output_path,
    ]
    _run(cmd, "Audio silence padding")


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

    # Reject if audio is far longer than video (loop threshold = 1.5×)
    if video_duration > 0 and audio_duration > video_duration * 1.5:
        raise ValueError(
            "Text too long for this meme — try shorter text. "
            f"Audio is {audio_duration:.1f}s but video is only {video_duration:.1f}s."
        )

    # Pad audio with silence if it is shorter than the video
    effective_audio_path = audio_path
    if video_duration > 0 and audio_duration < video_duration:
        logger.info("Padding audio to match video duration (%.2f s)", video_duration)
        padded_wav = os.path.join(job_dir, "audio_padded.wav")
        _pad_audio_with_silence(audio_path, video_duration, padded_wav)
        effective_audio_path = padded_wav

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
        "python", inference_script,
        "--checkpoint_path", MODEL_PATH,
        "--face", video_path,
        "--audio", effective_audio_path,
        "--outfile", output_path,
        "--pads", "0", "10", "0", "0",
        "--resize_factor", "1",
        "--nosmooth",
    ]

    logger.info("Running Wav2Lip inference")
    _run(cmd, "Wav2Lip inference", log_file=log_file)

    if not os.path.exists(output_path):
        raise RuntimeError(
            f"Wav2Lip did not produce an output file at {output_path}. "
            f"Check the log at {log_file} for details."
        )

    logger.info("Wav2Lip finished: %s", output_path)
    return output_path
