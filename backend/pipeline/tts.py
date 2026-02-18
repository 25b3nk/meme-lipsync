"""Stage 2: Text-to-speech using espeak-ng → 16 kHz mono WAV."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _get_audio_duration(wav_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        wav_path,
    ]
    result = _run(cmd, "ffprobe (audio duration)")
    return float(result.stdout.strip())


def generate_speech(
    text: str,
    output_wav_path: str,
    voice: str | None = None,
) -> float:
    """
    Convert text to a 16 kHz mono WAV using espeak-ng.

    Returns audio duration in seconds.
    """
    if not text or not text.strip():
        raise ValueError("Text for TTS cannot be empty.")

    os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)

    # espeak-ng writes a WAV directly; default sample rate is 22050 Hz
    logger.info("Synthesising speech with espeak-ng")
    raw_wav = output_wav_path + ".raw.wav"
    _run(["espeak-ng", "-w", raw_wav, text], "espeak-ng TTS")

    if not os.path.exists(raw_wav) or os.path.getsize(raw_wav) == 0:
        raise RuntimeError("espeak-ng produced an empty audio file.")

    # Resample to 16 kHz mono (required by Wav2Lip)
    logger.info("Resampling to 16 kHz mono WAV")
    _run([
        "ffmpeg", "-y",
        "-i", raw_wav,
        "-ar", "16000",
        "-ac", "1",
        output_wav_path,
    ], "FFmpeg WAV resample")

    os.unlink(raw_wav)

    duration = _get_audio_duration(output_wav_path)
    logger.info("TTS audio duration: %.2f s", duration)
    return duration
