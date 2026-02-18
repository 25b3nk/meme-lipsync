"""Stage 2: Text-to-speech using Edge TTS → 16 kHz mono WAV."""

import asyncio
import logging
import os
import subprocess
import tempfile

import edge_tts

from backend.config import DEFAULT_TTS_VOICE

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


def _get_audio_duration(wav_path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        wav_path,
    ]
    result = _run(cmd, "ffprobe (audio duration)")
    return float(result.stdout.strip())


async def _synthesise(text: str, voice: str, output_path: str) -> None:
    """Run Edge TTS and save the raw audio to output_path."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_speech(
    text: str,
    output_wav_path: str,
    voice: str | None = None,
) -> float:
    """
    Convert text to a 16 kHz mono WAV file using Edge TTS.

    Parameters
    ----------
    text : str
        The text to synthesise.
    output_wav_path : str
        Destination path for the final 16 kHz mono WAV.
    voice : str | None
        Edge TTS voice name; defaults to config DEFAULT_TTS_VOICE.

    Returns
    -------
    float
        Audio duration in seconds.
    """
    if not text or not text.strip():
        raise ValueError("Text for TTS cannot be empty.")

    voice = voice or DEFAULT_TTS_VOICE

    # Edge TTS outputs MP3 by default; write to a temp file first
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        raw_audio_path = tmp.name

    try:
        logger.info("Synthesising speech with voice '%s'", voice)
        asyncio.run(_synthesise(text, voice, raw_audio_path))

        if not os.path.exists(raw_audio_path) or os.path.getsize(raw_audio_path) == 0:
            raise RuntimeError("Edge TTS produced an empty audio file.")

        # Convert to 16 kHz mono WAV (required by Wav2Lip)
        logger.info("Converting TTS output to 16 kHz mono WAV")
        os.makedirs(os.path.dirname(output_wav_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_audio_path,
            "-ar", "16000",
            "-ac", "1",
            output_wav_path,
        ]
        _run(cmd, "FFmpeg MP3→WAV conversion")

        duration = _get_audio_duration(output_wav_path)
        logger.info("TTS audio duration: %.2f s", duration)
        return duration

    finally:
        if os.path.exists(raw_audio_path):
            os.unlink(raw_audio_path)
