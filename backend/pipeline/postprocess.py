"""Stage 4: Convert lip-synced MP4 back to an optimised GIF."""

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess and raise on non-zero exit."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def convert_to_gif(
    mp4_path: str,
    output_gif_path: str,
    original_fps: float,
) -> str:
    """
    Convert a lip-synced MP4 to an optimised GIF using a two-pass FFmpeg
    palette approach followed by Gifsicle compression.

    Parameters
    ----------
    mp4_path : str
        Path to the Wav2Lip output MP4.
    output_gif_path : str
        Destination path for the final optimised GIF.
    original_fps : float
        Frame rate to use for the output GIF (from the original upload).

    Returns
    -------
    str
        Path to the optimised GIF (same as output_gif_path).
    """
    output_dir = os.path.dirname(output_gif_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    fps = min(original_fps, 30.0)  # cap at 30 fps for reasonable GIF size

    palette_path = os.path.join(output_dir, "_palette.png")
    raw_gif_path = os.path.join(output_dir, "_raw.gif")

    # ── Pass 1: Generate a palette optimised for this video ──────────────────
    logger.info("GIF pass 1: generating palette (fps=%.2f)", fps)
    _run(
        [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-vf", (
                f"fps={fps},"
                "scale=480:-1:flags=lanczos,"
                "palettegen=stats_mode=diff"
            ),
            palette_path,
        ],
        "FFmpeg palette generation",
    )

    # ── Pass 2: Render GIF using that palette ────────────────────────────────
    logger.info("GIF pass 2: rendering GIF with palette")
    _run(
        [
            "ffmpeg", "-y",
            "-i", mp4_path,
            "-i", palette_path,
            "-lavfi", (
                f"fps={fps},"
                "scale=480:-1:flags=lanczos [x]; [x][1:v] "
                "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
            ),
            raw_gif_path,
        ],
        "FFmpeg GIF rendering",
    )

    # ── Gifsicle optimisation ────────────────────────────────────────────────
    if shutil.which("gifsicle"):
        logger.info("Running Gifsicle optimisation")
        _run(
            [
                "gifsicle",
                "-O3",
                "--lossy=80",
                raw_gif_path,
                "-o", output_gif_path,
            ],
            "Gifsicle optimisation",
        )
        os.unlink(raw_gif_path)
    else:
        logger.warning("gifsicle not found — skipping optimisation step")
        os.rename(raw_gif_path, output_gif_path)

    # Cleanup palette
    if os.path.exists(palette_path):
        os.unlink(palette_path)

    logger.info("GIF written to: %s", output_gif_path)
    return output_gif_path
