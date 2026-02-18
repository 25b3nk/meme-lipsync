"""Configuration loaded from environment variables with sensible defaults."""

import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
TEMP_DIR = os.environ.get("TEMP_DIR", "./temp")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/wav2lip_gan.pth")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
DEFAULT_TTS_VOICE = os.environ.get("DEFAULT_TTS_VOICE", "en-US-GuyNeural")

# Wav2Lip repo path (relative to project root)
WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "./wav2lip")

# Ensure working directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
