FROM python:3.10-slim

WORKDIR /app

# Install PyTorch CPU-only first (largest layer — cache before apt changes)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install wav2lip runtime dependencies (compatible versions for Python 3.10)
RUN pip install --no-cache-dir librosa==0.9.2 scipy tqdm

# Copy and install backend Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# System dependencies (after pip so changing apt doesn't bust pip cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gifsicle \
    git \
    libgl1 \
    libglib2.0-0 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the source
COPY . /app

# Make wav2lip importable if the repo has been cloned
ENV PYTHONPATH="/app:/app/wav2lip"

# Default: run the API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
