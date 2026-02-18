FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    gifsicle \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the source
COPY . /app

# Make wav2lip importable if the repo has been cloned
ENV PYTHONPATH="/app:/app/wav2lip"

# Default: run the API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
