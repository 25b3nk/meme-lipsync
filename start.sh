#!/usr/bin/env bash
# start.sh — launch all services locally using the venv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "ERROR: venv not found. Run: python3 -m venv .venv && pip install ..." >&2
  exit 1
fi

# Kill background children on exit
cleanup() {
  echo "Stopping services..."
  kill "$REDIS_PID" "$CELERY_PID" "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting Redis"
redis-server --daemonize no --loglevel warning &
REDIS_PID=$!
sleep 1

echo "==> Starting Celery worker"
source "$VENV"
PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/wav2lip" \
  celery -A backend.tasks worker --loglevel=info --concurrency=1 &
CELERY_PID=$!
sleep 2

echo "==> Starting API server"
PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/wav2lip" \
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo ""
echo "All services running. Press Ctrl+C to stop."
echo "  API:    http://localhost:8000"
echo "  Health: http://localhost:8000/health"
echo ""
wait
