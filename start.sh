#!/bin/sh

echo "Starting Deepfake API Server (Loading models into VRAM)..."
# Wrapped in a restart loop: an intermittent native segfault (no Python traceback)
# has been observed a few seconds after a request completes — traced to
# ela_detector.py relying on the garbage collector to close per-crop PIL Images/
# file handles instead of closing them deterministically (fixed separately), though
# the thread-limiting env vars above remain as a second line of defense against the
# broader class of native-library conflicts. If uvicorn ever dies anyway, this
# brings the API back up instead of leaving it dead for the rest of the container's
# life. Each restart re-loads every model from scratch (CLIP, EasyOCR), so a crashed
# request still costs ~15-30s of downtime for the next one, but the service
# self-heals instead of staying down.
(
  while true; do
    uvicorn main:app --host 0.0.0.0 --port 8000
    exit_code=$?
    echo "uvicorn exited (code $exit_code) - restarting in 2s..."
    sleep 2
  done
) &

echo "Warming up AI engines... Please wait."
sleep 15

echo "Starting Streamlit Frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0