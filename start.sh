#!/bin/sh

echo "Starting PDF Metadata API..."
(
  while true; do
    uvicorn main:app --host 0.0.0.0 --port 8000
    exit_code=$?
    echo "uvicorn exited (code $exit_code) - restarting in 2s..."
    sleep 2
  done
) &

sleep 2

echo "Starting Streamlit Frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
