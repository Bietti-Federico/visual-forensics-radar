#!/bin/sh

echo "Starting Deepfake API Server (Loading models into VRAM)..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "Warming up AI engines... Please wait."
sleep 15

echo "Starting Streamlit Frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0