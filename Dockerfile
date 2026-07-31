FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# torch and numpy (OpenBLAS) can each bring their own OpenMP runtime into the same
# process; left uncapped this is a well-known source of silent segfaults (no Python
# traceback) after a handful of inferences in Docker. Forcing 1 thread was the
# original, most conservative mitigation, but it also capped EasyOCR's CPU
# throughput hard — and a segfault still happened even with it in place (root cause
# traced instead to ela_detector.py's per-crop JPEG round-trip relying on the
# garbage collector for cleanup — fixed separately), so the safety this buys is
# limited. Now that start.sh auto-restarts the API on a crash, trading some of that
# margin for real speed (this host has 4 cores; leave one free for Streamlit/the OS)
# is a reasonable bet.
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV KMP_DUPLICATE_LIB_OK=TRUE
# transformers' Rust tokenizers (used by CLIPProcessor) warn/deadlock about
# parallelism once a second process/thread appears in the container (here,
# the Streamlit frontend running alongside uvicorn).
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache EasyOCR's detection/recognition model weights at build time
# (network access is available here) so the container never needs internet access or
# a slow first-request download at runtime.
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'])"

COPY backend ./backend
COPY frontend ./frontend
COPY main.py ./main.py
COPY start.sh ./start.sh

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]