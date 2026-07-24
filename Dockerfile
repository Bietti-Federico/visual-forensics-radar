FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# torch, numpy (OpenBLAS) and tesseract/leptonica can each bring their own OpenMP
# runtime into the same process; left uncapped this is a well-known source of
# silent segfaults (no Python traceback) after a handful of inferences in Docker.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
# transformers' Rust tokenizers (used by CLIPProcessor) warn/deadlock about
# parallelism once a second process/thread appears in the container (here,
# the Streamlit frontend running alongside uvicorn).
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY main.py ./main.py
COPY start.sh ./start.sh

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]