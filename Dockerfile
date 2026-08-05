FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 100 --retries 5 -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY main.py ./main.py
COPY start.sh ./start.sh

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]
