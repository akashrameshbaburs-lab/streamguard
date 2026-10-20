# StreamGuard — single image reused by all Python services:
#   api, consumer (scorer), notifier, simulator, trainer
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY static ./static

# Default command is the API; docker-compose overrides per service.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "3300"]
