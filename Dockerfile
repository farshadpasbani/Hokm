FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements-prod.txt .
# CPU-only torch keeps the image ~1 GB smaller than the CUDA default.
RUN pip install --no-cache-dir -r requirements-prod.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

COPY game_constants.py hokm.py enhanced_player.py baselines.py config.py \
     seed_utils.py telegram_auth.py game_service.py server.py ./
COPY static/ static/
COPY templates/miniapp.html templates/miniapp.html
COPY scripts/ scripts/
# Release checkpoint (if present): AI seats use it when MODEL_PATH points here.
COPY models_release/ models_release/

RUN useradd --create-home hokm && chown -R hokm:hokm /app
USER hokm

EXPOSE 8080

# Exactly ONE worker: game sessions live in process memory. Concurrency
# comes from threads; scale-out requires a shared session store first.
CMD exec gunicorn --workers 1 --threads 8 --timeout 60 \
    --bind 0.0.0.0:${PORT} server:app
