"""Central configuration, read once from environment variables.

All services share this module; each service reads only the keys it needs.
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    # --- infrastructure -----------------------------------------------------
    PG_DSN = os.environ.get(
        "PG_DSN", "postgresql://streamguard:streamguard@postgres:5432/streamguard"
    )
    AMQP_URL = os.environ.get(
        "AMQP_URL", "amqp://streamguard:streamguard@rabbitmq:5672/"
    )

    # --- streaming / scoring ------------------------------------------------
    EVENTS_PER_SEC = _i("EVENTS_PER_SEC", 20)
    # Threshold sweep on 21.5k scored events: F1 peaks flat between 0.72-0.78;
    # 0.75 maximizes precision near the F1 peak (fewer false alerts to review).
    ALERT_THRESHOLD = _f("ALERT_THRESHOLD", 0.75)
    # Fusion: risk = clip(max(rule, ML_WEIGHT*ml) + FUSION_GAMMA*min(rule, ml), 0, 1)
    # ML_WEIGHT < 1 discounts the (noisier) model score; FUSION_GAMMA is a
    # corroboration bonus when both engines agree. See app/model.py::fuse_risk.
    ML_WEIGHT = _f("ML_WEIGHT", 0.9)
    FUSION_GAMMA = _f("FUSION_GAMMA", 0.15)
    MODEL_POLL_SECONDS = _f("MODEL_POLL_SECONDS", 3.0)

    # --- model artefacts ----------------------------------------------------
    MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, "model.joblib"))
    METRICS_PATH = os.environ.get(
        "METRICS_PATH", os.path.join(MODEL_DIR, "metrics.json")
    )
    TRAIN_ROWS = _i("TRAIN_ROWS", 50000)
    RANDOM_SEED = _i("RANDOM_SEED", 42)

    # --- notifications (optional) -------------------------------------------
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

    # --- misc ---------------------------------------------------------------
    FRAUD_RATE = _f("FRAUD_RATE", 0.025)


settings = Settings()
