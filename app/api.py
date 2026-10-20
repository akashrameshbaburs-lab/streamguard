"""StreamGuard API + dashboard host.

Endpoints
---------
GET  /healthz                        liveness — always 200 if the process is up
GET  /healthz/ready                  readiness — checks Postgres AND RabbitMQ
GET  /api/metrics                    aggregate counters (see below)
GET  /api/alerts?limit=N             most recent alerts
POST /api/alerts/{id}/feedback       analyst TP/FP label -> labeled_feedback
POST /api/retrain                    retrain on synthetic + live labelled data
GET  /api/stream                     SSE: metrics ticks + new alerts
GET  /                               static dashboard (vanilla HTML/CSS/JS)

Why SSE polling instead of Redis pub/sub: this stack has no Redis by design.
The API watches Postgres for alerts newer than the last seen id every second
and pushes both metrics and alerts over a single SSE connection. Simple,
durable (Postgres is already the source of truth), and one fewer moving part.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import AsyncIterator

import pika
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from . import bus, db
from .config import settings
from .model import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("streamguard.api")

app = FastAPI(title="StreamGuard", version="1.0.0",
              description="Real-time fraud & anomaly detection")


@app.on_event("startup")
def _startup() -> None:
    try:
        db.init_db()
        log.info("database schema ready")
    except Exception as exc:
        log.error("startup DB init failed (will retry on first request): %s", exc)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "streamguard-api"}


@app.get("/healthz/ready")
def readiness() -> dict:
    checks: dict[str, str] = {}
    # Postgres
    try:
        row = db.fetchone("SELECT 1 AS ok")
        checks["postgres"] = "ok" if row and row["ok"] == 1 else "degraded"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"
    # RabbitMQ
    try:
        params = pika.URLParameters(settings.AMQP_URL)
        params.socket_timeout = 3
        conn = pika.BlockingConnection(params)
        conn.close()
        checks["rabbitmq"] = "ok"
    except Exception as exc:
        checks["rabbitmq"] = f"error: {exc}"
    ready = all(v == "ok" for v in checks.values())
    if not ready:
        raise HTTPException(status_code=503, detail={"ready": ready, "checks": checks})
    return {"ready": ready, "checks": checks}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@app.get("/api/metrics")
def metrics(window_seconds: int = Query(60, ge=5, le=600)) -> dict:
    totals = db.fetchone("""
        SELECT COUNT(*) AS processed,
               COALESCE(SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END), 0) AS flagged
        FROM scored_events
    """) or {"processed": 0, "flagged": 0}

    alert_stats = db.fetchone("""
        SELECT COUNT(*) AS alerts,
               COALESCE(AVG(detection_delay_ms), 0) AS avg_delay,
               COALESCE(MAX(detection_delay_ms), 0) AS max_delay
        FROM alerts
    """) or {"alerts": 0, "avg_delay": 0, "max_delay": 0}

    p95 = db.fetchone("""
        SELECT COALESCE(
            percentile_cont(0.95) WITHIN GROUP (ORDER BY detection_delay_ms), 0
        ) AS p95 FROM alerts
    """) or {"p95": 0}

    window = db.fetchone("""
        SELECT COUNT(*) AS n,
               COALESCE(SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END), 0) AS flagged
        FROM scored_events
        WHERE received_at > now() - make_interval(secs => %s)
    """, (window_seconds,)) or {"n": 0, "flagged": 0}

    model_row = db.fetchone("""
        SELECT version, metrics FROM model_versions ORDER BY version DESC LIMIT 1
    """) or {"version": 0, "metrics": {}}
    model_metrics = model_row["metrics"] if isinstance(model_row["metrics"], dict) else json.loads(
        model_row["metrics"] or "{}")

    processed = int(totals["processed"])
    alerts = int(alert_stats["alerts"])
    fraud_rate = (int(totals["flagged"]) / processed) if processed else 0.0

    return {
        "eventsProcessed": processed,
        "flaggedEvents": int(totals["flagged"]),
        "fraudRate": round(fraud_rate, 6),
        "alertsCount": alerts,
        "avgDetectionDelayMs": round(float(alert_stats["avg_delay"]), 1),
        "maxDetectionDelayMs": round(float(alert_stats["max_delay"]), 1),
        "p95DetectionDelayMs": round(float(p95["p95"]), 1),
        "eventsPerSec": round(int(window["n"]) / window_seconds, 2),
        "windowSeconds": window_seconds,
        "alertsLastWindow": int(window["flagged"]),
        "modelVersion": int(model_row["version"]),
        "modelPrimary": model_metrics.get("primary_model"),
        "modelF1": max(
            model_metrics.get("isolation_forest", {}).get("f1", 0.0),
            model_metrics.get("logistic_regression", {}).get("f1", 0.0),
        ),
        "threshold": settings.ALERT_THRESHOLD,
        "serverTime": time.time(),
    }


@app.get("/api/model/metrics")
def model_metrics() -> dict:
    """Full metrics of the newest trained model (precision/recall/F1 per model)."""
    row = db.fetchone("SELECT version, trained_at, metrics FROM model_versions ORDER BY version DESC LIMIT 1")
    if row is None:
        raise HTTPException(status_code=404, detail="no model trained yet")
    metrics = row["metrics"] if isinstance(row["metrics"], dict) else json.loads(row["metrics"] or "{}")
    return {
        "modelVersion": row["version"],
        "trainedAt": row["trained_at"].isoformat() if row["trained_at"] else None,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Alerts + feedback
# ---------------------------------------------------------------------------

class FeedbackBody(BaseModel):
    label: str

    @field_validator("label")
    @classmethod
    def _check(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("true", "false"):
            raise ValueError('label must be "true" (true positive) or "false" (false positive)')
        return v


@app.get("/api/alerts")
def list_alerts(limit: int = Query(25, ge=1, le=500)) -> dict:
    rows = db.fetchall("""
        SELECT a.id, a.event_id, a.user_id, a.amount, a.country, a.risk,
               a.ml_score, a.rule_score, a.reasons, a.model_version,
               a.detection_delay_ms, a.created_at, a.status,
               f.label AS feedback
        FROM alerts a
        LEFT JOIN labeled_feedback f ON f.alert_id = a.id
        ORDER BY a.id DESC
        LIMIT %s
    """, (limit,))
    alerts = []
    for r in rows:
        reasons = r["reasons"] if isinstance(r["reasons"], list) else json.loads(r["reasons"] or "[]")
        alerts.append({
            "id": r["id"],
            "eventId": r["event_id"],
            "userId": r["user_id"],
            "amount": float(r["amount"]),
            "country": r["country"],
            "risk": round(float(r["risk"]), 4),
            "mlScore": round(float(r["ml_score"] or 0), 4),
            "ruleScore": round(float(r["rule_score"] or 0), 4),
            "reasons": reasons,
            "modelVersion": r["model_version"],
            "detectionDelayMs": round(float(r["detection_delay_ms"] or 0), 1),
            "createdAt": r["created_at"].isoformat() if r["created_at"] else None,
            "status": r["status"],
            "feedback": ("true" if r["feedback"] else "false") if r["feedback"] is not None else None,
        })
    return {"alerts": alerts, "count": len(alerts)}


@app.post("/api/alerts/{alert_id}/feedback")
def submit_feedback(alert_id: int, body: FeedbackBody) -> dict:
    row = db.fetchone("SELECT id, event_id FROM alerts WHERE id = %s", (alert_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="alert not found")
    label = body.label == "true"
    db.execute("""
        INSERT INTO labeled_feedback (alert_id, event_id, label)
        VALUES (%s, %s, %s)
        ON CONFLICT (alert_id) DO UPDATE SET label = EXCLUDED.label,
                                              created_at = now()
    """, (alert_id, row["event_id"], label))
    return {"alertId": alert_id, "label": body.label,
            "recorded": True,
            "note": "feedback folds into the next retrain" if label else
                    "false-positive feedback folds into the next retrain"}


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------

_retrain_lock = asyncio.Lock()


@app.post("/api/retrain")
async def retrain() -> dict:
    """Retrain on synthetic data + live labelled rows (ground truth + feedback).

    Runs synchronously (a few tens of seconds); guarded by a lock so two
    concurrent requests cannot race on the model artifact.
    """
    if _retrain_lock.locked():
        raise HTTPException(status_code=409, detail="a retrain is already running")
    async with _retrain_lock:
        t0 = time.perf_counter()
        extra_X, extra_y = await asyncio.to_thread(_live_labelled_features)
        metrics = await asyncio.to_thread(
            train, None, None, None, "retrain via API", extra_X, extra_y
        )
        elapsed = time.perf_counter() - t0
        return {
            "modelVersion": metrics["model_version"],
            "primaryModel": metrics["primary_model"],
            "isolationForest": metrics["isolation_forest"],
            "logisticRegression": metrics["logistic_regression"],
            "trainRows": metrics["train_rows"],
            "testRows": metrics["test_rows"],
            "liveRowsUsed": len(extra_y),
            "durationSeconds": round(elapsed, 1),
        }


def _live_labelled_features() -> tuple[list[list[float]], list[int]]:
    """Load scored events with labels (feedback overrides ground truth)."""
    rows = db.fetchall("""
        SELECT s.features,
               COALESCE(f.label, g.is_fraud) AS label
        FROM scored_events s
        JOIN ground_truth g ON g.event_id = s.event_id
        LEFT JOIN labeled_feedback f ON f.event_id = s.event_id
        WHERE s.features IS NOT NULL
        ORDER BY s.ts
        LIMIT 100000
    """)
    X, y = [], []
    for r in rows:
        feats = r["features"] if isinstance(r["features"], list) else json.loads(r["features"])
        X.append([float(v) for v in feats])
        y.append(1 if r["label"] else 0)
    log.info("loaded %d live labelled rows (%d fraud)", len(y), sum(y))
    return X, y


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

@app.get("/api/stream")
async def stream(request: Request, window_seconds: int = Query(60, ge=5, le=600)):
    async def event_generator() -> AsyncIterator[str]:
        last_alert_id = 0
        row = db.fetchone("SELECT COALESCE(MAX(id), 0) AS m FROM alerts")
        last_alert_id = int(row["m"]) if row else 0
        tick = 0
        yield "retry: 2000\n\n"
        while True:
            if await request.is_disconnected():
                break
            tick += 1
            try:
                m = await asyncio.to_thread(_metrics_snapshot, window_seconds)
                yield _sse("metrics", m)
                new_alerts = await asyncio.to_thread(_new_alerts_since, last_alert_id)
                for alert in new_alerts:
                    last_alert_id = max(last_alert_id, alert["id"])
                    yield _sse("alert", alert)
                if tick % 15 == 0:
                    yield ": keepalive\n\n"
            except Exception as exc:
                log.warning("SSE tick failed: %s", exc)
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _metrics_snapshot(window_seconds: int) -> dict:
    # reuse the sync endpoint logic
    return metrics(window_seconds)


def _new_alerts_since(after_id: int) -> list[dict]:
    rows = db.fetchall("""
        SELECT id, event_id, user_id, amount, country, risk, ml_score, rule_score,
               reasons, model_version, detection_delay_ms, created_at, status
        FROM alerts WHERE id > %s ORDER BY id ASC LIMIT 50
    """, (after_id,))
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "eventId": r["event_id"],
            "userId": r["user_id"],
            "amount": float(r["amount"]),
            "country": r["country"],
            "risk": round(float(r["risk"]), 4),
            "mlScore": round(float(r["ml_score"] or 0), 4),
            "ruleScore": round(float(r["rule_score"] or 0), 4),
            "reasons": r["reasons"] if isinstance(r["reasons"], list) else json.loads(r["reasons"] or "[]"),
            "modelVersion": r["model_version"],
            "detectionDelayMs": round(float(r["detection_delay_ms"] or 0), 1),
            "createdAt": r["created_at"].isoformat() if r["created_at"] else None,
            "status": r["status"],
        })
    return out


# ---------------------------------------------------------------------------
# Static dashboard (mounted last so /api/* wins)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
