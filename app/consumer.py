"""Scoring consumer (the heart of StreamGuard).

Consumes payment events from queue ``raw_events``, enriches them with
per-user features, runs the deterministic rules engine and the ML scorer,
fuses both into a risk score, and:

  * writes every scored event to ``scored_events`` (for metrics windows),
  * when risk >= ALERT_THRESHOLD: inserts an ``alerts`` row (with reasons and
    measured detection latency), then publishes the alert to the ``alerts``
    exchange for the notifier.

The consumer NEVER reads ``ground_truth`` — labels stay on the simulator side
so evaluation is honest.

Model hot-reload: a small polling thread checks ``model_versions`` every
``MODEL_POLL_SECONDS`` and reloads the joblib artifact when the version moves
(e.g. after POST /api/retrain).
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from collections import deque

import psycopg

from . import bus, db
from .config import settings
from .features import FeatureExtractor
from .model import RiskScorer, ensure_model, fuse_risk
from .rules import evaluate as rules_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logging.getLogger("pika").setLevel(logging.WARNING)
log = logging.getLogger("streamguard.consumer")

SCORED_BATCH = 200        # scored_events rows per INSERT batch
FLUSH_INTERVAL = 0.5      # seconds between forced flushes


class ScoredBatches:
    """Buffers scored_events inserts; flushed in batches for throughput."""

    COLUMNS = ("event_id", "user_id", "amount", "merchant_id", "country",
               "device_fp", "ip", "ts", "ml_score", "rule_score", "risk",
               "is_flagged", "features")

    def __init__(self) -> None:
        self._buf: list[tuple] = []
        self._last_flush = time.monotonic()
        self.processed = 0

    def add(self, row: dict) -> None:
        self._buf.append(tuple(row[c] for c in self.COLUMNS))
        self.processed += 1

    def maybe_flush(self, force: bool = False) -> None:
        if not self._buf:
            return
        now = time.monotonic()
        if not force and len(self._buf) < SCORED_BATCH and now - self._last_flush < FLUSH_INTERVAL:
            return
        rows, self._buf = self._buf, []
        self._last_flush = now
        # ts is stored as timestamptz; the buffer carries epoch seconds, so
        # convert with to_timestamp() in the INSERT.
        sql = (
            "INSERT INTO scored_events (event_id, user_id, amount, merchant_id, country, "
            "device_fp, ip, ts, ml_score, rule_score, risk, is_flagged, features) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s),%s,%s,%s,%s,%s::jsonb) "
            "ON CONFLICT (event_id) DO NOTHING"
        )
        try:
            db.execute_many(sql, rows)
        except Exception as exc:
            log.warning("scored_events flush failed, dropping %d rows: %s", len(rows), exc)


class RateMeter:
    """Sliding-window rate meter for events/sec reporting."""

    def __init__(self, window: float = 60.0) -> None:
        self.window = window
        self._hits: deque[float] = deque()

    def hit(self) -> None:
        self._hits.append(time.monotonic())

    def rate(self, last: float = 10.0) -> float:
        cutoff = time.monotonic() - last
        n = sum(1 for t in self._hits if t > cutoff)
        return n / last


def process_event(event: dict, extractor: FeatureExtractor, scorer: RiskScorer,
                  channel, batches: ScoredBatches, meter: RateMeter) -> dict | None:
    """Score one event; insert + publish an alert if over threshold."""
    t0 = time.perf_counter()
    features, context = extractor.extract(event)
    rule_score, reasons = rules_evaluate(context, float(event["amount"]))
    ml_score, ml_details = scorer.score(features)
    risk = fuse_risk(rule_score, ml_score)
    extractor.update(event)
    meter.hit()

    flagged = risk >= settings.ALERT_THRESHOLD
    event_ts = float(event["ts"])
    now = time.time()
    delay_ms = (now - event_ts) * 1000.0

    batches.add({
        "event_id": event["eventId"],
        "user_id": event["userId"],
        "amount": float(event["amount"]),
        "merchant_id": event.get("merchantId"),
        "country": event.get("country"),
        "device_fp": event.get("deviceFp"),
        "ip": event.get("ip"),
        "ts": event_ts,
        "ml_score": ml_score,
        "rule_score": rule_score,
        "risk": risk,
        "is_flagged": flagged,
        "features": json.dumps(features),
    })

    alert = None
    if flagged:
        reason_list = list(reasons)
        if ml_score >= 0.6 and not any(r["rule"] == "ml_anomaly" for r in reason_list):
            reason_list.append({
                "rule": "ml_anomaly",
                "severity": round(ml_score, 3),
                "detail": f"{ml_details.get('model', 'ml')} score {ml_score:.2f}",
            })
        try:
            with db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO alerts (event_id, user_id, amount, merchant_id, country,
                                        device_fp, ip, risk, ml_score, rule_score, reasons,
                                        model_version, event_ts, detection_delay_ms)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,to_timestamp(%s),%s)
                    ON CONFLICT (event_id) DO NOTHING
                    RETURNING id
                    """,
                    (event["eventId"], event["userId"], float(event["amount"]),
                     event.get("merchantId"), event.get("country"), event.get("deviceFp"),
                     event.get("ip"), risk, ml_score, rule_score,
                     json.dumps(reason_list), scorer.version, event_ts, delay_ms),
                )
                row = cur.fetchone()
            if row:
                alert = {
                    "id": row["id"],
                    "eventId": event["eventId"],
                    "userId": event["userId"],
                    "amount": float(event["amount"]),
                    "country": event.get("country"),
                    "risk": round(risk, 4),
                    "mlScore": round(ml_score, 4),
                    "ruleScore": round(rule_score, 4),
                    "reasons": reason_list,
                    "modelVersion": scorer.version,
                    "detectionDelayMs": round(delay_ms, 1),
                    "ts": event_ts,
                }
                bus.publish_json(channel, bus.ALERTS_EXCHANGE, bus.ALERTS_ROUTING_KEY, alert)
        except Exception as exc:
            log.error("failed to record/publish alert for %s: %s", event["eventId"], exc)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if elapsed_ms > 100:
        log.warning("slow scoring for %s: %.1f ms", event["eventId"], elapsed_ms)
    return alert


def model_version_poller(scorer: RiskScorer, stop: threading.Event) -> None:
    """Poll model_versions and hot-reload the scorer when the version changes."""
    while not stop.wait(settings.MODEL_POLL_SECONDS):
        try:
            row = db.fetchone("SELECT MAX(version) AS v FROM model_versions")
            if row and row["v"] is not None:
                scorer.maybe_reload(int(row["v"]))
        except Exception as exc:
            log.debug("model poll failed: %s", exc)


def main() -> None:
    db.init_db()
    info = ensure_model()
    log.info("consumer starting with model %s", info)

    scorer = RiskScorer()
    if not scorer.load():
        log.error("no model available after ensure_model — exiting")
        sys.exit(1)

    extractor = FeatureExtractor()
    batches = ScoredBatches()
    meter = RateMeter()

    stop = threading.Event()
    threading.Thread(target=model_version_poller, args=(scorer, stop), daemon=True).start()

    def _shutdown(signum, frame):
        log.info("signal %d received, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    connection = bus.connect()
    channel = connection.channel()
    bus.declare_topology(channel)
    channel.basic_qos(prefetch_count=64)

    alerts_total = 0
    last_log = time.monotonic()

    def on_message(ch, method, properties, body):
        nonlocal alerts_total
        try:
            event = json.loads(body)
            alert = process_event(event, extractor, scorer, ch, batches, meter)
            if alert:
                alerts_total += 1
                log.info("ALERT #%d risk=%.2f user=%s reasons=%s delay=%.0fms",
                         alerts_total, alert["risk"], alert["userId"],
                         [r["rule"] for r in alert["reasons"]], alert["detectionDelayMs"])
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except json.JSONDecodeError:
            log.warning("dropping non-JSON message")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except psycopg.OperationalError as exc:
            log.error("DB error while scoring: %s (message requeued)", exc)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            time.sleep(1)
        except Exception as exc:
            log.exception("unhandled error while scoring: %s", exc)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=bus.EVENTS_QUEUE, on_message_callback=on_message)
    log.info("consuming from %s (threshold=%.2f, ml_weight=%.2f gamma=%.2f)",
             bus.EVENTS_QUEUE, settings.ALERT_THRESHOLD,
             settings.ML_WEIGHT, settings.FUSION_GAMMA)

    try:
        while not stop.is_set():
            connection.process_data_events(time_limit=1)
            batches.maybe_flush()
            if time.monotonic() - last_log >= 10:
                last_log = time.monotonic()
                q = _queue_depth(connection)
                log.info("stats: processed=%d alerts=%d eps=%.1f queue_depth=%s model=v%d",
                         batches.processed, alerts_total, meter.rate(), q, scorer.version)
    finally:
        batches.maybe_flush(force=True)
        try:
            connection.close()
        except Exception:
            pass
        log.info("consumer stopped: processed=%d alerts=%d", batches.processed, alerts_total)


def _queue_depth(connection) -> int | None:
    try:
        ch = connection.channel()
        res = ch.queue_declare(queue=bus.EVENTS_QUEUE, passive=True)
        return res.method.message_count
    except Exception:
        return None


if __name__ == "__main__":
    main()
