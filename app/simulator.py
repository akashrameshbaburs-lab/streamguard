"""Payment-event simulator (producer).

Publishes synthetic payment events to RabbitMQ exchange ``events`` (routing
key ``payment``) at ``EVENTS_PER_SEC`` and writes the GROUND-TRUTH label for
every event to Postgres table ``ground_truth`` — the scorer never reads this
table; it exists purely for offline evaluation (precision/recall) and for
folding labelled data into retrains.

Usage:
    python -m app.simulator                       # steady state (EVENTS_PER_SEC)
    python -m app.simulator --count 5000 --fast   # burst load, no rate limiting
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pika

from . import bus, db
from .config import settings
from .synth import EventFactory, build_user_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logging.getLogger("pika").setLevel(logging.WARNING)
log = logging.getLogger("streamguard.simulator")

BATCH_SIZE = 500  # ground-truth rows flushed per INSERT


class GroundTruthWriter:
    """Buffers ground-truth labels and flushes them in batches."""

    def __init__(self) -> None:
        self._buf: list[tuple[str, bool, float, str]] = []
        self._written = 0

    def add(self, event_id: str, is_fraud: bool, ts: float, pattern: str = "normal") -> None:
        self._buf.append((event_id, is_fraud, ts, pattern))
        if len(self._buf) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        rows, self._buf = self._buf, []
        try:
            db.execute_many(
                "INSERT INTO ground_truth (event_id, is_fraud, ts, pattern) "
                "VALUES (%s, %s, to_timestamp(%s), %s) ON CONFLICT (event_id) DO NOTHING",
                rows,
            )
            self._written += len(rows)
        except Exception as exc:
            log.warning("ground-truth flush failed (%d rows buffered again): %s", len(rows), exc)
            self._buf = rows + self._buf

    @property
    def written(self) -> int:
        return self._written + len(self._buf)


def publish_loop(count: int | None, fast: bool) -> None:
    db.init_db()
    connection = bus.connect()
    channel = connection.channel()
    bus.declare_topology(channel)

    rate = settings.EVENTS_PER_SEC
    # Size the benign user pool so per-user velocity stays well below the
    # 10 tx/60 s fraud rule under the configured rate (target <= ~2 tx/60 s).
    # A burst uses a large DISJOINT pool so compressed events don't overload
    # any one benign user and trip the velocity rule — that would be a
    # load-test artifact, not real fraud.
    if fast and count:
        pool_size = max(6000, count * 2)
        id_offset = 500_000
    else:
        pool_size = max(600, int(rate * 40))
        id_offset = 0
    users = build_user_pool(pool_size, id_offset=id_offset)
    factory = EventFactory(
        users,
        fraud_rate=settings.FRAUD_RATE,
        seed=42,
        id_prefix="live",
        # In a fast burst the wall clock never reaches the scheduled due-times
        # of multi-event fraud patterns (velocity legs, travel legs), so emit
        # them as soon as they are scheduled — otherwise a burst under-produces
        # fraud and the recall measurement is meaningless.
        flush_immediately=fast and bool(count),
    )
    truth = GroundTruthWriter()

    interval = 1.0 / rate if rate > 0 else 0.0
    log.info("simulator starting: count=%s fast=%s rate=%d eps pool=%d offset=%d fraud_rate=%.3f",
             count or "∞", fast, rate, pool_size, id_offset, settings.FRAUD_RATE)

    published = 0
    fraud_published = 0
    next_ts = time.time()

    def publish(event) -> None:
        nonlocal published, fraud_published
        channel.basic_publish(
            exchange=bus.EVENTS_EXCHANGE,
            routing_key=bus.EVENTS_ROUTING_KEY,
            body=json.dumps(event.to_payload()).encode(),
            properties=pika.BasicProperties(
                content_type="application/json", delivery_mode=2,
            ),
        )
        truth.add(event.event_id, event.is_fraud, event.ts, event.pattern)
        published += 1
        if event.is_fraud:
            fraud_published += 1

    try:
        while count is None or published < count:
            now = time.time()
            if not fast and interval:
                # pace to the configured rate
                if now < next_ts:
                    time.sleep(min(next_ts - now, 0.05))
                    now = time.time()
                next_ts += interval
                if next_ts < now - 5:  # fell way behind (GC pause etc.) — resync
                    next_ts = now

            event = factory.next_event(now)
            publish(event)

            if published % 1000 == 0:
                log.info("published %d events (fraud so far=%d)", published, fraud_published)
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        # Complete any multi-event attack that is still mid-pattern, so the
        # user's state (e.g. last country) ends where the attack left it.
        leftover = factory.drain_pending()
        if leftover:
            done = 0
            try:
                for ev in leftover:
                    publish(ev)
                    done += 1
            except Exception as exc:
                log.warning("shutdown drain stopped after %d/%d pending legs: %s",
                            done, len(leftover), exc)
        truth.flush()
        log.info("published %d events total (fraud=%d); ground truth rows written=%d",
                 published, fraud_published, truth.written)
        try:
            connection.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamGuard payment event simulator")
    parser.add_argument("--count", type=int, default=None,
                        help="publish exactly N events then exit (default: run forever)")
    parser.add_argument("--fast", action="store_true",
                        help="ignore EVENTS_PER_SEC pacing (burst mode for load tests)")
    args = parser.parse_args()
    publish_loop(args.count, args.fast)


if __name__ == "__main__":
    main()
