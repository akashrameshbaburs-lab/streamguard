"""RabbitMQ helpers: connection, topology declaration, publish helpers.

Topology (declared by every service that touches the bus, idempotently):

    exchange "events"  (direct)  ──▶ queue "raw_events"   (rk: payment)
    exchange "alerts"  (direct)  ──▶ queue "alerts"       (rk: alert)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import pika

from .config import settings

log = logging.getLogger("streamguard.bus")

EVENTS_EXCHANGE = "events"
EVENTS_QUEUE = "raw_events"
EVENTS_ROUTING_KEY = "payment"

ALERTS_EXCHANGE = "alerts"
ALERTS_QUEUE = "alerts"
ALERTS_ROUTING_KEY = "alert"


def connect(retries: int = 30, delay: float = 2.0) -> pika.BlockingConnection:
    """Connect to RabbitMQ with retries (services may start before the bus is up)."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            params = pika.URLParameters(settings.AMQP_URL)
            params.socket_timeout = 5
            params.blocked_connection_timeout = 5
            return pika.BlockingConnection(params)
        except Exception as exc:  # pragma: no cover - startup race
            last_exc = exc
            log.warning("RabbitMQ connect attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"could not connect to RabbitMQ: {last_exc}")


def declare_topology(channel) -> None:
    """Declare both exchanges and queues with bindings (idempotent)."""
    channel.exchange_declare(exchange=EVENTS_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=EVENTS_QUEUE, durable=True)
    channel.queue_bind(queue=EVENTS_QUEUE, exchange=EVENTS_EXCHANGE, routing_key=EVENTS_ROUTING_KEY)

    channel.exchange_declare(exchange=ALERTS_EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=ALERTS_QUEUE, durable=True)
    channel.queue_bind(queue=ALERTS_QUEUE, exchange=ALERTS_EXCHANGE, routing_key=ALERTS_ROUTING_KEY)


def publish_json(channel, exchange: str, routing_key: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
        body=body,
        properties=pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,  # persistent
        ),
    )
