"""Alert notifier.

Consumes from the ``alerts`` queue, marks the matching ``alerts`` row as
notified, logs the alert, and — if TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are
configured — posts a message to the Telegram Bot API (best effort: any
failure is logged and swallowed, the notifier never crashes on network
errors).

There is deliberately no Redis in this stack: the API's SSE endpoint polls
Postgres for new alerts (see docs/ARCHITECTURE.md "Why no Redis").
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import urllib.error
import urllib.parse
import urllib.request

from . import bus, db
from .config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logging.getLogger("pika").setLevel(logging.WARNING)
log = logging.getLogger("streamguard.notifier")


def send_telegram(text: str) -> bool:
    """Best-effort Telegram message. Returns True on success, never raises."""
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log.warning("telegram send failed (best effort): %s", exc)
        return False


def format_alert(alert: dict) -> str:
    reasons = ", ".join(r["rule"] for r in alert.get("reasons", []))
    return (
        f"🚨 <b>StreamGuard alert</b>\n"
        f"risk={alert['risk']:.2f} user={alert['userId']}\n"
        f"amount={alert['amount']:.2f} {alert.get('country') or '-'}\n"
        f"reasons: {reasons}\n"
        f"latency: {alert.get('detectionDelayMs', 0):.0f} ms"
    )


def mark_notified(alert_id: int) -> None:
    db.execute("UPDATE alerts SET status = 'notified', notified_at = now() WHERE id = %s", (alert_id,))


def main() -> None:
    db.init_db()
    connection = bus.connect()
    channel = connection.channel()
    bus.declare_topology(channel)
    channel.basic_qos(prefetch_count=16)

    delivered = 0

    def on_message(ch, method, properties, body):
        nonlocal delivered
        try:
            alert = json.loads(body)
        except json.JSONDecodeError:
            log.warning("dropping non-JSON alert message")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return
        delivered += 1
        log.info("ALERT delivered id=%s risk=%.2f user=%s reasons=%s",
                 alert.get("id"), alert.get("risk", 0.0), alert.get("userId"),
                 [r["rule"] for r in alert.get("reasons", [])])
        if alert.get("id") is not None:
            try:
                mark_notified(alert["id"])
            except Exception as exc:
                log.warning("could not mark alert %s notified: %s", alert.get("id"), exc)
        send_telegram(format_alert(alert))
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=bus.ALERTS_QUEUE, on_message_callback=on_message)
    log.info("notifier consuming from %s (telegram=%s)",
             bus.ALERTS_QUEUE, "on" if settings.TELEGRAM_BOT_TOKEN else "off")

    delivered = 0
    try:
        # process_data_events returns once idle; loop to keep the consumer alive
        while connection.is_open:
            connection.process_data_events(time_limit=1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            connection.close()
        except Exception:
            pass
        log.info("notifier stopped: delivered=%d", delivered)


if __name__ == "__main__":
    main()
