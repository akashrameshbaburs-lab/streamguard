"""PostgreSQL access helpers and schema management.

Every service calls ``init_db()`` on startup; it is idempotent (CREATE TABLE IF
NOT EXISTS) so concurrent starts are safe.
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS ground_truth (
    event_id   TEXT PRIMARY KEY,
    is_fraud   BOOLEAN NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    pattern    TEXT NOT NULL DEFAULT 'normal'  -- evaluation-only annotation (never read by the scorer)
);
ALTER TABLE ground_truth ADD COLUMN IF NOT EXISTS pattern TEXT NOT NULL DEFAULT 'normal';

CREATE TABLE IF NOT EXISTS scored_events (
    event_id    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    amount      DOUBLE PRECISION NOT NULL,
    merchant_id TEXT,
    country     TEXT,
    device_fp   TEXT,
    ip          TEXT,
    ts          TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ml_score    DOUBLE PRECISION,
    rule_score  DOUBLE PRECISION,
    risk        DOUBLE PRECISION,
    is_flagged  BOOLEAN NOT NULL DEFAULT FALSE,
    features    JSONB
);

CREATE INDEX IF NOT EXISTS idx_scored_events_ts ON scored_events (ts);
CREATE INDEX IF NOT EXISTS idx_scored_events_received ON scored_events (received_at);

CREATE TABLE IF NOT EXISTS alerts (
    id                 BIGSERIAL PRIMARY KEY,
    event_id           TEXT NOT NULL UNIQUE,
    user_id            TEXT NOT NULL,
    amount             DOUBLE PRECISION NOT NULL,
    merchant_id        TEXT,
    country            TEXT,
    device_fp          TEXT,
    ip                 TEXT,
    risk               DOUBLE PRECISION NOT NULL,
    ml_score           DOUBLE PRECISION,
    rule_score         DOUBLE PRECISION,
    reasons            JSONB NOT NULL,
    model_version      INTEGER NOT NULL,
    event_ts           TIMESTAMPTZ NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    detection_delay_ms DOUBLE PRECISION,
    status             TEXT NOT NULL DEFAULT 'open',
    notified_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts (created_at);

CREATE TABLE IF NOT EXISTS labeled_feedback (
    id         BIGSERIAL PRIMARY KEY,
    alert_id   BIGINT NOT NULL UNIQUE REFERENCES alerts (id) ON DELETE CASCADE,
    event_id   TEXT NOT NULL,
    label      BOOLEAN NOT NULL,   -- true = true positive, false = false positive
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    version    INTEGER PRIMARY KEY,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metrics    JSONB NOT NULL,
    notes      TEXT
);
"""


def connect() -> psycopg.Connection:
    """Open a dict-row connection using the configured DSN."""
    return psycopg.connect(settings.PG_DSN, row_factory=dict_row, connect_timeout=5)


@contextlib.contextmanager
def cursor() -> Iterator[psycopg.Cursor]:
    """Context manager yielding a dict-row cursor; commits on success."""
    with connect() as conn:
        with conn.cursor() as cur:
            yield cur
            conn.commit()


def init_db() -> None:
    with cursor() as cur:
        cur.execute(SCHEMA)


def fetchall(query: str, params: Any = None) -> list[dict]:
    with cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def fetchone(query: str, params: Any = None) -> dict | None:
    with cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def execute(query: str, params: Any = None) -> None:
    with cursor() as cur:
        cur.execute(query, params)


def execute_many(query: str, seq: list[tuple]) -> None:
    with cursor() as cur:
        cur.executemany(query, seq)
