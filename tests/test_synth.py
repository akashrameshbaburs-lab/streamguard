"""Unit tests for synthetic event generation (``app/synth.py``)."""
from __future__ import annotations

import collections

import pytest

from app.synth import (
    COUNTRY_RISK,
    MERCHANT_POOL,
    EventFactory,
    SynthEvent,
    build_user_pool,
    country_risk,
)

FRAUD_PATTERNS = {
    "velocity_burst",
    "amount_spike",
    "new_device_amount",
    "impossible_travel",
}
BENIGN_PATTERNS = {"normal", "hard_negative"}
BASE_TS = 1_700_000_000.0


def generate(seed: int, n: int = 300) -> list[SynthEvent]:
    """Run one factory for ``n`` events on a compressed simulated clock."""
    users = build_user_pool(200, id_offset=0)
    factory = EventFactory(users, fraud_rate=0.03, seed=seed, id_prefix="t")
    events: list[SynthEvent] = []
    ts = BASE_TS
    for _ in range(n):
        ts += 0.25
        events.append(factory.next_event(ts))
    return events


def comparable(events):
    """Wire-visible fields; ``event_id`` embeds a per-process uniqueness token."""
    return [
        (
            event.user_id,
            event.amount,
            event.merchant_id,
            event.country,
            event.device_fp,
            event.ip,
            event.ts,
            event.is_fraud,
            event.pattern,
        )
        for event in events
    ]


@pytest.fixture(scope="module")
def sample() -> list[SynthEvent]:
    """6000-event stream shaped like ``generate_training_frame``'s replay."""
    users = build_user_pool(600, id_offset=0)
    factory = EventFactory(users, fraud_rate=0.03, seed=42, id_prefix="t")
    events: list[SynthEvent] = []
    ts = BASE_TS
    chunk = 6000 // 24
    for index in range(6000):
        ts += 1.0 / 12.0
        if index and index % chunk == 0:
            ts += 3600.0  # jump an hour so the frame spans a whole day
        events.append(factory.next_event(ts))
    return events


def test_fixed_seed_produces_identical_rows():
    first = generate(seed=42)
    second = generate(seed=42)

    assert len(first) == len(second) == 300
    assert comparable(first) == comparable(second)
    # ...and even the deterministic counter suffix of the ids agrees; only the
    # per-process uniqueness token (which prevents cross-run id collisions)
    # may differ.
    assert [e.event_id.split("-")[2] for e in first] == [
        e.event_id.split("-")[2] for e in second
    ]
    assert first[0].event_id.split("-")[1] != second[0].event_id.split("-")[1]


def test_different_seed_produces_different_rows():
    assert comparable(generate(seed=42)) != comparable(generate(seed=43))


def test_fraud_rate_is_approximately_honoured(sample):
    fraud_rate = sum(event.is_fraud for event in sample) / len(sample)
    assert abs(fraud_rate - 0.03) < 0.01


def test_labels_match_injected_patterns(sample):
    seen = {event.pattern for event in sample}
    assert seen <= FRAUD_PATTERNS | BENIGN_PATTERNS
    assert FRAUD_PATTERNS <= seen          # every injected pattern materialises
    assert all(e.is_fraud for e in sample if e.pattern in FRAUD_PATTERNS)
    assert not any(e.is_fraud for e in sample if e.pattern in BENIGN_PATTERNS)


def test_velocity_bursts_cluster_in_the_60s_window(sample):
    by_user: dict[str, list[float]] = collections.defaultdict(list)
    for event in sample:
        if event.pattern == "velocity_burst":
            by_user[event.user_id].append(event.ts)

    assert by_user
    clustered = any(
        timestamps[i + 9] - timestamps[i] < 60.0
        for timestamps in (sorted(ts) for ts in by_user.values())
        for i in range(len(timestamps) - 9)
    )
    assert clustered


def test_payload_schema_excludes_the_label():
    payload = generate(seed=42)[0].to_payload()

    assert set(payload) == {
        "eventId",
        "userId",
        "amount",
        "merchantId",
        "country",
        "deviceFp",
        "ip",
        "ts",
    }
    assert "is_fraud" not in payload and "pattern" not in payload
    assert payload["amount"] == round(payload["amount"], 2)


def test_user_pool_is_deterministic_and_offset():
    pool_a = build_user_pool(20)
    assert pool_a == build_user_pool(20)
    assert build_user_pool(3, id_offset=100)[0].user_id == "u000100"

    assert all(user.country in COUNTRY_RISK for user in pool_a)
    assert all(
        user.devices and all(fp.startswith("fp-") for fp in user.devices)
        for user in pool_a
    )
    assert all(
        user.merchants and set(user.merchants) <= set(MERCHANT_POOL)
        for user in pool_a
    )
    assert all(user.activity > 0 for user in pool_a)


def test_country_risk_lookup_and_default():
    assert country_risk("US") == pytest.approx(0.10)
    assert country_risk("RU") == pytest.approx(0.55)
    assert country_risk("ZZ") == pytest.approx(0.20)   # unknown country -> default prior
