"""Unit tests for the deterministic rules engine (``app/rules.py``).

Each rule is exercised with a synthetic ``FeatureExtractor``-shaped context:
positive cases, benign cases, exact trigger boundaries and saturation caps.
"""
from __future__ import annotations

import pytest

from app import rules
from app.config import settings


def ctx(**overrides):
    """Benign context; no rule fires unless a signal is explicitly overridden."""
    base = {
        "amount_z": 0.0,
        "tx_last_60s": 1,
        "is_new_device": False,
        "country_change_30m": False,
    }
    base.update(overrides)
    return base


def rule_names(reasons):
    return [reason["rule"] for reason in reasons]


def test_benign_event_scores_zero_with_no_reasons():
    score, reasons = rules.evaluate(ctx(), 42.0)
    assert score == 0.0
    assert reasons == []


# --- velocity_burst ---------------------------------------------------------

def test_velocity_burst_fires_at_exact_trigger():
    score, reasons = rules.evaluate(ctx(tx_last_60s=rules.VELOCITY_TRIGGER), 20.0)
    assert score == pytest.approx(0.60)
    assert rule_names(reasons) == ["velocity_burst"]
    assert reasons[0]["severity"] == pytest.approx(0.60)
    assert reasons[0]["detail"] == "10 transactions in the last 60s"


def test_velocity_burst_just_below_trigger_is_benign():
    score, reasons = rules.evaluate(ctx(tx_last_60s=rules.VELOCITY_TRIGGER - 1), 20.0)
    assert score == 0.0
    assert reasons == []


def test_velocity_burst_severity_grows_then_saturates_at_cap():
    assert rules.evaluate(ctx(tx_last_60s=15), 20.0)[0] == pytest.approx(0.65)
    assert rules.evaluate(ctx(tx_last_60s=60), 20.0)[0] == pytest.approx(rules.SOLO_CAP)
    # a solo velocity hit proposes but must never alert on its own
    assert rules.SOLO_CAP < settings.ALERT_THRESHOLD


# --- amount_spike -----------------------------------------------------------

def test_amount_spike_fires_at_exact_z_trigger():
    score, reasons = rules.evaluate(ctx(amount_z=rules.AMOUNT_Z_TRIGGER), 2500.0)
    assert score == pytest.approx(0.62)
    assert rule_names(reasons) == ["amount_spike"]
    assert "amount=2500.00" in reasons[0]["detail"]


def test_amount_spike_just_below_trigger_is_benign():
    score, reasons = rules.evaluate(ctx(amount_z=rules.AMOUNT_Z_TRIGGER - 0.01), 2500.0)
    assert score == 0.0
    assert reasons == []


def test_amount_spike_severity_saturates_at_cap():
    assert rules.evaluate(ctx(amount_z=4.5), 800.0)[0] == pytest.approx(0.63)
    assert rules.evaluate(ctx(amount_z=100), 50_000.0)[0] == pytest.approx(rules.SOLO_CAP)


# --- new_device_amount ------------------------------------------------------

def test_new_device_amount_requires_the_absolute_floor():
    assert rules.evaluate(ctx(is_new_device=True), rules.NEW_DEVICE_MIN_AMOUNT - 0.01) == (0.0, [])

    score, reasons = rules.evaluate(ctx(is_new_device=True), rules.NEW_DEVICE_MIN_AMOUNT)
    assert score == pytest.approx(0.50)
    assert rule_names(reasons) == ["new_device_amount"]


def test_new_device_severity_scales_and_caps():
    assert rules.evaluate(ctx(is_new_device=True), 650.0)[0] == pytest.approx(0.56)
    assert rules.evaluate(ctx(is_new_device=True), 1000.0)[0] == pytest.approx(0.62)
    assert rules.evaluate(ctx(is_new_device=True), 50_000.0)[0] == pytest.approx(0.62)


def test_routine_new_device_small_amount_does_not_fire():
    score, reasons = rules.evaluate(ctx(is_new_device=True), 50.0)
    assert score == 0.0
    assert reasons == []


def test_known_device_high_amount_does_not_fire_new_device_rule():
    score, reasons = rules.evaluate(ctx(is_new_device=False), 50_000.0)
    assert score == 0.0
    assert reasons == []


# --- impossible_travel ------------------------------------------------------

def test_impossible_travel_alerts_on_its_own():
    score, reasons = rules.evaluate(ctx(country_change_30m=True), 30.0)
    assert score == pytest.approx(0.90)
    assert rule_names(reasons) == ["impossible_travel"]
    assert reasons[0]["severity"] == pytest.approx(0.90)
    assert score >= settings.ALERT_THRESHOLD


def test_impossible_travel_dominates_other_rules():
    score, reasons = rules.evaluate(
        ctx(tx_last_60s=12, amount_z=9.0, is_new_device=True, country_change_30m=True),
        2000.0,
    )
    assert score == pytest.approx(0.90)
    assert rule_names(reasons) == [
        "velocity_burst",
        "amount_spike",
        "new_device_amount",
        "impossible_travel",
    ]


def test_score_is_max_severity_across_firing_rules():
    # velocity at 10 -> 0.60, amount spike at z=5 -> 0.64
    score, reasons = rules.evaluate(ctx(tx_last_60s=10, amount_z=5.0), 900.0)
    assert score == pytest.approx(0.64)
    assert [reason["severity"] for reason in reasons] == pytest.approx([0.60, 0.64])
