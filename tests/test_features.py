"""Unit tests for per-user feature extraction (``app/features.py``)."""
from __future__ import annotations

import math

import pytest

from app import rules
from app.features import (
    FEATURE_NAMES,
    IMPOSSIBLE_TRAVEL_WINDOW_S,
    VELOCITY_WINDOW_S,
    FeatureExtractor,
    UserState,
)
from app.synth import country_risk


def event(user_id="u1", amount=100.0, country="US", device="fp-a", ts=1_000.0):
    return {
        "userId": user_id,
        "amount": amount,
        "country": country,
        "deviceFp": device,
        "ts": ts,
    }


def feed(extractor, ev):
    """extract-then-update, exactly the sequence the live scorer uses."""
    features, context = extractor.extract(ev)
    extractor.update(ev)
    return features, context


def test_feature_names_are_the_documented_ten():
    assert FEATURE_NAMES == [
        "log_amount",
        "amount_z",
        "tx_last_60s",
        "is_new_device",
        "country_change_30m",
        "country_risk",
        "hour_sin",
        "hour_cos",
        "log_device_count",
        "log_user_tx_count",
    ]
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == 10


def test_first_event_cold_start_baseline():
    extractor = FeatureExtractor()
    features, context = extractor.extract(event(amount=99.5, ts=6 * 3600))

    assert context == {
        "amount_z": 0.0,
        "tx_last_60s": 1,
        "is_new_device": False,
        "country_change_30m": False,
        "user_tx_count": 0,
    }
    assert len(features) == len(FEATURE_NAMES)
    assert features[0] == pytest.approx(math.log1p(99.5))
    assert features[2] == 1.0                    # this transaction counts itself
    assert features[3] == 0.0                    # a user's first device is not "new"
    assert features[5] == pytest.approx(country_risk("US"))
    assert features[6] == pytest.approx(1.0)     # 06:00 UTC -> sin(pi/2) = 1
    assert features[7] == pytest.approx(0.0, abs=1e-9)


def test_first_event_never_trips_a_rule():
    extractor = FeatureExtractor()
    _, context = extractor.extract(event(amount=5_000.0, ts=1_700_000_000.0))
    assert rules.evaluate(context, 5_000.0) == (0.0, [])


def test_update_folds_event_into_user_state():
    extractor = FeatureExtractor()
    feed(extractor, event(amount=80.0, ts=1_000.0))

    state = extractor.users["u1"]
    assert state.tx_count == 1
    assert state.amount_mean == pytest.approx(80.0)
    assert state.devices == {"fp-a"}
    assert state.last_country == "US"

    features, context = extractor.extract(event(amount=80.0, ts=1_001.0))
    assert context["user_tx_count"] == 1
    assert features[8] == pytest.approx(math.log1p(1))   # 1 prior device
    assert features[9] == pytest.approx(math.log1p(1))   # 1 prior transaction


def test_new_device_flag_tracks_fingerprints():
    extractor = FeatureExtractor()
    feed(extractor, event(device="fp-a", ts=1_000.0))

    features, context = extractor.extract(event(device="fp-b", ts=1_010.0))
    assert context["is_new_device"] is True
    assert features[3] == 1.0

    feed(extractor, event(device="fp-b", ts=1_010.0))
    _, context = extractor.extract(event(device="fp-a", ts=1_020.0))
    assert context["is_new_device"] is False

    _, context = extractor.extract(event(device="", ts=1_030.0))
    assert context["is_new_device"] is False


def test_velocity_counts_only_transactions_in_trailing_window():
    extractor = FeatureExtractor()
    for ts in (100.0, 110.0, 120.0):
        feed(extractor, event(ts=ts))

    _, context = extractor.extract(event(ts=125.0))
    assert context["tx_last_60s"] == 4          # 3 prior + this one

    feed(extractor, event(ts=125.0))
    _, context = extractor.extract(event(ts=1_000.0))
    assert context["tx_last_60s"] == 1          # everything aged out
    feed(extractor, event(ts=1_000.0))
    assert len(extractor.users["u1"].recent_ts) == 1   # state is trimmed, not accumulated

    # a transaction exactly VELOCITY_WINDOW_S old is outside the strict window
    boundary = FeatureExtractor()
    feed(boundary, event(ts=1_000.0))
    _, context = boundary.extract(event(ts=1_000.0 + VELOCITY_WINDOW_S))
    assert context["tx_last_60s"] == 1


def test_amount_z_needs_history_then_uses_floored_std():
    extractor = FeatureExtractor()
    amounts = [100.0, 110.0] * 4   # mean 105, tiny sample std
    for index, amount in enumerate(amounts[:7]):
        feed(extractor, event(amount=amount, ts=1_000.0 + index))

    _, context = extractor.extract(event(amount=1_000.0, ts=1_100.0))
    assert context["amount_z"] == 0.0           # only 7 prior txs -> no z yet

    feed(extractor, event(amount=amounts[7], ts=1_100.0))
    features, context = extractor.extract(event(amount=1_000.0, ts=1_200.0))
    expected_z = (1_000.0 - 105.0) / (0.10 * 105.0)   # std floored at 10% of mean
    assert context["amount_z"] == pytest.approx(expected_z)
    assert features[1] == 30.0                  # pathological z clipped for the model


def test_amount_z_is_negative_for_small_amounts():
    extractor = FeatureExtractor()
    for index, amount in enumerate([100.0, 110.0] * 4):
        feed(extractor, event(amount=amount, ts=1_000.0 + index))

    features, context = extractor.extract(event(amount=1.0, ts=1_200.0))
    assert context["amount_z"] == pytest.approx((1.0 - 105.0) / 10.5)
    assert features[1] == pytest.approx(context["amount_z"])


def test_amount_std_uses_welford_variance():
    extractor = FeatureExtractor()
    feed(extractor, event(amount=100.0, ts=1_000.0))
    assert extractor.users["u1"].amount_std() == 0.0     # a single sample has no spread

    feed(extractor, event(amount=120.0, ts=1_010.0))
    assert extractor.users["u1"].amount_std() == pytest.approx(20.0 / math.sqrt(2.0))


def test_amount_std_requires_two_samples():
    assert UserState().amount_std() == 0.0


def test_country_change_window_is_strict_and_handles_missing_country():
    extractor = FeatureExtractor()
    feed(extractor, event(country="US", ts=1_000.0))     # cold start: no previous country

    _, context = extractor.extract(event(country="US", ts=1_060.0))
    assert context["country_change_30m"] is False        # same country

    _, context = extractor.extract(
        event(country="FR", ts=1_000.0 + IMPOSSIBLE_TRAVEL_WINDOW_S)
    )
    assert context["country_change_30m"] is False        # exactly at the boundary is outside
    feed(extractor, event(country="FR", ts=1_000.0 + IMPOSSIBLE_TRAVEL_WINDOW_S))

    features, context = extractor.extract(
        event(country="US", ts=1_000.0 + IMPOSSIBLE_TRAVEL_WINDOW_S + 60)
    )
    assert context["country_change_30m"] is True
    assert features[4] == 1.0
    assert features[5] == pytest.approx(country_risk("US"))
    feed(extractor, event(country="US", ts=1_000.0 + IMPOSSIBLE_TRAVEL_WINDOW_S + 60))

    _, context = extractor.extract(
        event(country="", ts=1_000.0 + IMPOSSIBLE_TRAVEL_WINDOW_S + 120)
    )
    assert context["country_change_30m"] is False        # missing country never flags
