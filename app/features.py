"""Feature extraction shared by training (offline replay) and the live scorer.

The exact same code path computes features for the 50k-row training set and
for every live event, which eliminates train/serve skew by construction.

Feature vector (fixed order, see ``FEATURE_NAMES``):

  0 log_amount         log1p(amount)
  1 amount_z           z-score of amount vs the user's own history (0 if cold)
  2 tx_last_60s        user transactions in the trailing 60 seconds
  3 is_new_device      device fingerprint never seen for this user (0/1)
  4 country_change_30m country differs from the user's previous country
                       within the last 30 minutes (0/1)
  5 country_risk       static 0..1 risk prior for the transaction country
  6 hour_sin           sin(2*pi*hour/24) of the event hour (UTC)
  7 hour_cos           cos(2*pi*hour/24)
  8 log_device_count   log1p distinct devices the user has used
  9 log_user_tx_count  log1p user's total prior transactions

Users with no history ("cold start") get amount_z=0, tx_last_60s=1,
is_new_device=0 — the model then relies on amount/country features only.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .synth import country_risk

FEATURE_NAMES = [
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

VELOCITY_WINDOW_S = 60.0
IMPOSSIBLE_TRAVEL_WINDOW_S = 30 * 60.0


@dataclass
class UserState:
    """Rolling per-user history maintained by the scorer (and by training replay)."""
    tx_count: int = 0
    amount_mean: float = 0.0
    amount_m2: float = 0.0       # Welford accumulator for variance
    devices: set[str] = field(default_factory=set)
    last_country: str | None = None
    last_ts: float = 0.0
    recent_ts: list[float] = field(default_factory=list)  # trailing velocity window

    def amount_std(self) -> float:
        if self.tx_count < 2:
            return 0.0
        return math.sqrt(self.amount_m2 / (self.tx_count - 1))


class FeatureExtractor:
    """Stateful per-user feature extraction. Call ``extract`` then ``update``
    for every event, in timestamp order — identical to the training replay."""

    def __init__(self) -> None:
        self.users: dict[str, UserState] = {}

    def _state(self, user_id: str) -> UserState:
        st = self.users.get(user_id)
        if st is None:
            st = UserState()
            self.users[user_id] = st
        return st

    # -- pure feature computation (no state mutation) -----------------------

    def extract(self, event: dict) -> tuple[list[float], dict]:
        """Compute the feature vector plus rule-relevant context for an event.

        ``event`` keys: userId, amount, country, deviceFp, ts (epoch seconds).
        Returns (features, context) where context carries the raw signals the
        rules engine needs (z-score, velocity count, flags).
        """
        st = self._state(event["userId"])
        amount = float(event["amount"])
        ts = float(event["ts"])
        country = event.get("country") or ""
        device = event.get("deviceFp") or ""

        # z-score vs user history (before this tx).
        # Require a reasonably long history and floor the std so that a user
        # with a very tight spend distribution doesn't produce huge z-scores
        # for ordinary purchases (e.g. a $4 coffee after two $3 coffees).
        std = st.amount_std()
        if st.tx_count >= 8 and std > 1e-6:
            effective_std = max(std, 0.10 * st.amount_mean)
            amount_z = (amount - st.amount_mean) / effective_std
        else:
            amount_z = 0.0

        # velocity: count user txs in trailing 60 s (this tx included)
        cutoff = ts - VELOCITY_WINDOW_S
        tx_last_60s = 1 + sum(1 for t in st.recent_ts if t > cutoff)

        is_new_device = 1.0 if (st.tx_count >= 1 and device and device not in st.devices) else 0.0

        country_change = 0.0
        if st.last_country and st.last_ts and country and country != st.last_country \
                and (ts - st.last_ts) < IMPOSSIBLE_TRAVEL_WINDOW_S:
            country_change = 1.0

        hour = time.gmtime(ts).tm_hour + time.gmtime(ts).tm_min / 60.0

        features = [
            math.log1p(amount),
            max(min(amount_z, 30.0), -30.0),   # clip pathological z
            float(tx_last_60s),
            is_new_device,
            country_change,
            country_risk(country),
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.log1p(len(st.devices)),
            math.log1p(st.tx_count),
        ]
        context = {
            "amount_z": amount_z,
            "tx_last_60s": tx_last_60s,
            "is_new_device": bool(is_new_device),
            "country_change_30m": bool(country_change),
            "user_tx_count": st.tx_count,
        }
        return features, context

    # -- state mutation ------------------------------------------------------

    def update(self, event: dict) -> None:
        """Fold the event into the user's history (call AFTER ``extract``)."""
        st = self._state(event["userId"])
        amount = float(event["amount"])
        ts = float(event["ts"])

        st.tx_count += 1
        delta = amount - st.amount_mean
        st.amount_mean += delta / st.tx_count
        st.amount_m2 += delta * (amount - st.amount_mean)

        if event.get("deviceFp"):
            st.devices.add(event["deviceFp"])
        st.last_country = event.get("country") or st.last_country
        st.last_ts = max(st.last_ts, ts)
        st.recent_ts.append(ts)
        cutoff = ts - VELOCITY_WINDOW_S
        # keep the velocity window trimmed (list stays tiny at real rates)
        st.recent_ts = [t for t in st.recent_ts if t > cutoff]
