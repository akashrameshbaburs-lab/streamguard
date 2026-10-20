"""Deterministic rules engine.

Four hand-written fraud rules, each returning a severity in [0, 1]; the engine
returns the max severity as ``rule_score`` plus a human-readable ``reasons``
list. Severity design (tuned against the measured false-positive classes):
impossible travel keeps a strong solo severity (0.90) — a cross-border country
change within minutes is unambiguous and both cross-border legs are fraud by
design — while velocity/amount spike saturate BELOW the alert threshold (0.66)
because power users and the benign amount tail look identical to those rules;
they propose, the model disposes (see model.fuse_risk). New-device alone is
routine and stays weakest (0.62).
"""
from __future__ import annotations

from typing import Any

# Tunable severity parameters (documented in docs/ARCHITECTURE.md)
VELOCITY_TRIGGER = 10        # txs in 60 s before the velocity rule fires
AMOUNT_Z_TRIGGER = 4.0       # z-score before the amount-spike rule fires
NEW_DEVICE_MIN_AMOUNT = 300.0  # new device only matters for big tickets
SOLO_CAP = 0.66              # max severity for rules that must be ML-corroborated


def evaluate(context: dict[str, Any], amount: float) -> tuple[float, list[dict]]:
    """Return (rule_score, reasons) for one transaction.

    ``context`` comes from ``FeatureExtractor.extract`` and contains the raw
    signals: amount_z, tx_last_60s, is_new_device, country_change_30m.
    """
    reasons: list[dict] = []
    score = 0.0

    # 1) velocity burst: >=10 tx in 60 s — proposes only (capped sub-threshold)
    v = context["tx_last_60s"]
    if v >= VELOCITY_TRIGGER:
        sev = min(SOLO_CAP, 0.60 + 0.01 * (v - VELOCITY_TRIGGER))
        reasons.append({
            "rule": "velocity_burst",
            "severity": round(sev, 3),
            "detail": f"{v} transactions in the last 60s",
        })
        score = max(score, sev)

    # 2) amount spike: z > 4 vs user's own history — proposes only as well
    z = context["amount_z"]
    if z >= AMOUNT_Z_TRIGGER:
        sev = min(SOLO_CAP, 0.62 + 0.02 * (z - AMOUNT_Z_TRIGGER))
        reasons.append({
            "rule": "amount_spike",
            "severity": round(sev, 3),
            "detail": f"amount z-score {z:.1f} vs user history (amount={amount:.2f})",
        })
        score = max(score, sev)

    # 3) new device + high amount: capped at 0.62 — a single new-device hit
    #    never alerts on its own; it needs ML corroboration via the fusion.
    if context["is_new_device"] and amount >= NEW_DEVICE_MIN_AMOUNT:
        ratio = min(1.0, (amount - NEW_DEVICE_MIN_AMOUNT) / 700.0)
        sev = min(0.62, 0.50 + 0.12 * ratio)
        reasons.append({
            "rule": "new_device_amount",
            "severity": round(sev, 3),
            "detail": f"unseen device fingerprint with amount {amount:.2f}",
        })
        score = max(score, sev)

    # 4) impossible travel: country change within 30 minutes — strong enough
    #    to alert on its own (both cross-border legs are fraud by design).
    if context["country_change_30m"]:
        sev = 0.90
        reasons.append({
            "rule": "impossible_travel",
            "severity": sev,
            "detail": "country changed within the last 30 minutes",
        })
        score = max(score, sev)

    return score, reasons
