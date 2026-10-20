"""Diagnostic: replay synthetic traffic through the rules engine and report
which rules fire on benign vs fraudulent events. Used to sanity-check the
rule severities and the false-positive profile before a live run.

    docker compose exec -T api python scripts/diagnose_rules.py [--events 20000]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.features import FeatureExtractor  # noqa: E402
from app.rules import evaluate             # noqa: E402
from app.synth import EventFactory, build_user_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--users", type=int, default=800)
    args = ap.parse_args()

    users = build_user_pool(args.users, id_offset=0)
    factory = EventFactory(users, fraud_rate=0.025, seed=42, id_prefix="diag")
    extractor = FeatureExtractor()

    benign_hits: collections.Counter = collections.Counter()
    fraud_hits: collections.Counter = collections.Counter()
    benign_pattern_hits: collections.Counter = collections.Counter()
    benign_events = fraud_events = 0
    ts = 1_700_000_000.0
    for _ in range(args.events):
        ts += 1.0 / args.rate
        ev = factory.next_event(ts)
        payload = ev.to_payload()
        _feats, ctx = extractor.extract(payload)
        _score, reasons = evaluate(ctx, float(payload["amount"]))
        if ev.is_fraud:
            fraud_events += 1
            for r in reasons:
                fraud_hits[r["rule"]] += 1
        else:
            benign_events += 1
            for r in reasons:
                benign_hits[r["rule"]] += 1
                benign_pattern_hits[f"{ev.pattern}::{r['rule']}"] += 1
        extractor.update(payload)

    print(f"events={args.events} benign={benign_events} fraud={fraud_events} "
          f"({fraud_events / args.events * 100:.2f}% fraud)")
    print("benign rule hits:", dict(benign_hits))
    print("fraud  rule hits:", dict(fraud_hits))
    print("benign hits by pattern::rule:", dict(benign_pattern_hits))


if __name__ == "__main__":
    main()
