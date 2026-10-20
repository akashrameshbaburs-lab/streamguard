"""Offline evaluation — joins alerts against ground truth for REAL numbers.

    python scripts/evaluate.py [--window 60]

Runs inside the api container (has Postgres access + the app package):

    docker compose exec -T api python scripts/evaluate.py

Computes, at the configured ALERT_THRESHOLD:
  * precision / recall / F1 of the *deployed alerting pipeline* (rules+ML fusion),
    against the simulator's ground-truth labels,
  * average / p95 detection latency,
  * measured throughput (events/s) over the scored-event window,
  * alert counts and the confusion matrix.
Prints a JSON document; docs/METRICS.md quotes these numbers verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StreamGuard alerting quality")
    parser.add_argument("--window", type=int, default=60,
                        help="recent window (seconds) for throughput measurement")
    args = parser.parse_args()

    db.init_db()

    # confusion matrix: alerts vs ground truth
    cm = db.fetchone("""
        SELECT
            COUNT(*) FILTER (WHERE g.is_fraud)                AS tp,
            COUNT(*) FILTER (WHERE NOT g.is_fraud)            AS fp
        FROM alerts a
        JOIN ground_truth g ON g.event_id = a.event_id
    """) or {"tp": 0, "fp": 0}

    total_fraud = db.fetchone(
        "SELECT COUNT(*) AS n FROM ground_truth WHERE is_fraud"
    ) or {"n": 0}
    total_events = db.fetchone(
        "SELECT COUNT(*) AS n FROM ground_truth"
    ) or {"n": 0}
    scored = db.fetchone(
        "SELECT COUNT(*) AS n, COALESCE(MAX(received_at - ts), INTERVAL '0') AS span "
        "FROM scored_events"
    ) or {"n": 0, "span": 0}

    delays = db.fetchone("""
        SELECT COALESCE(AVG(detection_delay_ms), 0) AS avg_delay,
               COALESCE(MAX(detection_delay_ms), 0) AS max_delay,
               COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY detection_delay_ms), 0) AS p50,
               COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY detection_delay_ms), 0) AS p95,
               COUNT(*) AS n
        FROM alerts
    """) or {"avg_delay": 0, "max_delay": 0, "p50": 0, "p95": 0, "n": 0}

    window_stats = db.fetchone("""
        SELECT COUNT(*) AS n FROM scored_events
        WHERE received_at > now() - make_interval(secs => %s)
    """, (args.window,)) or {"n": 0}

    # rule breakdown of the alerts that were true positives vs false positives
    breakdown = db.fetchall("""
        SELECT r.value ->> 'rule' AS rule,
               COUNT(*) AS hits,
               COUNT(*) FILTER (WHERE g.is_fraud) AS tp,
               COUNT(*) FILTER (WHERE NOT g.is_fraud) AS fp
        FROM alerts a
        JOIN ground_truth g ON g.event_id = a.event_id,
        LATERAL jsonb_array_elements(a.reasons) AS r
        GROUP BY 1 ORDER BY hits DESC
    """)

    # per-pattern breakdown: how many fraud events of each pattern were caught
    pattern_recall = db.fetchall("""
        SELECT g.pattern,
               COUNT(*)                       AS fraud_events,
               COUNT(a.event_id)              AS alerted
        FROM ground_truth g
        LEFT JOIN alerts a ON a.event_id = g.event_id
        WHERE g.is_fraud
        GROUP BY g.pattern ORDER BY fraud_events DESC
    """)
    pattern_alerts = db.fetchall("""
        SELECT g.pattern,
               COUNT(*)                       AS alerts,
               COUNT(*) FILTER (WHERE g.is_fraud) AS tp
        FROM alerts a
        JOIN ground_truth g ON g.event_id = a.event_id
        GROUP BY g.pattern ORDER BY alerts DESC
    """)

    tp, fp = int(cm["tp"]), int(cm["fp"])
    fn = int(total_fraud["n"]) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / int(total_fraud["n"]) if total_fraud["n"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_pattern: dict[str, dict] = {}
    for r in pattern_recall:
        fe, alerted = int(r["fraud_events"]), int(r["alerted"])
        per_pattern[r["pattern"]] = {
            "fraudEvents": fe,
            "alerted": alerted,
            "recall": round(alerted / fe, 4) if fe else None,
            "alerts": 0,
            "precision": None,
        }
    for r in pattern_alerts:
        entry = per_pattern.setdefault(
            r["pattern"],
            {"fraudEvents": 0, "alerted": 0, "recall": None, "alerts": 0, "precision": None},
        )
        n_alerts, n_tp = int(r["alerts"]), int(r["tp"])
        entry["alerts"] = n_alerts
        entry["precision"] = round(n_tp / n_alerts, 4) if n_alerts else None

    result = {
        "evaluatedAt": None,  # filled below
        "alertThreshold": float(os.environ.get("ALERT_THRESHOLD", 0.75)),
        "events": {
            "totalGroundTruth": int(total_events["n"]),
            "fraudGroundTruth": int(total_fraud["n"]),
            "fraudRate": round(int(total_fraud["n"]) / int(total_events["n"]), 6) if total_events["n"] else 0.0,
            "scoredEvents": int(scored["n"]),
        },
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                       "recall": round(recall, 4), "f1": round(f1, 4)},
        "alerts": {"count": int(delays["n"])},
        "detectionLatencyMs": {
            "avg": round(float(delays["avg_delay"]), 1),
            "p50": round(float(delays["p50"]), 1),
            "p95": round(float(delays["p95"]), 1),
            "max": round(float(delays["max_delay"]), 1),
        },
        "throughput": {
            "measuredEventsPerSec": round(int(window_stats["n"]) / args.window, 2),
            "windowSeconds": args.window,
        },
        "ruleBreakdown": [dict(r) for r in breakdown],
        "perPattern": [dict(v, pattern=k) for k, v in sorted(per_pattern.items())],
    }

    import time
    result["evaluatedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
