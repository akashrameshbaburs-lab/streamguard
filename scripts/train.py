"""CLI training entrypoint.

    python scripts/train.py [--rows 50000] [--seed 42] [--notes "..."]

Generates synthetic transactions, trains IsolationForest + LogisticRegression,
evaluates on a held-out chronological split, writes model.joblib + metrics.json,
records the version in Postgres (if reachable) and prints the REAL numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db            # noqa: E402
from app.model import train   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StreamGuard models")
    parser.add_argument("--rows", type=int, default=None, help="synthetic training rows (default: TRAIN_ROWS env or 50000)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: RANDOM_SEED env or 42)")
    parser.add_argument("--notes", type=str, default="", help="free-text note stored with the model version")
    parser.add_argument("--skip-db", action="store_true", help="do not touch Postgres (offline training)")
    args = parser.parse_args()

    if not args.skip_db:
        try:
            db.init_db()
        except Exception as exc:
            print(f"WARNING: could not reach Postgres ({exc}); training anyway, version will not be recorded",
                  file=sys.stderr)

    metrics = train(n_rows=args.rows, seed=args.seed, notes=args.notes)

    print("\n================ StreamGuard training results ================")
    print(f"model_version      : {metrics['model_version']}")
    print(f"primary_model      : {metrics['primary_model']}")
    print(f"train/test rows    : {metrics['train_rows']} / {metrics['test_rows']}")
    print(f"synthetic fraud %  : {metrics['frame']['fraud_rate'] * 100:.2f}")
    for key in ("isolation_forest", "logistic_regression"):
        m = metrics[key]
        print(f"{key:20s}: precision={m['precision']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} (thr={m['threshold']}) tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    print(f"artifacts          : {os.environ.get('MODEL_PATH', '/app/models/model.joblib')}")
    print("==============================================================")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
