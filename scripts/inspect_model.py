"""Print the deployed LR's coefficients, means and scales — diagnostics for
train/serve skew investigations (feature-unit and distribution checks).

    docker compose exec -T api python scripts/inspect_model.py
"""
from __future__ import annotations

import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402


def main() -> None:
    art = joblib.load(settings.MODEL_PATH)
    pipe = art["lr_model"]
    if hasattr(pipe, "calibrated_classifiers_"):  # unwrap Platt calibration
        inner = pipe.calibrated_classifiers_[0]
        pipe = getattr(inner, "estimator", None) or inner.base_estimator
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    names = art["feature_names"]

    print(f"version={art['version']} primary={art['primary']} "
          f"lr_threshold={art['lr_threshold']:.4f} trained_at={art['trained_at']}")
    print(f"intercept={float(clf.intercept_[0]):+.4f}")
    print(f"{'feature':22s} {'coef':>9s} {'train_mean':>11s} {'train_std':>10s}")
    for n, c, m, s in zip(names, clf.coef_[0], scaler.mean_, scaler.scale_):
        print(f"{n:22s} {c:+9.4f} {m:11.4f} {s:10.4f}")

    iso = art["iso_model"]
    print(f"\nisolation_forest: contamination={iso.contamination} "
          f"threshold={art['iso_threshold']:.4f} scale={art['iso_scale']:.4f}")


if __name__ == "__main__":
    main()
