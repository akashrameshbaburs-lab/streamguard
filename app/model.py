"""Model training, persistence, scoring and risk fusion.

Training (scripts/train.py and POST /api/retrain both call ``train``):
  * generates ``n_rows`` synthetic transactions with the shared EventFactory,
  * replays them through FeatureExtractor for leak-free per-user features,
  * splits chronologically 60/20/20 train/val/test — val tunes the operating
    thresholds, test is only ever reported (no threshold-tuning leakage),
  * fits IsolationForest (unsupervised) on train features and
    LogisticRegression (supervised, class_weight=balanced) on train labels,
    with the LR calibrated (Platt / sigmoid) on a val half so probabilities
    reflect the true ~3 % fraud prior, not the balanced training prior,
  * evaluates BOTH on the held-out test split → real precision/recall/F1,
  * persists the best-F1 model as primary plus both models in one joblib file,
    with a sidecar metrics.json.

Scoring (``RiskScorer``):
  * loads the artifact, exposes ``score(features) -> ml_score in [0,1]``,
  * hot-reloads when the model file's recorded version changes (poll loop).

Fusion (documented in README and docs/ARCHITECTURE.md):

    risk = clip( max(rule_score, ML_WEIGHT * ml_score)
               + GAMMA * min(rule_score, ml_score),      # corroboration
               0, 1 )

Either engine can trigger an alert alone: an impossible-travel hit
(severity 0.90) crosses the 0.75 default threshold by itself, and a
confident model score (ml >= 0.84) does too. Velocity/amount-spike
severities are capped at 0.66 — BELOW the threshold — because power
users (10+ legit tx/60 s) and the benign amount tail (z >= 4) mimic
those rules; they propose, the model disposes. ML-only detections still
need a confident model output; weak signals need agreement.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import settings
from .features import FEATURE_NAMES, FeatureExtractor
from .synth import EventFactory, build_user_pool

log = logging.getLogger("streamguard.model")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def generate_training_frame(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a labelled feature frame by replaying synthetic events.

    Chronological replay through the same FeatureExtractor used online means
    features carry no future information (no leakage).
    """
    # Pool sized so the per-user event density (~12 eps / 600 users = 1.2
    # tx/min) matches live steady state (20 eps / 800 users = 1.5 tx/min).
    # A larger pool starves the benign velocity tail — the model would then
    # treat tx_last_60s=4 as fraud and ML-fire on ordinary heavy users.
    users = build_user_pool(600, id_offset=0)
    factory = EventFactory(users, fraud_rate=0.03, seed=seed, id_prefix=f"t{seed}")
    extractor = FeatureExtractor()

    ts = 1_700_000_000.0  # fixed epoch base: deterministic across runs
    features: list[list[float]] = []
    labels: list[int] = []
    patterns: list[str] = []
    events_by_second: dict[float, list[dict]] = {}

    # The simulated clock advances at ~12 events/s locally (realistic burst
    # and velocity spacing) but jumps ahead ~1 h every chunk so the frame
    # spans a full day. Without the jumps hour_sin/hour_cos are essentially
    # constant during training, StandardScaler amplifies the noise in those
    # near-zero-variance columns, and the fitted LR then scores EVERY live
    # event (different time of day) ~1e-80 — catastrophic train/serve skew.
    chunk = max(1, n_rows // 24)

    produced = 0
    while produced < n_rows:
        # advance simulated clock ~12 events/second to create realistic bursts
        ts += 1.0 / 12.0
        if produced and produced % chunk == 0:
            ts += 3600.0  # next hour of the day
        ev = factory.next_event(ts)
        events_by_second.setdefault(ev.ts, []).append(ev)
        produced += 1

    # replay strictly in timestamp order so user history is leak-free
    for ev_ts in sorted(events_by_second):
        for ev in events_by_second[ev_ts]:
            payload = ev.to_payload()
            feats, _ctx = extractor.extract(payload)
            extractor.update(payload)
            features.append(feats)
            labels.append(1 if ev.is_fraud else 0)
            patterns.append(ev.pattern)

    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64), {
        "fraud_rate": float(np.mean(labels)),
        "n_rows": int(len(labels)),
        "fraud_rows": int(np.sum(labels)),
    }


def train(
    n_rows: int | None = None,
    seed: int | None = None,
    version: int | None = None,
    notes: str = "",
    extra_features: np.ndarray | None = None,
    extra_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train both models, evaluate on the held-out split, persist artifacts.

    ``extra_features``/``extra_labels`` allow folding live labelled data
    (ground truth + analyst feedback) into the training set on retrain.
    Returns the metrics dict (also written to metrics.json and model_versions).
    """
    n_rows = n_rows or settings.TRAIN_ROWS
    seed = seed if seed is not None else settings.RANDOM_SEED

    X, y, frame_info = generate_training_frame(n_rows, seed)

    if extra_features is not None and extra_labels is not None and len(extra_labels) > 0:
        X = np.vstack([X, np.asarray(extra_features, dtype=np.float64)])
        y = np.concatenate([y, np.asarray(extra_labels, dtype=np.int64)])
        frame_info["extra_rows"] = int(len(extra_labels))

    # chronological 60/20/20 split (data is generated in time order).
    # train -> fit models, val -> tune operating thresholds, test -> report.
    # Tuning on val (not test) avoids leaking the test set into the decision
    # boundary, so the reported precision/recall/F1 are honest.
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # --- IsolationForest (unsupervised) ------------------------------------
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train)
    # decision_function: positive = inlier. Map to [0,1] anomaly score via a
    # logistic squash centred on 0 with scale = spread of train scores.
    iso_train_scores = iso.decision_function(X_train)
    iso_scale = float(np.std(iso_train_scores)) or 1.0
    # tune the operating threshold on the VALIDATION split only
    iso_val_anom = _squash_anomaly(iso.decision_function(X_val), iso_scale)
    iso_thr = _best_threshold(iso_val_anom, y_val)

    # report on the held-out TEST split
    iso_test_anom = _squash_anomaly(iso.decision_function(X_test), iso_scale)
    iso_pred = (iso_test_anom >= iso_thr).astype(int)
    iso_metrics = _prf(y_test, iso_pred)
    iso_metrics["threshold"] = round(float(iso_thr), 4)
    iso_metrics["model"] = "isolation_forest"

    # --- LogisticRegression (supervised baseline) --------------------------
    # class_weight=balanced counterfeits a ~50 % fraud prior, so raw
    # predict_proba is wildly over-confident (benign events land at 0.8+).
    # Calibrate with Platt scaling on the first half of the val split (which
    # carries the TRUE ~3 % prior), then tune the operating threshold on the
    # second half so calibration and threshold selection never see the same
    # rows. Calibrated probabilities are what the online fusion consumes.
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    lr.fit(X_train, y_train)
    half = len(X_val) // 2
    lr_model = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
    lr_model.fit(X_val[:half], y_val[:half])
    lr_val_prob = lr_model.predict_proba(X_val[half:])[:, 1]
    lr_thr = _best_threshold(lr_val_prob, y_val[half:])
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= lr_thr).astype(int)
    lr_metrics = _prf(y_test, lr_pred)
    lr_metrics["threshold"] = round(float(lr_thr), 4)
    lr_metrics["model"] = "logistic_regression"

    # --- choose primary by test F1 ----------------------------------------
    primary = "isolation_forest" if iso_metrics["f1"] >= lr_metrics["f1"] else "logistic_regression"
    if primary == "isolation_forest":
        iso_metrics["primary"] = True
    else:
        lr_metrics["primary"] = True

    version = version or _next_version()
    metrics = {
        "model_version": int(version),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "frame": frame_info,
        "isolation_forest": iso_metrics,
        "logistic_regression": lr_metrics,
        "primary_model": primary,
        "feature_names": FEATURE_NAMES,
        "notes": notes,
    }

    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    artifact = {
        "version": int(version),
        "primary": primary,
        "iso_model": iso,
        "iso_scale": iso_scale,
        "iso_threshold": float(iso_thr),
        "lr_model": lr_model,
        "lr_threshold": float(lr_thr),
        "feature_names": FEATURE_NAMES,
        "trained_at": metrics["trained_at"],
    }
    joblib.dump(artifact, settings.MODEL_PATH)
    with open(settings.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _record_version(version, metrics, notes)
    log.info(
        "trained model v%d primary=%s IF(p=%.3f r=%.3f f1=%.3f) LR(p=%.3f r=%.3f f1=%.3f)",
        version, primary,
        iso_metrics["precision"], iso_metrics["recall"], iso_metrics["f1"],
        lr_metrics["precision"], lr_metrics["recall"], lr_metrics["f1"],
    )
    return metrics


def _squash_anomaly(decision: np.ndarray, scale: float) -> np.ndarray:
    """Map IsolationForest decision_function (higher=more normal) to [0,1]
    anomaly score via a logistic squash; 0.5 ≈ the training decision boundary."""
    return 1.0 / (1.0 + np.exp(np.clip(decision / scale, -40, 40)))


def _best_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Pick the score threshold maximising F1 on the given split."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def _next_version() -> int:
    from . import db
    try:
        row = db.fetchone("SELECT COALESCE(MAX(version), 0) AS v FROM model_versions")
        return int(row["v"]) + 1 if row else 1
    except Exception:
        return int(time.time())


def _record_version(version: int, metrics: dict, notes: str) -> None:
    from . import db
    try:
        db.execute(
            """
            INSERT INTO model_versions (version, metrics, notes)
            VALUES (%s, %s::jsonb, %s)
            ON CONFLICT (version) DO UPDATE SET metrics = EXCLUDED.metrics,
                                                notes = EXCLUDED.notes
            """,
            (version, json.dumps(metrics), notes),
        )
    except Exception as exc:  # DB may be briefly unavailable during startup
        log.warning("could not record model version %d: %s", version, exc)


# ---------------------------------------------------------------------------
# Online scoring
# ---------------------------------------------------------------------------

class RiskScorer:
    """Loads the model artifact and scores feature vectors. Thread-safe."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.MODEL_PATH
        self._lock = threading.Lock()
        self._artifact: dict | None = None
        self.version: int = 0
        self.primary: str = "none"

    def load(self) -> bool:
        """(Re)load the artifact from disk. Returns True if a model is loaded."""
        if not os.path.exists(self.model_path):
            return False
        artifact = joblib.load(self.model_path)
        with self._lock:
            self._artifact = artifact
            self.version = int(artifact["version"])
            self.primary = artifact["primary"]
        log.info("loaded model v%d (primary=%s)", self.version, self.primary)
        return True

    def maybe_reload(self, current_db_version: int | None) -> bool:
        """Hot-reload if the DB says a newer model version exists."""
        if current_db_version and current_db_version != self.version:
            log.info("model version changed %d -> %d, hot-reloading", self.version, current_db_version)
            return self.load()
        return False

    def score(self, features: list[float]) -> tuple[float, dict]:
        """Return (ml_score in [0,1], details) using the primary model."""
        with self._lock:
            art = self._artifact
        if art is None:
            return 0.0, {"model_version": 0, "model": "none"}
        X = np.asarray(features, dtype=np.float64).reshape(1, -1)
        details: dict[str, Any] = {"model_version": self.version}
        if art["primary"] == "isolation_forest":
            d = float(art["iso_model"].decision_function(X)[0])
            score = float(_squash_anomaly(np.array([d]), art["iso_scale"])[0])
            details.update(model="isolation_forest", raw_decision=round(d, 4),
                           threshold=round(art["iso_threshold"], 4))
        else:
            proba = float(art["lr_model"].predict_proba(X)[0, 1])
            score = proba
            details.update(model="logistic_regression", probability=round(proba, 4),
                           threshold=art["lr_threshold"])
        return max(0.0, min(1.0, score)), details


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse_risk(rule_score: float, ml_score: float,
              ml_weight: float | None = None, gamma: float | None = None) -> float:
    """Combine rule and ML scores into a single risk in [0, 1].

    risk = clip( max(rule, ml_weight*ml) + gamma * min(rule, ml), 0, 1 )

    The max() term means EITHER engine can raise an alert on its own (a strong
    rule hit like an impossible-travel event, or a confident model score);
    the min() term is a corroboration bonus that pushes borderline cases over
    the threshold when both engines agree. ml_weight < 1 discounts the
    noisier model score relative to deterministic rules.

    Worked examples with defaults (ml_weight=0.9, gamma=0.15, threshold=0.75):
      rule=0.90 ml=0.30 -> max(0.90, 0.27) + 0.15*0.27 = 0.94  (travel: rule alert)
      rule=0.20 ml=0.90 -> max(0.20, 0.81) + 0.15*0.20 = 0.84  (ML-only alert)
      rule=0.64 ml=0.10 -> max(0.64, 0.09) + 0.15*0.10 = 0.655 (power user: no alert)
      rule=0.64 ml=0.80 -> max(0.64, 0.72) + 0.15*0.64 = 0.816 (burst: corroborated)
    """
    ml_weight = settings.ML_WEIGHT if ml_weight is None else ml_weight
    gamma = settings.FUSION_GAMMA if gamma is None else gamma
    raw = max(rule_score, ml_weight * ml_score) + gamma * min(rule_score, ml_score)
    return max(0.0, min(1.0, raw))


def ensure_model() -> dict:
    """Train on startup if no artifact exists (first container start)."""
    scorer = RiskScorer()
    if scorer.load():
        return {"model_version": scorer.version, "primary": scorer.primary, "freshly_trained": False}
    log.info("no model artifact found — training from scratch (this can take a minute)")
    metrics = train(n_rows=min(settings.TRAIN_ROWS, 20000), notes="bootstrap training on first start")
    return {"model_version": metrics["model_version"], "primary": metrics["primary_model"], "freshly_trained": True}
