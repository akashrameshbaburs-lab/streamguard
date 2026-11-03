"""Unit tests for risk fusion (``app/model.py``).

Only the pure ``fuse_risk`` maths and the no-artifact behaviour of
``RiskScorer`` are covered here, so no trained model, database or broker is
needed.
"""
from __future__ import annotations

import pytest

from app.config import settings
from app.model import RiskScorer, fuse_risk


def test_documented_defaults_are_loaded_from_config():
    assert settings.ML_WEIGHT == pytest.approx(0.9)
    assert settings.FUSION_GAMMA == pytest.approx(0.15)
    assert settings.ALERT_THRESHOLD == pytest.approx(0.75)


def test_documented_worked_examples():
    # risk = max(rule, 0.9 * ml) + 0.15 * min(rule, ml)
    assert fuse_risk(0.90, 0.30) == pytest.approx(0.945)   # travel: rule alert (doc rounds 0.94)
    assert fuse_risk(0.20, 0.90) == pytest.approx(0.840)   # ML-only alert
    assert fuse_risk(0.64, 0.10) == pytest.approx(0.655)   # power user: no alert
    assert fuse_risk(0.64, 0.80) == pytest.approx(0.816)   # burst: corroborated


def test_fusion_clips_to_unit_interval():
    assert fuse_risk(1.0, 1.0) == 1.0                      # 1 + 0.15 would exceed 1
    assert fuse_risk(0.95, 1.0) == 1.0
    assert fuse_risk(0.0, 0.0) == 0.0
    assert fuse_risk(-0.5, -0.2) == 0.0                    # defensive: never negative


def test_one_sided_rule_hit_alerts_alone():
    risk = fuse_risk(0.90, 0.05)
    assert risk == pytest.approx(0.9075)
    assert risk >= settings.ALERT_THRESHOLD


def test_one_sided_model_hit_alerts_alone():
    risk = fuse_risk(0.10, 0.90)
    assert risk == pytest.approx(0.825)
    assert risk >= settings.ALERT_THRESHOLD


def test_rule_only_decision_boundary_sits_at_alert_threshold():
    threshold = settings.ALERT_THRESHOLD

    at_boundary = fuse_risk(threshold, 0.0)
    assert at_boundary == pytest.approx(threshold)
    assert at_boundary >= threshold                        # exactly at threshold -> alert

    assert fuse_risk(threshold - 0.01, 0.0) < threshold


def test_ml_only_decision_boundary():
    threshold = settings.ALERT_THRESHOLD

    assert fuse_risk(0.0, threshold / settings.ML_WEIGHT) == pytest.approx(threshold)
    assert fuse_risk(0.0, 0.80) < threshold                # 0.720
    assert fuse_risk(0.0, 0.84) >= threshold               # 0.756


def test_corroboration_bonus_crosses_threshold_only_when_both_agree():
    threshold = settings.ALERT_THRESHOLD

    assert fuse_risk(0.64, 0.0) < threshold                # weak rule alone
    assert fuse_risk(0.0, 0.80) < threshold                # discounted ML alone
    assert fuse_risk(0.64, 0.80) >= threshold              # 0.816 together

    # capped velocity severity stays sub-threshold even with a weak ML nudge
    assert fuse_risk(0.66, 0.10) == pytest.approx(0.675)
    assert fuse_risk(0.66, 0.10) < threshold


def test_explicit_weights_override_settings():
    assert fuse_risk(0.5, 0.5, ml_weight=1.0, gamma=0.0) == pytest.approx(0.5)
    # max(0.4, 0.5*0.6) + 0.1*min(0.4, 0.6) = 0.4 + 0.04
    assert fuse_risk(0.4, 0.6, ml_weight=0.5, gamma=0.1) == pytest.approx(0.44)
    # the default path is identical to passing the configured knobs explicitly
    assert fuse_risk(0.3, 0.7) == pytest.approx(
        fuse_risk(0.3, 0.7, ml_weight=settings.ML_WEIGHT, gamma=settings.FUSION_GAMMA)
    )


def test_fused_risk_is_always_in_unit_interval():
    for rule_score in (0.0, 0.1, 0.25, 0.5, 0.64, 0.75, 0.9, 1.0):
        for ml_score in (0.0, 0.1, 0.25, 0.5, 0.64, 0.75, 0.9, 1.0):
            assert 0.0 <= fuse_risk(rule_score, ml_score) <= 1.0


def test_scorer_without_artifact_scores_zero(tmp_path):
    scorer = RiskScorer(model_path=str(tmp_path / "missing.joblib"))

    assert scorer.load() is False
    score, details = scorer.score([0.0] * 10)
    assert score == 0.0
    assert details == {"model_version": 0, "model": "none"}
