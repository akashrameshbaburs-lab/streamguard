"""Shared test configuration.

``app/config.py`` reads the environment **once at import time** (the ``Settings``
class body is evaluated on import), so the knobs the assertions depend on are
pinned here — before any test module imports ``app`` — to the documented
defaults from ``.env.example``. This keeps the fusion/threshold maths
deterministic whether the suite runs in CI, on a developer machine, or inside
the compose image.

The suite is intentionally offline: no RabbitMQ, no Postgres, no model
artifacts and no network access are required.
"""
from __future__ import annotations

import os

os.environ["ALERT_THRESHOLD"] = "0.75"
os.environ["ML_WEIGHT"] = "0.9"
os.environ["FUSION_GAMMA"] = "0.15"
os.environ["RANDOM_SEED"] = "42"
