# StreamGuard — Measured Metrics

Every number in this file was produced by `scripts/evaluate.py` (offline join
of `alerts × ground_truth`) or by the API endpoints during the final
end-to-end demo run on **2026-10-20**. Nothing is estimated or copied from
model-internal metrics — the detection numbers describe the **deployed
pipeline** (rules + ML + fusion at `ALERT_THRESHOLD=0.75`).

Reproduce everything with one command (clean volumes, rebuild, burst,
soak, all API checks, retrain, evaluate):

```powershell
.\scripts\demo.ps1 -Fresh          # final run: 19/19 checks PASS, exit 0
```

The evaluation JSON is saved verbatim to `docs/last-eval.json` by the demo;
the full console transcript is in `docs/demo-run.log`.

## 1. Headline results (final run: 5,000-event burst + 120 s steady soak)

| metric | value |
|---|---|
| Ground-truth events evaluated | **8,000** (5,000 burst + ~3,000 steady) |
| Fraud events (ground truth) | **176** (2.20 % fraud rate) |
| **Precision** | **0.758** (94 TP / 30 FP) |
| **Recall** | **0.534** (82 FN) |
| **F1** | **0.627** |
| Alerts raised | 125 |
| Detection latency avg | **32.0 ms** |
| Detection latency p50 / p95 / max | **23.0 / 109.2 / 150.8 ms** |
| Steady-state throughput (measured) | **19.95 evt/s** (simulator at 20 evt/s — consumer keeps up with zero backlog) |
| Burst throughput (publish) | ~830 evt/s (5,000 events in ~6 s), fully drained |
| API retrain | 8,000 live labelled rows folded in, 4.6 s, hot-reloaded live |

Detection latency is end-to-end and honest: `INSERT time − producer
timestamp`, including RabbitMQ queueing, feature enrichment, rules, ML
inference, fusion, and the Postgres write. No stage is measured in
isolation.

## 2. Where the detections come from

Rule breakdown of the 125 alerts (a single alert can carry several reasons):

| reason | hits | TP | FP | note |
|---|---|---|---|---|
| `ml_anomaly` | 124 | 94 | 30 | every alert had ML corroboration; 30 FPs are all ML-driven |
| `velocity_burst` | 51 | 40 | 11 | capped rule (0.66 max) — adds context, never alerts alone |
| `impossible_travel` | 42 | 42 | **0** | solo-alerting rule (0.90): perfect precision, as designed |
| `new_device_amount` | 8 | 8 | 0 | capped at 0.62 |
| `amount_spike` | 3 | 3 | 0 | capped at 0.66 |

Per-pattern recall (of all fraud events of that pattern, how many alerted):

| pattern | fraud events | alerted | recall | alert precision |
|---|---|---|---|---|
| `impossible_travel` | 32 | 32 | **1.000** | 1.000 |
| `new_device_amount` | 8 | 8 | **1.000** | 1.000 |
| `velocity_burst` | 114 | 49 | 0.430 | 1.000 |
| `amount_spike` | 22 | 5 | 0.227 | 1.000 |

**Why velocity/amount recall is bounded (honest structural limit):** the
velocity rule needs ≥10 transactions in 60 s and the amount z-score needs
≥8 historical transactions — the *first legs* of a burst attack have no
window/history yet, so they are undetectable in real time by construction.
This is the detection-window tradeoff every real-time velocity system
faces. Every alert the pipeline *did* raise on these patterns was correct
(precision 1.0); the misses are cold-start legs, not scoring errors.

The 30 false positives are the `normal` (benign) pattern: model-driven
alerts on the extreme benign tail (legitimate power users / large benign
purchases whose feature vectors resemble fraud). This is the measured cost
of the current operating point — each carries an `ml_anomaly` reason and
no solo rule hit.

## 3. Offline model metrics (held-out test set, chronological split)

Initial training (trainer one-shot, 50,000 synthetic rows, 60/20/20
chronological split, thresholds tuned on validation):

| model | precision | recall | F1 | threshold |
|---|---|---|---|---|
| **LogisticRegression** (primary) | **0.906** | 0.531 | **0.669** | 0.32 |
| IsolationForest (baseline) | 0.203 | 0.459 | 0.282 | 0.53 |

`POST /api/retrain` after the demo (50k synthetic + 8,000 live labelled
rows, mostly burst-pool cold-start users): LR test P=0.393 R=0.728
F1=0.511 — precision drops because the live rows are fraud-dense and
cold-start heavy; the retrain endpoint demonstrates the mechanism, and
v1 remained the scoring model for the evaluated alerts above.

Primary-model selection = best test F1. Both models and their metrics are
persisted (`models/model.joblib`, `models/metrics.json`,
`model_versions` table); the reported detection numbers are for the fused
pipeline, not for either model in isolation.

## 4. Alert-threshold sweep (why 0.75)

Offline rescore of **21,500 scored events** (519 fraud) from the
`scored_events` table — `rule_score` and `ml_score` are persisted per
event, so any fusion constants can be replayed exactly:

| threshold | alerts | TP | FP | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.60 | 727 | 325 | 402 | 0.447 | 0.626 | 0.522 |
| 0.65 | 629 | 311 | 318 | 0.494 | 0.599 | 0.542 |
| 0.68 | 540 | 301 | 239 | 0.557 | 0.580 | 0.569 |
| 0.70 | 488 | 291 | 197 | 0.596 | 0.561 | 0.578 |
| 0.72 | 443 | 290 | 153 | 0.655 | 0.559 | 0.603 |
| **0.75** | **403** | **278** | **125** | **0.690** | **0.536** | **0.603** |
| 0.78 | 365 | 264 | 101 | 0.723 | 0.509 | 0.597 |
| 0.80 | 351 | 259 | 92 | 0.738 | 0.499 | 0.595 |
| 0.85 | 312 | 238 | 74 | 0.763 | 0.459 | 0.573 |

F1 is flat between 0.72 and 0.78; **0.75** was chosen for the best
precision near the F1 peak — a fraud-alerting product optimizes the
analyst queue. (The sweep replays fusion with the shipped constants
`ML_WEIGHT=0.9`, `FUSION_GAMMA=0.15`.)

## 5. Tuning history (each cycle measured, each fix justified by data)

| cycle | change | measured result |
|---|---|---|
| 0 | initial (raw scores, uncapped rules) | precision **0.062** — velocity FP flood (power users) |
| 1 | chunked 24 h training clock; 3-leg travel pattern; busy-window labels; sub-threshold severity caps | recall 0.869, precision **0.145** (1,133 FPs — raw class-weighted LR scored benign rows ~0.8) |
| 2 | user pool 2,000 → 600 (match live tx density; starved benign velocity tail) | precision 0.213, recall 0.844, FP 681 |
| 3 | `CalibratedClassifierCV` (sigmoid, prefit on val-half; threshold tuned on val-half — no test leakage) | precision **0.718**, recall 0.560, F1 0.629 (at threshold 0.70) |
| 4 (final) | threshold 0.70 → 0.75 (sweep, §4); compose env fixed to real fusion keys (`ML_WEIGHT`/`FUSION_GAMMA=0.15`) | precision **0.758**, recall 0.534, F1 **0.627**; FP 30; latency avg 32 ms / p95 109 ms |

Root causes fixed along the way (all three were *measured*, not guessed):

1. **Near-constant clock features** — training spanned ~1.2 h, so
   `hour_sin/cos` were near-constant; StandardScaler amplified the noise
   and the fitted LR scored *every* live event ≈1e-82 (rules-only fusion).
2. **Starved benign velocity tail** — a 2,000-user training pool made
   `tx_last_60s=4` look fraudulent; LR fired on ordinary heavy users.
3. **Miscalibrated probabilities** — `class_weight=balanced` invents a
   ~50 % fraud prior; raw outputs put benign rows at 0.79–0.91, flooding
   `ml_anomaly` alerts. Fixed with sigmoid calibration on a leakage-free
   validation slice.

## 6. API verification (all PASS in the final demo)

`/healthz`, `/healthz/ready`, `/api/metrics`, `/api/alerts` (reasons[],
risk components, measured delays), `POST /api/alerts/{id}/feedback`,
SSE `/api/stream` (metrics + alert frames consumed by the dashboard),
static dashboard at `/` (KPIs + `EventSource` wiring verified),
`POST /api/retrain` (v1 → v2, 8,000 live rows, 4.6 s, consumer
hot-reloaded), `evaluate.py` exit 0. **19/19 checks PASS.**

## 7. Honest limitations

* **Synthetic data.** All traffic is generated (`app/synth.py`), ~2.2 %
  fraud, four patterns plus deliberately injected hard negatives
  (legitimate power-user bursts, benign large purchases). The numbers
  measure this pipeline honestly on this distribution; they are not
  claims about production fraud rates. Real deployment needs a real
  labelled dataset (e.g. Kaggle IEEE-CIS).
* **Cold-start recall ceiling** (§2): first legs of velocity/amount
  attacks are structurally undetectable in real time.
* **One live evaluation race:** 125 alerts existed at evaluation time,
  124 of which had a ground-truth row joinable at query time (one event's
  label landed between queries — inherent to scoring a live stream).
* **Feature state is in-memory**: a consumer restart rebuilds per-user
  windows from subsequent traffic (production: Redis / keyed partitions).
* **Single consumer**: throughput headroom is untested beyond ~830 evt/s
  burst publish; scaling out needs per-user state partitioning.
