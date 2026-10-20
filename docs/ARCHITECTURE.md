# StreamGuard — Architecture

Real-time fraud & anomaly detection on a payment-event stream. This document
explains the data flow, the design decisions, and — importantly — *why* the
rules engine and the ML model are combined the way they are. Every severity
cap and fusion constant below was tuned against measured false-positive
classes (see `docs/METRICS.md` for the numbers).

## 1. Topology

```
                                ┌──────────────────────────────────────────┐
                                │              PostgreSQL 16               │
                                │  events ground truth   (simulator-only) │
                                │  scored_events         (every score)    │
                                │  alerts                (risk ≥ thresh)  │
                                │  labeled_feedback      (analyst labels) │
                                │  model_versions        (artifact meta)  │
                                └──────▲───────────────▲──────────────▲────┘
                                       │ write labels   │ INSERT       │ read
                                       │ (never read    │ scored/alerts│ model,
                                       │  by scorer)    │              │ retrain
┌────────────┐  20 evt/s   ┌───────────┴───┐  batch   ┌─┴──────────────┴──┐
│ simulator  ├────────────▶│   RabbitMQ    ├─────────▶│  consumer/scorer  │
│ (synth +   │  exchange   │  events ──▶   │  ack per │  rules + ML +     │
│ 4 patterns)│  `events`   │  raw_events   │  message │  fusion → alerts  │
└────────────┘             └───────────────┘          └────────┬──────────┘
                                                              │ publish alert
                     ┌────────────────────────────────────────▼──────────┐
                     │ RabbitMQ alerts ──▶ alerts queue ──▶ notifier     │
                     │ (Telegram webhook if configured; logs otherwise)  │
                     └───────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ API (FastAPI, port 3300)                                                │
│   REST: /healthz, /healthz/ready, /api/metrics, /api/alerts,           │
│         /api/alerts/{id}/feedback, /api/retrain, /api/model/metrics    │
│   SSE:  /api/stream  ──▶  static dashboard (/, vanilla HTML/JS/SVG)    │
│   retrain runs in-process; consumer hot-reloads the new artifact       │
└─────────────────────────────────────────────────────────────────────────┘
```

Services (docker compose): `postgres`, `rabbitmq`, `trainer` (one-shot,
trains the first model before the API/consumer start), `api`, `consumer`,
`notifier`, `simulator`. One Dockerfile, different commands.

## 2. Event contract

```json
{
  "eventId": "burst-3fa9c1-00000042",
  "userId": 1042,
  "amount": 1299.99,
  "merchantId": "m-88",
  "country": "US",
  "deviceFp": "fp-77aa",
  "ip": "203.0.113.9",
  "ts": 1761000000.123
}
```

`ts` is the producer timestamp (epoch seconds). Detection latency is measured
end-to-end: `alerts.detection_delay_ms = INSERT time − event.ts` — it includes
queueing, feature enrichment, scoring, and the DB write. Nothing is
benchmarked in isolation.

## 3. Feature extraction (stateful, per user)

`FeatureExtractor` keeps a rolling in-memory window per `userId`:

| feature             | signal                                                        |
|---------------------|---------------------------------------------------------------|
| `amount_z`          | z-score of amount vs the user's own history (≥8 txs, std floored at 10% of mean to avoid cold-start explosions) |
| `tx_last_60s`       | transaction count in the trailing 60 s (velocity)             |
| `is_new_device`     | device fingerprint never seen for this user                   |
| `country_change_30m`| country differs from any of the user's txs in the last 30 min |
| `hour_sin/cos`      | time-of-day seasonality                                       |
| `amount_log`        | `log1p(amount)` — heavy tail compressed for the linear model  |

Ten features in total (see `app/features.py::FEATURE_NAMES`). The extractor
is updated **after** scoring, so a transaction never contaminates its own
history (no look-ahead bias).

## 4. Rules engine (`app/rules.py`)

Four deterministic rules, each emitting a severity in [0, 1]; the engine
returns `rule_score = max(severities)` plus machine-readable `reasons[]`.

| rule                 | trigger                          | severity                | rationale |
|----------------------|----------------------------------|-------------------------|-----------|
| `impossible_travel`  | country change within 30 min     | **0.90 (solo-alerting)**| A cross-border jump in minutes is unambiguous; both cross-border legs are fraud *by construction* in the simulator, so precision on this rule is 1.0 in measurement. |
| `velocity_burst`     | ≥10 tx in 60 s                   | `min(0.66, 0.60+0.01·(v−10))` — **capped below the alert threshold** | Power users legitimately hit 10+ tx/60 s. Measured FP class. The rule *proposes*; the model *disposes*. |
| `amount_spike`       | amount z-score ≥4 vs user history| `min(0.66, 0.62+0.02·(z−4))` — capped below threshold | The benign amount tail (big legit purchases) reaches z ≥ 4. Measured FP class. Also proposes only. |
| `new_device_amount`  | unseen device + amount ≥ 300     | `min(0.62, 0.50+0.12·ratio)` | A single new-device hit is routine (new phone). Needs ML corroboration. |

The sub-threshold caps are the single most important tuning decision in the
project. Before they were introduced, velocity/amount rules alone produced
hundreds of false alerts per 10 minutes (the FP flood documented in
`docs/METRICS.md`, tuning history). After capping, those rules add *context*
to alerts and push borderline fraud over the line **only when the model
agrees** — but can never alert by themselves.

## 5. ML models (`app/model.py`, `scripts/train.py`)

Two models trained on ~50k synthetic rows generated by the *same* generator
that drives the live simulator (same distribution, different random seed and
a disjoint user pool — no user overlap between train and live traffic):

* **IsolationForest** — unsupervised baseline; captures "odd shape" anomalies.
* **LogisticRegression** (on StandardScaler) — supervised, class-weighted.

Training protocol (designed after three measured failures, see METRICS.md):

1. **Chronological 60/20/20 split** (train/val/test) — no shuffling, so the
   test set is genuinely future data.
2. **Chunked 24 h clock**: synthetic rows span a full 24 h in 1-hour jumps,
   otherwise `hour_sin/cos` are near-constant, the scaler amplifies noise,
   and the deployed model scores everything ≈ 0 (observed in cycle 1).
3. **Calibrated probabilities**: `CalibratedClassifierCV(lr, "sigmoid",
   cv="prefit")` fit on the *first half* of the validation split; the alert
   threshold is tuned on the *second half* (no test leakage). Raw
   class-weighted LR outputs were miscalibrated (~0.8 on benign rows),
   flooding fusion with confident-looking garbage (observed in cycle 2).
4. Primary model selection = best test F1 (logistic regression wins; IF is
   kept as a reported baseline).

The artifact (`models/model.joblib`) carries its own metrics
(`models/metrics.json`); the consumer hot-reloads when the version in
`model_versions` moves, so `POST /api/retrain` takes effect live within
`MODEL_POLL_SECONDS` (3 s) without restarting anything.

## 6. Score fusion (`app/model.py::fuse_risk`)

```
risk = clip( max(rule_score, ML_WEIGHT · ml_score)
           + FUSION_GAMMA · min(rule_score, ml_score),   # corroboration
             0, 1 )
```

defaults: `ML_WEIGHT = 0.9`, `FUSION_GAMMA = 0.15`, `ALERT_THRESHOLD = 0.75`.

Design rationale:

* **max()** — either engine can raise an alert alone. Impossible travel
  (0.90 ≥ 0.75) alerts solo; a confident model score (ml ≥ 0.833 after the
  0.9 weight) alerts solo. Weak signals from one engine never alert alone.
* **min() bonus** — when both engines agree (e.g. velocity rule 0.64 +
  model 0.80 → 0.816), the corroboration pushes borderline cases over the
  threshold. This is where capped rules earn their keep: they contribute
  context *and* a small bonus without ever firing alone.
* **ML_WEIGHT < 1** — the model is noisier than deterministic rules; a
  discount keeps a confident-but-alone model score from equaling a rule hit.

Worked examples (defaults):

| rule | ml | risk | outcome |
|------|-----|------|---------|
| 0.90 | 0.30 | 0.94 | travel — rule solo alert |
| 0.20 | 0.90 | 0.84 | ML-only alert (confident model) |
| 0.64 | 0.10 | 0.655 | power-user burst — **suppressed** |
| 0.64 | 0.80 | 0.816 | burst corroborated by model — alert |

**Threshold selection**: sweeping the alert threshold over 21.5k scored
events (519 fraud), F1 peaks flat between 0.72 and 0.78; 0.75 was chosen at
the peak with the best precision there — fewer false alerts for an analyst
to review at no meaningful F1 cost. The sweep is reproducible from the
`scored_events` table (rule_score, ml_score columns are persisted).

## 7. Honest-labeling discipline

The scorer **never reads `ground_truth`**. Labels are written by the
simulator (which knows which events it made fraudulent) into a separate
table the consumer has no code path to touch. `scripts/evaluate.py` joins
`alerts` × `ground_truth` *offline* to compute precision/recall/F1 — the
same way you would score against human-reviewed labels in production.
Analyst feedback (`/api/alerts/{id}/feedback`) is stored separately and
folded into the next retrain.

## 8. Synthetic fraud patterns (`app/synth.py`)

~2.5% fraud rate, four patterns (each with a ground-truth `pattern` tag for
per-pattern recall):

1. **velocity_burst** — 12 transactions in ~10 s on one account.
2. **amount_spike** — one transaction at z ≥ 4 vs the user's history.
3. **new_device_amount** — new device fingerprint + amount ≥ 300.
4. **impossible_travel** — 3-leg round trip: home (normal) → abroad
   (fraud) → home (fraud). The return leg being fraud removes the
   "benign home transaction after an away leg" false-positive class that
   plagued the 2-leg design.

Hard negatives are injected deliberately: power-user bursts (10–14 tx/60 s
that are *legitimate*), large benign purchases, and new-device-but-small
amounts. These are the classes that make precision non-trivial — a
simulator with only obvious fraud would show meaningless 0.99 precision.

A "busy window" mechanic prevents the generator from drawing an innocent
transaction for a user who is mid-attack (that would create contradictory
labels); unfinished attacks are drained on shutdown so every injected leg
gets its label.

## 9. API & dashboard

* `/api/metrics` — live KPIs from Postgres (processed, flagged, alerts,
  avg/p95 latency, events/sec over a sliding window, current model info).
* `/api/alerts?limit=` — newest alerts with `reasons[]`, risk components,
  measured detection latency, and any analyst feedback.
* `/api/alerts/{id}/feedback` — record `{"label": "true"|"false"}`;
  stored in `labeled_feedback`, used by the next retrain.
* `/api/retrain` — synchronous retrain on synthetic data + live labelled
  rows (ground truth + feedback); returns per-model P/R/F1; guarded by a
  lock (concurrent call → 409).
* `/api/stream` — SSE: `metrics` frames every second + `alert` frames as
  they happen. The dashboard (static HTML/CSS/JS, no CDN, hand-rolled SVG
  sparklines) consumes it via `EventSource`.

**Why no Redis**: at this scale (one consumer, one API instance) SSE can
poll Postgres directly for new alerts and windowed metrics — the queries
are indexed range scans returning in well under a millisecond. Adding a
cache would add a consistency surface (cache vs table) for no measurable
latency win. At higher fan-out the right move is partitioned streams (see
Roadmap), not a cache in front of Postgres.

## 10. Failure handling & operational notes

* RabbitMQ topology is declared by both producer and consumer (idempotent);
  alerts and events exchanges are durable, queues persistent.
* Consumer acks per message; DB errors nack+requeue, malformed JSON is
  dropped with a warning; scoring exceptions are logged and acked (a poison
  message must not stall the stream).
* Scored-event inserts are batched (200 rows / 0.5 s) for throughput;
  alerts are inserted immediately (latency-sensitive).
* Health: `/healthz` (liveness), `/healthz/ready` (Postgres + RabbitMQ
  round-trip); compose healthchecks gate startup order (trainer completes
  before API/consumer start).
* Restart safety: all inserts are `ON CONFLICT DO NOTHING`; the simulator's
  run-id prefix prevents event-ID collisions between steady and burst runs.

## 11. Known limitations (deliberate, documented)

* **Synthetic data** — real deployment needs a real labelled dataset
  (e.g. Kaggle IEEE-CIS Fraud Detection). The generator's job is to make the
  *pipeline* and the *tuning methodology* honest, not to imitate reality
  perfectly.
* **Cold-start recall** — velocity/amount patterns require ≥10 tx/60 s or
  ≥8 historical txs; the *first* legs of an attack are structurally
  undetectable in real time by those rules, bounding per-pattern recall
  (measured: see METRICS.md). This is the detection-window tradeoff every
  real-time velocity system has.
* **In-memory feature state** — a consumer restart loses per-user windows
  (features rebuild as traffic flows). A production system would back this
  with Redis/Kafka state stores.
* **Single consumer** — horizontal scaling works (competing consumers on
  `raw_events`) but per-user feature state would need partitioning by
  `userId` to stay consistent.
