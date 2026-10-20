# StreamGuard — Real-Time Fraud & Anomaly Detection

A production-shaped streaming pipeline that scores payment events for fraud
in real time: a **deterministic rules engine** and **two ML models**
(IsolationForest + Logistic Regression) whose scores are **fused** into a
single risk score, alerted on when it crosses a sweep-calibrated threshold.
Everything — latency, precision/recall, throughput — is **measured**, not
claimed; see [docs/METRICS.md](docs/METRICS.md) for real numbers produced by
`scripts/evaluate.py` against simulator ground truth.

## The problem

Payment fraud must be caught *while the transaction is in flight* — after
the goods ship, the money is gone. That means scoring a high-volume event
stream with low latency, keeping false positives low enough that a human
analyst queue stays usable, and detecting attacks that look individually
innocuous (a burst of small charges, one unusually large purchase, a new
device) until you combine signals.

Rules alone catch the obvious (a login from two countries 20 minutes apart)
but drown in false positives on the ambiguous (10 transactions in a minute —
is that a fraud burst, or someone paying for a group dinner?). ML alone
catches the subtle but hallucinates on cold-start users. StreamGuard
combines both by design: rules provide explainability and hard triggers;
the model disambiguates the ambiguous; a fusion formula decides what an
analyst actually sees.

## Architecture

```
                                  ┌────────────────────────────────────────┐
                                  │            PostgreSQL 16               │
                                  │  ground_truth   (simulator-only)      │
                                  │  scored_events  (every score)         │
                                  │  alerts         (risk ≥ threshold)    │
                                  │  labeled_feedback / model_versions    │
                                  └─────▲────────────▲───────────▲────────┘
                                        │ labels      │ INSERT    │ model
                                        │ (never read │ scored +  │ artifact
                                        │  by scorer) │ alerts    │
 ┌────────────┐  events   ┌────────────┴───┐  batch  ┌┴───────────┴────────┐
 │ simulator  ├──────────▶│   RabbitMQ     ├────────▶│  consumer / scorer  │
 │ synth +    │  exchange │  raw_events    │  ack    │  rules + ML +       │
 │ 4 patterns │  `events` └────────────────┘  per tx │  fusion → alerts    │
 └────────────┘                                     └──────────┬───────────┘
                                                              │ alert JSON
 ┌───────────────────────────────────────────────────────────┐│
 │ API (FastAPI :3300)  REST + SSE /api/stream               │◀┘
 │   /healthz /api/metrics /api/alerts /api/alerts/{id}/     │
 │   feedback /api/retrain /api/model/metrics                │
 │   static dashboard (/, vanilla HTML/JS/SVG sparklines)    │
 └───────────────────────────────────────────────────────────┘

 RabbitMQ alerts exchange ──▶ alerts queue ──▶ notifier (Telegram or log)
```

Seven compose services: `postgres`, `rabbitmq`, `trainer` (one-shot first
training), `api`, `consumer`, `notifier`, `simulator`. Full detail —
including every tuning decision and *why* — in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Detection pipeline (per event)

1. **Features** — 10 per-user features computed by the *same code path* in
   training and serving (zero train/serve skew): amount z-score vs the
   user's own history, 60-second velocity, new-device flag, 30-minute
   country change, country risk prior, time-of-day, history sizes.
2. **Rules** — four deterministic rules produce `rule_score` + `reasons[]`.
   Velocity/amount severities are **capped below the alert threshold**
   (they *propose*, the model *disposes*); impossible-travel alerts solo.
3. **ML** — calibrated Logistic Regression (primary) + IsolationForest
   baseline produce `ml_score`.
4. **Fusion** —

   ```
   risk = clip( max(rule_score, 0.9 · ml_score) + 0.15 · min(rule_score, ml_score), 0, 1 )
   ```

   `max` = either engine can alert alone; `min` term = corroboration bonus
   when both agree; 0.9 discounts the noisier model. `risk ≥ 0.75`
   (threshold chosen by sweeping 21.5k scored events — F1 peak is flat
   0.72–0.78, best precision at the peak) → alert row + publish.

## Tech stack

| Layer        | Choice | Why |
|--------------|--------|-----|
| API          | FastAPI + uvicorn (Python 3.12) | async SSE, pydantic validation |
| Queue        | RabbitMQ 3.13 (management UI) | durable exchanges, per-message ack |
| Database     | PostgreSQL 16 | window functions for metrics, JSONB reasons |
| ML           | scikit-learn 1.6 (IsolationForest, LogisticRegression, CalibratedClassifierCV) | honest calibrated probabilities, no GPU needed |
| Persistence  | joblib artifact + metrics.json, hot-reload on version change | zero-downtime retrain |
| Dashboard    | vanilla HTML/CSS/JS, hand-rolled SVG sparklines, SSE | no CDN, works air-gapped |
| Orchestration| Docker Compose | one command up, healthcheck-gated startup |

## API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/healthz` | liveness |
| GET  | `/healthz/ready` | Postgres + RabbitMQ round-trip |
| GET  | `/api/metrics?window=60` | live KPIs: processed, flagged, alerts, avg/p95 latency, evt/s, model info |
| GET  | `/api/alerts?limit=25` | newest alerts with reasons[], risk components, latency, feedback |
| POST | `/api/alerts/{id}/feedback` | `{"label": "true"\|"false"}` — folds into next retrain |
| POST | `/api/retrain` | retrain on synthetic + live labelled rows; returns per-model P/R/F1 |
| GET  | `/api/model/metrics` | full metrics of the newest model |
| GET  | `/api/stream?window_seconds=60` | SSE: `metrics` frames + live `alert` frames |
| GET  | `/` | live dashboard |

## Quick start

```powershell
# prerequisites: Docker Desktop running
docker compose up -d --build

# wait for health, then open the dashboard
Start-Process http://localhost:3300

# drive a 5000-event fraud burst + verify everything end-to-end
.\scripts\demo.ps1            # or: demo.ps1 -Fresh for a clean slate
```

The stack trains its first model before going live (~1 min). The simulator
then streams 20 events/s (~2.5% fraud) forever; `scripts/demo.ps1` adds a
fast 5000-event burst and asserts every endpoint works.

RabbitMQ management UI: http://localhost:15673 (streamguard/streamguard).

## Measured results

Real numbers from `scripts/evaluate.py` (offline join of alerts × simulator
ground truth) — reproduction steps and the full breakdown including
per-pattern recall, latency percentiles, and the tuning history (three
measured failure modes) are in [docs/METRICS.md](docs/METRICS.md).

| metric | value |
|---|---|
| Precision / Recall / F1 (fused pipeline) | **0.758 / 0.534 / 0.627** |
| Alerts raised (8,000 events, 176 fraud) | 125 (94 TP, 30 FP) |
| Detection latency avg / p50 / p95 | **32 / 23 / 109 ms** (end-to-end, incl. queue + DB write) |
| Throughput | 20 evt/s sustained consumer; ~830 evt/s burst ingest, drained |
| Impossible-travel rule precision | 1.000 (42/42 alerts correct) |
| Live retrain | 8,000 labelled rows folded in 4.6 s, hot-reloaded, zero downtime |

Per-pattern recall, the threshold sweep that selected 0.75, and the full
four-cycle tuning history (each failure measured, each fix justified) are
in [docs/METRICS.md](docs/METRICS.md). `.\scripts\demo.ps1` reproduces
every number end-to-end (19/19 automated checks).

## Honest notes

* **Synthetic data.** The simulator generates events with ~2.5% fraud in
  four patterns plus deliberately injected *hard negatives* (legitimate
  power-user bursts, benign large purchases). The numbers above measure the
  pipeline honestly on this distribution — they are not claims about
  real-world fraud rates. A production deployment would swap in a real
  labelled dataset (e.g. Kaggle IEEE-CIS Fraud Detection).
* **Cold-start is a real ceiling.** Velocity/amount attacks need ≥10 tx/60 s
  or ≥8 historical transactions before their signals exist; the first legs
  of an attack are structurally undetectable in real time, which bounds
  per-pattern recall (see METRICS.md). This is the detection-window
  tradeoff every real-time velocity system faces, not a bug to tune away.
* **Feature state is in-memory.** A consumer restart rebuilds per-user
  windows from live traffic; production would back this with a state store
  (Redis, keyed Kafka partitions).
* **The scorer never reads labels.** Ground truth lives in a table the
  consumer has no code path to touch; evaluation is the offline join you
  would do with human-reviewed labels.

## Repository layout

```
app/           # config, db, bus, synth, features, rules, model, services
scripts/       # train.py, evaluate.py, demo.ps1 (+ diagnostics)
static/        # dashboard (index.html, app.js, style.css)
docs/          # ARCHITECTURE.md, METRICS.md
docker-compose.yml, Dockerfile, requirements.txt, .env.example
```

## Roadmap

* Kafka partitioning by `userId` for consistent per-user state at scale
* Feature state in Redis with backfill-on-restart
* Concept-drift monitoring: PSI on feature distributions + auto-retrain gates
  (the `/api/retrain` + hot-reload path is already the execution arm)
* Threshold as a function of alert budget (queue-depth-driven, not fixed)
* Real-dataset adapter (IEEE-CIS) for the training generator
