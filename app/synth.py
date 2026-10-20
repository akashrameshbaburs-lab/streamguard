"""Synthetic payment-event generation with injected fraud patterns.

This module is the single source of truth for what a "payment event" looks
like and how fraud is simulated. It is used by:

  * ``app/simulator.py``  — live event stream into RabbitMQ (real clock),
  * ``scripts/train.py``  — bulk generation for model training (simulated clock).

Fraud patterns injected (each labelled ``is_fraud=True``):

  1. ``velocity_burst``     — >=10 transactions by the same user inside 60 s
  2. ``amount_spike``       — amount z-score > 4 vs the user's own history
  3. ``new_device_amount``  — brand-new device fingerprint + high amount
  4. ``impossible_travel``  — country change for one user within 30 minutes

We also inject *hard negatives* (``hard_negative``): legitimate high-value
purchases from a genuinely new device (e.g. a spouse's phone). These are
labelled legitimate on purpose so that precision in the evaluation reflects
real-world ambiguity instead of a toy 100 %.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Static country fraud-risk scores (0 = very safe .. 1 = very risky).
# Values are synthetic but ordered plausibly; the model treats this as a prior.
COUNTRY_RISK: dict[str, float] = {
    "US": 0.10, "CA": 0.10, "GB": 0.15, "DE": 0.12, "FR": 0.12, "ES": 0.14,
    "IT": 0.16, "NL": 0.11, "SE": 0.08, "JP": 0.09, "AU": 0.11, "SG": 0.13,
    "IN": 0.30, "BR": 0.45, "MX": 0.38, "NG": 0.60, "RU": 0.55, "CN": 0.32,
    "TR": 0.42, "ZA": 0.48, "AE": 0.25, "ID": 0.35, "PH": 0.36, "VN": 0.33,
}

HOME_COUNTRY_WEIGHTS = [
    ("US", 22), ("GB", 10), ("DE", 9), ("FR", 7), ("IN", 8), ("CA", 6),
    ("JP", 5), ("BR", 5), ("AU", 4), ("ES", 4), ("IT", 3), ("NL", 3),
    ("MX", 3), ("SG", 2), ("SE", 2), ("CN", 3),
]

HIGH_RISK_FRAUD_COUNTRIES = ["NG", "RU", "BR", "ZA", "TR", "MX", "ID"]

N_MERCHANTS = 250
MERCHANT_POOL = [f"m{i:04d}" for i in range(N_MERCHANTS)]

# Average number of events produced by one fraud injection, across the four
# patterns (velocity_burst≈12, amount_spike=1, new_device_amount=1,
# impossible_travel=3), chosen uniformly. Used to convert a target
# event-level fraud rate into a per-draw injection probability.
_AVG_EVENTS_PER_INJECTION = (12 + 1 + 1 + 3) / 4


def country_risk(country: str) -> float:
    return COUNTRY_RISK.get(country, 0.2)


def _weighted(items: list[tuple[str, int]], rng: random.Random) -> str:
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    acc = 0.0
    for value, weight in items:
        acc += weight
        if r <= acc:
            return value
    return items[-1][0]


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@dataclass
class User:
    user_id: str
    country: str
    devices: list[str]            # trusted device fingerprints
    amount_mean: float            # lognormal median of their spend
    amount_sigma: float           # lognormal sigma (spread of their spend)
    activity: float               # relative tx rate (weight in the pool)
    ip: str
    merchants: list[str]          # affinity subset of the merchant pool


def _make_user(idx: int) -> User:
    """Deterministically build a user from their index (no RNG state kept)."""
    rng = random.Random(0x5EED + idx * 7919)
    uid = f"u{idx:06d}"
    country = _weighted(HOME_COUNTRY_WEIGHTS, rng)
    n_devices = rng.choices([1, 2, 3], weights=[55, 35, 10])[0]
    devices = [f"fp-{uid}-{chr(ord('a') + j)}" for j in range(n_devices)]
    amount_mean = rng.uniform(15.0, 180.0)
    amount_sigma = rng.uniform(0.35, 0.85)
    activity = rng.lognormvariate(0.0, 0.45)  # moderate heavy tail: power users
    # stay comfortably below the 10 tx/60s velocity rule as legitimate traffic;
    # real-world power users are modelled by this tail, not by rule violations
    ip = f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    n_affinity = rng.randint(3, 12)
    merchants = rng.sample(MERCHANT_POOL, n_affinity)
    return User(uid, country, devices, amount_mean, amount_sigma, activity, ip, merchants)


def build_user_pool(size: int, id_offset: int = 0) -> list[User]:
    """Build a deterministic pool of ``size`` users with ids offset by ``id_offset``."""
    return [_make_user(id_offset + i) for i in range(size)]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class SynthEvent:
    event_id: str
    user_id: str
    amount: float
    merchant_id: str
    country: str
    device_fp: str
    ip: str
    ts: float                     # epoch seconds
    is_fraud: bool
    pattern: str                  # "normal" | fraud pattern name | "hard_negative"

    def to_payload(self) -> dict:
        """Wire format published to RabbitMQ — deliberately excludes the label."""
        return {
            "eventId": self.event_id,
            "userId": self.user_id,
            "amount": round(self.amount, 2),
            "merchantId": self.merchant_id,
            "country": self.country,
            "deviceFp": self.device_fp,
            "ip": self.ip,
            "ts": self.ts,
        }


class EventFactory:
    """Generates labelled events for a user pool in chronological order.

    The simulator drives it tick by tick with a real clock; the trainer drives
    it in a tight loop over a compressed simulated clock. Either way the same
    distributions and fraud mix apply.
    """

    def __init__(
        self,
        users: list[User],
        fraud_rate: float = 0.025,
        seed: int = 42,
        id_prefix: str = "e",
        flush_immediately: bool = False,
    ) -> None:
        self.users = users
        self.fraud_rate = fraud_rate
        self.rng = random.Random(seed)
        self.id_prefix = id_prefix
        # In burst/`--fast` mode the publisher races through events in ~2s, so
        # wall-clock never reaches the scheduled due-time of multi-event fraud
        # patterns (velocity bursts, travel-away legs). Setting this flag makes
        # those events emit immediately instead of waiting for their due time;
        # they still carry close-together timestamps, so the 60s velocity
        # window fires exactly as it would in a real burst.
        self.flush_immediately = flush_immediately
        self._counter = 0
        # unique-per-process token embedded in event ids (see _next_id)
        self._run_id = uuid.uuid4().hex[:6]
        # per-run bookkeeping used by fraud patterns that need history
        self.last_tx_ts: dict[str, float] = {}
        self.tx_count: dict[str, int] = {}
        # users with an attack in progress: benign draws skip them so ground
        # truth never labels an unrelated draw as benign while the account is
        # inside an active fraud burst (the velocity window would make such
        # labels ambiguous — see docs/ARCHITECTURE.md)
        self.busy_until: dict[str, float] = {}
        # pending impossible-travel follow-up events (due_ts, event)
        self.pending: list[tuple[float, SynthEvent]] = []
        # A single fraud *injection* emits several events (a velocity burst is
        # 10-14 tx, impossible-travel is 2). To make `fraud_rate` mean the
        # fraction of EVENTS that are fraud rather than the fraction of draws,
        # divide by the average events per injection (≈4.0). The exact
        # realised rate is measured and reported by scripts/train.py; this
        # first-order correction lands it within a few tenths of a percent.
        self._inject_prob = fraud_rate / _AVG_EVENTS_PER_INJECTION

    # -- helpers ------------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        # `run_id` makes ids unique across processes (a steady simulator and a
        # `--fast` burst can run concurrently; a restarted simulator would
        # otherwise re-emit the same ids and overwrite earlier ground truth).
        return f"{self.id_prefix}-{self._run_id}-{self._counter:08d}"

    def _pick_user(self, min_history: int = 0, ts: float | None = None,
                   exclude_busy: bool = False) -> User:
        """Pick a weighted-random user; optionally require prior transactions.

        Fraud patterns that rely on a z-score need the CONSUMER to already
        have history for the target, so we prefer users we've transacted for
        before (mirrors the consumer's own per-user history).

        ``exclude_busy`` skips users with an attack in progress — used for
        benign draws so an unrelated purchase never lands inside a fraud
        burst's 60-second velocity window (its ground-truth label would be
        ambiguous: benign by generation, but on a compromised account).
        """
        candidates = self.users
        if exclude_busy and ts is not None and self.busy_until:
            free = [u for u in self.users if self.busy_until.get(u.user_id, 0.0) <= ts]
            if free:
                candidates = free
        if min_history > 0:
            rich = [u for u in candidates if self.tx_count.get(u.user_id, 0) >= min_history]
            if rich:
                weights = [u.activity for u in rich]
                return self.rng.choices(rich, weights=weights, k=1)[0]
        weights = [u.activity for u in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _normal_amount(self, user: User) -> float:
        # amount = user's median spend, spread by their personal sigma
        amount = user.amount_mean * self.rng.lognormvariate(0.0, user.amount_sigma)
        return min(max(amount, 1.0), 50_000.0)

    def _base_event(
        self,
        user: User,
        ts: float,
        *,
        amount: float | None = None,
        country: str | None = None,
        device_fp: str | None = None,
        merchant_id: str | None = None,
        is_fraud: bool = False,
        pattern: str = "normal",
    ) -> SynthEvent:
        event = SynthEvent(
            event_id=self._next_id(),
            user_id=user.user_id,
            amount=amount if amount is not None else self._normal_amount(user),
            merchant_id=merchant_id or self.rng.choice(user.merchants or MERCHANT_POOL),
            country=country or user.country,
            device_fp=device_fp or self.rng.choice(user.devices),
            ip=user.ip,
            ts=ts,
            is_fraud=is_fraud,
            pattern=pattern,
        )
        self.last_tx_ts[user.user_id] = ts
        self.tx_count[user.user_id] = self.tx_count.get(user.user_id, 0) + 1
        return event

    # -- fraud patterns -----------------------------------------------------

    def _fraud_event(self, ts: float, id_prefix: str | None = None) -> list[SynthEvent]:
        """Return 1..N events implementing one fraud pattern."""
        pattern = self.rng.choice(["velocity_burst", "amount_spike", "new_device_amount", "impossible_travel"])
        # amount_spike / new_device rely on a z-score, so prefer a target the
        # consumer already has history for; other patterns don't need it.
        need_history = 4 if pattern in ("amount_spike", "new_device_amount") else 0
        user = self._pick_user(min_history=need_history)

        if pattern == "velocity_burst":
            # 10-14 transactions in ~20 s; small amounts so only the velocity
            # rule (and possibly ML) can catch it.
            n = self.rng.randint(10, 14)
            out = []
            for j in range(n):
                ev = self._base_event(
                    user,
                    ts + j * self.rng.uniform(0.5, 2.0),
                    amount=user.amount_mean * self.rng.uniform(0.8, 1.6),
                    is_fraud=True,
                    pattern=pattern,
                )
                out.append(ev)
            return out

        if pattern == "amount_spike":
            # floor the absolute amount so it is catchable by log_amount even
            # when the target has no per-user history yet (cold-start z=0).
            ev = self._base_event(
                user,
                ts,
                amount=max(600.0, user.amount_mean * self.rng.uniform(6.0, 14.0)),
                is_fraud=True,
                pattern=pattern,
            )
            return [ev]

        if pattern == "new_device_amount":
            ev = self._base_event(
                user,
                ts,
                amount=max(600.0, user.amount_mean * self.rng.uniform(3.0, 8.0)),
                device_fp=f"fp-{user.user_id}-fraud{self.rng.randint(1000, 9999)}",
                country=self.rng.choice(HIGH_RISK_FRAUD_COUNTRIES),
                is_fraud=True,
                pattern=pattern,
            )
            return [ev]

        # impossible_travel: a full round trip inside a few minutes — a
        # legitimate-looking tx at home, a tx from a distant country seconds
        # later, and the return leg back home right after. Both cross-border
        # legs are fraud (the return is the attacker coming back, not a
        # benign purchase); keeping the user's last known country at home
        # afterwards also means a later benign home purchase can never
        # masquerade as 'impossible travel'.
        away_delay = self.rng.uniform(3.0, 25.0)
        home_tx = self._base_event(user, ts, is_fraud=False, pattern="normal")
        away_tx = self._base_event(
            user,
            ts + away_delay,
            country=self.rng.choice([c for c in HIGH_RISK_FRAUD_COUNTRIES if c != user.country]),
            is_fraud=True,
            pattern=pattern,
        )
        return_tx = self._base_event(
            user,
            ts + away_delay + self.rng.uniform(5.0, 20.0),
            is_fraud=True,
            pattern=pattern,
        )
        return [home_tx, away_tx, return_tx]

    def _hard_negative(self, ts: float) -> SynthEvent:
        """Legitimate big purchase from a genuinely new device — must look scary."""
        user = self._pick_user(ts=ts, exclude_busy=True)
        new_fp = f"fp-{user.user_id}-new{self.rng.randint(100, 999)}"
        return self._base_event(
            user,
            ts,
            amount=max(350.0, user.amount_mean * self.rng.uniform(2.5, 6.0)),
            device_fp=new_fp,
            is_fraud=False,
            pattern="hard_negative",
        )

    # -- public API ---------------------------------------------------------

    def _emit_scheduled(self, ts: float) -> SynthEvent | None:
        """Pop one already-scheduled fraud event, if any is ready to emit.

        In ``flush_immediately`` (burst) mode every scheduled event is due at
        once, so multi-event fraud patterns complete inside a single fast
        burst instead of waiting for a wall-clock delay that never arrives.
        """
        if not self.pending:
            return None
        if self.flush_immediately:
            _due, ev = self.pending.pop(0)
            # Burst mode compresses scheduled legs (due up to ~30 s ahead) into
            # "now". An event must never carry a timestamp later than the moment
            # it is emitted, or detection latency (created_at - ts) would be
            # negative. Clamp to the emit time; relative leg order is preserved
            # because pending is FIFO and wall-clock time advances between pops.
            ev.ts = min(ev.ts, ts)
        else:
            ready = [p for p in self.pending if p[0] <= ts]
            if not ready:
                return None
            _due, ev = ready[0]
            self.pending.remove((ready[0]))
        self.last_tx_ts[ev.user_id] = ev.ts
        self.tx_count[ev.user_id] = self.tx_count.get(ev.user_id, 0) + 1
        return ev

    def drain_pending(self, ts: float | None = None) -> list[SynthEvent]:
        """Return every still-scheduled fraud leg, timestamped at ``ts``.

        Called on shutdown so a multi-event attack that already started is
        completed instead of dropped mid-pattern — an unemitted return leg
        would leave the user's last country 'abroad' and make the next benign
        home purchase look like impossible travel.
        """
        ts = ts if ts is not None else time.time()
        out: list[SynthEvent] = []
        while self.pending:
            _due, ev = self.pending.pop(0)
            ev.ts = min(ev.ts, ts)
            self.last_tx_ts[ev.user_id] = ev.ts
            self.tx_count[ev.user_id] = self.tx_count.get(ev.user_id, 0) + 1
            out.append(ev)
        return out

    def next_event(self, ts: float | None = None) -> SynthEvent:
        """Generate one event; fraud injections may enqueue extra events."""
        ts = ts if ts is not None else time.time()

        scheduled = self._emit_scheduled(ts)
        if scheduled is not None:
            return scheduled

        if self.rng.random() < self._inject_prob:
            events = self._fraud_event(ts)
            # schedule follow-up events for patterns that span time
            for extra in events[1:]:
                self.pending.append((extra.ts, extra))
            first = events[0]
            self.last_tx_ts[first.user_id] = first.ts
            self.tx_count[first.user_id] = self.tx_count.get(first.user_id, 0) + 1
            # Lock the victim out of benign draws until the pattern's last
            # leg plus the full 60 s velocity window has elapsed — a benign
            # draw inside that window would sit on a compromised account and
            # make the velocity rule's ground-truth label ambiguous.
            last_due = max(ev.ts for ev in events)
            self.busy_until[first.user_id] = max(
                self.busy_until.get(first.user_id, 0.0), last_due + 60.0,
            )
            return first

        if self.rng.random() < 0.008:  # ~0.8 % hard negatives
            ev = self._hard_negative(ts)
            return ev

        user = self._pick_user(ts=ts, exclude_busy=True)
        return self._base_event(user, ts)
