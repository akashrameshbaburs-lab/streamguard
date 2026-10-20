/* StreamGuard dashboard — vanilla JS, no dependencies.
 * Connects to /api/stream (SSE): "metrics" ticks once per second, "alert"
 * events pushed as they are written to Postgres by the scorer.
 */
"use strict";

const $ = (id) => document.getElementById(id);

// rolling 60-sample window (1 sample/sec) of throughput for the sparkline
const HISTORY = 60;
const throughputHistory = [];
const alertRateHistory = [];

const renderedAlerts = new Set();
const feedbackState = new Map(); // alertId -> "true"|"false"

/* ---------------------------------------------------------------- format */
function fmtInt(n) {
  return Number(n).toLocaleString("en-US");
}
function fmtPct(x) {
  return (x * 100).toFixed(2) + "%";
}
function fmtMs(x) {
  const v = Number(x);
  if (v >= 1000) return (v / 1000).toFixed(2) + " s";
  return v.toFixed(0) + " ms";
}

/* ------------------------------------------------------------- kpi cards */
function renderMetrics(m) {
  $("kpi-events").textContent = fmtInt(m.eventsProcessed);
  $("kpi-eps").textContent = m.eventsPerSec.toFixed(1);
  $("kpi-fraud-rate").textContent = fmtPct(m.fraudRate);
  $("kpi-flagged").textContent = fmtInt(m.flaggedEvents);
  $("kpi-alerts").textContent = fmtInt(m.alertsCount);
  $("kpi-threshold").textContent = m.threshold.toFixed(2);
  $("kpi-avg-delay").textContent = fmtMs(m.avgDetectionDelayMs);
  $("kpi-p95-delay").textContent = fmtMs(m.p95DetectionDelayMs);

  $("model-badge").textContent =
    "model v" + m.modelVersion + (m.modelPrimary ? " · " + m.modelPrimary.replace(/_/g, " ") : "");

  throughputHistory.push(m.eventsPerSec);
  if (throughputHistory.length > HISTORY) throughputHistory.shift();
  // alerts-per-sec for the window: reuse window flag count
  alertRateHistory.push(m.windowSeconds ? m.alertsLastWindow / m.windowSeconds : 0);
  if (alertRateHistory.length > HISTORY) alertRateHistory.shift();
  renderSparkline();
}

/* ------------------------------------------------------------ sparkline */
function renderSparkline() {
  const svg = $("sparkline");
  const W = 600, H = 120, PAD = 4;
  const maxVal = Math.max(1, ...throughputHistory, ...alertRateHistory) * 1.15;

  const toPath = (arr, close) => {
    if (arr.length < 2) return "";
    const dx = (W - PAD * 2) / (HISTORY - 1);
    const pts = arr.map((v, i) => {
      const x = PAD + i * dx;
      const y = H - PAD - (v / maxVal) * (H - PAD * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    if (!close) return "M" + pts.join(" L");
    return "M" + pts.join(" L") + ` L${(PAD + (arr.length - 1) * dx).toFixed(1)},${H - PAD} L${PAD},${H - PAD} Z`;
  };

  // pad history to the right so the line "ages" left
  const padRight = (arr) => {
    const out = new Array(HISTORY - arr.length).fill(null);
    return arr.concat(out);
  };
  const t = padRight(throughputHistory);
  const a = padRight(alertRateHistory);
  const tPts = t.map((v, i) => ({ v, i })).filter((p) => p.v !== null);
  const aPts = a.map((v, i) => ({ v, i })).filter((p) => p.v !== null);
  const dx = (W - PAD * 2) / (HISTORY - 1);
  const xy = (p) => [PAD + p.i * dx, H - PAD - (p.v / maxVal) * (H - PAD * 2)];

  const line = (pts) => pts.map((p, i) => {
    const [x, y] = xy(p);
    return (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
  }).join(" ");

  const tLine = line(tPts);
  const aLine = line(aPts);
  const area = tPts.length > 1
    ? tLine + ` L${(PAD + (tPts[tPts.length - 1].i) * dx).toFixed(1)},${H - PAD} L${PAD},${H - PAD} Z`
    : "";

  svg.innerHTML = `
    <line class="grid" x1="${PAD}" y1="${H / 2}" x2="${W - PAD}" y2="${H / 2}"/>
    <line class="grid" x1="${PAD}" y1="${H - PAD}" x2="${W - PAD}" y2="${H - PAD}"/>
    ${area ? `<path class="area-throughput" d="${area}"/>` : ""}
    ${tLine ? `<path class="line-throughput" d="${tLine}"/>` : ""}
    ${aLine ? `<path class="line-alerts" d="${aLine}"/>` : ""}
    <text class="axis-label" x="${PAD}" y="12">max ${maxVal.toFixed(1)}/s</text>
    <text class="axis-label" x="${W - 44}" y="${H - 8}">-60s</text>
  `;
}

/* --------------------------------------------------------------- alerts */
function renderAlert(alert) {
  const container = $("alerts");
  const empty = container.querySelector(".empty");
  if (empty) empty.remove();

  if (renderedAlerts.has(alert.eventId)) return;
  renderedAlerts.add(alert.eventId);

  const row = document.createElement("div");
  const riskMed = alert.risk < 0.85;
  row.className = "alert-row" + (riskMed ? " risk-med" : "");

  const chips = alert.reasons.map((r) => {
    const cls = r.rule === "ml_anomaly" ? "chip chip-ml" : "chip";
    return `<span class="${cls}" title="${escapeHtml(r.detail)}">${escapeHtml(r.rule)} ${Number(r.severity).toFixed(2)}</span>`;
  }).join("");

  const fb = feedbackState.get(alert.id);
  row.innerHTML = `
    <div class="risk-pill${riskMed ? " risk-med" : ""}">${alert.risk.toFixed(2)}</div>
    <div class="alert-main">
      <div class="who"><span class="amount">$${alert.amount.toFixed(2)}</span> &middot; ${escapeHtml(alert.userId)}</div>
      <div class="meta">${escapeHtml(alert.country || "-")} &middot; evt ${escapeHtml(String(alert.eventId).slice(0, 18))} &middot; model v${alert.modelVersion}</div>
    </div>
    <div class="chips">${chips}</div>
    <div class="alert-actions">
      <span class="delay">${fmtMs(alert.detectionDelayMs)}</span>
      ${fb !== undefined
        ? `<span class="feedback-done${fb === "false" ? " fp" : ""}">${fb === "true" ? "✓ TP" : "✗ FP"}</span>`
        : `<button class="btn btn-tp" data-id="${alert.id}" data-label="true">True positive</button>
           <button class="btn btn-fp" data-id="${alert.id}" data-label="false">False positive</button>`}
    </div>
  `;

  row.querySelectorAll("button[data-label]").forEach((btn) => {
    btn.addEventListener("click", () => submitFeedback(btn.dataset.id, btn.dataset.label, row));
  });

  container.prepend(row);
  while (container.children.length > 50) container.lastChild.remove();

  const n = container.querySelectorAll(".alert-row").length;
  $("alert-count").textContent = n;
}

async function submitFeedback(alertId, label, row) {
  try {
    const resp = await fetch(`/api/alerts/${alertId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    feedbackState.set(alertId, label);
    row.querySelector(".alert-actions").innerHTML =
      `<span class="delay">&nbsp;</span><span class="feedback-done${label === "false" ? " fp" : ""}">${label === "true" ? "✓ TP" : "✗ FP"}</span>`;
  } catch (err) {
    alert("Feedback failed: " + err.message);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------- model / retrain */
async function loadModelPanel() {
  try {
    const resp = await fetch("/api/metrics");
    const m = await resp.json();
    $("m-version").textContent = "v" + m.modelVersion;
    $("m-primary").textContent = (m.modelPrimary || "-").replace(/_/g, " ");
    const detail = await fetchModelDetail();
    if (detail) {
      const ifM = detail.isolation_forest || {};
      const lrM = detail.logistic_regression || {};
      $("m-if-p").textContent = num(ifM.precision);
      $("m-if-r").textContent = num(ifM.recall);
      $("m-if-f1").textContent = num(ifM.f1);
      $("m-lr-p").textContent = num(lrM.precision);
      $("m-lr-r").textContent = num(lrM.recall);
      $("m-lr-f1").textContent = num(lrM.f1);
    }
  } catch (err) {
    console.warn("model panel load failed", err);
  }
}

async function fetchModelDetail() {
  try {
    const resp = await fetch("/api/model/metrics");
    if (resp.ok) return await resp.json();
  } catch (_) { /* older payloads */ }
  return null;
}

function num(x) {
  return x === undefined || x === null ? "–" : Number(x).toFixed(4);
}

function wireRetrain() {
  const btn = $("retrain-btn");
  const status = $("retrain-status");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    status.textContent = "retraining… generating synthetic data + training IF & LR (can take a minute)";
    try {
      const resp = await fetch("/api/retrain", { method: "POST" });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || "HTTP " + resp.status);
      }
      const r = await resp.json();
      status.textContent =
        `✓ v${r.modelVersion} trained in ${r.durationSeconds}s — ` +
        `IF F1=${r.isolationForest.f1.toFixed(3)}  LR F1=${r.logisticRegression.f1.toFixed(3)} ` +
        `(primary: ${r.primaryModel.replace(/_/g, " ")}, ${r.trainRows} rows)`;
      await loadModelPanel();
    } catch (err) {
      status.textContent = "✗ retrain failed: " + err.message;
    } finally {
      btn.disabled = false;
    }
  });
}

/* ------------------------------------------------------------------ SSE */
function connectSSE() {
  const es = new EventSource("/api/stream");
  const conn = $("conn");

  es.onopen = () => {
    conn.textContent = "live";
    conn.className = "badge badge-on";
  };
  es.onerror = () => {
    conn.textContent = "reconnecting…";
    conn.className = "badge badge-off";
  };
  es.addEventListener("metrics", (e) => {
    renderMetrics(JSON.parse(e.data));
  });
  es.addEventListener("alert", (e) => {
    renderAlert(JSON.parse(e.data));
  });
}

/* -------------------------------------------------------------- bootstrap */
async function loadInitialAlerts() {
  try {
    const resp = await fetch("/api/alerts?limit=25");
    const data = await resp.json();
    (data.alerts || []).reverse().forEach((a) => {
      if (a.feedback !== null && a.feedback !== undefined) feedbackState.set(a.id, a.feedback);
      renderAlert(a);
    });
    if (!data.alerts || data.alerts.length === 0) {
      $("alerts").innerHTML = '<div class="empty">No alerts yet — waiting for the scorer…</div>';
    }
  } catch (err) {
    console.warn("initial alerts load failed", err);
    $("alerts").innerHTML = '<div class="empty">No alerts yet — waiting for the scorer…</div>';
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadInitialAlerts();
  loadModelPanel();
  wireRetrain();
  connectSSE();
  setInterval(loadModelPanel, 10000);
});
