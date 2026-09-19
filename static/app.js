/*
 * Phase 9 — Frontend (Section 21).
 *
 * Plain HTML/JS, no build step, no framework — served directly by
 * FastAPI as static files (see src/api/main.py's StaticFiles mount).
 * Talks to the Phase 7/8 API only: POST /auth/register, POST
 * /checkins, GET /checkins, GET /checkins/{id}. No new backend
 * endpoints are added for the frontend.
 *
 * The "personal history chart" is a plain SVG built from the
 * REAL check-in rows returned by GET /checkins — a rolling average
 * and a delta-vs-previous, both computed here in JS. Per Section 4,
 * this is deliberately a non-ML display statistic, not a model
 * output: the model never predicts a trend, only a single score per
 * submitted check-in.
 */
(() => {
  "use strict";

  const STORAGE_KEY = "wellpulse_api_key";

  const $ = (id) => document.getElementById(id);

  const el = {
    authSection: $("auth-section"),
    tabRegister: $("tab-register"),
    tabLogin: $("tab-login"),
    registerForm: $("register-form"),
    loginForm: $("login-form"),
    registerError: $("register-error"),
    loginError: $("login-error"),

    keyBanner: $("save-key-banner"),
    saveKeyValue: $("save-key-value"),
    copyKeyBtn: $("copy-key-btn"),
    copyConfirm: $("copy-confirm"),
    keySavedBtn: $("key-saved-btn"),

    appSection: $("app-section"),
    logoutBtn: $("logout-btn"),

    checkinForm: $("checkin-form"),
    checkinSubmitBtn: $("checkin-submit-btn"),
    checkinError: $("checkin-error"),

    resultSection: $("result-section"),
    resultHeading: $("result-heading"),
    resultScore: $("result-score"),
    resultTier: $("result-tier"),
    resultCi: $("result-ci"),
    resultExplanation: $("result-explanation"),
    resultFactors: $("result-factors"),
    resultRecommendations: $("result-recommendations"),
    resultDisclaimer: $("result-disclaimer"),

    historySection: $("history-section"),
    historyEmpty: $("history-empty"),
    trendSummary: $("trend-summary"),
    chartWrap: $("chart-wrap"),
    historyList: $("history-list"),
  };

  const TIER_LABEL = {
    low_risk: "Low risk",
    medium_risk: "Medium risk",
    high_risk: "High risk",
  };

  // ---------------------------------------------------------------
  // API helper
  // ---------------------------------------------------------------

  function apiKey() {
    return localStorage.getItem(STORAGE_KEY);
  }

  async function api(path, { method = "GET", body, auth = true } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (auth) {
      const key = apiKey();
      if (key) headers["X-API-Key"] = key;
    }
    const res = await fetch(path, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = null;
    try {
      data = await res.json();
    } catch {
      /* no body */
    }
    if (!res.ok) {
      const err = new Error(extractErrorMessage(data, res.status));
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  function extractErrorMessage(data, status) {
    if (data && Array.isArray(data.detail)) {
      // FastAPI/Pydantic 422 shape: [{loc, msg, type}, ...]
      return data.detail
        .map((d) => {
          const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .join(" · ");
    }
    if (data && typeof data.detail === "string") return data.detail;
    if (status === 401) return "That API key isn't valid.";
    if (status === 503) return "The prediction model is temporarily unavailable, try again shortly.";
    return `Something went wrong (${status}).`;
  }

  // ---------------------------------------------------------------
  // Auth
  // ---------------------------------------------------------------

  function showAuthTab(which) {
    const isRegister = which === "register";
    el.tabRegister.classList.toggle("is-active", isRegister);
    el.tabLogin.classList.toggle("is-active", !isRegister);
    el.tabRegister.setAttribute("aria-selected", String(isRegister));
    el.tabLogin.setAttribute("aria-selected", String(!isRegister));
    el.registerForm.hidden = !isRegister;
    el.loginForm.hidden = isRegister;
  }

  el.tabRegister.addEventListener("click", () => showAuthTab("register"));
  el.tabLogin.addEventListener("click", () => showAuthTab("login"));

  el.registerForm.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    el.registerError.textContent = "";
    const email = $("register-email").value.trim();
    try {
      const data = await api("/auth/register", {
        method: "POST",
        body: { email },
        auth: false,
      });
      showSaveKeyBanner(data.api_key);
    } catch (err) {
      el.registerError.textContent =
        err.status === 409
          ? "That email is already registered, switch to \u201cI have a key\u201d if you saved it earlier."
          : err.message;
    }
  });

  el.loginForm.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    el.loginError.textContent = "";
    const key = $("login-key").value.trim();
    localStorage.setItem(STORAGE_KEY, key);
    try {
      await api("/checkins"); // cheapest authenticated call as a validity check
      enterApp();
    } catch (err) {
      localStorage.removeItem(STORAGE_KEY);
      el.loginError.textContent =
        err.status === 401 ? "That key wasn't recognized." : err.message;
    }
  });

  let pendingKey = null;

  function showSaveKeyBanner(key) {
    pendingKey = key;
    el.authSection.hidden = true;
    el.keyBanner.hidden = false;
    el.saveKeyValue.textContent = key;
    el.copyConfirm.textContent = "";
  }

  el.copyKeyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(el.saveKeyValue.textContent);
      el.copyConfirm.textContent = "Copied.";
    } catch {
      el.copyConfirm.textContent = "Copy failed, select the text and copy it manually.";
    }
  });

  el.keySavedBtn.addEventListener("click", () => {
    localStorage.setItem(STORAGE_KEY, pendingKey);
    el.keyBanner.hidden = true;
    enterApp();
  });

  el.logoutBtn.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    el.appSection.hidden = true;
    el.resultSection.hidden = true;
    el.checkinForm.reset();
    showAuthTab("register");
    el.authSection.hidden = false;
  });

  // ---------------------------------------------------------------
  // Check-in submission
  // ---------------------------------------------------------------

  function readCheckinForm() {
    return {
      Age: Number($("f-age").value),
      Gender: $("f-gender").value,
      Academic_Level: $("f-academic-level").value,
      Country: $("f-country").value.trim(),
      Avg_Daily_Usage_Hours: Number($("f-usage-hours").value),
      Most_Used_Platform: $("f-platform").value,
      Sleep_Hours_Per_Night: Number($("f-sleep-hours").value),
      Relationship_Status: $("f-relationship").value,
      Conflicts_Over_Social_Media: Number($("f-conflicts").value),
    };
  }

  el.checkinForm.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    el.checkinError.textContent = "";
    el.checkinSubmitBtn.disabled = true;
    el.checkinSubmitBtn.textContent = "Scoring…";
    try {
      const payload = readCheckinForm();
      const detail = await api("/checkins", { method: "POST", body: payload });
      renderResult(detail, { heading: "Your result" });
      el.resultSection.hidden = false;
      el.resultSection.scrollIntoView?.({ behavior: "smooth", block: "start" });
      await loadHistory();
    } catch (err) {
      if (err.status === 401) {
        el.checkinError.textContent = "Your session key was rejected, please log in again.";
        setTimeout(() => el.logoutBtn.click(), 1500);
      } else {
        el.checkinError.textContent = err.message;
      }
    } finally {
      el.checkinSubmitBtn.disabled = false;
      el.checkinSubmitBtn.textContent = "Get my check-in";
    }
  });

  // ---------------------------------------------------------------
  // Result rendering (shared by a fresh submission and a past check-in)
  // ---------------------------------------------------------------

  function renderResult(detail, { heading }) {
    el.resultHeading.textContent = heading;
    el.resultScore.textContent = detail.predicted_score.toFixed(1);
    el.resultTier.textContent = TIER_LABEL[detail.risk_tier] || detail.risk_tier;
    el.resultTier.className = `tier-badge ${detail.risk_tier}`;
    const [lo, hi] = detail.confidence_interval_68pct;
    el.resultCi.textContent = `Likely range: ${lo.toFixed(1)}–${hi.toFixed(1)} (68% confidence)`;
    el.resultExplanation.textContent = detail.explanation_sentence;

    el.resultFactors.innerHTML = "";
    detail.top_factors.forEach((f) => {
      const li = document.createElement("li");
      const arrow = document.createElement("span");
      arrow.className = `factor-direction ${f.direction}`;
      arrow.textContent = f.direction === "raising" ? "\u2191" : "\u2193";
      const label = document.createElement("span");
      label.textContent = f.pretty_name;
      li.append(arrow, label);
      el.resultFactors.appendChild(li);
    });

    el.resultRecommendations.innerHTML = "";
    detail.recommendations.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      el.resultRecommendations.appendChild(li);
    });

    el.resultDisclaimer.textContent = detail.disclaimer;
  }

  async function viewPastCheckin(id, createdAt) {
    try {
      const detail = await api(`/checkins/${id}`);
      renderResult(detail, {
        heading: `Check-in from ${formatDate(createdAt)}`,
      });
      el.resultSection.hidden = false;
      el.resultSection.scrollIntoView?.({ behavior: "smooth", block: "start" });
    } catch (err) {
      el.checkinError.textContent = err.message;
    }
  }

  // ---------------------------------------------------------------
  // History + non-ML trend stat (Section 4) + SVG chart
  // ---------------------------------------------------------------

  function formatDate(iso) {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }

  function round1(n) {
    return Math.round(n * 10) / 10;
  }

  async function loadHistory() {
    const history = await api("/checkins"); // newest-first, per the API
    if (!history.length) {
      el.historySection.hidden = true;
      el.historyEmpty.hidden = false;
      return;
    }
    el.historyEmpty.hidden = true;
    el.historySection.hidden = false;

    renderTrendSummary(history);
    el.chartWrap.innerHTML = buildTrendSvg([...history].reverse());
    renderHistoryList(history);
  }

  function renderTrendSummary(history) {
    const latest = history[0];
    const parts = [
      `Latest: ${latest.predicted_score.toFixed(1)} (${TIER_LABEL[latest.risk_tier]}).`,
    ];
    if (history.length >= 2) {
      const previous = history[1];
      const delta = round1(latest.predicted_score - previous.predicted_score);
      const dir = delta > 0 ? "up" : delta < 0 ? "down" : "unchanged since";
      const magnitude = delta === 0 ? "" : ` ${Math.abs(delta).toFixed(1)}`;
      parts.push(
        delta === 0
          ? "Unchanged from your previous check-in."
          : `${dir === "up" ? "Up" : "Down"}${magnitude} from your previous check-in.`
      );
      const windowSize = Math.min(3, history.length);
      const avg =
        history.slice(0, windowSize).reduce((sum, c) => sum + c.predicted_score, 0) / windowSize;
      parts.push(`Average over your last ${windowSize} check-ins: ${avg.toFixed(1)}.`);
    }
    el.trendSummary.textContent = parts.join(" ");
  }

  function renderHistoryList(history) {
    el.historyList.innerHTML = "";
    history.forEach((c) => {
      const li = document.createElement("li");

      const left = document.createElement("span");
      left.className = "history-date";
      left.textContent = formatDate(c.created_at);

      const mid = document.createElement("span");
      mid.className = "history-score";
      mid.textContent = `${c.predicted_score.toFixed(1)} · ${TIER_LABEL[c.risk_tier]}`;

      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "View";
      btn.addEventListener("click", () => viewPastCheckin(c.id, c.created_at));

      li.append(left, mid, btn);
      el.historyList.appendChild(li);
    });
  }

  // Plain hand-built SVG line chart — no charting library, per the
  // "plain HTML/JS, MVP scope" framework decision. Y-axis is fixed to
  // the model's known 1-10 score scale (not autoscaled to the data),
  // with background bands at the same risk-tier thresholds
  // src/features/engineering.py::compute_risk_tier uses, so the
  // chart's bands and the result view's tier badge always agree.
  function buildTrendSvg(pointsChronological) {
    const W = 600;
    const H = 220;
    const padL = 34;
    const padR = 16;
    const padT = 14;
    const padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const yMin = 1;
    const yMax = 10;
    const yToPx = (y) => padT + plotH * (1 - (y - yMin) / (yMax - yMin));

    const n = pointsChronological.length;
    const xToPx = (i) => (n === 1 ? padL + plotW / 2 : padL + (plotW * i) / (n - 1));

    const bands = [
      { from: 1, to: 5, color: "var(--high-tint)" },
      { from: 5, to: 7, color: "var(--medium-tint)" },
      { from: 7, to: 10, color: "var(--low-tint)" },
    ];
    const bandRects = bands
      .map(
        (b) =>
          `<rect x="${padL}" y="${yToPx(b.to)}" width="${plotW}" height="${
            yToPx(b.from) - yToPx(b.to)
          }" fill="${b.color}" />`
      )
      .join("");

    const tierColor = { low_risk: "var(--low)", medium_risk: "var(--medium)", high_risk: "var(--high)" };

    const linePoints = pointsChronological
      .map((c, i) => `${xToPx(i)},${yToPx(c.predicted_score)}`)
      .join(" ");

    const dots = pointsChronological
      .map(
        (c, i) =>
          `<circle cx="${xToPx(i)}" cy="${yToPx(c.predicted_score)}" r="4" fill="${
            tierColor[c.risk_tier] || "var(--accent)"
          }" stroke="var(--surface)" stroke-width="1.5" />`
      )
      .join("");

    const yLabels = [1, 5, 7, 10]
      .map(
        (v) =>
          `<text x="${padL - 8}" y="${yToPx(v) + 4}" text-anchor="end" font-size="10" fill="var(--text-muted)">${v}</text>`
      )
      .join("");

    // Only label the first and last date to avoid crowding.
    const xLabels = [0, n - 1]
      .filter((i, idx, arr) => arr.indexOf(i) === idx)
      .map(
        (i) =>
          `<text x="${xToPx(i)}" y="${H - 8}" text-anchor="${
            i === 0 ? "start" : "end"
          }" font-size="10" fill="var(--text-muted)">${formatDate(pointsChronological[i].created_at)}</text>`
      )
      .join("");

    return `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted wellbeing score over time">
        ${bandRects}
        ${yLabels}
        <polyline points="${linePoints}" fill="none" stroke="var(--accent-dark)" stroke-width="2" />
        ${dots}
        ${xLabels}
      </svg>
    `;
  }

  // ---------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------

  async function enterApp() {
    el.authSection.hidden = true;
    el.keyBanner.hidden = true;
    el.appSection.hidden = false;
    try {
      await loadHistory();
    } catch {
      /* non-fatal — the check-in form still works without history */
    }
  }

  async function init() {
    const key = apiKey();
    if (!key) {
      el.authSection.hidden = false;
      return;
    }
    try {
      await api("/checkins");
      enterApp();
    } catch {
      localStorage.removeItem(STORAGE_KEY);
      el.authSection.hidden = false;
    }
  }

  init();
})();
