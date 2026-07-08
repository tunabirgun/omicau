/* omicau local UI — single-page wizard.
   Phase 1 provides the shell + landing + navigation scaffold and the token-gated
   API helper; the wizard step content is wired to the local API in later phases. */

const TOKEN = (document.cookie.match(/omicau_token=([^;]+)/) || [])[1] ||
              new URLSearchParams(location.search).get("token") || "";

async function api(path, opts = {}) {
  const headers = Object.assign({ "X-Omicau-Token": TOKEN }, opts.headers || {});
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (!r.ok) {
    let msg;
    try { msg = (await r.json()).error || r.statusText; } catch { msg = r.statusText; }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

const STEPS = [
  { key: "files", label: "Add data files" },
  { key: "roles", label: "Assign omic roles" },
  { key: "orient", label: "Confirm orientation" },
  { key: "clinical", label: "Map clinical columns" },
  { key: "align", label: "Check alignment" },
  { key: "options", label: "Options" },
  { key: "run", label: "Run audit" },
  { key: "results", label: "Results" },
];

const state = {
  started: false,
  step: 0,
  session: null,     // session id from the server
  files: [],         // [{name, role, confidence, ...}]
  clinical: {},      // {target, group, batch, sample_id}
  runName: "",
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
};

function renderRail() {
  const rail = $("#rail");
  rail.innerHTML = "";
  document.querySelector(".app-shell").classList.toggle("solo", !state.started);
  if (!state.started) { rail.style.display = "none"; return; }
  rail.style.display = "";
  rail.append(el("h2", {}, "Audit steps"));
  STEPS.forEach((s, i) => {
    const cls = ["step-item"];
    if (i === state.step) cls.push("active");
    else if (i < state.step) cls.push("done");
    else cls.push("disabled");
    rail.append(el("div", { class: cls.join(" ") },
      el("span", { class: "num" }, i < state.step ? "✓" : String(i + 1)),
      el("span", {}, s.label)));
  });
}

// Step renderers are registered here; later phases fill in the wizard bodies.
const RENDERERS = {};

function renderContent() {
  const c = $("#content");
  c.innerHTML = "";
  if (!state.started) { c.append(renderLanding()); return; }
  const step = STEPS[state.step];
  const fn = RENDERERS[step.key];
  if (fn) { c.append(fn()); return; }
  c.append(el("h1", {}, step.label),
    el("p", { class: "lead" }, "This step is part of the wizard build."));
}

function renderLanding() {
  const wrap = el("div");
  wrap.append(el("h1", {}, "Audit your multi-omic dataset"));
  wrap.append(el("p", { class: "lead" },
    "Point omicau at your own files — one matrix per omic layer plus a clinical " +
    "table — and it runs the full leakage-safe audit and opens the interactive " +
    "report. Everything runs on this computer; no data is uploaded."));

  const card = el("div", { class: "hero-card" });
  card.append(el("span", { class: "eyebrow" }, "What omicau does"));
  const row = el("div", { class: "feature-row" });
  [
    ["Adversarial hygiene", "Tests for batch effects, target-linked missingness, and information leakage with shuffled-label controls."],
    ["Fusion benchmarks", "Group-aware cross-validated classical + neural models, with each layer's marginal gain and redundancy."],
    ["Provenance", "A value-level SHA-256 fingerprint of your exact inputs, so the report is tied to the data."],
  ].forEach(([k, v]) => row.append(el("div", { class: "feature" },
    el("div", { class: "k" }, k), el("div", { class: "v" }, v))));
  card.append(row);

  const btnRow = el("div", { class: "btn-row" });
  const start = el("button", { class: "btn-primary", onclick: startAudit }, "Start a new audit");
  btnRow.append(start, el("span", { class: "spacer" }),
    el("span", { class: "note-line", id: "health" }, ""));
  card.append(btnRow);
  wrap.append(card);
  return wrap;
}

async function startAudit() {
  try {
    // Later phases create a server-side session; the shell advances regardless.
    try { const s = await api("/api/session", { method: "POST" }); state.session = s.session; }
    catch (_) { /* session route arrives with the wizard build */ }
    state.started = true;
    state.step = 0;
    render();
  } catch (e) { alert("Could not start: " + e.message); }
}

function render() { renderRail(); renderContent(); }

async function init() {
  render();
  try {
    const h = await api("/api/health");
    const n = $("#health");
    if (n) n.textContent = `omicau v${h.version} · ready`;
  } catch (_) { /* health is best-effort */ }
}

init();
