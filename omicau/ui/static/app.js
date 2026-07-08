/* omicau local UI — single-page wizard.
   Every screen writes one part of the same config the CLI consumes and calls the
   identical run_audit; the UI never re-implements the science. All data stays on
   the machine (localhost + one-time token). */

const TOKEN = (document.cookie.match(/omicau_token=([^;]+)/) || [])[1] ||
              new URLSearchParams(location.search).get("token") || "";

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (!(opts.body instanceof FormData)) headers["X-Omicau-Token"] = TOKEN;
  else headers["X-Omicau-Token"] = TOKEN;
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (!r.ok) {
    let msg;
    try { msg = (await r.json()).detail || (await r.json()).error || r.statusText; }
    catch { msg = r.statusText; }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

const ROLE_OPTIONS = ["rna", "protein", "methylation", "mirna", "cnv", "metabolomics",
                      "mutation", "clinical", "other"];

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
  started: false, step: 0, session: null,
  files: [],                                   // upload info per file (+ role, orientation)
  rolesOk: false,
  clinicalCols: null,
  clinical: { target: "", sample_id: "", group: "", batch: "", task: "auto" },
  options: { run_name: "my_audit", n_splits: 5, neural: true },
  align: null, preflight: null, run: null,
};

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false)
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
};

const omics = () => state.files.filter((f) => f.role !== "clinical");
const clinicalFile = () => state.files.find((f) => f.role === "clinical");

// --------------------------------------------------------------------------- //
// Step rail + dispatch
// --------------------------------------------------------------------------- //
function renderRail() {
  const rail = $("#rail");
  rail.innerHTML = "";
  document.querySelector(".app-shell").classList.toggle("solo", !state.started);
  if (!state.started) { rail.style.display = "none"; return; }
  rail.style.display = "";
  rail.append(el("h2", {}, "Audit steps"));
  STEPS.forEach((s, i) => {
    const cls = ["step-item", i === state.step ? "active" : i < state.step ? "done" : "disabled"];
    rail.append(el("div", { class: cls.join(" ") },
      el("span", { class: "num" }, i < state.step ? "✓" : String(i + 1)),
      el("span", {}, s.label)));
  });
}

async function renderContent() {
  const c = $("#content");
  c.innerHTML = "";
  if (!state.started) { c.append(renderLanding()); return; }
  c.append(el("p", { class: "loading note-line" }, "Loading…"));
  try {
    const node = await RENDERERS[STEPS[state.step].key]();
    c.innerHTML = ""; c.append(node);
  } catch (e) {
    c.innerHTML = "";
    c.append(el("h1", {}, STEPS[state.step].label),
      el("div", { class: "msg-error" }, e.message), footer({ back: true }));
  }
}

function render() { renderRail(); return renderContent(); }
function goStep(i) { state.step = Math.max(0, Math.min(STEPS.length - 1, i)); return render(); }

function footer({ back = true, next = null, nextLabel = "Next", nextOk = true, extra = null } = {}) {
  const row = el("div", { class: "btn-row" });
  if (back) row.append(el("button", { class: "btn-ghost",
    onclick: () => goStep(state.step - 1) }, "Back"));
  row.append(el("span", { class: "spacer" }));
  if (extra) row.append(extra);
  if (next !== false) {
    const b = el("button", { class: "btn-primary", disabled: !nextOk,
      onclick: next || (() => goStep(state.step + 1)) }, nextLabel);
    row.append(b);
  }
  return row;
}

function h(title, sub) {
  return el("div", {}, el("h1", {}, title), sub ? el("p", { class: "lead" }, sub) : null);
}

// --------------------------------------------------------------------------- //
// Landing
// --------------------------------------------------------------------------- //
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
  [["Adversarial hygiene", "Tests for batch effects, target-linked missingness, and information leakage with shuffled-label controls."],
   ["Fusion benchmarks", "Group-aware cross-validated classical + neural models, with each layer's marginal gain and redundancy."],
   ["Provenance", "A value-level SHA-256 fingerprint of your exact inputs, so the report is tied to the data."]]
    .forEach(([k, v]) => row.append(el("div", { class: "feature" },
      el("div", { class: "k" }, k), el("div", { class: "v" }, v))));
  card.append(row);
  card.append(el("div", { class: "btn-row" },
    el("button", { class: "btn-primary", onclick: startAudit }, "Start a new audit"),
    el("span", { class: "spacer" }), el("span", { class: "note-line", id: "health" }, "")));
  wrap.append(card);
  return wrap;
}

async function startAudit() {
  try {
    const s = await api("/api/session", { method: "POST" });
    state.session = s.session; state.started = true; state.step = 0;
    render();
  } catch (e) { alert("Could not start: " + e.message); }
}

// --------------------------------------------------------------------------- //
// Step 1 — files
// --------------------------------------------------------------------------- //
const RENDERERS = {};

RENDERERS.files = async () => {
  const wrap = el("div");
  wrap.append(h("Add your data files",
    "One CSV/TSV per omic layer (RNA, proteomics, methylation, …) plus one clinical table. Orientation and delimiter are detected automatically."));
  const dz = el("div", { class: "dropzone" },
    el("div", { class: "big" }, "Drop CSV/TSV files here, or click to choose"),
    el("div", { class: "note-line" }, "Files stay on this machine."));
  const input = el("input", { type: "file", multiple: true, accept: ".csv,.tsv,.txt",
    style: "display:none" });
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("hover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("hover"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("hover"); upload(e.dataTransfer.files); });
  input.addEventListener("change", () => upload(input.files));
  wrap.append(dz, input);

  const list = el("div", { class: "file-list", id: "filelist" });
  state.files.forEach((f) => list.append(fileRow(f)));
  wrap.append(list);
  wrap.append(footer({ back: true, nextOk: state.files.length > 0 }));
  return wrap;

  async function upload(fileList) {
    if (!fileList || !fileList.length) return;
    const fd = new FormData();
    for (const f of fileList) fd.append("files", f);
    try {
      const res = await api(`/api/session/${state.session}/upload`, { method: "POST", body: fd });
      for (const info of res.files) {
        state.files = state.files.filter((x) => x.filename !== info.filename);
        state.files.push(Object.assign({ orientation: "samples_as_rows" }, info));
      }
      render();
    } catch (e) { alert("Upload failed: " + e.message); }
  }
};

function fileRow(f) {
  return el("div", { class: "file-row" },
    el("span", { class: "fname" }, f.filename),
    el("span", { class: "note-line" }, `${f.n_rows}×${f.n_cols} · ${f.delimiter}`),
    el("span", { class: "chip " + (f.confidence === "high" ? "hi" : "lo") },
      f.role + (f.confidence === "high" ? "" : "?")));
}

// --------------------------------------------------------------------------- //
// Step 2 — roles
// --------------------------------------------------------------------------- //
RENDERERS.roles = async () => {
  const wrap = el("div");
  wrap.append(h("Assign an omic role to each file",
    "We guessed from each file name — confirm or change. Exactly one file must be the clinical table; each omic layer needs a distinct role."));
  const list = el("div", { class: "file-list" });
  state.files.forEach((f) => {
    const sel = el("select");
    ROLE_OPTIONS.forEach((r) => sel.append(el("option", { value: r, selected: r === f.role }, r)));
    sel.addEventListener("change", () => { f.role = sel.value; validate(); });
    list.append(el("div", { class: "file-row" },
      el("span", { class: "fname" }, f.filename),
      el("span", { class: "chip " + (f.confidence === "high" ? "hi" : "lo") },
        "guess: " + f.role),
      sel));
  });
  wrap.append(list);
  const msg = el("div", { id: "rolemsg" });
  wrap.append(msg);
  const foot = footer({ back: true, nextOk: state.rolesOk, next: advance });
  wrap.append(foot);
  await validate();
  return wrap;

  async function validate() {
    const roles = Object.fromEntries(state.files.map((f) => [f.filename, f.role]));
    const r = await api(`/api/session/${state.session}/roles`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roles }) });
    state.rolesOk = r.ok;
    const m = $("#rolemsg"); if (m) {
      m.innerHTML = "";
      if (r.ok) m.append(el("div", { class: "msg-ok" }, "Roles look good — " + r.omics.join(", ") + " + clinical."));
      else r.errors.forEach((e) => m.append(el("div", { class: "msg-error" }, e)));
    }
    const b = document.querySelector(".btn-row .btn-primary");
    if (b) b.disabled = !r.ok;
  }
  async function advance() {
    await api(`/api/session/${state.session}/orient`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ orientation: Object.fromEntries(state.files.map((f) => [f.filename, f.orientation])) }) });
    goStep(state.step + 1);
  }
};

// --------------------------------------------------------------------------- //
// Step 3 — orientation
// --------------------------------------------------------------------------- //
RENDERERS.orient = async () => {
  const wrap = el("div");
  wrap.append(h("Confirm the orientation of each matrix",
    "omicau needs samples as rows. We auto-detect it; confirm the preview looks right, or flip. A silent wrong axis corrupts everything, so this is worth a glance."));
  omics().forEach((f) => {
    const card = el("div", { class: "hero-card", style: "margin-top:16px" });
    const flip = f.orientation === "samples_as_cols";
    card.append(el("div", { style: "display:flex;align-items:center;gap:16px" },
      el("span", { class: "chip hi" }, f.role), el("span", { class: "fname" }, f.filename),
      el("span", { class: "spacer", style: "flex:1" }),
      el("span", { class: "note-line" },
        flip ? "samples = columns" : "samples = rows"),
      el("button", { class: "btn-ghost", onclick: () => {
        f.orientation = flip ? "samples_as_rows" : "samples_as_cols"; render();
      } }, "Flip rows/columns")));
    card.append(previewTable(f, flip));
    card.append(el("div", { class: "consequence" },
      flip ? `Sample ids taken from the column headers (${(f.col_labels || []).slice(0, 3).join(", ")}…).`
           : `Sample ids taken from the first column (${(f.row_labels || []).slice(0, 3).join(", ")}…).`));
    wrap.append(card);
  });
  wrap.append(footer({ back: true, next: advance }));
  return wrap;

  async function advance() {
    await api(`/api/session/${state.session}/orient`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ orientation: Object.fromEntries(state.files.map((f) => [f.filename, f.orientation])) }) });
    goStep(state.step + 1);
  }
};

function previewTable(f, flip) {
  const box = el("div", { class: "preview-table", style: "margin-top:12px" });
  const table = el("table");
  const head = f.header || [];
  table.append(el("tr", {}, ...head.map((hh) => el("th", {}, hh))));
  (f.preview || []).forEach((r) => table.append(el("tr", {}, ...r.map((cell, i) =>
    el("td", {}, cell)))));
  box.append(table);
  return box;
}

// --------------------------------------------------------------------------- //
// Step 4 — clinical mapping
// --------------------------------------------------------------------------- //
RENDERERS.clinical = async () => {
  if (!clinicalFile()) throw new Error("No clinical table assigned — go back to roles.");
  if (!state.clinicalCols)
    state.clinicalCols = (await api(`/api/session/${state.session}/clinical`)).columns;
  const cols = state.clinicalCols;
  const idGuess = cols.find((c) => c.looks_like_id);
  if (!state.clinical.sample_id && idGuess) state.clinical.sample_id = idGuess.name;

  const wrap = el("div");
  wrap.append(h("Map the clinical columns",
    "Tell omicau which column is the outcome to predict, which identifies the sample, and (recommended) which groups repeated samples so cross-validation stays leakage-safe."));

  wrap.append(colSelect("target", "Outcome to predict (target)", cols, true, "target"));
  wrap.append(colSelect("sample_id", "Sample identifier", cols, true, null));
  wrap.append(colSelect("group", "Patient / group id (optional, prevents leakage)", cols, false, "group"));
  wrap.append(colSelect("batch", "Batch / site (optional)", cols, false, "batch"));

  const nextOk = () => state.clinical.target && state.clinical.sample_id;
  wrap.append(footer({ back: true, nextOk: nextOk(), next: advance }));
  // fire initial consequences
  ["target", "group", "batch"].forEach((k) => { if (state.clinical[k]) showConsequence(k); });
  return wrap;

  function colSelect(key, label, columns, required, conseqKind) {
    const box = el("div", { style: "margin:18px 0" });
    box.append(el("label", { style: "display:block;font-weight:600;margin-bottom:6px" },
      label + (required ? " *" : "")));
    const sel = el("select", { style: "min-width:280px" });
    sel.append(el("option", { value: "" }, required ? "— choose a column —" : "— none —"));
    columns.forEach((c) => sel.append(el("option", { value: c.name,
      selected: state.clinical[key] === c.name }, `${c.name}  (${c.kind}, ${c.n_unique} unique)`)));
    sel.addEventListener("change", async () => {
      state.clinical[key] = sel.value;
      await api(`/api/session/${state.session}/clinical-map`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.clinical) });
      if (conseqKind) showConsequence(key);
      const b = document.querySelector(".btn-row .btn-primary");
      if (b) b.disabled = !nextOk();
    });
    box.append(sel, el("div", { class: "consequence", id: "conseq-" + key }));
    return box;
  }

  async function showConsequence(key) {
    const node = $("#conseq-" + key); if (!node) return;
    node.textContent = "…";
    const kind = key === "sample_id" ? null : key;
    const params = new URLSearchParams({ column: state.clinical[key], kind: kind || "target" });
    if (key === "batch" && state.clinical.target) params.set("target", state.clinical.target);
    const c = await api(`/api/session/${state.session}/consequence?` + params.toString());
    node.textContent = c.message || "";
    node.className = "consequence" + (c.ok === false ? " msg-error" : "");
  }

  async function advance() {
    await api(`/api/session/${state.session}/clinical-map`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.clinical) });
    goStep(state.step + 1);
  }
};

// --------------------------------------------------------------------------- //
// Step 5 — alignment
// --------------------------------------------------------------------------- //
RENDERERS.align = async () => {
  const rep = await api(`/api/session/${state.session}/align`, { method: "POST" });
  state.align = rep;
  const wrap = el("div");
  wrap.append(h("Alignment check",
    "How many samples match across every layer and the clinical table. omicau flags and drops unmatched ids — it never edits your files."));
  const pct = Math.round(100 * rep.matched_all_layers / Math.max(1, rep.n_clinical));
  wrap.append(el("div", { class: "hero-card", style: "margin-top:8px" },
    el("div", { style: "font-size:26px;font-weight:600" },
      `${rep.matched_all_layers} of ${rep.n_clinical} samples matched across all layers (${pct}%)`),
    el("div", { class: "progress-bar", style: "margin-top:12px" },
      el("span", { style: `width:${pct}%;background:${pct >= 80 ? "var(--teal)" : pct >= 50 ? "var(--amber)" : "var(--vermillion)"}` })),
    ...rep.per_layer.map((l) => el("div", { class: "consequence" },
      `${l.name}: ${l.overlap_with_clinical}/${l.n_samples} overlap`
      + (l.unmatched_examples.length ? ` · unmatched e.g. ${l.unmatched_examples.slice(0, 3).join(", ")}` : ""))),
    rep.hint ? el("div", { class: "msg-error", style: "margin-top:14px" }, rep.hint) : null));
  wrap.append(footer({ back: true, nextOk: rep.ok,
    nextLabel: rep.ok ? "Next" : "Too few matches" }));
  return wrap;
};

// --------------------------------------------------------------------------- //
// Step 6 — options
// --------------------------------------------------------------------------- //
RENDERERS.options = async () => {
  const wrap = el("div");
  wrap.append(h("Options", "Sensible defaults are set; adjust if you like."));
  const o = state.options;
  const box = el("div", { class: "hero-card", style: "margin-top:8px" });
  const nameIn = el("input", { type: "text", value: o.run_name, style: "min-width:280px" });
  nameIn.addEventListener("input", () => (o.run_name = nameIn.value));
  const splitsIn = el("input", { type: "number", min: "2", max: "10", value: o.n_splits, style: "width:80px" });
  splitsIn.addEventListener("input", () => (o.n_splits = parseInt(splitsIn.value || "5", 10)));
  const neuralIn = el("input", { type: "checkbox" });
  if (o.neural) neuralIn.setAttribute("checked", "checked");
  neuralIn.addEventListener("change", () => (o.neural = neuralIn.checked));
  box.append(
    field("Audit name", nameIn),
    field("Cross-validation folds", splitsIn),
    field("Neural fusion model (slower, adds a deep-learning benchmark)", neuralIn));
  wrap.append(box);
  wrap.append(footer({ back: true, next: advance }));
  return wrap;

  function field(label, input) {
    return el("div", { style: "margin:14px 0;display:flex;align-items:center;gap:14px" },
      el("label", { style: "min-width:340px;font-weight:600" }, label), input);
  }
  async function advance() {
    await api(`/api/session/${state.session}/options`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(o) });
    goStep(state.step + 1);
  }
};

// --------------------------------------------------------------------------- //
// Step 7 — preflight + run + progress
// --------------------------------------------------------------------------- //
RENDERERS.run = async () => {
  const wrap = el("div");
  if (state.run && state.run.status === "running") return renderProgress(wrap);
  const pf = await api(`/api/session/${state.session}/preflight`, { method: "POST" });
  state.preflight = pf;
  wrap.append(h("Ready to run", "Confirm the summary below, then start the audit."));
  const feats = Object.entries(pf.feature_counts).map(([k, v]) => `${k}: ${v}`).join(" · ");
  wrap.append(el("div", { class: "hero-card", style: "margin-top:8px" },
    el("span", { class: "eyebrow" }, "Pre-flight"),
    el("div", { class: "feature-row" },
      feat("Samples", String(pf.n_samples)),
      feat("Task", pf.task),
      feat("Estimated time", pf.cost.human_readable)),
    el("div", { class: "consequence" }, "Layers — " + feats),
    el("div", { class: "consequence", style: "margin-top:10px" },
      el("span", { class: "chip" }, "SHA-256"), " ",
      el("code", { style: "font-family:var(--mono);font-size:12px" }, pf.provenance_hash),
      el("div", {}, "This fingerprint proves the report matches these exact files.")),
    el("div", { class: "btn-row" },
      el("button", { class: "btn-ghost", onclick: () => goStep(state.step - 1) }, "Back"),
      el("span", { class: "spacer" }),
      el("button", { class: "btn-primary", onclick: startRun }, "Run audit"))));
  return wrap;

  function feat(k, v) {
    return el("div", { class: "feature" }, el("div", { class: "k" }, k), el("div", { class: "v" }, v));
  }
};

async function startRun() {
  state.run = { status: "running", stages: [] };
  await api(`/api/session/${state.session}/run`, { method: "POST" });
  render();
  poll();
}

async function renderProgress(wrap) {
  wrap.append(h("Running the audit…", "This runs on your machine; you can leave this open."));
  const box = el("div", { class: "hero-card progress-wrap", id: "progbox" });
  wrap.append(box);
  fillProgress(box);
  return wrap;
}

function fillProgress(box) {
  const r = state.run || { stages: [], status: "running" };
  box.innerHTML = "";
  box.append(el("div", { class: "progress-bar" },
    el("span", { style: `width:${r.status === "done" ? 100 : Math.min(90, r.stages.length * 12)}%` })));
  const log = el("div", { class: "stage-log" });
  r.stages.forEach((s) => log.append(el("div", { class: s.includes("·") ? "done" : "" }, s)));
  box.append(log);
  if (r.status === "error") box.append(el("div", { class: "msg-error" }, "Run failed: " + r.error));
}

async function poll() {
  try {
    const p = await api(`/api/session/${state.session}/progress`);
    state.run = { status: p.status, stages: p.stages, error: p.error,
                  report_ready: p.report_ready, provenance: p.provenance };
    const box = $("#progbox"); if (box) fillProgress(box);
    if (p.status === "done") { state.step = 7; return render(); }
    if (p.status === "error") return;
    setTimeout(poll, 1000);
  } catch (e) { setTimeout(poll, 1500); }
}

// --------------------------------------------------------------------------- //
// Step 8 — results
// --------------------------------------------------------------------------- //
RENDERERS.results = async () => {
  const wrap = el("div");
  const reportUrl = `/api/session/${state.session}/report`;
  wrap.append(el("div", { style: "display:flex;align-items:center;gap:16px" },
    el("h1", { style: "margin:0" }, "Audit complete"),
    el("span", { class: "spacer", style: "flex:1" }),
    el("button", { class: "btn-ghost", onclick: () => window.open(reportUrl, "_blank") }, "Open in browser"),
    el("button", { class: "btn-primary", onclick: newAudit }, "New audit")));
  if (state.run && state.run.provenance)
    wrap.append(el("div", { class: "consequence" },
      el("span", { class: "chip" }, "SHA-256"), " ",
      el("code", { style: "font-family:var(--mono);font-size:12px" }, state.run.provenance)));
  wrap.append(el("iframe", { class: "result-frame", src: reportUrl, title: "omicau report" }));
  return wrap;

  async function newAudit() {
    const s = await api("/api/session", { method: "POST" });
    Object.assign(state, { session: s.session, step: 0, files: [], rolesOk: false,
      clinicalCols: null, clinical: { target: "", sample_id: "", group: "", batch: "", task: "auto" },
      align: null, preflight: null, run: null });
    render();
  }
};

// --------------------------------------------------------------------------- //
async function init() {
  await render();
  try {
    const hh = await api("/api/health");
    const n = $("#health"); if (n) n.textContent = `omicau v${hh.version} · ready`;
  } catch (_) { /* best-effort */ }
}
init();
