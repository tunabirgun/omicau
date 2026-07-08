/* omicau local UI — single-page wizard.
   Every screen writes one part of the same config the CLI consumes and calls the
   identical run_audit; the UI never re-implements the science. All data stays on
   the machine (localhost + one-time token). */

const TOKEN = (document.cookie.match(/omicau_token=([^;]+)/) || [])[1] ||
              new URLSearchParams(location.search).get("token") || "";
// The server already set the token as a cookie; drop it from the URL so it is
// not retained in browser history or sent as a Referer.
if (new URLSearchParams(location.search).get("token")) {
  try { history.replaceState(null, "", location.pathname); } catch (_) { /* ignore */ }
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (!(opts.body instanceof FormData)) headers["X-Omicau-Token"] = TOKEN;
  else headers["X-Omicau-Token"] = TOKEN;
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || j.error || r.statusText; } catch { /* keep statusText */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.text();
}

const ROLE_OPTIONS = ["rna", "protein", "methylation", "mirna", "cnv", "metabolomics",
                      "mutation", "clinical", "other"];

// Public data hubs surfaced in the wizard (same connectors the CLI `bootstrap` uses).
const HUBS = [
  { id: "mock", label: "Synthetic demo — offline, instant", param: null },
  { id: "tcga", label: "TCGA / cBioPortal (human)", param: "study", hint: "study id, e.g. laml_tcga" },
  { id: "ccle", label: "CCLE / DepMap cell lines", param: "target", hint: "gene, e.g. SOX10" },
  { id: "cptac", label: "CPTAC proteogenomics (human)", param: "cancer", hint: "cohort, e.g. Ucec" },
  { id: "openpbta", label: "OpenPBTA pediatric brain (human)", param: "target", hint: "e.g. broad_histology" },
  { id: "xena", label: "UCSC Xena (human)", param: "preset", hint: "preset, e.g. brca" },
  { id: "metabolomics", label: "Metabolomics Workbench", param: "study", hint: "e.g. ST000009" },
  { id: "expression_atlas", label: "EMBL-EBI Expression Atlas — any organism", param: "study", hint: "accession, e.g. E-GEOD-100100" },
];

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
  options: { run_name: "my_audit", n_splits: 5, neural: true,
             batch_blocked: false, batch_adjust_sensitivity: false, normalization: "none",
             llm: { enabled: false, provider: "anthropic", model: "claude-sonnet-5", base_url: "" } },
  align: null, preflight: null, run: null,
};

// The AI model API key lives ONLY in this in-memory variable — never in `state`
// that gets POSTed to /options, never persisted. It travels once, in the /run body.
let LLM_API_KEY = "";

// Friendly provider catalogue: label + default model + where to get a key.
const PROVIDERS = {
  anthropic: { label: "Claude (Anthropic)", model: "claude-sonnet-5", needsKey: true,
    keyUrl: "https://console.anthropic.com/settings/keys",
    keyLabel: "Anthropic API key", where: "console.anthropic.com → API Keys" },
  openai: { label: "ChatGPT (OpenAI)", model: "gpt-5.5", needsKey: true,
    keyUrl: "https://platform.openai.com/api-keys",
    keyLabel: "OpenAI API key", where: "platform.openai.com → API keys" },
  gemini: { label: "Gemini (Google)", model: "gemini-3.5-flash", needsKey: true,
    keyUrl: "https://aistudio.google.com/apikey",
    keyLabel: "Google AI Studio API key", where: "aistudio.google.com → Get API key" },
  local: { label: "Local model on this computer (Ollama / LM Studio)", model: "llama3.1",
    needsKey: false, base_url: "http://localhost:11434/v1",
    where: "install Ollama, then run: ollama pull llama3.1 — nothing leaves your machine" },
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
  if (!localStorage.getItem("omicau_ack")) return renderAck();
  try {
    const s = await api("/api/session", { method: "POST" });
    state.session = s.session; state.started = true; state.step = 0;
    render();
  } catch (e) { alert("Could not start: " + e.message); }
}

function renderAck() {
  const c = $("#content"); c.innerHTML = "";
  const card = el("div", { class: "hero-card" });
  card.append(el("span", { class: "eyebrow" }, "Before you begin"));
  card.append(el("h1", { style: "margin:8px 0 0" }, "Research use only"));
  card.append(el("p", { class: "lead" },
    "omicau audits data for predictive signal and data hygiene. It is not a " +
    "diagnostic device: it does not diagnose disease, recommend treatment, or " +
    "assess an individual patient's risk. Use its output to judge data quality " +
    "and the value of each omic layer for research — not for clinical decisions."));
  const cb = el("input", { type: "checkbox" });
  const btn = el("button", { class: "btn-primary", disabled: true, onclick: () => {
    localStorage.setItem("omicau_ack", new Date().toISOString()); startAudit();
  } }, "I understand — continue");
  cb.addEventListener("change", () => (btn.disabled = !cb.checked));
  const label = el("label", { style: "display:flex;gap:10px;align-items:center;margin:18px 0;font-size:15px" },
    cb, "I understand this is a research tool, not a diagnostic device.");
  card.append(label, el("div", { class: "btn-row" }, btn));
  c.append(el("h1", {}, "Audit your multi-omic dataset"), card);
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
  wrap.append(hubPanel());
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

// Load a ready-made public cohort instead of uploading files. Uses the identical
// connectors the CLI `bootstrap` command uses, so the two paths never diverge.
function hubPanel() {
  const card = el("div", { class: "hero-card", style: "margin-top:20px" });
  card.append(el("span", { class: "eyebrow" }, "No files handy? Load a public dataset"));
  card.append(el("p", { class: "consequence", style: "margin:8px 0 4px" },
    "Pull a ready-to-audit cohort from a public data hub — same data the command-line " +
    "tool uses. The synthetic demo is offline and instant; the others download on first use."));
  let hub = HUBS[0];
  const sel = el("select", { style: "min-width:340px" });
  HUBS.forEach((h) => sel.append(el("option", { value: h.id }, h.label)));
  const paramIn = el("input", { type: "text", style: "min-width:220px" });
  const paramWrap = el("span", {}, paramIn);
  const status = el("div", { class: "consequence", id: "hubstatus" });
  const btn = el("button", { class: "btn-primary" }, "Load dataset");

  const refresh = () => {
    hub = HUBS.find((h) => h.id === sel.value) || HUBS[0];
    paramWrap.style.display = hub.param ? "" : "none";
    paramIn.placeholder = hub.hint || "";
    paramIn.value = "";
  };
  sel.addEventListener("change", refresh);
  refresh();

  btn.addEventListener("click", async () => {
    const payload = { dataset: hub.id };
    if (hub.param && paramIn.value.trim()) payload[hub.param] = paramIn.value.trim();
    btn.disabled = true;
    status.className = "consequence";
    status.textContent = hub.id === "mock" ? "Assembling…" : "Downloading and assembling… (large hubs may take a minute)";
    try {
      const res = await api(`/api/session/${state.session}/hub`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload) });
      state.files = res.files.map((f) => Object.assign({ orientation: "samples_as_rows" }, f));
      const c = res.clinical || {};
      state.clinical = { target: c.target || "", sample_id: c.sample_id || "",
        group: c.group || "", batch: c.batch || "", task: c.task || "auto",
        time: c.time || "", event: c.event || "" };
      state.clinicalCols = null;
      state.options.run_name = res.run_name || state.options.run_name;
      goStep(1);                                   // review roles/orientation, then continue
    } catch (e) {
      btn.disabled = false;
      status.className = "consequence msg-error";
      status.textContent = "Could not load: " + e.message;
    }
  });

  card.append(el("div", { style: "display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:10px" },
    sel, paramWrap, btn));
  card.append(status);
  return card;
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
    // Include the file's current role even if it's a hub layer name not in the preset
    // list, so the dropdown reflects reality instead of silently showing the first option.
    const opts = ROLE_OPTIONS.includes(f.role) ? ROLE_OPTIONS : [f.role, ...ROLE_OPTIONS];
    opts.forEach((r) => sel.append(el("option", { value: r, selected: r === f.role }, r)));
    sel.addEventListener("change", () => {
      f.role = sel.value;
      state.clinicalCols = null;                        // the clinical file may have changed
      state.clinical = { target: "", sample_id: "", group: "", batch: "", task: "auto" };
      validate();
    });
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
    msg.innerHTML = "";                                  // local refs work before mount too
    if (r.ok) {
      msg.append(el("div", { class: "msg-ok" }, "Roles look good — " + r.omics.join(", ") + " + clinical."));
      if (r.single_modality)
        msg.append(el("div", { class: "consequence" },
          "One omic layer detected — omicau will run its single-modality leakage-safe honesty check (fusion and redundancy need 2+ layers)."));
    }
    else r.errors.forEach((e) => msg.append(el("div", { class: "msg-error" }, e)));
    const b = foot.querySelector(".btn-primary");
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

  const survival = state.clinical.task === "survival";
  const taskSel = el("select", { style: "min-width:280px" });
  [["auto", "Auto-detect (classification / regression)"], ["classification", "Classification"],
   ["regression", "Regression"], ["survival", "Survival (time-to-event)"]].forEach(([v, t]) => {
    taskSel.append(el("option", { value: v, selected: state.clinical.task === v }, t));
  });
  taskSel.addEventListener("change", async () => {
    state.clinical.task = taskSel.value;
    await api(`/api/session/${state.session}/clinical-map`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.clinical) });
    goStep(state.step);                        // re-render to show/hide survival mappers
  });
  wrap.append(el("div", { style: "margin:18px 0" },
    el("label", { style: "display:block;font-weight:600;margin-bottom:6px" }, "Prediction task"), taskSel));

  if (survival) {
    wrap.append(colSelect("time", "Time to event / follow-up (numeric)", cols, true, null));
    wrap.append(colSelect("event", "Event indicator (1 = event, 0 = censored)", cols, true, null));
  } else {
    wrap.append(colSelect("target", "Outcome to predict (target)", cols, true, "target"));
  }
  wrap.append(colSelect("sample_id", "Sample identifier", cols, true, null));
  wrap.append(colSelect("group", "Patient / group id (optional, prevents leakage)", cols, false, "group"));
  wrap.append(colSelect("batch", "Batch / site (optional)", cols, false, "batch"));

  const nextOk = () => state.clinical.sample_id && (survival
    ? (state.clinical.time && state.clinical.event)
    : state.clinical.target);
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
  const blockIn = el("input", { type: "checkbox" });
  if (o.batch_blocked) blockIn.setAttribute("checked", "checked");
  blockIn.addEventListener("change", () => (o.batch_blocked = blockIn.checked));
  const adjIn = el("input", { type: "checkbox" });
  if (o.batch_adjust_sensitivity) adjIn.setAttribute("checked", "checked");
  adjIn.addEventListener("change", () => (o.batch_adjust_sensitivity = adjIn.checked));
  const normSel = el("select");
  [["none", "Keep IDs verbatim (safest; non-human / case-sensitive)"],
   ["tcga", "TCGA mode: uppercase + collapse aliquot barcodes to patient"]].forEach(([v, t]) => {
    const opt = el("option", { value: v }, t);
    if (o.normalization === v) opt.setAttribute("selected", "selected");
    normSel.append(opt);
  });
  normSel.addEventListener("change", () => (o.normalization = normSel.value));
  box.append(
    field("Audit name", nameIn),
    field("Cross-validation folds", splitsIn),
    field("Neural fusion model (slower, adds a deep-learning benchmark)", neuralIn),
    field("Sample-ID matching", normSel),
    field("Cross-site stress test (batch-blocked CV; needs a batch column, 3+ batches)", blockIn),
    field("Batch-adjustment sensitivity probe (in-fold; auto-skipped if batch is confounded)", adjIn));
  wrap.append(box);
  wrap.append(el("p", { class: "consequence", style: "margin-top:10px" },
    "The stress test and batch-adjustment probe are optional robustness checks and need a mapped batch column. The probe never corrects your data — it reports alongside the standard result, and omicau refuses to run it when batch is confounded with the outcome."));
  wrap.append(llmCard(o));
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

// LLM verdict card (optional) — shown at the bottom of Options.
function llmCard(o) {
  const card = el("div", { class: "hero-card", style: "margin-top:20px" });
  card.append(el("span", { class: "eyebrow" }, "Plain-language summary — optional"));
  card.append(el("p", { class: "lead", style: "font-size:15px;margin:8px 0 0" },
    "Turn the audit numbers into a short written verdict using an AI model you choose — " +
    "Claude, ChatGPT, Gemini, or a model running locally on this computer."));
  card.append(el("p", { class: "consequence", style: "margin-top:8px" },
    "Only the audit's summary statistics are sent to the model — never your raw data. " +
    "If you skip this, omicau still writes its built-in rule-based summary."));
  const enable = el("input", { type: "checkbox", id: "llm-enable" });
  if (o.llm.enabled) enable.setAttribute("checked", "checked");
  card.append(el("label", { style: "display:flex;gap:10px;align-items:center;font-weight:600;margin-top:14px" },
    enable, "Add an AI plain-English verdict"));
  const body = el("div", { id: "llm-body", style: "margin-top:6px" });
  card.append(body);
  const rebuild = () => { body.innerHTML = ""; if (o.llm.enabled) fillLlmBody(body, o); };
  enable.addEventListener("change", () => { o.llm.enabled = enable.checked; rebuild(); });
  rebuild();
  return card;
}

function fillLlmBody(body, o) {
  const row = (label, help, control) => el("div", { style: "margin:16px 0" },
    el("label", { style: "display:block;font-weight:600;margin-bottom:4px" }, label),
    help ? el("div", { class: "note-line", style: "margin-bottom:6px" }, help) : null, control);

  // Provider picker
  const provSel = el("select", { style: "min-width:320px" });
  Object.entries(PROVIDERS).forEach(([k, p]) =>
    provSel.append(el("option", { value: k, selected: o.llm.provider === k }, p.label)));
  provSel.addEventListener("change", () => {
    o.llm.provider = provSel.value;
    const np = PROVIDERS[o.llm.provider];
    o.llm.model = np.model;                 // sensible default; user can edit
    o.llm.base_url = np.base_url || "";
    LLM_API_KEY = "";                       // never carry a key across providers
    fillLlmBody(body, o);                   // re-render for the new provider
  });
  const p = PROVIDERS[o.llm.provider] || PROVIDERS.anthropic;

  // Model id (editable)
  const modelIn = el("input", { type: "text", value: o.llm.model, style: "min-width:320px" });
  modelIn.addEventListener("input", () => (o.llm.model = modelIn.value));

  body.innerHTML = "";
  body.append(row("Which service", null, provSel));
  body.append(row("Model", "Paste any model name your account supports.", modelIn));

  if (p.needsKey) {
    const keyIn = el("input", { type: "password", value: LLM_API_KEY, autocomplete: "off",
      placeholder: "paste your key here", style: "min-width:320px" });
    keyIn.addEventListener("input", () => (LLM_API_KEY = keyIn.value));
    const help = el("div", { class: "note-line", style: "margin-bottom:6px" },
      "Held in memory for this one run only — never saved, logged, or written to the report. ",
      el("a", { href: p.keyUrl, target: "_blank", rel: "noopener" }, "Get a key: " + p.where));
    body.append(el("div", { style: "margin:16px 0" },
      el("label", { style: "display:block;font-weight:600;margin-bottom:4px" }, p.keyLabel),
      help, keyIn));
  } else {
    const urlIn = el("input", { type: "text", value: o.llm.base_url || "", style: "min-width:320px" });
    urlIn.addEventListener("input", () => (o.llm.base_url = urlIn.value));
    body.append(row("Server address", "Where your local model is listening. " + p.where, urlIn));
  }
}

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
  if (state.run && state.run.status === "running") return;   // ignore a double click
  state.run = { status: "running", stages: [] };
  render();
  try {
    // No-key runs stay byte-identical to before; the key only ever rides this one body.
    const opts = LLM_API_KEY
      ? { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ api_key: LLM_API_KEY }) }
      : { method: "POST" };
    await api(`/api/session/${state.session}/run`, opts);
    poll();
  } catch (e) {
    state.run = { status: "error", stages: [], error: e.message };
    render();
  }
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
    LLM_API_KEY = "";                          // drop any key from the finished run
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
