"""Interactive HTML dashboard and machine-readable asset compiler.

Produces a single self-contained ``.html`` file that is fully offline (Plotly
bundled inline; IBM Plex Sans + IBM Plex Mono embedded as @font-face data URIs),
styled with a humanist sans aesthetic and a color-blind-safe Okabe-Ito palette. Every data grid is sortable, text-filterable, and CSV/TSV-exportable via
dependency-free vanilla JavaScript. The workflow flowchart is embedded as a
scaling inline SVG. Alongside the dashboard it writes a raw JSON metadata object
and flat CSV summaries.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Template

from omicau.reporting._assets import (
    DASHBOARD_CSS, TOOLTIP_JS, GLOSSARY, SECTION_COPY, BADGES, FONT_FACES,
)

# --------------------------------------------------------------------------- #
# Color-blind-safe palette (Okabe-Ito) + semantic mapping
# --------------------------------------------------------------------------- #
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",   # cobalt -> standard / optimal
    "vermillion": "#D55E00",  # burnt orange -> warning / confounder
    "purple": "#CC79A7",
}
COBALT = OKABE_ITO["blue"]
VERMILLION = OKABE_ITO["vermillion"]
SLATE = "#4477AA"
TEAL = OKABE_ITO["green"]
AMBER = OKABE_ITO["orange"]
INK = "#1A202C"
BORDER = "#E2E8F0"

PLOTLY_FONT = "IBM Plex Sans, -apple-system, system-ui, sans-serif"
PLOTLY_CONFIG = {"responsive": True, "displaylogo": False,
                 "toImageButtonOptions": {"format": "svg", "scale": 2}}


# --------------------------------------------------------------------------- #
# Plain-language badge + answer-strip helpers (from the design system)
# --------------------------------------------------------------------------- #
def _verdict_status(verdict: str) -> str:
    v = (verdict or "").lower()
    if v.startswith("predictive"):
        return "predictive"
    if v.startswith("redundant"):
        return "redundant"
    if "batch-confounded" in v:
        return "batch_confounded"
    if "no detectable" in v or "control-like" in v:
        return "control_like"
    return "redundant"  # "informative but non-additive" reads as neutral


def _rating_status(rating: str) -> str:
    r = (rating or "").lower()
    if "high" in r:
        return "high"
    if "moderate" in r:
        return "moderate"
    return "clean"


def _badge(status: str) -> dict:
    return BADGES.get(status, {"label": status, "icon": "•", "css_class": "badge--neutral"})


def _answer_strip(util: dict, rating_status: str) -> list[dict]:
    gain = util.get("fusion_gain_over_best_single")
    leak = util.get("leakage_warning")
    ledger = util.get("modality_ledger", [])
    confounded = [m["modality"] for m in ledger if m.get("batch_confounded")]
    dead = [m["modality"] for m in ledger if "no detectable" in str(m.get("verdict", ""))]
    useful = [m["modality"] for m in ledger if str(m.get("verdict", "")).startswith("predictive")]

    if isinstance(gain, (int, float)) and gain > 0.02:
        q1 = ("Yes — fusion adds signal", "✚", "answer--added")
    elif isinstance(gain, (int, float)) and gain > 0.005:
        q1 = ("Marginally", "△", "answer--mid")
    else:
        q1 = ("No — one layer suffices", "•", "answer--neutral")

    if leak or rating_status == "high":
        q2 = ("No — serious flags", "▲", "answer--warn")
    elif rating_status == "moderate":
        q2 = ("With caveats", "△", "answer--mid")
    else:
        q2 = ("Yes — clean", "✓", "answer--ok")

    if leak:
        q3 = ("Re-check CV splits for leakage", "▲", "answer--warn")
    elif confounded:
        q3 = (f"Distrust the batch-confounded layer: {confounded[0]}", "▲", "answer--warn")
    elif dead:
        q3 = (f"Consider dropping {dead[0]}", "△", "answer--mid")
    elif useful and isinstance(gain, (int, float)) and gain > 0.005:
        q3 = ("Adopt the fusion model", "✓", "answer--ok")
    else:
        q3 = ("Prefer the best single layer", "•", "answer--neutral")

    return [
        {"q": "Does combining layers help?", "a": q1[0], "icon": q1[1], "cls": q1[2]},
        {"q": "Is the data trustworthy?", "a": q2[0], "icon": q2[1], "cls": q2[2]},
        {"q": "What should you do next?", "a": q3[0], "icon": q3[1], "cls": q3[2]},
    ]


def _trust_checklist(audit: dict, util: dict, missing: dict, batch: dict,
                     control_max) -> list[dict]:
    """A pass/caution/fail 'run report card' so non-experts trust the result
    only as far as the data warrants."""
    ds = audit.get("dataset", {})
    task = ds.get("task")
    n = ds.get("n_samples")
    ic = {"pass": "✓", "caution": "△", "fail": "▲"}
    checks: list[dict] = []

    leak = util.get("leakage_warning")
    ctrl = (f"shuffled-label control scored {control_max:.2f}"
            if control_max is not None else "controls unavailable")
    checks.append({"label": "Control baselines at chance (no leakage)",
                   "status": "fail" if leak else "pass",
                   "note": ctrl + (" — above chance; investigate leakage before trusting the model."
                                   if leak else ", so a random target is not predictable — as it should be.")})

    if n is not None:
        ss = "pass" if n >= 40 else "caution"
        checks.append({"label": "Adequate sample size", "status": ss,
                       "note": (f"{n} aligned samples." if ss == "pass"
                                else f"only {n} samples — estimates are unstable and easily over-fit.")})

    if task == "classification":
        bal = ds.get("class_balance") or {}
        vals = [v for v in bal.values() if isinstance(v, (int, float))]
        if vals:
            ok = min(vals) >= 0.4 * max(vals)
            checks.append({"label": "Balanced outcome classes",
                           "status": "pass" if ok else "caution",
                           "note": (f"classes are reasonably balanced ({min(vals)}–{max(vals)})." if ok
                                    else f"imbalanced (smallest {min(vals)}, largest {max(vals)}); read AUPRC / balanced accuracy, not raw accuracy.")})

    conf = [m["modality"] for m in util.get("modality_ledger", []) if m.get("batch_confounded")]
    checks.append({"label": "No batch confounding",
                   "status": "caution" if conf else "pass",
                   "note": (f"batch is confounded with the outcome in: {', '.join(conf)} — treat their apparent signal as untrustworthy."
                            if conf else "no layer's signal is dominated by batch structure.")})

    flagged = [t for t in missing.get("tests", []) if t.get("flag")]
    checks.append({"label": "No target-linked missingness",
                   "status": "caution" if flagged else "pass",
                   "note": (f"{len(flagged)} test(s) show missingness associated with the outcome (possible MNAR bias)."
                            if flagged else "missing values look unrelated to the outcome.")})

    for c in checks:
        c["icon"] = ic[c["status"]]
    return checks


# --------------------------------------------------------------------------- #
# Workflow flowchart (inline SVG + Mermaid source)
# --------------------------------------------------------------------------- #
FLOWCHART_STEPS = [
    ("Multi-modal ingestion", "auto-delimiter, orientation, fuzzy sample-name match", COBALT),
    ("Alignment & masking", "sample intersection, drop missing endpoints, NaN masks", COBALT),
    ("Provenance SHA-256", "immutable hash of sample index + feature footprints", TEAL),
    ("Cost / runtime estimate", "N x P_m, K folds, E epochs, device, cores", AMBER),
    ("Nested group-aware CV", "impute + scale + select fitted inside train folds only", COBALT),
    ("Fusion benchmarks", "classical concat + masked global-pooling neural network", SLATE),
    ("Leakage-safe XAI", "permutation importance on held-out folds", SLATE),
    ("Utility & redundancy audit", "marginal gain, CKA, batch/missingness/control gates", TEAL),
    ("Dual reporting", "clinical + research dashboard, multi-format docs", COBALT),
]


def flowchart_svg(width: int = 900) -> str:
    """Render the end-to-end pipeline as a responsive inline SVG (offline-safe)."""
    box_w, box_h, gap = 640, 62, 30
    pad_top, pad_side = 24, (width - box_w) // 2
    n = len(FLOWCHART_STEPS)
    height = pad_top * 2 + n * box_h + (n - 1) * gap
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="omicau workflow" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="{PLOTLY_FONT}" style="max-width:100%;height:auto;">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF"/>',
    ]
    for i, (title, sub, color) in enumerate(FLOWCHART_STEPS):
        y = pad_top + i * (box_h + gap)
        cx = pad_side + box_w / 2
        parts.append(
            f'<rect x="{pad_side}" y="{y}" width="{box_w}" height="{box_h}" rx="10" '
            f'fill="#FFFFFF" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{pad_side}" y="{y}" width="6" height="{box_h}" rx="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{y + 26}" text-anchor="middle" font-size="17" '
            f'font-weight="600" fill="{INK}">{html.escape(title)}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{y + 46}" text-anchor="middle" font-size="12.5" '
            f'font-family="IBM Plex Mono, monospace" fill="#4A5568">{html.escape(sub)}</text>'
        )
        if i < n - 1:
            ay = y + box_h
            parts.append(
                f'<line x1="{cx}" y1="{ay}" x2="{cx}" y2="{ay + gap - 6}" '
                f'stroke="#94A3B8" stroke-width="2"/>'
            )
            parts.append(
                f'<polygon points="{cx-5},{ay+gap-6} {cx+5},{ay+gap-6} {cx},{ay+gap} " '
                f'fill="#94A3B8"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def flowchart_mermaid() -> str:
    lines = ["flowchart TB"]
    for i, (title, _sub, _c) in enumerate(FLOWCHART_STEPS):
        lines.append(f'    S{i}["{title}"]')
        if i > 0:
            lines.append(f"    S{i-1} --> S{i}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotly figures
# --------------------------------------------------------------------------- #
def _fig_html(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=True if include_js else False,
                       config=PLOTLY_CONFIG, default_height="420px")


def _base_layout(fig: go.Figure, ytitle: str = "") -> go.Figure:
    fig.update_layout(
        template="simple_white",
        font=dict(family=PLOTLY_FONT, size=14, color=INK),
        margin=dict(l=60, r=30, t=20, b=90),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis_title=ytitle,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _classify_model(name: str) -> tuple[str, str]:
    if name.startswith("control::"):
        return "control", VERMILLION
    if name.endswith("::FUSION") or name == "neural::FUSION":
        return "fusion", COBALT
    if "FUSION-minus-" in name:
        return "leave-one-out", TEAL
    if name.startswith("neural::"):
        return "neural single", OKABE_ITO["purple"]
    return "single modality", SLATE


def fig_performance(models: dict, include_js: bool) -> str:
    metric = models.get("primary_metric", "score")
    rows = list(models.get("classical", [])) + list(models.get("neural", {}).get("results", []))
    rows += list(models.get("controls", []))
    rows = [r for r in rows if r.get("primary") is not None]
    rows.sort(key=lambda r: r["primary"], reverse=True)
    groups: dict[str, dict[str, list]] = {}
    for r in rows:
        label, color = _classify_model(r["name"])
        groups.setdefault(label, {"x": [], "y": [], "eplus": [], "eminus": [], "c": color})
        groups[label]["x"].append(r["name"])
        p = r["primary"]
        groups[label]["y"].append(p)
        # 95% bootstrap CI (asymmetric) where available; fall back to fold dispersion.
        lo, hi = r.get("ci_low"), r.get("ci_high")
        if lo is not None and hi is not None:
            groups[label]["eplus"].append(max(0.0, hi - p))
            groups[label]["eminus"].append(max(0.0, p - lo))
        else:
            disp = r.get("fold_dispersion") or r.get("primary_std") or 0.0
            groups[label]["eplus"].append(disp)
            groups[label]["eminus"].append(disp)
    fig = go.Figure()
    for label, g in groups.items():
        fig.add_bar(name=label, x=g["x"], y=g["y"], marker_color=g["c"],
                    error_y=dict(type="data", symmetric=False, array=g["eplus"],
                                 arrayminus=g["eminus"], visible=True, color="#64748B"))
    chance = 0.5 if models.get("task") == "classification" else 0.0
    fig.add_hline(y=chance, line_dash="dot", line_color="#94A3B8",
                  annotation_text="chance", annotation_position="top left")
    _base_layout(fig, ytitle=metric.upper())
    fig.update_layout(xaxis_tickangle=-40, barmode="group")
    return _fig_html(fig, include_js)


def fig_cka(util: dict, include_js: bool) -> str:
    rm = util.get("redundancy_matrix", {})
    mods = rm.get("modalities", [])
    mat = rm.get("cka", [])
    if not mods or not mat:
        return "<p class='muted'>No redundancy matrix available.</p>"
    z = [[(v if v is not None else 0.0) for v in row] for row in mat]
    fig = go.Figure(go.Heatmap(
        z=z, x=mods, y=mods, zmin=0, zmax=1,
        colorscale=[[0, "#F7FAFC"], [0.5, SLATE], [1, COBALT]],
        colorbar=dict(title="CKA"),
        text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(family="IBM Plex Mono, monospace", size=12),
    ))
    _base_layout(fig)
    fig.update_layout(margin=dict(l=90, r=30, t=20, b=90), showlegend=False)
    return _fig_html(fig, include_js)


def fig_missingness(missing: dict, include_js: bool) -> str:
    sm = missing.get("sample_missingness", {})
    by_mod = sm.get("by_modality", {})
    if not by_mod:
        return "<p class='muted'>No missingness matrix available.</p>"
    mods = list(by_mod.keys())
    z = [by_mod[m] for m in mods]  # rows = modalities, cols = samples
    fig = go.Figure(go.Heatmap(
        z=z, y=mods, zmin=0, zmax=max(0.01, max((max(r) for r in z), default=0.01)),
        colorscale=[[0, "#FFFFFF"], [0.5, AMBER], [1, VERMILLION]],
        colorbar=dict(title="missing frac"),
    ))
    _base_layout(fig)
    fig.update_layout(margin=dict(l=90, r=30, t=20, b=50), showlegend=False,
                      xaxis_title="samples", yaxis_title="")
    return _fig_html(fig, include_js)


def fig_marginal_gain(util: dict, include_js: bool) -> str:
    ledger = util.get("modality_ledger", [])
    if not ledger:
        return "<p class='muted'>No modality ledger available.</p>"
    x = [m["modality"] for m in ledger]
    y = [m.get("marginal_gain_classical") or 0.0 for m in ledger]
    colors = [COBALT if v > 0.01 else VERMILLION for v in y]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors))
    fig.add_hline(y=0, line_color="#94A3B8")
    _base_layout(fig, ytitle="marginal gain (fusion - leave-one-out)")
    fig.update_layout(showlegend=False)
    return _fig_html(fig, include_js)


def fig_attribution(models: dict, include_js: bool) -> str:
    fusion = next((r for r in models.get("classical", []) if r["name"].endswith("::FUSION")
                   and r.get("feature_importance")), None)
    if not fusion:
        return "<p class='muted'>No feature attribution available.</p>"
    imp = fusion["feature_importance"]
    top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top.reverse()
    labels = [k for k, _ in top]
    vals = [v for _, v in top]
    mod_of = [lbl.split("::", 1)[0] for lbl in labels]
    palette = [COBALT, TEAL, AMBER, OKABE_ITO["purple"], SLATE, VERMILLION]
    umods = list(dict.fromkeys(mod_of))
    cmap = {m: palette[i % len(palette)] for i, m in enumerate(umods)}
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=[cmap[m] for m in mod_of],
    ))
    _base_layout(fig)
    fig.update_layout(margin=dict(l=200, r=30, t=20, b=50), showlegend=False,
                      xaxis_title="permutation importance")
    return _fig_html(fig, include_js)


# --------------------------------------------------------------------------- #
# HTML tables (sortable / filterable / exportable via vanilla JS)
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return html.escape(str(v))


def render_table(table_id: str, columns: list[str], rows: list[list[Any]],
                 numeric_cols: set[int] | None = None) -> str:
    numeric_cols = numeric_cols or set()
    head = "".join(
        f'<th onclick="omicauSort(\'{table_id}\',{i})">{html.escape(c)}'
        f'<span class="sort-ind"></span></th>' for i, c in enumerate(columns)
    )
    body = []
    for row in rows:
        tds = "".join(
            f'<td class="{"num" if i in numeric_cols else ""}">{_fmt(v)}</td>'
            for i, v in enumerate(row)
        )
        body.append(f"<tr>{tds}</tr>")
    return f"""
<div class="table-wrap">
  <div class="table-controls">
    <input type="text" class="table-filter" placeholder="Filter rows…"
           oninput="omicauFilter('{table_id}', this.value)">
    <button class="btn" onclick="omicauExport('{table_id}', ',', 'csv')">Export CSV</button>
    <button class="btn" onclick="omicauExport('{table_id}', '\\t', 'tsv')">Export TSV</button>
  </div>
  <div class="table-scroll">
    <table id="{table_id}" class="omicau-table"><thead><tr>{head}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# Asset builders
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def _model_rows(models: dict) -> tuple[list[str], list[list[Any]]]:
    metric = models.get("primary_metric", "score")
    header = ["model", "type", metric, "95% CI", "n_features", "modalities", "folds"]
    rows = []
    allr = list(models.get("classical", [])) + list(models.get("neural", {}).get("results", []))
    allr += list(models.get("controls", []))
    for r in allr:
        label, _ = _classify_model(r["name"])
        lo, hi = r.get("ci_low"), r.get("ci_high")
        ci = f"{lo:.3f}–{hi:.3f}" if (lo is not None and hi is not None) else "—"
        rows.append([r["name"], label, r.get("primary"), ci,
                     r.get("n_features"), "+".join(r.get("modalities", [])), r.get("n_splits")])
    return header, rows


def build_report(audit: dict, out_dir: str | Path, config=None) -> dict[str, Path]:
    """Compile the HTML dashboard and machine-readable assets from an audit dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    # -- machine-readable assets ------------------------------------------- #
    json_path = out / "audit.json"
    json_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8", newline="")
    assets["json"] = json_path

    models = audit.get("models", {})
    util = audit.get("utility", {})
    missing = audit.get("diagnostics", {}).get("missingness", {})
    batch = audit.get("diagnostics", {}).get("batch", {})

    mheader, mrows = _model_rows(models)
    _write_csv(out / "model_metrics.csv", mheader, mrows)
    assets["model_metrics"] = out / "model_metrics.csv"

    led_header = ["modality", "n_features", "standalone", "marginal_gain", "gain_p",
                  "redundancy_cka", "redundant_with", "batch_confounded", "verdict"]
    led_rows = [[m["modality"], m.get("n_features"), m.get("standalone_primary"),
                 m.get("marginal_gain_classical"), m.get("marginal_gain_p"),
                 m.get("redundancy_max_cka"), m.get("redundant_with"),
                 m.get("batch_confounded"), m.get("verdict")] for m in util.get("modality_ledger", [])]
    _write_csv(out / "modality_ledger.csv", led_header, led_rows)
    assets["modality_ledger"] = out / "modality_ledger.csv"

    diag_header = ["modality", "test", "association", "statistic", "p_value", "p_adj", "flag"]
    diag_rows = [[t["modality"], t["test"], t["association"], t.get("statistic"),
                  t.get("p_value"), t.get("p_adj"), t.get("flag")] for t in missing.get("tests", [])]
    _write_csv(out / "missingness_tests.csv", diag_header, diag_rows)
    assets["missingness_tests"] = out / "missingness_tests.csv"

    # -- figures ----------------------------------------------------------- #
    perf_html = fig_performance(models, include_js=True)   # first figure bundles plotly.js
    cka_html = fig_cka(util, include_js=False)
    miss_html = fig_missingness(missing, include_js=False)
    gain_html = fig_marginal_gain(util, include_js=False)
    attr_html = fig_attribution(models, include_js=False)

    # -- tables ------------------------------------------------------------ #
    model_table = render_table("tbl-models", mheader, mrows, numeric_cols={2, 4, 6})
    ledger_table = render_table("tbl-ledger", led_header, led_rows, numeric_cols={1, 2, 3, 4, 5})
    diag_table = render_table("tbl-diag", diag_header, diag_rows, numeric_cols={3, 4, 5})
    attr_rows, attr_header = _attr_rows(models)
    attr_table = render_table("tbl-attr", attr_header, attr_rows, numeric_cols={2})

    # -- render template --------------------------------------------------- #
    control_vals = [c.get("primary") for c in util.get("controls", []) if c.get("primary") is not None]
    control_max = max(control_vals) if control_vals else None

    summary = audit.get("summary", {})
    rating_status = _rating_status(summary.get("data_hygiene_rating", ""))
    answers = _answer_strip(util, rating_status)
    ledger_items = [
        {**m, "badge": _badge(_verdict_status(m.get("verdict", "")))}
        for m in util.get("modality_ledger", [])
    ]

    checklist = _trust_checklist(audit, util, missing, batch, control_max)
    ctx = {
        "audit": audit,
        "control_max": control_max,
        "checklist": checklist,
        "dashboard_css": DASHBOARD_CSS,
        "font_faces": FONT_FACES,
        "tooltip_js": TOOLTIP_JS,
        "glossary": GLOSSARY,
        "section_copy": SECTION_COPY,
        "answers": answers,
        "rating_badge": _badge(rating_status),
        "ledger_items": ledger_items,
        "meta": audit.get("meta", {}),
        "env": audit.get("environment", {}),
        "dataset": audit.get("dataset", {}),
        "cost": audit.get("cost_estimate", {}),
        "summary": audit.get("summary", {}),
        "util": util,
        "missing": missing,
        "batch": batch,
        "models": models,
        "flowchart_svg": flowchart_svg(),
        "figs": {"performance": perf_html, "cka": cka_html, "missingness": miss_html,
                 "gain": gain_html, "attribution": attr_html},
        "tables": {"models": model_table, "ledger": ledger_table, "diag": diag_table,
                   "attr": attr_table},
        "config_json": html.escape(json.dumps(audit.get("config", {}), indent=2)),
    }
    html_out = Template(_TEMPLATE).render(**ctx)
    html_path = out / "report.html"
    html_path.write_text(html_out, encoding="utf-8", newline="")
    assets["html"] = html_path
    return assets


def _attr_rows(models: dict):
    header = ["rank", "feature (modality::name)", "importance"]
    fusion = next((r for r in models.get("classical", []) if r["name"].endswith("::FUSION")
                   and r.get("feature_importance")), None)
    if not fusion:
        return [], header
    imp = sorted(fusion["feature_importance"].items(), key=lambda kv: kv[1], reverse=True)[:40]
    return [[i + 1, k, v] for i, (k, v) in enumerate(imp)], header


# --------------------------------------------------------------------------- #
# Template (fully self-contained; fonts embedded as data URIs, Plotly bundled inline)
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>omicau audit — {{ meta.run_name }}</title>
<style>{{ font_faces|safe }}</style>
<style>{{ dashboard_css|safe }}</style>
<style>
.reportcard{background:var(--panel);border:1px solid var(--border);border-radius:var(--r2);box-shadow:var(--shadow-1);padding:var(--s6);margin:var(--s5) 0}
.reportcard .eyebrow{margin-bottom:var(--s3)}
.reportcard .rc-sub{color:var(--muted);font-size:14px;margin:0 0 var(--s4)}
.check{display:flex;gap:var(--s4);align-items:flex-start;padding:12px 0;border-top:1px solid var(--hairline)}
.check:first-of-type{border-top:0}
.check-ic{width:26px;height:26px;flex:none;border-radius:50%;display:grid;place-items:center;font-size:13px;font-weight:700}
.check--pass .check-ic{background:var(--teal-bg);color:var(--teal-ink)}
.check--caution .check-ic{background:var(--amber-bg);color:var(--amber-ink)}
.check--fail .check-ic{background:var(--vermillion-bg);color:var(--vermillion-ink)}
.check-label{font-weight:600;color:var(--ink)}
.check--fail .check-label{color:var(--vermillion-ink)}
.check-note{font-size:14px;color:var(--ink-soft);margin-top:2px;line-height:1.5}
</style>
</head>
<body>
{% macro tip(term) %}{% if glossary.get(term) %}<button type="button" class="info" aria-label="Define {{ term }}">i<span class="info__bubble">{{ glossary[term] }}</span></button>{% endif %}{% endmacro %}
{% macro means(key) %}{% set c = section_copy.get(key) %}{% if c %}<div class="callout callout--means"><div class="callout__label">What this shows</div><div class="callout__body"><p><strong>{{ c.what }}</strong></p><p>How to read it: {{ c.how_to_read }}</p><p>Decision: {{ c.what_to_do }}</p></div></div>{% endif %}{% endmacro %}
<div class="wrap">
<header class="masthead">
  <div class="brand">omicau · omics audit</div>
  <h1>{{ meta.run_name or "Multi-omic data audit" }}</h1>
  <div class="sub">{{ dataset.task }} · {{ dataset.n_samples }} samples · {{ dataset.modalities|length }} modalities · seed {{ meta.seed }}</div>
  <div class="hashline">provenance SHA-256: {{ meta.provenance_hash }}</div>
</header>

<div class="tabs">
  <button class="tab active" onclick="omicauTab(event,'exec')">Executive summary</button>
  <button class="tab" onclick="omicauTab(event,'research')">Research detail</button>
</div>

<!-- ===================== EXECUTIVE ===================== -->
<div id="exec" class="panel active">
  <div class="answer-strip">
    {% for a in answers %}
    <div class="answer {{ a.cls }}">
      <span class="answer-ic" aria-hidden="true">{{ a.icon }}</span>
      <div class="answer-q">{{ a.q }}</div>
      <div class="answer-a">{{ a.a }}</div>
    </div>
    {% endfor %}
  </div>

  <div class="verdict verdict--hero">
    <span class="verdict__eyebrow">Clinical verdict</span>
    <span class="badge {{ rating_badge.css_class }}">{{ rating_badge.icon }} {{ rating_badge.label }}</span>
    <p class="verdict__lead">{{ summary.get('clinical_verdict','No verdict available.') }}</p>
    <div class="verdict__meta">{{ dataset.n_samples }} samples · {{ dataset.task }} · interpretation source: {{ summary.get('source','rule_based') }}</div>
  </div>

  <div class="reportcard">
    <span class="eyebrow">Run report card</span>
    <p class="rc-sub">A quick trust check on the run itself — treat the result only as far as these allow.</p>
    {% for c in checklist %}
    <div class="check check--{{ c.status }}">
      <span class="check-ic" aria-hidden="true">{{ c.icon }}</span>
      <div><div class="check-label">{{ c.label }}</div><div class="check-note">{{ c.note }}</div></div>
    </div>
    {% endfor %}
  </div>

  <section>
    <span class="eyebrow">Headline numbers</span>
    <div class="grid cards">
      <div class="card card--optimal"><div class="k">Best fusion {{ models.primary_metric }} {% if models.task=='classification' %}{{ tip('AUROC') }}{% else %}{{ tip('R²') }}{% endif %}</div>
        <div class="v">{{ '%.3f'|format(util.best_model.primary) if util.best_model and util.best_model.primary is not none else '—' }}</div>
        <div class="plain">Higher is better; chance is about {{ '%.1f'|format(util.chance_level) }}.</div></div>
      <div class="card {{ 'card--positive' if (util.fusion_gain_over_best_single or 0) > 0.01 else 'card--neutral' }}"><div class="k">Fusion gain {{ tip('Fusion gain (leave-one-out)') }}</div>
        <div class="v {{ 'pos' if (util.fusion_gain_over_best_single or 0) > 0 else 'neg' }}">{{ '%+.3f'|format(util.fusion_gain_over_best_single) if util.fusion_gain_over_best_single is not none else '—' }}</div>
        <div class="plain">How much combining layers beats the best single layer.</div></div>
      <div class="card"><div class="k">Samples aligned</div>
        <div class="v">{{ dataset.n_samples }}</div>
        <div class="plain">{{ dataset.get('n_dropped',0) }} dropped for a missing outcome.</div></div>
      <div class="card {{ 'card--risk' if util.leakage_warning else 'card--optimal' }}"><div class="k">Control check {{ tip('Control baseline (shuffled target)') }}</div>
        <div class="v mono">{{ '%.3f'|format(control_max) if control_max is not none else '—' }}</div>
        <div class="plain">{{ 'Above chance — leakage flag.' if util.leakage_warning else 'Near chance — no leakage.' }}</div></div>
    </div>
  </section>

  <section>
    <h2>Modality utility ledger {{ tip('Layer verdicts') }}</h2>
    {% for m in ledger_items %}
    <div class="ledger-item">
      <div class="ledger-head">
        <span class="ledger-name">{{ m.modality }}</span>
        <span class="badge {{ m.badge.css_class }}">{{ m.badge.icon }} {{ m.badge.label }}</span>
      </div>
      <div class="ledger-stats">{{ m.n_features }} features · standalone {{ '%.3f'|format(m.standalone_primary) if m.standalone_primary is not none else '—' }} · marginal gain {{ '%+.3f'|format(m.marginal_gain_classical) if m.marginal_gain_classical is not none else '—' }}{% if m.redundant_with %} · overlaps {{ m.redundant_with }} (CKA {{ '%.2f'|format(m.redundancy_max_cka) }}){% endif %}</div>
      <div class="ledger-rec">{{ m.recommendation }}</div>
    </div>
    {% endfor %}
  </section>

  <section>
    <h2>Data-hygiene flags</h2>
    <ul class="flags">
      {% set allflags = (missing.get('flags',[]) + batch.get('flags',[]) + util.get('summary_flags',[])) %}
      {% if allflags %}{% for f in allflags %}<li>{{ f }}</li>{% endfor %}
      {% else %}<li class="clean">No missingness-bias or batch-effect flags raised.</li>{% endif %}
    </ul>
  </section>

  <section>
    <h2>Recommendations</h2>
    <ul class="recs">
      {% for r in summary.get('actionable_recommendations',[]) %}<li>{{ r }}</li>{% endfor %}
    </ul>
  </section>

  <details class="glossary">
    <summary>Reading guide — plain-language glossary</summary>
    <div class="gloss-body">
      {% for term, definition in glossary.items() %}<p><strong>{{ term }}</strong> — {{ definition }}</p>{% endfor %}
    </div>
  </details>
</div>

<!-- ===================== RESEARCH ===================== -->
<div id="research" class="panel">
  <section>
    <h2>Cross-modal performance</h2>
    {{ means('cross_modal_performance') }}
    <div class="reading-guide">
      <span class="legend-row"><span class="swatch" style="background:#0072B2"></span>fusion</span>
      <span class="legend-row"><span class="swatch" style="background:#4477AA"></span>single modality</span>
      <span class="legend-row"><span class="swatch" style="background:#009E73"></span>leave-one-out</span>
      <span class="legend-row"><span class="swatch" style="background:#D55E00"></span>control baseline</span>
    </div>
    <div class="figure">{{ figs.performance|safe }}</div>
    {{ tables.models|safe }}
  </section>

  <section>
    <h2>Modality redundancy</h2>
    {{ means('redundancy') }}
    <div class="figure">{{ figs.cka|safe }}</div>
    <h3>Marginal gain per modality</h3>
    <div class="figure">{{ figs.gain|safe }}</div>
    {{ tables.ledger|safe }}
  </section>

  <section>
    <h2>Missingness structure</h2>
    {{ means('missingness') }}
    <div class="figure">{{ figs.missingness|safe }}</div>
    {{ tables.diag|safe }}
  </section>

  <section>
    <h2>Feature attribution</h2>
    {{ means('feature_attribution') }}
    <div class="figure">{{ figs.attribution|safe }}</div>
    {{ tables.attr|safe }}
  </section>

  <section>
    <h2>Provenance &amp; environment</h2>
    {{ means('provenance') }}
    <div class="grid cards">
      <div class="card"><div class="k">Provenance hash {{ tip('Provenance hash') }}</div><div class="v mono" style="font-size:13px;word-break:break-all">{{ meta.provenance_hash }}</div></div>
      <div class="card"><div class="k">Device / cores</div><div class="v mono">{{ meta.device }} / {{ meta.cores }}</div></div>
      <div class="card"><div class="k">Python / torch</div><div class="v mono" style="font-size:16px">{{ env.python }} · {{ env.torch }}</div></div>
      <div class="card"><div class="k">Est. wall-time</div><div class="v mono" style="font-size:18px">{{ cost.get('human_readable','—') }}</div></div>
    </div>
    <h3>Resolved configuration</h3>
    <pre class="config">{{ config_json }}</pre>
  </section>
</div>

<footer>
  Generated by <strong>omicau</strong> v{{ meta.tool_version }} · {{ meta.created }}
</footer>
</div>

<script>
function omicauTab(ev, id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  ev.currentTarget.classList.add('active');
  document.getElementById(id).classList.add('active');
  window.dispatchEvent(new Event('resize'));
}
function omicauCellVal(td){
  const t = td.textContent.trim();
  const n = parseFloat(t.replace(/[^0-9eE.+-]/g,''));
  return (t!=='—' && t!=='' && !isNaN(n) && /[0-9]/.test(t)) ? n : t.toLowerCase();
}
function omicauSort(id, col){
  const tbl = document.getElementById(id);
  const tb = tbl.tBodies[0];
  const rows = Array.from(tb.rows);
  const cur = tbl.getAttribute('data-sort-col');
  const asc = !(cur == col && tbl.getAttribute('data-sort-dir') === 'asc');
  rows.sort((a,b)=>{
    const x = omicauCellVal(a.cells[col]), y = omicauCellVal(b.cells[col]);
    if(x<y) return asc?-1:1; if(x>y) return asc?1:-1; return 0;
  });
  rows.forEach(r=>tb.appendChild(r));
  tbl.setAttribute('data-sort-col', col);
  tbl.setAttribute('data-sort-dir', asc?'asc':'desc');
  tbl.querySelectorAll('.sort-ind').forEach(s=>s.textContent='');
  tbl.tHead.rows[0].cells[col].querySelector('.sort-ind').textContent = asc?' ▲':' ▼';
}
function omicauFilter(id, q){
  q = q.toLowerCase();
  const tbl = document.getElementById(id);
  Array.from(tbl.tBodies[0].rows).forEach(r=>{
    r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}
function omicauExport(id, sep, ext){
  const tbl = document.getElementById(id);
  const esc = v => {
    v = String(v);
    if(sep===',' && /[",\n]/.test(v)) return '"'+v.replace(/"/g,'""')+'"';
    return v;
  };
  const lines = [];
  const heads = Array.from(tbl.tHead.rows[0].cells).map(c=>esc(c.textContent.replace(/[▲▼]/g,'').trim()));
  lines.push(heads.join(sep));
  Array.from(tbl.tBodies[0].rows).forEach(r=>{
    if(r.style.display==='none') return;
    lines.push(Array.from(r.cells).map(c=>esc(c.textContent.trim())).join(sep));
  });
  const blob = new Blob([lines.join('\n')], {type:'text/'+ext});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = id.replace('tbl-','omicau_') + '.' + ext;
  a.click(); URL.revokeObjectURL(a.href);
}
{{ tooltip_js|safe }}
</script>
</body>
</html>"""
