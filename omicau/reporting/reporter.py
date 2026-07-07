"""Interactive HTML dashboard and machine-readable asset compiler.

Produces a single self-contained ``.html`` file (Plotly bundled inline, so it
works offline after the one-time Google Fonts load), styled with an editorial
serif aesthetic (EB Garamond + JetBrains Mono) and a color-blind-safe Okabe-Ito
palette. Every data grid is sortable, text-filterable, and CSV/TSV-exportable via
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

PLOTLY_FONT = "EB Garamond, Georgia, serif"
PLOTLY_CONFIG = {"responsive": True, "displaylogo": False,
                 "toImageButtonOptions": {"format": "svg", "scale": 2}}


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
            f'font-family="JetBrains Mono, monospace" fill="#4A5568">{html.escape(sub)}</text>'
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
        groups.setdefault(label, {"x": [], "y": [], "e": [], "c": color})
        groups[label]["x"].append(r["name"])
        groups[label]["y"].append(r["primary"])
        groups[label]["e"].append(r.get("primary_std") or 0.0)
    fig = go.Figure()
    for label, g in groups.items():
        fig.add_bar(name=label, x=g["x"], y=g["y"], marker_color=g["c"],
                    error_y=dict(type="data", array=g["e"], visible=True, color="#64748B"))
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
        textfont=dict(family="JetBrains Mono, monospace", size=12),
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
    header = ["model", "type", metric, f"{metric}_std", "n_features", "modalities", "folds"]
    rows = []
    allr = list(models.get("classical", [])) + list(models.get("neural", {}).get("results", []))
    allr += list(models.get("controls", []))
    for r in allr:
        label, _ = _classify_model(r["name"])
        rows.append([r["name"], label, r.get("primary"), r.get("primary_std"),
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
    model_table = render_table("tbl-models", mheader, mrows, numeric_cols={2, 3, 4, 6})
    ledger_table = render_table("tbl-ledger", led_header, led_rows, numeric_cols={1, 2, 3, 4, 5})
    diag_table = render_table("tbl-diag", diag_header, diag_rows, numeric_cols={3, 4, 5})
    attr_rows, attr_header = _attr_rows(models)
    attr_table = render_table("tbl-attr", attr_header, attr_rows, numeric_cols={2})

    # -- render template --------------------------------------------------- #
    control_vals = [c.get("primary") for c in util.get("controls", []) if c.get("primary") is not None]
    control_max = max(control_vals) if control_vals else None
    ctx = {
        "audit": audit,
        "control_max": control_max,
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
# Template (self-contained; fonts via CDN, Plotly bundled inline)
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>omicau audit — {{ meta.run_name }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --cobalt:#0072B2; --vermillion:#D55E00; --amber:#E69F00; --teal:#009E73;
  --slate:#4477AA; --ink:#1A202C; --muted:#64748B; --border:#E2E8F0;
  --bg:#FBFCFD; --panel:#FFFFFF;
  --serif:'EB Garamond',Georgia,'Times New Roman',serif;
  --mono:'JetBrains Mono','Fira Code','SF Mono',ui-monospace,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);
  font-size:18px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 80px}
header.masthead{border-bottom:1px solid var(--border);padding:34px 0 24px;margin-bottom:8px}
.brand{font-size:13px;letter-spacing:.22em;text-transform:uppercase;color:var(--cobalt);
  font-family:var(--mono);font-weight:600}
h1{font-size:38px;font-weight:600;margin:6px 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:16px}
.hashline{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:10px;
  word-break:break-all}
.tabs{display:flex;gap:6px;border-bottom:1px solid var(--border);margin:22px 0 26px;position:sticky;
  top:0;background:var(--bg);z-index:5;padding-top:6px}
.tab{appearance:none;border:none;background:none;font-family:var(--serif);font-size:17px;
  color:var(--muted);padding:12px 18px;cursor:pointer;border-bottom:2px solid transparent}
.tab.active{color:var(--ink);border-bottom-color:var(--cobalt);font-weight:600}
.panel{display:none}.panel.active{display:block}
.grid{display:grid;gap:18px}
.grid.cards{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:20px 22px}
.card .k{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted)}
.card .v{font-size:30px;font-weight:600;margin-top:6px;font-variant-numeric:tabular-nums}
.card .v.mono{font-family:var(--mono);font-size:22px}
.card .note{font-size:14px;color:var(--muted);margin-top:4px}
section{margin:34px 0}
h2{font-size:25px;font-weight:600;border-bottom:1px solid var(--border);padding-bottom:8px;
  margin-bottom:18px}
h3{font-size:19px;font-weight:600;margin:26px 0 10px}
.figure{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px 16px;
  overflow-x:auto}
.figcap{font-size:14px;color:var(--muted);margin:10px 4px 0}
.verdict{background:var(--panel);border:1px solid var(--border);border-left:5px solid var(--cobalt);
  border-radius:10px;padding:20px 24px;font-size:19px}
.pill{display:inline-block;font-family:var(--mono);font-size:12px;font-weight:600;padding:4px 10px;
  border-radius:999px;letter-spacing:.03em}
.pill.ok{background:#E6F0F7;color:var(--cobalt)}
.pill.warn{background:#FBEBDF;color:var(--vermillion)}
.pill.mid{background:#FCF4E1;color:#9A6A00}
.flags{list-style:none;padding:0;margin:14px 0}
.flags li{padding:10px 14px;border:1px solid var(--border);border-left:4px solid var(--amber);
  border-radius:8px;margin-bottom:8px;font-size:16px;background:var(--panel)}
.flags li.clean{border-left-color:var(--teal)}
ul.recs{padding-left:22px}ul.recs li{margin-bottom:8px}
.ledger-item{border:1px solid var(--border);border-radius:10px;padding:14px 18px;margin-bottom:10px;
  background:var(--panel)}
.ledger-item .name{font-weight:600;font-size:18px}
.ledger-item .rec{color:var(--muted);font-size:15px;margin-top:4px}
.table-wrap{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px}
.table-controls{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.table-filter{flex:1;min-width:180px;font-family:var(--serif);font-size:15px;padding:8px 12px;
  border:1px solid var(--border);border-radius:8px}
.btn{font-family:var(--mono);font-size:12.5px;font-weight:600;background:var(--cobalt);color:#fff;
  border:none;border-radius:8px;padding:8px 14px;cursor:pointer}
.btn:hover{background:#005a8c}
.table-scroll{overflow-x:auto}
table.omicau-table{width:100%;border-collapse:collapse;font-size:14.5px}
table.omicau-table th{font-family:var(--mono);font-size:11.5px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);text-align:left;padding:10px 12px;cursor:pointer;
  border-bottom:2px solid var(--border);white-space:nowrap;user-select:none}
table.omicau-table th:hover{color:var(--ink)}
table.omicau-table td{padding:9px 12px;border-bottom:1px solid var(--border)}
table.omicau-table td.num{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
table.omicau-table tbody tr:hover{background:#F7FAFC}
.sort-ind{margin-left:6px;color:var(--cobalt)}
.flow{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:22px}
pre.config{background:#0F172A;color:#E2E8F0;font-family:var(--mono);font-size:12.5px;padding:18px;
  border-radius:10px;overflow-x:auto;line-height:1.5}
.muted{color:var(--muted)}
.legend{font-size:13px;color:var(--muted);font-family:var(--mono);margin-top:6px}
footer{border-top:1px solid var(--border);margin-top:50px;padding-top:20px;color:var(--muted);
  font-size:14px}
</style>
</head>
<body>
<div class="wrap">
<header class="masthead">
  <div class="brand">omicau · omics audit</div>
  <h1>{{ meta.run_name or "Multi-omic data audit" }}</h1>
  <div class="sub">{{ dataset.task }} · {{ dataset.n_samples }} samples ·
    {{ dataset.modalities|length }} modalities · seed {{ meta.seed }}</div>
  <div class="hashline">provenance SHA-256: {{ meta.provenance_hash }}</div>
</header>

<div class="tabs">
  <button class="tab active" onclick="omicauTab(event,'exec')">Executive summary</button>
  <button class="tab" onclick="omicauTab(event,'research')">Research detail</button>
</div>

<!-- ===================== EXECUTIVE ===================== -->
<div id="exec" class="panel active">
  {% set rating = summary.get('data_hygiene_rating','') %}
  {% set pill = 'warn' if 'high' in rating else ('mid' if 'moderate' in rating else 'ok') %}
  <div class="verdict">
    <span class="pill {{ pill }}">{{ rating.split(':')[0] if rating else 'unrated' }}</span>
    <p style="margin:14px 0 0">{{ summary.get('clinical_verdict','No verdict available.') }}</p>
  </div>

  <section>
    <div class="grid cards">
      <div class="card"><div class="k">Best fusion {{ models.primary_metric }}</div>
        <div class="v">{{ '%.3f'|format(util.best_model.primary) if util.best_model and util.best_model.primary is not none else '—' }}</div>
        <div class="note">{{ util.best_model.name if util.best_model else '' }}</div></div>
      <div class="card"><div class="k">Fusion gain vs best single</div>
        <div class="v">{{ '%+.3f'|format(util.fusion_gain_over_best_single) if util.fusion_gain_over_best_single is not none else '—' }}</div>
        <div class="note">leave-one-modality-out delta</div></div>
      <div class="card"><div class="k">Samples aligned</div>
        <div class="v">{{ dataset.n_samples }}</div>
        <div class="note">{{ dataset.get('n_dropped',0) }} dropped in alignment</div></div>
      <div class="card"><div class="k">Control baseline max</div>
        <div class="v mono">{{ '%.3f'|format(control_max) if control_max is not none else '—' }}</div>
        <div class="note">{{ 'leakage flagged' if util.leakage_warning else 'near chance — no leakage' }}</div></div>
    </div>
  </section>

  <section>
    <h2>Modality utility ledger</h2>
    {% for m in util.modality_ledger %}
      {% set vc = 'warn' if m.batch_confounded or 'no detectable' in m.verdict else ('ok' if 'predictive' in m.verdict else 'mid') %}
      <div class="ledger-item">
        <span class="pill {{ vc }}">{{ m.verdict }}</span>
        <span class="name" style="margin-left:10px">{{ m.modality }}</span>
        <span class="muted" style="font-family:var(--mono);font-size:13px;margin-left:8px">
          {{ m.n_features }} features · standalone {{ '%.3f'|format(m.standalone_primary) if m.standalone_primary is not none else '—' }}
          · gain {{ '%+.3f'|format(m.marginal_gain_classical) if m.marginal_gain_classical is not none else '—' }}</span>
        <div class="rec">{{ m.recommendation }}</div>
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
    <p class="legend">interpretation source: {{ summary.get('source','rule_based') }}</p>
  </section>
</div>

<!-- ===================== RESEARCH ===================== -->
<div id="research" class="panel">
  <section>
    <h2>Cross-modal performance</h2>
    <div class="figure">{{ figs.performance|safe }}</div>
    <p class="figcap">Primary metric ({{ models.primary_metric|upper }}) per model. Cobalt = fusion,
      slate = single modality, green = leave-one-out, vermillion = control baseline. Error bars are
      the across-fold standard deviation; dotted line marks chance.</p>
    {{ tables.models|safe }}
  </section>

  <section>
    <h2>Modality redundancy (linear CKA)</h2>
    <div class="figure">{{ figs.cka|safe }}</div>
    <p class="figcap">Centered kernel alignment between modalities; high values indicate shared
      representation and probable redundancy.</p>
    <h3>Marginal gain per modality</h3>
    <div class="figure">{{ figs.gain|safe }}</div>
    {{ tables.ledger|safe }}
  </section>

  <section>
    <h2>Missingness structure</h2>
    <div class="figure">{{ figs.missingness|safe }}</div>
    <p class="figcap">Per-sample missing fraction by modality (burnt-orange = higher missingness).</p>
    {{ tables.diag|safe }}
  </section>

  <section>
    <h2>Leakage-safe feature attribution</h2>
    <div class="figure">{{ figs.attribution|safe }}</div>
    <p class="figcap">Permutation importance from the reference fusion model, computed on held-out
      folds; colored by source modality.</p>
    {{ tables.attr|safe }}
  </section>

  <section>
    <h2>Provenance &amp; environment</h2>
    <div class="grid cards">
      <div class="card"><div class="k">Provenance hash</div><div class="v mono" style="font-size:13px;word-break:break-all">{{ meta.provenance_hash }}</div></div>
      <div class="card"><div class="k">Device / cores</div><div class="v mono">{{ meta.device }} / {{ meta.cores }}</div></div>
      <div class="card"><div class="k">Python / torch</div><div class="v mono" style="font-size:16px">{{ env.python }} · {{ env.torch }}</div></div>
      <div class="card"><div class="k">Est. wall-time</div><div class="v mono" style="font-size:18px">{{ cost.get('human_readable','—') }}</div></div>
    </div>
    <h3>Resolved configuration</h3>
    <pre class="config">{{ config_json }}</pre>
  </section>
</div>

<footer>
  Generated by <strong>omicau</strong> v{{ meta.tool_version }} · {{ meta.created }} ·
  This report is self-contained. Fonts load from Google Fonts; all charts and data are embedded.
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
</script>
</body>
</html>"""
