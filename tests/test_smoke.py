"""End-to-end smoke tests for omicau on a minimal footprint.

Covers the whole pipeline on a small synthetic dataset: flexible ingestion
(row/column orientation, dirty headers, mixed delimiters), provenance hashing,
missingness and batch diagnostics, classical and neural fusion benchmarks,
leakage-safe feature attribution, the pre-flight cost estimator, and all report
compilers (HTML + JSON/CSV + Markdown/DOCX/LaTeX). The full pipeline is exercised
in both LLM states: once with a mocked Anthropic client and a simulated API key,
and once fully offline with the deterministic rule-based fallback.
"""

from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omicau.config import OmicauConfig
from omicau.data.alignment import align_modalities, read_matrix, compute_provenance_hash
from omicau.data.benchmark_data import make_mock_dataset, mock_config, write_mock_dataset
from omicau.diagnostics import batch_effect_diagnostics, missingness_diagnostics
from omicau.models.classical import run_classical_benchmarks
from omicau.models.neural import MaskedGlobalPoolingFusion, run_neural_benchmark
from omicau.interpretation.utility import build_utility_ledger
from omicau.interpretation.llm_summary import build_context, summarize
from omicau.reporting.reporter import build_report, flowchart_svg


# --------------------------------------------------------------------------- #
# Fixtures: a small, fast pipeline run shared across tests
# --------------------------------------------------------------------------- #
def _small_config(task: str = "classification") -> OmicauConfig:
    cfg = mock_config(task=task)
    cfg.classical.models = ["linear"]           # logistic/ridge only -> fast
    cfg.classical.max_features = None
    cfg.cv.n_splits = 3
    cfg.neural.epochs = 5
    cfg.neural.hidden_dim = 16
    cfg.neural.embed_dim = 8
    cfg.xai.permutation_repeats = 3
    cfg.compute.cores = 2
    return cfg


@pytest.fixture(scope="module")
def bundle():
    return make_mock_dataset(task="classification", n_samples=72, seed=123,
                             signal_features=12, redundant_features=8,
                             confounded_features=10, noise_features=6)


@pytest.fixture(scope="module")
def aligned(bundle):
    return align_modalities(bundle.modalities, bundle.clinical, _small_config())


@pytest.fixture(scope="module")
def pipeline(aligned):
    cfg = _small_config()
    missing = missingness_diagnostics(aligned)
    batch = batch_effect_diagnostics(aligned, seed=cfg.seed)
    classical = run_classical_benchmarks(aligned, cfg)
    neural = run_neural_benchmark(aligned, cfg)
    util = build_utility_ledger(aligned, classical, neural, batch, missing)
    return {"cfg": cfg, "missing": missing, "batch": batch,
            "classical": classical, "neural": neural, "util": util}


# --------------------------------------------------------------------------- #
# Ingestion / alignment / provenance
# --------------------------------------------------------------------------- #
def test_read_matrix_delimiters_and_sanitize(tmp_path):
    # semicolon-delimited, whitespace, European decimals, an NA token, a US column.
    csv = ("id;euro;us;const\n"
           " s1 ; 1,5 ; 2.0 ; 9\n"
           "s2;3,0; NA ;9\n"
           "s3;4,0;5.0;9\n")
    p = tmp_path / "m.csv"
    p.write_text(csv, encoding="utf-8", newline="")
    df = read_matrix(p)
    assert list(df.index) == ["s1", "s2", "s3"]
    assert df.loc["s1", "euro"] == pytest.approx(1.5)    # European decimal healed
    assert df.loc["s3", "euro"] == pytest.approx(4.0)
    assert df.loc["s1", "us"] == pytest.approx(2.0)      # US column preserved
    assert np.isnan(df.loc["s2", "us"])                  # NA token -> NaN
    assert df.dtypes.map(lambda d: d == np.float64).all()


def test_orientation_autodetect_and_dirty_headers(bundle):
    cfg = _small_config()
    # transpose the signal modality (genes as rows) and add whitespace to labels.
    t = bundle.modalities["signal"].T.copy()
    t.index = [f"  {g}  " for g in t.index]
    t.columns = [f"{s} " for s in t.columns]
    mods = dict(bundle.modalities)
    mods["signal"] = t
    ad = align_modalities(mods, bundle.clinical, cfg)
    assert ad.n_samples > 0
    assert "transposed" in ad.report["orientation"]["signal"]
    assert ad.modalities["signal"].shape[0] == ad.n_samples


def test_provenance_hash_deterministic(aligned):
    h1 = compute_provenance_hash(aligned.sample_ids, aligned.modalities, "label", aligned.task)
    h2 = compute_provenance_hash(aligned.sample_ids, aligned.modalities, "label", aligned.task)
    assert len(h1) == 64 and h1 == h2
    # dropping a modality changes the hash (structural).
    sub = {k: v for k, v in aligned.modalities.items() if k != "noise"}
    assert compute_provenance_hash(aligned.sample_ids, sub, "label", aligned.task) != h1


def test_provenance_hash_is_value_level(aligned):
    from omicau.data.alignment import ModalityMatrix
    base = compute_provenance_hash(aligned.sample_ids, aligned.modalities, "label", aligned.task)
    # perturb a single measurement -> the hash must change (tamper-evident).
    mods = dict(aligned.modalities)
    frame = mods["signal"].frame.copy()
    r, c = frame.index[0], frame.columns[0]
    frame.loc[r, c] = (frame.loc[r, c] or 0.0) + 1.0
    mods["signal"] = ModalityMatrix("signal", frame)
    assert compute_provenance_hash(aligned.sample_ids, mods, "label", aligned.task) != base


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_missingness_and_batch_diagnostics(pipeline):
    miss, batch = pipeline["missing"], pipeline["batch"]
    assert "tests" in miss and miss["overall"]["total_missing_fraction"] is not None
    json.dumps(miss)  # must be JSON-serializable
    assert "per_modality" in batch
    json.dumps(batch)
    # confounded modality should show the strongest batch structure.
    sils = {k: v.get("silhouette_batch") or -1 for k, v in batch["per_modality"].items()}
    assert sils["confounded"] == max(sils.values())


# --------------------------------------------------------------------------- #
# Models + XAI + cost
# --------------------------------------------------------------------------- #
def test_classical_benchmarks_and_attribution(pipeline):
    cl = pipeline["classical"]
    assert cl["results"] and cl["controls"]
    fusion = next(r for r in cl["results"] if r.name.endswith("::FUSION"))
    assert fusion.feature_importance, "reference fusion must carry attribution"
    assert np.isfinite(fusion.primary)


def test_neural_masked_pooling_ignores_missing():
    # A missing feature (mask=0) must not affect the pooled embedding.
    import torch
    torch.manual_seed(0)
    model = MaskedGlobalPoolingFusion({"m": 4}, embed_dim=5, hidden_dim=8, out_dim=2)
    x = torch.tensor([[1.0, 2.0, 3.0, 7.0]])
    mask_full = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask_drop = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    enc = model.encoders["m"]
    x2 = torch.tensor([[1.0, 2.0, 3.0, 999.0]])  # different value in the masked slot
    a = enc(x, mask_drop)
    b = enc(x2, mask_drop)
    assert torch.allclose(a, b, atol=1e-6)       # masked slot ignored
    assert not torch.allclose(enc(x, mask_full), a)  # observed slot matters


def test_neural_benchmark_runs(pipeline):
    nn = pipeline["neural"]
    assert nn["enabled"] and nn["results"]
    fusion = next(r for r in nn["results"] if r.name == "neural::FUSION")
    assert np.isfinite(fusion.primary)


def test_cost_estimate(aligned):
    from omicau.cli import estimate_runtime
    est = estimate_runtime(aligned, _small_config(), "cpu", 2)
    assert est["total_seconds"] > 0
    assert est["human_readable"].startswith("~")
    assert est["breakdown"]["classical_fits"] > 0


def test_utility_ledger(pipeline):
    util = pipeline["util"]
    assert util["modality_ledger"]
    assert set(util["redundancy_matrix"]["modalities"]) == {"signal", "redundant", "confounded", "noise"}
    for m in util["modality_ledger"]:
        assert m["verdict"]


# --------------------------------------------------------------------------- #
# Flowchart + document compilers
# --------------------------------------------------------------------------- #
def _audit(aligned, pipeline, summary):
    cfg = pipeline["cfg"]
    return {
        "meta": {"run_name": "smoke", "tool_version": "0.1.0", "created": "2026-07-07",
                 "provenance_hash": aligned.provenance_hash, "device": "cpu", "cores": 2,
                 "seed": cfg.seed},
        "environment": {"python": "3.12", "platform": "test", "machine": "x", "torch": "cpu",
                        "numpy": "x"},
        "dataset": {"n_samples": aligned.n_samples, "task": aligned.task,
                    "class_names": aligned.class_names,
                    "class_balance": aligned.report.get("class_balance"),
                    "feature_counts": aligned.feature_counts(), "n_dropped": 0,
                    "modalities": [{"name": n, "description": m.description, "n_features": m.shape[1]}
                                   for n, m in aligned.modalities.items()]},
        "cost_estimate": {"human_readable": "~5 s", "total_seconds": 5.0},
        "diagnostics": {"missingness": pipeline["missing"], "batch": pipeline["batch"]},
        "models": {"primary_metric": pipeline["classical"]["primary_metric"], "task": aligned.task,
                   "reference_estimator": pipeline["classical"]["reference_estimator"],
                   "classical": [r.to_dict() for r in pipeline["classical"]["results"]],
                   "controls": [r.to_dict() for r in pipeline["classical"]["controls"]],
                   "neural": {"enabled": True, "device": "cpu",
                              "results": [r.to_dict() for r in pipeline["neural"]["results"]]}},
        "utility": pipeline["util"], "summary": summary, "config": cfg.to_dict(),
    }


def test_flowchart_svg_offline():
    svg = flowchart_svg()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "Provenance SHA-256" in svg


def test_run_audit_in_process_contract(tmp_path):
    # The contract the optional UI binds to: build a config object, call
    # run_audit(config, echo=...) in-process, receive streamed named stages and
    # get back the audit dict (provenance hash) + asset paths — no CLI, no files
    # beyond the dataset.
    from omicau.cli import run_audit
    write_mock_dataset(tmp_path / "ds", task="classification", seed=3, n_samples=54)
    cfg = OmicauConfig.from_file(tmp_path / "ds" / "config.json")
    cfg.classical.models = ["linear"]
    cfg.cv.n_splits = 3
    cfg.neural.enabled = False
    cfg.classical.max_features = None
    cfg.xai.enabled = False
    cfg.reporting.docs = []

    stages: list[str] = []
    audit = run_audit(cfg, cores=2, device="cpu", llm=False, echo=lambda m: stages.append(str(m)))

    assert audit["meta"]["provenance_hash"]
    assert "ingest_align" in " ".join(stages)          # named stages streamed
    assert any("report" in s for s in stages)
    assets = audit.get("_assets", {})
    assert "html" in assets and Path(assets["html"]).exists()
    assert "json" in assets and Path(assets["json"]).exists()


def test_dashboard_offline_no_external_fonts():
    from omicau.reporting._assets import FONT_FACES
    assert FONT_FACES.count("@font-face") >= 6
    assert "data:font/woff2" in FONT_FACES
    assert "fonts.googleapis" not in FONT_FACES and "fonts.gstatic" not in FONT_FACES


def test_dashboard_css_integrity():
    # Guard the exact bug class where a stray */ in a comment closed the comment
    # early and silently broke the :root custom-property block.
    from omicau.reporting._assets import DASHBOARD_CSS as css
    assert css.count("{") == css.count("}"), "unbalanced CSS braces"
    assert css.count("/*") == css.count("*/"), "unbalanced CSS comments"
    assert ":root" in css and "--cobalt:#0072B2" in css.replace(" ", "")
    assert "</style>" not in css  # would prematurely close the <style> element


def _assert_valid_html(path):
    html = path.read_text(encoding="utf-8")
    assert html.strip().lower().startswith("<!doctype html>")
    assert html.count('class="omicau-table"') >= 3
    assert "plotly-graph-div" in html
    assert "omicauSort" in html and "omicauExport" in html
    assert "#0072B2" in html  # color-blind-safe palette present


def test_full_pipeline_offline(aligned, pipeline, tmp_path):
    summary = summarize(build_context(aligned, pipeline["util"], pipeline["missing"],
                                      pipeline["batch"]), pipeline["cfg"])
    assert summary["source"] == "rule_based"
    assert set(("clinical_verdict", "data_hygiene_rating", "modality_utility_ledger",
                "actionable_recommendations")) <= set(summary)
    audit = _audit(aligned, pipeline, summary)
    assets = build_report(audit, tmp_path / "offline")
    _assert_valid_html(assets["html"])
    assert json.loads(assets["json"].read_text(encoding="utf-8"))["meta"]["provenance_hash"]
    assert (tmp_path / "offline" / "model_metrics.csv").exists()


def test_full_pipeline_llm_mocked(aligned, pipeline, tmp_path, monkeypatch):
    # Install a fake anthropic module returning a valid JSON schema.
    canned = json.dumps({
        "clinical_verdict": "Fusion is justified; data hygiene is acceptable.",
        "data_hygiene_rating": "moderate concerns: one missingness flag.",
        "modality_utility_ledger": [{"modality": "signal", "verdict": "predictive",
                                     "recommendation": "retain"}],
        "actionable_recommendations": ["Correct batch effects in the confounded layer."],
    })

    class _Block:
        type = "text"
        text = canned

    class _Resp:
        stop_reason = "end_turn"
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            return _Resp()

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-xxxxx")

    cfg = _small_config()
    cfg.llm.enabled = True
    summary = summarize(build_context(aligned, pipeline["util"], pipeline["missing"],
                                      pipeline["batch"]), cfg)
    assert summary["source"].startswith("llm:")
    assert "Fusion is justified" in summary["clinical_verdict"]

    audit = _audit(aligned, pipeline, summary)
    assets = build_report(audit, tmp_path / "llm")
    _assert_valid_html(assets["html"])


def test_bootstrap_mock_writes_runnable_config(tmp_path):
    write_mock_dataset(tmp_path / "ds", task="classification", seed=1, n_samples=50)
    cfg = OmicauConfig.from_file(tmp_path / "ds" / "config.json")
    assert len(cfg.modalities) == 4
    # paths resolved against the config directory.
    from pathlib import Path
    assert Path(cfg.modalities[0].path).exists()
    from omicau.data.alignment import load_and_align
    ad = load_and_align(cfg)
    assert ad.n_samples > 0 and ad.provenance_hash
