"""Unit tests for the wizard's config-generation and validation core.

These run against real files (the mock dataset) and are independent of the web
layer, so they hold whether or not the optional [ui] extra is installed.
"""

from __future__ import annotations

import time

import pytest

from omicau.data.benchmark_data import write_mock_dataset
from omicau.ui import inspect as I


def _dataset(tmp_path):
    write_mock_dataset(tmp_path / "ds", task="classification", seed=7, n_samples=80)
    d = tmp_path / "ds"
    return d, {n: str(d / f"{n}.csv") for n in ("signal", "redundant", "confounded", "noise")}, \
        str(d / "clinical.csv")


def test_guess_role():
    assert I.guess_role("TCGA_rna_expression.csv")["role"] == "rna"
    assert I.guess_role("methylation_beta.tsv")["role"] == "methylation"
    assert I.guess_role("clinical.csv")["role"] == "clinical"
    assert I.guess_role("protein_rppa.csv")["role"] == "protein"
    assert I.guess_role("random_matrix.csv")["role"] == "other"


def test_inspect_matrix(tmp_path):
    _, mods, _ = _dataset(tmp_path)
    info = I.inspect_matrix(mods["signal"], "signal.csv")
    assert info["n_rows"] == 80 and info["n_cols"] > 0
    assert info["row_labels"][0].startswith("S")           # samples as rows
    assert info["col_labels"][0].startswith("SIG")         # feature columns
    assert len(info["preview"]) > 0 and info["delimiter"] == "comma"


def test_inspect_clinical(tmp_path):
    _, _, clin = _dataset(tmp_path)
    info = I.inspect_clinical(clin)
    names = {c["name"]: c for c in info["columns"]}
    assert {"sample_id", "label", "patient_id", "batch"} <= set(names)
    assert names["sample_id"]["looks_like_id"] is True
    assert names["label"]["kind"] == "categorical"


def test_target_group_batch_consequences(tmp_path):
    _, _, clin = _dataset(tmp_path)
    t = I.target_consequence(clin, "label")
    assert t["ok"] and t["task"] == "classification" and t["n_classes"] == 2

    g = I.group_consequence(clin, "patient_id")
    assert g["ok"] and g["n_groups"] < g["n_rows"] and g["repeated_measures"] is True

    b = I.batch_consequence(clin, "batch", target="label")
    assert b["ok"] and b["n_batches"] >= 2 and "crosstab" in b


def test_alignment_preview(tmp_path):
    _, mods, clin = _dataset(tmp_path)
    modalities = [{"name": n, "path": p, "orientation": "samples_as_rows"} for n, p in mods.items()]
    rep = I.alignment_preview(modalities, clin, "sample_id")
    assert rep["ok"] and rep["matched_all_layers"] == 80
    assert all(layer["overlap_with_clinical"] == 80 for layer in rep["per_layer"])


def test_alignment_preview_detects_mismatch(tmp_path):
    _, mods, clin = _dataset(tmp_path)
    # a modality whose ids don't match the clinical table -> low overlap + hint
    import pandas as pd
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(mods["signal"], index_col=0)
    df.index = [f"WRONG_{i}" for i in range(len(df))]
    df.to_csv(bad, index=True, index_label="sample_id")
    rep = I.alignment_preview([{"name": "bad", "path": str(bad), "orientation": "samples_as_rows"}],
                              clin, "sample_id")
    assert rep["ok"] is False and rep["matched_all_layers"] == 0 and rep["hint"]


def test_build_config_roundtrip(tmp_path):
    from omicau.config import OmicauConfig
    from omicau.data.alignment import load_and_align
    d, mods, clin = _dataset(tmp_path)
    session = {
        "run_name": "ui_test", "output_dir": str(d / "run"),
        "modalities": [{"name": n, "path": p, "orientation": "samples_as_rows"} for n, p in mods.items()],
        "clinical": {"path": clin, "target": "label", "sample_id": "sample_id",
                     "group": "patient_id", "batch": "batch", "task": "classification"},
        "n_splits": 3, "neural": False,
    }
    cfg_dict = I.build_config_dict(session)
    cfg = OmicauConfig.from_dict(cfg_dict)
    ad = load_and_align(cfg)                                  # the config actually runs
    assert ad.n_samples == 80 and ad.groups is not None and ad.batch is not None


def test_api_full_flow_end_to_end(tmp_path):
    # Exercises the whole wizard API path against the real library, on a tiny
    # dataset: upload -> roles -> clinical mapping + consequences -> alignment ->
    # options -> preflight -> run (background thread) -> progress -> report.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from omicau.ui.server import create_app

    write_mock_dataset(tmp_path / "ds", task="classification", seed=1, n_samples=48,
                       signal_features=8, redundant_features=6, confounded_features=6,
                       noise_features=4)
    d = tmp_path / "ds"
    hdr = {"X-Omicau-Token": "t"}
    client = TestClient(create_app(token="t", workspace=tmp_path / "ws"))

    sid = client.post("/api/session", headers=hdr).json()["session"]

    files = [("files", (f"{n}.csv", (d / f"{n}.csv").read_bytes(), "text/csv"))
             for n in ("signal", "redundant", "confounded", "noise", "clinical")]
    up = client.post(f"/api/session/{sid}/upload", files=files, headers=hdr).json()
    assert len(up["files"]) == 5

    roles = {"signal.csv": "rna", "redundant.csv": "protein", "confounded.csv": "methylation",
             "noise.csv": "mirna", "clinical.csv": "clinical"}
    r = client.post(f"/api/session/{sid}/roles", json={"roles": roles}, headers=hdr).json()
    assert r["ok"] is True

    cols = client.get(f"/api/session/{sid}/clinical", headers=hdr).json()
    assert any(c["name"] == "label" for c in cols["columns"])
    cons = client.get(f"/api/session/{sid}/consequence",
                      params={"column": "label", "kind": "target"}, headers=hdr).json()
    assert cons["ok"] and cons["task"] == "classification"

    client.post(f"/api/session/{sid}/clinical-map",
                json={"target": "label", "sample_id": "sample_id", "group": "patient_id",
                      "batch": "batch", "task": "classification"}, headers=hdr)
    al = client.post(f"/api/session/{sid}/align", headers=hdr).json()
    assert al["ok"] and al["matched_all_layers"] == 48

    client.post(f"/api/session/{sid}/options", json={"n_splits": 2, "neural": False}, headers=hdr)
    pf = client.post(f"/api/session/{sid}/preflight", headers=hdr).json()
    assert pf["ok"] and pf["n_samples"] == 48 and pf["provenance_hash"]

    assert client.post(f"/api/session/{sid}/run", headers=hdr).json()["started"] is True
    for _ in range(240):                                   # poll up to ~60s
        prog = client.get(f"/api/session/{sid}/progress", headers=hdr).json()
        if prog["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert prog["status"] == "done", prog.get("error")
    assert prog["report_ready"] and prog["provenance"] == pf["provenance_hash"]
    rep = client.get(f"/api/session/{sid}/report", headers=hdr)
    assert rep.status_code == 200 and "<!doctype html>" in rep.text.lower()
