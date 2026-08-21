from __future__ import annotations

import ast
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import omicau.models.fit_inventory as inventory_module
from omicau.models.fit_inventory import FitInventoryError, derive_fit_callsite_inventory


ROOT = Path(__file__).resolve().parents[1]


def _temporary_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    shutil.copytree(ROOT / "omicau", root / "omicau", ignore=shutil.ignore_patterns("__pycache__"))
    return root


def _replace(root: Path, path: str, old: str, new: str) -> None:
    target = root / path
    text = target.read_text(encoding="utf-8")
    assert old in text
    target.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")


def test_live_static_inventory_is_exact_and_development_only() -> None:
    receipt = derive_fit_callsite_inventory(ROOT)
    assert receipt["claim_id"] == "C08"
    assert receipt["decision"] == "development_only"
    assert receipt["fit_callsite_count"] == 14
    assert receipt["active_runtime_callsite_count"] == 13
    assert receipt["registered_runtime_callsite_count"] == 18
    assert receipt["inactive_registered_callsite_count"] == 5
    assert receipt["production_status"] == "pending_runtime_trace_and_frozen_inventory"
    assert receipt["schema_version"] == "c08_static_fit_callsite_inventory_v2"
    assert receipt["source_count"] == len(tuple((ROOT / "omicau").rglob("*.py")))
    assert len(receipt["fit_callsite_inventory_sha256"]) == 64
    assert len(receipt["source_bundle_sha256"]) == 64


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        ("omicau/models/base.py", "pipe.fit(X[train_idx], training_target)", "pipe.predict(X[train_idx])"),
        ("omicau/models/classical.py", "meta.fit(meta_train, y[outer_train])", "meta.predict(meta_train)"),
        ("omicau/models/neural.py", "mean, std = _masked_stats(raw[m], fit_idx)", "mean, std = (0.0, 1.0)"),
        ("omicau/models/neural.py", "opt.step()", "model.zero_grad()"),
        ("omicau/models/survival.py", "beta = cox_fit(", "beta = tuple("),
        ("omicau/diagnostics/batch.py", "pca.fit_transform(Xs)", "pca.transform(Xs)"),
        ("omicau/cli.py", ").fit(X0, y0)", ").predict(X0)"),
    ],
)
def test_omitted_registered_callsite_fails(tmp_path: Path, path: str, old: str, new: str) -> None:
    root = _temporary_repository(tmp_path)
    _replace(root, path, old, new)
    with pytest.raises(FitInventoryError, match="c08_static_runtime_inventory_mismatch") as error:
        derive_fit_callsite_inventory(root)
    assert error.value.invariant == "fit_callsite_set_exact"


def test_unregistered_fit_call_in_previously_unlisted_module_fails(tmp_path: Path) -> None:
    root = _temporary_repository(tmp_path)
    target = root / "omicau/config.py"
    target.write_text(target.read_text(encoding="utf-8") + "\ndef unregistered_training(model, X, y):\n    model.fit(X, y)\n", encoding="utf-8", newline="")
    with pytest.raises(FitInventoryError) as error:
        derive_fit_callsite_inventory(root)
    assert error.value.invariant == "fit_callsite_set_exact"


def test_new_package_module_fit_call_fails(tmp_path: Path) -> None:
    root = _temporary_repository(tmp_path)
    (root / "omicau/new_training.py").write_text("def train(model, X, y):\n    model.fit(X, y)\n", encoding="ascii")
    with pytest.raises(FitInventoryError) as error:
        derive_fit_callsite_inventory(root)
    assert error.value.invariant == "fit_callsite_set_exact"


def test_receiver_or_scope_drift_fails(tmp_path: Path) -> None:
    root = _temporary_repository(tmp_path)
    _replace(root, "omicau/models/base.py", "pipe.fit(X[train_idx], training_target)", "candidate.fit(X[train_idx], training_target)")
    with pytest.raises(FitInventoryError) as error:
        derive_fit_callsite_inventory(root)
    assert error.value.invariant == "fit_callsite_set_exact"


def test_source_parse_and_read_fail_closed(tmp_path: Path) -> None:
    root = _temporary_repository(tmp_path)
    (root / "omicau/config.py").write_text("def broken(:\n", encoding="ascii")
    with pytest.raises(FitInventoryError) as malformed:
        derive_fit_callsite_inventory(root)
    assert malformed.value.invariant == "source_ast_parse"
    with pytest.raises(FitInventoryError) as missing:
        derive_fit_callsite_inventory(tmp_path / "missing")
    assert missing.value.invariant in {"source_path_unreadable", "repository_root_unreadable"}


def test_semantic_spec_disposition_and_trace_plan_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    first = inventory_module._SPECS[0]
    monkeypatch.setattr(inventory_module, "_SPECS", (replace(first, disposition="traced_analysis_fit"), *inventory_module._SPECS[1:]))
    with pytest.raises(FitInventoryError) as error:
        derive_fit_callsite_inventory(ROOT)
    assert error.value.invariant in {"traced_spec_trace_plan_nonempty", "semantic_spec_sha256"}


def test_conditional_trace_plan_uses_registered_runtime_callsite(monkeypatch: pytest.MonkeyPatch) -> None:
    base = inventory_module._SPECS[3]
    changed = replace(base, trace_plan=(*base.trace_plan[:-1], inventory_module._TraceNode("base.feature_selector:when_present")))
    monkeypatch.setattr(inventory_module, "_SPECS", (*inventory_module._SPECS[:3], changed, *inventory_module._SPECS[4:]))
    with pytest.raises(FitInventoryError) as error:
        derive_fit_callsite_inventory(ROOT)
    assert error.value.invariant == "semantic_spec_callsite_registered"


def test_source_bundle_binds_all_package_sources_but_call_inventory_is_stable(tmp_path: Path) -> None:
    root = _temporary_repository(tmp_path)
    first = derive_fit_callsite_inventory(root)
    target = root / "omicau/config.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# inventory binding\n", encoding="utf-8")
    second = derive_fit_callsite_inventory(root)
    assert first["fit_callsite_inventory_sha256"] == second["fit_callsite_inventory_sha256"]
    assert first["source_bundle_sha256"] != second["source_bundle_sha256"]


def test_receipt_is_aggregate_and_deterministic() -> None:
    first = derive_fit_callsite_inventory(ROOT)
    second = derive_fit_callsite_inventory(ROOT)
    assert first == second
    rendered = repr(first)
    assert str(ROOT) not in rendered
    assert "sample" not in rendered.lower()
    assert "subject" not in rendered.lower()
    assert "seed" not in rendered.lower()


def test_public_module_has_no_dynamic_execution() -> None:
    tree = ast.parse((ROOT / "omicau/models/fit_inventory.py").read_text(encoding="utf-8"))
    names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not ({"eval", "exec", "compile"} & names)


def test_inventory_digests_are_not_seed_digests() -> None:
    receipt = derive_fit_callsite_inventory(ROOT)
    seed_digest = hashlib.sha256(b"31").hexdigest()
    assert receipt["fit_callsite_inventory_sha256"] != seed_digest
    assert receipt["source_bundle_sha256"] != seed_digest
