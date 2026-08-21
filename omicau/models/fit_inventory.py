"""Static inventory of data-dependent training calls used by C08 development checks."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from omicau.models.fit_trace import _CALLSITE_COMPONENTS


class FitInventoryError(ValueError):
    """Fixed public failure for a static fit-callsite inventory mismatch."""

    def __init__(self, invariant: str):
        self.code = "c08_static_runtime_inventory_mismatch"
        self.invariant = invariant
        super().__init__(self.code)


def _fail(invariant: str) -> None:
    raise FitInventoryError(invariant)


@dataclass(frozen=True)
class _TraceNode:
    callsite_id: str
    condition: str = "always"


@dataclass(frozen=True)
class _Spec:
    path: str
    scope: str
    operation: str
    target: str
    disposition: str
    trace_plan: tuple[_TraceNode, ...]


def _nodes(*callsite_ids: str) -> tuple[_TraceNode, ...]:
    return tuple(_TraceNode(callsite_id) for callsite_id in callsite_ids)


_PIPELINE_NODES = (
    *_nodes("base.imputer", "base.variance_filter", "base.scaler"),
    _TraceNode("base.feature_selector", "pipeline_step_present"),
    _TraceNode("base.model"),
)
_SPECS = (
    _Spec("omicau/cli.py", "estimate_runtime", "fit", "RandomForestClassifier(n_estimators=tr0, n_jobs=cores).fit", "excluded_runtime_probe", ()),
    _Spec("omicau/diagnostics/batch.py", "_pca_project", "fit_transform", "StandardScaler().fit_transform", "excluded_descriptive_diagnostic", ()),
    _Spec("omicau/diagnostics/batch.py", "_pca_project", "fit_transform", "pca.fit_transform", "excluded_descriptive_diagnostic", ()),
    _Spec("omicau/models/base.py", "cross_validate_estimator", "fit", "pipe.fit", "traced_analysis_fit", _PIPELINE_NODES),
    _Spec("omicau/models/classical.py", "_run_nested_stacking", "fit", "pipe.fit", "traced_inner_base_fit", _PIPELINE_NODES),
    _Spec("omicau/models/classical.py", "_run_nested_stacking", "fit", "base.fit", "traced_outer_base_fit", _PIPELINE_NODES),
    _Spec("omicau/models/classical.py", "_run_nested_stacking", "fit", "meta.fit", "traced_stacking_fit", _nodes("base.imputer", "base.variance_filter", "base.scaler", "stacking.stacker")),
    _Spec("omicau/models/classical.py", "_run_batch_adjusted_fusion", "fit_batch_centering", "fit_batch_centering", "traced_batch_adjustment_fit", _nodes("batch.adjuster")),
    _Spec("omicau/models/classical.py", "_run_batch_adjusted_fusion", "fit", "pipe.fit", "traced_batch_adjusted_model_fit", _PIPELINE_NODES),
    _Spec("omicau/models/neural.py", "_neural_cv", "_masked_stats", "_masked_stats", "traced_neural_scaler_fit", _nodes("neural.scaler")),
    _Spec("omicau/models/neural.py", "_train_fold", "step", "opt.step", "traced_neural_optimizer_step", _nodes("neural.optimizer")),
    _Spec("omicau/models/survival.py", "_Preproc.fit", "fit", "PCA(n_components=self.n_comp_, random_state=0).fit", "traced_survival_pca_fit", _nodes("survival.pca")),
    _Spec("omicau/models/survival.py", "_cv_cindex", "fit", "_Preproc(max_features).fit", "traced_survival_preprocessing_fit", _nodes("survival.imputer", "survival.scaler")),
    _Spec("omicau/models/survival.py", "_cv_cindex", "cox_fit", "cox_fit", "traced_survival_model_fit", _nodes("survival.cox")),
)
_TRAINING_OPERATIONS = frozenset({"_masked_stats", "cox_fit", "fit", "fit_batch_centering", "fit_transform", "step"})
_TRACE_CONDITIONS = frozenset({"always", "pipeline_step_present"})
_EXCLUDED_DISPOSITIONS = frozenset({"excluded_descriptive_diagnostic", "excluded_runtime_probe"})
_TRACED_DISPOSITIONS = frozenset({
    "traced_analysis_fit", "traced_batch_adjusted_model_fit", "traced_batch_adjustment_fit",
    "traced_inner_base_fit", "traced_neural_optimizer_step", "traced_neural_scaler_fit",
    "traced_outer_base_fit", "traced_stacking_fit", "traced_survival_model_fit",
    "traced_survival_pca_fit", "traced_survival_preprocessing_fit",
})
_EXPECTED_SPEC_SHA256 = "dbc3b403427de8e8a181681057abb5b391d43459d1b97241e5d1a35af21e0cff"


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.scope: list[str] = []
        self.calls: list[tuple[str, str, str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        operation = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else None
        if operation in _TRAINING_OPERATIONS:
            self.calls.append((self.path, ".".join(self.scope), operation, ast.unparse(function)))
        self.generic_visit(node)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")


def _spec_entries() -> list[dict[str, object]]:
    return [
        {
            "disposition": spec.disposition,
            "operation": spec.operation,
            "scope": spec.scope,
            "source_ref": spec.path,
            "target": spec.target,
            "trace_plan": [{"callsite_id": node.callsite_id, "condition": node.condition} for node in spec.trace_plan],
        }
        for spec in _SPECS
    ]


def _validate_spec_semantics() -> None:
    if len(_SPECS) != len({(spec.path, spec.scope, spec.operation, spec.target) for spec in _SPECS}):
        _fail("semantic_spec_unique")
    for spec in _SPECS:
        if spec.disposition in _EXCLUDED_DISPOSITIONS:
            if spec.trace_plan:
                _fail("excluded_spec_trace_plan_empty")
        elif spec.disposition in _TRACED_DISPOSITIONS:
            if not spec.trace_plan:
                _fail("traced_spec_trace_plan_nonempty")
        else:
            _fail("semantic_spec_disposition")
        ids = [node.callsite_id for node in spec.trace_plan]
        if len(ids) != len(set(ids)):
            _fail("semantic_spec_trace_plan_unique")
        for node in spec.trace_plan:
            if node.callsite_id not in _CALLSITE_COMPONENTS:
                _fail("semantic_spec_callsite_registered")
            if node.condition not in _TRACE_CONDITIONS:
                _fail("semantic_spec_condition_registered")
    if hashlib.sha256(_canonical_json(_spec_entries())).hexdigest() != _EXPECTED_SPEC_SHA256:
        _fail("semantic_spec_sha256")


def _is_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        _fail("source_path_unreadable")
    attributes = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _read_package_sources(repository_root: str | os.PathLike[str]) -> dict[str, bytes]:
    root = Path(repository_root)
    if _is_reparse(root):
        _fail("repository_root_reparse")
    try:
        root = root.resolve(strict=True)
    except OSError:
        _fail("repository_root_unreadable")
    package = root / "omicau"
    if not package.is_dir() or _is_reparse(package):
        _fail("package_root_invalid")
    try:
        paths = sorted(package.rglob("*.py"))
    except OSError:
        _fail("source_enumeration")
    if not paths:
        _fail("source_enumeration_empty")
    sources: dict[str, bytes] = {}
    for path in paths:
        if _is_reparse(path):
            _fail("source_reparse")
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
            before = resolved.stat()
            raw = resolved.read_bytes()
            after = resolved.stat()
        except (OSError, ValueError):
            _fail("source_read")
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity or len(raw) != before.st_size:
            _fail("source_read_race")
        sources[relative] = raw
    return sources


def derive_fit_callsite_inventory(repository_root: str | os.PathLike[str]) -> dict[str, object]:
    """Derive the package-wide inventory from one repository root."""
    _validate_spec_semantics()
    sources = _read_package_sources(repository_root)
    observed: list[tuple[str, str, str, str]] = []
    source_hashes: dict[str, str] = {}
    for path, raw in sources.items():
        try:
            tree = ast.parse(raw.decode("utf-8"), filename=path)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            _fail("source_ast_parse")
        visitor = _CallVisitor(path)
        visitor.visit(tree)
        observed.extend(visitor.calls)
        source_hashes[path] = hashlib.sha256(raw).hexdigest()
    expected = [(spec.path, spec.scope, spec.operation, spec.target) for spec in _SPECS]
    if sorted(observed) != sorted(expected):
        _fail("fit_callsite_set_exact")
    entries = sorted(_spec_entries(), key=_canonical_json)
    active_callsite_ids = {node.callsite_id for spec in _SPECS for node in spec.trace_plan}
    return {
        "active_runtime_callsite_count": len(active_callsite_ids),
        "claim_id": "C08",
        "decision": "development_only",
        "fit_callsite_count": len(entries),
        "fit_callsite_inventory_sha256": hashlib.sha256(_canonical_json(entries)).hexdigest(),
        "inactive_registered_callsite_count": len(_CALLSITE_COMPONENTS) - len(active_callsite_ids),
        "production_status": "pending_runtime_trace_and_frozen_inventory",
        "registered_runtime_callsite_count": len(_CALLSITE_COMPONENTS),
        "schema_version": "c08_static_fit_callsite_inventory_v2",
        "source_bundle_sha256": hashlib.sha256(_canonical_json(source_hashes)).hexdigest(),
        "source_count": len(sources),
    }
