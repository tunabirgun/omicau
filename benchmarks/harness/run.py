"""Benchmark harness: frozen splits, shared preprocessing, streaming disk use.

Design constraints are read from the machine-readable v1.0.0 protocol:

* every method sees byte-identical outer splits, loaded from benchmarks/splits/;
* all preprocessing is fitted inside the training fold (omicau's own pipeline is
  reused, so the tool and its baselines are preprocessed identically);
* out-of-fold predictions are persisted, because every paired comparison needs them;
* generated simulation matrices are disposable intermediates -- each regenerates
  exactly from its archived seed -- so they are deleted after a dataset is scored.
  Results, out-of-fold predictions and split files are never deleted.

Two passes, in protocol order: `--freeze-splits` writes and checksums the splits for a
dataset; `--run` regenerates the dataset, loads those splits, and scores every method.

Usage
    python run.py --freeze-splits --profile smoke
    python run.py --run           --profile smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
RECORD = REPO / "benchmark_record"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BENCH / "simulations"))

from generate import (STREAM_LABELS, Spec, UnitKey, generate_unit, protocol_units,
                      seed_for)  # noqa: E402

try:  # package import (tests/tools) and direct script execution
    from .audit import run_audit, summarize  # noqa: E402
    from .artifacts import (DatasetStage, atomic_move, atomic_write_text, marker_path,
                            quarantine, sha256_bytes, sha256_file, split_digest,
                            verify_marker, write_marker, MARKER_VERSION)  # noqa: E402
except ImportError:  # pragma: no cover - direct ``python run.py`` path
    from audit import run_audit, summarize  # noqa: E402
    from artifacts import (DatasetStage, atomic_move, atomic_write_text, marker_path,
                           quarantine, sha256_bytes, sha256_file, split_digest,
                           verify_marker, write_marker, MARKER_VERSION)  # noqa: E402

from omicau.models.base import make_pipeline, score_predictions  # noqa: E402

PROTOCOL = yaml.safe_load((RECORD / "benchmark_protocol.yaml").read_text(encoding="utf-8"))
PROTOCOL_VERSION = PROTOCOL["record"]["version"]
_SIM = PROTOCOL["simulation"]
_SIM_CV = _SIM["outer_cv"]
N_SPLITS = _SIM_CV["folds"]
N_REPEATS = _SIM_CV["repeats"]
MAX_FEATURES = 2000

SPLIT_DIR = BENCH / "splits"
RESULT_DIR = BENCH / "results" / "raw"
OOF_DIR = BENCH / "results" / "raw" / "oof"
FAIL_DIR = BENCH / "failures"
WORK_DIR = BENCH / "runs"
DEVELOPMENT_DIR = REPO / "local" / "benchmark_smoke"
ARTIFACT_ROOT = BENCH


class NotReadyError(RuntimeError):
    """A protocol cell is registered but its required input has not been supplied."""

WorkUnit = UnitKey


def protocol_work() -> list[WorkUnit]:
    """Expand to the registered independent experimental units.

    A group-leakage safe/unsafe pair is deliberately *one* unit: it regenerates
    one set of repeated measurements and evaluates both splitters against it.
    This makes the protocol's 3,890 units derivable from the YAML instead of a
    fragile hand-maintained total.
    """
    return protocol_units(PROTOCOL)


def smoke_work(families: set[str] | None = None) -> list[WorkUnit]:
    """Return one protocol-derived development unit per selected family.

    Without a selection this is the first registered unit, which keeps the default
    development check cheap. Naming families picks the first unit of each, so a
    structure such as the paired group-leakage arms can be exercised end to end.
    No smoke output is ever manuscript evidence.
    """
    first: dict[str, WorkUnit] = {}
    for unit in protocol_work():
        if families is not None and unit.family not in families:
            continue
        first.setdefault(unit.family, unit)
    selected = list(first.values())
    return selected if families is not None else selected[:1]


def core_work() -> list[WorkUnit]:
    return protocol_work()


# --------------------------------------------------------------------------- #
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_id(scenario: str, n: int, rep: int, overlay: str | None = None,
               family: str | None = None) -> str:
    stem = f"{scenario}_n{n}_r{rep}"
    if overlay:
        stem = f"{stem}_{overlay}"
    return f"{family}__{stem}" if family else stem


def unit_dataset_id(unit: WorkUnit) -> str:
    return dataset_id(unit.scenario_or_structure, unit.sample_size or 0,
                      unit.replicate_index, unit.condition_or_perturbation, unit.family)


def build(unit: WorkUnit, spec):
    """Regenerate a simulation unit from its archived seed."""
    if unit.family == "semi_synthetic_robustness":
        raise NotReadyError(
            "semi-synthetic robustness is registered but not runnable: it requires "
            f"the eligible {unit.scenario_or_structure!r} template and its frozen "
            "provenance/missingness manifest. No template adapter is present."
        )
    if unit.sample_size is None or not unit.scenario_or_structure.startswith("S"):
        raise NotReadyError(f"family {unit.family!r} has no registered generator adapter")
    condition = unit.condition_or_perturbation
    ds = generate_unit(unit, spec)
    ds.setdefault("overlay", condition)
    ds["family"] = unit.family
    ds["condition"] = condition
    ds["dataset_id"] = unit_dataset_id(unit)
    return ds


def safe_name(method: str) -> str:
    """Method ids carry '::' (single::rna_like); Windows rejects it in filenames."""
    return method.replace("::", "__")


def provenance_hash(mats: dict[str, np.ndarray], y: np.ndarray) -> str:
    h = hashlib.sha256()
    for name in sorted(mats):
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.nan_to_num(mats[name], nan=-9e18)).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def configure_output_paths(profile: str, root: Path | None = None) -> Path:
    """Route every emitted artifact through one profile-specific root."""
    global ARTIFACT_ROOT, SPLIT_DIR, RESULT_DIR, OOF_DIR, FAIL_DIR, WORK_DIR
    selected = (root if root is not None else
                (DEVELOPMENT_DIR if profile == "smoke" else BENCH)).resolve()
    if profile == "smoke":
        try:
            selected.relative_to(BENCH.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("smoke artifacts must not be written under definitive benchmarks/")
    ARTIFACT_ROOT = selected
    SPLIT_DIR = selected / "splits"
    RESULT_DIR = selected / "results" / "raw"
    OOF_DIR = RESULT_DIR / "oof"
    FAIL_DIR = selected / "failures"
    WORK_DIR = selected / "runs"
    return selected


def predictive_methods(spec: Spec) -> list[str]:
    """The executed method set.

    ``nested_best_single``, ``early_concat_elastic_net``,
    ``early_concat_random_forest``, ``late_stacking_fully_nested`` and
    ``omicau_masked_fusion`` are the registered primary comparators and the system
    under test (COMPARATOR_MANIFEST.yaml#comparators). The per-modality
    ``single::`` fits are the inputs ``nested_best_single`` selects among, retained
    so the selection is auditable. ``early_concat_hist_gb`` is NOT in the registered
    comparator set; it is retained as an exploratory arm and must not enter a
    primary ranking without a deviation record.
    """
    return [*(f"single::{name}" for name in spec.names),
            "nested_best_single",
            "early_concat_elastic_net", "early_concat_random_forest",
            "early_concat_hist_gb", "late_stacking_fully_nested",
            "omicau_masked_fusion"]


UNREGISTERED_METHODS = ("early_concat_hist_gb",)


def task_index_records(work: list[WorkUnit], spec: Spec,
                       profile: str) -> list[dict]:
    """Derive the execution contract from the selected protocol units."""
    phase = "development" if profile == "smoke" else "definitive"
    records: list[dict] = []
    for unit in work:
        ds_id = unit_dataset_id(unit)
        streams = {label: seed_for(unit, label) for label in STREAM_LABELS}
        arms = ("safe", "unsafe") if unit.family == "group_leakage" else ("standard",)
        for rep in range(N_REPEATS):
            for arm in arms:
                split_id = f"{ds_id}_rep{rep}_{arm}"
                for method in predictive_methods(spec):
                    records.append({
                        "dataset_id": ds_id, "method_id": method, "repeat": rep,
                        "split_id": split_id, "splitter_arm": arm,
                        "profile": profile, "phase": phase, "status": "expected",
                        "unit_key": unit.as_dict(), "stream_seeds": streams,
                    })
        records.append({
            "dataset_id": ds_id, "method_id": "omicau_audit", "repeat": 0,
            "split_id": f"{ds_id}_internal", "splitter_arm": "internal",
            "profile": profile, "phase": phase, "status": "expected",
            "unit_key": unit.as_dict(), "stream_seeds": streams,
        })
    return records


def write_task_index(work: list[WorkUnit], spec: Spec, profile: str) -> Path:
    records = task_index_records(work, spec, profile)
    path = ARTIFACT_ROOT / "task_index.json"
    payload = {"protocol_version": PROTOCOL_VERSION, "profile": profile,
               "tasks": records}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing task index disagrees with the selected work: {path}")
        return path
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def logical_task_key(record: dict) -> tuple[str, str, str, str | None]:
    """The identity a task index row, a result row and a failure record share.

    The splitter arm is carried by ``split_id``; without it the safe and unsafe
    arms of a group-leakage unit collapse onto one key and each looks like a
    duplicate of the other.
    """
    repeat = record.get("repeat", record.get("repeat_index", 0))
    split_id = record.get("split_id")
    return (str(record.get("dataset_id")), str(record.get("method_id")),
            str(0 if repeat is None else repeat),
            None if split_id is None else str(split_id))


def expected_keys_by_dataset(records: list[dict]) -> dict[str, set[tuple]]:
    index: dict[str, set[tuple]] = {}
    for record in records:
        index.setdefault(str(record["dataset_id"]), set()).add(logical_task_key(record))
    return index


def write_build_manifest(unit: WorkUnit, ds: dict, profile: str) -> Path:
    path = WORK_DIR / "build" / f"{unit_dataset_id(unit)}.json"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "profile": profile,
        "phase": "development" if profile == "smoke" else "definitive",
        "dataset_id": unit_dataset_id(unit),
        "unit_key": ds["unit_key"],
        "registry_unit_key": ds["registry_unit_key"],
        "stream_seeds": ds["stream_seeds"],
        "shapes": {name: list(values.shape) for name, values in ds["matrices"].items()},
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing build manifest disagrees with regenerated unit: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _split_path(ds_id: str, rep: int, splitter_arm: str | None) -> Path:
    arm_suffix = f"_{splitter_arm}" if splitter_arm else ""
    return SPLIT_DIR / f"{ds_id}_rep{rep}{arm_suffix}.json"


def freeze_splits(ds: dict, spec: Spec, splitter_arm: str | None = None) -> list[Path]:
    """One frozen split file per (dataset, repeat), written before any method runs."""
    y, groups = ds["y"], ds["groups"]
    out = []
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for rep in range(N_REPEATS):
        if rep != 0:
            raise RuntimeError("the seed registry provides one fold-assignment stream per unit")
        seed = int(ds["stream_seeds"]["fold_assignment"])
        k = min(N_SPLITS, int(np.bincount(y).min()))
        if k < 2:
            raise SystemExit(f"cannot split {ds.get('dataset_id', dataset_id(ds['scenario'], ds['n'], ds['replicate'], ds.get('overlay'), ds.get('family')))}: "
                             f"smallest class has {int(np.bincount(y).min())} members")
        if splitter_arm == "unsafe":
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.zeros(len(y)), y)
            splitter_name = "StratifiedKFold"
        else:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.zeros(len(y)), y, groups)
            splitter_name = "StratifiedGroupKFold"
        folds = [{"fold": i, "train": tr.tolist(), "test": te.tolist()}
                 for i, (tr, te) in enumerate(split_iter)]
        payload = {
            "dataset_id": ds.get("dataset_id", dataset_id(
                ds["scenario"], ds["n"], ds["replicate"], ds.get("overlay"), ds.get("family"))),
            "protocol_version": PROTOCOL_VERSION, "repeat": rep, "seed": seed,
            "splitter": splitter_name, "splitter_arm": splitter_arm or "standard",
            "n_splits": k, "shuffle": True,
            # The unsafe challenge arm deliberately ignores the subject column, so
            # naming one here would misdescribe the partition on its own manifest.
            "group_column": None if splitter_arm == "unsafe" else "subject",
            "n_samples": int(len(y)), "folds": folds,
        }
        payload["sha256"] = split_digest(payload)
        p = _split_path(payload["dataset_id"], rep, splitter_arm)
        atomic_write_text(p, json.dumps(payload))
        out.append(p)
    return out


def load_splits(ds_id: str, rep: int, splitter_arm: str | None = None) -> dict:
    """Load a frozen split and re-derive its digest rather than trusting the stored one."""
    p = _split_path(ds_id, rep, splitter_arm)
    if not p.exists():
        raise SystemExit(f"frozen split missing: {p}. Run --freeze-splits first.")
    payload = json.loads(p.read_text(encoding="utf-8"))
    recomputed = split_digest(payload)
    if payload.get("sha256") != recomputed:
        raise SystemExit(f"frozen split {p} does not match its own contents: "
                         f"stored {payload.get('sha256')!r}, recomputed {recomputed!r}")
    return payload


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def estimator(key: str, seed: int):
    if key == "elastic_net":
        # saga does not converge on 3.5k standardised features within 2000 iterations
        # (observed at n=500 during sizing). An unconverged baseline is a defect, not a
        # default, so the iteration cap is raised and convergence is asserted below.
        return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga",
                                  max_iter=8000, tol=1e-3,
                                  class_weight="balanced", random_state=seed)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=4,
                                      class_weight="balanced", random_state=seed)
    if key == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed)
    if key == "linear":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    raise ValueError(key)


def cv_predict(X: np.ndarray, y: np.ndarray, folds: list[dict], est_key: str, seed: int) -> np.ndarray:
    """Out-of-fold class-1 probabilities on the frozen folds, preprocessing in-fold."""
    oof = np.full(len(y), np.nan)
    for f in folds:
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        pipe = make_pipeline(estimator(est_key, seed), "classification", X.shape[1], MAX_FEATURES, seed)
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def _inner_splitter(k: int, seed: int, group_aware: bool):
    """Inner splitter matching the arm: the unsafe arm must be naive throughout.

    A group-aware inner loop inside the unsafe challenge would partially repair
    the leakage the arm exists to measure, and only the two methods that have an
    inner loop would be repaired -- shrinking the registered naive-minus-group-
    aware gap by a method-dependent amount.
    """
    if group_aware:
        return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def _inner_split_iter(splitter, n: int, y: np.ndarray, groups: np.ndarray, group_aware: bool):
    return (splitter.split(np.zeros(n), y, groups) if group_aware
            else splitter.split(np.zeros(n), y))


def nested_stacking(mats: dict, y: np.ndarray, folds: list[dict], groups: np.ndarray,
                    seed: int, group_aware: bool = True) -> np.ndarray:
    """Fully nested late stacking: meta-features come from an inner loop only."""
    inner_k = PROTOCOL["real_cohorts"]["inner_cv"]["folds"]
    oof = np.full(len(y), np.nan)
    names = sorted(mats)
    for f in folds:
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        y_tr = y[tr]
        k = min(inner_k, int(np.bincount(y_tr).min()))
        if k < 2:
            # Skipping the fold would return a vector with unpredicted positions.
            # A method that cannot score a fold is a retained failure with a stated
            # disposition, not a silently incomplete prediction.
            raise RuntimeError(
                f"inner cross-validation is infeasible for fold {f['fold']}: the "
                f"smallest class in the outer training fold has {int(np.bincount(y_tr).min())} "
                "member(s)"
            )
        inner = _inner_splitter(k, seed, group_aware)
        meta_tr = np.zeros((len(tr), len(names)))
        for i, m in enumerate(names):
            Xm = mats[m][tr]
            for itr, ite in _inner_split_iter(inner, len(tr), y_tr, groups[tr], group_aware):
                pipe = make_pipeline(estimator("linear", seed), "classification",
                                     Xm.shape[1], MAX_FEATURES, seed)
                pipe.fit(Xm[itr], y_tr[itr])
                meta_tr[ite, i] = pipe.predict_proba(Xm[ite])[:, 1]
        meta_te = np.zeros((len(te), len(names)))
        for i, m in enumerate(names):
            pipe = make_pipeline(estimator("linear", seed), "classification",
                                 mats[m].shape[1], MAX_FEATURES, seed)
            pipe.fit(mats[m][tr], y_tr)                    # refit on the full outer-train
            meta_te[:, i] = pipe.predict_proba(mats[m][te])[:, 1]
        meta = make_pipeline(estimator("linear", seed), "classification", len(names), None, seed)
        meta.fit(meta_tr, y_tr)
        oof[te] = meta.predict_proba(meta_te)[:, 1]
    return oof


def nested_best_single(mats: dict, y: np.ndarray, folds: list[dict], groups: np.ndarray,
                       seed: int, group_aware: bool = True) -> np.ndarray:
    """The registered ``nested_best_single`` comparator.

    COMPARATOR_MANIFEST.yaml states the procedure verbatim: fit the same
    leakage-safe elastic-net pipeline to every modality inside the outer training
    set, choose the modality on inner-CV AUROC only, refit on the complete outer
    training set, and score the untouched outer test set. The manifest's
    ``oracle_outer_test_selection: false`` is what the inner loop buys -- picking
    the modality on outer-test AUROC would be selection on the outcome.
    """
    from sklearn.metrics import roc_auc_score

    inner_k = PROTOCOL["real_cohorts"]["inner_cv"]["folds"]
    names = sorted(mats)
    oof = np.full(len(y), np.nan)
    for f in folds:
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        y_tr = y[tr]
        k = min(inner_k, int(np.bincount(y_tr).min()))
        if k < 2:
            raise RuntimeError(
                f"inner cross-validation is infeasible for fold {f['fold']}: the smallest "
                f"class in the outer training fold has {int(np.bincount(y_tr).min())} member(s)"
            )
        inner = _inner_splitter(k, seed, group_aware)
        scores: dict[str, float] = {}
        for m in names:
            Xm = mats[m][tr]
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in _inner_split_iter(inner, len(tr), y_tr, groups[tr], group_aware):
                pipe = make_pipeline(estimator("elastic_net", seed), "classification",
                                     Xm.shape[1], MAX_FEATURES, seed)
                pipe.fit(Xm[itr], y_tr[itr])
                inner_oof[ite] = pipe.predict_proba(Xm[ite])[:, 1]
            observed = np.isfinite(inner_oof)
            scores[m] = (float(roc_auc_score(y_tr[observed], inner_oof[observed]))
                         if observed.any() and len(np.unique(y_tr[observed])) > 1 else float("nan"))
        scored = [m for m in names if not np.isnan(scores[m])]
        if not scored:
            # With no usable inner score there is no evidence to select on, and a
            # tie-break by modality name would return a prediction that looks
            # chosen but was not. That is a retained failure, not a result.
            raise RuntimeError(
                f"no modality produced a usable inner-CV score on fold {f['fold']}; "
                "modality selection has no evidence to act on"
            )
        # Ties break on the sorted modality name, which is fixed before any fit.
        chosen = min(scored, key=lambda m: (-scores[m], m))
        pipe = make_pipeline(estimator("elastic_net", seed), "classification",
                             mats[chosen].shape[1], MAX_FEATURES, seed)
        pipe.fit(mats[chosen][tr], y_tr)
        oof[te] = pipe.predict_proba(mats[chosen][te])[:, 1]
    return oof


def inner_validation_indices(outer_train: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             seed: int, group_aware: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return a validation subset drawn only from an outer training fold.

    ``group_aware`` follows the arm: the unsafe challenge is naive throughout, so
    its early-stopping split ignores subjects exactly as its outer split does.
    """
    outer_train = np.asarray(outer_train, dtype=int)
    y_train = y[outer_train]
    class_counts = np.bincount(y_train)
    class_counts = class_counts[class_counts > 0]
    k = min(3, int(class_counts.min()) if class_counts.size else 0,
            len(np.unique(groups[outer_train])) if group_aware else len(outer_train))
    if k < 2:
        # Tiny development cells can be too small for a second group split.  Reusing
        # outer-train remains leakage-safe; crucially, the outer test fold is never
        # used to select an early-stopping epoch.
        return outer_train, outer_train
    splitter = _inner_splitter(k, seed, group_aware)
    fit_local, validation_local = next(_inner_split_iter(
        splitter, len(outer_train), y_train, groups[outer_train], group_aware))
    return outer_train[fit_local], outer_train[validation_local]


def omicau_fusion(mats: dict, y: np.ndarray, folds: list[dict], groups: np.ndarray,
                  seed: int, group_aware: bool = True) -> np.ndarray:
    """omicau's masked global-pooling fuser, driven on the frozen folds.

    Uses the package's own per-fold training entry point so the model, its masked
    standardization and its early stopping are the shipped ones, not a reimplementation.
    """
    import torch
    from omicau.config import NeuralSpec
    from omicau.models.neural import (_masked_stats, _standardize, _to_batch,
                                      _train_fold, resolve_device)

    cfg = NeuralSpec()
    device = resolve_device("cpu")
    dims = {m: mats[m].shape[1] for m in mats}
    oof = np.full(len(y), np.nan)
    for fold_index, f in enumerate(folds):
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        fit_idx, validation_idx = inner_validation_indices(tr, y, groups, seed + fold_index,
                                                           group_aware)
        arrays = {}
        for m, X in mats.items():
            mean, std = _masked_stats(X, fit_idx)
            xz, mask = _standardize(X, mean, std)
            arrays[m] = {"Xs": xz, "mask": mask}   # key name required by _to_batch
        model = _train_fold(arrays, y, fit_idx, validation_idx, dims, "classification",
                            int(len(np.unique(y))), cfg, device, seed + fold_index, cfg.batch_size)
        model.eval()
        with torch.no_grad():
            logits = model(_to_batch(arrays, te, device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        oof[te] = prob
    return oof


# --------------------------------------------------------------------------- #
def _peak_rss_mib() -> float | None:
    """Best available process peak RSS on Unix and Windows, without a hard dependency."""
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "rss", None)
        return round(float(peak) / (1024 ** 2), 3) if peak is not None else None
    except (ImportError, OSError):
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux returns KiB; macOS returns bytes.  This harness runs on Windows,
            # where psutil above supplies peak_wset, but retain a portable fallback.
            if sys.platform == "darwin":
                peak /= 1024
            return round(float(peak) / 1024, 3)
        except (ImportError, AttributeError, OSError):
            return None


def measure_call(fn):
    """Measure one fit attempt, retaining fields needed by the resource endpoint."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        value = fn()
    except Exception as exc:
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        setattr(exc, "benchmark_cost", {
            "wall_time_seconds": round(wall, 3), "cpu_time_seconds": round(cpu, 3),
            "peak_rss_mib": _peak_rss_mib(), "measured_exclusively": False,
        })
        raise
    return value, {
        "wall_time_seconds": round(time.perf_counter() - wall_start, 3),
        "cpu_time_seconds": round(time.process_time() - cpu_start, 3),
        "peak_rss_mib": _peak_rss_mib(), "measured_exclusively": False,
    }


def score_row(ds, ds_id, rep, split, method, oof, y, groups, mats, cost, profile,
              splitter_arm: str = "standard", group_auroc_gap: float | None = None) -> dict:
    metrics = score_predictions(y, oof, None, "classification")
    return {
        "run_id": f"{ds_id}|rep{rep}|{splitter_arm}|{method}",
        "protocol_version": PROTOCOL_VERSION,
        "profile": profile,
        "phase": "definitive" if profile != "smoke" else "development",
        "dataset_id": ds_id,
        "simulation": {
            "scenario": ds["scenario"], "sample_size": ds["n"], "replicate": ds["replicate"],
            "seed": ds["seed"], "overlay": ds.get("overlay"), "ground_truth_roles": ds["roles"],
            "unit_key": ds["unit_key"], "stream_seeds": ds["stream_seeds"],
            "missingness_mechanism": ds.get("missingness_mechanism"),
        },
        "task": "classification_binary",
        "method_id": method,
        "tuning_arm": "defaults",
        "split_id": f"{ds_id}_rep{rep}_{splitter_arm}",
        "split_sha256": split["sha256"],
        "splitter_arm": splitter_arm,
        "unsafe_challenge": splitter_arm == "unsafe",
        "naive_minus_group_aware_auroc": group_auroc_gap,
        "repeat": rep,
        "primary_metric": "auroc",
        "primary_value": metrics.get("auroc"),
        "secondary_metrics": {k: v for k, v in metrics.items() if k != "auroc"},
        "n_subjects": int(len(np.unique(groups))),
        "n_samples": int(len(y)),
        "n_features_per_modality": {m: int(v.shape[1]) for m, v in mats.items()},
        "modalities": sorted(mats),
        "oof_predictions_path": f"results/raw/oof/{ds_id}_rep{rep}_{splitter_arm}_{safe_name(method)}.npy",
        "cost": {**cost, "configuration": {
            "profile": profile, "splitter": split["splitter"], "splitter_arm": splitter_arm,
            "fold_assignment_seed": split["seed"],
            "model_initialization_seed": ds["stream_seeds"]["model_initialization"],
            "outer_folds": split["n_splits"],
        }, "recovery_attempts": []},
        "software": {"python": sys.version.split()[0], "omicau": __import__("omicau").__version__},
        "provenance_hash": provenance_hash(mats, y),
        "timestamp_utc": utcnow(),
    }


def path_component(ds_id: str) -> str:
    """A filesystem-safe stem for one dataset, confined to the harness work tree."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", ds_id).strip("._")
    if not component:
        raise RuntimeError("refusing to derive a path from an empty dataset identifier")
    return component


def open_stage(profile: str, ds_id: str) -> DatasetStage:
    """Return this dataset's staging area, confined below the harness work root."""
    component = path_component(ds_id)
    root = (WORK_DIR / "staging").resolve()
    candidate = (root / component).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"refusing to stage outside the harness work directory: {candidate}") from exc
    return DatasetStage(ds_id, component, candidate)


def run_dataset(unit: WorkUnit, spec, profile, stage: DatasetStage,
                keep_matrices=False) -> list[dict]:
    """Score one dataset entirely inside ``stage``; nothing is promoted here."""
    ds = build(unit, spec)
    ds_id = unit_dataset_id(unit)
    mats, y, groups = ds["matrices"], ds["y"], ds["groups"]
    rows = []
    model_seed = int(ds["stream_seeds"]["model_initialization"])
    for rep in range(N_REPEATS):
        arms = ("safe", "unsafe") if unit.family == "group_leakage" else ("standard",)
        arm_outputs: dict[str, dict[str, tuple[np.ndarray, dict, dict]]] = {}
        for arm in arms:
            split = load_splits(ds_id, rep, None if arm == "standard" else arm)
            folds, split_seed = split["folds"], split["seed"]
            jobs = {
                **{f"single::{m}": (lambda m=m: cv_predict(mats[m], y, folds, "random_forest", model_seed))
                   for m in mats},
                "early_concat_elastic_net": lambda: cv_predict(np.hstack([mats[m] for m in sorted(mats)]), y, folds, "elastic_net", model_seed),
                "early_concat_random_forest": lambda: cv_predict(np.hstack([mats[m] for m in sorted(mats)]), y, folds, "random_forest", model_seed),
                "nested_best_single": lambda: nested_best_single(
                    mats, y, folds, groups, model_seed, group_aware=(arm != "unsafe")),
                "early_concat_hist_gb": lambda: cv_predict(np.hstack([mats[m] for m in sorted(mats)]), y, folds, "hist_gb", model_seed),
                "late_stacking_fully_nested": lambda: nested_stacking(
                    mats, y, folds, groups, model_seed, group_aware=(arm != "unsafe")),
                "omicau_masked_fusion": lambda: omicau_fusion(
                    mats, y, folds, groups, model_seed, group_aware=(arm != "unsafe")),
            }
            arm_outputs[arm] = {}
            for method, fn in jobs.items():
                try:
                    oof, cost = measure_call(fn)
                except Exception as exc:                   # failures are retained outcomes
                    stage.fail_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(stage.fail_dir / f"{ds_id}_rep{rep}_{arm}_{safe_name(method)}.json", json.dumps({
                        "run_id": f"{ds_id}|rep{rep}|{arm}|{method}",
                        "protocol_version": PROTOCOL_VERSION, "profile": profile,
                        "phase": "development" if profile == "smoke" else "definitive",
                        "dataset_id": ds_id, "method_id": method, "repeat": rep,
                        "task": "classification_binary",
                        "splitter_arm": arm, "unsafe_challenge": arm == "unsafe",
                        "disposition": "numerical_failure", "error_class": type(exc).__name__,
                        "error_message": str(exc)[:400], "cost": getattr(exc, "benchmark_cost", None),
                        "split_id": f"{ds_id}_rep{rep}_{arm}",
                        "configuration": {"model_seed": model_seed,
                                          "fold_assignment_seed": split_seed,
                                          "splitter": split["splitter"]},
                        "recovery_attempts": [], "timestamp_utc": utcnow(),
                    }, indent=1), encoding="utf-8")
                    print(f"    {arm}/{method:<24} FAILED {type(exc).__name__}: {exc}", flush=True)
                    continue
                stage.oof_dir.mkdir(parents=True, exist_ok=True)
                np.save(stage.oof_dir / f"{ds_id}_rep{rep}_{arm}_{safe_name(method)}.npy",
                        oof.astype(np.float32))
                arm_outputs[arm][method] = (oof, cost, split)

        for arm, outputs in arm_outputs.items():
            for method, (oof, cost, split) in outputs.items():
                gap = None
                # The gap is one quantity per (unit, method). Writing it on both
                # arms would let any mean over rows count each unit twice, so it
                # is recorded on the unsafe arm, which is the naive term.
                if (arm == "unsafe" and unit.family == "group_leakage"
                        and method in arm_outputs.get("safe", {})
                        and method in arm_outputs.get("unsafe", {})):
                    safe_oof = arm_outputs["safe"][method][0]
                    unsafe_oof = arm_outputs["unsafe"][method][0]
                    gap = float(score_predictions(y, unsafe_oof, None, "classification").get("auroc")
                                - score_predictions(y, safe_oof, None, "classification").get("auroc"))
                row = score_row(ds, ds_id, rep, split, method, oof, y, groups, mats, cost,
                                profile, arm, gap)
                rows.append(row)
                print(f"    {arm}/{method:<24} AUROC {row['primary_value']:.3f}  "
                      f"{cost['wall_time_seconds']:6.1f}s", flush=True)
    # --- audit pass: the primary outcomes of Track A ------------------------
    # Run once per dataset, not per repeat: the tool's audit carries its own
    # cross-validation, and it is the object under test rather than a comparator.
    try:
        a, audit_cost = measure_call(lambda: run_audit(
            mats, y, groups, ds.get("batch"), model_seed, neural=True))
        s = summarize(a, ds["roles"], bool(ds.get("leakage_present", False)))
        best = (a["ledger"].get("best_model") or {}).get("primary")
        rows.append({
            "run_id": f"{ds_id}|audit", "protocol_version": PROTOCOL_VERSION,
            "profile": profile, "phase": "definitive" if profile != "smoke" else "development",
            "dataset_id": ds_id,
            "simulation": {"scenario": ds["scenario"], "sample_size": ds["n"],
                           "replicate": ds["replicate"], "seed": ds["seed"],
                           "overlay": ds.get("overlay"), "ground_truth_roles": ds["roles"],
                           "unit_key": ds["unit_key"], "stream_seeds": ds["stream_seeds"],
                           "missingness_mechanism": ds.get("missingness_mechanism")},
            "task": "classification_binary", "method_id": "omicau_audit",
            "tuning_arm": "defaults", "split_id": f"{ds_id}_internal", "repeat": 0,
            "primary_metric": "role_macro_f1" if unit.family == "role_recovery" else "auroc",
            "primary_value": s["role_macro_f1"] if unit.family == "role_recovery"
            else (float(best) if best is not None else None),
            "audit": {
                "leakage_alarm": s["leakage_alarm"],
                "batch_confounded": any(s["batch_confounded_called"].values()),
                "batch_confounded_by_modality": s["batch_confounded_called"],
                "batch_structured_by_modality": s["batch_structured_called"],
                "missingness_bias": s["missingness"]["target_associated_warning"],
                "missingness_batch_associated": s["missingness"]["batch_associated_warning"],
                "missingness_by_modality": s["missingness"]["per_modality"],
                "grouping_warning": s["grouping_warning"],
                "grouping_warning_classes": s["grouping_warning_classes"],
                "modality_verdicts": s["modality_roles_called"],
                "marginal_gain": s["marginal_gain"], "marginal_gain_p": s["marginal_gain_p"],
                "cka": s["cka"],
            },
            "secondary_metrics": {"role_accuracy": s["role_accuracy"],
                                  "role_macro_f1": s["role_macro_f1"]},
            "control_baselines": s["controls"],
            "n_subjects": int(len(np.unique(groups))), "n_samples": int(len(y)),
            "n_features_per_modality": {m: int(v.shape[1]) for m, v in mats.items()},
            "modalities": sorted(mats), "oof_predictions_path": None,
            "cost": {**audit_cost, "configuration": {
                "profile": profile, "seed": model_seed, "neural_enabled": True,
                "early_stopping_validation": "internal_training_subset",
            }, "recovery_attempts": []},
            "software": {"python": sys.version.split()[0],
                         "omicau": __import__("omicau").__version__},
            "provenance_hash": provenance_hash(mats, y), "timestamp_utc": utcnow(),
            "notes": json.dumps({"leakage_present": bool(ds.get("leakage_present", False)),
                                 "group_leakage_present": bool(ds.get("group_leakage_present", False)),
                                 "false_certification": s["false_certification"],
                                 "false_alarm": s["false_alarm"],
                                 "grouping_warning": s["grouping_warning"],
                                 "unmapped_verdicts": s["unmapped_verdicts"]}),
        })
        print(f"    {'omicau_audit':<32} role_macro_f1 {s['role_macro_f1']:.2f}  "
              f"alarm {s['leakage_alarm']}  {audit_cost['wall_time_seconds']:6.1f}s", flush=True)
    except Exception as exc:
        stage.fail_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(stage.fail_dir / f"{ds_id}_audit.json", json.dumps({
            "run_id": f"{ds_id}|audit", "protocol_version": PROTOCOL_VERSION,
            "profile": profile, "phase": "development" if profile == "smoke" else "definitive",
            "dataset_id": ds_id, "method_id": "omicau_audit", "repeat": 0,
            "split_id": f"{ds_id}_internal",
            "task": "classification_binary", "disposition": "numerical_failure",
            "error_class": type(exc).__name__, "error_message": str(exc)[:400],
            "cost": getattr(exc, "benchmark_cost", None), "recovery_attempts": [],
            "timestamp_utc": utcnow()}, indent=1))
        print(f"    {'omicau_audit':<32} FAILED {type(exc).__name__}: {exc}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# atomic completion (COMPUTE_PLAN.md §5)
# --------------------------------------------------------------------------- #
def validate_stage(stage: DatasetStage, rows: list[dict],
                   expected: set[tuple]) -> tuple[list[str], set[tuple]]:
    """Check a staged dataset against its task-index contract before promotion.

    Everything here is checked against the staged files, so a dataset that fails
    is never promoted and therefore never looks complete to a later pass.  The
    second return value is the set of indexed tasks the staged output accounts
    for, by a result row or by a retained failure record.
    """
    problems: list[str] = []
    failures = sorted(stage.fail_dir.glob("*.json")) if stage.fail_dir.exists() else []
    observed: dict[tuple, str] = {}
    for row in rows:
        key = logical_task_key(row)
        if key in observed:
            problems.append(f"duplicate staged row for {key}")
        observed[key] = "result"
    for path in failures:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            problems.append(f"unreadable staged failure record {path.name}: {type(exc).__name__}")
            continue
        key = logical_task_key(record)
        if key in observed:
            problems.append(f"{key} has both a result row and a failure record")
        observed[key] = "failure"

    def _order(keys: set[tuple]) -> list[tuple]:
        return sorted(keys, key=lambda k: tuple(map(str, k)))

    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing:
        problems.append(f"{len(missing)} indexed task(s) produced neither a row nor a "
                        f"failure record: {_order(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} staged task(s) are absent from the task index: "
                        f"{_order(extra)[:3]}")

    referenced: set[str] = set()
    for row in rows:
        relative = row.get("oof_predictions_path")
        if row.get("method_id") == "omicau_audit":
            continue
        if not relative:
            problems.append(f"{logical_task_key(row)} has no out-of-fold path")
            continue
        referenced.add(Path(str(relative)).name)
        staged = stage.oof_dir / Path(str(relative)).name
        if not staged.is_file():
            problems.append(f"staged out-of-fold file is missing: {relative}")
            continue
        values = np.load(staged, allow_pickle=False)
        if values.ndim != 1 or not np.isfinite(values).all():
            problems.append(f"staged out-of-fold file is not a finite vector: {relative}")
        elif row.get("n_samples") is not None and values.size != int(row["n_samples"]):
            problems.append(f"staged out-of-fold length {values.size} != n_samples "
                            f"{row['n_samples']}: {relative}")
        stored = row.get("split_sha256")
        arm = row.get("splitter_arm", "standard")
        split_file = _split_path(str(row["dataset_id"]), int(row.get("repeat", 0)),
                                 None if arm == "standard" else arm)
        if not split_file.is_file():
            problems.append(f"frozen split artifact is missing for {logical_task_key(row)}")
        else:
            payload = json.loads(split_file.read_text(encoding="utf-8"))
            recomputed = split_digest(payload)
            if recomputed != payload.get("sha256"):
                problems.append(f"frozen split {split_file.name} disagrees with its own digest")
            if recomputed != stored:
                problems.append(f"row split_sha256 disagrees with the recomputed digest "
                                f"for {logical_task_key(row)}")

    # An out-of-fold file no row names would be promoted and then checksummed into
    # the marker as verified evidence for a prediction nothing cites.
    orphans = sorted(path.name for path in stage.oof_dir.glob("*.npy")
                     if path.name not in referenced)
    if orphans:
        problems.append(f"{len(orphans)} staged out-of-fold file(s) are named by no row: "
                        f"{orphans[:3]}")
    return problems, set(observed)


def _split_files_for(rows: list[dict]) -> dict[Path, str]:
    """The frozen split manifests these rows depend on, with recomputed digests."""
    found: dict[Path, str] = {}
    for row in rows:
        if row.get("method_id") == "omicau_audit":
            continue
        arm = row.get("splitter_arm", "standard")
        path = _split_path(str(row["dataset_id"]), int(row.get("repeat", 0)),
                           None if arm == "standard" else arm)
        if path.is_file():
            found[path] = sha256_file(path)
    return found


def promote_dataset(stage: DatasetStage, rows: list[dict], expected: set[tuple],
                    task_index_sha256: str, profile: str) -> Path:
    """Replace the dataset's final artifacts atomically, then mark it complete.

    Ordering is deliberate. Nothing already on disk is destroyed until every new
    artifact is in place, so a promotion that fails part-way leaves the previous
    attempt's retained failure records intact rather than deleting evidence on
    behalf of a run that never finished.
    """
    stage.write_rows(rows)          # staged first, so a rejected dataset can be inspected
    problems, covered = validate_stage(stage, rows, expected)
    if problems:
        raise RuntimeError(f"{stage.dataset_id} failed pre-promotion validation: "
                           + "; ".join(problems[:5]))

    artifacts: dict[str, str] = {}
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for staged in sorted(stage.oof_dir.glob("*.npy")):
        final = atomic_move(staged, OOF_DIR / staged.name)
        artifacts[final.relative_to(ARTIFACT_ROOT).as_posix()] = sha256_file(final)
    # The frozen split is the one input every row cites by digest, so it belongs
    # in the checksum set: a split edited after promotion must invalidate the
    # dataset rather than be skipped as complete on the next resume.
    for path, digest in _split_files_for(rows).items():
        artifacts[path.relative_to(ARTIFACT_ROOT).as_posix()] = digest
    failures: dict[str, str] = {}
    staged_failures = sorted(stage.fail_dir.glob("*.json")) if stage.fail_dir.exists() else []
    if staged_failures:
        FAIL_DIR.mkdir(parents=True, exist_ok=True)
    for staged in staged_failures:
        final = atomic_move(staged, FAIL_DIR / staged.name)
        failures[final.relative_to(ARTIFACT_ROOT).as_posix()] = sha256_file(final)

    text = "".join(json.dumps(row) + "\n" for row in rows)
    shard = atomic_write_text(RESULT_DIR / "datasets" / f"{stage.component}.jsonl", text)

    # Only now, with every replacement in place, is a superseded record from an
    # earlier attempt removed -- and only if this attempt did not rewrite it.
    promoted = {Path(name).name for name in failures}
    if FAIL_DIR.exists():
        for stale in FAIL_DIR.glob(f"{stage.dataset_id}_*.json"):
            if stale.name not in promoted:
                stale.unlink()

    marker = write_marker(marker_path(WORK_DIR, stage.component), {
        "marker_version": MARKER_VERSION,
        "dataset_id": stage.dataset_id,
        "protocol_version": PROTOCOL_VERSION,
        "profile": profile,
        "phase": "development" if profile == "smoke" else "definitive",
        "task_index_sha256": task_index_sha256,
        "rows": {"path": shard.relative_to(ARTIFACT_ROOT).as_posix(),
                 "sha256": sha256_bytes(text.encode("utf-8")), "count": len(rows)},
        "artifacts": artifacts,
        "failures": failures,
        "covered_task_keys": [list(key) for key in
                              sorted(covered, key=lambda k: tuple(map(str, k)))],
        "completed_utc": utcnow(),
    })
    leftover = stage.discard()
    if leftover:
        print(f"    staging directory could not be removed: {leftover[0]}", flush=True)
    return marker


def rebuild_aggregate(out: Path, order: list[str], task_index_sha256: str | None = None,
                      expected_by_dataset: dict[str, set[tuple]] | None = None) -> int:
    """Rewrite the aggregate row file from the datasets whose markers verify.

    The shards are the record of truth; the aggregate exists so validators and the
    monitor can read one file. Rewriting it atomically means an interrupted pass
    can never leave a half-written line behind, and gating on the marker means the
    aggregate cannot advertise a dataset that resume still considers outstanding.
    """
    parts: list[str] = []
    written = 0
    for ds_id in order:
        component = path_component(ds_id)
        shard = RESULT_DIR / "datasets" / f"{component}.jsonl"
        if not shard.is_file():
            continue
        if task_index_sha256 is not None:
            expected = (expected_by_dataset or {}).get(ds_id, set())
            if verify_marker(marker_path(WORK_DIR, component), ARTIFACT_ROOT,
                             task_index_sha256, expected):
                continue
        text = shard.read_text(encoding="utf-8")
        parts.append(text if text.endswith("\n") or not text else text + "\n")
        written += sum(1 for line in text.splitlines() if line.strip())
    atomic_write_text(out, "".join(parts))
    return written


def warn_about_stale_aggregates(out: Path, profile: str) -> list[Path]:
    """Name sibling aggregates this pass does not maintain, without deleting them.

    Changing the shard layout between passes leaves an older aggregate behind, and
    the phase gate concatenates every ``{profile}*_rows.jsonl`` it finds. Deleting
    another shard's output would be worse than reporting it, so this only reports.
    """
    siblings = [path for path in sorted(RESULT_DIR.glob(f"{profile}*_rows.jsonl"))
                if path != out]
    for path in siblings:
        print(f"warning: {path.name} is not written by this pass; the phase gate reads "
              "it too. Remove it if it is from a superseded shard layout.", flush=True)
    return siblings


def record_harness_defect(ds_id: str, profile: str, exc: BaseException) -> Path:
    """Retain a promotion failure as an outcome instead of abandoning the campaign."""
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(FAIL_DIR / f"{ds_id}_promotion.json", json.dumps({
        "run_id": f"{ds_id}|promotion", "protocol_version": PROTOCOL_VERSION,
        "profile": profile, "phase": "development" if profile == "smoke" else "definitive",
        "dataset_id": ds_id, "method_id": "harness_promotion", "repeat": 0,
        "split_id": f"{ds_id}_internal", "task": "classification_binary",
        "disposition": "harness_defect", "error_class": type(exc).__name__,
        "error_message": str(exc)[:400], "recovery_attempts": [],
        "timestamp_utc": utcnow(),
    }, indent=1))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-only", action="store_true",
                        help="generate and verify units without freezing splits or fitting models")
    action.add_argument("--freeze-splits", action="store_true")
    action.add_argument("--run", action="store_true")
    ap.add_argument("--profile", default="smoke", choices=["smoke", "core", "full"])
    ap.add_argument("--root", type=Path, default=None,
                    help="artifact root; smoke defaults to local/benchmark_smoke")
    ap.add_argument("--keep-matrices", action="store_true")
    ap.add_argument("--shard", default=None, help="i/N -- process only this slice")
    ap.add_argument("--dataset", action="append", default=None,
                    help="scenario:n:replicate, repeatable; overrides the profile work list")
    ap.add_argument("--family", action="append", default=None,
                    help="restrict the work list to these registered families, repeatable")
    ap.add_argument("--aggregate-every", type=int, default=25,
                    help="datasets between atomic rebuilds of the aggregate row file")
    ap.add_argument("--max-consecutive-failures", type=int, default=5,
                    help="stop after this many datasets in a row fail promotion")
    ap.add_argument("--readiness-phase", default=None,
                    help="gate on one execution phase's own prerequisites instead of "
                         "the overall readiness verdict; requires a matching --family")
    args = ap.parse_args()

    # A smoke pass is an explicitly local development check.  Any core/full action
    # is definitive infrastructure and must not create a single split or output
    # before the read-only readiness gate passes.
    if args.profile != "smoke":
        from benchmark_record.tools.readiness import PHASE_FAMILIES, check_readiness
        readiness = check_readiness(REPO)
        if args.readiness_phase:
            # Opt-in, per-phase gate: the phase must clear the prerequisites it
            # actually depends on, and the selected families must belong to it.
            # Phases whose prerequisites are outstanding stay blocked.
            phase = readiness["phases"].get(args.readiness_phase)
            if phase is None:
                raise SystemExit(f"unknown readiness phase {args.readiness_phase!r}; "
                                 f"known: {', '.join(readiness['phases'])}")
            allowed = set(PHASE_FAMILIES.get(args.readiness_phase, ()))
            selected = set(args.family or ())
            if not selected:
                raise SystemExit(f"--readiness-phase requires --family; phase "
                                 f"{args.readiness_phase!r} covers: {', '.join(sorted(allowed))}")
            outside = sorted(selected - allowed)
            if outside:
                raise SystemExit(
                    f"families {', '.join(outside)} are outside readiness phase "
                    f"{args.readiness_phase!r}, whose families are {', '.join(sorted(allowed))}"
                )
            if phase["status"] != "pass":
                detail = "; ".join(phase["blockers"][:3])
                raise SystemExit(f"phase {args.readiness_phase!r} is blocked: {detail}")
            print(f"readiness phase {args.readiness_phase}: PASS "
                  f"(overall readiness is {readiness['status'].upper()}; "
                  "the remaining phases stay blocked)")
        elif readiness["status"] != "pass":
            detail = "; ".join(readiness["blockers"][:3])
            raise SystemExit(
                "core/full execution is blocked by benchmark readiness: "
                f"{detail or 'unknown readiness failure'}"
            )
    configure_output_paths(args.profile, args.root)

    spec = Spec.load()
    families = set(args.family) if args.family else None
    if families:
        registered = {unit.family for unit in protocol_work()}
        unknown = sorted(families - registered)
        if unknown:
            raise SystemExit(f"unregistered family selection: {', '.join(unknown)}; "
                             f"registered families are {', '.join(sorted(registered))}")
    if args.dataset:
        work = []
        for d in args.dataset:
            parts = d.split(":")
            work.append(WorkUnit(
                "manual", parts[0], parts[3] if len(parts) > 3 and parts[3] else None,
                int(parts[1]), int(parts[2])))
        index_work = work
    elif args.profile == "smoke":
        # A development root holds exactly the units it was asked for, so the
        # smoke selection is also its own completeness contract.
        work = index_work = smoke_work(families)
    else:
        # For a definitive profile the contract is the whole protocol: a family
        # executed in a later phase stays visibly outstanding rather than
        # disappearing from the index that validates completeness.
        index_work = core_work()
        work = [unit for unit in index_work if families is None or unit.family in families]

    # Concurrent shards must not race to create the index: os.replace over a file
    # another process has open fails on Windows, and the loser dies before doing
    # any work. A sharded pass therefore requires an index a prior pass wrote.
    index_path = ARTIFACT_ROOT / "task_index.json"
    if args.shard and not index_path.exists():
        raise SystemExit(
            f"sharded execution requires an existing task index: {index_path}. "
            "Run --freeze-splits (or --build-only) once for the whole profile first."
        )
    index_path = write_task_index(index_work, spec, args.profile)
    index_sha256 = sha256_file(index_path)
    print(f"task index: {index_path} ({len(index_work)} unit(s))")
    if families:
        print(f"family selection: {', '.join(sorted(families))} -> {len(work)} unit(s); "
              "omitted families remain registered and unexecuted")
    if args.shard:
        i, n_shards = (int(x) for x in args.shard.split("/"))
        work = [w for j, w in enumerate(work) if j % n_shards == i]
    unavailable = [u for u in work if u.family == "semi_synthetic_robustness"]
    if unavailable:
        structures = ", ".join(sorted({u.scenario_or_structure for u in unavailable}))
        raise SystemExit(
            "semi-synthetic cells are explicitly not ready: missing eligible frozen "
            f"template adapter(s) for {structures}. No cells were omitted; select the "
            "runnable families explicitly with --family once the omission is logged as "
            "a deviation."
        )
    if args.build_only:
        for unit in work:
            ds = build(unit, spec)
            manifest = write_build_manifest(unit, ds, args.profile)
            print(f"{unit_dataset_id(unit)}: build verified -> {manifest}")
        return 0
    if args.freeze_splits:
        for unit in work:
            ds = build(unit, spec)
            arms = ("safe", "unsafe") if unit.family == "group_leakage" else (None,)
            paths = [p for arm in arms for p in freeze_splits(ds, spec, arm)]
            print(f"{unit_dataset_id(unit)}: {len(paths)} split files frozen")
        return 0

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard.replace('/', 'of')}" if args.shard else ""
    out = RESULT_DIR / f"{args.profile}{suffix}_rows.jsonl"
    expected_by_dataset = expected_keys_by_dataset(task_index_records(work, spec, args.profile))

    # Resume is decided by the completion marker and the checksums it records, never
    # by the presence of rows: an interrupted dataset leaves rows nowhere a validator
    # can see them, so a partial attempt can never make a dataset look finished.
    completed: list[str] = []
    pending: list[WorkUnit] = []
    for unit in work:
        ds_id = unit_dataset_id(unit)
        component = path_component(ds_id)
        reasons = verify_marker(marker_path(WORK_DIR, component), ARTIFACT_ROOT,
                                index_sha256, expected_by_dataset.get(ds_id, set()))
        if reasons:
            if marker_path(WORK_DIR, component).exists():
                print(f"re-running {ds_id}: {reasons[0]}", flush=True)
                marker_path(WORK_DIR, component).unlink()
            # The shard goes with the marker. Left behind, it would keep feeding
            # the aggregate rows for a dataset that is queued for a re-run.
            shard = RESULT_DIR / "datasets" / f"{component}.jsonl"
            if shard.exists():
                shard.unlink()
            stage_root = (WORK_DIR / "staging" / component)
            if stage_root.exists():
                moved = quarantine(stage_root, WORK_DIR / "quarantine",
                                   f"incomplete staging for {ds_id}: {reasons[0]}")
                print(f"quarantined orphaned staging -> {moved}", flush=True)
            pending.append(unit)
        else:
            completed.append(ds_id)
    if completed:
        print(f"resuming: {len(completed)} dataset(s) verified complete, "
              f"{len(pending)} outstanding", flush=True)

    order = [unit_dataset_id(unit) for unit in work]
    warn_about_stale_aggregates(out, args.profile)
    since_rebuild = 0
    consecutive_defects = 0
    defects: list[str] = []
    try:
        for unit in pending:
            ds_id = unit_dataset_id(unit)
            print(f"\n{ds_id}", flush=True)
            stage = open_stage(args.profile, ds_id).create()
            try:
                rows = run_dataset(unit, spec, args.profile, stage, args.keep_matrices)
                promote_dataset(stage, rows, expected_by_dataset.get(ds_id, set()),
                                index_sha256, args.profile)
                consecutive_defects = 0
            except Exception as exc:
                # One dataset that cannot be promoted is a retained outcome, not a
                # reason to abandon the remaining units. A run of them is a systemic
                # fault and does stop the pass.
                quarantine(stage.root, WORK_DIR / "quarantine",
                           f"promotion did not complete for {ds_id}: {exc}")
                record_harness_defect(ds_id, args.profile, exc)
                defects.append(ds_id)
                consecutive_defects += 1
                print(f"    PROMOTION FAILED {type(exc).__name__}: {exc}", flush=True)
                if consecutive_defects >= args.max_consecutive_failures:
                    raise SystemExit(
                        f"stopping: {consecutive_defects} consecutive datasets failed "
                        f"promotion (last: {ds_id}). Retained failure records are in "
                        f"{FAIL_DIR}."
                    ) from exc
            since_rebuild += 1
            if args.aggregate_every > 0 and since_rebuild >= args.aggregate_every:
                rebuild_aggregate(out, order, index_sha256, expected_by_dataset)
                since_rebuild = 0
    finally:
        total = rebuild_aggregate(out, order, index_sha256, expected_by_dataset)
        print(f"\nwrote {out.relative_to(REPO)} — {total} rows from "
              f"{len(order)} selected dataset(s)", flush=True)
        if defects:
            print(f"promotion failed for {len(defects)} dataset(s), retained as "
                  f"harness_defect records: {defects[:5]}", flush=True)
    return 1 if defects else 0


if __name__ == "__main__":
    raise SystemExit(main())
