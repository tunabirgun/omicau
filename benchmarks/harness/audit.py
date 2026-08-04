"""Run omicau's own audit on a simulated dataset and score it against the truth.

Track A's primary outcomes are audit outcomes, not predictive ones: leakage-alarm
sensitivity, clean-condition specificity, false-certification rate, modality-role
recovery, redundancy calls. Those come from the tool's own ledger, so this drives the
shipped pipeline rather than reimplementing its decisions.

Note on splits: the cross-method predictive comparison uses the frozen splits, because
that comparison is only meaningful paired. The audit runs the tool's internal
cross-validation, seeded identically -- the audit is the object under test, not a
comparator, and substituting our splits for its own would evaluate something the tool
does not ship.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from omicau.config import OmicauConfig
from omicau.data.alignment import AlignedDataset, ModalityMatrix, check_grouping
from omicau.diagnostics.batch import batch_effect_diagnostics
from omicau.diagnostics.missingness import missingness_diagnostics
from omicau.interpretation.utility import build_utility_ledger
from omicau.models.classical import run_classical_benchmarks
from omicau.models.neural import run_neural_benchmark


def to_aligned(mats: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray,
               batch: np.ndarray | None) -> AlignedDataset:
    ids = [f"S{i:05d}" for i in range(len(y))]
    mods = {}
    for name, X in mats.items():
        cols = [f"{name}::f{j}" for j in range(X.shape[1])]
        mods[name] = ModalityMatrix(name=name,
                                    frame=pd.DataFrame(np.asarray(X, dtype=np.float64),
                                                       index=ids, columns=cols))
    return AlignedDataset(
        modalities=mods,
        y=pd.Series(np.asarray(y), index=ids),
        y_raw=pd.Series(np.asarray(y), index=ids),   # batch confounding tests cross the raw target
        class_names=[str(c) for c in sorted(set(np.asarray(y).tolist()))],
        task="classification",
        sample_ids=ids,
        groups=pd.Series(np.asarray(groups), index=ids),
        batch=None if batch is None else pd.Series(np.asarray(batch), index=ids),
    )


def run_audit(mats, y, groups, batch, seed: int, neural: bool = True) -> dict:
    """Return the tool's audit output for one dataset."""
    aligned = to_aligned(mats, y, groups, batch)
    # The grouping preflight is one of the two registered group-leakage outputs
    # (benchmark_protocol.yaml#diagnostic_interpretation.group_leakage.assessed_by).
    # The CLI runs it; the harness must too, or the output has no source.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            check_grouping(aligned)
            grouping_error = None
        except Exception as exc:
            grouping_error = f"{type(exc).__name__}: {exc}"
    grouping_warnings = [str(item.message) for item in caught]
    cfg = OmicauConfig()
    cfg.seed = seed
    cfg._propagate_seed()
    cfg.xai.enabled = False              # attribution is a separate outcome, priced separately
    cfg.neural.enabled = neural
    # One thread per worker: parallelism comes from running many shards, which keeps
    # every core on independent work instead of splitting one model across threads.
    cfg.compute.cores = 4

    missing_diag = missingness_diagnostics(aligned)
    batch_diag = batch_effect_diagnostics(aligned, seed=seed) if batch is not None else None
    classical = run_classical_benchmarks(aligned, cfg, batch_diag=batch_diag)
    neural_out = run_neural_benchmark(aligned, cfg) if neural else {"enabled": False}
    ledger = build_utility_ledger(aligned, classical, neural_out, batch_diag, missing_diag)
    return {"classical": classical, "neural": neural_out, "ledger": ledger,
            "batch_diag": batch_diag, "missing_diag": missing_diag,
            "grouping": {**classify_grouping_warnings(grouping_warnings),
                         "messages": grouping_warnings, "error": grouping_error}}


# check_grouping emits several distinct warnings. Collapsing them into one boolean
# would be misleading: the "grouping is a no-op" warning fires precisely on the
# clean cells, where every subject contributes one specimen, and stays silent on
# the repeated-specimen cells where grouping is doing real work.
GROUPING_WARNING_CLASSES = (
    ("no_group_column", "No 'group' column set"),
    ("grouping_is_noop", "about one level per sample"),
    ("composite_group", "Composite group"),
)


def classify_grouping_warnings(messages: list[str]) -> dict:
    classified = {name: any(fragment in message for message in messages)
                  for name, fragment in GROUPING_WARNING_CLASSES}
    known = {fragment for _, fragment in GROUPING_WARNING_CLASSES}
    classified["other_warning"] = any(
        not any(fragment in message for fragment in known) for message in messages)
    classified["any_warning"] = bool(messages)
    return classified


# omicau states verdicts in prose for its readers. Scoring needs the four role
# classes the generator constructs, so the prose is mapped back onto them. The mapping
# is by leading phrase, taken from _verdict()/_verdict_single() in the package.
VERDICT_TO_ROLE = [
    ("batch-confounded", "batch_confounded"),
    ("predictive (adds marginal signal)", "predictive"),
    ("predictive (standalone)", "predictive"),
    ("informative alone (fusion gain not significant)", "not_additive"),
    ("informative but non-additive", "not_additive"),
    ("redundant", "not_additive"),
    ("no detectable signal", "control_like"),
]


def verdict_to_role(verdict: str | None) -> str | None:
    if not verdict:
        return None
    for prefix, role in VERDICT_TO_ROLE:
        if verdict.startswith(prefix):
            return role
    return None          # unmapped prose is reported, never silently scored as wrong


def missingness_flags(audit: dict) -> dict:
    """Extract the registered missingness risk flags from the diagnostic itself.

    ``SIMULATION_DESIGN.yaml#missingness_design`` registers a target-associated
    and a batch-associated warning. Both are decided by the diagnostic's own
    FDR-adjusted tests, so they are read from those tests rather than inferred
    from verdict prose.
    """
    diagnostic = audit.get("missing_diag") or {}
    tests = diagnostic.get("tests") or []
    per_modality: dict[str, dict] = {}
    for test in tests:
        modality = test.get("modality")
        if modality is None:
            continue
        record = per_modality.setdefault(
            modality, {"target_associated": False, "batch_associated": False,
                       "min_p_adj_target": None, "min_p_adj_batch": None})
        against = test.get("against")
        if against not in {"target", "batch"}:
            continue
        key = "target_associated" if against == "target" else "batch_associated"
        record[key] = bool(record[key] or test.get("flag"))
        p_adj = test.get("p_adj")
        bucket = "min_p_adj_target" if against == "target" else "min_p_adj_batch"
        if isinstance(p_adj, (int, float)):
            current = record[bucket]
            record[bucket] = p_adj if current is None else min(current, float(p_adj))
    return {
        "alpha": diagnostic.get("alpha"),
        "per_modality": per_modality,
        "target_associated_warning": any(v["target_associated"] for v in per_modality.values()),
        "batch_associated_warning": any(v["batch_associated"] for v in per_modality.values()),
        "overall_missing_fraction": ((diagnostic.get("overall") or {})
                                     .get("total_missing_fraction")),
    }


def summarize(audit: dict, truth_roles: dict, leakage_present: bool) -> dict:
    """Compress the audit into the fields the result schema records, and score the
    verdicts against the generator's ground truth."""
    led = audit["ledger"]
    entries = {e["modality"]: e for e in led.get("modality_ledger", [])}
    verdicts = {m: e.get("verdict") for m, e in entries.items()}
    gains = {m: e.get("gain") for m, e in entries.items()}
    gain_p = {m: e.get("gain_p") for m, e in entries.items()}
    # The ledger carries the batch decision as a boolean. Re-deriving it from the
    # verdict headline loses every case where a batch-confounded modality also
    # clears the gain test and the headline reports the gain instead.
    batch_called = {m: bool(e.get("batch_confounded")) for m, e in entries.items()}
    batch_structured = {m: bool(e.get("batch_structured")) for m, e in entries.items()}

    roles_called = {m: verdict_to_role(v) for m, v in verdicts.items()}
    unmapped = sorted({v for m, v in verdicts.items() if roles_called.get(m) is None and v})
    correct = {m: (roles_called.get(m) == truth_roles.get(m)) for m in truth_roles}
    # The registered role-recovery endpoint is macro-F1 over ground-truth roles
    # represented in the cell.  An unmapped verdict remains an explicit error class;
    # it cannot disappear from scoring by being dropped from the denominator.
    truth = [truth_roles[m] for m in truth_roles]
    called = [roles_called.get(m) or "__unmapped__" for m in truth_roles]
    registered_roles = sorted(set(truth))
    role_macro_f1 = (float(f1_score(truth, called, labels=registered_roles,
                                    average="macro", zero_division=0))
                     if registered_roles else None)
    alarm = bool(led.get("leakage_warning"))
    grouping = audit.get("grouping") or {}
    return {
        "leakage_alarm": alarm,
        "leakage_present": bool(leakage_present),
        "leakage_correct": alarm == bool(leakage_present),
        "false_certification": (leakage_present and not alarm),
        "false_alarm": ((not leakage_present) and alarm),
        "batch_confounded_called": batch_called,
        "batch_structured_called": batch_structured,
        "missingness": missingness_flags(audit),
        "grouping_warning": bool(grouping.get("any_warning")),
        "grouping_warning_classes": {name: bool(grouping.get(name)) for name in
                                     (*(n for n, _ in GROUPING_WARNING_CLASSES),
                                      "other_warning")},
        "grouping_messages": grouping.get("messages") or [],
        "modality_verdicts": verdicts,
        "modality_roles_called": roles_called,
        "unmapped_verdicts": unmapped,
        "ground_truth_roles": truth_roles,
        "role_correct": correct,
        "role_accuracy": (sum(correct.values()) / len(correct)) if correct else None,
        "role_macro_f1": role_macro_f1,
        "marginal_gain": gains,
        "marginal_gain_p": gain_p,
        "controls": {c["name"]: c["primary"] for c in led.get("controls", [])},
        "fusion_gain_over_best_single": led.get("fusion_gain_over_best_single"),
        "best_single_modality": led.get("best_single_modality"),
        "cka": led.get("redundancy_matrix"),
    }
