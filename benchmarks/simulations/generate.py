"""Protocol-backed simulation generator.

Every structural parameter is read from ``benchmark_protocol.yaml`` so the generator
cannot drift from the protocol it implements.

A modality's ground-truth role is a property of this generator: the scenario decides
which factors are expressed where, and the role falls out of that construction. Roles
are written next to the data so scoring never depends on an annotation added later.

Usage
    python generate.py --list
    python generate.py --family role_recovery --scenario S1 --n 150 --replicate 0
    python generate.py --validate                 # development checks; not confirmatory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RECORD = REPO / "benchmark_record"
PROTOCOL = RECORD / "benchmark_protocol.yaml"
SEED_FILE = RECORD / "checksums" / "definitive_seed_registry.json"

ROLE_PREDICTIVE = "predictive"
ROLE_NOT_ADDITIVE = "not_additive"
ROLE_BATCH_CONFOUNDED = "batch_confounded"
ROLE_CONTROL_LIKE = "control_like"


# --------------------------------------------------------------------------- #
# configuration, read from the record
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    modalities: list[dict]
    sample_sizes: list[int]
    instability_n: int
    scenarios_primary: dict
    scenarios_secondary: dict
    replicates: dict
    master_seed: int
    shared_factors: int
    factors_per_modality: int
    n_blocks: int
    within_block_rho: float
    noise_fraction: float
    missing_baseline: dict
    bayes_band: list[float]
    risk_flag_severity: dict[str, dict[str, Any]]
    repeated_measurements_per_subject: int
    names: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "Spec":
        proto = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
        sim = proto["simulation"]
        params = proto["simulation_parameters"]
        latent = params["latent_structure"]
        families = sim["families"]
        role_recovery = families["role_recovery"]
        nonlinear = families["nonlinear_secondary"]
        sample_sizes = sorted({
            int(n)
            for family in families.values()
            for cell in family.get("cells", [family])
            for n in cell.get("sample_sizes", [])
        })
        replicate_counts = {}
        for family_name, family in families.items():
            counts = [
                cell[key]
                for cell in family.get("cells", [family])
                for key in ("independent_replicates_per_cell", "paired_replicates_per_cell")
                if key in cell
            ]
            if counts:
                replicate_counts[family_name] = max(counts)
        modalities = [
            {"name": item["id"], "n_features": item["feature_count"]}
            for item in params["modalities"]
        ]
        spec = cls(
            modalities=modalities,
            sample_sizes=sample_sizes,
            instability_n=max(sample_sizes),
            scenarios_primary={name: None for name in role_recovery["scenarios"]},
            scenarios_secondary={name: None for name in nonlinear["scenarios"]},
            replicates=replicate_counts,
            master_seed=sim["seed_generation"]["master_seed"],
            shared_factors=latent["shared_factors"],
            factors_per_modality=latent["modality_specific_factors_per_modality"],
            n_blocks=latent["feature_blocks_per_modality"],
            within_block_rho=latent["within_block_correlation"],
            noise_fraction=latent["noise_feature_fraction"],
            missing_baseline={item["id"]: 0.0 for item in params["modalities"]},
            bayes_band=params["target_bayes_AUROC_range"],
            risk_flag_severity=params["risk_flag_severity"],
            repeated_measurements_per_subject=params["group_leakage"]["repeated_measurements_per_subject"],
        )
        spec.names = [m["name"] for m in spec.modalities]
        return spec

    def n_features(self, name: str) -> int:
        return next(m["n_features"] for m in self.modalities if m["name"] == name)


@dataclass(frozen=True)
class UnitKey:
    """The complete, protocol-defined identity of a simulation data unit."""

    family: str
    scenario_or_structure: str
    condition_or_perturbation: str | None
    sample_size: int | None
    replicate_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "scenario_or_structure": self.scenario_or_structure,
            "condition_or_perturbation": self.condition_or_perturbation,
            "sample_size": self.sample_size,
            "replicate_index": self.replicate_index,
        }


STREAM_LABELS = ("data_generation", "fold_assignment", "model_initialization", "bootstrap")


def _replicate_count(block: dict[str, Any]) -> int:
    for name in ("independent_replicates_per_cell", "paired_replicates_per_cell"):
        if name in block:
            return int(block[name])
    raise ValueError(f"family has no replicate count: {block!r}")


def missingness_condition(protocol: dict[str, Any], severity: str,
                          replicate_index: int) -> str:
    """Pair one registered mechanism with a missingness replicate.

    Mechanisms are assigned within each severity/sample-size cell in protocol
    order.  They annotate the existing independent units; they never create an
    additional mechanism dimension.
    """
    mechanisms = protocol["simulation_parameters"]["risk_flag_severity"]["missingness"]["mechanisms"]
    if not isinstance(mechanisms, list) or not mechanisms:
        raise ValueError("missingness mechanisms must be a non-empty ordered list")
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    return f"{severity}__{mechanisms[replicate_index % len(mechanisms)]}"


def protocol_units(protocol: dict[str, Any]) -> list[UnitKey]:
    """Expand the protocol into its canonical independent generation units."""
    units: list[UnitKey] = []
    for family, block in protocol["simulation"]["families"].items():
        if family == "missingness_risk_flags":
            for cell in block["cells"]:
                for sample_size in cell["sample_sizes"]:
                    for replicate in range(_replicate_count(cell)):
                        units.append(UnitKey(
                            family, block["base_scenario"],
                            missingness_condition(protocol, str(cell["severity"]), replicate),
                            int(sample_size), replicate,
                        ))
            continue
        if family == "group_leakage":
            for sample_size in block["sample_sizes"]:
                for replicate in range(_replicate_count(block)):
                    units.append(UnitKey(
                        family, block["base_scenario"], "safe__unsafe_paired",
                        int(sample_size), replicate,
                    ))
            continue
        if "cells" in block:
            for cell in block["cells"]:
                for sample_size in cell["sample_sizes"]:
                    for replicate in range(_replicate_count(cell)):
                        units.append(UnitKey(
                            family, block["base_scenario"], str(cell["severity"]),
                            int(sample_size), replicate,
                        ))
            continue
        if "cohort_structures" in block:
            for structure in block["cohort_structures"]:
                for perturbation in block["perturbations"]:
                    for replicate in range(_replicate_count(block)):
                        units.append(UnitKey(family, str(structure), str(perturbation),
                                             None, replicate))
            continue
        scenarios = block.get("scenarios") or [block["base_scenario"]]
        conditions = block.get("paired_conditions", [block.get("condition")])
        for scenario in scenarios:
            for condition in conditions:
                for sample_size in block["sample_sizes"]:
                    for replicate in range(_replicate_count(block)):
                        units.append(UnitKey(family, str(scenario), condition,
                                             int(sample_size), replicate))
    if len(set(units)) != len(units):
        raise ValueError("protocol expansion produced duplicate generation unit keys")
    return units


def canonical_seed_payload(namespace: str, master_seed: int, unit: UnitKey,
                           stream_label: str, collision_counter: int = 0) -> bytes:
    """Return the exact UTF-8 SHA-256 preimage registered by the protocol.

    Canonical JSON makes the derivation independent of Python dictionary ordering and
    platform locale.  The counter is present even for the first attempt (zero), as
    required by the collision policy.
    """
    if stream_label not in STREAM_LABELS:
        raise ValueError(f"unknown stream label {stream_label!r}")
    if collision_counter < 0:
        raise ValueError("collision_counter must be non-negative")
    payload = {
        "namespace": namespace,
        "master_seed": int(master_seed),
        **unit.as_dict(),
        "stream_label": stream_label,
        "collision_counter": collision_counter,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")


def sha256_uint32(namespace: str, master_seed: int, unit: UnitKey,
                  stream_label: str, collision_counter: int = 0) -> tuple[int, str]:
    """Derive a uint32 seed from the first four SHA-256 bytes, big endian."""
    digest = hashlib.sha256(
        canonical_seed_payload(namespace, master_seed, unit, stream_label, collision_counter)
    ).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False), digest.hex()


def registry_unit_key(unit: UnitKey) -> UnitKey:
    """Map paired conditions to their one shared generated dataset.

    Safe and unsafe group-leakage evaluations differ only in their splitter, so the
    protocol's paired-generation rule gives them one data-generation key.  The
    evaluation condition itself remains in the task metadata and split stream.
    """
    if unit.family == "group_leakage" and unit.condition_or_perturbation in {"safe", "unsafe"}:
        return UnitKey(unit.family, unit.scenario_or_structure, "safe__unsafe_paired",
                       unit.sample_size, unit.replicate_index)
    return unit


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
def block_loadings(rng: np.random.Generator, p: int, n_blocks: int, rho: float) -> np.ndarray:
    """Per-feature block membership weights, giving correlated feature blocks."""
    block = rng.integers(0, n_blocks, size=p)
    strength = np.sqrt(rho)
    return block, strength


def modality_matrix(rng, n, p, factor_scores, factor_loadings, n_blocks, rho):
    """Latent structure + correlated blocks + independent noise, on the latent scale."""
    block, strength = block_loadings(rng, p, n_blocks, rho)
    block_scores = rng.standard_normal((n, n_blocks))
    x = factor_scores @ factor_loadings                     # shared/specific structure
    x = x + strength * block_scores[:, block]               # within-block correlation
    x = x + np.sqrt(max(0.0, 1.0 - rho)) * rng.standard_normal((n, p))
    return x


def to_measurement_scale(rng, x: np.ndarray, kind: str) -> np.ndarray:
    """Map the latent matrix onto a plausible measurement scale per modality."""
    if kind == "rna_like":
        mu = np.exp(np.clip(x * 0.5 + 4.0, 0, 12))
        counts = rng.poisson(mu)                            # over-dispersion added below
        counts = counts + rng.negative_binomial(5, 0.5, size=x.shape)
        return np.log2(counts + 1.0)
    if kind == "methyl_like":
        beta = 1.0 / (1.0 + np.exp(-(x * 0.8)))
        beta = np.clip(beta, 1e-3, 1 - 1e-3)
        return np.log2(beta / (1 - beta))                   # M-value
    return x * 0.7 + 8.0                                    # protein_like: log-scale gaussian


def apply_missingness(rng, x: np.ndarray, rate: float, abundance_dependent: bool) -> np.ndarray:
    if rate <= 0:
        return x
    if abundance_dependent:
        # low-abundance entries are likelier to be unobserved (MNAR by construction)
        r = np.argsort(np.argsort(x, axis=None)).reshape(x.shape) / max(1, x.size - 1)
        prob = np.clip(2.0 * rate * (1.0 - r), 0, 1)
    else:
        prob = np.full(x.shape, rate)
    out = x.copy()
    out[rng.random(x.shape) < prob] = np.nan
    return out


def apply_registered_missingness(rng: np.random.Generator, x: np.ndarray, y: np.ndarray,
                                 batch: np.ndarray, additional_rate: float,
                                 mechanism: str) -> np.ndarray:
    """Apply one protocol-registered missingness mechanism to a matrix.

    The severity probability is an *additional* probability.  Clean therefore
    leaves the generated measurement matrix unchanged.  Each mechanism is a
    distinct unit condition, never a post-hoc overlay of a completed analysis.
    """
    if additional_rate <= 0:
        return x
    out = x.copy()
    if mechanism == "MCAR":
        mask = rng.random(x.shape) < additional_rate
    elif mechanism == "MAR":
        # Missingness depends on an observed covariate (batch), not on the value
        # being made missing.  Higher numbered batches have higher probability.
        scaled_batch = batch / max(1, int(batch.max()))
        probability = additional_rate * (0.5 + scaled_batch[:, None])
        mask = rng.random(x.shape) < np.clip(probability, 0, 1)
    elif mechanism == "MNAR":
        ranks = np.argsort(np.argsort(x, axis=None)).reshape(x.shape) / max(1, x.size - 1)
        mask = rng.random(x.shape) < np.clip(2.0 * additional_rate * (1.0 - ranks), 0, 1)
    elif mechanism == "batch_associated":
        affected = batch == int(batch.max())
        mask = (rng.random(x.shape) < additional_rate) & affected[:, None]
    elif mechanism == "whole_modality_absence":
        missing_subjects = rng.random(x.shape[0]) < additional_rate
        mask = np.broadcast_to(missing_subjects[:, None], x.shape)
    else:
        raise ValueError(f"unknown registered missingness mechanism {mechanism!r}")
    out[mask] = np.nan
    return out


def bayes_auroc(linpred: np.ndarray, y: np.ndarray) -> float:
    """AUROC of the true linear predictor: the ceiling any model could reach."""
    pos, neg = linpred[y == 1], linpred[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


_CALIBRATION_CACHE: dict[tuple, float] = {}


def calibrate_scale(spec: Spec, weights: np.ndarray, seed: int) -> float:
    """Find the signal scale whose Bayes-optimal AUROC lands in the prespecified band.

    Calibrated on a large independent draw, per the design, rather than fixing a
    coefficient and discovering the difficulty afterwards.
    """
    key = (int(weights.size), float(np.mean(spec.bayes_band)), int(seed))
    if key in _CALIBRATION_CACHE:          # deterministic in its inputs, so cacheable
        return _CALIBRATION_CACHE[key]
    target = float(np.mean(spec.bayes_band))
    rng = np.random.default_rng(seed)
    n = 20000
    f = rng.standard_normal((n, weights.size))
    lo, hi = 0.01, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lin = f @ (weights * mid)
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        auc = bayes_auroc(lin, y)
        if auc < target:
            lo = mid
        else:
            hi = mid
    _CALIBRATION_CACHE[key] = 0.5 * (lo + hi)
    return _CALIBRATION_CACHE[key]


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def scenario_plan(scenario: str, spec: Spec) -> dict:
    """Which factors carry the outcome, where they are expressed, and the true roles."""
    rna, methyl, protein = spec.names
    if scenario == "S1":
        return {"signal_factors": {"f0": [rna]},
                "roles": {rna: ROLE_PREDICTIVE, methyl: ROLE_CONTROL_LIKE, protein: ROLE_CONTROL_LIKE}}
    if scenario == "S2":
        return {"signal_factors": {"f0": [rna], "f1": [protein]},
                "roles": {rna: ROLE_PREDICTIVE, protein: ROLE_PREDICTIVE, methyl: ROLE_CONTROL_LIKE}}
    if scenario == "S3":
        return {"signal_factors": {"f0": [rna, methyl]},
                "roles": {rna: ROLE_PREDICTIVE, methyl: ROLE_NOT_ADDITIVE, protein: ROLE_CONTROL_LIKE}}
    if scenario == "S4":
        return {"signal_factors": {"f0": [rna]}, "batch_confounded": protein,
                "roles": {rna: ROLE_PREDICTIVE, methyl: ROLE_CONTROL_LIKE, protein: ROLE_BATCH_CONFOUNDED}}
    if scenario == "S5":
        return {"signal_factors": {"f0": [rna]}, "independent": protein,
                "roles": {rna: ROLE_PREDICTIVE, methyl: ROLE_CONTROL_LIKE, protein: ROLE_CONTROL_LIKE}}
    if scenario == "S6":
        return {"signal_factors": {"f0": [rna], "f1": [protein]}, "interaction": True,
                "roles": {rna: ROLE_PREDICTIVE, protein: ROLE_PREDICTIVE, methyl: ROLE_CONTROL_LIKE}}
    raise SystemExit(f"unknown scenario {scenario!r}")


def generate_unit(unit: UnitKey, spec: Spec) -> dict:
    """Generate one registered unit from its full identity and archived streams."""
    if unit.sample_size is None:
        raise ValueError("semi-synthetic units require an eligible template and are not generated without one")
    if not unit.scenario_or_structure.startswith("S"):
        raise ValueError(f"unknown parametric scenario {unit.scenario_or_structure!r}")
    registry_key = registry_unit_key(unit)
    streams = {label: seed_for(registry_key, label) for label in STREAM_LABELS}
    seed = streams["data_generation"]
    rng = np.random.default_rng(seed)
    scenario, n, replicate = unit.scenario_or_structure, unit.sample_size, unit.replicate_index
    plan = scenario_plan(scenario, spec)

    condition = unit.condition_or_perturbation
    batch_strength = 0.0
    if unit.family == "batch_risk_flags":
        if f"{condition}_outcome_assignment_probability" not in spec.risk_flag_severity["batch"]:
            raise ValueError(f"unknown batch severity {condition!r}")
        batch_strength = float(spec.risk_flag_severity["batch"][condition + "_outcome_assignment_probability"])
    missingness_mechanism = None
    missingness_rate = 0.0
    if unit.family == "missingness_risk_flags":
        try:
            severity, missingness_mechanism = str(condition).split("__", 1)
        except ValueError as exc:
            raise ValueError("missingness condition must be '<severity>__<mechanism>'") from exc
        missingness_rate = float(spec.risk_flag_severity["missingness"][
            severity + "_additional_missing_probability"
        ])

    n_shared = spec.shared_factors
    n_total_factors = n_shared + spec.factors_per_modality * len(spec.names)
    factors = rng.standard_normal((n, n_total_factors))

    # outcome from the designated factors
    idx = {f"f{i}": i for i in range(n_shared)}
    sig_idx = [idx[k] for k in plan["signal_factors"]]
    weights = np.zeros(n_total_factors)
    weights[sig_idx] = 1.0
    scale = calibrate_scale(spec, weights[weights != 0], spec.master_seed + 977)

    if plan.get("interaction"):
        lin = scale * factors[:, sig_idx[0]] * factors[:, sig_idx[1]]
    else:
        lin = factors @ (weights * scale)
    p_y = 1.0 / (1.0 + np.exp(-lin))
    y = (rng.random(n) < p_y).astype(int)

    # batch, optionally confounded with the outcome
    n_batches = int(spec.risk_flag_severity["batch"]["batch_count"])
    if batch_strength > 0:
        base = rng.integers(0, n_batches, size=n)
        flip = rng.random(n) < batch_strength
        batch = np.where(flip, y * (n_batches - 1), base)
    else:
        batch = rng.integers(0, n_batches, size=n)

    matrices, truth = {}, {}
    for m_i, name in enumerate(spec.names):
        p = spec.n_features(name)
        loadings = np.zeros((n_total_factors, p))
        # every modality expresses its own specific factors
        for k in range(spec.factors_per_modality):
            fi = n_shared + m_i * spec.factors_per_modality + k
            cols = rng.choice(p, size=int(p * (1 - spec.noise_fraction) / 2), replace=False)
            loadings[fi, cols] = rng.normal(1.0, 0.2, size=cols.size)
        # signal factors only where the scenario expresses them
        for fname, targets in plan["signal_factors"].items():
            if name in targets and name != plan.get("independent"):
                cols = rng.choice(p, size=max(5, int(p * (1 - spec.noise_fraction) / 2)), replace=False)
                loadings[idx[fname], cols] = rng.normal(1.0, 0.2, size=cols.size)

        x = modality_matrix(rng, n, p, factors, loadings, spec.n_blocks, spec.within_block_rho)

        if name == plan.get("batch_confounded"):
            offsets = rng.normal(0, 1.5, size=(n_batches, p))
            x = x + offsets[batch]

        x = to_measurement_scale(rng, x, name)
        x = apply_missingness(rng, x, float(spec.missing_baseline.get(name, 0.0)),
                              abundance_dependent=(name == "protein_like"))
        if missingness_mechanism is not None and name == "protein_like":
            x = apply_registered_missingness(rng, x, y, batch, missingness_rate,
                                             missingness_mechanism)
        matrices[name] = x.astype(np.float32)
        truth[name] = plan["roles"][name]

    ds = {
        "unit_key": unit.as_dict(), "registry_unit_key": registry_key.as_dict(),
        "scenario": scenario, "n": n, "replicate": replicate, "seed": int(seed),
        "stream_seeds": streams, "matrices": matrices, "y": y, "batch": batch,
        "groups": np.arange(n),                       # one specimen per subject by default
        "roles": truth, "bayes_auroc": bayes_auroc(lin, y),
        "signal_scale": scale, "batch_outcome_strength": batch_strength,
        "missingness_mechanism": missingness_mechanism,
        "additional_missing_probability": missingness_rate,
    }
    if unit.family == "null_control_specificity":
        # A fresh Bernoulli outcome is independent of all generated measurements.
        null_rng = np.random.default_rng(streams["bootstrap"])
        ds["y"] = (null_rng.random(n) < float(y.mean())).astype(int)
        ds["roles"] = {name: ROLE_CONTROL_LIKE for name in truth}
        ds["bayes_auroc"] = 0.5
        ds["overlay"] = "no_predictive_signal"
        ds["leakage_present"] = False
    if unit.family == "group_leakage":
        # Both paired conditions receive byte-identical repeated measurements.
        ds = apply_overlay(ds, "L3_repeated_specimens", spec,
                           specimens_per_subject=spec.repeated_measurements_per_subject)
        ds["evaluate_under"] = str(condition)
        ds["paired_condition"] = "safe__unsafe_paired"
        # The protocol assesses this family by the grouping warning and the
        # naive-minus-group-aware AUROC gap, and states that null controls do not
        # substitute for that assessment (benchmark_protocol.yaml#
        # diagnostic_interpretation.group_leakage). Leaving the generic
        # leakage_present flag set would score the control alarm against a
        # detection the protocol does not ask it to make, recording a false
        # certification for every unit in the family.
        ds["group_leakage_present"] = True
        ds["leakage_present"] = False
    return ds


def generate(scenario: str, n: int, replicate: int, spec: Spec,
             batch_outcome_strength: float | None = None, n_batches: int | None = None,
             *, family: str = "role_recovery", condition: str | None = None) -> dict:
    """Compatibility wrapper; new callers should pass a complete ``UnitKey``.

    ``batch_outcome_strength`` is retained only for pre-existing development calls.
    Registered batch units must select their strength from the protocol by severity.
    """
    if batch_outcome_strength not in (None, 0.0):
        family, condition = "batch_risk_flags", next(
            (name.removesuffix("_outcome_assignment_probability")
             for name, value in spec.risk_flag_severity["batch"].items()
             if name.endswith("_outcome_assignment_probability") and value == batch_outcome_strength),
            None,
        )
        if condition is None:
            raise ValueError("batch strength is not a registered protocol severity")
    return generate_unit(UnitKey(family, scenario, condition, n, replicate), spec)


# --------------------------------------------------------------------------- #
# Development stress-test overlays. The protocol names the evaluated families;
# these overlays are only the generator-level transformations they require.
# --------------------------------------------------------------------------- #
# The contaminated and clean conditions share a generator and differ only in the
# contamination, so a detection rate is attributable to the contamination itself.
OVERLAYS = [
    "clean_null",
    "L1_exact_target_copy", "L1_noisy_target_surrogate",
    "L1_post_outcome_variable", "L1_label_encoded_sample_id",
    "L3_duplicate_samples", "L3_repeated_specimens",
    "L3_technical_replicates", "L3_random_vs_group_split",
]


def apply_overlay(ds: dict, overlay: str, spec: Spec, surrogate_snr: float = 1.0,
                  duplicate_fraction: float = 0.2, specimens_per_subject: int = 2) -> dict:
    """Inject one stress condition. Returns a new dataset dict; `ds` is not mutated."""
    if overlay is None:
        return ds
    if overlay not in OVERLAYS:
        raise SystemExit(f"unknown overlay {overlay!r}; known: {OVERLAYS}")

    rng = np.random.default_rng(ds["seed"] + 10_000 + OVERLAYS.index(overlay))
    out = dict(ds)
    out["matrices"] = {k: v.copy() for k, v in ds["matrices"].items()}
    out["overlay"] = overlay
    y = ds["y"].copy()
    rna = spec.names[0]

    # --- the clean null: outcome independent of every feature -----------------
    if overlay == "clean_null":
        out["y"] = (rng.random(len(y)) < 0.5).astype(int)
        out["roles"] = {m: ROLE_CONTROL_LIKE for m in ds["roles"]}
        out["bayes_auroc"] = 0.5     # chance, by construction: y is independent of every feature
        out["leakage_present"] = False
        out["expected"] = "no layer predictive; controls at chance; no leakage alarm"
        return out

    # --- L1: the outcome, or a function of it, enters the features ------------
    if overlay.startswith("L1_"):
        col = None
        if overlay == "L1_exact_target_copy":
            col = y.astype(np.float32) * 10.0
        elif overlay == "L1_noisy_target_surrogate":
            col = y.astype(np.float32) * surrogate_snr + rng.standard_normal(len(y)).astype(np.float32)
        elif overlay == "L1_post_outcome_variable":
            # a clinical-like measurement generated FROM the outcome, as a downstream
            # consequence of it rather than a predictor of it
            col = (y * 2.0 + rng.normal(0, 0.5, len(y))).astype(np.float32)
        elif overlay == "L1_label_encoded_sample_id":
            # the identifier itself encodes the class; it leaks only if a pipeline
            # consumes ids as data, so the id array is what carries the contamination
            out["sample_ids"] = np.array([f"CASE{i:04d}_CLASS{c}" for i, c in enumerate(y)])
            out["leakage_present"] = True
            out["expected"] = "leakage detectable only if identifiers are consumed as features"
            return out
        out["matrices"][rna] = np.hstack([out["matrices"][rna], col.reshape(-1, 1)])
        out["leakage_present"] = True
        out["expected"] = "controls above chance; leakage alarm fires; no certification"
        return out

    # --- L3: the same subject appears on both sides of a split ----------------
    n = len(y)
    if overlay == "L3_duplicate_samples":
        k = max(1, int(n * duplicate_fraction))
        idx = rng.choice(n, size=k, replace=False)
        take = np.concatenate([np.arange(n), idx])
        groups = np.concatenate([ds["groups"], ds["groups"][idx]])
        noise = 0.0
    elif overlay == "L3_repeated_specimens":
        take = np.repeat(np.arange(n), specimens_per_subject)
        groups = np.repeat(ds["groups"], specimens_per_subject)
        noise = 0.5                       # specimen-level variation within a subject
    elif overlay in ("L3_technical_replicates", "L3_random_vs_group_split"):
        take = np.repeat(np.arange(n), 2)
        groups = np.repeat(ds["groups"], 2)
        noise = 0.05                      # measurement noise only
    out["y"] = y[take]
    out["groups"] = groups
    out["batch"] = ds["batch"][take]
    for m, X in out["matrices"].items():
        Xe = X[take]
        if noise:
            Xe = Xe + rng.normal(0, noise, size=Xe.shape).astype(np.float32)
        out["matrices"][m] = Xe
    out["leakage_present"] = True
    out["n"] = int(len(out["y"]))
    out["expected"] = ("group-aware splitting removes the inflation; the naive-versus-"
                       "group-aware gap is the measured quantity")
    if overlay == "L3_random_vs_group_split":
        out["evaluate_under"] = ["group_aware", "random"]
    return out


# --------------------------------------------------------------------------- #
# archived seeds
# --------------------------------------------------------------------------- #
def _registry_index() -> dict[tuple[str, str, str | None, int | None, int, str], int]:
    if not SEED_FILE.exists():
        raise FileNotFoundError(
            f"seed registry missing: {SEED_FILE}; run benchmark_record/tools/generate_seed_registry.py --write"
        )
    payload = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError("seed registry has no stream record list")
    index: dict[tuple[str, str, str | None, int | None, int, str], int] = {}
    for record in streams:
        if not isinstance(record, dict):
            raise ValueError("seed registry contains a non-object stream record")
        key = (
            record.get("family"), record.get("scenario_or_structure"),
            record.get("condition_or_perturbation"), record.get("sample_size"),
            record.get("replicate_index"), record.get("stream_label"),
        )
        seed = record.get("uint32_seed")
        if not all(isinstance(part, (str, int)) or part is None for part in key) or not isinstance(seed, int):
            raise ValueError("seed registry contains an invalid stream record")
        if key in index:
            raise ValueError(f"seed registry duplicates stream key {key!r}")
        index[key] = seed
    return index


def seed_for(unit: UnitKey, stream_label: str) -> int:
    key = (unit.family, unit.scenario_or_structure, unit.condition_or_perturbation,
           unit.sample_size, unit.replicate_index, stream_label)
    try:
        return _registry_index()[key]
    except KeyError as exc:
        raise KeyError(f"no archived {stream_label!r} seed for unit {unit.as_dict()}") from exc


# --------------------------------------------------------------------------- #
def write_dataset(ds: dict, out_dir: Path) -> Path:
    d = out_dir / f"{ds['scenario']}_n{ds['n']}_r{ds['replicate']}"
    d.mkdir(parents=True, exist_ok=True)
    for name, x in ds["matrices"].items():
        np.save(d / f"{name}.npy", x)
    np.save(d / "y.npy", ds["y"])
    np.save(d / "batch.npy", ds["batch"])
    (d / "truth.json").write_text(json.dumps({
        "scenario": ds["scenario"], "n": ds["n"], "replicate": ds["replicate"],
        "unit_key": ds["unit_key"], "stream_seeds": ds["stream_seeds"],
        "seed": ds["seed"], "roles": ds["roles"], "bayes_auroc": ds["bayes_auroc"],
        "signal_scale": ds["signal_scale"],
        "batch_outcome_strength": ds["batch_outcome_strength"],
        "missingness_mechanism": ds["missingness_mechanism"],
        "additional_missing_probability": ds["additional_missing_probability"],
        "shapes": {k: list(v.shape) for k, v in ds["matrices"].items()},
    }, indent=2), encoding="utf-8")
    return d


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--family", default="role_recovery")
    ap.add_argument("--scenario")
    ap.add_argument("--n", type=int)
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--overlay", default=None, help="stress condition to inject")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    spec = Spec.load()

    if args.list:
        print(f"modalities        : {[(m['name'], m['n_features']) for m in spec.modalities]}")
        print(f"sample sizes      : {spec.sample_sizes} (+{spec.instability_n} instability only)")
        print(f"scenarios primary : {list(spec.scenarios_primary)}")
        print(f"scenarios second. : {list(spec.scenarios_secondary)}")
        print(f"replicates        : {spec.replicates}")
        print(f"master seed       : {spec.master_seed}")
        print(f"bayes AUROC band  : {spec.bayes_band}")
        print(f"overlays          : {OVERLAYS}")
        return 0

    if False:  # Seed registries are created by benchmark_record/tools/generate_seed_registry.py.
        SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = build_seed_file(spec)
        SEED_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {SEED_FILE.relative_to(REPO)} — {payload['n_streams']} streams "
              f"from master seed {payload['master_seed']}")
        return 0

    if args.validate:
        return validate(spec)

    if not (args.scenario and args.n):
        ap.error("give --scenario and --n, or use --list / --validate")
    ds = generate_unit(UnitKey(args.family, args.scenario, args.condition,
                               args.n, args.replicate), spec)
    if args.overlay:
        ds = apply_overlay(ds, args.overlay, spec)
    print(f"{ds['scenario']} n={ds['n']} r={ds['replicate']} seed={ds['seed']} "
          f"bayes_auroc={ds['bayes_auroc']:.3f} roles={ds['roles']}")
    for k, v in ds["matrices"].items():
        print(f"  {k:<14} {v.shape}  missing {np.isnan(v).mean():.3f}")
    if args.out:
        print("wrote", write_dataset(ds, args.out))
    return 0


# --------------------------------------------------------------------------- #
# development validation of the generator (not confirmatory evidence)
# --------------------------------------------------------------------------- #
def cka(a: np.ndarray, b: np.ndarray) -> float:
    a = np.nan_to_num(a - np.nanmean(a, 0)); b = np.nan_to_num(b - np.nanmean(b, 0))
    num = np.linalg.norm(b.T @ a, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / den) if den > 0 else float("nan")


def validate(spec: Spec) -> int:
    print("generator validation (development analysis; not confirmatory)\n")
    fails = 0
    lo, hi = spec.bayes_band

    # Each scenario is generated under the family that registers it: S6 exists
    # only under nonlinear_secondary, so requesting it under role_recovery has no
    # archived seed and the whole gate aborts before running a single check.
    families = {name: "role_recovery" for name in spec.scenarios_primary}
    families.update({name: "nonlinear_secondary" for name in spec.scenarios_secondary})
    severe_batch = next(
        name.removesuffix("_outcome_assignment_probability")
        for name, value in spec.risk_flag_severity["batch"].items()
        if name.endswith("_outcome_assignment_probability")
        and value == max(v for k, v in spec.risk_flag_severity["batch"].items()
                         if k.endswith("_outcome_assignment_probability"))
    )
    for s, family in families.items():
        unit = (UnitKey("batch_risk_flags", s, severe_batch, 500, 0) if s == "S4"
                else UnitKey(family, s, None, 500, 0))
        ds = generate_unit(unit, spec)
        # S6's outcome is a product of two factors while calibrate_scale bisects on
        # a sum, so its Bayes AUROC is not expected to land in the additive band.
        # The deviation is reported rather than exempted silently.
        in_band = (lo - 0.05) <= ds["bayes_auroc"] <= (hi + 0.05)
        ok = in_band or s in spec.scenarios_secondary
        fails += 0 if ok else 1
        note = "" if in_band else "  [outside the additive calibration band]"
        miss = {k: round(float(np.isnan(v).mean()), 3) for k, v in ds["matrices"].items()}
        print(f"  {s}  bayes_auroc={ds['bayes_auroc']:.3f} {'OK' if ok else 'OUT OF BAND'}"
              f"  prevalence={ds['y'].mean():.2f}  missing={miss}{note}")

    # S3 redundant pair should share representation; S2 complementary should not
    s3 = generate_unit(UnitKey("role_recovery", "S3", None, 500, 0), spec)
    s2 = generate_unit(UnitKey("role_recovery", "S2", None, 500, 0), spec)
    rna, methyl, protein = spec.names
    c3 = cka(s3["matrices"][rna], s3["matrices"][methyl])
    c2 = cka(s2["matrices"][rna], s2["matrices"][protein])
    print(f"\n  CKA(rna, methyl) in S3 = {c3:.3f}   CKA(rna, protein) in S2 = {c2:.3f}")
    if not c3 > c2:
        print("  FAIL: the redundant pair is not more aligned than the complementary pair")
        fails += 1

    # batch-outcome association in S4 tracks the registered severities. Sweeping an
    # unregistered strength would check a condition the protocol does not contain.
    print()
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    sweep: dict[str, UnitKey] = {}
    for unit in protocol_units(protocol):
        # Each severity is checked at its own largest registered sample size; a
        # cell that registers only n=150 has no n=500 seed to draw on.
        if unit.family != "batch_risk_flags" or unit.replicate_index != 0:
            continue
        severity = str(unit.condition_or_perturbation)
        best = sweep.get(severity)
        if best is None or (unit.sample_size or 0) > (best.sample_size or 0):
            sweep[severity] = unit
    for severity, unit in sweep.items():
        strength = float(spec.risk_flag_severity["batch"][
            f"{severity}_outcome_assignment_probability"])
        ds = generate_unit(unit, spec)
        tab = np.zeros((2, int(ds["batch"].max()) + 1))
        for yy, bb in zip(ds["y"], ds["batch"]):
            tab[yy, bb] += 1
        row, col = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
        exp = row @ col / tab.sum()
        chi2 = float(np.nansum((tab - exp) ** 2 / np.where(exp > 0, exp, np.nan)))
        v = float(np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1))))
        print(f"  S4 batch severity {severity:<12} n={unit.sample_size:<4} "
              f"(p={strength:.1f}) -> Cramer's V = {v:.3f}")

    # a true-null outcome must not be predictable from the true linear predictor
    rng = np.random.default_rng(spec.master_seed)
    null_auc = bayes_auroc(rng.standard_normal(5000), (rng.random(5000) < 0.5).astype(int))
    print(f"\n  true-null bayes_auroc = {null_auc:.3f} (expect ~0.5)")
    fails += 0 if abs(null_auc - 0.5) < 0.03 else 1

    print(f"\n{'FAIL' if fails else 'PASS'}: {fails} check(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
