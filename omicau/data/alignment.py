"""Flexible multi-omic ingestion, alignment, and cryptographic provenance.

The ingestion layer adapts to non-standardized matrices: it auto-detects the
column delimiter, decides matrix orientation by overlap scoring (transposing a
genes-as-rows matrix automatically), fuzzily normalizes sample names to strip
batch prefixes and aliquot suffixes, coerces text-corrupted numbers, and marks
missing entries as ``NaN`` (true masks -- never imputed at ingest). After
alignment it computes an immutable SHA-256 signature of the sample index and
per-modality feature footprints to lock provenance for the study lifetime.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Tokens frequently used to mark missing values across heterogeneous exports.
NA_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", ".", "-", "?", "#n/a", "#na",
    "missing", "nd", "n.d.", "inf", "-inf", "+inf", "infinity",
}

_CANDIDATE_DELIMS = ["\t", ",", ";", "|"]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class ModalityMatrix:
    """A single aligned modality: samples (rows) x features (columns)."""

    name: str
    frame: pd.DataFrame  # index = sample ids, float64, NaN for missing
    description: str = ""

    @property
    def X(self) -> np.ndarray:
        """Feature values as a contiguous float64 array (NaN where missing)."""
        return np.ascontiguousarray(self.frame.to_numpy(dtype=np.float64))

    @property
    def mask(self) -> np.ndarray:
        """Observed mask (1.0 observed, 0.0 missing) aligned to :pyattr:`X`."""
        return (~np.isnan(self.X)).astype(np.float64)

    @property
    def feature_names(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.shape


@dataclass
class AlignedDataset:
    """The frozen, aligned multi-omic study state consumed by every stage."""

    modalities: dict[str, ModalityMatrix]
    y: pd.Series  # numeric-encoded target (survival: the time-to-event), index = sample ids
    task: str  # "classification" | "regression" | "survival"
    sample_ids: list[str]
    y_raw: pd.Series | None = None
    class_names: list[str] | None = None
    groups: pd.Series | None = None
    batch: pd.Series | None = None
    event: pd.Series | None = None      # survival only: 1 = event, 0 = right-censored
    time_unit: str = ""                 # survival only: free-text label for the report
    provenance_hash: str = ""
    report: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #
    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    @property
    def modality_names(self) -> list[str]:
        return list(self.modalities.keys())

    def feature_counts(self) -> dict[str, int]:
        return {name: m.shape[1] for name, m in self.modalities.items()}

    def modality_arrays(self, names: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Return ``{name: {"X", "mask", "features"}}`` for the given modalities."""
        names = names or self.modality_names
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            m = self.modalities[name]
            out[name] = {"X": m.X, "mask": m.mask, "features": m.feature_names}
        return out

    def concat_matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Early-fusion concatenation (NaN preserved for fold-internal handling)."""
        names = names or self.modality_names
        blocks, feats = [], []
        for name in names:
            m = self.modalities[name]
            blocks.append(m.X)
            feats.extend(f"{name}::{f}" for f in m.feature_names)
        if not blocks:
            return np.empty((self.n_samples, 0)), []
        return np.concatenate(blocks, axis=1), feats


# --------------------------------------------------------------------------- #
# Robust reading and sanitization
# --------------------------------------------------------------------------- #
def _detect_delimiter(sample_text: str) -> str:
    """Infer the column delimiter from a text sample (self-repairing)."""
    lines = [ln for ln in sample_text.splitlines() if ln.strip()][:20]
    if not lines:
        return ","
    # Prefer csv.Sniffer, but fall back to frequency counting on failure.
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters="".join(_CANDIDATE_DELIMS))
        if dialect.delimiter in _CANDIDATE_DELIMS:
            return dialect.delimiter
    except (csv.Error, Exception):  # noqa: BLE001 - sniffing is best-effort
        pass
    header = lines[0]
    counts = {d: header.count(d) for d in _CANDIDATE_DELIMS}
    best = max(counts, key=counts.get)
    if counts[best] > 0:
        return best
    # No standard delimiter in the header -> whitespace-separated columns.
    if re.search(r"\S\s+\S", header):
        return r"\s+"
    return ","


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce a text column to float64, healing common numeric corruptions."""
    text = s.astype("string").str.strip()
    lowered = text.str.lower()
    text = text.mask(lowered.isin(NA_TOKENS))

    direct = pd.to_numeric(text, errors="coerce")
    n_ok = int(direct.notna().sum())

    # European decimals / thousands separators: try a comma->dot repair and keep
    # whichever interpretation recovers more finite numbers.
    if text.notna().any():
        repaired_txt = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        repaired = pd.to_numeric(repaired_txt, errors="coerce")
        if int(repaired.notna().sum()) > n_ok:
            direct = repaired

    out = direct.astype("float64")  # nullable -> numpy float64 (pd.NA -> np.nan)
    # Non-finite guards: treat +/-inf as missing to protect variance tracking.
    return out.replace([np.inf, -np.inf], np.nan)


def read_matrix(
    source: str | Path | pd.DataFrame,
    *,
    orientation: str = "auto",
    id_regex: str | None = None,
) -> pd.DataFrame:
    """Read one modality matrix into a numeric ``samples? x features?`` frame.

    Orientation is resolved later against the reference sample ids; this returns
    the raw parse with the first column used as the row index. Passing a
    ``DataFrame`` (in-memory ingestion) sanitizes it in place.
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
        if raw.index.name is None and raw.columns.size and raw.index.equals(pd.RangeIndex(len(raw))):
            # Treat a positional index frame as already row-indexed by its index.
            pass
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        delim = _detect_delimiter(text[:8192])
        raw = pd.read_csv(
            io.StringIO(text),
            sep=delim,
            engine="python",
            dtype="string",
            index_col=0,
            header=0,
            skipinitialspace=True,
        )

    # Clean row/column labels defensively.
    raw.index = pd.Index([str(i).strip() for i in raw.index], name=raw.index.name)
    raw.columns = pd.Index([str(c).strip() for c in raw.columns])

    # Drop unnamed / fully empty structural artifacts.
    raw = raw.loc[~raw.index.isin(["", "nan", "None"])]
    raw = raw.loc[:, ~raw.columns.str.match(r"^(Unnamed.*)?$", na=False) | (raw.notna().any())]

    # Coerce to numeric in one flattened vectorized pass (fast on wide matrices);
    # only columns that lose a real value fall back to the per-column
    # European-decimal / thousands-separator repair.
    shape = raw.shape
    flat = pd.Series(raw.to_numpy().ravel()).astype("string").str.strip()
    flat = flat.mask(flat.str.lower().isin(NA_TOKENS))
    text_present = flat.notna().to_numpy().reshape(shape)
    num_arr = np.array(pd.to_numeric(flat, errors="coerce").astype("float64").to_numpy(),
                       dtype=np.float64, copy=True).reshape(shape)
    num_arr[~np.isfinite(num_arr)] = np.nan
    numeric = pd.DataFrame(num_arr, index=raw.index, columns=raw.columns)

    # Columns where stripped text was present but did not parse -> repair.
    needs_repair = np.where((text_present & np.isnan(num_arr)).any(axis=0))[0]
    for c in needs_repair:
        numeric.iloc[:, c] = _coerce_numeric_series(raw.iloc[:, c]).to_numpy()

    # Drop all-NaN rows/cols produced by structural noise.
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if id_regex:
        numeric.index = _apply_id_regex(numeric.index, id_regex)

    if orientation == "samples_as_cols":
        numeric = numeric.T
    return numeric


def _apply_id_regex(labels: pd.Index, pattern: str) -> pd.Index:
    rx = re.compile(pattern)
    out = []
    for lab in labels:
        m = rx.search(str(lab))
        out.append(m.group(1) if (m and m.groups()) else str(lab))
    return pd.Index(out)


# --------------------------------------------------------------------------- #
# Sample-name normalization + orientation
# --------------------------------------------------------------------------- #
def normalize_names(labels, spec) -> list[str]:
    """Apply fuzzy normalization (whitespace, case, prefix/suffix stripping)."""
    prefixes = [re.compile(p) for p in getattr(spec, "strip_prefix_regex", [])]
    suffixes = [re.compile(p) for p in getattr(spec, "strip_suffix_regex", [])]
    out = []
    for lab in labels:
        s = str(lab)
        if getattr(spec, "strip_whitespace", True):
            s = s.strip()
        for rx in prefixes:
            s = rx.sub("", s, count=1)
        for rx in suffixes:
            s = rx.sub("", s, count=1)
        if getattr(spec, "uppercase", False):
            s = s.upper()
        out.append(s.strip())
    return out


def _overlap_score(labels: list[str], reference: set[str]) -> float:
    if not labels:
        return 0.0
    return len(set(labels) & reference) / max(1, len(set(labels)))


def resolve_orientation(
    frame: pd.DataFrame, reference: set[str], norm_spec, forced: str = "auto"
) -> tuple[pd.DataFrame, str]:
    """Transpose ``frame`` so samples are rows, by overlap with reference ids."""
    if forced == "samples_as_rows":
        return frame, "kept (forced)"
    if forced == "samples_as_cols":
        return frame.T, "transposed (forced)"

    row_score = _overlap_score(normalize_names(frame.index, norm_spec), reference)
    col_score = _overlap_score(normalize_names(frame.columns, norm_spec), reference)
    if col_score > row_score:
        return frame.T, f"transposed (row={row_score:.2f} < col={col_score:.2f})"
    return frame, f"kept (row={row_score:.2f} >= col={col_score:.2f})"


def _collapse_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Average rows that share a normalized sample id (aliquot collapse)."""
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0, sort=False).mean()
    return frame


# --------------------------------------------------------------------------- #
# Target encoding and task inference
# --------------------------------------------------------------------------- #
def _infer_task(y: pd.Series, declared: str) -> str:
    if declared in {"classification", "regression", "survival"}:
        return declared
    numeric = pd.to_numeric(y, errors="coerce")
    n_unique = int(y.nunique(dropna=True))
    if numeric.notna().all() and n_unique > max(12, int(0.2 * len(y))):
        return "regression"
    return "classification"


def _encode_target(
    y_raw: pd.Series, task: str, positive_label: str | None
) -> tuple[pd.Series, list[str] | None]:
    if task == "regression":
        return pd.to_numeric(y_raw, errors="coerce").astype("float64"), None

    labels = y_raw.astype("string")
    classes = sorted([c for c in labels.dropna().unique()], key=str)
    if positive_label is not None and str(positive_label) in classes:
        # Ensure the positive class encodes to the largest code for binary tasks.
        classes = [c for c in classes if c != str(positive_label)] + [str(positive_label)]
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = labels.map(mapping).astype("float64")
    return encoded, classes


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _matrix_digest(frame: pd.DataFrame, sample_ids: list[str]) -> str:
    """Content SHA-256 of an aligned matrix (canonical row/column order)."""
    sub = frame.reindex(index=sample_ids)
    sub = sub.reindex(sorted(sub.columns), axis=1)
    arr = np.ascontiguousarray(sub.to_numpy(dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _resolve_group_series(clinical, group_spec, sample_ids):
    """Build the group Series from a str or list[str] spec.

    A list is combined column-wise into one categorical whose levels block on the
    coarsest shared unit (composite grouping for nested / repeated-measure designs).
    Missing parts are filled 'NA' before joining so a NaN in one column does not
    silently merge distinct units. Returns (series_or_None, info_or_None)."""
    if not group_spec:
        return None, None
    cols = [group_spec] if isinstance(group_spec, str) else list(group_spec)
    present = [c for c in cols if c in clinical.columns]
    missing = [c for c in cols if c not in clinical.columns]
    if not present:
        return None, {"requested": cols, "used": [], "missing": missing, "composite": len(cols) > 1}
    parts = [clinical[c].astype("string").fillna("NA") for c in present]
    g = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=" | ")
    g.index = pd.Index(sample_ids)
    if len(present) > 1:
        g = g.rename(" | ".join(present))
    return g, {"requested": cols, "used": present, "missing": missing,
               "composite": len(present) > 1, "n_units": int(g.nunique()),
               "label": " | ".join(present)}


def check_grouping(aligned) -> None:
    """Preflight on the grouping column, run before any cross-validation.

    The whole leakage guarantee rests on the user naming the ``group`` column
    correctly, so this makes the common mistakes loud instead of silent:
    * no group column -> warn that every row is treated as independent (scores
      can be inflated by pseudoreplication);
    * ~one group per sample -> warn that grouping is a no-op;
    * a class confined to a single group -> raise a plain message, since
      leakage-safe (group-aware) CV would hold that whole class out and the
      classifier would train on a single class (the raw sklearn error otherwise).
    """
    groups = aligned.groups
    n = aligned.n_samples
    if groups is None:
        warnings.warn(
            "No 'group' column set: cross-validation treats every row as an independent "
            "sample. If several rows share a subject (repeated tissues, timepoints, replicates, "
            "or a litter / cage / plot), set clinical.group -- otherwise scores may be "
            "optimistically inflated by pseudoreplication.", stacklevel=2)
        return
    g = np.asarray(groups)
    n_groups = len(np.unique(g))
    if n_groups >= n:
        warnings.warn(
            f"The 'group' column has about one level per sample ({n_groups} groups for {n} "
            "samples), so group-aware splitting does nothing. Point it at the shared unit "
            "(subject / litter / cage / plot), not a per-sample id.", stacklevel=2)
        return
    gi = (aligned.report or {}).get("grouping") or {}
    if gi.get("composite"):
        warnings.warn(
            f"Composite group {gi['used']} yields {gi['n_units']} units. If the same subject "
            "appears in more than one of these columns (e.g. one animal across several runs), "
            "combining them SPLITS that subject across folds and re-opens the leak. Use a list "
            "only for nested factors whose coarsest shared block is their combination; otherwise "
            "set group to the single outermost unit.", stacklevel=2)
    if aligned.task == "classification" and aligned.y is not None:
        y = np.asarray(aligned.y)
        groups_per_class = pd.DataFrame({"y": y, "g": g}).groupby("y")["g"].nunique()
        if (groups_per_class < 2).any():
            raise ValueError(
                "The 'group' column is confounded with the outcome: at least one outcome class "
                "comes from a single group, so leakage-safe (group-aware) cross-validation would "
                "hold that whole class out and cannot train. Group by the independent replicate "
                "unit (culture / colony / subject / plot), not by the strain or condition that "
                "defines the outcome.")


def compute_provenance_hash(
    sample_ids: list[str],
    modalities: dict[str, ModalityMatrix],
    target_name: str,
    task: str,
    target_values=None,
) -> str:
    """SHA-256 over the aligned sample index, feature footprints, and matrix content.

    Tamper-evident at the value level: the manifest includes a per-modality
    content digest of the numeric matrix (canonical row/column order) and a
    digest of the encoded target, so changing any measurement -- not only the
    samples or features -- changes the hash.
    """
    h = hashlib.sha256()
    manifest: dict = {
        "samples": list(sample_ids),
        "target": target_name,
        "task": task,
        "modalities": {
            name: {
                "features": m.feature_names,
                "shape": list(m.shape),
                "content_sha256": _matrix_digest(m.frame, sample_ids),
            }
            for name, m in sorted(modalities.items())
        },
    }
    if target_values is not None:
        y_arr = np.ascontiguousarray(np.asarray(target_values, dtype=np.float64))
        manifest["target_sha256"] = hashlib.sha256(y_arr.tobytes()).hexdigest()

    import json

    h.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def align_modalities(
    modalities: dict[str, pd.DataFrame],
    clinical: pd.DataFrame,
    config,
) -> AlignedDataset:
    """Align modalities and clinical data into a frozen :class:`AlignedDataset`.

    Steps: normalize ids, resolve orientation per modality, intersect samples
    across all modalities and the clinical target, drop target-missing records,
    namespace features to block cross-modality duplicates, encode the target,
    and hash provenance.
    """
    clin_spec = config.clinical
    norm = config.normalization
    report: dict[str, Any] = {"orientation": {}, "dropped": {}, "notes": []}

    # -- clinical target / metadata ---------------------------------------- #
    clinical = clinical.copy()
    clinical.columns = [str(c).strip() for c in clinical.columns]
    if clin_spec.sample_id and clin_spec.sample_id in clinical.columns:
        clinical = clinical.set_index(clin_spec.sample_id)
    clinical.index = pd.Index(
        normalize_names(clinical.index, norm) if norm.enabled else [str(i).strip() for i in clinical.index]
    )
    clinical = clinical[~clinical.index.duplicated(keep="first")]

    if clin_spec.target not in clinical.columns:
        raise ValueError(
            f"Target column '{clin_spec.target}' not found in clinical table "
            f"(columns: {list(clinical.columns)[:20]})."
        )

    y_all = clinical[clin_spec.target]
    if clin_spec.drop_missing_target:
        keep = y_all.notna() & (y_all.astype("string").str.strip().str.lower() != "")
        n_drop = int((~keep).sum())
        if n_drop:
            report["dropped"]["missing_target"] = n_drop
        clinical = clinical.loc[keep]

    reference = set(clinical.index.astype(str))
    if not reference:
        raise ValueError("No samples remain after dropping records with a missing target.")

    # -- per-modality ingestion and orientation ---------------------------- #
    prepared: dict[str, pd.DataFrame] = {}
    mod_specs = {m.name: m for m in config.modalities} if config.modalities else {}
    for name, frame in modalities.items():
        spec = mod_specs.get(name)
        forced = spec.orientation if spec else "auto"
        oriented, note = resolve_orientation(frame, reference, norm, forced)
        report["orientation"][name] = note
        if norm.enabled:
            oriented = oriented.copy()
            oriented.index = pd.Index(normalize_names(oriented.index, norm))
        oriented = _collapse_duplicates(oriented)
        # De-duplicate feature columns within a modality (keep first).
        if oriented.columns.has_duplicates:
            oriented = oriented.loc[:, ~oriented.columns.duplicated(keep="first")]
        prepared[name] = oriented

    # -- intersect samples across all modalities and clinical -------------- #
    common = set(clinical.index.astype(str))
    for frame in prepared.values():
        common &= set(frame.index.astype(str))
    if not common:
        raise ValueError(
            "Sample-id intersection across modalities and clinical is empty. "
            "Check id formatting / normalization or the sample-id column."
        )
    sample_ids = sorted(common)
    report["n_samples_aligned"] = len(sample_ids)
    report["n_samples_clinical"] = len(reference)

    # -- build aligned modality matrices ----------------------------------- #
    aligned_mods: dict[str, ModalityMatrix] = {}
    for name, frame in prepared.items():
        sub = frame.reindex(sample_ids)
        sub = sub.astype("float64")
        # Structural gate: block modalities that carry no numeric signal.
        finite_cols = sub.notna().any(axis=0)
        if not bool(finite_cols.any()):
            report["notes"].append(f"Modality '{name}' dropped: no numeric data after alignment.")
            continue
        sub = sub.loc[:, finite_cols]
        desc = mod_specs[name].description if name in mod_specs else ""
        aligned_mods[name] = ModalityMatrix(name=name, frame=sub, description=desc)

    if not aligned_mods:
        raise ValueError("All modalities were dropped during alignment (no numeric data).")

    # -- target / groups / batch ------------------------------------------- #
    clinical = clinical.loc[sample_ids]
    is_survival = clin_spec.task == "survival"
    if is_survival:
        if not clin_spec.time or clin_spec.time not in clinical.columns:
            raise ValueError("task='survival' requires clinical.time (a numeric time-to-event column).")
        if not clin_spec.event or clin_spec.event not in clinical.columns:
            raise ValueError("task='survival' requires clinical.event (1 = event, 0 = right-censored).")
        target_col = clin_spec.time
        task = "survival"
        y_enc, class_names = pd.to_numeric(clinical[clin_spec.time], errors="coerce").astype("float64"), None
    else:
        target_col = clin_spec.target
        task = _infer_task(clinical[clin_spec.target], clin_spec.task)
        y_enc, class_names = _encode_target(clinical[clin_spec.target], task, clin_spec.positive_label)
    y_enc.index = pd.Index(sample_ids)

    # Drop samples whose encoded target is NaN (regression coercion failures).
    valid = y_enc.notna()
    if not bool(valid.all()):
        keep_ids = [s for s, v in zip(sample_ids, valid) if v]
        report["dropped"]["uncodable_target"] = int((~valid).sum())
        sample_ids = keep_ids
        y_enc = y_enc.loc[sample_ids]
        for name in list(aligned_mods.keys()):
            aligned_mods[name] = ModalityMatrix(
                name=name,
                frame=aligned_mods[name].frame.loc[sample_ids],
                description=aligned_mods[name].description,
            )
        clinical = clinical.loc[sample_ids]

    groups, group_info = _resolve_group_series(clinical, clin_spec.group, sample_ids)
    if group_info is not None:
        report["grouping"] = group_info
        if group_info["missing"]:
            report.setdefault("notes", []).append(
                f"Grouping columns not found and ignored: {group_info['missing']}.")
    batch = None
    if clin_spec.batch and clin_spec.batch in clinical.columns:
        batch = clinical[clin_spec.batch].astype("string").fillna("NA")
        batch.index = pd.Index(sample_ids)

    event = None
    if is_survival:
        event = pd.to_numeric(clinical[clin_spec.event], errors="coerce").fillna(0.0).astype("float64")
        event.index = pd.Index(sample_ids)
        event = event.loc[sample_ids]

    y_raw = clinical[target_col].copy()
    y_raw.index = pd.Index(sample_ids)

    tv = y_enc.reindex(sample_ids).to_numpy()
    if is_survival and event is not None:                 # event is part of the target -> hash it too
        tv = np.concatenate([tv, event.reindex(sample_ids).to_numpy()])
    prov = compute_provenance_hash(sample_ids, aligned_mods, target_col, task, target_values=tv)

    if task == "classification":
        counts = y_enc.astype("int64").value_counts().to_dict()
        report["class_balance"] = {str(k): int(v) for k, v in counts.items()}
    if is_survival:
        report["survival"] = {"n_events": int(event.sum()), "n_censored": int((event == 0).sum()),
                              "time_unit": clin_spec.time_unit}

    return AlignedDataset(
        modalities=aligned_mods,
        y=y_enc.astype("float64"),
        task=task,
        sample_ids=list(sample_ids),
        y_raw=y_raw,
        class_names=class_names,
        groups=groups,
        batch=batch,
        event=event,
        time_unit=clin_spec.time_unit,
        provenance_hash=prov,
        report=report,
    )


def load_and_align(config) -> AlignedDataset:
    """Read every modality + clinical file referenced by ``config`` and align."""
    if not config.modalities:
        raise ValueError("Config declares no modalities.")
    if not config.clinical.path:
        raise ValueError("Config declares no clinical.path.")

    modalities: dict[str, pd.DataFrame] = {}
    for spec in config.modalities:
        if not spec.path:
            warnings.warn(f"Modality '{spec.name}' has no path; skipping.", stacklevel=2)
            continue
        modalities[spec.name] = read_matrix(
            spec.path, orientation="auto", id_regex=spec.id_regex
        )

    clin_path = Path(config.clinical.path)
    text = clin_path.read_text(encoding="utf-8-sig", errors="replace")
    delim = _detect_delimiter(text[:8192])
    clinical = pd.read_csv(io.StringIO(text), sep=delim, engine="python", dtype="string")

    return align_modalities(modalities, clinical, config)
