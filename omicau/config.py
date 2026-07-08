"""Configuration schema and loader for omicau.

The configuration is intentionally forgiving: unknown keys are ignored with a
warning, missing keys fall back to documented defaults, and the file may be
JSON, TOML, or YAML (YAML requires the optional ``pyyaml`` dependency). This
keeps configuration friction-free for non-technical users while remaining fully
reproducible -- every resolved value is serialized back into the run manifest.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Nested specification dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class ModalitySpec:
    """One omic modality (a sample x feature matrix on disk or in memory)."""

    name: str
    path: str | None = None
    #: "auto" lets the alignment engine decide via overlap scoring; "samples_as_rows"
    #: or "samples_as_cols" force an orientation.
    orientation: str = "auto"
    #: Optional regex whose first capture group extracts the canonical sample id.
    id_regex: str | None = None
    #: Human description carried into reports.
    description: str = ""


@dataclass
class ClinicalSpec:
    """The clinical / phenotype table carrying the prediction target."""

    path: str | None = None
    target: str = "target"
    #: Column holding sample identifiers; None => the table index / first column.
    sample_id: str | None = None
    #: Column defining groups for group-aware (leakage-safe) cross-validation,
    #: e.g. patient id when multiple samples share a patient.
    group: str | None = None
    #: Column defining batches for batch-effect diagnostics.
    batch: str | None = None
    #: For binary classification, the label treated as the positive class.
    positive_label: str | None = None
    #: "auto" | "classification" | "regression".
    task: str = "auto"
    drop_missing_target: bool = True


@dataclass
class CVSpec:
    n_splits: int = 5
    seed: int = 42
    shuffle: bool = True
    #: Group-level bootstrap resamples for the primary-metric confidence interval.
    n_bootstrap: int = 1000
    #: Opt-in cross-site stress test: also cross-validate the reference fusion with
    #: folds blocked on the batch column (leave-one-batch-out), giving an honest
    #: new-batch generalization estimate alongside the standard CV.
    batch_blocked: bool = False


@dataclass
class NeuralSpec:
    enabled: bool = True
    epochs: int = 60
    batch_size: int = 32
    hidden_dim: int = 64
    embed_dim: int = 32
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    pooling: str = "mean"  # "mean" | "max" | "attention"


@dataclass
class ClassicalSpec:
    enabled: bool = True
    #: Estimator keys resolved in models.classical (see that module for the map).
    models: list[str] = field(default_factory=lambda: ["linear", "random_forest"])
    #: Optional univariate pre-filter cap fitted strictly inside train folds.
    max_features: int | None = 2000


@dataclass
class XAISpec:
    enabled: bool = True
    permutation_repeats: int = 8
    top_k_features: int = 25


@dataclass
class ControlSpec:
    enabled: bool = True
    shuffle_target: bool = True
    shuffle_features: bool = True
    random_noise: bool = True


@dataclass
class ComputeSpec:
    #: None => auto (physical-core aware, leaves headroom for the OS).
    cores: int | None = None
    #: "auto" resolves to MPS > CUDA > CPU at runtime.
    device: str = "auto"
    #: PyTorch DataLoader workers (0 is safest / headless-cluster friendly).
    torch_workers: int = 0
    #: Opt-in strict determinism: enables torch deterministic algorithms and sets
    #: CUBLAS_WORKSPACE_CONFIG (off by default because some ops lack deterministic
    #: kernels; enabled with warn_only so it never hard-fails a run).
    deterministic: bool = False


@dataclass
class LLMSpec:
    enabled: bool = False
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 2000


@dataclass
class ReportingSpec:
    html: bool = True
    json: bool = True
    csv: bool = True
    #: The multi-format documentation compiler is a private manuscript tool and
    #: is not part of the app; the app emits only the HTML dashboard + JSON/CSV.
    docs: list[str] = field(default_factory=list)


#: Named normalization presets. "none" is the neutral default (whitespace-only,
#: case-preserving, no suffix stripping) so non-human / case-sensitive sample
#: ids are never silently altered. "tcga" opts into the legacy behavior:
#: uppercase + collapse a TCGA aliquot barcode to its patient stem.
_TCGA_ALIQUOT_SUFFIX = r"(-\d{2}[A-Z])?(-\d{2}[A-Z]-\d{4}-\d{2})?$"
NORMALIZATION_PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    "tcga": {"uppercase": True, "strip_suffix_regex": [_TCGA_ALIQUOT_SUFFIX]},
}


@dataclass
class NormalizationSpec:
    """Fuzzy sample-name normalization applied before alignment."""

    #: Named preset resolved at load time (see NORMALIZATION_PRESETS). Explicit
    #: fields below always override the preset. "none" keeps ids verbatim except
    #: for whitespace; "tcga" restores uppercase + aliquot-suffix collapse.
    preset: str = "none"
    enabled: bool = True
    #: Case-fold ids. Default False so case-sensitive ids are preserved; the
    #: "tcga" preset sets this True.
    uppercase: bool = False
    strip_whitespace: bool = True
    #: Regexes removed from the *start* of each id (batch prefixes).
    strip_prefix_regex: list[str] = field(default_factory=list)
    #: Regexes removed from the *end* of each id (barcode / vial suffixes).
    #: Empty by default; the "tcga" preset adds the aliquot-barcode collapse.
    strip_suffix_regex: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Top-level configuration
# --------------------------------------------------------------------------- #
_NESTED = {
    "clinical": ClinicalSpec,
    "cv": CVSpec,
    "neural": NeuralSpec,
    "classical": ClassicalSpec,
    "xai": XAISpec,
    "controls": ControlSpec,
    "compute": ComputeSpec,
    "llm": LLMSpec,
    "reporting": ReportingSpec,
    "normalization": NormalizationSpec,
}


@dataclass
class OmicauConfig:
    """Fully-resolved run configuration."""

    run_name: str = "omicau_run"
    output_dir: str = "omicau_output"
    #: Study organism, free-text, carried into audit.json / DOME / model card /
    #: report — never into the provenance hash (a mutable annotation, not data).
    #: "unspecified" so no organism is assumed; human hub clients stamp
    #: "Homo sapiens". Examples: "Mus musculus", "Danio rerio", "Zea mays".
    organism: str = "unspecified"
    modalities: list[ModalitySpec] = field(default_factory=list)
    clinical: ClinicalSpec = field(default_factory=ClinicalSpec)
    cv: CVSpec = field(default_factory=CVSpec)
    neural: NeuralSpec = field(default_factory=NeuralSpec)
    classical: ClassicalSpec = field(default_factory=ClassicalSpec)
    xai: XAISpec = field(default_factory=XAISpec)
    controls: ControlSpec = field(default_factory=ControlSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    llm: LLMSpec = field(default_factory=LLMSpec)
    reporting: ReportingSpec = field(default_factory=ReportingSpec)
    normalization: NormalizationSpec = field(default_factory=NormalizationSpec)
    #: Global master seed; individual components derive from it.
    seed: int = 42

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OmicauConfig":
        """Build a config from a plain dict, tolerating missing/unknown keys."""

        data = dict(data or {})
        kwargs: dict[str, Any] = {}

        # modalities: list of dict/str.
        raw_mods = data.pop("modalities", []) or []
        mods: list[ModalitySpec] = []
        for m in raw_mods:
            if isinstance(m, str):
                mods.append(ModalitySpec(name=Path(m).stem, path=m))
            elif isinstance(m, dict):
                mods.append(_filter_construct(ModalitySpec, m))
            elif isinstance(m, ModalitySpec):
                mods.append(m)
        kwargs["modalities"] = mods

        # nested specs.
        for key, spec_cls in _NESTED.items():
            if key in data:
                sub = data.pop(key)
                if isinstance(sub, dict):
                    if key == "normalization":
                        preset = NORMALIZATION_PRESETS.get(sub.get("preset", "none"), {})
                        sub = {**preset, **sub}  # explicit user keys win over preset
                    kwargs[key] = _filter_construct(spec_cls, sub)
                else:
                    kwargs[key] = sub

        # scalar top-level keys.
        top_names = {f.name for f in fields(cls)}
        for key in list(data.keys()):
            if key in top_names and key not in kwargs:
                kwargs[key] = data.pop(key)

        if data:
            warnings.warn(f"omicau config: ignoring unknown keys {sorted(data)}", stacklevel=2)

        cfg = cls(**kwargs)
        cfg._propagate_seed()
        return cfg

    @classmethod
    def from_file(cls, path: str | Path) -> "OmicauConfig":
        """Load a config from a .json, .toml, .yaml/.yml file (format by extension)."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "YAML config requires the optional 'pyyaml' dependency "
                    "(pip install omicau[yaml]); or use a .json/.toml config."
                ) from exc
            data = yaml.safe_load(text) or {}
        elif suffix == ".toml":
            try:
                import tomllib  # Python 3.11+
            except ModuleNotFoundError:  # pragma: no cover - py3.10
                import tomli as tomllib  # type: ignore
            data = tomllib.loads(text)
        else:  # default to JSON
            data = json.loads(text)

        cfg = cls.from_dict(data)
        cfg._resolve_paths(path.parent)
        return cfg

    def _resolve_paths(self, base: Path) -> None:
        """Resolve relative modality/clinical paths against the config directory."""
        base = Path(base)
        for spec in self.modalities:
            if spec.path and not Path(spec.path).is_absolute():
                candidate = base / spec.path
                if candidate.exists():
                    spec.path = str(candidate)
        if self.clinical.path and not Path(self.clinical.path).is_absolute():
            candidate = base / self.clinical.path
            if candidate.exists():
                self.clinical.path = str(candidate)
        # keep output_dir relative to the config too, unless absolute.
        if self.output_dir and not Path(self.output_dir).is_absolute():
            self.output_dir = str(base / self.output_dir)

    # -- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8", newline=""
        )

    # -- helpers ------------------------------------------------------------ #
    def _propagate_seed(self) -> None:
        """Keep the CV seed aligned with the master seed unless overridden."""

        if self.cv.seed == CVSpec().seed and self.seed != OmicauConfig().seed:
            self.cv.seed = self.seed

    @staticmethod
    def example() -> dict[str, Any]:
        """A ready-to-edit example configuration (used by ``bootstrap``)."""

        return {
            "run_name": "example_audit",
            "output_dir": "omicau_output",
            "organism": "Homo sapiens",
            "seed": 42,
            "modalities": [
                {"name": "rna", "path": "rna.csv", "description": "RNA-seq log-TPM"},
                {"name": "protein", "path": "protein.csv", "description": "Proteomics"},
                {"name": "noise", "path": "noise.csv", "description": "Negative control"},
            ],
            "clinical": {
                "path": "clinical.csv",
                "target": "label",
                "sample_id": "sample_id",
                "group": "patient_id",
                "batch": "batch",
                "task": "classification",
            },
            "cv": {"n_splits": 5, "seed": 42, "shuffle": True},
            "neural": {"enabled": True, "epochs": 60, "batch_size": 32},
            "classical": {"enabled": True, "models": ["linear", "random_forest"]},
            "xai": {"enabled": True, "permutation_repeats": 8, "top_k_features": 25},
            "controls": {"enabled": True},
            "compute": {"cores": None, "device": "auto"},
            "llm": {"enabled": False, "model": "claude-sonnet-5"},
            "reporting": {"html": True, "json": True, "csv": True},
        }


def _filter_construct(spec_cls: type, data: dict[str, Any]):
    """Construct a dataclass from ``data``, ignoring keys it does not define."""

    valid = {f.name for f in fields(spec_cls)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    unknown = set(data) - valid
    if unknown:
        warnings.warn(
            f"omicau config: ignoring unknown keys {sorted(unknown)} for {spec_cls.__name__}",
            stacklevel=2,
        )
    return spec_cls(**kwargs)
