# Omicau

Omicau v0.6.0 is a local command-line and optional browser workflow for assessing what an assay adds to a predictive model. It aligns multi-omic inputs, examines missingness and batch information, compares single-assay and fusion models, and produces an HTML report with JSON and CSV outputs.

[Start a run](https://tunabirgun.github.io/omicau/guide.html) · [Read a report](https://tunabirgun.github.io/omicau/interpretation.html) · [Inspect the evaluation](https://tunabirgun.github.io/omicau/reproducibility.html)

## Installation

Download the wheel from the [v0.6.0 release](https://github.com/tunabirgun/omicau/releases/tag/v0.6.0). With Python 3.10 or later, run:

```bash
python -m pip install ./omicau-0.6.0-py3-none-any.whl
```

For the browser interface, extract the release source archive and run from its directory:

```bash
python -m pip install ".[ui]"
omicau ui
```

Open the local address printed in the terminal. Keep the terminal running while using the interface. The interface exposes a subset of the configuration options. Optional source-install extras include `data` for public-data connectors, `cptac` for CPTAC access, and `dev` for testing; CPTAC dependencies may require a compiler.

Desktop bundles support Windows x86-64, Linux x86-64 and macOS ARM64. Windows binaries are unsigned; the macOS bundle is not signed or notarized. Install from the release assets: v0.6.0 was not published to PyPI or a Conda channel.

## Minimal usage

Create an offline synthetic example, run it, and check its recorded input fingerprint:

```bash
omicau check-env
omicau bootstrap --dataset mock --out-dir demo
omicau run --config demo/config.json --cores 8
omicau verify --config demo/config.json --audit demo/run/audit.json
```

Open `demo/run/report.html`. Choose a core count that fits available memory and workload. The mock example is separate from the archived evaluation datasets. Use `omicau <command> --help` for command options.

## Inputs

Provide one numeric CSV or TSV matrix per assay and a clinical table containing sample identifiers and the outcome. Samples may occupy matrix rows or columns; retain missing values. Set `clinical.group` to the outermost independent unit so that repeated observations from the same participant or source stay together across assessment partitions.

## Minimal configuration

Save this as `study/config.json`, alongside the input files. Replace the example column names and omit `batch` when no batch variable is available.

```json
{
  "run_name": "my_study",
  "output_dir": "run",
  "seed": 42,
  "modalities": [
    {"name": "rna", "path": "rna.csv"},
    {"name": "protein", "path": "protein.csv"}
  ],
  "clinical": {
    "path": "clinical.csv",
    "sample_id": "sample_id",
    "target": "outcome",
    "group": "patient_id",
    "batch": "batch",
    "task": "classification"
  },
  "cv": {"n_splits": 5, "n_bootstrap": 1000},
  "compute": {"cores": 8, "device": "auto"}
}
```

Run `omicau run --config study/config.json`. Choose a fold count suitable for the outcome and grouping structure. Without a grouping variable, set `independent_rows` to `true` under `cv` only when the study design supports independent rows.

## Methodology

Models compare each assay alone, full fusion, and refitted assay-omission subsets on shared assessment partitions. Learned model preprocessing is fitted within the relevant training folds. Transformations performed before importing data remain the user's responsibility.

### Reading layer results

The ledger uses the classical reference estimator declared before fitting. Its contrast is the full-model score minus the omitted-assay score: AUROC for classification and R-squared for regression. Standalone scores, neural results, feature attributions and diagnostic warnings do not determine its category.

The **unrounded family decision interval** determines increased performance (entirely above zero), reduced performance (entirely below zero), or unresolved (crosses or touches zero). Invalid or missing pairs are unavailable, with a reason. The ordinary 95% interval is separate. With one assay, contribution is not applicable; survival is unsupported by this paired rule.

Intervals resample saved held-out predictions at the declared independent-unit level. They exclude retraining, feature-reselection and partition-selection variability. The family quantiles have no established population-coverage or family-error guarantee. Read contrasts alongside absolute scores and diagnostics; unresolved does not mean an absence of biological information.

Research use only. Results depend on the cohort, endpoint, assays and reference model; they do not establish clinical utility or causal biology. The [interpretation guide](https://tunabirgun.github.io/omicau/interpretation.html) explains these distinctions.

## Fixed split manifests and controls

Stacking requires a validated nested plan through `cv.split_manifest` and `cv.inner_splits`; ordinary dynamic-split runs report it as unavailable. The manifest format is `omicau_public_split_manifest_v2`, bound to aligned values, group identities and declared design strata.

Fixed-plan target-permutation controls require `clinical.permutation_strata` declared before fitting. They permute endpoint-constant groups within training strata under an exchangeability assumption. Feature-shuffling, noise and legacy dynamic-split controls are descriptive stress checks, not universal leakage guarantees.

## Outputs and verification

| Output | Purpose |
| --- | --- |
| `report.html` | Interactive report. |
| `audit.json` | Configuration, diagnostics, metrics, provenance and environment record. |
| `model_metrics.csv`, `modality_ledger.csv` | Model scores, paired layer assessments and diagnostic flags. |
| `missingness_tests.csv` | Missingness tests and adjusted P values. |
| `MODEL_CARD.md`, `runtime_log.txt` | Analysis scope and execution information. |

Retain configuration, inputs and outputs together. `omicau verify --config study/config.json --audit study/run/audit.json` compares the aligned-input fingerprint, including measurements and encoded outcomes. It does not replay model fitting or validate every configuration setting, environment detail or scientific conclusion.

## Evaluation and reproducibility

| Resource | Scope |
| --- | --- |
| [Results v3.0.0](https://doi.org/10.5281/zenodo.23444757) | Current aggregates, environment specifications and numbered reproduction scripts. |
| [Protocol v6.3.0](https://doi.org/10.5281/zenodo.23444777) | Executed-study consolidation and deviations; not a new preregistration. |
| [Original release validation archive](https://github.com/tunabirgun/omicau/releases/download/v0.6.0/omicau-validation-0.6.0.zip) | Materials bundled on 15 October 2026; later comparisons are in the current results archive. |

The evaluation includes 737 completed simulation datasets out of 740 intended, 20 matched preprocessing-fault pairs, 40 conditional draws on one molecular background, and five installed-package random-forest applications. The core extension records 443 of 445 fits and 164 of 165 contrasts. The expanded native panel has 303 qualified slots out of 360, 16 complete series out of 24, and 32 numerical comparisons out of 48. Unavailable outcomes remain recorded; incomplete series are not pooled.

Comparisons concern the evaluated configurations, including dense DIABLO-framework PLS-DA and component-level fuseMLR and IntegratedLearner adapters. No IntegratedLearner series met the complete-series requirement. Expanded-panel comparisons are secondary and descriptive; they do not establish a general method ranking. See the [evaluation guide](https://tunabirgun.github.io/omicau/reproducibility.html#evaluation) for tuning, eligibility and uncertainty.

Archived evaluation scripts are separate from ordinary CLI and browser runs. Start with the archive README and `protocol/EXECUTED_STUDY.md`. Molecular matrices, individual outcomes, identifiers and predictions are excluded; prediction-level reproduction requires obtaining the documented sources under their access terms and generating records locally. See [reproduction scope](https://tunabirgun.github.io/omicau/reproducibility.html#reproduce) and [earlier records](https://tunabirgun.github.io/omicau/reproducibility.html#history).

## Development

From the source directory:

```bash
python -m pip install ".[dev]"
python -m pytest
```

## License

Omicau is distributed under the [MIT License](LICENSE). Source datasets retain their own licences and access conditions.
