# omicau

`omicau` is a local-first command-line tool for leakage-aware multi-omics data auditing and fusion benchmarking. It aligns molecular layers, checks missingness and batch structure, evaluates classical and neural fusion under group-aware cross-validation, records value-level provenance, and produces a self-contained HTML report with machine-readable outputs.

Research use only. Predictive performance does not establish clinical utility or causal biology.

## Main capabilities

- CSV and TSV ingestion with orientation detection and cross-file sample alignment.
- Group-aware cross-validation for repeated measures and related samples.
- Missingness-bias, batch-confounding, and group-structure diagnostics.
- Classical single-modality and fusion models plus availability-aware gated residual neural fusion.
- Fold-local preprocessing, fixed random seeds, and provenance hashes.
- Modality utility, redundancy, permutation importance, and negative controls.
- Self-contained HTML, JSON, CSV, runtime log, and model-card outputs.
- Optional local web interface and public-data connectors.

## Neural fusion

By default, neural benchmarks use availability-aware gated residual fusion: each modality is encoded with masked pooling and a nonlinear projection, produces a unimodal prediction, and contributes through a normalized gate across observed modalities; a zero-initialized residual adds cross-modal interactions. Set `"neural": {"architecture": "legacy"}` to use the previous masked global-pooling fusion.

## Installation

Python 3.10 or later is required.

```bash
python -m pip install omicau
```

Install optional features only when needed:

```bash
python -m pip install "omicau[ui]"       # local browser interface
python -m pip install "omicau[data]"     # public-data connectors
python -m pip install "omicau[dev]"      # tests and coverage
```

From a source checkout:

```bash
python -m pip install .
```

## Quick start

```bash
omicau check-env
omicau bootstrap --dataset mock --out-dir demo
omicau run --config demo/config.json --cores 8
omicau verify --config demo/config.json --audit demo/run/audit.json
```

Open `demo/run/report.html` after the run.

## Input layout

Provide one numeric matrix per modality and one clinical table. Matrices may use samples as rows or columns. Missing values should remain missing; learned preprocessing is fitted inside training folds.

```text
study/
  rna.csv
  protein.csv
  clinical.csv
  config.json
```

Minimal configuration:

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

Set `clinical.group` to the outermost independent unit. All rows from the same group remain on one side of each cross-validation split.

## Commands

| Command | Purpose |
| --- | --- |
| `omicau check-env` | Report compute and optional-dependency availability. |
| `omicau bootstrap` | Create a mock dataset or retrieve a supported public cohort. |
| `omicau run` | Run the audit and write the report assets. |
| `omicau verify` | Recompute data provenance and verify a stored audit. |
| `omicau ui` | Start the optional local browser interface. |

Use `omicau <command> --help` for the complete option list.

## Outputs

| File | Content |
| --- | --- |
| `report.html` | Self-contained interactive report. |
| `audit.json` | Configuration, diagnostics, metrics, provenance, and environment record. |
| `model_metrics.csv` | Per-model metrics and confidence intervals. |
| `modality_ledger.csv` | Standalone scores, marginal gains, redundancy, and verdicts. |
| `missingness_tests.csv` | Missingness-bias tests and adjusted p-values. |
| `MODEL_CARD.md` | Intended use, data, methods, metrics, and limitations. |
| `runtime_log.txt` | Stage-level wall time and normalized compute information. |

## Methodological safeguards

Imputation, scaling, variance filtering, feature selection, batch adjustment, calibration, thresholds, and stacking features are fitted inside training data only. Model assessment uses shared group-aware folds. Null controls, target shuffling, and negative-control modalities are retained as controls rather than presented as recommended analyses. Failed methods and incomplete runs are reported instead of silently removed.

Run identities include aligned-value SHA-256 provenance, configuration, relevant dependency versions, and implementation checks.

## Development

```bash
python -m pip install "omicau[dev]"
python -m pytest
```

Documentation: [tunabirgun.github.io/omicau](https://tunabirgun.github.io/omicau/)

License: [MIT](LICENSE)

## Article

The companion research article is a **work in progress**. Results and claims will be added only after the registered benchmark and independent validation gates are complete.
