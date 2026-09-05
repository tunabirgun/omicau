# omicau

`omicau` v0.4.0 is a local-first command-line tool for leakage-aware multi-omics data auditing and fusion benchmarking. It aligns molecular layers, audits missingness and batch structure, evaluates models with group-aware resampling, records value-level provenance, and writes a self-contained HTML report with machine-readable results.

Source and release code: [github.com/tunabirgun/omicau](https://github.com/tunabirgun/omicau). Frozen benchmark suite: [https://doi.org/10.5281/zenodo.22304152](https://doi.org/10.5281/zenodo.22304152).

Research use only. Predictive performance does not establish clinical utility, individual risk, treatment benefit, or causal biology.

## Scope

- Ingest CSV or TSV modality matrices and a clinical table, with orientation detection and sample alignment.
- Audit missingness, batch structure, group structure, and modality availability before interpreting model results.
- Compare single-modality, classical fusion, and neural fusion models under shared group-aware splits.
- Produce an HTML report, auditable JSON and CSV outputs, a model card, runtime record, and provenance checks.

## Completed benchmark summary

Omicau v0.4.0 was evaluated under the frozen suite at [DOI 10.5281/zenodo.22304152](https://doi.org/10.5281/zenodo.22304152). That pre-execution archive contains the protocol and executable suite, not benchmark results. The completed analysis recorded 13,507 of 13,507 registered units with zero execution failures across binary and continuous synthetic families, five internal real cohorts, and an exploratory CPTAC cohort.

| Benchmark check | Recorded result |
| --- | --- |
| Execution completeness | 13,507/13,507 registered units; zero execution failures |
| Synthetic coverage | Binary and continuous families |
| Real-data coverage | Five internal cohorts; CPTAC exploratory (27 units) |
| Required-method null safeguards | Each required method: 0/40 false alarms in both synthetic families |
| Required-method signal safeguards | Each required method: 40/40 detections in both synthetic families |
| Supplemental synthetic controls | DIABLO: 0/40 false alarms and 40/40 detections in the binary family; block-PLS: 0/40 and 33/40 in the continuous family |
| Omicau 0.3.0/0.4.0 fitting-time ratio | 15.2–25.2× across matched 45-unit internal workloads |
| Selected established-software comparator/Omicau 0.4.0 runtime ratio | 14.2–17.4× per registered unit on internal cohorts |
| Public-release mapping | 73/73 submitted source files matched the PyPI source distribution; 47/47 implementation files matched the source distribution, wheel, and release tag |
| Released-software tests | 244/244 passed on the exact PyPI source distribution |

The efficiency ratios are registered-environment, complete-pipeline fitting-time measurements for the stated benchmark design. They are not end-to-end, energy-use, or hardware-general estimates. Predictive effects were cohort-specific; they do not establish overall predictive superiority for omicau or any fusion method.

## Installation

Python 3.10 or later is required.

```bash
python -m pip install "omicau==0.4.0"
```

From a source checkout:

```bash
python -m pip install .
```

Install optional capabilities only when needed:

```bash
python -m pip install "omicau[ui]"
python -m pip install "omicau[data]"
python -m pip install "omicau[cptac]"
python -m pip install "omicau[dev]"
```

`ui` provides a local browser interface. `data` enables supported public-data connectors. `cptac` is separate because its dependency requirements may need a compiler toolchain. `dev` installs test tooling.

## Minimal usage

Check the installed environment, create an offline example, run it, then verify the stored provenance.

```bash
omicau check-env
omicau bootstrap --dataset mock --out-dir demo
omicau run --config demo/config.json --cores 8
omicau verify --config demo/config.json --audit demo/run/audit.json
```

Open `demo/run/report.html` after the run. Use `omicau <command> --help` for the complete command reference.

## Inputs

Provide one numeric matrix per modality and one clinical table. Matrices may have samples in rows or columns. Retain missing values as missing values; transformations and learned preprocessing are fitted within the relevant training data.

```text
study/
  rna.csv
  protein.csv
  clinical.csv
  config.json
```

The clinical table identifies the sample, prediction target, and outermost independent unit. Set `clinical.group` to the person, specimen source, family, repeated-measure unit, or other unit that must not cross a resampling split.

## Minimal configuration

```json
{
  "run_name": "my_study",
  "output_dir": "run",
  "seed": 42,
  "modalities": [
    {
      "name": "rna",
      "path": "rna.csv"
    },
    {
      "name": "protein",
      "path": "protein.csv"
    }
  ],
  "clinical": {
    "path": "clinical.csv",
    "sample_id": "sample_id",
    "target": "outcome",
    "group": "patient_id",
    "batch": "batch",
    "task": "classification"
  },
  "cv": {
    "n_splits": 5,
    "n_bootstrap": 1000
  },
  "compute": {
    "cores": 8,
    "device": "auto"
  }
}
```

`seed` anchors stochastic steps. Choose `n_splits` only when each training and assessment partition remains suitable for the target prevalence and grouping structure. Set `cores` to a value compatible with available memory; a higher value is not automatically a faster or safer analysis.

## Commands

| Command | Purpose |
| --- | --- |
| `omicau check-env` | Report compute status, directory access, and optional dependency availability. |
| `omicau bootstrap` | Create a mock dataset or assemble a supported public cohort. |
| `omicau run` | Run the audit and write report assets. |
| `omicau verify` | Recompute or compare the recorded provenance hash. |
| `omicau ui` | Start the optional local browser interface. |

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

## Audit safeguards

- Sample alignment and input orientation are checked before model fitting.
- Imputation, scaling, variance filtering, feature selection, calibration, thresholds, and stacking features are fitted within training data.
- Shared group-aware folds preserve the outermost independent unit across model comparisons.
- Missingness, batch, and group diagnostics are reported as risks to investigate, not as causal findings.
- Null controls, target shuffling, and negative-control modalities remain controls rather than recommended analyses.
- Failed methods and incomplete runs are recorded rather than silently removed.
- Each required gate method produced 0/40 null false alarms and 40/40 injected detections in both synthetic families. Supplemental block-PLS produced 0/40 false alarms and 33/40 continuous detections; supplemental DIABLO produced 0/40 and 40/40 in its eligible binary family.

These safeguards reduce specific, testable sources of optimistic bias. They do not make an observational dataset clinically valid, eliminate confounding, or convert a model association into a biological mechanism.

## Reproducibility and public release mapping

| Public item | Identifier | Role |
| --- | --- | --- |
| Software release | [`omicau` v0.4.0](https://github.com/tunabirgun/omicau/releases/tag/v0.4.0) | Installable command-line software and archived source code. |
| Source repository | [github.com/tunabirgun/omicau](https://github.com/tunabirgun/omicau) | Release code, configuration schema, and documentation. |
| Frozen benchmark suite | [DOI 10.5281/zenodo.22304152](https://doi.org/10.5281/zenodo.22304152) | Pre-execution protocol, code, partitions, and environment specification. |

Each run records the resolved configuration, aligned-value SHA-256 provenance, relevant dependency versions, diagnostics, and output paths in `audit.json`. Re-run `omicau verify` against the source configuration and stored audit to detect changes in aligned inputs or feature footprints.

The benchmark mapping compared extracted file bytes without normalization. All 73 submitted source files matched the PyPI source distribution, and all 47 implementation files matched across the source distribution, wheel, and v0.4.0 release tag. The exact PyPI source distribution passed 244 software tests; those tests establish defined behavior and failure handling, not biological or clinical validity.

The aggregate results on this page summarize the completed analysis prepared for the companion article; they are distinct from the frozen pre-execution Zenodo archive.

For development checks:

```bash
python -m pip install "omicau[dev]"
python -m pytest
```

## Research-use limitations

- omicau is not a diagnostic device and must not guide clinical decisions or treatment.
- Model estimates are conditional on the available cohort, endpoint definition, data processing, grouping, and validation design.
- Cohort-specific predictive effects must not be generalized as universal method superiority.
- Diagnostics flag potential data-quality or design problems; they do not prove their cause or correct the underlying data.
- External validation, domain review, and appropriate governance remain the user's responsibility.

## License

omicau is distributed under the [MIT License](LICENSE).

## Article

Work in progress.
