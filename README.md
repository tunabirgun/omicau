# omicau

`omicau` v0.5.2 is a local command-line and browser-based workflow for multi-omics data auditing and fusion evaluation. It aligns molecular layers, examines missingness and batch information, compares single-layer and fusion models with group-aware resampling, and writes a self-contained HTML report with machine-readable results. The report distinguishes a layer's standalone predictive signal from its added contribution to a specified model and endpoint.

Source and release code: [github.com/tunabirgun/omicau](https://github.com/tunabirgun/omicau). Frozen benchmark suite: [https://doi.org/10.5281/zenodo.22304152](https://doi.org/10.5281/zenodo.22304152).

Ordinary users run `omicau run` or the optional local browser interface. The bundled `benchmarks/` harness and `benchmark_record/` files are archived evaluation materials; they are not invoked by the ordinary run or UI paths.

Research use only. Predictive performance does not establish clinical utility, individual risk, treatment benefit, or causal biology.

## Scope

- Ingest CSV or TSV modality matrices and a clinical table, with orientation detection and sample alignment.
- Audit missingness, batch structure, group structure, and modality availability before interpreting model results.
- Compare single-modality, classical fusion, and neural fusion models under shared group-aware splits.
- Produce an HTML report, auditable JSON and CSV outputs, a model card, runtime record, and provenance checks.

## Evaluation materials

The [frozen benchmark suite](https://doi.org/10.5281/zenodo.22304152) provides the registered evaluation protocol and executable materials. Its archived scripts are separate from the ordinary `omicau run` and browser workflows. The report from an ordinary run includes its own split, model, control, and runtime records so that a user can assess the specific analysis performed.

## Installation

Python 3.10 or later is required. Install the wheel supplied with the release:

```bash
python -m pip install ./omicau-0.5.2-py3-none-any.whl
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

## Fixed split manifests and controls

For a pre-specified nested evaluation, set `cv.split_manifest` and `cv.inner_splits` before fitting. A public manifest uses `omicau_public_split_manifest_v2` and is bound to aligned values, ordered group identities, and declared design strata. Stacking requires this validated nested plan; ordinary dynamic-split runs record stacking as unavailable rather than producing a nonnested estimate.

If a fixed-plan target-permutation control is enabled, declare `clinical.permutation_strata` before analysis. The control permutes endpoint-constant groups within declared training strata and is conditional on its exchangeability assumption. Feature shuffling, random noise, and legacy dynamic-split controls are descriptive stress controls; they do not establish group-safe randomization, universal leakage safety, causal validity, or external generalization.

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

These safeguards reduce specific, testable sources of optimistic bias. They do not make an observational dataset clinically valid, eliminate confounding, or convert a model association into a biological mechanism.

## Reproducibility and release materials

| Public item | Identifier | Role |
| --- | --- | --- |
| Software release | [`omicau` v0.5.2](https://github.com/tunabirgun/omicau/releases/tag/v0.5.2) | Installable command-line software and archived source code. |
| Source repository | [github.com/tunabirgun/omicau](https://github.com/tunabirgun/omicau) | Release code, configuration schema, and documentation. |
| Frozen benchmark suite | [DOI 10.5281/zenodo.22304152](https://doi.org/10.5281/zenodo.22304152) | Pre-execution protocol, code, partitions, and environment specification. |

Each run records the resolved configuration, aligned-value SHA-256 provenance, relevant dependency versions, diagnostics, and output paths in `audit.json`. Re-run `omicau verify` against the source configuration and stored audit to detect changes in aligned inputs or feature footprints.

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
