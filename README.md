# Omicau

Local CLI and optional browser interface for auditing multi-omic inputs and assessing what an assay adds to a predictive model. Outputs include an HTML report, model metrics, paired assay assessments and input/provenance records.

[Run guide](https://tunabirgun.github.io/omicau/guide.html) · [Interpretation](https://tunabirgun.github.io/omicau/interpretation.html) · [Evaluation](https://tunabirgun.github.io/omicau/reproducibility.html)

## Install

With Python 3.10+, install the wheel from the [v0.6.0 release](https://github.com/tunabirgun/omicau/releases/tag/v0.6.0):

```bash
python -m pip install ./omicau-0.6.0-py3-none-any.whl
```

For the browser interface, extract the release source and run:

```bash
python -m pip install ".[ui]"
omicau ui
```

Keep the terminal open and use its printed local address. Release assets also include Windows x86-64, Linux x86-64 and macOS ARM64 desktop bundles; Windows is unsigned and macOS is not signed/notarized. Version 0.6.0 was not published to PyPI or Conda.

## Try an offline example

```bash
omicau check-env
omicau bootstrap --dataset mock --out-dir demo
omicau run --config demo/config.json --cores 1
omicau verify --config demo/config.json --audit demo/run/audit.json
```

Open `demo/run/report.html`. This is synthetic demonstration data, separate from the archived evaluation. Increase concurrency only within available resources.

Real runs use one numeric matrix per assay and a clinical table defining identifiers and outcome. Preserve missing values and declare the outermost independent unit through `clinical.group`; repeated observations must stay grouped. See the run guide for configuration and supported designs.

## Interpretation and verification

The assay ledger compares a prespecified classical reference model with refitted assay-omission models on shared assessment partitions. Its unrounded family decision interval determines increased, reduced or unresolved performance; unavailable pairs retain reasons. Standalone, neural and feature-attribution results do not determine the ledger category.

Intervals resample saved held-out predictions and exclude retraining, feature-reselection and partition-selection variability. Family quantiles have no established population-coverage or family-error guarantee. One-assay contribution is inapplicable, and survival is unsupported by this paired rule. Results establish neither clinical utility nor causal biology.

`omicau verify` checks the aligned-input fingerprint; it does not replay fitting or validate every configuration, environment detail or conclusion. Retain inputs, configuration and outputs together. Stacking and fixed-plan controls require the contracts in [fixed-split controls](docs/fixed-split-controls.md).

## Reproducibility

- [Results v3.0.0](https://doi.org/10.5281/zenodo.23444757)
- [Protocol v6.3.0](https://doi.org/10.5281/zenodo.23444777): Executed-study consolidation and deviations, not a new preregistration.
- [Benchmark-record guide](docs/benchmark-records.md): Current deposits versus legacy repository materials.

Archived evaluations are separate from ordinary application runs. Their incomplete cases and comparator restrictions remain part of the evidence; secondary comparisons do not establish a general method ranking. Prediction-level reproduction requires obtaining the documented data under its access terms.

## Development and license

```bash
python -m pip install ".[dev]"
python -m pytest
```

[MIT](LICENSE). Source datasets retain their own licences and access conditions.
