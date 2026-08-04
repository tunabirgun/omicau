# Preprocessing rules

## 1. The rule

Everything learned from data is learned **inside the training fold only** and applied
unchanged to the held-out fold:

imputation · scaling · variance filtering · feature selection · batch-adjustment
parameters · latent factors · model parameters · hyperparameters · calibration ·
stacking meta-features · decision thresholds.

No statistic computed on a validation or test row may influence training. This
applies identically to omicau and to every comparator; a comparator whose official
implementation cannot honour it is recorded in `COMPARATOR_MANIFEST.yaml` as
transductive and is either excluded or reported separately, never quietly included.

## 2. Ingestion, before any modelling

Ingestion is dataset-level and outcome-blind: delimiter inference, orientation
resolution, sample-name normalization, numeric coercion, and intersection of samples
across modalities and the clinical table. None of these use the outcome, so they do
not leak. Missing values are kept as true `NaN` masks at ingest and are **not**
imputed there.

Feature-level filtering that uses the outcome — differential expression, univariate
selection, correlation with the target — is modelling, not ingestion, and is subject
to §1.

## 3. Per-fold pipeline

Fitted inside each training fold, in order: median imputation with training-fold
medians; zero-variance filtering; standardization with training-fold mean and
standard deviation; optional univariate selection capped at
`tuning.omicau_search_space_when_tuned["classical.max_features"]` in the tuned arm
and at the package default in the primary arm.

The neural path standardizes over observed entries only, with training-fold
statistics, so masked entries contribute nothing to the fitted scale.

## 4. Missing-value handling is a comparison, not a default

Two handlings are compared in the ablations: the masked neural path, in which a
missing feature contributes nothing to the pooled embedding, and median imputation.
Neither is assumed superior; the M1 stress tests measure how each behaves under
MCAR, MAR, MNAR, batch-associated missingness and whole-modality absence.

## 5. Batch adjustment

Batch adjustment is **not** applied as a default correction. Where it is used as a
sensitivity probe, the adjustment is fitted inside the training fold and applied to
that fold's validation rows, and it is hard-gated off when batch is confounded with
the outcome, since removing a confounded batch effect inflates downstream confidence
rather than repairing it. The probe emits no corrected dataset.

## 6. Deliberately unsafe variants

The L2 stress tests construct pipelines that violate §1 on purpose: scaling before
cross-validation, feature selection on the complete dataset, imputation on the
complete dataset, latent factors fitted on the complete dataset. They exist to
measure whether the audit notices. Every table, figure and caption that reports them
labels them as challenge pipelines. They are never presented as a recommended method,
and their inflated scores are never compared against correctly fitted pipelines
without that label.
