# Prespecified hypotheses and estimands

**Protocol:** omicau benchmark v1.0.0. All numeric design values and decision limits
are referenced from `benchmark_protocol.yaml`. Hypothesis labels are immutable after
freeze.

## Simulation hypotheses

### H1 — modality-utility recovery (primary)

- **Question:** Does the utility ledger recover the known role of each modality across
  the prespecified simulation cells?
- **Unit:** independently generated dataset.
- **Estimand:** cell-specific proportion of datasets for which the complete ledger
  classification equals the generator truth, with component-wise rates for
  predictive, not-additive, redundant and control-like modalities.
- **Decision rule:** report each rate with a Wilson interval and Monte Carlo SE; no
  post-result pass threshold may be introduced.
- **Boundary:** this assesses the frozen generators and thresholds only; it does not
  establish a universal biological definition of modality utility.

### H2 — risk-flag operating characteristics (secondary)

- **Question:** Under the prespecified batch and missingness mechanisms, how often do
  the corresponding audit flags agree with the known generator state?
- **Unit:** independently generated dataset.
- **Estimands:** sensitivity and false-warning rate by mechanism and severity, plus
  the proportion of indeterminate results.
- **Decision rule:** report rates with Holm adjustment inside the batch and
  missingness families under `statistical_analysis.multiplicity`.
- **Boundary:** these outputs are risk flags, not tests of a causal batch or
  missingness mechanism. Even in simulation, recovery is limited to the implemented
  data-generating process.

### H3 — target/pipeline sanity controls (secondary)

- **Question:** Do shuffled-target, column-shuffled and synthetic-noise controls show
  task-appropriate null behavior when the complete pipeline is nested correctly?
- **Unit:** independently generated dataset-control pair.
- **Estimand:** false-alarm rate and metric excess over the null reference for each
  control family.
- **Decision rule:** use the role and permitted conclusions under
  `diagnostic_interpretation.null_controls`.
- **Boundary:** these controls test the executed target/pipeline path. They are not
  arbitrary leakage detectors and cannot certify the absence of group leakage or
  every possible dependence error.

### H4 — group-leakage vulnerability (secondary)

- **Question:** In repeated-subject challenge cells, does ignoring the known subject
  grouping produce optimistic performance relative to group-aware CV?
- **Unit:** independently generated dataset; paired methods share the same generated
  data and outer assignment basis.
- **Estimands:** naive-minus-group-aware primary-metric gap, warning rate when a group
  field is absent or invalid, and their joint reporting completeness.
- **Decision rule:** jointly report both outputs required by
  `diagnostic_interpretation.group_leakage`; the gap is not converted into a universal
  detector threshold.
- **Boundary:** the measured gap pertains to the simulated grouping mechanism. A
  small gap is not proof that rows in an arbitrary real dataset are independent.

## Primary real-cohort hypothesis

### H5 — externally defined endpoint prediction (primary)

- **Question:** Does the frozen omicau workflow provide reproducible out-of-fold
  prediction for the externally defined primary endpoints under group-aware nested
  resampling?
- **Unit:** subject within cohort; cohorts are not treated as exchangeable replicates
  unless a model explicitly says so.
- **Estimand:** cohort-specific AUROC for each externally defined binary endpoint, as
  registered under `scientific_scope.primary_claims`; secondary metrics include AUPRC
  and calibration summaries.
- **Uncertainty:** subject-level paired bootstrap jointly across all outer repeats.
- **Decision rule:** report descriptive estimates and intervals separately by cohort;
  no single favorable cohort establishes general superiority.
- **Boundary:** TCGA-BRCA PAM50 is excluded from H5 because it is a secondary positive
  control, not a primary externally defined endpoint.

## Secondary hypotheses

### H6 — selected-cell external method comparison

- **Question:** On `simulation.external_comparator_subset`, what are the paired differences in
  predictive performance, failure rate, wall time and peak memory between omicau and
  each eligible external method?
- **Unit:** independent simulation dataset.
- **Estimands:** paired metric difference and resource ratio by method and selected
  cell; failures remain in the denominator and are separately classified.
- **Interpretation:** `audit_thresholds.USEFUL_MARGIN` supports
  only a descriptive statement of practical similarity. This is not a formal
  noninferiority test and no noninferiority margin is asserted.
- **Scope:** no external-method claim is made outside the selected simulation cells.

### H7 — execution integrity

- **Question:** Do interruption, resume and clean rerun produce complete,
  provenance-matched results without duplicate work or altered summaries?
- **Unit:** each prespecified clean-run, interruption/resume, cache and re-aggregation
  execution test.
- **Estimands:** artifact completeness, checksum identity where determinism is
  required, prediction tolerance where floating-point identity is not required, and
  duplicated-task count.
- **Decision rule:** all applicable checks under `freeze_gates` must pass.

## Positive control and exploratory analyses

### PC1 — TCGA-BRCA PAM50

TCGA-BRCA PAM50 evaluates whether the workflow can recover a known, strong
transcriptome-linked signal and is summarized by four-class macro-F1 after the
prespecified exclusions. It is reported as a secondary positive control. Its
performance cannot support H5, a broad real-world generalization claim, or a claim
that multi-omics fusion is necessary.

### E1 — single-case-study XAI

Permutation importance is run only for `xai.case_study_dataset_selection`. Outputs assess stability
across held-out predictions and generate hypotheses about influential features.
They are descriptive, conditional on the fitted model and non-causal. There is no
multiple-cohort XAI claim.

### E2 — Bayesian hierarchical summary

If run as a secondary or post hoc analysis, a Bayesian hierarchical model may
summarize method differences across eligible datasets using an explicit
practical-equivalence region. It remains descriptive
([Corani et al., 2017](REFERENCES.md#corani-2017)). There is no primary Bayesian
AUROC interval-coverage hypothesis.

## Explicitly absent hypotheses

The protocol does not test primary frequentist or Bayesian interval coverage; does
not claim that risk flags establish causation; does not claim that null controls
detect arbitrary leakage; does not claim formal noninferiority; and does not treat CV
folds or repeats as independent sample units.
