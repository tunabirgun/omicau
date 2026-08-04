# Expected definitive outputs

This inventory defines the minimum publication and audit trail. Paths containing
numeric choices are derived from `benchmark_protocol.yaml`; aggregate counts are not
hardcoded, consistent with `machine_summary.generation_rule`.

## 1. Pre-freeze record

| Artifact | Required content |
| --- | --- |
| `benchmark_record/benchmark_protocol.yaml` | canonical version, scope, parameters, thresholds, seeds and freeze gates |
| `benchmark_record/checksums/definitive_seed_registry.json` | expanded deterministic seed registry from `simulation.seed_generation` |
| `benchmark_record/checksums/seed_overlap_audit.json` | definitive and pilot inventory digests, counts, explicit overlap list and pass/fail result |
| `benchmark_record/checksums/protocol_checksums.sha256` | checksums for every archived protocol component |
| `benchmark_record/environment/` | operating system, hardware, package locks and comparator environments |
| dataset and comparator manifests | eligibility status, provenance, endpoint source, versions, licences and reasons for exclusions |
| frozen real-cohort split manifests | subject-safe outer and inner assignments with checksums |
| freeze report | result of every item under `freeze_gates`, frozen commit and new Zenodo DOI |

The freeze report must fail if a definitive result directory is non-empty, the DOI
or publication date is pending, a reference does not resolve, or the seed audit does
not satisfy `independence.seed_audit`.

## 2. Simulation outputs

| Artifact | Grain |
| --- | --- |
| `benchmarks/results/simulation/task_index.parquet` | one row per expected dataset-method-profile task, including status |
| `benchmarks/results/simulation/dataset_manifest.parquet` | one row per independent generated dataset with generator cell, seed identifiers and provenance digest |
| `benchmarks/results/simulation/predictions/` | compressed out-of-fold predictions indexed by dataset, split, arm and method |
| `benchmarks/results/simulation/metrics.parquet` | per-dataset metrics and paired differences; folds are retained as partition metadata, not independent rows for inference |
| `benchmarks/results/simulation/role_recovery.parquet` | truth, audit verdict and component agreement per modality and dataset |
| `benchmarks/results/simulation/risk_flags.parquet` | batch and missingness generator state, flag, effect size and adjusted p-value |
| `benchmarks/results/simulation/null_controls.parquet` | control type, task-specific null reference, metric and sanity-check disposition |
| `benchmarks/results/simulation/group_leakage.parquet` | grouping warning and paired naive-minus-group-aware AUROC gap |
| `benchmarks/results/simulation/external_comparators.parquet` | selected-cell paired metrics, resource use, status and failure disposition only |
| `benchmarks/results/simulation/cell_summary.csv` | Wilson intervals and Monte Carlo SE for rates; continuous summaries over independent datasets |

Every table exposes planned, successful, failed and indeterminate denominators.

## 3. Real-cohort outputs

| Artifact | Grain |
| --- | --- |
| `benchmarks/results/real/cohort_manifest.csv` | cohort, externally defined endpoint, role, group field, batch/site field, eligibility and provenance |
| `benchmarks/results/real/predictions/` | compressed subject-level outer out-of-fold predictions for each repeat and method |
| `benchmarks/results/real/metrics.csv` | cohort-specific primary and secondary metrics with subject-level bootstrap intervals |
| `benchmarks/results/real/paired_differences.csv` | subject-paired method differences jointly resampled across repeats |
| `benchmarks/results/real/four_class_positive_control.csv` | TCGA-BRCA PAM50 four-class macro-F1, explicitly labelled secondary positive control |
| `benchmarks/results/real/binary_endpoints.csv` | each externally defined binary endpoint as a separate task with AUROC, AUPRC and calibration summaries |
| `benchmarks/results/real/risk_flags.csv` | batch and missingness risk flags with non-causal wording |
| `benchmarks/results/real/xai_case_study/` | held-out permutation importance for the single prespecified case study only |

External-method results on real cohorts are not expected because
`methods.external_methods_on_real_cohorts` is false.

## 4. Execution, failures and deviations

| Artifact | Required content |
| --- | --- |
| `benchmarks/results/resources.parquet` | observed fields under `tuning.resource_budget`, plus worker/cache/checkpoint metadata |
| `benchmarks/failures/` | one schema-valid record per failed task under [FAILURE_REPORTING_POLICY.md](FAILURE_REPORTING_POLICY.md) |
| `benchmarks/deviations/` | one schema-valid record per departure under [DEVIATION_POLICY.md](DEVIATION_POLICY.md) |
| `benchmarks/results/log_index.csv` | task-to-bundled-log mapping, warning counts and bundle checksum |
| `benchmarks/results/cache_manifest.csv` | cache key, content digest, creator/reuser, validation and quarantine status |
| `benchmarks/results/reproducibility.json` | clean/resume/re-aggregation checks and tolerances |

Repeated identical warnings are counted rather than copied indefinitely. The first
occurrence remains available with context; final counts are included in the bundle
index.

## 5. Publication bundle

The release bundle contains the archived protocol, manifests, split checksums,
aggregate simulation and real-cohort tables, permitted de-identified predictions,
failure and deviation logs, environment lock, machine-readable claim registry,
figure/table source data and a bundle index with member checksums. Compression is
lossless. Raw omics matrices, direct identifiers and prohibited clinical fields are
excluded.

Required human-readable summaries include:

- a flow diagram of eligibility and completed/failed tasks;
- claim-indexed primary and secondary tables;
- rate plots with Wilson intervals and Monte Carlo SE;
- real-cohort estimates with subject-level bootstrap intervals;
- external comparisons confined to the selected simulation subset;
- compute and failure summaries; and
- the complete pilot disclosure and deviation table.

No report may use fold or repeat counts as independent sample sizes. No expected
output includes a primary interval-coverage table or a primary Bayes-AUROC-coverage
analysis.
