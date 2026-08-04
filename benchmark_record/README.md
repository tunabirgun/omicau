# omicau benchmark protocol v1.0.0

This is the prospective record for a new, independent, compute-efficient benchmark
of omicau. It is a protocol, not a results deposit. All runs made before this record
is frozen and archived are pilot analyses only; their outputs and seeds are excluded
from definitive analyses and cannot support confirmatory claims.

## Scope at a glance

- Simulated datasets are independent experimental units. Each receives one frozen
  five-fold cross-validation partition.
- Real cohorts use five-fold, three-repeat group-aware outer cross-validation, with
  three-fold inner cross-validation for tuning.
- External-method comparisons are limited to the prespecified simulation subset.
- Explainability analysis is disabled except for one prespecified case study.
- Interval coverage is not a primary claim.
- Null controls are pipeline sanity checks, not generic leakage detectors.
- Group leakage is evaluated through the grouping warning and the performance gap
  between naive and group-aware resampling.
- Batch and missingness findings are risk flags, not evidence of causation.
- TCGA-BRCA PAM50 is a secondary positive control. Every primary real-cohort endpoint
  is defined externally to omicau.

The machine-readable source of truth is `benchmark_protocol.yaml`. Reader-facing
documents refer to its keys rather than maintaining separate numeric copies.

## Record integrity

`RECORD_FREEZE.json` records the prospective freeze state, source revision,
threshold check, and the audited zero overlap between definitive and retained pilot
seeds. `RECORD_MANIFEST.sha256` gives a SHA-256 digest for every other archive member.

This archive contains reader-facing protocol material only. It excludes packaging
tools, machine-local paths and environment captures, results, splits, data, logs,
and pilot material. A `PENDING` DOI means this is the pre-upload package; the newly
reserved Zenodo DOI can be inserted without relating the record to any prior deposit.

## Interpretation boundary

This record prespecifies evaluation. It does not claim that omicau is superior, that
pilot behavior will reproduce, or that an audit warning establishes a causal
mechanism. Definitive claims depend only on analyses run after this version is frozen
and archived, using the frozen design and non-overlapping seeds.
