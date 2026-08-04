# Compute plan

**Protocol:** omicau benchmark v1.0.0. Resource values and stop limits are read from
`benchmark_protocol.yaml`; this document explains their use.

## 1. Resource envelope

The definitive local execution profile uses six concurrent process workers and the
per-fit limit under `tuning.resource_budget.cpu_threads_per_fit`, yielding the
six-worker, four-thread profile selected from pilot throughput on the declared
machine. The pilot informed scheduling feasibility only; no pilot metric, output,
threshold or seed contributes to the definitive benchmark.

Before each run, the harness records logical and physical CPU count, available RAM,
swap, free disk, operating system, accelerator state and package environments. It
refuses to start when the fit limits under `tuning.resource_budget` cannot be met.
Thread-related environment variables are set per worker before numerical libraries
load, preventing nested pools from oversubscribing the machine.

## 2. Unit of scheduling

The schedulable unit is one dataset-method-profile task with fixed data digest,
split manifest and configuration digest. Independent simulation datasets are
distributed across workers. Inner fits remain local to that task so tuning state and
temporary data are not shared across processes.

Real cohorts are scheduled conservatively because repeated nested CV increases memory
residency. External methods obey their
own measured or documented thread limits in
`environment/comparator-environments.yaml`.

## 3. Compute-efficiency decisions

- Independent simulations use one outer CV partition under
  `simulation.outer_cv`; precision comes from independent generated datasets, not
  repeated CV on the same dataset.
- Full external comparators run only on `simulation.external_comparator_subset`. Other cells
  retain omicau arms and low-cost baselines needed for the registered estimands.
- Permutation-importance XAI is disabled except for `xai.case_study_dataset_selection`.
- Tuning spaces and fit caps are fixed under `tuning`; no method receives additional
  searches because early results are promising.
- Cheap integrity checks run before expensive fits, and a failed prerequisite blocks
  dependent tasks.

These choices reduce redundant fitting while preserving independent simulation
replication, honest real-cohort uncertainty and matched comparisons.

## 4. Provenance-keyed cache

Only deterministic, outcome-independent artifacts may be reused across tasks. The
cache includes aligned-input provenance and modality concatenations. A cache key is
derived from protocol version, data-value digest, ordered sample digest, ordered and
namespaced feature digest, modality set, endpoint-independent ingestion options and
code/schema version.

Cached concatenations contain no fitted imputation, scaling, filtering, feature
selection, latent factors or model state. Those remain fold-local. A key or checksum
mismatch is a cache miss; stale entries are quarantined rather than overwritten. Each
result records whether a cache entry was created or reused and the digest it verified.

## 5. Atomic checkpoints and resume

Each task writes to a uniquely named temporary checkpoint in its target directory,
flushes and validates the payload, then replaces the final checkpoint atomically.
The completion marker is written last. A task is complete only when its expected
files, schema checks and checksums all pass.

On resume, complete matching tasks are skipped; valid partial tasks continue from the
last validated atomic unit; corrupt, mismatched or orphaned temporary files are
quarantined and recorded. Resume never changes a seed, split, method configuration or
tuning path. Failure dispositions follow [FAILURE_REPORTING_POLICY.md](FAILURE_REPORTING_POLICY.md).

## 6. Output volume and bundling

Tabular results use lossless compressed formats; raw repeated predictions are stored
once and summaries are derived from them. Small metadata and log artifacts are bundled
by run to avoid excessive file
counts. Every bundle has an index, member checksums, compression method and schema
version. A bundle is not the only copy of a manifest required to resume a task.

No raw multi-omics matrices, direct subject identifiers or prohibited clinical fields
enter a public bundle. Publication bundles contain aggregate results or permitted
pseudonymous prediction records.

## 7. Logging without warning spam

Logs are structured by run and task. Repeated warnings with the same normalized
warning class, source and message are emitted once, followed by a counter in the task
summary. The first occurrence retains full context and stack information when
available. Suppression never applies to distinct errors, failed validation, changed
inputs or final failure dispositions.

Progress reporting is rate-limited, while heartbeat records
remain sufficient to distinguish slow work from a stalled task. Console output stays
ASCII-safe; UTF-8 log files retain full text.

## 8. Monitoring and stop conditions

The harness records wall time, CPU time, peak resident memory, disk growth, retries,
cache status and exit disposition per task. It stops scheduling new work when a hard
limit under `tuning.resource_budget` is crossed and lets only explicitly safe
in-flight checkpoints finish. Out-of-memory, disk exhaustion and wall-time failures
are retained as benchmark outcomes.

Pilot throughput supports the chosen parallel profile but does not guarantee the
definitive duration. The pre-flight estimate and observed throughput are reported
with assumptions; neither is used to alter the registered simulation cells after
results become visible.

## 9. Reproducibility checks

The execution tests cover clean run, interrupted run,
resume, cache reuse and re-aggregation from stored predictions. Required comparisons
use exact checksums for deterministic artifacts and an explicitly recorded tolerance
only where platform-level floating-point identity
is not promised. The environment, random-state registry and deterministic settings
are captured for every test.
