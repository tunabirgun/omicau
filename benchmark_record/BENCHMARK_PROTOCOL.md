# omicau benchmark protocol v1.0.0

**Status:** prospective draft for a new, independent Zenodo record. Definitive
analysis is prohibited until the record has passed its freeze checks and has been
archived.

## 1. Purpose and evidential boundary

This protocol evaluates whether omicau provides a reproducible, leakage-safe and
computationally practical audit of matched multi-omics prediction studies. The
benchmark is designed to support carefully bounded claims suitable for peer review,
not a universal ranking of multi-omics methods. Its design follows guidance on
prospective method benchmarking, representative data-generating mechanisms and
transparent reporting ([Brooks et al., 2024](REFERENCES.md#brooks-2024);
[Weber et al., 2019](REFERENCES.md#weber-2019);
[Morris et al., 2019](REFERENCES.md#morris-2019)).

This is a new protocol, not a continuation or version of either deleted earlier
Zenodo record. All analyses performed before this record is frozen are pilot or
development analyses. They cannot supply confirmatory results, tune the definitive
analysis, or enter the manuscript Results or Abstract. The complete history and the
mandatory seed exclusion are in [PILOT_DISCLOSURE.md](PILOT_DISCLOSURE.md).

`benchmark_protocol.yaml` is the sole source of truth for numeric design values,
thresholds, seeds, resource limits and tuning budgets. Prose cites keys rather than
copying values. A mismatch between this record, the executable configuration and the
installed package is a failed pre-flight check.

## 2. Questions and claims

The prespecified questions are:

1. Do audit verdicts recover known modality utility and known risk conditions in
   independently generated simulations?
2. Do null controls behave as pipeline sanity checks under correctly nested
   analysis?
3. Does a grouping warning, together with the naive-minus-group-aware performance
   gap, reveal vulnerability to non-independent samples?
4. Does omicau achieve useful predictive performance on externally defined real
   endpoints without preferential dataset selection?
5. On the prespecified simulation subset, how do omicau and eligible external
   methods compare in predictive performance, failure rate and compute cost?
6. Are outputs complete, provenance-linked and reproducible under interruption and
   restart?

The formal hypotheses and estimands are in [HYPOTHESES.md](HYPOTHESES.md); permitted
wording is in [CLAIM_REGISTRY.md](CLAIM_REGISTRY.md). No primary claim concerns
confidence-interval coverage, and there is no primary Bayesian AUROC-coverage claim.

## 3. Benchmark domains

### 3.1 Independent simulations

Each generated dataset is one independent experimental unit. Data-generating cells
span the sample-size, dimensionality, class-balance, signal, redundancy, batch,
missingness and repeated-subject factors declared under `simulation.families` and
`simulation_parameters`. Seeds come only from `simulation.seed_generation`; the zero-overlap
audit against all pilot seeds must pass before generation.

Simulation aims, data-generating mechanisms, estimands, methods and performance
measures follow the ADEMP structure. Recovery rates are summarized over independent
datasets within a cell. CV folds within a dataset are computational partitions, not
replicates. Accordingly, each simulation uses the single outer partition specified
by `simulation.outer_cv`, with no repeated CV. This preserves independent
simulation replication while avoiding unnecessary refits.

Null controls use corrupted targets or inputs to verify that the fitted pipeline
does not manufacture predictive signal. They are not claimed to detect every form
of leakage. In particular, group leakage is tested separately as specified in
Section 6.

### 3.2 Real cohorts

Eligibility and exclusions are fixed before outcome-dependent modeling. Primary
real endpoints must be defined by an external clinical, pathological or experimental
criterion named in `real_cohorts.primary_candidate_ids`; they cannot be derived from the same
predictor features being evaluated. Cohort provenance, endpoint source, group field,
batch/site field and exclusions are recorded before splitting.

Real-cohort evaluation uses the repeated, group-aware nested design under
`real_cohorts.outer_cv` and `real_cohorts.inner_cv`: outer splits estimate performance and inner splits perform all
tuning. The independent unit is the subject or the coarsest prespecified grouping
unit, never a row, aliquot or repeated specimen.

TCGA-BRCA PAM50 is only the secondary positive control under
`real_cohorts.secondary_positive_control_ids`. Because PAM50 is transcriptionally
defined, it cannot support the primary real-world generalization claim and is always
labelled as a positive control.

## 4. Methods

The omicau arms, ablations, baselines and tuning spaces are declared under `methods`
and `tuning`. All data-dependent operations—including imputation, scaling, filtering,
feature selection, batch-adjustment probes, latent factors, calibration, stacking,
thresholds and early stopping—are fitted using training data only. The operational
rules are in [PREPROCESSING_RULES.md](PREPROCESSING_RULES.md).

External comparisons are limited to the prespecified simulation cells under
`simulation.external_comparator_subset`. Eligibility is assessed without seeing definitive
results under [COMPARATOR_ELIGIBILITY_RULES.md](COMPARATOR_ELIGIBILITY_RULES.md).
Candidate method families include DIABLO, MOGONET and MOFA+ using their documented
roles and an explicitly recorded inductive prediction procedure
([Singh et al., 2019](REFERENCES.md#singh-2019);
[Wang et al., 2021](REFERENCES.md#wang-2021);
[Argelaguet et al., 2020](REFERENCES.md#argelaguet-2020)). A method that cannot use
the frozen splits or operate inductively is excluded from the paired comparison or
reported in a separate transductive column. It is never silently adapted.

Default and tuned arms are kept distinct. Tuning budgets are matched under
`tuning.maximum_configurations_per_method_per_outer_training_set` and
`tuning.resource_budget`; the entire selection procedure is repeated inside every outer
training set ([Varma & Simon, 2006](REFERENCES.md#varma-2006)).

## 5. Outcomes

Primary estimands are declared under `scientific_scope.primary_claims`. The
TCGA-BRCA PAM50 four-class positive control is evaluated by macro-F1 so each class
contributes equally. Prespecified binary endpoints are separate tasks with their own
AUROC, AUPRC, calibration and classification summaries; they are not post hoc
dichotomizations of the four-class result.

Simulation endpoints include correct verdict rates, false-warning rates, predictive
metrics, naive-minus-group-aware gaps, method failures, wall time and peak memory.
Batch and missingness diagnostics are risk flags. They indicate compatibility with
batch structure, outcome confounding or informative missingness; they do not prove a
causal mechanism or establish that a dataset is unusable.

Permutation-importance XAI is disabled by default under `xai.default` and is
run only for the single case study selected under `xai.case_study_dataset_selection`. Its rankings are
descriptive and non-causal.

## 6. Leakage and controls

Target shuffling, column shuffling and synthetic-noise controls are evaluated against
their task-appropriate null behavior. Passing means that these controls found no
evidence of target or pipeline signal under the tested configuration; it never means
that all possible leakage has been excluded ([Kapoor & Narayanan, 2023](REFERENCES.md#kapoor-2023)).

Group leakage is evaluated by two jointly reported outputs:

- a grouping-status warning stating whether the declared rows are demonstrably
  independent; and
- the paired difference between naive row-wise CV and group-aware CV in the
  prespecified repeated-subject challenge cells.

The gap quantifies optimism associated with ignoring the known grouping structure.
It is not inferred from null controls and is not generalized to unmeasured forms of
dependence.

## 7. Statistical analysis

The complete analysis is fixed in [STATISTICAL_ANALYSIS_PLAN.md](STATISTICAL_ANALYSIS_PLAN.md).
Folds and repeats overlap and are correlated; they are never counted as independent
observations ([Bates et al., 2024](REFERENCES.md#bates-2024)). Real-cohort uncertainty
uses a subject-level paired bootstrap that resamples subjects once per bootstrap draw
and applies that draw jointly to all repeats and compared methods. Simulation rates
receive Wilson intervals and Monte Carlo standard errors across independent datasets.

Multiplicity is controlled by Holm adjustment within the prespecified families under
`statistical_analysis.multiplicity`. The practical tolerance under
`audit_thresholds.USEFUL_MARGIN` is descriptive: results within it may be described
as practically similar, but this protocol makes no formal noninferiority claim.
Bayesian hierarchical comparisons may be reported only as secondary descriptive
analyses, with their model and prior recorded; they do not replace the primary
estimands ([Corani et al., 2017](REFERENCES.md#corani-2017)).

## 8. Efficient execution and reproducibility

The execution contract is in [COMPUTE_PLAN.md](COMPUTE_PLAN.md). Efficiency comes
from one partition per independent simulation, full external comparators only on
selected cells, XAI in one case study, provenance-keyed caches for immutable inputs
and concatenations, atomic checkpoints, resumable tasks, compressed/bundled outputs
and warning de-duplication. Parallelism uses the prespecified six-worker profile and
the per-fit limit in `tuning.resource_budget.cpu_threads_per_fit`, selected from pilot throughput and resource checks;
pilot performance estimates are not reused.

Every result carries the protocol version, frozen commit, environment, data digest,
configuration digest, split identifier, seed, method version and status. Expected
artifacts are enumerated in [EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md). Failed runs
remain visible under [FAILURE_REPORTING_POLICY.md](FAILURE_REPORTING_POLICY.md).

## 9. Freeze, deviations and interpretation

Before the first definitive fit, pre-flight must verify the new Zenodo DOI, clean
protocol state, complete manifests, exact threshold parity, empty definitive result
directories, valid frozen splits, successful seed non-overlap, sufficient resources
and passing smoke tests. No result becomes confirmatory merely because it was run
with the intended code; it must be run after the archived freeze against the frozen
assets.

After freeze, this record is immutable. Every departure is logged before rerunning
the affected analysis under [DEVIATION_POLICY.md](DEVIATION_POLICY.md) and summarized
in [DEVIATIONS.md](DEVIATIONS.md). Post hoc analyses remain labelled post hoc in every
appearance and cannot support a primary claim.

Interpretation is bounded to the prespecified generators, cohorts, endpoints,
methods, resource envelope and software versions. A risk flag is not causal proof, a
sanity-control pass is not proof that leakage is impossible, practical similarity is
not noninferiority, and predictive association is not clinical utility.
