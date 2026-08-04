# Resampling and split specification

**Protocol:** omicau benchmark v1.0.0. `benchmark_protocol.yaml` controls every
numeric split parameter and seed. Generated split manifests are immutable after
freeze.

## 1. Independent unit

The split unit is the subject or the coarsest prespecified grouping identifier. Every
row, aliquot, specimen, time point or assay from that unit must remain on one side of
every split. When no valid grouping identifier exists, the dataset is not silently
treated as independent: the grouping status is flagged and eligibility follows
[EXCLUSION_CRITERIA.md](EXCLUSION_CRITERIA.md).

Class labels may guide stratification but never grouping. Batch/site may guide a
prespecified blocked sensitivity analysis but never be used to leak outcome
information into split construction.

## 2. Simulation datasets

Each independently generated simulation dataset receives exactly the single outer CV
partition specified by `simulation.outer_cv`. There are no repeated outer
partitions. All methods, ablations and controls evaluated on that dataset receive the
same split manifest.

This design spends computation on independent generated datasets rather than on
repeated partitions of the same dataset. Folds are used to obtain out-of-fold
predictions and do not increase the simulation replicate count. A tuned simulation
arm performs its matched selection using outer-training data only and the budget
under `tuning`; it may not inspect the held-out fold.

## 3. Real cohorts

Real cohorts use the repeated group-aware outer design under `real_cohorts.outer_cv`
and the inner design under `real_cohorts.inner_cv`. Each outer repeat has a
deterministic seed derived from `simulation.seed_generation`; each subject appears in
one outer test fold per repeat.
All methods share each outer and inner assignment.

Classification uses group-preserving stratification where feasible. Regression and
survival preserve groups and follow the eligibility and fold-choice rules under
`real_cohorts.dataset_rules`. If exact stratification is impossible, the recorded
deterministic fold-choice rule is used; the fold count is not changed after inspecting
model results.

All tuning occurs in the inner CV of the current outer-training set. Selecting
hyperparameters on outer-test performance is prohibited. The final outer prediction
is produced once for that split after inner selection
([Varma & Simon, 2006](REFERENCES.md#varma-2006)).

## 4. Four-class and binary endpoints

The four-class endpoint uses one common set of group-aware outer assignments and is
scored by macro-F1. Each separately prespecified binary endpoint has its own eligible
subject set and therefore its own manifest. Binary endpoints are not created by
choosing favorable class combinations after the four-class results are seen.

A valid classification manifest must satisfy the class-support constraints under
`real_cohorts.dataset_rules` in every test fold and in every inner-training/validation
pair. Otherwise the declared fallback or exclusion rule is applied before modeling.

## 5. Group-leakage challenge

Only the challenge cells under `simulation.families.group_leakage` run both:

- the correct group-aware split, with all rows from a subject kept together; and
- a naive row-wise split that preserves class stratification but ignores subject
  identity.

The paired comparison uses the same generated dataset, model specification and
outer-assignment seed namespace. Results include the grouping-status warning and the
naive-minus-group-aware performance gap. Null controls are not used as a substitute
for this assessment.

The naive arm is a deliberately unsafe challenge pipeline. Its results are labelled
unsafe in every file, table, figure and caption and are never presented as a
recommended analysis.

## 6. Split generation order

1. Verify the aligned-data provenance digest and endpoint definition.
2. Resolve the independent-unit identifier and assert that each row maps to exactly
   one unit.
3. Apply only the frozen, outcome-independent eligibility and exclusion rules.
4. Load a seed from the definitive split registry; never generate one ad hoc.
5. Construct outer assignments from subject-level labels and groups.
6. Construct inner assignments separately inside each outer-training set.
7. Validate disjointness, coverage, class support and shared-method identity.
8. Write the manifest atomically and checksum it before any fit begins.

No preprocessing statistic, feature value, model result or pilot split may influence
assignment.

## 7. Seed separation and audit

Definitive generator, split, model and bootstrap seeds occupy the namespaces under
`simulation.seed_generation`. Before freeze, the seed-audit tool compares the expanded
definitive registry with every recoverable pilot seed inventory. The intersection
must be empty. It writes the compared inventories, their digests, counts and overlap
list to the required artifact under `independence.seed_audit.archived_output`.

An overlap of any size is a hard stop. Missing pilot seed provenance is also a hard
stop until a conservative namespace exclusion has been applied and documented under
`independence.seed_audit`. Pilot seeds are never recycled because their corresponding
results were favorable, unfavorable or apparently unused.

## 8. Manifest schema and validation

Each split row records protocol version, dataset digest, endpoint, repeat, outer fold,
subject/group identifier digest, class or balancing stratum, split role, split seed
identifier and parent inner split when applicable. Raw subject identifiers are not
required in the public artifact; stable salted digests may be used in public outputs.

Validation must establish:

- no group appears on both sides of a split;
- every eligible subject appears exactly once as outer test per repeat;
- inner splits contain only subjects from their outer-training set;
- all compared methods reference byte-identical assignments;
- each assignment seed is in the definitive registry and absent from the pilot
  registry; and
- regenerated manifests match their frozen checksums.

Because CV estimates are correlated, split manifests support prediction generation,
not an assumption that folds or repeats are independent
([Bates et al., 2024](REFERENCES.md#bates-2024)).
