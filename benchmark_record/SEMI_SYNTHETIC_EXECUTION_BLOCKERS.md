# Semi-synthetic robustness: why the family cannot run yet

Working analysis, not part of the published v1.0.0 record. It states exactly what
stops `simulation.families.semi_synthetic_robustness` from executing, separates the
two causes, and says which of them requires a protocol amendment.

## What the family is

`benchmark_protocol.yaml#simulation.families.semi_synthetic_robustness` registers
2 cohort structures x 3 perturbations x 30 replicates = 180 independent units, with
no `sample_sizes` key: the sample size is a property of the template, not of the
protocol. The family carries a secondary claim (`C8`, CLAIM_REGISTRY.md) and is
explicitly not eligible for a primary claim.

`SIMULATION_DESIGN.yaml#semi_synthetic_design` selects each structure's template
from a real cohort, with a fallback:

- `acute_infection_like` <- first eligible COVID_ICU harmonized predictor matrix
- `solid_tumour_like` <- first eligible of CPTAC_UCEC_MSIH_VS_CNVL, then TCGA_BRCA_ER
- either, if ineligible <- "prespecified synthetic template matching manifest
  dimension and missingness targets"

## Blocker A: the template descriptor does not exist

Both branches, including the synthetic fallback, are defined by reference to
*manifest dimension and missingness targets*. Those quantities are the subject
count, per-modality feature counts, marginal distributions, within- and
cross-modality correlation, and missingness mask geometry of a harmonized cohort.

`DATASET_MANIFEST.yaml` states where they come from and, deliberately, that they
are not stored in the record:

    aggregate_subject_counts: derived_after_harmonization_not_stored_here

No such quantities appear anywhere else in the published record. Grepping
`COMPUTE_PLAN.md`, `EXPECTED_OUTPUTS.md`, `HYPOTHESES.md`,
`STATISTICAL_ANALYSIS_PLAN.md`, `RESAMPLING_AND_SPLIT_SPECIFICATION.md` and
`benchmark_protocol.yaml` returns no subject count, feature count or missingness
target for either structure. `benchmarks/datasets/` is empty; no cohort has been
acquired or harmonized.

**This is an input dependency, not a protocol gap.** The fallback is fully
specified relative to an artifact the protocol expects to exist by this point;
that artifact simply has not been produced. Inventing a dimension would substitute
an unregistered parameter for a registered derivation, which
`benchmark_protocol.yaml#machine_summary` and `SIMULATION_DESIGN.yaml#machine_summary`
both forbid.

The contract for the missing artifact is now explicit:
`schemas/semi-synthetic-template.schema.json`. One descriptor per structure, at
`benchmarks/datasets/manifests/semi_synthetic_template_<structure>.json`, resolves
this blocker. `tools/readiness.py` checks for them and blocks definitive readiness
while they are absent.

Resolution path: acquire and outcome-blind-harmonize COVID_ICU and the solid-tumour
candidate, derive both descriptors from the harmonized predictor matrices, and
record the harmonization manifest checksum in each descriptor's `origin` block.
Blocker A then clears without any change to the protocol.

## Blocker B: the perturbation magnitudes are genuinely unregistered

`SIMULATION_DESIGN.yaml` names the three operations and nothing more:

| perturbation | registered operation | what is missing |
| --- | --- | --- |
| `covariate_shift` | shift one latent non-outcome factor between training and test groups | which factor, and by how much |
| `blockwise_missingness` | remove prespecified feature blocks with group-dependent probability | how many blocks, and both group probabilities |
| `batch_prevalence_shift` | alter batch prevalence between training and test without changing the conditional outcome model | the shifted prevalence vector |

All three are defined relative to "training and test groups", which do not exist
until the partition is frozen. `simulation.split_reuse` requires
`one_frozen_partition_shared_by_all_methods_and_arms_for_that_unit`, so the only
reading consistent with the rest of the protocol is that the perturbation is
applied to the frozen outer test folds after the split is written. That coupling
rule is also unstated.

**This is a protocol gap and it does require an amendment.** Four scientific
parameters and one procedural rule would have to be prespecified. They can still
be chosen honestly, because no semi-synthetic result exists to choose them from,
but `DEVIATION_POLICY.md` is explicit that a frozen record is immutable and that a
deviation "produces a new Zenodo version of the record, referencing the previous
version, with the deviation log attached".

That is an external publication action. It is not taken here, and no deviation
record has been filed.

## Draft deviation record, pending authorization

Filed only on explicit instruction, to `benchmarks/deviations/`, before any
semi-synthetic unit runs. Magnitudes below are placeholders marked `TO_BE_CHOSEN`;
they are not proposed values and must be fixed before the record is written.

```json
{
  "deviation_id": "DEV-001-semi-synthetic-perturbation-magnitudes",
  "date": "TO_BE_SET_ON_FILING",
  "protocol_version": "1.0.0",
  "protocol_section": "SIMULATION_DESIGN.yaml#semi_synthetic_design.perturbations",
  "specified": "Three perturbation operations are named without magnitudes, and without a rule coupling 'training and test groups' to the frozen outer partition.",
  "implemented": "TO_BE_CHOSEN: latent factor index and shift in standard deviations; number of feature blocks removed and both group-dependent probabilities; the held-out batch prevalence vector; and the coupling rule 'perturbation applied to the frozen outer test folds'.",
  "reason": "The registered operations are not executable as written. The family is a registered secondary claim and cannot be silently omitted; the alternative to prespecifying these values is dropping 180 registered units.",
  "decided_by": "TO_BE_SET_ON_FILING",
  "affected_runs": ["simulation.families.semi_synthetic_robustness (180 units)"],
  "results_seen_before_change": "no",
  "rerun_plan": "No semi-synthetic unit has been executed, so nothing is invalidated. The family runs for the first time under the amended record version.",
  "timestamp_utc": "TO_BE_SET_ON_FILING"
}
```

## Effect on the rest of the benchmark

None, provided the family is skipped explicitly rather than silently. The runner
refuses to start when a semi-synthetic unit is in the work list and requires the
runnable families to be named:

```bash
python benchmarks/harness/run.py --run --profile core --family role_recovery --family batch_risk_flags
```

The task index is still written for the complete profile, so every unexecuted
semi-synthetic task stays outstanding in the completeness contract and in every
phase-gate report. The family cannot disappear by being skipped.
