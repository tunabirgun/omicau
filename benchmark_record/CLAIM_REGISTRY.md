# Claim registry

**Protocol:** omicau benchmark v1.0.0. Claims are bounded by the estimands in
`benchmark_protocol.yaml`; no unregistered primary claim may be introduced after
freeze.

| ID | Status | Evidence and estimand | Permitted conclusion | Prohibited extension |
| --- | --- | --- | --- | --- |
| C1 role recovery | Primary | `scientific_scope.primary_claims[id=role_recovery]`; independent simulations from `simulation.families.role_recovery` | Report cell-specific recovery of known simulated modality roles with uncertainty. | “The ledger identifies true biological utility in all datasets.” |
| C2 predictive discrimination | Primary | `scientific_scope.primary_claims[id=predictive_discrimination]`; paired binary AUROC differences in `simulation.families.predictive_performance` | Report paired discrimination differences for the frozen simulation cells. | General superiority outside the generators or a claim based on selected favorable cells. |
| C3 real-cohort generalization | Primary | `scientific_scope.primary_claims[id=real_cohort_generalization]`; subject-level out-of-fold AUROC within each eligible cohort in `real_cohorts.primary_candidate_ids` | Report performance separately for each externally defined primary endpoint under repeated group-aware CV. | Clinical utility, cross-disease generalization, or treating cohorts/folds/repeats as independent replicates. |
| C4 external comparators | Secondary | Eligible methods on `simulation.external_comparator_subset` only | Describe paired performance, failure and resource differences on the selected simulation subset. | External-method rankings on untested cells or real cohorts; formal noninferiority. |
| C5 risk flags | Secondary | `simulation.families.batch_risk_flags` and `simulation.families.missingness_risk_flags` | Report sensitivity and false-warning rates for the implemented risk conditions. | Causal proof of batch effects, confounding or missingness mechanism. |
| C6 group leakage | Secondary | `simulation.families.group_leakage`; grouping warning plus naive-minus-group-aware AUROC gap | Quantify optimism from ignoring the known simulated subject grouping. | Inferring group leakage from null controls or claiming all dependence has been excluded. |
| C7 null controls | Secondary sanity check | `simulation.families.null_control_specificity` and `diagnostic_interpretation.null_controls` | “No sanity-check failure detected” or “sanity check failed” for the tested pipeline. | “No leakage exists” or “the control detects arbitrary leakage.” |
| C8 nonlinear and semi-synthetic behavior | Secondary | `simulation.families.nonlinear_secondary` and `simulation.families.semi_synthetic_robustness` | Describe behavior under the named generators and perturbations. | Treating semi-synthetic outcomes as clinical validation. |
| C9 computational resources | Secondary | observed fields under `tuning.resource_budget` | Report wall time, CPU, peak memory, configurations and failures under the frozen hardware/software profile. | Universal speed, scalability or deployability claims. |
| C10 TCGA-BRCA PAM50 | Secondary positive control | `real_cohorts.secondary_positive_control_ids`; four-class macro-F1 after prespecified exclusions | Report recovery of a known transcriptome-linked positive-control signal. | Primary predictive adequacy, necessity of fusion, or independent clinical endpoint validity. |
| C11 XAI case study | Exploratory | the one case selected by `xai.case_study_dataset_selection`; held-out permutation importance | Describe model-conditional importance and stability as hypothesis-generating. | Feature discovery, mechanism, causality or multi-cohort replication. |
| C12 reproducibility | Secondary | provenance, checksum, interruption/resume and output-completeness checks | State which frozen artifacts reproduced under the declared tolerance. | Reproducibility outside the archived environment or for untested platforms. |

## Statistical language rules

- Wilson intervals and Monte Carlo SE summarize rates over independent simulation
  datasets. The number of CV folds is never used as the inferential sample size.
- Real-cohort intervals and method differences use subject-level paired bootstrap
  draws applied jointly across repeats. Folds and repeats are correlated.
- Holm adjustment is applied within each prespecified claim family as required by
  `statistical_analysis.multiplicity`.
- `audit_thresholds.USEFUL_MARGIN` may be used as a descriptive practical tolerance
  for task-scale score differences where prespecified. “Within the practical
  tolerance” is permitted; “noninferior,” “equivalent” and “proved the same” are not.
- The primary four-class positive-control summary is macro-F1. Primary real binary
  endpoints remain separate and use their registered binary metrics; neither may be
  substituted after results are seen.
- There is no primary interval-coverage or Bayes-AUROC-coverage claim, as fixed by
  `statistical_analysis.primary_interval_coverage_analysis` and
  `scientific_scope.excluded_claims`.

## Mandatory caveats

Every public summary must state that batch and missingness outputs are risk flags,
null controls are pipeline sanity checks, group leakage is assessed separately, and
TCGA-BRCA PAM50 is a secondary positive control. Every external comparison must name
the prespecified simulation subset. Every XAI output must be labelled descriptive and
non-causal.

## Pilot boundary

No claim may use the 1,030-dataset pilot, either deleted earlier Zenodo record, or any
pilot output or seed. [PILOT_DISCLOSURE.md](PILOT_DISCLOSURE.md) supplies the required
wording. Pilot-informed threshold changes become prospective only after the new
v1.0.0 freeze.
