# Protocol-layer findings requiring a decision before definitive execution

Working analysis, not part of the published v1.0.0 record. Every item here is a
disagreement between the registered protocol and what the registered design can
actually measure. None can be fixed in the harness without changing what the
protocol says, so none has been changed. Code-level defects found in the same
review were fixed and are not listed here.

`DEVIATION_POLICY.md` is explicit that a frozen record is immutable and that a
deviation produces a new Zenodo version referencing the previous one. Acting on
any item below is therefore an external publication step and needs authorization.

## 1. The severe missingness condition cannot be positive for one of its two outputs

`SIMULATION_DESIGN.yaml#missingness_design` registers two outputs,
`target_associated_missingness_warning` and `batch_associated_missingness_warning`,
with `severity: severe` as the positive condition.

The family's base scenario is S5 (`benchmark_protocol.yaml:82`). In S5,
`protein_like` is the scenario's `independent` modality — it expresses no outcome
factor (`generate.py:401-403`). The generator applies the registered missingness
mechanisms to `protein_like` only (`generate.py:491-493`).

The consequence is structural. The diagnostic tests missingness rate against the
target and against batch. Measured over the registered `severe` cells at n=500,
10 replicates per mechanism, using per-subject missing fraction in `protein_like`
and a tie-corrected AUROC:

| mechanism | added rate | AUROC(missingness, y) | AUROC(missingness, top batch) |
| --- | --- | --- | --- |
| MCAR | 0.400 | 0.489 | 0.510 |
| MAR | 0.401 | 0.505 | 0.999 |
| MNAR | 0.400 | 0.504 | 0.500 |
| batch_associated | 0.103 | 0.501 | 1.000 |
| whole_modality_absence | 0.398 | 0.501 | 0.504 |

No mechanism produces a target-associated condition: every value in the third
column is at chance. The registered sensitivity for
`target_associated_missingness_warning` is therefore bounded at zero by the
generator rather than measured on the tool, and pooled sensitivity across the five
mechanisms is bounded the same way. The batch output does have a positive
condition, but in only two of the five mechanisms.

The fourth column corrects an earlier prediction in this document: MNAR was
expected to trip the batch test incidentally, and it does not (0.500). MNAR
depends on `protein_like` values, which under S5 carry neither outcome nor batch
structure.

Related, same location: `batch_associated` masks only the highest-numbered batch,
giving a measured added rate of 0.103 against ~0.400 for the other four, so
severity is not matched across mechanisms — the `severe` label denotes a
four-fold weaker treatment there. And because a zero additional rate returns
immediately, the five `clean__*` conditions apply one identical null treatment;
they draw different data from different seeds, so they are 500 independent null
units, but a per-mechanism breakdown of clean specificity partitions them
arbitrarily.

Measurement note: `generate.bayes_auroc` breaks ties by argsort order rather than
averaging them. On the heavily tied missing-fraction vector it reported 0.337 for
`batch_associated` where the tie-corrected value is 0.501. Production use is
unaffected — the function is only ever applied to a continuous linear predictor,
where ties have measure zero — but it must not be reused on a discrete statistic.

Options: (a) amend the base scenario so missingness lands on a signal-bearing
modality; (b) amend the registered output set for this family; (c) run as
registered and report the bound explicitly as a property of the design. Option (c)
needs no amendment but must be stated wherever the missingness sensitivity is
reported.

## 1b. S4 labels a modality `batch_confounded` where batch is independent of the outcome

Measured, not inferred. `generate_unit` sets the batch-outcome association only
for the `batch_risk_flags` family; every other family leaves
`batch_outcome_strength` at zero. `scenario_plan("S4")` nonetheless assigns
`protein_like` the ground-truth role `batch_confounded` in every S4 unit, because
that role is a property of the scenario rather than of the severity.

Observed Cramér's V between batch and outcome, n=500, replicate 0:

| unit | batch-outcome strength | Cramér's V | ground-truth role for `protein_like` |
| --- | --- | --- | --- |
| `role_recovery` S4 | 0.0 | 0.098 | `batch_confounded` |
| `batch_risk_flags` S4 clean | 0.0 | 0.045 | `batch_confounded` |
| `batch_risk_flags` S4 severe | 0.8 | 0.855 | `batch_confounded` |

In the first two rows `protein_like` carries per-batch offsets but batch is
independent of the outcome, so there is nothing to confound. omicau computes
`batch_confounded = batch_structured AND confounded_globally` and correctly
reports no confounding — and is scored as wrong against the label.

This reaches a **primary** claim. `role_recovery` registers S4 across three sample
sizes at 100 replicates each: 300 units in which one of three modalities has a
ground-truth role the correct answer cannot match. A further 300 `batch_risk_flags`
clean units are affected in the same way. The batch risk-flag specificity endpoint
itself is unaffected — the clean cell is correctly a negative for the *flag*; only
the *role* label is wrong.

Options: (a) make the role severity-dependent, labelling `protein_like`
`control_like` where batch is not outcome-associated; (b) amend the role vocabulary
so `batch_confounded` denotes batch structure rather than batch-outcome
confounding; (c) run as registered and report a known floor on S4 role recovery.
Only (c) requires no amendment, and it must be stated wherever S4 role recovery is
reported.

## 2. S6's registered roles contradict its own generator

`SIMULATION_DESIGN.yaml:70` describes S6 as
`multiplicative_interaction_without_marginal_main_effects`, and then labels both
`rna_like` and `protein_like` `predictive_additive` (`:72-74`).

With `y ~ Bernoulli(sigmoid(s * f0 * f1))` and `f1` symmetric about zero,
`sigmoid(u) + sigmoid(-u) = 1` gives `E[y | f0] = 1/2` for every `f0`. Since `y`
is binary, that constant conditional mean makes `y` exactly independent of `f0`,
and likewise of `f1`. Each modality alone therefore carries zero information;
only the pair carries any.

Measured at n=500, best univariate AUROC over each modality's features (the most
generous marginal detector available, and inflated by taking a maximum over ~1000
features): `rna_like` 0.571, `protein_like` 0.566, `methyl_like` 0.587. The
registered control modality scores highest of the three, so the two signal-bearing
modalities are marginally indistinguishable from a control. Joint Bayes AUROC is
0.716, so the signal is present — just not marginally.

Running the audit on that unit confirms the consequence directly. All three
modalities return `no detectable signal (control-like)`; role accuracy is 1/3 and
macro-F1 is 0.25, with the only "correct" call being `methyl_like`, the true
control. Every path in `_verdict` to a non-control verdict requires
`standalone_useful` (`utility.py:385-410`), so the significant-gain route cannot
rescue the label either. omicau describes the data correctly and is scored wrong
against the registered labels.

S6 sits outside `role_recovery`, so no primary claim is affected, but role
metrics are still written for S6 rows and would enter any table that aggregates
audit rows by scenario.

Do not add main effects to the generator: that would destroy the scenario's
purpose. The fix is to the labels or to the role vocabulary.

## 3. S6's difficulty calibration is additive; its outcome is multiplicative

`calibrate_scale` bisects on `f @ (weights * scale)`, a sum, and caches on
`(number of signal weights, band mean, seed)`. S6 then applies the result as
`scale * f0 * f1`. The measured Bayes AUROC is **0.716** against a registered band
of `[0.72, 0.82]` — 0.004 below the lower bound, inside the development gate's
±0.05 tolerance but outside the band as registered.

Small, and now measurable: the generator-integrity gate could not run at all
before this session (see below), so the value had never been observed. Either
calibrate on the scenario's own linear predictor or register the exemption.

## 4. `early_concat_hist_gb` is not a registered comparator

`COMPARATOR_MANIFEST.yaml#comparators` registers `nested_best_single`,
`early_elastic_net`, `random_forest_early_fusion`, `fully_nested_stacking`,
`DIABLO`, `additional_supervised_eligibility_slot` and `MOFA_plus`. The harness
also fits a histogram gradient-boosting early-fusion arm, which appears nowhere in
the manifest.

It is retained and labelled in `run.predictive_methods` as exploratory, since
`comparison_rules.no_method_dropped_after_results_seen` makes dropping it after
results are visible impossible and no results exist yet. It must not enter a
primary ranking without either a deviation record adding it or its removal now.

`nested_best_single` was registered as a primary comparator but had no
implementation; it is now implemented exactly as the manifest's `procedure` field
states, with modality selection on inner-CV AUROC and
`oracle_outer_test_selection: false`.

## 5. The control baseline differs from the registered one

`SIMULATION_DESIGN.yaml:129,131` register
`target_permutation_within_outer_training_fold` and
`permuted_feature_rows_within_modality_and_outer_training_fold`. The package
permutes the target globally before cross-validation
(`omicau/models/classical.py:205`).

Global permutation is the more common construction and is arguably the stronger
control, but it is not the registered one. A deviation record naming the
substitution would resolve it.

## 6. The control leakage gate is a decision rule, not a 5% test

The working tree changes the gate to require **both** a bootstrap CI lower bound
above chance **and** a point estimate above `CONTROL_MARGIN`, with the interval
Bonferroni-corrected across the control family. Two mechanical consequences:

- The AND strictly shrinks the alarm region: specificity rises, sensitivity falls.
  For a leakage detector a miss is the costlier error, and it is the direction
  that flatters the clean-condition specificity endpoint. State this wherever the
  alarm's operating characteristics are reported.
- `bootstrap_ci` takes two-sided percentiles at `alpha/2`, while the gate compares
  only the lower bound. At `alpha = 0.05/3` the per-control one-sided level is
  0.00833 and the family-wise one-sided level is 0.025, not the 0.05 that
  `CONTROL_FAMILY_ALPHA` advertises. Either pass `2 * CONTROL_FAMILY_ALPHA / k` or
  document the gate as one-sided at 0.025 family-wise.

The combination rule is not a numeric constant, so `check_thresholds.py` cannot
see a change to it. The freeze gate `audit_thresholds_match_imported_code_constants`
covers the eight mirrored constants and nothing else.

## 7. The generator-integrity gate could not run

Not a protocol issue — recorded here because `STATISTICAL_ANALYSIS_PLAN.md`
defines the analysis population as datasets that pass it. `generate.py --validate`
requested S6 under `role_recovery`, a combination with no archived seed, so it
raised before executing a single check. Fixed: each scenario is now generated
under the family that registers it, and the batch sweep uses only registered
severities at their own registered sample sizes. The gate passes.
