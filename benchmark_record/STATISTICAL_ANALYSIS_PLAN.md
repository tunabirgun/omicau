# Statistical analysis plan

**Protocol:** omicau benchmark v1.0.0. This plan is prospective. Numeric values are
read from `benchmark_protocol.yaml`; prose names the controlling keys.

## 1. Analysis populations and units

The simulation analysis population contains every definitively generated dataset
that passes the generator-integrity checks. Each dataset is one independent unit.
Folds within a dataset are not replicates.

The real-cohort population contains every eligible subject after the prespecified
exclusions. Subject is the analysis and bootstrap unit; all rows belonging to that
subject remain together. Cohorts are summarized separately and are not treated as
exchangeable replicates.

The comparator population is restricted to `simulation.external_comparator_subset` and to
methods whose frozen eligibility status permits the task. Failed runs are retained
under their recorded disposition and are not converted to successful scores.

## 2. Prediction summaries

For every outer split, preprocessing, feature selection, model fitting, tuning,
calibration and threshold selection use only the outer-training data. Hyperparameter
selection uses `real_cohorts.inner_cv` and the objective under `tuning.selection_metric_ref`.
This nesting is required because tuning and evaluating on the same CV results is
optimistically biased ([Varma & Simon, 2006](REFERENCES.md#varma-2006)).

Out-of-fold predictions are retained at subject level. Within each real-data repeat,
each eligible subject contributes once to the pooled outer prediction vector. Metrics
are computed from pooled predictions within each repeat and then averaged across
repeats; fold metrics are retained only for quality control and descriptive fold dispersion.

The TCGA-BRCA PAM50 four-class positive control uses macro-F1. Each primary binary
endpoint is analyzed as a distinct endpoint, with AUROC registered under
`scientific_scope.primary_claims`; AUPRC and calibration are secondary. Binary
results are not inferred from or substituted for the multiclass analysis.

## 3. Dependence created by cross-validation

Training sets overlap across folds, and the same subjects recur across repeated CV.
Consequently, fold and repeat estimates are correlated and are never entered as if
they were independent observations. Naive standard errors, paired tests over folds
and confidence intervals based on fold SD are prohibited because they can materially
understate uncertainty ([Bates et al., 2024](REFERENCES.md#bates-2024)).

The split structure is therefore used to generate honest out-of-fold predictions,
not to inflate the inferential sample size.

## 4. Real-cohort uncertainty and paired comparisons

Real-cohort intervals and method differences use the paired subject-level bootstrap
at the level under `statistical_analysis.confidence_level`:

1. draw subjects with replacement within the prespecified stratification unit;
2. use the same subject multiplicities for every compared method;
3. apply that one draw jointly to each subject's out-of-fold predictions from all
   repeats;
4. recompute each repeat-specific metric and its across-repeat mean for each method; and
5. store the paired difference before constructing the interval.

This preserves pairing between methods and does not pretend that repeats are
independent. Percentile intervals are reported; degenerate resamples are redrawn,
with redraw counts reported.

For the naive-versus-group-aware challenge, the estimand is the subject-level paired
difference `metric_naive - metric_group_aware`; positive values indicate optimism
from row-wise splitting. The grouping-status warning is reported beside the gap and
is not collapsed into a single leakage label.

## 5. Simulation summaries

Within each simulation cell, binary operating characteristics—including correct
verdict, warning, false warning, control alarm and run success—are proportions over
independent datasets. Each proportion receives a Wilson score interval at
`statistical_analysis.confidence_level`. Its Monte Carlo standard error is
`sqrt(p_hat * (1 - p_hat) / R)`, where `R` is the number of successfully evaluated
independent datasets contributing to that estimand. The planned and observed `R`,
failures and indeterminate outcomes are shown together.

Continuous simulation estimands are summarized by the mean, median, empirical
quantiles and Monte Carlo SE over independent datasets. Paired method differences are formed within a
dataset before aggregation. No uncertainty calculation uses the number of CV folds.

Interval coverage is not a primary endpoint. Any interval-coverage diagnostic added
later is explicitly secondary or post hoc and cannot replace the registered claims.

## 6. Multiplicity

Hypothesis tests are grouped exactly as listed under
`statistical_analysis.multiplicity`. Raw p-values and Holm-adjusted p-values are both
reported. Holm adjustment is applied within each family and never reconstructed from
rounded values. No unregistered regrouping is permitted after results are seen.

Confidence intervals describe estimand uncertainty; they are not silently converted
into multiple-testing decisions. Descriptive tables not attached to a claim are
labelled descriptive.

## 7. Practical tolerance and method comparison

The score-scale value under `audit_thresholds.USEFUL_MARGIN` defines a descriptive
region within which observed differences may be called practically small. Reports
must show the estimate and uncertainty interval, not only whether the estimate falls
inside the region.

This tolerance is not a noninferiority margin: the design is not powered as a formal
noninferiority trial, no one-sided type-I error claim is made, and the permitted phrase
is “within the prespecified practical tolerance,” not “noninferior” or “equivalent.”

Eligible external methods are compared only on `simulation.external_comparator_subset`.
Predictive differences, wall-time ratios, peak-memory ratios and failure counts are
reported together to avoid a performance-only ranking. Dataset or method selection
after results are observed is prohibited, consistent with concerns about optimistic
method evaluation ([Buchka et al., 2021](REFERENCES.md#buchka-2021)).

## 8. Diagnostic interpretation

Null controls are pipeline sanity checks. Their alarm rates estimate how often the
tested corrupted inputs yield unexpected signal under the executed pipeline. They do
not estimate sensitivity to arbitrary leakage and cannot rule out group leakage.

Batch and missingness results are reported as risk flags with effect sizes, adjusted
p-values and sample support. Phrases implying a proven causal mechanism—such as
“batch caused the prediction” or “missingness is MNAR”—are prohibited. Group leakage
is assessed only from grouping status and the prespecified naive-versus-group-aware
gap challenge.

## 9. Missing, failed and indeterminate results

No metric is imputed for a failed method. Denominators are displayed for every rate.
An undefined metric caused by absent classes or a degenerate prediction is recorded as
indeterminate with its reason, not dropped. The primary presentation includes success,
failure and indeterminate counts.

## 10. Secondary Bayesian analysis

No Bayesian hierarchical model is part of the primary analysis. Any secondary or
post hoc model may summarize paired differences across eligible independent datasets
and a declared practical-equivalence region ([Corani et al., 2017](REFERENCES.md#corani-2017)). Its
likelihood, priors, convergence diagnostics and posterior summaries must be archived.
It is secondary, is not used to tune methods or choose datasets, and supports no
primary Bayes-AUROC coverage claim.

## 11. Reporting precision and reproducibility

Machine-readable outputs retain full precision. Tables round only for display. Every
estimate is linked to the data digest, split identifier,
method/configuration digest, seed registry and code/environment snapshot. Analysis
code reads all numeric choices from YAML and fails on missing or unknown keys.
