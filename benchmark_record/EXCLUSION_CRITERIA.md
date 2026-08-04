# Exclusion criteria

Frozen before eligibility is assessed. Every exclusion actually applied is logged
with its rule, its count and its date in `benchmarks/results/harmonized/exclusions.csv`.

## 1. Sample-level exclusions

| Rule | Applies to | Reason |
| --- | --- | --- |
| Missing outcome | all datasets | the endpoint cannot be defined for the row |
| Not present in every required modality after intersection | all datasets | the matched-multi-omics premise fails |
| Missing subject identifier | all primary datasets | group-aware splitting becomes unverifiable |
| Duplicate row with identical provenance digest | all datasets | an exact duplicate is a leakage source, not a sample |
| PAM50 Normal-like | R1 primary endpoint | the category is sensitive to tumour purity and sample composition; retained in the secondary five-class analysis |
| Intermediate or ambiguous consensus diagnosis | R2 | the binary endpoint is not defined; retained only in the secondary ordinal analysis |
| Class with fewer than `dataset_rules.min_class_size` members | all classification datasets | per-class metrics are not estimable |
| Non-primary histology outside the frozen endpoint definition | R3 | the endpoint is a two-class contrast defined before the data were inspected |

## 2. Feature-level exclusions

- Features constant across the training fold (zero variance) are dropped inside the
  fold, never dataset-wide.
- Features whose identifier collides across modalities are namespaced, not dropped.
- No feature is excluded on the basis of its association with the outcome outside a
  training fold.

## 3. Dataset-level exclusions

A candidate dataset is excluded when, after intersection, it falls below
`dataset_rules.min_samples_after_intersection`, when fewer than
`dataset_rules.min_modalities` layers survive, when no subject identifier is
recoverable, when the target is derivable from the predictor features themselves, or
when access terms prohibit the reproducible processing this protocol requires.

**Exclusion is decided on these rules alone.** A dataset is never excluded because
omicau performed poorly on it, and a dataset excluded after a definitive run has been
seen is a deviation and is reported as one.

## 4. Run-level exclusions

None. A run that fails is a failure record (`FAILURE_REPORTING_POLICY.md`), not an
exclusion. Failed runs remain in the results with their status.
