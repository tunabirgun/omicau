# Comparator eligibility rules

Frozen before any comparator is assessed. The assessment itself is recorded in
`COMPARATOR_MANIFEST.yaml`.

## 1. The four statuses

| Status | Meaning |
| --- | --- |
| `primary_eligible` | supports the task, runs inductively on the frozen splits, and can be tuned within the matched budget |
| `secondary_eligible` | usable but with a stated limitation — for example unsupervised, or requiring a workaround that changes its intended use |
| `ineligible_for_task` | the official implementation does not offer this task; recorded, not forced |
| `implementation_unsuccessful` | supports the task in principle but could not be run correctly here after documented attempts |

## 2. What is recorded for every comparator, before it runs

Task support · supervised or unsupervised · inductive or transductive · missingness
handling · group-aware resampling compatibility · tuning requirements · official
implementation and its provenance · version as installed · licence · installation
status · known limitations.

Versions and licences come from the installed artefact, not from documentation about
it.

## 3. Fairness rules

1. **A method is never forced into a task it was not designed for.** An unsupervised
   factor model is not a classifier; it enters only through a stated protocol
   (fit on training data, project test data, classify on training-derived factors) and
   only if that protocol is implementable without transductive leakage.
2. **The official implementation is used** wherever one exists. A reimplementation is
   permitted only when the official one cannot run here, and it is then labelled a
   reimplementation everywhere it appears, with the differences described.
3. **Tuning budgets are matched** within an arm (`tuning.budget`). A comparator tuned
   over more configurations than omicau, or the reverse, invalidates the comparison
   — this is the "unequal tuning" outcome that the protocol declares unacceptable.
4. **Identical frozen splits** are used by every comparator. A method that cannot
   consume externally supplied splits is not compared in the paired analysis; it is
   reported separately with that reason.
5. **Default settings are honoured in the primary arm.** Where a method's documented
   defaults are not usable at all (DIABLO's component and feature counts), that fact
   is recorded and the method appears only in the tuned arm, with the reason stated.
6. **Failure is reported, not hidden** (`FAILURE_REPORTING_POLICY.md`). A comparator
   that will not install is a finding about deployability.
7. **No comparator is dropped after its results are seen.** Dropping one at that point
   is a deviation and is reported as one.

## 4. Transductive operation

A method that uses test-set features during fitting — jointly embedding train and
test, or building a graph over all samples — is transductive. Transductive results
are not comparable to inductive ones and are either excluded or reported in a
separate, clearly labelled column. The manifest records this per method, and the
determination is made by reading the implementation, not the paper.

## 5. Version pinning

Every comparator's environment is pinned and recorded in
`environment/comparator-environments.yaml`. A version change after freeze is a
deviation.
