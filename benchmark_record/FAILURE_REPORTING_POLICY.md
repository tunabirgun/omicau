# Failure reporting policy

## Principle

A method that will not install, will not run, or will not support a task is
reporting something real about that method. Failures are results and are published.

## What is recorded

Every failed run is written to `benchmarks/failures/` as one record validating
against `schemas/failed-run.schema.json`, carrying: dataset, method, task, profile,
split identifier, software version, the error class and message, the resource limits
in force, the wall-clock time to failure, how many recovery attempts were made and
what they changed, and the final disposition.

## Dispositions

| Disposition | Meaning |
| --- | --- |
| `unsupported_task` | the official implementation does not offer this task; not a defect |
| `implementation_unsuccessful` | the method supports the task in principle but could not be made to run correctly here, after documented attempts |
| `resource_exhausted` | out of memory, out of disk, or exceeded the wall-clock limit in `COMPUTE_PLAN.md` |
| `numerical_failure` | the run completed but produced a degenerate or non-finite result |
| `harness_defect` | the failure was ours; fixed, and the fix is a deviation record with the affected runs rerun |

## Reporting

- Table 5 of the report states failure counts per method and dataset with their
  dispositions.
- A method excluded from a ranking because of failures is still shown in the results
  table with its failure count, never omitted.
- `implementation_unsuccessful` is reported as a limitation of this study's use of
  the method, not as a claim about the method's quality, and the attempts made are
  described so a reader can judge.
- Recovery attempts that succeeded are also recorded, so that a run reported as
  successful on the third attempt is visibly that.

## What is not permitted

Dropping a failing method from a table; rerunning until a run succeeds and reporting
only that run; substituting a different configuration for the failing one without a
deviation record; describing a failure as "not applicable" when the method does
support the task.
