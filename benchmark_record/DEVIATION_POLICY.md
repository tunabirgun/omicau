# Deviation policy

## What counts as a deviation

Any difference between what this record specifies and what was done. That includes:
a changed parameter; a dataset, endpoint or comparator added, dropped or redefined; a
metric added or removed; a threshold altered; an analysis added after results were
seen; a bug fix that changes any definitive result; a change of environment or
hardware for runs already completed.

Fixing a defect is legitimate. Fixing it silently is not.

## Procedure

1. Write the deviation record **before** rerunning the affected analysis. It
   validates against `schemas/deviation.schema.json` and lands in
   `benchmarks/deviations/`.
2. Record: date, protocol section affected, what the protocol said, what was done
   instead, why, who decided, which runs are invalidated, and whether the change was
   made before or after the affected results were seen.
3. Rerun what the change invalidates. A partial rerun is itself recorded, with the
   reason the remainder was not rerun.
4. Every deviation appears in the report — in the Methods where it changes what
   was done, and in full in the supplementary deviation log.

## The before/after-seeing-results field is not optional

A change made before the affected results were examined and a change made after are
different kinds of evidence, and the reader is entitled to know which this was. The
field is mandatory, and "unclear" is an allowed value with an explanation — it is
better than a confident wrong answer.

## Post hoc analyses

An analysis not specified in this record may still be worth doing. It is reported as
*post hoc* in every table, figure and sentence where it appears, and it never
supports a primary claim.

## Threshold changes

If the definitive results show that the audit thresholds produce excessive false
warnings or modality-role errors, the thresholds may be revised — transparently, in the
source, with the revision recorded. The confirmatory benchmark is then **repeated under
a new prospectively archived protocol version**. Revised thresholds evaluated on the
data that motivated the revision are not confirmatory evidence, and the report does not
present them as such.

## Protocol versioning

A frozen record is immutable. A deviation does not edit it: it produces a new Zenodo
version of the record, referencing the previous version, with the deviation log
attached. The report cites the version under which each analysis was run.
