# Deviation register

**Protocol:** omicau benchmark v1.0.0  
**Current status:** no deviations from this protocol have been recorded.

## Scope

This file is the human-readable index of schema-valid records stored under
`benchmarks/deviations/`. It does not replace the procedure in
[DEVIATION_POLICY.md](DEVIATION_POLICY.md). After freeze, every difference between
the archived protocol and the executed definitive analysis is recorded before the
affected analysis is rerun.

The following are deviations when they affect definitive work: changing a dataset,
endpoint, comparator, split, seed, generator, threshold, metric, tuning budget,
resource limit, exclusion, preprocessing step, multiplicity family or claim;
changing code or environment in a way that changes a result; adding an analysis after
results are seen; or omitting a failed task.

## Register

| Deviation ID | Date | Protocol key or section | Before/after affected results seen | Affected runs | Disposition | Public report location |
| --- | --- | --- | --- | --- | --- | --- |
| None | — | — | — | — | No deviation recorded | — |

The placeholder row is removed when the first deviation record is added. Counts and
lists in the publication are derived from the machine-readable records, not copied
into this file by hand.

## Pilot history is not a v1.0.0 deviation

The exploratory 1,030-dataset pilot, post-result threshold revision and deletion of
two earlier Zenodo records occurred before this new independent protocol. They are
fully disclosed in [PILOT_DISCLOSURE.md](PILOT_DISCLOSURE.md) and excluded under
`independence`; they are not deviations from a protocol that did not yet exist.

Reusing any pilot output, result, split or seed in definitive v1.0.0 work would,
however, violate the protocol and block freeze or require a recorded deviation and a
new prospective protocol before rerun. The zero-overlap audit under
`independence.seed_audit` is mandatory.

## Reporting rule

Every recorded deviation is reproduced in the supplementary release and summarized
where it affects interpretation. A post hoc analysis is labelled post hoc in every
table, figure and sentence. A frozen record is never silently edited, and a bug fix
that changes definitive results invalidates the affected runs until they are rerun
under the documented disposition.
