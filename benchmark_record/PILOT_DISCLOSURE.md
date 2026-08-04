# Pilot disclosure and independence statement

## What happened before this protocol

Before benchmark protocol v1.0.0 was drafted, the development program ran an
exploratory pilot covering **1,030 datasets**. That work was used to debug the
harness, inspect failure modes, estimate throughput and refine the audit. It was not
prospectively registered and is therefore pilot evidence only.

Audit thresholds were revised after pilot results had been seen. This is a
post-result threshold revision. The revised values are prospective only for this new
record: they are defined in `audit_thresholds` in `benchmark_protocol.yaml`, checked
against `omicau.interpretation.utility` before freeze, and are not validated by the
pilot data that motivated the revision.

Two earlier Zenodo records were created and subsequently deleted. This v1.0.0 record
is a brand-new, independent record, not a new version of either deleted record. Their
identifiers are deliberately omitted as required by
`independence.historical_identifiers_must_not_appear`; neither record can be cited as
the prospective registration for this benchmark.

## Complete exclusion from definitive evidence

The exclusions under `independence` are absolute:

- every run started before the successful v1.0.0 freeze is pilot-only;
- no pilot score, interval, figure, table, ranking or narrative conclusion may enter
  a definitive result;
- no pilot output may select a dataset, endpoint, comparator, simulation cell,
  hyperparameter, practical tolerance or analysis method;
- no pilot subject assignment or generated dataset may be reused; and
- no seed used or reserved in the pilot may be used in a definitive generator,
  split, model-initialization or bootstrap stream.

Pilot throughput may justify the execution profile, including the six-worker by
four-thread schedule, because that is a resource-planning decision rather than a
scientific result. The definitive task counts, seeds and outputs are generated anew
from the archived v1.0.0 configuration.

## Zero-overlap seed audit

Before freeze, the complete definitive registry generated under
`simulation.seed_generation` is compared with all known pilot seed inventories. The
audit requirement and failure action are defined under `independence.seed_audit`.
The archived output is written to the path in
`independence.seed_audit.archived_output` and must show the overlap count required by
`independence.seed_audit.required_overlap_count`.

The compared pilot inventory includes generator, fold-assignment, model,
hyperparameter-search, control and bootstrap seeds where recoverable. If a pilot
range cannot be reconstructed exactly, its entire plausible namespace is excluded.
A missing inventory is not interpreted as zero overlap. Any overlap blocks freeze;
the definitive namespace must be re-keyed and regenerated before any definitive run.

## Reporting language

Permitted:

> An exploratory 1,030-dataset pilot informed software debugging, threshold revision
> and compute planning. Because thresholds were revised after pilot results were
> observed, all pilot outputs and seeds were excluded. Definitive analyses were run
> only after a new independent v1.0.0 protocol was archived and a zero-overlap seed
> audit passed.

Prohibited:

- describing any pilot result as confirmatory, definitive, replicated or
  prospectively validated;
- combining pilot and definitive estimates;
- choosing definitive cells or comparators because of pilot performance;
- calling a post-result threshold revision prespecified for the pilot; or
- implying that the deleted records prospectively registered this benchmark.

This disclosure accompanies the protocol archive, manuscript Methods and any public
benchmark release.
