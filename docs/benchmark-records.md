# Benchmark records

## Current evaluation

Use these versioned resources for the executed Omicau 0.6.0 evaluation:

| Resource | Role |
| --- | --- |
| [Results v3.0.0](https://doi.org/10.5281/zenodo.23444757) | Aggregate results, unavailable dispositions, environment specifications and numbered reproduction scripts. |
| [Protocol v6.3.0](https://doi.org/10.5281/zenodo.23444777) | Executed-study consolidation, native configurations and numbered deviations. This consolidation is post-execution, not a new preregistration. |
| [Software v0.6.0](https://github.com/tunabirgun/omicau/releases/tag/v0.6.0) | Installable software and original release validation bundle. That bundle predates the expanded comparisons in the current results deposit. |

Start with the downloaded archive's README and `protocol/EXECUTED_STUDY.md`. Earlier design documents preserve their freeze-time wording; the executed-study record and numbered deviations establish the final scope. The [evaluation guide](https://tunabirgun.github.io/omicau/reproducibility.html) explains the numerical coverage, comparator configurations and conditional interpretation.

Only complete registered series contribute pooled comparisons. Unavailable cases remain explicit. The public archives contain aggregates and scientific scripts, not molecular matrices, individual outcomes, identifiers or prediction vectors. Prediction-level reproduction requires obtaining the documented source data under their access terms and generating records locally.

## Repository history

The repository retains a historical protocol tree under `benchmark_record/` and a legacy execution harness under `benchmarks/`. Their version labels, schedules, threshold rules and draft or pending metadata belong to those earlier records. The repository tree is not the current Zenodo archive or an execution plan for the current study.

In particular, the historical `benchmark_record/CITATION.cff` and pre-upload metadata must not be used as citation metadata for the current evaluation. Use the versioned DOI and citation supplied by the relevant Zenodo record. Each record also provides its version history.

Historical material is preserved for traceability. Reproduction of the current evaluation starts from the deposited scripts and environment specifications above, rather than the legacy harness or the unpinned repository environment. The ordinary application quickstart does not invoke the archived study scripts.
