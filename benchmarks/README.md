# Legacy benchmark harness

This directory retains execution code for the historical protocol in `benchmark_record/`. It is separate from ordinary `omicau run` and browser workflows and is not the reproduction entry point for the current Omicau 0.6.0 evaluation.

For the current study, use [results v3.0.0](https://doi.org/10.5281/zenodo.23444757) and [protocol v6.3.0](https://doi.org/10.5281/zenodo.23444777). Begin with the downloaded README and `protocol/EXECUTED_STUDY.md`. The [benchmark-record guide](../docs/benchmark-records.md) distinguishes those deposits from the repository's historical material.

## What is stored here

| Path | Role |
| --- | --- |
| `harness/` | Legacy execution, artifact validation, harmonization and phase-checking code. |
| `simulations/generate.py` | Historical simulation generator source. |
| Runtime directories | Locally generated inputs, splits, runs, results, failures and logs; excluded from Git. |

Configuration and dataset-manifest directories described by earlier protocols are not a complete distributed dataset. Their absence does not imply that all prerequisites for an execution have been met. Source files under `simulations/` are tracked; generated datasets are local runtime material.

## Historical execution contracts

The legacy harness uses its matching protocol, scenario definitions, split registry and environment. The smoke, core and full profiles have different evidentiary roles; smoke outputs are development checks. Changing the installed application or adopting current thresholds does not convert an earlier run into a current-version evaluation.

For this harness, a completion marker must agree with its referenced artifacts and split manifests; the presence of result rows alone is insufficient. Interrupted staging directories and failure records are retained, and aggregates are assembled from validated dataset shards. Read the source and matching historical protocol before attempting a replay. This documentation update does not certify that the legacy harness executes against the current package.

The current Zenodo archives provide their own numbered scripts, scientific configurations, environment specifications and unavailable-case accounting. Use those materials for the reported evaluation rather than launching an old core/full profile.
