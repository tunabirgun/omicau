# omicau — project instructions

Project rules for this repository. They extend the global defaults in
`~/.Codex/AGENTS.md` and override them where they conflict. Nothing in this
file is machine-enforced; the enforcement lives in `.gitignore`, the freeze
tooling under `benchmark_record/tools/`, and the test suite.

## What this repository is

`omicau` is a leakage-safe multi-omics audit CLI: ingestion and alignment,
value-level SHA-256 provenance, missingness-bias and batch-confounding
diagnostics, classical and neural fusion benchmarks under group-aware
cross-validation, permutation-importance attribution, a modality-utility ledger,
and a self-contained HTML dashboard. `README.md` §Methodology is the canonical
description of the statistics; do not paraphrase it from memory when writing
documentation or manuscript text — read it.

## Repository map

| Path | Tracked | Purpose |
| --- | --- | --- |
| `omicau/` | yes | the package (`data/`, `diagnostics/`, `models/`, `interpretation/`, `reporting/`, `ui/`) |
| `tests/` | yes | smoke + UI core tests |
| `docs/` | yes | GitHub Pages documentation site |
| `packaging/` | yes | conda-forge recipe material |
| `benchmark_record/` | yes | **prospective** benchmark protocol, frozen and archived to Zenodo before any definitive run |
| `benchmarks/` | source + scaffolding | benchmark harness, generators, configs and dataset metadata are tracked; runs, results, logs, splits and derived data stay local |
| `manuscript/` | no (gitignored) | private article workspace, figures, tables and the documentation compiler |
| `local/` | no (gitignored) | consolidated working material: `demo/`, `demo_vid/`, `design_assets/`, `example_output/` — see `local/README.md` |
| `tests_real/` | no (gitignored) | real-dataset test material; large, left at the root on purpose |
| `outdated/` | no | superseded files land here; nothing is deleted |

## Conventions that override the global defaults

- Never stage `manuscript/`, `demo/`, `demo_vid/`, `design_assets/`,
  `example_output/`, `tests_real/`, or benchmark runtime outputs. Under
  `benchmarks/`, only `README.md`, `configs/`, `datasets/manifests/`,
  `datasets/download_scripts/`, `harness/` and `simulations/*.py` are source.
- The documentation compiler (`manuscript/docs_generator.py`) is private author
  tooling, not a package feature. Do not re-add it to `omicau/` or the CLI.
- The Windows console is cp1252: no non-ASCII glyphs in CLI stdout. Scripts that
  print fetched text must `sys.stdout.reconfigure(encoding="utf-8")` first.
- pandas 3.0 copy-on-write: `to_numpy()` returns a read-only view — copy before
  mutating; `Float64` needs an explicit cast to numpy `float64`.

## Benchmark discipline

The publication benchmark is prospectively registered. The complete protocol is
`benchmark_record/BENCHMARK_PROTOCOL.md`; every numeric parameter has its single
source of truth in `benchmark_record/benchmark_protocol.yaml`. Prose documents
cite parameter keys rather than restating values, so a number cannot drift
between documents.

Binding rules:

1. **Development analyses and definitive analyses are different things.** Any run
   made before the protocol freeze is a development analysis: it may debug code,
   check data formats, size compute, and verify that figures render. It may never
   be reported as confirmatory evidence, and its numbers may not enter the
   manuscript's Results or Abstract. See `benchmark_record/BENCHMARK_PROTOCOL.md`
   §2.3.
2. **Nothing is frozen until `benchmark_record/tools/freeze_record.py` succeeds.**
   The script derives the commit hash, working-tree status, package versions and
   per-file checksums; it refuses to freeze while any protocol file carries a
   `STATUS: NOT DRAFTED` marker, while the Zenodo DOI field is `PENDING`, or while
   any benchmark result directory is non-empty.
3. **Definitive runs happen only after the freeze**, against the frozen splits in
   `benchmarks/splits/`, with every method receiving the identical splits.
4. **Every deviation from the frozen protocol is logged** in
   `benchmarks/deviations/` against `benchmark_record/schemas/deviation.schema.json`,
   with the reason and the date, before the affected analysis is rerun.
5. **Failed runs are retained, not dropped.** A comparator that crashes, a method
   that does not support a task, an out-of-memory kill: all are recorded under
   `benchmarks/failures/` and reported in the manuscript.
6. **No dataset, endpoint, or comparator is selected on the basis of favourable
   omicau performance.** Eligibility rules are frozen before eligibility is
   assessed.
7. **All preprocessing is fitted inside training folds only** — imputation,
   scaling, variance filtering, feature selection, batch adjustment, latent
   factors, calibration, stacking meta-features, thresholds. Any deliberately
   unsafe variant exists only as a labelled challenge pipeline in the leakage
   stress tests and is never presented as a recommended method.

## Thresholds: the code is the source of truth

The audit's decision cut-offs are defined once, in
`omicau/interpretation/utility.py`:

`USEFUL_MARGIN`, `GAIN_EPS`, `GAIN_STRONG`, `GAIN_ALPHA`, `CKA_REDUNDANT`,
`CONTROL_MARGIN`, `SUBGROUP_MIN_N`.

`omicau/reporting/reporter.py` imports the gain bands rather than restating them,
so the executive strip, the headline card and the summary flags cannot diverge.
Keep it that way: a new surface that needs a cut-off imports the constant.

The protocol mirrors these values in `benchmark_protocol.yaml:audit_thresholds`.
`benchmark_record/tools/check_thresholds.py` imports the constants from the
installed package and fails when the mirror disagrees. Run it before freezing and
before reporting any threshold in the manuscript. Never transcribe a threshold
into a document by hand.

## Manuscript

- Target venue: *Computational Biology and Chemistry* (Elsevier). Formatting
  decisions are recorded in `manuscript/format-profile.md`.
- `manuscript/manuscript.md` is canonical; the two `.docx` renditions are built
  from it by `manuscript/build_docx.py` and are never hand-edited.
- Unfilled result slots use the greppable token `[[PENDING: <slug>]]`. A draft is
  not submittable while any remains.
- Author identity, ORCID and affiliations come from the `author-profile` skill,
  copied character for character. Do not reconstruct them from memory.

## Environment

Python 3.12.10, Windows 11, torch CPU build, pandas 3.0, python-docx 1.2.0.
No pandoc on this machine; MiKTeX is available for LaTeX. `omicau` is installed
in the working environment, so `import omicau` works from tooling scripts.
