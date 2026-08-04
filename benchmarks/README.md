# benchmarks/ — execution tree

Where the benchmark specified in `../benchmark_record/` actually runs. Empty by
design: the protocol is frozen and archived **before** anything lands here, and
`benchmark_record/tools/freeze_record.py` refuses to freeze while `results/`,
`runs/`, `figures/` or `tables/` contains a file.

## Layout

```text
benchmarks/
├── configs/                     resolved run configurations, one per run        [tracked]
├── datasets/
│   ├── manifests/               per-dataset manifests + checksums               [tracked]
│   ├── download_scripts/        reproducible access scripts                     [tracked]
│   └── derived_nonrestricted/   derived matrices that may be shared             [local]
├── simulations/                 generator, seeds, generated datasets            [local]
├── splits/                      frozen split index files + checksums            [local]
├── comparators/                 comparator wrappers and environments            [local]
├── runs/{smoke,core,full}/      raw run outputs                                 [local]
├── results/{raw,harmonized,statistical,frozen}/                                 [local]
├── failures/                    failed-run records                              [local]
├── deviations/                  deviation records                               [local]
├── logs/                                                                        [local]
├── figures/{main,supplementary}/                                                [local]
├── tables/{main,supplementary}/                                                 [local]
└── manuscript_assets/                                                           [local]
```

`[local]` directories are git-ignored. That is deliberate and not only about size:
ROSMAP-derived material is access-controlled and must never reach a public remote.
The ignore rules are in `../.gitignore` and were added before this tree existed.

## Order of operations

1. Freeze and publish the record (`benchmark_record/`). Nothing below happens first.
2. Generate and freeze the splits and the simulated datasets; checksum both.
3. Run the **smoke** profile. Its outputs validate the harness and are never
   manuscript evidence.
4. Run the **core** profile — the minimum publication-grade benchmark.
5. Harmonize into one long-format table, validate against
   `../benchmark_record/schemas/benchmark-result.schema.json`, then freeze into
   `results/frozen/`.
6. Build every figure and table from `results/frozen/` alone, so no figure can
   contain a number that is not in the frozen results.
7. Run the **full** profile for the supplementary evidence.

## Rules that bind every run here

- Identical frozen splits for every method; the split file's SHA-256 goes into the
  run configuration.
- Every preprocessing step fitted inside the training fold only.
- Failed runs are recorded, never dropped.
- Any departure from the frozen protocol gets a deviation record **before** the
  affected analysis is rerun.
- Development runs are labelled `phase: development` in their result rows and can
  never support a manuscript claim.

## Completion and resume

A dataset is complete only when `runs/complete/<dataset>.json` exists **and** every
checksum it records still reproduces: the row shard, each out-of-fold file, each
retained failure record, and the frozen split manifests the rows cite. Row presence
alone never means complete — an interrupted dataset keeps its work in
`runs/staging/<dataset>/`, where no validator can see it, and a resume quarantines
that directory under `runs/quarantine/` before re-running.

`results/raw/datasets/<dataset>.jsonl` is the record of truth. The aggregate
`{profile}_rows.jsonl` is rebuilt atomically from the shards whose markers verify,
so it can never advertise a dataset that resume still considers outstanding.

## Running a phase

The task index is the completeness contract and is written for the whole profile,
so a family executed later stays visibly outstanding rather than disappearing:

```bash
python benchmarks/harness/run.py --run --profile core --family role_recovery --family group_leakage
```

`--readiness-phase <phase>` gates on the prerequisites that phase actually depends
on instead of the overall readiness verdict, and requires a matching `--family`.
Without it the strict overall gate applies. `benchmark_record/tools/readiness.py`
prints both verdicts; `phase_gate.py --family` scopes the between-phase check to
the families executed so far.
