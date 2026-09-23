# Fixed splits and controls

Use the normal workflow when a study-specific split plan is not pre-specified:

```bash
omicau check-env
omicau bootstrap --dataset mock --out-dir demo
omicau run --config demo/config.json --cores 8 --device cpu --no-llm
omicau verify --config demo/config.json --audit demo/run/audit.json
```

Install the wheel from the GitHub release:

```bash
python -m pip install ./omicau-0.5.2-py3-none-any.whl
```

This installation route uses the release asset directly.

`omicau ui --host 127.0.0.1` starts the optional local browser interface after installation with `omicau[ui]`. The interface binds to localhost; it does not provide an external-holdout workflow.

The repository's `benchmarks/` harness and `benchmark_record/` are archived evaluation materials. They are separate from the ordinary `omicau run` and UI execution paths.

For a pre-specified nested evaluation, create and freeze a JSON manifest before model fitting, then add its relative or absolute path as `cv.split_manifest` and its nested fold count as `cv.inner_splits`. A public manifest has exactly these top-level fields:

```json
{
  "schema_version": "omicau_public_split_manifest_v2",
  "aligned_provenance_sha256": "...",
  "aligned_group_sha256": "...",
  "aligned_permutation_strata_sha256": "... or null",
  "outer_folds": []
}
```

Each outer fold records `train`, `assessment`, and `inner_folds` in the canonical aligned order. The implementation verifies the value, group, and strata bindings before model construction. The executable construction example is `tests/test_public_fixed_split_cli.py`; it derives the bindings with `omicau.models.split_plan.public_manifest_identity_binding` after alignment.

Stacking fusion is available only after this nested plan validates. Without a validated nested plan, the ordinary dynamic-split route omits stacking and records an unavailable status rather than producing a nonnested estimate.

To request the fixed-plan group target-permutation stress control, set `clinical.group`, `clinical.permutation_strata`, `cv.split_manifest`, and `cv.inner_splits`. Strata must be declared clinical design variables and must not be derived from the endpoint. The endpoint must be constant within each group. The control permutes whole groups inside declared training strata while preserving assessment outcomes. Its result is conditional on the stated exchangeability assumption; it is not automatically a global-chance result. Read the control receipt before interpretation.

Feature shuffling and random-noise controls are descriptive stress controls. Ordinary dynamic-split target shuffling remains descriptive and is not group-safe. No control result proves absence of leakage, bias, or confounding.
