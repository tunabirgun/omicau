# Fixed splits and controls

This guide describes the public fixed-partition interface in Omicau v0.6.0. For installation and an ordinary synthetic run, use the [quickstart](../README.md#minimal-usage) or [browser guide](https://tunabirgun.github.io/omicau/guide.html).

## Optional browser interface

From the extracted release source directory, install the interface and launch it locally:

```bash
python -m pip install ".[ui]"
omicau ui --host 127.0.0.1
```

Use the versioned release source; v0.6.0 was not published to PyPI. The browser exposes a subset of configuration options and does not provide an external-holdout workflow. Advanced split configuration is supplied through the JSON configuration and manifest.

## Define a fixed nested plan

The v0.6.0 CLI accepts fixed manifests for classification and regression and requires `clinical.group`. The `cv.independent_rows` option does not replace that requirement on the fixed-plan route.

Create the plan before model fitting. Set `cv.split_manifest` to its JSON path, `cv.n_splits` to its outer-fold count, and `cv.inner_splits` to its nested-fold count. Relative manifest paths are resolved from the configuration directory. The envelope has exactly these five fields:

```json
{
  "schema_version": "omicau_public_split_manifest_v2",
  "aligned_provenance_sha256": "<aligned-input digest>",
  "aligned_group_sha256": "<aligned-group digest>",
  "aligned_permutation_strata_sha256": null,
  "outer_folds": []
}
```

This shows the structure, not a runnable manifest: replace the digest placeholders and empty fold list with values derived from the aligned inputs. The strata field is `null` only when no permutation strata are declared; otherwise it contains their ordered-identity digest.

Each outer fold has `train`, `assessment` and `inner_folds`. Each inner fold has `train` and `assessment`. All indices are zero-based positions in the full canonical aligned sample order, including inner-fold indices; they are not sample identifiers or positions relative to the outer-training subset. Inner folds partition only their outer-training rows. The validator checks group separation, exact assessment coverage, fold counts and task-specific outcome support before constructing models.

Use `omicau.models.split_plan.public_manifest_identity_binding` after alignment to derive group and strata bindings. The [v0.6.0 synthetic construction example](https://github.com/tunabirgun/omicau/blob/v0.6.0/tests/test_public_fixed_split_cli.py) demonstrates the format and rejection of changed input, group, strata or fold bindings. Tests are available in the repository, rather than the source distribution. Adapt the split design to the study; the test fixture is not a study partition plan.

Stacking is available only after the nested plan validates. Ordinary dynamic-split runs record stacking as unavailable instead of supplying a nonnested estimate.

## Permutation and stress controls

When `controls.enabled` and `controls.shuffle_target` are both true, a fixed-plan run also requires `clinical.permutation_strata`. Declare those design variables before analysis; they must not be endpoint columns or derived from the outcome. The endpoint must be constant within each group. To omit this control, set `controls.shuffle_target` to `false` before fitting and retain that choice in the configuration.

The fixed-plan control permutes whole groups within declared training strata while preserving assessment outcomes. Its interpretation depends on exchangeability within those strata. It is not automatically a global-chance result; inspect the recorded control eligibility and scope.

Feature-shuffling and random-noise controls are descriptive stress checks. Ordinary dynamic-split target shuffling is also descriptive and is not group-safe. No control result proves the absence of leakage, bias or confounding.

## Application and study records

An ordinary run records the actual split and control information in its audit. `omicau verify` compares the aligned-input fingerprint; it does not revalidate the whole split design, replay fitting or establish scientific validity. The [current evaluation scripts](https://github.com/tunabirgun/omicau/blob/main/docs/benchmark-records.md) are a separate study workflow, with their own recorded configuration and reproduction scope.
