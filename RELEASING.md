# Releasing omicau (automatic PyPI + conda sync)

Shipping a new version is one deliberate act — **bump the version and merge to
`main`** — and pip (then conda) update on their own. No manual `twine`, no tags to
push, no tokens anywhere in the repo (publishing uses OIDC trusted publishing).

## Why a version bump is unavoidable

PyPI **permanently rejects re-uploading an existing version** (`0.1.0` is frozen),
and conda-forge tracks PyPI. So "publish on every push" is impossible: a push that
does not change the version has nothing new to ship. The version bump *is* the
release signal.

## One-time setup (~2 minutes, once ever)

Register this repo as a PyPI **trusted publisher** so CI uploads with no stored
token: <https://pypi.org/manage/account/publishing/> →

| Field | Value |
| --- | --- |
| PyPI project | `omicau` |
| Owner | `tunabirgun` |
| Repository | `omicau` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

The binding is keyed to the workflow **filename + environment** — never rename
`publish-pypi.yml` or add a second publishing workflow, or OIDC will fail.

## Shipping an update (the automatic part)

```bash
# edit pyproject.toml:  version = "0.1.1"
git checkout -b release-0.1.1
git commit -am "release: v0.1.1"
git push -u origin release-0.1.1
# open a PR, review, merge to main
```

On merge, [`publish-pypi.yml`](.github/workflows/publish-pypi.yml) detects that
`pyproject.toml`'s version changed (git-diff against the previous commit), builds
the sdist + wheel, runs `twine check`, and publishes to PyPI via OIDC. Within a
minute or two `pip install --upgrade omicau` resolves the new version for
everyone. (`workflow_dispatch` is available as a manual override; `skip-existing`
makes re-runs no-ops.)

Keeping `main` behind PR review is the safety gate: a version bump ships only
after the PR that carries it is merged.

## Desktop installers are separate (and deliberate)

[`release.yml`](.github/workflows/release.yml) builds the signed Windows / macOS /
Linux desktop apps. It runs **only on a published GitHub Release** (or manual
dispatch) — *not* on a version bump — so the code-signing / notarization pipeline
never fires by accident. Cut a GitHub Release when you actually want new installers.

## conda: automatic *after* a one-time human gate

conda-forge follows **PyPI**, not this repo. The recipe is prepared at
[`packaging/conda-forge/meta.yaml`](packaging/conda-forge/meta.yaml) with the real
sdist sha256. To create the feedstock (one time):

1. **Verify the current recipe format first.** conda-forge is migrating toward a v1
   `recipe.yaml`; confirm whether staged-recipes today expects the classic
   `recipes/omicau/meta.yaml` or the v1 schema, per its live CONTRIBUTING docs.
2. Lint locally: `pipx run conda-smithy recipe-lint packaging/conda-forge/meta.yaml`.
3. Fork <https://github.com/conda-forge/staged-recipes>, add the recipe under
   `recipes/omicau/`, open a PR, and answer the bot + reviewer thread until it
   merges. This lists you as a standing recipe maintainer, so it needs your own
   account and consent — do it yourself rather than delegate it.

After the feedstock exists it is automatic: the conda-forge **autotick bot** opens
a version-bump PR on every new PyPI release; enable *automerge* and those merge
themselves once CI passes, so `conda install -c conda-forge omicau` stays current
with no manual step.

## Summary

| Channel | Automatic on a version bump merged to `main`? | Manual step |
| --- | --- | --- |
| PyPI (`pip` / `pipx`) | **Yes** — CI builds + publishes via OIDC | bump version; one-time trusted-publisher setup |
| conda-forge (`conda` / `mamba`) | **Yes, after the feedstock exists** — autotick bot + automerge | one-time staged-recipes PR (you, maintainer-reviewed) |
| Desktop installers | No — deliberate, on a GitHub Release | cut a Release when you want installers |
