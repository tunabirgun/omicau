# Releasing omicau (automatic PyPI + conda sync)

The goal: push a version tag and have `pip install omicau` (and, downstream,
`conda install omicau`) update on their own. Here is exactly how automatic each
piece is, and the one manual step that cannot be removed.

## The one unavoidable manual step: the version number

PyPI (and therefore conda-forge) **never accepts a re-upload of an existing
version** — `0.1.0` is permanent. So every release must carry a new version.
There is no "publish the same version on every push"; a push that does not change
the version has nothing new to publish. Bump `version` in `pyproject.toml` each
release.

## One-time setup (do this once, ~2 minutes)

Register this repo as a PyPI **trusted publisher** so CI can upload with no stored
token: <https://pypi.org/manage/account/publishing/> →

| Field | Value |
| --- | --- |
| PyPI project | `omicau` |
| Owner | `tunabirgun` |
| Repository | `omicau` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

(Alternative: store a PyPI API token as the `PYPI_API_TOKEN` Actions secret and
switch the publish step to use it. Trusted publishing is preferred — no secret.)

## Shipping an update (this is the automatic part)

```bash
# 1. bump the version
#    edit pyproject.toml:  version = "0.1.1"
git commit -am "release: v0.1.1"
git push

# 2. tag it -> CI publishes to PyPI automatically
git tag v0.1.1
git push --tags
```

Pushing the `v*` tag triggers [`.github/workflows/publish-pypi.yml`](.github/workflows/publish-pypi.yml),
which checks the tag matches `pyproject.toml`, builds the sdist + wheel, runs
`twine check`, and publishes to PyPI via OIDC. Within a minute or two,
`pip install --upgrade omicau` resolves the new version for everyone. No manual
`twine upload`.

> Note: a `v*` tag also triggers `release.yml` (the desktop-app installers). That
> is separate from PyPI and needs the signing secrets; ignore or disable it if you
> only want the Python package.

## conda: automatic *after* a one-time human gate

conda-forge tracks **PyPI**, not this repo. The flow:

1. **One time:** open a PR adding [`packaging/conda-forge/meta.yaml`](packaging/conda-forge/meta.yaml)
   to <https://github.com/conda-forge/staged-recipes>. A conda-forge maintainer
   reviews and merges it, creating the `omicau-feedstock`. This human review is
   required and cannot be automated away.
2. **After that, it is automatic:** the conda-forge **autotick bot** watches PyPI
   and opens a version-bump PR on the feedstock whenever a new PyPI release lands.
   Enable *automerge* on the feedstock and those bot PRs merge themselves once CI
   passes — so a new PyPI version flows to `conda install -c conda-forge omicau`
   with no manual step.

## Summary

| Channel | Automatic on a version tag? | Manual step |
| --- | --- | --- |
| PyPI (`pip` / `pipx`) | **Yes** — CI builds + publishes | bump version; one-time trusted-publisher setup |
| conda-forge (`conda` / `mamba`) | **Yes, after the feedstock exists** — autotick bot + automerge | one-time staged-recipes PR (maintainer-merged) |
| `environment.yml` (conda today) | n/a — always installs current `main` via pip | — |
