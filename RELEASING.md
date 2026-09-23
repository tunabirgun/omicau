# Releasing omicau

The [GitHub v0.5.2 release](https://github.com/tunabirgun/omicau/releases/tag/v0.5.2) contains a Python wheel and source archive, a portable Windows x86-64 ZIP, a Linux x86-64 AppImage, and a macOS Apple Silicon disk image. As checked on 23 September 2026, [PyPI](https://pypi.org/project/omicau/) still lists v0.4.0. A GitHub release and a PyPI publication are separate actions.

## Python package

1. Update and review the version in `pyproject.toml`. PyPI does not allow replacing files for an already published version.
2. Merge the reviewed version change to `main`. The [`publish-pypi` workflow](.github/workflows/publish-pypi.yml) detects a version change, builds the wheel and source distribution, checks their metadata, and stores them as CI artifacts. This push run is validation only.
3. Review the build and validation results. When publication is authorized, dispatch the same workflow with `publish=true`. It rebuilds and publishes through the `pypi` environment using OIDC trusted publishing.
4. Verify the resulting version and files on PyPI. Until that check succeeds, installation instructions for a newer GitHub release should use its release wheel rather than an unpinned `pip install omicau`.

A manual dispatch with `publish=false` builds validation artifacts without publishing.

## Desktop bundles

The [`build-desktop` workflow](.github/workflows/release.yml) runs when a GitHub Release is published or when manually dispatched. It builds and smoke-tests frozen bundles on Windows, Linux, and macOS. The workflow stores the products as CI artifacts and has read-only repository permissions; it does not attach files to a GitHub Release.

Review the platform artifacts and their smoke results before attaching any desktop bundles to a release. Describe the actual attached files. The v0.5.2 Windows asset is a portable ZIP rather than an installer, and the macOS arm64 disk image is not notarized. Local build commands and outputs are documented in [packaging/README.md](packaging/README.md).

## Conda-forge

The repository contains a proposed [conda-forge recipe](packaging/conda-forge/meta.yaml). A recipe in this repository does not itself publish a conda package or keep one synchronized. Check current staged-recipes requirements and the package's actual channel availability before submitting or describing a conda-forge distribution. Feedstock creation and later updates follow conda-forge's own review and automation.
