# Releasing omicau

GitHub release assets and PyPI publication are separate actions. Use the versioned release wheel or source archive for the current workflow, and describe only files that have actually been attached after validation.

The current distribution is [v0.6.0](https://github.com/tunabirgun/omicau/releases/tag/v0.6.0). It provides a wheel, source archive, Windows portable ZIP, Linux AppImage, macOS ARM64 disk image and the original validation archive. It was not published to PyPI or a Conda channel. Current study materials are [results v3.0.0](https://doi.org/10.5281/zenodo.23444757) and [protocol v6.3.0](https://doi.org/10.5281/zenodo.23444777); their versions are independent of the software version.

## Python package

1. Update and review the version in `pyproject.toml` and `omicau/__init__.py`, keeping distribution metadata and CLI output consistent. Review version-specific documentation and the proposed Conda recipe separately. PyPI does not allow replacing files for an already published version.
2. Merge the reviewed version change to `main`. The [`publish-pypi` workflow](.github/workflows/publish-pypi.yml) detects a version change, builds the wheel and source distribution, checks their metadata, and stores them as CI artifacts. This push run is validation only.
3. Review the build and validation results. When publication is authorized and the repository is configured as a PyPI trusted publisher, dispatch the same workflow with `publish=true` on the reviewed source revision. It rebuilds and publishes through the `pypi` environment using OIDC trusted publishing; a previous validation run does not cover a different revision.
4. Verify the resulting version and files on PyPI. Until that check succeeds, installation instructions for a newer GitHub release should use its release wheel rather than an unpinned `pip install omicau`.

A manual dispatch with `publish=false` builds validation artifacts without publishing.

## Desktop bundles

The [`build-desktop` workflow](.github/workflows/release.yml) runs when a GitHub Release is published or when manually dispatched. It builds and smoke-tests frozen bundles on Windows, Linux, and macOS. The workflow stores the products as CI artifacts and has read-only repository permissions; it does not attach files to a GitHub Release.

Review the platform artifacts and their smoke results before attaching any desktop bundles to a release. Describe the actual attached files. The Windows bundle may be a portable ZIP when an installer is not built. State the actual signing and notarization status of the macOS disk image. Local build commands and outputs are documented in [packaging/README.md](packaging/README.md).

For v0.6.0, the Windows binaries are unsigned and the macOS bundle is not signed or notarized. The published validation ZIP contains the original release evaluation, not the later comparative extension in Zenodo. Preserve versioned assets and point readers to the appropriate study record when documentation changes.

## Conda-forge

The repository contains a proposed [conda-forge recipe](packaging/conda-forge/meta.yaml). A recipe in this repository does not itself publish a conda package or keep one synchronized. Check current staged-recipes requirements and the package's actual channel availability before submitting or describing a conda-forge distribution. Feedstock creation and later updates follow conda-forge's own review and automation.
