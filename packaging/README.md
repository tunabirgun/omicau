# Desktop packaging

Versioned desktop bundles are listed on the [current GitHub release](https://github.com/tunabirgun/omicau/releases/latest) after the platform build and frozen-application smoke checks pass. Only the files attached to that release are available for download.

| Platform | Bundle | Use |
| --- | --- | --- |
| Windows x86-64 | Versioned portable ZIP | Extract the bundle and run omicau.exe. |
| Linux x86-64 | AppImage | Make it executable and run it. |
| macOS Apple Silicon | Disk image | Open the image and run the bundled app. |

Opening a frozen app without arguments starts the local browser interface. Arguments use the same run, bootstrap, verify, check-env and ui commands as the Python package. Signing and notarization status must be stated for the actual attached artifact.

## Build locally

PyInstaller builds each platform on its own operating system. The build scripts install CPU-only PyTorch where applicable and use [`omicau.spec`](omicau.spec) to produce a directory bundle.

| Platform | Command | Output |
| --- | --- | --- |
| Windows | `powershell -File packaging/build-windows.ps1` | `dist/omicau/`; optionally `packaging/Output/omicau-setup-*.exe` if Inno Setup is installed. |
| Linux | `bash packaging/build-linux.sh` | `dist/omicau-x86_64.AppImage`. |
| macOS arm64 | `bash packaging/build-macos.sh` | `dist/omicau-arm64.dmg`. |

Signing is conditional on the local signing environment. The Windows script signs the executable only when a signing certificate is configured; the macOS script signs or notarizes only when its corresponding credentials are configured. A local build does not establish that the published asset was signed or notarized.

The [`build-desktop` workflow](../.github/workflows/release.yml) runs on a published GitHub Release or manual dispatch. It uploads build products as CI artifacts and runs frozen-app smoke checks. Its permissions do not attach assets to the GitHub Release; attaching reviewed release files is a separate step.

## Conda recipe

The proposed conda-forge recipe is scoped to the current source archive. Its source checksum is required by the recipe format. It has not been submitted to a channel or validated with a conda build; use the release wheel or source for the tested installation route.
