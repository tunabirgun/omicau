# Desktop packaging

The [Omicau v0.5.2 release](https://github.com/tunabirgun/omicau/releases/tag/v0.5.2) includes these desktop bundles alongside its Python wheel and source archive:

| Platform | Release asset | Use |
| --- | --- | --- |
| Windows x86-64 | `omicau-0.5.2-windows-x86_64.zip` | Extract the portable bundle and run `omicau.exe`. |
| Linux x86-64 | `omicau-x86_64.AppImage` | Make the AppImage executable and run it. |
| macOS Apple Silicon | `omicau-arm64.dmg` | Open the disk image and run the bundled app. |

Opening a frozen app without command-line arguments starts the local browser interface. Passing arguments uses the same `run`, `bootstrap`, `verify`, `check-env`, and `ui` commands as the Python package. The Windows release is a portable ZIP, not an installer. The v0.5.2 macOS disk image is not notarized.

## Build locally

PyInstaller builds each platform on its own operating system. The build scripts install CPU-only PyTorch where applicable and use [`omicau.spec`](omicau.spec) to produce a directory bundle.

| Platform | Command | Output |
| --- | --- | --- |
| Windows | `powershell -File packaging/build-windows.ps1` | `dist/omicau/`; optionally `packaging/Output/omicau-setup-*.exe` if Inno Setup is installed. |
| Linux | `bash packaging/build-linux.sh` | `dist/omicau-x86_64.AppImage`. |
| macOS arm64 | `bash packaging/build-macos.sh` | `dist/omicau-arm64.dmg`. |

Signing is conditional on the local signing environment. The Windows script signs the executable only when a signing certificate is configured; the macOS script signs or notarizes only when its corresponding credentials are configured. A local build does not establish that the published asset was signed or notarized.

The [`build-desktop` workflow](../.github/workflows/release.yml) runs on a published GitHub Release or manual dispatch. It uploads build products as CI artifacts and runs frozen-app smoke checks. Its permissions do not attach assets to the GitHub Release; attaching reviewed release files is a separate step.
