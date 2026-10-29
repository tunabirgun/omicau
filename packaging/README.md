# Desktop packaging

The [v0.6.0 release](https://github.com/tunabirgun/omicau/releases/tag/v0.6.0) supplies these desktop bundles. The Python wheel and source archive are on the same release page.

| Platform | Bundle | Use |
| --- | --- | --- |
| Windows x86-64 | `omicau-0.6.0-windows-x86_64.zip` | Extract the entire bundle and run `omicau.exe`. |
| Linux x86-64 | `omicau-0.6.0-linux-x86_64.AppImage` | Make it executable and run it. |
| macOS ARM64 | `omicau-0.6.0-macos-arm64.dmg` | Open the image and run the bundled `omicau` executable. |

Opening a frozen app without arguments starts the local browser interface. Arguments use the same `run`, `bootstrap`, `verify`, `check-env` and `ui` commands as the Python package. The v0.6.0 Windows binaries are unsigned, and the macOS bundle is not signed or notarized. No Windows installer or Intel macOS bundle is attached to this release.

## Build locally

Run the scripts from the repository root in a dedicated build environment; they install or update dependencies. The CI workflow uses Python 3.12. PyInstaller builds each platform on its own operating system and uses [`omicau.spec`](omicau.spec) to produce a directory bundle. Windows and Linux scripts install PyTorch from its CPU wheel index; macOS builds require Apple Silicon.

| Platform | Command | Output |
| --- | --- | --- |
| Windows | `powershell -File packaging/build-windows.ps1` | `dist/omicau/`; optionally `packaging/Output/omicau-setup-*.exe` if Inno Setup is installed. |
| Linux | `bash packaging/build-linux.sh` | `dist/omicau-x86_64.AppImage`. |
| macOS arm64 | `bash packaging/build-macos.sh` | `dist/omicau-arm64.dmg`. |

Linux requires `appimagetool` on `PATH`; the script fails if it is absent or if `dist/omicau.AppDir` already exists. Preserve or relocate previous build products before rebuilding. Windows requires Inno Setup's `iscc` only for the optional installer, and `signtool` when executable signing is requested. macOS packaging uses `hdiutil`; optional signing and notarization require the corresponding Apple tools and configured identities.

Signing is conditional on the build environment. The Windows script can sign the inner executable; it does not automatically sign the optional Inno Setup installer. The macOS script signs or notarizes when the corresponding credentials are configured. Verify the actual output before describing its signing status. A successful smoke run establishes basic operation, not scientific validity of every analysis path.

The [`build-desktop` workflow](../.github/workflows/release.yml) runs on a published GitHub Release or manual dispatch. It uploads build products as CI artifacts and runs frozen-app smoke checks. Its permissions do not attach assets to the GitHub Release; attaching reviewed release files is a separate step.

## Conda recipe

The proposed conda-forge recipe is scoped to the current source archive. Its source checksum is required by the recipe format. It has not been submitted to a channel or validated with a conda build; use the release wheel or source for the tested installation route.

The root [`environment.yml`](../environment.yml) is an unpinned convenience specification, not the study environment lock or a published Conda package. For scientific reproduction, use the environment specifications in the [current evaluation records](../docs/benchmark-records.md).
