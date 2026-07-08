# Packaging omicau as a desktop app

omicau is primarily a `pip`-installable CLI. This directory builds an optional
**no-install desktop app** that bundles a private Python runtime (including
CPU-only PyTorch) so a non-coder downloads one installer and double-clicks —
double-clicking opens the local web UI in a browser; the same binary also works
as the `omicau` CLI.

PyInstaller **cannot cross-compile**, so each OS is built on its own machine (or
CI runner). The canonical path is CI: `.github/workflows/release.yml` builds all
three on `windows-latest`, `ubuntu-latest`, and `macos-14` (arm64) on a `v*` tag.

## Build locally

| OS | Command | Output |
| --- | --- | --- |
| Windows | `powershell -File packaging/build-windows.ps1` | `dist/omicau/` + `packaging/Output/omicau-setup-*.exe` (Inno Setup) |
| Linux | `bash packaging/build-linux.sh` | `dist/omicau/` + `dist/omicau-x86_64.AppImage` |
| macOS (arm64) | `bash packaging/build-macos.sh` | `dist/omicau-arm64.dmg` |

All three drive the same [`omicau.spec`](omicau.spec): **onedir** (never onefile —
torch would unpack hundreds of MB to temp on every launch and trip AV), **no UPX**
(it corrupts torch's shared libraries), `torchvision`/`torchaudio` excluded, built
against **CPU-only torch** (`--index-url https://download.pytorch.org/whl/cpu`).

## Size — state it honestly

torch is the whole size story. CPU-only wheels: ~109 MB (Windows), ~75 MB
(macOS arm64), ~170–190 MB (Linux, vs ~900 MB with the default CUDA wheel).
Realistic installed footprint ~450–850 MB; compressed installer download
~250–450 MB. That floor cannot be shrunk further and should be stated on the
download page.

## Signing (required for a smooth non-coder install)

An unsigned double-click is a hard stop at Gatekeeper / SmartScreen. Reuse the
BulkSeq Studio signing identities so per-certificate reputation carries over.

- **Windows** — sign the inner `omicau.exe` and the installer (`signtool`, or
  Azure Trusted Signing). Expect one SmartScreen prompt on a brand-new
  certificate that fades as reputation accrues. Set `OMICAU_SIGN_CERT` /
  `OMICAU_SIGN_PASS`.
- **macOS** — **arm64 only** (no modern x86_64 torch wheel). Sign every nested
  torch `.dylib`/`.so` inside-out, then the app, with a Developer ID + hardened
  runtime + secure timestamp; `notarytool submit --wait` then `stapler staple`
  so first launch works offline. Set `OMICAU_SIGN_ID` / `OMICAU_NOTARY_PROFILE`.
- **Linux** — AppImage needs no signing.

## Non-goal

The desktop GUI is not for HPC/headless clusters — the `pip` package + `omicau
run` CLI already serve those. Do not ship the GUI there.
