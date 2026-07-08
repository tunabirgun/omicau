#!/usr/bin/env bash
# Build the macOS (Apple Silicon) app, sign, notarize, and package a .dmg.
#   bash packaging/build-macos.sh
#
# ARM64 ONLY: PyTorch stopped shipping macOS x86_64 wheels after 2.2.2 (Jan 2024),
# so a universal2 binary with modern torch is impossible. State arm64-only in the
# release notes; do not imply Intel support.
#
# Signing/notarization env (set before running):
#   OMICAU_SIGN_ID     "Developer ID Application: Your Name (TEAMID)"
#   OMICAU_NOTARY_PROFILE  a notarytool keychain profile (xcrun notarytool store-credentials)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ "$(uname -m)" = "arm64" ] || { echo "must build on Apple Silicon (arm64)"; exit 1; }

echo "== CPU torch (arm64 wheel is already CPU) =="
pip install --upgrade torch
pip install ".[ui]" pyinstaller

echo "== PyInstaller (onedir) =="
pyinstaller packaging/omicau.spec --noconfirm --clean
APP=dist/omicau
test -x "$APP/omicau" || { echo "build failed"; exit 1; }

if [ -n "${OMICAU_SIGN_ID:-}" ]; then
  echo "== codesign nested dylibs inside-out, then the app =="
  # sign every nested Mach-O (torch .dylib/.so) before the outer bundle
  find "$APP" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 \
    | xargs -0 -I{} codesign --force --timestamp --options runtime -s "$OMICAU_SIGN_ID" {}
  codesign --force --timestamp --options runtime -s "$OMICAU_SIGN_ID" "$APP/omicau"
fi

echo "== .dmg =="
DMG=dist/omicau-arm64.dmg
rm -f "$DMG"
hdiutil create -volname omicau -srcfolder "$APP" -ov -format UDZO "$DMG"

if [ -n "${OMICAU_SIGN_ID:-}" ]; then codesign --force --timestamp -s "$OMICAU_SIGN_ID" "$DMG"; fi
if [ -n "${OMICAU_NOTARY_PROFILE:-}" ]; then
  echo "== notarize + staple =="
  xcrun notarytool submit "$DMG" --keychain-profile "$OMICAU_NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
echo "Built $DMG (arm64 only)"
