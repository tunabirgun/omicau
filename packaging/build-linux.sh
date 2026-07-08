#!/usr/bin/env bash
# Build the Linux onedir app + AppImage.
#   bash packaging/build-linux.sh
# CPU-only torch is essential on Linux: the default manylinux torch wheel bundles
# CUDA (~900 MB); the CPU index drops it to ~170-190 MB.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== CPU-only torch =="
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
pip install ".[ui]" pyinstaller

echo "== PyInstaller (onedir) =="
pyinstaller packaging/omicau.spec --noconfirm --clean
test -x dist/omicau/omicau || { echo "build failed"; exit 1; }

echo "== AppImage =="
APPDIR=dist/omicau.AppDir
rm -rf "$APPDIR"; mkdir -p "$APPDIR/usr/bin"
cp -r dist/omicau/* "$APPDIR/usr/bin/"
cat > "$APPDIR/omicau.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=omicau
Exec=omicau
Icon=omicau
Categories=Science;
Terminal=true
DESK
# a placeholder icon keeps appimagetool happy
touch "$APPDIR/omicau.png"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/omicau" "$@"
RUN
chmod +x "$APPDIR/AppRun"
if command -v appimagetool >/dev/null 2>&1; then
  appimagetool "$APPDIR" dist/omicau-x86_64.AppImage
  echo "Built dist/omicau-x86_64.AppImage"
else
  echo "appimagetool not found — AppDir is ready at $APPDIR (run appimagetool to package)"
fi
