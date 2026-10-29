# Build the Windows directory bundle and optional installer.
#   powershell -File packaging/build-windows.ps1
# Run from the repository root with Python 3.12, as in CI.
# Inno Setup 6 (iscc on PATH) enables the optional installer step.
# Executable signing additionally requires signtool and a configured certificate.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "== CPU-only torch =="
pip install --upgrade "torch" --index-url https://download.pytorch.org/whl/cpu
pip install ".[ui]" pyinstaller

Write-Host "== PyInstaller (onedir) =="
Push-Location $root
pyinstaller packaging/omicau.spec --noconfirm --clean
Pop-Location

$dist = Join-Path $root "dist/omicau"
if (-not (Test-Path (Join-Path $dist "omicau.exe"))) { throw "build failed: omicau.exe missing" }
Write-Host "Built $dist"

# Optional: sign the inner executable. This does not sign the installer.
if ($env:OMICAU_SIGN_CERT) {
  Write-Host "== signing omicau.exe =="
  signtool sign /fd SHA256 /f $env:OMICAU_SIGN_CERT /p $env:OMICAU_SIGN_PASS /tr http://timestamp.digicert.com /td SHA256 (Join-Path $dist "omicau.exe")
}

# Installer (Inno Setup). Skips gracefully if iscc is not installed.
if (Get-Command iscc -ErrorAction SilentlyContinue) {
  Write-Host "== Inno Setup installer =="
  # Inject the version from pyproject.toml so the installer label never drifts.
  $ver = (Select-String -Path (Join-Path $root "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
  iscc "/DAppVersion=$ver" packaging/omicau.iss
} else {
  Write-Host "iscc (Inno Setup) not found on PATH - skipping installer; onedir is in dist/omicau"
}
