# Build the Windows onedir app + installer.
#   powershell -File packaging/build-windows.ps1
# Prereqs: Python 3.12, and (for signing) the code-signing identity used for
# BulkSeq Studio so SmartScreen reputation carries over. Inno Setup 6 on PATH
# (iscc) is needed for the installer step.
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

# Optional: sign the inner exe (reuse the BulkSeq identity or Azure Trusted Signing)
if ($env:OMICAU_SIGN_CERT) {
  Write-Host "== signing omicau.exe =="
  signtool sign /fd SHA256 /f $env:OMICAU_SIGN_CERT /p $env:OMICAU_SIGN_PASS /tr http://timestamp.digicert.com /td SHA256 (Join-Path $dist "omicau.exe")
}

# Installer (Inno Setup). Skips gracefully if iscc is not installed.
if (Get-Command iscc -ErrorAction SilentlyContinue) {
  Write-Host "== Inno Setup installer =="
  iscc packaging/omicau.iss
} else {
  Write-Host "iscc (Inno Setup) not found on PATH — skipping installer; onedir is in dist/omicau"
}
