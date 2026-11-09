<#
  Apsi 2.0.0 — one-shot setup for Windows 10/11 (run from the source folder).

    powershell -ExecutionPolicy Bypass -File setup.ps1               # install Apsi for development + run tests
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Installer    # also build the setup wizard (ApsiSetup-*.exe)
    powershell -ExecutionPolicy Bypass -File setup.ps1 -Libs ai      # also install libraries: gpu, onnx, torch, transformers, sci, data, ai, ai-gpu, science, all

  Needs: Python 3.12+ (python.org or winget install Python.Python.3.12).
  For -Installer also Node.js 20+ (winget install OpenJS.NodeJS.LTS).
#>
param(
  [switch]$Installer,
  [switch]$SkipTests,
  [string[]]$Libs = @()
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($t) { Write-Host "`n==> $t" -ForegroundColor Cyan }
function Need($cmd, $hint) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { Write-Host "Missing: $cmd. $hint" -ForegroundColor Red; exit 1 }
}

Step "Checking Python"
$py = $null
foreach ($c in @("py -3.12", "py -3", "python")) {
  try { $v = & cmd /c "$c -c ""import sys;print(sys.version_info>=(3,12))"" 2>nul"; if ($v -eq "True") { $py = $c; break } } catch {}
}
if (-not $py) { Write-Host "Python 3.12+ not found. Install it: winget install Python.Python.3.12" -ForegroundColor Red; exit 1 }
Write-Host "using $py"

Step "Creating a private environment (.venv)"
if (-not (Test-Path .venv)) { & cmd /c "$py -m venv .venv"; if ($LASTEXITCODE) { exit 1 } }
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $vpy -m pip install --disable-pip-version-check -q --upgrade pip
& $vpy -m pip install --disable-pip-version-check -q --only-binary=:all: -r requirements.txt
& $vpy -m pip install --disable-pip-version-check -q -e .
& $vpy -c "from apsi.security import write_manifest; write_manifest()"

if ($Libs.Count -gt 0) {
  Step "Installing libraries: $($Libs -join ', ')"
  & $vpy -m apsi pkg install @Libs
}

if (-not $SkipTests) {
  Step "Running the test suite"
  & $vpy -m apsi test tests/apsi examples --allow-all
  if ($LASTEXITCODE) { exit 1 }
  & $vpy -W ignore -m unittest discover -s tests -q
  if ($LASTEXITCODE) { Write-Host "Some toolchain tests failed (see above); continuing." -ForegroundColor Yellow }
  & $vpy -m apsi scan examples
}

if ($Installer) {
  Need node "Install Node.js 20+: winget install OpenJS.NodeJS.LTS"

  Step "Building the VS Code extension (optional)"
  Push-Location editors\vscode
  try {
    $vsixOut = Join-Path $PSScriptRoot "dist\apsi-language-2.0.0.vsix"
    New-Item -ItemType Directory -Force (Split-Path $vsixOut) | Out-Null
    npm install --no-audit --no-fund --silent
    npx --yes @vscode/vsce@2 package --allow-missing-repository --skip-license -o $vsixOut
  } catch { Write-Host "VS Code extension skipped: $_" -ForegroundColor Yellow }
  Pop-Location

  Step "Building the Windows payload (bundled Python 3.12 + numpy)"
  Push-Location installer
  if (Test-Path payload) { Remove-Item payload -Recurse -Force }
  & $vpy ..\packaging\build_payload.py --platform windows --out payload
  if ($LASTEXITCODE) { Pop-Location; exit 1 }

  Step "Building the setup wizard"
  # npm install (not ci) so the Windows builds of esbuild/rollup are used even if the lock file came from Linux
  npm install --no-audit --no-fund
  if ($LASTEXITCODE) { Pop-Location; exit 1 }
  npm run build:ui
  if ($LASTEXITCODE) { Pop-Location; exit 1 }
  npx electron-builder --win portable --x64 --publish never
  if ($LASTEXITCODE) { Pop-Location; exit 1 }
  Pop-Location
  Get-ChildItem installer\release\*.exe | ForEach-Object { Write-Host "built $($_.FullName)" -ForegroundColor Green }
  Write-Host "Run it:     .\installer\release\ApsiSetup-2.0.0-x64.exe" -ForegroundColor Green
}

Step "Done"
Write-Host "Activate:   .venv\Scripts\Activate.ps1"
Write-Host "Try:        apsi run examples\01_hello.apsi"
Write-Host "Libraries:  apsi pkg list   |   apsi pkg install gpu onnx torch transformers"
Write-Host "Security:   apsi scan <folder>   |   apsi audit <file>   |   apsi config --profile strict"
