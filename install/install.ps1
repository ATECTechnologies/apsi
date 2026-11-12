# Apsi installer for Windows 10 / 11 (PowerShell)
#   powershell -ExecutionPolicy Bypass -File install\install.ps1
#   powershell -ExecutionPolicy Bypass -File install\install.ps1 -Uninstall
#
# Installs into %LOCALAPPDATA%\Apsi (own virtual environment) and adds "apsi" to the user PATH.
param([switch]$Uninstall)
$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $PSScriptRoot
$Prefix = Join-Path $env:LOCALAPPDATA "Apsi"
$Bin    = Join-Path $Prefix "bin"

function Remove-FromPath($dir) {
    $p = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($p) {
        $parts = $p.Split(";") | Where-Object { $_ -and ($_.TrimEnd("\") -ne $dir.TrimEnd("\")) }
        [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    }
}

if ($Uninstall) {
    Remove-FromPath $Bin
    if (Test-Path $Prefix) { Remove-Item -Recurse -Force $Prefix }
    Write-Host "Apsi uninstalled."
    exit 0
}

$build = [Environment]::OSVersion.Version.Build
if ($build -lt 10240) { Write-Warning "Apsi targets Windows 10 or newer (this is build $build)." }

# find Python 3.12+ : prefer the py launcher
$candidates = @(
    @{ Exe = "py";      Args = @("-3") },
    @{ Exe = "python";  Args = @() },
    @{ Exe = "python3"; Args = @() }
)
$PyExe = $null; $PyArgs = @()
foreach ($c in $candidates) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $a = $c.Args
        $ok = & $c.Exe @a -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)" 2>$null
        if ("$ok".Trim() -eq "1") { $PyExe = $c.Exe; $PyArgs = $c.Args; break }
    } catch { }
}
if (-not $PyExe) {
    Write-Host "Python 3.12+ was not found. Install it from https://www.python.org/downloads/windows/"
    Write-Host "(tick 'Add python.exe to PATH'), or run:  winget install Python.Python.3.12"
    exit 1
}
Write-Host ("Using " + (& $PyExe @PyArgs --version))

New-Item -ItemType Directory -Force -Path $Bin | Out-Null
& $PyExe @PyArgs -m venv (Join-Path $Prefix "venv")
if ($LASTEXITCODE -ne 0) { Write-Host "Could not create a virtual environment."; exit 1 }
$vpy = Join-Path $Prefix "venv\Scripts\python.exe"
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet "$Root"
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed."; exit 1 }

$cmd = "@echo off`r`nchcp 65001 >nul`r`n`"$vpy`" -m apsi %*`r`n"
Set-Content -Path (Join-Path $Bin "apsi.cmd") -Value $cmd -Encoding ASCII

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $userPath) { $userPath = "" }
if (-not ($userPath.Split(";") -contains $Bin)) {
    [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";" + $Bin).TrimStart(";"), "User")
    Write-Host "Added $Bin to your PATH (open a new terminal)."
}
& (Join-Path $Bin "apsi.cmd") version
Write-Host "Ready: apsi run hello.apsi"
