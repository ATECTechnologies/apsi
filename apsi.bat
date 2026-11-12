@echo off
rem Run Apsi on Windows 10/11 without installing:  apsi.bat run examples\01_hello.apsi
rem Uses the first Python whose NumPy loads: "python" on PATH (venvs, CI, python.org installs), then the py launcher's
rem 3.14, 3.13 and 3.12 (a free-threaded "3.14t" build on PATH often has no working NumPy).
setlocal EnableDelayedExpansion
chcp 65001 >nul
set "APSI_DIR=%~dp0"
set "PYTHONPATH=%APSI_DIR%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "PY="
for %%P in ("python" "py -3.14" "py -3.13" "py -3.12" "py -3") do (
    if not defined PY (
        %%~P -c "import numpy" >nul 2>nul && set "PY=%%~P"
    )
)
if not defined PY (
    echo apsi: no Python with a working NumPy was found. Install Python 3.12-3.14 from python.org ^(not the free-threaded build^), then: py -m pip install numpy 1>&2
    endlocal & exit /b 1
)
!PY! -m apsi %*
endlocal & exit /b %ERRORLEVEL%
