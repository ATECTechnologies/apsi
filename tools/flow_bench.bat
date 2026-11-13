@echo off
rem Time a training step on this computer's NVIDIA card; the report goes to flow-bench.txt in the Apsi folder.
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "OUT=%CD%\flow-bench.txt"
set "PY="
for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
    if not defined PY (
        %%~P -c "import numpy" >nul 2>nul && set "PY=%%~P"
    )
)
!PY! tools\flow_bench.py > "%OUT%" 2>&1
echo == done >> "%OUT%"
endlocal
