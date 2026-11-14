@echo off
rem Times every matrix product of the 127 M GPT on this card (seconds). Results: gemm-bench.txt in the Apsi folder.
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "APSI_NATIVE_QUIET=1"
set "PY="
for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
    if not defined PY (
        %%~P -c "import numpy, llvmlite" >nul 2>nul && set "PY=%%~P"
    )
)
%PY% tools\gemm_bench.py
