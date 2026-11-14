@echo off
rem Finds the fastest settings of Apsi's GPU trainer for this card: the 127 M GPT (apsi gpu bench) under each setting,
rem once, the card cooled (to 50 C, at most 90 s) before every run and no temperature control during the short run (so every
rem setting meets the same card). About 10 minutes. The card's power limit is shown first and set to the
rem recommended sustained-training value (80 % on an RTX 30; the GPU Tuner's value) so no earlier run can slow it. Results: tune-results\summary.txt in the Apsi folder.
rem   default           what Apsi chooses by itself
rem   splitk off / 2 / 4  how the weight-gradient products split their long k (auto picks from the card's size)
rem   variant x / w / h   one build of the tiled tensor-core product for every shape (auto times them all)
rem   compensated       2.6.0's earlier default (compensated weights on in bf16), for comparison
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "APSI=%CD%"
set "PYTHONPATH=%APSI%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "APSI_NATIVE_QUIET=1"
set "OUT=%APSI%\tune-results"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\summary.txt"
set "PY="
for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
    if not defined PY (
        %%~P -c "import numpy, llvmlite" >nul 2>nul && set "PY=%%~P"
    )
)
if not defined PY (
    echo No Python with NumPy and llvmlite was found: py -m pip install numpy llvmlite ml_dtypes
    pause
    exit /b 1
)
echo Apsi GPU tuning on this computer, %DATE% %TIME% > "%LOG%"
echo -- the card before >> "%LOG%"
!PY! -m apsi gpu >> "%LOG%" 2>&1
!PY! -m apsi gpu power 80%% >> "%LOG%" 2>&1
echo -- the card now >> "%LOG%"
!PY! -m apsi gpu >> "%LOG%" 2>&1
type "%LOG%"
set "APSI_GPU_TEMP=off"
for %%R in (1 2) do (
    for %%C in (default qkv-off) do (
        call :cool
        set "APSI_SPLITK=" & set "APSI_GEMM_VARIANT=" & set "APSI_GPU_COMPENSATED=" & set "APSI_QKV="
        if "%%C"=="splitk-off" set "APSI_SPLITK=off"
        if "%%C"=="splitk-2" set "APSI_SPLITK=2"
        if "%%C"=="splitk-4" set "APSI_SPLITK=4"
        if "%%C"=="variant-x" set "APSI_GEMM_VARIANT=x"
        if "%%C"=="variant-w" set "APSI_GEMM_VARIANT=w"
        if "%%C"=="variant-h" set "APSI_GEMM_VARIANT=h"
        if "%%C"=="compensated" set "APSI_GPU_COMPENSATED=1"
        if "%%C"=="qkv-off" set "APSI_QKV=0"
        echo == round %%R: %%C
        !PY! -m apsi gpu bench 127m --steps 60 > "%OUT%\%%C-%%R.txt" 2>&1
        echo -- round %%R %%C >> "%LOG%"
        findstr /c:"tokens/s" /c:"{" "%OUT%\%%C-%%R.txt" >> "%LOG%"
        findstr /c:"tokens/s over" "%OUT%\%%C-%%R.txt"
    )
)
echo -- the card after >> "%LOG%"
!PY! -m apsi gpu >> "%LOG%" 2>&1
echo.
echo Done: %LOG%
pause
exit /b 0

:cool
powershell -NoProfile -Command "$t0=Get-Date; while (((Get-Date)-$t0).TotalSeconds -lt 90) { $c=[int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0); if ($c -le 50) { break }; Start-Sleep 5 }"
exit /b 0
