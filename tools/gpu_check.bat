@echo off
rem Check Apsi on this computer's NVIDIA card: every GPU kernel against the exact reference, GLA, gpu_tensor and the
rem tests that need a real card. The whole report goes to gpu-check.txt in the Apsi folder.
rem   Double-click it, or: tools\gpu_check.bat
rem It uses the first Python whose NumPy loads (a free-threaded "3.14t" build often has no working NumPy).
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONPATH=%CD%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "OUT=%CD%\gpu-check.txt"
echo Apsi GPU check, %DATE% %TIME% > "%OUT%"
set "PY="
for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
    if not defined PY (
        %%~P -c "import numpy" >nul 2>nul && set "PY=%%~P"
    )
)
if not defined PY (
    echo No Python with a working NumPy was found. Install Python 3.12-3.14 from python.org ^(not the free-threaded build^), then: py -m pip install numpy >> "%OUT%"
    type "%OUT%"
    pause
    exit /b 1
)
echo Python: !PY! >> "%OUT%"
echo == packages >> "%OUT%"
!PY! -m pip install --quiet --disable-pip-version-check --no-warn-script-location numpy ml_dtypes llvmlite pytest >> "%OUT%" 2>&1
!PY! -c "import sys, numpy, llvmlite; print(sys.version); print('numpy', numpy.__version__, 'llvmlite', llvmlite.__version__)" >> "%OUT%" 2>&1
echo == device >> "%OUT%"
!PY! -m apsi doctor >> "%OUT%" 2>&1
echo == every kernel against the reference >> "%OUT%"
!PY! -m apsi gpu bench check >> "%OUT%" 2>&1
echo == tests on the card >> "%OUT%"
!PY! -m pytest -q -p no:cacheprovider tests/test_gpu_real.py tests/test_gpu_train.py tests/test_gla.py tests/test_training_modes.py tests/test_gpu_tensor.py >> "%OUT%" 2>&1
echo == done >> "%OUT%"
echo The report is in %OUT%
endlocal
