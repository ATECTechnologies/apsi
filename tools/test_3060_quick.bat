@echo off
rem Apsi 2.6.0: the short release check on this computer's NVIDIA card (about 15 minutes).
rem   Double-click it. Close games and other heavy programs first: the card must be idle.
rem The results go to test-results-2.6.0\summary.txt in the Apsi folder (send that file back).
rem What it runs:
rem   1. the machine:             apsi doctor
rem   2. correctness on the card: every GPU kernel against the exact reference (apsi gpu bench check)
rem   3. the tests that use the card, and the new parts (layouts, sealed packages, enum / match, graphs)
rem   4. AI training speed:       the 10.9 M GPT, the 127 M GPT and the 110 M LLaMA/Qwen3-style model, 45 steps
rem                               each (tokens/s, temperature, power)
rem Everything else is tested on every push on Linux, macOS and Windows (no card needed).
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "APSI=%CD%"
set "PYTHONPATH=%APSI%"
set "PYTHONUTF8=1"
set "APSI_NATIVE_QUIET=1"
set "OUT=%APSI%\test-results-2.6.0"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\summary.txt"
echo Apsi 2.6.0 release check on this computer, %DATE% %TIME% > "%LOG%"
set "PY=%APSI%\.venv\Scripts\python.exe"
if not exist "!PY!" set "PY="
if not defined PY (
    for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
        if not defined PY (
            %%~P -c "import numpy" >nul 2>nul && set "PY=%%~P"
        )
    )
)
if not defined PY (
    echo No Python with NumPy was found. Install Python 3.12-3.14, then: py -m pip install numpy >> "%LOG%"
    type "%LOG%"
    pause
    exit /b 1
)
echo Python: !PY! >> "%LOG%"
"!PY!" -m pip install --quiet --disable-pip-version-check --no-warn-script-location numpy ml_dtypes llvmlite pytest >nul 2>&1
timeout /t 10 /nobreak >nul

echo == 1. the machine
echo == 1. the machine >> "%LOG%"
"!PY!" -m apsi doctor >> "%LOG%" 2>&1

echo == 2. correctness on the card (2-3 minutes)
echo == 2. correctness on the card >> "%LOG%"
"!PY!" -m apsi gpu bench check >> "%LOG%" 2>&1

echo == 3. tests (3-6 minutes)
echo == 3. tests >> "%LOG%"
"!PY!" -m pytest -q -p no:cacheprovider tests/test_gpu_real.py tests/test_gpu_train.py tests/test_training_modes.py tests/test_gpu_tensor.py tests/test_steps_resume.py tests/test_thermal.py tests/test_layout.py tests/test_modelpack.py tests/test_language_more.py tests/test_token_focus.py >> "%LOG%" 2>&1

echo == 4. AI training speed (about 5 minutes: three models)
echo == 4. AI training speed >> "%LOG%"
for %%F in (11m 127m modern) do (
    "!PY!" -m apsi gpu bench %%F --steps 45 > "%OUT%\gpu-bench-%%F.txt" 2>&1
    echo -- %%F >> "%LOG%"
    findstr /i /c:"tokens/s" /c:"C mean" /c:"launches" /c:"{" "%OUT%\gpu-bench-%%F.txt" >> "%LOG%"
)

echo. >> "%LOG%"
echo finished %DATE% %TIME% >> "%LOG%"
echo.
echo Done: %LOG%
findstr /i /c:"FAIL" /c:"failed" /c:"error" "%LOG%" >nul && (echo Something did not pass: see the file above.) || (echo Everything passed.)
pause
