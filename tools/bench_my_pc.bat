@echo off
rem Apsi 2.6.0 on this computer: every benchmark and test that needs your own processor and NVIDIA card.
rem   Double-click it. Takes about 30-60 minutes. Close other heavy programs first (games, browsers with video).
rem The results go to the folder bench-results-2.6.0 in the Apsi folder; send that folder back (or zip it).
rem What it runs:
rem   1. the machine:              apsi doctor (processor, memory, card, driver)
rem   2. correctness on the card:  every GPU kernel against the exact reference, GLA and a transformer trained twice
rem                                (identical weights), steps / resume / NaN guard, the GPU tests
rem   3. AI training speed:        the 10.9 M and 127 M GPT and the 127 M LLaMA/Qwen3-style model on the card
rem                                (tokens/s, temperature, power, energy per token)
rem   4. a training step, measured: queued against waiting copies, the kernels' shares
rem   5. CPU programs:             the eight heavy programs in Apsi and every other language installed here
rem                                (C/C++ with gcc or clang, Rust, Go, Java, Node.js, C#, Julia: missing ones are skipped)
rem   6. Apsi 2.6.0 against 2.4.4: the same programs, start-up and checking, on this computer
rem                                (needs Downloads\apsi-2.4.4-source.zip, which you already have)
rem Not run here (not part of this test): real language models from Hugging Face (use hub), the AI server, databases.
setlocal EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0.."
set "APSI=%CD%"
set "PYTHONPATH=%APSI%;%PYTHONPATH%"
set "PYTHONUTF8=1"
set "APSI_NATIVE_QUIET=1"
set "OUT=%APSI%\bench-results-2.6.0"
if not exist "%OUT%" mkdir "%OUT%"
set "LOG=%OUT%\summary.txt"
echo Apsi 2.6.0 benchmarks on this computer, %DATE% %TIME% > "%LOG%"

set "PY="
for %%P in ("py -3.14" "py -3.13" "py -3.12" "python") do (
    if not defined PY (
        %%~P -c "import numpy" >nul 2>nul && set "PY=%%~P"
    )
)
if not defined PY (
    echo No Python with a working NumPy was found. Install Python 3.12-3.14 from python.org, then: py -m pip install numpy >> "%LOG%"
    type "%LOG%"
    pause
    exit /b 1
)
echo Python: !PY! >> "%LOG%"
echo == 0. packages
!PY! -m pip install --quiet --disable-pip-version-check --no-warn-script-location numpy ml_dtypes llvmlite pytest >> "%LOG%" 2>&1

echo == 1. the machine
echo == 1. the machine >> "%LOG%"
!PY! -m apsi doctor > "%OUT%\doctor.txt" 2>&1
type "%OUT%\doctor.txt" >> "%LOG%"

echo == 2. correctness on the card
echo == 2. correctness on the card >> "%LOG%"
!PY! -m apsi gpu bench check > "%OUT%\gpu-check.txt" 2>&1
!PY! -m pytest -q -p no:cacheprovider tests/test_gpu_real.py tests/test_gpu_train.py tests/test_gla.py tests/test_training_modes.py tests/test_gpu_tensor.py tests/test_steps_resume.py tests/test_token_focus.py tests/test_bpe.py >> "%OUT%\gpu-check.txt" 2>&1
type "%OUT%\gpu-check.txt" >> "%LOG%"

echo == 3. AI training speed on the card (three models, a few minutes each)
echo == 3. AI training speed >> "%LOG%"
!PY! -m apsi gpu bench 11m > "%OUT%\gpu-bench-11m.txt" 2>&1
!PY! -m apsi gpu bench > "%OUT%\gpu-bench-127m.txt" 2>&1
!PY! -m apsi gpu bench modern > "%OUT%\gpu-bench-modern.txt" 2>&1
for %%F in (11m 127m modern) do (
    echo -- %%F >> "%LOG%"
    findstr /i /c:"tokens/s" /c:"{" "%OUT%\gpu-bench-%%F.txt" >> "%LOG%"
)

echo == 4. a training step, measured
echo == 4. a training step >> "%LOG%"
!PY! tools\flow_bench.py > "%OUT%\flow-bench.txt" 2>&1
type "%OUT%\flow-bench.txt" >> "%LOG%"

echo == 5. CPU programs against the other languages (15-30 minutes)
echo == 5. CPU programs >> "%LOG%"
!PY! benchmarks\heavy\run.py --skip Mojo --out "%OUT%\heavy-2.6.0" > "%OUT%\heavy-2.6.0.txt" 2>&1
type "%OUT%\heavy-2.6.0\results.md" >> "%LOG%" 2>nul

echo == 6. Apsi 2.6.0 against 2.4.4
echo == 6. 2.6.0 against 2.4.4 >> "%LOG%"
set "OLDZIP=%USERPROFILE%\Downloads\apsi-2.4.4-source.zip"
set "OLD=%TEMP%\apsi-bench-2.4.4"
if exist "%OLDZIP%" (
    if not exist "%OLD%\apsi-2.4.4\apsi\__init__.py" powershell -NoProfile -Command "Expand-Archive -LiteralPath '%OLDZIP%' -DestinationPath '%OLD%' -Force"
    for /d %%D in ("%OLD%\*") do if exist "%%D\apsi\__init__.py" set "OLDROOT=%%D"
)
if defined OLDROOT (
    !PY! benchmarks\heavy\run.py --apsi-root "!OLDROOT!" --only Apsi --label "Apsi 2.4.4" --out "%OUT%\heavy-2.4.4" > "%OUT%\heavy-2.4.4.txt" 2>&1
    !PY! benchmarks\versions\compare.py --old "!OLDROOT!" --new "%APSI%" --out "%OUT%\versions" > "%OUT%\versions.txt" 2>&1
    type "%OUT%\versions\versions.md" >> "%LOG%" 2>nul
) else (
    echo Downloads\apsi-2.4.4-source.zip not found: step 6 skipped >> "%LOG%"
)

echo == done >> "%LOG%"
echo.
echo Finished. The results are in %OUT%  (summary.txt has everything in one file).
pause
endlocal
