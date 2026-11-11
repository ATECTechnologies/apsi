@echo off
rem Run Apsi on Windows 10/11 without installing:  apsi.bat run examples\01_hello.apsi
rem Uses the first Python that has NumPy: "python" on PATH (venvs, CI, python.org installs), then the py launcher.
setlocal
chcp 65001 >nul
set "APSI_DIR=%~dp0"
set "PYTHONPATH=%APSI_DIR%;%PYTHONPATH%"
set "PYTHONUTF8=1"
python -c "import numpy" >nul 2>nul
if %ERRORLEVEL%==0 (
    python -m apsi %*
    goto :done
)
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 -m apsi %*
) else (
    python -m apsi %*
)
:done
endlocal & exit /b %ERRORLEVEL%
