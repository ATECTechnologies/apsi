@echo off
rem Run Apsi on Windows 10/11 without installing:  apsi.bat run examples\01_hello.apsi
setlocal
chcp 65001 >nul
set "APSI_DIR=%~dp0"
set "PYTHONPATH=%APSI_DIR%;%PYTHONPATH%"
set "PYTHONUTF8=1"
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 -m apsi %*
) else (
    python -m apsi %*
)
endlocal & exit /b %ERRORLEVEL%
