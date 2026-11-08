@echo off
rem Double-click friendly wrapper around install.ps1 (Windows 10/11)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
