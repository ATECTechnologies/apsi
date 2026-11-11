@echo off
rem Uploads Apsi 2.4.4 and newer (installers and the GPU Tuner) from your Downloads folder to apsi.dev.
rem Asks for an API token (Settings -> Developer, "release"). Files already on the site are skipped.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload-releases.ps1" %*
pause
