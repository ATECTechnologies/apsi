@echo off
rem Builds the Windows installer of this Apsi version from scratch, in one run:
rem   installer\release\ApsiSetup-<version>-x64.exe   (+ the VS Code extension in dist\)
rem Old build output is removed first, so nothing from an earlier version can end up in it.
rem Needs Python 3.12+ and Node.js 20+ (winget install OpenJS.NodeJS.LTS).
setlocal
cd /d "%~dp0"
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(Select-String -Path apsi\__init__.py -Pattern '__version__ = \"(.+)\"').Matches[0].Groups[1].Value"`) do set "V=%%v"
echo == Apsi %V%: cleaning old build output
if exist installer\release rmdir /s /q installer\release
if exist installer\payload rmdir /s /q installer\payload
if exist installer\dist rmdir /s /q installer\dist
if exist dist\apsi-language-*.vsix del /q dist\apsi-language-*.vsix
echo == building (10-20 minutes)
powershell -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -Installer || goto fail
if not exist "installer\release\ApsiSetup-%V%-x64.exe" goto fail
echo.
echo Done: %CD%\installer\release\ApsiSetup-%V%-x64.exe
echo Next: install it, then double-click Downloads\upload-releases.bat
pause
exit /b 0
:fail
echo.
echo The build failed (see above).
pause
exit /b 1
