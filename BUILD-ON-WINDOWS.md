# Building Apsi 2.6.0 on Windows

Requirements: Windows 10 or 11, Python 3.12 or newer (`winget install Python.Python.3.12`), and Node.js 20 or newer
for the setup wizard (`winget install OpenJS.NodeJS.LTS`).

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1                 # virtual environment, Apsi, all tests, Apsi Guard scan
powershell -ExecutionPolicy Bypass -File setup.ps1 -Installer      # + installer\release\ApsiSetup-2.6.0-x64.exe
```

Then check the result:

```powershell
.\apsi.bat version
.\apsi.bat doctor
.\apsi.bat run examples\01_hello.apsi
```

Publish the installer on the website with the server command:

```bash
node scripts/admin.js release add ApsiSetup-2.6.0-x64.exe --version 2.6.0 --platform windows --title "Windows installer"
```

Details: `docs/BUILD.md`, `docs/INSTALL.md` and `docs/TROUBLESHOOTING.md`.

Or from any computer, straight to the website (token with the `release` permission):

```powershell
.\apsi.bat lib login --token apsi_…
.\apsi.bat release upload installer\release\ApsiSetup-2.6.0-x64.exe --version 2.6.0
```
