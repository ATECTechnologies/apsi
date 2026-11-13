# Building Apsi 2.6.0 from source

## Requirements

| | Windows 10/11 | Linux |
|---|---|---|
| Python 3.12+ | `winget install Python.Python.3.12` | `sudo apt install python3 python3-venv` |
| Node.js 20+ (installer only) | `winget install OpenJS.NodeJS.LTS` | `sudo apt install nodejs npm` (or nodejs.org) |
| NSIS + zip (classic .exe/.deb/zip only) | – | `sudo apt install nsis zip` |

## One command

```powershell
# Windows (PowerShell, in the source folder)
powershell -ExecutionPolicy Bypass -File setup.ps1                  # .venv + Apsi + tests + malware scan of examples
powershell -ExecutionPolicy Bypass -File setup.ps1 -Installer       # + installer\release\ApsiSetup-2.6.0-x64.exe
powershell -ExecutionPolicy Bypass -File setup.ps1 -Libs ai,gpu     # + libraries
```

```bash
# Linux
./setup.sh                    # .venv + Apsi + tests + malware scan of examples
./setup.sh --installer        # + installer/release/*.AppImage
./setup.sh --release          # + dist/: wheel, .deb, Windows zip, NSIS exe, SHA256SUMS
./setup.sh --libs ai,gpu      # + libraries
```

Linux release files that need **no internet and no Node.js** (any Linux, also WSL):

```bash
python3 packaging/make_linux_installer.py dist/linux   # dist/linux/ApsiSetup-2.6.0-linux.sh (self-extracting installer)
python3 packaging/make_wheel.py dist/linux              # dist/linux/apsi-2.6.0-py3-none-any.whl (standard library only)
SKIP_WINDOWS=1 SKIP_ELECTRON=1 sh packaging/build_all.sh   # dist/apsi_2.6.0_all.deb + wheel
```

The AppImage wizard and the portable `tar.gz` with its own Python download Electron and Python, so they need
internet: `bash packaging/build_linux_release.sh` builds all Linux files into `dist/linux/`.

## Libraries (`apsi pkg`)

| Bundle | Installs |
|---|---|
| `gpu` | cupy-cuda12x (NVIDIA, compute 7.5+: GTX 1660 and newer) |
| `onnx` / `onnx-gpu` | onnxruntime / onnxruntime-gpu |
| `torch-auto` | PyTorch, the build that fits the machine (by NVIDIA driver version; ROCm on AMD/Linux; else CPU) |
| `torch` | PyTorch, CPU build |
| `torch-cuda` · `torch-cu128` · `torch-cu124` · `torch-cu118` | PyTorch with CUDA 12.6 (default) · 12.8 (RTX 50) · 12.4 · 11.8 — one build at a time; an installed build is kept, another one is replaced |
| `transformers` | transformers, tokenizers, safetensors, huggingface_hub |
| `sci` · `data` · `plot` · `crypto` | scipy · pandas + pyarrow · matplotlib · cryptography |
| `audit` | pip-audit |
| groups | `ai` = onnx torch-auto transformers · `ai-gpu` = gpu onnx-gpu torch-cuda transformers · `science` = sci data plot · `all` |

```
apsi pkg list                       # what is available and installed
apsi pkg install ai                 # binary wheels only, over HTTPS
apsi pkg install some-package       # any PyPI package
apsi pkg lock                       # apsi.lock with exact versions + sha256
apsi pkg install -r apsi.lock       # reproducible install, hashes enforced
apsi pkg audit                      # conflicts + known vulnerabilities
apsi pkg remove torch
```

## Step by step (what the scripts do)

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt && pip install -e .
python -c "from apsi.security import write_manifest; write_manifest()"
apsi test tests/apsi examples --allow-all && python -m unittest discover -s tests

cd installer
python ../packaging/build_payload.py --platform windows --out payload   # or --platform linux
npm ci && npm test && npm run build:ui
npx electron-builder --win portable --x64       # Windows → release/ApsiSetup-2.6.0-x64.exe
npx electron-builder --linux AppImage           # Linux   → release/*.AppImage
npm run dev                                     # UI preview in a browser (mock system)
```

The payload contains a relocatable CPython 3.12 (python-build-standalone), numpy, Apsi, the
examples, the docs, the VS Code extension and the brand icons. The build downloads the
runtime once into `packaging/.cache/`.

## Signing (recommended for public releases)

```powershell
$env:CSC_LINK = "C:\keys\apsi-codesign.pfx"; $env:CSC_KEY_PASSWORD = "…"
npm run dist:win
```

## Unattended install (IT deployment)

```
ApsiSetup-2.6.0-x64.exe --silent [--dir PATH] [--profile sandbox|strict|standard|developer]
                         [--no-examples] [--no-docs] [--no-editors] [--no-path] [--no-fileAssociation]
                         [--no-startMenu] [--desktop-shortcut] [--with-gpu] [--with-onnx] [--with-torch]
                         [--with-transformers]
ApsiSetup-2.6.0-x64.exe --silent --uninstall
```
