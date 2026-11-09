# Installing Apsi on Windows and Linux

## Supported systems

| | Target |
|---|---|
| Operating system | **Windows 10 / 11** (build 10240 or newer), **Linux** kernel 4.15+ (Ubuntu 18.04+, Debian 10+, Fedora, Arch, …) |
| CPU | x86-64-v3 (AVX2 + FMA): Intel Core i5-7500 / i7-6700 or newer, any AMD Ryzen, or ARM64 |
| GPU (optional) | NVIDIA compute capability 7.5+ (GTX 1660 / RTX 20xx or newer) + CuPy |
| Python | 3.12 or newer (tested with 3.12, 3.13 and 3.14; the Windows installer and the portable Linux build bring their own 3.12) |

After installing, run `apsi doctor` to check your machine against these targets.

---

## The easy way: the ready-made installers (recommended)

| System | File | How |
|---|---|---|
| Windows 10/11 (64-bit) | `ApsiSetup-2.4.4-x64.exe` | double-click (contains Python 3.12 + numpy; no administrator rights needed) |
| Windows, portable | `apsi-2.4.4-windows-x64.zip` | unzip and run `apsi.cmd` (works from a USB stick) |
| Ubuntu / Debian / Mint | `apsi_2.4.4_all.deb` | `sudo apt install ./apsi_2.4.4_all.deb` |
| Any system with Python 3.12+ | `apsi-2.4.4-py3-none-any.whl` | `pip install apsi-2.4.4-py3-none-any.whl` |

These files are in the `dist/` folder.

### What the Windows setup wizard does

1. Shows *Preparing Apsi Setup…* immediately while it unpacks, then keeps that window on top of other windows until
   the wizard has finished loading, closes it and opens the wizard in its place.
2. Lets you choose **Express** (recommended defaults) or **Custom** (folder, components, optional AI packages, security profile).
3. Installs to `%LOCALAPPDATA%\Programs\Apsi` — per user, no administrator rights.
4. Adds `apsi` to your user PATH. Windows that were already open keep the old PATH: open a new terminal (quit VS Code
   completely) or see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for a command that fixes the current window.
5. Registers `.apsi` files: the Apsi icon, **Run with Apsi** (double-click), **Edit** (VS Code when installed, otherwise Notepad),
   **Test with Apsi**, and **New → Apsi source file** in Explorer's right-click menu.
6. Creates Start-menu entries: Apsi REPL, Examples, Documentation, Uninstall.
7. Adds Apsi to the code editors it finds — VS Code, Cursor, Windsurf, VSCodium, VS Code Insiders, Trae, Sublime Text,
   Neovim, Vim and Notepad++ — and lists them on the *System check* page (see [`EDITORS.md`](EDITORS.md)).
8. Registers in *Settings → Apps* so it can be removed normally. Every change is recorded and undone on uninstall;
   a failed install rolls back completely.

Unattended install: `ApsiSetup-2.4.4-x64.exe --silent [--dir PATH] [--profile strict] [--no-editors] [--with-torch]`.
Windows older than 10 and 32-bit Windows are not supported.

### What the .deb package does

- installs `python3-numpy` from apt automatically
- creates the `/usr/bin/apsi` command
- puts the examples and documentation in `/usr/share/apsi/examples` and `/usr/share/apsi/docs`

---

## Windows 10 / 11 from source

### 1. Install Python (if you do not have it)

```powershell
winget install Python.Python.3.12
```

Or download it from python.org and **tick "Add python.exe to PATH"** during setup.

### 2. Install Apsi

Double-click `install\install.bat`, or run in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install\install.ps1
```

The script:

- finds Python 3.12+ (`py -3` first, then `python`)
- creates a separate virtual environment in `%LOCALAPPDATA%\Apsi\venv`, so other Pythons on the machine are untouched
- creates the `apsi` command (`%LOCALAPPDATA%\Apsi\bin\apsi.cmd`) and adds it to your user PATH, without administrator rights
- sets the console to UTF-8 so symbols such as ✓ ± ² display correctly

Open a new terminal and try:

```powershell
apsi version
apsi run examples\01_hello.apsi
apsi doctor
```

Uninstall: `powershell -ExecutionPolicy Bypass -File install\install.ps1 -Uninstall`

### Without installing

```bat
pip install numpy
apsi.bat run examples\01_hello.apsi
```

### Windows tips

- **Windows Terminal** (included with Windows 11) shows Unicode symbols such as ✓ ± ² best
- `.apsi` files saved by Notepad (UTF-8 with BOM, CRLF line endings) work as they are
- the REPL understands Windows paths directly: `cd C:\Users\me\project`, `run app.apsi`, `ls`
- compiling C for the FFI: `examples\native\build.bat` works with MSVC (`cl`) and MinGW-w64 (`gcc`)
- `cload("native/vecmath")` finds `vecmath.dll` on Windows and `libvecmath.so` on Linux
- shell completion: `apsi completion powershell | Out-String | Invoke-Expression` (add it to your `$PROFILE`)

---

## Linux from source

### 1. Install Python and venv

```bash
# Ubuntu / Debian
sudo apt install python3 python3-venv
# Fedora
sudo dnf install python3
# Arch
sudo pacman -S python
```

### 2. Install Apsi

```bash
sh install/install.sh
```

- installs into `~/.apsi/venv` and creates `~/.local/bin/apsi` without sudo
- if `~/.local/bin` is not on your PATH yet, the script prints the line to add to `~/.bashrc`

Uninstall: `sh install/install.sh --uninstall`

### Without installing

```bash
pip install numpy
./apsi.sh run examples/01_hello.apsi
```

### Executable scripts

```bash
#!/usr/bin/env apsi
```

Put this line at the top of a file and `chmod +x` it to run `.apsi` files like any other program (a `#` line is a comment).

---

## Optional extras

| You want | Install | Used with |
|---|---|---|
| NVIDIA GPU | `apsi pkg install gpu` | `use gpu` → `to_gpu(x)` |
| PyTorch (picks the CUDA build that matches your NVIDIA driver, or CPU) | `apsi pkg install torch-auto` | `fit(…, backend="torch")` on the GPU, `torch_module`, `to_torch` (`--allow-python`) — see [`BACKENDS.md`](BACKENDS.md) |
| PyTorch + ONNX Runtime + Transformers | `apsi pkg install ai` (or `ai-gpu`) | `use models` (`--allow-python`) |
| Editor support | `apsi editors install` | VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim, Notepad++ — see [`EDITORS.md`](EDITORS.md) |
| Tab completion and history in the REPL on Windows | `pip install pyreadline3` (included in the setup wizard) | `apsi` |

## Problems after installing

`apsi` not recognized, no colours in VS Code, `needs permission 'python'`: see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Checking the installation

```bash
apsi doctor                                   # hardware and operating system
apsi verify                                   # runtime files match the release manifest
apsi test examples tests/apsi --allow-all     # the whole language test suite
python -m unittest discover tests             # the toolchain test suite
```

If you keep the code on GitHub, `.github/workflows/ci.yml` runs every test automatically on
Windows Server 2022/2025 and Ubuntu 22.04/24.04 with Python 3.12, 3.13 and 3.14.
