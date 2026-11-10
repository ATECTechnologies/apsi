# Apsi 2.4.4 for Linux

Works on every x86-64 distribution with glibc 2.17 or newer, such as Ubuntu 18.04+, Debian 10+, Fedora, Arch,
openSUSE, Mint and Pop!_OS (the shell installer also on ARM, and in WSL on Windows).

**Download:** `ApsiSetup-2.4.4-linux.sh` and `apsi_2.4.4_all.deb` are on [apsi.dev/download](https://apsi.dev/download).
The other files below are made when you build Apsi yourself ([BUILD.md](BUILD.md)).

| File | Best for | Bundled Python |
|---|---|---|
| `ApsiSetup-2.4.4-linux.sh` | **simplest** — one-file installer for every distribution, no root, works offline | uses the system python3 (installs numpy privately if missing) |
| `ApsiSetup-2.4.4-linux-x86_64.AppImage` | **recommended** — graphical setup (Express/Custom, security, Apsi Guard) | ✓ 3.12 + numpy |
| `apsi-2.4.4-linux-x86_64.tar.gz` | portable — unpack and use, no install, no root | ✓ 3.12 + numpy |
| `apsi_2.4.4_all.deb` | Ubuntu/Debian system install with apt | uses the system python3 |
| `apsi-2.4.4-py3-none-any.whl` | an existing Python 3.12+ or a venv | – |
| `apsi-language-2.4.4.vsix` | VS Code extension | – |

## 0) Shell installer (`ApsiSetup-2.4.4-linux.sh`)

```bash
sh ApsiSetup-2.4.4-linux.sh                    # install for the current user
sh ApsiSetup-2.4.4-linux.sh --no-editors       # without adding Apsi to code editors
sh ApsiSetup-2.4.4-linux.sh --prefix ~/apsi    # somewhere else (default: ~/.local/share/apsi)
~/.local/share/apsi/uninstall.sh                # remove everything again
```

It needs `python3` 3.12 or newer (Ubuntu 24.04+, Debian 13+, Fedora 39+, Arch, openSUSE Tumbleweed; on older systems install one with `uv python install 3.12` or use the portable build, which brings its own Python). It installs:

- Apsi with its documentation and examples into `~/.local/share/apsi`
- the commands `apsi`, `apsi-repl` and `apsi-run` into `~/.local/bin` (it tells you if that folder is not on your
  `PATH` yet)
- the `.apsi` file type with its icon and *Run with Apsi*, *Apsi REPL* in the application menu, and *Apsi file* in
  *New Document* when `~/Templates` exists
- Apsi support in the code editors it finds (VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim)

numpy comes from the system when it is installed (`python3-numpy`); otherwise the installer creates a private Python
environment and installs numpy into it with pip. Running it again updates Apsi in place. In WSL (Linux on Windows)
it works the same way.

## 1) AppImage setup (graphical)

```bash
chmod +x ApsiSetup-2.4.4-linux-x86_64.AppImage
./ApsiSetup-2.4.4-linux-x86_64.AppImage
```

If you get a FUSE error (Ubuntu 22.04 and newer), do one of these:

```bash
sudo apt install libfuse2                     # Ubuntu 24.04+: sudo apt install libfuse2t64
./ApsiSetup-2.4.4-linux-x86_64.AppImage --appimage-extract-and-run   # or run without FUSE
```

Install without a window (servers, SSH):

```bash
./ApsiSetup-2.4.4-linux-x86_64.AppImage --appimage-extract-and-run --silent --profile strict
```

Apsi is installed in `~/.local/share/apsi` and the `apsi` command in `~/.local/bin`. The setup also registers the
`text/x-apsi` file type (icon and *Open with Apsi*), adds *Apsi REPL* to the application menu and, when
`~/Templates` exists, an *Apsi file* entry to *New Document*.
Uninstall with `~/.local/share/apsi/uninstall.sh`, or open the AppImage again and choose Uninstall.

## 2) Portable (tar.gz)

```bash
tar xzf apsi-2.4.4-linux-x86_64.tar.gz
cd apsi-2.4.4-linux-x86_64
./bin/apsi version
./bin/apsi run examples/01_hello.apsi
./install.sh            # (optional) link the apsi command into ~/.local/bin
```

## 3) .deb package

```bash
sudo apt install ./apsi_2.4.4_all.deb
apsi version
```

Examples and documentation are in `/usr/share/apsi/`. Remove with `sudo apt remove apsi`.

## 4) pip / venv

```bash
python3 -m venv ~/apsi-env && . ~/apsi-env/bin/activate
pip install apsi-2.4.4-py3-none-any.whl
```

## After installing

```bash
apsi doctor                                  # CPU, GPU and RAM against Apsi's targets
apsi pkg install ai                          # PyTorch + ONNX + Transformers
apsi pkg install ai-gpu                      # instead of the line above, with an NVIDIA card (PyTorch + CUDA 12.6)
apsi scan ~/Downloads/some-project           # malware scan (add --av to also run ClamAV)
apsi config --profile strict                 # default security level
eval "$(apsi completion bash)"               # tab completion (add to ~/.bashrc)
code --install-extension apsi-language-2.4.4.vsix
```

## Problems after installing

`apsi: command not found`, no colours in VS Code, `needs permission 'python'`: see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

## Checking downloads

```bash
sha256sum -c SHA256SUMS.txt
```
