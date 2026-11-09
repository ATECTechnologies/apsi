# Troubleshooting

This page covers the problems people most often hit right after installing Apsi, with the exact commands to fix them.

- [`apsi` is not recognized / `command not found`](#apsi-is-not-recognized--command-not-found)
- [VS Code shows no syntax colours, or Ctrl+F5 does nothing](#vs-code-shows-no-syntax-colours-or-ctrlf5-does-nothing)
- [`needs permission 'python'`](#needs-permission-python)
- [A function prints "runs on the bytecode engine"](#a-function-prints-runs-on-the-bytecode-engine)
- [`Apsi 2.4.4 needs Python 3.12 or newer`](#apsi-244-needs-python-312-or-newer)
- [A converted PyTorch program reports notes](#a-converted-pytorch-program-reports-notes)

---

## `apsi` is not recognized / `command not found`

**Typical messages**

```text
apsi : The term 'apsi' is not recognized as the name of a cmdlet, function, script file, or operable program.   (PowerShell)
'apsi' is not recognized as an internal or external command, operable program or batch file.                   (Command Prompt)
bash: apsi: command not found                                                                                     (Linux)
```

**Why it happens.** The installer adds the folder that contains the `apsi` command to your `PATH`. A program reads
`PATH` once, when it starts. Every terminal window that was already open before the installation finished still has
the old `PATH`, so it cannot find `apsi`, even though the installation succeeded.

**The fix.** Close the terminal and open a new one.

- **VS Code:** its built-in terminal inherits `PATH` from VS Code itself, so opening a new terminal tab is not
  enough. Quit VS Code completely (**File → Exit**, so that every VS Code window is closed) and start it again.
  The same applies to Cursor, Windsurf, VSCodium and other editors with a built-in terminal.
- **Windows Terminal:** close the whole Windows Terminal application, not just one tab.

### Use Apsi in the current window without restarting it

These commands update only the window you type them in. New windows pick up the change by themselves.

**PowerShell** (reloads `PATH` exactly as Windows now has it saved):

```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
apsi version
```

**Command Prompt** (`cmd.exe` cannot reload `PATH`, so add the Apsi folder for this session):

```bat
set "PATH=%LOCALAPPDATA%\Programs\Apsi\bin;%PATH%"
apsi version
```

If you installed from source with `setup.ps1`, the folder is `%LOCALAPPDATA%\Apsi\bin` instead.

**Linux** (shell installer, AppImage setup and `pip --user` installs put the command in `~/.local/bin`):

```bash
export PATH="$HOME/.local/bin:$PATH"
hash -r
apsi version
```

To make this permanent on Linux, add the `export` line to `~/.bashrc` (or `~/.zshrc`). Many distributions only add
`~/.local/bin` to `PATH` at login when the folder already existed, so logging out and back in once also fixes it.
The `.deb` package installs `/usr/bin/apsi`, which is always on `PATH`.

### Check whether Apsi is on PATH

**PowerShell**

```powershell
Get-Command apsi                                                     # shows the full path of the apsi command, if found
[Environment]::GetEnvironmentVariable('Path', 'User') -split ';' | Select-String -SimpleMatch 'Apsi'
Test-Path "$env:LOCALAPPDATA\Programs\Apsi\bin\apsi.cmd"             # True when the command file exists
```

**Command Prompt**

```bat
where apsi
echo %PATH%
```

**Linux**

```bash
command -v apsi                                          # prints the path of the apsi command, if found
echo "$PATH" | tr ':' '\n' | grep -x "$HOME/.local/bin"  # prints the folder when it is on PATH
ls -l ~/.local/bin/apsi                                  # the command file itself
```

How to read the result:

| Saved PATH contains the Apsi folder | Command file exists | What to do |
|---|---|---|
| yes | yes | Only the window is out of date: open a new window, or run the command above for your shell. |
| no | yes | The PATH entry was not written (for example, *Add apsi to PATH* was unticked in Custom setup). Run the setup again, or add the folder to your user PATH under **Settings → System → About → Advanced system settings → Environment Variables**. |
| no | no | Apsi is not installed for this user. Run the installer again. |

You can always start Apsi without `PATH` by using the full path, for example
`& "$env:LOCALAPPDATA\Programs\Apsi\bin\apsi.cmd" version` in PowerShell or `~/.local/bin/apsi version` on Linux.

---

## VS Code shows no syntax colours, or Ctrl+F5 does nothing

**Why it happens.** VS Code loads extensions, and reads `PATH`, when it starts. If it was open while Apsi was being
installed, it neither knows about the Apsi extension nor can it find the `apsi` command that **Apsi: Run File**
(`Ctrl+F5`) uses.

**The fix.**

1. Quit VS Code completely (**File → Exit**) and start it again.
2. If `.apsi` files are still shown as plain text, install the extension again from a terminal:

   ```bash
   apsi editors install vscode
   ```

   Then reload the window: press `Ctrl+Shift+P`, type **Developer: Reload Window**, and press Enter.

**Check that it worked.**

- The language indicator in the bottom-right corner of the status bar shows **Apsi** when an `.apsi` file is open.
  If it shows **Plain Text**, click it and choose **Apsi**.
- `code --list-extensions` lists `apsi.apsi-language`.
- `apsi editors list` shows which editors Apsi found and whether Apsi support is installed in each.

If `Ctrl+F5` still does nothing while colours work, the extension cannot find the `apsi` command: follow
[the PATH section above](#apsi-is-not-recognized--command-not-found), then restart VS Code once more.
Another extension can also take over `Ctrl+F5`; the same command is available as **Apsi: Run File** from the
Command Palette and as `Ctrl+Alt+N`.

---

## `needs permission 'python'`

**Typical message**

```text
app.apsi:3:1: PermissionError: pyimport('pandas') needs permission 'python'; run with --allow-python or --allow-all
```

**Why it happens.** Apsi programs cannot use Python libraries (`pyimport`, `use models`, the PyTorch backend,
`to_torch`, ...) unless you allow it, because Python code can do things that Apsi's permission system cannot check.
The error means your program uses Python and the permission was not given.

**The fix.** Add `--allow-python` to the command:

```bash
apsi run app.apsi --allow-python
apsi test --allow-python
```

`--allow-python` allows ordinary libraries such as numpy, pandas, torch and transformers. `--allow-python=all` also
allows system modules such as `subprocess` and `os`, which can bypass Apsi's file and network permissions; use it only
for code you trust. Avoid `--allow-all` unless you need every permission.

To avoid typing the flag every time in a project, list the permission in the project's `apsi.toml`:

```toml
[run]
allow = ["python"]
```

Executables made with `apsi build` take the same flag when they are built: `apsi build app.apsi --allow-python`.

---

## A function prints "runs on the bytecode engine"

```text
apsi: note: render() runs on the bytecode engine: lists (line 12)
```

This is a note, not an error: the program keeps running with correct results. A function marked `@native` uses
something the native compiler does not handle yet (the note names it and gives the line), so that function runs on
the bytecode engine instead of as machine code. Rewrite that line using the constructs listed in
[`NATIVE.md`](NATIVE.md) to get machine-code speed, use `@native("strict")` to turn the note into an error, or
`@native("quiet")` (or `APSI_NATIVE_QUIET=1`) to hide it. `apsi native explain app.apsi` lists every such function.

## `Apsi 2.4.4 needs Python 3.12 or newer`

Apsi runs on Python 3.12 and every newer release. This message means the `apsi` command, `pip` or your editor
started an older Python.

- **Windows:** the installer (`ApsiSetup-2.4.4-x64.exe`) brings its own Python 3.12, so nothing else is needed.
  To use `pip install` instead, install a current Python first:

  ```powershell
  winget install Python.Python.3.13
  py -3.13 -m pip install apsi-2.4.4-py3-none-any.whl
  ```

- **Linux:** Ubuntu 24.04+, Debian 13+, Fedora 39+, Arch and openSUSE Tumbleweed ship Python 3.12 or newer. On
  older systems, use the portable build (`apsi-2.4.4-linux-x86_64.tar.gz`, which brings its own Python), or install
  a newer Python next to the system one:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.12
  uv venv --python 3.12 ~/.apsi-venv
  ~/.apsi-venv/bin/pip install apsi-2.4.4-py3-none-any.whl
  ```

- **Check which Python is used:**

  ```bash
  python3 --version            # Linux
  py -0p                       # Windows: every installed Python
  ```

## A converted PyTorch program reports notes

`apsi convert` turns PyTorch programs into native Apsi and lists, with line numbers, the few things it cannot
translate exactly (for example learning-rate schedulers or `nn.MultiheadAttention`). The rest of the program is
complete and runs without Python. Review the listed lines; [`PYTHON.md`](PYTHON.md#pytorch-programs--native-apsi)
explains each one.
