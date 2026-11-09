# Editors and developer tools

Apsi works in the editors people already use. Each one gets Apsi's own colours, the Apsi file icon where the editor
supports one, run/test commands and — through the language server — live errors, hover, go to definition,
completion and formatting.

## Supported editors

| Editor | How Apsi gets there | What you get |
|---|---|---|
| **Visual Studio Code** | setup wizard / `apsi editors install vscode` | everything: colours, icon, themes, snippets, run/test/REPL commands, tasks, language server |
| **Cursor** | setup wizard / `apsi editors install cursor` | the same extension as VS Code |
| **Windsurf** | setup wizard / `apsi editors install windsurf` | the same extension as VS Code |
| **VSCodium**, **VS Code Insiders**, **Trae** | setup wizard / `apsi editors install vscodium` … | the same extension as VS Code |
| **Sublime Text 3/4** | setup wizard / `apsi editors install sublime` | colours, build system (`Ctrl+B`: run, test, check, format, bench), clickable errors, comments, indentation, Goto Symbol, snippets, command palette; language server with the LSP package |
| **Neovim** | setup wizard / `apsi editors install neovim` | filetype, colours, indentation, `:make`, mappings, and the **language server starts automatically** |
| **Vim** | setup wizard / `apsi editors install vim` | filetype, colours, indentation, `:make`, mappings |
| **Notepad++** (Windows) | setup wizard / `apsi editors install notepadpp` | Apsi language with its own colours, picked automatically for `.apsi` files |
| **JetBrains IDEs** (IntelliJ IDEA, PyCharm, CLion, WebStorm, …) | one manual step (below) | colours, comments and snippets from the TextMate bundle; language server through LSP4IJ |
| **Helix**, **Zed**, **Emacs** | one manual step (below) | language server; Emacs also gets `apsi-mode` with colours and indentation |

### `apsi editors`

```text
apsi editors                        list editors: found on this computer? Apsi installed?
apsi editors install                add Apsi to every editor that was found
apsi editors install cursor vscode  only these (ids: vscode vscode-insiders cursor windsurf vscodium trae
                                    sublime neovim vim notepadpp, or all)
apsi editors uninstall [NAMES]      remove it again
apsi editors path                   the folder that holds the integrations (…/editors)
```

The setup wizard runs the same command: its **Code editor support** component lists the editors it found, and the
uninstaller removes Apsi from them again. Nothing outside each editor's own extension/package folder is touched.
Restart editors that were open so they load Apsi.

How each one is installed:

- **VS Code family** — with the editor's command-line tool (`code`, `cursor`, `windsurf`, `codium`, …) when it is
  available; otherwise the extension is unpacked straight into the editor's extension folder (`~/.vscode/extensions`,
  `~/.cursor/extensions`, `~/.windsurf/extensions`, `~/.vscode-oss/extensions`, …), so it also works when the CLI is
  not on PATH.
- **Sublime Text** — `Packages/Apsi` in the Sublime data folder.
- **Neovim / Vim** — a native package in `…/site/pack/apsi/start/apsi` (Neovim) or `~/.vim/pack/apsi/start/apsi`
  (`~/vimfiles/…` on Windows); no plugin manager and no change to your `init.lua`/`.vimrc`.
- **Notepad++** — `%APPDATA%\Notepad++\userDefineLangs\Apsi.xml`.

### Manual steps

| Editor | Step |
|---|---|
| JetBrains | *Settings → Editor → TextMate Bundles → +* → choose `<Apsi folder>/editors/textmate/Apsi.tmbundle`. For live errors install the **LSP4IJ** plugin and add a language server with the command `apsi lsp` for `*.apsi`. |
| Helix | append `editors/helix/languages.toml` to `~/.config/helix/languages.toml` (`%AppData%\helix\` on Windows) |
| Zed | merge `editors/zed/settings.json` into Zed's `settings.json` |
| Emacs | `(load "<Apsi folder>/editors/emacs/apsi-mode.el")` in your init file; `M-x eglot` starts the language server |

`apsi editors` prints these steps with the real paths of your installation.

### Neovim and Vim details

| Key | Does |
|---|---|
| `<LocalLeader>r` | run the file |
| `<LocalLeader>t` | test the file |
| `<LocalLeader>c` | check the file (`:make`, errors in the quickfix list) |
| `<LocalLeader>f` | format the file |
| `<LocalLeader>i` | Apsi REPL in a terminal |

Neovim settings: `vim.g.apsi_lsp = false` keeps the language server off; `vim.g.apsi_command = { "py", "-3", "-m", "apsi" }`
starts Apsi another way. `:help apsi` has the details.

### Sublime Text details

`Ctrl+B` runs the file, `Ctrl+Shift+B` chooses *Test*, *Check*, *Format*, *Benchmark*, *Run Project* or *Test Project*,
`F4` jumps to the next error. For the language server, install the **LSP** package and add
`{ "clients": { "apsi": { "enabled": true, "command": ["apsi", "lsp"], "selector": "source.apsi" } } }` to its settings.

### One grammar everywhere

`editors/vscode/syntaxes/apsi.tmLanguage.json` and `editors/vscode/snippets/apsi.json` are the source of truth;
`python tools/build_editors.py` regenerates the Sublime Text and TextMate/JetBrains versions from them, so every editor
colours Apsi the same way. Vim, Notepad++ and Emacs use hand-written equivalents.

## Language server (`apsi lsp`)

Apsi has a standard LSP language server built in, so it works with almost every editor:

- **errors while you type**: unit mismatches, undeclared names, assignments to `let`, wrong arguments, `evolve` rules, syntax errors
- **hover**: a variable shows its physical type, e.g. `let speed: Velocity [m/s]`; a function shows its signature
- **go to definition**
- **completion**: names in the file, built-ins, modules, physical types, and **unit names inside `[...]`**
- **format document** (the same formatter as `apsi fmt`)
- **outline** (document symbols)

## VS Code, Cursor, Windsurf, VSCodium

Everything in this section applies to all editors of the VS Code family — they share one extension.
The setup wizard installs it automatically into every one it finds. To install it by hand:
Extensions → `…` → **Install from VSIX…** → `dist/apsi-language-2.4.4.vsix`.

The extension finds Apsi by itself, in this order: `apsi.command` from Settings, the wizard's install
(`%LOCALAPPDATA%\Programs\Apsi` / `~/.local/share/apsi`), the script install (`%LOCALAPPDATA%\Apsi` / `~/.apsi`),
`apsi` on PATH, `py -3 -m apsi`, `python -m apsi`. Set it yourself with Settings → `apsi.command`, e.g.
`["C:\\tools\\python\\python.exe", "-m", "apsi"]`.

### What you get

| Feature | Details |
|---|---|
| Syntax colours | keywords, `let`/`var`/`fn`/`struct`, definitions vs. calls, built-ins, parameters, types, **units in their own colour** (`9.81[m/s^2]`, `as [km]`), `±`, `x'` derivatives in `evolve`, f-string interpolation, decorators, `TODO` in comments; Apsi code blocks in Markdown are coloured too |
| Themes | **Apsi Dark** and **Apsi Light** (optional — every other theme works as well) |
| File icon | the Apsi logo, letters only, on a transparent background |
| Run / test | ▶ button in the editor title bar, status-bar button, right-click menus in the editor and the Explorer |
| REPL | **Apsi: Open REPL**; `Shift+Enter` sends the selection or the current line (blocks are sent whole) |
| Tasks | `Terminal → Run Task → apsi` lists *test*, *check*, *format*, *run project* and every task from `apsi.toml`; errors in the output appear in the Problems panel (`$apsi` problem matcher) |
| Projects | **Apsi: New Project…** creates a project from a template and opens it |
| Python → Apsi | **Apsi: Convert Python File to Apsi** (right-click a `.py` file or folder, or the editor title bar) — the result opens beside it |
| Snippets | `fn`, `fnb`, `struct`, `test`, `for`, `fore`, `if`, `ife`, `evolve`, `grad`, `train`, `route`, `main`, `use`, `import`, `@prec`, `pf`, … |
| Language server | everything in the list above |

### Commands and keys

| Command | Key |
|---|---|
| Apsi: Run File | `Ctrl+F5` or `Ctrl+Alt+N` |
| Apsi: Run File with Arguments… | – (remembers the last arguments per file) |
| Apsi: Test File | `Ctrl+Alt+T` |
| Apsi: Open REPL | `Ctrl+Alt+R` |
| Apsi: Run Selection / Line in REPL | `Shift+Enter` |
| Apsi: Load File into REPL | – |
| Apsi: Test Project · Check File · Watch File · Benchmark File · Show Permissions the File Needs | – |
| Apsi: Run Project Task… | – (pick from `apsi.toml` `[tasks]`) |
| Apsi: Run Apsi Command… | – (type any command line, e.g. `pkg list`) |
| Apsi: Show Documentation for Symbol | – (`apsi doc` for the word under the cursor) |
| Apsi: New Project… · Restart Language Server · Show Language Server Output | – |

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `apsi.command` | `[]` (auto-detect) | how to start Apsi |
| `apsi.run.arguments` | `[]` | extra flags for Run and Test, e.g. `["--allow-net"]` |
| `apsi.run.saveBeforeRun` | `true` | save before running |
| `apsi.run.clearTerminal` | `false` | clear the terminal before each run |
| `apsi.languageServer.enabled` | `true` | diagnostics, hover, completion, formatting |
| `apsi.statusBar.enabled` | `true` | version and Run button in the status bar |

### Tasks in `tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    { "type": "apsi", "command": "run", "file": "src/main.apsi", "args": ["--allow-net"], "label": "serve" },
    { "type": "apsi", "command": "task", "task": "ci", "label": "ci", "group": "test" }
  ]
}
```

`apsi new` writes a ready `.vscode/tasks.json` (run, test, check, format, watch) for every new project.

Build the `.vsix` from source:

```bash
cd editors/vscode && npm install && npx @vscode/vsce package
```

---

## Formatter (`apsi fmt`)

```bash
apsi fmt                     # every .apsi file under the current folder
apsi fmt examples            # one folder
apsi fmt --check src         # check only (for CI): exit code 1 when a file would change
```

- 4-space indentation, a space after `,`, spaces around `= == -> => |> ±`, two spaces before an end-of-line comment
- unit literals (`9.81[m/s^2]`), ranges (`0..10`) and keyword arguments (`f(x=1)`) have no spaces
- **100 % safe**: a file is written only when its syntax tree after formatting is identical to the one before

## Other highlighters

The TextMate grammar `editors/vscode/syntaxes/apsi.tmLanguage.json` also works in Sublime Text, JetBrains
(TextMate Bundles plugin), GitHub Linguist and Shiki (for highlighting on a website).
