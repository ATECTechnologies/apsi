# Apsi for VS Code

Language support for [Apsi](https://github.com/apsi-lang/apsi) — the language for AI, physics and servers.

- **Syntax colours** for `.apsi` files: definitions and calls, built-ins, parameters, types, **units** (`9.81[m/s^2]`,
  `as [km]`), `±`, `x'` derivatives, f-strings, decorators — and for Apsi code blocks in Markdown
- **Apsi Dark** and **Apsi Light** themes, snippets, smart indentation, folding
- **Live diagnostics** from the Apsi checker (unit errors, shape errors, undefined names, …)
- **Hover** with the physical dimension of every variable and function signatures; go to definition; completion
  (including unit names inside `[...]`); document outline; **Format Document** (`apsi fmt`)
- **Run / Test / Check / Watch / Benchmark** the current file (▶ button, `Ctrl+F5`, `Ctrl+Alt+T`, right-click menus)
- **REPL**: *Apsi: Open REPL* (`Ctrl+Alt+R`), `Shift+Enter` sends the selection or line
- **Projects and tasks**: *Apsi: New Project…*, `apsi.toml` tasks in *Run Task*, errors in the Problems panel

The extension finds Apsi automatically (setup wizard, script install, `apsi` on PATH, `python -m apsi`).
Set `apsi.command` to choose one yourself. Full guide: `docs/EDITORS.md` in the Apsi folder.

Install Apsi first (the setup wizard installs this extension for you), or install the `.vsix` by hand:
Extensions view → `...` → *Install from VSIX…*
