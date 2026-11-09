# Apsi for Sublime Text

Installed by `apsi editors install sublime` (or the Apsi setup wizard) into `Packages/Apsi`.

- syntax colours for `.apsi` files (the same grammar as the VS Code extension)
- **Tools → Build** (`Ctrl+B`) runs the file; `Ctrl+Shift+B` picks Test, Check, Format, Benchmark, Run/Test Project;
  errors are clickable (F4 jumps to the next one)
- command palette: *Apsi: Run File*, *Apsi: Test File*, …
- comments (`Ctrl+/`), indentation after `:`, functions and structs in *Goto Symbol* (`Ctrl+R`), snippets

Language server (live errors, hover, completion): install the **LSP** package, then add to
*Preferences → Package Settings → LSP → Settings*:

```json
{ "clients": { "apsi": { "enabled": true, "command": ["apsi", "lsp"], "selector": "source.apsi" } } }
```
