# Debugging

Apsi has a real debugger: breakpoints, stepping, the call stack, variables and tensors — in VS Code and the other
VS Code-family editors (F5) or in the terminal. It works on the compiled engine: Apsi lines are mapped back from
the generated code, so breakpoints, `step` and the stack follow your source, also through imported `.apsi` files and
converted Python programs.

## In VS Code

Install the Apsi extension (`apsi editors install vscode`), open a `.apsi` file, click left of a line number to set a
breakpoint and press **F5**. The extension starts `apsi dap` (the Debug Adapter Protocol server) for you.

* **Breakpoints:** plain, conditional (`x > 10`), hit counts (`5`, `>= 100`), and log points (`loss = {loss}`).
* **Stop on uncaught errors** (on by default): the program pauses where the error happened, with all variables.
* **Variables:** locals and globals of each frame; lists, dicts, records and objects expand; values can be changed
  while paused.
* **Tensors** show shape, dtype, min/max/mean and the first values instead of a wall of numbers.
* **Watch / Debug Console:** any Apsi expression in the paused frame.
* Program output appears in the Debug Console.

Without a `launch.json`, F5 debugs the open file. With one:

```json
{ "type": "apsi", "request": "launch", "name": "Run app.apsi", "program": "${file}",
  "args": ["--epochs", "3"], "stopOnEntry": false }
```

## In the terminal

```bash
apsi debug app.apsi                      # stops at the first line
apsi debug app.apsi -b 42 -b "57 if loss > 1" -- --epochs 3
```

| command | |
|---|---|
| `c` | continue |
| `n` | next line (step over) |
| `s` | step into a call |
| `o` | step out of the function |
| `b LINE`, `b FILE:LINE`, `b LINE if COND` | add a breakpoint |
| `p EXPR` | print an expression (tensors summarised) |
| `l` | local variables |
| `bt` | call stack |
| `q` | quit |

## Without a debugger

* `print(x)` of a tensor gives its shape, dtype and values.
* Errors name the file, line and column and show the line with a caret, plus the chain of Apsi function calls.
* `apsi check app.apsi` finds type, unit and name errors without running anything.
