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

## Breakpoints in detail

| Kind | In VS Code | In the terminal | Stops when |
|---|---|---|---|
| Line | click left of the line number | `b 42` or `b model.apsi:42` | the line is reached |
| Conditional | right-click → *Add Conditional Breakpoint* → expression | `b 42 if loss > 1` | the expression is true in that frame |
| Hit count | right-click → *Hit Count* → `5` or `>= 5`, `> 5`, `== 5`, `% 10` | — | from the 5th hit on, after the 5th, only the 5th, every 10th |
| Log point | right-click → *Add Logpoint* → `epoch {epoch}: loss = {loss}` | — | never: the message is printed and the program continues |
| Uncaught error | *Breakpoints* panel → *Uncaught errors* (on by default) | always on | an error is about to end the program |

Conditions and log-point expressions are Apsi expressions evaluated in the paused frame: names, attributes,
indexing, arithmetic, comparisons and calls. A condition that cannot be evaluated (a typo, a name
that does not exist yet) stops at the breakpoint, so you see the problem instead of silently missing the stop.

## What the variables view shows

* **Locals and globals** of every frame in the call stack. Module-level globals are limited to the names your
  program declared, not everything imported.
* **Tensors and arrays:** `tensor (64, 128) float32  min -0.91 max 1.2 mean 0.003  [0.12, -0.04, …]`. Expand
  them to see rows.
* **Quantities with units** show as Apsi prints them (`9.81 m/s^2`).
* **Changing a value:** double-click it in the VS Code variables view and type any Apsi expression; the program
  continues with the new value.

## Debugging servers, tests and converted Python

* **Web servers:** set a breakpoint inside a route and send a request; the request waits while you are paused.
* **Tests:** `apsi debug tests/test_model.apsi` runs the test file under the debugger; a failing `assert` stops
  with its values visible when *Uncaught errors* is on.
* **Converted Python:** `apsi convert script.py` then `apsi debug script.apsi`. Breakpoints and the stack refer to
  the Apsi lines, which match the Python structure line for line.
* **Permissions** are the same as for `apsi run`: pass `--allow-read`, `--allow-net` and so on after the file.

## Cost

The debugger traces only frames of Apsi code (not the standard library or NumPy), and only while it is attached.
`apsi run` has no tracing at all, so there is no reason to remove breakpoints or debug code before deploying.
