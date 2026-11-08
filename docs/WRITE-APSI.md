# You already know Apsi — three ways to write it (1.10.0)

You do not have to learn a new language before you are productive. If you can read Python, you can read Apsi:

| Way | Best when | What you do |
|---|---|---|
| **1. Write it like Python** | you start something new | the Python you know, with the small changes below |
| **2. Let an AI write it** | you want speed | give an AI assistant the brief below |
| **3. Write Python and convert** | you already have Python code | `apsi convert script.py` |

## 1. Write it like Python

Indentation, `if/elif/else`, `for`/`while`, `break`/`continue`, lists, dicts, tuples, slicing, comprehensions,
f-strings, lambdas and the string and list methods work as in Python. The differences:

| Python | Apsi | Why |
|---|---|---|
| `x = 1` | `let x = 1` (constant) or `var x = 1` (variable) | accidental reassignment is caught before running |
| `def f(a, b=2):` | `fn f(a, b = 2):` — or one line: `fn f(a) = a * 2` | |
| `class P:` + `__init__` | `struct P:` with `name: Type` fields and methods | construction is checked |
| `lambda x: x * 2` | `fn(x) => x * 2` | |
| `range(n)` | `0..n` | ranges also work with units: `0s..10s` |
| `None True False` | `none true false` | |
| `raise ValueError(…)` | `error("…")` (stops the program) | `try/except` arrives in 2.0.0 |
| `import numpy as np` | built in: `tensor`, `zeros`, `@`, `sum` … | |

Plus physical units, which Python lacks: `let v = 3[m/s]`, `v * 2s` is `6 m`; `v + 2s` is an error before running.

## 2. Let an AI write it

```text
Write the program in Apsi 1.10.0 (https://apsi.dev), a Python-like language for AI and physics.
Rules: indentation like Python; `let` constants, `var` variables; `fn name(a, b = 2):` or `fn f(x) = expr`;
lambdas `fn(x) => expr`; `struct Name:` with `field: Type = default` and methods taking `self`; ranges `0..n`;
`none/true/false`. No try/except, yield or *args in 1.10.0: use error("message") to stop.
Units: 9.81[m/s^2], 2kg, 3s. Tensors built in: tensor, zeros, randn, @, sum, mean. `use nn` for AI.
Built-ins: https://apsi.dev/docs/1.10.0/reference
```

Then `apsi check program.apsi` points at the exact line of any mistake.

## 3. Write Python and convert it

```bash
apsi convert script.py           # writes script.apsi and checks it
apsi convert train.py --run
```

The 1.10.0 converter translates the core language, classes, numpy, math, random, json and PyTorch models. It keeps
`try` blocks' protected code and turns `except` parts into comments, collects generators into lists and reports
every such place with its line number. Apsi 2.0.0 converts these exactly — see [versions](/docs/1.10.0/versions).

## Freedom, not a cage

Any Python library still works with `pyimport("name")` (`--allow-python`), C libraries with `cload`, and your own
libraries can be shared with `apsi pack && apsi publish`.
