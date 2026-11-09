# You already know Apsi — three ways to write it

Apsi is designed so that **you do not have to learn a new language before you are productive**. If you can read
Python, you can read Apsi; if you can write Python, there are three ways to get working Apsi code today:

| Way | Best when | What you do |
|---|---|---|
| **1. Write it like Python** | you are starting something new | write the Python you know, with four small changes (below) |
| **2. Let an AI write it** | you want speed | give an AI assistant the short brief below and ask for Apsi |
| **3. Write Python and convert** | you already have Python code | `apsi convert script.py` — then run it |

All three end in the same place: a program that is **checked before it runs** (units, types, shapes, names) and
runs on Apsi's fast engines.

## 1. Write it like Python

Almost everything is the same: indentation, `if/elif/else`, `for`/`while`, `break`/`continue`, lists, dicts,
tuples, slicing, comprehensions, f-strings, `try/except/finally`, `raise`, `yield`, lambdas, `*args/**kwargs`,
string and list methods. The differences that matter:

| Python | Apsi | Why it is better |
|---|---|---|
| `x = 1` | `let x = 1` (never changes) or `var x = 1` (changes) | accidental reassignment is caught before running |
| `def f(a, b=2):` | `fn f(a, b = 2):` — or one line: `fn f(a) = a * 2` | short functions read like formulas |
| `class P:` + `__init__` | `struct P:` with `name: Type` fields | fields are declared once; construction is checked |
| `lambda x: x * 2` | `fn(x) => x * 2` | |
| `range(n)` | `0..n` (or `range(n)`, both work) | ranges also work with units: `0s..10s` |
| `None True False` | `none true false` | |
| `import numpy as np` | built in: `tensor`, `zeros`, `@`, `sum`, `linspace` … | no imports for maths, tensors, AI, ODEs |

And one thing Python does not have at all — **physical units**: `let v = 3[m/s]`, `let t = 2s`, `v * t` is `6 m`;
`v + t` is reported as an error before the program runs.

```apsi
# Python habits work: classes, errors, generators, comprehensions
struct Account:
    owner: Str
    balance: Float = 0.0
    fn withdraw(self, amount):
        if amount > self.balance:
            raise ValueError(f"only {self.balance} left")
        self.balance -= amount

let acc = Account("ana", 100.0)
try:
    acc.withdraw(250.0)
except ValueError as e:
    print("refused:", e.message)
```

## 2. Let an AI write it

Modern assistants (Claude, ChatGPT, Gemini, Copilot, Cursor …) write good Apsi when they get this brief first.
Copy it, then describe your program:

```text
Write the program in Apsi (https://apsi.dev), a Python-like language for AI and physics.
Rules: indentation blocks like Python; `let` for constants and `var` for variables; `fn name(a, b = 2):`
or one-line `fn f(x) = expr`; lambdas `fn(x) => expr`; `struct Name:` with `field: Type = default` and methods
taking `self`; ranges `0..n`; `none/true/false`; try/except/finally, raise, yield, *args/**kwargs as in Python.
Physical quantities carry units: 9.81[m/s^2], 2kg, 3s; annotate with Length, Mass, Time, Velocity, Energy …
Tensors are built in: tensor, zeros, randn, @, sum, mean; `use nn`, `use rl`, `use torch` for AI.
Built-ins and modules: see https://apsi.dev/docs/2.4.4/reference . Prefer built-ins over Python libraries.
```

The site publishes [`/llms.txt`](/llms.txt) and [`/llms-full.txt`](/llms-full.txt) — the whole documentation in
plain text — so assistants that browse can read everything themselves. After the AI answers, run
`apsi check program.apsi`: the checker points at the exact line of anything the AI got wrong, and you (or the AI)
fix it in seconds.

## 3. Write Python and convert it

```bash
apsi convert script.py                 # writes script.apsi next to it and checks it
apsi convert src/ -o apsi_src/         # a whole folder
apsi convert train.py --run            # convert, check and run
```

The converter understands Python's meaning, not only its spelling: classes become structs with their methods,
`__add__`/`__eq__`/`__len__`/`__str__` become the struct methods that Apsi's operators call, exceptions keep their
kinds (`except KeyError` still catches a missing key), generators stay lazy, `for … else` keeps its rule, numpy
becomes Apsi tensors and **whole PyTorch training programs become native Apsi that runs without Python or
PyTorch** — matching PyTorch's numbers to 1e-9. Pure standard modules (`collections`, `heapq`, `itertools`, `re`,
`datetime` …) are used as they are. Every converted test program in `tests/pycompat/` prints exactly what Python
prints. Details and the complete mapping: [From Python and PyTorch](/docs/2.4.4/python).

## Freedom, not a cage

Apsi gives you safe defaults — units, checks, a sandbox — and then gets out of the way:

- **Any Python library** still works: `let pd = pyimport("pandas")` (run with `--allow-python`).
- **C and C++** libraries: `cload("libfoo.so")` and call functions directly ([interop](/docs/2.4.4/interop)).
- **Your own operators, iterators, printing** on your types (struct methods `add`, `iter`, `to_str` …).
- **Your own libraries** for everyone: `apsi pack && apsi publish` ([libraries](/docs/2.4.4/libraries)).
- **Servers, tools, data pipelines, simulations, models** — the same language for all of them.
