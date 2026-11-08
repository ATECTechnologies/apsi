# From Python to Apsi (`apsi convert`)

`apsi convert` translates Python files — or whole folders — into Apsi, maps the libraries you use onto Apsi's own
built-ins, checks the result and tells you exactly which lines need a human look.

```bash
apsi convert script.py                 # writes script.apsi next to it, then checks it
apsi convert script.py --run           # convert, check and run
apsi convert src/ -o apsi_src/         # a whole project, keeping the folder layout
apsi convert script.py --stdout        # print the Apsi code
```

In VS Code, Cursor and the other VS Code-family editors: right-click a `.py` file (or a folder) →
**Apsi: Convert Python File to Apsi** — the result opens beside the Python file.

Typical output:

```text
✓ demo.py → demo.apsi  (2 notes)
    line 48: generator squares(): values are collected into a list
    line 92: try/except: Apsi stops with a clear error message instead of raising exceptions; …
```

## What is translated

| Python | Apsi |
|---|---|
| `x = 1` (assigned once) / assigned again or `+=` | `let x = 1` / `var x = 1` — chosen automatically |
| a name first assigned inside `if`/`for`/`with`/`try` and used after it | declared before the block (`var x = none`), because Apsi blocks have their own scope |
| a parameter that is reassigned | `var p = p` at the start of the function (Apsi parameters are constants) |
| `def f(a, b=2, *, c=3)` | `fn f(a, b = 2, c = 3)` — one-line bodies become `fn f(a) = expr` |
| `lambda x: x * 2` | `fn(x) => x * 2` |
| `@functools.lru_cache` / `@cache` | `@memo` |
| generators (`yield`) | the values are collected into a list that the function returns |
| `class` with `__init__` | `struct` with the fields found in `__init__`, methods, plus `Name_init` / `Name_new` (calls `Name(...)` become `Name_new(...)`) |
| `@dataclass` | a `struct` with typed fields and defaults (constructed directly) |
| inheritance from a class in the same file | fields and methods are merged; `super().__init__(...)` calls the base's `_init` |
| `@staticmethod` | a function `Class_method` |
| `if`/`elif`/`else`, `while`, `for`, `break`, `continue`, `pass`, `assert` | the same |
| `for i in range(n)` / `range(a, b)` | `for i in 0..n` / `a..b` |
| nested unpacking `for k, (a, b) in …`, `first, *rest = xs` | temporary variable + element access |
| list / dict / set comprehensions, generator expressions | comprehensions (`set([...])` for sets) |
| f-strings (also with format specs) | f-strings; when an expression contains quotes, string concatenation with `format()` |
| `with open(p) as f: f.read()` / `f.write()` / `print(..., file=f)` / `json.load(f)` / `json.dump(x, f)` / `for line in f` | `read_text`, `write_text`, `load_json`, `save_json`, `read_text(p).splitlines()` |
| `try` / `except` / `finally` | the protected code and the `finally` code are kept; `except` blocks become comments (noted) |
| `raise ValueError("msg")` | `error("ValueError: " + str("msg"))` |
| `isinstance(x, int)` / `(int, float)` / own classes | `type(x) == "Int"` / `… or …` / `type(x) == "ClassName"` |
| `max(xs, key=f)` / `min(…)` | `sorted(xs, key=f)[-1]` / `[0]` |
| `a & b`, `a \| b` on booleans or boolean arrays | `where(a, b, false)`, `where(a, true, b)` (element-wise, correct for both) |
| `is None` / `is not None` | `== none` / `!= none` |
| `if __name__ == "__main__":` | the body runs at the end of the file |
| docstrings and comments | comments, in the same places |
| names that are Apsi keywords (`var`, `let`, `fn`, `struct`, `test`, `step`, …) | renamed with a trailing `_` |

## Libraries

| Python | Apsi |
|---|---|
| **numpy**: `array`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace`, `sum`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `cumsum`, `sqrt`, `exp`, `log`, trig and hyperbolic functions, `clip`, `where`, `dot`, `matmul`, `outer`, `cross`, `transpose`, `reshape`, `concatenate`, `stack`, `vstack`, `hstack`, `meshgrid`, `sort`, `roll`, `einsum`, `maximum`, `minimum`, `square`, `sign`, `median` | built-ins of the same meaning (`concatenate` → `concat`, `arcsin` → `asin`, …) |
| `np.random.seed/randn/rand/randint/normal/uniform/random/permutation/choice` | `seed`, `randn`, `rand`, `randint`, … (Apsi's random generator is a different stream: random numbers differ, statistics do not) |
| `np.linalg.inv/det/solve/eig/eigh/svd/norm`, `np.fft.*`, `np.trapz`, `np.interp` | `use linalg`, `use sci` |
| `dtype=np.float32` / `torch.float16` … | `dtype=f32` / `f16` … |
| **math**: `sqrt`, `exp`, `log` (also with a base), trig, `hypot`, `radians`, `degrees`, `factorial`, `isnan`, `isinf`, `fsum`, `pi`, `e`, `tau`, `inf` | built-ins |
| **random**: `random`, `seed`, `randint` (inclusive), `randrange`, `uniform`, `gauss`, `choice` | `rand`, `seed`, `randint(a, b + 1)`, … |
| **json** `loads`/`dumps`/`load`/`dump`, **time** `time`/`perf_counter`/`sleep`, **os.path** `exists`/`join`, **sys** `argv`/`exit`, **statistics** `mean`/`median`/`pstdev` | `json_parse`, `json_str`, `load_json`, `save_json`, `clock()`, `parallel.sleep`, `exists`, `args()`, `exit` |
| **PyTorch** tensors: `torch.tensor/zeros/ones/randn/arange/linspace/matmul/…`, `dim=`, `keepdim=` | the numpy equivalents, `axis=`, `keepdims=` |
| `torch.nn.Linear/Conv2d/LSTM/GRU/Embedding/LayerNorm/BatchNorm/Dropout/MultiheadAttention/Flatten/Sequential` and activations | `use layers`: `linear`, `conv2d_layer`, `lstm_layer`, … `sequential([...])`, `activation("relu")` |
| `torch.nn.functional.relu/gelu/softmax/log_softmax/mse_loss/cross_entropy/dropout` | built-ins and `use nn` |
| **any other library** (pandas, scipy.special, requests, matplotlib, …) | loaded with `pyimport("name")`; the program then needs `--allow-python` (`=all` for libraries that reach the network or the operating system). The note tells you which. |

## What needs a human look

The converter never silently drops behaviour; everything below is reported with its line number:

- `try`/`except` (Apsi reports errors itself and stops), `for … else`, `while … else`
- `*args` / `**kwargs` (become a list/dict parameter), `*` / `**` in calls, the `:=` operator, `match`
- bitwise operators on integers (`^`, `<<`, `>>`, `~`), complex numbers, `async`/`await`
- `random.shuffle` and other in-place random operations, `functools.reduce`
- special methods such as `__add__` (kept as normal methods; operators do not call them), `super().method()`
- base classes from other modules (for example `torch.nn.Module` — rewrite the model with `use layers`; it trains on
  PyTorch again with `fit(…, backend="torch")`, see [`BACKENDS.md`](BACKENDS.md))
- decorators that have no Apsi equivalent

After converting, run `apsi check file.apsi` (done automatically) — and consider adding units to the physics
parts: `let g = 9.81` → `let g = 9.81[m/s^2]` turns every later formula into a checked one.
