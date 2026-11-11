# From Python to Apsi (`apsi convert`)

`apsi convert` translates Python files — or whole projects — into Apsi **with Python's exact semantics**: the converted
program prints the same text, raises the same errors, trains to the same losses and writes the same files as the
Python program, byte for byte. Large real programs convert in seconds and run immediately.

```bash
apsi convert script.py                 # writes script.apsi (and .apsi files for the modules it imports), checks them
apsi convert script.py --run           # convert, check and run
apsi convert src/ -o apsi_src/         # a whole project, keeping the folder layout
apsi convert script.py --stdout        # print the Apsi code
apsi convert script.py --idiomatic     # map numpy/math/torch onto Apsi's own built-ins (small numeric scripts)
```

In VS Code, Cursor and the other VS Code-family editors: right-click a `.py` file (or a folder) →
**Apsi: Convert Python File to Apsi**.

Typical output:

```text
✓ build_graph_cache.py → build_graph_cache.apsi  (0 notes)
✓ GLA.py → GLA.apsi  (0 notes)
    run with --allow-python=all (it uses subprocess)
✓ gla_brain_dual.py → gla_brain_dual.apsi  (0 notes)

converted 5 files, 0 notes
```

Modules that sit next to the program (`import GLA`, `from gla_brain_dual import X`, packages with `__init__.py`,
`from . import x`) are converted with it and imported as Apsi files, and so are the installed libraries it imports
that are written in Python (next section).

## Libraries: what converts and what does not

Python is needed only while converting. What `apsi convert` produces runs on Apsi.

**Converted completely — your code and every library written in Python.** The libraries a program imports are
looked up in the program's virtual environment (`.venv` or `venv` next to it, made with **any** Python version:
3.8 … 3.14, free-threaded too) and in the Python running the conversion. A library made only of Python files —
`requests`, `urllib3`, `idna`, `certifi`, `pyyaml`'s pure part, most of PyPI — is converted into Apsi with the
program, into `apsi_packages/` next to it, together with its data files (certificates, tables, templates) and its
version information:

```text
✓ fetch.py → fetch.apsi  (0 notes)
✓ library certifi: 2 modules converted into apsi_packages/certifi
✓ library idna: 6 modules converted into apsi_packages/idna
✓ library requests: 18 modules converted into apsi_packages/requests
✓ library urllib3: 36 modules converted into apsi_packages/urllib3

converted 1 file and 62 library modules, 0 notes
```

The converted program then runs without that environment and without `pip`: copy the folder (program plus
`apsi_packages/`) to any computer with Apsi and `apsi run fetch.apsi`. Imports done at run time
(`importlib.import_module`, `__import__`), `importlib.resources` and `importlib.metadata` find the converted
libraries too. `--no-packages` keeps them as Python imports instead.

**Not converted — libraries made of compiled code (C, C++, Rust).** `numpy`, `pyarrow`, `torch`, `pandas`,
`tokenizers` and similar ship machine code (`.pyd` / `.so` files), not Python; there is nothing to translate. Such
an import stays a Python import (`pyimport`), the conversion notes name it, and the program needs that library
installed for Apsi's Python to use it. Where Apsi has its own tool for the same job, the note names it, so the part
that matters can be rewritten on Apsi alone:

| compiled library | Apsi's own tool |
|---|---|
| `numpy` | tensors (`apsi convert --idiomatic` maps small numeric scripts onto them) |
| `torch`, `tensorflow`, `jax` | layers and `fit()`, GPU training without PyTorch ([AI](AI.md), [GPU](GPU.md)) |
| `transformers` | `use hub`: models from Hugging Face without Python ([Hub](HUB.md)) |
| `datasets`, `pyarrow` | `use hub` (`hub.download(repo, file, kind="dataset")`), `use data`, `use dataframe` (CSV, JSON, Parquet) |
| `pandas`, `polars` | `use dataframe` ([DataFrame](DATAFRAME.md)) |
| `scipy` | the `sci` built-ins (integrate, minimize, fft, curve_fit) ([Science](SCIENCE.md)) |
| `requests` (when kept), `aiohttp` | `use web` (`http_get`, `http_post_json`) |
| `safetensors` | `use models` ([model files](MODEL-FILES.md)) |

**So, for data collection and AI training programs** — the ones that lean on `datasets`, `pyarrow`, `torch` and
`transformers` — the conversion is complete for the program's own logic and its pure-Python libraries, and partial
for the compiled ones: those keep running through Python until that part uses Apsi's tools above. `--idiomatic`
does that rewrite automatically for numeric and PyTorch model code; for the rest the notes say which tool to use.

## How exact is exact?

The converted file starts with `use pycompat`. That line gives the file Python's object model and built-ins while the
text stays Apsi (`fn`, `let`/`var`, `none`/`true`/`false`, `import "x.apsi"`). Concretely:

| Python | Apsi (with `use pycompat`) | behaves |
|---|---|---|
| `class Net(nn.Module):` with `__init__`, `super().__init__()`, `forward`, `@property`, `@staticmethod`, `@classmethod`, `__slots__`, dunder methods, metaclasses, class decorators, `__init_subclass__` | `class Net(nn.Module):` with `fn __init__(self, …):` … | a real class: inherits from any library class, attribute hooks (`__setattr__`, `__getattr__`) run, pickling works |
| `@dataclass(frozen=True, order=True)` with `x: int = 0`, `field(default_factory=list)` | the same decorator and `x: int = 0` fields | Python's dataclass (`asdict`, `fields`, `replace`, frozen errors …) |
| `with open(p) as f, lock:` / `async with` | `with open(p) as f, lock:` | `__enter__`/`__exit__`, exceptions suppressed when `__exit__` returns true |
| `try / except (A, B) as e / else / finally`, `raise X from Y`, own exception classes | the same | the real exception object is bound (`e.args`, `e.__cause__`, `isinstance`) |
| generators, `x = yield v`, `yield from`, generator expressions | the same | lazy, `send()` / `throw()` / return values |
| `global`, `nonlocal`, closures, default values evaluated once, `*args` (a tuple), keyword-only and positional-only parameters | the same | Python's scoping: one namespace per function, loop variables live after the loop |
| `a, *rest = xs`, `for i, (a, b) in …`, `a = b = 0`, `x += [1]` (in place) | the same | |
| `{1, 2}`, set/dict comprehensions, `{**a, **b}`, `[*xs, y]` | the same | |
| `is`, `is not`, `a < b < c` (b evaluated once), `| & ^ << >> ~`, `:=` | the same | |
| `for … else`, `while … else`, `del`, `assert`, `pass` | the same | |
| f-strings with `!r`, `{x=}`, nested specs `{v:{w}.{p}f}` | f-strings, or `format()` calls when an expression contains quotes | Python's `format()` |
| `print`, `str`, `repr`, `len`, `range`, `map`, `zip`, `sorted`, `int`, `isinstance`, `super`, … | Python's own built-ins | `print([None, True])` prints `[None, True]` |
| `import x.y as z`, `from x import y` | `let z = pyimport("x.y")`, `let y = pyfrom("x", "y")` | submodules found as Python does |
| `if __name__ == "__main__":` | the same | `__name__` is `"__main__"` only for the program you run; imported files get their module name |
| docstrings | kept (they are docstrings again: `help()` and `__doc__` work) | |
| comments, blank lines, statement order | kept | |
| `match` / `case` with literal, capture, sequence (`[x, *rest]`), mapping (`{"k": v, **rest}`), class (`Point(x=0, y=y)`), or (`1 \| 2`), `as` and guard patterns | the same | Python's matching rules, including `__match_args__` |
| `except*` and exception groups | the same | |
| `except KINDS as e:` where `KINDS` is a variable or call | an `isinstance` test on the caught error | the same handler runs |
| `async def`, `await`, `async for`, `async with`, async comprehensions | the same | runs on asyncio |
| `[b.v for b.v in xs]`, `for st.line, x in …` (attribute and index targets) | the same | |
| `type Pair = tuple[int, int]`, `def first[T](xs: list[T]) -> T`, `class Box[T]:` | `TypeAliasType`, `TypeVar` and `Generic[T]` objects | |
| `self.__secret` inside `class Vault` | the same | stored as `_Vault__secret` (Python's name mangling) |
| `from math import *`, `from helper import *` | the module's public names (its `__all__` when it has one) | |
| `a[1:2, ...]`, `a[*idx]`, `a[1,]` | the same | |
| reading a name before it is bound | the same | `UnboundLocalError` / `NameError`, as in Python |
| names that are Apsi keywords (`let`, `var`, `fn`, `struct`, `use`, `none`, `true`, `false`, `match`, `type` …) | written in backticks: `` `let` ``, `` `fn` `` | the same name — `getattr`, keyword arguments and pickling see `let`, not a renamed copy |

Converted programs run at the speed of the Python program — attribute access, calls and loops compile to the same
Python bytecode — and keep every Apsi feature: `use rt`, `use hp`, `@native` functions, units, the checker.
Measured on 2 cores (`benchmarks/python/cmp.sh`, best of five, interleaved):

| program | CPython 3.13 | converted, `apsi run` |
|---|---|---|
| objects: trees, dataclasses, dicts, `Counter` | 0.295 s | **0.271 s** (8 % faster) |
| strings: regex, `split`, JSON, sets | 0.271 s | 0.273 s |
| n-body, 200,000 steps (floats, lists) | 0.928 s | 0.941 s |

Programs that create many objects are faster because Apsi freezes everything loaded before the program starts (the
cycle collector never scans it again) and collects the youngest generation every 50,000 allocations instead of
2,000; reference counting is untouched, so objects are still freed the moment they are no longer used
(`APSI_GC=default` restores Python's settings). Pure arithmetic in Python loops runs on the same bytecode
interpreter as CPython, so it runs at the same speed; to go far beyond it, put `@native` on the hot function
(machine code through LLVM, see [native code](NATIVE.md)).

Conversion itself: the 23,769-line program and three of its modules (29,022 lines) convert and check in 4.5 s.

## Permissions — what a converted program may touch

Apsi's sandbox stays on. Without any flag a converted program can use:

* every **pure** standard module (`collections`, `itertools`, `functools`, `re`, `json`, `dataclasses`, `enum`,
  `typing`, `math`, `random`, `decimal`, `time`, `argparse`, `threading`, `queue`, `datetime`, `hashlib` …)
* **`sys`, `os`, `os.path`, `pathlib` and `numpy` as permission-checked stand-ins**: the same names and behaviour,
  but file access asks `--allow-read`/`--allow-write`, environment variables ask `--allow-env`, and the parts that
  would escape the sandbox (`os.system`, `subprocess`, `sys.modules`, `numpy.ctypeslib`, `np.load(allow_pickle=True)`,
  `array.tofile`) are not there
* `open()` checked against `--allow-read` / `--allow-write`

Libraries converted into `apsi_packages/` are Apsi files and follow the same rules (a converted `requests` uses
`socket` and `ssl`, so a program using it runs with `--allow-python=all`). Compiled
libraries kept as Python imports need `--allow-python` (for example `torch`, `scipy`, `pandas`), and libraries that reach the operating
system or the network (`subprocess`, `socket`, `http`, `ctypes`, `multiprocessing`, `shutil` …) need
`--allow-python=all`; with `=all` every module is the real one and nothing is filtered. `apsi convert` prints the flag
a file needs, and `apsi run` says so when one is missing:

```bash
apsi run train.apsi --allow-python                     # torch, numpy, scipy …
apsi run GLA.apsi --allow-python=all -- train --epochs 3   # arguments after -- go to the program (sys.argv)
```

Introspection that leads to interpreter internals (`__globals__`, `__subclasses__`, `__builtins__`, frame objects,
`object.__getattribute__`, `__dict__` of library classes, `operator.attrgetter("__…")`, `eval`/`exec`/`__import__`)
is refused unless the program was started with `--allow-python=all`.

## Checked with real programs

* **The Python standard library itself.** `tests/test_stdlib_convert.py` copies 34 pure-Python standard modules next to
  a driver program — `_pydecimal` (6,357 lines of arbitrary-precision decimal arithmetic), `_pydatetime`, `argparse`,
  `configparser`, `tomllib`, `json` (a package with relative imports), `pickle`, `dataclasses`, `enum`, `random`,
  `statistics`, `fractions`, `difflib`, `ipaddress`, `textwrap`, `pprint`, `calendar`, `shlex`, `graphlib`, `heapq`,
  `bisect`, `html.parser`, `urllib.parse`, `base64`, `fnmatch`, `string`, `copy`, `contextlib`, `operator`, `reprlib`,
  `numbers`, `_py_abc`, `colorsys`, `keyword` — converts all of them with the program, runs it, and requires **byte-identical output** to
  CPython running the original files. About 38,000 lines of metaclasses, descriptors, generators, `__slots__`, name
  mangling, `__class_getitem__`, exception chaining and 50-digit arithmetic.
* `tests/pycompat/` — programs with classes, inheritance from library classes, dataclasses, enums, abstract base
  classes, context managers, generators with `send`, closures, threads, `collections`, `heapq`, `functools`, `re`,
  `json`, numpy and every operator. `tests/test_python_compat.py` converts each one and requires **byte-identical
  output**.
* A 23,769-line AI training program with its four companion modules (graph building for 30 languages, web search,
  safety filter, a dual-brain model) converts in about 3 seconds and checks cleanly; its command-line tools, its
  multi-process graph-cache builder (results pickled across worker processes, resumable output) and its safety filter
  produce byte-identical files and output to Python.
* A PyTorch-style training program (an `nn.Module` hierarchy that registers parameters through `__setattr__`,
  autograd, Adam with weight decay, cosine schedule, early stopping, checkpoint) trains to bit-identical losses.

## Coverage

Every statement and expression of Python 3.13 converts: there is no "kept as a comment" list any more. Two things
differ by design and are not part of a program's behaviour:

* **Tracebacks** of an uncaught error name the `.apsi` file and line (`train.apsi:212:1: TypeError: …`) instead of
  the `.py` file; the error type and message are Python's. Programs that catch errors see the real exception objects.
* **The sandbox.** Without `--allow-python=all`, introspection that leads to interpreter internals is refused (see
  above). With it, `eval`, `exec`, `globals()`, frames and every module behave exactly as in Python.

A converted module is reused while it is newer than its `.py` file and was written by the same Apsi version; a new
Apsi version converts it again.

## `--idiomatic`: PyTorch programs → native Apsi (no Python, no PyTorch)

A PyTorch training script becomes an Apsi program that runs **without Python and without PyTorch**. The converter
targets `use torch`, Apsi's own implementation of the PyTorch API: the same layer names, constructor arguments,
parameter names (`fc1.weight` stored as `[out, in]`), buffers, losses, optimizers and learning-rate schedulers,
implemented natively on Apsi's automatic differentiation. Nothing is approximated and nothing is left for you to
finish by hand.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, d_out)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        return self.out(self.drop(F.relu(self.bn(self.fc1(x)))))

X = torch.randn(256, 4)
y = (X[:, 0] > 0).long()
model = MLP(4, 32, 2)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
for epoch in range(200):
    opt.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
model.eval()
print(loss.item())
torch.save(model.state_dict(), "mlp.safetensors")
```

`apsi convert mlp.py --idiomatic` writes:

```apsi
# Converted from mlp.py by `apsi convert`.
use models as models
use torch as torch

var _aux3 = none
var _grads2 = none
var _loss_of1 = none
var loss = none

fn MLP(d_in, hidden, d_out):
    let fc1 = torch.Linear(d_in, hidden)
    let bn = torch.BatchNorm1d(hidden)
    let out = torch.Linear(hidden, d_out)
    let drop = torch.Dropout(0.1)
    fn init_(seed = 0) = record(fc1=fc1.init(seed=seed * 1009 + 1), bn=bn.init(seed=seed * 1009 + 2), out=out.init(seed=seed * 1009 + 3), drop=drop.init(seed=seed * 1009 + 4))
    fn forward_(p, x, train = false) = out(p.out, drop(p.drop, torch.relu(bn(p.bn, fc1(p.fc1, x, train=train), train=train)), train=train), train=train)
    return torch.Module(init_, forward_, "MLP", record(fc1=fc1, bn=bn, out=out, drop=drop))
let X = randn(256, 4)
let y = tensor(X[:, 0] > 0, dtype=i64)
let model = MLP(4, 32, 2)
var model_params = model.init(seed=0)
let opt = torch.AdamW(lr=0.01)
let sched = torch.CosineAnnealingLR(opt, T_max=200)
let loss_fn = torch.CrossEntropyLoss(label_smoothing=0.1)
for epoch in 0..200:
    fn _loss_of1(_p):
        let logits = model(_p, X)
        let loss = loss_fn(logits, y)
        return loss
    loss, _aux3, _grads2 = torch.value_and_grad_aux(_loss_of1)(model_params)
    _grads2 = torch.clip_grad_norm(_grads2, 1.0)
    model_params = opt.step(model_params, _grads2)
    sched.step()
model.eval()
print(loss)
models.save_safetensors("mlp.safetensors", torch.state_dict(model, model_params))
```

The saved file is a standard safetensors file with PyTorch's names (`bn.running_mean`, `fc1.weight`, …):
`model.load_state_dict(load_file("mlp.safetensors"), strict=True)` in PyTorch accepts it unchanged, and the other
direction works the same way.

### What is converted exactly

| PyTorch | Apsi (`use torch`) |
|---|---|
| `class Net(nn.Module)` with `__init__`, `forward` and helper methods; sub-modules of your own classes; `nn.Sequential`, `nn.ModuleList` | `fn Net(...)` returning `torch.Module(init, forward)`; parameters are an explicit tree (`p.fc1.weight`) |
| `nn.Linear`, `Bilinear`, `Conv1d`, `Conv2d` (stride, padding including `"same"`, dilation, groups), `MaxPool1d/2d`, `AvgPool1d/2d` (`ceil_mode`, `count_include_pad`), `AdaptiveAvgPool1d/2d`, `AdaptiveMaxPool1d/2d`, `Embedding` (`padding_idx`), `Dropout`, `Flatten`, `Unflatten`, `Identity` | the same layers with the same arguments |
| `BatchNorm1d/2d` (running statistics, `momentum`, `momentum=None`, `track_running_stats`), `LayerNorm`, `RMSNorm`, `GroupNorm` | identical training and evaluation behaviour, buffers included in `state_dict` |
| `LSTM`, `GRU`, `RNN` (`num_layers`, `bidirectional`, `batch_first`, `dropout`, initial states) | returns the same `(output, (h_n, c_n))` tuples |
| `MultiheadAttention` (`kdim`/`vdim`, `key_padding_mask`, boolean or float `attn_mask`, `need_weights`, `average_attn_weights`, `is_causal`), `TransformerEncoder(Layer)`, `TransformerDecoder(Layer)`, `Transformer` (`norm_first`, activation) | same results, including attention weights |
| all activation modules and `F.` functions (`gelu` exact or `tanh`, `silu`, `mish`, `softplus`, `glu`, …), `F.pad`, `F.normalize`, `F.one_hot`, `F.scaled_dot_product_attention`, `F.cosine_similarity` | same names under `torch.` |
| `CrossEntropyLoss` (`weight`, `ignore_index`, `label_smoothing`, probability targets, N-d inputs), `NLLLoss`, `MSELoss`, `L1Loss`, `SmoothL1Loss`, `HuberLoss`, `BCELoss`, `BCEWithLogitsLoss` (`pos_weight`), `KLDivLoss`; every `reduction` | exact formulas |
| `torch.optim.SGD` (momentum, Nesterov), `Adam` (`amsgrad`), `AdamW`, `Adamax`, `RMSprop`, `Adagrad`, `NAdam`, `RAdam` | identical update rules |
| `lr_scheduler.StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `LinearLR`, `ConstantLR`, `LambdaLR`, `MultiplicativeLR`, `PolynomialLR`, `OneCycleLR` (with momentum cycling), `CyclicLR`, `ReduceLROnPlateau`, `SequentialLR`, `ChainedScheduler` | identical learning-rate sequences; `sched.step()` is kept |
| `opt.zero_grad(); loss.backward(); clip_grad_norm_ / clip_grad_value_; opt.step()` | one `value_and_grad_aux` call, clipping, `opt.step(params, grads)`; values computed in the forward pass (logits, accuracy) remain available |
| `model.train()` / `model.eval()`, `torch.no_grad()`, `.to(device)`, `.item()`, `.detach()` | kept or removed as appropriate |
| `DataLoader(TensorDataset(...))`, `DataLoader` over your own `Dataset` class (`__len__`, `__getitem__`), `batch_size`, `shuffle`, `drop_last`, `collate_fn` | `torch.DataLoader` with the same behaviour |
| `torch.utils.checkpoint.checkpoint` | `torch.checkpoint` — recomputes activations in the backward pass to save memory |
| `model.state_dict()`, `load_state_dict(...)`, `torch.save` / `torch.load`, `safetensors.torch.save_file` / `load_file` | `torch.state_dict`, `torch.load_state_dict`; files ending in `.safetensors` stay safetensors, others become `.apsim` |
| tensor methods and functions: `view`, `reshape`, `flatten`, `contiguous`, `size`, `float`, `long`, `sum(dim=)`, `mean`, `argmax`, `max(dim=)`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `clamp`, `exp`, `matmul`, `torch.cat`, `torch.stack` | the matching Apsi functions |

**Verified exactness.** The test suite (`tests/test_convert_torch.py`, `tests/test_torch_compat.py`) runs every layer,
loss, optimizer and scheduler above next to real PyTorch with identical weights and compares outputs, gradients and
learning-rate sequences. A complete training run — convolution, BatchNorm, LSTM, multi-head attention, AdamW,
cosine annealing, gradient clipping and label smoothing — ends with the same weights as PyTorch to within 1e-9.

Random initialisation uses Apsi's generator, so a model trained *from scratch* starts from different random weights
than in PyTorch. To reproduce a PyTorch run exactly, load its initial `state_dict`.

### Memory

- `torch.checkpoint_layer(layer, params, x)` or `torch.checkpoint(fn, x)` stores only the block inputs and
  recomputes the rest during the backward pass: about a third of the peak memory for about twice the compute.
- Create parameters with `dtype=f32` (or `fit(…, precision="fp32")`); gradients and Adam state then stay float32.
- `DataLoader` builds one batch at a time, so datasets larger than memory can be streamed from a `Dataset` class.


After converting with `--idiomatic`, run `apsi check file.apsi` (done automatically) — and consider adding units to
the physics parts: `let g = 9.81` → `let g = 9.81[m/s^2]` turns every later formula into a checked one.
