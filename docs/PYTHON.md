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
| `def f(*args, **kwargs)`, `f(*xs, **d)` | the same: `fn f(*args, **kwargs)`, `f(*xs, **d)` |
| `lambda x: x * 2` | `fn(x) => x * 2` |
| `@functools.lru_cache` / `@cache` | `@memo` |
| generators (`yield`, `yield from`) | Apsi generators: values are produced lazily, endless generators work |
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
| `try` / `except A, B as e` / `else` / `finally` | the same statements; kinds keep their Python names, and errors Apsi raises itself (division by zero, missing key …) match them |
| `raise ValueError("msg")`, bare `raise`, own exception classes | `raise ValueError("msg")`, `raise`, an error struct with a `message` field |
| `for … else`, `while … else` | a `broke` flag, so `else` runs exactly when Python runs it |
| `__add__`, `__eq__`, `__lt__`, `__len__`, `__getitem__`, `__str__` … | struct methods `add`, `eq`, `lt`, `len`, `get`, `to_str` … that operators and built-ins call |
| class attributes (`Counter.count += 1`) | a module variable `Counter_count` |
| `collections`, `heapq`, `itertools`, `functools`, `bisect`, `re`, `string`, `copy`, `datetime`, `fractions`, `decimal` … | `pyimport("…")` of the pure standard module — **no `--allow-python` needed** (they cannot touch files, network or processes) |
| `@dataclass` with `field(default_factory=list)` | a fresh `[]` default for every instance |
| `[x for a in A for b in B]`, `{k: v for … for …}` | `chain_lists([...])` / `dict(...)` with the same order |
| `isinstance(x, int)` / `(int, float)` / own classes | `type(x) == "Int"` / `… or …` / `type(x) == "ClassName"` |
| `max(xs, key=f)` / `min(…)` | `sorted(xs, key=f)[-1]` / `[0]` |
| `a & b`, `a \| b`, `a ^ b`, `<<`, `>>` | `band`, `bor`, `bxor`, `shl`, `shr`: integers bit by bit, sets, dict merge, boolean tensors — exactly like Python |
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
| `torch.nn.Linear/Conv2d/LSTM/GRU/Embedding/LayerNorm/BatchNorm/Dropout/MultiheadAttention/Flatten/Sequential` and activations | `use layers`: `linear`, `conv2d_layer`, `lstm_layer`, … `sequential([...])`, `activation("relu")` — with PyTorch's defaults (for example `Conv2d` padding 0, exact `GELU`) |
| whole PyTorch programs (models, training loops, data loaders) | native Apsi — see the next section |
| `torch.nn.functional.relu/gelu/softmax/log_softmax/mse_loss/cross_entropy/dropout` | built-ins and `use nn` |
| **any other library** (pandas, scipy.special, requests, matplotlib, …) | loaded with `pyimport("name")`; the program then needs `--allow-python` (`=all` for libraries that reach the network or the operating system). The note tells you which. |

## PyTorch programs → native Apsi

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

`apsi convert mlp.py` writes:

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

## Checked with real programs

`tests/pycompat/` holds Python programs that use classes with inheritance and operator methods, dataclasses,
custom exceptions, `try/except/else/finally`, generators (also endless ones), closures with `nonlocal`,
`*args/**kwargs`, `collections`, `heapq`, `itertools`, `re`, `json`, string formatting (`%`, `.format`, f-strings),
comprehensions with several `for`, sets, slicing, sorting with keys and numpy. `tests/test_python_compat.py`
converts each one and checks that the Apsi program prints exactly what Python prints. All of them pass.

## What needs a human look

The converter never silently drops behaviour. These general Python constructs have no direct Apsi equivalent and
are reported with their line number so that you can check the result:

- `match` statements (rewrite as `if`/`elif`), `async`/`await`
- `random.shuffle` and other in-place random operations
- `super().method()` other than `super().__init__()`, metaclasses, `__getattr__`
- base classes from other modules (classes derived from `torch.nn.Module` and `Dataset` are converted, see above)
- decorators that have no Apsi equivalent

After converting, run `apsi check file.apsi` (done automatically) — and consider adding units to the physics
parts: `let g = 9.81` → `let g = 9.81[m/s^2]` turns every later formula into a checked one.
