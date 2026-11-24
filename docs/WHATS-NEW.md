# What's new in Apsi 2.6.6

2.6.6 is the largest change to Apsi since its first version. It keeps everything that runs today and adds a second
way to run Apsi: **native programs** that are compiled to machine code and carry Apsi's own runtime, so they need no
Python. Apsi also works directly with **C and C++** source files, and the language picks up many habits that Python
programmers expect. Programs written for 2.6.2, 2.6.0, 2.4.4, 2.0.0 and 1.10.0 run unchanged.

> Measurements for 2.6.6 will be published with the final release. The numbers on the
> [benchmarks page](https://apsi.dev/benchmarks) are from 2.6.2 and earlier.

- [A new architecture: native programs](#a-new-architecture-native-programs)
- [C and C++ next to Apsi](#c-and-c-next-to-apsi)
- [Stable by design](#stable-by-design)
- [The language](#the-language)
- [Modules and libraries](#modules-and-libraries)
- [Web server](#web-server)
- [GPU training](#gpu-training)
- [Upgrading](#upgrading)

## A new architecture: native programs

Until 2.6.2, every Apsi program ran on a Python host: the checker verified it, the compiler translated it into
Python, and NumPy did the arithmetic. 2.6.6 adds a compiler that goes all the way down:

```
app.apsi ──► parser ──► checker ──┬──► Python host .................. apsi run   (unchanged)
                                  │
                                  └──► C ──► clang / gcc ──► app.exe   apsi build --native
                                              + Apsi runtime (C)
                                              + standard library (Apsi)
```

```bash
apsi build --native app.apsi        # app.exe on Windows, app on Linux and macOS
./app                               # no Python, no Apsi installation needed to run it
```

**What is inside a native program**

| Part | What it does |
|---|---|
| Apsi's runtime (C) | values, memory management (a collector that runs at safe points), strings, lists, dicts, sets, closures, classes and structs, exceptions, generators, quantities with units |
| Tensors (C) | float64 / float32 / int64 / bool arrays with NumPy's broadcasting and indexing, reductions, matrix products on every core, vectorised `exp` / `tanh` |
| Automatic differentiation (C) | `grad`, `value_and_grad`, gradients of gradients, on numbers, quantities, tensors and trees of parameters |
| GPU (C + PTX) | tensors on NVIDIA cards through the driver alone: element-wise maths, reductions, softmax, matrix products, `where`; cuBLAS and Tensor Cores when CUDA 11–13 is installed |
| Numerics (C) | LU, Cholesky, QR, eigenproblems and SVD; the FFT; incomplete gamma and beta functions; convolution and pooling with gradients |
| Systems (C) | an HTTP/1.1 server on an event loop, SHA-2 / SHA-3 / BLAKE2 / HMAC / scrypt, safetensors and `.npy` files, files and folders |
| Standard library (Apsi) | `linalg`, `stats`, `sci`, `data`, `layers`, `models`, `web`, `crypto`, `asyncio`, `nn`, the functional helpers, the `evolve` integrators, physical constants |

**What runs natively.** The whole language: functions with default, keyword, `*args` and `**kwargs` parameters,
closures, classes with inheritance and `super()`, structs, exceptions with `try` / `except` / `else` / `finally`,
generators with `yield`, `yield from` and `send`, `async fn` and `await`, comprehensions, f-strings with format
specs, `match`, `evolve`, units and shape parameters (`fn f[B](x: Tensor[B, _])`). Modules: the program's own files
and packages, `math`, `random`, `io`, `sys`, `collections`, `nn`, `linalg`, `stats`, `sci`, `data`, `layers`,
`models`, `web`, `crypto`, `asyncio`, `physics.constants`. Native programs print the same text and report errors the
same way as `apsi run`.

```apsi
use layers
use web

let X = load_csv("features.csv")    # [n, 4] measurements
let y = tensor([int(v) for v in read_text("labels.txt").split()])
let model = mlp([4, 32, 3])
let trained = fit(model, model.init(seed=0), X, y, loss="cross_entropy", epochs=200, verbose=false)

let app = server()
app.post("/predict", fn(req) => {"class": int(argmax(predict(model, trained.params, tensor([req.json()["x"]])), axis=1)[0])})
app.run(port=8080)                  # apsi build --native: one executable, no Python
```

**Fast on the machine it runs on.** A native build is compiled for the processor of the computer that builds it
(`-march=native`; set `APSI_PORTABLE=1` for a program that runs on any x86-64). Matrix products use packed,
register-blocked kernels that use 512-bit registers when the CPU has them. Element-wise maths runs in vectorised
loops. A temporary that only an operator made is reused in place, and tensor buffers are recycled. `reshape` returns
a view, as in NumPy.

**Considerate with the computer.** The worker threads leave one core to the operating system and run slightly below
normal priority, so the desktop stays responsive while a program computes. Threads with no work sleep instead of
spinning. Card memory is returned at the next safe point when tensors are no longer used.

**Nothing to install.** On Windows the installer brings a C toolchain (clang and lld from llvm-mingw, x86-64 only).
A source checkout downloads the same toolchain once, the first time it is needed. `apsi toolchain` shows which
compiler is used. Linux and macOS use the system's `clang` or `gcc`.

**Tested against `apsi run`.** Every native test program must print exactly what `apsi run` prints, gradients are
checked against numerical derivatives, and the runtime runs clean under AddressSanitizer and UBSan.

See [Native programs](NATIVE.md#native-programs) for the details.

## C and C++ next to Apsi

`cload` now takes a C or C++ **source file**:

```c
// vecmath.c
#include <stdint.h>
double dot(const double *a, const double *b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
void scale(double *a, int64_t n, double k) { for (int64_t i = 0; i < n; i++) a[i] *= k; }
```

```cpp
// geometry.cpp
#include <cmath>
extern "C" double hyp(double x, double y) { return std::hypot(x, y); }
```

```apsi
let vec = cload("vecmath.c")
let dot = vec.fn("dot", [ptr_f64, ptr_f64, i64], f64)
let scale = vec.fn("scale", [ptr_f64, i64, f64], void)

let a = tensor([1.0, 2.0, 3.0])
print(dot(a, a, len(a)))        # 14.0
scale(a, len(a), 10.0)          # C writes straight into the tensor
print(a)                        # [10., 20., 30.]

let geo = cload("geometry.cpp")
print(geo.fn("hyp", [f64, f64], f64)(3.0, 4.0))   # 5.0
```

- With `apsi run --allow-ffi`, the file is compiled once into a shared library with Apsi's toolchain and cached
  (keyed by its contents).
- With `apsi build --native`, the file is compiled into the program itself. Each `lib.fn(...)` becomes a typed C call
  generated at compile time, so the C compiler handles the calling convention on every platform.
- Tensors are passed as pointers without a copy, so C can write its results into them. Strings are passed as
  `char *`, and quantities as `x.magnitude`, so you choose the unit deliberately.
- Shared libraries (`.dll`, `.so`, `.dylib`) load as before.

See [Using Apsi with other languages](INTEROP.md#c-and-c-source-files).

## Stable by design

Libraries that break between versions are a common way projects in new languages stop working. 2.6.6 prevents this
in four ways:

- **No binary ABI between Apsi libraries.** Apsi modules, including the standard library written in Apsi, are
  compiled from source together with the program. A new Apsi never meets an old compiled library.
- **A pinned toolchain.** The C compiler for native programs is a fixed version shipped with Apsi.
- **One file to ship.** Native Windows programs are linked statically.
- **One checker, two engines.** `apsi run` and native builds share the parser and checker, and native programs are
  tested for the same output.

Fixed while building the native runtime (these also affect correctness in general):

- Element-wise maths on two tensors with the same number of elements but different shapes (for example
  `[N, 1] * [1, N]`) could take the same-shape path.
- Gradients through `concat` and `stack` now reach every joined part.
- `logsumexp` in `apsi run` now carries gradients.

## The language

### `match`

```apsi
fn describe(v):
    match v:
        case 0:
            return "zero"
        case [first, *rest] if len(rest) > 0:
            return f"{first} and {len(rest)} more"
        case {"name": n, **others}:
            return f"named {n} with {others}"
        case Point(x=0, y=y):
            return f"on the y axis at {y}"
        case int() | float() as x:
            return f"the number {x}"
        case none:
            return "nothing"
        case _:
            return "something else"
```

Literals, captures, sequences with `*rest`, maps with `**rest`, class patterns (positional fields of structs and
`__match_args__`, keyword fields), built-in type patterns, alternatives with `|`, `as` and guards.

### Python habits that just work

- Single-quoted strings `'text'`, `'''…'''`, `f'…'`, `b'…'`; quotes inside f-string fields.
- `def name(...):` next to `fn`; `lambda x, y=1: expr` next to `fn(x, y=1) => expr`.
- Python's type names in annotations (`int`, `float`, `str`, `list`, …) and typing names (`List[int]`,
  `Optional[str]`).
- Plain assignment declares a variable (`x = 1`), with Python's scoping. `let` and `var` keep Apsi's rules.
- `import json`, `import numpy as np`, `from dataclasses import dataclass`; imports inside `try` and `if`.
- `None` / `True` / `False`, raw strings `r"\d+"`, `...`, `x: int = 5`, `a, *rest = xs`, `@dataclass`.

### Units on tensors, everywhere

```apsi
let speeds = tensor([10.0, 20.0, 30.0]) * u.km / u.h
print(speeds as [m/s])                 # [2.77778, 5.55556, 8.33333] m/s
let path = tensor([3.0, 4.0]) * u.m
print(norm(path), dot(path, path))     # 5.0 m 25.0 m^2
```

`u.<unit>` gives a unit as a value. `sum`, `mean`, `max`, `norm`, `sqrt`, `stack`, `dot` and the others keep or
combine units, and comparing quantities with different units is an error.

## Modules and libraries

- `use NAME` finds the program's own files (`use helpers`, `use shapes.circle`), and packages are folders with
  `__init__.apsi`, with relative imports.
- `from M import a, b as c`, `__all__`, `apsi doc` for API documentation of your own code, and
  `apsi new NAME --template lib`.
- `apsi build` carries the program's packages, local modules and installed libraries.

## Web server

- The server takes the visitor's address and scheme from trusted proxies (Cloudflare built in) with
  `app.trust_proxy(...)`, and supports `app.allowed_hosts(...)`, `app.force_https()` and
  `app.production(domain=..., cloudflare=true)`.
- `apsi tunnel server.apsi --port 8080` puts a program on the internet through a Cloudflare Tunnel.
- The header-injection guard is stricter.

## GPU training

- Training with batches of many shapes stays inside the card's memory budget.
- `fit_batches(..., verbose=...)` reports where each step's time and memory go.
- `fit_batches(..., prefetch=true)` makes the next epoch's batches while the card trains.
- `apsi gpu test` checks everything Apsi does on the card in one report, and `apsi gpu bench max` finds the largest
  model that trains on it.

## Upgrading

Install 2.6.6 in place of (or next to) 2.6.2 and run `apsi test`. Nothing needs changing. To try native programs:

```bash
apsi build --native examples/08_neural_net.apsi
```

What native programs do not cover yet is listed in [Native programs](NATIVE.md#what-is-not-native-yet). Those
programs keep running with `apsi run`. Every change is listed in the [changelog](../CHANGELOG.md).
