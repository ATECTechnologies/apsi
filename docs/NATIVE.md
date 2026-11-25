# Native code

Apsi has two ways to turn a program into machine code:

- **Native programs** (`apsi build --native`, new in 2.6.6): the whole program is compiled into an executable with
  Apsi's own runtime. It contains no Python.
- **`@native` functions** inside `apsi run`: single numeric functions compiled with LLVM while the rest of the
  program runs on the Python host.

## Native programs

```bash
apsi build --native app.apsi             # -> app.exe (Windows) / app (Linux, macOS)
apsi build --native app.apsi -o dist     # into another folder
apsi build --native app.apsi --emit-c    # also keep the generated C next to the program
apsi toolchain                           # which C compiler is used
```

The program is checked first, exactly as `apsi run` checks it, and nothing is built if the checker finds an error.
It is then translated to C and compiled together with Apsi's runtime:

```
app.apsi ─► parser ─► checker ─► C ─► clang / gcc ─► app.exe
                                       ▲
              Apsi runtime (C): values, memory, exceptions, generators, units,
              tensors, automatic differentiation, GPU
              standard library: linalg, stats, sci, data, layers, models, web, crypto,
              asyncio, nn, and the everyday built-ins
```

The runtime is compiled once per compiler and machine and kept (in `~/.apsi/cache/rt`), so after the first build a
program's build compiles only the program.

### What you get

- **One file.** On Windows the program is linked statically and needs nothing beside it. On Linux and macOS it
  uses the system's C library.
- **The same behaviour as `apsi run`.** Output, number formatting, error messages with their line and column, and
  the functions named in the traceback all match.
- **Compiled for this machine.** Programs are built for the processor of the computer that builds them. To build one
  that runs on any x86-64 computer, set `APSI_PORTABLE=1`.
- **Cross-building.** `--target x86_64-pc-windows-msvc` (or another clang target triple) builds for another system.

### How values live

Every value a function holds is kept in a slot of its frame, and frames are linked, so the collector can run at any
safe point: on entry to a function and at the end of each loop iteration. It frees what nothing reaches any more,
including tensor buffers and GPU memory. A variable captured by an inner function lives in a cell. A loop variable
captured by a `fn(...) => ...` gets a fresh cell on each iteration, as in `apsi run`. Exceptions unwind frames and
handlers together. Generators run on stacks of their own (fibers on Windows), and the collector sees their
suspended frames.

### Tensors and the GPU

Native tensors are float64, float32, int64 or bool arrays with NumPy's broadcasting, indexing (slices, masks,
position lists, blocks of positions) and printing. `reshape` returns a view. Matrix products use packed kernels on
every core. The worker threads leave one core to the operating system and run slightly below normal priority.

```apsi
use sys
if gpu_available():
    let a = to_gpu(randn(1024, 1024))
    let b = to_gpu(randn(1024, 1024))
    print(sum(a @ b), gpu_info())
```

On the card, element-wise maths, reductions, softmax, permutes, `where`, gathers, scatters, optimizer steps and
matrix products run as Apsi's own kernels, built into the program as PTX. Only the NVIDIA driver is needed, and the
driver compiles the kernels for whatever card it finds. Operations the card does not have yet run on the processor
and the result returns to the card. `to_host` brings tensors and whole parameter trees back.

When cuBLAS is installed (any CUDA 11, 12 or 13: a CUDA Toolkit, or the `nvidia-cublas-cu12` / `-cu13` wheels in a
`.venv`), matrix products go to it. `APSI_GPU_MATH=tf32`, `bf16` or `f16` lets it use Tensor Cores; the default
stays exact float32. `apsi cuda` shows what is found and `apsi cuda install` adds cuBLAS (see
[GPU.md](GPU.md#cuda-libraries)). Freed card memory is kept for reuse (up to a quarter of the card, or
`APSI_GPU_CACHE_MB`) and given back when an allocation would fail.

### Programs converted from Python

`apsi convert prog.py` writes `use pycompat` at the top; such a program builds natively too and prints what Python
prints: `None` / `True`, `'strings'` inside lists, CPython's order for sets of Ints, `%`-formatting, `str.format`,
`type(x)` as the type, `@property` (with setters), `@staticmethod`, `@classmethod`, `super().__init__()`, `__name__`,
`frozenset`, `issubclass`, and native stand-ins for `math`, `collections` (Counter, defaultdict, deque, namedtuple,
OrderedDict), `functools`, `itertools`, `typing`, `dataclasses`, `copy`, `sys`, `random` (CPython's Mersenne Twister:
the same numbers as Python for a seed), `heapq`, `bisect`, `string`, `operator`, `json`, `time`, `os` / `os.path`,
`abc` and `statistics`.

### Numbers

Ints have no size limit, as in `apsi run`: a result beyond 64 bits becomes an Int of any size (`2 ** 200`,
`math.factorial(100)`, `int("…")` of a long text), and goes back to the fast 64-bit form when it fits again. `use hp`
gives decimals of any precision with Python's decimal arithmetic.

### Gradients

`grad`, `value_and_grad` and `stop_gradient` work on numbers, quantities with units, tensors and trees of parameters
(lists, dicts, records, structs). A gradient can itself be differentiated, which physics-informed networks need. The
backward pass walks its graph iteratively, so very deep graphs cannot overflow the stack.

### C and C++

`cload("file.c")` or `cload("file.cpp")` compiles the source file into the program, and each
`lib.fn("name", [types], result)` becomes a typed C call. See [INTEROP.md](INTEROP.md#c-and-c-source-files).

### The standard library in native programs

| Module | In native programs |
|---|---|
| `linalg` | solve, inv, det, slogdet, cholesky, qr, eig, eigh, eigvals, svd, svdvals, pinv, lstsq, matrix_rank, cond, norm, trace, matrix_power, kron, cross |
| `stats` | every distribution, test, model and correction of `apsi run`, with the same results (the test suite compares them to 6 digits) |
| `sci` | fft, ifft, rfft, irfft, fftfreq, power_spectrum, trapz, cumtrapz, integrate, interp, find_root, bisect, minimize (L-BFGS, Adam), curve_fit, solve_ivp (TR-BDF2, Dormand–Prince), einsum |
| `data` | the pipelines (sources, map, filter, batch, shuffle, window, shard, interleave, cache…), `.npy` files, glob patterns |
| `layers` | dense, mlp, cnn, lstm / gru layers, attention (grouped-query, RoPE, QK norm), transformer blocks, encoders and language models, normalisations, residual and parallel blocks, `fit`, `predict`, `summary` |
| `models` | safetensors (BF16 / F16 decoded on load) and `.apsim` checkpoints with their SHA-256, readable by `apsi run` and back |
| `web` | `server()` with routes, hooks, CORS, rate limits, static files, JWT and API keys, on Apsi's own HTTP/1.1 event loop |
| `crypto` | SHA-2, SHA-3, BLAKE2, SHA-1, MD5, HMAC, scrypt and PBKDF2 password hashes, Base64, secure random numbers |
| `asyncio` | `import asyncio` / `pyimport("asyncio")`: `async fn`, `await`, run, sleep, gather, create_task, wait_for, Event, Lock, Queue |
| `special` | gamma, beta, incomplete gamma and beta with their inverses, erf / erfinv, Bessel J and I, elliptic K and E, Legendre, Hermite, Laguerre, Chebyshev, spherical harmonics, zeta, E1 |
| `signal` | Butterworth filters as second-order sections, `filtfilt` (zero phase, as SciPy), notch, windows, convolution, peaks, Welch spectra, spectrograms, Savitzky–Golay, resampling, Hilbert envelope |
| `config` | TOML (read by Apsi's own parser), JSON, INI and `.env` settings, the environment and `--set` arguments, defaults with types, `${...}` references |
| `physics` | constants, the formulas with their units, `fields` (gradient, laplacian), `meshgrid` |
| `pde` | tridiagonal solves, heat (Crank–Nicolson, ADI), waves, advection, fast Poisson solvers, pseudo-spectral Navier–Stokes, CG, BiCGSTAB |
| `mechanics` | integrators, Hamiltonian / Lagrangian flows (by autodiff), N-body, orbits, rigid bodies, normal modes, beams, trusses |
| `precision` | FP16, BF16, FP8 (E4M3 / E5M2), FP6, FP4, INT8 / INT4 / UINT8: bit-exact rounding, quantised tensors, `round_to` with straight-through gradients (its names work without `use`, as in `apsi run`) |
| `lab` | seeds, provenance, experiment logs, `format_measurement`, Markdown / LaTeX / CSV tables |
| `sys`, `gpu` | `device`, `to_device`, `hardware`, `cpu_threads`, GPU info |
| `tokens` | `pack` / `unpack`, `dedupe` / `expand`, `diff` / `patch`, `best`, `count` (the estimate), `savings` |
| `dataframe` | typed columns, expressions, filter / select / group by / join / sort / window, CSV (parsed in C), JSON, the same table printout |
| `db` | SQLite through the SQLite library the system already has (Windows 10+, Linux, macOS; `APSI_SQLITE` names another); PostgreSQL (the wire protocol: SCRAM-SHA-256 / md5 / password, prepared statements, arrays, json, bytea, LISTEN / NOTIFY) and Redis (pipelines, pub/sub, every command by name) written in Apsi on native TCP connections. `numeric` comes back as a float and dates as text |
| `net`, `http_get`, `http_post_json` | HTTP and HTTPS requests: WinHTTP on Windows; sockets and the system's OpenSSL on Linux and macOS (`HTTPS_PROXY` / `NO_PROXY`, `APSI_CA_FILE`); redirects, `HTTPError` |
| `quantum` | grid Schrödinger solver, split-operator evolution, operators, `expm`, spins, qubits and gates, density matrices, entanglement, hydrogen (complex128 tensors; `measure` draws from the native generator) |
| `rl` | CartPole, GridWorld and bandit environments, discounted returns, GAE, PPO, DQN, REINFORCE, evaluation, preference / DPO / IPO losses, KL-penalised rewards, GRPO advantages, PPO clip loss, sequence log-probabilities, MCTS, beam search, best-of-n |
| `torch` | PyTorch's layers (Linear … Conv1d/2d with stride, padding, dilation and groups, pooling with ceil mode, batch / layer / group / RMS norm with running statistics, LSTM / GRU / RNN, MultiheadAttention, Transformer encoder / decoder), the functional forms, every loss with weights / ignore_index / label smoothing, SGD … RAdam with PyTorch's update rules, all the learning-rate schedulers, DataLoader, state_dict / load_state_dict; `init(seed)` draws from the native generator |
| `parallel` | tasks (`spawn`, `wait_all`, `wait_any`), channels, locks, atomics, `parallel_for` / `parallel_reduce`, `sleep`, `set_threads`, `bench`: tasks take turns on stacks of their own (tensor kernels inside them use every core) |
| `ffi` | `ffi.cload` and the C type names, the same as the built-in `cload` |
| `scale` | training plans for 1B … 2T-parameter models (TP × PP × DP layout, recomputation, memory per GPU, GPUs, days), inference memory, the GPU table |
| `hp` | decimals of 2 to 100 000 significant digits with Python's decimal arithmetic, rounding and text: roots, exp / ln / log10, trigonometric and hyperbolic functions, exact sums and dot products, LU solve / det / inv, factorial, gamma |
| `layout` | C structs with natural or packed alignment, little / big endian, bit fields, arrays, nested structs, `pack` / `unpack`, views over a `bytearray`, C headers with static assertions, device register maps |
| `medical` | pharmacokinetics and NCA, SIR / SEIR, epidemiology, diagnostic accuracy, ROC with DeLong, Kaplan–Meier, log-rank, Cox regression, clinical formulas, ECG R-peaks and HRV, DICOM images and series |
| `env`, `log` | `.env` files and secrets; leveled logging to the console and rotating text / JSON files, timers, reading logs back |

Convolutions and max pooling run as C kernels (image columns and the packed matrix product) with their gradients,
about 1.7× faster than `apsi run` on a small CNN.

### What is not native yet

These programs keep running with `apsi run`. A native build that reaches one of them stops with a message that says so.

- graph networks, mixtures of experts and `fit_batches` in `layers`; GGUF, ONNX and PyTorch files in `models`;
  `text_dataset`, `train_tokenizer` and Parquet in `data`; websockets and TLS in `web` (serve HTTPS through a
  reverse proxy); `tokens.focus` / `fit_chat` / `outline` (they read Python syntax trees); `db.mysql` / `mongo`, `db.pool`, PostgreSQL / Redis over TLS on Windows; `layout`'s `numpy_dtype()`
- the modules `hub`, `memory`, `rt` (camera and video), `python`; in programs converted from Python, Python modules
  other than the stand-ins listed under "Programs converted from Python" (for example `re`)
- Python modules (`pyimport`, `import numpy`), except `asyncio`
- random numbers: native programs use their own generator, so a seed gives different numbers than in `apsi run`
  (the same seed always gives the same numbers in native programs)

### Errors

A native program that stops on an error prints the same report as `apsi run`, and exits with code 1. A misspelt
attribute or method gets a suggestion (`'str' object has no attribute 'uper'; did you mean 'upper'?`), and so does a
misspelt module name at build time:

```
app.apsi:12:5: RuntimeError: IndexError: index 5 is out of range for an axis of size 3
    let y = x[5]
    ^
  in step()
```

## `@native` functions

Apsi compiles functions to machine code. A function marked `@native` is specialised on the types and units of
its arguments, translated to LLVM IR, optimised at `-O3` for the exact CPU it runs on (AVX2, AVX-512, FMA, NEON), and
called directly — no interpreter in between. `parallel for` spreads a loop over all CPU cores, and `@gpu` runs it on
NVIDIA GPUs.

```apsi
use physics.constants

@native
fn orbit_speed(m: Mass, r: Length) -> Velocity = sqrt(G * m / r)

@native
fn energy(x, v, k: Float) -> Float:
    var e = 0.0
    parallel for i in 0..len(x):              # all CPU cores
        e += 0.5 * v[i] * v[i] + 0.5 * k * x[i] * x[i]
    return e

print(orbit_speed(5.972e24kg, 384400[km]) as [km/s])     # 1.01828904573 km/s
```

Nothing else changes: native functions take and return ordinary Apsi values, call and are called by any other
function, and work inside `evolve`. Results are identical to the bytecode engine.

- [How it works](#how-it-works)
- [Performance](#performance)
- [What compiles](#what-compiles)
- [Semantics and safety](#semantics-and-safety)
- [Tensors](#tensors)
- [Units](#units)
- [`parallel for`](#parallel-for)
- [`@gpu`](#gpu)
- [Tools and settings](#tools-and-settings)
- [Limits](#limits)

---

## How it works

```text
 .apsi source ──▶ parser ──▶ checker (names, units, types) ──▶ checked syntax tree
                                                                    │
         first call with new argument types (Int, Float, f32, tensors of a dtype/rank/layout, units)
                                                                    ▼
   specialiser: infers every variable's type by abstract execution, checks units statically and erases them,
   proves index ranges, fuses tensor expressions, outlines parallel loops
                                                                    │
                                                                    ▼
   LLVM IR ──▶ LLVM -O3 (inlining, loop + SLP vectorisation, unrolling) for the host CPU ──▶ machine code
                                                                    │        (signed on-disk cache)
                                                                    ▼
   NativeFunction: checks argument types/units at the boundary, calls the machine code, restores units on the result
```

- **Specialisation.** `f(2.0)` and `f(tensor)` get separate machine code, each as fast as hand-written code for
  that type. The first call compiles (typically 10–50 ms); later calls go straight to machine code.
- **Machine-code cache.** Compiled code is stored in `~/.apsi/cache/native`, signed with HMAC-SHA256 using your
  per-user key; an entry whose signature does not verify is never loaded. The cache key covers the IR, the CPU
  model and features, and the LLVM and Apsi versions, so code is only reused where it is valid.
- **Fast calls.** A cheap check of the argument types finds the machine code directly; numbers are passed in
  registers and tensors as the NumPy object itself (the machine code reads its data pointer, shape and strides from
  NumPy's struct, whose layout is measured and verified at start-up). A call costs ~0.9 µs, ~1.1 µs with a tensor
  (it was ~8–10 µs in the first 2.0 build).
- **Never breaks a program.** If a function uses something the native compiler does not handle, Apsi prints one note
  with the reason and line, and that function runs on the bytecode engine. Arguments native code cannot take
  (lists, records, structs, values being differentiated by `grad`) also use the bytecode engine, so `grad(f)` of a
  native `f` works.
- **Translated once, then direct.** A processor only executes its own machine instructions, so every language is
  translated at some point: C, C++, Rust and Go before you ship, Apsi the first time a function is called with new
  argument types. LLVM IR is the in-memory form the optimiser works on, not an extra layer at run time. After that
  first call — and in every later run, through the signed cache — the call goes straight into machine code with no
  translation, interpretation or Python in between. Measured on a 2-vCPU machine: the first compilation of a
  small loop takes about 90 ms; later runs load it from the cache in about 30 ms, most of which is loading the
  LLVM library once per process; each call then costs under 1 µs.
- **One program, two engines.** Native and bytecode functions call each other freely. Calls between native
  functions are direct machine-code calls, usually inlined.

## Performance

Same algorithms, same inputs, best of 5 runs, milliseconds (lower is better). Every language produces the same
checksums. Measured on a 2-vCPU cloud machine (Intel Xeon, Emerald Rapids, AVX-512); GCC 13.3, Clang 18.1,
Rust 1.97, Go 1.24, Apsi 2.0 with LLVM 22.

| Benchmark | Apsi 1.x bytecode engine | Apsi `@native` | Apsi `@native("fast")` | C (gcc -O3) | C++ (g++ -O3) | C (clang -O3) | Rust (-O3) | Go | Apsi fast + `parallel for` |
|---|---|---|---|---|---|---|---|---|---|
| fib(32), recursion | 1421 | 14.1 | 6.7 | 3.5 | 3.7 | 5.2 | 5.0 | 10.0 | – |
| Mandelbrot 1000×1000, 50 it. | 6302 | 60.1 | **46.9** | 61.8 | 65.8 | 57.3 | 59.5 | 62.5 | **24.5** |
| spectral-norm 1000 | 16425 | 50.1 | **24.9** | 24.9 | 25.1 | 49.7 | 46.8 | 49.6 | **12.6** |
| sieve 10M | 13545 | 61.9 | **50.9** | 53.6 | 55.0 | 51.0 | 57.6 | 59.0 | – |
| n-body, 1M steps | 63520 | 67.0 | 45.9 | 57.6 | 55.7 | **38.7** | 65.5 | 79.1 | – |
| matmul 512, naive i-k-j | 102456 | 35.6 | 28.9 | **27.4** | 41.0 | 36.7 | 46.7 | 144.5 | **15.0** |
| `sum(sqrt(a*a + b*b))`, 4M | 46 | 11.1 | 4.0 | 4.7 | 6.6 | 10.2 | 11.1 | 10.4 | **2.2** (automatic) |

The Apsi columns before the last one are single-core. The last column uses both cores of this machine: through
`parallel for`, or automatically for large fused tensor expressions and reductions (the `sum(sqrt(…))` row runs
on both cores without any change to the code; `@native` without `fast` takes 4.95 ms there). C, C++, Rust and Go
were measured as written, single-threaded.

How to read it:

- **Against the 1.x bytecode engine:** 100× to 3,500× faster on loop-heavy code.
- **Against C, C++, Rust and Go:** `@native("fast")` is the fastest or tied single-threaded result on four of the
  seven benchmarks and within 6–20 % of the best on n-body and matmul. Recursive `fib` remains about 2× behind GCC,
  which applies recursion-specific transformations, and is faster than Go.
- **Default `@native`** keeps checks those languages' release builds do not have (64-bit integer overflow, every
  index, integer division by zero) and still lands in the same range as Clang, Rust and Go on most benchmarks.
- **Why no compiler is 100× faster than C on the same algorithm:** C, C++ and Rust already compile to machine code
  that runs close to what the processor can do; Apsi uses the same LLVM optimiser. Beyond that point, more speed
  comes from more hardware: `parallel for` scales with the number of cores (this machine has 2; a 16-core desktop
  gives up to about 16× over single-threaded C), and `@gpu` uses the thousands of cores of a GPU. Apsi does this
  for you: large tensor kernels split across cores automatically, while in C, Rust or Go you would add threads by
  hand.

Reproduce: `sh benchmarks/run_all.sh` (needs gcc, g++, rustc, go; clang optional).


### Heavy workloads against C, C++, Rust, Go and Mojo

A second suite (`benchmarks/heavy/`) uses larger, harder programs: dense all-pairs gravity, a PDE stencil, a
1024² matrix product, LU factorisation with pivoting, Ising Monte Carlo, a 100-million-cell dynamic program, a
20-million-step Runge–Kutta integration and k-means clustering. Each is written the same way in Apsi (`@native`,
strict IEEE), C, C++, Rust, Go and Mojo 1.2 (with raw pointers, as performance-oriented Mojo is written); the runner
refuses to report times unless all of them produce the same checksum. Apsi 2.4.4 on a GitHub Actions runner
(AMD EPYC 7763), single core, milliseconds, best of 5 rounds in which Apsi 2.4.4 and 2.0.0 ran in turn on the same
runner (geometric mean: Apsi takes 0.81× the time of C):

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 46.5 | **44.1** | **44.1** | 181.9 | 317.4 |
| jacobi-2d 1024x200 | **151.4** | 167.1 | 157.3 | 166.6 | 700.7 |
| matmul 1024 | 177.6 | 199.6 | **164.8** | 182 | 479.4 |
| lu-pivot 800 | **40** | 40.3 | 40.1 | 41.2 | 116.5 |
| ising 512x50 | **81.5** | 90.6 | 90.4 | 83.6 | 145.9 |
| levenshtein 10k | **195.1** | 809.3 | 305.6 | 268.7 | 1,736 |
| ode-rk4 20M | 354.9 | **288.7** | 288.8 | 354.9 | 355.2 |
| kmeans 200k k16 x20 | **51.3** | 60.5 | 59 | 64.2 | 213.3 |

The same runner, Apsi 2.0.0 against 2.4.4 (geometric mean: 2.4.4 is 1.29× faster; `@native` code was already at C
speed in 2.0.0):

| Benchmark | Apsi 2.0.0 | Apsi 2.4.4 | 2.4.4 against 2.0.0 |
|---|---:|---:|---:|
| nbody-dense 2000x10 | 182.7 | 46.5 | **3.93× faster** |
| jacobi-2d 1024x200 | 150.5 | 151.4 | same (0.99×) |
| matmul 1024 | 178.9 | 177.6 | same (1.01×) |
| lu-pivot 800 | 40.3 | 40 | same (1.01×) |
| ising 512x50 | 81.5 | 81.5 | same (1.00×) |
| levenshtein 10k | 193.8 | 195.1 | same (0.99×) |
| ode-rk4 20M | 533.3 | 354.9 | **1.50× faster** |
| kmeans 200k k16 x20 | 65.3 | 51.3 | **1.27× faster** |

jacobi-2d is limited by memory bandwidth, which other jobs on the shared runner also use, so one round can be much
slower than the others (one of the five rounds took 261 ms); the best of several rounds is the reliable number.
C's ode-rk4 is faster because gcc fuses multiply-adds by default; with `@native("fma")` Apsi takes 263 ms there.
Results on another runner model (EPYC 9V45, AVX-512) and Mojo results from an earlier machine are in
`benchmarks/heavy/results.md` and `benchmarks/heavy/versions/`.

Run it on your own machine with `python3 benchmarks/heavy/run.py` (languages whose compiler is missing are skipped).


## What compiles

| Area | Supported in native code |
|---|---|
| Values | `Int` (64-bit), `Float` (f64), `f32` scalars, `Bool`, tensors of `f64 f32 i64 i32 i16 i8 uint8 bool` of any rank (contiguous or strided views), tuples, physical quantities |
| Statements | `let`, `var`, assignment (also to tensor elements and sub-tensors), `+= -= *= /= //= %= **=`, `if/elif/else`, `while`, `for` over ranges / `range()` / tensors / `enumerate(tensor)`, `parallel for`, `break`, `continue`, `return` (also of tuples and tensors), `assert`, `print` |
| Expressions | arithmetic, comparisons, `and`/`or`/`not`, `a if c else b`, `~=`, `as [unit]`, indexing `a[i]`, `a[i, j]`, `a[i][j]`, negative indices, slices `a[lo:hi:step]`, `a.shape`, `a.size`, `a.ndim`, `a.T` |
| Built-ins | `sqrt cbrt exp log log10 log2 sin cos tan asin acos atan atan2 sinh cosh tanh abs floor ceil round min max pow float int bool len shape ndim sum mean prod argmin argmax dot norm clip where relu sigmoid magnitude zeros ones full zeros_like ones_like` and the constants `pi e tau inf nan` |
| Matrices | `a @ b` for vectors and matrices (f64 or f32) runs on the multi-threaded BLAS, the same kernels as NumPy |
| Calls | other Apsi functions (compiled together, recursion included), physics constants and other global numbers and tensors, `module.name` constants such as `u.au` |

Not compiled yet (the function runs on the bytecode engine and the note says which): strings other than in
`print`/`assert`, lists, dicts, records, structs, lambdas and closures as values, comprehensions, `evolve` *inside* a
native function (an `evolve` that *calls* native functions is fine), `@` on tensors with more than 2 axes, reductions with
`axis=`, reading a global declared with `var` (pass it as an argument instead), and calls to Python libraries.

## Semantics and safety

Native code means the same thing as the bytecode engine. Default `@native` keeps every check:

| Situation | Native behaviour |
|---|---|
| `Int` arithmetic | 64-bit; overflow is an error with the source line (bytecode Ints never overflow); an argument too big for 64 bits runs the call on the bytecode engine |
| `//` and `%` | floor division and modulo with the sign of the divisor, exactly as Apsi defines them, for Int and Float |
| Division by zero | exactly as in the bytecode engine: for ordinary numbers `/`, `//`, `%` by zero and `0.0 ** -1` are `ZeroDivisionError`; for tensor elements (NumPy scalars) they give `inf`/`nan`, and `Int // 0` gives 0 |
| Number origin | a value read from a tensor is a NumPy scalar in the bytecode engine, with NumPy's rules (above; Int overflow wraps around). Native code tracks where every value came from, at compile time or with a one-bit run-time flag when it depends on the path, and applies the same rule. The result even comes back as the same Python class (`float` or `numpy.float64`) |
| Indexing | every index is checked; negative indices count from the end; loops whose index range can be proven safe up front run without per-element checks (the loop is versioned, so behaviour is unchanged) |
| `min`/`max` | the same choice as the bytecode engine, including the order of equal values |
| `round(x)` | round half to even, returns an Int |
| Float arithmetic | strict IEEE 754, no reassociation and no FMA contraction, so results match the bytecode engine bit for bit |
| Verified by | `tests/test_native.py` (every construct against the bytecode engine) and `tests/test_native_fuzz.py`: random Int/Float programs with Python and NumPy arguments must give the same values, the same errors and the same result classes — 0 differences in 14,400 calls with the default seeds and 8 more |
| `Int ** Int` | exact; an exponent that turns out negative at run time is an error (write it as a Float), because the result would have to change type |
| Errors | `assert` messages, line and column, and the call chain are reported as usual |

`@native("fast")` is for code that is known to be correct and must be as fast as possible: integer overflow and
index checks are removed, and floating-point reassociation and FMA are allowed, so sums may differ in the last
digits (like C with `-O3 -ffast-math`). Use it after the program works with `@native`.

`@native("fma")` keeps every check and the order of every operation, and only allows `a * b + c` to become one
fused multiply-add instruction (one rounding instead of two). That is what gcc and clang do for C by default, so it
is the like-for-like setting against C. Results differ from strict IEEE in the last bits (usually they are slightly
more accurate). On a chain of dependent arithmetic it is much faster: the ODE benchmark runs in 288 ms instead of
440 ms on a Xeon 8573C, against 387 ms for gcc -O3 (`benchmarks/heavy/fma.apsi`).

`@native("strict")` turns "cannot compile" into an error instead of a note — useful in tests and CI.

## Tensors

- **Element loops** compile to direct memory access: `a[i, j]` on a contiguous tensor is one address computation.
  Tensors passed in are used in place (no copy); writes are visible to the caller, as in Apsi.
- **Views:** `m[i]` (a row) and `a[lo:hi]` are views of the same memory, without copying.
- **Fused expressions:** `sqrt(a * a + b * b) + 0.5 * a` is computed in one pass with no temporary tensors.
  Shapes broadcast like NumPy (`m * v` with `m: (6, 5)` and `v: (5,)`); incompatible shapes are an error.
- **Reductions fuse too:** `sum(a * b)`, `mean`, `min`, `max`, `argmin`, `argmax`, `prod`, `dot`, `norm` read their
  input once without building it.
- **All cores automatically:** fused expressions and `sum`/`mean`/`prod`/`min`/`max` over 131,072 elements or more
  are split into the same fixed 256 chunks as `parallel for` and run on every core, without any change to the code
  (`sum(sqrt(a*a + b*b))` over 4M elements: 11.3 ms on one core, 5.2 ms on two). Sums are combined chunk by chunk
  in a fixed order, so the result is identical for every thread count; it can differ in the last bits from a
  strictly sequential sum, as NumPy's pairwise sums do.
- **New tensors:** `zeros(n, m)`, `ones`, `full`, `zeros_like`, results of tensor expressions. Tensors created inside
  a loop iteration are freed at the end of the iteration when they do not outlive it; a tensor that is returned
  becomes a normal Apsi tensor with no copy.

## Units

Units are checked when a function is compiled and then erased: the machine code works on plain SI numbers, so
physical units cost nothing at run time, and the result gets its unit back.

```apsi
@native
fn kinetic(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2
print(kinetic(2kg, 3[m/s]))          # 9.0 J

@native
fn wrong(x: Length, t: Time) = x + t # UnitError: cannot add Length and Time (at compile time)
```

Arguments are checked against the parameter annotations on every call, so `kinetic(2kg, 3s)` is still a
`UnitError`. Calls from unit-erased code (inside `evolve`) use a separate specialisation without units.

## `parallel for`

```apsi
@native
fn mean_and_max(x):
    var total = 0.0
    var biggest = -inf
    parallel for i in 0..len(x):
        total += x[i]
        biggest = max(biggest, x[i])
    return total / len(x), biggest
```

- Iterations run on a pool of threads, one per core (`APSI_THREADS=n`, `native_threads(n)` or
  `apsi run --threads n` to change). The dispatcher and the worker loop are machine code: workers stay awake for a
  few milliseconds after a loop, spinning on a shared counter with the CPU's `pause` instruction, so back-to-back
  parallel loops start in about **5 µs** with no operating-system call (Python's thread pool needed ~70 µs). Idle
  workers then sleep and use no CPU (`APSI_SPIN` sets the spin length). Threads claim several consecutive chunks per
  atomic operation, so cores do not fight over one cache line, and every field threads share has its own cache line.
- Reductions (`+=`, `-=`, `*=`, `x = min(x, …)`, `x = max(x, …)`) use private copies that are combined in a fixed
  order. The work is always split into the same 256 chunks, so **the result is bit-identical for any number of
  threads and on every machine**.
- Outer variables that are not reductions are read-only inside the loop; a loop that assigns them, or contains
  `return`, `break` or `print`, runs sequentially, and the note says why.
- A `parallel for` inside another one runs sequentially in the thread that reaches it.
- Outside native code `parallel for` is an ordinary `for`, so programs run unchanged on every engine.

## `@gpu`

```apsi
@gpu
fn saxpy(a: Float, x, y):
    parallel for i in 0..len(x):
        y[i] = a * x[i] + y[i]

@gpu
fn sum_of_squares(x) -> Float:
    var s = 0.0
    parallel for i in 0..len(x):
        s += x[i] * x[i]
    return s
```

A `@gpu` function runs its `parallel for` with one GPU thread per iteration.

- **Requirements:** an NVIDIA GPU with compute capability 7.5 or newer (GTX 1660 / RTX 20xx and every newer
  generation) and the normal NVIDIA display driver, on Windows 10/11 or Linux. No CUDA toolkit: Apsi generates PTX
  itself (LLVM NVPTX backend, `sm_75`), and the driver compiles it for the installed GPU and caches it.
- **Several GPUs:** the iterations are split evenly across all GPUs (choose them with `APSI_GPUS=0,1`). When every
  tensor write is to row `i` of the loop (`out[i] = …`, `out[i, j] = …`), each GPU copies back only its own rows;
  otherwise one GPU is used. Reductions from all GPUs are added together.
- **Shape of a GPU function:** optional `let`/`var` lines, one `parallel for` over a range, then optionally
  `return` of a `+=` reduction variable. The loop body may use arithmetic, comparisons, `if`/`while`/`for`, tensor
  element reads and writes, `sqrt abs floor ceil min max where clip round int float` and functions that follow the
  same rules. `exp`, `log`, trigonometric functions, `**` with exponents other than 1 and 2 (any integer up to 16
  with `@gpu("fast")`), Float `//` and `%`, `print`
  and creating tensors are not available inside GPU kernels yet.
- **Always runs:** without a usable GPU, or for a function outside these rules, the same function runs as native
  code on all CPU cores, with the same results; one note says why.
- **One thread per element:** when the body of the `parallel for` over rows is a single inner loop whose
  iterations are independent (`c[i, j] = …`), the two loops become one loop over rows × columns, so a 2048 × 2048
  result gets 4 million GPU threads instead of 2,048. Nothing to write differently.
- **CPU or GPU, chosen per call size:** a launch of fewer than 4,096 iterations runs on the CPU (the GPU's fixed
  costs would dominate). For larger sizes the choice is measured per size class (powers of two): the first GPU
  call is a warm-up (it loads the kernel and is not counted), then one GPU call and one CPU call are timed, and
  later calls use the faster side. Every 64 calls the slower side is timed once more, so a choice made while the
  machine was busy does not stick. Each call runs exactly once, so side effects never repeat. `@gpu("strict")` or `APSI_GPU_AUTO=0` always uses the GPU; `APSI_GPU_MIN=n` changes the threshold.
- **Matrix products and Tensor Cores:** a `@gpu` function whose loop nest is a matrix product runs as one cuBLAS
  GEMM when cuBLAS is installed (`pip install nvidia-cublas-cu12`): Tensor Cores with f32 accumulation for f16 and
  bf16 tensors, SGEMM / DGEMM for f32 / f64. See [Number formats](NUMBERS.md).
- **Number formats:** f64, f32, f16, bf16, FP8 (E4M3, E5M2) and int8 tensors; 16- and 8-bit values are computed
  in f32 and rounded once when stored ([Number formats](NUMBERS.md)). `var acc: f32 = 0.0` keeps an accumulator in
  f32, which consumer GPUs run up to 64× faster than f64.
- **Data movement:** ordinary tensor arguments are copied to GPU memory for each call and written tensors are
  copied back. `to_gpu(x)` keeps a tensor in GPU memory instead: kernels read and write it in place, with no copy
  per call, and `to_host(g)` brings the values back when you need them. Without a GPU, `to_gpu` returns the tensor
  itself, so the program runs everywhere.
- **Precision:** IEEE arithmetic; `sqrt` is correctly rounded, as on the CPU.
- **Status:** PTX generation, the kernel rules and the CPU path are covered by the test suite. The launch path
  uses the CUDA driver API (`nvcuda.dll` / `libcuda.so.1`) and needs a GPU to run, so it is not exercised by
  continuous integration on GPU-less machines. `apsi native info` and `gpu_info()` show the GPUs Apsi can use;
  `apsi doctor` reports them too.

For training neural networks on the GPU, the PyTorch backend (`fit(…, backend="torch")`, see
[`BACKENDS.md`](BACKENDS.md)) remains the right tool: it brings cuDNN and fused training kernels. `@gpu` is for your
own loops and kernels, including matrix products on Tensor Cores.

**Other GPU APIs.** `@gpu` targets NVIDIA GPUs through CUDA. Vulkan, DirectX 12 and Metal compute backends (which
would add AMD, Intel and Apple GPUs) are planned; until then those GPUs are used through the PyTorch backend
(ROCm, XPU, MPS).

## Tools and settings

| Command / setting | Effect |
|---|---|
| `apsi native info` | CPU model, SIMD features, threads, GPUs and the cache location (`--json` for scripts) |
| `apsi native explain app.apsi` | runs the program and lists each native function: compiled for which argument types, or why it runs on the bytecode engine (`--all` tries every function) |
| `apsi native ir app.apsi f` / `asm` | the optimised LLVM IR or the machine code of function `f` |
| `apsi native clear-cache` | empty the machine-code cache |
| `apsi run --native app.apsi` | try every function natively (silent fallback), as if each had `@native` |
| `apsi run --threads 8 app.apsi` | threads for `parallel for` |
| `native_info()`, `gpu_info()`, `native_threads(n)` | the same from inside a program |
| `APSI_THREADS=n` | default thread count |
| `APSI_NATIVE=auto` | same as `--native` |
| `APSI_NATIVE_QUIET=1` | hide "runs on the bytecode engine" notes |
| `APSI_NO_CACHE=1` | do not read or write the compile and machine-code caches |
| `APSI_NO_GPU=1` / `APSI_GPUS=0,2` | ignore GPUs / use only these GPUs |

The native compiler needs `llvmlite` (installed with Apsi; it contains LLVM, nothing else to install). Without it,
`@native` functions run on the bytecode engine and `apsi native info` says why.

## Limits

- Native `Int` is 64-bit; bigger integers stay on the bytecode engine.
- Functions with shape parameters (`fn f[N](...)`) and parameters typed with structs run on the bytecode engine.
- Several Python threads calling native functions at the same moment share one error slot; run parallel work with
  `parallel for` instead of Python threads.
- The playground (WebAssembly) has no native compiler; `@native` functions run on the bytecode engine there.
