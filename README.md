<p align="center"><img src="assets/brand/apsi-logo-transparent.png" alt="Apsi" width="220"></p>

<p align="center"><strong>A programming language for AI, physics and servers — that reads like Python and checks your work before it runs.</strong></p>

<p align="center">
  <a href="https://github.com/ATECTechnologies/apsi/actions/workflows/linux-x64.yml"><img src="https://github.com/ATECTechnologies/apsi/actions/workflows/linux-x64.yml/badge.svg" alt="linux-x64"></a>
  <a href="https://github.com/ATECTechnologies/apsi/actions/workflows/darwin-x64.yml"><img src="https://github.com/ATECTechnologies/apsi/actions/workflows/darwin-x64.yml/badge.svg" alt="darwin-x64"></a>
  <a href="https://github.com/ATECTechnologies/apsi/actions/workflows/windows-x64.yml"><img src="https://github.com/ATECTechnologies/apsi/actions/workflows/windows-x64.yml/badge.svg" alt="windows-x64"></a>
</p>
<p align="center">
  <a href="https://github.com/ATECTechnologies/apsi/actions/workflows/linux-arm64.yml"><img src="https://github.com/ATECTechnologies/apsi/actions/workflows/linux-arm64.yml/badge.svg" alt="linux-arm64"></a>
  <a href="https://github.com/ATECTechnologies/apsi/actions/workflows/darwin-arm64.yml"><img src="https://github.com/ATECTechnologies/apsi/actions/workflows/darwin-arm64.yml/badge.svg" alt="darwin-arm64"></a>
</p>
<p align="center">
  <a href="https://apsi.dev/download"><img src="https://img.shields.io/badge/download-2.6.6-2ea44f" alt="Download Apsi 2.6.6"></a>
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776ab" alt="Python 3.12 | 3.13 | 3.14">
  <img src="https://img.shields.io/badge/native-executables%20%7C%20AVX2%20%7C%20AVX--512%20%7C%20NEON-555" alt="native code">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20Tensor%20Cores-76b900" alt="GPU">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apsi%202.0-blue" alt="license"></a>
</p>

<p align="center">
  <a href="https://apsi.dev">Website</a> ·
  <a href="https://apsi.dev/docs">Documentation</a> ·
  <a href="docs/WHATS-NEW.md">What's new in 2.6.6</a> ·
  <a href="https://apsi.dev/download">Download</a>
</p>

---

## What is Apsi?

Apsi is a programming language with the syntax of Python and the safety net of a compiler. Physical units, tensor
shapes and types are checked before the program starts, so `5m + 2s` or a matrix of the wrong size is an error on
the line where you wrote it, not a wrong number an hour later. Since 2.6.6, an Apsi program can also be compiled
into a standalone executable that runs without Python, and it works with C and C++ code directly.

```apsi
use physics.constants

fn kinetic(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2

print(kinetic(2kg, 3[m/s]))            # 9.0 J
print(m_e * c ** 2 as [MeV])           # 0.510998949996 MeV
let wrong = 5m + 2s                    # ✗ UnitError before running: cannot add Length and Time
```

```bash
apsi run physics.apsi                  # run it straight away
apsi build --native physics.apsi       # or make physics.exe: machine code, no Python inside
```

## Apsi 2.6.6: a new architecture

Until 2.6.2, every Apsi program ran on top of a Python host. 2.6.6 adds a second way to run Apsi, built from the
ground up for it:

```
            ┌──────────── apsi run ──────────────┐
 app.apsi → │ checker → Python host (NumPy, …)   │   quick to start, every library, the REPL and debugger
            └────────────────────────────────────┘
            ┌──────── apsi build --native ───────┐
 app.apsi → │ checker → C → machine code         │   app.exe: one file, no Python, Apsi's own runtime
            └────────────────────────────────────┘
```

- **Apsi's own runtime.** Values, memory management, strings, collections, closures, classes, exceptions,
  generators, units, tensors, automatic differentiation and the GPU are implemented in Apsi's runtime, written in C.
  A native program contains no Python interpreter.
- **The standard library in Apsi.** The functional helpers, `nn` (layers, losses, optimizers, CNNs, LSTMs, GRUs),
  the `evolve` integrators and the physical constants are written in Apsi and compiled with your program.
- **Built for the machine it runs on.** Matrix products use packed, register-blocked kernels on every core;
  element-wise maths, `exp` and `tanh` use the CPU's widest vector unit.
- **The GPU without a toolkit.** Kernels are built into the program, and the NVIDIA driver is the only thing the
  card needs.
- **C and C++ next to Apsi.** `cload("vecmath.c")` or `cload("model.cpp")` takes the source file itself. A native
  build compiles it into the program, and `apsi run` builds it once and caches it. Tensors are passed to C without a
  copy.
- **Nothing to install.** On Windows, Apsi carries its own C toolchain (clang and lld), so no compiler setup is needed.
- **The same answers.** Native programs are tested against `apsi run`: same output, same error messages, same
  numbers, and they run clean under AddressSanitizer.

`apsi run` stays as it was, so existing programs, Python libraries and tools keep working. Both ways share one
checker, so a program that passes it behaves the same either way. The full list is in
[What's new in 2.6.6](docs/WHATS-NEW.md), and the native compiler is covered in [docs/NATIVE.md](docs/NATIVE.md).

## How Apsi is different

| | Python | Apsi |
|---|---|---|
| Units (`9.81[m/s^2]`, `5km`) | a library, checked at run time | part of the language, checked before running |
| Tensor shapes | found when the program crashes | `fn f[B, T](x: Tensor[B, T])` names the wrong dimension before running |
| Speed | an interpreter | `apsi build --native`: a compiled executable; `@native` functions inside `apsi run` |
| Shipping a program | Python and its packages on every machine | one executable |
| C and C++ | ctypes, cffi or an extension module | `cload("file.c")`, the source file itself |
| Existing Python code | — | `apsi convert` turns it into Apsi that behaves the same |
| Permissions | a script can read any file and open any socket | files, network and Python modules are off until you allow them |
| Built in | — | autodiff, layers, RL, real-time loops, tables, databases, an HTTP and WebSocket server |

## Install

Download: [apsi.dev/download](https://apsi.dev/download) — every version, with the SHA-256 of each file.

| System | File | Install |
|---|---|---|
| Windows 10/11 (x64) | `ApsiSetup-2.6.6-x64.exe` | double-click (brings its own Python and C toolchain; no administrator rights) |
| Any Linux, also WSL | `ApsiSetup-2.6.6-linux.sh` | `sh ApsiSetup-2.6.6-linux.sh` (no root) |
| Ubuntu / Debian | `apsi_2.6.6_all.deb` | `sudo apt install ./apsi_2.6.6_all.deb` |

From this repository: `pip install numpy`, then `./apsi.sh run examples/01_hello.apsi` (Linux, macOS) or
`apsi.bat run examples\01_hello.apsi` (Windows). Native builds on Linux and macOS use the system's `clang` or
`gcc`; on Windows the toolchain is fetched once on first use when Apsi was not installed with the installer.
Building the installers: [docs/BUILD.md](docs/BUILD.md).

## Quick start

```bash
apsi new hello && cd hello
apsi run                        # runs src/main.apsi
apsi test                       # runs the tests next to it
apsi build --native             # src/main.apsi -> an executable
apsi convert script.py          # an existing Python program -> script.apsi, same behaviour
apsi debug app.apsi             # breakpoints and stepping (or F5 in VS Code)
```

Then read [Getting started](docs/GETTING-STARTED.md) (15 minutes) or [Write Apsi your way](docs/WRITE-APSI.md).

## A tour in code

**Train a network** — layers, losses and optimizers from `nn`, gradients from the language:

```apsi
use nn

let X, y = spiral_data(n_per_class=100, classes=3, seed=0)
var params = mlp_init([2, 64, 64, 3], seed=0)
fn loss_fn(p, xb, yb):
    var h = xb
    for i, layer in enumerate(p):
        h = h @ layer.W + layer.b
        if i < len(p) - 1:
            h = relu(h)
    return cross_entropy(h, yb)

let result = train(loss_fn, params, (X, y), adam(lr=0.01), epochs=200, batch_size=64)
print("final loss:", result.history[-1])
```

**Solve physics** — an ODE with units, integrated by the language:

```apsi
use physics.constants

var pos = tensor([0.0, 1.0]) * u.m
var vel = tensor([30.0, 30.0]) * u.m / u.s
evolve pos, vel over t in 0[s]..4[s] step 1[ms] using rk4 record every 100 into path:
    pos' = vel
    vel' = tensor([0.0, -1.0]) * g0
print("highest point:", max(path.pos[:, 1]))
```

**Call C** — the source file next to your program, compiled for you:

```c
// vecmath.c
#include <stdint.h>
double dot(const double *a, const double *b, int64_t n) {
    double s = 0; for (int64_t i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
```

```apsi
let lib = cload("vecmath.c")
let dot = lib.fn("dot", [ptr_f64, ptr_f64, i64], f64)
let a = tensor([1.0, 2.0, 3.0])
print(dot(a, a, len(a)))            # 14.0
```

**Match on structure** — patterns on lists, maps, types and classes:

```apsi
fn describe(v):
    match v:
        case [first, *rest] if len(rest) > 0:
            return f"{first} and {len(rest)} more"
        case {"name": n, **others}:
            return f"named {n}"
        case int() | float() as x:
            return f"the number {x}"
        case _:
            return "something else"
```

## Train your own language model

A tokenizer, a LLaMA / Qwen3-style decoder, training and text written by the model — the whole program
(`examples/47_train_your_llm.apsi`), seconds on a laptop:

```apsi
use layers
use data
use nn

let corpus = ["The planet orbits the star once a year. " * 4, "Energy is mass times the speed of light squared. " * 4]
let tok = data.train_tokenizer(corpus, vocab_size=300)                    # byte-level BPE, as GPT-2 / Llama 3 / Qwen
let batches = [b for b in (data.text_dataset(corpus, tokenizer=tok, seq_len=24) |> batch(8))]

let lm = transformer_lm(tok.vocab_size, d=64, heads=4, layers=2, context=24,
                        kv_heads=2, rope=true, qk_norm=true, norm="rms", ffn="swiglu")
let r = fit_batches(lm, lm.init(seed=0), batches, loss="cross_entropy",
                    opt=adamw(lr=3e-3, state="bf16"), epochs=40, clip=1.0)
print(f"loss {r.history[0]:.2f} -> {r.history[-1]:.2f}")
```

The same lines train larger models on an NVIDIA card (`backend="gpu"`), with checkpoints that resume exactly after
any stop. See [Training](docs/TRAINING.md) and [GPU](docs/GPU.md).

## Stable by design

Breaking changes in libraries are a common reason projects in new languages stop working. Apsi avoids them in
these ways:

- **No binary ABI between Apsi libraries.** Apsi modules are compiled from source together with your program, so a
  new Apsi version never meets an old compiled library.
- **A pinned toolchain.** The C compiler that builds native programs is a fixed version shipped with Apsi, not
  whatever happens to be installed.
- **One file to ship.** Native Windows programs are linked statically, so nothing else needs installing next to them.
- **One checker, two engines.** `apsi run` and native builds share the parser and the checker, and native programs
  are tested for the same output as `apsi run`.
- **Published libraries cannot change under you.** A version on the hub is never replaced, so builds stay
  reproducible.

## Supported platforms

| | |
|---|---|
| Operating systems | Windows 10 / 11 (x64), Linux x86-64 and ARM64 (Ubuntu 22.04+, Debian 12+, Fedora, Arch, WSL), macOS on Apple silicon and Intel — each tested on every push |
| Python (host runtime) | 3.12, 3.13, 3.14 |
| Native programs | `apsi build --native`: Windows x64 (the C toolchain comes with Apsi), Linux and macOS (the system's clang or gcc); x86-64 with AVX2 / AVX-512, ARM64 with NEON |
| `@native` functions | inside `apsi run`, through LLVM (`llvmlite`) |
| GPUs | NVIDIA GTX 1660 and newer (training on Apsi's own tensor-core kernels; only the NVIDIA driver needed); AMD ROCm, Apple MPS, Intel XPU through the PyTorch backend |
| Editors | VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim, Notepad++, JetBrains, Helix, Zed, Emacs |

## Documentation

| Start | Language | Topics |
|---|---|---|
| [Getting started](docs/GETTING-STARTED.md) | [Language reference](docs/LANGUAGE.md) | [AI](docs/AI.md) · [Layers](docs/LAYERS.md) · [Training](docs/TRAINING.md) · [GPU](docs/GPU.md) · [RL](docs/RL.md) |
| [What's new](docs/WHATS-NEW.md) | [Everything on one page](docs/REFERENCE.md) | [From Python](docs/PYTHON.md) · [Native programs](docs/NATIVE.md) · [C and C++](docs/INTEROP.md) · [Embedded](docs/EMBEDDED.md) |
| [Install](docs/INSTALL.md) · [Linux](docs/INSTALL-LINUX.md) | [Project overview](docs/OVERVIEW.md) | [Real time](docs/REALTIME.md) · [Memory](docs/MEMORY.md) · [Tokens](docs/TOKENS.md) · [Hub](docs/HUB.md) · [Graphs](docs/GRAPHS.md) · [Number formats](docs/NUMBERS.md) |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | [Security](docs/SECURITY.md) | [Tables](docs/DATAFRAME.md) · [Databases](docs/DATABASES.md) · [Servers](docs/SERVER.md) |
| [Versions compared](docs/VERSIONS.md) | [Editors](docs/EDITORS.md) | [Science](docs/SCIENCE.md) · [Debugging](docs/DEBUGGING.md) · [Configuration](docs/ENV.md) |

The same documentation, searchable, is at [apsi.dev/docs](https://apsi.dev/docs).

## Repository layout

```
apsi/          the language: lexer, parser, checker, compiler, runtime, standard library, tools
apsi/aot/      the native compiler: Apsi → C → machine code, its C runtime (values, memory, tensors, GPU) and the
               standard library written in Apsi
apsi/native/   @native functions inside apsi run (LLVM) and GPU code generation for the trainer
docs/          documentation (also published on apsi.dev)
examples/      runnable examples, also used as tests
tests/         the test suite (python -m unittest discover tests; apsi test tests/apsi examples)
benchmarks/    benchmark programs in Apsi, C, C++, Rust, Go, Java, JavaScript, C#, Julia, Mojo, plus pandas, Node.js and Python comparisons
editors/       editor extensions (VS Code family, Sublime Text, Vim, Notepad++ …)
installer/     the Windows setup wizard;  packaging/  wheel, .deb and Linux installer builders
playground/    the WebAssembly playground
```

## Contributing

Bug reports and pull requests are welcome. Before sending a pull request, run the tests:

```bash
python -m unittest discover tests
apsi test tests/apsi examples
apsi fmt --check examples tests/apsi
```

Security problems: please report them privately as described in [`docs/SECURITY.md`](docs/SECURITY.md) or at
[apsi.dev/security](https://apsi.dev/security), not in public issues.

## Credits

Apsi is designed and developed by **ATEC**. Its development is assisted by Claude, the AI model made by Anthropic.
Thank you to everyone who reports problems, writes libraries and shares their work with the community.

## License

Apsi is developed by **ATEC** and published under the [Apsi License 2.0](LICENSE). In short: use it for anything,
including commercial work; the programs, libraries and models you build with it are yours; Apsi itself may not be
sold or offered as a paid service without a licence from ATEC; copies and changed versions of Apsi credit ATEC and
stay under the same licence. Plain-language guide with examples: [`docs/LICENSING.md`](docs/LICENSING.md).
