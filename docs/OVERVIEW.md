# Apsi 2.4.4 — project overview

**Apsi** is a programming language for **AI**, **physics** and **servers**. Physical units, tensor-shape
checking, automatic differentiation, number formats from FP64 down to FP4, data pipelines and a hardened
HTTP server are built into the language. It reads like Python, but catches mistakes before the program runs,
and since 2.0 it compiles to **native machine code** through LLVM, running on all CPU cores and on NVIDIA GPUs.

```apsi
use physics.constants

fn kinetic(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2

print(kinetic(2kg, 3[m/s]))          # 9.0 J
print(m_e * c ** 2 as [MeV])         # 0.510998949996 MeV

let wrong = 5m + 2s                  # ✗ UnitError before running: cannot add Length and Time

@native                              # machine code for this CPU, units checked then erased
fn energy(x, v, k: Float) -> Float:
    var e = 0.0
    parallel for i in 0..len(x):     # every CPU core; @gpu runs the same loop on the GPU
        e += 0.5 * v[i] * v[i] + 0.5 * k * x[i] * x[i]
    return e
```

## Install

| System | File in `dist/` |
|---|---|
| Windows 10/11 (x64) | `ApsiSetup-2.4.4-x64.exe` — setup wizard with its own Python; no administrator rights needed |
| Any Linux (also WSL) | `sh ApsiSetup-2.4.4-linux.sh` — one-file installer, no root; commands, `.apsi` file type with icon, editor support |
| Ubuntu / Debian | `sudo apt install ./apsi_2.4.4_all.deb` |
| Any system with Python 3.12+ | `pip install apsi-2.4.4-py3-none-any.whl` |
| Code editors | the setup wizard adds Apsi to VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim and Notepad++ it finds (`apsi editors install` does the same later) |

The installer adds `apsi` to your PATH, registers `.apsi` files (icon, *Run with Apsi*, *Edit*, and
*New → Apsi source file* in Explorer), creates Start-menu / application-menu entries and adds Apsi to your code editors. Details: [`INSTALL.md`](INSTALL.md) · [`INSTALL-LINUX.md`](INSTALL-LINUX.md)

If `apsi` is not recognized right after installing, open a new terminal (quit VS Code completely); other first-run
problems and their fixes are in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

Without installing: `pip install numpy`, then use `apsi.bat …` (Windows) or `./apsi.sh …` (Linux) from this folder.

### Build from source

| | Command |
|---|---|
| Windows: development setup + tests | `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Windows: + setup wizard | `powershell -ExecutionPolicy Bypass -File setup.ps1 -Installer` |
| Linux: development setup + tests | `./setup.sh` |
| Linux: + AppImage wizard / all release files | `./setup.sh --installer` / `./setup.sh --release` |
| Optional libraries | add `-Libs ai` (Windows) or `--libs ai` (Linux): `gpu`, `onnx`, `torch-auto` (the right PyTorch build), `torch`, `torch-cuda`, `transformers`, `sci`, `data`, `ai`, `ai-gpu`, `science`, `all` |

Full details: [`BUILD.md`](BUILD.md)

## Your first program

```bash
apsi new hello          # create a project (templates: app, lib, physics, ai, server)
cd hello
apsi run                # runs src/main.apsi
apsi test               # runs the tests in tests/
apsi                    # interactive REPL
```

## Highlights

| Area | What Apsi gives you |
|---|---|
| **Physical units** | `9.81[m/s^2]`, `2kg`, `5[km]` are checked before the program runs; convert with `as [km]`; every SI unit and prefix plus CODATA constants |
| **Tensor shapes** | `fn f[B, T, D](x: Tensor[B, T, D])` — a mismatch names the exact dimension that is wrong |
| **Autodiff** | `grad(f)`, `value_and_grad(f)` on tensors, structs and records, unit-aware (`grad(energy)` has the unit of a force); any order: `hessian`, `jacobian`, `laplace`, `grad(grad(f))` — enough for physics-informed neural networks |
| **AI** | `use layers`: **any architecture as a graph of layers** (`sequential`, `residual`, `parallel`, `dag`), transformers, CNNs, LSTM/GRU, autoencoders/VAEs, **graph neural networks** (GCN, GraphSAGE, GAT, GIN), a model zoo and `fit()`; plus Adam/AdamW/SGD, schedules, gradient clipping, save/load — reference: [`LAYERS.md`](LAYERS.md) |
| **GPU training (PyTorch)** | the same models train on PyTorch with `fit(…, backend="torch")` — NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU or CPU; fp64/fp32/tf32, bf16/fp16 mixed precision, `torch.compile`, fused attention, cuDNN LSTMs; `torch_module()` gives a real `torch.nn.Module`; `check_torch()` verifies every layer against NumPy; `apsi pkg install torch-auto` picks the right CUDA build — [`BACKENDS.md`](BACKENDS.md) |
| **Python, exactly** | `apsi convert` turns any Python 3.13 program into Apsi that behaves **byte for byte** like the original: classes, metaclasses, dataclasses, `match`, `async`, `except*`, generics, star imports. Verified on 34 standard-library modules (≈38,000 lines) and a 23,769-line AI training program — identical output, files and training losses — [`PYTHON.md`](PYTHON.md) |
| **Real time** | `use rt`: control loops at 1–2 kHz with microsecond lateness; video pipelines at any size from 640×480 to 3840×2160 (named sizes `vga` … `4k`) with newest-frame hand-off, adaptive resolution and per-frame latency reports; CPU or GPU — [`REALTIME.md`](REALTIME.md) |
| **RAM, SSD, GPU** | `use memory`: arrays that move to the SSD instead of running out of RAM, out-of-core `chunks`/`map` on all cores, two-tier RAM/SSD caches, batches that shrink on out-of-memory, the GPU with the most free memory — [`MEMORY.md`](MEMORY.md) |
| **Fewer AI tokens** | `use tokens`: exactly reversible packing for prompts — record lists as tables (−57 to −63 %), repeated lines once (−59 % on logs), only the changed lines of an edited document (−99 %) — [`TOKENS.md`](TOKENS.md) |
| **Tables** | `use dataframe`: a columnar engine — filters 24×, group-by 4×, joins 3× faster than pandas, 4× less memory — [`DATAFRAME.md`](DATAFRAME.md) |
| **Databases** | `use db`: PostgreSQL, Redis and SQLite with no driver to install — [`DATABASES.md`](DATABASES.md) |
| **Debugger** | breakpoints, stepping, variables and tensors in VS Code (F5) or `apsi debug` — [`DEBUGGING.md`](DEBUGGING.md) |
| **From PyTorch to native** | `apsi convert --idiomatic`: **PyTorch programs become native Apsi** on `use torch`, Apsi's own implementation of the PyTorch API: `nn.Module` classes, every common layer (BatchNorm, LSTM, attention, transformers), losses with all options, optimizers, learning-rate schedulers, `DataLoader`s, `state_dict`s and the training loop. Whole training runs end with the same weights as PyTorch (to 1e-9), with no Python or PyTorch at run time. numpy, math, random and json map onto Apsi too; any other library keeps working through `pyimport` — [`PYTHON.md`](PYTHON.md) |
| **Reinforcement learning** | `use rl`: vectorised environments, GAE, PPO, double DQN, REINFORCE, DPO / IPO / GRPO / RLHF objectives, Monte Carlo tree search, beam search — [`RL.md`](RL.md) |
| **Libraries** | `.apsil` packages: `apsi install NAME` gets the newest version with dependencies and a lock file, verifies every file's SHA-256 and scans it; `apsi pack` + `apsi publish` with an API token — [`LIBRARIES.md`](LIBRARIES.md) |
| **Science and medicine** | `use special`, `stats`, `signal`, `mechanics`, `pde`, `quantum`, `medical`, `lab`: special functions, distributions and 19 hypothesis tests, regression, power analysis, zero-phase filters and spectra, Lagrangian/Hamiltonian dynamics by autodiff, N-body, orbits, beams and trusses, heat/wave/Poisson/Navier–Stokes solvers, Schrödinger equation and qubits, pharmacokinetics, survival analysis (Kaplan–Meier, log-rank, Cox), ROC/DeLong, ECG/HRV, DICOM, reproducible experiment logs. Verified against SciPy, statsmodels, lifelines, scikit-learn and pydicom — [`SCIENCE.md`](SCIENCE.md) |
| **Physics** | `sci.solve_ivp` with an implicit TR-BDF2 solver for **stiff** systems and adaptive RK45; `evolve` solves differential equations with euler / midpoint / rk4 / adaptive **rk45** / symplectic / verlet; Laplacians for PDEs; linalg (eigh, solve, …); measurement uncertainty `9.81 ± 0.02` that propagates through formulas with correct correlations |
| **Native code** | `@native` compiles a function to machine code through LLVM for the exact CPU (AVX2/AVX-512), specialised on argument types and units; on par with C, C++ and Rust and 100–3,500× faster than the 1.x bytecode engine on loops ([benchmarks](NATIVE.md#performance)); overflow, index and division checks stay on by default; results identical to the bytecode engine |
| **Large models** | `apsi plan 1T --active 32B` / `use scale`: memory per GPU, tensor × pipeline × data layout, GPU count, days and serving memory for 1B–2T-parameter models (dense or mixture-of-experts, BF16/FP8) — [`SCALE.md`](SCALE.md) |
| **Parallel & GPU** | `parallel for` runs iterations on all CPU cores with deterministic reductions (bit-identical for any thread count); `@gpu` runs it on every NVIDIA GPU in the machine (GTX 1660 and newer, multi-GPU split, no CUDA toolkit) and falls back to all CPU cores — [`NATIVE.md`](NATIVE.md) |
| **Speed (rest of the language)** | the compiler turns the whole program into bytecode; every function has an "erased twin" without unit bookkeeping for hot loops; parsed programs are kept in a signed compile cache; tensor work runs on multi-threaded BLAS, on the GPU through CuPy, and neural networks through PyTorch |
| **Executables** | `apsi build app.apsi` → `app.exe` (Windows) / `app` (Linux) with the program, its imports, Apsi as bytecode and fixed permissions; `--standalone` adds its own Python runtime for PCs without Apsi |
| **Precision** | `@precision(exact)` computes with exact fractions; `@precision(f32)` / `f16` for AI |
| **Security** | permission system (Python, C, network are off by default), Apsi Guard pre-run scan, and a `crypto` module (scrypt, HMAC, SHA-3, tokens, constant-time compare) |
| **Other languages** | Python/PyTorch (`pyimport`, `to_torch`, `torch_module`), C/C++ (`cload`, zero-copy tensors), JS/Next.js/Java through `apsi serve` (JSON HTTP API) |
| **Number formats** | FP64, FP32, FP16, BF16, FP8 (E4M3/E5M2), FP6, FP4, INT8, INT4 — bit-exact with current GPU formats, stored at their real size, quantization-aware training |
| **Data** | streaming pipelines: parallel map, prefetch, shuffle, vectorized batching (~2 M rows/s), sliding windows, sharding, interleaving; CSV, JSON Lines, Parquet, `.npy`; memory-mapped tensors on SSD, signed on-disk cache |
| **Accuracy** | exact fractions, correctly rounded sums (`fsum`), measurement uncertainty, unit checking, bit-exact low-precision formats |
| **Concurrency** | tasks, channels, locks, atomics, `parallel_for` / `parallel_reduce`, `pmap`, BLAS thread control |
| **Servers** | `use web`: asyncio protocol + worker pool + multi-process, 20,000+ req/s on 2 vCPU at 50–1,000 connections; WebSocket routes, rooms and client; `.env` files and secrets (`use env`); protection against slowloris, request smuggling, path traversal, header injection and DoS; JWT, API keys, rate limits, CORS, TLS, metrics |
| **AI models** | `use models`: Apsi checkpoints (`.apsim`: units, compressed tensors, SHA-256, signatures, sharding), GGUF (bit-exact Q8_0/Q4_0), safetensors read/write, memory-mapped loading, INT8/FP8/FP4 compression (4–16× less RAM), `qlinear`; ONNX Runtime, Hugging Face and PyTorch models served as an API in a few lines |
| **Tooling** | `run`, `check`, `test`, `fmt`, `watch`, `bench`, `build`, `compile`, `convert`, `new`, `task`, `doc`, `lsp`, `repl`, `serve`, `doctor`, `editors`, `pkg`; extensions for VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim and Notepad++ with live diagnostics, unit hovers, go to definition, tasks, snippets and themes; web playground (WebAssembly) |
| **Platforms** | Windows 10/11 and Linux installers; CI on Windows Server 2022/2025 and Ubuntu 22.04/24.04; Python 3.12 and newer (3.12, 3.13, 3.14) |

## Commands

```text
apsi                          interactive REPL (cd, ls, run FILE, !cmd and :help work inside it)
apsi FILE.apsi [ARGS]         run a program (same as apsi run)
apsi -e "CODE"                evaluate code and print the result
apsi run [FILE] [ARGS]        check, then run (default: the project's main file); --watch, --time
apsi check [PATHS]            check files or folders without running them
apsi test [PATHS]             run every test "..." block
apsi fmt [PATHS]              format code (--check for CI)
apsi watch [PATHS]            re-run, re-test or re-check on every save
apsi bench FILE               time a program over several runs
apsi build FILE               make an executable: app.exe (Windows) / app (Linux); --standalone, --allow-…
apsi compile [PATHS]          compile programs ahead of time into the signed compile cache (--stats, --clear)
apsi new NAME / apsi init     create a project with apsi.toml, src/, tests/ and VS Code tasks
apsi task [NAME] / apsi NAME  run a task from apsi.toml
apsi doc [NAME]               documentation for built-ins and modules
apsi audit / scan / verify    permissions a program needs / malware scan / installation integrity
apsi serve FILE               expose a program's functions as a JSON HTTP API
apsi install NAME             install an Apsi library (.apsil) from the registry: NAME@^1.2, a file, or [dependencies]
apsi pack / apsi publish      build dist/NAME-VERSION.apsil / upload it with an API token
apsi search WORDS · apsi lib list | info | registry | login   the library registry
apsi pkg install torch-auto   optional Python libraries (torch-auto, gpu, onnx, transformers, ai, …)
apsi config --profile strict  default security profile
apsi completion powershell    shell completion (bash, zsh, fish, powershell)
apsi editors install          Apsi support in VS Code, Cursor, Windsurf, VSCodium, Sublime, Vim/Neovim, Notepad++
apsi convert app.py           translate Python (and whole folders) into Apsi
apsi native info | explain FILE  the native compiler: CPU, SIMD, GPUs; which functions run as machine code
apsi run --native FILE        compile every function that allows it to machine code
apsi plan 1T --active 32B     plan training of a large model: GPUs, layout, memory per GPU, days
apsi lsp | doctor | ast | tokens | version
```

Every command and flag is described in the *Command line* chapter of [`LANGUAGE.md`](LANGUAGE.md).

## Examples (`examples/`)

| File | Shows |
|---|---|
| `01_hello` | language basics: let/var, functions, structs, comprehensions, pipes |
| `02_units` | physical units, conversions, constants |
| `03_autodiff` | derivatives, force from potential energy, linear regression |
| `04_projectile` | projectile with air drag (rk4) |
| `05_orbit` | one year of Earth's orbit; energy drift only 1e-14 (verlet) |
| `06_heat_equation` | 2-D heat equation |
| `07_quantum_box` | Schrödinger equation; energy levels match theory |
| `08_neural_net` | MLP trained from scratch to 99 % accuracy |
| `09_transformer` | character-level mini-GPT written from scratch |
| `10_cnn` | convolutional network |
| `11_precision` | exact / f32 / memo |
| `12_security` | passwords, HMAC, hashing, sandbox |
| `13_interop` | Python / PyTorch / C |
| `14_api_server` | HTTP API for Next.js / Java |
| `15_native_c` | calling your own compiled C without copying data |
| `16_nbody` | vectorized gravity for 200 bodies |
| `17_wave` | wave on a string returns to its shape after one period |
| `18_pinn` | physics-informed network learns cos(ωt) from the equation alone |
| `19_measurement` | measurement uncertainty + rk45 |
| `20_spectrum` | FFT finds a signal's frequency (in Hz) |
| `21_curve_fit` | fit a decay to get τ ± σ; minimize, root finding, integration |
| `22_lstm_sequence` | LSTM time-series prediction |
| `23_web_api` | API server: AI model, orbit simulation, JWT login, rate limit |
| `24_low_precision` | FP64 → FP4/INT4 comparison + quantization-aware training |
| `25_data_pipeline` | parallel SSD data pipeline + prefetch + disk_cache |
| `26_model_serving` | train → safetensors → INT8 load → API |
| `27_pretrained_models` | Hugging Face / ONNX → API |
| `28_any_architecture` | `use layers`: residual MLP, a home-made DAG, a graph attention network, a small GPT |
| `29_stiff_chemistry` | stiff Robertson kinetics and an RC circuit with `sci.solve_ivp` |
| `30_gpu_training` | training on the GPU through the PyTorch backend |
| `31_native_speed` | `@native` machine code, units at zero cost, fused tensor kernels, `parallel for` on all cores |
| `32_gpu` | `@gpu` kernels: SAXPY, a reduction and a matrix multiply on the GPU(s), or all CPU cores without one |
| `33_model_files` | Apsi checkpoints (units, compression, SHA-256, signatures, sharding) and GGUF export |
| `34_statistics` | a two-group study: descriptive statistics, Welch t, Mann–Whitney, power, regression, FDR, tables |
| `35_mechanics` | a double pendulum from its Lagrangian, an exact pendulum period, a Hohmann transfer, a cantilever |
| `36_quantum` | harmonic-oscillator states, a moving wave packet, a Bell pair and its entanglement |
| `37_medical` | pharmacokinetics and NCA, Kaplan–Meier, log-rank, Cox regression, diagnostic accuracy, eGFR |
| `38_reinforcement_learning` | PPO learns CartPole; Monte Carlo tree search finds the winning move |
| `39_pytorch_converted` | a PyTorch script with BatchNorm, AdamW, cosine schedule, label smoothing and clipping after `apsi convert` |

## Tests

```bash
apsi test examples tests/apsi --allow-all   # test blocks written in Apsi (pass on --engine compile and interp)
python -m unittest discover tests           # toolchain, REPL, CLI, LSP, formatter, Windows/Linux, web server
```

## Development

Development started on **Sunday, 8 November 2026 at about 04:00** (Bangkok time, UTC+7); this release was completed the same morning at about **09:00** — about **5 hours** of development.

## Documentation

- [`WHATS-NEW.md`](WHATS-NEW.md) — **everything new in 2.4.4**, with examples, measurements and limits
- [`GETTING-STARTED.md`](GETTING-STARTED.md) — install, first program, first model, first library
- [`NATIVE.md`](NATIVE.md) — **native code**: `@native`, `parallel for`, `@gpu`, benchmarks against C, C++, Rust, Go and Mojo
- [`LIBRARIES.md`](LIBRARIES.md) — `.apsil` libraries: install, create, pack, publish, the registry API
- [`RL.md`](RL.md) — reinforcement learning and learning from preferences
- [`SCALE.md`](SCALE.md) — planning 21B–2T-parameter training runs (`apsi plan`, `use scale`)
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — `apsi` not recognized, VS Code without colours, `needs permission 'python'`
- [`LANGUAGE.md`](LANGUAGE.md) — **the complete language reference**: syntax, types, units, functions, structs, modules, every built-in, the REPL and the command line
- [`AI.md`](AI.md) — building AI with Apsi: any architecture, smaller RAM/VRAM, pretrained models, serving
- [`LAYERS.md`](LAYERS.md) — **`use layers` reference**: every layer with shapes and parameters, `dag`, graph neural networks, the model zoo, `fit`, recipes
- [`BACKENDS.md`](BACKENDS.md) — NumPy and PyTorch backends: CUDA, ROCm, Apple MPS, Intel XPU, bf16/fp16 mixed precision, `torch.compile`, `torch.nn.Module` export
- [`PYTHON.md`](PYTHON.md) — converting Python and PyTorch programs with `apsi convert`
- [`REALTIME.md`](REALTIME.md) — 1–2 kHz loops and video from 640×480 to 4K (`use rt`)
- [`MEMORY.md`](MEMORY.md) — RAM budgets, SSD spill, out-of-core data, OOM-safe batches (`use memory`)
- [`DATAFRAME.md`](DATAFRAME.md) — tables (`use dataframe`)
- [`DATABASES.md`](DATABASES.md) — PostgreSQL, Redis, SQLite (`use db`)
- [`DEBUGGING.md`](DEBUGGING.md) — the debugger
- [`ENV.md`](ENV.md) — `.env` files, secrets, Python virtual environments
- [`TOKENS.md`](TOKENS.md) — fewer LLM tokens with exactly reversible transforms (`use tokens`)
- [`SERVER.md`](SERVER.md) — secure, high-throughput servers
- [`PERFORMANCE.md`](PERFORMANCE.md) — number formats, CPU, GPU, PyTorch, RAM, SSD
- [`INTEROP.md`](INTEROP.md) — Python, PyTorch, C/C++, JS/Next.js, Java
- [`SECURITY.md`](SECURITY.md) — permissions, Apsi Guard, profiles
- [`EDITORS.md`](EDITORS.md) — VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Neovim/Vim, Notepad++, JetBrains, Helix, Zed, Emacs
- [`INSTALL.md`](INSTALL.md) · [`INSTALL-LINUX.md`](INSTALL-LINUX.md) · [`BUILD.md`](BUILD.md)
- [`PLAYGROUND.md`](PLAYGROUND.md) — the web playground
- [`ROADMAP.md`](ROADMAP.md) · [`CHANGELOG.md`](https://github.com/ATECTechnologies/apsi/blob/main/CHANGELOG.md)

## Source layout

```
apsi/            the language
  lexer.py       text → tokens (indentation, Unicode, unit literals)
  parser.py      tokens → syntax tree
  checker.py     names, units, argument counts and evolve rules, checked before running
  interpreter.py reference tree-walking runtime
  native/        the native compiler (LLVM): compiler.py (specialiser, units, fusion, parallel for → IR),
                 engine.py (JIT, -O3 for the host CPU, signed machine-code cache, thread pool),
                 __init__.py (@native, calling convention), gpu.py + cuda.py (@gpu: PTX, multi-GPU, CUDA driver)
  codegen.py     Apsi → Python bytecode compiler + erased twins
  compiler.py    closure compiler for evolve in the interpreter engine
  cli.py         the apsi command
  repl.py        the interactive REPL
  project.py     apsi new / init / task and apsi.toml
  editors.py     apsi editors: integrations for VS Code-family editors, Sublime, Vim/Neovim, Notepad++
  convert.py     apsi convert: Python → Apsi translator with library mapping
  layers.py      use layers: composable layers, DAG models, GNNs, model zoo, fit
  runner.py      runs programs on a large stack (deep recursion), forwards Ctrl+C
  tools.py       apsi eval / watch / bench / doc / completion
  integrators.py rk45 (Dormand–Prince)
  uncertain.py   measurement uncertainty and error propagation
  sci.py         einsum, FFT, integrate, find_root, minimize, curve_fit
  formatter.py   apsi fmt
  lsp.py         language server
  playground.py  runner for the web playground
  precision.py   BF16/FP8/FP6/FP4/INT8/INT4
  data.py        data pipelines and SSD storage
  concurrency.py tasks, channels, parallel loops
  web.py         HTTP server
  models.py      safetensors, model compression, ONNX / Hugging Face / PyTorch
  modelio.py     GGUF and Apsi checkpoints (.apsim), apsi model
  special.py stats.py dsp.py mechanics.py pde.py quantum.py physics.py medical.py lab.py
                 science, engineering and medical modules (docs/SCIENCE.md)
  convert_torch.py  PyTorch programs -> native Apsi (used by apsi convert)
  units.py       SI dimensions and the Quantity type
  autodiff.py    reverse-mode autodiff (incl. conv2d and attention ops)
  ops.py         math/tensor operations for numbers, tensors, quantities and autodiff nodes
  stdlib.py      built-ins and the nn / physics / linalg / random modules
  interop.py     Python, PyTorch, C FFI, files, network (through the permission system)
  crypto.py      security module
  security.py    permissions, profiles, release manifest
  guard.py       Apsi Guard pre-run threat scan
  hardware.py    apsi doctor and the GPU backend
  server.py      apsi serve
examples/        39 examples (+ native/: C code and build scripts for Windows/Linux)
tests/           test suite (tests/test_native.py: native vs bytecode differential tests)
benchmarks/      the same benchmarks in Apsi, C, C++, Rust and Go (sh benchmarks/run_all.sh);
                 heavy/: eight large workloads also in Mojo (python3 benchmarks/heavy/run.py)
docs/            documentation
editors/         editor integrations: vscode/ (VS Code, Cursor, Windsurf, VSCodium), sublime/, vim/ (Vim + Neovim),
                 notepad++/, textmate/ (JetBrains, TextMate), helix/, zed/, emacs/
install/         script installers for Windows (install.ps1 / install.bat) and Linux (install.sh)
installer/       the setup wizard (Electron + React)
playground/      browser playground
packaging/       release builds (.exe, .deb, zip, wheel)
tools/           playground build and test scripts
.github/         CI on Windows and Linux
```

Test the playground in real WebAssembly:

```bash
npm install pyodide@0.27.7 && node tools/test_playground.mjs playground/apsi_bundle.js
```

## License and credit

Apsi is developed by **ATEC** and distributed under the [Apsi License 2.0](https://github.com/ATECTechnologies/apsi/blob/main/LICENSE): free to use for any purpose,
including commercial work; the programs, libraries and models you build with Apsi are yours to license and sell;
Apsi itself may not be sold or offered as a paid service without a licence from ATEC; copies and changed versions of
Apsi credit ATEC and stay under the same licence. Plain-language guide with examples:
[`LICENSING.md`](LICENSING.md).
