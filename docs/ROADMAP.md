# Apsi roadmap

## ✅ v0.1 — the language and the reference runtime
- complete syntax: let/var, functions, closures, lambdas, pipes, structs, records, comprehensions, f-strings, Unicode
- the checker verifies names, units, constants, arguments, return values and `evolve` rules before running
- SI units, shape generics, unit-aware autodiff
- `evolve` with 5 integrators and a hot-loop compiler that erases units
- nn: dense, conv2d, pooling, layer-norm, attention, optimizers, schedulers
- physics: constants, fields, unit-preserving linalg
- permissions, crypto, Python/PyTorch/C FFI, HTTP API, `apsi doctor`, VS Code

## ✅ v0.2 — compiler and accuracy
- Apsi → Python bytecode compiler as the main engine, 5–20× faster than v0.1
- an erased twin of every function, so units are removed along the whole call path in hot loops
- higher-order autodiff (Hessian, Jacobian, Laplacian, PINNs)
- `±` uncertainty that tracks every source separately
- adaptive integrator `rk45 tol ...`
- `pmap` for parallel work

## ✅ v0.3 — real use on Windows/Linux and complete tooling
- Windows 10/11 (PowerShell) and Linux installers, BOM/CRLF support, automatic `.dll`/`.so` selection
- `apsi doctor` checks the Windows build / Linux kernel / CPU / GPU
- CI on Windows Server 2022/2025 and Ubuntu 22.04/24.04 with Python 3.9–3.13
- language server + VS Code extension (.vsix), `apsi fmt`
- web playground (WebAssembly), every example tested in it
- `use sci` (einsum, FFT, integrate, find_root, minimize, curve_fit)
- nn: LSTM, GRU, batch/RMS norm, RMSprop, Adagrad, `train()`, sample datasets

## ✅ 1.10.0 — first production release with installers (8 November 2026, about 04:00–09:00 Bangkok time)
- Windows setup wizard (.exe with its own Python), portable zip, .deb for Debian/Ubuntu, wheel
- FP64 → FP4/INT4 number formats, bit-exact with the standards; quantization-aware training
- streaming data pipelines, memory-mapped tensors on SSD, signed on-disk cache
- concurrency (tasks, channels, atomics, parallel_for)
- secure, high-concurrency HTTP server (`use web`)
- full REPL (shell commands, `:commands`, history, completion), projects (`apsi new`, `apsi.toml`, tasks),
  `watch`, `bench`, `doc`, `eval`, shell completion; complete language reference
- editor extensions installed by the setup wizard: VS Code, Cursor, Windsurf, VSCodium, Sublime Text, Vim/Neovim,
  Notepad++ (plus configurations for JetBrains, Helix, Zed, Emacs)
- `apsi convert`: Python files and folders → Apsi, with numpy/math/torch library mapping
- `use layers`: any architecture as a graph of layers, transformers, graph neural networks (GCN, GraphSAGE, GAT,
  GIN), model zoo, `fit`
- **PyTorch backend**: the same models on CUDA / ROCm / MPS / XPU with bf16/fp16 mixed precision and
  `torch.compile`; `apsi pkg install torch-auto` picks the matching CUDA build
- `sci.solve_ivp`: implicit TR-BDF2 for stiff systems and adaptive RK45
- `apsi build` → executables (`.exe`, `--standalone` with its own Python); signed compile cache, `apsi compile`

## Phase 2 (continued) — IR compiler and speed
- **Apsi IR**: lower the checked AST to an SSA IR carrying types, shapes and units
- erase units across the whole program (not just in `evolve`): check once, run on plain numbers
- kernel fusion: merge chains of element-wise ops into one loop to cut memory traffic
- IR-level (source-to-source) autodiff, so no tape is built at run time

## Phase 3 — native backends (per the hardware targets)
- CPU through LLVM: **x86-64-v3** (AVX2 + FMA, Intel Core i5-7500 / Ryzen and newer) and ARMv8.2 NEON
- GPU through MLIR → PTX: **sm_75 and newer** (Turing GTX 1660 / RTX 20xx → Ada/Blackwell), Tensor Cores for
  f16/bf16, and native FP8/FP4 on sm_89/sm_90/sm_100 (the formats are already bit-exact but computed through FP32)
- `@shard(devices=...)` to spread work over several GPUs and `@pipeline` for pipeline parallelism
- automatic GPU memory management: memory pool, async copies overlapping computation
- `apsi build --native`: machine code instead of bytecode inside the executable, and `apsi build --lib`: a shared
  library callable from C/C++/Python/JS (the executable itself exists since 1.10.0)

## Phase 4 — advanced physics and AI
- `field` and PDEs in the syntax (`∂t u = α ∇²u`) with several kinds of boundary conditions
- higher-order symplectic integrators; implicit solvers for PDEs (stiff ODEs are covered by `solve_ivp` since 1.10.0)
- diffusion and physics-informed network libraries (transformers and GNNs are in `use layers` since 1.10.0)
- an Apsi package manager for Apsi libraries (`apsi add <package>`)

## Phase 5 — ecosystem
- language server: rename, find references, tensor shapes on hover
- linter (unused variables, unreachable code)
- official website (the playground is ready to use)
