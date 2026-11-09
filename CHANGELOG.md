# Changelog

## 2.0.0 — native machine code, all CPU cores and NVIDIA GPUs

Apsi 2.0 replaces "everything runs on the Python VM" with a native compiler. Python stays where it helps (the
tooling, libraries, the rest of the language), while numeric code runs as machine code built through LLVM.

### Python compatibility — write Apsi the way you write Python
- **Exceptions:** `try / except Kind, Other as e / else / finally` and `raise` (with Python's kind names; errors Apsi
  raises itself — division by zero, missing keys, bad indices, unit mismatches — match them too). Own error types
  are structs with a `message` field.
- **Generators:** `yield` and `yield from` produce values lazily; endless generators work.
- **`*args` / `**kwargs`** in definitions and `*xs` / `**d` in calls.
- **Operators on structs:** methods `add sub mul div eq lt len get contains iter to_str call hash …` are used by
  `+ - * / == < len() [] in for print() …` (the conversion of Python's special methods).
- **Pure standard modules without permissions:** `pyimport("collections")`, heapq, itertools, functools, bisect,
  re, string, copy, datetime, fractions, decimal and more need no `--allow-python` (only public, non-module
  members are exposed, so nothing reaches files, network or processes).
- New built-ins `bor band bxor shl shr bnot` (Python's `| & ^ << >> ~` for ints, sets, dicts and boolean tensors),
  `chain_lists`, `set`, `frozenset`; `json_str(…, indent=, sort_keys=)`; named tuples print as `Point(x=1, y=2)`.
- **`apsi convert` keeps behaviour instead of commenting it out:** `try/except` blocks, `raise`, generators, variadic
  functions, `for/while … else`, class attributes, special methods, `field(default_factory=…)`, nested
  comprehensions, `filter(None, …)`, `sum(xs, start)`, `nonlocal`. `tests/test_python_compat.py` converts real
  Python programs and requires identical output.

### Website commands (from any computer)
- `apsi release upload FILE… --version V` (platform detected from the file name, resumable 32 MB pieces),
  `apsi release list / delete ID / hide / unhide`, `apsi release bench results.json --version V`.
- `apsi yank NAME VERSION [--undo]`, `apsi unpublish NAME VERSION`.
- `benchmarks/heavy/run.py`: one command, live-updating table in the terminal, `--publish V` updates
  apsi.dev/benchmarks while it runs, `--apsi-root` measures another Apsi version.

### Native compiler (`@native`) — new architecture
- **Checked AST → specialised, unit-erased LLVM IR → machine code for the host CPU** (`-O3`, loop and SLP
  vectorisation, AVX2 / AVX-512 / FMA / NEON as available). Each function is specialised on the types, tensor
  dtypes/ranks/layouts and physical units of its arguments; types are inferred by abstract execution.
- **Units at zero cost:** dimensional analysis happens at compile time (wrong units are a `UnitError`), the machine
  code works on SI numbers, and results get their units back. Argument units are still checked on every call.
- **Safe by default:** 64-bit integer overflow, index range and integer/float floor-division-by-zero checks with
  source lines; Python semantics for `//`, `%`, negative indices, `min`/`max`, `round`. Strict IEEE arithmetic, so
  results are bit-identical to the bytecode engine. Index checks are **hoisted out of inner loops** by proving
  their range before the loop (the loop is versioned, behaviour unchanged), which lets LLVM vectorise.
- **`@native("fast")`**: checks off, `nsw` integers, fast-math — for proven code. **`@native("strict")`**: an error
  instead of a fallback.
- **Tensors:** direct element access on contiguous and strided tensors, row and slice views without copies,
  **fused** element-wise expressions with NumPy broadcasting (one pass, no temporaries), fused reductions
  (`sum mean min max argmin argmax prod dot norm`), `zeros/ones/full/*_like`, zero-copy tensor returns, and
  per-iteration memory arenas so temporaries created in loops are freed every iteration.
- **Never breaks a program:** unsupported constructs or argument kinds fall back to the bytecode engine with one
  note naming the construct and line; `grad` of a native function, `evolve` calling native right-hand sides, and
  native ↔ bytecode calls all work.
- **Signed machine-code cache** (`~/.apsi/cache/native`, HMAC-SHA256 with the per-user key), keyed by IR, CPU,
  CPU features, LLVM and Apsi versions.
- **Benchmarks** (`benchmarks/`, [`docs/NATIVE.md`](docs/NATIVE.md#performance)): 100–3,500× faster than 1.10;
  `@native("fast")` fastest or tied single-threaded on Mandelbrot, spectral-norm, sieve and fused-array sums
  against GCC, G++, Clang, Rust and Go, within 6–20 % on n-body and matmul.

### Exact semantics
- Native code now follows the bytecode engine **exactly**, including where Python numbers and NumPy scalars
  (tensor elements) obey different rules (division by zero, Int overflow, `//` and `%` by zero, `0.0 ** -1`): every
  value's origin is tracked at compile time, or with a one-bit run-time flag when it depends on the path taken,
  through variables, branches, `min`/`max` and function returns. Results come back as the same Python class.
- `tests/test_native_fuzz.py`: seeded differential fuzzing of random programs with Python and NumPy arguments —
  values, errors and result classes must all match (0 differences in 14,400 calls). Bugs it found and that are fixed:
  `-0.0 // nan` (CPython treats NaN as true), `floor`/`ceil` return Ints like the bytecode engine, provenance lost
  through `abs` and Bool → Int.

### Parallelism
- **Native thread runtime**: dispatcher and workers in machine code, spinning workers with `pause`, a seqlock-style
  job hand-off, one cache line per shared field and guided chunk claiming — a parallel region starts in ~5 µs
  instead of ~70 µs; parallel speed-ups improved accordingly (spectral-norm 15.9 → 12.6 ms, matmul 20.9 → 15.0 ms
  on 2 cores). Stress-tested with hundreds of thousands of regions, errors mid-region, nesting, thread-count changes
  and concurrent callers, including under gdb; this found and fixed a use-after-return race in the error path and
  an out-of-bounds read.
- **Call latency** from bytecode into native code: ~0.9 µs (was ~8 µs) through a signature cache, a direct typed
  entry point and tensors passed as the NumPy object (struct layout measured and verified at start-up); Ints beyond
  64 bits are routed to the bytecode engine instead of being truncated.
- **`parallel for`** (soft keyword): iterations on all CPU cores in native code; deterministic reductions
  (`+= -= *=`, `x = min(x, …)`, `x = max(x, …)`) that give **bit-identical results for any thread count**; one
  thread wake-up per core per loop (chunks are claimed by native workers with an atomic counter). An ordinary `for`
  on the other engines, so programs stay portable.
- **Automatic multi-core kernels:** fused tensor expressions and `sum`/`mean`/`prod`/`min`/`max` over 131,072 or more
  elements run on all cores without any code change, with deterministic chunked combination;
  `native_threads(n)`, `APSI_THREADS`, `apsi run --threads N`.

### GPUs (`@gpu`)
- A `parallel for` becomes a CUDA kernel: Apsi generates **PTX for sm_75** (GTX 1660 / RTX 20xx and newer) with
  LLVM's NVPTX backend and loads it through the **CUDA driver API** (`nvcuda.dll` / `libcuda.so.1`) — no CUDA
  toolkit needed. `+=` reductions use GPU atomics.
- **Multi-GPU:** iterations are split across all GPUs (`APSI_GPUS` to choose); each GPU copies back only the rows it
  wrote. Without a GPU, or for kernels outside the GPU rules, the function runs on all CPU cores instead.
- `gpu_info()`; `use gpu`'s `gpu_info()` now works with only the NVIDIA driver (it required CuPy and crashed without it).

### Large models
- **`apsi plan` / `use scale`**: plans training of 1B–2T-parameter models, dense or mixture-of-experts, BF16 or
  FP8, on 13 GPU types: tensor × pipeline × data-parallel layout, recomputation, memory per GPU, GPU count, nodes,
  days, GPU-hours, dataset size and serving memory, with the standard published accounting
  ([`docs/SCALE.md`](docs/SCALE.md)).
- **`@` in native code** runs on the multi-threaded BLAS (the same kernels as NumPy), so matrix-heavy native
  functions get BLAS speed plus fused element-wise work around it.

### Science, engineering and medicine
- New standard modules, all verified against reference implementations (`tests/test_science.py`):
  - `use special`: gamma, digamma, beta, error functions and inverses, regularized incomplete gamma/beta and
    inverses, Bessel J and I, elliptic K and E, Legendre/associated Legendre, Hermite, Laguerre, Chebyshev,
    spherical harmonics, ζ, E1. Relative errors are 1e-16 to 1e-13 against SciPy.
  - `use stats`: 11 distributions (pdf/pmf, cdf, sf, ppf, sample), descriptive statistics, t-tests, ANOVA,
    χ², Fisher's exact test, Mann–Whitney, Wilcoxon, Kruskal–Wallis, exact two-sample KS, Jarque–Bera,
    Pearson/Spearman/Kendall, binomial test, Wilson and Clopper–Pearson intervals, OLS and logistic regression,
    power and sample size (exact noncentral t), bootstrap, permutation tests, Bonferroni/Holm/BH/BY. Results agree
    with SciPy and statsmodels to 1e-10 or better.
  - `use signal`: Butterworth design (second-order sections), zero-phase filtering with SciPy-identical edges,
    notch, windows, convolution, peak detection (height, distance, prominence), Welch, spectrogram,
    Savitzky–Golay, resampling, Hilbert envelope.
  - `use mechanics`: Verlet, 4th-order Yoshida and RK4 integration; Hamiltonian and Lagrangian dynamics whose
    equations of motion come from automatic differentiation; N-body gravity with energy and momentum
    tracking; Kepler's equation, orbital elements, Hohmann transfers; inertia tensors, quaternions and torqued
    rigid bodies; normal modes; Euler–Bernoulli beams and plane trusses by finite elements.
  - `use pde`: heat (Crank–Nicolson, ADI), wave, advection, Poisson by fast sine/Fourier transforms, 2-D
    incompressible Navier–Stokes (pseudo-spectral), conjugate gradients, BiCGSTAB, batched tridiagonal solver.
  - `use quantum`: Schrödinger bound states (fourth-order) and split-operator time evolution, expectation values
    and uncertainties, operators, spin matrices, qubit gates and circuits, measurement sampling, density matrices,
    partial trace, entropy, fidelity, concurrence, hydrogen.
  - `use physics` gains formulas with units: Planck, Wien, Stefan–Boltzmann, Lorentz transformations, Doppler,
    Snell/Fresnel/Brewster, thin lens, diffraction, Coulomb fields, exact Biot–Savart, Lorentz force, kinetic
    theory, decay.
  - `use medical`: one- and two-compartment pharmacokinetics with repeated dosing, non-compartmental analysis;
    SIR/SEIR, final size, incidence rates, RR/OR/RD/NNT, Mantel–Haenszel; diagnostic accuracy with confidence
    intervals, ROC with DeLong intervals and paired comparison, Cohen's kappa, Brier score; Kaplan–Meier, log-rank,
    Cox regression (Efron ties) and Harrell's C, matching lifelines; eGFR (CKD-EPI 2021), Cockcroft–Gault, QTc, BSA,
    BMI; ECG R-peak detection and HRV; DICOM reading (pixel-identical to pydicom, identifiers left out by default)
    with 3-D series and windowing. Marked as research and education software, not a medical device.
  - `use lab`: one seed for every generator, provenance records, JSON-lines experiment logs, measurement
    formatting, Markdown/LaTeX/CSV tables.

### PyTorch programs → native Apsi
- New module `use torch`: PyTorch's API implemented natively with PyTorch's parameter names and exact results —
  layers (convolution with groups/dilation/`"same"`, pooling, BatchNorm with running statistics, LayerNorm, RMSNorm,
  GroupNorm, Embedding, multi-layer and bidirectional LSTM/GRU/RNN, MultiheadAttention with every mask option,
  Transformer encoder/decoder), losses with all options (`weight`, `ignore_index`, `label_smoothing`, `pos_weight`,
  `reduction`), eight optimizers, fifteen learning-rate schedulers including `OneCycleLR` and `ReduceLROnPlateau`,
  `DataLoader` over tensors or your own `Dataset` classes, `state_dict` / `load_state_dict`, gradient clipping and
  gradient checkpointing.
- `apsi convert` now targets `use torch`, so schedulers, BatchNorm, attention, RNN states, custom datasets and loss
  options are converted exactly instead of being reported for review. A full training run (convolution, BatchNorm,
  LSTM, attention, AdamW, cosine annealing, clipping, label smoothing) ends with the same weights as PyTorch to 1e-9.
  `torch.save` / `safetensors` files keep PyTorch's names, so weights move between PyTorch and Apsi unchanged.
- Training state is computed once per step (`value_and_grad_aux`): values from the forward pass stay available
  without a second forward pass.
- New: `layers.custom_layer`, `layers.load_torch_weights`, `nn.loader`, more `nn` losses, the built-ins `swapaxes`,
  `permute`, `unsqueeze`, `squeeze`, a differentiable `erf` and `clip`, and the exact GELU.

### Reinforcement learning (`use rl`)
- Vectorised environments (CartPole identical to Gymnasium's, grid world, bandits), GAE, PPO (reaches the maximum
  CartPole return), double DQN, REINFORCE, DPO / IPO / Bradley–Terry / GRPO / KL-penalised RLHF objectives,
  Monte Carlo tree search, beam search and best-of-n. Guide: `docs/RL.md`.

### Memory
- Gradient checkpointing (`torch.checkpoint`, `layers.checkpoint`): about a third of the peak memory for twice the
  compute, with identical gradients (dropout masks are replayed).
- float32 training keeps parameters, gradients and optimizer state in float32 end to end (`fit(…, precision="fp32")`).

### Libraries (`.apsil`)
- New package format and registry client: `apsi install NAME` (newest version, semantic-version ranges, dependencies,
  lock file), `apsi uninstall`, `apsi pack` (checks the code, builds a reproducible `dist/NAME-VERSION.apsil`),
  `apsi publish` with API tokens, `apsi search`, `apsi lib list / info / registry / login`; the official registry is https://apsi.dev. Every download is
  verified by SHA-256 per file and scanned by Apsi Guard. Installed libraries are used with `use NAME`.
- `use NAME` now also makes the module available as `NAME.member` (unless the name is taken).
- `apsi new NAME --template lib` creates a publishable library.

### Benchmarks
- New heavy suite in `benchmarks/heavy/` against C, C++, Rust, Go and Mojo with checksum verification.

### Requirements
- Apsi now needs **Python 3.12 or newer** (tested with 3.12, 3.13 and 3.14), NumPy 1.26+ and llvmlite 0.43+. The
  Windows installer and the portable Linux build bring their own Python 3.12; `import apsi` on an older Python
  gives a clear message.

### Model files
- **Apsi checkpoints (`.apsim`)**: `save_model` / `load_model` / `model_info`. Standard safetensors files that
  also keep physical units, compressed tensors (INT8/INT4/FP8/FP6/FP4/BF16) at their real size, and the parameter
  tree. A SHA-256 over the tensor table and data is checked when loading. Optional HMAC-SHA256 signatures use a
  per-user key or a shared key. Sharding with a Hugging Face style index supports 70B–2T-parameter checkpoints.
  Writes are atomic, and loading is memory-mapped.
- **GGUF**: `save_gguf` / `load_gguf` / `gguf_info`, version 3, F32/F16/BF16/Q8_0/Q4_0 and integer tensors, all
  metadata types. Q8_0 and Q4_0 output is bit-identical to the reference `gguf` package (0 of 1,331,200 bytes
  differ over 200 random matrices).
- `apsi model info | verify | convert` converts between safetensors, GGUF and `.apsim`.
- Hardened parsers: header sizes, offsets and tensor sizes are checked before mapping, and fuzzed files produce
  clear errors instead of crashes. The model functions are covered by permission checks and `apsi audit`.

### Tools
- `apsi native info | explain FILE | ir FILE FN | asm FILE FN | clear-cache`; `apsi run --native` (every function
  that allows it becomes machine code); `apsi doctor` reports the native compiler and the GPUs.
- New built-ins `native`, `gpu`, `native_threads`, `native_info`, `gpu_info`; editor grammars and snippets
  (`native`, `pfor`, `gpu`) for `parallel`, `@native` and `@gpu` in every supported editor.
- Examples `31_native_speed.apsi` and `32_gpu.apsi`; `tests/test_native.py` runs every native construct against the
  bytecode engine and requires identical results and errors.
- [`docs/NATIVE.md`](docs/NATIVE.md) (native code reference and benchmarks) and
  [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) (`apsi` not recognized and PATH, VS Code colours and
  Ctrl+F5, `needs permission 'python'`).

### REPL
- The prompt shows the current folder, like a shell: `C:\Users\me\project apsi>` (`~/project apsi>` on Linux; long
  paths shortened in the middle; `APSI_PROMPT=short` for the plain `apsi>`). `cd` no longer prints the folder,
  since the prompt shows it.
- `ls` / `dir` print an aligned table: folders first, then files, with modification time and size, a header and a
  summary line (`1 folder, 2 files (2.9 KB), 1 hidden (ls -a)`); empty folders say so.
- A line ending in an operator (`let x = 1 +`) now continues correctly: the REPL already waited for the next line,
  but the two lines were then rejected as a syntax error.

### Fixes
- **Windows setup wizard build:** the wheel builder read `pyproject.toml` with a regular expression and wrote the
  dependency `pyreadline3>=3.4; sys_platform == \"win32\"` with the escaping left in, which pip rejects
  (`invalid metadata: Expected a marker variable or quoted string`). It now uses a real TOML parser (`tomllib`,
  with a correct fallback for Python 3.9/3.10) and refuses to write malformed requirements.
- The wheel builder, `apsi build`, the installation manifest (`apsi verify`) and the playground bundle now include
  subpackages (previously only top-level files were packaged and verified).
- Installers say clearly that terminals and editors opened before installation keep the old PATH, and the Linux
  installer prints the command that fixes the current shell.

### Packaging
- New dependency: `llvmlite` (prebuilt wheels for Windows and Linux; it contains LLVM). The Windows payload and the
  Linux portable build bundle it; the shell installer installs it into a private environment when the system lacks
  it; the `.deb` recommends `python3-llvmlite`. Without it Apsi still runs, on the bytecode engine.

## 1.10.0 — first production release with installers

Development started on **Sunday, 8 November 2026 at about 04:00** (Bangkok time, UTC+7); this release was completed the same morning at about **09:00** — about **5 hours** of development.

### Command line and REPL
- **New REPL**: shell words work as in a terminal (`cd C:\Users\me\project`, `cd -`, `pwd`, `ls`/`dir`, `clear`/`cls`,
  `run app.apsi`, `exit`), any operating-system command with `!cmd`, and 26 `:commands` (`:help`, `:vars`, `:type`,
  `:doc`, `:time`, `:bench`, `:load`, `:reload`, `:run`, `:check`, `:test`, `:fmt`, `:edit`, `:paste`, `:save`,
  `:history`, `:ast`, `:tokens`, `:reset`, `:perms`, `:env`, …)
  - automatic multi-line input (open brackets, trailing `:`, `\`, `=`, `,` or an operator), `_` holds the last result
  - persistent history, Tab completion of names, fields, `:commands` and paths (pyreadline3 on Windows),
    colours (respects `NO_COLOR`), Ctrl+C cancels the current input instead of quitting, helpful hints after errors
- **Shortcuts**: `apsi` alone starts the REPL (or runs a program piped on stdin), `apsi FILE.apsi ARGS` runs a file,
  `apsi -e "CODE"` evaluates code, `apsi NAME` runs an `apsi.toml` task, mistyped commands get "did you mean"
  - permission flags may come first and never swallow the file: `apsi --allow-python app.apsi`,
    `apsi run --allow-read app.apsi`, `apsi --allow-python -e "CODE"`
- **Projects**: `apsi new NAME --template app|lib|physics|ai|server`, `apsi init`, `apsi.toml`
  (`[project] main`, `[run] profile/allow`, `[tasks]`), `apsi task [NAME]`; `apsi run` / `apsi test` without
  arguments use the project
- **New commands**: `apsi eval`, `apsi watch` (re-run / re-test / re-check on save), `apsi bench`, `apsi doc`,
  `apsi completion bash|zsh|fish|powershell`, `apsi help COMMAND`
- `apsi run`: program arguments (read them with `args()`), `--time`, `--watch`, `-` for stdin;
  `apsi check` and `apsi fmt` accept folders and default to the current folder; coloured error messages
- New built-ins: `args()`, `input(prompt)`, `exit(code)`
- **`apsi build app.apsi`** makes an executable (`app.exe` on Windows, `app` on Linux) containing the program, its
  imports, the parsed program, Apsi as bytecode and the permissions chosen at build time; `--standalone` (Windows)
  adds its own Python runtime with the packages the program uses, `--windowed`, `--add` for data
- **compile cache**: every program file is parsed once per process (previously the Guard scan, the checker and the run
  each parsed it) and kept in a signed on-disk cache (`~/.apsi/cache/compiled`); `apsi compile PATHS` fills it ahead
  of time, `--runtime`, `--stats`, `--clear`, `APSI_NO_CACHE=1`

### Python to Apsi
- **`apsi convert`** translates Python files and whole folders: let/var inference, block-scope hoisting,
  functions, closures, lambdas, classes and dataclasses (with inheritance), generators, comprehensions, f-strings,
  nested and starred unpacking, `with open`, try/except, comments and docstrings
- library mapping for numpy, math, random, json, time, os.path, sys, statistics, PyTorch tensors, `torch.nn`
  layers (→ `use layers`) and `torch.nn.functional`; every other library keeps working through `pyimport`
- every place that is not an exact translation is reported with its line; the result is checked automatically;
  `--run` converts, checks and runs; **Apsi: Convert Python File to Apsi** in VS Code/Cursor

### AI
- **`use layers`**: any architecture as a graph of layers — `sequential`, `residual`, `parallel`, `dag` (any directed
  acyclic graph with several inputs/outputs), dense, convolution, pooling, LSTM/GRU, multi-head and cross attention,
  transformer blocks, normalisation, dropout, 12 activations
- **graph neural networks**: `graph()`, GCN, GraphSAGE, GAT, GIN, graph pooling and differentiable
  `segment_sum/mean/max/softmax` (gradients verified against finite differences)
- model zoo: `mlp`, `resnet_mlp`, `cnn`, `lstm_classifier`, `transformer_lm`, `transformer_classifier`, `autoencoder`,
  `vae`, `gcn_net`, `gat_net`, `gin_net`, `siamese`; `fit` (mini-batches, graph masks, validation), `predict`, `summary`
- **PyTorch backend**: every `use layers` model trains and runs on PyTorch with `fit(…, backend="torch")` — CPU,
  NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU; precisions fp64/fp32/tf32 and bf16/fp16 mixed precision (automatic
  loss scaling), `torch.compile`, fused attention, cuDNN LSTMs; `backend="auto"`, `devices()`,
  `torch_module(model, params)` (a real `torch.nn.Module`) and `params_from_torch`; `check_torch()` compares every
  layer's outputs and gradients with NumPy. Needs `--allow-python`. Guide: `docs/BACKENDS.md`
- `apsi pkg install torch-auto` (also used by the `ai` group and the setup wizard) picks the PyTorch build from the
  NVIDIA driver version — CUDA 12.8 / 12.6 / 12.4 / 11.8 — ROCm for AMD on Linux, else CPU;
  explicit builds: `apsi pkg install torch-cuda` (CUDA 12.6) · `torch-cu128` (RTX 50) · `torch-cu124` · `torch-cu118` · `torch` (CPU):
  PyTorch is taken from PyTorch's own index (a CPU-only wheel from PyPI can no longer slip in), an installed build is
  kept, a different build replaces it
- `fit` mini-batches models with several inputs (`dag(inputs=[…])`, `siamese`); new `mean_pool(axis)` layer
- new complete reference `docs/LAYERS.md` (every layer with shapes and parameters, recipes, error messages)

### Physics, mathematics, data
- `sci.solve_ivp`: implicit, L-stable, adaptive TR-BDF2 for stiff systems and adaptive Dormand–Prince RK45, with units
  and dense output (`t_eval`); reproduces the Robertson kinetics reference values
- `fsum` (correctly rounded sums) and Python-compatible built-ins (`any`, `all`, `divmod`, `pow`, `ord`, `chr`, `dict`,
  `set`, `tuple`, `iter`, `next`, `format`, `getattr`, `hasattr`, …)
- data pipelines: `from_jsonl`, `from_parquet`, `interleave`, `window` (sliding windows), `shard`
- new examples `28_any_architecture` and `29_stiff_chemistry`

### Editors
- **one command for every editor**: `apsi editors [list|install|uninstall]`, also run by the setup wizard (component
  *Code editor support*, listed on the *System check* page) and undone by the uninstaller
- VS Code family — **VS Code, Cursor, Windsurf, VSCodium, VS Code Insiders, Trae** — get the same extension (through
  their command-line tool, or unpacked straight into their extension folder when the tool is missing)
- **Sublime Text** package: colours, build system with variants and clickable errors, command palette, comments,
  indentation, Goto Symbol, snippets
- **Neovim / Vim** package: filetype, colours, indentation, `:make`, mappings, help file; the language server starts
  automatically in Neovim
- **Notepad++** user-defined language with Apsi colours
- **JetBrains** TextMate bundle, **Helix**, **Zed** and **Emacs** (`apsi-mode`) configurations
- `tools/build_editors.py` keeps the Sublime/TextMate grammar and snippets identical to the VS Code ones

### VS Code extension
- richer grammar: definitions vs. calls, built-ins, parameters, types and generic types, units in their own colour,
  `as [unit]`, `±`, derivatives, f-string formats, triple-quoted strings, decorators, `TODO`, Apsi code blocks in Markdown
- **Apsi Dark** / **Apsi Light** themes, snippets, smarter indentation and folding
- commands: Run, Run with Arguments, Run Project, Test, Test Project, Check, Watch, Benchmark, Audit, REPL,
  Run Selection in REPL (`Shift+Enter`), Load File into REPL, Run Project Task, New Project, Show Documentation,
  Run Apsi Command, Restart Language Server; ▶ button, status-bar button, right-click menus, key bindings
- task provider for `apsi.toml` tasks and an `$apsi` problem matcher
- the file icon is now the Apsi logo letters only, on a transparent background

### Installers
- **`ApsiSetup-1.10.0-linux.sh`**: one-file installer for every Linux distribution (and WSL), no root, built offline
  by `packaging/make_linux_installer.py`: commands, `.apsi` file type and icons, *Run with Apsi*, templates, editor
  support, `uninstall.sh`; uses the system Python and installs numpy privately only when it is missing
- the wheel is built with the standard library only (`packaging/make_wheel.py`), so release builds work offline;
  the `.deb` now carries the release manifest (`apsi verify` passes)
- Windows: deep recursion works again — the program thread asked for a 256 MiB stack, which Windows refuses (the
  limit is below 256 MiB), so programs fell back to a 1,000-level limit; now 255 MiB with smaller fall-backs
- **Windows 10/11**: `ApsiSetup-1.10.0-x64.exe` contains Python 3.12 and numpy — nothing else to install, no administrator rights
  - adds `apsi` to PATH, creates Start-menu entries and registers `.apsi` files
  - `.apsi` files: logo icon (letters only, transparent, every standard size 16–256), *Run with Apsi*, *Edit*
    (VS Code when installed), *Test with Apsi*, and **New → Apsi source file**; Explorer refreshes the icon immediately
  - uninstall from Settings → Apps
- the setup wizard shows its *Preparing Apsi Setup…* window immediately and keeps it on top of other windows until the
  wizard is ready, then opens the wizard in its place; a balanced 900 × 610 window; software rendering, crash and
  load-failure recovery, and helpers that can never crash Setup (more stable on every GPU driver); English-only locales
  make the package smaller and faster to unpack
- Linux: `.apsi` file icons in every size, *New Document → Apsi file* when `~/Templates` exists, more terminals
  for *Open the REPL*
- **Portable Windows**: `apsi-1.10.0-windows-x64.zip` — unzip and use
- **Linux (Debian/Ubuntu)**: `sudo apt install ./apsi_1.10.0_all.deb`
- **Any system**: `pip install apsi-1.10.0-py3-none-any.whl`

### Security (see docs/SECURITY.md)
- **Apsi Guard** scans programs automatically before they run; `apsi scan` scans on demand (Microsoft Defender/ClamAV with
  `--av`, YARA rules and block lists)
  - detects Trojan Source, look-alike characters, password or key theft, data exfiltration, start-up persistence,
    crypto-miners, ransomware patterns, sandbox escapes and obfuscated code
- **Ransomware shield**: stops a program that rewrites or deletes many existing files in a short time, or touches a decoy file
- **Protected locations**: no program may overwrite `~/.apsi`, the runtime, system folders or start-up files; reading
  password stores needs an explicit, specific permission
- **Network guard**: blocks SSRF (cloud metadata, link-local), checks every redirect and limits response sizes
- `--allow-python` allows ordinary libraries only; subprocess, os and ctypes need `--allow-python=all`
  - `--allow-ffi=lib@sha256` pins native libraries by hash; libraries in temp folders are refused
- the installed runtime runs Python isolated (`-I`)
- **Setup**: checks the SHA-256 of every payload file before installing, validates the target folder, enables Electron fuses,
  checks the sender of every IPC message and whitelists every option

### Setup wizard (Electron + React)
- Express and Custom modes: system check, folder (validated while you type), components, optional packages
  (GPU/ONNX/PyTorch/Transformers), security profile and Apsi Guard, summary, progress with rollback
- maintenance mode: Modify, Repair, Uninstall; `--silent` for unattended installs
- the Apsi logo in every standard size (ICO 16–256, PNG 16–1024) for Setup, NSIS, `.apsi` files, VS Code and the playground

### Package commands
- `apsi pkg list | install | remove | lock | audit` installs library bundles (gpu, onnx, torch, transformers, sci, data, ai, all)
  - binary wheels only, locked with hashes
- `apsi scan`, `apsi audit`, `apsi verify`, `apsi config`
- `setup.ps1` / `setup.sh` install, test and build the installers in one command

### Language and libraries
- `use precision`: FP64, FP32, FP16, BF16, FP8 (E4M3/E5M2), FP6 (E3M2/E2M3), FP4 (E2M1), INT8, INT4, UINT8
  - bit-exact with the standards, bit-packing, per-channel quantization, quantization-aware training and `qmatmul`
- `use data`: streaming pipelines (parallel map, prefetch, shuffle, batch), memory-mapped tensors on SSD, signed disk cache
- `use parallel`: tasks, channels, locks, atomics, parallel_for/reduce, set_threads, bench
- `use web`: HTTP server for many concurrent requests (asyncio + worker pool + multi-process), secure by default
  - JWT, API keys, rate limits, CORS, static files, metrics, TLS
- `use gpu`: `device()`, `to_device()`, `gpu_info()`; autodiff works on CuPy tensors
- `use models`: memory-mapped safetensors read/write, compressed loading (INT8/FP8/FP4), `quantize_model`, `qlinear`,
  ONNX Runtime, Hugging Face, PyTorch
- a line starting with `|>` continues the previous line
- `from_tensors(..., batch=N, shuffle=true)` yields vectorized batches
- new examples: `23_web_api`, `24_low_precision`, `25_data_pipeline`, `26_model_serving`, `27_pretrained_models`

### Documentation
- everything is in English
- [`docs/LANGUAGE.md`](docs/LANGUAGE.md): one complete reference for the syntax, the type system, units, functions,
  structs, modules, every built-in and module member, the REPL, the command line and `apsi.toml`
  (replaces `SPEC.md` and `STDLIB.md`)

## 0.3.0
- script installers for Windows/Linux, CI, language server, VS Code extension, `apsi fmt`, web playground, `use sci`, LSTM/GRU

## 0.2.0
- Apsi → bytecode compiler, higher-order autodiff, `±` uncertainty, `rk45`

## 0.1.0
- the language, units, shape generics, autodiff, `evolve`, nn, permissions, interop
