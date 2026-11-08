# Changelog

## 1.10.0 — first production release with installers

### Website and library commands (added to 1.10.0 for apsi.dev)
- **Libraries from the registry:** `apsi install NAME` (also `NAME@^1.2`, local `.apsil` files, `[dependencies]` of
  `apsi.toml` + `apsi-packages.lock`), `apsi uninstall`, `apsi pack`, `apsi publish` (resumable upload for large
  libraries), `apsi search`, `apsi lib list | info | login | logout | registry`; installed libraries load with `use NAME`.
- **Removing things again:** `apsi yank NAME VERSION [--undo]`, `apsi unpublish NAME VERSION`.
- **Releases from any computer (administrators):** `apsi release upload FILE… --version 1.10.0`,
  `apsi release list`, `apsi release delete ID`, `apsi release hide|unhide ID`,
  `apsi release bench results.json --version 1.10.0`. On the server itself: `node scripts/admin.js release …`.

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
