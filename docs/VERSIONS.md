# Versions: what each one can do

| | **2.4.4** (current) | **2.0.0** | **1.10.0** |
|---|---|---|---|
| Released | 10 November 2026 | November 2026 | 8 November 2026 |
| Engines | bytecode engine + native compiler + Python-exact mode (`use pycompat`) | bytecode engine **+ native compiler** (`@native`, LLVM machine code), `parallel for`, NVIDIA GPU kernels | bytecode engine |
| Speed on the [benchmarks](/benchmarks) | as 2.0.0; HTTP server about 3× faster, cold compile 32 % faster | next to C, C++, Rust; ahead of Go | about 500× slower than 2.0.0 (geometric mean; 12× to 2,400× per program) |
| Errors | as 2.0.0, plus `except*` and exception groups | `try / except / else / finally`, `raise`, Python kind names | `error("…")` stops the program |
| Generators, `*args/**kwargs`, operator methods on structs | yes, plus real classes, `with`, `match`, `async` | yes | no |
| Python conversion | **byte-identical behaviour** for any Python 3.13 program (verified on 34 standard-library modules) | keeps exceptions, generators, classes with operators, `for … else`, pure stdlib modules | core language and numpy; exceptions become comments |
| PyTorch | as 2.0.0 | exact conversion of whole training programs, native layers/optimizers/schedulers | `use nn` and `use layers` (PyTorch backend), conversion of models |
| Reinforcement learning, science modules | as 2.0.0 | `use rl`, special functions, statistics, DSP, PDE, quantum, mechanics, medical, lab | AI and physics basics |
| Large models | as 2.0.0 | `apsi plan`, model files (GGUF, safetensors, `.apsim`), sharding | — |
| Libraries (`apsi install / pack / publish / yank / unpublish`) | as 2.0.0 | yes | yes |
| Website commands (`apsi release …`) | as 2.0.0 | yes | yes |
| Standard modules | 42 modules, 1092 members, 298 built-ins | 34 modules, 833 members, 253 built-ins | 22 modules, 360 members, 201 built-ins |
| Real time, video | `use rt`: 1–2 kHz loops, video 640×480 → 4K at any size | — | — |
| RAM / SSD / GPU management | `use memory` | — | — |
| Fewer AI tokens | `use tokens` (exactly reversible) | — | — |
| Tables, databases | `use dataframe`, `use db` (PostgreSQL, Redis, SQLite) | — | — |
| WebSocket, `.env`, debugger | yes | — | — |
| Documentation | [2.4.4 docs](/docs/2.4.4) | [2.0.0 docs](/docs/2.0.0) | [1.10.0 docs](/docs/1.10.0) |

## Which one should I use?

**2.4.4** for everything new. Programs written for 2.0.0 and 1.10.0 run on 2.4.4 unchanged — each version only adds.

Keep **2.0.0** or **1.10.0** only if a machine cannot install 2.0.0's native compiler dependency (`llvmlite`); even then 2.0.0
works without it, on the bytecode engine, so the reasons are rare.

## Moving from 2.0.0 to 2.4.4

Install 2.4.4 (it replaces 2.0.0) and run `apsi test`. Python files you converted with 2.0.0 can be converted again
(`apsi convert file.py`) to get Python's exact behaviour.

## Moving from 1.10.0 to 2.0.0

1. Install 2.0.0 (it replaces 1.10.0).
2. Run your tests: `apsi test`.
3. Add `@native` to the functions that do the heavy loops — see [maximum performance](/docs/2.0.0/best-performance).
4. If you used `error()` for things you would like to recover from, switch to `raise` and `try / except`.
