# Versions: what each one can do

| | **2.0.0** (current) | **1.10.0** |
|---|---|---|
| Released | November 2026 | 8 November 2026 |
| Engines | bytecode engine **+ native compiler** (`@native`, LLVM machine code), `parallel for`, NVIDIA GPU kernels | bytecode engine |
| Speed on the [benchmarks](/benchmarks) | next to C, C++, Rust; ahead of Go | about 500× slower than 2.0.0 (geometric mean; 12× to 2,400× per program) |
| Errors | `try / except / else / finally`, `raise`, Python kind names | `error("…")` stops the program |
| Generators, `*args/**kwargs`, operator methods on structs | yes | no |
| Python conversion | keeps exceptions, generators, classes with operators, `for … else`, pure stdlib modules | core language and numpy; exceptions become comments |
| PyTorch | exact conversion of whole training programs, native layers/optimizers/schedulers | `use nn` and `use layers` (PyTorch backend), conversion of models |
| Reinforcement learning, science modules | `use rl`, special functions, statistics, DSP, PDE, quantum, mechanics, medical, lab | AI and physics basics |
| Large models | `apsi plan`, model files (GGUF, safetensors, `.apsim`), sharding | — |
| Libraries (`apsi install / pack / publish / yank / unpublish`) | yes | yes |
| Website commands (`apsi release …`) | yes | yes |
| Standard modules | 34 modules, 833 members, 253 built-ins | 22 modules, 360 members, 201 built-ins |
| Documentation | [2.0.0 docs](/docs/2.0.0) | [1.10.0 docs](/docs/1.10.0) |

## Which one should I use?

**2.0.0** for everything new. Programs written for 1.10.0 run on 2.0.0 unchanged — 2.0.0 only adds.

Keep **1.10.0** only if a machine cannot install 2.0.0's native compiler dependency (`llvmlite`); even then 2.0.0
works without it, on the bytecode engine, so the reasons are rare.

## Moving from 1.10.0 to 2.0.0

1. Install 2.0.0 (it replaces 1.10.0).
2. Run your tests: `apsi test`.
3. Add `@native` to the functions that do the heavy loops — see [maximum performance](/docs/2.0.0/best-performance).
4. If you used `error()` for things you would like to recover from, switch to `raise` and `try / except`.
