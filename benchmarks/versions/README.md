# One Apsi version against another

```bash
python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . --web
```

Runs `programs.apsi` (code that runs unchanged on both versions), the start-up time, `apsi check` of every example
with an empty cache and, with `--web`, the HTTP server under load, for both versions on the same machine, and writes
`versions.json` and `versions.md`. Rows whose printed checks differ between the versions are left out.

## 2.4.4 against 2.0.0

2-core cloud VM (Intel Xeon @ 2.30 GHz), Linux, Python 3.13 (`versions-2.4.4-vs-2.0.0.json`):

| Program | 2.0.0 | 2.4.4 | |
|---|---:|---:|---:|
| matmul 300, loops (compiled engine) | 23,648 ms | 3.7 ms | **6,374×** |
| matmul 512, `@native` loops ¹ | 125,377 ms | 417 ms | **300×** |
| 200k objects: create and sum | 2,715 ms | 293 ms | **9.3×** |
| start-up: `apsi run hello.apsi` | 894 ms | 133 ms | **6.7×** |
| `apsi check`, 41 example programs, empty cache | 35,869 ms | 7,319 ms | **4.9×** |
| web server, 256 connections | 6,140 req/s | 19,553 req/s | **3.2×** |
| web server, 64 connections | 6,501 req/s | 19,919 req/s | **3.1×** |
| fib(25), recursive calls | 55.0 ms | 24.4 ms | **2.3×** |
| word count, 260k words | 201 ms | 143 ms | 1.4× |
| 200k f-strings and join | 163 ms | 129 ms | 1.3× |
| harmonic sum 50M, `@native` ¹ | 1,704 ms | 1,650 ms | 1.0× |
| integer loop, 3M iterations | 111 ms | 113 ms | 1.0× |

Geometric mean: **7.3× faster**. Every row printed the same check value on both versions.

¹ This machine has no `llvmlite`, so both versions ran the `@native` functions without the native compiler. The
`bench` workflow (`.github/workflows/bench.yml`) measures them with the native compiler on a GitHub runner, together
with the heavy CPU benchmarks of both versions against C, C++, Rust and Go.
