# One Apsi version against another

```bash
python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . --web
```

Runs `programs.apsi` (code that runs unchanged on both versions), the start-up time, `apsi check` of every example
with an empty cache and, with `--web`, the HTTP server under load, for both versions on the same machine, and writes
`versions.json` and `versions.md`. Rows whose printed checks differ between the versions are left out.

## 2.4.4 against 2.0.0

Measured twice, each time with both versions on the same machine:

* **runner** — a GitHub Actions runner (AMD EPYC 9V45, 4 threads, Python 3.13) with `llvmlite` installed, so
  `@native` functions run as machine code (`versions-2.4.4-vs-2.0.0-runner.json`, from `.github/workflows/bench.yml`);
* **2-core VM** — Intel Xeon @ 2.30 GHz, Python 3.13, without `llvmlite` (`versions-2.4.4-vs-2.0.0.json`).

The 2.0.0 and 2.4.4 columns are the runner's.

| Program | 2.0.0 | 2.4.4 | runner | 2-core VM ¹ |
|---|---:|---:|---:|---:|
| matmul 300, loops (compiled engine) | 20,950 ms | 3.2 ms | **6567.32×** | 6374.16× |
| 200k objects: create and sum | 2,130 ms | 262 ms | **8.12×** | 9.26× |
| fib(25), recursive calls | 41.0 ms | 12.2 ms | **3.37×** | 2.26× |
| web server, 64 connections | 11,970 req/s | 39,964 req/s | **3.34×** | 3.06× |
| web server, 256 connections | 13,144 req/s | 37,956 req/s | **2.89×** | 3.18× |
| matmul 512, @native loops | 14.6 ms | 6.5 ms | **2.26×** | 300.48× |
| word count, 260k words | 155 ms | 102 ms | **1.51×** | 1.41× |
| start-up: apsi run hello.apsi | 181 ms | 134 ms | **1.35×** | 6.69× |
| 200k f-strings and join | 116 ms | 93.8 ms | **1.24×** | 1.26× |
| integer loop, 3M iterations | 87.2 ms | 80.9 ms | **1.08×** | 0.99× |
| apsi check, 41 example programs (1,520 lines), empty cache | 6,759 ms | 6,499 ms | **1.04×** | 4.9× |
| harmonic sum 50M, @native | 44.7 ms | 46.7 ms | **0.96×** | 1.03× |

Geometric mean: **3.85× faster** on the runner, **7.31×** on the VM. Every row printed the same check
value on both versions.

¹ Without `llvmlite`, both versions ran the `@native` functions without the native compiler, and 2.0.0 started
much more slowly (894 ms against 133 ms), which also dominates its `apsi check` time there.

`@native` machine code was already at C speed in 2.0.0 and stays there: the heavy CPU benchmarks of both versions on
the same runner are in `benchmarks/heavy/versions/2.4.4-runner.json` and `2.0.0-runner.json`.
