# One Apsi version against another

```bash
python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . --web
```

Runs `programs.apsi` (code that runs unchanged on both versions), the start-up time, `apsi check` of every example
with an empty cache and, with `--web`, the HTTP server under load, for both versions on the same machine, and writes
`versions.json` and `versions.md`. Rows whose printed checks differ between the versions are left out.

## 2.4.4 against 2.0.0

Measured twice, each time with both versions on the same machine:

* **runner** — a GitHub Actions runner (AMD EPYC 7763, 4 threads, Python 3.13) with `llvmlite` installed, so
  `@native` functions run as machine code (`versions-2.4.4-vs-2.0.0-runner.json`, from `.github/workflows/bench.yml`);
* **2-core VM** — Intel Xeon @ 2.30 GHz, Python 3.13, without `llvmlite` (`versions-2.4.4-vs-2.0.0.json`).

The 2.0.0 and 2.4.4 columns are the runner's.

| Program | 2.0.0 | 2.4.4 | runner | 2-core VM ¹ |
|---|---:|---:|---:|---:|
| matmul 300, loops (compiled engine) | 37,909 ms | 5.0 ms | **7658.43×** | 6374.16× |
| 200k objects: create and sum | 3,793 ms | 419 ms | **9.05×** | 9.26× |
| fib(25), recursive calls | 82.3 ms | 24.4 ms | **3.37×** | 2.26× |
| matmul 512, @native loops | 26.7 ms | 9.8 ms | **2.73×** | 300.48× |
| web server, 64 connections | 5,184 req/s | 12,554 req/s | **2.42×** | 3.06× |
| web server, 256 connections | 6,037 req/s | 12,250 req/s | **2.03×** | 3.18× |
| word count, 260k words | 324 ms | 202 ms | **1.6×** | 1.41× |
| start-up: apsi run hello.apsi | 232 ms | 173 ms | **1.34×** | 6.69× |
| 200k f-strings and join | 234 ms | 189 ms | **1.23×** | 1.26× |
| integer loop, 3M iterations | 198 ms | 171 ms | **1.16×** | 0.99× |
| apsi check, 43 example programs (1,594 lines), empty cache | 9,967 ms | 9,613 ms | **1.04×** | 4.9× |
| harmonic sum 50M, @native | 70.1 ms | 70.1 ms | **1.0×** | 1.03× |

Geometric mean: **3.83× faster** on the runner, **7.31×** on the VM. Every row printed the same check
value on both versions.

¹ Without `llvmlite`, both versions ran the `@native` functions without the native compiler, and 2.0.0 started
much more slowly (894 ms against 133 ms), which also dominates its `apsi check` time there.

`@native` machine code was already at C speed in 2.0.0 and stays there: the heavy CPU benchmarks of both versions on
the same runner are in `benchmarks/heavy/versions/2.4.4-runner.json` and `2.0.0-runner.json`.
