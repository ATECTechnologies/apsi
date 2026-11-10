# One Apsi version against another

```bash
python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . --web
```

Runs `programs.apsi` (code that runs unchanged on both versions), the start-up time, `apsi check` of every example
with an empty cache and, with `--web`, the HTTP server under load, for both versions on the same machine, and writes
`versions.json` and `versions.md`. Rows whose printed checks differ between the versions are left out.

## 2.4.4 against 2.0.0

Measured twice, each time with both versions on the same machine (on the runner: best of two runs):

* **runner** — a GitHub Actions runner (AMD EPYC 7763, 4 threads, Python 3.13) with `llvmlite` installed, so
  `@native` functions run as machine code (`versions-2.4.4-vs-2.0.0-runner.json`, from `.github/workflows/bench.yml`);
* **2-core VM** — Intel Xeon @ 2.30 GHz, Python 3.13, without `llvmlite` (`versions-2.4.4-vs-2.0.0.json`).

The 2.0.0 and 2.4.4 columns are the runner's.

| Program | 2.0.0 | 2.4.4 | runner | 2-core VM ¹ |
|---|---:|---:|---:|---:|
| matmul 300, loops (compiled engine) | 38,157 ms | 5.0 ms | **7708.43×** | 6374.16× |
| 200k objects: create and sum | 3,758 ms | 434 ms | **8.65×** | 9.26× |
| fib(25), recursive calls | 82.8 ms | 24.5 ms | **3.37×** | 2.26× |
| matmul 512, @native loops | 24.7 ms | 9.6 ms | **2.56×** | 300.48× |
| web server, 64 connections | 5,124 req/s | 13,949 req/s | **2.72×** | 3.06× |
| web server, 256 connections | 5,848 req/s | 14,052 req/s | **2.4×** | 3.18× |
| word count, 260k words | 320 ms | 205 ms | **1.56×** | 1.41× |
| start-up: apsi run hello.apsi | 231 ms | 173 ms | **1.34×** | 6.69× |
| 200k f-strings and join | 236 ms | 192 ms | **1.23×** | 1.26× |
| integer loop, 3M iterations | 171 ms | 168 ms | **1.02×** | 0.99× |
| apsi check, 43 example programs (1,594 lines), empty cache | 9,865 ms | 9,526 ms | **1.04×** | 4.9× |
| harmonic sum 50M, @native | 70.1 ms | 70.1 ms | **1.0×** | 1.03× |

Geometric mean: **3.84× faster** on the runner, **7.31×** on the VM. Every row printed the same check
value on both versions.

¹ Without `llvmlite`, both versions ran the `@native` functions without the native compiler, and 2.0.0 started
much more slowly (894 ms against 133 ms), which also dominates its `apsi check` time there.

`@native` machine code was already at C speed in 2.0.0 and stays there: the heavy CPU benchmarks of both versions on
the same runner are in `benchmarks/heavy/versions/2.4.4-runner.json` and `2.0.0-runner.json`.
