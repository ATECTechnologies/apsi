Apsi 2.4.4 on a GitHub Actions runner (AMD EPYC 7763 64-Core Processor), one core, times in ms (best of runs); all checks agree: True.

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 127.8 | **44.2** | 44.3 | 181.9 | 318.9 |
| jacobi-2d 1024x200 | 175.7 | 168.7 | 189.8 | **168.2** | 703.2 |
| matmul 1024 | 196.5 | 201.9 | **166.6** | 183.7 | 700.2 |
| lu-pivot 800 | 40.7 | 40.8 | **40.6** | 41.4 | 116.5 |
| ising 512x50 | **81.7** | 90.6 | 90.6 | 84.6 | 146.5 |
| levenshtein 10k | **194.1** | 818.7 | 306 | 268.7 | 1736.2 |
| ode-rk4 20M | 355.1 | **288.8** | 289.1 | 355.1 | 355.5 |
| kmeans 200k k16 x20 | **51.5** | 61 | 60.8 | 74.5 | 213.3 |

Mojo is not installed on the runner. Earlier results (another machine, with Mojo) are in `versions/`.
