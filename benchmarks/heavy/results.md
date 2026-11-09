Apsi 2.4.4 on a GitHub Actions runner (AMD EPYC 9V45 96-Core Processor), one core, times in ms (best of runs); all checks agree: True.

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 115.3 | 14.3 | **14.2** | 111.2 | 245.6 |
| jacobi-2d 1024x200 | 102.9 | 101.8 | **100** | 103.7 | 473.4 |
| matmul 1024 | 105.2 | 103.8 | 111.4 | **103.2** | 542.4 |
| lu-pivot 800 | 20.4 | **20** | **20** | **20** | 88.6 |
| ising 512x50 | 56 | 64.9 | 62.6 | **55.9** | 113.3 |
| levenshtein 10k | **178.6** | 508 | 183 | 285.1 | 1469.3 |
| ode-rk4 20M | 371.9 | **227.9** | 236.6 | 241.3 | 256.7 |
| kmeans 200k k16 x20 | 43.2 | **34.6** | 35 | 41.3 | 137.1 |

Mojo is not installed on the runner. Earlier results (another machine, with Mojo) are in `versions/`.
