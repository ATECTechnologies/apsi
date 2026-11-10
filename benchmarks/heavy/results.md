Apsi 2.4.4 on a GitHub Actions runner (AMD EPYC 7763 64-Core Processor), one core, times in ms (best of runs); all checks agree: True.

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 46.8 | **44.1** | 44.3 | 182.2 | 319.1 |
| jacobi-2d 1024x200 | 207 | 173.9 | **161.4** | 177.8 | 707.1 |
| matmul 1024 | 196.2 | 202.8 | **186.6** | 191.1 | 700.7 |
| lu-pivot 800 | 41.2 | **40.5** | 41.4 | 42.4 | 116.9 |
| ising 512x50 | **81.7** | 90.8 | 90.6 | 84.7 | 147.3 |
| levenshtein 10k | **195.3** | 811.2 | 306.4 | 268.9 | 1739.1 |
| ode-rk4 20M | 355.2 | **289** | 289.1 | 355.2 | 355.7 |
| kmeans 200k k16 x20 | **51.5** | 60.8 | 60.5 | 64.4 | 213.6 |

The same programs on another runner model (AMD EPYC 9V45 96-Core Processor, AVX-512), where every language is faster:

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 28.7 | **14.3** | 14.6 | 110.3 | 240.9 |
| jacobi-2d 1024x200 | 98.7 | **95.9** | 100.8 | 118.1 | 479 |
| matmul 1024 | **94.4** | 98.4 | 110.3 | 100.9 | 533.8 |
| lu-pivot 800 | 19.9 | 20.6 | 20.9 | **19.8** | 86.1 |
| ising 512x50 | **55.5** | 64.9 | 64.3 | 58.8 | 110.6 |
| levenshtein 10k | **176.1** | 493.1 | 183.8 | 300.1 | 1434.1 |
| ode-rk4 20M | 247.7 | **231.3** | 239.1 | 256 | 246.1 |
| kmeans 200k k16 x20 | **32.6** | 33.5 | 34.3 | 42.1 | 133.8 |

And on a third model (Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz, AVX-512):

| Benchmark | Apsi | C | C++ | Rust | Go |
|---|---:|---:|---:|---:|---:|
| nbody-dense 2000x10 | 58.7 | **57.7** | 57.8 | 134.6 | 288.9 |
| jacobi-2d 1024x200 | 280.7 | **278.9** | 298.9 | 301.6 | 888.3 |
| matmul 1024 | 299.3 | **298.9** | 299.1 | 303.7 | 654 |
| lu-pivot 800 | **53.3** | 56 | 56.6 | 53.6 | 135.5 |
| ising 512x50 | 100 | 100.9 | 100.1 | **89.4** | 176.3 |
| levenshtein 10k | 363 | 747.4 | **362.9** | 501.1 | 1193.7 |
| ode-rk4 20M | 438.2 | 313 | **312.7** | 437.9 | 444.1 |
| kmeans 200k k16 x20 | **68.1** | 69.2 | 69.6 | 75.7 | 188.1 |

GitHub gives runs different processor models, so only numbers from the same run can be compared.

ode-rk4 with `@native("fma")` (fused multiply-adds, as gcc does for C by default), first runner:

```
ode-rk4 20M: 355.3 ms  [check 1.3041989311869047]
ode-rk4 20M, @native("fma"): 263.3 ms  [check 1.3041989311869153]
```

Mojo is not installed on the runner. Earlier results (another machine, with Mojo) are in `versions/`.
