# Matrix multiplication

```bash
apsi run benchmarks/matmul/mm.apsi                      # Apsi: loops (two forms) and a @ b
APSI_IDIOMS=0 apsi run benchmarks/matmul/mm.apsi        # the same loops without the matrix-product recognition
gcc -O3 -march=native benchmarks/matmul/mm.c -o mm && ./mm
python3 benchmarks/matmul/mm.py                         # NumPy and pure-Python loops
```

Measured on a 2-core cloud VM (Intel Xeon @ 2.30 GHz), float64, best of 3:

| n | C loops, -O3 -march=native | NumPy `a @ b` | Apsi loops (ikj) | Apsi loops (dot form) | Apsi `a @ b` |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.90 ms (11.6 GFLOP/s) | 0.58 ms | **0.63 ms** (53.6 GFLOP/s) | 0.75 ms | 0.54 ms |
| 512 | 28.96 ms (9.3 GFLOP/s) | 4.03 ms | **4.30 ms** (62.4 GFLOP/s) | 4.45 ms | 4.09 ms |
| 1024 | 251.1 ms (8.6 GFLOP/s) | 29.9 ms | **33.0 ms** (65.0 GFLOP/s) | 31.2 ms | 32.9 ms |
| 2048 | 2,409 ms (7.1 GFLOP/s) | 229.9 ms | **234.2 ms** (73.3 GFLOP/s) | 233.7 ms | 230.8 ms |

Pure Python loops: 75 ms for n = 128 (0.056 GFLOP/s), so about 1,000 times slower than the same loops in Apsi.
With `APSI_IDIOMS=0` the Apsi loops of n = 300 take 25.4 s instead of 2.8 ms.

Why the loops are fast: the compiler recognises loop nests that compute a matrix product and calls the same tuned
BLAS kernel NumPy uses (cache blocking, SIMD, all cores). The loops remain in the program and run, unchanged,
whenever that would not give the loops' result (see [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md#matrix-products-written-as-loops)).
The C column is the same loop nest compiled by GCC; a C program that wants this speed links a BLAS library itself.
