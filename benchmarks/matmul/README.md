# Matrix multiplication

```bash
apsi run benchmarks/matmul/mm.apsi                      # Apsi: loops (two forms) and a @ b
APSI_IDIOMS=0 apsi run benchmarks/matmul/mm.apsi        # the same loops without the matrix-product recognition
gcc -O3 -march=native benchmarks/matmul/mm.c -o mm && ./mm
python3 benchmarks/matmul/mm.py                         # NumPy and pure-Python loops
```

Measured on an AMD EPYC 7763 GitHub Actions runner (4 vCPU), float64, best of 3:

| n | C loops, -O3 -march=native | NumPy `a @ b` | Apsi loops (ikj) | Apsi loops (dot form) | Apsi `a @ b` |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.28 ms (14.7 GFLOP/s) | 0.61 ms | **0.55 ms** (61.0 GFLOP/s) | 0.45 ms | 0.46 ms |
| 512 | 18.85 ms (14.2 GFLOP/s) | 3.20 ms | **3.40 ms** (79.0 GFLOP/s) | 3.20 ms | 3.17 ms |
| 1024 | 155.2 ms (13.8 GFLOP/s) | 23.98 ms | **24.27 ms** (88.5 GFLOP/s) | 23.77 ms | 23.49 ms |
| 2048 | 2,111 ms (8.1 GFLOP/s) | 190.7 ms | **192.4 ms** (89.3 GFLOP/s) | 189.2 ms | 190.2 ms |

Pure Python loops: 114.0 ms for n = 128 (0.037 GFLOP/s), so about 2,400 times slower than the same loops in Apsi.
With `APSI_IDIOMS=0` the Apsi loops of n = 300 take 25.4 s instead of 2.8 ms.

Why the loops are fast: the compiler recognises loop nests that compute a matrix product and calls the same tuned
BLAS kernel NumPy uses (cache blocking, SIMD, all cores). The loops remain in the program and run, unchanged,
whenever that would not give the loops' result (see [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md#matrix-products-written-as-loops)).
The C column is the same loop nest compiled by GCC; a C program that wants this speed links a BLAS library itself.
