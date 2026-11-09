"""NumPy (the same BLAS) and pure-Python loops for comparison: python3 benchmarks/matmul/mm.py"""
import time
import numpy as np


def best(f, reps=3):
    t_best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        f()
        t_best = min(t_best, time.perf_counter() - t)
    return t_best


for n in (256, 512, 1024, 2048):
    a, b = np.random.default_rng(1).standard_normal((2, n, n))
    t = best(lambda: a @ b)
    print(f"n={n}  NumPy a @ b {t * 1000:8.2f} ms {2 * n ** 3 / t / 1e9:6.1f} GFLOP/s")

n = 128
a = np.random.default_rng(1).standard_normal((n, n)).tolist()
b = np.random.default_rng(2).standard_normal((n, n)).tolist()


def py_loops():
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            ci, bk = c[i], b[k]
            for j in range(n):
                ci[j] += aik * bk[j]


t = best(py_loops)
print(f"n={n}  Python loops {t * 1000:8.2f} ms {2 * n ** 3 / t / 1e9:6.3f} GFLOP/s")
