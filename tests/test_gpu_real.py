"""@gpu on a real NVIDIA GPU: results against NumPy, the work really running on the GPU, many calls in a row.

Skipped on machines without llvmlite or without a usable GPU (continuous integration has none). On a machine with a
GPU, run it on its own to see each check and the timings:

    python -m unittest tests.test_gpu_real -v
"""
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                       # noqa: E402
from apsi import native as N                        # noqa: E402

SRC = r'''
@gpu("strict")
fn saxpy(a: Float, x, y):
    parallel for i in 0..len(x):
        y[i] = a * x[i] + y[i]

@gpu("strict")
fn sumsq(x) -> Float:
    var s = 0.0
    parallel for i in 0..len(x):
        s += x[i] * x[i]
    return s

@gpu("strict")
fn row_sums(m, out):
    parallel for i in 0..m.shape[0]:
        var acc = 0.0
        for j in 0..m.shape[1]:
            acc += m[i, j]
        out[i] = acc

@gpu("strict")
fn matmul_rows(a, b, c):
    let k = a.shape[1]
    parallel for i in 0..a.shape[0]:
        for j in 0..b.shape[1]:
            var acc = 0.0
            for p in 0..k:
                acc += a[i, p] * b[p, j]
            c[i, j] = acc

@gpu("strict")
fn stencil(u, out):
    let n = len(u)
    parallel for i in 1..n - 1:
        out[i] = 0.25 * u[i - 1] + 0.5 * u[i] + 0.25 * u[i + 1]

@gpu("strict")
fn clamp_count(x, lo: Float, hi: Float) -> Float:
    var inside = 0.0
    parallel for i in 0..len(x):
        if x[i] >= lo and x[i] <= hi:
            inside += 1.0
    return inside
'''

HAVE = N.available()


def _gpu():
    if not HAVE:
        return None, "llvmlite is not installed"
    try:
        from apsi.native.gpu import info
        g = info()
    except Exception as e:                              # noqa: BLE001
        return None, f"no GPU: {e}"
    return (g, "") if g.get("available") else (None, g.get("reason", "no GPU"))


GPU, WHY = _gpu()


@unittest.skipUnless(GPU, f"needs an NVIDIA GPU ({WHY})")
class RealGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Interpreter().run(SRC, "<gpu-real>")
        print("\nGPUs:", GPU["devices"], file=sys.stderr)

    def g(self, name):
        return self.env.lookup(name)

    def gpu_calls(self):
        from apsi.native.gpu import STATS
        return STATS["gpu_calls"]

    def check_on_gpu(self, before):
        self.assertGreater(self.gpu_calls(), before, "the call did not run on the GPU")

    def test_saxpy_large(self):
        n = 10_000_000
        x = np.linspace(0, 1, n)
        y = np.ones(n)
        before = self.gpu_calls()
        self.g("saxpy")(2.0, x, y)
        self.check_on_gpu(before)
        self.assertTrue(np.allclose(y, 2.0 * x + 1.0, rtol=0, atol=1e-12))

    def test_reduction(self):
        x = np.random.default_rng(1).standard_normal(5_000_000)
        before = self.gpu_calls()
        got = self.g("sumsq")(x)
        self.check_on_gpu(before)
        self.assertAlmostEqual(got / float(np.sum(x * x)), 1.0, places=9)

    def test_rows_and_odd_sizes(self):
        for shape in [(1, 1), (7, 3), (1000, 1), (1, 1000), (4097, 33)]:
            m = np.random.default_rng(2).standard_normal(shape)
            out = np.zeros(shape[0])
            self.g("row_sums")(m, out)
            self.assertTrue(np.allclose(out, m.sum(axis=1), rtol=1e-12, atol=1e-12), shape)

    def test_matmul(self):
        rng = np.random.default_rng(3)
        for m, k, n in [(128, 64, 32), (513, 257, 129), (1024, 1024, 1024)]:
            a, b = rng.standard_normal((m, k)), rng.standard_normal((k, n))
            c = np.zeros((m, n))
            before = self.gpu_calls()
            t = time.perf_counter()
            self.g("matmul_rows")(a, b, c)
            dt = time.perf_counter() - t
            self.check_on_gpu(before)
            self.assertTrue(np.allclose(c, a @ b, rtol=1e-10, atol=1e-10), (m, k, n))
            print(f"  matmul {m}x{k}x{n} on the GPU: {dt * 1000:.1f} ms (first call includes compiling)",
                  file=sys.stderr)

    def test_stencil_and_branches(self):
        u = np.random.default_rng(4).standard_normal(1_000_001)
        out = np.zeros_like(u)
        self.g("stencil")(u, out)
        want = np.zeros_like(u)
        want[1:-1] = 0.25 * u[:-2] + 0.5 * u[1:-1] + 0.25 * u[2:]
        self.assertTrue(np.allclose(out, want, rtol=0, atol=1e-12))
        self.assertEqual(self.g("clamp_count")(u, -1.0, 1.0), float(np.count_nonzero((u >= -1) & (u <= 1))))

    def test_float32(self):
        x = np.linspace(0, 1, 1_000_000, dtype=np.float32)
        y = np.ones_like(x)
        self.g("saxpy")(2.0, x, y)
        self.assertTrue(np.allclose(y, 2.0 * x + 1.0, rtol=1e-6))

    def test_many_calls_are_stable(self):
        x = np.linspace(0, 1, 100_000)
        f = self.g("saxpy")
        for _ in range(500):
            y = np.zeros_like(x)
            f(1.0, x, y)
        self.assertTrue(np.array_equal(y, x))

    def test_timing_against_the_cpu(self):
        n = 2048
        rng = np.random.default_rng(5)
        a, b = rng.standard_normal((n, 256)), rng.standard_normal((256, n))
        c = np.zeros((n, n))
        f = self.g("matmul_rows")
        f(a, b, c)                                          # compile and warm up
        t = time.perf_counter()
        f(a, b, c)
        gpu_t = time.perf_counter() - t
        cpu = f.cpu
        cpu(a, b, c)
        t = time.perf_counter()
        cpu(a, b, c)
        cpu_t = time.perf_counter() - t
        print(f"  {n}x256x{n}: GPU {gpu_t * 1000:.1f} ms, the same function on the CPU {cpu_t * 1000:.1f} ms",
              file=sys.stderr)
        self.assertTrue(np.allclose(c, a @ b, rtol=1e-10, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
