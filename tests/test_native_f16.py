"""Small floats in @native and @gpu code: float16, bfloat16 and FP8 (e4m3fn, e5m2), stored small, computed in 32 bits.

Conversions must agree bit for bit with NumPy (float16) and ml_dtypes (bfloat16): every one of the 65,536 16-bit
patterns is widened, and special values, subnormals, overflow and ties are narrowed.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                       # noqa: E402
from apsi import native as N                        # noqa: E402

try:
    import ml_dtypes
    BF16 = np.dtype(ml_dtypes.bfloat16)
except ImportError:                                 # pragma: no cover - installed with Apsi's requirements
    BF16 = None

SRC = r'''
fn copy(x, y):
    for i in 0..len(x):
        y[i] = x[i]

fn scale(x, y, a: Float):
    for i in 0..len(x):
        y[i] = x[i] * a

fn total(x) -> Float:
    var s = 0.0
    for i in 0..len(x):
        s += x[i]
    return s

fn axpy2d(a: Float, x, y):
    for i in 0..x.shape[0]:
        for j in 0..x.shape[1]:
            y[i, j] = a * x[i, j] + y[i, j]

fn make_half(n: Int):
    let z = zeros(n, dtype=f16)
    for i in 0..n:
        z[i] = i * 0.5
    return z

fn make_bf16(n: Int):
    let z = zeros(n, dtype=bf16)
    for i in 0..n:
        z[i] = i * 0.25
    return z

@gpu
fn gpu_scale(x, y, a: Float):
    parallel for i in 0..len(x):
        y[i] = x[i] * a

@gpu
fn gpu_mm(a, b, c):
    let k = a.shape[1]
    parallel for i in 0..a.shape[0]:
        for j in 0..b.shape[1]:
            var acc = 0.0
            for p in 0..k:
                acc += a[i, p] * b[p, j]
            c[i, j] = acc
'''

SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, 0.1, 1 / 3, 65504.0, 65519.99, 65520.0, 1e6, -1e6, 6.1e-5, 6.0e-5,
                    5.96e-8, 2.98e-8, 2.99e-8, 1e-10, np.inf, -np.inf, np.nan, 3.0e38, 1.17e-38, 1e-45],
                   dtype=np.float32)


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and (a.view(np.uint16) == b.view(np.uint16)).all()


@unittest.skipUnless(N.available(), "llvmlite is not installed")
class Native16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APSI_NATIVE_QUIET"] = "1"
        cls.env = Interpreter().run(SRC, "<f16>")

    def nat(self, name):
        return N.native("strict")(self.env.lookup(name))

    def test_half_widening_every_pattern(self):
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        x = bits.view(np.float16)
        y = np.zeros(65536, np.float32)
        self.nat("copy")(x, y)
        want = x.astype(np.float32)
        ok = (y == want) | (np.isnan(y) & np.isnan(want))
        self.assertTrue(ok.all())
        self.assertTrue((np.signbit(y) == np.signbit(want)).all())

    def test_half_narrowing_rounds_like_numpy(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([SPECIAL] + [rng.standard_normal(50000).astype(np.float32) * s
                                        for s in (1e-7, 1e-4, 1, 300, 6e4, 1e5)])
        y = np.zeros(len(x), np.float16)
        self.nat("copy")(x, y)
        want = x.astype(np.float16)
        nan = np.isnan(want)
        self.assertTrue(same_bits(y[~nan], want[~nan]))
        self.assertTrue(np.isnan(y[nan]).all())

    def test_math_in_32_bits(self):
        x = np.linspace(-10, 10, 1001).astype(np.float16)
        y = np.zeros_like(x)
        self.nat("scale")(x, y, 2.0)
        self.assertTrue(same_bits(y, (x.astype(np.float32) * 2).astype(np.float16)))
        self.nat("scale")(x, y, 0.1)
        want = (x.astype(np.float64) * 0.1).astype(np.float16)
        self.assertTrue(np.allclose(y.astype(np.float32), want.astype(np.float32), rtol=1e-3, atol=1e-7))
        big = np.full(100000, 0.1, np.float16)                  # 16-bit accumulation would stall at 256
        self.assertAlmostEqual(self.nat("total")(big), 100000 * float(np.float16(0.1)), delta=1e-6 * 10000)

    def test_2d_and_creation(self):
        x = np.random.default_rng(2).standard_normal((33, 17)).astype(np.float16)
        y = np.ones_like(x)
        self.nat("axpy2d")(2.0, x, y)
        want = (np.float32(2.0) * x.astype(np.float32) + np.float32(1.0)).astype(np.float16)
        self.assertTrue(same_bits(y, want))
        z = self.nat("make_half")(10)
        self.assertEqual(z.dtype, np.float16)
        self.assertTrue((z == np.arange(10) * 0.5).all())

    @unittest.skipUnless(BF16 is not None, "ml_dtypes is not installed")
    def test_bfloat16(self):
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        x = bits.view(BF16)
        y = np.zeros(65536, np.float32)
        self.nat("copy")(x, y)
        want = x.astype(np.float32)
        self.assertTrue(((y == want) | (np.isnan(y) & np.isnan(want))).all())
        rng = np.random.default_rng(3)
        f = np.concatenate([SPECIAL] + [rng.standard_normal(50000).astype(np.float32) * s for s in (1e-30, 1, 1e30)])
        b = np.zeros(len(f), BF16)
        self.nat("copy")(f, b)
        want = f.astype(BF16)
        nan = np.isnan(f)
        self.assertTrue(same_bits(b[~nan], want[~nan]))
        self.assertTrue(np.isnan(b[nan].astype(np.float32)).all())
        z = self.nat("make_bf16")(8)
        self.assertEqual(z.dtype, BF16)
        self.assertTrue((z.astype(np.float32) == np.arange(8) * 0.25).all())

    @unittest.skipUnless(BF16 is not None, "ml_dtypes is not installed")
    def test_fp8(self):
        rng = np.random.default_rng(5)
        for name in ("float8_e4m3fn", "float8_e5m2"):
            dt = np.dtype(getattr(ml_dtypes, name))
            codes = np.arange(256, dtype=np.uint16).astype(np.uint8)
            x = codes.view(dt)
            y = np.zeros(256, np.float32)
            self.nat("copy")(x, y)
            want = x.astype(np.float32)
            self.assertTrue(((y == want) | (np.isnan(y) & np.isnan(want))).all(), name)
            f = np.concatenate([SPECIAL, want[~np.isnan(want)]] +
                               [rng.standard_normal(40000).astype(np.float32) * s for s in (1e-6, 1e-2, 1, 100, 1e4, 1e6)])
            mids = (want[~np.isnan(want)][:-1] + want[~np.isnan(want)][1:]) / 2
            f = np.concatenate([f, mids.astype(np.float32), -f])
            q = np.zeros(len(f), dt)
            self.nat("copy")(f, q)
            ref = f.astype(dt)
            nan = np.isnan(ref.astype(np.float32))
            self.assertTrue((q.view(np.uint8)[~nan] == ref.view(np.uint8)[~nan]).all(), name)
            self.assertTrue(np.isnan(q[nan].astype(np.float32)).all(), name)

    def test_gpu_kernels_are_generated(self):
        x = np.zeros(8, np.float16)
        ptx = self.env.lookup("gpu_scale").ptx(x, x, 2.0)
        self.assertIn(".entry apsi_kernel", ptx)
        a = np.zeros((4, 3), np.float16)
        b = np.zeros((3, 5), np.float16)
        c = np.zeros((4, 5), np.float16)
        self.assertIn(".entry apsi_kernel", self.env.lookup("gpu_mm").ptx(a, b, c))
        # the CPU fallback (no GPU on this machine) gives the same answers
        rng = np.random.default_rng(4)
        a = rng.standard_normal((20, 7)).astype(np.float16)
        b = rng.standard_normal((7, 9)).astype(np.float16)
        c = np.zeros((20, 9), np.float16)
        self.env.lookup("gpu_mm")(a, b, c)
        want = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
        self.assertTrue(np.allclose(c.astype(np.float32), want.astype(np.float32), rtol=2e-3, atol=2e-3))


class Collapse(unittest.TestCase):
    """GPU kernels: a parallel for over rows whose body is one independent inner loop gets one thread per element."""

    def shape(self, text):
        from apsi.parser import parse
        from apsi.native.gpu import collapse
        return collapse(parse(text).body[0])[1] is not None

    def test_collapses_independent_elements(self):
        self.assertTrue(self.shape("fn f(a, b, c):\n    parallel for i in 0..a.shape[0]:\n        for j in 0..4:\n"
                                   "            var acc = 0.0\n            for p in 0..3:\n"
                                   "                acc += a[i, p] * b[p, j]\n            c[i, j] = acc\n"))
        self.assertTrue(self.shape("fn f(m) -> Float:\n    var s = 0.0\n    parallel for i in 0..3:\n"
                                   "        for j in 0..3:\n            s += m[i, j]\n    return s\n"))

    def test_keeps_loops_that_depend_on_each_other(self):
        for text in [
            "fn f(m, out):\n    parallel for i in 0..3:\n        for j in 0..3:\n            out[i] += m[i, j]\n",
            "fn f(u, o):\n    parallel for i in 0..3:\n        for j in 1..3:\n            o[i, j] = o[i, j - 1] + u[i, j]\n",
            "fn f(o):\n    parallel for i in 0..4:\n        for j in 0..i:\n            o[i, j] = 1.0\n",
            "fn f(o):\n    parallel for i in 0..4:\n        for j in 0..4:\n            if j > 2:\n                break\n"
            "            o[i, j] = 1.0\n",
        ]:
            self.assertFalse(self.shape(text), text)


if __name__ == "__main__":
    unittest.main()
