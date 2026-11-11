"""Loop nests recognised as matrix products run with BLAS in both engines, and fall back to the loops exactly."""
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                        # noqa: E402
from apsi import idioms                             # noqa: E402
from apsi.errors import ApsiError                   # noqa: E402
from apsi.parser import parse                       # noqa: E402
from apsi import native as N                        # noqa: E402

SRC = r'''
fn ikj(a, b, c):
    let n = a.shape[0]
    for i in 0..n:
        for k in 0..a.shape[1]:
            let aik = a[i, k]
            for j in 0..b.shape[1]:
                c[i, j] += aik * b[k, j]

fn ijk(a, b, c, m: Int, n: Int, k: Int):
    for i in 0..m:
        for j in range(n):
            for p in 0..k:
                c[i, j] += b[p, j] * a[i, p]

fn dot_rows(a, b, c):
    parallel for i in 0..a.shape[0]:
        for j in 0..b.shape[1]:
            var s = 0.0
            for k in 0..b.shape[0]:
                s += a[i, k] * b[k, j]
            c[i, j] = s
'''


def nest(text):
    return idioms.match_gemm(parse(text).body[0])


class Matcher(unittest.TestCase):
    def test_shapes_that_match(self):
        g = nest("for i in 0..n:\n for k in 0..m:\n  let t = a[i, k]\n  for j in 0..p:\n   c[i, j] += t * b[k, j]")
        self.assertEqual((g.a, g.b, g.c, g.mode), ("a", "b", "c", 0))
        self.assertEqual((g.M.id, g.N.id, g.K.id), ("n", "p", "m"))
        g = nest("for j in 0..n:\n for k in 0..m:\n  for i in 0..p:\n   c[i, j] += b[k, j] * a[i, k]")
        self.assertEqual((g.a, g.b, g.M.id, g.N.id, g.K.id), ("a", "b", "p", "n", "m"))
        g = nest("for i in 0..n:\n for j in 0..m:\n  var s = 0.0\n  for k in 0..p:\n   s += a[i, k] * b[k, j]\n"
                 "  c[i, j] = s")
        self.assertEqual(g.mode, 1)
        self.assertIsNotNone(nest("for i in 0..n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] += a[i, k] * a[k, j]"))

    def test_shapes_that_do_not(self):
        bad = [
            "for i in 0..n:\n for j in 0..n:\n  for k in 0..n:\n   a[i, j] += a[i, k] * b[k, j]",     # writes an input
            "for i in 0..n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] += a[k, i] * b[k, j]",     # a transposed
            "for i in 0..n:\n for j in 0..i:\n  for k in 0..n:\n   c[i, j] += a[i, k] * b[k, j]",     # triangular
            "for i in 1..n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] += a[i, k] * b[k, j]",     # offset
            "for i in 0..=n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] += a[i, k] * b[k, j]",    # inclusive
            "for i in 0..n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] -= a[i, k] * b[k, j]",     # minus
            "for i in 0..n:\n for j in 0..n:\n  for k in 0..f(n):\n   c[i, j] += a[i, k] * b[k, j]",  # a call
            "for i in 0..n:\n for j in 0..n:\n  var s = 1.0\n  for k in 0..n:\n   s += a[i, k] * b[k, j]\n"
            "  c[i, j] = s",                                                                     # s starts at 1
            "for i in 0..n:\n for j in 0..n:\n  for k in 0..n:\n   c[i, j] += a[i, k] * b[k, j]\n  print(i)",
        ]
        for text in bad:
            self.assertIsNone(nest(text), text)


class Engines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Interpreter().run(SRC, "<idioms>")

    def fns(self, name):
        f = self.env.lookup(name)
        out = [("compiled", f)]
        if N.available():
            out.append(("native", N.native("strict")(f)))
        return out

    def loops(self, name, *args):
        idioms.ENABLED = False
        try:
            env = Interpreter().run(SRC, "<loops>")
            env.lookup(name)(*args)
        finally:
            idioms.ENABLED = True

    def test_results_match_the_loops(self):
        rng = np.random.default_rng(3)
        for dt, tol in ((np.float64, 1e-12), (np.float32, 1e-4)):
            a = rng.standard_normal((37, 23)).astype(dt)
            b = rng.standard_normal((23, 41)).astype(dt)
            want = np.zeros((37, 41), dt) + 1
            self.loops("ikj", a, b, want)
            for how, f in self.fns("ikj"):
                c = np.zeros((37, 41), dt) + 1
                f(a, b, c)
                self.assertTrue(np.allclose(c, want, rtol=tol, atol=tol), (how, dt))
            for how, f in self.fns("ijk"):
                c = np.zeros((40, 45), dt)
                f(a, b, c, 30, 20, 10)                   # a sub-block: c[:30, :20] += a[:30, :10] @ b[:10, :20]
                ref = np.zeros((40, 45), dt)
                ref[:30, :20] = a[:30, :10].astype(np.float64) @ b[:10, :20].astype(np.float64)
                self.assertTrue(np.allclose(c, ref, rtol=tol, atol=tol), (how, dt))
            for how, f in self.fns("dot_rows"):
                c = np.full((37, 41), 7, dt)
                f(a, b, c)
                self.assertTrue(np.allclose(c, a.astype(np.float64) @ b.astype(np.float64), rtol=tol, atol=tol),
                                (how, dt))

    def test_fallbacks_behave_like_the_loops(self):
        rng = np.random.default_rng(4)
        a = rng.standard_normal((6, 6))
        for how, f in self.fns("ijk"):
            x = a.copy()
            want = a.copy()
            self.loops("ijk", want, want, want, 6, 6, 6)     # c is also an input: element by element
            f(x, x, x, 6, 6, 6)
            self.assertTrue(np.array_equal(x, want), how)
            with self.assertRaises((ApsiError, IndexError), msg=how):
                f(a, a, np.zeros((3, 3)), 6, 6, 6)          # out of bounds: the loops raise
            c = np.zeros((6, 6))
            f(a, a, c, 0, 6, 6)                              # empty: nothing happens
            self.assertFalse(c.any())
            ints = np.arange(36).reshape(6, 6)
            ci = np.zeros((6, 6), dtype=ints.dtype)
            f(ints, ints, ci, 6, 6, 6)                       # integers: the loops, exact
            self.assertTrue(np.array_equal(ci, ints @ ints), how)

    def test_much_faster_than_loops(self):
        n = 160
        a, b = np.random.default_rng(5).standard_normal((2, n, n))
        f = self.env.lookup("ikj")
        t = time.perf_counter()
        f(a, b, np.zeros((n, n)))
        fast = time.perf_counter() - t
        idioms.ENABLED = False
        try:
            g = Interpreter().run(SRC, "<loops>").lookup("ikj")
            t = time.perf_counter()
            g(a, b, np.zeros((n, n)))
            slow = time.perf_counter() - t
        finally:
            idioms.ENABLED = True
        self.assertLess(fast * 20, slow)


if __name__ == "__main__":
    unittest.main()
