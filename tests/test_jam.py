"""Unroll-and-jam of independent outer loops (apsi/native/jam.py): only safe nests, and bit-identical results."""
import contextlib
import io
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                       # noqa: E402
from apsi import native as N                        # noqa: E402
from apsi.native import jam as J                    # noqa: E402
from apsi.parser import parse                       # noqa: E402

PROGRAMS = r'''
fn forces(x, y, m, ax, ay):
    let n = len(x)
    for i in 0..n:
        var fx = 0.0
        var fy = 0.0
        let xi = x[i]
        let yi = y[i]
        for j in 0..n:
            let dx = x[j] - xi
            let dy = y[j] - yi
            let d2 = dx * dx + dy * dy + 0.01
            let inv = m[j] / (d2 * sqrt(d2))
            fx += dx * inv
            fy += dy * inv
        ax[i] = fx
        ay[i] = fy

fn assign(px, cx, sums, cnt) -> Float:
    var total = 0.0
    for i in 0..len(px):
        var best = 0
        var bd = 1e300
        for c in 0..len(cx):
            let d = (px[i] - cx[c]) * (px[i] - cx[c])
            if d < bd:
                bd = d
                best = c
        sums[best] += px[i]
        cnt[best] += 1.0
        total += bd
    return total

fn rowdot(a, v, out):
    for i in 0..a.shape[0]:
        var s = 0.0
        for k in 0..a.shape[1]:
            s += a[i, k] * v[k]
        out[i] = s
'''


def nest(text):
    return J.jam(parse(text).body[0].body[0])


class Matcher(unittest.TestCase):
    def test_applies(self):
        fns = {f.name: f for f in parse(PROGRAMS).body}
        for name in ("forces", "assign", "rowdot"):
            body = fns[name].body
            loop = [s for s in body if type(s).__name__ == "For"][0]
            self.assertIsNotNone(J.jam(loop), name)

    def test_refuses_unsafe_nests(self):
        bad = [
            # a running total read inside the loop
            "fn f(x, o):\n    for i in 0..4:\n        var s = 0.0\n        for j in 0..4:\n"
            "            s += x[j] + o[0]\n        o[0] += s\n",
            # the inner loop writes another lane's row
            "fn f(x, o):\n    for i in 0..4:\n        var s = 0.0\n        for j in 0..4:\n"
            "            s += x[j]\n            o[i + 1] = s\n        o[i] = s\n",
            # inner bounds depend on i
            "fn f(x, o):\n    for i in 0..4:\n        var s = 0.0\n        for j in 0..i:\n"
            "            s += x[j]\n        o[i] = s\n",
            # calls a user function
            "fn f(x, o):\n    for i in 0..4:\n        var s = 0.0\n        for j in 0..4:\n"
            "            s += g(x[j])\n        o[i] = s\n",
            # reads a row written by another lane
            "fn f(x, o):\n    for i in 1..4:\n        var s = 0.0\n        for j in 0..4:\n"
            "            s += x[j]\n        o[i] = s + o[i - 1]\n",
            # writes an outer scalar inside the inner loop
            "fn f(x, o):\n    var t = 0.0\n    for i in 0..4:\n        var s = 0.0\n        for j in 0..4:\n"
            "            s += x[j]\n            t += 1.0\n        o[i] = s\n",
        ]
        for text in bad:
            fd = parse(text).body[0]
            loop = [s for s in fd.body if type(s).__name__ == "For"][0]
            self.assertIsNone(J.jam(loop), text)


def run_jammed(src, call):
    """Run src with every jammable loop rewritten, in the bytecode engine (no native compiler needed)."""
    import apsi.codegen as CG
    orig = CG.CodeGen.s_For

    def s_For(self, s, scope):
        rep = J.jam(s)
        if rep is not None:
            for st in rep:
                self.stmt(st, scope)
            return
        return orig(self, s, scope)
    CG.CodeGen.s_For = s_For
    try:
        env = Interpreter().run(src, "<jam>")
        return call(env)
    finally:
        CG.CodeGen.s_For = orig


class SameResults(unittest.TestCase):
    def call(self, env, native=False):
        rng = np.random.default_rng(1)
        res = []
        for n in (1, 3, 4, 5, 8, 37):
            x, y = rng.standard_normal(n), rng.standard_normal(n)
            m = rng.random(n)
            ax, ay = np.zeros(n), np.zeros(n)
            f = env.lookup("forces")
            (N.native("strict")(f) if native else f)(x, y, m, ax, ay)
            res += [ax.copy(), ay.copy()]
            cx = rng.standard_normal(5)
            sums, cnt = np.zeros(5), np.zeros(5)
            g = env.lookup("assign")
            t = (N.native("strict")(g) if native else g)(rng.standard_normal(n * 7), cx, sums, cnt)
            res += [np.array([t]), sums.copy(), cnt.copy()]
            a, v, o = rng.standard_normal((n, 6)), rng.standard_normal(6), np.zeros(n)
            h = env.lookup("rowdot")
            (N.native("strict")(h) if native else h)(a, v, o)
            res.append(o.copy())
        return res

    def plain(self):
        J.ENABLED = False
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return self.call(Interpreter().run(PROGRAMS, "<plain>"))
        finally:
            J.ENABLED = True

    def test_rewrite_is_bit_identical(self):
        plain = self.plain()
        with contextlib.redirect_stderr(io.StringIO()):
            jammed = run_jammed(PROGRAMS, self.call)
        for a, b in zip(plain, jammed):
            self.assertTrue(np.array_equal(a.view(np.uint64), b.view(np.uint64)))

    @unittest.skipUnless(N.available(), "llvmlite is not installed")
    def test_native_is_bit_identical(self):
        plain = self.plain()
        env = Interpreter().run(PROGRAMS, "<native>")
        native = self.call(env, native=True)
        for a, b in zip(plain, native):
            self.assertTrue(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


VEC_PROGRAMS = r'''
fn energy(x, m, inner: Int) -> Float:
    var total = 0.0
    for i in 0..len(x):
        var f = 0.0
        var g = 1.0
        let xi = x[i]
        for j in 0..inner:
            let d = x[j] - xi
            let r = sqrt(d * d + 0.5)
            f += -m[j] * d / (r * r * r)
            g *= abs(d) + 1.0
        total += f + g
    return total

fn matvec(a, v) -> Float:
    var total = 0.0
    for i in 0..a.shape[0]:
        var acc = 0.0
        for k in 0..a.shape[1]:
            acc += a[i, k] * v[k]
        total += acc * (i + 1)
    return total

fn by_python_float(x, s: Float) -> Float:
    var total = 0.0
    for i in 0..len(x):
        var f = 0.0
        for j in 0..len(x):
            f += (x[j] - x[i]) / s
        total += f
    return total
'''


@unittest.skipUnless(N.available(), "llvmlite is not installed")
class VectorLanes(unittest.TestCase):
    """The jammed inner loop as SIMD vectors (Spec.jam_vector_for): same bits, same Python/NumPy scalar types."""

    def run_all(self, env):
        rng = np.random.default_rng(3)
        out = []
        for n in (0, 1, 3, 4, 5, 9, 33):
            x, m = rng.standard_normal(n), rng.random(n)
            for inner in (0, 1, n):
                out.append(env["energy"](x, m, min(inner, n)))
            out.append(env["by_python_float"](x, 0.75))
            out.append(env["matvec"](rng.standard_normal((n, 7)), rng.standard_normal(7)))
        return out

    def test_vector_lanes_match_the_interpreter_and_scalar_lanes(self):
        plain_env = Interpreter().run(VEC_PROGRAMS, "<plain>")
        J.ENABLED = False
        try:
            plain = self.run_all({k: plain_env.lookup(k) for k in ("energy", "by_python_float", "matvec")})
        finally:
            J.ENABLED = True
        results = {}
        for mode in ("1", "0"):
            os.environ["APSI_JAM_VEC"] = mode
            try:
                env = Interpreter().run(VEC_PROGRAMS, "<native>")
                fns = {k: N.native("strict")(env.lookup(k)) for k in ("energy", "by_python_float", "matvec")}
                results[mode] = self.run_all(fns)
                if mode == "1":
                    ir_text = fns["energy"].llvm_ir(np.zeros(4), np.zeros(4), 4)
                    self.assertIn("<4 x double>", ir_text)
                    self.assertIn("<4 x double>", fns["matvec"].llvm_ir(np.zeros((4, 4)), np.zeros(4)))
            finally:
                os.environ.pop("APSI_JAM_VEC", None)
        for got in results.values():
            for a, b in zip(plain, got):
                self.assertEqual(type(a), type(b))
                self.assertEqual(np.float64(a).view(np.uint64), np.float64(b).view(np.uint64))


if __name__ == "__main__":
    unittest.main()
