"""The native tier must give exactly the results of the bytecode engine (differential tests).

Every function below is run twice: as compiled by the bytecode engine, and through `@native("strict")`
(machine code; `strict` makes any fallback a test failure). Results must be identical - same type, same bits for
floats, same tensors - and errors must be raised in both or neither.
"""
import io
import os
import sys
import contextlib
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                       # noqa: E402
from apsi.errors import ApsiError, UnitError       # noqa: E402
from apsi.units import Quantity                    # noqa: E402
from apsi import native as N                        # noqa: E402

HAVE = N.available()

SRC = r'''
use physics.constants

let SCALE = 2.5
let TABLE = linspace(0, 1, 11)
var counter_arr = zeros(4)

fn add(a, b) = a + b
fn ifloor(a: Int, b: Int) = a // b
fn imod(a: Int, b: Int) = a % b
fn ffloor(a: Float, b: Float) = a // b
fn fmod_(a: Float, b: Float) = a % b
fn power(a, b) = a ** b
fn tdiv(a, b) = a / b
fn cmp(a, b) = (a < b, a <= b, a == b, a != b, a > b, a >= b)
fn logic(a: Bool, b: Bool) = (a and b, a or b, not a)
fn minmax(a, b, c) = (min(a, b, c), max(a, b, c))
fn rnd(x: Float) = round(x)
fn conv(x: Float) = (int(x), float(int(x)))
fn absval(x) = abs(x)
fn ifx(x) = 1.5 if x > 0 else -2.5
fn neg_index(a, i: Int) = a[i]
fn scaled(x) = x * SCALE
fn table_at(i: Int) = TABLE[i]
fn approx_eq(a: Float, b: Float) = a ~= b

fn collatz(n: Int) -> Int:
    var steps = 0
    var x = n
    while x != 1:
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        steps += 1
    return steps

fn primes_below(n: Int) -> Int:
    var count = 0
    for k in 2..n:
        var p = true
        var d = 2
        while d * d <= k:
            if k % d == 0:
                p = false
                break
            d += 1
        if not p:
            continue
        count += 1
    return count

fn ack(m: Int, n: Int) -> Int:
    if m == 0:
        return n + 1
    if n == 0:
        return ack(m - 1, 1)
    return ack(m - 1, ack(m, n - 1))

fn sum_sq(a):
    var s = 0.0
    for i in 0..len(a):
        s += a[i] * a[i]
    return s

fn sum_range_step(n: Int) -> Int:
    var s = 0
    for i in range(n, 0, -3):
        s += i
    return s

fn matvec(m, v):
    let out = zeros(len(m))
    for i in 0..len(m):
        var acc = 0.0
        for j in 0..len(v):
            acc += m[i, j] * v[j]
        out[i] = acc
    return out

fn rows_sum(m):
    let out = zeros(m.shape[0])
    for i in 0..m.shape[0]:
        out[i] = sum(m[i])
    return out

fn stencil(u):
    let n = len(u)
    let out = zeros(n)
    for i in 1..n - 1:
        out[i] = u[i - 1] - 2.0 * u[i] + u[i + 1]
    return out

fn periodic(u):
    let n = len(u)
    let out = zeros(n)
    for i in 0..n:
        out[i] = u[i - 1] + u[(i + 1) % n]
    return out

fn fused(a, b) = sqrt(a * a + b * b) + 0.5 * a
fn fused_sum(a, b) = sum(a * b + 1.0)
fn bcast(m, v) = m * v + 1.0
fn red(a) = (sum(a), mean(a), min(a), max(a), argmin(a), argmax(a), prod(a))
fn dotn(a, b) = (dot(a, b), norm(a))
fn wclip(a) = where(a > 0.5, a, 0.0) + clip(a, 0.2, 0.8)
fn inplace(a):
    a[1] = 7.0
    a[2] += 1.5
    return a
fn slice_sum(a) = sum(a[2:7]) + sum(a[::2]) + sum(a[-3:])
fn row_assign(m):
    m[1] = m[0] * 2.0
    return m
fn mk(n: Int, m: Int):
    let z = zeros(n, m)
    for i in 0..n:
        for j in 0..m:
            z[i, j] = i * 10 + j
    return z
fn iter_elems(a):
    var s = 0.0
    for x in a:
        s += x
    for i, x in enumerate(a):
        s += i * x
    return s
fn int_arr(a):
    var s = 0
    for i in 0..len(a):
        s += a[i] * 3
    return s
fn f32_arr(a) = sum(a * 2.0)
fn bool_arr(a):
    var c = 0
    for x in a:
        if x:
            c += 1
    return c
fn tup(a: Float, b: Float):
    let s, p = a + b, a * b
    return s, p
fn use_tup(x: Float):
    let s, p = tup(x, 2.0)
    return s * p

fn energy(m: Mass, v: Velocity) -> Energy = 0.5 * m * v ** 2
fn speed(d: Length, t: Time) = d / t
fn rest_energy(m: Mass) = m * c ** 2
fn grav(m1: Mass, m2: Mass, r: Length) = G * m1 * m2 / r ** 2
fn unit_sum(xs) = sum(xs)

fn par_sum(n: Int) -> Float:
    var s = 0.0
    parallel for i in 0..n:
        s += 1.0 / (i + 1)
    return s

fn par_fill(a):
    parallel for i in 0..len(a):
        a[i] = sin(i * 0.001) * 2.0
    return a

fn par_minmax(a):
    var lo = 1e300
    var hi = -1e300
    parallel for i in 0..len(a):
        lo = min(lo, a[i])
        hi = max(hi, a[i])
    return lo, hi

fn par_count(n: Int) -> Int:
    var c = 0
    parallel for i in 0..n:
        if i % 3 == 0:
            c += 1
    return c

fn par_matmul(a, b):
    let n = a.shape[0]
    let m = b.shape[1]
    let k = a.shape[1]
    let out = zeros(n, m)
    parallel for i in 0..n:
        for j in 0..m:
            var acc = 0.0
            for p in 0..k:
                acc += a[i, p] * b[p, j]
            out[i, j] = acc
    return out

fn global_arr_rw(i: Int):
    counter_arr[i] += 1.0
    return counter_arr[i]

fn chk_assert(x: Float) -> Float:
    assert x > 0, "x must be positive"
    return sqrt(x)

fn overflow(x: Int) -> Int = x * x * x * x
fn idx_err(a, i: Int) = a[i]
fn zdiv(a: Int, b: Int) = a // b
fn shape_err(a, b) = a + b

fn alloc_loop(n: Int) -> Float:
    var s = 0.0
    for k in 0..n:
        let tmp = zeros(8)
        tmp[k % 8] = k
        s += sum(tmp)
    return s

fn returns_view(a) = a[1]
fn mm(a, b) = a @ b
fn mm_relu(a, b) = sum(where(a @ b > 0.0, a @ b, 0.0))
fn printer(x: Float):
    print("value", x, f"{x:.3f}")
    print(f"half={x / 2}")
    return x
'''


def _same(a, b):
    if isinstance(a, Quantity) or isinstance(b, Quantity):
        return isinstance(a, Quantity) and isinstance(b, Quantity) and a.dim == b.dim and _same(a.v, b.v)
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and (np.array_equal(a, b, equal_nan=True) or np.allclose(a, b, rtol=1e-12,
                                                                                            atol=1e-300))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b) and type(a) is type(b)
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, float) and isinstance(b, (float, np.floating)):
        return (a == b) or (a != a and b != b)
    if isinstance(a, np.floating) and isinstance(b, (float, np.floating)):
        return float(a) == float(b) or (a != a and b != b)
    return a == b


@unittest.skipUnless(HAVE, "llvmlite is not installed")
class NativeDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.it = Interpreter()
        cls.env = cls.it.run(SRC, "<native-test>")

    def fn(self, name):
        return self.env.lookup(name)

    def both(self, name, *args, approx=False, copy=True):
        f = self.fn(name)
        g = N.native("strict")(f)
        cp = (lambda xs: [x.copy() if isinstance(x, np.ndarray) else x for x in xs]) if copy else (lambda x: x)
        try:
            want = f(*cp(args))
            werr = None
        except Exception as e:                       # called directly, the bytecode side may raise Python errors
            want, werr = None, e
        try:
            got = g(*cp(args))
            gerr = None
        except ApsiError as e:
            got, gerr = None, e
        if werr is not None or gerr is not None:
            self.assertIsNotNone(werr, f"{name}{args}: only native raised {gerr!r}")
            self.assertIsNotNone(gerr, f"{name}{args}: only bytecode raised {werr!r}")
            return werr, gerr
        if approx:
            tol = 1e-12 if approx is True else approx
            self.assertTrue(np.allclose(np.asarray(_raw(want), dtype=float), np.asarray(_raw(got), dtype=float),
                                        rtol=tol), f"{name}{args}: {want!r} vs {got!r}")
        else:
            self.assertTrue(_same(want, got), f"{name}{args}: bytecode {want!r} != native {got!r}")
        return want, got

    # ---------------------------------------------------------------- scalars
    def test_arith(self):
        for a, b in [(3, 4), (-7, 2), (2.5, -1.25), (1, 2.5), (True, 3), (10**15, 3)]:
            self.both("add", a, b)
            self.both("tdiv", a, b)
            self.both("cmp", a, b)
        for a in range(-9, 10):
            for b in (-4, -3, -1, 1, 2, 5):
                self.both("ifloor", a, b)
                self.both("imod", a, b)
        for a in (-7.5, -0.0, 0.0, 3.25, 1e300, -2.0):
            for b in (-2.0, 0.75, 3.0, -0.5):
                self.both("ffloor", a, b)
                self.both("fmod_", a, b)
        for a, b in [(2, 10), (3, 0), (-3, 3), (2.0, 0.5), (9, 0.5), (2.0, -2), (1.5, 3)]:
            self.both("power", a, b)
        # Int ** (negative Int known only at run time) would change type; native says so clearly
        with self.assertRaises(ApsiError) as cm:
            N.native("strict")(self.fn("power"))(2, -2)
        self.assertIn("negative exponent", cm.exception.msg)
        for a, b in [(True, False), (True, True), (False, False)]:
            self.both("logic", a, b)
        self.both("minmax", 3, -1, 2)
        self.both("minmax", 2.5, 2.5, 1.0)
        for x in (0.5, 1.5, 2.5, -0.5, -1.5, 2.675, 1e10):
            self.both("rnd", x)
            self.both("conv", x)
        for x in (-3, 4, -2.5, 0.0):
            self.both("absval", x)
            self.both("ifx", x)
        self.both("approx_eq", 1.0, 1.0 + 1e-9)
        self.both("approx_eq", 1.0, 1.1)

    def test_control_flow(self):
        for n in (1, 7, 27, 97):
            self.both("collatz", n)
        self.both("primes_below", 500)
        self.both("ack", 2, 3)
        self.both("sum_range_step", 20)
        self.both("sum_range_step", 0)

    def test_globals(self):
        self.both("scaled", 4)
        self.both("table_at", 3)
        self.both("table_at", -1)

    # ---------------------------------------------------------------- tensors
    def test_tensors(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        m = rng.standard_normal((6, 5))
        self.both("sum_sq", a)
        self.both("neg_index", a, -1)
        self.both("neg_index", a, -50)
        self.both("matvec", m, rng.standard_normal(5))
        self.both("rows_sum", m)
        self.both("stencil", a)
        self.both("periodic", a)
        self.both("fused", a, b)
        self.both("fused_sum", a, b, approx=True)
        self.both("bcast", m, rng.standard_normal(5))
        self.both("bcast", m, rng.standard_normal((6, 1)))
        self.both("red", a, approx=True)
        self.both("red", np.arange(1, 9))
        self.both("dotn", a, b, approx=True)
        self.both("wclip", rng.random(30))
        self.both("inplace", a)
        self.both("slice_sum", a, approx=True)
        self.both("row_assign", m)
        self.both("mk", 3, 4)
        self.both("iter_elems", a, approx=True)
        self.both("int_arr", np.arange(20, dtype=np.int64))
        self.both("int_arr", np.arange(20, dtype=np.int32))
        self.both("f32_arr", rng.random(40).astype(np.float32), approx=1e-5)
        self.both("bool_arr", rng.random(40) > 0.5)
        self.both("tup", 2.0, 3.0)
        self.both("use_tup", 1.5)
        # non-contiguous inputs (transposed / strided views)
        self.both("sum_sq", a[::3])
        self.both("matvec", m.T.copy().T, rng.standard_normal(5))
        self.both("rows_sum", m[:, ::2])
        self.both("returns_view", m)
        self.both("mm", m, rng.standard_normal((5, 7)), approx=True)
        self.both("mm", m, rng.standard_normal(5), approx=True)
        self.both("mm", rng.standard_normal(6), m, approx=True)
        self.both("mm", a, b, approx=True)
        self.both("mm", m.astype(np.float32), rng.standard_normal((5, 3)).astype(np.float32), approx=1e-5)
        self.both("mm", m, rng.standard_normal((4, 2)))            # shape error in both engines
        self.both("mm_relu", m, rng.standard_normal((5, 7)), approx=True)

    def test_in_place_semantics(self):
        f = N.native("strict")(self.fn("inplace"))
        a = np.zeros(5)
        r = f(a)
        self.assertEqual(a[1], 7.0)                  # the caller's tensor is modified, as in Apsi
        self.assertIs(r, a) if r is a else self.assertTrue(np.shares_memory(r, a))

    def test_units(self):
        from apsi.units import D
        self.both("energy", Quantity(2.0, D(kg=1)), Quantity(3.0, D(m=1, s=-1)))
        self.both("speed", Quantity(100.0, D(m=1)), Quantity(9.58, D(s=1)))
        self.both("rest_energy", Quantity(9.1093837015e-31, D(kg=1)))
        self.both("grav", Quantity(5.97e24, D(kg=1)), Quantity(7.35e22, D(kg=1)), Quantity(3.84e8, D(m=1)))
        self.both("unit_sum", Quantity(np.arange(4.0), D(m=1)))
        g = N.native("strict")(self.fn("energy"))
        with self.assertRaises(ApsiError):             # a wrong unit is still caught at the boundary
            g(Quantity(2.0, D(kg=1)), Quantity(3.0, D(s=1)))

    def test_static_unit_errors(self):
        it = Interpreter()
        env = it.run("fn bad(x: Length, t: Time) = x + t\n", "<u>", check=False)
        g = N.native("strict")(env.lookup("bad"))
        from apsi.units import D
        with self.assertRaises(UnitError):
            g(Quantity(1.0, D(m=1)), Quantity(1.0, D(s=1)))

    def test_parallel(self):
        rng = np.random.default_rng(2)
        self.both("par_sum", 100000, approx=True)
        self.both("par_fill", np.zeros(5000))
        self.both("par_minmax", rng.standard_normal(10000))
        self.both("par_count", 100001)
        self.both("par_matmul", rng.standard_normal((17, 9)), rng.standard_normal((9, 13)), approx=True)
        # bit-identical across thread counts (fixed chunking)
        g = N.native("strict")(self.fn("par_sum"))
        old = N.threads()
        try:
            N.set_threads(1)
            r1 = g(1234567)
            N.set_threads(4)
            r4 = g(1234567)
        finally:
            N.set_threads(old)
        self.assertEqual(r1, r4)

    def test_global_tensor_writes(self):
        g = N.native("strict")(self.fn("global_arr_rw"))
        self.assertEqual(g(2), 1.0)
        self.assertEqual(g(2), 2.0)
        self.assertEqual(self.env.lookup("counter_arr")[2], 2.0)

    def test_errors(self):
        werr, gerr = self.both("chk_assert", -1.0)
        self.assertIn("positive", gerr.msg)
        self.both("chk_assert", 4.0)
        with self.assertRaises(ApsiError) as cm:
            N.native("strict")(self.fn("overflow"))(1 << 20)
        self.assertIn("overflow", cm.exception.msg)
        self.both("idx_err", np.zeros(3), 3)
        self.both("idx_err", np.zeros(3), -4)
        self.both("zdiv", 1, 0)
        self.both("shape_err", np.zeros(3), np.zeros(4))
        e = self.both("idx_err", np.zeros(3), 5)[1]
        self.assertTrue(e.line)                        # errors point at the Apsi source line

    def test_fast_call_path(self):
        g = N.native("strict")(self.fn("add"))
        for _ in range(3):                                      # 1st call compiles, later calls take the fast path
            self.assertEqual(g(2, 3), 5)
        self.assertEqual(g(2**70, 1), 2**70 + 1)                # beyond 64 bits: the bytecode engine, not truncation
        self.assertIsInstance(g(np.float64(1.5), np.float64(2.0)), np.float64)
        self.assertIsInstance(g(1.5, 2.0), float)
        self.assertIs(g(True, False).__class__, int)
        h = N.native("strict")(self.fn("sum_sq"))
        a = np.arange(5.0)
        for _ in range(3):
            self.assertEqual(h(a), 30.0)
        self.assertEqual(h(a[::2]), 20.0)                       # strided: the general path
        self.assertEqual(h(np.arange(3.0, dtype=np.float32)), 5.0)
        with self.assertRaises(ApsiError):
            N.native("strict")(self.fn("idx_err"))(np.zeros(3), 7)

    def test_parallel_runtime_stress(self):
        """Back-to-back regions, errors mid-region, nesting, thread-count changes and concurrent callers."""
        import random
        import threading
        env = Interpreter().run(
            "fn psum(n: Int) -> Int:\n    var s = 0\n    parallel for i in 0..n:\n        s += i\n    return s\n"
            "fn perr(n: Int, bad: Int) -> Int:\n    var s = 0\n    parallel for i in 0..n:\n"
            "        assert i != bad, \"bad index\"\n        s += 1\n    return s\n"
            "fn nested(n: Int) -> Int:\n    var s = 0\n    parallel for i in 0..n:\n        s += psum(i)\n    return s\n",
            "<stress>")
        psum, perr, nested = (N.native("strict")(env.lookup(k)) for k in ("psum", "perr", "nested"))
        R = random.Random(7)
        old = N.threads()
        try:
            for it in range(3000):
                n = R.choice([0, 1, 2, 7, 255, 256, 257, 3000])
                self.assertEqual(psum(n), n * (n - 1) // 2)
                if it % 7 == 0:
                    with self.assertRaises(ApsiError):
                        perr(R.choice([50, 300]), R.randint(0, 49))
                if it % 300 == 0:
                    N.set_threads(R.choice([1, 2, 3, 4]))
                    self.assertEqual(nested(30), sum(i * (i - 1) // 2 for i in range(30)))
        finally:
            N.set_threads(old)
        errs = []

        def caller():
            for _ in range(200):
                if psum(5000) != 12497500:
                    errs.append(1)
        ts = [threading.Thread(target=caller) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(errs, [])

    def test_memory(self):
        self.both("alloc_loop", 1000)

    def test_print(self):
        f = self.fn("printer")
        g = N.native("strict")(f)
        o1, o2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o1):
            f(1.23456)
        with contextlib.redirect_stdout(o2):
            g(1.23456)
        self.assertEqual(o1.getvalue(), o2.getvalue())


def _raw(x):
    if isinstance(x, Quantity):
        return x.v
    if isinstance(x, tuple):
        return [float(_raw(y)) for y in x]
    return x


@unittest.skipUnless(HAVE, "llvmlite is not installed")
class NativeLanguage(unittest.TestCase):
    def run_src(self, src):
        it = Interpreter()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            env = it.run(src, "<t>")
        return env, out.getvalue()

    def test_decorator_and_fallback(self):
        env, out = self.run_src(
            "@native\nfn f(x: Float) -> Float = x * 2.0\n"
            "@native(\"quiet\")\nfn g(x) = [x, x]\n"
            "print(f(2.0))\nprint(g(1))\n")
        self.assertEqual(out.split(), ["4.0", "[1,", "1]"])

    def test_strict_reports(self):
        with self.assertRaises(ApsiError):
            self.run_src("@native(\"strict\")\nfn g(x) = [x, x]\nprint(g(1))\n")

    def test_parallel_syntax_in_bytecode(self):
        env, out = self.run_src("var s = 0\nparallel for i in 0..10:\n    s += i\nprint(s)\nlet parallel = 2\n"
                                "print(parallel)\n")
        self.assertEqual(out.split(), ["45", "2"])

    def test_evolve_calls_native(self):
        env, out = self.run_src(
            "@native\nfn accel(x: Float) -> Float = -x\n"
            "var x = 1.0\nvar v = 0.0\n"
            "evolve x, v over t in 0.0..1.0 step 0.001 using rk4:\n    x' = v\n    v' = accel(x)\n"
            "print(x ~= cos(1.0))\n")
        self.assertEqual(out.strip(), "true")

    def test_grad_through_native(self):
        env, out = self.run_src("@native\nfn f(x) = x * x * 3.0\nprint(grad(f)(2.0))\n")
        self.assertEqual(out.strip(), "12.0")


if __name__ == "__main__":
    unittest.main()


GPU_SRC = r'''
@gpu
fn saxpy(a: Float, x, y):
    parallel for i in 0..len(x):
        y[i] = a * x[i] + y[i]

@gpu
fn sumsq(x) -> Float:
    var s = 0.0
    parallel for i in 0..len(x):
        s += x[i] * x[i]
    return s

@gpu
fn rows(m, out):
    parallel for i in 0..m.shape[0]:
        var acc = 0.0
        for j in 0..m.shape[1]:
            acc += m[i, j]
        out[i] = acc

@gpu
fn uses_exp(x):
    parallel for i in 0..len(x):
        x[i] = exp(x[i])

@gpu
fn scattered(x, y):
    parallel for i in 0..len(x):
        y[len(x) - 1 - i] = x[i]

@gpu
fn no_loop(x) = x * 2.0
'''


@unittest.skipUnless(HAVE, "llvmlite is not installed")
class Gpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APSI_NATIVE_QUIET"] = "1"
        cls.env = Interpreter().run(GPU_SRC, "<gpu-test>")

    def g(self, name):
        return self.env.lookup(name)

    def test_ptx_is_generated_for_sm75(self):
        for name, args in [("saxpy", (2.0, np.zeros(8), np.zeros(8))), ("sumsq", (np.zeros(8),)),
                           ("rows", (np.zeros((3, 4)), np.zeros(3)))]:
            ptx = self.g(name).ptx(*args)
            self.assertIn(".target sm_75", ptx)
            self.assertIn(".entry apsi_kernel", ptx)
        self.assertIn("atom.global.add.f64", self.g("sumsq").ptx(np.zeros(8)))

    def test_unsupported_kernels_are_refused_with_a_reason(self):
        from apsi.native.compiler import Unsupported
        with self.assertRaises(Unsupported) as cm:
            self.g("uses_exp").ptx(np.zeros(4))
        self.assertIn("exp", str(cm.exception))
        with self.assertRaises(Unsupported):
            self.g("no_loop").ptx(np.zeros(4))

    def test_cpu_fallback_gives_the_right_results(self):
        x = np.linspace(0, 1, 1001)
        y = np.ones(1001)
        self.g("saxpy")(2.0, x, y)
        self.assertTrue(np.allclose(y, 2.0 * x + 1.0))
        self.assertAlmostEqual(self.g("sumsq")(x), float(np.sum(x * x)), places=9)
        m = np.arange(12.0).reshape(3, 4)
        out = np.zeros(3)
        self.g("rows")(m, out)
        self.assertTrue(np.array_equal(out, m.sum(axis=1)))
        z = np.array([0.0, 1.0])
        self.g("uses_exp")(z)
        self.assertTrue(np.allclose(z, [1.0, np.e]))

    def test_multi_gpu_row_split_analysis(self):
        from apsi.native.gpu import _written_arrays, kernel_shape
        _, loop, _ = kernel_shape(self.g("saxpy").fndef)
        self.assertEqual(_written_arrays(loop.body, "i"), {"y": True})       # each GPU writes its own rows
        _, loop, _ = kernel_shape(self.g("scattered").fndef)
        self.assertEqual(_written_arrays(loop.body, "i"), {"y": False})      # not row-wise: one GPU only


class NativeCli(unittest.TestCase):
    def apsi(self, *argv):
        import subprocess
        env = dict(os.environ, PYTHONPATH=ROOT, APSI_NO_GPU="1")
        return subprocess.run([sys.executable, "-m", "apsi", *argv], capture_output=True, text=True, env=env,
                              timeout=300)

    def test_info(self):
        r = self.apsi("native", "info")
        self.assertIn("native compiler" if HAVE else "unavailable", r.stdout)

    @unittest.skipUnless(HAVE, "llvmlite is not installed")
    def test_explain_and_run_native(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.apsi")
            with open(p, "w") as fh:
                fh.write("@native\nfn f(n: Int) -> Int = n * 2\n@native\nfn g(n) = [n]\nprint(f(21))\nprint(g(1))\n"
                         "fn h(x: Float) -> Float = x + 1.0\nprint(h(1.0))\n")
            r = self.apsi("native", "explain", p)
            self.assertIn("native   f", r.stdout)
            self.assertIn("bytecode g: lists", r.stdout)
            r = self.apsi("run", "--native", p)
            self.assertEqual(r.stdout.split(), ["42", "[1]", "2.0"])
            r = self.apsi("native", "asm", p, "f")
            self.assertEqual(r.returncode, 0)
