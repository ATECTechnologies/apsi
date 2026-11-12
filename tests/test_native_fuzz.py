"""Differential fuzzing: random Int/Float programs must behave identically on the native and bytecode engines.

Values, errors and even the scalar class (Python number or NumPy scalar) must match, for Python arguments and for
NumPy scalars (tensor elements), whose rules differ (x/0.0 -> inf, Int // 0 -> 0, wrap-around). Seeded, so a
failure always reproduces. Set APSI_FUZZ_SEEDS=1,2,3,... to run more.
"""
import math
import os
import random
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter      # noqa: E402
from apsi import native as N      # noqa: E402
from apsi.errors import ApsiError  # noqa: E402


def program(R, n_fns):
    def expr(depth, kind):
        if depth == 0 or R.random() < 0.25:
            if R.random() < 0.5:
                return R.choice(["a", "b", "c"])
            if kind == "int":
                return str(R.randint(-9, 9)) if R.random() < .8 else str(R.choice([2**31, -2**40, 2**62]))
            return repr(R.choice([0.5, -1.25, 3.0, 1e-3, 7.75, -0.0, 2.0]))
        r = R.random()
        if r < 0.6:
            op = R.choice(["+", "-", "*", "//", "%"] if kind == "int" else ["+", "-", "*", "/", "//", "%"])
            return f"({expr(depth - 1, kind)} {op} {expr(depth - 1, kind)})"
        if r < 0.7:
            return f"({expr(depth - 1, kind)} ** {R.choice([0, 1, 2, 3])})"
        if r < 0.8:
            return f"{R.choice(['min', 'max'])}({expr(depth - 1, kind)}, {expr(depth - 1, kind)})"
        if r < 0.88:
            return f"abs({expr(depth - 1, kind)})"
        cmp = R.choice(["<", "<=", "==", "!=", ">", ">="])
        return (f"({expr(depth - 1, kind)} if {expr(depth - 1, kind)} {cmp} {expr(depth - 1, kind)} "
                f"else {expr(depth - 1, kind)})")
    src = []
    for i in range(n_fns):
        kind = "int" if i % 2 == 0 else "float"
        t = "Int" if kind == "int" else "Float"
        zero = "0" if kind == "int" else "0.0"
        src.append(f"fn f{i}(a: {t}, b: {t}, c: {t}):\n    var s = {zero}\n    for k in 0..{R.randint(1, 6)}:\n"
                   f"        s = s + {expr(2, kind)}\n        if s > {R.randint(-50, 50)}:\n"
                   f"            s = s - {expr(1, kind)}\n    return s + {expr(3, kind)}\n")
    return src


@unittest.skipUnless(N.available(), "llvmlite is not installed")
class NativeFuzz(unittest.TestCase):
    def run_seed(self, seed, n_fns=60, calls=8):
        R = random.Random(seed)
        src = program(R, n_fns)
        env = Interpreter().run("\n".join(src), f"<fuzz {seed}>", check=False)
        problems = []
        for i in range(n_fns):
            f = env.lookup(f"f{i}")
            g = N.native("strict")(f)
            for _ in range(calls):
                if i % 2 == 0:
                    args = [R.choice([0, 1, -1, 2, 3, -7, 10, 2**20, -2**33, R.randint(-100, 100)]) for _ in range(3)]
                else:
                    args = [R.choice([0.0, -0.0, 1.0, -2.5, 0.1, 1e10, -3.75, R.uniform(-10, 10)]) for _ in range(3)]
                if R.random() < 0.35:
                    args = [(np.int64(x) if isinstance(x, int) else np.float64(x)) if R.random() < 0.6 else x
                            for x in args]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        want, werr = f(*args), None
                    except Exception as e:
                        want, werr = None, e
                    try:
                        got, gerr = g(*args), None
                    except ApsiError as e:
                        got, gerr = None, e
                if werr is not None or gerr is not None:
                    # a native overflow error where the bytecode engine's unlimited Python Int kept going
                    if gerr is not None and werr is None and "overflow" in gerr.msg:
                        continue
                    if (werr is None) != (gerr is None):
                        problems.append((f"f{i}", args, werr, gerr))
                    continue
                same_nan = isinstance(want, float) and isinstance(got, float) and math.isnan(want) and math.isnan(got)
                if type(want) is not type(got) or not (want == got or same_nan) or (
                        isinstance(want, float) and want == 0 and math.copysign(1, want) != math.copysign(1, got)):
                    problems.append((f"f{i}", args, want, got))
        self.assertEqual(problems, [], "\n" + "\n".join(map(repr, problems[:5])) +
                         "\n" + "\n".join(src[int(p[0][1:])] for p in problems[:2]))

    def test_seeds(self):
        seeds = [int(x) for x in os.environ.get("APSI_FUZZ_SEEDS", "101,102,103").split(",")]
        for seed in seeds:
            with self.subTest(seed=seed):
                self.run_seed(seed)


if __name__ == "__main__":
    unittest.main()
