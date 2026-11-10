"""@native("fma"): checks stay on, a * b + c may become one fused multiply-add."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apsi import Interpreter                       # noqa: E402
from apsi import native as N                        # noqa: E402

SRC = r'''
fn horner(x: Float, n: Int) -> Float:
    var y = 0.0
    for k in 0..n:
        y = y * x + 1.0 / (k + 1)
    return y

fn safe_div(a: Float, b: Float) -> Float = a / b
'''


@unittest.skipUnless(N.available(), "llvmlite is not installed")
class Fma(unittest.TestCase):
    def setUp(self):
        self.env = Interpreter().run(SRC, "<fma>")

    def test_contracts_only_multiply_add(self):
        f = N.native("fma")(self.env.lookup("horner"))
        ir_text = f.llvm_ir(0.5, 10)
        self.assertIn("contract", ir_text)
        self.assertNotIn(" fast ", ir_text)
        exact = N.native(self.env.lookup("horner"))(0.37, 200)
        self.assertTrue(np.isclose(f(0.37, 200), exact, rtol=1e-14, atol=0))

    def test_checks_stay_on(self):
        g = N.native("fma")(self.env.lookup("safe_div"))
        with self.assertRaises(Exception):
            g(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
