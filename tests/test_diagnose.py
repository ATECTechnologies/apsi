"""diagnose(): the checks for a model built from scratch, and `apsi cache`."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from apsi import layers as L
from apsi.stdlib import spiral_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Diagnose(unittest.TestCase):
    def setUp(self):
        self.X, self.y = spiral_data(50, 3, 0.2, 0)

    def test_healthy(self):
        m = L.mlp([2, 32, 3])
        r = L.diagnose(m, m.init(1), self.X, self.y, loss="cross_entropy", lr=0.01, verbose=False)
        self.assertTrue(r.ok, r.problems)
        self.assertEqual(r.warnings, [])
        self.assertLess(r.losses[-1], r.losses[0])

    def test_too_few_outputs(self):
        m = L.mlp([2, 32, 2])
        r = L.diagnose(m, m.init(1), self.X, self.y, loss="cross_entropy", verbose=False)
        self.assertFalse(r.ok)
        self.assertIn("needs 3 outputs", r.problems[0])

    def test_nan_and_scale(self):
        X = np.asarray(self.X, float).copy()
        X[3, 1] = np.nan
        m = L.mlp([2, 8, 3])
        r = L.diagnose(m, m.init(1), X, self.y, loss="cross_entropy", verbose=False)
        self.assertTrue(any("NaN" in p for p in r.problems))
        r = L.diagnose(m, m.init(1), np.asarray(self.X) * 1e5, self.y, loss="cross_entropy", verbose=False)
        self.assertTrue(any("scale them" in w for w in r.warnings))


class Cache(unittest.TestCase):
    def test_show_and_clean(self):
        with tempfile.TemporaryDirectory() as home:
            gpu = os.path.join(home, "cache", "gpu")
            os.makedirs(gpu)
            old = os.path.join(gpu, "hub-kernels-000000.ptx")
            with open(old, "w") as fh:
                fh.write("x" * 1000)
            env = {**os.environ, "APSI_HOME": home, "APSI_HUB_CACHE": os.path.join(home, "hub"), "PYTHONPATH": ROOT}
            run = lambda *a: subprocess.run([sys.executable, "-m", "apsi", "cache", *a], capture_output=True,  # noqa
                                            text=True, env=env, cwd=ROOT)
            out = run().stdout
            self.assertIn("1 stale item", out)
            self.assertIn("would remove", run("clean", "--dry-run").stdout)
            self.assertTrue(os.path.exists(old))
            run("clean")
            self.assertFalse(os.path.exists(old))


if __name__ == "__main__":
    unittest.main()
