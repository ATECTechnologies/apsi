"""Models must actually learn: small tasks trained from scratch reach their targets."""
import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Learning(unittest.TestCase):
    def test_sequence_cross_entropy(self):
        from apsi.stdlib import cross_entropy
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(2, 5, 7))
        labels = rng.integers(0, 7, size=(2, 5))
        flat = cross_entropy(logits.reshape(-1, 7), labels.reshape(-1))
        self.assertAlmostEqual(float(cross_entropy(logits, labels)), float(flat), places=12)

    @unittest.skipIf(sys.platform.startswith("win"), "data-parallel fit needs fork")
    def test_data_parallel_fit_matches_one_process(self):
        from apsi import layers
        from apsi.stdlib import Adam
        rng = np.random.default_rng(1)
        X = rng.normal(size=(600, 8))
        y = (X[:, 0] > 0).astype(np.int64)
        net = layers.mlp([8, 16, 2])
        a = layers.fit(net, net.init(seed=0), X, y, loss="cross_entropy", opt=Adam(1e-2), epochs=3, batch_size=100,
                       verbose=False, precision="fp64")
        b = layers.fit(net, net.init(seed=0), X, y, loss="cross_entropy", opt=Adam(1e-2), epochs=3, batch_size=100,
                       verbose=False, precision="fp64", workers=3)
        np.testing.assert_allclose(a.history, b.history, rtol=1e-9)

    def test_three_models_learn(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        out = subprocess.run([sys.executable, "-m", "apsi", "run", os.path.join(ROOT, "benchmarks", "ai", "learn.apsi"),
                              "--allow-python"], capture_output=True, text=True, env=env, timeout=900).stdout
        lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}
        self.assertIn("-> 1.000", lines["spiral"])
        self.assertIn("-> 1.000", lines["LSTM"])
        gpt = lines["char"].split()
        self.assertLess(float(gpt[gpt.index("->") + 1]), 0.2, out)


if __name__ == "__main__":
    unittest.main()
