"""PyTorch backend of `use layers`: parity with NumPy, training, nn.Module round trip.
Skipped where PyTorch is not installed.  Run:  python -m unittest tests.test_torch"""
import importlib.util
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import layers as L                      # noqa: E402
from apsi.errors import RuntimeErr                # noqa: E402
from apsi.stdlib import adam                      # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class TorchBackend(unittest.TestCase):
    def test_every_layer_matches_numpy(self):
        r = L.check_torch(device="cpu", verbose=False)
        bad = [c.name for c in r.cases if c.output_diff > 1e-8 or c.grad_diff > 1e-6]
        self.assertTrue(r.ok, f"mismatch in {bad} (largest {r.worst:.2e})")

    def test_fit_learns_on_torch(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        m = L.mlp([2, 32, 32, 2])
        r = L.fit(m, m.init(1), X, y, loss="cross_entropy", opt=adam(0.02), epochs=60, batch_size=50,
                  backend="torch", device="cpu", verbose=False)
        self.assertLess(r.history[-1], r.history[0] * 0.5)
        acc = np.mean(np.argmax(L.predict(m, r.params, X), -1) == y)          # trained on torch, run on NumPy
        self.assertGreater(acc, 0.9)

    def test_graph_training_and_module_round_trip(self):
        g = L.graph(edges=[[0, 1], [1, 2], [2, 3], [3, 0]], n=4, undirected=True)
        x = np.random.default_rng(1).standard_normal((4, 3))
        net = L.gat_net(3, 4, 2, heads=2, dropout=0.0)
        p = net.init(0)
        r = L.fit(net, p, x, np.array([0, 1, 0, 1]), loss="cross_entropy", graph=g, epochs=5, backend="torch",
                  device="cpu", precision="fp64", verbose=False)
        self.assertEqual(len(r.history), 5)
        mod = L.torch_module(net, p, precision="fp64")
        out = mod(x, graph=g).detach().numpy()
        np.testing.assert_allclose(out, L.predict(net, p, x, graph=g), atol=1e-10)
        back = L.params_from_torch(net, mod)
        self.assertEqual(L.count_params(back), L.count_params(p))


@unittest.skipIf(HAVE_TORCH, "PyTorch is installed")
class WithoutTorch(unittest.TestCase):
    def test_clear_message_and_auto_fallback(self):
        m = L.mlp([2, 2])
        with self.assertRaises(RuntimeErr) as cm:
            L.fit(m, m.init(0), np.zeros((2, 2)), np.zeros((2, 2)), backend="torch", verbose=False)
        self.assertIn("pip install torch", str(cm.exception))
        r = L.fit(m, m.init(0), np.zeros((2, 2)), np.zeros((2, 2)), backend="auto", epochs=1, verbose=False)
        self.assertEqual(len(r.history), 1)
        self.assertEqual([d.name for d in L.devices()], ["numpy"])


if __name__ == "__main__":
    unittest.main()
