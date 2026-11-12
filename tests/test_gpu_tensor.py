"""gpu_tensor: tensors on the card with Apsi's own kernels (checked on the exact emulator and on the processor), and
any function that takes a tensor still working on them."""
import subprocess
import sys
import os
import unittest

import numpy as np

from apsi.gpu_tensor import gpu_tensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


class GpuTensor(unittest.TestCase):
    def test_kernels_give_the_processors_values(self):
        rng = np.random.default_rng(0)
        a, w, b, c = rng.normal(size=(5, 7)), rng.normal(size=(7, 4)), rng.normal(size=4), rng.normal(size=(3, 5, 7))
        for dev in ("emulator", "cpu"):
            A, W = gpu_tensor(a, dev), gpu_tensor(w, dev)
            C = gpu_tensor(c, dev)
            cases = [((A @ W + b).gelu(), gelu(a @ w + b)), (A.T @ A, a.T @ a), (A @ A.T, a @ a.T), (C @ W, c @ w),
                     (C @ gpu_tensor(np.swapaxes(c, 1, 2), dev), c @ np.swapaxes(c, 1, 2)),
                     (2.0 - A * A + 1.0 - A / 4, 2.0 - a * a + 1.0 - a / 4), (A.sum(-1), a.sum(-1)),
                     (A.sum(0), a.sum(0)), (A.softmax(), np.exp(a) / np.exp(a).sum(-1, keepdims=True)),
                     (A.layer_norm(), (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)),
                     (W.rows([3, 0, 3]), w[[3, 0, 3]]), (A.exp(), np.exp(a)), (A @ W - b, a @ w - b),
                     (A.reshape(7, 5), a.reshape(7, 5)), (A.tanh().sigmoid(), 1 / (1 + np.exp(-np.tanh(a))))]
            for got, ref in cases:
                np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-5, atol=1e-5)
            self.assertAlmostEqual(A.sum(), a.sum(), places=4)

    def test_from_apsi(self):
        src = ("let x = randn(4, 6, seed=1)\nlet w = randn(6, 3, seed=2)\nlet h = (gpu_tensor(x) @ w + 0.5).gelu()\n"
               "assert max(abs(h.numpy() - gelu(x @ w + 0.5))) < 1e-5\nprint(sum(exp(h)) > 0.0)\n")
        r = subprocess.run([sys.executable, "-m", "apsi", "run", "-"], input=src, capture_output=True, text=True,
                           cwd=ROOT)
        if "true" not in r.stdout.lower():
            path = os.path.join(ROOT, "tests", "_gt_tmp.apsi")
            with open(path, "w") as f:
                f.write(src)
            try:
                r = subprocess.run([sys.executable, "-m", "apsi", "run", path], capture_output=True, text=True,
                                   cwd=ROOT)
            finally:
                os.remove(path)
        self.assertIn("true", r.stdout.lower(), r.stderr)


if __name__ == "__main__":
    unittest.main()
