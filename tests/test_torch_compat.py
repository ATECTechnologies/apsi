"""The native `use torch` module against real PyTorch: every layer (outputs and gradients), loss, optimizer and
scheduler. Skipped when PyTorch is not installed. The reference scripts are in tests/torchref/."""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import torch  # noqa: F401
except Exception:
    torch = None


class AgainstPyTorch(unittest.TestCase):
    def _run(self, name):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "torchref", name)], capture_output=True,
                           text=True, timeout=1200)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("BAD: []", r.stdout, r.stdout[-3000:])

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_layers(self):
        self._run("layers_vs_torch.py")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_losses_optimizers_schedulers(self):
        self._run("training_vs_torch.py")


class WithoutPyTorch(unittest.TestCase):
    def test_module_runs_natively(self):
        sys.path.insert(0, ROOT)
        from apsi import Interpreter
        src = """use torch
let net = Sequential(Linear(4, 8), ReLU(), BatchNorm1d(8), Linear(8, 3))
var p = net.init(seed=0)
let opt = AdamW(lr=0.01)
let sched = StepLR(opt, 2, 0.5)
let x = randn(16, 4)
let y = tensor([i % 3 for i in 0..16])
for step in 0..6:
    let loss, aux, g = value_and_grad_aux(fn(q) => cross_entropy(net(q, x), y, label_smoothing=0.1))(p)
    p = opt.step(p, clip_grad_norm(g, 1.0))
    sched.step()
assert abs(opt.lr - 0.00125) < 1e-12
net.eval()
assert shape(net(p, x)) == (16, 3)
let sd = state_dict(net, p)
let p2 = load_state_dict(net, p, sd)
assert parameters_count(p2) == 4 * 8 + 8 + 8 + 8 + 8 * 3 + 3
"""
        Interpreter().run(src, "<t>")


if __name__ == "__main__":
    unittest.main()
