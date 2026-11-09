"""PyTorch programs converted to native Apsi: they run without Python or PyTorch, and (when PyTorch is installed)
give the same outputs and gradients as PyTorch for the same weights."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter                             # noqa: E402
from apsi.convert import convert_source                 # noqa: E402

try:
    import torch
except Exception:
    torch = None

NETS = os.path.join(ROOT, "tests", "data", "torch_nets.py")


def _code(src):
    out = convert_source(src)
    return out[0] if isinstance(out, tuple) else out


class TorchConversion(unittest.TestCase):
    def test_training_script_becomes_native_apsi(self):
        src = open(NETS, encoding="utf-8").read().replace("range(15)", "range(4)").replace(
            'torch.save(model.state_dict(), "net.pt")', "")
        code = _code(src)
        for absent in ("pyimport", "torch_py", "backward", "zero_grad", "NOTE: not converted"):
            self.assertNotIn(absent, code)
        for present in ("use torch as torch", "torch.Module(", "torch.value_and_grad_aux", "torch.AdamW(",
                        "torch.DataLoader(torch.TensorDataset(X, y)", "torch.clip_grad_norm", "model.eval()",
                        "model(model_params, X)", "param_count(model_params)"):
            self.assertIn(present, code)
        it = Interpreter()
        env = it.run(code, "<converted>")
        vars_ = {**it.globals.vars, **getattr(env, "vars", {})}
        self.assertEqual(int(vars_["n_params"]), 2386)                 # same architecture as PyTorch
        self.assertGreater(float(vars_["acc"]), 0.8)                  # and it learns

    def test_tensor_methods(self):
        src = """import torch
x = torch.arange(24.0).reshape(2, 3, 4)
a = x.permute(0, 2, 1)
b = x.transpose(1, 2)
c = x.unsqueeze(-1).squeeze(-1)
v, i = x.max(dim=1)
m = x.mean(dim=1, keepdim=True)
s = x.size(0)
"""
        it = Interpreter()
        env = it.run(_code(src), "<t>")
        g = {**it.globals.vars, **getattr(env, "vars", {})}
        x = np.arange(24.0).reshape(2, 3, 4)
        self.assertTrue(np.array_equal(g["a"], x.transpose(0, 2, 1)))
        self.assertTrue(np.array_equal(g["b"], x.swapaxes(1, 2)))
        self.assertTrue(np.array_equal(g["c"], x))
        self.assertTrue(np.array_equal(g["v"], x.max(1)) and np.array_equal(g["i"], x.argmax(1)))
        self.assertEqual(np.shape(g["m"]), (2, 1, 4))
        self.assertEqual(int(g["s"]), 2)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_same_outputs_and_gradients_as_pytorch(self):
        from apsi.torch_compat import load_state_dict
        from apsi.stdlib import value_and_grad, cross_entropy
        src = open(NETS, encoding="utf-8").read()
        defs = src.split("torch.manual_seed(0)")[0]
        ns = {}
        exec(compile(defs, NETS, "exec"), ns)
        it = Interpreter()
        env = it.run(_code(defs + "\nnet = Net(6, 16, 2, 2)\ncnn = CNN(10)\nseqc = SeqClassifier(50, 8, 12, 3)\n"),
                     "<t>")
        g = {**it.globals.vars, **getattr(env, "vars", {})}
        torch.manual_seed(1)
        cases = [("net", ns["Net"](6, 16, 2, 2), torch.randn(5, 6, dtype=torch.float64)),
                 ("cnn", ns["CNN"](10), torch.randn(3, 1, 16, 16, dtype=torch.float64)),
                 ("seqc", ns["SeqClassifier"](50, 8, 12, 3), torch.randint(0, 50, (4, 7)))]
        for name, tm, x in cases:
            tm = tm.double().eval()
            with torch.no_grad():
                for p in tm.parameters():
                    p.add_(0.1 * torch.randn_like(p))
            am = g[name]
            ap = load_state_dict(am, g[name + "_params"], tm.state_dict())
            am.eval()
            out_t = tm(x)
            out_a = np.asarray(am(ap, x.numpy()))
            self.assertLess(np.max(np.abs(out_t.detach().numpy() - out_a)), 1e-12, name)
            yv = torch.randint(0, out_t.shape[1], (out_t.shape[0],))
            torch.nn.functional.cross_entropy(out_t, yv).backward()
            _, ga = value_and_grad(lambda p: cross_entropy(am(p, x.numpy()), yv.numpy()))(ap)
            last = [k for k, _ in tm.named_parameters()][-1]                 # the output layer's bias
            sub = ga
            for part in last.split("."):
                sub = sub[int(part)] if part.isdigit() else sub[part]
            self.assertLess(np.max(np.abs(dict(tm.named_parameters())[last].grad.numpy() - np.asarray(sub))),
                            1e-12, name)


if __name__ == "__main__":
    unittest.main()


class TrainingEquivalence(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_converted_training_gives_the_same_weights(self):
        import shutil
        import subprocess
        import tempfile
        try:
            import safetensors  # noqa: F401
        except ImportError:
            self.skipTest("safetensors is not installed")
        d = tempfile.mkdtemp()
        try:
            src = open(os.path.join(ROOT, "tests", "data", "torch_train_exact.py"), encoding="utf-8").read()
            torch.set_default_dtype(torch.float64)
            try:
                ns = {"__name__": "defs"}
                exec(compile(src.split("torch.manual_seed(0)")[0], "defs", "exec"), ns)
                torch.manual_seed(1)
                m = ns["Model"](5, 8, 3)
                from safetensors.torch import save_file
                save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(d, "init.safetensors"))
                save_file({"X": torch.randn(64, 7, 5), "y": torch.randint(0, 3, (64,)).double()},
                          os.path.join(d, "data.safetensors"))
                cwd = os.getcwd()
                os.chdir(d)
                try:
                    exec(compile(src, "torch_train_exact.py", "exec"), {"__name__": "__main__"})
                finally:
                    os.chdir(cwd)
            finally:
                torch.set_default_dtype(torch.float32)
            os.replace(os.path.join(d, "final.safetensors"), os.path.join(d, "final_torch.safetensors"))
            code = _code(src)
            self.assertNotIn("NOTE", code)
            with open(os.path.join(d, "train.apsi"), "w", encoding="utf-8") as fh:
                fh.write(code)
            r = subprocess.run([sys.executable, "-m", "apsi", "run", "train.apsi", "--allow-read", "--allow-write"],
                               cwd=d, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            from safetensors.numpy import load_file
            a, b = load_file(os.path.join(d, "final.safetensors")), load_file(os.path.join(d, "final_torch.safetensors"))
            self.assertEqual(set(a), set(b))
            for k in b:
                self.assertLess(float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))), 1e-9, k)
        finally:
            shutil.rmtree(d, ignore_errors=True)
