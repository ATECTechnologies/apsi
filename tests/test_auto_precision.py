"""precision="auto": bf16 where the GPU has it, fp16 on older GPUs, fp32 on the CPU; overrides and custom pairs."""
import os
import types
import unittest
from unittest import mock

from apsi import torch_backend as tb
from apsi.errors import RuntimeErr


class _DT:
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return self.n


def fake_torch(bf16):
    t = types.SimpleNamespace()
    t.dtype = _DT
    for n in ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn"):
        setattr(t, n, _DT(n))
    t.cuda = types.SimpleNamespace(is_bf16_supported=lambda: bf16)
    t.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()),
                                       cudnn=types.SimpleNamespace())
    return t


def dev(kind):
    return types.SimpleNamespace(type=kind)


class AutoPrecision(unittest.TestCase):
    def run_with(self, bf16, name, kind, env=None):
        with mock.patch.object(tb, "_torch", fake_torch(bf16)), \
                mock.patch.dict(os.environ, {"APSI_PRECISION": env} if env else {}, clear=False):
            if not env:
                os.environ.pop("APSI_PRECISION", None)
            store, amp = tb._precision(name, dev(kind))
            return str(store), (str(amp) if amp is not None else None)

    def test_auto_choice(self):
        self.assertEqual(self.run_with(True, "auto", "cuda"), ("float32", "bfloat16"))     # RTX 30xx and newer
        self.assertEqual(self.run_with(False, "auto", "cuda"), ("float32", "float16"))     # GTX 10xx / RTX 20xx
        self.assertEqual(self.run_with(True, "auto", "cpu"), ("float32", None))
        self.assertEqual(self.run_with(True, "auto", "mps"), ("float32", "float16"))
        self.assertEqual(self.run_with(True, None, "xpu"), ("float32", "bfloat16"))

    def test_override_and_custom(self):
        self.assertEqual(self.run_with(True, "fp32", "cuda"), ("float32", None))
        self.assertEqual(self.run_with(True, "auto", "cuda", env="fp16"), ("float32", "float16"))
        self.assertEqual(self.run_with(True, {"store": "bfloat16", "compute": "bfloat16"}, "cuda"), ("bfloat16", None))
        self.assertEqual(self.run_with(True, {"compute": "float8_e4m3fn"}, "cuda"), ("float32", "float8_e4m3fn"))
        with self.assertRaises(RuntimeErr):
            self.run_with(True, {"compute": "float7"}, "cuda")
        with self.assertRaises(RuntimeErr):
            self.run_with(True, "fp12", "cuda")

    def test_low_bit_formats(self):
        # fp8 / fp6 / fp4 / int8 / int4 weights train with a float32 master and 16-bit products on the GPU
        for name in ("fp8", "fp6", "fp6_e2m3", "fp4", "int8", "int4", "mxfp4"):
            self.assertEqual(self.run_with(True, name, "cuda"), ("float32", "bfloat16"), name)
            self.assertEqual(self.run_with(False, name, "cuda"), ("float32", "float16"), name)
            self.assertEqual(self.run_with(True, name, "cpu"), ("float32", None), name)
        self.assertEqual(self.run_with(True, "auto", "cuda", env="fp4"), ("float32", "bfloat16"))


if __name__ == "__main__":
    unittest.main()
