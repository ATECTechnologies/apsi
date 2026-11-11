"""int8 / int4 weights on the GPU: the quantization, the products and a whole model, against bfloat16."""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_hub_models import tiny_llama, _on          # noqa: E402
from apsi import hub as H, hub_gpu                    # noqa: E402


class Quantized(unittest.TestCase):
    def test_float_formats_round_like_ml_dtypes(self):
        try:
            import ml_dtypes
        except ImportError:
            self.skipTest("ml_dtypes not installed")
        rng = np.random.default_rng(0)
        v = np.concatenate([np.arange(0, 449, 0.0625), rng.uniform(-448, 448, 20000), rng.uniform(-0.02, 0.02, 20000)])
        v = np.concatenate([v, -v]).astype(np.float32)
        np.testing.assert_array_equal(hub_gpu.fp_round(v, 8), v.astype(ml_dtypes.float8_e4m3fn).astype(np.float32))
        if hasattr(ml_dtypes, "float4_e2m1fn"):
            t = np.concatenate([np.arange(-6, 6.001, 0.125), rng.uniform(-6, 6, 20000)]).astype(np.float32)
            np.testing.assert_array_equal(hub_gpu.fp_round(t, 4), t.astype(ml_dtypes.float4_e2m1fn).astype(np.float32))

    def test_quantization_and_products(self):
        e = hub_gpu.Emulator()
        rng = np.random.default_rng(3)
        w = rng.standard_normal((96, 128)).astype(np.float32) * 0.05
        x = rng.standard_normal((3, 128)).astype(np.float32)
        ref = x @ hub_gpu.from_bf16_pairs(hub_gpu.to_bf16_pairs(w), w.shape).T
        for kind, tol in (("int8", 0.01), ("int4", 0.12), ("fp8", 0.04), ("fp4", 0.15)):
            m = e.qmatrix(w, kind)
            self.assertEqual(m.kind, kind)
            X = e.alloc(x.size)
            e.put(X, x.reshape(-1))
            for T in (3, 12):                               # matrix-vector, and the tiled product (dequantized)
                xs = np.tile(x, (4, 1))[:T]
                e.put(X, xs.reshape(-1)) if T <= 3 else None
                X2 = e.alloc(xs.size)
                e.put(X2, xs.reshape(-1))
                Y = e.alloc(T * 96)
                e.linear(m, X2, Y, T)
                want = xs @ hub_gpu.from_bf16_pairs(hub_gpu.to_bf16_pairs(w), w.shape).T
                err = np.abs(e.get(Y, T * 96).reshape(T, 96) - want).max() / np.abs(want).max()
                self.assertLess(err, tol, (kind, T, err))

    def test_whole_model_in_int8_and_int4(self):
        def go():
            with tempfile.TemporaryDirectory() as td:
                tiny_llama(td, "qwen2", ctx=300, exact=True, dtype="BF16", hd=64, inter=128)
                ids = [3, 5, 7, 11, 2, 9] * 20
                base = H.Model(td)
                a = base.net.forward(np.array([ids]), base.net.new_cache(), 0)
                self.assertEqual(base.plan["gpu"]["weights"], "bf16")
                # a tiny random network magnifies weight errors about 10x (int8: 0.4 % per weight -> ~4 % here);
                # real language models are far less sensitive
                for kind, tol in (("int8", 0.1), ("int4", 1.5), ("fp8", 0.4), ("fp4", 1.5)):
                    with mock.patch.dict(os.environ, {"APSI_GPU_WEIGHTS": kind}):
                        q = H.Model(td)
                    self.assertEqual(q.plan["gpu"]["weights"], kind)
                    b = q.net.forward(np.array([ids]), q.net.new_cache(), 0)
                    err = np.abs(b - a).max() / np.abs(a).max()
                    self.assertLess(err, tol, (kind, err))
                    agree = (b[0].argmax(-1) == a[0].argmax(-1)).mean()
                    self.assertGreater(agree, 0.9 if kind == "int8" else 0.5, (kind, agree))
                    self.assertTrue(len(q.generate("hello", max_new_tokens=5)) > 0)
                # a card too small for bfloat16 gets int8 by itself
                small = hub_gpu.Emulator()
                need_bf16 = sum(base.net._layer_values(i) for i in range(base.net.L)) * 2
                kv = 2 * 2 * 2 * 16 * 300 * base.net.L                 # float16 cache of every layer
                head = 2 * int(np.prod(base.net._wshape(None)))
                small.free_memory = lambda: int(need_bf16 * 0.3) + (384 << 20) + kv + head + 4 * 128 * 64 * 2
                auto = _on(small, lambda: H.Model(td))
                self.assertRegex(auto.plan["gpu"]["weights"], r"^int[48]")       # int8 where room is left, int4 elsewhere
                self.assertEqual(auto.plan["gpu"]["layers"], auto.net.L)
        _on(hub_gpu.Emulator(), go)

    def test_real_gpu_matches_the_emulator(self):
        r, why = hub_gpu.runner("auto")
        if r is None:
            self.skipTest(why)
        e = hub_gpu.Emulator()
        e.tc = r.tc                                     # prompts on tensor cores: the same input rounding
        rng = np.random.default_rng(5)
        try:
            for O, I in ((96, 128), (300, 896), (8, 4864)):
                w = (rng.standard_normal((O, I)) * 0.05).astype(np.float32)
                for kind in ("int8", "int4", "fp8", "fp4"):
                    mg, me = r.qmatrix(w, kind), e.qmatrix(w, kind)
                    np.testing.assert_array_equal(r.get(mg.scales, O * I // 32), me.scales.a[:O * I // 32])
                    for T in (1, 3, 40):
                        x = rng.standard_normal((T, I)).astype(np.float32)
                        ln = (1 + 0.1 * rng.standard_normal(I)).astype(np.float32)
                        for norm in (False, True):
                            out = []
                            for dev, m in ((r, mg), (e, me)):
                                X, Y = dev.alloc(T * I), dev.alloc(T * O)
                                dev.put(X, x.reshape(-1))
                                L = dev.vec(ln) if norm else None
                                dev.begin()
                                dev.linear(m, X, Y, T, ln=L)
                                out.append(dev.get(Y, T * O))
                            np.testing.assert_allclose(out[0], out[1], rtol=2e-3, atol=2e-3 * np.abs(out[1]).max(),
                                                       err_msg=f"{kind} {O}x{I} T={T} norm={norm}")
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
