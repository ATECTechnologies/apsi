"""The processor path of hub models: int8 matrices (Q8_0 layout) and the native thread team."""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from apsi import hub_cpu


def bits(a):
    return (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


@unittest.skipUnless(hub_cpu.available(), "native compiler not available")
class Kernels(unittest.TestCase):
    def test_products_match_numpy_at_odd_sizes(self):
        r = np.random.default_rng(0)
        for O, I, T in [(1, 32, 1), (17, 64, 1), (1000, 96, 3), (4864, 896, 1), (130, 4864, 2), (64, 128, 30)]:
            w = bits(r.standard_normal((O, I)) * 0.05)
            x = r.standard_normal((T, I)).astype(np.float32)
            dense = (w.astype(np.uint32) << np.uint32(16)).view(np.float32)
            ref = x @ dense.T
            np.testing.assert_allclose(hub_cpu.matmul(x, w), ref, rtol=1e-4, atol=1e-4)
            q = hub_cpu.Q8.from_bf16(w)
            self.assertEqual(q.codes.dtype, np.int8)
            np.testing.assert_allclose(hub_cpu.matmul(x, q), x @ q.dense().T, rtol=1e-4, atol=1e-4)
            # int8 error stays at the level of bfloat16 rounding
            self.assertLess(np.abs(hub_cpu.matmul(x, q) - ref).max(), 0.02 * np.abs(ref).max() + 1e-6)

    def test_many_calls_from_several_threads(self):
        import threading
        r = np.random.default_rng(1)
        w = hub_cpu.Q8.from_bf16(bits(r.standard_normal((512, 256))))
        x = r.standard_normal((1, 256)).astype(np.float32)
        want = x @ w.dense().T
        bad = []

        def go():
            for _ in range(200):
                if not np.allclose(hub_cpu.matmul(x, w), want, rtol=1e-4, atol=1e-4):
                    bad.append(1)
        ts = [threading.Thread(target=go) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(bad, [])


@unittest.skipUnless(hub_cpu.available(), "native compiler not available")
class Int8Model(unittest.TestCase):
    def test_int8_model_close_to_bf16_and_cached(self):
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from test_hub_models import write_safetensors
        from apsi import hub as H
        r = np.random.default_rng(2)
        V, Hd, nh, kvh, I, L = 64, 64, 4, 2, 128, 2
        W = {"model.embed_tokens.weight": r.standard_normal((V, Hd)) * 0.3, "model.norm.weight": np.ones(Hd)}
        for i in range(L):
            p = f"model.layers.{i}."
            W[p + "input_layernorm.weight"] = np.ones(Hd)
            W[p + "post_attention_layernorm.weight"] = np.ones(Hd)
            for n, o in (("q", Hd), ("k", 32), ("v", 32)):
                W[p + f"self_attn.{n}_proj.weight"] = r.standard_normal((o, Hd)) * 0.1
            W[p + "self_attn.o_proj.weight"] = r.standard_normal((Hd, Hd)) * 0.1
            W[p + "mlp.gate_proj.weight"] = r.standard_normal((I, Hd)) * 0.1
            W[p + "mlp.up_proj.weight"] = r.standard_normal((I, Hd)) * 0.1
            W[p + "mlp.down_proj.weight"] = r.standard_normal((Hd, I)) * 0.1
        W = {k: np.asarray(v, np.float32) for k, v in W.items()}
        cfg = {"model_type": "llama", "hidden_size": Hd, "num_attention_heads": nh, "num_key_value_heads": kvh,
               "num_hidden_layers": L, "intermediate_size": I, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
               "max_position_embeddings": 128, "eos_token_id": 1, "tie_word_embeddings": True,
               "hidden_act": "silu"}
        import json
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as home:
            write_safetensors(os.path.join(d, "model.safetensors"), W, "BF16")
            with open(os.path.join(d, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            ids = np.array([[3, 5, 7, 11, 2, 9]])
            with mock.patch.dict(os.environ, {"APSI_HOME": home, "APSI_CPU_WEIGHTS": "bf16"}):
                ref = H.Model(d, device="cpu")
                a = ref.net.forward(ids, ref.net.new_cache(), 0)
            outs = []
            for _ in range(2):                                  # made once, then read from the cache
                with mock.patch.dict(os.environ, {"APSI_HOME": home, "APSI_CPU_WEIGHTS": "int8"}):
                    m = H.Model(d, device="cpu")
                    outs.append(m.net.forward(ids, m.net.new_cache(), 0))
                    self.assertEqual(m.net.cpu_weights, "int8")
            cached = [f for _, _, fs in os.walk(os.path.join(home, "cache", "cpu-int8")) for f in fs]
            self.assertIn("complete", cached)
            np.testing.assert_array_equal(outs[0], outs[1])
            self.assertLess(np.abs(outs[0] - a).max(), 0.03 * np.abs(a).max())
            self.assertTrue((outs[0][0, -1].argmax() == a[0, -1].argmax()))


if __name__ == "__main__":
    unittest.main()
