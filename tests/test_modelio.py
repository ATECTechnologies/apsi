"""Model files: GGUF (checked against the reference `gguf` package when installed) and Apsi checkpoints."""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter, modelio as M                     # noqa: E402
from apsi.errors import ApsiError                               # noqa: E402
from apsi.models import load_safetensors                        # noqa: E402
from apsi.precision import QTensor, quantize, dequantize        # noqa: E402
from apsi.units import Quantity, D                              # noqa: E402
from apsi.values import Record                                  # noqa: E402

try:
    import gguf
except ImportError:
    gguf = None


def tree():
    rng = np.random.default_rng(0)
    return Record(layers=[Record(W=rng.standard_normal((64, 96)).astype(np.float32), b=rng.standard_normal(96)),
                          Record(W=rng.standard_normal((96, 32)), b=np.zeros(32))],
                  emb=rng.standard_normal((10, 33)), steps=np.arange(5, dtype=np.int32),
                  mass=Quantity(np.array([1.0, 2.0]), D(kg=1)))


class ModelFiles(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="apsi-model-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def p(self, name):
        return os.path.join(self.d, name)

    # ------------------------------------------------------------------ GGUF
    def test_gguf_round_trip(self):
        t = tree()
        for dt, tol in [("f32", 0), ("f16", 1e-3), ("bf16", 8e-3), ("q8_0", 1.5e-2), ("q4_0", 0.25)]:
            path = M.save_gguf(self.p(f"m_{dt}.gguf"), t, metadata={"llama.context_length": 4096, "tags": ["a"]},
                               dtype=dt)
            r = M.load_gguf(path)
            W = t.layers[0].W
            err = np.max(np.abs(r.params.layers[0].W - W)) / np.max(np.abs(W))
            self.assertLessEqual(err, tol, dt)
            self.assertTrue(np.array_equal(r.params.steps, t.steps))
            self.assertEqual(r.params.mass.dim, D(kg=1))
            self.assertEqual(r.metadata.llama_context_length, 4096)
        info = M.gguf_info(self.p("m_q4_0.gguf"))
        self.assertEqual(info.version, 3)
        self.assertEqual(info.tensors.layers_0_W.type, "Q4_0")
        self.assertEqual(info.tensors.emb.type, "F16")          # 33 columns: not a multiple of 32

    @unittest.skipIf(gguf is None, "the reference gguf package is not installed")
    def test_gguf_matches_reference_implementation(self):
        from gguf.quants import quantize as gq, dequantize as gdq
        rng = np.random.default_rng(3)
        for _ in range(30):
            a = (rng.standard_normal((8, 128)) * rng.choice([1e-4, 1, 1e3])).astype(np.float32)
            for f, qt in ((M._q8_0, gguf.GGMLQuantizationType.Q8_0), (M._q4_0, gguf.GGMLQuantizationType.Q4_0)):
                self.assertTrue(np.array_equal(np.frombuffer(f(a), np.uint8),
                                               np.asarray(gq(a, qt)).reshape(-1).view(np.uint8)))
        path = M.save_gguf(self.p("ref.gguf"), tree(), dtype="q8_0")
        r = gguf.GGUFReader(path)
        ours = M.load_gguf(path, nested=False).params
        for t in r.tensors:
            ref = np.asarray(t.data) if t.tensor_type.name.startswith("I") else gdq(t.data, t.tensor_type)
            o = ours[t.name.replace(".", "_")]
            o = o.v if isinstance(o, Quantity) else o
            self.assertTrue(np.array_equal(np.asarray(o, np.float64).reshape(-1),
                                           np.asarray(ref, np.float64).reshape(-1)), t.name)

    # ------------------------------------------------------------------ Apsi checkpoints
    def test_apsim_round_trip_units_and_compression(self):
        t = tree()
        t.fields["q8"] = quantize(np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32), "int8",
                                  axis=0)
        t.fields["q4"] = quantize(np.random.default_rng(2).standard_normal((32, 16)).astype(np.float32), "int4",
                                  axis=0)
        t.fields["f8"] = quantize(np.random.default_rng(3).standard_normal((32, 16)).astype(np.float32), "fp8_e4m3")
        path = M.save_model(self.p("m.apsim"), t, metadata={"arch": "mlp", "layers": 2})
        r = M.load_model(path)
        self.assertTrue(np.array_equal(r.layers[0].W, t.layers[0].W))
        self.assertTrue(np.array_equal(r.steps, t.steps))
        self.assertEqual(r.mass.dim, D(kg=1))
        for k in ("q8", "q4", "f8"):
            self.assertIsInstance(r[k], QTensor)
            self.assertTrue(np.array_equal(dequantize(r[k]), dequantize(t[k])), k)     # stored at real size
        info = M.model_info(path)
        self.assertEqual((info.format, info.compressed, info.units), ("apsi-model", 3, 1))
        self.assertEqual(info.metadata.arch, "mlp")
        plain = load_safetensors(path)                       # a plain safetensors reader opens it too
        self.assertTrue(np.array_equal(plain.layers[0].W, t.layers[0].W))

    def test_apsim_sharding(self):
        t = tree()
        idx = M.save_model(self.p("big.apsim"), t, shard="20KB")
        self.assertTrue(idx.endswith(".index.json"))
        shards = [f for f in os.listdir(self.d) if f.startswith("big-0")]
        self.assertGreater(len(shards), 1)
        r = M.load_model(idx)
        self.assertTrue(np.array_equal(r.layers[1].W, t.layers[1].W))
        self.assertTrue(np.array_equal(r.emb, t.emb))
        self.assertEqual(M.model_info(idx).files, len(shards))

    def test_apsim_detects_damage_and_checks_signatures(self):
        path = M.save_model(self.p("s.apsim"), tree(), sign="team-secret")
        self.assertIsNotNone(M.load_model(path, key="team-secret"))
        with self.assertRaises(ApsiError):
            M.load_model(path, key="other-secret")
        unsigned = M.save_model(self.p("u.apsim"), tree())
        with self.assertRaises(ApsiError):
            M.load_model(unsigned, key="team-secret")
        with open(path, "r+b") as fh:                         # flip one byte of tensor data
            fh.seek(-5, 2)
            b = fh.read(1)
            fh.seek(-5, 2)
            fh.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(ApsiError):
            M.load_model(path)
        idx = M.save_model(self.p("sh.apsim"), tree(), shard="20KB", sign=True)
        self.assertIsNotNone(M.load_model(idx, key=True))

    def test_apsim_protects_layout_and_index(self):
        path = M.save_model(self.p("l.apsim"), tree())
        raw = open(path, "rb").read()
        swapped = raw.replace(b'"shape":[64,96]', b'"shape":[96,64]', 1)    # same bytes, different meaning
        self.assertNotEqual(raw, swapped)
        with open(path, "wb") as fh:
            fh.write(swapped)
        with self.assertRaises(ApsiError):
            M.load_model(path)
        idx = M.save_model(self.p("sx.apsim"), tree(), shard="20KB", sign="k")
        text = open(idx).read()
        self.assertIn('\\"0\\", \\"1\\"', text)
        with open(idx, "w") as fh:
            fh.write(text.replace('\\"0\\", \\"1\\"', '\\"1\\", \\"0\\"', 1))      # kg -> m
        with self.assertRaises(ApsiError):
            M.load_model(idx, key="k")
        with open(idx, "w") as fh:
            fh.write("{not json")
        with self.assertRaises(ApsiError):
            M.load_model(idx)

    def test_apsim_rejects_malformed_files(self):
        path = M.save_model(self.p("m.apsim"), tree())
        raw = open(path, "rb").read()
        bad = [b"", raw[:5], (10 ** 12).to_bytes(8, "little") + raw[8:], raw[:8] + b"[" + raw[9:]]
        n = int.from_bytes(raw[:8], "little")
        header = raw[8:8 + n].replace(b'"data_offsets":[0,', b'"data_offsets":[9999999999,', 1)
        bad.append(len(header).to_bytes(8, "little") + header + raw[8 + n:])
        for i, b in enumerate(bad):
            with open(self.p(f"bad{i}.apsim"), "wb") as fh:
                fh.write(b)
            with self.assertRaises(ApsiError, msg=i):
                M.load_model(self.p(f"bad{i}.apsim"), verify=False)
        rng = np.random.default_rng(11)
        for i in range(300):                      # random damage anywhere in the header
            b = bytearray(raw)
            for _ in range(3):
                b[int(rng.integers(0, n + 8))] = int(rng.integers(32, 127))
            with open(self.p("fz.apsim"), "wb") as fh:
                fh.write(b)
            for verify in (True, False):
                try:
                    M.load_model(self.p("fz.apsim"), verify=verify)
                except ApsiError:
                    pass

    def test_gguf_rejects_malformed_files(self):
        path = M.save_gguf(self.p("m.gguf"), tree(), dtype="q8_0")
        raw = open(path, "rb").read()
        rng = np.random.default_rng(7)
        cases = [b"", raw[:30], raw[:len(raw) // 2], raw[:8] + (2 ** 60).to_bytes(8, "little") + raw[16:]]
        for _ in range(200):                      # random byte damage in the header
            b = bytearray(raw)
            for _ in range(4):
                b[int(rng.integers(4, 600))] = int(rng.integers(0, 256))
            cases.append(bytes(b))
        for i, b in enumerate(cases):
            with open(self.p("bad.gguf"), "wb") as fh:
                fh.write(b)
            try:
                M.load_gguf(self.p("bad.gguf"))
            except ApsiError:
                pass                              # a clear error; anything else (crash, IndexError …) fails

    def test_command_line_convert_info_verify(self):
        import subprocess
        env = dict(os.environ, PYTHONPATH=ROOT)
        run = lambda *a: subprocess.run([sys.executable, "-m", "apsi", "model", *a], capture_output=True, text=True,
                                        env=env, cwd=self.d)
        M.save_model(self.p("t.apsim"), tree())
        r = run("convert", "t.apsim", "t.gguf", "--dtype", "q8_0")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Q8_0", run("info", "t.gguf").stdout)
        self.assertEqual(run("convert", "t.gguf", "s.apsim", "--key", "k").returncode, 0)
        self.assertEqual(run("verify", "s.apsim", "--key", "k").returncode, 0)
        self.assertNotEqual(run("verify", "s.apsim", "--key", "x").returncode, 0)
        self.assertEqual(run("convert", "s.apsim", "s.safetensors").returncode, 0)
        self.assertTrue(np.allclose(load_safetensors(self.p("s.safetensors")).emb, tree().emb, atol=0.05))

    def test_permissions_apply_in_programs(self):
        src = f'use models\nsave_model("{self.p("x.apsim")}", record(w=ones(3)))\n'
        with self.assertRaises(ApsiError):
            Interpreter(permissions=set()).run(src, "<t>")


if __name__ == "__main__":
    unittest.main()
