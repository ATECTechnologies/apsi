"""Sealed model packages and the libraries built around them."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from apsi import layers as L, modelpack as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Cipher(unittest.TestCase):
    def test_rfc8439_vector(self):
        """RFC 8439, 2.4.2: the sunscreen test vector, both the C path (when installed) and the NumPy path."""
        key, nonce = bytes(range(32)), bytes.fromhex("000000000000004a00000000")
        pt = (b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, "
              b"sunscreen would be it.")
        want = ("6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0bf91b65c5524733ab8f593dabcd62b3571639"
                "d624e65152ab8f530c359f0861d807ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab77937365af90bbf"
                "74a35be6b40b8eedf2785e42874d")
        self.assertEqual(P.chacha20(key, nonce, pt, 1).hex(), want)
        blocks = P._chacha_blocks(key, nonce, 1, 2)
        ks = np.frombuffer(blocks, np.uint8)[:len(pt)]
        self.assertEqual((np.frombuffer(pt, np.uint8) ^ ks).tobytes().hex(), want)


class Packages(unittest.TestCase):
    def models(self):
        rng = np.random.default_rng(0)
        mlp = L.Sequential([L.Linear(6, 16), L.Activation("gelu"), L.LayerNorm(16), L.Linear(16, 3)])
        lm = L.transformer_lm(50, d=32, heads=4, layers=2, context=16)
        return [(mlp, rng.normal(size=(5, 6))), (lm, rng.integers(0, 50, (2, 16)))]

    def test_round_trip_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            for i, (m, x) in enumerate(self.models()):
                p = m.init(seed=i)
                want = np.asarray(L.predict(m, p, x))
                path = os.path.join(d, f"m{i}.apsipkg")
                P.pack_model(path, m, p, password="correct horse", metadata={"task": "test"}, public={"name": "m"})
                info = P.package_info(path)
                self.assertTrue(info.password)
                self.assertEqual(info.public["name"], "m")
                got = P.load_package(path, password="correct horse")
                np.testing.assert_allclose(np.asarray(got.predict(x)), want, rtol=1e-6, atol=1e-7)
                self.assertEqual(got.metadata["task"], "test")
                with self.assertRaises(P.PackageError):
                    P.load_package(path, password="wrong")
                raw = bytearray(open(path, "rb").read())
                raw[len(raw) // 2] ^= 1
                bad = os.path.join(d, "bad.apsipkg")
                open(bad, "wb").write(bytes(raw))
                with self.assertRaises(P.PackageError):
                    P.load_package(bad, password="correct horse")
                with open(path, "rb") as f:
                    blob = f.read()
                self.assertNotIn(b"linear", blob.lower())               # the architecture is not readable
                self.assertNotIn(np.asarray(p.layers[0].W if hasattr(p, "layers") else 0.0).tobytes()[:16] or b"x",
                                 blob)

    def test_key_and_library(self):
        m, x = self.models()[0]
        p = m.init(seed=3)
        want = np.asarray(L.predict(m, p, x))
        key = P.new_key()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.apsipkg")
            P.pack_model(path, m, p, key=key)
            with self.assertRaises(P.PackageError):
                P.load_package(path)
            np.testing.assert_allclose(np.asarray(P.load_package(path, key=key).predict(x)), want, rtol=1e-6)
            out = os.path.join(d, "lib")
            P.build_library(path, out, "mymodel")
            np.save(os.path.join(d, "x.npy"), x)
            code = ("import numpy as np, mymodel; mymodel.load(key=%r); "
                    "print(np.asarray(mymodel.predict(np.load(%r))).tolist())" % (key, os.path.join(d, "x.npy")))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([out, ROOT]))
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=d)
            self.assertEqual(r.returncode, 0, r.stderr)
            np.testing.assert_allclose(np.array(eval(r.stdout)), want, rtol=1e-6)

    def test_own_classes_are_refused(self):
        class Mine(L.Linear):
            pass
        Mine.__module__ = "my_program"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(P.PackageError):
                P.pack_model(os.path.join(d, "x.apsipkg"), Mine(2, 2), Mine(2, 2).init(), password="x")


if __name__ == "__main__":
    unittest.main()
