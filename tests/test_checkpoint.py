"""save_checkpoint / load_checkpoint: weights, optimizer, step, config and files in one file or in shards."""
import os
import tempfile
import unittest

import numpy as np

from apsi import modelio
from apsi.stdlib import AdamW
from apsi.values import Record


def tree(seed):
    r = np.random.default_rng(seed)
    return Record(w=r.standard_normal((32, 48)).astype(np.float32),
                  layers=[Record(b=r.standard_normal(48).astype(np.float32)) for _ in range(3)])


class Checkpoint(unittest.TestCase):
    def roundtrip(self, shard):
        with tempfile.TemporaryDirectory() as d:
            p, g = tree(0), tree(1)
            opt = AdamW(1e-2)
            p = opt.step(p, g)
            tok = os.path.join(d, "tokenizer.json")
            with open(tok, "w") as fh:
                fh.write('{"model": "x"}')
            out = modelio.save_checkpoint(os.path.join(d, "ck"), p, opt=opt, step=7,
                                          config={"d": 48, "name": "t"}, files={"tokenizer.json": tok,
                                                                               "notes": "hello"}, shard=shard)
            back = modelio.load_checkpoint(out)
            self.assertEqual(back.step, 7)
            self.assertEqual(back.config.d, 48)
            self.assertEqual(back.files["tokenizer.json"], b'{"model": "x"}')
            self.assertEqual(back.files["notes"], b"hello")
            np.testing.assert_array_equal(back.params.w, p.w)
            self.assertEqual(len(back.params.layers), 3)
            # resuming gives exactly the step the original optimizer would have taken
            a = back.optimizer.step(back.params, g)
            b = opt.step(p, g)
            np.testing.assert_allclose(a.w, b.w, rtol=0, atol=0)
            return out, d

    def test_one_file(self):
        out, _ = self.roundtrip("4GB")
        self.assertTrue(out.endswith(".apsim"))

    def test_shards_load_from_any_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = tree(2)
            idx = modelio.save_checkpoint(os.path.join(d, "big"), p, step=1, shard="2KB")
            self.assertTrue(idx.endswith(".index.json"))
            shards = sorted(f for f in os.listdir(d) if f.endswith(".apsim"))
            self.assertGreaterEqual(len(shards), 2)
            for name in (idx, os.path.join(d, "big"), os.path.join(d, shards[1])):
                np.testing.assert_array_equal(modelio.load_checkpoint(name).params.w, p.w)

    def test_plain_model_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = tree(3)
            modelio.save_model(os.path.join(d, "m"), p)
            back = modelio.load_checkpoint(os.path.join(d, "m"))
            self.assertIsNone(back.step)
            np.testing.assert_array_equal(back.params.w, p.w)


if __name__ == "__main__":
    unittest.main()
