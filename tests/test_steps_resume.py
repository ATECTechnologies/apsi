"""Training by steps, checkpoints, exact resume after a stop or a crash, and the guard against non-finite steps:
the GPU trainer (on the Emulator) and the NumPy backend."""
import os
import tempfile
import unittest

import numpy as np

from apsi import layers as L, gpu_train as GT
from apsi.stdlib import Adam
from apsi.values import tree_leaves


def same(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(tree_leaves(a), tree_leaves(b)))


def finite(p):
    return all(np.all(np.isfinite(np.asarray(x))) for x in tree_leaves(p) if isinstance(x, np.ndarray))


class OnTheCard(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.lm = L.transformer_lm(30, d=32, heads=4, layers=1, context=8)
        self.bs = [(rng.integers(0, 30, (4, 8)), rng.integers(0, 30, (4, 8))) for _ in range(6)]
        self.ck = os.path.join(tempfile.mkdtemp(), "run.npz")

    def run_(self, ops, **kw):
        return GT.fit_batches(self.lm, self.lm.init(seed=0), self.bs, "cross_entropy", opt=Adam(1e-2), ops=ops,
                              verbose=False, precision="fp32", **kw)

    def test_steps_and_exact_resume(self):
        for acc in (1, 2):
            full = self.run_(GT.Emulator(), steps=10 // acc, accumulate=acc)
            if os.path.exists(self.ck):
                os.remove(self.ck)
            self.run_(GT.Emulator(), steps=4 // acc, accumulate=acc, checkpoint=self.ck)
            back = self.run_(GT.Emulator(), steps=10 // acc, accumulate=acc, checkpoint=self.ck)
            self.assertTrue(same(full.params, back.params), acc)

    def test_a_crash_keeps_what_was_learned(self):
        class Crash(GT.Emulator):
            n = 0

            def ce(self, *a):
                Crash.n += 1
                if Crash.n == 8:
                    raise RuntimeError("card lost")
                return super().ce(*a)
        full = self.run_(GT.Emulator(), steps=12, accumulate=2)
        with self.assertRaisesRegex(RuntimeError, "card lost"):
            self.run_(Crash(), steps=12, accumulate=2, checkpoint=self.ck)
        self.assertTrue(os.path.exists(self.ck))
        back = self.run_(GT.Emulator(), steps=12, accumulate=2, checkpoint=self.ck)
        self.assertTrue(same(full.params, back.params))

    def test_save_every_and_callback(self):
        seen = []
        self.run_(GT.Emulator(), steps=9, checkpoint=self.ck, save_every=3, callback=seen.append)
        self.assertEqual(seen[-1].step, 9)
        with np.load(self.ck) as z:
            self.assertEqual(int(z["meta"][2]), 9)

    def test_a_non_finite_batch_is_skipped(self):
        class Nan(GT.Emulator):
            k = 0

            def ce(self, L_, LAB, DL, LOSS, M, V, wg, wl):
                super().ce(L_, LAB, DL, LOSS, M, V, wg, wl)
                Nan.k += 1
                if Nan.k == 3:
                    self.v(DL, M * V)[0] = np.nan
        r = self.run_(Nan(), epochs=2)
        self.assertTrue(finite(r.params))
        self.assertLess(r.history[-1], r.history[0])

    def test_another_model_is_refused(self):
        self.run_(GT.Emulator(), steps=2, checkpoint=self.ck)
        other = L.transformer_lm(30, d=16, heads=4, layers=1, context=8)
        with self.assertRaises(GT.TrainError):
            GT.fit_batches(other, other.init(seed=0), self.bs, "cross_entropy", ops=GT.Emulator(), steps=2,
                           checkpoint=self.ck, verbose=False)


class OnNumPy(unittest.TestCase):
    def test_steps_resume_and_guard(self):
        rng = np.random.default_rng(1)
        m = L.mlp([4, 16, 3])
        bs = [(rng.normal(size=(8, 4)), rng.integers(0, 3, 8)) for _ in range(5)]
        ck = os.path.join(tempfile.mkdtemp(), "np.npz")
        kw = dict(loss="cross_entropy", backend="numpy", verbose=False)
        full = L.fit_batches(m, m.init(seed=0), bs, opt=Adam(1e-2), steps=9, **kw)
        L.fit_batches(m, m.init(seed=0), bs, opt=Adam(1e-2), steps=4, checkpoint=ck, **kw)
        back = L.fit_batches(m, m.init(seed=0), bs, opt=Adam(1e-2), steps=9, checkpoint=ck, **kw)
        self.assertTrue(same(full.params, back.params))
        bad = list(bs)
        bad[2] = (np.full((8, 4), np.inf), bs[2][1])
        r = L.fit_batches(m, m.init(seed=0), bad, opt=Adam(1e-2), epochs=2, **kw)
        self.assertTrue(finite(r.params))


if __name__ == "__main__":
    unittest.main()
