"""Training modes on every backend: any optimizer on the GPU trainer (adam, adamw and sgd with kernels of their own,
the others on the processor), frozen parts (freeze= / train_only=), losses the card has no kernel for, and fit() on
many cards for what its kernels do not do - each the same training as NumPy."""
import unittest

import numpy as np

from apsi import layers as L, gpu_train as GT, gla as GL
from apsi.stdlib import Adam, AdamW, RMSprop, SGD


def data(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(64, 8)), rng.integers(0, 3, 64)


def mlp():
    return L.Sequential([L.Linear(8, 32), L.Activation("tanh"), L.Linear(32, 32), L.Activation("gelu"),
                         L.Linear(32, 3)])


class Modes(unittest.TestCase):
    def test_any_optimizer_and_frozen_parts_with_fit(self):
        X, y = data()
        m = mlp()
        for opt, kw in ((lambda: Adam(0.01), dict(freeze="[0]")),
                        (lambda: AdamW(0.01, weight_decay=0.1), dict(train_only="[4]", clip=0.5)),
                        (lambda: RMSprop(0.003), {}), (lambda: SGD(0.05, momentum=0.9), dict(freeze=["[2].b"]))):
            a = L.fit(m, m.init(seed=0), X, y, "cross_entropy", opt=opt(), epochs=4, batch_size=16, verbose=False,
                      precision="fp64", **kw)
            b = GT.fit(m, m.init(seed=0), X, y, "cross_entropy", opt=opt(), epochs=4, batch_size=16, verbose=False,
                       precision="fp32", ops=GT.Emulator(), **kw)
            np.testing.assert_allclose(a.history, b.history, rtol=1e-5)
            p0 = dict(L.param_paths(m.init(seed=0)))
            for path, w in L.param_paths(b.params):
                frozen = L.frozen_paths(m.init(seed=0), kw.get("freeze"), kw.get("train_only"))
                if path in frozen:
                    self.assertLess(float(np.abs(np.asarray(w) - p0[path]).max()), 1e-6)

    def test_fit_on_many_cards_for_what_its_kernels_do_not_do(self):
        X, y = data(1)
        m = mlp()
        a = L.fit(m, m.init(seed=0), X, y, "cross_entropy", opt=RMSprop(0.003), epochs=3, batch_size=16,
                  verbose=False, precision="fp64", freeze="[2]")
        b = GT.fit(m, m.init(seed=0), X, y, "cross_entropy", opt=RMSprop(0.003), epochs=3, batch_size=16,
                   verbose=False, precision="fp32", ops=[GT.Emulator() for _ in range(2)], freeze="[2]")
        np.testing.assert_allclose(a.history, b.history, rtol=1e-5)

    def test_a_loss_the_card_has_no_kernel_for(self):
        rng = np.random.default_rng(2)
        X, y = rng.normal(size=(32, 8)), rng.normal(size=(32, 3))
        m = mlp()
        for loss in ("mae", lambda p, t: L.ops.mean(L.ops.abs_(p - t) ** 3)):
            a = L.fit(m, m.init(seed=0), X, y, loss, opt=Adam(0.01), epochs=3, verbose=False, precision="fp64")
            b = GT.fit(m, m.init(seed=0), X, y, loss, opt=Adam(0.01), epochs=3, verbose=False, precision="fp32",
                       ops=GT.Emulator())
            np.testing.assert_allclose(a.history, b.history, rtol=1e-5)

    def test_phases_of_a_graph_and_language_model_across_cards(self):
        """GLA's phases: one brain and the writer learn while the rest stays; any optimizer; split across cards."""
        m = GL.gla("tiny")
        batches = [GL.gla_batch(GL.gla_task("tiny", int(b), seed=s)) for s, b in enumerate([3, 2, 4, 3, 2, 3, 2, 4])]
        for opt, kw in ((lambda: Adam(0.003), dict(freeze=["brain_a_*", "enc"])),
                        (lambda: SGD(0.05, momentum=0.9), dict(train_only="w*", clip=0.5))):
            a = L.fit_batches(m, m.init(seed=0), batches, GL.gla_loss, opt=opt(), epochs=2, backend="numpy",
                              verbose=False, accumulate=4, **kw)
            for n, st in ((1, None), (3, None), (4, 2)):
                ops = GT.Emulator() if n == 1 else [GT.Emulator() for _ in range(n)]
                b = GT.fit_batches(m, m.init(seed=0), batches, GL.gla_loss, opt=opt(), epochs=2, verbose=False,
                                   precision="fp32", ops=ops, accumulate=4, stages=st, **kw)
                np.testing.assert_allclose(a.history, b.history, rtol=1e-5)
            moved = {p for p, w in L.param_paths(b.params)
                     if float(np.abs(np.asarray(w) - dict(L.param_paths(m.init(seed=0)))[p]).max()) > 1e-6}
            frozen = L.frozen_paths(m.init(seed=0), kw.get("freeze"), kw.get("train_only"))
            self.assertFalse(moved & frozen)
            self.assertTrue(moved)

    def test_variants_without_a_kernel_run_on_the_processor_inside_the_model(self):
        rng = np.random.default_rng(0)
        N = 30
        g = L.graph(edges=rng.integers(0, N, (60, 2)).tolist(), n=N, batch=np.repeat([0, 1, 2], 10))
        X, Yn, Yg = rng.normal(size=(N, 8)), rng.integers(0, 3, N), rng.integers(0, 3, 3)
        for m, Y in ((L.Sequential([L.Linear(8, 16), L.Activation("mish"), L.Linear(16, 3)]), Yn),
                     (L.Sequential([L.Linear(8, 16), L.GraphPool("max"), L.Linear(16, 3)]), Yg)):
            a = L.fit_batches(m, m.init(seed=0), [(X, Y, g)], "cross_entropy", opt=Adam(0.01), epochs=3,
                              backend="numpy", verbose=False)
            b = GT.fit_batches(m, m.init(seed=0), [(X, Y, g)], "cross_entropy", opt=Adam(0.01), epochs=3,
                               verbose=False, precision="fp32", ops=GT.Emulator())
            np.testing.assert_allclose(a.history, b.history, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
