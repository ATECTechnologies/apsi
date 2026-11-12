"""Apsi's own GPU trainer: gradients against Apsi's autodiff (on the Emulator), training curves against the NumPy
backend, and - when there is an NVIDIA card - every kernel and a whole training run against the Emulator."""
import unittest

import numpy as np

from apsi import layers as L, gpu_train as GT, ops as aops
from apsi.stdlib import Adam, SGD, adamw, value_and_grad
from apsi.values import tree_leaves


def grads_match(tc, model, X, Y, loss, prec="fp32"):
    p = model.init(seed=0)
    lf = L._loss_fn(loss)
    _, g = value_and_grad(lambda q: lf(model(q, X, train=True), Y))(p)
    em = GT.Emulator(tc=True)
    net = GT.Net(em, model, p, prec)
    ids = isinstance(model.layers[0], L.Embedding)
    Xa = GT._inputs(X, ids)
    Xd = net.arena.get("X", Xa.size)
    em.put(Xd[0], Xa.reshape(-1))
    Ya = np.asarray(Y).astype(np.int32 if loss == "cross_entropy" else np.float32)
    Yd = net.arena.get("Y", Ya.size)
    em.put(Yd[0], Ya.reshape(-1))
    em.fill((net.G, 0), net.n, 0)
    out, shape = net.forward(Xd, Xa.shape)
    rows = GT._rows(shape)
    d = net.arena.get("d", rows * shape[-1])
    if loss == "cross_entropy":
        w = 1 / (Ya >= 0).sum()
        em.ce(out, Yd, d, (net.LOSS, 0), rows, shape[-1], w, w)
    else:
        m = rows * shape[-1]
        em.mse(out, Yd, d, (net.LOSS, 0), m, 1 / m, 1 / m)
    net.backward(d)
    G = em.get(net.G, net.n)
    for a, ga in zip(tree_leaves(p), tree_leaves(g)):
        o = net.offsets[id(a)]
        ref = np.asarray(aops.raw(ga))
        tc.assertLess(np.abs(G[o:o + a.size].reshape(a.shape) - ref).max() / (np.abs(ref).max() + 1e-12), 1e-5)


class Trainer(unittest.TestCase):
    def test_gradients_are_exact(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(16, 10))
        grads_match(self, L.mlp([10, 32, 3], "gelu"), X, rng.integers(0, 3, 16), "cross_entropy")
        grads_match(self, L.Sequential([L.Linear(10, 16), L.LayerNorm(16), L.Activation("silu"), L.Linear(16, 4),
                                        L.RMSNorm(4)]), X, rng.normal(size=(16, 4)), "mse")
        grads_match(self, L.resnet_mlp(10, 3, width=16, blocks=2), X, rng.normal(size=(16, 3)), "mse")
        lab = rng.integers(0, 23, (3, 12))
        lab[0, :3] = -100
        grads_match(self, L.transformer_lm(23, d=32, heads=4, layers=2, context=12), rng.integers(0, 23, (3, 12)),
                    lab, "cross_entropy")

    def test_modern_blocks_are_exact(self):
        """Grouped-query attention, rotary positions, per-head query / key norms, RMS norm and SwiGLU: the gradients
        on the GPU trainer equal Apsi's autodiff; training equals the NumPy backend, fused attention or not, on one
        card or split across three."""
        rng = np.random.default_rng(0)
        for kw in (dict(kv_heads=2), dict(rope=True), dict(qk_norm=True), dict(norm="rms", ffn="swiglu"),
                   dict(kv_heads=1, rope=True, qk_norm=True, norm="rms", ffn="swiglu"),
                   dict(kv_heads=2, rope=True, qk_norm=True, ffn="swiglu", activation="relu")):
            lab = rng.integers(0, 23, (3, 12))
            lab[0, :3] = -100
            grads_match(self, L.transformer_lm(23, d=32, heads=4, layers=2, context=12, **kw),
                        rng.integers(0, 23, (3, 12)), lab, "cross_entropy")
        grads_match(self, L.Sequential([L.Linear(10, 16), L.SwiGLU(16, 24, d_out=5)]), rng.normal(size=(16, 10)),
                    rng.normal(size=(16, 5)), "mse")
        lm = L.transformer_lm(30, d=128, heads=4, layers=2, context=40, kv_heads=2, rope=True, qk_norm=True,
                              norm="rms", ffn="swiglu")
        ids = rng.integers(0, 30, (24, 40))
        lab = np.roll(ids, -1, 1)
        kw = dict(loss="cross_entropy", opt=None, epochs=3, batch_size=8, verbose=False)
        ref = L.fit(lm, lm.init(seed=0), ids, lab, precision="fp32", **dict(kw, opt=Adam(3e-3))).history
        for ops in (GT.Emulator(), [GT.Emulator() for _ in range(3)]):
            r = GT.fit(lm, lm.init(seed=0), ids, lab, precision="fp32", ops=ops, parallel="pipeline",
                       **dict(kw, opt=Adam(3e-3))).history
            np.testing.assert_allclose(ref, r, rtol=1e-5)
        e = GT.Emulator()
        e.flash = True                                              # the fused kernel's path (bfloat16 inputs)
        r = GT.fit(lm, lm.init(seed=0), ids, lab, ops=e, **dict(kw, opt=Adam(3e-3))).history
        np.testing.assert_allclose(ref, r, rtol=3e-3)

    def test_graph_networks_are_exact(self):
        """Sparse graphs on the GPU trainer (GCN, GraphSAGE, GIN, GAT, pooling per graph; CSR kernels with no
        atomics): node and graph tasks, masks, validation and predict equal the NumPy backend."""
        rng = np.random.default_rng(0)
        N = 40
        e = rng.integers(0, N, (120, 2)).tolist()
        g = L.graph(edges=e, n=N)
        X, y, yr = rng.normal(size=(N, 8)), rng.integers(0, 3, N), rng.normal(size=(N, 2))
        mask = rng.random(N) < 0.5
        cases = (("gat", L.gat_net(8, 8, 3, dropout=0.0), "cross_entropy", y),
                 ("gcn", L.gcn_net(8, 16, 3, dropout=0.0), "cross_entropy", y),
                 ("sage", L.Sequential([L.SAGE(8, 16), L.Activation("mish"), L.SAGE(16, 2, "sum")]), "mse", yr),
                 ("gin", L.Sequential([L.GIN(8, 16), L.Activation("elu"), L.GIN(16, 3)]), "cross_entropy", y),
                 ("gat mean", L.Sequential([L.GAT(8, 4, heads=3), L.Activation("softplus"),
                                            L.GAT(12, 2, heads=2, concat=False)]), "mse", yr))
        for name, m, loss, yy in cases:
            kw = dict(loss=loss, epochs=4, verbose=False, precision="fp32", graph=g, mask=mask, val=(X, yy))
            a = L.fit(m, m.init(seed=0), X, yy, opt=Adam(0.01), **kw)
            b = GT.fit(m, m.init(seed=0), X, yy, opt=Adam(0.01), ops=GT.Emulator(), **kw)
            np.testing.assert_allclose(a.history, b.history, rtol=1e-5, err_msg=name)
            np.testing.assert_allclose(a.val_history, b.val_history, rtol=1e-5, err_msg=name)
            np.testing.assert_allclose(np.asarray(L.predict(m, a.params, X, graph=g)),
                                       GT.predict(m, a.params, X, ops=GT.Emulator(), graph=g, precision="fp32"),
                                       rtol=1e-4, atol=1e-5, err_msg=name)
        gb = L.graph(edges=e, n=N, batch=np.repeat(np.arange(8), 5))
        for pool in ("sum", "mean"):
            m = L.gin_net(8, 16, 2, pool=pool)
            yg = rng.normal(size=(8, 2))
            a = L.fit(m, m.init(seed=0), X, yg, loss="mse", opt=Adam(0.01), epochs=4, verbose=False,
                      precision="fp32", graph=gb)
            b = GT.fit(m, m.init(seed=0), X, yg, loss="mse", opt=Adam(0.01), epochs=4, verbose=False,
                       precision="fp32", graph=gb, ops=GT.Emulator())
            np.testing.assert_allclose(a.history, b.history, rtol=1e-5)
        with self.assertRaises(GT.TrainError):
            GT.fit(L.Sequential([L.SAGE(8, 3, "max")]), L.Sequential([L.SAGE(8, 3, "max")]).init(), X, y,
                   loss="cross_entropy", epochs=1, verbose=False, graph=g, ops=GT.Emulator())

    def test_training_matches_the_numpy_backend(self):
        rng = np.random.default_rng(0)
        lm = L.transformer_lm(23, d=32, heads=4, layers=2, context=12)
        ids = rng.integers(0, 23, (40, 12))
        lab = np.roll(ids, -1, 1)
        for opt in (lambda: Adam(1e-2), lambda: adamw(lr=1e-2), lambda: SGD(0.1, momentum=0.9)):
            a = L.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=opt(), epochs=3, batch_size=8,
                      verbose=False, precision="fp32", clip=1.0)
            b = GT.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=opt(), epochs=3, batch_size=8,
                       verbose=False, precision="fp32", clip=1.0, ops=GT.Emulator())
            np.testing.assert_allclose(a.history, b.history, rtol=1e-4)

    def test_several_cards_train_like_one(self):
        """Data-parallel training on several cards (here 3 emulated) gives the result of one card with the batch."""
        from apsi.values import tree_leaves
        rng = np.random.default_rng(5)
        X, y = rng.normal(size=(50, 6)), rng.integers(0, 3, 50)
        cases = (("cross_entropy", y, 3, lambda: Adam(0.01)), ("mse", rng.normal(size=(50, 2)), 2, lambda: Adam(0.01)),
                 ("cross_entropy", y, 3, lambda: adamw(lr=0.01, weight_decay=0.1)),
                 ("cross_entropy", y, 3, lambda: SGD(0.1, momentum=0.9)))
        for lossname, yy, last, mk in cases:
            m = L.mlp([6, 16, last], "gelu")
            p0 = m.init(seed=1)
            from apsi.values import tree_map
            fisher = tree_map(lambda a: np.full_like(np.asarray(a), 0.5) if isinstance(a, np.ndarray) else a, p0)
            kw = dict(loss=lossname, epochs=3, batch_size=16, verbose=False, precision="fp32", clip=0.05,
                      ewc=(p0, fisher, 0.3))
            one = GT.fit(m, p0, X, yy, opt=mk(), ops=GT.Emulator(), **kw)
            many = GT.fit(m, p0, X, yy, opt=mk(), ops=[GT.Emulator(), GT.Emulator(), GT.Emulator()], ema=0.9, **kw)
            for a, b in zip(tree_leaves(one.params), tree_leaves(many.params)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-4, atol=2e-5)
            np.testing.assert_allclose(one.history, many.history, rtol=1e-4)
            self.assertIn("3 GPUs", many.device)
            self.assertIsNotNone(many.ema_params)

    def test_one_model_split_across_cards_trains_like_one_card(self):
        """Model parallel: the model cut into stages on several (emulated) cards, micro-batches in a pipeline, and the
        pipeline copied when there are more cards than stages: the same weights as one card with the whole batch."""
        from apsi.values import tree_map
        rng = np.random.default_rng(0)
        lm = L.transformer_lm(23, d=32, heads=4, layers=3, context=12)
        ids = rng.integers(0, 23, (40, 12))
        lab = np.roll(ids, -1, 1)
        kw = dict(loss="cross_entropy", epochs=3, batch_size=8, verbose=False, precision="fp32", clip=1.0)
        one = GT.fit(lm, lm.init(seed=0), ids, lab, opt=Adam(1e-2), ops=GT.Emulator(), **kw)
        for cards, stages, want in ((3, None, (3, 1)), (4, 2, (2, 2)), (3, [3, 4, 3], (3, 1))):
            r = GT.fit(lm, lm.init(seed=0), ids, lab, opt=Adam(1e-2), ops=[GT.Emulator() for _ in range(cards)],
                       parallel="pipeline", stages=stages, val=(ids, lab), **kw)
            self.assertEqual((len(r.stages), r.replicas), want)
            self.assertIn("split into", r.device)
            np.testing.assert_allclose(one.history, r.history, rtol=1e-5)
            for a, b in zip(tree_leaves(one.params), tree_leaves(r.params)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=2e-5)
            self.assertEqual(len(r.val_history), 3)
        rng = np.random.default_rng(5)
        X, y = rng.normal(size=(50, 6)), rng.integers(0, 3, 50)
        for lossname, yy, last, mk in (("mse", rng.normal(size=(50, 2)), 2, lambda: Adam(0.01)),
                                       ("cross_entropy", y, 3, lambda: adamw(lr=0.01, weight_decay=0.1)),
                                       ("cross_entropy", y, 3, lambda: SGD(0.1, momentum=0.9))):
            m = L.mlp([6, 16, 16, 12, last], "gelu")
            p0 = m.init(seed=1)
            fisher = tree_map(lambda a: np.full_like(np.asarray(a), 0.5) if isinstance(a, np.ndarray) else a, p0)
            kw = dict(loss=lossname, epochs=3, batch_size=16, verbose=False, precision="fp32", clip=0.05,
                      ewc=(p0, fisher, 0.3), ema=0.9)
            one = GT.fit(m, p0, X, yy, opt=mk(), ops=GT.Emulator(), **kw)
            for cards, stages, micro in ((3, None, None), (4, 2, 3), (2, None, 16)):
                r = GT.fit(m, p0, X, yy, opt=mk(), ops=[GT.Emulator() for _ in range(cards)], parallel="pipeline",
                           stages=stages, micro_batches=micro, **kw)
                for a, b in zip(tree_leaves(one.params) + tree_leaves(one.ema_params),
                                tree_leaves(r.params) + tree_leaves(r.ema_params)):
                    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-4, atol=2e-6)

    def test_split_model_details(self):
        from apsi import gpu_pipeline as GP
        self.assertEqual(GP.partition([1, 1, 1, 1, 8, 1, 1, 1, 1], [1] * 9, 3), [4, 1, 4])
        self.assertEqual(GP.partition([1] * 4, [10, 10, 10, 10], 2, 1.0, budget=30), [2, 2])
        self.assertIsNone(GP.partition([1] * 4, [10, 10, 10, 10], 2, 1.0, budget=15))
        self.assertEqual(GP.schedule(0, 3, 4), [("F", 0), ("F", 1), ("F", 2), ("B", 0), ("F", 3), ("B", 1),
                                                ("B", 2), ("B", 3)])
        self.assertEqual(GP.schedule(2, 3, 2), [("F", 0), ("B", 0), ("F", 1), ("B", 1)])
        lm = L.transformer_lm(23, d=32, heads=4, layers=2, context=12)
        self.assertEqual(len(GP.units(lm, lm.init(seed=0))), 8)    # embedding, positions, 2 x (attention, MLP), ...
        rng = np.random.default_rng(1)
        ids = rng.integers(0, 23, (16, 12))
        with self.assertRaises(GT.TrainError):
            GT.fit(lm, lm.init(seed=0), ids, ids, loss="cross_entropy", epochs=1, verbose=False,
                   ops=[GT.Emulator(), GT.Emulator()], stages=[1, 2])
        for prec in ("auto", "int4"):                               # mixed / low bit and dropout on a split model
            d = L.transformer_lm(23, d=32, heads=4, layers=2, context=12, dropout=0.1)
            r = GT.fit(d, d.init(seed=0), ids, np.roll(ids, -1, 1), loss="cross_entropy", opt=Adam(1e-2), epochs=4,
                       batch_size=8, verbose=False, precision=prec, ops=[GT.Emulator() for _ in range(3)],
                       parallel="pipeline")
            self.assertLess(r.history[-1], r.history[0] - 0.2, prec)

        class Broken(GT.Emulator):
            def ce(self, *a):
                raise RuntimeError("the last card failed")
        with self.assertRaisesRegex(RuntimeError, "the last card failed"):     # a failure stops every card at once
            GT.fit(lm, lm.init(seed=0), ids, ids, loss="cross_entropy", epochs=1, batch_size=8, verbose=False,
                   ops=[GT.Emulator(), GT.Emulator(), Broken()], parallel="pipeline")

    def test_split_model_cards_work_at_the_same_time(self):
        """With cards whose products take time (the processor free meanwhile, as with real cards), four stages
        finish a step much sooner than one card doing all of it."""
        import time

        class Card(GT.Emulator):
            def _gemm(self, A, B, Cv, bias, M, N, K, *a, **k):
                time.sleep(M * N * K * a[5] * a[6] * 2e-8)
                return super()._gemm(A, B, Cv, bias, M, N, K, *a, **k)
        rng = np.random.default_rng(0)
        lm = L.transformer_lm(23, d=32, heads=4, layers=4, context=12)
        ids = rng.integers(0, 23, (32, 12))
        kw = dict(loss="cross_entropy", epochs=1, verbose=False, opt=Adam(1e-2))
        t = time.perf_counter()
        GT.fit(lm, lm.init(seed=0), ids, np.roll(ids, -1, 1), ops=Card(), **kw)
        one = time.perf_counter() - t
        t = time.perf_counter()
        GT.fit(lm, lm.init(seed=0), ids, np.roll(ids, -1, 1), ops=[Card() for _ in range(4)], parallel="pipeline",
               micro_batches=8, **kw)
        self.assertLess(time.perf_counter() - t, one / 1.5)

    def test_ema_and_ewc(self):
        """EMA of the weights (the GLA project's teacher weights) and EWC (keep what an earlier task needed)."""
        from apsi.values import tree_leaves
        rng = np.random.default_rng(3)
        m = L.mlp([6, 16, 3], "gelu")
        X, y = rng.normal(size=(32, 6)), rng.integers(0, 3, 32)
        p0 = m.init(seed=0)
        one = GT.fit(m, p0, X, y, loss="cross_entropy", opt=SGD(0.1), epochs=1, verbose=False, ema=0.9,
                     precision="fp32", ops=GT.Emulator())                  # one full-batch step
        for e, a, b in zip(tree_leaves(one.ema_params), tree_leaves(p0), tree_leaves(one.params)):
            np.testing.assert_allclose(e, 0.9 * np.asarray(a) + 0.1 * np.asarray(b), rtol=1e-5, atol=1e-6)
        fisher = [np.ones_like(np.asarray(a)) for a in tree_leaves(p0)]
        from apsi.values import tree_map
        it = iter(fisher)
        fisher = tree_map(lambda a: next(it) if isinstance(a, np.ndarray) else a, p0)
        free = GT.fit(m, p0, X, y, loss="cross_entropy", opt=SGD(0.1), epochs=20, verbose=False, precision="fp32",
                      ops=GT.Emulator())
        held = GT.fit(m, p0, X, y, loss="cross_entropy", opt=SGD(0.1), epochs=20, verbose=False, precision="fp32",
                      ewc=(p0, fisher, 5.0), ops=GT.Emulator())
        drift = lambda r: sum(float(np.abs(np.asarray(a) - np.asarray(b)).sum())
                              for a, b in zip(tree_leaves(r.params), tree_leaves(p0)))
        self.assertLess(drift(held), 0.5 * drift(free))
        with self.assertRaises(GT.TrainError):
            GT.fit(m, p0, X, y, loss="cross_entropy", epochs=1, verbose=False, ema=1.5, ops=GT.Emulator())

    def test_mixed_and_low_bit_precision_learn(self):
        rng = np.random.default_rng(1)
        lm = L.transformer_lm(23, d=32, heads=4, layers=2, context=12)
        ids = rng.integers(0, 23, (40, 12))
        lab = np.roll(ids, -1, 1)
        for prec in ("auto", "fp8", "fp6_e2m3", "fp4", "int8", "int4"):
            r = GT.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=Adam(1e-2), epochs=4, batch_size=8,
                       verbose=False, precision=prec, ops=GT.Emulator(), val=(ids, lab))
            self.assertLess(r.history[-1], r.history[0] - 0.3, prec)
            self.assertEqual(len(r.val_history), 4)

    def test_fused_attention_trains_like_the_separate_products(self):
        rng = np.random.default_rng(7)
        lm = L.transformer_lm(30, d=64, heads=2, layers=1, context=40)            # head size 32: the fused path
        ids = rng.integers(0, 30, (24, 40))
        lab = np.roll(ids, -1, 1)
        runs = []
        for flash in (True, False):
            e = GT.Emulator()
            e.flash = flash
            runs.append(GT.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=Adam(3e-3), epochs=3,
                               batch_size=8, verbose=False, ops=e).history)
        np.testing.assert_allclose(runs[0], runs[1], rtol=2e-2)
        self.assertLess(runs[0][-1], runs[0][0])

    def test_unsupported_layers_say_so(self):
        self.assertIsNone(GT.supports(L.transformer_lm(30, d=16, heads=2, layers=1, context=8)))
        self.assertIn("conv", GT.supports(L.Sequential([L.Conv2D(1, 2)])))


@unittest.skipIf(GT.device()[0] is None, "no NVIDIA GPU")
class RealGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, _ = GT.device()

    def test_every_kernel_matches_the_emulator(self):
        g, e = self.g, GT.Emulator(tc=self.g.tc)
        rng = np.random.default_rng(2)

        def pair(a):
            a = np.ascontiguousarray(a)
            bg, be = g.alloc(a.size), e.alloc(a.size)
            g.put(bg, a.reshape(-1))
            e.put(be, a.reshape(-1))
            return (bg, 0), (be, 0)

        def same(vg, ve, n, tol, what):
            a, b = g.get(vg[0], n, vg[1]), e.get(ve[0], n, ve[1])
            err = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
            self.assertLess(err, tol, what)
        # products: plain, transposed operands, batched heads, bias, accumulate, alpha; both precisions
        M, N, K = 70, 45, 37
        A, B, Cc, bias = pair(rng.normal(size=(M, K))), pair(rng.normal(size=(K, N))), pair(rng.normal(size=(M, N))), \
            pair(rng.normal(size=N))
        At = pair(rng.normal(size=(K, M)))
        for passes in (1, 3):
            for dv, i in ((g, 0), (e, 1)):
                dv.gemm(A[i], B[i], Cc[i], M, N, K, K, 1, N, 1, N, alpha=0.5, beta=True, bias=bias[i], passes=passes)
                dv.gemm(At[i], B[i], Cc[i], M, N, K, 1, M, N, 1, N, beta=True, passes=passes)   # A transposed
            same(Cc[0], Cc[1], M * N, 2e-3 if passes == 1 else 1e-5, f"gemm passes={passes}")
        Bh, T, hd = 3, 20, 8
        D = 2 * hd
        q, kk, S = pair(rng.normal(size=(Bh, T, D))), pair(rng.normal(size=(Bh, T, D))), pair(np.zeros(Bh * 2 * T * T))
        for dv, (qq, kq, ss) in ((g, (q[0], kk[0], S[0])), (e, (q[1], kk[1], S[1]))):
            dv.gemm(qq, kq, ss, T, T, hd, D, 1, 1, D, T, alpha=0.3, batch=(Bh, 2), sa=(T * D, hd), sb=(T * D, hd),
                    sc=(2 * T * T, T * T), passes=3)
        same(S[0], S[1], Bh * 2 * T * T, 1e-5, "batched attention product")
        n = 1000
        x, y, out = pair(rng.normal(size=n)), pair(rng.normal(size=n)), pair(np.zeros(n))
        for kind in range(10):
            g.act_f(x[0], out[0], n, kind)
            e.act_f(x[1], out[1], n, kind)
            same(out[0], out[1], n, 1e-4, f"act_f {kind}")
            g.act_b(x[0], y[0], out[0], n, kind)
            e.act_b(x[1], y[1], out[1], n, kind)
            same(out[0], out[1], n, 1e-4, f"act_b {kind}")
        M, N = 33, 300
        X, Gm, Bt, Y2 = pair(rng.normal(size=(M, N))), pair(rng.normal(size=N)), pair(rng.normal(size=N)), pair(np.zeros(M * N))
        mean, rstd, dX = pair(np.zeros(M)), pair(np.zeros(M)), pair(np.zeros(M * N))
        dG, dB = pair(np.zeros(N)), pair(np.zeros(N))
        res = pair(rng.normal(size=(M, N)))
        for rms in (False, True):
            for dv, i in ((g, 0), (e, 1)):
                dv.ln_f(X[i], Gm[i], Bt[i], Y2[i], mean[i], rstd[i], M, N, 1e-5, rms)
                dv.ln_b(X[i], Gm[i], Y2[i], mean[i], rstd[i], dX[i], M, N, rms, res[i] if rms else None)
                dv.colsum(Y2[i], X[i], mean[i], rstd[i], dG[i], dB[i], M, N, 2 if rms else 1)
            for v, nn, w in ((Y2, M * N, "ln_f"), (dX, M * N, "ln_b"), (dG, N, "colsum g"), (dB, N, "colsum b")):
                same(v[0], v[1], nn, 1e-4, f"{w} rms={rms}")
        R, n2, T2 = 24, 12, 12
        Sx, P, dP, dS = pair(rng.normal(size=R * n2)), pair(np.zeros(R * n2)), pair(rng.normal(size=R * n2)), \
            pair(np.zeros(R * n2))
        for causal in (False, True):
            for dv, i in ((g, 0), (e, 1)):
                dv.softmax_f(Sx[i], P[i], R, n2, T2, causal)
                dv.softmax_b(P[i], dP[i], dS[i], R, n2)
            same(P[0], P[1], R * n2, 1e-5, "softmax_f")
            same(dS[0], dS[1], R * n2, 1e-4, "softmax_b")
        V, Mr = 50, 40
        lab = rng.integers(-1, V, Mr).astype(np.int32)
        Lg, Lb, dL, loss = pair(rng.normal(size=(Mr, V))), pair(lab), pair(np.zeros(Mr * V)), pair(np.zeros(1))
        for dv, i in ((g, 0), (e, 1)):
            dv.ce(Lg[i], Lb[i], dL[i], loss[i], Mr, V, 0.1, 0.2)
        same(dL[0], dL[1], Mr * V, 1e-5, "ce grad")
        same(loss[0], loss[1], 1, 1e-5, "ce loss")
        # rotary positions (forward and inverse), sparse graph kernels, dot / scale / column copies
        rows, hd, Hh, Tt = 2 * 12 * 3, 16, 3, 12
        Xr, CS, SN = pair(rng.normal(size=rows * hd)), pair(rng.normal(size=Tt * hd // 2)), \
            pair(rng.normal(size=Tt * hd // 2))
        for sign in (1.0, -1.0):
            for dv, i in ((g, 0), (e, 1)):
                dv.rope(Xr[i], CS[i], SN[i], rows, hd, Hh, Tt, sign)
            same(Xr[0], Xr[1], rows * hd, 1e-5, f"rope {sign}")
        Nn, Ee, Hh, Ff = 37, 150, 3, 5
        dst = np.sort(rng.integers(0, Nn, Ee))
        off = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=Nn))]).astype(np.int32)
        col = rng.integers(0, Nn, Ee).astype(np.int32)
        perm = rng.permutation(Ee).astype(np.int32)
        COL, OFF, PERM, ROW = pair(col), pair(off), pair(perm), pair(dst.astype(np.int32))
        Xg, VAL = pair(rng.normal(size=(Nn, Hh * Ff))), pair(rng.normal(size=(Ee, Hh)))
        Yg = pair(rng.normal(size=(Nn, Hh * Ff)))
        for vals, vp, add in ((False, False, False), (True, False, False), (True, True, True)):
            for dv, i in ((g, 0), (e, 1)):
                dv.spmm(Xg[i], COL[i], VAL[i] if vals else None, PERM[i] if vp else None, OFF[i], Yg[i], Nn, Hh, Ff,
                        add)
            same(Yg[0], Yg[1], Nn * Hh * Ff, 1e-5, f"spmm {vals} {vp} {add}")
        DA, Sg, Ag, DS = pair(np.zeros(Ee * Hh)), pair(rng.normal(size=Ee * Hh)), pair(np.zeros(Ee * Hh)), \
            pair(np.zeros(Ee * Hh))
        for dv, i in ((g, 0), (e, 1)):
            dv.edge_dot(Yg[i], Xg[i], ROW[i], COL[i], DA[i], Ee, Hh, Ff)
            dv.seg_softmax(Sg[i], OFF[i], Ag[i], Nn, Hh, 0.2)
            dv.seg_softmax_b(Sg[i], Ag[i], DA[i], OFF[i], DS[i], Nn, Hh, 0.2)
        same(DA[0], DA[1], Ee * Hh, 1e-5, "edge_dot")
        same(Ag[0], Ag[1], Ee * Hh, 1e-5, "seg_softmax")
        same(DS[0], DS[1], Ee * Hh, 1e-4, "seg_softmax_b")
        acc, sc1 = pair(np.zeros(1)), pair(np.array([0.3]))
        for dv, i in ((g, 0), (e, 1)):
            dv.dot_acc(x[i], y[i], n, acc[i])
            dv.scale_add(out[i], x[i], sc1[i], n, 1.0)
        same(acc[0], acc[1], 1, 1e-5, "dot_acc")
        same(out[0], out[1], n, 1e-5, "scale_add")
        wide, part = pair(rng.normal(size=(20, 30))), pair(rng.normal(size=(20, 7)))
        for back in (0, 1):
            for dv, i in ((g, 0), (e, 1)):
                dv.cols(part[i], wide[i], 20, 7, 30, 11, back)
            same(wide[0], wide[1], 600, 1e-6, "cols")
            same(part[0], part[1], 140, 1e-6, "cols back")
        W, WQ = pair(rng.normal(size=(70, 48))), pair(np.zeros(70 * 48))
        for kind in range(6):
            g.fakeq(W[0], WQ[0], 70, 48, kind)
            e.fakeq(W[1], WQ[1], 70, 48, kind)
            np.testing.assert_array_equal(g.get(WQ[0][0], 70 * 48), e.get(WQ[1][0], 70 * 48), f"fakeq {kind}")

    def test_tiled_products(self):
        """The shared-memory tensor-core products: every layout, edges, batches of heads and split k; both builds
        (the card picks the faster per shape, so both must be right)."""
        import os
        from unittest import mock
        for variant in ("a", "x", "w", "h"):
            if variant != "a" and f"gemm_tkk#{variant}" not in self.g.fn:
                continue
            with self.subTest(variant=variant), mock.patch.dict(os.environ, {"APSI_GEMM_VARIANT": variant}):
                self._tiled_body()

    def _tiled_body(self):
        g, e = self.g, GT.Emulator(tc=True)
        if not g.tiled:
            self.skipTest(f"no tiled products here: {g.tc_why}")
        rng = np.random.default_rng(5)

        def pair(a):
            a = np.ascontiguousarray(a, np.float32)
            bg, be = g.alloc(a.size), e.alloc(a.size)
            g.put(bg, a.reshape(-1))
            e.put(be, a.reshape(-1))
            return (bg, 0), (be, 0)

        def check(M, N, K, ta, tb, beta=False, bias=False, alpha=1.0, passes=1):
            A = pair(rng.normal(size=(K, M) if ta else (M, K)))
            B = pair(rng.normal(size=(N, K) if tb else (K, N)))
            Cc = pair(rng.normal(size=(M, N)))
            bs = pair(rng.normal(size=N)) if bias else (None, None)
            sam, sak = (1, M) if ta else (K, 1)
            sbk, sbn = (1, K) if tb else (N, 1)
            self.assertIsNotNone(g._tiled_ok(A[0], B[0], M, N, K, sam, sak, sbk, sbn, (0, 0), (0, 0)))
            for dv, i in ((g, 0), (e, 1)):
                dv.gemm(A[i], B[i], Cc[i], M, N, K, sam, sak, sbk, sbn, N, alpha=alpha, beta=beta, bias=bs[i],
                        passes=passes)
            a, b_ = g.get(Cc[0][0], M * N), e.get(Cc[1][0], M * N)
            err = np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9)
            self.assertLess(err, 2e-3 if passes == 1 else 5e-5, (M, N, K, ta, tb, beta, bias, passes, err))
        for ta in (False, True):
            for tb in (False, True):
                check(200, 136, 96, ta, tb)
                check(128, 128, 32, ta, tb, beta=True, bias=True, alpha=0.5)
                check(4, 4, 4, ta, tb)
                check(384, 1536, 4096, ta, tb, beta=True)          # split k
                check(256, 384, 2048, ta, tb)                      # split k into a cleared C
                check(200, 136, 96, ta, tb, beta=True, bias=True, alpha=0.5, passes=3)   # float32: 3 bfloat16 products
                check(384, 384, 4096, ta, tb, passes=3)
        # the last step fused in: activation (and its pre-activation kept), its derivative, a residual
        M, N, K = 256, 384, 128
        A, Bm, Bt = pair(rng.normal(size=(M, K))), pair(rng.normal(size=(K, N))), pair(rng.normal(size=(N, K)))
        X2, Cc = pair(rng.normal(size=(M, N))), pair(np.zeros((M, N)))
        bias = pair(rng.normal(size=N))
        for epi in (1, 2, 3):
            for act in (0, 1, 2):
                for dv, i in ((g, 0), (e, 1)):
                    if epi == 2:                       # as in backward: dy W^T, W read transposed
                        dv.gemm(A[i], Bt[i], Cc[i], M, N, K, K, 1, 1, K, N, epi=epi, X2=X2[i], act=act)
                    else:
                        dv.gemm(A[i], Bm[i], Cc[i], M, N, K, K, 1, N, 1, N, bias=bias[i], epi=epi, X2=X2[i], act=act)
                for view, what in ((Cc, "C"), (X2, "X2")):
                    a, b_ = g.get(view[0][0], M * N), e.get(view[1][0], M * N)
                    err = np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9)
                    self.assertLess(err, 3e-3, (epi, act, what, err))
                for dv, i in ((g, 0), (e, 1)):            # the same in float32 (three products, the last fused)
                    if epi == 2:
                        dv.gemm(A[i], Bt[i], Cc[i], M, N, K, K, 1, 1, K, N, epi=epi, X2=X2[i], act=act, passes=3)
                    else:
                        dv.gemm(A[i], Bm[i], Cc[i], M, N, K, K, 1, N, 1, N, bias=bias[i], epi=epi, X2=X2[i], act=act,
                                passes=3)
                for view, what in ((Cc, "C"), (X2, "X2")):
                    a, b_ = g.get(view[0][0], M * N), e.get(view[1][0], M * N)
                    err = np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9)
                    self.assertLess(err, 1e-4, (epi, act, what, "float32", err))
        # A read through an activation (A holds pre-activations): act(h)^T dy for a weight gradient
        Hm, Dy, Wg, tmp = pair(rng.normal(size=(M, K))), pair(rng.normal(size=(M, N))), pair(np.zeros((K, N))), \
            pair(np.zeros(M * K))
        for act in (0, 1, 2):
            for dv, i in ((g, 0), (e, 1)):
                dv.gemm(Hm[i], Dy[i], Wg[i], K, N, M, 1, K, N, 1, N, a_act=act, tmp=(tmp[i], M * K))
            a, b_ = g.get(Wg[0][0], K * N), e.get(Wg[1][0], K * N)
            self.assertLess(np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9), 3e-3, ("a_act", act))
        # the bias gradient taken by the weight-gradient product (and split k with it)
        for Mx in (512, 4096):
            Xm, Dy2 = pair(rng.normal(size=(Mx, K))), pair(rng.normal(size=(Mx, N)))
            Wg2, db = pair(np.zeros((K, N))), pair(np.zeros(N))
            for dv, i in ((g, 0), (e, 1)):
                dv.gemm(Xm[i], Dy2[i], Wg2[i], K, N, Mx, 1, K, N, 1, N, beta=True, db=db[i])
            for view, size, what in ((Wg2, K * N, "dW"), (db, N, "db")):
                a, b_ = g.get(view[0][0], size), e.get(view[1][0], size)
                self.assertLess(np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9), 3e-3, (what, Mx))
        # attention heads: batched, offsets inside the rows
        Bh, T, hd, H = 2, 128, 64, 3
        D = H * hd
        q, k, S = pair(rng.normal(size=(Bh, T, D))), pair(rng.normal(size=(Bh, T, D))), pair(np.zeros(Bh * H * T * T))
        for dv, i in ((g, 0), (e, 1)):
            dv.gemm(q[i], k[i], S[i], T, T, hd, D, 1, 1, D, T, alpha=0.125, batch=(Bh, H), sa=(T * D, hd),
                    sb=(T * D, hd), sc=(H * T * T, T * T))
        a, b_ = g.get(S[0][0], Bh * H * T * T), e.get(S[1][0], Bh * H * T * T)
        self.assertLess(np.abs(a - b_).max() / np.abs(b_).max(), 2e-3)

    def test_fused_attention(self):
        g, e = self.g, GT.Emulator(tc=True)
        if not g.flash:
            self.skipTest(f"no fused attention here: {g.tc_why}")
        rng = np.random.default_rng(8)
        for (B, T, H, hd, causal) in ((2, 128, 3, 64, True), (1, 100, 2, 32, False), (2, 64, 6, 64, False),
                                      (1, 200, 1, 64, True)):
            n = B * T * H * hd
            views = {}
            for name in ("q", "k", "v", "do"):
                a = rng.normal(size=n).astype(np.float32)
                views[name] = [(g.alloc(n), 0), (e.alloc(n), 0)]
                g.put(views[name][0][0], a)
                e.put(views[name][1][0], a)
            for name, size in (("o", n), ("L", B * H * T), ("dd", B * H * T), ("dq", n), ("dk", n), ("dv", n)):
                views[name] = [(g.alloc(size), 0), (e.alloc(size), 0)]
            for i, dv_ in ((0, g), (1, e)):
                V = {k_: v_[i] for k_, v_ in views.items()}
                dv_.flash_fwd(V["q"], V["k"], V["v"], V["o"], V["L"], B, T, H, hd, causal, 0.125)
                dv_.flash_dot(V["do"], V["o"], V["dd"], B, T, H, hd)
                dv_.flash_bwd(V["q"], V["k"], V["v"], V["do"], V["L"], V["dd"], V["dq"], V["dk"], V["dv"], B, T, H,
                              hd, causal, 0.125)
            for name, size in (("o", n), ("L", B * H * T), ("dq", n), ("dk", n), ("dv", n)):
                a, b_ = g.get(views[name][0][0], size), e.get(views[name][1][0], size)
                err = np.abs(a - b_).max() / (np.abs(b_).max() + 1e-9)
                self.assertLess(err, 2e-2, (name, B, T, H, hd, causal, err))

    def test_training_on_the_card_matches_the_emulator(self):
        rng = np.random.default_rng(3)
        lm = L.transformer_lm(40, d=64, heads=4, layers=2, context=32)
        ids = rng.integers(0, 40, (64, 32))
        lab = np.roll(ids, -1, 1)
        for prec in ("fp32", "auto", "fp4"):
            a = GT.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=adamw(lr=3e-3), epochs=3,
                       batch_size=16, verbose=False, precision=prec, ops=GT.Emulator(tc=self.g.tc))
            b = GT.fit(lm, lm.init(seed=0), ids, lab, loss="cross_entropy", opt=adamw(lr=3e-3), epochs=3,
                       batch_size=16, verbose=False, precision=prec, ops=self.g)
            np.testing.assert_allclose(a.history, b.history, rtol=2e-3 if prec == "fp32" else 2e-2, err_msg=prec)


class Kernels(unittest.TestCase):
    """Every kernel module builds (no GPU needed): a mistake in one of them must fail here, not on the card."""

    def test_every_module_builds(self):
        try:
            import llvmlite  # noqa: F401
        except ImportError:
            self.skipTest("llvmlite is not installed here (the GPU kernels are generated with it)")
        from apsi import gpu_train_kernels as KS
        for which, build in (("main", KS._main_module), ("tc", KS._tc_module), ("tiled", KS._tiled_module),
                             ("tiled_x", lambda: KS._tiled_module(True)),
                             ("tiled_w", lambda: KS._tiled_module(True, True)),
                             ("tiled_h", lambda: KS._tiled_module(h=True)), ("flash", KS._flash_module)):
            text = build()
            self.assertIn(".entry", text, which)
        self.assertIn(".minnctapersm 2", KS._tiled_module(True))
        self.assertIn("cp.async.wait_group 2", KS._tiled_module(h=True))
        self.assertIn(".entry to_bf16", KS._tiled_module())
        self.assertIn(".entry flash_f64", KS._flash_module())


if __name__ == "__main__":
    unittest.main()
