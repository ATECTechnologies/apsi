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
        for kind in range(7):
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
