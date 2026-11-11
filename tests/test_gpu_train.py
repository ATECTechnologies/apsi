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
        for rms in (False, True):
            for dv, i in ((g, 0), (e, 1)):
                dv.ln_f(X[i], Gm[i], Bt[i], Y2[i], mean[i], rstd[i], M, N, 1e-5, rms)
                dv.ln_b(X[i], Gm[i], Y2[i], mean[i], rstd[i], dX[i], M, N, rms)
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


if __name__ == "__main__":
    unittest.main()
