"""Apsi's own GPU training: fit(..., backend="gpu") trains on an NVIDIA card with Apsi's kernels (gpu_train_kernels)
through the CUDA driver - no PyTorch, no CUDA toolkit, nothing to download.

Supported layers: sequential, linear, activation (relu, gelu, silu/swish, tanh, sigmoid, leaky_relu, identity),
dropout, layer_norm, rms_norm, embedding, positional_encoding (learned or fixed), residual (with pre_norm),
multi_head_attention (self-attention, causal or not), transformer_block - so mlp(), resnet_mlp(), transformer_lm()
and transformer_encoder() train here. Losses: "cross_entropy" (negative labels left out) and "mse". Optimizers:
Adam, AdamW, SGD (with momentum), with clip= (global gradient norm) and schedule=.

Precision: "auto" / "bf16" / "fp16" multiply on bfloat16 tensor cores with float32 sums (Ampere and newer; the
float32 kernel on older cards); "fp32" / "tf32" use three tensor-core passes (about float32 accuracy). Parameters,
gradients and the optimizer state are float32. "fp8", "fp6", "fp6_e2m3", "fp4", "int8", "int4": the weights are
rounded to that format on the card for every forward pass (lowbit.py), the master copy stays float32.

Memory: parameters, gradients and Adam state live in three flat arrays (one optimizer kernel for all of them, one
clear of the gradients per step); activations are kept per layer in buffers reused every step. The loss is read
back once per epoch, so the card never waits for the processor inside an epoch.

Everything runs the same on the Emulator (NumPy), which the tests use to check the whole training against Apsi's
autodiff, and on any real GPU against the Emulator.
"""
import ctypes
import math
import os
import time

import numpy as np

from . import gpu_train_kernels as KS
from .values import Record, tree_leaves, tree_map



def _keeps_frozen(fn):
    def run(*a, **kw):
        from .layers import keeps_frozen
        return keeps_frozen(fn)(*a, **kw)
    run.__name__, run.__doc__, run.__wrapped__ = fn.__name__, fn.__doc__, fn
    return run

class TrainError(Exception):
    pass


class Buf:
    __slots__ = ("ptr", "n", "a")

    def __init__(self, ptr, n, a=None):
        self.ptr, self.n, self.a = ptr, n, a


# ============================================================================== devices
class _Ops:
    """Kernel launches in terms of (buffer, offset) views; sizes in float32 elements."""

    tc = False

    def _ring_ready(self):                               # pinned staging memory: the real card only
        return False

    def drop_ring(self):
        pass
    _db = None

    def gemm(self, A, B, Cv, M, N, K, sam, sak, sbk, sbn, ldc, alpha=1.0, beta=False, bias=None, batch=(1, 1),
             sa=(0, 0), sb=(0, 0), sc=(0, 0), passes=1, epi=0, X2=None, act=0, a_act=None, tmp=None, db=None):
        """C = alpha A B (+ bias) (+ C), then epi: 1 X2 <- C, C <- act(C); 2 C <- C * act'(X2); 3 C <- C + X2.
        A kernel that can do epi in its last step does (one pass over C); otherwise a second kernel does it."""
        nb1, nb2 = batch
        if db is not None:                               # + the column sums of B (dy) into db: the bias gradient
            self._db = db
            self.gemm(A, B, Cv, M, N, K, sam, sak, sbk, sbn, ldc, alpha, beta, bias, batch, sa, sb, sc, passes, epi,
                      X2, act, a_act, tmp)
            if self._db is not None:                     # the product did not take it: a pass of its own
                self._db = None
                self.colsum(B, None, None, None, None, db, K, N, 0)
            return
        if a_act is not None:                            # A holds pre-activations: act(A) B
            if self._gemm(A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta,
                          passes, a_act=a_act):
                return
            self.act_f(A, tmp[0], tmp[1], a_act)                # (composite: act(A) into tmp first)
            A = tmp[0]
        args = (A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes)
        if not epi:
            self._gemm(*args)
            return
        if self._gemm(*args, epi=epi, X2=X2, act=act):
            return
        n = M * N
        if epi == 1:
            self._gemm(A, B, X2, *args[3:])
            self.act_f(X2, Cv, n, act)
        elif epi == 2:
            self._gemm(*args)
            self.act_b(X2, Cv, Cv, n, act)
        else:
            self._gemm(*args)
            self.axpy(Cv, X2, n)


def _guard():
    """Steps with a non-finite gradient (inf / NaN anywhere) are skipped on the card: the weights, the optimizer's
    moments and the carried rounding error stay as they were, and the step is counted (the default; one sum of
    squares of the gradient per step). APSI_GPU_GUARD=0 turns it off."""
    return os.environ.get("APSI_GPU_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")


def _deterministic():
    """The same bits every run (the default): kernels add in a fixed order, never atomically. APSI_GPU_DETERMINISTIC=0
    lets the fused attention's backward add dQ atomically (one kernel instead of two)."""
    return os.environ.get("APSI_GPU_DETERMINISTIC", "1").strip().lower() not in ("0", "false", "no", "off")


def _bf16(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)


class Emulator(_Ops):
    """The kernels in NumPy, element for element (bfloat16 rounding of the tensor-core inputs included)."""
    name = "emulator"

    def __init__(self, tc=True):
        self.tc = tc
        self.used = 0

    def alloc(self, n):
        self.used += 4 * n
        return Buf(0, n, np.zeros(max(1, n), np.float32))

    def free(self, b):
        self.used -= 4 * b.n

    def put(self, b, a, off=0):
        a = np.ascontiguousarray(a)
        b.a[off:off + a.size] = a.view(np.float32) if a.dtype in (np.int32, np.uint32) else a

    def get(self, b, n, off=0):
        return b.a[off:off + n].copy()

    def sync(self):
        pass

    @staticmethod
    def v(view, n):
        b, off = view
        return b.a[off:off + n]

    def fill(self, view, n, value):
        self.v(view, n)[:] = value

    def axpy(self, Y, X, n, a=1.0):
        y = self.v(Y, n)
        y[:] = np.float32(a) * self.v(X, n) + y

    def pull(self, dst, src_ops, src, n):
        """dst (on this device) <- src (on src_ops' device): n float32."""
        self.v(dst, n)[:] = src_ops.v(src, n)

    def tanh_s(self, OUT, W):
        self.v(OUT, 1)[0] = np.float32(math.tanh(float(self.v(W, 1)[0])))

    def axpy_s(self, Y, X, n, S):
        y = self.v(Y, n)
        y[:] = (np.float64(self.v(S, 1)[0]) * self.v(X, n).astype(np.float64) + y).astype(np.float32)

    def gate_g(self, G, DG, TH):
        th = float(self.v(TH, 1)[0])
        g = self.v(G, 1)
        g[0] = g[0] + np.float32(float(self.v(DG, 1)[0]) * (1.0 - th * th))

    def add3(self, Z, X, Y, n):
        self.v(Z, n)[:] = self.v(X, n) + self.v(Y, n)

    def mul3(self, Z, X, Y, n):
        self.v(Z, n)[:] = self.v(X, n) * self.v(Y, n)

    @staticmethod
    def _act(x, kind):
        x = x.astype(np.float32)
        if kind == 0:
            return np.maximum(x, 0), (x > 0).astype(np.float32)
        if kind == 1:
            c = np.float32(0.7978845608028654)
            t = np.tanh(c * (x + np.float32(0.044715) * x ** 3))
            return 0.5 * x * (1 + t), 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * np.float32(0.044715) * x * x)
        if kind == 2:
            s = 1 / (1 + np.exp(-x))
            return x * s, s + x * s * (1 - s)
        if kind == 3:
            t = np.tanh(x)
            return t, 1 - t * t
        if kind == 4:
            s = 1 / (1 + np.exp(-x))
            return s, s * (1 - s)
        if kind == 5:
            return x, np.ones_like(x)
        if kind == 7:
            e = np.exp(np.minimum(x, 0))
            return np.where(x > 0, x, e - 1), np.where(x > 0, np.float32(1), e).astype(np.float32)
        if kind in (10, 11):
            red = x - np.float32(2 * np.pi) * np.rint(x / np.float32(2 * np.pi))
            sn, cs = np.sin(red), np.cos(red)
            return (sn, cs) if kind == 10 else (cs, -sn)
        if kind in (8, 9):
            s = 1 / (1 + np.exp(-x))
            sp = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
            if kind == 8:
                return sp, s
            t = np.tanh(sp)
            return x * t, t + x * (1 - t * t) * s
        return np.where(x > 0, x, 0.01 * x), np.where(x > 0, 1, 0.01).astype(np.float32)

    def act_f(self, X, Y, n, kind):
        self.v(Y, n)[:] = self._act(self.v(X, n), kind)[0]

    def act_b(self, X, DY, DX, n, kind):
        self.v(DX, n)[:] = self.v(DY, n) * self._act(self.v(X, n), kind)[1]

    def colsum(self, DY, X, MEAN, RSTD, DG, DB, M, N, mode):
        dy = self.v(DY, M * N).reshape(M, N)
        if mode != 2:
            self.v(DB, N)[:] += dy.sum(0)
        if mode:
            x = self.v(X, M * N).reshape(M, N)
            mu = self.v(MEAN, M)[:, None] if mode == 1 else 0.0
            self.v(DG, N)[:] += (dy * (x - mu) * self.v(RSTD, M)[:, None]).sum(0)

    def ln_f(self, X, Gm, Bt, Y, MEAN, RSTD, M, N, eps, rms):
        x = self.v(X, M * N).reshape(M, N)
        mu = np.zeros((M, 1), np.float32) if rms else x.mean(1, keepdims=True)
        var = ((x - mu) ** 2).mean(1, keepdims=True)
        r = 1 / np.sqrt(var + np.float32(eps))
        y = (x - mu) * r * self.v(Gm, N)
        if not rms:
            y = y + self.v(Bt, N)
        self.v(Y, M * N)[:] = y.reshape(-1)
        self.v(MEAN, M)[:] = mu[:, 0]
        self.v(RSTD, M)[:] = r[:, 0]

    def ln_b(self, X, Gm, DY, MEAN, RSTD, DX, M, N, rms, RES=None):
        x = self.v(X, M * N).reshape(M, N)
        mu = 0.0 if rms else self.v(MEAN, M)[:, None]
        r = self.v(RSTD, M)[:, None]
        xh = (x - mu) * r
        gdy = self.v(DY, M * N).reshape(M, N) * self.v(Gm, N)
        m1 = 0.0 if rms else gdy.mean(1, keepdims=True)
        m2 = (gdy * xh).mean(1, keepdims=True)
        dx = (r * (gdy - m1 - xh * m2)).reshape(-1)
        self.v(DX, M * N)[:] = dx + self.v(RES, M * N) if RES is not None else dx

    def embed_f(self, E, IDS, Y, M, D):
        ids = self.v(IDS, M).view(np.int32)
        self.v(Y, M * D)[:] = self.v(E, (int(ids.max()) + 1) * D).reshape(-1, D)[ids].reshape(-1)

    def embed_b(self, DY, IDS, DE, M, D, V):
        ids = self.v(IDS, M).view(np.int32)
        de = self.v(DE, V * D).reshape(V, D)
        np.add.at(de, ids, self.v(DY, M * D).reshape(M, D))

    def pos_f(self, X, P, Y, M, D, T):
        p = self.v(P, T * D).reshape(T, D)
        self.v(Y, M * D)[:] = (self.v(X, M * D).reshape(-1, T, D) + p).reshape(-1)

    def pos_b(self, DY, DP, M, D, T):
        self.v(DP, T * D)[:] += self.v(DY, M * D).reshape(-1, T, D).sum(0).reshape(-1)

    def softmax_f(self, S, P, rows, n, T, causal):
        s = self.v(S, rows * n).reshape(rows, n).copy()
        if causal:
            q = (np.arange(rows) % T)[:, None]
            s = np.where(np.arange(n)[None, :] <= q, s, -np.inf)
        e = np.exp(s - s.max(1, keepdims=True))
        self.v(P, rows * n)[:] = (e / e.sum(1, keepdims=True)).reshape(-1)

    def softmax_b(self, P, DP, DS, rows, n):
        p = self.v(P, rows * n).reshape(rows, n)
        dp = self.v(DP, rows * n).reshape(rows, n)
        self.v(DS, rows * n)[:] = (p * (dp - (p * dp).sum(1, keepdims=True))).reshape(-1)

    def ce(self, L, LAB, DL, LOSS, M, V, wg, wl):
        lg = self.v(L, M * V).reshape(M, V)
        lab = self.v(LAB, M).view(np.int32)
        mx = lg.max(1, keepdims=True)
        e = np.exp(lg - mx)
        Z = e.sum(1, keepdims=True)
        p = e / Z
        ok = lab >= 0
        p[np.arange(M)[ok], lab[ok]] -= 1
        p[~ok] = 0
        self.v(DL, M * V)[:] = (p * np.float32(wg)).reshape(-1)
        lse = (mx + np.log(Z))[:, 0]
        self.v(LOSS, 1)[0] += np.float32(wl) * (lse[ok] - lg[np.arange(M)[ok], lab[ok]]).sum()

    def loss_ew(self, Y, T, W, DY, LOSS, n, wg, wl, kind):
        y, t = self.v(Y, n), self.v(T, n)
        w = self.v(W, n) if W is not None else np.float32(1)
        if kind == 0:
            d = y - t
            lv, gv = np.abs(d), np.sign(d)
        else:
            sg = 1 / (1 + np.exp(-y))
            a, c = sg + np.float32(1e-12), 1 - sg + np.float32(1e-12)
            lv = -(t * np.log(a) + (1 - t) * np.log(c))
            ds = sg * (1 - sg)
            gv = -(t * ds / a - (1 - t) * ds / c)
        self.v(DY, n)[:] = gv * w * np.float32(wg)
        self.v(LOSS, 1)[0] += np.float32(wl) * (lv * w).sum()

    def mse(self, Y, T, DY, LOSS, n, wg, wl):
        d = self.v(Y, n) - self.v(T, n)
        self.v(DY, n)[:] = 2 * d * np.float32(wg)
        self.v(LOSS, 1)[0] += np.float32(wl) * (d * d).sum()

    def sumsq(self, X, n, ACC):
        x = self.v(X, n)
        self.v(ACC, 1)[0] += (x * x).sum()

    def _kahan(self, p, new, delta, COMP, n):
        if COMP is None:
            p[:] = new
            return
        c = self.v(COMP, n)
        y = delta - c
        t = p + y
        c[:] = (t - p) - y
        p[:] = t

    def adam(self, Pm, Gr, Mm, Vm, SQ, n, n_decay, lr, b1, b2, eps, c1, c2, wd, clip, COMP=None):
        if not np.isfinite(self.v(SQ, 1)[0]):              # a non-finite gradient: the step is skipped
            self.v(Gr, n)[:] = 0.0
            return
        g = self.v(Gr, n) * self._clip(SQ, clip)
        p, m, v = self.v(Pm, n), self.v(Mm, n), self.v(Vm, n)
        w = p.copy()
        w[:n_decay] *= np.float32(1 - lr * wd)
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * g * g
        upd = np.float32(lr) * (m / np.float32(c1)) / (np.sqrt(v / np.float32(c2)) + np.float32(eps))
        dec = np.zeros_like(p)
        dec[:n_decay] = p[:n_decay] * np.float32(lr * wd)
        self._kahan(p, w - upd, -(dec + upd), COMP, n)
        self.v(Gr, n)[:] = 0.0                    # the kernel clears the gradient it used

    def sgd(self, Pm, Gr, VEL, SQ, n, lr, mom, clip, COMP=None):
        if not np.isfinite(self.v(SQ, 1)[0]):
            self.v(Gr, n)[:] = 0.0
            return
        vel = self.v(VEL, n)
        vel[:] = np.float32(mom) * vel + self.v(Gr, n) * self._clip(SQ, clip)
        p = self.v(Pm, n)
        step = np.float32(lr) * vel
        self._kahan(p, p - step, -step, COMP, n)
        self.v(Gr, n)[:] = 0.0

    def _clip(self, SQ, clip):
        if clip <= 0:
            return np.float32(1)
        return np.float32(min(1.0, clip / math.sqrt(float(self.v(SQ, 1)[0]) + 1e-30)))

    # ---------------------------------------------------------------- sparse graphs
    def spmm(self, X, COL, VAL, VPERM, OFF, Y, N, H, F, add=False):
        off = self.v(OFF, N + 1).view(np.int32).astype(np.int64)
        E = int(off[-1])
        HF = H * F
        col = self.v(COL, max(E, 1)).view(np.int32)[:E].astype(np.int64)
        x = X[0].a[X[1]:].view(np.float32)
        rowsx = x[(col[:, None] * HF + np.arange(HF)[None, :])].reshape(E, H, F) if E else np.zeros((0, H, F),
                                                                                                     np.float32)
        if VAL is not None:
            ve = np.arange(E) if VPERM is None else self.v(VPERM, max(E, 1)).view(np.int32)[:E].astype(np.int64)
            val = VAL[0].a[VAL[1]:].view(np.float32)
            rowsx = rowsx * val[ve[:, None] * H + np.arange(H)[None, :]][:, :, None]
        seg = np.repeat(np.arange(N), np.diff(off))
        out = np.zeros((N, HF), np.float64)
        np.add.at(out, seg, rowsx.reshape(E, HF).astype(np.float64))
        y = self.v(Y, N * HF)
        y[:] = (out.reshape(-1) + (y if add else 0)).astype(np.float32)

    def edge_dot(self, DY, X, ROW, COL, DA, E, H, F):
        row = self.v(ROW, max(E, 1)).view(np.int32)[:E].astype(np.int64)
        col = self.v(COL, max(E, 1)).view(np.int32)[:E].astype(np.int64)
        HF = H * F
        dy = DY[0].a[DY[1]:].view(np.float32)
        x = X[0].a[X[1]:].view(np.float32)
        a = dy[row[:, None] * HF + np.arange(HF)[None, :]].reshape(E, H, F)
        bb = x[col[:, None] * HF + np.arange(HF)[None, :]].reshape(E, H, F)
        self.v(DA, E * H)[:] = (a.astype(np.float64) * bb).sum(-1).reshape(-1)

    def seg_softmax(self, S, OFF, A, N, H, slope):
        off = self.v(OFF, N + 1).view(np.int32).astype(np.int64)
        E = int(off[-1])
        z = self.v(S, E * H).reshape(E, H)
        z = np.where(z > 0, z, z * np.float32(slope))
        seg = np.repeat(np.arange(N), np.diff(off))
        mx = np.full((N, H), -3.0e38, np.float32)
        np.maximum.at(mx, seg, z)
        e = np.exp(z - mx[seg])
        den = np.zeros((N, H), np.float32)
        np.add.at(den, seg, e)
        self.v(A, E * H)[:] = (e / (den[seg] + np.float32(1e-16))).reshape(-1)

    def seg_softmax_b(self, S, A, DA, OFF, DS, N, H, slope):
        off = self.v(OFF, N + 1).view(np.int32).astype(np.int64)
        E = int(off[-1])
        a, da, sc = (self.v(t, E * H).reshape(E, H) for t in (A, DA, S))
        seg = np.repeat(np.arange(N), np.diff(off))
        t = np.zeros((N, H), np.float32)
        np.add.at(t, seg, a * da)
        g = a * (da - t[seg])
        self.v(DS, E * H)[:] = np.where(sc > 0, g, g * np.float32(slope)).reshape(-1)

    def dot_acc(self, X, Y, n, ACC):
        self.v(ACC, 1)[0] += (self.v(X, n).astype(np.float64) * self.v(Y, n)).sum()

    def scale_add(self, Z, X, S, n, c):
        z = self.v(Z, n)
        z[:] = z + self.v(X, n) * (self.v(S, 1)[0] + np.float32(c))

    def acc2(self, HI, LO, Gr, n, clear=True):
        g, h, lo = self.v(Gr, n), self.v(HI, n), self.v(LO, n)
        s_ = (h + g).astype(np.float32)
        bp = (s_ - h).astype(np.float32)
        e = ((h - (s_ - bp)) + (g - bp)).astype(np.float32)
        h[:] = s_
        lo[:] = lo + e
        if clear:
            g[:] = 0.0

    def ew1(self, X, Y, n, kind):
        x = self.v(X, n)
        with np.errstate(all="ignore"):
            f = [np.exp, np.log, np.sqrt, np.abs, np.square, lambda a: np.float32(1) / a, np.negative,
                 lambda a: np.float32(1) / np.sqrt(a)][kind]
            self.v(Y, n)[:] = f(x).astype(np.float32)

    def ew2(self, Z, X, Y, n, kind):
        x, y = self.v(X, n), self.v(Y, n)
        with np.errstate(all="ignore"):
            f = [np.divide, np.maximum, np.minimum, np.power][kind]
            self.v(Z, n)[:] = f(x, y).astype(np.float32)

    def fin2(self, HI, LO, Gr, n):
        self.v(Gr, n)[:] = self.v(HI, n) + self.v(LO, n)
        self.v(HI, n)[:] = 0.0
        self.v(LO, n)[:] = 0.0

    def cols(self, X, Y, rows, w, ld, off, back):
        y = Y[0].a[Y[1]:Y[1] + rows * ld].reshape(rows, ld)
        x = self.v(X, rows * w).reshape(rows, w)
        if back:
            x[:] = y[:, off:off + w]
        else:
            y[:, off:off + w] = x

    # ---------------------------------------------------------------- physics-informed networks (Taylor mode)
    @staticmethod
    def _derivs(z, kind):
        if kind == 0:
            t = np.tanh(z)
            t1 = 1 - t * t
            return t, t1, -2 * t * t1, t1 * (6 * t * t - 2)
        if kind in (1, 6):
            red = z - np.float32(2 * np.pi) * np.rint(z / np.float32(2 * np.pi))
            sn, cs = np.sin(red), np.cos(red)
            return (sn, cs, -sn, -cs) if kind == 1 else (cs, -sn, -cs, sn)
        s = 1 / (1 + np.exp(-z))
        s1 = s * (1 - s)
        s2 = s1 * (1 - 2 * s)
        s3 = s2 * (1 - 2 * s) - 2 * s1 * s1
        if kind == 2:
            return s, s1, s2, s3
        if kind == 3:
            return z * s, s + z * s1, 2 * s1 + z * s2, 3 * s2 + z * s3
        if kind == 4:
            return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z))), s, s1, s2
        one, zero = np.ones_like(z), np.zeros_like(z)
        return z, one, zero, zero

    def taylor_f(self, Z, Y, N, F, d, kind):
        z = self.v(Z, (1 + 2 * d) * N * F).reshape(1 + 2 * d, N * F)
        y = self.v(Y, (1 + 2 * d) * N * F).reshape(1 + 2 * d, N * F)
        f0, f1, f2, _ = self._derivs(z[0], kind)
        j, h = z[1:1 + d], z[1 + d:]
        y[0] = f0
        y[1:1 + d] = f1 * j
        y[1 + d:] = f2 * j * j + f1 * h

    def taylor_b(self, Z, DY, DZ, N, F, d, kind):
        z = self.v(Z, (1 + 2 * d) * N * F).reshape(1 + 2 * d, N * F)
        dy = self.v(DY, (1 + 2 * d) * N * F).reshape(1 + 2 * d, N * F)
        dz = self.v(DZ, (1 + 2 * d) * N * F).reshape(1 + 2 * d, N * F)
        _, f1, f2, f3 = self._derivs(z[0], kind)
        j, h = z[1:1 + d], z[1 + d:]
        dj, dh = dy[1:1 + d], dy[1 + d:]
        dz0 = dy[0] * f1 + (dj * f2 * j + dh * (f3 * j * j + f2 * h)).sum(0)
        dz[1:1 + d] = dj * f1 + dh * 2 * f2 * j
        dz[1 + d:] = dh * f1
        dz[0] = dz0

    def rowbias(self, Y, B, rows, F):
        y = self.v(Y, rows * F).reshape(rows, F)
        y += self.v(B, F)

    # ---------------------------------------------------------------- recurrent cells
    def _rv(self, X, B, T, w):
        return self.v(X, B * T * w).reshape(B, T, w)

    @staticmethod
    def _sg(x):
        return 1 / (1 + np.exp(-x))

    def lstm_f(self, Z, CS, HS, B, H, T, t):
        z = self._rv(Z, B, T, 4 * H)[:, t]
        cs, hs = self._rv(CS, B, T, H), self._rv(HS, B, T, H)
        cp = cs[:, t - 1] if t else 0
        c = self._sg(z[:, H:2 * H]) * cp + self._sg(z[:, :H]) * np.tanh(z[:, 2 * H:3 * H])
        cs[:, t] = c
        hs[:, t] = self._sg(z[:, 3 * H:]) * np.tanh(c)

    def lstm_b(self, Z, CS, DY, DHN, DCN, DZ, B, H, T, t):
        z = self._rv(Z, B, T, 4 * H)[:, t]
        cs = self._rv(CS, B, T, H)
        i_, f_, g_, o_ = self._sg(z[:, :H]), self._sg(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), self._sg(z[:, 3 * H:])
        c, cp = cs[:, t], (cs[:, t - 1] if t else np.zeros_like(cs[:, t]))
        dhn, dcn = self.v(DHN, B * H).reshape(B, H), self.v(DCN, B * H).reshape(B, H)
        dh = self._rv(DY, B, T, H)[:, t] + dhn
        tc = np.tanh(c)
        dc = dh * o_ * (1 - tc * tc) + dcn
        dz = self._rv(DZ, B, T, 4 * H)
        dz[:, t] = np.concatenate([dc * g_ * i_ * (1 - i_), dc * cp * f_ * (1 - f_), dc * i_ * (1 - g_ * g_),
                                   dh * tc * o_ * (1 - o_)], 1)
        dcn[:] = dc * f_

    def gru_a(self, ZR, HS, RH, B, H, T, t):
        zr = self._rv(ZR, B, T, 2 * H)
        zr[:, t] = self._sg(zr[:, t])
        hp = self._rv(HS, B, T, H)[:, t - 1] if t else 0
        self._rv(RH, B, T, H)[:, t] = zr[:, t, H:] * hp

    def gru_b(self, CX, ZR, HS, B, H, T, t):
        cx = self._rv(CX, B, T, H)
        cx[:, t] = np.tanh(cx[:, t])
        hs = self._rv(HS, B, T, H)
        z = self._rv(ZR, B, T, 2 * H)[:, t, :H]
        hp = hs[:, t - 1] if t else 0
        hs[:, t] = (1 - z) * hp + z * cx[:, t]

    def gru_b1(self, CX, ZR, HS, DY, DHN, DCX, DZR, DHP, B, H, T, t):
        cand = self._rv(CX, B, T, H)[:, t]
        z = self._rv(ZR, B, T, 2 * H)[:, t, :H]
        hp = self._rv(HS, B, T, H)[:, t - 1] if t else 0
        dh = self._rv(DY, B, T, H)[:, t] + self.v(DHN, B * H).reshape(B, H)
        self._rv(DCX, B, T, H)[:, t] = dh * z * (1 - cand * cand)
        self._rv(DZR, B, T, 2 * H)[:, t, :H] = dh * (cand - hp) * z * (1 - z)
        self.v(DHP, B * H)[:] = (dh * (1 - z)).reshape(-1)

    def gru_b2(self, DRH, ZR, HS, DZR, DHP, B, H, T, t):
        r = self._rv(ZR, B, T, 2 * H)[:, t, H:]
        d = self.v(DRH, B * H).reshape(B, H)
        hp = self._rv(HS, B, T, H)[:, t - 1] if t else 0
        self._rv(DZR, B, T, 2 * H)[:, t, H:] = d * hp * r * (1 - r)
        dhp = self.v(DHP, B * H)
        dhp[:] = dhp + (d * r).reshape(-1)

    # ---------------------------------------------------------------- images
    def im2col(self, X, COL, N, C, H, W, kh, kw, s, p, OH, OW):
        x = self.v(X, N * C * H * W).reshape(N, C, H, W)
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        out = np.empty((N, OH, OW, C, kh, kw), np.float32)
        for i in range(kh):
            for j in range(kw):
                out[:, :, :, :, i, j] = xp[:, :, i:i + s * OH:s, j:j + s * OW:s].transpose(0, 2, 3, 1)
        self.v(COL, out.size)[:] = out.reshape(-1)

    def col2im(self, DCOL, DX, N, C, H, W, kh, kw, s, p, OH, OW):
        g = self.v(DCOL, N * OH * OW * C * kh * kw).reshape(N, OH, OW, C, kh, kw)
        dxp = np.zeros((N, C, H + 2 * p, W + 2 * p), np.float64)
        for i in range(kh):
            for j in range(kw):
                dxp[:, :, i:i + s * OH:s, j:j + s * OW:s] += g[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        self.v(DX, N * C * H * W)[:] = dxp[:, :, p:p + H, p:p + W].reshape(-1)

    def perm_nc(self, Z, Y, BIAS, N, Cc, S, back):
        if back:
            self.v(Z, N * S * Cc)[:] = self.v(Y, N * Cc * S).reshape(N, Cc, S).transpose(0, 2, 1).reshape(-1)
        else:
            y = self.v(Z, N * S * Cc).reshape(N, S, Cc).transpose(0, 2, 1)
            if BIAS is not None:
                y = y + self.v(BIAS, Cc)[None, :, None]
            self.v(Y, N * Cc * S)[:] = y.reshape(-1)

    def pool_f(self, X, Y, NC, H, W, kk, mode):
        H2, W2 = H // kk, W // kk
        xr = self.v(X, NC * H * W).reshape(NC, H, W)[:, :H2 * kk, :W2 * kk].reshape(NC, H2, kk, W2, kk)
        self.v(Y, NC * H2 * W2)[:] = (xr.max(axis=(2, 4)) if mode == 0 else xr.mean(axis=(2, 4))).reshape(-1)

    def pool_b(self, X, Y, DY, DX, NC, H, W, kk, mode):
        H2, W2 = H // kk, W // kk
        x = self.v(X, NC * H * W).reshape(NC, H, W)
        dy = self.v(DY, NC * H2 * W2).reshape(NC, H2, 1, W2, 1)
        full = np.zeros((NC, H, W), np.float32)
        if mode == 0:
            xr = x[:, :H2 * kk, :W2 * kk].reshape(NC, H2, kk, W2, kk)
            m = (xr == self.v(Y, NC * H2 * W2).reshape(NC, H2, 1, W2, 1)).astype(np.float32)
            m = m / m.sum(axis=(2, 4), keepdims=True)
            full[:, :H2 * kk, :W2 * kk] = (m * dy).reshape(NC, H2 * kk, W2 * kk)
        else:
            full[:, :H2 * kk, :W2 * kk] = np.broadcast_to(dy / (kk * kk), (NC, H2, kk, W2, kk)).reshape(
                NC, H2 * kk, W2 * kk)
        self.v(DX, NC * H * W)[:] = full.reshape(-1)

    def topk_gate(self, Lg, IDX, GATE, M, E, k):
        lg = self.v(Lg, M * E).reshape(M, E)
        idx = np.argsort(-lg, axis=1, kind="stable")[:, :k]
        sel = np.take_along_axis(lg, idx, 1)
        e = np.exp(sel - sel[:, :1])
        self.v(IDX, M * k)[:] = idx.astype(np.int32).reshape(-1).view(np.float32)
        self.v(GATE, M * k)[:] = (e / e.sum(1, keepdims=True)).reshape(-1)

    def topk_gate_b(self, GATE, DG, IDX, DL, M, E, k):
        g = self.v(GATE, M * k).reshape(M, k)
        dg = self.v(DG, M * k).reshape(M, k)
        idx = self.v(IDX, M * k).view(np.int32).reshape(M, k)
        dl = np.zeros((M, E), np.float32)
        np.put_along_axis(dl, idx.astype(np.int64), g * (dg - (g * dg).sum(1, keepdims=True)), 1)
        self.v(DL, M * E)[:] = dl.reshape(-1)

    def colmul(self, X, S, Y, n, D):
        self.v(Y, n)[:] = (self.v(X, n).reshape(-1, D) * self.v(S, D)).reshape(-1)

    def rope(self, X, CS, SN, rows, hd, H, T, sign):
        half = hd // 2
        x = self.v(X, rows * hd).reshape(rows, hd)
        t = (np.arange(rows) // H) % T
        c = self.v(CS, T * half).reshape(T, half)[t]
        sn = self.v(SN, T * half).reshape(T, half)[t] * np.float32(sign)
        x1, x2 = x[:, :half].copy(), x[:, half:].copy()
        x[:, :half] = x1 * c - x2 * sn
        x[:, half:] = x2 * c + x1 * sn

    def gather(self, SRC, IDX, DST, rows, width, start):
        idx = self.v(IDX, start + rows).view(np.int32)[start:]
        src = SRC[0].a[SRC[1]:].view(np.float32)
        self.v(DST, rows * width)[:] = src[(idx[:, None].astype(np.int64) * width + np.arange(width)).reshape(-1)]

    def drop_f(self, X, Y, MASK, n, p, seed):
        i = np.arange(n, dtype=np.uint32)
        h = (i * np.uint32(0x9E3779B1)) ^ np.uint32(seed & 0xFFFFFFFF)
        h ^= h >> 16
        h *= np.uint32(0x7FEB352D)
        h ^= h >> 15
        h *= np.uint32(0x846CA68B)
        h ^= h >> 16
        u = (h >> 8).astype(np.float32) * np.float32(1.0 / (1 << 24))
        keep = np.where(u >= np.float32(p), np.float32(1) / np.float32(1 - p), np.float32(0))
        self.v(MASK, n)[:] = keep
        self.v(Y, n)[:] = self.v(X, n) * keep

    # fused attention: q, k, v, o in [B, T, H*hd]; L (log2 of each row's softmax denominator) [B*H, T]
    flash = True

    def _heads(self, view, B, T, H, hd):
        return self.v(view, B * T * H * hd).reshape(B, T, H, hd).transpose(0, 2, 1, 3)

    def _probs(self, q, k, T, causal, scale):
        s = np.einsum("bhtd,bhsd->bhts", _bf16(q).astype(np.float64), _bf16(k).astype(np.float64)) * scale
        s2 = (s * 1.4426950408889634).astype(np.float32)
        if causal:
            s2 = np.where(np.tril(np.ones((T, T), bool)), s2, -np.inf)
        return s2

    def flash_fwd(self, Q, K, V, O, L, B, T, H, hd, causal, scale):
        q, k, v = (self._heads(x, B, T, H, hd) for x in (Q, K, V))
        s2 = self._probs(q, k, T, causal, scale)
        m = s2.max(-1, keepdims=True)
        pu = np.exp2(s2 - m)
        lsum = pu.sum(-1, keepdims=True)
        o = np.einsum("bhts,bhsd->bhtd", _bf16(pu).astype(np.float64), _bf16(v).astype(np.float64)) / lsum
        self.v(O, B * T * H * hd)[:] = o.transpose(0, 2, 1, 3).astype(np.float32).reshape(-1)
        self.v(L, B * H * T)[:] = (m + np.log2(lsum)).astype(np.float32).reshape(-1)

    def flash_dot(self, DO, O, DD, B, T, H, hd):
        do, o = self._heads(DO, B, T, H, hd), self._heads(O, B, T, H, hd)
        self.v(DD, B * H * T)[:] = (do * o).sum(-1).astype(np.float32).reshape(-1)

    def flash_bwd(self, Q, K, V, DO, L, DD, DQ, DK, DV, B, T, H, hd, causal, scale):
        q, k, v, do = (self._heads(x, B, T, H, hd) for x in (Q, K, V, DO))
        s2 = self._probs(q, k, T, causal, scale)
        p = np.exp2(s2 - self.v(L, B * H * T).reshape(B, H, T, 1))
        f = lambda x: _bf16(x).astype(np.float64)                                      # noqa: E731
        dv = np.einsum("bhts,bhtd->bhsd", f(p), f(do))
        dp = np.einsum("bhtd,bhsd->bhts", f(do), f(v))
        ds = p * (dp - self.v(DD, B * H * T).reshape(B, H, T, 1))
        dq = scale * np.einsum("bhts,bhsd->bhtd", f(ds), f(k))
        dk = scale * np.einsum("bhts,bhtd->bhsd", f(ds), f(q))
        for view, val in ((DQ, dq), (DK, dk), (DV, dv)):
            self.v(view, B * T * H * hd)[:] = val.transpose(0, 2, 1, 3).astype(np.float32).reshape(-1)

    def fakeq(self, W, WQ, K, N, kind):
        from .lowbit import round_weights
        name = {v: k for k, v in KS.FAKEQ.items()}[kind]
        self.v(WQ, K * N)[:] = round_weights(self.v(W, K * N).reshape(K, N), name).reshape(-1)

    def _gemm(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes,
              epi=0, X2=None, act=0, a_act=None):
        if epi or a_act is not None:
            return False                                 # the composite path (the same numbers)
        ab, ao = A
        bb, bo = B
        cb, co = Cv
        mi, ki, ni = np.arange(M)[:, None], np.arange(K)[None, :], np.arange(N)[None, :]
        for z in range(nb1 * nb2):
            z1, z2 = divmod(z, nb2)
            oa = ao + z1 * sa[0] + z2 * sa[1]
            ob = bo + z1 * sb[0] + z2 * sb[1]
            oc = co + z1 * sc[0] + z2 * sc[1]
            a = ab.a[oa + mi * sam + ki * sak]
            bm = bb.a[ob + np.arange(K)[:, None] * sbk + ni * sbn]
            if self.tc and passes == 1:
                c = _bf16(a).astype(np.float64) @ _bf16(bm).astype(np.float64)
            else:
                c = a.astype(np.float64) @ bm.astype(np.float64)
            c = (np.float32(alpha) * c.astype(np.float32))
            if bias is not None:
                c = c + bias[0].a[bias[1]:bias[1] + N]
            idx = oc + np.arange(M)[:, None] * ldc + ni
            if beta:
                c = c + cb.a[idx]
            cb.a[idx] = c


class Gpu(_Ops):
    """The kernels on a real NVIDIA GPU."""

    def __init__(self, dev):
        self.dev = dev
        self.name = dev.name
        self.used = 0
        lib = dev.drv.lib
        dev.activate()
        lib.cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        for fname, at in (("cuMemcpyDtoDAsync_v2", [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]),
                          ("cuMemcpyPeerAsync", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                                 ctypes.c_size_t, ctypes.c_void_p])):
            try:
                getattr(lib, fname).argtypes = at
            except AttributeError:
                pass
        mod, first = dev.load("apsi-train-1", KS.ptx("main"))
        self.fn = {"fill": first}
        for name in KS.MAIN[1:]:
            f = ctypes.c_void_p()
            dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), f"cuModuleGetFunction({name})")
            self.fn[name] = f
        self.tc = False
        self.tc_why = None
        if tuple(dev.cc) >= (8, 0) and os.environ.get("APSI_TENSOR_CORES", "1").strip() not in ("0", "false", "no"):
            try:
                tmod, self.fn["gemm_tc"] = dev.load("apsi-train-tc-1", KS.ptx("tc"))
                f = ctypes.c_void_p()
                dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), tmod, b"gemm_tc3"), "cuModuleGetFunction")
                self.fn["gemm_tc3"] = f
                self.tc = True
            except Exception as e:                       # the float32 kernel then
                self.tc_why = f"{type(e).__name__}: {e}"
        elif tuple(dev.cc) < (8, 0):
            self.tc_why = f"compute capability {dev.cc[0]}.{dev.cc[1]}: bfloat16 tensor cores need 8.0 (Ampere)"
        self.tiled = False
        if self.tc and os.environ.get("APSI_TILED", "1").strip() not in ("0", "false", "no"):
            try:                                         # shared-memory tiles: the fast products
                tmod, self.fn["gemm_tkn"] = dev.load("apsi-train-tiled-1", KS.ptx("tiled"))
                for name in KS.TILED[1:] + tuple(e[0] for e in KS.EPI_KERNELS) + \
                        tuple(f"gemm_tmn_aa{j}" for j in KS.EPI_ACTS) + ("bf_lo", "to_bf16"):
                    f = ctypes.c_void_p()
                    dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), tmod, name.encode()), "cuModuleGetFunction")
                    self.fn[name] = f
                self.tiled = True
            except Exception as e:
                self.tc_why = f"tiled products unavailable: {type(e).__name__}: {e}"
        if self.tiled and os.environ.get("APSI_GEMM_VARIANT", "auto").strip().lower() != "a":
            # more builds of the same products: "x" loads first (2 blocks of 8 warps per SM), "w" 4 warps of 64 x 64
            # over slices of 16 (the shape of the fast GEMM libraries); the card times them per shape
            for tag, which in (("x", "tiled_x"), ("w", "tiled_w"), ("h", "tiled_h")):
                try:
                    xmod, first = dev.load(f"apsi-train-tiled-{tag}", KS.ptx(which))
                    xf = {f"gemm_tkn#{tag}": first}
                    for name in KS.TILED[1:] + tuple(e[0] for e in KS.EPI_KERNELS) + \
                            (tuple(f"gemm_tmn_aa{j}" for j in KS.EPI_ACTS) if tag != "h" else ()):
                        f = ctypes.c_void_p()
                        dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), xmod, name.encode()),
                                      "cuModuleGetFunction")
                        xf[f"{name}#{tag}"] = f
                    self.fn.update(xf)
                except Exception as e:                   # noqa: BLE001  (the other builds then)
                    self.variant_why = f"build {tag}: {type(e).__name__}: {e}"
        self.sms = int(getattr(dev, "sms", 0) or 28)
        self.flash = False
        if self.tc and os.environ.get("APSI_FLASH", "1").strip() not in ("0", "false", "no"):
            try:                                         # attention in one kernel each way
                fmod, self.fn["flash_dot"] = dev.load("apsi-train-flash-1", KS.ptx("flash"))
                for hd in KS.FLASH_HD:
                    for kind in "fbkq":
                        f = ctypes.c_void_p()
                        dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), fmod, f"flash_{kind}{hd}".encode()),
                                      "cuModuleGetFunction")
                        self.fn[f"flash_{kind}{hd}"] = f
                self.flash = True
            except Exception as e:
                self.tc_why = f"fused attention unavailable: {type(e).__name__}: {e}"
        self._calls = {}
        self._scr = {}
        self._occ = {}
        self._choice = {}
        self._wcache, self._wrange, self._wgen = {}, None, 0
        self.profile = None                              # {kernel: (seconds, launches)} while profiling
        self._pending = []
        self._lib = lib
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        try:
            lib.cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                                                                         ctypes.c_int, ctypes.c_size_t]
        except AttributeError:
            pass
        lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        self._launch = lib.cuLaunchKernel

    _CT = {"p": ctypes.c_uint64, "u": ctypes.c_uint32, "i": ctypes.c_int32, "f": ctypes.c_float}

    def _go(self, name, grid, block, *args):
        gx, gy, gz = (tuple(grid) + (1, 1))[:3] if isinstance(grid, tuple) else (grid, 1, 1)
        key = (name, gx, gy, gz, block, args)
        hit = self._calls.get(key)
        if hit is None:
            objs = [self._CT[k](v) for k, v in args]
            arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
            if len(self._calls) > 16384:
                self._calls.clear()
            hit = self._calls[key] = (objs, arr)
        self.dev.ensure()
        prof = self.profile is not None
        if prof:                                         # CUDA events around the launch: the card's own clock
            e0, e1 = self._event(), self._event()
            self._lib.cuEventRecord(e0, None)
        r = self._launch(self.fn[name], max(1, gx), max(1, gy), max(1, gz), block, 1, 1, 0, None, hit[1], None)
        if prof:
            self._lib.cuEventRecord(e1, None)
            label = name if not name.startswith("gemm") else \
                f"{name} {args[4][1]}x{args[5][1]}x{args[6][1]}" + (f" b{gy}" if gy > 1 else "") + (f" k/{gz}" if gz > 1 else "")
            self._pending.append((label, e0, e1))
        if r:
            self.dev.drv.check(r, f"launching {name}")

    def _event(self):
        e = ctypes.c_void_p()
        self.dev.drv.check(self._lib.cuEventCreate(ctypes.byref(e), 0), "cuEventCreate")
        return e

    def start_profile(self):
        self.profile, self._pending = {}, []

    def stop_profile(self):
        """{kernel: (seconds on the card, launches)} since start_profile()."""
        self.sync()
        out = {}
        ms = ctypes.c_float()
        for label, e0, e1 in self._pending:
            self._lib.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
            t, c = out.get(label, (0.0, 0))
            out[label] = (t + ms.value / 1000.0, c + 1)
            self._lib.cuEventDestroy_v2(e0)
            self._lib.cuEventDestroy_v2(e1)
        self.profile, self._pending = None, []
        return out

    def kernel_info(self):
        """Registers per thread and static shared memory of every kernel (what limits how many run at once)."""
        info = {}
        v = ctypes.c_int()
        for name, f in self.fn.items():
            regs = self._lib.cuFuncGetAttribute(ctypes.byref(v), 4, f) == 0 and v.value
            smem = self._lib.cuFuncGetAttribute(ctypes.byref(v), 1, f) == 0 and v.value
            local = self._lib.cuFuncGetAttribute(ctypes.byref(v), 3, f) == 0 and v.value   # spilled registers
            info[name] = {"registers": regs, "shared_bytes": smem, "local_bytes": local}
        return info

    def alloc(self, n):
        try:
            p = self.dev.alloc(4 * max(1, n))
        except Exception as e:                           # noqa: BLE001
            if "OUT_OF_MEMORY" not in str(e).upper() and "out of memory" not in str(e).lower():
                raise
            raise TrainError(f"the card ran out of memory ({self.used / 2 ** 30:.2f} GB in use, "
                             f"{4 * n / 2 ** 20:.0f} MB more asked, {self.dev.memory / 2 ** 30:.1f} GB on the card): "
                             "a smaller batch_size (with accumulate= for the same step), fewer or narrower layers, a "
                             "shorter context, or more cards "
                             "(device=\"cuda:all\" splits the model across them)") from None
        self.used += 4 * n
        return Buf(p, n)

    def free(self, b):
        self.used -= 4 * b.n
        self.dev.free(b.ptr)
        self._calls.clear()

    RING = 64 << 20                                      # pinned staging memory for copies to the card

    def put(self, b, a, off=0):
        """Host -> card without waiting: the values go through pinned memory and a copy queued behind the kernels
        already launched, so the processor prepares the next batch while the card works (the order on the card is
        the order of the calls, as before). Large arrays (over a quarter of the ring) copy directly."""
        a = np.ascontiguousarray(a)
        if a.dtype not in (np.float32, np.int32, np.uint32):
            a = a.astype(np.float32)
        n = a.nbytes
        if n and n <= self.RING // 4 and self._ring_ready():
            self._put_async(b.ptr + 4 * off, a, n)
            return
        self.dev.to_device(b.ptr + 4 * off, a.ctypes.data, n)

    def _ring_ready(self):
        r = self.__dict__.get("_ring")
        if r is None:
            r = False
            if os.environ.get("APSI_GPU_ASYNC_COPY", "1").strip().lower() not in ("0", "false", "no", "off"):
                try:
                    lib = self._lib
                    lib.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
                    lib.cuMemcpyHtoDAsync_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
                                                         ctypes.c_void_p]
                    lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
                    lib.cuEventQuery.argtypes = [ctypes.c_void_p]
                    self.dev.ensure()
                    ptr = ctypes.c_void_p()
                    t0 = time.perf_counter()
                    self.dev.drv.check(lib.cuMemHostAlloc(ctypes.byref(ptr), self.RING, 1), "cuMemHostAlloc")
                    r = {"ptr": ptr.value, "head": 0, "q": [], "free": [], "made_s": time.perf_counter() - t0}
                except Exception:                        # noqa: BLE001  (no pinned memory: plain copies)
                    r = False
            self._ring = r
        return bool(r)

    def _put_async(self, dptr, a, n):
        r, lib = self._ring, self._lib
        size = (n + 255) & ~255
        start = r["head"] if r["head"] + size <= self.RING else 0
        end = start + size
        q = r["q"]
        last = -1                                        # the newest copy still using any of [start, end)
        for i, (s0, e0, _) in enumerate(q):
            if s0 < end and start < e0:
                last = i
        if last >= 0:                                    # copies run in order: waiting for it covers the older ones
            self.dev.drv.check(lib.cuEventSynchronize(q[last][2]), "cuEventSynchronize")
            r["free"].extend(ev for _, _, ev in q[:last + 1])
            del q[:last + 1]
        while len(q) > 64 and lib.cuEventQuery(q[0][2]) == 0:     # finished copies leave the queue
            r["free"].append(q.pop(0)[2])
        ctypes.memmove(r["ptr"] + start, a.ctypes.data, n)
        self.dev.ensure()
        self.dev.drv.check(lib.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dptr), ctypes.c_void_p(r["ptr"] + start),
                                                    ctypes.c_size_t(n), None), "cuMemcpyHtoDAsync")
        ev = r["free"].pop() if r["free"] else self._event_light()
        self.dev.drv.check(lib.cuEventRecord(ev, None), "cuEventRecord")
        q.append((start, end, ev))
        r["head"] = end

    def drop_ring(self):
        """Give the pinned staging memory back (after the copies in flight finish)."""
        r = self.__dict__.pop("_ring", None)
        if r:
            self.sync()
            for _, _, ev in r["q"]:
                self._lib.cuEventDestroy_v2(ev)
            for ev in r["free"]:
                self._lib.cuEventDestroy_v2(ev)
            self._lib.cuMemFreeHost(ctypes.c_void_p(r["ptr"]))

    def _event_light(self):
        e = ctypes.c_void_p()
        self.dev.drv.check(self._lib.cuEventCreate(ctypes.byref(e), 2), "cuEventCreate")     # no timing
        return e

    def get(self, b, n, off=0):
        out = np.empty(n, np.float32)
        self.dev.to_host(out.ctypes.data, b.ptr + 4 * off, out.nbytes)
        return out

    def sync(self):
        self.dev.ensure()
        self.dev.drv.check(self.dev.drv.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    @staticmethod
    def p(view):
        return ("p", view[0].ptr + 4 * view[1]) if view is not None else ("p", 0)

    @staticmethod
    def blocks(n):
        return max(1, min((n + 255) // 256, 65536))

    def fill(self, view, n, value):
        self._go("fill", self.blocks(n), 256, self.p(view), ("u", n), ("f", value))

    def axpy(self, Y, X, n, a=1.0):
        self._go("axpy", self.blocks(n), 256, self.p(Y), self.p(X), ("u", n), ("f", a))

    def pull(self, dst, src_ops, src, n):
        """dst (on this card) <- src (on src_ops' card): n float32, card to card (NVLink / PCIe peer access) when the
        driver allows it, else through the processor's memory. Queued on this card after its earlier work; the
        source must be finished (the caller syncs it)."""
        self.dev.ensure()
        dptr, sptr = self.p(dst)[1], src_ops.p(src)[1]
        self._wgen += 1                                  # (the weights may be what changes)
        if src_ops is self or src_ops.dev is self.dev:
            self.dev.drv.check(self._lib.cuMemcpyDtoDAsync_v2(ctypes.c_uint64(dptr), ctypes.c_uint64(sptr),
                                                              ctypes.c_size_t(4 * n), None), "cuMemcpyDtoDAsync")
            return
        try:
            r = self._lib.cuMemcpyPeerAsync(ctypes.c_uint64(dptr), self.dev.ctx, ctypes.c_uint64(sptr),
                                            src_ops.dev.ctx, ctypes.c_size_t(4 * n), None)
        except AttributeError:
            r = 1
        if r != 0:                                       # no peer copy: through the host
            self.put(dst[0], src_ops.get(src[0], n, src[1]), dst[1])

    def add3(self, Z, X, Y, n):
        self._go("add3", self.blocks(n), 256, self.p(Z), self.p(X), self.p(Y), ("u", n))

    def mul3(self, Z, X, Y, n):
        self._go("mul3", self.blocks(n), 256, self.p(Z), self.p(X), self.p(Y), ("u", n))

    def act_f(self, X, Y, n, kind):
        self._go("act_f", self.blocks(n), 256, self.p(X), self.p(Y), ("u", n), ("u", kind))

    def act_b(self, X, DY, DX, n, kind):
        self._go("act_b", self.blocks(n), 256, self.p(X), self.p(DY), self.p(DX), ("u", n), ("u", kind))

    def colsum(self, DY, X, MEAN, RSTD, DG, DB, M, N, mode):
        """Column sums with no atomic additions: blocks of 32 columns x 8 lanes of rows; enough chunks of rows to
        fill the card (at most 64 rows a lane), each chunk's sums added in order by the last chunk to finish, so the
        result is the same every run (deterministic training) in one launch."""
        cb = -(-N // 32)
        chunks = max(-(-M // 512), min(-(-4 * self.sms // cb), -(-M // 32)), 1)
        chunks = min(chunks, 65535)
        rows = -(-M // chunks)
        chunks = -(-M // rows)
        part = chunks > 1
        WB = self._scratch("colsum_wb", chunks * N if part and mode != 2 else 1)
        WG = self._scratch("colsum_wg", chunks * N if part and mode != 0 else 1)
        self._go("colsum", (cb, chunks), 256, self.p(DY), self.p(X), self.p(MEAN),
                 self.p(RSTD), self.p(DG), self.p(DB), ("u", M), ("u", N), ("u", rows), ("u", mode), self.p(WB),
                 self.p(WG), ("u", 1 if part else 0), self._cnt())

    def ln_f(self, X, Gm, Bt, Y, MEAN, RSTD, M, N, eps, rms):
        self._go("ln_f", -(-M // 8), 256, self.p(X), self.p(Gm), self.p(Bt), self.p(Y), self.p(MEAN), self.p(RSTD),
                 ("u", M), ("u", N), ("f", eps), ("u", 1 if rms else 0))

    def ln_b(self, X, Gm, DY, MEAN, RSTD, DX, M, N, rms, RES=None):
        self._go("ln_b", -(-M // 8), 256, self.p(X), self.p(Gm), self.p(DY), self.p(MEAN), self.p(RSTD), self.p(DX),
                 ("u", M), ("u", N), ("u", 1 if rms else 0), self.p(RES), ("u", 1 if RES is not None else 0))

    def embed_f(self, E, IDS, Y, M, D):
        self._go("embed_f", self.blocks(M * D), 256, self.p(E), self.p(IDS), self.p(Y), ("u", M), ("u", D))

    def embed_b(self, DY, IDS, DE, M, D, V):
        """DE[IDS[i]] += DY[i] with no atomic additions: the positions of each row in order (made on the processor
        from the M ids), one thread per (row, column) sums them: the same result every run."""
        if M <= 0:
            return
        ids = np.ascontiguousarray(self.get(IDS[0], M, IDS[1])).view(np.int32).astype(np.int64)
        order = np.argsort(ids, kind="stable")
        rows, cnt = np.unique(ids[order], return_counts=True)
        off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
        U = len(rows)
        pos, offb, rowb = self._scratch("eb_pos", M), self._scratch("eb_off", U + 1), self._scratch("eb_rows", U)
        self.put(pos[0], order.astype(np.int32), pos[1])
        self.put(offb[0], off, offb[1])
        self.put(rowb[0], rows.astype(np.int32), rowb[1])
        self._go("embed_bs", self.blocks(U * D), 256, self.p(DY), self.p(pos), self.p(offb), self.p(rowb),
                 self.p(DE), ("u", U), ("u", D))

    def pos_f(self, X, P, Y, M, D, T):
        self._go("pos_f", self.blocks(M * D), 256, self.p(X), self.p(P), self.p(Y), ("u", M), ("u", D), ("u", T))

    def pos_b(self, DY, DP, M, D, T):
        self._go("pos_b", self.blocks(T * D), 256, self.p(DY), self.p(DP), ("u", M), ("u", D), ("u", T))

    def softmax_f(self, S, P, rows, n, T, causal):
        self._go("softmax_f", rows, 256, self.p(S), self.p(P), ("u", rows), ("u", n), ("u", T), ("u", 1 if causal else 0))

    def softmax_b(self, P, DP, DS, rows, n):
        self._go("softmax_b", rows, 256, self.p(P), self.p(DP), self.p(DS), ("u", rows), ("u", n))

    def ce(self, L, LAB, DL, LOSS, M, V, wg, wl):
        part = self._scratch("loss_part", max(1, M))
        self._go("ce", M, 256, self.p(L), self.p(LAB), self.p(DL), self.p(LOSS), ("u", M), ("u", V), ("f", wg),
                 ("f", wl), self.p(part), self._cnt())

    def loss_ew(self, Y, T, W, DY, LOSS, n, wg, wl, kind):
        nb = self.blocks(n)
        part = self._scratch("loss_part", nb)
        self._go("loss_ew", nb, 256, self.p(Y), self.p(T), self.p(W if W is not None else Y), self.p(DY),
                 self.p(LOSS), ("u", n), ("f", wg), ("f", wl), ("u", kind), ("u", 1 if W is not None else 0),
                 self.p(part), self._cnt())

    def mse(self, Y, T, DY, LOSS, n, wg, wl):
        nb = self.blocks(n)
        part = self._scratch("loss_part", nb)
        self._go("mse", nb, 256, self.p(Y), self.p(T), self.p(DY), self.p(LOSS), ("u", n), ("f", wg),
                 ("f", wl), self.p(part), self._cnt())

    def sumsq(self, X, n, ACC):
        nb = min(self.blocks(n), 1024)
        part = self._scratch("sumsq_part", nb)
        self._go("sumsq", nb, 256, self.p(X), ("u", n), self.p(ACC), self.p(part), self._cnt())

    def adam(self, Pm, Gr, Mm, Vm, SQ, n, n_decay, lr, b1, b2, eps, c1, c2, wd, clip, COMP=None):
        self._wgen += 1
        self._go("adam", self.blocks(n), 256, self.p(Pm), self.p(Gr), self.p(Mm), self.p(Vm), self.p(SQ), ("u", n),
                 ("u", n_decay), ("f", lr), ("f", b1), ("f", b2), ("f", eps), ("f", c1), ("f", c2), ("f", wd),
                 ("f", clip), self.p(COMP if COMP is not None else Pm), ("u", 1 if COMP is not None else 0))

    def sgd(self, Pm, Gr, VEL, SQ, n, lr, mom, clip, COMP=None):
        self._wgen += 1
        self._go("sgd", self.blocks(n), 256, self.p(Pm), self.p(Gr), self.p(VEL), self.p(SQ), ("u", n), ("f", lr),
                 ("f", mom), ("f", clip), self.p(COMP if COMP is not None else Pm), ("u", 1 if COMP is not None else 0))

    def drop_f(self, X, Y, MASK, n, p, seed):
        self._go("drop_f", self.blocks(n), 256, self.p(X), self.p(Y), self.p(MASK), ("u", n), ("f", p),
                 ("i", int(np.int32(np.uint32(seed & 0xFFFFFFFF)))))

    def spmm(self, X, COL, VAL, VPERM, OFF, Y, N, H, F, add=False):
        mode = (1 if VAL is not None else 0) | (2 if VPERM is not None else 0) | (4 if add else 0)
        self._go("spmm", self.blocks(N * H * F), 256, self.p(X), self.p(COL), self.p(VAL if VAL is not None else X),
                 self.p(VPERM if VPERM is not None else COL), self.p(OFF), self.p(Y), ("u", N), ("u", H), ("u", F),
                 ("u", mode))

    def edge_dot(self, DY, X, ROW, COL, DA, E, H, F):
        self._go("edge_dot", self.blocks(E * H), 256, self.p(DY), self.p(X), self.p(ROW), self.p(COL), self.p(DA),
                 ("u", E), ("u", H), ("u", F))

    def seg_softmax(self, S, OFF, A, N, H, slope):
        self._go("seg_softmax", self.blocks(N * H), 256, self.p(S), self.p(OFF), self.p(A), ("u", N), ("u", H),
                 ("f", slope))

    def seg_softmax_b(self, S, A, DA, OFF, DS, N, H, slope):
        self._go("seg_softmax_b", self.blocks(N * H), 256, self.p(S), self.p(A), self.p(DA), self.p(OFF), self.p(DS),
                 ("u", N), ("u", H), ("f", slope))

    def dot_acc(self, X, Y, n, ACC):
        nb = min(self.blocks(n), 1024)
        part = self._scratch("dot_part", nb)
        self._go("dot_acc", nb, 256, self.p(X), self.p(Y), ("u", n), self.p(ACC), self.p(part), self._cnt())

    def scale_add(self, Z, X, S, n, c):
        self._go("scale_add", self.blocks(n), 256, self.p(Z), self.p(X), self.p(S), ("u", n), ("f", c))

    def acc2(self, HI, LO, Gr, n, clear=True):
        self._go("acc2", self.blocks(n), 256, self.p(HI), self.p(LO), self.p(Gr), ("u", n), ("u", 1 if clear else 0))

    def tanh_s(self, OUT, W):
        self._go("tanh_s", 1, 32, self.p(OUT), self.p(W))

    def axpy_s(self, Y, X, n, S):
        self._go("axpy_s", self.blocks(n), 256, self.p(Y), self.p(X), ("u", n), self.p(S))

    def gate_g(self, G, DG, TH):
        self._go("gate_g", 1, 32, self.p(G), self.p(DG), self.p(TH))

    def ew1(self, X, Y, n, kind):
        self._go("ew1", self.blocks(n), 256, self.p(X), self.p(Y), ("u", n), ("u", kind))

    def ew2(self, Z, X, Y, n, kind):
        self._go("ew2", self.blocks(n), 256, self.p(Z), self.p(X), self.p(Y), ("u", n), ("u", kind))

    def fin2(self, HI, LO, Gr, n):
        self._go("fin2", self.blocks(n), 256, self.p(HI), self.p(LO), self.p(Gr), ("u", n))

    def cols(self, X, Y, rows, w, ld, off, back):
        self._go("cols", self.blocks(rows * w), 256, self.p(X), self.p(Y), ("u", rows), ("u", w), ("u", ld),
                 ("u", off), ("u", 1 if back else 0))

    def taylor_f(self, Z, Y, N, F, d, kind):
        self._go("taylor_f", self.blocks(N * F), 256, self.p(Z), self.p(Y), ("u", N), ("u", F), ("u", d), ("u", kind))

    def taylor_b(self, Z, DY, DZ, N, F, d, kind):
        self._go("taylor_b", self.blocks(N * F), 256, self.p(Z), self.p(DY), self.p(DZ), ("u", N), ("u", F), ("u", d),
                 ("u", kind))

    def rowbias(self, Y, B, rows, F):
        self._go("rowbias", self.blocks(rows * F), 256, self.p(Y), self.p(B), ("u", rows), ("u", F))

    def _rec(self, name, B, H, *bufs, T, t):
        self._go(name, self.blocks(B * H), 256, *[self.p(x) for x in bufs], ("u", B), ("u", H), ("u", T), ("u", t))

    def lstm_f(self, Z, CS, HS, B, H, T, t):
        self._rec("lstm_f", B, H, Z, CS, HS, T=T, t=t)

    def lstm_b(self, Z, CS, DY, DHN, DCN, DZ, B, H, T, t):
        self._rec("lstm_b", B, H, Z, CS, DY, DHN, DCN, DZ, T=T, t=t)

    def gru_a(self, ZR, HS, RH, B, H, T, t):
        self._rec("gru_a", B, H, ZR, HS, RH, T=T, t=t)

    def gru_b(self, CX, ZR, HS, B, H, T, t):
        self._rec("gru_b", B, H, CX, ZR, HS, T=T, t=t)

    def gru_b1(self, CX, ZR, HS, DY, DHN, DCX, DZR, DHP, B, H, T, t):
        self._rec("gru_b1", B, H, CX, ZR, HS, DY, DHN, DCX, DZR, DHP, T=T, t=t)

    def gru_b2(self, DRH, ZR, HS, DZR, DHP, B, H, T, t):
        self._rec("gru_b2", B, H, DRH, ZR, HS, DZR, DHP, T=T, t=t)

    def im2col(self, X, COL, N, C, H, W, kh, kw, s, p, OH, OW):
        self._go("im2col", self.blocks(N * OH * OW * C * kh * kw), 256, self.p(X), self.p(COL),
                 *[("u", v) for v in (N, C, H, W, kh, kw, s, p, OH, OW)])

    def col2im(self, DCOL, DX, N, C, H, W, kh, kw, s, p, OH, OW):
        self._go("col2im", self.blocks(N * C * H * W), 256, self.p(DCOL), self.p(DX),
                 *[("u", v) for v in (N, C, H, W, kh, kw, s, p, OH, OW)])

    def perm_nc(self, Z, Y, BIAS, N, Cc, S, back):
        self._go("perm_nc", self.blocks(N * Cc * S), 256, self.p(Z), self.p(Y), self.p(BIAS if BIAS is not None else Z),
                 ("u", N), ("u", Cc), ("u", S), ("u", 1 if back else 0), ("u", 1 if BIAS is not None else 0))

    def pool_f(self, X, Y, NC, H, W, kk, mode):
        self._go("pool_f", self.blocks(NC * (H // kk) * (W // kk)), 256, self.p(X), self.p(Y), ("u", NC), ("u", H),
                 ("u", W), ("u", kk), ("u", mode))

    def pool_b(self, X, Y, DY, DX, NC, H, W, kk, mode):
        self._go("pool_b", self.blocks(NC * H * W), 256, self.p(X), self.p(Y), self.p(DY), self.p(DX), ("u", NC),
                 ("u", H), ("u", W), ("u", kk), ("u", mode))

    def topk_gate(self, Lg, IDX, GATE, M, E, k):
        self._go("topk_gate", self.blocks(M), 256, self.p(Lg), self.p(IDX), self.p(GATE), ("u", M), ("u", E), ("u", k))

    def topk_gate_b(self, GATE, DG, IDX, DL, M, E, k):
        self._go("topk_gate_b", self.blocks(M), 256, self.p(GATE), self.p(DG), self.p(IDX), self.p(DL), ("u", M),
                 ("u", E), ("u", k))

    def colmul(self, X, S, Y, n, D):
        self._go("colmul", self.blocks(n), 256, self.p(X), self.p(S), self.p(Y), ("u", n), ("u", D))

    def rope(self, X, CS, SN, rows, hd, H, T, sign):
        self._go("rope", self.blocks(rows * (hd // 2)), 256, self.p(X), self.p(CS), self.p(SN), ("u", rows), ("u", hd),
                 ("u", H), ("u", T), ("f", sign))

    def gather(self, SRC, IDX, DST, rows, width, start):
        self._go("gather", self.blocks(rows * width), 256, self.p(SRC), self.p(IDX), self.p(DST), ("u", rows),
                 ("u", width), ("u", start))

    def flash_fwd(self, Q, K, V, O, L, B, T, H, hd, causal, scale):
        self._go(f"flash_f{hd}", (-(-T // KS.BQ), B * H), 128, self.p(Q), self.p(K), self.p(V), self.p(O), self.p(L),
                 ("u", T), ("u", H * hd), ("u", H), ("f", scale * 1.4426950408889634), ("u", 1 if causal else 0))

    def flash_dot(self, DO, O, DD, B, T, H, hd):
        rows = B * H * T
        self._go("flash_dot", -(-rows // 4), 128, self.p(DO), self.p(O), self.p(DD), ("u", T), ("u", H * hd),
                 ("u", H), ("u", hd), ("u", rows))

    def flash_bwd(self, Q, K, V, DO, L, DD, DQ, DK, DV, B, T, H, hd, causal, scale):
        if _deterministic():                             # dK, dV per key block; dQ per query block: no atomics
            grid, s2 = (-(-T // KS.BQ), B * H), scale * 1.4426950408889634
            c = 1 if causal else 0
            self._go(f"flash_k{hd}", grid, 128, self.p(Q), self.p(K), self.p(V), self.p(DO), self.p(L), self.p(DD),
                     self.p(DQ), self.p(DK), self.p(DV), ("u", T), ("u", H * hd), ("u", H), ("f", scale), ("f", s2),
                     ("u", c))
            self._go(f"flash_q{hd}", grid, 128, self.p(Q), self.p(K), self.p(V), self.p(DO), self.p(L), self.p(DD),
                     self.p(DQ), ("u", T), ("u", H * hd), ("u", H), ("f", scale), ("f", s2), ("u", c))
            return
        self.fill(DQ, B * T * H * hd, 0.0)
        self._go(f"flash_b{hd}", (-(-T // KS.BQ), B * H), 128, self.p(Q), self.p(K), self.p(V), self.p(DO),
                 self.p(L), self.p(DD), self.p(DQ), self.p(DK), self.p(DV), ("u", T), ("u", H * hd), ("u", H),
                 ("f", scale), ("f", scale * 1.4426950408889634), ("u", 1 if causal else 0))

    def fakeq(self, W, WQ, K, N, kind):
        self._go("fakeq", self.blocks(((K + 31) // 32) * N), 256, self.p(W), self.p(WQ), ("u", K), ("u", N),
                 ("u", kind))

    def _gemm_padded(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, alpha, beta):
        """A product whose sizes are not multiples of 4 (a SwiGLU width of 2730, say) on the tiled tensor-core
        kernels, through zero-padded copies, instead of the slower general kernel. Same values up to the order of the
        sums (zeros add nothing); the same bits every run."""
        if os.environ.get("APSI_GEMM_PAD", "1").strip().lower() in ("0", "false", "no", "off"):
            return False
        return _gemm_padded_impl(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, alpha, beta)

    def _tiled_ok(self, A, B, M, N, K, sam, sak, sbk, sbn, sa, sb):
        def q(*v):
            return all(int(x) % 4 == 0 for x in v)
        if sak == 1:
            if not q(K, sam, A[1], *sa):
                return None
            ka = "k"
        elif sam == 1:
            if not q(M, sak, A[1], *sa):
                return None
            ka = "m"
        else:
            return None
        if sbn == 1:
            if not q(N, sbk, B[1], *sb):
                return None
            kb = "n"
        elif sbk == 1:
            if not q(K, sbn, B[1], *sb):
                return None
            kb = "k"
        else:
            return None
        return f"gemm_t{ka}{kb}"

    @staticmethod
    def _threads(name):
        return 128 if name.endswith("#w") else 256

    def _per_sm(self, name):
        """Blocks of this kernel that fit on one multiprocessor at once (registers, shared memory)."""
        n = self._occ.get(name)
        if n is None:
            v = ctypes.c_int()
            try:
                ok = self._lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(v), self.fn[name],
                                                                          self._threads(name),
                                                                          ctypes.c_size_t(0)) == 0
            except AttributeError:
                ok = False
            n = self._occ[name] = max(1, v.value) if ok else 2
        return n

    def _h_ready(self, name, args, M, N, K, nb):
        """Can the bfloat16-operand build run this product? (dense operands, dimensions in 8s, no bias gradient)"""
        if f"{name}#h" not in self.fn or nb != 1 or args[27][1] or "to_bf16" not in self.fn:
            return False
        sam, sak, sbk, sbn = (a[1] for a in args[7:11])
        da = (sak == 1 and sam == K and K % 8 == 0) or (sam == 1 and sak == M and M % 8 == 0)
        db = (sbn == 1 and sbk == N and N % 8 == 0) or (sbk == 1 and sbn == K and K % 8 == 0)
        return da and db and args[0][1] % 8 == 0 and args[1][1] % 8 == 0

    def weights_are(self, buf, n):
        """The network's weights (their bfloat16 copies are made once per update, not once per product)."""
        for bf in self._wcache.values():
            self.free(bf[1])
        self._wcache = {}
        self._wrange = (buf.ptr, buf.ptr + 4 * n) if buf is not None else None

    def _h_args(self, args, M, N, K):
        """The operands rounded to bfloat16 (one pass each), and the arguments that point there. A weight matrix is
        rounded once after each update and reused by every product until the next one (forward and backward)."""
        out = list(args)
        for i, n, key in ((0, M * K, "h_a"), (1, K * N, "h_b")):
            ptr = args[i][1]
            wr = self._wrange
            if wr is not None and wr[0] <= ptr < wr[1]:
                hit = self._wcache.get((ptr, n))
                if hit is None or hit[0] != self._wgen:
                    dst = hit[1] if hit is not None else self.alloc((n + 1) // 2)
                    self._go("to_bf16", self.blocks((n + 1) // 2), 256, args[i], self.p((dst, 0)), ("u", n))
                    self._wcache[(ptr, n)] = hit = (self._wgen, dst)
                out[i] = self.p((hit[1], 0))
                continue
            dst = self._scratch(key, (n + 1) // 2)
            self._go("to_bf16", self.blocks((n + 1) // 2), 256, args[i], self.p(dst), ("u", n))
            out[i] = self.p(dst)
        return out

    def _variant(self, name, grid, args, M, N, ldc, nb):
        """Which build of a tiled product runs: the one the card itself measured faster for this shape (each is
        timed twice on scratch outputs the first time the shape comes; APSI_GEMM_VARIANT=a or x fixes it)."""
        K = args[6][1]
        cands = [name] + [f"{name}#{t}" for t in ("x", "w") if f"{name}#{t}" in self.fn]
        if self._h_ready(name, args, M, N, K, nb):
            cands.append(f"{name}#h")
        force = os.environ.get("APSI_GEMM_VARIANT", "auto").strip().lower()
        if force in ("x", "w", "h"):
            return f"{name}#{force}" if f"{name}#{force}" in cands else name
        if force == "a" or len(cands) == 1 or nb != 1:
            return name
        key = (name, grid, tuple(a for a in args if a[0] != "p"))
        pick = self._choice.get(key)
        if pick is not None:
            return pick
        prof, self.profile = self.profile, None
        targs = list(args)
        targs[2] = self.p(self._scratch("tune_c", M * ldc + N))
        if targs[23][1]:
            targs[23] = self.p(self._scratch("tune_x", max(1, grid[2] if len(grid) > 2 else 1) * M * ldc + N))
        if targs[26][1]:
            targs[26] = self.p(self._scratch("tune_db", N))
        best, best_t = name, None
        ms = ctypes.c_float()
        for cand in cands + cands:
            e0, e1 = self._event(), self._event()
            self._lib.cuEventRecord(e0, None)
            self._go(cand, grid, self._threads(cand), *(self._h_args(targs, M, N, K) if cand.endswith("#h") else targs))
            self._lib.cuEventRecord(e1, None)
            self._lib.cuEventSynchronize(e1)
            self._lib.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
            self._lib.cuEventDestroy_v2(e0)
            self._lib.cuEventDestroy_v2(e1)
            if best_t is None or ms.value < best_t:
                best, best_t = cand, ms.value
        self.profile = prof
        self._choice[key] = best
        return best

    def _cnt(self):
        """Counters of the kernels whose last block finishes the sum (zero between launches: that block resets its
        own), one per column block: 65536 of them, made once."""
        c = self.__dict__.get("_counters")
        if c is None:
            c = self._counters = (self.alloc(65536), 0)
            self.fill(c, 65536, 0.0)
        return self.p(c)

    def _scratch(self, key, n):
        bf = self._scr.get(key)
        if bf is None or bf.n < n:
            if bf is not None:
                self.free(bf)
            bf = self._scr[key] = self.alloc(n)
        return (bf, 0)

    def _gemm3(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, sa, sb, sc, alpha, beta, epi, X2, act):
        """float32 products on the shared-memory tensor-core kernels: hi(A) hi(B) + lo(A) hi(B) + hi(A) lo(B)."""
        dense_a = (sak == 1 and sam == K) or (sam == 1 and sak == M)
        dense_b = (sbn == 1 and sbk == N) or (sbk == 1 and sbn == K)
        if not (dense_a and dense_b and "bf_lo" in self.fn) or \
                self._tiled_ok(A, B, M, N, K, sam, sak, sbk, sbn, sa, sb) is None:
            return False
        if epi:
            name = self._tiled_ok(A, B, M, N, K, sam, sak, sbk, sbn, sa, sb)
            if ldc != N or f"{name}_e{epi}" + (f"a{act}" if epi in (1, 2) else "") not in self.fn:
                return False
        lo_a, lo_b = self._scratch("lo_a", M * K), self._scratch("lo_b", K * N)
        self._go("bf_lo", self.blocks(M * K), 256, self.p(A), self.p(lo_a), ("u", M * K))
        self._go("bf_lo", self.blocks(K * N), 256, self.p(B), self.p(lo_b), ("u", K * N))
        common = (M, N, K, sam, sak, sbk, sbn, ldc, 1, 1, sa, sb, sc, alpha)
        self._gemm(A, B, Cv, bias, *common, beta, 1)
        self._gemm(lo_a, B, Cv, None, *common, True, 1)
        return self._gemm(A, lo_b, Cv, None, *common, True, 1, epi=epi, X2=X2, act=act)

    def _gemm(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes,
              epi=0, X2=None, act=0, a_act=None):
        if passes == 3 and self.tiled and a_act is None and nb1 * nb2 == 1 and \
                os.environ.get("APSI_FP32_TILED", "0").strip() in ("1", "true", "yes") and \
                self._gemm3(A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, sa, sb, sc, alpha, beta, epi, X2, act):
            return True
        name = self._tiled_ok(A, B, M, N, K, sam, sak, sbk, sbn, sa, sb) if self.tiled and passes == 1 else None
        if a_act is not None:
            if name != "gemm_tmn" or f"gemm_tmn_aa{a_act}" not in self.fn or epi:
                return False
            name = f"gemm_tmn_aa{a_act}"
        if epi:
            ename = f"{name}_e{epi}" + (f"a{act}" if epi in (1, 2) else "") if name else None
            if name is None or ldc != N or nb1 * nb2 != 1 or ename not in self.fn:
                return False
            name = ename
        if name is not None:
            tiles = ((M + KS.BM - 1) // KS.BM) * ((N + KS.BN - 1) // KS.BN)
            nz, kchunk = 1, K
            blocks = tiles * nb1 * nb2
            # few output tiles and a long k (gradients of weights): split k so the blocks fill whole waves of the
            # card (2 blocks per multiprocessor): 9 tiles split 7 ways is 63 blocks = 2 waves of 56, the second
            # almost empty; split 6 ways it is one full wave
            if blocks < 4 * self.sms and nb1 * nb2 == 1 and K >= 512 and (beta or ldc == N) and not epi:
                want = _split_k(blocks, K, self._per_sm(name) * self.sms)
                if want > 1:
                    kchunk = -(-(-(-K // want)) // KS.BK) * KS.BK
                    nz = -(-K // kchunk)
                    if nz > 1 and not beta:
                        self.fill(Cv, M * N, 0.0)
            db = self._db if name.startswith("gemm_tmn") and sbn == 1 and nz == 1 else None
            if nz > 1:                                   # the parts of k into a workspace, added in order after
                X2 = self._scratch("splitk", nz * M * ldc)
            args = [self.p(A), self.p(B), self.p(Cv), self.p(bias), ("u", M),
                    ("u", N), ("u", K), ("u", sam), ("u", sak), ("u", sbk), ("u", sbn), ("u", ldc), ("u", nb2),
                    ("u", sa[0]), ("u", sa[1]), ("u", sb[0]), ("u", sb[1]), ("u", sc[0]), ("u", sc[1]),
                    ("f", alpha), ("u", 1 if beta else 0), ("u", 1 if bias is not None else 0), ("u", kchunk),
                    self.p(X2), ("u", epi), ("u", act), self.p(db), ("u", 1 if db is not None else 0), self._cnt()]
            grid = (tiles, nb1 * nb2, nz)
            use = self._variant(name, grid, args, M, N, ldc, nb1 * nb2)
            if use.endswith("#h"):
                args = self._h_args(args, M, N, K)
            self._go(use, grid, self._threads(use), *args)         # split k: its last part adds the parts
            if db is not None:
                self._db = None                          # taken
            return True
        if passes == 1 and self.tiled and a_act is None and not epi and nb1 * nb2 == 1 and ldc == N and \
                M * N * K >= (1 << 21) and self._gemm_padded(A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, alpha, beta):
            return True
        tiles = ((M + KS.TILE - 1) // KS.TILE) * ((N + KS.TILE - 1) // KS.TILE)
        args = (self.p(A), self.p(B), self.p(Cv), self.p(bias), ("u", M), ("u", N), ("u", K), ("u", sam),
                ("u", sak), ("u", sbk), ("u", sbn), ("u", ldc), ("u", nb2), ("u", sa[0]), ("u", sa[1]),
                ("u", sb[0]), ("u", sb[1]), ("u", sc[0]), ("u", sc[1]), ("f", alpha), ("u", 1 if beta else 0),
                ("u", 1 if bias is not None else 0))
        if self.tc:
            self._go("gemm_tc" if passes == 1 else "gemm_tc3", (tiles, nb1 * nb2), 128, *args)
        else:
            self._go("gemm_simt", (tiles, nb1 * nb2), 256, *args)


def _r4(n):
    return (int(n) + 3) // 4 * 4


def _pad_into(ops, key, src, rows, cols, rows_p, cols_p):
    """src (rows x cols, dense) into a zeroed scratch buffer of rows_p x cols_p: the tiled kernels read 16 bytes at a
    time and need sizes that are multiples of 4; zeros add nothing to a product."""
    dst = ops._scratch(key, rows_p * cols_p)
    ops.fill(dst, rows_p * cols_p, 0.0)
    ops.cols(src, dst, rows, cols, cols_p, 0, False)
    return dst


def _gemm_padded_impl(ops, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, alpha, beta):
    if not ((sak == 1 and sam == K) or (sam == 1 and sak == M)) or not ((sbn == 1 and sbk == N) or
                                                                      (sbk == 1 and sbn == K)):
        return False
    Mp, Np, Kp = _r4(M), _r4(N), _r4(K)
    if (Mp, Np, Kp) == (M, N, K):
        return False
    if sak == 1:                                       # A stored M x K
        Ap = _pad_into(ops, "pad_a", A, M, K, Mp, Kp)
        a_str = (Kp, 1)
    else:                                              # A stored K x M (a transposed view)
        Ap = _pad_into(ops, "pad_a", A, K, M, Kp, Mp)
        a_str = (1, Mp)
    if sbn == 1:                                       # B stored K x N
        Bp = _pad_into(ops, "pad_b", B, K, N, Kp, Np)
        b_str = (Np, 1)
    else:                                              # B stored N x K
        Bp = _pad_into(ops, "pad_b", B, N, K, Np, Kp)
        b_str = (1, Kp)
    if beta:
        Cp = _pad_into(ops, "pad_c", Cv, M, N, Mp, Np)
    else:
        Cp = ops._scratch("pad_c", Mp * Np)
    biasp = _pad_into(ops, "pad_bias", bias, 1, N, 1, Np) if bias is not None else None
    if ops._tiled_ok(Ap, Bp, Mp, Np, Kp, a_str[0], a_str[1], b_str[0], b_str[1], (0, 0), (0, 0)) is None:
        return False
    ops._gemm(Ap, Bp, Cp, biasp, Mp, Np, Kp, a_str[0], a_str[1], b_str[0], b_str[1], Np, 1, 1, (0, 0), (0, 0),
              (0, 0), alpha, beta, 1)
    ops.cols(Cv, Cp, M, N, Np, 0, True)                 # the M x N corner back into C
    return True


def _split_k(blocks, K, slots):
    """How many parts to split k into: the fewest waves times the k each block works through, with a small cost per
    extra part (the atomic additions and zeroing C)."""
    best, best_cost = 1, None
    for w in range(1, 17):
        if w > 1 and K // w < 128:
            break
        steps = -(-(-(-K // w)) // KS.BK)
        cost = -(-blocks * w // slots) * steps * (1.0 + 0.02 * (w - 1))
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = w, cost
    return best


def devices_for(name):
    """The cards of "cuda:all" or "cuda:0,1,2" (several cards: data parallel, or the model split across them), as a
    list of ops; or None."""
    d = str(name or "").lower().replace(" ", "")
    if not d.startswith("cuda:") or ("," not in d and d != "cuda:all"):
        return None
    from .native import cuda
    drv = cuda.driver()
    if drv is None:
        return []
    idx = [x.index for x in drv.devices] if d == "cuda:all" else [int(t) for t in d[5:].split(",") if t]
    out = []
    for i in idx:
        o, _ = device(f"cuda:{i}")
        if o is not None:
            out.append(o)
    return out


def device(name="cuda"):
    """(ops, why not): the GPU training kernels on `name` ("cuda", "cuda:N", "auto"), or None and the reason."""
    from .native import cuda
    drv = cuda.driver()
    if drv is None:
        return None, cuda.unavailable_reason() or "no NVIDIA GPU"
    idx = 0
    d = str(name or "cuda").lower()
    if d.startswith("cuda:"):
        try:
            idx = int(d[5:])
        except ValueError:
            return None, f"unknown device '{name}'"
    pick = [x for x in drv.devices if x.index == idx]
    if not pick:
        return None, f"no GPU {idx}"
    try:
        return Gpu(pick[0]), ""
    except Exception as e:                                   # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ============================================================================== the network on the device
class _Arena:
    """Activation and gradient buffers by name, kept across steps (grown when a larger batch comes)."""

    def __init__(self, ops):
        self.ops, self.bufs = ops, {}

    def get(self, key, n):
        b = self.bufs.get(key)
        if b is None or b.n < n:
            if b is not None:
                self.ops.free(b)
            b = self.bufs[key] = self.ops.alloc(n)
        return (b, 0)

    def close(self):
        for b in self.bufs.values():
            self.ops.free(b)
        self.bufs = {}


class _Param:
    """A parameter: its place in the flat parameter / gradient arrays."""

    def __init__(self, net, shape, off):
        self.net, self.shape, self.off = net, tuple(shape), off
        self.n = int(np.prod(shape)) if shape else 1

    @property
    def w(self):                                    # the weights the forward pass uses (rounded in low-bit mode)
        q = self.net.qoff.get(self.off)
        return (self.net.WQ, q) if q is not None else (self.net.P, self.off)

    @property
    def g(self):
        return (self.net.G, self.off)


class _Node:
    def __init__(self, net, key):
        self.net, self.key = net, key

    def gbuf(self, name, n):
        """A buffer of the backward pass: shared by the layers that are two apart in a sequence (a gradient lives
        only until the layer before has used it), so the gradients take the memory of two layers, not of all."""
        return self.net.garena.get(("g", getattr(self, "gkey", self.key), name), n)

    def buf(self, name, n):
        return self.net.arena.get((self.key, name), n)


def _rows(shape):
    return int(np.prod(shape[:-1])) if len(shape) > 1 else 1


class _Linear(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.N = layer.n_in, layer.n_out
        self.W = net.param(p.W)
        self.b = net.param(p.b) if layer.bias else None

    act = None                                          # set when an activation follows (fused into this product)
    share_out = False                                   # act(h) is only read by the next product: not kept
    x_pre = None                                        # the layer before keeps h instead of my input act(h)

    @property
    def takes_residual(self):
        return self.act is None

    def fwd(self, x, shape, residual=None):
        o, M = self.net.ops, _rows(shape)
        self.x, self.M = x, M
        y = self.net.arena.get(("shared", "act"), M * self.N) if self.share_out else self.buf("y", M * self.N)
        if self.act is not None:                        # y = act(x W + b), the pre-activation kept for backward
            self.h = self.buf("h", M * self.N)
            o.gemm(x, self.W.w, y, M, self.N, self.K, self.K, 1, self.N, 1, self.N,
                   bias=self.b.w if self.b else None, passes=self.net.passes, epi=1, X2=self.h, act=self.act)
        else:                                           # y = x W + b (+ the residual input)
            o.gemm(x, self.W.w, y, M, self.N, self.K, self.K, 1, self.N, 1, self.N,
                   bias=self.b.w if self.b else None, passes=self.net.passes, epi=3 if residual else 0, X2=residual)
        return y, shape[:-1] + (self.N,)

    def bwd(self, dy, need_dx=True, into_act=None):
        """into_act=(kind, h): the layer before is a linear layer with that activation; dx comes out already
        multiplied by act'(h) (that layer's dy) in the same pass."""
        o, M, K, N = self.net.ops, self.M, self.K, self.N
        if self.act is not None and not getattr(self, "dh_given", False):
            dh = self.gbuf("dh", M * N)
            o.act_b(self.h, dy, dh, M * N, self.act)
            dy = dh
        self.dh_given = False
        # dW[K, N] += x^T dy: A(k, m) = x[m * K + k]
        db = self.b.g if self.b is not None else None   # the bias gradient comes with the weight gradient
        if self.x_pre is not None:                      # x = act(h): applied while the product reads h
            pre = self.x_pre
            o.gemm(pre.h, dy, self.W.g, K, N, M, 1, K, N, 1, N, beta=True, passes=self.net.passes, a_act=pre.act,
                   tmp=(self.net.arena.get(("shared", "act"), M * K), M * K), db=db)
        else:
            o.gemm(self.x, dy, self.W.g, K, N, M, 1, K, N, 1, N, beta=True, passes=self.net.passes, db=db)
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * K)
        if into_act is not None:
            o.gemm(dy, self.W.w, dx, M, K, N, N, 1, 1, N, K, passes=self.net.passes, epi=2, X2=into_act[1],
                   act=into_act[0])
        else:
            o.gemm(dy, self.W.w, dx, M, K, N, N, 1, 1, N, K, passes=self.net.passes)       # dy W^T
        return dx


class _Act(_Node):
    def __init__(self, net, key, kind):
        super().__init__(net, key)
        self.kind = kind

    def fwd(self, x, shape):
        n = int(np.prod(shape))
        self.x, self.n = x, n
        y = self.buf("y", n)
        self.net.ops.act_f(x, y, n, self.kind)
        return y, shape

    def bwd(self, dy, need_dx=True):
        dx = self.gbuf("dx", self.n)
        self.net.ops.act_b(self.x, dy, dx, self.n, self.kind)
        return dx


class _Dropout(_Node):
    def __init__(self, net, key, p):
        super().__init__(net, key)
        self.p = float(p)

    def fwd(self, x, shape):
        self.on = self.net.train and self.p > 0
        if not self.on:
            return x, shape
        n = self.n = int(np.prod(shape))
        y, self.mask = self.buf("y", n), self.buf("mask", n)
        self.net.seed = (self.net.seed * 1103515245 + 12345) & 0x7FFFFFFF
        self.net.ops.drop_f(x, y, self.mask, n, self.p, self.net.seed)
        return y, shape

    def bwd(self, dy, need_dx=True):
        if not self.on:
            return dy
        dx = self.gbuf("dx", self.n)
        self.net.ops.mul3(dx, dy, self.mask, self.n)
        return dx


class _Norm(_Node):
    def __init__(self, net, key, layer, p, rms):
        super().__init__(net, key)
        self.N, self.eps, self.rms = layer.d, layer.eps, rms
        self.gm = net.param(p.gamma)
        self.bt = None if rms else net.param(p.beta)

    def fwd(self, x, shape):
        M = self.M = _rows(shape)
        self.x = x
        y = self.buf("y", M * self.N)
        self.mean, self.rstd = self.buf("mean", M), self.buf("rstd", M)
        self.net.ops.ln_f(x, self.gm.w, self.bt.w if self.bt else None, y, self.mean, self.rstd, M, self.N, self.eps,
                          self.rms)
        return y, shape

    def bwd(self, dy, need_dx=True, residual=None):
        """residual: a gradient to add to dx (the skip connection around this norm), in the same pass."""
        o = self.net.ops
        o.colsum(dy, self.x, self.mean, self.rstd, self.gm.g, self.bt.g if self.bt else None, self.M, self.N,
                 2 if self.rms else 1)
        dx = self.gbuf("dx", self.M * self.N)
        o.ln_b(self.x, self.gm.w, dy, self.mean, self.rstd, dx, self.M, self.N, self.rms, residual)
        return dx


class _Embedding(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.V, self.D = layer.vocab, layer.d
        self.E = net.param(p.E)

    def fwd(self, x, shape):
        M = self.M = int(np.prod(shape))
        self.ids = x
        y = self.buf("y", M * self.D)
        self.net.ops.embed_f(self.E.w, x, y, M, self.D)
        return y, tuple(shape) + (self.D,)

    def bwd(self, dy, need_dx=True):
        self.net.ops.embed_b(dy, self.ids, self.E.g, self.M, self.D, self.V)
        return None


class _PosEnc(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.D, self.learned, self.max_len = layer.d, layer.learned, layer.max_len
        if self.learned:
            self.P = net.param(p.P)
        else:
            pos = np.arange(self.max_len)[:, None]
            i = np.arange(self.D)[None, :]
            ang = pos / np.power(10000.0, (2 * (i // 2)) / self.D)
            pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang)).astype(np.float32)
            self.fixed = net.ops.alloc(pe.size)
            net.ops.put(self.fixed, pe.reshape(-1))
            net.owned.append(self.fixed)

    def fwd(self, x, shape):
        T = shape[-2]
        if T > self.max_len:
            raise TrainError(f"positional_encoding: sequences of {T} are longer than its {self.max_len}")
        M = self.M = _rows(shape)
        self.T = T
        y = self.buf("y", M * self.D)
        self.net.ops.pos_f(x, self.P.w if self.learned else (self.fixed, 0), y, M, self.D, T)
        return y, shape

    def bwd(self, dy, need_dx=True):
        if self.learned:
            self.net.ops.pos_b(dy, self.P.g, self.M, self.D, self.T)
        return dy


class _Seq(_Node):
    def __init__(self, net, key, nodes):
        super().__init__(net, key)
        self.nodes = nodes

    @property
    def takes_residual(self):
        return bool(self.nodes) and getattr(self.nodes[-1], "takes_residual", False)

    def fwd(self, x, shape, residual=None):
        last = len(self.nodes) - 1
        for i, nd in enumerate(self.nodes):
            x, shape = nd.fwd(x, shape, residual) if (i == last and residual is not None) else nd.fwd(x, shape)
        return x, shape

    def bwd(self, dy, need_dx=True):
        for i in range(len(self.nodes) - 1, -1, -1):
            nd, before = self.nodes[i], (self.nodes[i - 1] if i > 0 else None)
            if isinstance(nd, _Linear) and isinstance(before, _Linear) and before.act is not None:
                dy = nd.bwd(dy, True, into_act=(before.act, before.h))      # the activation's backward fused in
                before.dh_given = True
            else:
                dy = nd.bwd(dy, need_dx or i > 0)
            if dy is None:
                return None
        return dy


class _Residual(_Node):
    def __init__(self, net, key, inner, norm):
        super().__init__(net, key)
        self.inner, self.norm = inner, norm

    def fwd(self, x, shape):
        h, hs = self.norm.fwd(x, shape) if self.norm else (x, shape)
        self.n = int(np.prod(shape))
        if getattr(self.inner, "takes_residual", False):  # the inner layer's last product adds x itself
            y, _ = self.inner.fwd(h, hs, residual=x)
            return y, shape
        f, fs = self.inner.fwd(h, hs)
        y = self.buf("y", self.n)
        self.net.ops.add3(y, x, f, self.n)
        return y, shape

    def bwd(self, dy, need_dx=True):
        dh = self.inner.bwd(dy, True)
        if self.norm:
            return self.norm.bwd(dh, True, residual=dy)  # dx = dy + the norm's gradient, one pass
        dx = self.gbuf("dx", self.n)
        self.net.ops.add3(dx, dy, dh, self.n)
        return dx


class _Attention(_Node):
    """Self-attention: q k v projections, softmax(q k^T / sqrt(hd)) (causal or not) per head, the output
    projection. Scores and probabilities are kept for the backward pass ([batch x heads, T, T]) unless the fused
    kernel runs. Grouped-query attention (kv_heads): each key / value head is copied to the query heads it serves
    (one gather) and their gradients summed back (one scatter-add); qk_norm: an RMS norm over each head; rope:
    rotary positions applied in place (the backward pass applies the inverse rotation)."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.D, self.h, self.causal = layer.d, layer.h, layer.causal
        self.kvh = int(getattr(layer, "kvh", layer.h))
        self.rope = bool(getattr(layer, "rope", False))
        self.rope_base = float(getattr(layer, "rope_base", 10000.0))
        self.qkn = bool(getattr(layer, "qk_norm", False))
        self.eps = float(getattr(layer, "eps", 1e-6))
        self.Wq, self.Wk, self.Wv, self.Wo = (net.param(getattr(p, n)) for n in ("Wq", "Wk", "Wv", "Wo"))
        self.bo = net.param(p.bo) if getattr(layer, "bias", True) else None
        self.gated = getattr(layer, "gate", None) is not None       # output * tanh(gate) (no output bias)
        self.g = net.param(p.gate) if self.gated else None
        self.mmask = bool(getattr(layer, "memory_mask", False))      # the memory's last column: 1 read, 0 not
        if self.qkn:
            self.gq, self.gk = net.param(p.q_norm), net.param(p.k_norm)
        self.tabs, self.idx = {}, {}

    takes_residual = True

    def _tables(self, T):
        """cos and sin of the rotary angles for positions 0..T-1, [T, hd / 2], on the card (made once per length)."""
        t = self.tabs.get(T)
        if t is None:
            hd = self.D // self.h
            inv = 1.0 / (self.rope_base ** (np.arange(hd // 2) * 2.0 / hd))
            ang = np.arange(T)[:, None] * inv[None, :]
            o = self.net.ops
            cs, sn = o.alloc(ang.size), o.alloc(ang.size)
            o.put(cs, np.cos(ang).astype(np.float32).reshape(-1))
            o.put(sn, np.sin(ang).astype(np.float32).reshape(-1))
            self.net.owned += [cs, sn]
            t = self.tabs[T] = ((cs, 0), (sn, 0))
        return t

    def _kv_index(self, M):
        """For each (token, query head): the row of its key / value head (rows of the head size)."""
        ix = self.idx.get(M)
        if ix is None:
            g = self.h // self.kvh
            rows = (np.arange(M)[:, None] * self.kvh + np.arange(self.h)[None, :] // g).reshape(-1).astype(np.int32)
            b = self.net.ops.alloc(rows.size)
            self.net.ops.put(b, rows)
            self.net.owned.append(b)
            ix = self.idx[M] = (b, 0)
        return ix

    def fwd(self, x, shape, residual=None, mem=None):
        """mem=(x_mem, shape_mem): cross-attention, the keys and values from x_mem ([batch, S, features])."""
        if len(shape) != 3:
            raise TrainError("multi_head_attention on the GPU takes [batch, time, features]")
        o, (B, T, D), h, kvh = self.net.ops, shape, self.h, self.kvh
        xm, S = (x, T) if mem is None else (mem[0], int(mem[1][1]))
        Dm = D + 1 if self.mmask else D
        if mem is not None and (len(mem[1]) != 3 or mem[1][0] != B or mem[1][2] != Dm):
            raise TrainError(f"cross-attention: the memory is {tuple(mem[1])}, the query {tuple(shape)}")
        self.mb = None
        if self.mmask and mem is not None:              # split the memory: its features, and a bias row per batch
            a = o.get(xm[0], B * S * Dm, xm[1]).reshape(B * S, Dm)
            xm = _put_buf(self, "xmf", a[:, :D], np.float32)
            self.mb = _put_buf(self, "mb", (a[:, D].astype(np.float64) - 1.0) * 1e9, np.float32)
            self.ones = _put_buf(self, "ones", np.ones(T), np.float32)
        hd = D // h
        M, Mk, Dkv = B * T, B * S, kvh * hd
        self.x, self.xm, self.cross, self.B, self.T, self.S = x, xm, mem is not None, B, T, S
        ps = self.net.passes
        q, k, v = self.buf("q", M * D), self.buf("k", Mk * Dkv), self.buf("v", Mk * Dkv)
        o.gemm(x, self.Wq.w, q, M, D, D, D, 1, D, 1, D, passes=ps)
        for W, out in ((self.Wk, k), (self.Wv, v)):
            o.gemm(xm, W.w, out, Mk, Dkv, D, D, 1, Dkv, 1, Dkv, passes=ps)
        if self.qkn:                                    # an RMS norm of every head's query and key
            self.qraw, self.kraw = q, k
            self.qm, self.qr = self.buf("qm", M * h), self.buf("qr", M * h)
            self.km, self.kr = self.buf("km", Mk * kvh), self.buf("kr", Mk * kvh)
            qn, kn = self.buf("qn", M * D), self.buf("kn", Mk * Dkv)
            o.ln_f(q, self.gq.w, None, qn, self.qm, self.qr, M * h, hd, self.eps, True)
            o.ln_f(k, self.gk.w, None, kn, self.km, self.kr, Mk * kvh, hd, self.eps, True)
            q, k = qn, kn
        if self.rope:
            cs, sn = self._tables(T)
            o.rope(q, cs, sn, M * h, hd, h, T, 1.0)
            cs, sn = self._tables(S)
            o.rope(k, cs, sn, Mk * kvh, hd, kvh, S, 1.0)
        if kvh != h:                                    # each key / value head to the query heads it serves
            ix = self._kv_index(Mk)
            kf, vf = self.buf("kf", Mk * D), self.buf("vf", Mk * D)
            o.gather(k, ix, kf, Mk * h, hd, 0)
            o.gather(v, ix, vf, Mk * h, hd, 0)
            k, v = kf, vf
        self.q, self.k, self.v = q, k, v
        sc = 1.0 / math.sqrt(hd)
        self.fused = ps == 1 and getattr(o, "flash", False) and hd in KS.FLASH_HD and D % 4 == 0 and not self.cross \
            and not self.gated
        if self.fused:                                  # scores, softmax and the weighted sum in one kernel
            att, self.L = self.buf("att", M * D), self.buf("lse", B * h * T)
            o.flash_fwd(q, k, v, att, self.L, B, T, h, hd, self.causal, sc)
            self.att = att
            y = self.buf("y", M * D)
            o.gemm(att, self.Wo.w, y, M, D, D, D, 1, D, 1, D, bias=self.bo.w if self.bo else None, passes=ps, epi=3 if residual else 0,
                   X2=residual)
            return y, shape
        Sc, P = self.buf("s", B * h * T * S), self.buf("p", B * h * T * S)
        # S[b, h](t, s) = q[b, t, h, :] . k[b, s, h, :] * sc
        o.gemm(q, k, Sc, T, S, hd, D, 1, 1, D, S, alpha=sc, batch=(B, h), sa=(T * D, hd), sb=(S * D, hd),
               sc=(h * T * S, T * S), passes=ps)
        if self.mb is not None:                         # + the memory's bias (0 or -1e9) on every query row
            o.gemm(self.ones, self.mb, Sc, T, S, 1, 1, 1, S, 1, S, beta=True, batch=(B, h), sa=(0, 0), sb=(S, 0),
                   sc=(h * T * S, T * S), passes=3)
        o.softmax_f(Sc, P, B * h * T, S, T, self.causal)
        self.P = P
        att = self.buf("att", M * D)                    # [b, t, h, :] = P[b, h] @ v[b, :, h, :]
        o.gemm(P, v, att, T, hd, S, S, 1, D, 1, D, batch=(B, h), sa=(h * T * S, T * S), sb=(S * D, hd),
               sc=(T * D, hd), passes=ps)
        self.att = att
        y = self.buf("y", M * D)
        if self.gated:                                  # y = tanh(gate) (att Wo) (+ residual)
            z = self.buf("z", M * D)
            o.gemm(att, self.Wo.w, z, M, D, D, D, 1, D, 1, D, passes=ps)
            self.z, self.th = z, self.buf("th", 1)      # tanh(gate) on the card: no wait for the card here
            o.tanh_s(self.th, self.g.w)
            if residual is not None:
                o.pull(y, o, residual, M * D)
            else:
                o.fill(y, M * D, 0.0)
            o.axpy_s(y, z, M * D, self.th)
            return y, shape
        o.gemm(att, self.Wo.w, y, M, D, D, D, 1, D, 1, D, bias=self.bo.w if self.bo else None, passes=ps, epi=3 if residual else 0,
               X2=residual)
        return y, shape

    def bwd(self, dy, need_dx=True):
        out = self._bwd(dy, need_dx)
        if self.cross and self.mmask:                   # the memory's gradient: none for its last column
            dx, dm = out
            Mk, D = self.B * self.S, self.D
            a = np.zeros((Mk, D + 1), np.float32)
            a[:, :D] = self.net.ops.get(dm[0], Mk * D, dm[1]).reshape(Mk, D)
            return dx, _put_buf(self, "dmem1", a, np.float32, grad=True)
        return out

    def _bwd(self, dy, need_dx=True):
        """The input's gradient; for cross-attention (the query's gradient, the memory's gradient)."""
        o, B, T, S, D, h = self.net.ops, self.B, self.T, self.S, self.D, self.h
        hd, M, Mk, ps = D // h, B * T, B * S, self.net.passes
        sc = 1.0 / math.sqrt(hd)
        if self.gated:                                  # d gate = (1 - tanh^2) dy . z; the rest sees tanh(gate) dy
            dg = self.gbuf("dg", 1)
            o.edge_dot(dy, self.z, _put_buf(self, "zero1", [0]), _put_buf(self, "zero2", [0]), dg, 1, 1, M * D)
            o.gate_g(self.g.g, dg, self.th)
            dz = self.gbuf("dz", M * D)
            o.fill(dz, M * D, 0.0)
            o.axpy_s(dz, dy, M * D, self.th)
            dy = dz
        o.gemm(self.att, dy, self.Wo.g, D, D, M, 1, D, D, 1, D, beta=True, passes=ps, db=self.bo.g if self.bo else None)
        datt = self.gbuf("datt", M * D)
        o.gemm(dy, self.Wo.w, datt, M, D, D, D, 1, 1, D, D, passes=ps)
        if self.fused:
            dd = self.gbuf("dd", B * h * T)
            o.flash_dot(datt, self.att, dd, B, T, h, hd)
            dq, dk, dv = self.gbuf("dq", M * D), self.gbuf("dk", M * D), self.gbuf("dv", M * D)
            o.flash_bwd(self.q, self.k, self.v, datt, self.L, dd, dq, dk, dv, B, T, h, hd, self.causal, sc)
            return self._grads_in(dq, dk, dv, M, D, ps)
        dP = self.gbuf("dp", B * h * T * S)               # dP = datt v^T
        o.gemm(datt, self.v, dP, T, S, hd, D, 1, 1, D, S, batch=(B, h), sa=(T * D, hd), sb=(S * D, hd),
               sc=(h * T * S, T * S), passes=ps)
        dv = self.gbuf("dv", Mk * D)                      # dv = P^T datt
        o.gemm(self.P, datt, dv, S, hd, T, 1, S, D, 1, D, batch=(B, h), sa=(h * T * S, T * S), sb=(T * D, hd),
               sc=(S * D, hd), passes=ps)
        dS = self.gbuf("ds", B * h * T * S)
        o.softmax_b(self.P, dP, dS, B * h * T, S)
        dq, dk = self.gbuf("dq", M * D), self.gbuf("dk", Mk * D)
        o.gemm(dS, self.k, dq, T, hd, S, S, 1, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * S, T * S),
               sb=(S * D, hd), sc=(T * D, hd), passes=ps)                                   # dq = dS k
        o.gemm(dS, self.q, dk, S, hd, T, 1, S, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * S, T * S),
               sb=(T * D, hd), sc=(S * D, hd), passes=ps)                                   # dk = dS^T q
        return self._grads_in(dq, dk, dv, M, D, ps)

    def _grads_in(self, dq, dk, dv, M, D, ps):
        o, h, kvh = self.net.ops, self.h, self.kvh
        hd = D // h
        Dkv = kvh * hd
        Mk = self.B * self.S
        if kvh != h:                                    # the copies' gradients summed back into each kv head
            ix = self._kv_index(Mk)
            dks, dvs = self.gbuf("dks", Mk * Dkv), self.gbuf("dvs", Mk * Dkv)
            o.fill(dks, Mk * Dkv, 0.0)
            o.fill(dvs, Mk * Dkv, 0.0)
            o.embed_b(dk, ix, dks, Mk * h, hd, Mk * kvh)
            o.embed_b(dv, ix, dvs, Mk * h, hd, Mk * kvh)
            dk, dv = dks, dvs
        if self.rope:                                   # the inverse rotation
            cs, sn = self._tables(self.T)
            o.rope(dq, cs, sn, M * h, hd, h, self.T, -1.0)
            cs, sn = self._tables(self.S)
            o.rope(dk, cs, sn, Mk * kvh, hd, kvh, self.S, -1.0)
        if self.qkn:
            dq2, dk2 = self.gbuf("dqn", M * D), self.gbuf("dkn", Mk * Dkv)
            o.colsum(dq, self.qraw, self.qm, self.qr, self.gq.g, None, M * h, hd, 2)
            o.ln_b(self.qraw, self.gq.w, dq, self.qm, self.qr, dq2, M * h, hd, True, None)
            o.colsum(dk, self.kraw, self.km, self.kr, self.gk.g, None, Mk * kvh, hd, 2)
            o.ln_b(self.kraw, self.gk.w, dk, self.km, self.kr, dk2, Mk * kvh, hd, True, None)
            dq, dk = dq2, dk2
        o.gemm(self.x, dq, self.Wq.g, D, D, M, 1, D, D, 1, D, beta=True, passes=ps)
        for W, d in ((self.Wk, dk), (self.Wv, dv)):
            o.gemm(self.xm, d, W.g, D, Dkv, Mk, 1, D, Dkv, 1, Dkv, beta=True, passes=ps)
        dx = self.gbuf("dx", M * D)
        o.gemm(dq, self.Wq.w, dx, M, D, D, D, 1, 1, D, D, passes=ps)
        dm = dx if not self.cross else self.gbuf("dmem", Mk * D)
        first = self.cross
        for W, d in ((self.Wk, dk), (self.Wv, dv)):
            o.gemm(d, W.w, dm, Mk, D, Dkv, Dkv, 1, 1, Dkv, D, beta=not first, passes=ps)
            first = False
        return (dx, dm) if self.cross else dx


class _Identity(_Node):
    """Identity, flatten, reshape: the same values, another shape."""

    def __init__(self, net, key, layer=None):
        super().__init__(net, key)
        self.layer = layer

    def fwd(self, x, shape):
        from . import layers as L
        l_ = self.layer
        if isinstance(l_, L.Flatten):
            return x, (shape[0], int(np.prod(shape[1:])))
        if isinstance(l_, L.Reshape):
            if int(np.prod(l_.shape)) != int(np.prod(shape[1:])):
                raise TrainError(f"reshape: {tuple(shape)} cannot become (N, {', '.join(map(str, l_.shape))})")
            return x, (shape[0],) + tuple(l_.shape)
        return x, shape

    def bwd(self, dy, need_dx=True):
        return dy


class _MeanPool(_Node):
    """[B, T, D] -> [B, D], the mean over time (one product with a column of 1 / T each way)."""

    def __init__(self, net, key, layer):
        super().__init__(net, key)
        if int(layer.axis) not in (1, -2):
            raise TrainError(f"mean_pool(axis={layer.axis}) is not on the GPU trainer yet (axis=1)")
        self.ones = {}

    def _ones(self, T):
        b = self.ones.get(T)
        if b is None:
            b = self.net.ops.alloc(T)
            self.net.ops.fill((b, 0), T, 1.0 / T)
            self.net.owned.append(b)
            b = self.ones[T] = (b, 0)
        return b

    def fwd(self, x, shape):
        if len(shape) != 3:
            raise TrainError("mean_pool on the GPU takes [batch, time, features]")
        B, T, D = shape
        self.B, self.T, self.D = B, T, D
        y = self.buf("y", B * D)
        self.net.ops.gemm(self._ones(T), x, y, 1, D, T, T, 1, D, 1, D, batch=(B, 1), sa=(0, 0), sb=(T * D, 0),
                          sc=(D, 0), passes=3)
        return y, (B, D)

    def bwd(self, dy, need_dx=True):
        B, T, D = self.B, self.T, self.D
        dx = self.gbuf("dx", B * T * D)
        self.net.ops.gemm(self._ones(T), dy, dx, T, D, 1, 1, 1, D, 1, D, batch=(B, 1), sa=(0, 0), sb=(D, 0),
                          sc=(T * D, 0), passes=3)
        return dx


class _Scale(_Node):
    """x * s (+ t), per feature."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.D = layer.d
        self.s = net.param(p.s)
        self.t = net.param(p.t) if layer.shift else None
        self.consts = {}

    def _consts(self, M):
        c = self.consts.get(M)
        if c is None:
            o = self.net.ops
            z, one = o.alloc(M), o.alloc(M)
            o.fill((z, 0), M, 0.0)
            o.fill((one, 0), M, 1.0)
            self.net.owned += [z, one]
            c = self.consts[M] = ((z, 0), (one, 0))
        return c

    def fwd(self, x, shape):
        o, M, D = self.net.ops, _rows(shape), self.D
        self.x, self.M = x, M
        y = self.buf("y", M * D)
        o.colmul(x, self.s.w, y, M * D, D)
        if self.t is not None:
            o.pos_f(y, self.t.w, y, M, D, 1)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, M, D = self.net.ops, self.M, self.D
        z, one = self._consts(M)                        # sum dy * x (and sum dy): the column sums of a norm
        o.colsum(dy, self.x, z, one, self.s.g, self.t.g if self.t is not None else None, M, D,
                 1 if self.t is not None else 2)
        dx = self.gbuf("dx", M * D)
        o.colmul(dy, self.s.w, dx, M * D, D)
        return dx


class _Combine(_Node):
    """add, mul, mean or concat (last axis) of several inputs."""

    def __init__(self, net, key, how):
        super().__init__(net, key)
        if how not in ("add", "mul", "mean", "concat"):
            raise TrainError(f"combine '{how}' is not on the GPU trainer (add, mul, mean, concat)")
        self.how = how

    def fwd_multi(self, ins):
        o = self.net.ops
        shapes = [s_ for _, s_ in ins]
        self.ins, self.shapes = [x for x, _ in ins], shapes
        if self.how == "concat":
            M = _rows(shapes[0])
            if any(_rows(s_) != M or tuple(s_[:-1]) != tuple(shapes[0][:-1]) for s_ in shapes):
                raise TrainError(f"concatenate: the shapes {shapes} differ before the last axis")
            W = sum(s_[-1] for s_ in shapes)
            y = self.buf("y", M * W)
            off = 0
            for x, s_ in ins:
                o.cols(x, y, M, s_[-1], W, off, 0)
                off += s_[-1]
            self.M, self.W = M, W
            return y, tuple(shapes[0][:-1]) + (W,)
        if any(tuple(s_) != tuple(shapes[0]) for s_ in shapes):
            raise TrainError(f"{self.how}: the inputs' shapes differ: {shapes}")
        n = self.n = int(np.prod(shapes[0]))
        y = self.buf("y", n)
        if self.how == "mean":
            o.fill(y, n, 0.0)
            for x in self.ins:
                o.axpy(y, x, n, 1.0 / len(self.ins))
        elif self.how == "add":
            o.add3(y, self.ins[0], self.ins[1], n) if len(self.ins) > 1 else o.pull(y, o, self.ins[0], n)
            for x in self.ins[2:]:
                o.axpy(y, x, n, 1.0)
        else:
            o.pull(y, o, self.ins[0], n)
            for x in self.ins[1:]:
                o.mul3(y, y, x, n)
        return y, shapes[0]

    def bwd_multi(self, dy):
        o = self.net.ops
        if self.how == "concat":
            outs, off = [], 0
            for i, s_ in enumerate(self.shapes):
                dx = self.gbuf(("dx", i), self.M * s_[-1])
                o.cols(dx, dy, self.M, s_[-1], self.W, off, 1)
                off += s_[-1]
                outs.append(dx)
            return outs
        n = self.n
        if self.how == "add":
            return [dy] * len(self.ins)
        if self.how == "mean":
            dx = self.gbuf("dx", n)
            o.fill(dx, n, 0.0)
            o.axpy(dx, dy, n, 1.0 / len(self.ins))
            return [dx] * len(self.ins)
        outs = []
        for i in range(len(self.ins)):
            dx = self.gbuf(("dx", i), n)
            o.pull(dx, o, dy, n)
            for j, x in enumerate(self.ins):
                if j != i:
                    o.mul3(dx, dx, x, n)
            outs.append(dx)
        return outs


class _Grads:
    """The gradients of named values in a graph of layers: the first contribution kept as it is, later ones added
    into a buffer of the owner's own (never into a buffer another layer gave)."""

    def __init__(self, node, shapes):
        self.node, self.shapes, self.g, self.own = node, shapes, {}, set()
        self.lo = {}                                      # name -> the low part of an exact (double-float) sum

    def add(self, name, dx):
        """Several contributions are summed exactly (double-float) and rounded once when read, so the gradient of a
        value does not depend on the order its users' gradients arrive in (one card or many, any cut)."""
        if dx is None:
            return
        if name not in self.g:
            self.g[name] = dx
            return
        o, n = self.node.net.ops, int(np.prod(self.shapes[name]))
        if name not in self.own:
            hi, lo = self.node.gbuf(("acc", name), n), self.node.gbuf(("acclo", name), n)
            o.pull(hi, o, self.g[name], n)
            o.fill(lo, n, 0.0)
            self.g[name], self.lo[name], _ = hi, lo, self.own.add(name)
        o.acc2(self.g[name], self.lo[name], dx, n, False)

    def seed(self, name, hi, lo=None):
        """A gradient that arrives as an exact sum (hi, lo) from another card: summed on exactly."""
        if lo is None:
            return self.add(name, hi)
        o, n = self.node.net.ops, int(np.prod(self.shapes[name]))
        if name in self.g:
            self.add(name, hi)
            o.acc2(self.g[name], self.lo[name], lo, n, False)
            return
        h2, l2 = self.node.gbuf(("acc", name), n), self.node.gbuf(("acclo", name), n)
        o.pull(h2, o, hi, n)
        o.pull(l2, o, lo, n)
        self.g[name], self.lo[name], _ = h2, l2, self.own.add(name)

    def raw(self):
        """name -> (hi, lo or None): the gradients left, not rounded (to carry an exact sum to another card)."""
        return {k: (v, self.lo.get(k)) for k, v in self.g.items()}

    def take(self, name):
        """The gradient of `name`, its exact sum rounded once."""
        g = self.g.pop(name, None)
        lo = self.lo.pop(name, None)
        if g is not None and lo is not None:
            o, n = self.node.net.ops, int(np.prod(self.shapes[name]))
            o.axpy(g, lo, n, 1.0)                          # hi + lo, one rounding
        return g

    def done(self):
        """Every gradient left (the inputs'), rounded."""
        for name in list(self.lo):
            self.g[name] = self.take(name)
        return self.g


def _fwd_multi(nd, ins):
    if hasattr(nd, "fwd_multi"):
        return nd.fwd_multi(ins)
    if isinstance(nd, _Attention) and len(ins) == 2:
        return nd.fwd(ins[0][0], ins[0][1], mem=ins[1])
    raise TrainError(f"a layer of the GPU trainer got {len(ins)} inputs (combine layers and cross-attention take "
                     "several)")


def _bwd_multi(nd, dy):
    return nd.bwd_multi(dy) if hasattr(nd, "bwd_multi") else list(nd.bwd(dy, True))


class _DAG(_Node):
    """Any directed acyclic graph of layers (dag(...)): values by name, the gradients added where a value feeds
    several layers."""

    def __init__(self, net, key, layer, p, gk):
        super().__init__(net, key)
        self.inputs, self.out = list(layer.inputs), layer.output
        self.multi_out = not isinstance(self.out, str)           # several outputs: the model's root only
        self.inp = self.inputs[0]
        before, net.dag_params = getattr(net, "dag_params", None), p      # tied_output finds its owner's weights
        try:
            self.items = [(name, net.compile(l_, getattr(p, name), f"{key}.{name}", f"{gk}.{name}"), srcs, single)
                          for name, l_, srcs, single in layer.nodes]
        finally:
            net.dag_params = before

    def fwd(self, x, shape):
        if len(self.inputs) != 1:
            raise TrainError(f"this dag takes {len(self.inputs)} inputs {self.inputs}: give them as a list")
        return self._run({self.inp: (x, shape)})

    def fwd_multi(self, ins):
        if len(ins) != len(self.inputs):
            raise TrainError(f"this dag takes {len(self.inputs)} inputs {self.inputs}, got {len(ins)}")
        return self._run(dict(zip(self.inputs, ins)))

    def bwd_multi(self, dy):
        self._back(dy, True)
        g = self.G.done()
        return [g.get(n) for n in self.inputs]

    def run_all(self, vals):
        """Every node from the given values (name -> (view, shape)); all the values by name."""
        self._run(vals)
        return self.vals

    def back_from(self, seeds, seeds_lo=None, raw=False):
        """Gradients from the given ones (name -> view; seeds_lo: the low parts of exact sums): the gradients of
        every value, by name (raw: name -> (hi, lo or None), not rounded)."""
        G = self.G = _Grads(self, self.shapes)
        for k_, v in seeds.items():
            G.seed(k_, v, (seeds_lo or {}).get(k_))
        self._walk_back(G, True)
        return G.raw() if raw else G.done()

    def _run(self, vals):
        self.vals = vals
        for name, nd, srcs, single in self.items:
            vals[name] = nd.fwd(*vals[srcs[0]]) if single else _fwd_multi(nd, [vals[s_] for s_ in srcs])
        self.shapes = {k_: v[1] for k_, v in vals.items()}
        if self.multi_out:
            return [vals[o][0] for o in self.out], [vals[o][1] for o in self.out]
        return vals[self.out]

    def bwd(self, dy, need_dx=True):
        self._back(dy, need_dx)
        return self.G.done().get(self.inp)

    def _back(self, dy, need_dx):
        G = self.G = _Grads(self, self.shapes)
        if self.multi_out:                                       # one gradient per output (None: not in the loss)
            for o_, d_ in zip(self.out, dy):
                G.add(o_, d_)
        else:
            G.g[self.out] = dy
        self._walk_back(G, need_dx)

    def _walk_back(self, G, need_dx):
        for name, nd, srcs, single in reversed(self.items):
            g = G.take(name)
            if g is None:
                continue
            if single:
                G.add(srcs[0], nd.bwd(g, need_dx or srcs[0] not in self.inputs))
            else:
                for s_, dx in zip(srcs, _bwd_multi(nd, g)):
                    G.add(s_, dx)


class _TiedOutput(_Node):
    """tied_output: x @ E^T with the table E of an embedding of the same dag (one parameter, two users: both add
    their gradients into it)."""

    def __init__(self, net, key, layer, dag_params):
        super().__init__(net, key)
        if dag_params is None or not hasattr(dag_params, layer.owner):
            raise TrainError(f"tied_output('{layer.owner}'): no embedding of that name in the dag")
        E = getattr(dag_params, layer.owner).E
        self.V, self.D = E.shape
        self.E = net.param(E)

    def fwd(self, x, shape):
        M, V, D = _rows(shape), self.V, self.D
        self.x, self.M = x, M
        y = self.buf("y", M * V)
        self.net.ops.gemm(x, self.E.w, y, M, V, D, D, 1, 1, D, V, passes=self.net.passes)    # x E^T
        return y, tuple(shape[:-1]) + (V,)

    def bwd(self, dy, need_dx=True):
        o, M, V, D, ps = self.net.ops, self.M, self.V, self.D, self.net.passes
        o.gemm(dy, self.x, self.E.g, V, D, M, 1, V, D, 1, D, beta=True, passes=ps)        # dE += dy^T x
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * D)
        o.gemm(dy, self.E.w, dx, M, D, V, V, 1, D, 1, D, passes=ps)                       # dx = dy E
        return dx


class _Parallel(_Node):
    """Branches on the same input, combined (concat, add, mul, mean)."""

    def __init__(self, net, key, layer, p, gk):
        super().__init__(net, key)
        self.branches = [net.compile(b_, pp, f"{key}.{i}", f"{gk}.{i}") for i, (b_, pp) in
                         enumerate(zip(layer.branches, p))]
        self.comb = _Combine(net, key + ".c", layer.combine)
        self.comb.gkey = gk + ".c"

    def fwd(self, x, shape):
        self.shape = shape
        return self.comb.fwd_multi([b_.fwd(x, shape) for b_ in self.branches])

    def bwd(self, dy, need_dx=True):
        G = _Grads(self, {"x": self.shape})
        for b_, d in zip(self.branches, self.comb.bwd_multi(dy)):
            G.add("x", b_.bwd(d, need_dx))
        return G.done().get("x")


def _ofs(view, k):
    return (view[0], view[1] + int(k))


class _MoE(_Node):
    """Mixture of experts. The router's scores and the top-k choice with its softmax are made on the card (one
    kernel); the choice comes back to the processor (M x k integers) to group the rows by expert; every expert runs
    its gated feed-forward block on its rows (products on slices, no padding, no row dropped); the outputs are
    summed back into the rows weighted by the gates (a sparse product, no atomics)."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.F, self.E, self.k = layer.d, layer.ff, layer.E, layer.k
        self.act = KS.ACTS[layer.act]
        self.Wr, self.Wg, self.Wu, self.Wd = (net.param(getattr(p, n)) for n in ("Wr", "Wg", "Wu", "Wd"))

    def _put(self, name, a):
        a = np.ascontiguousarray(a.astype(np.int32))
        b = self.buf(name, max(1, a.size))
        if a.size:
            self.net.ops.put(b[0], a, b[1])
        return b

    def fwd(self, x, shape):
        o, ps = self.net.ops, self.net.passes
        M, K, F, E, k = _rows(shape), self.K, self.F, self.E, self.k
        self.x, self.M, self.shape = x, M, shape
        lg, idx, gate = self.buf("lg", M * E), self.buf("idx", M * k), self.buf("gate", M * k)
        o.gemm(x, self.Wr.w, lg, M, E, K, K, 1, E, 1, E, passes=ps)
        o.topk_gate(lg, idx, gate, M, E, k)
        self.lg, self.idx, self.gate = lg, idx, gate
        choice = o.get(idx[0], M * k, idx[1]).view(np.int32).astype(np.int64)
        order = np.argsort(choice, kind="stable")              # (row, slot) entries grouped by expert
        counts = np.bincount(choice, minlength=E)
        pos = np.empty(M * k, np.int64)
        pos[order] = np.arange(M * k)
        self.counts = counts
        self.rows = self._put("rows", order // k)              # the token of each grouped entry
        self.slot = self._put("slot", order)                    # its (row, slot) index
        self.pos = self._put("pos", pos)                        # where each (row, slot) went
        self.tok = self._put("tok", np.repeat(np.arange(M), k))
        self.off = self._put("off", np.arange(M + 1) * k)
        self.one = self._put("one", np.arange(M * k + 1))
        n = M * k
        xs = self.buf("xs", n * K)
        o.gather(x, self.rows, xs, n, K, 0)
        self.xs = xs
        self.h, self.a, self.u, self.z = (self.buf(t, n * F) for t in ("h", "a", "u", "z"))
        ys = self.buf("ys", n * K)
        r0 = 0
        for e in range(E):                                      # every expert on its rows
            ne = int(counts[e])
            if not ne:
                continue
            xe, he, ae, ue, ze = _ofs(xs, r0 * K), _ofs(self.h, r0 * F), _ofs(self.a, r0 * F), \
                _ofs(self.u, r0 * F), _ofs(self.z, r0 * F)
            o.gemm(xe, _ofs(self.Wg.w, e * K * F), ae, ne, F, K, K, 1, F, 1, F, passes=ps, epi=1, X2=he, act=self.act)
            o.gemm(xe, _ofs(self.Wu.w, e * K * F), ue, ne, F, K, K, 1, F, 1, F, passes=ps)
            o.mul3(ze, ae, ue, ne * F)
            o.gemm(ze, _ofs(self.Wd.w, e * F * K), _ofs(ys, r0 * K), ne, K, F, F, 1, K, 1, K, passes=ps)
            r0 += ne
        self.ys = ys
        y = self.buf("y", M * K)                                # sum over each row's experts, weighted by the gates
        o.spmm(ys, self.pos, gate, None, self.off, y, M, 1, K)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        M, K, F, E, k = self.M, self.K, self.F, self.E, self.k
        n = M * k
        dys = self.gbuf("dys", n * K)                           # each grouped entry: gate * dy[its row]
        o.spmm(dy, self.rows, self.gate, self.slot, self.one, dys, n, 1, K)
        dg = self.gbuf("dg", n)                                 # each gate: dy[row] . ys[its entry]
        o.edge_dot(dy, self.ys, self.tok, self.pos, dg, n, 1, K)
        dl = self.gbuf("dl", M * E)
        o.topk_gate_b(self.gate, dg, self.idx, dl, M, E, k)
        o.gemm(self.x, dl, self.Wr.g, K, E, M, 1, K, E, 1, E, beta=True, passes=ps)
        dxs = self.gbuf("dxs", n * K)
        dz, du, dh = self.gbuf("dz", n * F), self.gbuf("du", n * F), self.gbuf("dh", n * F)
        r0 = 0
        for e in range(E):
            ne = int(self.counts[e])
            if not ne:
                continue
            xe, dye = _ofs(self.xs, r0 * K), _ofs(dys, r0 * K)
            ae, ue, ze, he = _ofs(self.a, r0 * F), _ofs(self.u, r0 * F), _ofs(self.z, r0 * F), _ofs(self.h, r0 * F)
            dze, due, dhe = _ofs(dz, r0 * F), _ofs(du, r0 * F), _ofs(dh, r0 * F)
            o.gemm(ze, dye, _ofs(self.Wd.g, e * F * K), F, K, ne, 1, F, K, 1, K, beta=True, passes=ps)
            o.gemm(dye, _ofs(self.Wd.w, e * F * K), dze, ne, F, K, K, 1, 1, K, F, passes=ps)
            o.mul3(due, dze, ae, ne * F)
            o.mul3(dze, dze, ue, ne * F)
            o.act_b(he, dze, dhe, ne * F, self.act)
            o.gemm(xe, dhe, _ofs(self.Wg.g, e * K * F), K, F, ne, 1, K, F, 1, F, beta=True, passes=ps)
            o.gemm(xe, due, _ofs(self.Wu.g, e * K * F), K, F, ne, 1, K, F, 1, F, beta=True, passes=ps)
            dxe = _ofs(dxs, r0 * K)
            o.gemm(dhe, _ofs(self.Wg.w, e * K * F), dxe, ne, K, F, F, 1, 1, F, K, passes=ps)
            o.gemm(due, _ofs(self.Wu.w, e * K * F), dxe, ne, K, F, F, 1, 1, F, K, beta=True, passes=ps)
            r0 += ne
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * K)
        o.gemm(dl, self.Wr.w, dx, M, K, E, E, 1, 1, E, K, passes=ps)
        o.embed_b(dxs, self.rows, dx, n, K, M)
        return dx


class _Conv2D(_Node):
    """[N, C, H, W] -> [N, O, OH, OW]: the windows unfolded into rows (im2col), one product with the weights, the
    result put back in channel order with the bias; backward the same products and the rows folded back (col2im,
    a thread an input value, no atomics)."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.C, self.O, self.k, self.s, self.p = layer.c_in, layer.c_out, layer.k, layer.stride, layer.padding
        self.W, self.b = net.param(p.W), net.param(p.b)

    def fwd(self, x, shape):
        if len(shape) != 4 or shape[1] != self.C:
            raise TrainError(f"conv2d({self.C} -> {self.O}) got {tuple(shape)} (want [N, {self.C}, H, W])")
        o, ps = self.net.ops, self.net.passes
        N, C, H, W = shape
        k, s, p = self.k, self.s, self.p
        OH, OW = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
        R, CK, O = N * OH * OW, C * k * k, self.O
        self.dims = (N, C, H, W, OH, OW, R, CK)
        col = self.buf("col", R * CK)
        o.im2col(x, col, N, C, H, W, k, k, s, p, OH, OW)
        self.col = col
        z = self.buf("z", R * O)                        # [rows, O] = col @ W^T
        o.gemm(col, self.W.w, z, R, O, CK, CK, 1, 1, CK, O, passes=ps)
        y = self.buf("y", R * O)
        o.perm_nc(z, y, self.b.w, N, O, OH * OW, 0)
        return y, (N, O, OH, OW)

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        N, C, H, W, OH, OW, R, CK = self.dims
        O, k = self.O, self.k
        dz = self.gbuf("dz", R * O)
        o.perm_nc(dz, dy, None, N, O, OH * OW, 1)
        o.colsum(dz, None, None, None, None, self.b.g, R, O, 0)
        o.gemm(dz, self.col, self.W.g, O, CK, R, 1, O, CK, 1, CK, beta=True, passes=ps)     # dW = dz^T col
        if not need_dx:
            return None
        dcol = self.gbuf("dcol", R * CK)
        o.gemm(dz, self.W.w, dcol, R, CK, O, O, 1, CK, 1, CK, passes=ps)
        dx = self.gbuf("dx", N * C * H * W)
        o.col2im(dcol, dx, N, C, H, W, k, k, self.s, self.p, OH, OW)
        return dx


_HOST_LOCK = __import__("threading").RLock()      # the processor's part of a model: one card's thread at a time


class _HostLayer(_Node):
    """Any other layer (a custom layer, a lambda, a layer the card has no kernel for): run on the processor with
    Apsi's autodiff, its input and the current weights brought from the card, the output and the gradients sent
    back. The rest of the model stays on the card; only this layer pays the copies."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.layer, self.p = layer, p
        self.leaves = [a for a in tree_leaves(p) if isinstance(a, np.ndarray) and id(a) in net.offsets]
        net.host_layers = getattr(net, "host_layers", 0) + 1

    def _params(self):
        o, net = self.net.ops, self.net
        now = {}
        for a in self.leaves:
            w = net.param(a).w
            now[id(a)] = o.get(w[0], a.size, w[1]).reshape(a.shape).astype(np.float64)
        return tree_map(lambda a: now.get(id(a), a) if isinstance(a, np.ndarray) else a, self.p)

    def _ctx(self):
        return {"graph": self.net.graph} if self.net.graph is not None else {}

    def _get(self, x, shape, i=0):
        a = self.net.ops.get(x[0], int(np.prod(shape)), x[1]).reshape(shape)
        ids = getattr(self.layer, "takes_ids", False)          # integer inputs (ids) travel as their bits
        if ids is True or (isinstance(ids, (list, tuple)) and i < len(ids) and ids[i]):
            return np.ascontiguousarray(a).view(np.int32).astype(np.int64)
        return a.astype(np.float64)

    def _run(self, xs):
        with _HOST_LOCK:
            return self._run_locked(xs)

    def _run_locked(self, xs):
        from . import ops as aops
        self.rng = aops._RNG[0].bit_generator.state
        p = self._params()
        y = self.layer(p, xs if self.multi else xs[0], train=self.net.train, **self._ctx())
        y = np.asarray(aops.raw(y), np.float32)
        self.pnow, self.out_shape = p, tuple(y.shape)
        out = self.buf("y", max(1, y.size))
        self.net.ops.put(out[0], y.reshape(-1), out[1])
        return out, tuple(y.shape)

    def fwd(self, x, shape):
        self.multi, self.shapes = False, [shape]
        self.xs = [self._get(x, shape)]
        return self._run(self.xs)

    def fwd_multi(self, ins):
        self.multi, self.shapes = True, [s_ for _, s_ in ins]
        self.xs = [self._get(x, s_, i) for i, (x, s_) in enumerate(ins)]
        return self._run(self.xs)

    def _grads(self, dy):
        with _HOST_LOCK:
            return self._grads_locked(dy)

    def _grads_locked(self, dy):
        from . import ops as aops
        from .stdlib import value_and_grad
        p = self.pnow
        layer, ctx, train, multi = self.layer, self._ctx(), self.net.train, self.multi
        n = int(np.prod(self.out_shape))
        dyh = self.net.ops.get(dy[0], n, dy[1]).reshape(self.out_shape).astype(np.float64)

        def f(p_, *xs):
            y = layer(p_, list(xs) if multi else xs[0], train=train, **ctx)
            return aops.sum_(y * dyh)
        g0 = aops._RNG[0].bit_generator.state
        aops._RNG[0].bit_generator.state = self.rng       # the same random numbers as the forward pass
        ids = getattr(layer, "takes_ids", False)
        isid = [ids is True or (isinstance(ids, (list, tuple)) and i < len(ids) and bool(ids[i]))
                for i in range(len(self.xs))]
        wrt = (0,) + tuple(1 + i for i in range(len(self.xs)) if not isid[i])
        try:
            _, g = value_and_grad(f, wrt=wrt)(p, *self.xs)
        finally:
            aops._RNG[0].bit_generator.state = g0
        g = list(g) if isinstance(g, (list, tuple)) and len(wrt) > 1 else [g]
        gp, rest = g[0], list(g[1:])
        gxs = [None if isid[i] else rest.pop(0) for i in range(len(self.xs))]
        o, net = self.net.ops, self.net
        flat = {id(a): t for a, t in zip(tree_leaves(p), tree_leaves(gp))}
        orig = {id(b_): a for a, b_ in zip(tree_leaves(self.p), tree_leaves(p))}
        for b_id, t in flat.items():
            a = orig.get(b_id)
            if a is None or not isinstance(a, np.ndarray) or id(a) not in net.offsets:
                continue
            tmp = self.gbuf(("pg", id(a)), a.size)
            o.put(tmp[0], np.asarray(aops.raw(t), np.float32).reshape(-1), tmp[1])
            o.axpy(net.param(a).g, tmp, a.size, 1.0)
        outs = []
        for i, (gx, s_) in enumerate(zip(gxs, self.shapes)):
            if gx is None:
                outs.append(None)
                continue
            d = self.gbuf(("dx", i), int(np.prod(s_)))
            o.put(d[0], np.asarray(aops.raw(gx), np.float32).reshape(-1), d[1])
            outs.append(d)
        return outs

    def bwd(self, dy, need_dx=True):
        return self._grads(dy)[0]

    def bwd_multi(self, dy):
        return self._grads(dy)


class _LastStep(_Node):
    def fwd(self, x, shape):
        B, T, D = shape
        self.shape = shape
        y = self.buf("y", B * D)
        self.net.ops.cols(y, x, B, D, T * D, (T - 1) * D, 1)
        return y, (B, D)

    def bwd(self, dy, need_dx=True):
        B, T, D = self.shape
        dx = self.gbuf("dx", B * T * D)
        self.net.ops.fill(dx, B * T * D, 0.0)
        self.net.ops.cols(dy, dx, B, D, T * D, (T - 1) * D, 0)
        return dx


class _Recurrent(_Node):
    """lstm / gru over [B, T, n_in]: the input products for every step at once, then a step at a time the product
    with the previous state and one kernel for the cell; backward the same in reverse, the weight gradients of the
    inputs in one product."""

    def __init__(self, net, key, layer, p, gru):
        super().__init__(net, key)
        self.n, self.H, self.seq, self.gru = layer.n_in, layer.n_hidden, layer.seq, gru
        if gru:
            self.Wzr, self.bzr, self.Wh, self.bh = (net.param(getattr(p, n)) for n in ("Wzr", "bzr", "Wh", "bh"))
        else:
            self.W, self.b = net.param(p.W), net.param(p.b)

    def fwd(self, x, shape):
        if len(shape) != 3 or shape[2] != self.n:
            raise TrainError(f"{'gru' if self.gru else 'lstm'}({self.n}, {self.H}) got {tuple(shape)}")
        o, ps = self.net.ops, self.net.passes
        B, T, n = shape
        H, M = self.H, B * T
        self.x, self.B, self.T = x, B, T
        HS = self.buf("hs", M * H)
        if not self.gru:
            Z, CS = self.buf("z", M * 4 * H), self.buf("cs", M * H)
            o.gemm(x, self.W.w, Z, M, 4 * H, n, n, 1, 4 * H, 1, 4 * H, bias=self.b.w, passes=ps)
            Wh = _ofs(self.W.w, n * 4 * H)
            for t in range(T):
                if t:
                    o.gemm(_ofs(HS, (t - 1) * H), Wh, _ofs(Z, t * 4 * H), B, 4 * H, H, T * H, 1, 4 * H, 1, T * 4 * H,
                           beta=True, passes=ps)
                o.lstm_f(Z, CS, HS, B, H, T, t)
            self.Z, self.CS = Z, CS
        else:
            ZR, RH, CX = self.buf("zr", M * 2 * H), self.buf("rh", M * H), self.buf("cx", M * H)
            o.gemm(x, self.Wzr.w, ZR, M, 2 * H, n, n, 1, 2 * H, 1, 2 * H, bias=self.bzr.w, passes=ps)
            o.gemm(x, self.Wh.w, CX, M, H, n, n, 1, H, 1, H, bias=self.bh.w, passes=ps)
            for t in range(T):
                if t:
                    o.gemm(_ofs(HS, (t - 1) * H), _ofs(self.Wzr.w, n * 2 * H), _ofs(ZR, t * 2 * H), B, 2 * H, H,
                           T * H, 1, 2 * H, 1, T * 2 * H, beta=True, passes=ps)
                o.gru_a(ZR, HS, RH, B, H, T, t)
                if t:
                    o.gemm(_ofs(RH, t * H), _ofs(self.Wh.w, n * H), _ofs(CX, t * H), B, H, H, T * H, 1, H, 1, T * H,
                           beta=True, passes=ps)
                o.gru_b(CX, ZR, HS, B, H, T, t)
            self.ZR, self.RH, self.CX = ZR, RH, CX
        self.HS = HS
        if self.seq:
            return HS, (B, T, H)
        y = self.buf("last", B * H)
        o.cols(y, HS, B, H, T * H, (T - 1) * H, 1)
        return y, (B, H)

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        B, T, H, n = self.B, self.T, self.H, self.n
        M = B * T
        if not self.seq:                                # only the last step had a gradient
            full = self.gbuf("dyfull", M * H)
            o.fill(full, M * H, 0.0)
            o.cols(dy, full, B, H, T * H, (T - 1) * H, 0)
            dy = full
        dhn = self.gbuf("dhn", B * H)
        o.fill(dhn, B * H, 0.0)
        HS = self.HS
        if not self.gru:
            DZ, dcn = self.gbuf("dz", M * 4 * H), self.gbuf("dcn", B * H)
            o.fill(dcn, B * H, 0.0)
            Wh = _ofs(self.W.w, n * 4 * H)
            for t in range(T - 1, -1, -1):
                o.lstm_b(self.Z, self.CS, dy, dhn, dcn, DZ, B, H, T, t)
                if t:
                    o.gemm(_ofs(DZ, t * 4 * H), Wh, dhn, B, H, 4 * H, T * 4 * H, 1, 1, 4 * H, H, passes=ps)
            o.colsum(DZ, None, None, None, None, self.b.g, M, 4 * H, 0)
            o.gemm(self.x, DZ, self.W.g, n, 4 * H, M, 1, n, 4 * H, 1, 4 * H, beta=True, passes=ps)
            Wg_h = _ofs(self.W.g, n * 4 * H)
            for t in range(1, T):
                o.gemm(_ofs(HS, (t - 1) * H), _ofs(DZ, t * 4 * H), Wg_h, H, 4 * H, B, 1, T * H, T * 4 * H, 1, 4 * H,
                       beta=True, passes=ps)
            if not need_dx:
                return None
            dx = self.gbuf("dx", M * n)
            o.gemm(DZ, self.W.w, dx, M, n, 4 * H, 4 * H, 1, 1, 4 * H, n, passes=ps)
            return dx
        DCX, DZR = self.gbuf("dcx", M * H), self.gbuf("dzr", M * 2 * H)
        dhp, drh = self.gbuf("dhp", B * H), self.gbuf("drh", B * H)
        for t in range(T - 1, -1, -1):
            o.gru_b1(self.CX, self.ZR, HS, dy, dhn, DCX, DZR, dhp, B, H, T, t)
            o.gemm(_ofs(DCX, t * H), _ofs(self.Wh.w, n * H), drh, B, H, H, T * H, 1, 1, H, H, passes=ps)
            o.gru_b2(drh, self.ZR, HS, DZR, dhp, B, H, T, t)
            o.gemm(_ofs(DZR, t * 2 * H), _ofs(self.Wzr.w, n * 2 * H), dhp, B, H, 2 * H, T * 2 * H, 1, 1, 2 * H, H,
                   beta=True, passes=ps)
            dhn, dhp = dhp, dhn
        o.colsum(DZR, None, None, None, None, self.bzr.g, M, 2 * H, 0)
        o.colsum(DCX, None, None, None, None, self.bh.g, M, H, 0)
        o.gemm(self.x, DZR, self.Wzr.g, n, 2 * H, M, 1, n, 2 * H, 1, 2 * H, beta=True, passes=ps)
        o.gemm(self.x, DCX, self.Wh.g, n, H, M, 1, n, H, 1, H, beta=True, passes=ps)
        o.gemm(self.RH, DCX, _ofs(self.Wh.g, n * H), H, H, M, 1, H, H, 1, H, beta=True, passes=ps)
        for t in range(1, T):
            o.gemm(_ofs(HS, (t - 1) * H), _ofs(DZR, t * 2 * H), _ofs(self.Wzr.g, n * 2 * H), H, 2 * H, B, 1, T * H,
                   T * 2 * H, 1, 2 * H, beta=True, passes=ps)
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * n)
        o.gemm(DZR, self.Wzr.w, dx, M, n, 2 * H, 2 * H, 1, 1, 2 * H, n, passes=ps)
        o.gemm(DCX, self.Wh.w, dx, M, n, H, H, 1, 1, H, n, beta=True, passes=ps)
        return dx


class _Pool(_Node):
    """max_pool / avg_pool (k x k windows, the rest of the rows and columns left out), global_avg_pool."""

    def __init__(self, net, key, layer, mode, glob=False):
        super().__init__(net, key)
        self.kk, self.mode, self.glob = int(getattr(layer, "k", 0) or 0), mode, glob
        self.ones = {}

    def fwd(self, x, shape):
        if len(shape) != 4:
            raise TrainError(f"pooling on the GPU takes [N, C, H, W], got {tuple(shape)}")
        o = self.net.ops
        N, C, H, W = shape
        self.x, self.shape = x, shape
        if self.glob:
            S = H * W
            one = self.ones.get(S)
            if one is None:
                one = o.alloc(S)
                o.fill((one, 0), S, 1.0 / S)
                self.net.owned.append(one)
                one = self.ones[S] = (one, 0)
            self.one = one
            y = self.buf("y", N * C)
            o.gemm(x, one, y, N * C, 1, S, S, 1, 1, 1, 1, passes=3)
            return y, (N, C)
        kk = self.kk
        y = self.buf("y", N * C * (H // kk) * (W // kk))
        o.pool_f(x, y, N * C, H, W, kk, self.mode)
        self.y = y
        return y, (N, C, H // kk, W // kk)

    def bwd(self, dy, need_dx=True):
        o = self.net.ops
        N, C, H, W = self.shape
        dx = self.gbuf("dx", N * C * H * W)
        if self.glob:
            S = H * W
            o.gemm(dy, self.one, dx, N * C, S, 1, 1, 1, 0, 1, S, passes=3)
            return dx
        o.pool_b(self.x, self.y, dy, dx, N * C, H, W, self.kk, self.mode)
        return dx


class _GraphDev:
    """A graph on the card, in the forms the layers need, each made once: the edges in the order of their target
    (CSR rows = targets) and in the order of their source (the transpose, for the backward pass), edge weights,
    the edges without self loops, the nodes of each graph (pooling)."""

    def __init__(self, net, g):
        self.net, self.g, self.cache, self.bufs = net, g, {}, []
        self.N, self.E = int(g.n), int(g.edges)

    def _put(self, a, dtype=np.int32):
        o = self.net.ops
        a = np.ascontiguousarray(np.asarray(a).astype(dtype).reshape(-1))
        b = o.alloc(max(1, a.size))
        if a.size:
            o.put(b, a)
        self.bufs.append(b)
        return (b, 0)

    def close(self):
        for b in self.bufs:
            self.net.ops.free(b)
        self.bufs, self.cache = [], {}

    def csr(self, which="all"):
        """(forward, transpose): each {col, off, n, vperm, row, ids}; forward rows are targets, columns sources."""
        hit = self.cache.get(("csr", which))
        if hit is not None:
            return hit
        g = self.g
        src, dst = np.asarray(g.src, np.int64), np.asarray(g.dst, np.int64)
        rel = np.asarray(g.rel, np.int64)
        hop = np.asarray(g.hop, np.int64)
        if isinstance(which, tuple):                     # ("noself", relation ids): the edges of those relations,
            from .graphs import owned_edges              # renumbered (a layer that owns some relations)
            src, dst, rel = owned_edges(g, 0, which[1])
            hop = np.ones(len(src), np.int64)
        elif which == "noself":
            keep = src != dst
            src, dst, rel, hop = src[keep], dst[keep], rel[keep], hop[keep]
        N, E = self.N, len(src)
        order = np.argsort(dst, kind="stable")
        s_, d_ = src[order], dst[order]
        off = np.concatenate([[0], np.cumsum(np.bincount(d_, minlength=N))])
        tperm = np.argsort(s_, kind="stable")              # edges (in target order) grouped by source
        toff = np.concatenate([[0], np.cumsum(np.bincount(s_, minlength=N))])
        fwd = dict(col=self._put(s_), off=self._put(off), row=self._put(d_), ids=self._put(np.arange(E)), n=N, E=E,
                   src=s_, dst=d_, rel=rel[order], hop=hop[order], eoff=self._put(np.arange(E + 1)))
        bwd = dict(col=self._put(d_[tperm]), off=self._put(toff), vperm=self._put(tperm), n=N, E=E)
        hit = self.cache[("csr", which)] = (fwd, bwd)
        return hit

    def custom(self, key, src, dst):
        """A CSR of my own edges (src -> dst), cached under key: e.g. every node to every node of its graph."""
        hit = self.cache.get(("custom", key))
        if hit is None:
            src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
            order = np.argsort(dst, kind="stable")
            s_, d_ = src[order], dst[order]
            E = len(s_)
            off = np.concatenate([[0], np.cumsum(np.bincount(d_, minlength=self.N))])
            hit = self.cache[("custom", key)] = dict(col=self._put(s_), off=self._put(off), row=self._put(d_),
                                                      ids=self._put(np.arange(E)), eoff=self._put(np.arange(E + 1)),
                                                      n=self.N, E=E, src=s_, dst=d_)
        return hit

    def values(self, kind, which="all"):
        """Per edge (target order): "gcn" 1 / sqrt(deg(target) deg(source)); "mean" 1 / deg(target)."""
        hit = self.cache.get(("val", kind, which))
        if hit is None:
            f, _ = self.csr(which)
            deg = np.asarray(self.g.in_degree, np.float64)
            if kind == "gcn":
                v = 1.0 / np.sqrt(np.maximum(deg[f["dst"]] * deg[f["src"]], 1.0))
            else:
                cnt = np.bincount(f["dst"], minlength=self.N).astype(np.float64)
                v = 1.0 / np.maximum(cnt, 1.0)[f["dst"]]
            hit = self.cache[("val", kind, which)] = self._put(v, np.float32)
        return hit

    def pool(self, how):
        """The nodes of each graph: forward rows are graphs; values 1 / count (mean) or none (sum)."""
        hit = self.cache.get(("pool", how))
        if hit is None:
            batch = np.asarray(self.g.batch, np.int64)
            G = int(self.g.graphs)
            order = np.argsort(batch, kind="stable")
            cnt = np.bincount(batch, minlength=G)
            off = np.concatenate([[0], np.cumsum(cnt)])
            pos = np.empty(len(order), np.int64)
            pos[order] = np.arange(len(order))
            val = (1.0 / np.maximum(cnt, 1))[batch[order]] if how == "mean" else None
            fwd = dict(col=self._put(order), off=self._put(off), n=G, val=None if val is None else
                       self._put(val, np.float32))
            bwd = dict(col=self._put(batch), off=self._put(np.arange(self.N + 1)), vperm=self._put(pos), n=self.N)
            hit = self.cache[("pool", how)] = (fwd, bwd, G)
        return hit


def _need_graph(net, what):
    if net.graph is None:
        raise TrainError(f"{what} needs the graph: fit(..., graph=g)")
    if net.gdev is None:
        net.gdev = _GraphDev(net, net.graph)
    return net.gdev


class _GCN(_Node):
    """h = x W, then each node sums its in-edges' h[source] / sqrt(deg deg) (+ b)."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.F = layer.n_in, layer.n_out
        self.W = net.param(p.W)
        self.b = net.param(p.b) if layer.bias else None

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "gcn")
        o, M, K, F = self.net.ops, _rows(shape), self.K, self.F
        self.x, self.M = x, M
        h, y = self.buf("h", M * F), self.buf("y", M * F)
        o.gemm(x, self.W.w, h, M, F, K, K, 1, F, 1, F, passes=self.net.passes)
        f, _ = gd.csr()
        o.spmm(h, f["col"], gd.values("gcn"), None, f["off"], y, M, 1, F)
        if self.b is not None:
            o.pos_f(y, self.b.w, y, M, F, 1)
        return y, shape[:-1] + (F,)

    def bwd(self, dy, need_dx=True):
        gd = self.net.gdev
        o, M, K, F = self.net.ops, self.M, self.K, self.F
        if self.b is not None:
            o.pos_b(dy, self.b.g, M, F, 1)
        _, t = gd.csr()
        dh = self.gbuf("dh", M * F)
        o.spmm(dy, t["col"], gd.values("gcn"), t["vperm"], t["off"], dh, M, 1, F)
        o.gemm(self.x, dh, self.W.g, K, F, M, 1, K, F, 1, F, beta=True, passes=self.net.passes)
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * K)
        o.gemm(dh, self.W.w, dx, M, K, F, F, 1, 1, F, K, passes=self.net.passes)
        return dx


class _SAGE(_Node):
    """x Ws + aggregate_{j -> i}(x_j) Wn + b, aggregate the mean or the sum of the in-edges' sources."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        if layer.agg not in ("mean", "sum"):
            raise TrainError(f"sage(aggregate=\"{layer.agg}\") is not on the GPU trainer yet (mean, sum)")
        self.K, self.F, self.mean = layer.n_in, layer.n_out, layer.agg == "mean"
        self.Ws, self.Wn, self.b = net.param(p.Ws), net.param(p.Wn), net.param(p.b)

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "sage")
        o, M, K, F, ps = self.net.ops, _rows(shape), self.K, self.F, self.net.passes
        self.x, self.M = x, M
        f, _ = gd.csr()
        self.m = self.buf("m", M * K)
        o.spmm(x, f["col"], gd.values("mean") if self.mean else None, None, f["off"], self.m, M, 1, K)
        y = self.buf("y", M * F)
        o.gemm(x, self.Ws.w, y, M, F, K, K, 1, F, 1, F, bias=self.b.w, passes=ps)
        o.gemm(self.m, self.Wn.w, y, M, F, K, K, 1, F, 1, F, beta=True, passes=ps)
        return y, shape[:-1] + (F,)

    def bwd(self, dy, need_dx=True):
        gd = self.net.gdev
        o, M, K, F, ps = self.net.ops, self.M, self.K, self.F, self.net.passes
        o.gemm(self.x, dy, self.Ws.g, K, F, M, 1, K, F, 1, F, beta=True, passes=ps, db=self.b.g)
        o.gemm(self.m, dy, self.Wn.g, K, F, M, 1, K, F, 1, F, beta=True, passes=ps)
        if not need_dx:
            return None
        dm, dx = self.gbuf("dm", M * K), self.gbuf("dx", M * K)
        o.gemm(dy, self.Wn.w, dm, M, K, F, F, 1, 1, F, K, passes=ps)
        o.gemm(dy, self.Ws.w, dx, M, K, F, F, 1, 1, F, K, passes=ps)
        _, t = gd.csr()
        o.spmm(dm, t["col"], gd.values("mean") if self.mean else None, t["vperm"] if self.mean else None, t["off"],
               dx, M, 1, K, add=True)
        return dx


class _GIN(_Node):
    """mlp((1 + eps) x + sum of the neighbours' x), the self loops left out."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K = layer.n_in
        self.eps = net.param(p.eps)
        self.mlp = net.compile(layer.mlp, p.mlp, key + ".m")

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "gin")
        o, M, K = self.net.ops, _rows(shape), self.K
        self.x, self.M = x, M
        f, _ = gd.csr("noself")
        z = self.buf("z", M * K)
        o.spmm(x, f["col"], None, None, f["off"], z, M, 1, K)
        o.scale_add(z, x, self.eps.w, M * K, 1.0)
        return self.mlp.fwd(z, shape)

    def bwd(self, dy, need_dx=True):
        gd = self.net.gdev
        o, M, K = self.net.ops, self.M, self.K
        dz = self.mlp.bwd(dy, True)
        o.dot_acc(dz, self.x, M * K, self.eps.g)
        if not need_dx:
            return None
        _, t = gd.csr("noself")
        dx = self.gbuf("dx", M * K)
        o.spmm(dz, t["col"], None, None, t["off"], dx, M, 1, K)
        o.scale_add(dx, dz, self.eps.w, M * K, 1.0)
        return dx


class _GAT(_Node):
    """Graph attention: h = x W (heads x features); a score per edge and head, leaky(a_src . h[source] +
    a_dst . h[target]); a softmax over each node's in-edges; the weighted sum of the sources' h; heads concatenated
    or averaged; + b."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.F, self.H, self.concat, self.slope = layer.n_in, layer.n_out, layer.h, layer.concat, \
            float(layer.slope)
        self.W, self.a_src, self.a_dst, self.b = (net.param(getattr(p, n)) for n in ("W", "a_src", "a_dst", "b"))
        self.Mavg = None

    def _avg(self):
        if self.Mavg is None:                            # [H F, F]: the mean over the heads as a product
            m = np.tile(np.eye(self.F), (self.H, 1)) / self.H
            b = self.net.ops.alloc(m.size)
            self.net.ops.put(b, m.astype(np.float32).reshape(-1))
            self.net.owned.append(b)
            self.Mavg = (b, 0)
        return self.Mavg

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "gat")
        o, M, K, F, H, ps = self.net.ops, _rows(shape), self.K, self.F, self.H, self.net.passes
        HF = H * F
        f, _ = gd.csr()
        E = f["E"]
        self.x, self.M = x, M
        h = self.h = self.buf("h", M * HF)
        o.gemm(x, self.W.w, h, M, HF, K, K, 1, HF, 1, HF, passes=ps)
        es, ed = self.buf("es", M * H), self.buf("ed", M * H)
        for a, out in ((self.a_src, es), (self.a_dst, ed)):     # per head: h[n, head, :] . a[head, :]
            o.gemm(h, a.w, out, M, 1, F, HF, 1, 1, 1, H, batch=(1, H), sa=(0, F), sb=(0, F), sc=(0, 1), passes=ps)
        sc, tmp = self.buf("s", E * H), self.buf("st", E * H)
        o.gather(es, f["col"], sc, E, H, 0)
        o.gather(ed, f["row"], tmp, E, H, 0)
        o.add3(sc, sc, tmp, E * H)
        self.s, self.al = sc, self.buf("al", E * H)
        o.seg_softmax(sc, f["off"], self.al, M, H, self.slope)
        y = self.buf("agg", M * HF)
        o.spmm(h, f["col"], self.al, None, f["off"], y, M, H, F)
        if not self.concat:
            self.agg = y
            y = self.buf("y", M * F)
            o.gemm(self.agg, self._avg(), y, M, F, HF, HF, 1, F, 1, F, passes=3)
        width = HF if self.concat else F
        o.pos_f(y, self.b.w, y, M, width, 1)
        return y, shape[:-1] + (width,)

    def bwd(self, dy, need_dx=True):
        gd = self.net.gdev
        o, M, K, F, H, ps = self.net.ops, self.M, self.K, self.F, self.H, self.net.passes
        HF = H * F
        f, t = gd.csr()
        E = f["E"]
        width = HF if self.concat else F
        o.pos_b(dy, self.b.g, M, width, 1)
        if not self.concat:
            dfull = self.gbuf("dfull", M * HF)
            o.gemm(dy, self._avg(), dfull, M, HF, F, F, 1, 1, F, HF, passes=3)
            dy = dfull
        dh = self.gbuf("dh", M * HF)                    # through the weighted sum: the sources' h
        o.spmm(dy, t["col"], self.al, t["vperm"], t["off"], dh, M, H, F)
        dal = self.gbuf("dal", E * H)                   # and the weights
        o.edge_dot(dy, self.h, f["row"], f["col"], dal, E, H, F)
        ds = self.gbuf("ds", E * H)
        o.seg_softmax_b(self.s, self.al, dal, f["off"], ds, M, H, self.slope)
        des, ded = self.gbuf("des", M * H), self.gbuf("ded", M * H)
        o.spmm(ds, t["vperm"], None, None, t["off"], des, M, 1, H)     # sum over each node's out-edges
        o.spmm(ds, f["ids"], None, None, f["off"], ded, M, 1, H)       # and its in-edges
        for a, de in ((self.a_src, des), (self.a_dst, ded)):
            o.gemm(self.h, de, a.g, F, 1, M, 1, HF, H, 1, 1, beta=True, batch=(1, H), sa=(0, F), sb=(0, 1),
                   sc=(0, F), passes=3)
            o.gemm(de, a.w, dh, M, F, 1, H, 1, F, 1, HF, beta=True, batch=(1, H), sa=(0, 1), sb=(0, F), sc=(0, F),
                   passes=3)
        o.gemm(self.x, dh, self.W.g, K, HF, M, 1, K, HF, 1, HF, beta=True, passes=ps)
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * K)
        o.gemm(dh, self.W.w, dx, M, K, HF, HF, 1, 1, HF, K, passes=ps)
        return dx


class _EdgeAttention:
    """Attention along given edges, a softmax over each node's in-edges: the core of graph_transformer (keys and
    values by relation, a bias by relation and by hop) and global_attention (every node of a graph to every
    other). q [N, d]; the keys / values of edge e are rows kidx[e] of KR / VR ([rows, d]); scores
    q[target] . k / sqrt(hd) (+ bias[e, head]); msg [N, d] = the weighted sum of the values."""

    @staticmethod
    def fwd(node, f, q, KR, VR, kidx, N, H, hd, bias_parts=(), vidx=None):
        o, E = node.net.ops, f["E"]
        d = H * hd
        KE, VE = node.buf("ke", E * d), node.buf("ve", E * d)
        o.gather(KR, kidx, KE, E, d, 0)
        o.gather(VR, kidx if vidx is None else vidx, VE, E, d, 0)
        sc = node.buf("sc", E * H)
        o.edge_dot(q, KE, f["row"], f["ids"], sc, E, H, hd)              # q[target] . k
        o.axpy(sc, sc, E * H, 1.0 / math.sqrt(hd) - 1.0)
        if bias_parts:
            tmp = node.buf("sb", E * H)
            for table, ids in bias_parts:                                 # + a bias per relation / hop and head
                o.gather(table, ids, tmp, E, H, 0)
                o.add3(sc, sc, tmp, E * H)
        al = node.buf("al", E * H)
        o.seg_softmax(sc, f["off"], al, N, H, 1.0)
        msg = node.buf("msg", N * d)
        o.spmm(VE, f["ids"], al, None, f["off"], msg, N, H, hd)
        node.ea = (KE, VE, sc, al)
        return msg

    @staticmethod
    def bwd(node, f, q, kidx, dmsg, N, H, hd, rows, bias_grads=(), vidx=None, vrows=None):
        """(dq, dKR, dVR): dKR / dVR [rows, d] (rows of the key / value tables)."""
        o, E = node.net.ops, f["E"]
        d = H * hd
        KE, VE, sc, al = node.ea
        dVE = node.gbuf("dve", E * d)                                     # each edge's value: alpha dmsg[target]
        o.spmm(dmsg, f["row"], al, None, f["eoff"], dVE, E, H, hd)
        dal = node.gbuf("dal", E * H)
        o.edge_dot(dmsg, VE, f["row"], f["ids"], dal, E, H, hd)
        ds = node.gbuf("ds", E * H)
        o.seg_softmax_b(sc, al, dal, f["off"], ds, N, H, 1.0)
        for gtable, ids, rows_ in bias_grads:
            o.embed_b(ds, ids, gtable, E, H, rows_)
        o.axpy(ds, ds, E * H, 1.0 / math.sqrt(hd) - 1.0)
        dq = node.gbuf("dq", N * d)
        o.spmm(KE, f["ids"], ds, None, f["off"], dq, N, H, hd)
        dKE = node.gbuf("dke", E * d)
        o.spmm(q, f["row"], ds, None, f["eoff"], dKE, E, H, hd)
        vrows = rows if vrows is None else vrows
        dKR, dVR = node.gbuf("dkr", rows * d), node.gbuf("dvr", vrows * d)
        o.fill(dKR, rows * d, 0.0)
        o.fill(dVR, vrows * d, 0.0)
        o.embed_b(dKE, kidx, dKR, E, d, rows)
        o.embed_b(dVE, kidx if vidx is None else vidx, dVR, E, d, vrows)
        return dq, dKR, dVR


class _GraphTransformer(_Node):
    """graph_transformer: x + attention over the in-edges (no self loops; keys and values by relation; a bias by
    relation and by hop), then x + mlp(norm(x))."""

    def __init__(self, net, key, layer, p, gk):
        from . import layers as L
        super().__init__(net, key)
        self.d, self.H, self.R, self.max_hop = layer.d, layer.h, layer.R, layer.max_hop
        self.n1 = _Norm(net, key + ".n1", L.LayerNorm(layer.d), p.n1, False)
        self.n2 = _Norm(net, key + ".n2", L.LayerNorm(layer.d), p.n2, False)
        self.n1.gkey, self.n2.gkey = gk + ".n1", gk + ".n2"
        self.Wq, self.Wk, self.Wv, self.Wo, self.bo = (net.param(getattr(p, n)) for n in ("Wq", "Wk", "Wv", "Wo",
                                                                                          "bo"))
        self.rb, self.hb = net.param(p.rel_bias), net.param(p.hop_bias)
        self.mlp = net.compile(layer.mlp, p.mlp, key + ".m", gk + ".m")
        self.ids = self.ids_for = None

    def _ids(self, gd, f):
        hit = gd.cache.get(("gt", id(self)))
        if hit is None:
            if gd.g.relations > self.R:
                raise TrainError(f"graph_transformer(relations={self.R}) got a graph with {gd.g.relations} relations")
            hit = gd.cache[("gt", id(self))] = (gd._put(f["src"] * self.R + f["rel"]), gd._put(f["rel"]),
                                                 gd._put(np.minimum(f["hop"], self.max_hop)))
        return hit

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "graph_transformer")
        o, ps = self.net.ops, self.net.passes
        N, d, H, R = _rows(shape), self.d, self.H, self.R
        hd = d // H
        self.x, self.N = x, N
        f, _ = gd.csr("noself")
        self.E = f["E"]
        h, _ = self.n1.fwd(x, shape)
        self.h = h
        x1 = self.buf("x1", N * d)
        if self.E:
            kidx, rid, hid = self._ids(gd, f)
            q = self.q = self.buf("q", N * d)
            o.gemm(h, self.Wq.w, q, N, d, d, d, 1, d, 1, d, passes=ps)
            KR, VR = self.buf("kr", N * R * d), self.buf("vr", N * R * d)
            for W, out in ((self.Wk, KR), (self.Wv, VR)):               # every node's key / value in every relation
                o.gemm(h, W.w, out, N, d, d, d, 1, d, 1, R * d, batch=(1, R), sa=(0, 0), sb=(0, d * d), sc=(0, d),
                       passes=ps)
            self.msg = _EdgeAttention.fwd(self, f, q, KR, VR, kidx, N, H, hd, ((self.rb.w, rid), (self.hb.w, hid)))
            o.gemm(self.msg, self.Wo.w, x1, N, d, d, d, 1, d, 1, d, bias=self.bo.w, passes=ps, epi=3, X2=x)
        else:
            o.pull(x1, o, x, N * d)
        self.x1 = x1
        h2, _ = self.n2.fwd(x1, shape)
        f2, _ = self.mlp.fwd(h2, shape)
        y = self.buf("y", N * d)
        o.add3(y, x1, f2, N * d)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        N, d, H, R = self.N, self.d, self.H, self.R
        hd = d // H
        dx1 = self.n2.bwd(self.mlp.bwd(dy, True), True, residual=dy)
        if not self.E:
            return dx1
        gd = self.net.gdev
        f, _ = gd.csr("noself")
        kidx, rid, hid = self._ids(gd, f)
        o.gemm(self.msg, dx1, self.Wo.g, d, d, N, 1, d, d, 1, d, beta=True, passes=ps, db=self.bo.g)
        dmsg = self.gbuf("dmsg", N * d)
        o.gemm(dx1, self.Wo.w, dmsg, N, d, d, d, 1, 1, d, d, passes=ps)
        dq, dKR, dVR = _EdgeAttention.bwd(self, f, self.q, kidx, dmsg, N, H, hd, N * R,
                                          ((self.rb.g, rid, R), (self.hb.g, hid, self.max_hop + 1)))
        dh = self.gbuf("dh", N * d)
        o.gemm(self.h, dq, self.Wq.g, d, d, N, 1, d, d, 1, d, beta=True, passes=ps)
        o.gemm(dq, self.Wq.w, dh, N, d, d, d, 1, 1, d, d, passes=ps)
        for W, dR in ((self.Wk, dKR), (self.Wv, dVR)):
            for r in range(R):
                o.gemm(self.h, (dR[0], dR[1] + r * d), (W.g[0], W.g[1] + r * d * d), d, d, N, 1, d, R * d, 1, d,
                       beta=True, passes=ps)
                o.gemm((dR[0], dR[1] + r * d), (W.w[0], W.w[1] + r * d * d), dh, N, d, d, R * d, 1, 1, d, d,
                       beta=True, passes=ps)
        return self.n1.bwd(dh, True, residual=dx1)


class _TypeGroups:
    """The nodes of a graph grouped by type on the card: perm (sorted by type), inv (back), rows of each type."""

    def __init__(self, gd, T, ids=None):
        from .graphs import local_types
        t = local_types(gd.g, gd.N, T, ids)
        perm = np.argsort(t, kind="stable")
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        self.counts = [int(c) for c in np.bincount(np.minimum(t, T), minlength=T + 1)[:T]]
        self.perm, self.inv = gd._put(perm), gd._put(inv)
        self.known = (t < T).astype(np.float32)
        self.known_d = gd._put(self.known, np.float32)


def _groups(net, T, ids=None):
    gd = _need_graph(net, "typed layers")
    key = ("types", T, ids)
    if key not in gd.cache:
        gd.cache[key] = _TypeGroups(gd, T, ids)
    return gd.cache[key]


def _diag(node, name, x, vals, N, W, grad=False):
    """rows of x scaled by vals[row] (a 0 / 1 mask): one sparse product with a diagonal"""
    gd = node.net.gdev
    if ("eye", N) not in gd.cache:
        gd.cache[("eye", N)] = (gd._put(np.arange(N)), gd._put(np.arange(N + 1)))
    col, off = gd.cache[("eye", N)]
    y = (node.gbuf if grad else node.buf)(name, N * W)
    node.net.ops.spmm(x, col, vals, None, off, y, N, 1, W)
    return y


class _TypedLin:
    """typed_linear inside a node: rows sorted by type, one product per type, back in order."""

    def __init__(self, node, tag, W, b, K, F, T):
        self.node, self.tag, self.W, self.b, self.K, self.F, self.T = node, tag, W, b, K, F, T

    def fwd(self, x, N, grp):
        nd, o, K, F = self.node, self.node.net.ops, self.K, self.F
        xs, ys = nd.buf(self.tag + "xs", N * K), nd.buf(self.tag + "ys", N * F)
        o.gather(x, grp.perm, xs, N, K, 0)
        o.fill(ys, N * F, 0.0)
        r0 = 0
        for t, c in enumerate(grp.counts):
            if c:
                o.gemm(_ofs(xs, r0 * K), _ofs(self.W.w, t * K * F), _ofs(ys, r0 * F), c, F, K, K, 1, F, 1, F,
                       passes=nd.net.passes)
                if self.b is not None:
                    o.rowbias(_ofs(ys, r0 * F), _ofs(self.b.w, t * F), c, F)
            r0 += c
        self.xs, self.grp = xs, grp
        y = nd.buf(self.tag + "y", N * F)
        o.gather(ys, grp.inv, y, N, F, 0)
        return y

    def bwd(self, dy, N):
        nd, o, K, F, grp = self.node, self.node.net.ops, self.K, self.F, self.grp
        dys, dxs = nd.gbuf(self.tag + "dys", N * F), nd.gbuf(self.tag + "dxs", N * K)
        o.gather(dy, grp.perm, dys, N, F, 0)
        o.fill(dxs, N * K, 0.0)
        r0 = 0
        for t, c in enumerate(grp.counts):
            if c:
                o.gemm(_ofs(self.xs, r0 * K), _ofs(dys, r0 * F), _ofs(self.W.g, t * K * F), K, F, c, 1, K, F, 1, F,
                       beta=True, passes=nd.net.passes)
                if self.b is not None:
                    o.colsum(_ofs(dys, r0 * F), None, None, None, None, _ofs(self.b.g, t * F), c, F, 0)
                o.gemm(_ofs(dys, r0 * F), _ofs(self.W.w, t * K * F), _ofs(dxs, r0 * K), c, K, F, F, 1, 1, F, K,
                       passes=nd.net.passes)
            r0 += c
        dx = nd.gbuf(self.tag + "dx", N * K)
        o.gather(dxs, grp.inv, dx, N, K, 0)
        return dx


class _TypedNormOp:
    """typed_norm inside a node: a layer norm per type on the rows of that type; other rows zero."""

    def __init__(self, node, tag, g, b, D, T, eps):
        self.node, self.tag, self.g, self.b, self.D, self.T, self.eps = node, tag, g, b, D, T, eps

    def fwd(self, x, N, grp):
        nd, o, D = self.node, self.node.net.ops, self.D
        xs, ys = nd.buf(self.tag + "xs", N * D), nd.buf(self.tag + "ys", N * D)
        mean, rstd = nd.buf(self.tag + "mu", N), nd.buf(self.tag + "rs", N)
        o.gather(x, grp.perm, xs, N, D, 0)
        o.fill(ys, N * D, 0.0)
        r0 = 0
        for t, c in enumerate(grp.counts):
            if c:
                o.ln_f(_ofs(xs, r0 * D), _ofs(self.g.w, t * D), _ofs(self.b.w, t * D), _ofs(ys, r0 * D),
                       _ofs(mean, r0), _ofs(rstd, r0), c, D, self.eps, False)
            r0 += c
        self.xs, self.mean, self.rstd, self.grp = xs, mean, rstd, grp
        y = nd.buf(self.tag + "y", N * D)
        o.gather(ys, grp.inv, y, N, D, 0)
        return y

    def bwd(self, dy, N):
        nd, o, D, grp = self.node, self.node.net.ops, self.D, self.grp
        dys, dxs = nd.gbuf(self.tag + "dys", N * D), nd.gbuf(self.tag + "dxs", N * D)
        o.gather(dy, grp.perm, dys, N, D, 0)
        o.fill(dxs, N * D, 0.0)
        r0 = 0
        for t, c in enumerate(grp.counts):
            if c:
                a = lambda v, w: _ofs(v, r0 * w)         # noqa: E731
                o.colsum(a(dys, D), a(self.xs, D), a(self.mean, 1), a(self.rstd, 1), _ofs(self.g.g, t * D),
                         _ofs(self.b.g, t * D), c, D, 1)
                o.ln_b(a(self.xs, D), _ofs(self.g.w, t * D), a(dys, D), a(self.mean, 1), a(self.rstd, 1),
                       a(dxs, D), c, D, False, None)
            r0 += c
        dx = nd.gbuf(self.tag + "dx", N * D)
        o.gather(dxs, grp.inv, dx, N, D, 0)
        return dx


class _TypedLinearNode(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.op = _TypedLin(self, "", net.param(p.W), net.param(p.b) if layer.bias else None, layer.n_in,
                            layer.n_out, layer.T)
        self.ids = layer.type_ids

    def fwd(self, x, shape):
        self.N = _rows(shape)
        return self.op.fwd(x, self.N, _groups(self.net, self.op.T, self.ids)), shape[:-1] + (self.op.F,)

    def bwd(self, dy, need_dx=True):
        return self.op.bwd(dy, self.N)


class _TypedNormNode(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.op = _TypedNormOp(self, "", net.param(p.gamma), net.param(p.beta), layer.d, layer.T, layer.eps)
        self.ids = layer.type_ids

    def fwd(self, x, shape):
        self.N = _rows(shape)
        return self.op.fwd(x, self.N, _groups(self.net, self.op.T, self.ids)), shape

    def bwd(self, dy, need_dx=True):
        return self.op.bwd(dy, self.N)


class _HGT(_Node):
    """hgt: typed norm, typed q / k / v / o, keys through each edge's relation (one batched product for every
    relation and head), one softmax over a node's in-edges, residual for the nodes with in-edges, a shared feed-forward
    block for the nodes of the layer's types."""

    def __init__(self, net, key, layer, p, gk):
        super().__init__(net, key)
        self.d, self.H, self.T, self.R = layer.d, layer.h, layer.T, layer.R
        T, d = self.T, self.d
        self.norm = _TypedNormOp(self, "n.", net.param(p.norm.gamma), net.param(p.norm.beta), d, T, layer.norm.eps)
        self.lin = {n: _TypedLin(self, n + ".", net.param(getattr(p, n).W), net.param(getattr(p, n).b), d, d, T)
                    for n in ("q", "k", "v", "o")}
        self.rel = net.param(p.rel)
        self.ffn = net.compile(layer.ffn, p.ffn, key + ".f", gk + ".f") if layer.ffn is not None else None
        self.ids = self.ids_for = None
        self.type_ids, self.rel_ids = layer.type_ids, layer.rel_ids
        self.which = "noself" if self.rel_ids is None else ("noself", self.rel_ids)

    def _ids(self, gd, f):
        key = ("hgt", id(self), self.which)
        if key not in gd.cache:
            if len(f["rel"]) and f["rel"].max() >= self.R:
                raise TrainError(f"hgt(relations={self.R}) got a graph with relation {int(f['rel'].max())}")
            has_in = (np.bincount(f["dst"], minlength=gd.N) > 0).astype(np.float32)
            gd.cache[key] = (gd._put(f["src"] * self.R + f["rel"]), gd._put(f["src"]), gd._put(has_in, np.float32))
        return gd.cache[key]

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "hgt")
        o, ps = self.net.ops, self.net.passes
        N, d, H, R, T = _rows(shape), self.d, self.H, self.R, self.T
        hd = d // H
        grp = _groups(self.net, T, self.type_ids)
        f, _ = gd.csr(self.which)
        self.N, self.E, self.x, self.grp = N, f["E"], x, grp
        xn = self.norm.fwd(x, N, grp)
        q, k, v = (self.lin[n].fwd(xn, N, grp) for n in ("q", "k", "v"))
        self.q, self.k = q, k
        x1 = self.buf("x1", N * d)
        if self.E:
            kidx, vidx, hasin = self._ids(gd, f)
            KR = self.buf("kr", N * R * d)
            o.gemm(k, self.rel.w, KR, N, hd, hd, d, 1, hd, 1, R * d, batch=(R, H), sa=(0, hd),
                   sb=(H * hd * hd, hd * hd), sc=(d, hd), passes=ps)
            msg = _EdgeAttention.fwd(self, f, q, KR, v, kidx, N, H, hd, vidx=vidx)
            om = _diag(self, "om", self.lin["o"].fwd(msg, N, grp), hasin, N, d)
            o.add3(x1, x, om, N * d)
        else:
            o.pull(x1, o, x, N * d)
        self.x1 = x1
        if self.ffn is None:
            return x1, shape
        fo, _ = self.ffn.fwd(x1, shape)
        fm = _diag(self, "fm", fo, grp.known_d, N, d)
        y = self.buf("y", N * d)
        o.add3(y, x1, fm, N * d)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        N, d, H, R = self.N, self.d, self.H, self.R
        hd = d // H
        gd, grp = self.net.gdev, self.grp
        if self.ffn is None:
            dx1 = dy
        else:
            dfm = _diag(self, "dfm", dy, grp.known_d, N, d, grad=True)
            dx1 = self.gbuf("dx1", N * d)
            o.add3(dx1, dy, self.ffn.bwd(dfm, True), N * d)
        dx = self.gbuf("dx", N * d)
        if not self.E:
            dxn_parts = []
        else:
            f, _ = gd.csr(self.which)
            kidx, vidx, hasin = self._ids(gd, f)
            dom = _diag(self, "dom", dx1, hasin, N, d, grad=True)
            dmsg = self.lin["o"].bwd(dom, N)
            dq, dKR, dv = _EdgeAttention.bwd(self, f, self.q, kidx, dmsg, N, H, hd, N * R, vidx=vidx, vrows=N)
            dk = self.gbuf("dk", N * d)
            for r in range(R):                             # back through the relation transforms
                o.gemm(_ofs(dKR, r * d), _ofs(self.rel.w, r * H * hd * hd), dk, N, hd, hd, R * d, 1, 1, hd, d,
                       beta=r > 0, batch=(1, H), sa=(0, hd), sb=(0, hd * hd), sc=(0, hd), passes=ps)
            o.gemm(self.k, dKR, self.rel.g, hd, hd, N, 1, d, R * d, 1, hd, beta=True, batch=(R, H), sa=(0, hd),
                   sb=(d, hd), sc=(H * hd * hd, hd * hd), passes=ps)
            dxn_parts = [self.lin["q"].bwd(dq, N), self.lin["k"].bwd(dk, N), self.lin["v"].bwd(dv, N)]
        if dxn_parts:
            dxn = self.gbuf("dxn", N * d)
            o.add3(dxn, dxn_parts[0], dxn_parts[1], N * d)
            o.axpy(dxn, dxn_parts[2], N * d, 1.0)
            o.add3(dx, dx1, self.norm.bwd(dxn, N), N * d)
        else:
            o.pull(dx, o, dx1, N * d)
        return dx


class _GlobalAttention(_Node):
    """global_attention: x + attention of every node over the nodes of its graph, as edges on the card. Options as
    the layer: fewer heads than the model (ratio), biases on q / k / v, learned keys (every value scaled by
    2 sigmoid(score); above max_keys the best-scored keys), only the nodes of some types."""

    def __init__(self, net, key, layer, p, gk):
        from . import layers as L
        super().__init__(net, key)
        self.d, self.H, self.max_keys = layer.d, getattr(layer, "H", layer.h), layer.max_keys
        self.hd = layer.d // layer.h
        self.inner = self.H * self.hd
        self.layer = layer
        self.n = _Norm(net, key + ".n", L.LayerNorm(layer.d), p.n, False)
        self.n.gkey = gk + ".n"
        self.Wq, self.Wk, self.Wv, self.Wo, self.bo = (net.param(getattr(p, n)) for n in ("Wq", "Wk", "Wv", "Wo",
                                                                                          "bo"))
        self.bias = bool(getattr(layer, "bias", False))
        self.bq, self.bk, self.bv = ((net.param(p.bq), net.param(p.bk), net.param(p.bv)) if self.bias else
                                     (None, None, None))
        self.learned = bool(getattr(layer, "learned", False))
        if self.learned:
            self.ksw, self.ksb = net.param(p.ks_w), net.param(p.ks_b)
        self.T, self.type_ids = getattr(layer, "T", None), getattr(layer, "type_ids", None)

    def _own(self, N):
        if self.T is None:
            return np.ones(N, bool)
        from .graphs import _owned_mask
        return _owned_mask(self.net.graph, N, self.T, self.type_ids).reshape(-1)

    def _gd(self, N):
        g = self.net.graph
        if g is None:
            if self.net.gdev is None or self.net.gdev.N != N:
                from . import layers as L
                if self.net.gdev is not None:
                    self.net.gdev.close()
                self.net.gdev = _GraphDev(self.net, L.Graph(np.zeros(0, int), np.zeros(0, int), N, self_loops=False))
            return self.net.gdev, np.zeros(N, np.int64)
        return _need_graph(self.net, "global_attention"), np.asarray(g.batch, np.int64)

    def _edges(self, N, score):
        gd, batch = self._gd(N)
        own = self._own(N)
        src, dst, picked = [], [], False
        for b_ in np.unique(batch[own]):
            idx = np.nonzero((batch == b_) & own)[0]
            keys = self.layer.keys_of(idx, score) if hasattr(self.layer, "keys_of") else (
                idx if len(idx) <= self.max_keys else idx[np.linspace(0, len(idx) - 1, self.max_keys).astype(int)])
            picked = picked or (self.learned and len(keys) < len(idx))
            src.append(np.tile(keys, len(idx)))
            dst.append(np.repeat(idx, len(keys)))
        src = np.concatenate(src) if src else np.zeros(0, np.int64)
        dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)
        if picked:                                         # keys chosen by this step's scores: edges of this pass
            order = np.argsort(dst, kind="stable")
            s_, d_ = src[order], dst[order]
            E = len(s_)
            put = lambda name, a: _put_buf(self, name, a)              # noqa: E731
            f = dict(col=put("e_col", s_), off=put("e_off", np.concatenate([[0], np.cumsum(np.bincount(
                d_, minlength=N))])), row=put("e_row", d_), ids=put("e_ids", np.arange(E)),
                eoff=put("e_eoff", np.arange(E + 1)), n=N, E=E, src=s_, dst=d_)
            return gd, f, put("e_kid", s_)
        f = gd.custom(("global", self.max_keys, self.type_ids), src, dst)
        ck = ("ga", self.max_keys, self.type_ids)
        if ck not in gd.cache:
            gd.cache[ck] = gd._put(f["src"])
        return gd, f, gd.cache[ck]

    def fwd(self, x, shape):
        o, ps = self.net.ops, self.net.passes
        N, d, H, hd, I = _rows(shape), self.d, self.H, self.hd, self.inner
        self.x, self.N = x, N
        self._gd(N)
        h, _ = self.n.fwd(x, shape)
        self.h = h
        q, KR, VR = self.buf("q", N * I), self.buf("kr", N * I), self.buf("vr", N * I)
        for W, b, out in ((self.Wq, self.bq, q), (self.Wk, self.bk, KR), (self.Wv, self.bv, VR)):
            o.gemm(h, W.w, out, N, I, d, d, 1, I, 1, I, bias=b.w if b is not None else None, passes=ps)
        score = None
        if self.learned:
            sc = self.buf("ks", max(1, N))
            o.gemm(h, self.ksw.w, sc, N, 1, d, d, 1, 1, 1, 1, bias=self.ksb.w, passes=3)
            score = o.get(sc[0], N, sc[1]).astype(np.float64)
            sig = 1.0 / (1.0 + np.exp(-score))
            self.sig = sig
            self.s2 = _put_buf(self, "s2", 2.0 * sig, np.float32)
            self.VR = VR
            VR = _diag(self, "vrs", VR, self.s2, N, I)
        gd, f, kid = self._edges(N, score)
        self.f, self.kid, self.q = f, kid, q
        self.msg = _EdgeAttention.fwd(self, f, q, KR, VR, kid, N, H, hd)
        y = self.buf("y", N * d)
        if self.T is None:
            o.gemm(self.msg, self.Wo.w, y, N, d, I, I, 1, d, 1, d, bias=self.bo.w, passes=ps, epi=3, X2=x)
        else:
            self.own = _put_buf(self, "own", self._own(N), np.float32)
            dl = self.buf("dl", N * d)
            o.gemm(self.msg, self.Wo.w, dl, N, d, I, I, 1, d, 1, d, bias=self.bo.w, passes=ps)
            o.add3(y, x, _diag(self, "dlm", dl, self.own, N, d), N * d)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        N, d, H, hd, I = self.N, self.d, self.H, self.hd, self.inner
        f, kid = self.f, self.kid
        dd = dy if self.T is None else _diag(self, "ddl", dy, self.own, N, d, grad=True)
        o.gemm(self.msg, dd, self.Wo.g, I, d, N, 1, I, d, 1, d, beta=True, passes=ps, db=self.bo.g)
        dmsg = self.gbuf("dmsg", N * I)
        o.gemm(dd, self.Wo.w, dmsg, N, I, d, d, 1, 1, d, I, passes=ps)
        dq, dK, dV = _EdgeAttention.bwd(self, f, self.q, kid, dmsg, N, H, hd, N)
        dh = self.gbuf("dh", N * d)
        if self.learned:                                   # v = v_raw 2 sigmoid(s): back to v_raw and to s
            dvs = self.gbuf("dvs", max(1, N))
            o.edge_dot(dV, self.VR, self.n_ids(N), self.n_ids(N), dvs, N, 1, I)
            g_ = o.get(dvs[0], N, dvs[1]).astype(np.float64) * 2.0 * self.sig * (1.0 - self.sig)
            dsc = _put_buf(self, "dsc", g_, np.float32, grad=True)
            dV = _diag(self, "dvr", dV, self.s2, N, I, grad=True)
            o.gemm(self.h, dsc, self.ksw.g, d, 1, N, 1, d, 1, 1, 1, beta=True, passes=3)
            o.colsum(dsc, None, None, None, None, self.ksb.g, N, 1, 0)
        first = True
        for W, b, dW in ((self.Wq, self.bq, dq), (self.Wk, self.bk, dK), (self.Wv, self.bv, dV)):
            o.gemm(self.h, dW, W.g, d, I, N, 1, d, I, 1, I, beta=True, passes=ps,
                   db=b.g if b is not None else None)
            o.gemm(dW, W.w, dh, N, d, I, I, 1, 1, I, d, beta=not first, passes=ps)
            first = False
        if self.learned:
            o.gemm(dsc, self.ksw.w, dh, N, d, 1, 1, 1, 1, 1, d, beta=True, passes=3)
        return self.n.bwd(dh, True, residual=dy)

    def n_ids(self, N):
        gd = self.net.gdev
        if ("eye", N) not in gd.cache:
            gd.cache[("eye", N)] = (gd._put(np.arange(N)), gd._put(np.arange(N + 1)))
        return gd.cache[("eye", N)][0]


class _Seg:
    """Nodes grouped into segments on the card (a node in segment seg[i], or none: -1): the sum / mean of each
    segment (rows = segments) and the way back (rows = nodes)."""

    def __init__(self, gd, seg, S):
        seg = np.asarray(seg, np.int64)
        N = len(seg)
        inn = seg >= 0
        cnt = np.bincount(seg[inn], minlength=S).astype(np.float64)
        order = np.nonzero(inn)[0][np.argsort(seg[inn], kind="stable")]
        self.S, self.N, self.cnt = S, N, cnt
        self.f_col, self.f_off = gd._put(order), gd._put(np.concatenate([[0], np.cumsum(cnt.astype(np.int64))]))
        inv = 1.0 / np.maximum(cnt, 1.0)
        self.f_mean = gd._put(inv[seg[order]] if len(order) else np.zeros(1), np.float32)
        self.f_one = gd._put(np.ones(max(1, len(order))), np.float32)
        self.b_col = gd._put(np.where(inn, seg, 0)[inn])
        self.b_off = gd._put(np.concatenate([[0], np.cumsum(inn.astype(np.int64))]))
        self.b_one = gd._put(np.ones(max(1, int(inn.sum()))), np.float32)
        self.b_mean = gd._put(inv[seg[inn]] if inn.any() else np.zeros(1), np.float32)

    def reduce(self, o, x, y, F, mean):
        o.spmm(x, self.f_col, self.f_mean if mean else self.f_one, None, self.f_off, y, self.S, 1, F)

    def spread(self, o, s, y, F, mean, add=False):
        o.spmm(s, self.b_col, self.b_mean if mean else self.b_one, None, self.b_off, y, self.N, 1, F, add=add)


def _seg(net, key, seg, S):
    gd = net.gdev
    k = ("seg", key)
    if k not in gd.cache:
        gd.cache[k] = _Seg(gd, seg, S)
    return gd.cache[k]


class _Context:
    """GLA's graph context inside a brain: the mean of each graph's owned nodes (on the card), projected (on the
    card), its component along the mean removed and scaled by tanh(gate) (on the processor: graphs x d numbers), added
    back to the owned nodes (on the card)."""

    def __init__(self, node, W, b, gate, li, d):
        self.node, self.W, self.b, self.gate, self.li, self.d = node, W, b, gate, li, d

    def fwd(self, x, N, seg):
        nd, o, d, S = self.node, self.node.net.ops, self.d, seg.S
        t = f"c{self.li}."
        mu = nd.buf(t + "mu", S * d)
        seg.reduce(o, x, mu, d, True)
        gp = nd.buf(t + "gp", S * d)
        o.gemm(mu, self.W.w, gp, S, d, d, d, 1, d, 1, d, bias=self.b.w, passes=nd.net.passes)
        M = o.get(mu[0], S * d, mu[1]).reshape(S, d).astype(np.float64)
        Gp = o.get(gp[0], S * d, gp[1]).reshape(S, d).astype(np.float64)
        gam = float(o.get(self.gate.w[0], 1, self.gate.w[1] + self.li)[0])
        n = np.sqrt((M * M).sum(-1, keepdims=True))
        big = n > 1e-6
        nn = np.where(big, n, 1e-6)
        u = M / nn
        gu = (Gp * u).sum(-1, keepdims=True)
        gc = Gp - gu * u
        th = math.tanh(gam)
        self.st = (M, Gp, u, nn, big, gu, gc, th)
        out = _put_buf(nd, t + "out", th * gc, np.float32)
        y = nd.buf(t + "y", N * d)
        nd.net.ops.pull(y, o, x, N * d)
        seg.spread(o, out, y, d, False, add=True)
        self.mu = mu
        return y

    def bwd(self, dy, N, seg):
        nd, o, d, S = self.node, self.node.net.ops, self.d, seg.S
        t = f"c{self.li}."
        dsum = nd.gbuf(t + "ds", S * d)
        seg.reduce(o, dy, dsum, d, False)
        D = o.get(dsum[0], S * d, dsum[1]).reshape(S, d).astype(np.float64)
        M, Gp, u, nn, big, gu, gc, th = self.st
        dgam = (1.0 - th * th) * float((D * gc).sum())
        tmp = nd.gbuf(t + "dg1", 1)
        o.put(tmp[0], np.array([dgam], np.float32), tmp[1])
        o.axpy((self.gate.g[0], self.gate.g[1] + self.li), tmp, 1, 1.0)
        dgc = th * D
        du_ = (dgc * u).sum(-1, keepdims=True)
        dG = dgc - du_ * u
        du = -(gu * dgc + du_ * Gp)
        dM = np.where(big, (du - (du * u).sum(-1, keepdims=True) * u) / nn, du / nn)
        dgp = _put_buf(nd, t + "dgp", dG, np.float32, grad=True)
        o.gemm(self.mu, dgp, self.W.g, d, d, S, 1, d, d, 1, d, beta=True, passes=nd.net.passes, db=self.b.g)
        dmu = _put_buf(nd, t + "dmu", dM, np.float32, grad=True)
        o.gemm(dgp, self.W.w, dmu, S, d, d, d, 1, 1, d, d, beta=True, passes=nd.net.passes)       # + dgp W^T
        dx = nd.gbuf(t + "dx", N * d)
        nd.net.ops.pull(dx, o, dy, N * d)
        seg.spread(o, dmu, dx, d, True, add=True)
        return dx


class _Brain(_Node):
    """hgt_brain on the card: the typed input projection (gelu) of the owned nodes, the hgt layers, the global
    attention layers, the graph context after every layer, the final norm of the owned nodes."""

    def __init__(self, net, key, layer, p, gk):
        from . import layers as L
        super().__init__(net, key)
        self.layer, self.d, self.T, self.ids = layer, layer.d, layer.T, layer.type_ids
        d = self.d
        self.inp = _TypedLin(self, "in.", net.param(p.input.W), net.param(p.input.b), d, d, self.T)
        self.layers = [net.compile(l_, pp, f"{key}.l{i}", f"{gk}.l{i}") for i, (l_, pp) in
                       enumerate(zip(layer.layers, p.layers))]
        order = sorted(layer.ga)
        self.ga = {li: net.compile(layer.ga[li], p.attend[order.index(li)], f"{key}.g{li}", f"{gk}.g{li}")
                   for li in order}
        self.ctx = [_Context(self, net.param(p.ctx_W), net.param(p.ctx_b), net.param(p.ctx_gate), li, d)
                    for li in range(layer.L)] if layer.context else []
        self.final = _Norm(net, key + ".fn", L.LayerNorm(d), p.final, False)
        self.final.gkey = gk + ".fn"
        self.nodes = self.layers + [self.ga[k] for k in order] + [self.final]

    def _masks(self, N):
        gd = _need_graph(self.net, "hgt_brain")
        k = ("brain-own", self.T, self.ids)
        if k not in gd.cache:
            from .graphs import _owned_mask
            own = _owned_mask(self.net.graph, N, self.T, self.ids).reshape(-1)
            batch = np.asarray(self.net.graph.batch, np.int64)
            G = max(int(self.net.graph.graphs), int(batch.max(initial=-1)) + 1)
            gd.cache[k] = (gd._put(own, np.float32), gd._put(~own, np.float32), _Seg(gd, np.where(own, batch, -1), G))
        return gd.cache[k]

    def fwd(self, x, shape):
        o = self.net.ops
        N, d = _rows(shape), self.d
        own, notown, seg = self._masks(N)
        grp = _groups(self.net, self.T, self.ids)
        self.N, self.x = N, x
        y = self.inp.fwd(x, N, grp)
        a = self.buf("a", N * d)
        o.act_f(y, a, N * d, KS.ACTS["gelu"])
        self.y = y
        h = self.buf("h0", N * d)
        o.add3(h, _diag(self, "ha", a, own, N, d), _diag(self, "hx", x, notown, N, d), N * d)
        self.trace = []
        for li, nd in enumerate(self.layers):
            h, _ = nd.fwd(h, shape)
            if li in self.ga:
                h, _ = self.ga[li].fwd(h, shape)
            if self.ctx:
                h = self.ctx[li].fwd(h, N, seg)
        self.hlast = h
        f, _ = self.final.fwd(h, shape)
        out = self.buf("out", N * d)
        o.add3(out, _diag(self, "fo", f, own, N, d), _diag(self, "fx", h, notown, N, d), N * d)
        return out, shape

    def bwd(self, dy, need_dx=True):
        o = self.net.ops
        N, d = self.N, self.d
        own, notown, seg = self._masks(N)
        df = _diag(self, "dfo", dy, own, N, d, grad=True)
        dh = self.final.bwd(df, True, residual=_diag(self, "dfx", dy, notown, N, d, grad=True))
        for li in range(len(self.layers) - 1, -1, -1):
            if self.ctx:
                dh = self.ctx[li].bwd(dh, N, seg)
            if li in self.ga:
                dh = self.ga[li].bwd(dh, True)
            dh = self.layers[li].bwd(dh, True)
        da = _diag(self, "dha", dh, own, N, d, grad=True)
        dyp = self.gbuf("dy", N * d)
        o.act_b(self.y, da, dyp, N * d, KS.ACTS["gelu"])
        dx = self.gbuf("dx", N * d)
        o.add3(dx, self.inp.bwd(dyp, N), _diag(self, "dhx", dh, notown, N, d, grad=True), N * d)
        return dx


def _brain_masks(node, N, T, ids):
    gd = _need_graph(node.net, "a brain")
    k = ("brain-own", T, ids)
    if k not in gd.cache:
        from .graphs import _owned_mask
        own = _owned_mask(node.net.graph, N, T, ids).reshape(-1)
        batch = np.asarray(node.net.graph.batch, np.int64)
        G = max(int(node.net.graph.graphs), int(batch.max(initial=-1)) + 1)
        gd.cache[k] = (gd._put(own, np.float32), gd._put(~own, np.float32), _Seg(gd, np.where(own, batch, -1), G))
    return gd.cache[k]


class _BrainInput(_Node):
    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.T, self.ids, self.d = layer.T, layer.type_ids, layer.d
        self.inp = _TypedLin(self, "in.", net.param(p.input.W), net.param(p.input.b), self.d, self.d, self.T)

    def fwd(self, x, shape):
        o, N, d = self.net.ops, _rows(shape), self.d
        own, notown, _ = _brain_masks(self, N, self.T, self.ids)
        self.N = N
        self.y = self.inp.fwd(x, N, _groups(self.net, self.T, self.ids))
        a = self.buf("a", N * d)
        o.act_f(self.y, a, N * d, KS.ACTS["gelu"])
        h = self.buf("h", N * d)
        o.add3(h, _diag(self, "ha", a, own, N, d), _diag(self, "hx", x, notown, N, d), N * d)
        return h, shape

    def bwd(self, dh, need_dx=True):
        o, N, d = self.net.ops, self.N, self.d
        own, notown, _ = _brain_masks(self, N, self.T, self.ids)
        da = _diag(self, "dha", dh, own, N, d, grad=True)
        dyp = self.gbuf("dy", N * d)
        o.act_b(self.y, da, dyp, N * d, KS.ACTS["gelu"])
        dx = self.gbuf("dx", N * d)
        o.add3(dx, self.inp.bwd(dyp, N), _diag(self, "dhx", dh, notown, N, d, grad=True), N * d)
        return dx


class _BrainLayer(_Node):
    def __init__(self, net, key, layer, p, gk, dag_params):
        super().__init__(net, key)
        self.T, self.ids, self.d = layer.T, layer.type_ids, layer.hgt.d
        self.hgt = net.compile(layer.hgt, p.hgt, key + ".h", gk + ".h")
        self.ga = net.compile(layer.ga, p.attend, key + ".g", gk + ".g") if layer.ga is not None else None
        self.ctx = None
        if layer.context:
            if dag_params is None or not hasattr(dag_params, layer.owner):
                raise TrainError(f"brain_layer: no node '{layer.owner}' with the brain's shared weights")
            sp = getattr(dag_params, layer.owner)
            self.ctx = _Context(self, net.param(sp.ctx_W), net.param(sp.ctx_b), net.param(sp.ctx_gate), layer.li,
                                self.d)
        self.nodes = [self.hgt] + ([self.ga] if self.ga is not None else [])

    def fwd(self, x, shape):
        N = _rows(shape)
        self.N = N
        h, _ = self.hgt.fwd(x, shape)
        if self.ga is not None:
            h, _ = self.ga.fwd(h, shape)
        if self.ctx is not None:
            h = self.ctx.fwd(h, N, _brain_masks(self, N, self.T, self.ids)[2])
        return h, shape

    def bwd(self, dh, need_dx=True):
        N = self.N
        if self.ctx is not None:
            dh = self.ctx.bwd(dh, N, _brain_masks(self, N, self.T, self.ids)[2])
        if self.ga is not None:
            dh = self.ga.bwd(dh, True)
        return self.hgt.bwd(dh, True)


class _BrainOutput(_Node):
    def __init__(self, net, key, layer, p):
        from . import layers as L
        super().__init__(net, key)
        self.T, self.ids, self.d = layer.T, layer.type_ids, layer.d
        self.final = _Norm(net, key + ".fn", L.LayerNorm(layer.d), p.final, False)

    def fwd(self, x, shape):
        o, N, d = self.net.ops, _rows(shape), self.d
        own, notown, _ = _brain_masks(self, N, self.T, self.ids)
        self.N = N
        f, _ = self.final.fwd(x, shape)
        out = self.buf("out", N * d)
        o.add3(out, _diag(self, "fo", f, own, N, d), _diag(self, "fx", x, notown, N, d), N * d)
        return out, shape

    def bwd(self, dy, need_dx=True):
        N, d = self.N, self.d
        own, notown, _ = _brain_masks(self, N, self.T, self.ids)
        df = _diag(self, "dfo", dy, own, N, d, grad=True)
        return self.final.bwd(df, True, residual=_diag(self, "dfx", dy, notown, N, d, grad=True))


class _TypedPoolNode(_Node):
    """typed_pool: the mean of every (graph, owned type) on the card; the mix with exp(log_weight) on the processor
    (graphs x types x d numbers); the way back spread on the card."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.layer, self.lw, self.T = layer, net.param(p.log_weight), layer.T

    def fwd(self, x, shape):
        o = self.net.ops
        N, d = _rows(shape), shape[-1]
        gd = _need_graph(self.net, "typed_pool")
        loc, batch, G, cnt, on = self.layer.plan(self.net.graph, N)
        T = self.T
        k = ("tpool", T, self.layer.type_ids, self.layer.presence)
        if k not in gd.cache:
            gd.cache[k] = _Seg(gd, np.where(loc < T, batch * T + loc, -1), G * T)
        seg = gd.cache[k]
        Mb = self.buf("M", G * T * d)
        seg.reduce(o, x, Mb, d, True)
        M = o.get(Mb[0], G * T * d, Mb[1]).reshape(G, T, d).astype(np.float64)
        w = np.exp(o.get(self.lw.w[0], T, self.lw.w[1]).astype(np.float64))
        wo = w[None, :] * on                                       # [G, T]
        den = np.maximum(wo.sum(1, keepdims=True), 1e-8)
        Y = (wo[:, :, None] * M).sum(1) / den
        self.st, self.seg, self.N, self.d, self.G = (M, w, on, den, Y), seg, N, d, G
        return _put_buf(self, "y", Y, np.float32), (G, d)

    def bwd(self, dy, need_dx=True):
        o = self.net.ops
        G, d, T, N = self.G, self.d, self.T, self.N
        M, w, on, den, Y = self.st
        D = o.get(dy[0], G * d, dy[1]).reshape(G, d).astype(np.float64)
        coef = (w[None, :] * on) / den                              # [G, T]
        dM = coef[:, :, None] * D[:, None, :]
        dlw = (coef * ((M - Y[:, None, :]) * D[:, None, :]).sum(-1)).sum(0)
        tmp = self.gbuf("dlw", T)
        o.put(tmp[0], dlw.astype(np.float32), tmp[1])
        o.axpy(self.lw.g, tmp, T, 1.0)
        dMb = _put_buf(self, "dM", dM, np.float32, grad=True)
        dx = self.gbuf("dx", N * d)
        o.fill(dx, N * d, 0.0)
        self.seg.spread(o, dMb, dx, d, True, add=True)
        return dx


class _NodeEncoder(_Node):
    """node_encoder on the card: eleven table lookups (the ids' columns split on the processor: nodes x 11 integers),
    the gated sums, the mix (two products, gelu), the output product scaled per feature, plus the gated words."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.layer, self.d = layer, layer.d
        self.tabs = [net.param(a) for a in (p.tok, p.sub, p.ntype, p.slot, p.fam, p.extra)] + \
            [net.param(a) for a in p.struct]
        self.sizes = list(layer.tables)
        self.ss, self.lg, self.gain = net.param(p.struct_scale), net.param(p.lang_gate), net.param(p.out_gain)
        self.mW, self.mb, self.oW, self.ob = (net.param(a) for a in (p.mix.W, p.mix.b, p.out.W, p.out.b))

    def _scal(self, P, i):
        return float(self.net.ops.get(P.w[0], 1, P.w[1] + i)[0])

    def _dot(self, a, b, n, name):
        o = self.net.ops
        r = self.gbuf(name, 1)
        z = _put_buf(self, "z0", [0])
        o.edge_dot(a, b, z, z, r, 1, 1, n)
        return float(o.get(r[0], 1, r[1])[0])

    def _add_scalar_grad(self, P, i, v, name):
        o = self.net.ops
        t = self.gbuf(name, 1)
        o.put(t[0], np.array([v], np.float32), t[1])
        o.axpy((P.g[0], P.g[1] + i), t, 1, 1.0)

    def fwd(self, x, shape):
        o, d, ps = self.net.ops, self.d, self.net.passes
        N = int(shape[0])
        ids = np.ascontiguousarray(o.get(x[0], N * 11, x[1])).view(np.int32).reshape(N, 11).astype(np.int64)
        self.cols = [_put_buf(self, ("c", i), np.clip(ids[:, i], 0, n - 1)) for i, n in enumerate(self.sizes)]
        E = [self.buf(("e", i), N * d) for i in range(11)]
        for i in range(11):
            o.embed_f(self.tabs[i].w, self.cols[i], E[i], N, d)
        self.E, self.N = E, N
        g = [self._scal(self.lg, i) for i in range(4)]
        ss = self._scal(self.ss, 0)
        words = self.buf("words", N * d)
        o.add3(words, E[0], E[1], N * d)
        where = self.buf("where", N * d)
        o.add3(where, E[6], E[7], N * d)
        for i in (8, 9, 10):
            o.axpy(where, E[i], N * d, 1.0)
        what = self.buf("what", N * d)
        o.pull(what, o, E[2], N * d)
        for k, i in enumerate((3, 4, 5)):
            o.axpy(what, E[i], N * d, g[k])
        o.axpy(what, where, N * d, ss)
        hp = self.buf("hp", N * d)
        o.gemm(words, self.mW.w, hp, N, d, d, d, 1, d, 1, d, bias=self.mb.w, passes=ps)
        o.gemm(what, _ofs(self.mW.w, d * d), hp, N, d, d, d, 1, d, 1, d, beta=True, passes=ps)
        h = self.buf("h", N * d)
        o.act_f(hp, h, N * d, KS.ACTS["gelu"])
        ov = self.buf("ov", N * d)
        o.gemm(h, self.oW.w, ov, N, d, d, d, 1, d, 1, d, bias=self.ob.w, passes=ps)
        gt = self.buf("gt", N * d)
        o.gather(self.gain.w, _put_buf(self, "zeros", np.zeros(N)), gt, N, d, 0)
        y = self.buf("y", N * d)
        o.mul3(y, ov, gt, N * d)
        o.axpy(y, words, N * d, g[3])
        self.st = (g, ss, words, where, what, hp, h, ov, gt)
        return y, (N, d)

    def bwd(self, dy, need_dx=True):
        o, d, ps, N = self.net.ops, self.d, self.net.passes, self.N
        g, ss, words, where, what, hp, h, ov, gt = self.st
        n = N * d
        self._add_scalar_grad(self.lg, 3, self._dot(dy, words, n, "d3"), "s3")
        t = self.gbuf("t", n)
        o.mul3(t, dy, ov, n)
        o.colsum(t, None, None, None, None, self.gain.g, N, d, 0)          # d gain
        do = self.gbuf("do", n)
        o.mul3(do, dy, gt, n)
        o.gemm(h, do, self.oW.g, d, d, N, 1, d, d, 1, d, beta=True, passes=ps, db=self.ob.g)
        dh = self.gbuf("dh", n)
        o.gemm(do, self.oW.w, dh, N, d, d, d, 1, 1, d, d, passes=ps)
        dhp = self.gbuf("dhp", n)
        o.act_b(hp, dh, dhp, n, KS.ACTS["gelu"])
        o.gemm(words, dhp, self.mW.g, d, d, N, 1, d, d, 1, d, beta=True, passes=ps, db=self.mb.g)
        o.gemm(what, dhp, _ofs(self.mW.g, d * d), d, d, N, 1, d, d, 1, d, beta=True, passes=ps)
        dwords = self.gbuf("dwords", n)
        o.gemm(dhp, self.mW.w, dwords, N, d, d, d, 1, 1, d, d, passes=ps)
        o.axpy(dwords, dy, n, g[3])
        dwhat = self.gbuf("dwhat", n)
        o.gemm(dhp, _ofs(self.mW.w, d * d), dwhat, N, d, d, d, 1, 1, d, d, passes=ps)
        for i in (0, 1):
            o.embed_b(dwords, self.cols[i], self.tabs[i].g, N, d, self.sizes[i])
        o.embed_b(dwhat, self.cols[2], self.tabs[2].g, N, d, self.sizes[2])
        sc = self.gbuf("sc", n)
        for k, i in enumerate((3, 4, 5)):
            self._add_scalar_grad(self.lg, k, self._dot(dwhat, self.E[i], n, ("dg", k)), ("sg", k))
            o.fill(sc, n, 0.0)
            o.axpy(sc, dwhat, n, g[k])
            o.embed_b(sc, self.cols[i], self.tabs[i].g, N, d, self.sizes[i])
        self._add_scalar_grad(self.ss, 0, self._dot(dwhat, where, n, "dss"), "sss")
        o.fill(sc, n, 0.0)
        o.axpy(sc, dwhat, n, ss)
        for i in range(6, 11):
            o.embed_b(sc, self.cols[i], self.tabs[i].g, N, d, self.sizes[i])
        return None


class _BrainMemory(_Node):
    """brain_memory on the card: the scores of every node (one product each brain), the k best of every graph chosen
    on the processor from those N numbers, the chosen rows gathered, projected and scaled, the summaries and the null
    slot, all placed by one gather and RMS-normed; the slot mask column added on the way out (graphs x slots)."""

    def __init__(self, net, key, layer, p):
        from . import layers as L
        super().__init__(net, key)
        self.layer, self.bd, self.d = layer, layer.bd, layer.d
        P = lambda a: net.param(a)                                 # noqa: E731
        self.sa_w, self.sa_b, self.sc_w, self.sc_b = P(p.score_a.W), P(p.score_a.b), P(p.score_c.W), P(p.score_c.b)
        self.na, self.nc, self.ua, self.uc = P(p.node_a), P(p.node_c), P(p.sum_a), P(p.sum_c)
        self.temb, self.semb, self.null = P(p.type_emb), P(p.slot_emb), P(p.null)
        self.norm = _Norm(net, key + ".n", L.RMSNorm(layer.d), Record(gamma=p.norm), True)

    def _pick(self, h, sw, sb, ids, T, K, batch, G, N, tag):
        from .graphs import _owned_mask
        o = self.net.ops
        s = self.buf(tag + "s", max(1, N))
        o.gemm(h, sw.w, s, N, 1, self.bd, self.bd, 1, 1, 1, 1, bias=sb.w, passes=3)
        raw = o.get(s[0], N, s[1]).astype(np.float64)
        own = _owned_mask(self.net.graph, N, T, ids).reshape(-1)
        rows, pos = [], []
        for b_ in range(G):
            idx = np.nonzero((batch == b_) & own)[0]
            top = idx[np.argsort(-raw[idx], kind="stable")[:K]]
            rows.append(top)
            pos.append(b_ * K + np.arange(len(top)))
        rows = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
        pos = np.concatenate(pos).astype(np.int64) if pos else np.zeros(0, np.int64)
        return rows, pos, 1.0 / (1.0 + np.exp(-raw[rows])), s

    def fwd_multi(self, ins):
        o, ps, bd, d, L_ = self.net.ops, self.net.passes, self.bd, self.d, self.layer
        (ha, sh), (hc, _), (sa, ssh), (sc, _) = ins
        N, G = int(sh[0]), int(ssh[0])
        g = self.net.graph
        batch = np.asarray(g.batch, np.int64)
        ntype = np.asarray(g.ntype, np.int64)
        ka, kc = L_.ka, L_.kc
        M = ka + kc + 3
        ra, pa, siga, _ = self._pick(ha, self.sa_w, self.sa_b, L_.ids_a, L_.Ta, ka, batch, G, N, "a")
        rc, pc, sigc, _ = self._pick(hc, self.sc_w, self.sc_b, L_.ids_c, L_.Tc, kc, batch, G, N, "c")
        Ra, Rc = len(ra), len(rc)
        P = Ra + Rc + 2 * G + 2                          # [vals a, vals c, summary a, summary c, null, zero]
        pool = self.buf("pool", P * d)
        o.fill(pool, P * d, 0.0)
        st = {}
        for tag, h, rows, sig, W, slot, off in (("a", ha, ra, siga, self.na, 0, 0), ("c", hc, rc, sigc, self.nc, 1, Ra)):
            R = len(rows)
            if not R:
                st[tag] = None
                continue
            rid = _put_buf(self, tag + "rows", rows)
            hr = self.buf(tag + "hr", R * bd)
            o.gather(h, rid, hr, R, bd, 0)
            pr = self.buf(tag + "pr", R * d)
            o.gemm(hr, W.w, pr, R, d, bd, bd, 1, d, 1, d, passes=ps)
            sv = _put_buf(self, tag + "sig", sig, np.float32)
            dst = _ofs(pool, off * d)
            o.spmm(pr, _put_buf(self, tag + "eye", np.arange(R)), sv, None, _put_buf(self, tag + "eo", np.arange(R + 1)),
                   dst, R, 1, d)
            tid = _put_buf(self, tag + "tid", np.minimum(ntype[rows], L_.NT - 1))
            te = self.buf(tag + "te", R * d)
            o.embed_f(self.temb.w, tid, te, R, d)
            o.axpy(dst, te, R * d, 1.0)
            o.rowbias(dst, _ofs(self.semb.w, slot * d), R, d)
            st[tag] = (rows, rid, hr, pr, sv, sig, tid, h)
        for k, (s_in, U, slot) in enumerate(((sa, self.ua, 2), (sc, self.uc, 3))):
            dst = _ofs(pool, (Ra + Rc + k * G) * d)
            o.gemm(s_in, U.w, dst, G, d, bd, bd, 1, d, 1, d, passes=ps)
            o.rowbias(dst, _ofs(self.semb.w, slot * d), G, d)
        nl = _ofs(pool, (Ra + Rc + 2 * G) * d)
        o.axpy(nl, self.null.w, d, 1.0)
        o.axpy(nl, _ofs(self.semb.w, 4 * d), d, 1.0)
        zero_row = Ra + Rc + 2 * G + 1
        idx = np.full((G, M), zero_row, np.int64)
        keep = np.zeros((G, M))
        idx.reshape(-1)[pa // ka * M + pa % ka if ka else []] = np.arange(Ra)
        idx.reshape(-1)[pc // kc * M + ka + pc % kc if kc else []] = Ra + np.arange(Rc)
        keep.reshape(-1)[pa // ka * M + pa % ka if ka else []] = 1.0
        keep.reshape(-1)[pc // kc * M + ka + pc % kc if kc else []] = 1.0
        idx[:, ka + kc] = Ra + Rc + np.arange(G)
        idx[:, ka + kc + 1] = Ra + Rc + G + np.arange(G)
        idx[:, ka + kc + 2] = Ra + Rc + 2 * G
        keep[:, ka + kc:] = 1.0
        self.idx = _put_buf(self, "idx", idx)
        pre = self.buf("pre", G * M * d)
        o.gather(pool, self.idx, pre, G * M, d, 0)
        mem, _ = self.norm.fwd(pre, (G * M, d))
        a = o.get(mem[0], G * M * d, mem[1]).reshape(G * M, d)
        y = _put_buf(self, "y", np.concatenate([a, keep.reshape(-1, 1)], 1), np.float32)
        self.st, self.dims = st, (N, G, M, P, Ra, Rc, sa, sc)
        return y, (G, M, d + 1)

    def bwd_multi(self, dy):
        o, ps, bd, d = self.net.ops, self.net.passes, self.bd, self.d
        N, G, M, P, Ra, Rc, sa, sc = self.dims
        a = o.get(dy[0], G * M * (d + 1), dy[1]).reshape(G * M, d + 1)[:, :d]
        dmem = _put_buf(self, "dmem", a, np.float32, grad=True)
        dpre = self.norm.bwd(dmem, True)
        dpool = self.gbuf("dpool", P * d)
        o.fill(dpool, P * d, 0.0)
        o.embed_b(dpre, self.idx, dpool, G * M, d, P)
        nl = _ofs(dpool, (Ra + Rc + 2 * G) * d)
        o.axpy(self.null.g, nl, d, 1.0)
        o.axpy(_ofs(self.semb.g, 4 * d), nl, d, 1.0)
        dsum = []
        for k, (s_in, U, slot) in enumerate(((sa, self.ua, 2), (sc, self.uc, 3))):
            src = _ofs(dpool, (Ra + Rc + k * G) * d)
            o.gemm(s_in, src, U.g, bd, d, G, 1, bd, d, 1, d, beta=True, passes=ps)
            o.colsum(src, None, None, None, None, _ofs(self.semb.g, slot * d), G, d, 0)
            ds = self.gbuf(("dsum", k), G * bd)
            o.gemm(src, U.w, ds, G, bd, d, d, 1, 1, d, bd, passes=ps)
            dsum.append(ds)
        dh = []
        for tag, W, sw, sb, slot, off in (("a", self.na, self.sa_w, self.sa_b, 0, 0), ("c", self.nc, self.sc_w,
                                                                                     self.sc_b, 1, Ra)):
            dhx = self.gbuf(("dh", tag), N * bd)
            o.fill(dhx, N * bd, 0.0)
            dh.append(dhx)
            s_ = self.st[tag]
            if s_ is None:
                continue
            rows, rid, hr, pr, sv, sig, tid, h = s_
            R = len(rows)
            dv = _ofs(dpool, off * d)
            o.embed_b(dv, tid, self.temb.g, R, d, self.layer.NT)
            o.colsum(dv, None, None, None, None, _ofs(self.semb.g, slot * d), R, d, 0)
            dsg = self.gbuf(tag + "dsg", R)
            o.edge_dot(dv, pr, _put_buf(self, tag + "e1", np.arange(R)), _put_buf(self, tag + "e2", np.arange(R)),
                       dsg, R, 1, d)
            dscore = o.get(dsg[0], R, dsg[1]).astype(np.float64) * sig * (1.0 - sig)
            dpr = self.gbuf(tag + "dpr", R * d)
            o.spmm(dv, _put_buf(self, tag + "e3", np.arange(R)), sv, None, _put_buf(self, tag + "e4", np.arange(R + 1)),
                   dpr, R, 1, d)
            o.gemm(hr, dpr, W.g, bd, d, R, 1, bd, d, 1, d, beta=True, passes=ps)
            dhr = self.gbuf(tag + "dhr", R * bd)
            o.gemm(dpr, W.w, dhr, R, bd, d, d, 1, 1, d, bd, passes=ps)
            o.embed_b(dhr, rid, dhx, R, bd, N)
            dsN = np.zeros(N)
            np.add.at(dsN, rows, dscore)
            dsc = _put_buf(self, tag + "dsN", dsN, np.float32, grad=True)
            o.gemm(h, dsc, sw.g, bd, 1, N, 1, bd, 1, 1, 1, beta=True, passes=3)
            o.colsum(dsc, None, None, None, None, sb.g, N, 1, 0)
            o.gemm(dsc, sw.w, dhx, N, bd, 1, 1, 1, 1, 1, bd, beta=True, passes=3)
        return [dh[0], dh[1], dsum[0], dsum[1]]


class _MixHop(_Node):
    """concat_k (A^k x) W_k + b, A the mean over each node's in-neighbours (no self loops)."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.F, self.hops = layer.n_in, layer.n_out, layer.hops
        self.W, self.b = net.param(p.W), net.param(p.b)

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "mix_hop")
        o, ps = self.net.ops, self.net.passes
        N, K, F, J = _rows(shape), self.K, self.F, len(self.hops)
        f, _ = gd.csr("noself")
        self.x, self.N = x, N
        self.pw = [x]
        for k in range(1, max(self.hops) + 1):
            nxt = self.buf(("p", k), N * K)
            o.spmm(self.pw[-1], f["col"], gd.values("mean", "noself"), None, f["off"], nxt, N, 1, K)
            self.pw.append(nxt)
        y = self.buf("y", N * F * J)
        for i, k in enumerate(self.hops):
            o.gemm(self.pw[k], (self.W.w[0], self.W.w[1] + i * K * F), (y[0], y[1] + i * F), N, F, K, K, 1, F, 1,
                   F * J, passes=ps)
        o.pos_f(y, self.b.w, y, N, F * J, 1)
        return y, shape[:-1] + (F * J,)

    def bwd(self, dy, need_dx=True):
        gd = self.net.gdev
        o, ps = self.net.ops, self.net.passes
        N, K, F, J = self.N, self.K, self.F, len(self.hops)
        _, t = gd.csr("noself")
        o.pos_b(dy, self.b.g, N, F * J, 1)
        top = max(self.hops)
        dp = [None] * (top + 1)
        for i, k in enumerate(self.hops):
            o.gemm(self.pw[k], (dy[0], dy[1] + i * F), (self.W.g[0], self.W.g[1] + i * K * F), K, F, N, 1, K, F * J,
                   1, F, beta=True, passes=ps)
            if dp[k] is None:
                dp[k] = self.gbuf(("dp", k), N * K)
                o.fill(dp[k], N * K, 0.0)
            o.gemm((dy[0], dy[1] + i * F), (self.W.w[0], self.W.w[1] + i * K * F), dp[k], N, K, F, F * J, 1, 1, F,
                   K, beta=True, passes=ps)
        for k in range(top, 0, -1):                     # back along the powers: A^T
            if dp[k] is None:
                continue
            if dp[k - 1] is None:
                dp[k - 1] = self.gbuf(("dp", k - 1), N * K)
                o.fill(dp[k - 1], N * K, 0.0)
            o.spmm(dp[k], t["col"], gd.values("mean", "noself"), t["vperm"], t["off"], dp[k - 1], N, 1, K, add=True)
        return dp[0]


class _GraphMemory(_Node):
    """graph_memory: the k best-scored nodes of every graph, each scaled by the sigmoid of its score: [graphs, k, d].
    The scores are made on the card; the choice (not differentiable) on the processor from N numbers; the rows are
    gathered and scaled in one sparse product, and sent back the same way."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.d, self.k = layer.d, layer.k
        self.w, self.b = net.param(p.w), net.param(p.b)

    def fwd(self, x, shape):
        o, ps = self.net.ops, self.net.passes
        N, d, k = _rows(shape), self.d, self.k
        g = self.net.graph
        batch = np.asarray(g.batch, np.int64) if g is not None else np.zeros(N, np.int64)
        G = int(batch.max()) + 1 if N else 0
        sc = self.buf("sc", max(1, N))
        o.gemm(x, self.w.w, sc, N, 1, d, d, 1, 1, 1, 1, bias=self.b.w, passes=3)
        raw = o.get(sc[0], N, sc[1]).astype(np.float64)
        rows, valid = [], np.zeros((G, k))
        for b_ in range(G):
            idx = np.nonzero(batch == b_)[0]
            top = idx[np.argsort(-raw[idx], kind="stable")[:k]]
            valid[b_, :len(top)] = 1.0
            rows.append(np.concatenate([top, np.full(k - len(top), top[0] if len(top) else 0)]))
        sel = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
        sig = 1.0 / (1.0 + np.exp(-raw[sel]))
        wv = (sig * valid.reshape(-1)).astype(np.float32)
        R = G * k
        order = np.argsort(sel, kind="stable")
        self.N, self.G, self.R, self.x = N, G, R, x
        self.sig, self.valid, self.sel_h = sig, valid.reshape(-1), sel
        put = lambda name, a, dt=np.int32: _put_buf(self, name, a, dt)   # noqa: E731
        self.sel = put("sel", sel)
        self.wv = put("wv", wv, np.float32)
        self.ids = put("ids", np.arange(R + 1))
        self.tcol = put("tcol", order)                     # rows of each node (for the way back)
        self.toff = put("toff", np.concatenate([[0], np.cumsum(np.bincount(sel, minlength=N))]))
        y = self.buf("y", max(1, R * d))
        o.spmm(x, self.sel, self.wv, None, self.ids, y, R, 1, d)
        return y, (G, k, d)

    def bwd(self, dy, need_dx=True):
        o, ps = self.net.ops, self.net.passes
        N, R, d = self.N, self.R, self.d
        dwv = self.gbuf("dwv", max(1, R))
        o.edge_dot(dy, self.x, self.ids, self.sel, dwv, R, 1, d)       # d weight of each row
        dw = o.get(dwv[0], R, dwv[1]).astype(np.float64)
        ds = np.zeros(N)
        np.add.at(ds, self.sel_h, dw * self.valid * self.sig * (1 - self.sig))
        dsc = _put_buf(self, "dsc", ds, np.float32, grad=True)
        o.gemm(self.x, dsc, self.w.g, d, 1, N, 1, d, 1, 1, 1, beta=True, passes=3)     # dw += x^T dscore
        o.colsum(dsc, None, None, None, None, self.b.g, N, 1, 0)
        if not need_dx:
            return None
        dx = self.gbuf("dx", N * d)
        o.spmm(dy, self.tcol, self.wv, self.tcol, self.toff, dx, N, 1, d)
        o.gemm(dsc, self.w.w, dx, N, d, 1, 1, 1, 1, 1, d, beta=True, passes=3)          # + dscore w^T
        return dx


def _put_buf(node, name, a, dtype=np.int32, grad=False):
    a = np.ascontiguousarray(np.asarray(a).astype(dtype).reshape(-1))
    b = (node.gbuf if grad else node.buf)(name, max(1, a.size))
    if a.size:
        node.net.ops.put(b[0], a, b[1])
    return b


class _GraphPool(_Node):
    """One row per graph: the mean or the sum of its nodes."""

    def __init__(self, net, key, layer):
        super().__init__(net, key)
        if layer.how not in ("mean", "sum"):
            raise TrainError(f"graph_pool(\"{layer.how}\") is not on the GPU trainer yet (mean, sum)")
        self.how = layer.how

    def fwd(self, x, shape):
        gd = _need_graph(self.net, "graph_pool")
        f, _, G = gd.pool(self.how)
        D = shape[-1]
        self.M, self.D = _rows(shape), D
        y = self.buf("y", G * D)
        self.net.ops.spmm(x, f["col"], f["val"], None, f["off"], y, G, 1, D)
        return y, (G, D)

    def bwd(self, dy, need_dx=True):
        f, t, _ = self.net.gdev.pool(self.how)
        dx = self.gbuf("dx", self.M * self.D)
        self.net.ops.spmm(dy, t["col"], f["val"], t["vperm"] if f["val"] is not None else None, t["off"], dx,
                          self.M, 1, self.D)
        return dx


class _SwiGLU(_Node):
    """down(act(x Wg) * (x Wu)): the gate product keeps its pre-activation and writes the activation in the same
    pass; the backward pass needs no other buffers than the gate, the up product and their product."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.K, self.F, self.N = layer.d, layer.ff, layer.d_out
        self.act = KS.ACTS[layer.act]
        self.Wg, self.Wu, self.Wd = net.param(p.Wg), net.param(p.Wu), net.param(p.Wd)

    @property
    def takes_residual(self):
        return self.N == self.K

    def fwd(self, x, shape, residual=None):
        o, M, ps = self.net.ops, _rows(shape), self.net.passes
        K, F, N = self.K, self.F, self.N
        self.x, self.M = x, M
        self.h, self.a, self.u = self.buf("h", M * F), self.buf("a", M * F), self.buf("u", M * F)
        o.gemm(x, self.Wg.w, self.a, M, F, K, K, 1, F, 1, F, passes=ps, epi=1, X2=self.h, act=self.act)
        o.gemm(x, self.Wu.w, self.u, M, F, K, K, 1, F, 1, F, passes=ps)
        self.z = self.buf("z", M * F)
        o.mul3(self.z, self.a, self.u, M * F)
        y = self.buf("y", M * N)
        o.gemm(self.z, self.Wd.w, y, M, N, F, F, 1, N, 1, N, passes=ps, epi=3 if residual else 0, X2=residual)
        return y, shape[:-1] + (N,)

    def bwd(self, dy, need_dx=True):
        o, M, ps = self.net.ops, self.M, self.net.passes
        K, F, N = self.K, self.F, self.N
        o.gemm(self.z, dy, self.Wd.g, F, N, M, 1, F, N, 1, N, beta=True, passes=ps)      # dWd += z^T dy
        dz = self.gbuf("dz", M * F)
        o.gemm(dy, self.Wd.w, dz, M, F, N, N, 1, 1, N, F, passes=ps)                     # dz = dy Wd^T
        du, dg = self.gbuf("du", M * F), self.gbuf("dg", M * F)
        o.mul3(du, dz, self.a, M * F)                                                     # du = dz * act(h)
        o.mul3(dz, dz, self.u, M * F)                                                     # da = dz * u
        o.act_b(self.h, dz, dg, M * F, self.act)                                          # dg = da * act'(h)
        o.gemm(self.x, dg, self.Wg.g, K, F, M, 1, K, F, 1, F, beta=True, passes=ps)
        o.gemm(self.x, du, self.Wu.g, K, F, M, 1, K, F, 1, F, beta=True, passes=ps)
        if not need_dx:
            return None
        dx = self.gbuf("dx", M * K)
        o.gemm(dg, self.Wg.w, dx, M, K, F, F, 1, 1, F, K, passes=ps)
        o.gemm(du, self.Wu.w, dx, M, K, F, F, 1, 1, F, K, beta=True, passes=ps)
        return dx


class Net:
    """A model compiled for the device: parameters in one flat array, a tree of nodes."""

    def __init__(self, ops, model, params, precision="auto", graph=None):
        from . import lowbit
        self.ops = ops
        self.graph, self.gdev, self.gkeep = graph, None, {}
        self.arena = _Arena(ops)
        self.garena = self.arena                        # the backward pass's buffers (the same arena on one card)
        self.owned = []
        self.train = True
        self.seed = 12345
        prec = str(precision or "auto").lower()
        self.kind = lowbit.parse(prec)
        self.passes = 3 if prec in ("fp32", "float32", "f32", "tf32", "fp64", "float64", "f64") else 1
        leaves = [a for a in tree_leaves(params) if isinstance(a, np.ndarray)]
        # matrices first (weight decay and low-bit rounding apply to them), then vectors
        order = sorted(range(len(leaves)), key=lambda i: leaves[i].ndim < 2)
        self.offsets, off = {}, 0
        for i in order:
            self.offsets[id(leaves[i])] = off
            off += leaves[i].size
        self.n = off
        self.n_decay = sum(a.size for a in leaves if a.ndim >= 2)
        self.leaves, self.params_tree = leaves, params
        self.P, self.G = ops.alloc(self.n), ops.alloc(self.n)
        if hasattr(ops, "weights_are"):
            ops.weights_are(self.P, self.n)
        flat = np.zeros(self.n, np.float32)
        for a in leaves:
            o = self.offsets[id(a)]
            flat[o:o + a.size] = np.asarray(a, np.float32).reshape(-1)
        ops.put(self.P, flat)
        self.mats = [(self.offsets[id(a)], a.shape) for a in leaves if a.ndim >= 2]
        self.qoff = {}
        if self.kind:
            self.WQ = ops.alloc(self.n_decay)
            self.qoff = {o: o for o, _ in self.mats}         # the matrices come first: same offsets in WQ
        self.LOSS = ops.alloc(4)
        self.fuse = os.environ.get("APSI_FUSE", "1").strip() not in ("0", "false", "no")
        self.root = self.compile(model, params, "m")

    def param(self, a):
        if not isinstance(a, np.ndarray) or id(a) not in self.offsets:
            raise TrainError("a parameter of the model is not an array")
        return _Param(self, a.shape, self.offsets[id(a)])

    def compile(self, layer, p, key, gk=None):
        nd = self._compile(layer, p, key, gk or key)
        if getattr(nd, "gkey", None) is None:
            nd.gkey = gk or key
        return nd

    def _compile(self, layer, p, key, gk):
        try:
            return self._compile_native(layer, p, key, gk)
        except TrainError as e:
            if "not on the GPU trainer" not in str(e) or \
                    os.environ.get("APSI_GPU_HOST_LAYERS", "1").strip() in ("0", "false", "no"):
                raise
            return _HostLayer(self, key, layer, p)       # a variant the card has no kernel for: on the processor

    def _compile_native(self, layer, p, key, gk):
        from . import layers as L
        kind = type(layer).__name__
        if isinstance(layer, L.Sequential):
            nodes = []
            for i, (l_, pp) in enumerate(zip(layer.layers, p)):
                if isinstance(l_, L.Dropout) and not float(getattr(l_, "p", getattr(l_, "rate", 0.0)) or 0.0):
                    continue                            # dropout 0: nothing to do
                nd = self.compile(l_, pp, f"{key}.{i}", f"{gk}.s{i % 2}")
                if isinstance(nd, _Act) and nodes and isinstance(nodes[-1], _Linear) and nodes[-1].act is None \
                        and self.fuse:
                    nodes[-1].act = nd.kind             # linear + activation: one product with the activation
                    continue
                nodes.append(nd)
            for a_, b_ in zip(nodes, nodes[1:]):        # linear + act, then linear: keep h only
                if isinstance(a_, _Linear) and a_.act is not None and isinstance(b_, _Linear) and \
                        b_.act is None and type(b_) is _Linear:
                    a_.share_out, b_.x_pre = True, a_
            return _Seq(self, key, nodes)
        if isinstance(layer, L.Linear):
            return _Linear(self, key, layer, p)
        if isinstance(layer, L.Activation):
            name = str(layer.name).lower()
            if name not in KS.ACTS:
                raise TrainError(f"activation '{layer.name}' is not on the GPU trainer "
                                 f"({', '.join(sorted(KS.ACTS))})")
            return _Act(self, key, KS.ACTS[name])
        if isinstance(layer, L.Dropout):
            return _Dropout(self, key, getattr(layer, "p", getattr(layer, "rate", 0.0)))
        if isinstance(layer, L.LayerNorm):
            return _Norm(self, key, layer, p, False)
        if isinstance(layer, L.RMSNorm):
            return _Norm(self, key, layer, p, True)
        if isinstance(layer, L.Embedding):
            return _Embedding(self, key, layer, p)
        if isinstance(layer, L.PositionalEncoding):
            return _PosEnc(self, key, layer, p)
        if isinstance(layer, L.Residual):
            return _Residual(self, key, self.compile(layer.inner, p.inner, key + ".f", gk + ".f"),
                             self.compile(layer.norm, p.norm, key + ".n", gk + ".n") if layer.norm else None)
        if isinstance(layer, L.TransformerBlock):
            return _Seq(self, key, [self.compile(layer.attn, p.attn, key + ".a", gk + ".a"),
                                    self.compile(layer.mlp, p.mlp, key + ".m", gk + ".m")])
        if isinstance(layer, L.MultiHeadAttention) and (getattr(layer, "gate", None) is None or
                                                        not getattr(layer, "bias", True)):
            return _Attention(self, key, layer, p)
        if isinstance(layer, L.TiedOutput):
            return _TiedOutput(self, key, layer, self.dag_params)
        if isinstance(layer, L.SwiGLU):
            return _SwiGLU(self, key, layer, p)
        if isinstance(layer, L.MoE):
            return _MoE(self, key, layer, p)
        if isinstance(layer, L.DAG):
            return _DAG(self, key, layer, p, gk)
        if isinstance(layer, L.Parallel):
            if layer.combine == "list":
                raise TrainError("parallel(combine=\"list\") is not on the GPU trainer (concat, add, mul, mean)")
            return _Parallel(self, key, layer, p, gk)
        if isinstance(layer, L.Combine):
            return _Combine(self, key, layer.how)
        if isinstance(layer, (L.Identity, L.Flatten, L.Reshape)):
            return _Identity(self, key, layer)
        if isinstance(layer, L.MeanPool):
            return _MeanPool(self, key, layer)
        if isinstance(layer, L.Conv2D):
            return _Conv2D(self, key, layer, p)
        if isinstance(layer, L.LSTMLayer):
            return _Recurrent(self, key, layer, p, False)
        if isinstance(layer, L.GRULayer):
            return _Recurrent(self, key, layer, p, True)
        if isinstance(layer, L.LastStep):
            return _LastStep(self, key)
        if isinstance(layer, L.MaxPool):
            return _Pool(self, key, layer, 0)
        if isinstance(layer, L.AvgPool):
            return _Pool(self, key, layer, 1)
        if isinstance(layer, L.GlobalAvgPool):
            return _Pool(self, key, layer, 1, glob=True)
        if isinstance(layer, L.Scale):
            return _Scale(self, key, layer, p)
        from . import graphs as GR
        if isinstance(layer, GR.GraphTransformer):
            return _GraphTransformer(self, key, layer, p, gk)
        if isinstance(layer, GR.GlobalAttention):
            return _GlobalAttention(self, key, layer, p, gk)
        if isinstance(layer, GR.MixHop):
            return _MixHop(self, key, layer, p)
        if isinstance(layer, GR.GraphMemory):
            return _GraphMemory(self, key, layer, p)
        if isinstance(layer, GR.HGT):
            return _HGT(self, key, layer, p, gk)
        if isinstance(layer, GR.TypedLinear):
            return _TypedLinearNode(self, key, layer, p)
        if isinstance(layer, GR.TypedNorm):
            return _TypedNormNode(self, key, layer, p)
        from . import gla as GLA
        if isinstance(layer, GLA.ParamBank):
            return _Identity(self, key)                  # held weights: nothing to compute, no gradient
        if isinstance(layer, GLA.BrainMemory):
            return _BrainMemory(self, key, layer, p)
        if isinstance(layer, GLA.NodeEncoder):
            return _NodeEncoder(self, key, layer, p)
        if isinstance(layer, GR.BrainInput):
            return _BrainInput(self, key, layer, p)
        if isinstance(layer, GR.BrainShared):
            return _Identity(self, key)
        if isinstance(layer, GR.BrainLayer):
            return _BrainLayer(self, key, layer, p, gk, getattr(self, "dag_params", None))
        if isinstance(layer, GR.BrainOutput):
            return _BrainOutput(self, key, layer, p)
        if isinstance(layer, GR.HGTBrain):
            return _Brain(self, key, layer, p, gk)
        if isinstance(layer, GR.TypedPool):
            return _TypedPoolNode(self, key, layer, p)
        if isinstance(layer, L.GCN):
            return _GCN(self, key, layer, p)
        if isinstance(layer, L.SAGE):
            return _SAGE(self, key, layer, p)
        if isinstance(layer, L.GIN):
            return _GIN(self, key, layer, p)
        if isinstance(layer, L.GAT):
            return _GAT(self, key, layer, p)
        if isinstance(layer, L.GraphPool):
            return _GraphPool(self, key, layer)
        if os.environ.get("APSI_GPU_HOST_LAYERS", "1").strip() in ("0", "false", "no"):
            raise TrainError(f"layer '{getattr(layer, 'kind', kind)}' is not on the GPU trainer yet "
                             "(backend=\"torch\" or \"numpy\" trains it)")
        return _HostLayer(self, key, layer, p)              # on the processor, the rest on the card

    def round_weights(self):
        if self.kind:
            from .lowbit import KINDS                     # noqa: F401
            for off, shape in self.mats:
                K = int(np.prod(shape[:-1]))
                self.ops.fakeq((self.P, off), (self.WQ, off), K, int(shape[-1]), KS.FAKEQ[self.kind])

    def forward(self, x, shape):
        if isinstance(x, list):                         # several inputs: a dag with as many
            return _fwd_multi(self.root, list(zip(x, shape)))
        return self.root.fwd(x, shape)

    def backward(self, dy):
        self.root.bwd(dy, False)

    def set_graph(self, g, keep=False):
        """Another graph for the next steps (a batch of graphs): its forms on the card are made when needed. keep:
        the forms of the graphs used since clear_graphs() stay on the card (micro-batches in flight)."""
        if g is self.graph:
            return
        if keep:
            if self.gdev is not None:
                self.gkeep[id(self.graph)] = (self.graph, self.gdev)
            hit = self.gkeep.get(id(g))
            self.graph, self.gdev = g, (hit[1] if hit else None)
            return
        if self.gdev is not None:
            self.gdev.close()
        self.graph, self.gdev = g, None

    def clear_graphs(self):
        for g_, gd in list(self.gkeep.values()):
            if gd is not self.gdev:
                gd.close()
        self.gkeep = {}

    def flat_like(self, tree):
        """A tree shaped like the parameters, flattened to this network's layout (float32)."""
        flat = np.zeros(self.n, np.float32)
        mine, theirs = tree_leaves(self.params_tree), tree_leaves(tree)
        if len(mine) != len(theirs):
            raise TrainError("ewc: the anchor and fisher must have the shapes of the parameters")
        for a, t in zip(mine, theirs):
            if isinstance(a, np.ndarray) and id(a) in self.offsets:
                o = self.offsets[id(a)]
                flat[o:o + a.size] = np.asarray(t, np.float32).reshape(-1)
        return flat

    def params(self, buf=None):
        flat = self.ops.get(buf if buf is not None else self.P, self.n)

        def back(a):
            if isinstance(a, np.ndarray) and id(a) in self.offsets:
                o = self.offsets[id(a)]
                return flat[o:o + a.size].reshape(a.shape).astype(a.dtype if a.dtype.kind == "f" else np.float32)
            return a
        return tree_map(back, self.params_tree)

    def close(self):
        self.arena.close()
        self.clear_graphs()
        if self.gdev is not None:
            self.gdev.close()
        if hasattr(self.ops, "weights_are"):
            self.ops.weights_are(None, 0)
        for b in [self.P, self.G, self.LOSS] + self.owned + ([self.WQ] if self.kind else []):
            self.ops.free(b)


# ============================================================================== fit / predict
def supports(model):
    """None when the GPU trainer can take `model`, else the reason."""
    try:
        Net(_Probe(), model, model.init(seed=0))
        return None
    except TrainError as e:
        return str(e)
    except Exception as e:                                # noqa: BLE001
        return f"{type(e).__name__}: {e}"


class _Probe(Emulator):
    def alloc(self, n):
        return Buf(0, n, np.zeros(1, np.float32))

    def put(self, b, a, off=0):
        pass


def _inputs(X, ids):
    a = np.asarray(X)
    if ids:
        return np.ascontiguousarray(a.astype(np.int32))
    return np.ascontiguousarray(a.astype(np.float32))


def _batch_parts(item):
    """(inputs, targets, graph) of one batch given as (inputs, targets) or (inputs, targets, graph) or a record."""
    if hasattr(item, "inputs") or hasattr(item, "x"):
        X = getattr(item, "inputs", None)
        X = getattr(item, "x", None) if X is None else X
        return X, getattr(item, "y", getattr(item, "targets", None)), getattr(item, "graph", None)
    item = list(item)
    if len(item) == 2:
        return item[0], item[1], None
    if len(item) == 3:
        return item[0], item[1], item[2]
    raise TrainError("a batch is (inputs, targets) or (inputs, targets, graph)")


def _as_input(a):
    a = np.asarray(_raw_of(a))
    return np.ascontiguousarray(a.astype(np.int32) if a.dtype.kind in "iub" else a.astype(np.float32))


def _raw_of(a):
    from . import ops as aops
    return aops.raw(a)


@_keeps_frozen
def fit_batches(model, params, batches, loss="cross_entropy", opt=None, epochs=1, verbose=True, clip=None,
                schedule=None, device="cuda", precision="auto", ops=None, compensated=None, every=None,
                accumulate=1, stages=None, freeze=None, train_only=None, exact_sums=None, steps=None,
                checkpoint=None, save_every=None, resume=True, callback=None):
    """Train on batches that each bring their own shapes: batches is a list of (inputs, targets[, graph]) or a
    function epoch -> such a list. inputs may be a list (a dag with several inputs: token ids, node features, ...);
    each batch may have its own graph (a batch of graphs), its own number of rows and its own lengths. One card."""
    from . import ops as aops
    from .stdlib import Adam, AdamW, SGD, value_and_grad
    if ops is None:
        many = devices_for(device) if str(device).lower().startswith("cuda:") else None
        if many and len(many) > 1:
            ops = many
        else:
            ops, why = globals()["device"](device)
            if ops is None:
                raise TrainError(f"no GPU to train on: {why}")
    if isinstance(ops, (list, tuple)):
        if len(ops) > 1 and (steps or checkpoint):
            raise TrainError("steps= and checkpoint= train on one card for now: give one card (device=\"cuda:0\"), "
                             "or epochs= on several")
        if len(ops) > 1:                                 # the model split across the cards, copied over the rest
            from .gpu_pipeline import fit_batches_pipeline
            return fit_batches_pipeline(list(ops), model, params, batches, loss, opt, epochs, verbose, clip, schedule,
                                        precision, stages, max(1, int(accumulate or 1)),
                                        _want_comp(compensated, params), every, freeze, train_only,
                                        _want_exact(exact_sums, params))
        ops = ops[0]
    loss = _host_loss(loss)
    custom = callable(loss)
    loss_fn, loss = (loss, "custom") if custom else (None, str(loss))
    opt = opt or Adam(1e-3)
    adam = isinstance(opt, Adam)
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    E = int(epochs)
    net = Net(ops, model, params, precision)
    n = net.n
    st1, st2, SQ = ops.alloc(n), (ops.alloc(n) if adam else None), ops.alloc(1)
    COMP = ops.alloc(n) if _want_comp(compensated, params) else None
    LOSS1 = ops.alloc(1)
    LOSSB = None                                         # each batch's loss, read once an epoch (no wait a batch)
    for b_ in (st1, st2, COMP):
        if b_ is not None:
            ops.fill((b_, 0), n, 0.0)
    ops.fill((net.G, 0), n, 0.0)
    history, t, step = [], int(getattr(opt, "t", 0)), 0
    every = int(every or max(1, E // 10))
    net.freeze = _Freeze(ops, net, params, freeze, train_only)
    EXACT = None
    if max(1, int(accumulate or 1)) > 1 and _want_exact(exact_sums, params):
        EXACT = (ops.alloc(n), ops.alloc(n))             # the gradients of a step summed in double-float
        for b_ in EXACT:
            ops.fill((b_, 0), n, 0.0)
    net.exact = EXACT
    want = int(steps) if steps else None                 # a number of optimizer steps instead of epochs
    start_ep, skip = 0, 0
    if checkpoint and resume and os.path.exists(str(checkpoint)):
        t, step, start_ep, skip = _ckpt_load(str(checkpoint), ops, net, st1, st2, COMP)
        opt.t = t
        if verbose:
            print(f"resumed from {checkpoint}: step {step}, epoch {start_ep}, batch {skip}")
    last = start_ep + E if want is None else 1 << 30
    pos = [start_ep, skip]                               # where the next step starts (epoch, batch)

    def saved():
        _ckpt_save(str(checkpoint), ops, net, st1, st2, COMP, t, step, pos[0], pos[1])

    def after_step(ep_, bi_):                            # True: stop here (steps reached)
        pos[0], pos[1] = ep_, bi_ + 1
        if checkpoint and save_every and step % int(save_every) == 0:
            saved()
        return want is not None and step >= want
    stop = False
    try:
        for ep in range(start_ep, last):
            if stop:
                break
            items = list(batches(ep) if callable(batches) else batches)
            if not items:
                raise TrainError("no batches")
            tot, rows_all = 0.0, 0
            if LOSSB is None or LOSSB.n < len(items):
                if LOSSB is not None:
                    ops.free(LOSSB)
                LOSSB = ops.alloc(len(items))
            ops.fill((LOSSB, 0), len(items), 0.0)
            on_card = []                                 # the batches whose loss is in LOSSB
            A = max(1, int(accumulate or 1))
            for bi, item in enumerate(items):
                if ep == start_ep and bi < skip:             # resumed: these batches were already trained on
                    continue
                group = min(A, len(items) - (bi // A) * A)       # the batches of this step
                X, Y, g = _batch_parts(item)
                multi = isinstance(X, (list, tuple))
                xs = [_as_input(a) for a in (X if multi else [X])]
                net.set_graph(g)
                net.train = True
                net.round_weights()
                views = []
                for i, a in enumerate(xs):
                    v = net.arena.get(("in", i), a.size)
                    ops.put(v[0], a.reshape(-1), v[1])
                    views.append(v)
                if multi:
                    out, shape = net.forward(views, [a.shape for a in xs])
                else:
                    out, shape = net.forward(views[0], xs[0].shape)
                Yr = Y if custom and isinstance(Y, Record) else np.asarray(aops.raw(Y))
                if isinstance(out, list):                    # a dag with several outputs: a loss of your own
                    if not custom:
                        raise TrainError("a model with several outputs trains with a loss function of your own")
                    lv, dout = _multi_loss(ops, net, out, shape, loss_fn, Yr, group)
                    net.backward(dout)
                    if EXACT is not None:                         # this batch's gradients, summed exactly
                        ops.acc2((EXACT[0], 0), (EXACT[1], 0), (net.G, 0), n)
                    tot += lv
                    rows_all += 1
                    if (bi + 1) % A and bi + 1 < len(items):
                        continue
                    t, step = _apply_step(ops, net, opt, adam, wd, clip, schedule, step, t, st1, st2, SQ, COMP, n)
                    if after_step(ep, bi):
                        stop = True
                        break
                    continue
                rows, C = _rows(shape), shape[-1]
                m = rows * C
                dout = net.arena.get("dout", m)
                if custom:
                    oh = ops.get(out[0], m, out[1]).reshape(shape).astype(np.float64)
                    lv, gh = value_and_grad(lambda o_: loss_fn(o_, Yr))(oh)
                    ops.put(dout[0], (np.asarray(aops.raw(gh), np.float32) / group).reshape(-1), dout[1])
                    lv = float(aops.raw(lv))
                else:
                    ya = np.ascontiguousarray(Yr.astype(np.int32) if loss == "cross_entropy" else
                                              Yr.astype(np.float32)).reshape(-1)
                    yd = net.arena.get("Y", ya.size)
                    ops.put(yd[0], ya, yd[1])
                    if loss == "cross_entropy":
                        if ya.size != rows:
                            raise TrainError(f"cross_entropy: {ya.size} labels for {rows} rows of output")
                        w = 1.0 / max(int((ya >= 0).sum()), 1)
                        ops.ce(out, yd, dout, (LOSSB, bi), rows, C, w / group, w)
                    else:
                        if ya.size != m:
                            raise TrainError(f"{loss}: targets of {ya.size} values for an output of {m}")
                        _regress(ops, loss, out, yd, dout, (LOSSB, bi), m, 1.0 / (m * group), 1.0 / m)
                    on_card.append(bi)
                    lv = None
                net.backward(dout)
                if EXACT is not None:                         # this batch's gradients, summed exactly
                    ops.acc2((EXACT[0], 0), (EXACT[1], 0), (net.G, 0), n)
                if lv is not None:
                    tot += lv
                rows_all += 1
                if (bi + 1) % A and bi + 1 < len(items):
                    continue                                 # gradients add up until the step's last batch
                t, step = _apply_step(ops, net, opt, adam, wd, clip, schedule, step, t, st1, st2, SQ, COMP, n)
                if after_step(ep, bi):
                    stop = True
                    break
            if not stop:
                pos[0], pos[1] = ep + 1, 0
            if on_card:
                lb = ops.get(LOSSB, len(items))
                for i in on_card:
                    tot += float(lb[i])
            history.append(tot / max(rows_all, 1))
            if callback is not None:
                callback(Record(epoch=ep, step=step, loss=history[-1]))
            if verbose and (ep % every == 0 or ep == last - 1 or stop):
                print(f"epoch {ep:4}  loss {history[-1]:.5f}  ({len(items)} batches, step {step}, GPU)")
        opt.t = t
        out_params = net.params()
        if checkpoint:
            saved()
    except BaseException:                                # a crash or Ctrl+C: what was learned is kept
        if checkpoint:
            try:
                ops.fill((net.G, 0), n, 0.0)
                saved()
                if verbose:
                    print(f"training stopped at step {step}; saved to {checkpoint} (run again to resume)")
            except Exception:                            # noqa: BLE001  (the card itself may be gone)
                pass
        raise
    finally:
        for b_ in (st1, st2, SQ, COMP, LOSS1, LOSSB):
            if b_ is not None:
                ops.free(b_)
        net.freeze.close()
        for b_ in (EXACT or ()):
            ops.free(b_)
        net.close()
    return Record(params=out_params, history=history, device=getattr(ops, "name", "?"))


def _ckpt_save(path, ops, net, st1, st2, COMP, t, step, epoch, batch=0):
    """The flat weights, the optimizer's moments and the carried rounding error, written atomically."""
    n = net.n
    data = {"P": ops.get(net.P, n), "meta": np.array([n, t, step, epoch, batch], np.int64)}
    for name, b_ in (("M", st1), ("V", st2), ("C", COMP)):
        if b_ is not None:
            data[name] = ops.get(b_, n)
    master = getattr(net, "master", None)
    if master is not None:
        arrs = [np.asarray(a, np.float64).reshape(-1) for a in tree_leaves(master) if isinstance(a, np.ndarray)]
        data["P64"] = np.concatenate(arrs) if arrs else np.zeros(0)   # the float64 master copy, in tree order
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, path)


def _ckpt_load(path, ops, net, st1, st2, COMP):
    with np.load(path) as z:
        meta = [int(v) for v in z["meta"]] + [0]
        n, t, step, epoch, batch = meta[:5]
        if n != net.n:
            raise TrainError(f"the checkpoint {path} is of a model with {n:,} weights, this one has {net.n:,}: "
                             "a different model (delete the file or give another checkpoint=)")
        ops.put(net.P, z["P"].astype(np.float32))
        for name, b_ in (("M", st1), ("V", st2), ("C", COMP)):
            if b_ is not None and name in z.files:
                ops.put(b_, z[name].astype(np.float32))
        if "P64" in z.files:
            flat = z["P64"]
            leaves, off = [], 0
            for a in tree_leaves(net.params_tree):
                if isinstance(a, np.ndarray):
                    leaves.append(flat[off:off + a.size].reshape(a.shape))
                    off += a.size
                else:
                    leaves.append(a)
            it = iter(leaves)
            net.master = tree_map(lambda a: next(it), net.params_tree)
    return t, step, epoch, batch


def _multi_loss(ops, net, outs, shapes, loss_fn, Yr, group, arena=None):
    """The loss of several outputs: (value, [gradient view of each output]). A loss with `on_card = (i, labels_of,
    rest)` keeps output i's cross-entropy on the card (labels_of(targets): class ids, -1 ignored; or None for the
    processor) and sends only the other outputs to the processor, where rest(outputs, targets) adds the rest of
    the loss; any other loss runs on the processor over every output."""
    from . import ops as aops
    from .stdlib import value_and_grad
    arena = arena if arena is not None else net.arena
    douts, fn, card, lossv = [None] * len(outs), loss_fn, set(), None
    spec = getattr(loss_fn, "on_card", None)
    if spec is not None:
        i_ce, labels_of, rest = spec
        lab = labels_of(Yr)
        if lab is not None:
            lab = np.ascontiguousarray(np.asarray(lab).reshape(-1).astype(np.int32))
            rows, V = _rows(shapes[i_ce]), int(shapes[i_ce][-1])
            if lab.size != rows:
                raise TrainError(f"cross_entropy on the card: {lab.size} labels for {rows} rows of output {i_ce}")
            yd = arena.get("ce_labels", lab.size)
            ops.put(yd[0], lab, yd[1])
            d = arena.get(("dout", i_ce), rows * V)
            lossv = arena.get("ce_loss", 1)
            ops.fill(lossv, 1, 0.0)
            w = 1.0 / max(int((lab >= 0).sum()), 1)
            ops.ce(outs[i_ce], yd, d, lossv, rows, V, w / group, w)
            douts[i_ce], fn, card = d, rest, {i_ce}
    keep = [i for i in range(len(outs)) if i not in card]
    hs = [ops.get(outs[i][0], int(np.prod(shapes[i])), outs[i][1]).reshape(shapes[i]).astype(np.float64)
          for i in keep]

    def f(*os_):
        full = [None] * len(outs)
        for j, i in enumerate(keep):
            full[i] = os_[j]
        return fn(full, Yr)
    lv, gh = value_and_grad(f, wrt=tuple(range(len(hs))))(*hs)
    gh = gh if isinstance(gh, (list, tuple)) else [gh]
    for i, g_ in zip(keep, gh):
        a = np.asarray(aops.raw(g_), np.float32).reshape(-1) / group
        d = arena.get(("dout", i), max(1, a.size))
        ops.put(d[0], a, d[1])
        douts[i] = d
    total = float(aops.raw(lv))
    if lossv is not None:
        total += float(ops.get(lossv[0], 1, lossv[1])[0])
    return total, douts


def _rows_of(X):
    from . import ops as aops
    return len(np.asarray(aops.raw(X[0] if isinstance(X, (list, tuple)) else X)))


def _host_loss(loss):
    """A loss the card has no kernel for (any name fit() knows, e.g. "vae", or a function): on the processor."""
    if callable(loss) or str(loss) in LOSSES:
        return loss
    from . import layers as L
    try:
        return L._loss_fn(str(loss))
    except Exception:                                   # noqa: BLE001
        raise TrainError(f"loss '{loss}' is unknown ({', '.join(LOSSES)} on the card, or a function)") from None


def native_opt(opt):
    from .stdlib import Adam, SGD
    return type(opt) in (Adam, SGD) or type(opt).__name__ == "AdamW" and isinstance(opt, Adam)


class _Freeze:
    """Weights that stay as they are (freeze= / train_only=): no gradient, and their values put back after every
    update (weight decay or momentum cannot move them)."""

    def __init__(self, ops, net, params, freeze=None, train_only=None):
        self.ops, self.net, self.ranges, self.buf = ops, net, [], None
        if freeze is None and train_only is None:
            return
        from .layers import frozen_ids
        ids = frozen_ids(params, freeze, train_only)
        spans = sorted((net.offsets[i], a.size) for i, a in ((id(a), a) for a in net.leaves) if i in ids)
        for off, k in spans:
            if self.ranges and self.ranges[-1][0] + self.ranges[-1][1] == off:
                self.ranges[-1] = (self.ranges[-1][0], self.ranges[-1][1] + k)
            else:
                self.ranges.append((off, k))
        total = sum(k for _, k in self.ranges)
        self.buf = ops.alloc(max(1, total)) if total else None

    def zero_grads(self):
        for off, k in self.ranges:
            self.ops.fill((self.net.G, off), k, 0.0)

    def save(self):
        o = 0
        for off, k in self.ranges:
            self.ops.pull((self.buf, o), self.ops, (self.net.P, off), k)
            o += k

    def restore(self):
        o = 0
        for off, k in self.ranges:
            self.ops.pull((self.net.P, off), self.ops, (self.buf, o), k)
            o += k

    def close(self):
        if self.buf is not None:
            self.ops.free(self.buf)
            self.buf = None


def _host_step(ops, net, opt, clip, lr):
    """Any optimizer with step(params, grads): the weights and gradients in the model's own tree on the processor,
    the new weights back to the card (the optimizer sees what it sees on NumPy)."""
    import math as _m
    master = getattr(net, "master", None)
    if master is None:                                   # float64 weights kept for the optimizer, as on NumPy
        master = tree_map(lambda a: np.asarray(a, np.float64) if isinstance(a, np.ndarray) else a, net.params_tree)
    P, G = master, net.params(net.G)
    if _guard() and not all(np.all(np.isfinite(a)) for a in tree_leaves(G) if isinstance(a, np.ndarray)):
        ops.fill((net.G, 0), net.n, 0.0)                # a non-finite gradient: the step is skipped
        return
    if clip:
        nrm = _m.sqrt(sum(float(np.sum(np.asarray(a, np.float64) ** 2)) for a in tree_leaves(G)
                          if isinstance(a, np.ndarray)))
        if nrm > clip:
            G = tree_map(lambda a: a * (clip / nrm) if isinstance(a, np.ndarray) else a, G)
    P = tree_map(lambda a: a.astype(np.float64) if isinstance(a, np.ndarray) else a, P)
    G = tree_map(lambda a: a.astype(np.float64) if isinstance(a, np.ndarray) else a, G)
    if hasattr(opt, "lr"):
        opt.lr = lr
    from . import ops as aops
    new = tree_map(lambda a: np.asarray(aops.raw(a), np.float64) if hasattr(a, "shape") or
                   isinstance(a, np.ndarray) else a, opt.step(P, G))
    fz = getattr(net, "freeze", None)
    if fz is not None and fz.ranges:                     # frozen weights keep their values in the master copy too
        spans = fz.ranges
        old_l, new_l = tree_leaves(P), tree_leaves(new)
        keep = []
        for a, o_, n_ in zip(tree_leaves(net.params_tree), old_l, new_l):
            off = net.offsets.get(id(a)) if isinstance(a, np.ndarray) else None
            keep.append(o_ if off is not None and any(lo <= off < lo + k for lo, k in spans) else n_)
        it = iter(keep)
        new = tree_map(lambda a: next(it), new)
    net.master = new
    ops.put(net.P, net.flat_like(new))
    ops.fill((net.G, 0), net.n, 0.0)                    # as the kernels do: the next step's gradients start at 0


def _apply_step(ops, net, opt, adam, wd, clip, schedule, step, t, st1, st2, SQ, COMP, n):
    from . import ops as aops
    t += 1
    lr = float(aops.raw(schedule(step))) if schedule is not None else float(opt.lr)
    ex = getattr(net, "exact", None)
    if ex is not None:                                   # the step's exact sum, rounded once
        ops.fin2((ex[0], 0), (ex[1], 0), (net.G, 0), n)
    fz = getattr(net, "freeze", None)
    if fz is not None and fz.ranges:
        fz.zero_grads()
        fz.save()
    if not native_opt(opt):
        _host_step(ops, net, opt, clip, lr)
        if COMP is not None:                             # the carried rounding error belongs to the kernels' updates
            ops.fill((COMP, 0), n, 0.0)
        if fz is not None and fz.ranges:
            fz.restore()
        return t, step + 1
    _native_step(ops, net, opt, adam, wd, clip, lr, t, st1, st2, SQ, COMP, n)
    if fz is not None and fz.ranges:
        fz.restore()
    return t, step + 1


def _native_step(ops, net, opt, adam, wd, clip, lr, t, st1, st2, SQ, COMP, n):
    ops.fill((SQ, 0), 1, 0.0)
    if clip or _guard():
        ops.sumsq((net.G, 0), n, (SQ, 0))
    if adam:
        ops.adam((net.P, 0), (net.G, 0), (st1, 0), (st2, 0), (SQ, 0), n, net.n_decay, lr, opt.b1, opt.b2,
                 opt.eps, 1 - opt.b1 ** t, 1 - opt.b2 ** t, wd, float(clip or 0.0), (COMP, 0) if COMP else None)
    else:
        ops.sgd((net.P, 0), (net.G, 0), (st1, 0), (SQ, 0), n, lr, float(opt.momentum), float(clip or 0.0),
                (COMP, 0) if COMP else None)


def _want_exact(exact, params):
    """Exact sums of the gradients of a step's batches (and of the cards that share a step): on by default below
    2 B parameters (8 more bytes per parameter). APSI_GPU_EXACT_SUMS=0 / 1 overrides."""
    if exact is not None:
        return bool(exact)
    env = os.environ.get("APSI_GPU_EXACT_SUMS", "").strip().lower()
    if env in ("0", "off", "false", "no"):
        return False
    if env in ("1", "on", "true", "yes"):
        return True
    return sum(int(np.size(a)) for a in tree_leaves(params) if isinstance(a, np.ndarray)) < 2_000_000_000


def _want_comp(compensated, params):
    """Compensated weight updates: on by default below 2 B parameters (4 more bytes per parameter)."""
    if compensated is not None:
        return bool(compensated)
    env = os.environ.get("APSI_GPU_COMPENSATED", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return sum(int(np.asarray(a).size) for a in tree_leaves(params) if isinstance(a, np.ndarray)) < 2_000_000_000


LOSSES = ("cross_entropy", "mse", "mae", "bce")


def _regress(o, loss, out, Yd, dout, lossv, m, wg, wl, W=None):
    """The element-wise losses: mse, mae, bce (W: a weight per element, e.g. a mask)."""
    if loss == "mse":
        o.mse(out, Yd, dout, lossv, m, wg, wl)
    else:
        o.loss_ew(out, Yd, W, dout, lossv, m, wg, wl, {"mae": 0, "bce": 1}[loss])


def _too_big(ops_list, params, opt):
    """True when the weights, gradients and optimizer state alone would take most of the smallest card."""
    from .stdlib import Adam
    mems = [getattr(getattr(o, "dev", None), "memory", None) for o in ops_list]
    if not all(mems):
        return False
    count = sum(int(np.asarray(a).size) for a in tree_leaves(params) if isinstance(a, np.ndarray))
    return count * (16 if isinstance(opt, Adam) or opt is None else 12) > 0.6 * min(mems)


def _check_size(ops, params, opt, shards=1):
    """An estimate, not a limit: weights, gradients and the optimizer's state take about 8 + 8 / N bytes per parameter
    on each card; what the activations take depends on the architecture (an LLM's on the context and the batch, other
    models on their own layers), so nothing is refused here. When the estimate alone exceeds the card, a warning
    says so with the numbers; if the card then really runs out, the error says what to change."""
    import warnings
    from .stdlib import Adam
    total = getattr(getattr(ops, "dev", None), "memory", None)
    if not total:
        return
    count = sum(int(np.asarray(a).size) for a in tree_leaves(params) if isinstance(a, np.ndarray))
    per = 8 + (8 if isinstance(opt, Adam) or opt is None else 4) / max(1, shards)
    need = count * per
    if need > total:
        warnings.warn(f"{count / 1e9:.2f} B parameters take about {need / 2 ** 30:.1f} GB per card for training "
                      f"(weights, gradients, the optimizer's share) before activations; this card has "
                      f"{total / 2 ** 30:.1f} GB: it may not fit (more cards: device=\"cuda:all\" splits the model across them; a smaller batch)",
                      RuntimeWarning, stacklevel=3)


@_keeps_frozen
def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, val=None, device="cuda", precision="auto", ops=None, ema=None, ewc=None,
        parallel="auto", stages=None, micro_batches=None, graph=None, mask=None, compensated=None, freeze=None,
        train_only=None):
    """Several cards (device="cuda:all", "cuda:0,1,2"): parallel="data" copies the model to every card and shares
    the batch out; "pipeline" splits the model across the cards (stages: how many pieces, or a list with the units of
    each); "auto" (the default) copies it when it fits on one card, else splits it into the fewest pieces that fit and
    copies those over the cards left. micro_batches: the pieces of a batch that flow through a split model.
    compensated: the weights keep the rounding error of every update (compensated summation, 4 more bytes per
    parameter): updates far smaller than float32's step at a weight's size still add up (default: on below 2 B
    parameters).

    ema=0.999: also keeps an exponential moving average of the weights (record.ema_params), the smoother set
    that often generalizes better. ewc=(anchor_params, fisher, strength): elastic weight consolidation, a pull of
    every weight toward anchor_params in proportion to its importance (fisher, the same shapes; e.g. the mean
    squared gradients on the old task), so a model learns a new task without forgetting the old one."""
    from . import layers as L, ops as aops
    from .stdlib import Adam, AdamW, SGD
    if ops is None:
        many = devices_for(device)
        if many is not None:
            if not many:
                raise TrainError(f"no GPU to train on for device='{device}'")
            ops = many if len(many) > 1 else many[0]
    mode = str(parallel or "auto").lower()
    if mode not in ("auto", "data", "pipeline"):
        raise TrainError(f"parallel='{parallel}': auto, data or pipeline")
    if graph is not None:                              # every step takes the whole graph: one card
        if isinstance(ops, (list, tuple)):
            if mode == "pipeline" or stages is not None:
                raise TrainError("a graph model trains on one card (its nodes depend on each other in every step)")
            ops = ops[0]
        batch_size = None
    loss = _host_loss(loss)
    opt = opt or Adam(1e-3)
    if isinstance(ops, (list, tuple)) and len(ops) > 1 and graph is None and mask is None and val is None and \
            ema is None and ewc is None and (callable(loss) or not native_opt(opt) or freeze is not None or
                                             train_only is not None):
        # what the many-card kernels of fit() do not do (a loss function, another optimizer, frozen parts): the
        # same training as batches, on the cards of fit_batches
        bs = int(batch_size or _rows_of(X))
        Xa, Ya = np.asarray(aops.raw(X)), np.asarray(aops.raw(X if y is None else y))
        rng = np.random.default_rng(int(seed))

        def epoch_batches(ep):
            idx = rng.permutation(len(Xa)) if batch_size else np.arange(len(Xa))
            return [(Xa[idx[i:i + bs]], Ya[idx[i:i + bs]]) for i in range(0, len(Xa), bs)]
        return fit_batches(model, params, epoch_batches, loss, opt, epochs, verbose, clip, schedule, ops=ops,
                           precision=precision, compensated=compensated, stages=stages, freeze=freeze,
                           train_only=train_only, accumulate=micro_batches or 1)
    if isinstance(ops, (list, tuple)) and callable(loss):
        if len(ops) > 1:
            raise TrainError("a loss written as a function trains on one card (device=\"cuda\")")
        ops = ops[0]
    if isinstance(ops, (list, tuple)):
        if len(ops) > 1 and (mode == "pipeline" or stages is not None or (mode == "auto" and _too_big(ops, params, opt))):
            from .gpu_pipeline import fit_pipeline
            return fit_pipeline(list(ops), model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip,
                                schedule, val, precision, ema, ewc,
                                "all" if stages is None and mode == "pipeline" else stages, micro_batches,
                                _want_comp(compensated, params))
        if len(ops) > 1:
            return _fit_parallel(list(ops), model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip,
                                 schedule, val, precision, ema, ewc, _want_comp(compensated, params))
        ops = ops[0]
    if ops is None:
        ops, why = globals()["device"](device)
        if ops is None:
            raise TrainError(f"no GPU to train on: {why}")
        if str(device or "cuda").lower() in ("cuda", "auto", "gpu") and mode != "data" and _too_big([ops], params, opt):
            many = devices_for("cuda:all") or []          # too large for the first card: every card, the model split
            if len(many) > 1:
                from .gpu_pipeline import fit_pipeline
                return fit_pipeline(many, model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip,
                                    schedule, val, precision, ema, ewc, stages, micro_batches,
                                    _want_comp(compensated, params))
    custom = callable(loss)                              # a loss function: evaluated on the processor
    if custom:
        if isinstance(ops, (list, tuple)) and len(ops) > 1:
            raise TrainError("a loss written as a function trains on one card (device=\"cuda\")")
        loss_fn, loss = loss, "custom"
    loss = str(loss) if not custom else loss
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    ids = isinstance(first, L.Embedding)
    Xa = _inputs(aops.raw(X), ids)
    Ya = np.asarray(aops.raw(X if y is None else y))
    ce = loss == "cross_entropy"
    Yorig = np.asarray(aops.raw(X if y is None else y)) if custom else None
    if custom and Yorig.dtype.kind not in "fiub":
        raise TrainError("a loss function on the GPU trainer takes numeric targets")
    Ya = np.ascontiguousarray(Ya.astype(np.int32) if ce else Ya.astype(np.float32))
    sel = None
    if mask is not None:                               # only these rows count in the loss
        sel = np.asarray(aops.raw(mask)).astype(bool).reshape(-1)
        if ce:
            Ya = Ya.copy()
            Ya[~sel] = -1
    n = Xa.shape[0]
    _check_size(ops, params, opt)
    net = Net(ops, model, params, precision, graph)
    fz = net.freeze = _Freeze(ops, net, params, freeze, train_only)
    if fz.buf is not None:
        net.owned.append(fz.buf)                         # freed with the net
    rng = np.random.default_rng(int(seed))
    net.seed = int(seed) * 7919 + 1
    adam = isinstance(opt, Adam)
    guard = _guard()
    st1, st2 = ops.alloc(net.n), (ops.alloc(net.n) if adam else None)
    ops.fill((st1, 0), net.n, 0.0)
    if st2 is not None:
        ops.fill((st2, 0), net.n, 0.0)
    SQ = ops.alloc(1)
    COMP = None
    if _want_comp(compensated, params):
        COMP = ops.alloc(net.n)
        ops.fill((COMP, 0), net.n, 0.0)
        net.owned.append(COMP)
    EMA = EWC_A = EWC_F = EWC_T = None
    if ema is not None:
        ema = float(ema)
        if not 0.0 < ema < 1.0:
            raise TrainError(f"ema={ema}: give the decay, between 0 and 1 (0.999 is usual)")
        EMA = ops.alloc(net.n)
        ops.fill((EMA, 0), net.n, 0.0)
        ops.axpy((EMA, 0), (net.P, 0), net.n, 1.0)
        net.owned.append(EMA)
    if ewc is not None:
        try:
            anchor, fisher, lam = ewc
        except (TypeError, ValueError):
            raise TrainError("ewc=(anchor_params, fisher, strength)") from None
        EWC_A, EWC_F, EWC_T = ops.alloc(net.n), ops.alloc(net.n), ops.alloc(net.n)
        ops.put(EWC_A, net.flat_like(anchor))
        ops.put(EWC_F, net.flat_like(fisher))
        net.owned += [EWC_A, EWC_F, EWC_T]
        lam = float(lam)
    t_step = int(getattr(opt, "t", 0))
    bs = int(batch_size) if batch_size else n
    xw, yw = int(np.prod(Xa.shape[1:])), int(np.prod(Ya.shape[1:]))
    Xd = net.arena.get("X", min(bs, n) * xw)
    Yd = net.arena.get("Y", min(bs, n) * yw if graph is None else Ya.size)
    # the whole data set on the card when it is small next to the card (one copy, then each batch is gathered
    # there from a permutation uploaded once an epoch: no copy from the processor inside an epoch)
    E = int(epochs)
    orders = [rng.permutation(n) if batch_size else np.arange(n) for _ in range(E)]
    resident = (Xa.nbytes + Ya.nbytes) < min(1 << 30, getattr(ops, "budget", 1 << 30))
    if resident:                 # the data and every epoch's order go to the card once: nothing waits between epochs
        XA, YA, IDX = ops.alloc(Xa.size), ops.alloc(Ya.size), ops.alloc(max(1, n * E))
        ops.put(XA, Xa.reshape(-1))
        ops.put(YA, Ya.reshape(-1))
        if E:
            ops.put(IDX, np.concatenate(orders).astype(np.int32))
        net.owned += [XA, YA, IDX]
    LOSSE = ops.alloc(max(1, E))                             # one loss per epoch, read once at the end
    ops.fill((LOSSE, 0), max(1, E), 0.0)
    net.owned.append(LOSSE)
    marks = []                                               # the end of each epoch on the card's own clock
    can_mark = hasattr(ops, "_event") and not isinstance(ops, Emulator)
    if can_mark:
        e0 = ops._event()
        ops._lib.cuEventRecord(e0, None)
        marks.append(e0)
    history, val_hist, step = [], [], 0
    host_loss = [0.0] * max(1, E)
    INV = MT = None
    ops.fill((net.G, 0), net.n, 0.0)
    from . import thermal
    gov = thermal.governor(getattr(ops, "dev", None))        # keeps the card near APSI_GPU_TEMP (77 C)
    paced0 = (gov.status() or {}).get("paced_s") or 0.0
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    ep_seconds, ep_temp, ep_t0 = [], [], time.perf_counter()
    try:
        for ep in range(E):
            order = orders[ep]
            lossv = (LOSSE, ep)
            for s in range(0, n, bs):
                b = order[s:s + bs]
                xb, yb = Xa[b], (Ya[b] if graph is None else Ya)
                if graph is not None:                       # the whole graph (targets per node or per graph)
                    if ep == 0 and s == 0 or sel is not None:
                        ops.put(Xd[0], xb.reshape(-1))
                        ops.put(Yd[0], yb.reshape(-1))
                elif resident:
                    ops.gather((XA, 0), (IDX, 0), Xd, len(b), xw, ep * n + s)
                    ops.gather((YA, 0), (IDX, 0), Yd, len(b), yw, ep * n + s)
                else:
                    ops.put(Xd[0], xb.reshape(-1))
                    ops.put(Yd[0], yb.reshape(-1))
                lr = float(aops.raw(schedule(step))) if schedule is not None else float(opt.lr)
                net.round_weights()                         # (the optimizer kernel left the gradient cleared)
                net.train = True
                out, shape = net.forward(Xd, xb.shape)
                rows = _rows(shape)
                dout = net.arena.get("dout", rows * shape[-1])
                if custom:                                  # the loss and its gradient on the processor
                    from .stdlib import value_and_grad
                    oh = ops.get(out[0], rows * shape[-1], out[1]).reshape(shape).astype(np.float64)
                    yh = yb if Yorig is None else (Yorig[b] if graph is None else Yorig)
                    pick = np.nonzero(sel)[0] if sel is not None else None

                    def lf(o_):
                        return loss_fn(o_[pick], yh[pick]) if pick is not None else loss_fn(o_, yh)
                    lv, gh = value_and_grad(lf)(oh)
                    ops.put(dout[0], np.asarray(aops.raw(gh), np.float32).reshape(-1), dout[1])
                    host_loss[ep] += float(aops.raw(lv)) * (1.0 if graph is not None else len(b) / n)
                elif ce:
                    valid = int((yb >= 0).sum())
                    w = 1.0 / max(valid, 1)
                    if yb.size != rows:
                        raise TrainError(f"cross_entropy: {yb.size} labels for {rows} rows of logits")
                    ops.ce(out, Yd, dout, lossv, rows, shape[-1], w, w * len(b) / n)
                elif sel is not None and loss != "mse":     # the rows outside the mask weigh nothing
                    m = rows * shape[-1]
                    if INV is None:
                        INV = ops.alloc(m)
                        ops.put(INV, np.repeat(sel.astype(np.float32), m // max(1, len(sel))))
                        net.owned.append(INV)
                    me = max(1, int(sel.sum())) * (m // max(1, len(sel)))
                    _regress(ops, loss, out, Yd, dout, lossv, m, 1.0 / me, 1.0 / me, (INV, 0))
                elif sel is not None:                       # the rows outside the mask: target = output
                    m = rows * shape[-1]
                    if INV is None:
                        INV = ops.alloc(m)
                        ops.put(INV, np.repeat((~sel).astype(np.float32), m // max(1, len(sel))))
                        net.owned.append(INV)
                        MT = net.arena.get("mt", m)
                    ops.pull(MT, ops, out, m)
                    ops.axpy(MT, Yd, m, -1.0)
                    ops.mul3(MT, MT, (INV, 0), m)
                    ops.axpy(Yd, MT, m, 1.0)
                    me = max(1, int(sel.sum())) * (m // max(1, len(sel)))
                    ops.mse(out, Yd, dout, lossv, m, 1.0 / me, 1.0 / me)
                else:
                    m = rows * shape[-1]
                    _regress(ops, loss, out, Yd, dout, lossv, m, 1.0 / m, len(b) / (n * m))
                net.backward(dout)
                if EWC_A is not None:                       # g += strength * F * (w - anchor)
                    ops.fill((EWC_T, 0), net.n, 0.0)
                    ops.axpy((EWC_T, 0), (net.P, 0), net.n, 1.0)
                    ops.axpy((EWC_T, 0), (EWC_A, 0), net.n, -1.0)
                    ops.mul3((EWC_T, 0), (EWC_T, 0), (EWC_F, 0), net.n)
                    ops.axpy((net.G, 0), (EWC_T, 0), net.n, lam)
                if clip or guard:
                    ops.fill((SQ, 0), 1, 0.0)
                    ops.sumsq((net.G, 0), net.n, (SQ, 0))
                t_step += 1
                if fz.ranges:
                    fz.zero_grads()
                    fz.save()
                    if clip or guard:
                        ops.fill((SQ, 0), 1, 0.0)
                        ops.sumsq((net.G, 0), net.n, (SQ, 0))
                if not native_opt(opt):
                    _host_step(ops, net, opt, clip, lr)
                    if COMP is not None:
                        ops.fill((COMP, 0), net.n, 0.0)
                elif adam:
                    c1, c2 = 1 - opt.b1 ** t_step, 1 - opt.b2 ** t_step
                    ops.adam((net.P, 0), (net.G, 0), (st1, 0), (st2, 0), (SQ, 0), net.n, net.n_decay, lr, opt.b1,
                             opt.b2, opt.eps, c1, c2, wd, float(clip or 0.0), (COMP, 0) if COMP else None)
                else:
                    ops.sgd((net.P, 0), (net.G, 0), (st1, 0), (SQ, 0), net.n, lr, float(opt.momentum),
                            float(clip or 0.0), (COMP, 0) if COMP else None)
                if fz.ranges:
                    fz.restore()
                step += 1
                if EMA is not None:                         # ema <- d ema + (1 - d) w
                    ops.axpy((EMA, 0), (EMA, 0), net.n, ema - 1.0)
                    ops.axpy((EMA, 0), (net.P, 0), net.n, 1.0 - ema)
                gov.tick(ops.sync)
            if can_mark:
                ev = ops._event()
                ops._lib.cuEventRecord(ev, None)
                marks.append(ev)
            else:
                now = time.perf_counter()
                ep_seconds.append(round(now - ep_t0, 3))
                ep_t0 = now
            st = gov.status()
            ep_temp.append(st.get("temperature_c") if st else None)
            show = verbose and (ep % max(1, E // 10) == 0 or ep == E - 1)
            extra = ""
            if val is not None:                             # (these wait for the card; without them nothing does)
                if custom:
                    vx_, vy_ = val
                    vo = _forward_batches(net, ops, _inputs(aops.raw(vx_), ids), bs if graph is None else 1 << 62)
                    vl = float(aops.raw(loss_fn(vo.astype(np.float64), np.asarray(aops.raw(vy_)))))
                else:
                    vl = _eval_loss(net, ops, val, ids, loss, bs if graph is None else 1 << 62)
                val_hist.append(vl)
                extra = f"  val {vl:.5f}"
            if show:
                print(f"epoch {ep:4}  loss {float(ops.get(LOSSE, 1, ep)[0]):.5f}{extra}")
        history = [float(v) for v in ops.get(LOSSE, max(1, E))[:E]] if not custom else list(host_loss)
        if can_mark:
            ops.sync()
            ms = ctypes.c_float()
            for a_, b_ in zip(marks, marks[1:]):
                ops._lib.cuEventElapsedTime(ctypes.byref(ms), a_, b_)
                ep_seconds.append(round(ms.value / 1000.0, 3))
            for e_ in marks:
                ops._lib.cuEventDestroy_v2(e_)
        opt.t = t_step
        peak = ops.used
        out_params = net.params()
        ema_params = net.params(EMA) if EMA is not None else None
    finally:
        for bf in (st1, st2, SQ):
            if bf is not None:
                ops.free(bf)
        net.close()
    temp = gov.status()
    if temp:
        temp = dict(temp, paced_s=round(temp["paced_s"] - paced0, 1))     # this run's pauses only
    if verbose and temp and temp["paced_s"]:
        print(f"GPU kept near {temp['target_c']:.0f} C (highest {temp['max_c']:.0f} C, paced {temp['paced_s']} s)")
    return Record(params=out_params, history=history, val_history=val_hist, gpu_memory_mb=round(peak / 2 ** 20),
                  gpu_temperature=temp, epoch_seconds=ep_seconds, epoch_temperature_c=ep_temp, ema_params=ema_params,
                  device=f"gpu ({getattr(ops, 'name', '?')}, Apsi kernels"
                         f"{', tensor cores' if ops.tc else ''})")


def _fit_parallel(ops_list, model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip, schedule, val,
                  precision, ema, ewc, comp=False):
    """Data-parallel training on several cards: every card keeps the whole model and its optimizer state, takes its
    share of each batch, and the gradients are summed across the cards (card to card where the driver allows it)
    before the identical update on each, so the result is that of one card with the whole batch. Throughput grows
    with the cards; the model must fit on one of them."""
    from . import layers as L, ops as aops, thermal
    from .stdlib import Adam, AdamW, SGD
    loss = str(loss)
    if loss not in LOSSES:
        raise TrainError(f"loss '{loss}' is not on the GPU trainer ({', '.join(LOSSES)})")
    opt = opt or Adam(1e-3)
    if not isinstance(opt, (Adam, SGD)):
        raise TrainError(f"optimizer {type(opt).__name__} is not on the GPU trainer (adam, adamw, sgd)")
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    ids = isinstance(first, L.Embedding)
    Xa = _inputs(aops.raw(X), ids)
    Ya = np.asarray(aops.raw(X if y is None else y))
    ce = loss == "cross_entropy"
    Ya = np.ascontiguousarray(Ya.astype(np.int32) if ce else Ya.astype(np.float32))
    n, E = Xa.shape[0], int(epochs)
    bs = int(batch_size) if batch_size else n
    rng = np.random.default_rng(int(seed))
    orders = [rng.permutation(n) if batch_size else np.arange(n) for _ in range(E)]
    xw, yw = int(np.prod(Xa.shape[1:])), int(np.prod(Ya.shape[1:]))
    adam = isinstance(opt, Adam)
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    R = []
    try:
        for r, o in enumerate(ops_list):
            _check_size(o, params, opt, len(ops_list))
            net = Net(o, model, params, precision)
            net.seed = int(seed) * 7919 + 1 + 104729 * r
            # the optimizer's state is shared out: card r keeps (and updates) only its slice of the parameters
            lo, hi = net.n * r // len(ops_list), net.n * (r + 1) // len(ops_list)
            ln = max(1, hi - lo)
            rep = {"ops": o, "net": net, "lo": lo, "hi": hi, "st1": o.alloc(ln), "st2": o.alloc(ln) if adam else None,
                   "COMP": o.alloc(ln) if comp else None,
                   "SQ": o.alloc(1), "LOSSE": o.alloc(max(1, E)), "gov": thermal.governor(getattr(o, "dev", None))}
            o.fill((rep["st1"], 0), ln, 0.0)
            if rep["COMP"] is not None:
                o.fill((rep["COMP"], 0), ln, 0.0)
            if rep["st2"] is not None:
                o.fill((rep["st2"], 0), ln, 0.0)
            o.fill((rep["LOSSE"], 0), max(1, E), 0.0)
            rep["XA"], rep["YA"], rep["IDX"] = o.alloc(Xa.size), o.alloc(Ya.size), o.alloc(max(1, n * E))
            o.put(rep["XA"], Xa.reshape(-1))
            o.put(rep["YA"], Ya.reshape(-1))
            if E:
                o.put(rep["IDX"], np.concatenate(orders).astype(np.int32))
            rep["Xd"] = net.arena.get("X", min(bs, n) * xw)
            rep["Yd"] = net.arena.get("Y", min(bs, n) * yw)
            R.append(rep)
        r0 = R[0]
        nP = r0["net"].n
        for rep in R:
            o = rep["ops"]
            rep["TMP"] = o.alloc(max(1, rep["hi"] - rep["lo"]))
            rep["SQX"], rep["SQ2"] = o.alloc(1), o.alloc(1)
            o.fill((rep["SQ2"], 0), 1, 0.0)
        EMA = EWC = None
        if ema is not None:
            ema = float(ema)
            if not 0.0 < ema < 1.0:
                raise TrainError(f"ema={ema}: give the decay, between 0 and 1 (0.999 is usual)")
            EMA = r0["ops"].alloc(nP)
            r0["ops"].fill((EMA, 0), nP, 0.0)
            r0["ops"].axpy((EMA, 0), (r0["net"].P, 0), nP, 1.0)
        if ewc is not None:
            anchor, fisher, lam = ewc
            for rep in R:
                o = rep["ops"]
                rep["EA"], rep["EF"], rep["ET"] = o.alloc(nP), o.alloc(nP), o.alloc(max(1, rep["hi"] - rep["lo"]))
                o.put(rep["EA"], rep["net"].flat_like(anchor))
                o.put(rep["EF"], rep["net"].flat_like(fisher))
            EWC = float(lam)
        t_step, step, history, val_hist, ep_seconds = int(getattr(opt, "t", 0)), 0, [], [], []
        for ep in range(E):
            t0 = time.perf_counter()
            for s in range(0, n, bs):
                b = orders[ep][s:s + bs]
                yb = Ya[b]
                lr = float(aops.raw(schedule(step))) if schedule is not None else float(opt.lr)
                parts = np.array_split(np.arange(len(b)), len(R))
                valid = max(int((yb >= 0).sum()), 1) if ce else None
                m_all = None
                for rep, part in zip(R, parts):              # every card starts its share at once
                    o, net = rep["ops"], rep["net"]
                    o.fill((net.G, 0), net.n, 0.0)
                    if not len(part):
                        continue
                    k = len(part)
                    o.gather((rep["XA"], 0), (rep["IDX"], 0), rep["Xd"], k, xw, ep * n + s + int(part[0]))
                    o.gather((rep["YA"], 0), (rep["IDX"], 0), rep["Yd"], k, yw, ep * n + s + int(part[0]))
                    net.round_weights()
                    net.train = True
                    xshape = (k,) + Xa.shape[1:]
                    out, shape = net.forward(rep["Xd"], xshape)
                    rows = _rows(shape)
                    dout = net.arena.get("dout", rows * shape[-1])
                    lossv = (rep["LOSSE"], ep)
                    if ce:
                        w = 1.0 / valid
                        o.ce(out, rep["Yd"], dout, lossv, rows, shape[-1], w, w * len(b) / n)
                    else:
                        m_all = len(b) * int(np.prod(shape[1:]))
                        _regress(o, loss, out, rep["Yd"], dout, lossv, rows * shape[-1], 1.0 / m_all,
                                 len(b) / (n * m_all))
                    net.backward(dout)
                o0 = r0["ops"]
                for rep in R:                                # every card's gradient is complete
                    rep["ops"].sync()
                # reduce-scatter: card r sums only its own slice of the gradient, pulled from every other card (the
                # cards pull at the same time, each on its own queue); no card needs the whole sum
                for rep in R:
                    o, lo, k = rep["ops"], rep["lo"], rep["hi"] - rep["lo"]
                    if k <= 0:
                        continue
                    for q in R:
                        if q is not rep:
                            o.pull((rep["TMP"], 0), q["ops"], (q["net"].G, lo), k)
                            o.axpy((rep["net"].G, lo), (rep["TMP"], 0), k, 1.0)
                t_step += 1
                for rep in R:
                    o, net, lo, k = rep["ops"], rep["net"], rep["lo"], rep["hi"] - rep["lo"]
                    if EWC is not None and k > 0:            # g += strength * F * (w - anchor), on the slice
                        o.fill((rep["ET"], 0), k, 0.0)
                        o.axpy((rep["ET"], 0), (net.P, lo), k, 1.0)
                        o.axpy((rep["ET"], 0), (rep["EA"], lo), k, -1.0)
                        o.mul3((rep["ET"], 0), (rep["ET"], 0), (rep["EF"], lo), k)
                        o.axpy((net.G, lo), (rep["ET"], 0), k, EWC)
                    if clip or _guard():
                        o.fill((rep["SQ"], 0), 1, 0.0)
                        if k > 0:
                            o.sumsq((net.G, lo), k, (rep["SQ"], 0))
                if clip:                                     # the norm of the whole gradient: the slices' sums added
                    for rep in R:
                        rep["ops"].sync()
                    for rep in R:
                        o = rep["ops"]
                        for q in R:
                            if q is not rep:
                                o.pull((rep["SQX"], 0), q["ops"], (q["SQ"], 0), 1)
                                o.axpy((rep["SQ2"], 0), (rep["SQX"], 0), 1, 1.0)
                for rep in R:                                # each card updates its slice
                    o, net = rep["ops"], rep["net"]
                    lo, k = rep["lo"], rep["hi"] - rep["lo"]
                    if clip:
                        o.axpy((rep["SQ2"], 0), (rep["SQ"], 0), 1, 1.0)
                    if k <= 0:
                        continue
                    sq = (rep["SQ2"], 0) if clip else (rep["SQ"], 0)
                    if adam:
                        c1, c2 = 1 - opt.b1 ** t_step, 1 - opt.b2 ** t_step
                        nd = min(max(net.n_decay - lo, 0), k)
                        o.adam((net.P, lo), (net.G, lo), (rep["st1"], 0), (rep["st2"], 0), sq, k,
                               nd, lr, opt.b1, opt.b2, opt.eps, c1, c2, wd, float(clip or 0.0),
                               (rep["COMP"], 0) if rep["COMP"] else None)
                    else:
                        o.sgd((net.P, lo), (net.G, lo), (rep["st1"], 0), sq, k, lr, float(opt.momentum),
                              float(clip or 0.0), (rep["COMP"], 0) if rep["COMP"] else None)
                    if clip:
                        o.fill((rep["SQ2"], 0), 1, 0.0)
                for rep in R:                                # every card gets the slices the others updated
                    rep["ops"].sync()
                for rep in R:
                    for q in R:
                        if q is not rep and q["hi"] > q["lo"]:
                            rep["ops"].pull((rep["net"].P, q["lo"]), q["ops"], (q["net"].P, q["lo"]), q["hi"] - q["lo"])
                if EMA is not None:
                    o0.axpy((EMA, 0), (EMA, 0), nP, ema - 1.0)
                    o0.axpy((EMA, 0), (r0["net"].P, 0), nP, 1.0 - ema)
                step += 1
                for rep in R:
                    rep["gov"].tick(rep["ops"].sync)
            for rep in R:
                rep["ops"].sync()
            ep_seconds.append(round(time.perf_counter() - t0, 3))
            if val is not None:
                val_hist.append(_eval_loss(r0["net"], o0, val, ids, loss, bs))
            if verbose and (ep % max(1, E // 10) == 0 or ep == E - 1):
                lv = sum(float(rep["ops"].get(rep["LOSSE"], 1, ep)[0]) for rep in R)
                print(f"epoch {ep:4}  loss {lv:.5f}  ({len(R)} cards)")
        history = [float(sum(v)) for v in zip(*[rep["ops"].get(rep["LOSSE"], max(1, E))[:E] for rep in R])]
        opt.t = t_step
        out_params = r0["net"].params()
        ema_params = r0["net"].params(EMA) if EMA is not None else None
        peak = max(rep["ops"].used for rep in R)
        temps = [rep["gov"].status() for rep in R]
    finally:
        for rep in R:
            o = rep["ops"]
            for key in ("st1", "st2", "SQ", "LOSSE", "XA", "YA", "IDX", "EA", "EF", "ET", "TMP", "SQX", "SQ2", "COMP"):
                if rep.get(key) is not None:
                    o.free(rep[key])
            rep["net"].close()
    return Record(params=out_params, history=history, val_history=val_hist, gpu_memory_mb=round(peak / 2 ** 20),
                  gpu_temperature=temps[0], gpu_temperatures=temps, epoch_seconds=ep_seconds,
                  epoch_temperature_c=[t.get("temperature_c") if t else None for t in temps], ema_params=ema_params,
                  device=f"{len(R)} GPUs ({', '.join(getattr(rep['ops'], 'name', '?') for rep in R)}), Apsi kernels, "
                         "data-parallel")


def _forward_batches(net, ops, Xa, bs):
    outs = []
    net.train = False
    Xd = net.arena.get("Xp", min(bs, len(Xa)) * int(np.prod(Xa.shape[1:])))
    for s in range(0, len(Xa), bs):
        xb = Xa[s:s + bs]
        ops.put(Xd[0], xb.reshape(-1))
        net.round_weights()
        out, shape = net.forward(Xd, xb.shape)
        outs.append(ops.get(out[0], int(np.prod(shape)), out[1]).reshape(shape))
    return np.concatenate(outs, 0)


def _eval_loss(net, ops, val, ids, ce, bs):
    return _eval_loss_host(lambda Xa: _forward_batches(net, ops, Xa, bs), val, ids, ce, None)


def _eval_loss_host(forward, val, ids, ce, bs):
    """The loss on (inputs, targets): forward(inputs) gives the model's output (bs: in batches of bs here)."""
    from . import ops as aops
    vx, vy = val
    Xa = _inputs(aops.raw(vx), ids)
    if bs:
        out = np.concatenate([forward(Xa[s:s + bs]) for s in range(0, len(Xa), bs)], 0).astype(np.float64)
    else:
        out = forward(Xa).astype(np.float64)
    vy = np.asarray(aops.raw(vy))
    if ce is True or ce == "cross_entropy":
        lg = out.reshape(-1, out.shape[-1])
        lab = vy.reshape(-1).astype(np.int64)
        ok = lab >= 0
        mx = lg.max(1, keepdims=True)
        lse = (mx + np.log(np.exp(lg - mx).sum(1, keepdims=True)))[:, 0]
        return float((lse[ok] - lg[np.arange(len(lab))[ok], lab[ok]]).mean())
    vy = vy.astype(np.float64).reshape(out.shape)
    if ce == "mae":
        return float(np.abs(out - vy).mean())
    if ce == "bce":
        sg = 1 / (1 + np.exp(-out))
        return float(-(vy * np.log(sg + 1e-12) + (1 - vy) * np.log(1 - sg + 1e-12)).mean())
    return float(((out - vy) ** 2).mean())


def predict(model, params, X, device="cuda", precision="auto", batch_size=64, ops=None, graph=None):
    from . import layers as L, ops as aops
    if ops is None:
        ops, why = globals()["device"](device)
        if ops is None:
            raise TrainError(f"no GPU: {why}")
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    net = Net(ops, model, params, precision, graph)
    try:
        return _forward_batches(net, ops, _inputs(aops.raw(X), isinstance(first, L.Embedding)),
                                int(batch_size) if graph is None else 1 << 62)
    finally:
        net.close()
