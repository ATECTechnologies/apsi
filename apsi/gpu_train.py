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

    def mse(self, Y, T, DY, LOSS, n, wg, wl):
        d = self.v(Y, n) - self.v(T, n)
        self.v(DY, n)[:] = 2 * d * np.float32(wg)
        self.v(LOSS, 1)[0] += np.float32(wl) * (d * d).sum()

    def sumsq(self, X, n, ACC):
        x = self.v(X, n)
        self.v(ACC, 1)[0] += (x * x).sum()

    def adam(self, Pm, Gr, Mm, Vm, SQ, n, n_decay, lr, b1, b2, eps, c1, c2, wd, clip):
        g = self.v(Gr, n) * self._clip(SQ, clip)
        p, m, v = self.v(Pm, n), self.v(Mm, n), self.v(Vm, n)
        p[:n_decay] *= np.float32(1 - lr * wd)
        m[:] = b1 * m + (1 - b1) * g
        v[:] = b2 * v + (1 - b2) * g * g
        p[:] = p - np.float32(lr) * (m / np.float32(c1)) / (np.sqrt(v / np.float32(c2)) + np.float32(eps))

    def sgd(self, Pm, Gr, VEL, SQ, n, lr, mom, clip):
        vel = self.v(VEL, n)
        vel[:] = np.float32(mom) * vel + self.v(Gr, n) * self._clip(SQ, clip)
        p = self.v(Pm, n)
        p[:] = p - np.float32(lr) * vel

    def _clip(self, SQ, clip):
        if clip <= 0:
            return np.float32(1)
        return np.float32(min(1.0, clip / math.sqrt(float(self.v(SQ, 1)[0]) + 1e-30)))

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
                        tuple(f"gemm_tmn_aa{j}" for j in KS.EPI_ACTS) + ("bf_lo",):
                    f = ctypes.c_void_p()
                    dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), tmod, name.encode()), "cuModuleGetFunction")
                    self.fn[name] = f
                self.tiled = True
            except Exception as e:
                self.tc_why = f"tiled products unavailable: {type(e).__name__}: {e}"
        if self.tiled and os.environ.get("APSI_GEMM_VARIANT", "auto").strip().lower() != "a":
            # more builds of the same products: "x" loads first (2 blocks of 8 warps per SM), "w" 4 warps of 64 x 64
            # over slices of 16 (the shape of the fast GEMM libraries); the card times them per shape
            for tag, which in (("x", "tiled_x"), ("w", "tiled_w")):
                try:
                    xmod, first = dev.load(f"apsi-train-tiled-{tag}", KS.ptx(which))
                    xf = {f"gemm_tkn#{tag}": first}
                    for name in KS.TILED[1:] + tuple(e[0] for e in KS.EPI_KERNELS) + \
                            tuple(f"gemm_tmn_aa{j}" for j in KS.EPI_ACTS):
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
                    for kind in "fb":
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
        p = self.dev.alloc(4 * max(1, n))
        self.used += 4 * n
        return Buf(p, n)

    def free(self, b):
        self.used -= 4 * b.n
        self.dev.free(b.ptr)
        self._calls.clear()

    def put(self, b, a, off=0):
        a = np.ascontiguousarray(a)
        if a.dtype not in (np.float32, np.int32, np.uint32):
            a = a.astype(np.float32)
        self.dev.to_device(b.ptr + 4 * off, a.ctypes.data, a.nbytes)

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
        rows = 64
        self._go("colsum", ((N + 255) // 256, (M + rows - 1) // rows), 256, self.p(DY), self.p(X), self.p(MEAN),
                 self.p(RSTD), self.p(DG), self.p(DB), ("u", M), ("u", N), ("u", rows), ("u", mode))

    def ln_f(self, X, Gm, Bt, Y, MEAN, RSTD, M, N, eps, rms):
        self._go("ln_f", -(-M // 8), 256, self.p(X), self.p(Gm), self.p(Bt), self.p(Y), self.p(MEAN), self.p(RSTD),
                 ("u", M), ("u", N), ("f", eps), ("u", 1 if rms else 0))

    def ln_b(self, X, Gm, DY, MEAN, RSTD, DX, M, N, rms, RES=None):
        self._go("ln_b", -(-M // 8), 256, self.p(X), self.p(Gm), self.p(DY), self.p(MEAN), self.p(RSTD), self.p(DX),
                 ("u", M), ("u", N), ("u", 1 if rms else 0), self.p(RES), ("u", 1 if RES is not None else 0))

    def embed_f(self, E, IDS, Y, M, D):
        self._go("embed_f", self.blocks(M * D), 256, self.p(E), self.p(IDS), self.p(Y), ("u", M), ("u", D))

    def embed_b(self, DY, IDS, DE, M, D, V):
        self._go("embed_b", self.blocks(M * D), 256, self.p(DY), self.p(IDS), self.p(DE), ("u", M), ("u", D))

    def pos_f(self, X, P, Y, M, D, T):
        self._go("pos_f", self.blocks(M * D), 256, self.p(X), self.p(P), self.p(Y), ("u", M), ("u", D), ("u", T))

    def pos_b(self, DY, DP, M, D, T):
        self._go("pos_b", self.blocks(T * D), 256, self.p(DY), self.p(DP), ("u", M), ("u", D), ("u", T))

    def softmax_f(self, S, P, rows, n, T, causal):
        self._go("softmax_f", rows, 256, self.p(S), self.p(P), ("u", rows), ("u", n), ("u", T), ("u", 1 if causal else 0))

    def softmax_b(self, P, DP, DS, rows, n):
        self._go("softmax_b", rows, 256, self.p(P), self.p(DP), self.p(DS), ("u", rows), ("u", n))

    def ce(self, L, LAB, DL, LOSS, M, V, wg, wl):
        self._go("ce", M, 256, self.p(L), self.p(LAB), self.p(DL), self.p(LOSS), ("u", M), ("u", V), ("f", wg),
                 ("f", wl))

    def mse(self, Y, T, DY, LOSS, n, wg, wl):
        self._go("mse", self.blocks(n), 256, self.p(Y), self.p(T), self.p(DY), self.p(LOSS), ("u", n), ("f", wg),
                 ("f", wl))

    def sumsq(self, X, n, ACC):
        self._go("sumsq", min(self.blocks(n), 1024), 256, self.p(X), ("u", n), self.p(ACC))

    def adam(self, Pm, Gr, Mm, Vm, SQ, n, n_decay, lr, b1, b2, eps, c1, c2, wd, clip):
        self._go("adam", self.blocks(n), 256, self.p(Pm), self.p(Gr), self.p(Mm), self.p(Vm), self.p(SQ), ("u", n),
                 ("u", n_decay), ("f", lr), ("f", b1), ("f", b2), ("f", eps), ("f", c1), ("f", c2), ("f", wd),
                 ("f", clip))

    def sgd(self, Pm, Gr, VEL, SQ, n, lr, mom, clip):
        self._go("sgd", self.blocks(n), 256, self.p(Pm), self.p(Gr), self.p(VEL), self.p(SQ), ("u", n), ("f", lr),
                 ("f", mom), ("f", clip))

    def drop_f(self, X, Y, MASK, n, p, seed):
        self._go("drop_f", self.blocks(n), 256, self.p(X), self.p(Y), self.p(MASK), ("u", n), ("f", p),
                 ("i", int(np.int32(np.uint32(seed & 0xFFFFFFFF)))))

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
        self.fill(DQ, B * T * H * hd, 0.0)
        self._go(f"flash_b{hd}", (-(-T // KS.BQ), B * H), 128, self.p(Q), self.p(K), self.p(V), self.p(DO),
                 self.p(L), self.p(DD), self.p(DQ), self.p(DK), self.p(DV), ("u", T), ("u", H * hd), ("u", H),
                 ("f", scale), ("f", scale * 1.4426950408889634), ("u", 1 if causal else 0))

    def fakeq(self, W, WQ, K, N, kind):
        self._go("fakeq", self.blocks(((K + 31) // 32) * N), 256, self.p(W), self.p(WQ), ("u", K), ("u", N),
                 ("u", kind))

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

    def _variant(self, name, grid, args, M, N, ldc, nb):
        """Which build of a tiled product runs: the one the card itself measured faster for this shape (each is
        timed twice on scratch outputs the first time the shape comes; APSI_GEMM_VARIANT=a or x fixes it)."""
        cands = [name] + [f"{name}#{t}" for t in ("x", "w") if f"{name}#{t}" in self.fn]
        force = os.environ.get("APSI_GEMM_VARIANT", "auto").strip().lower()
        if force in ("x", "w"):
            return f"{name}#{force}" if f"{name}#{force}" in self.fn else name
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
            targs[23] = self.p(self._scratch("tune_x", M * ldc + N))
        if targs[26][1]:
            targs[26] = self.p(self._scratch("tune_db", N))
        best, best_t = name, None
        ms = ctypes.c_float()
        for cand in cands + cands:
            e0, e1 = self._event(), self._event()
            self._lib.cuEventRecord(e0, None)
            self._go(cand, grid, self._threads(cand), *targs)
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
            db = self._db if name.startswith("gemm_tmn") and sbn == 1 else None
            args = [self.p(A), self.p(B), self.p(Cv), self.p(bias), ("u", M),
                    ("u", N), ("u", K), ("u", sam), ("u", sak), ("u", sbk), ("u", sbn), ("u", ldc), ("u", nb2),
                    ("u", sa[0]), ("u", sa[1]), ("u", sb[0]), ("u", sb[1]), ("u", sc[0]), ("u", sc[1]),
                    ("f", alpha), ("u", 1 if beta else 0), ("u", 1 if bias is not None else 0), ("u", kchunk),
                    self.p(X2), ("u", epi), ("u", act), self.p(db), ("u", 1 if db is not None else 0)]
            grid = (tiles, nb1 * nb2, nz)
            use = self._variant(name, grid, args, M, N, ldc, nb1 * nb2)
            self._go(use, grid, self._threads(use), *args)
            if db is not None:
                self._db = None                          # taken
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
    """The cards of "cuda:all" or "cuda:0,1,2" (several cards: data-parallel training), as a list of ops; or None."""
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
        return self.net.arena.get(("g", getattr(self, "gkey", self.key), name), n)

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
    projection. Scores and probabilities are kept for the backward pass ([batch x heads, T, T])."""

    def __init__(self, net, key, layer, p):
        super().__init__(net, key)
        self.D, self.h, self.causal = layer.d, layer.h, layer.causal
        self.Wq, self.Wk, self.Wv, self.Wo = (net.param(getattr(p, n)) for n in ("Wq", "Wk", "Wv", "Wo"))
        self.bo = net.param(p.bo)

    takes_residual = True

    def fwd(self, x, shape, residual=None):
        if len(shape) != 3:
            raise TrainError("multi_head_attention on the GPU takes [batch, time, features]")
        o, (B, T, D), h = self.net.ops, shape, self.h
        hd = D // h
        M = B * T
        self.x, self.B, self.T = x, B, T
        ps = self.net.passes
        q, k, v = self.buf("q", M * D), self.buf("k", M * D), self.buf("v", M * D)
        for W, out in ((self.Wq, q), (self.Wk, k), (self.Wv, v)):
            o.gemm(x, W.w, out, M, D, D, D, 1, D, 1, D, passes=ps)
        self.q, self.k, self.v = q, k, v
        sc = 1.0 / math.sqrt(hd)
        self.fused = ps == 1 and getattr(o, "flash", False) and hd in KS.FLASH_HD and D % 4 == 0
        if self.fused:                                  # scores, softmax and the weighted sum in one kernel
            att, self.L = self.buf("att", M * D), self.buf("lse", B * h * T)
            o.flash_fwd(q, k, v, att, self.L, B, T, h, hd, self.causal, sc)
            self.att = att
            y = self.buf("y", M * D)
            o.gemm(att, self.Wo.w, y, M, D, D, D, 1, D, 1, D, bias=self.bo.w, passes=ps, epi=3 if residual else 0,
                   X2=residual)
            return y, shape
        S, P = self.buf("s", B * h * T * T), self.buf("p", B * h * T * T)
        # S[b, h](t, s) = q[b, t, h, :] . k[b, s, h, :] * sc
        o.gemm(q, k, S, T, T, hd, D, 1, 1, D, T, alpha=sc, batch=(B, h), sa=(T * D, hd), sb=(T * D, hd),
               sc=(h * T * T, T * T), passes=ps)
        o.softmax_f(S, P, B * h * T, T, T, self.causal)
        self.P = P
        att = self.buf("att", M * D)                    # [b, t, h, :] = P[b, h] @ v[b, :, h, :]
        o.gemm(P, v, att, T, hd, T, T, 1, D, 1, D, batch=(B, h), sa=(h * T * T, T * T), sb=(T * D, hd),
               sc=(T * D, hd), passes=ps)
        self.att = att
        y = self.buf("y", M * D)
        o.gemm(att, self.Wo.w, y, M, D, D, D, 1, D, 1, D, bias=self.bo.w, passes=ps, epi=3 if residual else 0,
               X2=residual)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, B, T, D, h = self.net.ops, self.B, self.T, self.D, self.h
        hd, M, ps = D // h, B * T, self.net.passes
        sc = 1.0 / math.sqrt(hd)
        o.gemm(self.att, dy, self.Wo.g, D, D, M, 1, D, D, 1, D, beta=True, passes=ps, db=self.bo.g)
        datt = self.gbuf("datt", M * D)
        o.gemm(dy, self.Wo.w, datt, M, D, D, D, 1, 1, D, D, passes=ps)
        if self.fused:
            dd = self.gbuf("dd", B * h * T)
            o.flash_dot(datt, self.att, dd, B, T, h, hd)
            dq, dk, dv = self.gbuf("dq", M * D), self.gbuf("dk", M * D), self.gbuf("dv", M * D)
            o.flash_bwd(self.q, self.k, self.v, datt, self.L, dd, dq, dk, dv, B, T, h, hd, self.causal, sc)
            return self._grads_in(dq, dk, dv, M, D, ps)
        dP = self.gbuf("dp", B * h * T * T)               # dP = datt v^T
        o.gemm(datt, self.v, dP, T, T, hd, D, 1, 1, D, T, batch=(B, h), sa=(T * D, hd), sb=(T * D, hd),
               sc=(h * T * T, T * T), passes=ps)
        dv = self.gbuf("dv", M * D)                       # dv = P^T datt
        o.gemm(self.P, datt, dv, T, hd, T, 1, T, D, 1, D, batch=(B, h), sa=(h * T * T, T * T), sb=(T * D, hd),
               sc=(T * D, hd), passes=ps)
        dS = self.gbuf("ds", B * h * T * T)
        o.softmax_b(self.P, dP, dS, B * h * T, T)
        dq, dk = self.gbuf("dq", M * D), self.gbuf("dk", M * D)
        o.gemm(dS, self.k, dq, T, hd, T, T, 1, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * T, T * T),
               sb=(T * D, hd), sc=(T * D, hd), passes=ps)                                   # dq = dS k
        o.gemm(dS, self.q, dk, T, hd, T, 1, T, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * T, T * T),
               sb=(T * D, hd), sc=(T * D, hd), passes=ps)                                   # dk = dS^T q
        return self._grads_in(dq, dk, dv, M, D, ps)

    def _grads_in(self, dq, dk, dv, M, D, ps):
        o = self.net.ops
        dx = self.gbuf("dx", M * D)
        first = True
        for W, d in ((self.Wq, dq), (self.Wk, dk), (self.Wv, dv)):
            o.gemm(self.x, d, W.g, D, D, M, 1, D, D, 1, D, beta=True, passes=ps)
            o.gemm(d, W.w, dx, M, D, D, D, 1, 1, D, D, beta=not first, passes=ps)
            first = False
        return dx


class Net:
    """A model compiled for the device: parameters in one flat array, a tree of nodes."""

    def __init__(self, ops, model, params, precision="auto"):
        from . import lowbit
        self.ops = ops
        self.arena = _Arena(ops)
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
        if isinstance(layer, L.MultiHeadAttention):
            return _Attention(self, key, layer, p)
        raise TrainError(f"layer '{getattr(layer, 'kind', kind)}' is not on the GPU trainer yet "
                         "(backend=\"torch\" or \"numpy\" trains it)")

    def round_weights(self):
        if self.kind:
            from .lowbit import KINDS                     # noqa: F401
            for off, shape in self.mats:
                K = int(np.prod(shape[:-1]))
                self.ops.fakeq((self.P, off), (self.WQ, off), K, int(shape[-1]), KS.FAKEQ[self.kind])

    def forward(self, x, shape):
        return self.root.fwd(x, shape)

    def backward(self, dy):
        self.root.bwd(dy, False)

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


def _check_size(ops, params, opt, shards=1):
    """Training keeps 4 float32 numbers per parameter on the card (weights, gradient, AdamW's two moments: 16 bytes);
    a model that cannot fit is refused at once with the numbers, rather than failing half-way."""
    from .stdlib import Adam
    total = getattr(getattr(ops, "dev", None), "memory", None)
    if not total:
        return
    count = sum(int(np.asarray(a).size) for a in tree_leaves(params) if isinstance(a, np.ndarray))
    per = 8 + (8 if isinstance(opt, Adam) or opt is None else 4) / max(1, shards)
    need = count * per
    if need > 0.92 * total:
        fits = int(0.92 * total / per * 0.75)
        raise TrainError(
            f"{count / 1e9:.2f} B parameters need {need / 2 ** 30:.1f} GB on each card for training (weights, "
            f"gradients and the optimizer's state, {per:g} bytes each; with several cards the optimizer's state is "
            f"shared out among them) before any activations; this card has {total / 2 ** 30:.1f} GB. Models up to "
            f"about {fits / 1e6:,.0f} M parameters train on it; larger ones need cards with more memory or more "
            f"cards (device=\"cuda:all\"), and run here with use hub in int4 / int8")


def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, val=None, device="cuda", precision="auto", ops=None, ema=None, ewc=None):
    """ema=0.999: also keeps an exponential moving average of the weights (record.ema_params), the smoother set
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
    if isinstance(ops, (list, tuple)):
        if len(ops) > 1:
            return _fit_parallel(list(ops), model, params, X, y, loss, opt, epochs, batch_size, seed, verbose, clip,
                                 schedule, val, precision, ema, ewc)
        ops = ops[0]
    if ops is None:
        ops, why = globals()["device"](device)
        if ops is None:
            raise TrainError(f"no GPU to train on: {why}")
    loss = str(loss)
    if loss not in ("cross_entropy", "mse"):
        raise TrainError(f"loss '{loss}' is not on the GPU trainer (cross_entropy, mse)")
    opt = opt or Adam(1e-3)
    if not isinstance(opt, (Adam, SGD)):
        raise TrainError(f"optimizer {type(opt).__name__} is not on the GPU trainer (adam, adamw, sgd)")
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    ids = isinstance(first, L.Embedding)
    Xa = _inputs(aops.raw(X), ids)
    Ya = np.asarray(aops.raw(X if y is None else y))
    ce = loss == "cross_entropy"
    Ya = np.ascontiguousarray(Ya.astype(np.int32) if ce else Ya.astype(np.float32))
    n = Xa.shape[0]
    _check_size(ops, params, opt)
    net = Net(ops, model, params, precision)
    rng = np.random.default_rng(int(seed))
    net.seed = int(seed) * 7919 + 1
    adam = isinstance(opt, Adam)
    st1, st2 = ops.alloc(net.n), (ops.alloc(net.n) if adam else None)
    ops.fill((st1, 0), net.n, 0.0)
    if st2 is not None:
        ops.fill((st2, 0), net.n, 0.0)
    SQ = ops.alloc(1)
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
    Yd = net.arena.get("Y", min(bs, n) * yw)
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
                xb, yb = Xa[b], Ya[b]
                if resident:
                    ops.gather((XA, 0), (IDX, 0), Xd, len(b), xw, ep * n + s)
                    ops.gather((YA, 0), (IDX, 0), Yd, len(b), yw, ep * n + s)
                else:
                    ops.put(Xd[0], xb.reshape(-1))
                    ops.put(Yd[0], yb.reshape(-1))
                lr = float(aops.raw(schedule(step))) if schedule is not None else float(opt.lr)
                ops.fill((net.G, 0), net.n, 0.0)
                net.round_weights()
                net.train = True
                out, shape = net.forward(Xd, xb.shape)
                rows = _rows(shape)
                dout = net.arena.get("dout", rows * shape[-1])
                if ce:
                    valid = int((yb >= 0).sum())
                    w = 1.0 / max(valid, 1)
                    if yb.size != rows:
                        raise TrainError(f"cross_entropy: {yb.size} labels for {rows} rows of logits")
                    ops.ce(out, Yd, dout, lossv, rows, shape[-1], w, w * len(b) / n)
                else:
                    m = rows * shape[-1]
                    ops.mse(out, Yd, dout, lossv, m, 1.0 / m, len(b) / (n * m))
                net.backward(dout)
                if EWC_A is not None:                       # g += strength * F * (w - anchor)
                    ops.fill((EWC_T, 0), net.n, 0.0)
                    ops.axpy((EWC_T, 0), (net.P, 0), net.n, 1.0)
                    ops.axpy((EWC_T, 0), (EWC_A, 0), net.n, -1.0)
                    ops.mul3((EWC_T, 0), (EWC_T, 0), (EWC_F, 0), net.n)
                    ops.axpy((net.G, 0), (EWC_T, 0), net.n, lam)
                if clip:
                    ops.fill((SQ, 0), 1, 0.0)
                    ops.sumsq((net.G, 0), net.n, (SQ, 0))
                t_step += 1
                if adam:
                    c1, c2 = 1 - opt.b1 ** t_step, 1 - opt.b2 ** t_step
                    ops.adam((net.P, 0), (net.G, 0), (st1, 0), (st2, 0), (SQ, 0), net.n, net.n_decay, lr, opt.b1,
                             opt.b2, opt.eps, c1, c2, wd, float(clip or 0.0))
                else:
                    ops.sgd((net.P, 0), (net.G, 0), (st1, 0), (SQ, 0), net.n, lr, float(opt.momentum),
                            float(clip or 0.0))
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
                vl = _eval_loss(net, ops, val, ids, ce, bs)
                val_hist.append(vl)
                extra = f"  val {vl:.5f}"
            if show:
                print(f"epoch {ep:4}  loss {float(ops.get(LOSSE, 1, ep)[0]):.5f}{extra}")
        history = [float(v) for v in ops.get(LOSSE, max(1, E))[:E]]
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
                  precision, ema, ewc):
    """Data-parallel training on several cards: every card keeps the whole model and its optimizer state, takes its
    share of each batch, and the gradients are summed across the cards (card to card where the driver allows it)
    before the identical update on each, so the result is that of one card with the whole batch. Throughput grows
    with the cards; the model must fit on one of them."""
    from . import layers as L, ops as aops, thermal
    from .stdlib import Adam, AdamW, SGD
    loss = str(loss)
    if loss not in ("cross_entropy", "mse"):
        raise TrainError(f"loss '{loss}' is not on the GPU trainer (cross_entropy, mse)")
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
                   "SQ": o.alloc(1), "LOSSE": o.alloc(max(1, E)), "gov": thermal.governor(getattr(o, "dev", None))}
            o.fill((rep["st1"], 0), ln, 0.0)
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
        TMP = r0["ops"].alloc(r0["net"].n)
        nP = r0["net"].n
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
                rep["EA"], rep["EF"], rep["ET"] = o.alloc(nP), o.alloc(nP), o.alloc(nP)
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
                        o.mse(out, rep["Yd"], dout, lossv, rows * shape[-1], 1.0 / m_all, len(b) / (n * m_all))
                    net.backward(dout)
                o0 = r0["ops"]                               # the sum of the gradients, on the first card
                for rep in R[1:]:
                    rep["ops"].sync()
                    o0.pull((TMP, 0), rep["ops"], (rep["net"].G, 0), nP)
                    o0.axpy((r0["net"].G, 0), (TMP, 0), nP, 1.0)
                o0.sync()
                for rep in R[1:]:                            # and back to every card
                    rep["ops"].pull((rep["net"].G, 0), o0, (r0["net"].G, 0), nP)
                t_step += 1
                for rep in R:                                # the same update everywhere
                    o, net = rep["ops"], rep["net"]
                    if EWC is not None:
                        o.fill((rep["ET"], 0), nP, 0.0)
                        o.axpy((rep["ET"], 0), (net.P, 0), nP, 1.0)
                        o.axpy((rep["ET"], 0), (rep["EA"], 0), nP, -1.0)
                        o.mul3((rep["ET"], 0), (rep["ET"], 0), (rep["EF"], 0), nP)
                        o.axpy((net.G, 0), (rep["ET"], 0), nP, EWC)
                    if clip:
                        o.fill((rep["SQ"], 0), 1, 0.0)
                        o.sumsq((net.G, 0), nP, (rep["SQ"], 0))
                    lo, k = rep["lo"], rep["hi"] - rep["lo"]
                    if k <= 0:
                        continue
                    if adam:
                        c1, c2 = 1 - opt.b1 ** t_step, 1 - opt.b2 ** t_step
                        nd = min(max(net.n_decay - lo, 0), k)
                        o.adam((net.P, lo), (net.G, lo), (rep["st1"], 0), (rep["st2"], 0), (rep["SQ"], 0), k,
                               nd, lr, opt.b1, opt.b2, opt.eps, c1, c2, wd, float(clip or 0.0))
                    else:
                        o.sgd((net.P, lo), (net.G, lo), (rep["st1"], 0), (rep["SQ"], 0), k, lr, float(opt.momentum),
                              float(clip or 0.0))
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
                val_hist.append(_eval_loss(r0["net"], o0, val, ids, ce, bs))
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
            for key in ("st1", "st2", "SQ", "LOSSE", "XA", "YA", "IDX", "EA", "EF", "ET"):
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
    from . import ops as aops
    vx, vy = val
    out = _forward_batches(net, ops, _inputs(aops.raw(vx), ids), bs).astype(np.float64)
    vy = np.asarray(aops.raw(vy))
    if ce:
        lg = out.reshape(-1, out.shape[-1])
        lab = vy.reshape(-1).astype(np.int64)
        ok = lab >= 0
        mx = lg.max(1, keepdims=True)
        lse = (mx + np.log(np.exp(lg - mx).sum(1, keepdims=True)))[:, 0]
        return float((lse[ok] - lg[np.arange(len(lab))[ok], lab[ok]]).mean())
    return float(((out - vy) ** 2).mean())


def predict(model, params, X, device="cuda", precision="auto", batch_size=64, ops=None):
    from . import layers as L, ops as aops
    if ops is None:
        ops, why = globals()["device"](device)
        if ops is None:
            raise TrainError(f"no GPU: {why}")
    first = model.layers[0] if isinstance(model, L.Sequential) and model.layers else model
    net = Net(ops, model, params, precision)
    try:
        return _forward_batches(net, ops, _inputs(aops.raw(X), isinstance(first, L.Embedding)), int(batch_size))
    finally:
        net.close()
