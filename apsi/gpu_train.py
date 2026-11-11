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

    def gemm(self, A, B, Cv, M, N, K, sam, sak, sbk, sbn, ldc, alpha=1.0, beta=False, bias=None, batch=(1, 1),
             sa=(0, 0), sb=(0, 0), sc=(0, 0), passes=1):
        nb1, nb2 = batch
        self._gemm(A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes)


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

    def ln_b(self, X, Gm, DY, MEAN, RSTD, DX, M, N, rms):
        x = self.v(X, M * N).reshape(M, N)
        mu = 0.0 if rms else self.v(MEAN, M)[:, None]
        r = self.v(RSTD, M)[:, None]
        xh = (x - mu) * r
        gdy = self.v(DY, M * N).reshape(M, N) * self.v(Gm, N)
        m1 = 0.0 if rms else gdy.mean(1, keepdims=True)
        m2 = (gdy * xh).mean(1, keepdims=True)
        self.v(DX, M * N)[:] = (r * (gdy - m1 - xh * m2)).reshape(-1)

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

    def fakeq(self, W, WQ, K, N, kind):
        from .lowbit import round_weights
        name = {v: k for k, v in KS.FAKEQ.items()}[kind]
        self.v(WQ, K * N)[:] = round_weights(self.v(W, K * N).reshape(K, N), name).reshape(-1)

    def _gemm(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes):
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
        self._calls = {}
        self._launch = lib.cuLaunchKernel

    _CT = {"p": ctypes.c_uint64, "u": ctypes.c_uint32, "i": ctypes.c_int32, "f": ctypes.c_float}

    def _go(self, name, grid, block, *args):
        gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
        key = (name, gx, gy, block, args)
        hit = self._calls.get(key)
        if hit is None:
            objs = [self._CT[k](v) for k, v in args]
            arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
            if len(self._calls) > 16384:
                self._calls.clear()
            hit = self._calls[key] = (objs, arr)
        self.dev.ensure()
        r = self._launch(self.fn[name], max(1, gx), max(1, gy), 1, block, 1, 1, 0, None, hit[1], None)
        if r:
            self.dev.drv.check(r, f"launching {name}")

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
        self._go("ln_f", M, 256, self.p(X), self.p(Gm), self.p(Bt), self.p(Y), self.p(MEAN), self.p(RSTD),
                 ("u", M), ("u", N), ("f", eps), ("u", 1 if rms else 0))

    def ln_b(self, X, Gm, DY, MEAN, RSTD, DX, M, N, rms):
        self._go("ln_b", M, 256, self.p(X), self.p(Gm), self.p(DY), self.p(MEAN), self.p(RSTD), self.p(DX),
                 ("u", M), ("u", N), ("u", 1 if rms else 0))

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

    def fakeq(self, W, WQ, K, N, kind):
        self._go("fakeq", self.blocks(((K + 31) // 32) * N), 256, self.p(W), self.p(WQ), ("u", K), ("u", N),
                 ("u", kind))

    def _gemm(self, A, B, Cv, bias, M, N, K, sam, sak, sbk, sbn, ldc, nb1, nb2, sa, sb, sc, alpha, beta, passes):
        tiles = ((M + KS.TILE - 1) // KS.TILE) * ((N + KS.TILE - 1) // KS.TILE)
        args = (self.p(A), self.p(B), self.p(Cv), self.p(bias), ("u", M), ("u", N), ("u", K), ("u", sam),
                ("u", sak), ("u", sbk), ("u", sbn), ("u", ldc), ("u", nb2), ("u", sa[0]), ("u", sa[1]),
                ("u", sb[0]), ("u", sb[1]), ("u", sc[0]), ("u", sc[1]), ("f", alpha), ("u", 1 if beta else 0),
                ("u", 1 if bias is not None else 0))
        if self.tc:
            self._go("gemm_tc" if passes == 1 else "gemm_tc3", (tiles, nb1 * nb2), 128, *args)
        else:
            self._go("gemm_simt", (tiles, nb1 * nb2), 256, *args)


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

    def fwd(self, x, shape):
        o, M = self.net.ops, _rows(shape)
        self.x, self.M = x, M
        y = self.buf("y", M * self.N)
        o.gemm(x, self.W.w, y, M, self.N, self.K, self.K, 1, self.N, 1, self.N, bias=self.b.w if self.b else None,
               passes=self.net.passes)
        return y, shape[:-1] + (self.N,)

    def bwd(self, dy, need_dx=True):
        o, M, K, N = self.net.ops, self.M, self.K, self.N
        # dW[K, N] += x^T dy: A(k, m) = x[m * K + k]
        o.gemm(self.x, dy, self.W.g, K, N, M, 1, K, N, 1, N, beta=True, passes=self.net.passes)
        if self.b is not None:
            o.colsum(dy, None, None, None, None, self.b.g, M, N, 0)
        if not need_dx:
            return None
        dx = self.buf("dx", M * K)
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
        dx = self.buf("dx", self.n)
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
        dx = self.buf("dx", self.n)
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

    def bwd(self, dy, need_dx=True):
        o = self.net.ops
        o.colsum(dy, self.x, self.mean, self.rstd, self.gm.g, self.bt.g if self.bt else None, self.M, self.N,
                 2 if self.rms else 1)
        dx = self.buf("dx", self.M * self.N)
        o.ln_b(self.x, self.gm.w, dy, self.mean, self.rstd, dx, self.M, self.N, self.rms)
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

    def fwd(self, x, shape):
        for nd in self.nodes:
            x, shape = nd.fwd(x, shape)
        return x, shape

    def bwd(self, dy, need_dx=True):
        for i in range(len(self.nodes) - 1, -1, -1):
            dy = self.nodes[i].bwd(dy, need_dx or i > 0)
            if dy is None:
                return None
        return dy


class _Residual(_Node):
    def __init__(self, net, key, inner, norm):
        super().__init__(net, key)
        self.inner, self.norm = inner, norm

    def fwd(self, x, shape):
        h, hs = self.norm.fwd(x, shape) if self.norm else (x, shape)
        f, fs = self.inner.fwd(h, hs)
        self.n = int(np.prod(shape))
        y = self.buf("y", self.n)
        self.net.ops.add3(y, x, f, self.n)
        return y, shape

    def bwd(self, dy, need_dx=True):
        dh = self.inner.bwd(dy, True)
        if self.norm:
            dh = self.norm.bwd(dh, True)
        dx = self.buf("dx", self.n)
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

    def fwd(self, x, shape):
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
        S, P = self.buf("s", B * h * T * T), self.buf("p", B * h * T * T)
        sc = 1.0 / math.sqrt(hd)
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
        o.gemm(att, self.Wo.w, y, M, D, D, D, 1, D, 1, D, bias=self.bo.w, passes=ps)
        return y, shape

    def bwd(self, dy, need_dx=True):
        o, B, T, D, h = self.net.ops, self.B, self.T, self.D, self.h
        hd, M, ps = D // h, B * T, self.net.passes
        sc = 1.0 / math.sqrt(hd)
        o.gemm(self.att, dy, self.Wo.g, D, D, M, 1, D, D, 1, D, beta=True, passes=ps)
        o.colsum(dy, None, None, None, None, self.bo.g, M, D, 0)
        datt = self.buf("datt", M * D)
        o.gemm(dy, self.Wo.w, datt, M, D, D, D, 1, 1, D, D, passes=ps)
        dP = self.buf("dp", B * h * T * T)               # dP = datt v^T
        o.gemm(datt, self.v, dP, T, T, hd, D, 1, 1, D, T, batch=(B, h), sa=(T * D, hd), sb=(T * D, hd),
               sc=(h * T * T, T * T), passes=ps)
        dv = self.buf("dv", M * D)                       # dv = P^T datt
        o.gemm(self.P, datt, dv, T, hd, T, 1, T, D, 1, D, batch=(B, h), sa=(h * T * T, T * T), sb=(T * D, hd),
               sc=(T * D, hd), passes=ps)
        dS = self.buf("ds", B * h * T * T)
        o.softmax_b(self.P, dP, dS, B * h * T, T)
        dq, dk = self.buf("dq", M * D), self.buf("dk", M * D)
        o.gemm(dS, self.k, dq, T, hd, T, T, 1, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * T, T * T),
               sb=(T * D, hd), sc=(T * D, hd), passes=ps)                                   # dq = dS k
        o.gemm(dS, self.q, dk, T, hd, T, 1, T, D, 1, D, alpha=sc, batch=(B, h), sa=(h * T * T, T * T),
               sb=(T * D, hd), sc=(T * D, hd), passes=ps)                                   # dk = dS^T q
        dx = self.buf("dx", M * D)
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
        self.root = self.compile(model, params, "m")

    def param(self, a):
        if not isinstance(a, np.ndarray) or id(a) not in self.offsets:
            raise TrainError("a parameter of the model is not an array")
        return _Param(self, a.shape, self.offsets[id(a)])

    def compile(self, layer, p, key):
        from . import layers as L
        kind = type(layer).__name__
        if isinstance(layer, L.Sequential):
            return _Seq(self, key, [self.compile(l_, pp, f"{key}.{i}") for i, (l_, pp) in enumerate(zip(layer.layers, p))])
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
            return _Residual(self, key, self.compile(layer.inner, p.inner, key + ".f"),
                             self.compile(layer.norm, p.norm, key + ".n") if layer.norm else None)
        if isinstance(layer, L.TransformerBlock):
            return _Seq(self, key, [self.compile(layer.attn, p.attn, key + ".a"),
                                    self.compile(layer.mlp, p.mlp, key + ".m")])
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

    def params(self):
        flat = self.ops.get(self.P, self.n)

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


def fit(model, params, X, y=None, loss="mse", opt=None, epochs=10, batch_size=None, seed=0, verbose=True,
        clip=None, schedule=None, val=None, device="cuda", precision="auto", ops=None):
    from . import layers as L, ops as aops
    from .stdlib import Adam, AdamW, SGD
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
    net = Net(ops, model, params, precision)
    rng = np.random.default_rng(int(seed))
    net.seed = int(seed) * 7919 + 1
    adam = isinstance(opt, Adam)
    st1, st2 = ops.alloc(net.n), (ops.alloc(net.n) if adam else None)
    ops.fill((st1, 0), net.n, 0.0)
    if st2 is not None:
        ops.fill((st2, 0), net.n, 0.0)
    SQ = ops.alloc(1)
    t_step = int(getattr(opt, "t", 0))
    bs = int(batch_size) if batch_size else n
    xw, yw = int(np.prod(Xa.shape[1:])), int(np.prod(Ya.shape[1:]))
    Xd = net.arena.get("X", min(bs, n) * xw)
    Yd = net.arena.get("Y", min(bs, n) * yw)
    # the whole data set on the card when it is small next to the card (one copy, then each batch is gathered
    # there from a permutation uploaded once an epoch: no copy from the processor inside an epoch)
    resident = (Xa.nbytes + Ya.nbytes) < min(1 << 30, getattr(ops, "budget", 1 << 30))
    if resident:
        XA, YA, IDX = ops.alloc(Xa.size), ops.alloc(Ya.size), ops.alloc(n)
        ops.put(XA, Xa.reshape(-1))
        ops.put(YA, Ya.reshape(-1))
        net.owned += [XA, YA, IDX]
    history, val_hist, step = [], [], 0
    from . import thermal
    gov = thermal.governor(getattr(ops, "dev", None))        # keeps the card near APSI_GPU_TEMP (77 C)
    wd = float(getattr(opt, "wd", 0.0)) if isinstance(opt, AdamW) else 0.0
    try:
        for ep in range(int(epochs)):
            order = rng.permutation(n) if batch_size else np.arange(n)
            ops.fill((net.LOSS, 0), 1, 0.0)
            if resident:
                ops.put(IDX, order.astype(np.int32))
            for s in range(0, n, bs):
                b = order[s:s + bs]
                xb, yb = Xa[b], Ya[b]
                if resident:
                    ops.gather((XA, 0), (IDX, 0), Xd, len(b), xw, s)
                    ops.gather((YA, 0), (IDX, 0), Yd, len(b), yw, s)
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
                    ops.ce(out, Yd, dout, (net.LOSS, 0), rows, shape[-1], w, w * len(b) / n)
                else:
                    m = rows * shape[-1]
                    ops.mse(out, Yd, dout, (net.LOSS, 0), m, 1.0 / m, len(b) / (n * m))
                net.backward(dout)
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
                gov.tick(ops.sync)
            history.append(float(ops.get(net.LOSS, 1)[0]))          # the one wait of the epoch
            extra = ""
            if val is not None:
                vl = _eval_loss(net, ops, val, ids, ce, bs)
                val_hist.append(vl)
                extra = f"  val {vl:.5f}"
            if verbose and (ep % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1):
                print(f"epoch {ep:4}  loss {history[-1]:.5f}{extra}")
        opt.t = t_step
        peak = ops.used
        out_params = net.params()
    finally:
        for bf in (st1, st2, SQ):
            if bf is not None:
                ops.free(bf)
        net.close()
    temp = gov.status()
    if verbose and temp and temp["paced_s"]:
        print(f"GPU kept near {temp['target_c']:.0f} C (highest {temp['max_c']:.0f} C, paced {temp['paced_s']} s)")
    return Record(params=out_params, history=history, val_history=val_hist, gpu_memory_mb=round(peak / 2 ** 20),
                  gpu_temperature=temp,
                  device=f"gpu ({getattr(ops, 'name', '?')}, Apsi kernels"
                         f"{', tensor cores' if ops.tc else ''})")


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
