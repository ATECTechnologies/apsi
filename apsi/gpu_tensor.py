"""Tensors on the card with Apsi's own kernels (the GPU trainer's library: no CUDA toolkit, no CuPy, no PyTorch).

    let a = gpu_tensor(x)              # on the first NVIDIA card; on the processor when there is none
    let b = (a @ w + bias).gelu()      # products, element-wise arithmetic, activations stay on the card
    let p = softmax(b)                 # any other function: computed on the processor, the program runs the same
    print(b.numpy())

What has a kernel stays on the card: @ (2-D and batched 3-D, transposes without copies), + - * with a tensor of the
same shape, a row vector or a number, unary minus, relu, gelu, silu, tanh, sigmoid, softplus, mish, elu, leaky_relu,
sin, cos, softmax over the last axis, layer_norm / rms_norm over the last axis, sum and mean over an axis or all,
rows by index (gather), reshape. Everything else - any Apsi function that takes a tensor - sees the values as an
ordinary tensor (the copy to the processor is made for it), so code written for the processor runs unchanged.
"""
import math

import numpy as np

from .errors import RuntimeErr, ShapeError

_DEV = {}


def _device(name="cuda"):
    """(ops, label): Apsi's GPU kernels on a card, the exact emulator ("emulator"), or None for the processor."""
    key = str(name or "cuda").lower()
    if key in ("cpu", "processor", "numpy"):
        return None, "cpu"
    if key in _DEV:
        return _DEV[key]
    from . import gpu_train as GT
    if key in ("emulator", "emulate"):
        hit = (GT.Emulator(), "emulator")
    else:
        ops, why = GT.device(key if key.startswith("cuda") else "cuda")
        hit = (ops, getattr(ops, "name", key)) if ops is not None else (None, "cpu")
    _DEV[key] = hit
    return hit


class GTensor:
    """A float32 tensor in a card's memory (or, without a card, an ordinary array with the same methods)."""
    __array_priority__ = 1000

    def __init__(self, ops, buf, shape, host=None, label="cpu", trans=False):
        self.ops, self.buf, self.shape, self.host, self.device, self.trans = ops, buf, tuple(int(s) for s in shape), \
            host, label, trans

    # ------------------------------------------------------------ moving
    @staticmethod
    def of(x, device="cuda"):
        if isinstance(x, GTensor):
            return x
        from . import ops as aops
        a = np.ascontiguousarray(np.asarray(aops.raw(x), np.float32))
        ops, label = _device(device)
        if ops is None:
            return GTensor(None, None, a.shape, host=a, label="cpu")
        b = ops.alloc(max(1, a.size))
        if a.size:
            ops.put(b, a.reshape(-1))
        return GTensor(ops, b, a.shape, label=label)

    def _new(self, shape):
        b = self.ops.alloc(max(1, int(np.prod(shape))))
        return GTensor(self.ops, b, shape, label=self.device)

    def numpy(self):
        """The values as an ordinary tensor (float32)."""
        if self.ops is None:
            return self.host
        a = self.ops.get(self.buf, int(np.prod(self.shape))).reshape(self.shape[::-1] if self.trans else self.shape)
        return np.ascontiguousarray(a.T) if self.trans else a

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a.astype(dtype) if dtype is not None else a

    def cpu(self):
        return self.numpy()

    def __del__(self):
        try:
            if self.ops is not None and self.buf is not None and not self.__dict__.get("_view"):
                self.ops.free(self.buf)
        except Exception:                                 # noqa: BLE001 - interpreter shutting down
            pass

    def __repr__(self):
        return f"gpu_tensor({list(self.shape)}, on {self.device})"

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def ndim(self):
        return len(self.shape)

    def _v(self):
        return (self.buf, 0)

    def _host(self, fn, *others):
        """Anything without a kernel: on the processor, back to where self lives."""
        args = [o.numpy() if isinstance(o, GTensor) else o for o in others]
        out = fn(self.numpy(), *args)
        return GTensor.of(out, self.device) if self.ops is not None else GTensor(None, None, np.shape(out),
                                                                                  host=np.asarray(out, np.float32))

    def _plain(self):
        """A contiguous copy when self is a transposed view."""
        if not self.trans:
            return self
        return GTensor.of(self.numpy(), self.device)

    # ------------------------------------------------------------ arithmetic
    def _binary(self, other, how, swap=False):
        if self.ops is None:
            a, b = (self.host, np.asarray(other.numpy() if isinstance(other, GTensor) else other, np.float32))
            if swap:
                a, b = b, a
            r = {"+": np.add, "-": np.subtract, "*": np.multiply}[how](a, b)
            return GTensor(None, None, r.shape, host=r.astype(np.float32))
        x = self._plain()
        o = x.ops
        n = x.size
        if isinstance(other, GTensor) or hasattr(other, "shape") and np.ndim(other) > 0:
            y = GTensor.of(other, x.device)._plain() if not isinstance(other, GTensor) else other._plain()
            if y.ops is not o:
                y = GTensor.of(y.numpy(), x.device)
            if y.shape == x.shape:
                out = x._new(x.shape)
                if how == "+":
                    o.add3(out._v(), x._v(), y._v(), n)
                elif how == "*":
                    o.mul3(out._v(), x._v(), y._v(), n)
                else:                                       # x - y, or y - x
                    o.pull(out._v(), o, (y if swap else x)._v(), n)
                    o.axpy(out._v(), (x if swap else y)._v(), n, -1.0)
                return out
            if how in ("+", "-") and y.ndim == 1 and x.ndim >= 1 and y.shape[0] == x.shape[-1] and not swap:
                out = x._new(x.shape)
                o.pull(out._v(), o, x._v(), n)
                if how == "-":
                    neg = y._new(y.shape)
                    o.fill(neg._v(), y.size, 0.0)
                    o.axpy(neg._v(), y._v(), y.size, -1.0)
                    y = neg
                o.rowbias(out._v(), y._v(), n // x.shape[-1], x.shape[-1])
                return out
            fn = {"+": np.add, "-": np.subtract, "*": np.multiply}[how]
            return x._host(lambda a, b: fn(b, a) if swap else fn(a, b), y)
        s = float(other)
        out = x._new(x.shape)
        if how == "*":
            o.fill(out._v(), n, 0.0)
            o.axpy(out._v(), x._v(), n, s)
            return out
        if how == "-" and swap:                             # s - x
            o.fill(out._v(), n, 0.0)
            o.axpy(out._v(), x._v(), n, -1.0)
        else:
            o.pull(out._v(), o, x._v(), n)
            s = -s if how == "-" else s
        c = x._new((1,))
        o.fill(c._v(), 1, s)
        o.rowbias(out._v(), c._v(), n, 1)
        return out

    def __add__(self, o):
        return self._binary(o, "+")

    def __radd__(self, o):
        return self._binary(o, "+", True)

    def __sub__(self, o):
        return self._binary(o, "-")

    def __rsub__(self, o):
        return self._binary(o, "-", True)

    def __mul__(self, o):
        return self._binary(o, "*")

    def __rmul__(self, o):
        return self._binary(o, "*", True)

    def __truediv__(self, o):
        if not isinstance(o, GTensor) and np.ndim(o) == 0:
            return self * (1.0 / float(o))
        return self._host(lambda a, b: a / b, o)

    def __rtruediv__(self, o):
        return self._host(lambda a, b: b / a, o)

    def __neg__(self):
        return self * -1.0

    def __pow__(self, p):
        if p == 2:
            return self * self
        return self._host(lambda a: a ** p)

    # ------------------------------------------------------------ products
    @property
    def T(self):
        if self.ndim != 2:
            return self._host(lambda a: a.T)
        if self.ops is None:
            return GTensor(None, None, self.shape[::-1], host=np.ascontiguousarray(self.host.T))
        t = GTensor(self.ops, self.buf, self.shape[::-1], label=self.device, trans=not self.trans)
        t._owner = self                                     # the same memory: keep it alive
        t.__dict__["_view"] = True
        return t

    def _strides(self):
        """(rows, cols, row stride, col stride) of a 2-D tensor or view."""
        R, C = self.shape
        return (R, C, 1, R) if self.trans else (R, C, C, 1)

    def __matmul__(self, other):
        y = other if isinstance(other, GTensor) else GTensor.of(other, self.device)
        if self.ops is None:
            r = self.host @ np.asarray(y.numpy(), np.float32)
            return GTensor(None, None, r.shape, host=r.astype(np.float32))
        o = self.ops
        if y.ops is not o:
            y = GTensor.of(y.numpy(), self.device)
        if self.ndim == 2 and y.ndim == 2:
            M, K, sam, sak = self._strides()
            K2, N, sbk, sbn = y._strides()
            if K != K2:
                raise ShapeError(f"@: {list(self.shape)} @ {list(y.shape)}")
            out = self._new((M, N))
            o.gemm(self._v(), y._v(), out._v(), M, N, K, sam, sak, sbk, sbn, N, passes=3)
            return out
        if self.ndim == 3 and y.ndim in (2, 3):
            x = self._plain()
            B, M, K = x.shape
            if y.ndim == 2:                                 # [B, M, K] @ [K, N]: one product of B*M rows
                K2, N, sbk, sbn = y._strides()
                if K != K2:
                    raise ShapeError(f"@: {list(x.shape)} @ {list(y.shape)}")
                out = x._new((B, M, N))
                o.gemm(x._v(), y._v(), out._v(), B * M, N, K, K, 1, sbk, sbn, N, passes=3)
                return out
            y = y._plain()
            B2, K2, N = y.shape
            if B != B2 or K != K2:
                raise ShapeError(f"@: {list(x.shape)} @ {list(y.shape)}")
            out = x._new((B, M, N))
            o.gemm(x._v(), y._v(), out._v(), M, N, K, K, 1, N, 1, N, batch=(B, 1), sa=(M * K, 0), sb=(K * N, 0),
                   sc=(M * N, 0), passes=3)
            return out
        return self._host(lambda a, b: a @ b, y)

    def __rmatmul__(self, other):
        return GTensor.of(other, self.device) @ self

    # ------------------------------------------------------------ functions
    def _act(self, name):
        if self.ops is None:
            from .layers import _activation
            return GTensor(None, None, self.shape, host=np.asarray(_activation(name)(self.host), np.float32))
        from . import gpu_train_kernels as KS
        x = self._plain()
        out = x._new(x.shape)
        x.ops.act_f(x._v(), out._v(), x.size, KS.ACTS[name])
        return out

    def relu(self):
        return self._act("relu")

    def gelu(self):
        return self._act("gelu")

    def silu(self):
        return self._act("silu")

    def tanh(self):
        return self._act("tanh")

    def sigmoid(self):
        return self._act("sigmoid")

    def softplus(self):
        return self._act("softplus")

    def mish(self):
        return self._act("mish")

    def elu(self):
        return self._act("elu")

    def leaky_relu(self):
        return self._act("leaky_relu")

    def sin(self):
        return self._act("sin")

    def cos(self):
        return self._act("cos")

    def exp(self):
        return self._host(np.exp)

    def log(self):
        return self._host(np.log)

    def sqrt(self):
        return self._host(np.sqrt)

    def softmax(self, axis=-1):
        if axis not in (-1, self.ndim - 1) or self.ops is None:
            return self._host(lambda a: _np_softmax(a, axis))
        x = self._plain()
        F = x.shape[-1]
        out = x._new(x.shape)
        x.ops.softmax_f(x._v(), out._v(), x.size // F, F, 1, False)
        return out

    def layer_norm(self, gamma=None, beta=None, eps=1e-5, rms=False):
        F = self.shape[-1]
        if self.ops is None:
            a = self.host
            if rms:
                y = a / np.sqrt((a * a).mean(-1, keepdims=True) + eps)
            else:
                y = (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + eps)
            y = y * (1 if gamma is None else np.asarray(gamma)) + (0 if beta is None or rms else np.asarray(beta))
            return GTensor(None, None, y.shape, host=y.astype(np.float32))
        x = self._plain()
        g = GTensor.of(np.ones(F) if gamma is None else gamma, x.device)
        b = None if rms else GTensor.of(np.zeros(F) if beta is None else beta, x.device)
        M = x.size // F
        out, mu, rs = x._new(x.shape), x._new((M,)), x._new((M,))
        x.ops.ln_f(x._v(), g._v(), None if b is None else b._v(), out._v(), mu._v(), rs._v(), M, F, eps, rms)
        return out

    def rms_norm(self, gamma=None, eps=1e-6):
        return self.layer_norm(gamma, None, eps, rms=True)

    def sum(self, axis=None, keepdims=False):
        if self.ops is None:
            return _wrap_host(self.host.sum(axis=axis, keepdims=keepdims))
        x = self._plain()
        o = x.ops
        if axis is None:                                    # all: one product with a column of ones
            ones = GTensor.of(np.ones(x.size), x.device)
            out = x._new((1,))
            o.gemm(x._v(), ones._v(), out._v(), 1, 1, x.size, x.size, 1, 1, 1, 1, passes=3)
            return float(out.numpy()[0])
        ax = axis % x.ndim
        if ax == x.ndim - 1:
            F = x.shape[-1]
            ones = GTensor.of(np.ones(F), x.device)
            out = x._new(x.shape[:-1] + ((1,) if keepdims else ()))
            o.gemm(x._v(), ones._v(), out._v(), x.size // F, 1, F, F, 1, 1, 1, 1, passes=3)
            return out
        if ax == 0 and x.ndim == 2:
            R, C = x.shape
            out = x._new((1, C) if keepdims else (C,))
            o.fill(out._v(), C, 0.0)
            o.colsum(x._v(), None, None, None, None, out._v(), R, C, 0)
            return out
        return x._host(lambda a: a.sum(axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis, keepdims)
        n = self.size if axis is None else self.shape[axis % self.ndim]
        return s / n

    def rows(self, ids):
        """Rows of a 2-D tensor at integer indices (an embedding lookup)."""
        ids = np.asarray(ids).astype(np.int64)
        if self.ops is None:
            return GTensor(None, None, ids.shape + self.shape[1:], host=self.host[ids])
        x = self._plain()
        D = int(np.prod(x.shape[1:]))
        ib = x.ops.alloc(max(1, ids.size))
        x.ops.put(ib, ids.reshape(-1).astype(np.int32))
        out = x._new(ids.shape + x.shape[1:])
        x.ops.embed_f(x._v(), (ib, 0), out._v(), ids.size, D)
        x.ops.free(ib)
        return out

    def reshape(self, *shape):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else tuple(shape)
        if -1 in shape:
            k = shape.index(-1)
            rest = int(np.prod([s for s in shape if s != -1]))
            shape = shape[:k] + (self.size // max(1, rest),) + shape[k + 1:]
        if int(np.prod(shape)) != self.size:
            raise ShapeError(f"reshape {list(self.shape)} -> {list(shape)}")
        if self.ops is None:
            return GTensor(None, None, shape, host=self.host.reshape(shape))
        x = self._plain()
        out = x._new(shape)
        x.ops.pull(out._v(), x.ops, x._v(), x.size)
        return out


def _np_softmax(a, axis):
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _wrap_host(r):
    return float(r) if np.ndim(r) == 0 else GTensor(None, None, np.shape(r), host=np.asarray(r, np.float32))


def gpu_tensor(x, device="cuda"):
    """x on the card (Apsi's own kernels); on the processor when there is no card. device: "cuda", "cuda:1", "cpu"."""
    return GTensor.of(x, device)


def gpu_device(device="cuda"):
    """Where gpu_tensor puts tensors: the card's name, or "cpu"."""
    ops, label = _device(device)
    return label if ops is not None else "cpu"


def names():
    return dict(gpu_tensor=gpu_tensor, gpu_device=gpu_device)
