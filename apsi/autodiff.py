"""Reverse-mode automatic differentiation with higher-order support.

Every differentiable op records its inputs and a vector-Jacobian product (vjp)
written in terms of these same ops. A normal backward pass runs the vjps on raw
numpy values (fast). With create_graph=True the vjps run on Nodes, so the
gradient is itself differentiable — this gives Hessians, Laplacians of networks
(physics-informed nets), and grad(grad(f)) to any order.
"""
import builtins
from fractions import Fraction

import numpy as np

_OK = (int, float, np.ndarray, np.generic, Fraction)
try:                                   # GPU arrays (CuPy) take part in autodiff too
    import importlib.util as _ilu
    if _ilu.find_spec("cupy") is not None:
        import cupy as _cp
        _OK = _OK + (_cp.ndarray,)
except Exception:                      # CuPy installed without a working CUDA driver
    pass


class HigherOrderError(Exception):
    pass


class Node:
    __array_ufunc__ = None
    __array_priority__ = 999
    __slots__ = ("value", "inputs", "vjp", "raw_only", "grad")

    def __init__(self, value, inputs=(), vjp=None, raw_only=False):
        self.value = value
        self.inputs = inputs
        self.vjp = vjp
        self.raw_only = raw_only
        self.grad = None

    # arithmetic -------------------------------------------------------
    def __add__(self, o): return add(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __radd__(self, o): return add(o, self) if isinstance(o, _OK) else NotImplemented
    def __sub__(self, o): return sub(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __rsub__(self, o): return sub(o, self) if isinstance(o, _OK) else NotImplemented
    def __mul__(self, o): return mul(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __rmul__(self, o): return mul(o, self) if isinstance(o, _OK) else NotImplemented
    def __truediv__(self, o): return div(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __rtruediv__(self, o): return div(o, self) if isinstance(o, _OK) else NotImplemented
    def __pow__(self, o): return power(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __rpow__(self, o): return power(o, self) if isinstance(o, _OK) else NotImplemented
    def __matmul__(self, o): return matmul(self, o) if isinstance(o, (Node,) + _OK) else NotImplemented
    def __rmatmul__(self, o): return matmul(o, self) if isinstance(o, _OK) else NotImplemented
    def __neg__(self): return neg(self)
    def __pos__(self): return self
    def __abs__(self): return abs_(self)
    def __getitem__(self, idx): return getitem(self, idx)

    # comparisons work on values (no gradient)
    def __lt__(self, o): return self.value < _v(o)
    def __le__(self, o): return self.value <= _v(o)
    def __gt__(self, o): return self.value > _v(o)
    def __ge__(self, o): return self.value >= _v(o)
    def __eq__(self, o): return self.value == _v(o)
    def __ne__(self, o): return self.value != _v(o)
    __hash__ = object.__hash__

    def __len__(self): return len(self.value)
    def __bool__(self): return bool(self.value)
    def __float__(self): return float(self.value)

    @property
    def shape(self): return tuple(np.shape(self.value))

    @property
    def ndim(self): return np.ndim(self.value)

    @property
    def size(self): return int(np.size(self.value))

    @property
    def T(self): return transpose(self)

    def __repr__(self): return f"Node({self.value!r})"


def _v(x):
    if isinstance(x, Node):
        return x.value
    if isinstance(x, Fraction):
        return float(x)
    return x


def _op(fwd, vjp, inputs, raw_only=False):
    vals = [_v(i) for i in inputs]
    out = fwd(*vals)
    if not builtins.any(isinstance(i, Node) for i in inputs):
        return out
    return Node(out, tuple(inputs), vjp, raw_only)


def const(x):
    """Detach from the graph (stop gradient)."""
    return _v(x)


def ident(x):
    """A fresh differentiable leaf that stays connected to x's graph."""
    return Node(_v(x), (x,), lambda g, out, need, a: (g,))


# ------------------------------------------------------------- shape helpers
def unbroadcast(g, shape):
    shape = tuple(shape)
    gs = np.shape(_v(g))
    if gs == shape:
        return g
    if shape == ():
        return sum(g)
    extra = len(gs) - len(shape)
    if extra > 0:
        g = sum(g, axis=tuple(range(extra)))
        gs = np.shape(_v(g))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and gs[i] != 1)
    if axes:
        g = sum(g, axis=axes, keepdims=True)
    return g


def broadcast_to(x, shape):
    shape = tuple(shape)
    xs = np.shape(_v(x))
    return _op(lambda a: np.broadcast_to(a, shape).copy() if shape else a,
               lambda g, out, need, a: (unbroadcast(g, xs),), [x])


# ------------------------------------------------------------- binary ops
def add(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    return _op(lambda x, y: x + y,
               lambda g, out, need, x, y: (unbroadcast(g, sa) if need[0] else None,
                                           unbroadcast(g, sb) if need[1] else None), [a, b])


def sub(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    return _op(lambda x, y: x - y,
               lambda g, out, need, x, y: (unbroadcast(g, sa) if need[0] else None,
                                           unbroadcast(neg(g), sb) if need[1] else None), [a, b])


def mul(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    return _op(lambda x, y: x * y,
               lambda g, out, need, x, y: (unbroadcast(mul(g, y), sa) if need[0] else None,
                                           unbroadcast(mul(g, x), sb) if need[1] else None), [a, b])


def div(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    return _op(lambda x, y: x / y,
               lambda g, out, need, x, y: (unbroadcast(div(g, y), sa) if need[0] else None,
                                           unbroadcast(neg(div(mul(g, out), y)), sb) if need[1] else None),
               [a, b])


def power(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))

    def vjp(g, out, need, x, y):
        ga = unbroadcast(mul(g, mul(y, power(x, sub(y, 1)))), sa) if need[0] else None
        gb = None
        if need[1]:
            safe = where(np.asarray(_v(x)) > 0, x, 1.0)
            gb = unbroadcast(mul(g, mul(out, log(safe))), sb)
        return ga, gb
    return _op(lambda x, y: x ** y, vjp, [a, b])


def _swap(x):
    n = np.ndim(_v(x))
    axes = list(range(n))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return transpose(x, tuple(axes))


def matmul(a, b):
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    na, nb = len(sa), len(sb)

    def vjp(g, out, need, x, y):
        ga = gb = None
        if na == 1 and nb == 1:
            ga = mul(g, y) if need[0] else None
            gb = mul(g, x) if need[1] else None
        elif nb == 1:
            if need[0]:
                ga = mul(reshape(g, np.shape(_v(g)) + (1,)), y)
            if need[1]:
                r = matmul(_swap(x), reshape(g, np.shape(_v(g)) + (1,)))
                gb = unbroadcast(reshape(r, np.shape(_v(r))[:-1]), sb)
        elif na == 1:
            if need[0]:
                ga = matmul(y, g)
            if need[1]:
                gb = mul(reshape(x, (sa[0], 1)), reshape(g, (1, -1)))
        else:
            if need[0]:
                ga = unbroadcast(matmul(g, _swap(y)), sa)
            if need[1]:
                gb = unbroadcast(matmul(_swap(x), g), sb)
        return ga, gb
    return _op(lambda x, y: np.matmul(x, y), vjp, [a, b])


def where(cond, a, b):
    cond = np.asarray(_v(cond))
    sa, sb = np.shape(_v(a)), np.shape(_v(b))
    return _op(lambda x, y: np.where(cond, x, y),
               lambda g, out, need, x, y: (unbroadcast(where(cond, g, 0.0), sa) if need[0] else None,
                                           unbroadcast(where(cond, 0.0, g), sb) if need[1] else None),
               [a, b])


# ------------------------------------------------------------- unary ops
def _un(fwd, dfn):
    def f(x):
        return _op(fwd, lambda g, out, need, a: (dfn(g, out, a),), [x])
    return f


neg = _un(lambda a: -a, lambda g, out, a: neg(g))
exp = _un(np.exp, lambda g, out, a: mul(g, out))
log = _un(np.log, lambda g, out, a: div(g, a))
log10 = _un(np.log10, lambda g, out, a: div(g, mul(a, np.log(10.0))))
sin = _un(np.sin, lambda g, out, a: mul(g, cos(a)))
cos = _un(np.cos, lambda g, out, a: neg(mul(g, sin(a))))
tan = _un(np.tan, lambda g, out, a: div(g, power(cos(a), 2)))
asin = _un(np.arcsin, lambda g, out, a: div(g, sqrt(sub(1.0, mul(a, a)))))
acos = _un(np.arccos, lambda g, out, a: neg(div(g, sqrt(sub(1.0, mul(a, a))))))
atan = _un(np.arctan, lambda g, out, a: div(g, add(1.0, mul(a, a))))
sinh = _un(np.sinh, lambda g, out, a: mul(g, cosh(a)))
cosh = _un(np.cosh, lambda g, out, a: mul(g, sinh(a)))
tanh = _un(np.tanh, lambda g, out, a: mul(g, sub(1.0, mul(out, out))))
sqrt = _un(np.sqrt, lambda g, out, a: div(mul(g, 0.5), out))
abs_ = _un(np.abs, lambda g, out, a: mul(g, np.sign(_v(a))))
relu = _un(lambda a: np.maximum(a, 0), lambda g, out, a: mul(g, (np.asarray(_v(a)) > 0).astype(float)))
sigmoid = _un(lambda a: 1.0 / (1.0 + np.exp(-a)), lambda g, out, a: mul(g, mul(out, sub(1.0, out))))
softplus = _un(lambda a: np.logaddexp(0, a), lambda g, out, a: mul(g, sigmoid(a)))


# ------------------------------------------------------------- reductions
def _norm_axes(axis, nd):
    if axis is None:
        return tuple(range(nd))
    if isinstance(axis, (tuple, list)):
        return tuple(int(a) % nd for a in axis)
    return (int(axis) % nd,)


def _kd_shape(shape, axes):
    return tuple(1 if i in axes else s for i, s in enumerate(shape))


def sum(x, axis=None, keepdims=False):
    shape = np.shape(_v(x))
    axes = _norm_axes(axis, len(shape))
    kd = _kd_shape(shape, axes)
    ax = tuple(axes) if axis is not None else None
    return _op(lambda a: np.sum(a, axis=ax, keepdims=keepdims),
               lambda g, out, need, a: (broadcast_to(reshape(g, kd), shape),), [x])


def mean(x, axis=None, keepdims=False):
    shape = np.shape(_v(x))
    axes = _norm_axes(axis, len(shape))
    n = 1
    for i in axes:
        n *= shape[i]
    return div(sum(x, axis, keepdims), float(n))


def _extreme(x, axis, keepdims, fn):
    xv = np.asarray(_v(x))
    full = fn(xv, axis=axis, keepdims=True)
    mask = (xv == full).astype(float)
    mask /= mask.sum(axis=axis, keepdims=True)
    shape = xv.shape
    axes = _norm_axes(axis, len(shape))
    kd = _kd_shape(shape, axes)
    return _op(lambda a: fn(a, axis=axis, keepdims=keepdims),
               lambda g, out, need, a: (mul(broadcast_to(reshape(g, kd), shape), mask),), [x])


def max(x, axis=None, keepdims=False):
    return _extreme(x, axis, keepdims, np.max)


def min(x, axis=None, keepdims=False):
    return _extreme(x, axis, keepdims, np.min)


# ------------------------------------------------------------- shape ops
def transpose(x, axes=None):
    inv = None if axes is None else tuple(np.argsort(axes))
    return _op(lambda a: np.transpose(a, axes),
               lambda g, out, need, a: (transpose(g, inv),), [x])


def reshape(x, shape):
    orig = np.shape(_v(x))
    return _op(lambda a: np.reshape(a, shape),
               lambda g, out, need, a: (reshape(g, orig),), [x])


def _is_advanced(idx):
    items = idx if isinstance(idx, tuple) else (idx,)
    return builtins.any(isinstance(i, (list, np.ndarray)) for i in items)


def scatter(g, idx, shape):
    """Inverse of getitem: place g into zeros(shape) at idx (adding duplicates)."""
    def fwd(gv):
        z = np.zeros(shape)
        if _is_advanced(idx):
            np.add.at(z, idx, gv)
        else:
            z[idx] += gv
        return z
    return _op(fwd, lambda h, out, need, a: (getitem(h, idx),), [g])


def getitem(x, idx):
    shape = np.shape(_v(x))
    return _op(lambda a: np.asarray(a)[idx],
               lambda g, out, need, a: (scatter(g, idx, shape),), [x])


def roll(x, shift, axis=None):
    return _op(lambda a: np.roll(a, shift, axis),
               lambda g, out, need, a: (roll(g, -shift, axis) if not isinstance(shift, tuple)
                                        else roll(g, tuple(-s for s in shift), axis),), [x])


def softmax(x, axis=-1):
    def fwd(a):
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)
    return _op(fwd, lambda g, out, need, a:
               (mul(out, sub(g, sum(mul(g, out), axis=axis, keepdims=True))),), [x])


def log_softmax(x, axis=-1):
    def fwd(a):
        m = np.max(a, axis=axis, keepdims=True)
        return a - (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)))
    return _op(fwd, lambda g, out, need, a:
               (sub(g, mul(exp(out), sum(g, axis=axis, keepdims=True))),), [x])


def stack(items, axis=0):
    items = list(items)
    n_out = np.ndim(_v(items[0])) + 1
    ax = axis % n_out

    def vjp(g, out, need, *xs):
        res = []
        for k in range(len(xs)):
            if need[k]:
                idx = (slice(None),) * ax + (k,)
                res.append(getitem(g, idx))
            else:
                res.append(None)
        return tuple(res)
    return _op(lambda *vals: np.stack(vals, axis=axis), vjp, items)


def concat(items, axis=0):
    items = list(items)
    sizes = [np.shape(_v(i))[axis] for i in items]
    bounds = np.cumsum([0] + sizes)
    nd = np.ndim(_v(items[0]))
    ax = axis % nd

    def vjp(g, out, need, *xs):
        return tuple(getitem(g, (slice(None),) * ax + (slice(int(bounds[k]), int(bounds[k + 1])),))
                     if need[k] else None for k in range(len(xs)))
    return _op(lambda *vals: np.concatenate(vals, axis=axis), vjp, items)


# ------------------------------------------------------------- CNN ops (first order)
def _im2col(x, kh, kw, stride):
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    win = win[:, :, ::stride, ::stride]
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5))


def conv2d(x, w, stride=1, padding=0):
    """2-D convolution. x: (N, C, H, W), w: (O, C, kh, kw) -> (N, O, OH, OW)."""
    p = int(padding)

    def fwd(xv, wv):
        xv = np.asarray(xv, dtype=float)
        xp = np.pad(xv, ((0, 0), (0, 0), (p, p), (p, p))) if p else xv
        cols = _im2col(xp, wv.shape[2], wv.shape[3], stride)
        return np.tensordot(cols, wv, axes=([3, 4, 5], [1, 2, 3])).transpose(0, 3, 1, 2)

    def vjp(g, out, need, xv, wv):
        xv = np.asarray(xv, dtype=float)
        xp = np.pad(xv, ((0, 0), (0, 0), (p, p), (p, p))) if p else xv
        O, C, kh, kw = wv.shape
        cols = _im2col(xp, kh, kw, stride)
        OH, OW = cols.shape[1:3]
        gx = gw = None
        if need[0]:
            gcols = np.tensordot(g.transpose(0, 2, 3, 1), wv, axes=([3], [0]))
            dxp = np.zeros_like(xp)
            for i in range(kh):
                for j in range(kw):
                    dxp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += \
                        gcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
            gx = dxp[:, :, p:p + xv.shape[2], p:p + xv.shape[3]] if p else dxp
        if need[1]:
            gw = np.tensordot(g.transpose(1, 0, 2, 3), cols, axes=([1, 2, 3], [0, 1, 2]))
        return gx, gw
    return _op(fwd, vjp, [x, w], raw_only=True)


def max_pool2d(x, k=2):
    xv0 = np.asarray(_v(x), dtype=float)
    N, C, H, W = xv0.shape
    H2, W2 = H // k, W // k
    xr = xv0[:, :, :H2 * k, :W2 * k].reshape(N, C, H2, k, W2, k)
    out0 = xr.max(axis=(3, 5))
    mask = (xr == out0[:, :, :, None, :, None])
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)

    def vjp(g, out, need, a):
        full = np.zeros_like(xv0)
        full[:, :, :H2 * k, :W2 * k] = (mask * g[:, :, :, None, :, None]).reshape(N, C, H2 * k, W2 * k)
        return (full,)
    return _op(lambda a: out0, vjp, [x], raw_only=True)


# ------------------------------------------------------------- backward
def _topo(root):
    order, seen, stack_ = [], set(), [(root, False)]
    while stack_:
        node, done = stack_.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack_.append((node, True))
        for p in node.inputs:
            if isinstance(p, Node) and id(p) not in seen:
                stack_.append((p, False))
    return order


def gradients(root, create_graph=False):
    """Return {id(node): d root / d node} for every node in root's graph."""
    order = _topo(root)
    seed = np.ones_like(root.value, dtype=float) if np.ndim(root.value) else 1.0
    grads = {id(root): seed}
    for node in reversed(order):
        g = grads.get(id(node))
        if g is None or node.vjp is None:
            continue
        need = tuple(isinstance(i, Node) for i in node.inputs)
        if create_graph and not node.raw_only:
            gs = node.vjp(g, node, need, *node.inputs)
        else:
            if create_graph and node.raw_only:
                raise HigherOrderError("second-order derivatives through conv2d/max_pool2d are not supported yet")
            gs = node.vjp(_v(g), node.value, need, *[_v(i) for i in node.inputs])
        for inp, gi in zip(node.inputs, gs):
            if gi is not None and isinstance(inp, Node):
                k = id(inp)
                grads[k] = gi if k not in grads else add(grads[k], gi)
    return grads


def backward(root):
    """Compatibility helper: store gradients in .grad of every node."""
    grads = gradients(root)
    for node in _topo(root):
        node.grad = grads.get(id(node))
