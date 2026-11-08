"""Generic math/tensor operations that work uniformly on numbers, tensors,
traced values (autodiff Nodes) and physical Quantities."""
import builtins
import math
from fractions import Fraction

import numpy as np

from . import autodiff as ad
from .autodiff import Node
from .errors import UnitError, ShapeError, RuntimeErr
from .units import Quantity, make, DIMLESS, dim_name, Dim, _is_zero, parse_unit
from .values import RangeVal, Dtype
from .uncertain import Uncertain, DERIVS

NUMERIC = (int, float, Fraction, np.number, np.ndarray, Node, bool, np.bool_, Uncertain)


def py(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def raw(x):
    """Strip Quantity and Node wrappers."""
    if isinstance(x, Quantity):
        x = x.v
    if isinstance(x, Node):
        x = x.value
    return x


def dim_of(x):
    if isinstance(x, Quantity):
        return x.dim
    if isinstance(x, NUMERIC):
        return DIMLESS
    return None


def _np_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, Dtype):
        return dtype.np
    return np.dtype(dtype)


def _arr(x):
    if isinstance(x, (list, tuple, RangeVal)):
        return tensor(list(x) if isinstance(x, RangeVal) else x)
    return x


# ------------------------------------------------------------- elementwise
def _dimless_unary(name, npf, adf):
    def f(x):
        x = _arr(x)
        if isinstance(x, Quantity):
            raise UnitError(f"{name}() needs a dimensionless argument, but got {dim_name(x.dim)}; "
                            f"divide by a reference value first (e.g. x / 1[m])")
        if isinstance(x, Node):
            return adf(x)
        if isinstance(x, Uncertain):
            if name not in DERIVS:
                raise RuntimeErr(f"{name}() does not support uncertain values yet")
            return x.apply(*DERIVS[name])
        if isinstance(x, Fraction):
            x = float(x)
        return py(npf(x))
    f.__name__ = name
    return f


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


exp = _dimless_unary("exp", np.exp, ad.exp)
log = _dimless_unary("log", np.log, ad.log)
log10 = _dimless_unary("log10", np.log10, ad.log10)
sin = _dimless_unary("sin", np.sin, ad.sin)
cos = _dimless_unary("cos", np.cos, ad.cos)
tan = _dimless_unary("tan", np.tan, ad.tan)
asin = _dimless_unary("asin", np.arcsin, ad.asin)
acos = _dimless_unary("acos", np.arccos, ad.acos)
atan = _dimless_unary("atan", np.arctan, ad.atan)
sinh = _dimless_unary("sinh", np.sinh, ad.sinh)
cosh = _dimless_unary("cosh", np.cosh, ad.cosh)
tanh = _dimless_unary("tanh", np.tanh, ad.tanh)
sigmoid = _dimless_unary("sigmoid", _np_sigmoid, ad.sigmoid)
softplus = _dimless_unary("softplus", lambda x: np.logaddexp(0, x), ad.softplus)


def log2(x):
    return log(x) / math.log(2.0)


def gelu(x):
    x = _arr(x)
    return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def sqrt(x):
    x = _arr(x)
    if isinstance(x, Quantity):
        return make(sqrt(x.v), x.dim ** Fraction(1, 2))
    if isinstance(x, Node):
        return ad.sqrt(x)
    if isinstance(x, Uncertain):
        return x.apply(*DERIVS["sqrt"])
    if isinstance(x, Fraction):
        x = float(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return math.sqrt(x) if x >= 0 else float("nan")
    return py(np.sqrt(x))


def cbrt(x):
    x = _arr(x)
    if isinstance(x, Quantity):
        return make(cbrt(x.v), x.dim ** Fraction(1, 3))
    if isinstance(x, Node):
        return x ** (1.0 / 3.0)
    return py(np.cbrt(x))


def abs_(x):
    x = _arr(x)
    if isinstance(x, Quantity):
        return Quantity(abs_(x.v), x.dim)
    if isinstance(x, Node):
        return ad.abs_(x)
    if isinstance(x, np.ndarray):
        return np.abs(x)
    return builtins.abs(x)


def relu(x):
    x = _arr(x)
    if isinstance(x, Quantity):
        return Quantity(relu(x.v), x.dim)
    if isinstance(x, Node):
        return ad.relu(x)
    return py(np.maximum(x, 0))


def softmax(x, axis=-1):
    x = _arr(x)
    if isinstance(x, Quantity):
        raise UnitError("softmax() needs dimensionless logits")
    if isinstance(x, Node):
        return ad.softmax(x, axis)
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = _arr(x)
    if isinstance(x, Quantity):
        raise UnitError("log_softmax() needs dimensionless logits")
    if isinstance(x, Node):
        return ad.log_softmax(x, axis)
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    return x - (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)))


def floor(x):
    q = isinstance(x, Quantity)
    r = np.floor(raw(x))
    r = py(r)
    if isinstance(r, float) and r.is_integer():
        r = int(r)
    return Quantity(r, x.dim) if q else r


def ceil(x):
    q = isinstance(x, Quantity)
    r = py(np.ceil(raw(x)))
    if isinstance(r, float) and r.is_integer():
        r = int(r)
    return Quantity(r, x.dim) if q else r


def round_(x, digits=0):
    if isinstance(x, Quantity):
        return Quantity(round_(x.v, digits), x.dim)
    r = py(np.round(raw(x), digits))
    if digits == 0 and isinstance(r, float) and r.is_integer():
        return int(r)
    return r


def atan2(y, x):
    if isinstance(y, Quantity) or isinstance(x, Quantity):
        if dim_of(y) != dim_of(x):
            raise UnitError(f"atan2() needs both arguments in the same units, got "
                            f"{dim_name(dim_of(y))} and {dim_name(dim_of(x))}")
    return py(np.arctan2(raw(y), raw(x)))


# ------------------------------------------------------------- reductions
def _reduce(name, npf, adf):
    def f(x, axis=None, keepdims=False):
        if isinstance(x, (list, tuple)) and axis is None and not keepdims:
            items = list(x)
            if name == "sum":
                total = 0
                for it in items:
                    total = total + it
                return total
            if not items:
                raise RuntimeErr("mean() of an empty list")
            total = 0
            for it in items:
                total = total + it
            return total / len(items)
        x = _arr(x)
        if isinstance(x, Quantity):
            return Quantity(f(x.v, axis, keepdims), x.dim)
        if isinstance(x, Node):
            return adf(x, axis=axis, keepdims=keepdims)
        if name == "sum" and not keepdims and isinstance(x, np.ndarray) and axis is not None:
            r = _small_axis_sum(x, axis)
            if r is not None:
                return r
        return py(npf(x, axis=axis, keepdims=keepdims))
    f.__name__ = name
    return f


def _small_axis_sum(x, axis):
    """Reducing a short axis (e.g. xyz components) with numpy's generic reduce is slow;
    unrolled slice additions are several times faster and give identical results."""
    nd = x.ndim
    if not isinstance(axis, (int, np.integer)) or nd < 2 or x.dtype.kind not in "fc":
        return None
    ax = axis % nd
    k = x.shape[ax]
    if k > 8 or k == 0:
        return None
    idx = [slice(None)] * nd
    idx[ax] = 0
    out = x[tuple(idx)].copy()
    for i in range(1, k):
        idx[ax] = i
        out += x[tuple(idx)]
    return out


sum_ = _reduce("sum", np.sum, ad.sum)
mean = _reduce("mean", np.mean, ad.mean)


def _extreme(name, npf, adf, pyf):
    def f(*args, axis=None, keepdims=False):
        if len(args) > 1:
            return pyf(args)
        if not args:
            raise RuntimeErr(f"{name}() needs at least one argument")
        x = args[0]
        if isinstance(x, (list, tuple, RangeVal)):
            items = list(x)
            if any(isinstance(i, (np.ndarray, Node)) for i in items):
                x = tensor(items)
            else:
                return pyf(items)
        if isinstance(x, Quantity):
            return Quantity(f(x.v, axis=axis, keepdims=keepdims), x.dim)
        if isinstance(x, Node):
            return adf(x, axis=axis, keepdims=keepdims)
        if isinstance(x, np.ndarray):
            return py(npf(x, axis=axis, keepdims=keepdims))
        return x
    f.__name__ = name
    return f


max_ = _extreme("max", np.max, ad.max, builtins.max)
min_ = _extreme("min", np.min, ad.min, builtins.min)


def var(x, axis=None):
    x = _arr(x)
    m = mean(x, axis=axis, keepdims=True)
    d = x - m
    return mean(d * d, axis=axis)


def std(x, axis=None):
    return sqrt(var(x, axis))


def prod(x, axis=None):
    x = _arr(x)
    if isinstance(x, Quantity):
        n = np.size(raw(x)) if axis is None else np.shape(raw(x))[axis]
        return make(py(np.prod(x.v, axis=axis)), x.dim ** n)
    return py(np.prod(raw(x), axis=axis))


def cumsum(x, axis=None):
    x = _arr(x)
    if isinstance(x, Quantity):
        return Quantity(np.cumsum(raw(x.v), axis=axis), x.dim)
    return np.cumsum(raw(x), axis=axis)


def argmax(x, axis=None):
    return py(np.argmax(raw(_arr(x)), axis=axis))


def argmin(x, axis=None):
    return py(np.argmin(raw(_arr(x)), axis=axis))


# ------------------------------------------------------------- construction
def tensor(x, dtype=None):
    dt = _np_dtype(dtype)
    if isinstance(x, Quantity):
        return x
    if isinstance(x, Node):
        return x
    if isinstance(x, np.ndarray):
        return x.astype(dt) if dt is not None else x
    if isinstance(x, RangeVal):
        x = list(x)
    if isinstance(x, (list, tuple)):
        elems = [tensor(e) if isinstance(e, (list, tuple)) else e for e in x]
        if any(isinstance(e, (Quantity, Node)) for e in elems):
            return stack(elems)
        a = np.array(elems, dtype=dt)
        if a.dtype == object:
            raise ShapeError("tensor() needs a rectangular list of numbers")
        return a
    if isinstance(x, Fraction):
        x = float(x)
    return np.array(x, dtype=dt)


def stack(items, axis=0):
    items = [tensor(i) if isinstance(i, (list, tuple)) else i for i in items]
    if not items:
        raise ShapeError("stack() needs at least one item")
    qs = [i for i in items if isinstance(i, Quantity)]
    if qs:
        d = qs[0].dim
        vals = []
        for i in items:
            if isinstance(i, Quantity):
                if i.dim != d:
                    raise UnitError(f"cannot stack {dim_name(d)} with {dim_name(i.dim)}")
                vals.append(i.v)
            elif _is_zero(i):
                vals.append(i)
            else:
                raise UnitError(f"cannot stack {dim_name(d)} with a dimensionless value")
        return Quantity(stack(vals, axis), d)
    if any(isinstance(i, Node) for i in items):
        return ad.stack(items, axis)
    return np.stack([np.asarray(i, dtype=float) if isinstance(i, Fraction) else np.asarray(i)
                     for i in items], axis=axis)


def concat(items, axis=0):
    items = [tensor(i) if isinstance(i, (list, tuple)) else i for i in items]
    qs = [i for i in items if isinstance(i, Quantity)]
    if qs:
        d = qs[0].dim
        if any(not isinstance(i, Quantity) or i.dim != d for i in items):
            raise UnitError("concat() needs all parts in the same units")
        return Quantity(concat([i.v for i in items], axis), d)
    if any(isinstance(i, Node) for i in items):
        return ad.concat(items, axis)
    return np.concatenate([np.asarray(i) for i in items], axis=axis)


def _shape_args(shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    for s in shape:
        if not isinstance(s, (int, np.integer)) or isinstance(s, bool):
            raise ShapeError(f"shape entries must be integers, got {s!r}")
    return tuple(int(s) for s in shape)


def zeros(*shape, dtype=None):
    return np.zeros(_shape_args(shape), dtype=_np_dtype(dtype) or float)


def ones(*shape, dtype=None):
    return np.ones(_shape_args(shape), dtype=_np_dtype(dtype) or float)


def full(shape, value):
    shape = _shape_args(shape if isinstance(shape, (list, tuple)) else (shape,))
    if isinstance(value, Quantity):
        return Quantity(np.full(shape, value.v, dtype=float), value.dim)
    return np.full(shape, value, dtype=float)


def zeros_like(x):
    if isinstance(x, Quantity):
        return Quantity(np.zeros_like(raw(x.v), dtype=float), x.dim)
    return np.zeros_like(raw(x), dtype=float)


def ones_like(x):
    if isinstance(x, Quantity):
        return Quantity(np.ones_like(raw(x.v), dtype=float), x.dim)
    return np.ones_like(raw(x), dtype=float)


def eye(n):
    return np.eye(int(n))


def _same_dims(*xs):
    ds = {dim_of(x) for x in xs if isinstance(x, Quantity)}
    if not ds:
        return None
    if len(ds) > 1 or any(not isinstance(x, Quantity) and not _is_zero(x) for x in xs):
        raise UnitError("all endpoints must have the same units")
    return ds.pop()


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    d = _same_dims(start, stop, step) if any(isinstance(x, Quantity) for x in (start, stop, step)) else None
    r = np.arange(raw(start), raw(stop), raw(step))
    return Quantity(r, d) if d is not None else r


def linspace(start, stop, n=50):
    d = _same_dims(start, stop) if any(isinstance(x, Quantity) for x in (start, stop)) else None
    r = np.linspace(raw(start), raw(stop), int(n))
    return Quantity(r, d) if d is not None else r


def meshgrid(x, y):
    xx, yy = np.meshgrid(raw(x), raw(y), indexing="ij")
    if isinstance(x, Quantity):
        xx = Quantity(xx, x.dim)
    if isinstance(y, Quantity):
        yy = Quantity(yy, y.dim)
    return (xx, yy)


_RNG = [np.random.default_rng(0)]


def seed(n):
    _RNG[0] = np.random.default_rng(int(n))


def _rng(s):
    return np.random.default_rng(int(s)) if s is not None else _RNG[0]


def randn(*shape, seed=None):
    return _rng(seed).standard_normal(_shape_args(shape)) if shape else py(_rng(seed).standard_normal())


def rand(*shape, seed=None):
    return _rng(seed).random(_shape_args(shape)) if shape else py(_rng(seed).random())


def randint(lo, hi, *shape, seed=None):
    if shape:
        return _rng(seed).integers(lo, hi, _shape_args(shape))
    return py(_rng(seed).integers(lo, hi))


def permutation(n, seed=None):
    return _rng(seed).permutation(int(n))


# ------------------------------------------------------------- shape ops
def shape(x):
    return tuple(np.shape(raw(_arr(x))))


def ndim(x):
    return len(shape(x))


def reshape(x, *shp):
    shp = _shape_args(shp) if not (len(shp) == 1 and isinstance(shp[0], int)) else (shp[0],)
    x = _arr(x)
    if isinstance(x, Quantity):
        return Quantity(reshape(x.v, *shp), x.dim)
    if isinstance(x, Node):
        return ad.reshape(x, shp)
    return np.reshape(x, shp)


def flatten(x):
    return reshape(x, -1)


def transpose(x, axes=None):
    x = _arr(x)
    if isinstance(x, Quantity):
        return Quantity(transpose(x.v, axes), x.dim)
    if isinstance(x, Node):
        return ad.transpose(x, axes)
    return np.transpose(x, axes)


def roll(x, shift, axis=None):
    if isinstance(x, Quantity):
        return Quantity(roll(x.v, shift, axis), x.dim)
    if isinstance(x, Node):
        return ad.roll(x, shift, axis)
    return np.roll(x, shift, axis)


def where(cond, a, b):
    cond = raw(cond)
    if isinstance(a, Quantity) or isinstance(b, Quantity):
        d = _same_dims(a, b)
        return Quantity(where(cond, raw(a) if not isinstance(a, Quantity) else a.v,
                              raw(b) if not isinstance(b, Quantity) else b.v), d)
    if isinstance(a, Node) or isinstance(b, Node):
        return ad.where(cond, a, b)
    return np.where(cond, a, b)


def clip(x, lo, hi):
    if isinstance(x, Quantity):
        return Quantity(np.clip(raw(x.v), raw(lo), raw(hi)), x.dim)
    return py(np.clip(raw(x), lo, hi))


def sort(x):
    if isinstance(x, list):
        return sorted(x)
    if isinstance(x, Quantity):
        return Quantity(np.sort(raw(x.v)), x.dim)
    return np.sort(raw(x))


# ------------------------------------------------------------- linear algebra
def matmul(a, b):
    return _arr(a) @ _arr(b)


def dot(a, b):
    a, b = _arr(a), _arr(b)
    if ndim(a) > 1 or ndim(b) > 1:
        return a @ b
    return sum_(a * b)


def outer(a, b):
    a, b = _arr(a), _arr(b)
    return reshape(a, -1, 1) * reshape(b, 1, -1)


def cross(a, b):
    a, b = _arr(a), _arr(b)
    if shape(a)[-1] != 3 or shape(b)[-1] != 3:
        raise ShapeError("cross() needs 3-component vectors")
    E = Ellipsis
    a0, a1, a2 = a[E, 0], a[E, 1], a[E, 2]
    b0, b1, b2 = b[E, 0], b[E, 1], b[E, 2]
    return stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def norm(x, axis=None):
    x = _arr(x)
    return sqrt(sum_(x * x, axis=axis))


def _linalg_raw(name, x):
    if isinstance(x, Node) or (isinstance(x, Quantity) and isinstance(x.v, Node)):
        raise RuntimeErr(f"{name}() is not differentiable yet")
    return raw(_arr(x))


def inv(a):
    r = np.linalg.inv(_linalg_raw("inv", a))
    return Quantity(r, DIMLESS / a.dim) if isinstance(a, Quantity) else r


def det(a):
    r = py(np.linalg.det(_linalg_raw("det", a)))
    return make(r, a.dim ** shape(a)[0]) if isinstance(a, Quantity) else r


def solve(a, b):
    r = np.linalg.solve(_linalg_raw("solve", a), _linalg_raw("solve", b))
    return make(r, dim_of(b) / dim_of(a))


def eigh(a):
    vals, vecs = np.linalg.eigh(_linalg_raw("eigh", a))
    return (make(vals, dim_of(a)), vecs)


def eig(a):
    vals, vecs = np.linalg.eig(_linalg_raw("eig", a))
    return (make(vals, dim_of(a)), vecs)


def svd(a):
    u, s, vt = np.linalg.svd(_linalg_raw("svd", a), full_matrices=False)
    return (u, make(s, dim_of(a)), vt)


def trace(a):
    return make(py(np.trace(raw(a))), dim_of(a))


# ------------------------------------------------------------- physics fields
def laplacian(u, dx, boundary="periodic"):
    """Discrete Laplacian (∇²u) on a regular grid with spacing dx."""
    n = ndim(u)
    total = 0
    for ax in range(n):
        total = total + (roll(u, 1, ax) + roll(u, -1, ax) - 2 * u)
    lap = total / (dx * dx)
    if boundary == "fixed":
        mask = np.zeros(shape(u))
        mask[tuple(slice(1, -1) for _ in range(n))] = 1.0
        lap = lap * mask
    elif boundary != "periodic":
        raise RuntimeErr("boundary must be \"periodic\" or \"fixed\"")
    return lap


def gradient(u, dx, axis=0):
    """Central-difference derivative of a field along one axis (periodic)."""
    return (roll(u, -1, axis) - roll(u, 1, axis)) / (2 * dx)


# ------------------------------------------------------------- comparisons
def approx(a, b, rel=1e-6, atol=1e-12):
    if isinstance(a, Quantity) or isinstance(b, Quantity):
        da, db = dim_of(a), dim_of(b)
        if da != db and not (_is_zero(a) or _is_zero(b)):
            raise UnitError(f"cannot compare {dim_name(da)} and {dim_name(db)}")
    ra, rb = raw(a), raw(b)
    if isinstance(ra, Uncertain):
        ra = ra.x
    if isinstance(rb, Uncertain):
        rb = rb.x
    if isinstance(ra, Fraction):
        ra = float(ra)
    if isinstance(rb, Fraction):
        rb = float(rb)
    return bool(np.allclose(ra, rb, rtol=rel, atol=atol))


# ------------------------------------------------------------- uncertainty
def plusminus(value, sigma):
    """value ± sigma  -> an independent measurement with standard uncertainty sigma."""
    if isinstance(value, Quantity) or isinstance(sigma, Quantity):
        if dim_of(value) != dim_of(sigma):
            raise UnitError(f"value and uncertainty must have the same units: "
                            f"{dim_name(dim_of(value))} ± {dim_name(dim_of(sigma))}")
        return Quantity(Uncertain.measure(value.v, sigma.v), value.dim, value.disp)
    return Uncertain.measure(raw(value), raw(sigma))


def nominal(x):
    if isinstance(x, Quantity):
        return Quantity(nominal(x.v), x.dim, x.disp)
    return x.x if isinstance(x, Uncertain) else x


def sigma(x):
    if isinstance(x, Quantity):
        return Quantity(sigma(x.v), x.dim, x.disp)
    return x.sigma if isinstance(x, Uncertain) else 0.0


def relative_uncertainty(x):
    v = raw(x)
    return v.sigma / abs(v.x) if isinstance(v, Uncertain) and v.x else 0.0
