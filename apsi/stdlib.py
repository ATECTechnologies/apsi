"""Apsi standard library: global builtins and the importable modules
(math, physics.*, nn, linalg, random)."""
import math
import sys
import time as _time
from fractions import Fraction

import numpy as np

from . import ops
from . import autodiff as ad
from .autodiff import Node
from .errors import RuntimeErr, ShapeError, UnitError
from .fmt import fmt, type_name
from .units import Quantity, make, DIMLESS, dim_name, parse_unit
from .values import (Record, Dtype, UnitNamespace, tree_map, tree_leaves)

# =============================================================== autodiff API
_TRACE_DEPTH = [0]


def _wrap_leaf(x):
    if isinstance(x, Quantity):
        inner = _wrap_leaf(x.v)
        return Quantity(inner, x.dim) if isinstance(inner, Node) else x
    if isinstance(x, Node):
        return ad.ident(x)                    # nested grad: stay connected to the outer graph
    if isinstance(x, (bool, np.bool_)):
        return x
    if isinstance(x, (int, float, Fraction, np.floating, np.integer)):
        return Node(float(x))
    if isinstance(x, np.ndarray) and x.dtype.kind in "fiu":
        return Node(x if x.dtype.kind == "f" else x.astype(float))    # float32 stays float32 (half the memory)
    return x


def _unwrap_leaf(grads, out_dim):
    def get(n):
        g = grads.get(id(n))
        if g is None:
            g = np.zeros_like(n.value, dtype=float) if np.ndim(n.value) else 0.0
        return ops.py(g) if not isinstance(g, Node) else g

    def f(x):
        if isinstance(x, Quantity) and isinstance(x.v, Node):
            return make(get(x.v), out_dim / x.dim)
        if isinstance(x, Node):
            return make(get(x), out_dim)
        return None
    return f


def _value_and_grad(f, args, kwargs, wrt):
    idxs = [wrt] if isinstance(wrt, int) else list(wrt)
    args = list(args)
    nested = _TRACE_DEPTH[0] > 0
    for i in idxs:
        if i >= len(args):
            raise RuntimeErr(f"grad: argument {i} was not supplied")
        args[i] = tree_map(_wrap_leaf, args[i])
    _TRACE_DEPTH[0] += 1
    try:
        out = f(*args, **kwargs)
    finally:
        _TRACE_DEPTH[0] -= 1
    dim, ov = DIMLESS, out
    if isinstance(out, Quantity):
        dim, ov = out.dim, out.v
    grads = {}
    if isinstance(ov, Node):
        if np.size(ov.value) != 1:
            raise ShapeError(f"grad() needs a function returning a scalar, got shape {np.shape(ov.value)}")
        try:
            grads = ad.gradients(ov, create_graph=nested)
        except ad.HigherOrderError as e:
            raise RuntimeErr(str(e))
        value = ov if nested else ops.py(ov.value)
    else:
        value = ops.py(ov)
    g = [tree_map(_unwrap_leaf(grads, dim), args[i]) for i in idxs]
    return make(value, dim), (g[0] if isinstance(wrt, int) else tuple(g))


def grad(f, wrt=0):
    """grad(f) -> function returning df/dx (reverse mode, unit-aware, any order)."""
    def g(*args, **kwargs):
        return _value_and_grad(f, args, kwargs, wrt)[1]
    g.__name__ = f"grad({getattr(f, 'name', getattr(f, '__name__', 'f'))})"
    return g


def value_and_grad(f, wrt=0):
    def g(*args, **kwargs):
        return _value_and_grad(f, args, kwargs, wrt)
    g.__name__ = f"value_and_grad({getattr(f, 'name', 'f')})"
    return g


def derivative(f, x):
    return grad(f)(x)


def jacobian(f):
    """jacobian(f)(x) -> d f_i / d x_j  (f: Tensor[M] -> Tensor[M], x: Tensor[N]) as Tensor[M, N]."""
    def j(x):
        y = f(x)
        m = int(np.size(ops.raw(y)))
        rows = [grad(lambda z, i=i: ops.flatten(f(z))[i])(x) for i in range(m)]
        return ops.reshape(ops.stack(rows), *(np.shape(ops.raw(y)) + np.shape(ops.raw(x))))
    j.__name__ = f"jacobian({getattr(f, 'name', 'f')})"
    return j


def hessian(f):
    """hessian(f)(x) -> matrix of second derivatives of a scalar f."""
    return jacobian(grad(f))


def laplace(f):
    """laplace(f)(x) -> Σ ∂²f/∂x_i²  (trace of the Hessian), e.g. for PDE residuals."""
    def lap(x):
        if np.ndim(ops.raw(x)) == 0:
            return grad(grad(f))(x)
        n = int(np.size(ops.raw(x)))
        df = grad(f)
        total = 0
        for i in range(n):
            total = total + ops.flatten(grad(lambda z, i=i: ops.flatten(df(z))[i])(x))[i]
        return total
    lap.__name__ = f"laplace({getattr(f, 'name', 'f')})"
    return lap


def stop_gradient(x):
    if isinstance(x, Quantity):
        return Quantity(ad.const(x.v), x.dim, x.disp)
    return tree_map(lambda v: ad.const(v) if isinstance(v, Node) else v, x)


# =============================================================== decorators
F16, F32, F64 = Dtype("f16", np.float16), Dtype("f32", np.float32), Dtype("f64", np.float64)
I32, I64 = Dtype("i32", np.int32), Dtype("i64", np.int64)
EXACT = Dtype("exact", None)
DTYPES = {"f16": F16, "f32": F32, "f64": F64, "i32": I32, "i64": I64, "exact": EXACT}


def precision(dtype):
    """@precision(f32) / @precision(exact): numeric precision of a function."""
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]
    if dtype is _exact:            # `exact` is both a builtin and a precision mode
        dtype = EXACT
    if not isinstance(dtype, Dtype):
        raise RuntimeErr("precision() takes a numeric format: f64, f32, f16, bf16, fp8_e4m3, fp8_e5m2, "
                         "fp6_e3m2, fp6_e2m3, fp4_e2m1, int8, int4 or exact")
    from . import precision as _P
    low = dtype.name not in ("f16", "f32", "f64", "i32", "i64", "exact")
    if low:
        _P.get_format(dtype.name)          # validates the name

    def deco(fn):
        from .values import Function
        if low:
            def lcast(v):
                if isinstance(v, (np.ndarray, Node)) or (isinstance(v, Quantity) and np.ndim(ops.raw(v))):
                    return _P.round_to(v, dtype.name)
                return v

            def lwrapper(*args, **kwargs):
                return lcast(fn(*[lcast(a) for a in args], **{k: lcast(v) for k, v in kwargs.items()}))
            for attr in ("name", "_apsi_e", "_apsi_def"):
                if hasattr(fn, attr):
                    setattr(lwrapper, attr, getattr(fn, attr))
            lwrapper.precision = dtype.name
            return lwrapper
        if isinstance(fn, Function) or dtype.name == "exact":
            fn.precision = dtype.name      # interpreter engine / compile-time exact literals
            return fn
        npd = dtype.np

        def cast(v):
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                return v.astype(npd)
            if isinstance(v, Quantity) and isinstance(v.v, np.ndarray):
                return Quantity(v.v.astype(npd), v.dim, v.disp)
            return v

        def wrapper(*args, **kwargs):
            return cast(fn(*[cast(a) for a in args], **{k: cast(v) for k, v in kwargs.items()}))
        for attr in ("name", "_apsi_e", "_apsi_def"):
            if hasattr(fn, attr):
                setattr(wrapper, attr, getattr(fn, attr))
        wrapper.precision = dtype.name
        return wrapper
    return deco


def memo(fn):
    """@memo: cache results by argument values."""
    cache = {}

    def wrapper(*args, **kwargs):
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hash(key)
        except TypeError:
            key = repr((args, kwargs))
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    wrapper.__name__ = getattr(fn, "name", "memo")
    wrapper.name = wrapper.__name__
    return wrapper


def timed(fn):
    """@timed: print how long each call takes."""
    def wrapper(*args, **kwargs):
        t0 = _time.perf_counter()
        r = fn(*args, **kwargs)
        dt = _time.perf_counter() - t0
        print(f"[timed] {getattr(fn, 'name', 'fn')}: {dt * 1000:.3f} ms")
        return r
    wrapper.__name__ = getattr(fn, "name", "timed")
    wrapper.name = wrapper.__name__
    return wrapper


# =============================================================== misc builtins
def _print(*args, sep=" ", end="\n", flush=False, file=None):
    out = sys.stderr if file in ("stderr", sys.stderr) else sys.stdout
    out.write(str(sep).join(fmt(a) for a in args) + str(end))
    if flush:
        out.flush()


def _write_out(text):
    """Write text without a line break (Python's sys.stdout.write)."""
    sys.stdout.write(str(text))
    return len(str(text))


def _write_err(text):
    sys.stderr.write(str(text))
    return len(str(text))


def _flush_out():
    sys.stdout.flush()


def _set_recursion_limit(n):
    """Allow deeper recursion (Python's sys.setrecursionlimit), capped at 1 000 000; the stack grows with it."""
    import threading
    n = max(1000, min(int(n), 1_000_000))
    sys.setrecursionlimit(n)
    try:
        threading.stack_size(min(512 * 1024 * 1024, max(threading.stack_size(), n * 1024)))
    except (ValueError, RuntimeError):
        pass
    return n


def _platform():
    import platform
    return {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform) + "-" + platform.machine().lower()


def _str(x):
    t = type(x)
    if t is int:
        return str(x)
    if t is str:
        return x
    return fmt(x)


def _int(x, base=None):
    if base is not None:
        return int(str(x), int(base))
    return _int1(x)


def _int1(x):
    if isinstance(x, Quantity):
        raise UnitError(f"cannot convert {dim_name(x.dim)} to Int")
    if isinstance(x, str):
        return int(x)
    return int(ops.raw(x))


def _float(x):
    if isinstance(x, Quantity):
        raise UnitError(f"cannot convert {dim_name(x.dim)} to Float; use x.magnitude")
    if isinstance(x, str):
        return float(x)
    return float(ops.raw(x))


def _range(a, b=None, step=1):
    if b is None:
        return range(int(a))
    return range(int(a), int(b), int(step))


def _len(x):
    return len(x)


def _error(msg):
    raise RuntimeErr(str(msg))


def _platform():
    """"windows", "linux" or "macos"."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


PROGRAM_ARGS = []      # filled by `apsi run FILE ARGS...`


def _args():
    """Command-line arguments after the program name: `apsi run app.apsi a b` -> ["a", "b"]."""
    return list(PROGRAM_ARGS)


def _input(prompt=""):
    """Read one line from standard input (without the newline); returns none at end of input."""
    sys.stdout.write(fmt(prompt) if not isinstance(prompt, str) else prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        return None
    return line.rstrip("\n").rstrip("\r")


def _exit(code=0):
    """Stop the program with an exit status (0 = success)."""
    raise SystemExit(int(code))


def _getattr(obj, name, *default):
    """getattr(obj, "field"[, default]); names starting with _ are never reachable."""
    from .values import StructInstance, Record, Namespace, BoundMethod
    name = str(name)
    if name.startswith("_"):
        raise RuntimeErr(f"getattr(): attribute '{name}' is private")
    if isinstance(obj, (StructInstance, Record)):
        if name in obj.fields:
            return obj.fields[name]
        m = obj.type.methods.get(name) if isinstance(obj, StructInstance) else None
        if m is not None:
            return BoundMethod(m, obj)
        if default:
            return default[0]
        raise RuntimeErr(f"no field or method '{name}'")
    if isinstance(obj, Namespace):
        if name in obj.members:
            return obj.members[name]
        if default:
            return default[0]
        raise RuntimeErr(f"module '{obj.name}' has no member '{name}'")
    return getattr(obj, name, *default)


def _hasattr(obj, name):
    try:
        _getattr(obj, name)
        return True
    except Exception:                      # noqa: BLE001
        return False


_TYPE_OF_CONVERTER = {}


def _isinstance(x, t):
    """isinstance(x, T): T is a type name ("Int", "Tensor", "Length"…), int/float/str/bool/list/dict/tuple/set,
    a struct, an error kind (ValueError…), or a tuple/list of those."""
    import builtins as b
    from .values import StructInstance, StructType
    if isinstance(t, (tuple, list)):
        return any(_isinstance(x, u) for u in t)
    if isinstance(t, str):
        return type_name(x) == t or (t == "Float" and type_name(x) == "Int") or (
            t in ("Number", "Any")) and isinstance(x, (int, float, complex)) or t == "Any"
    py = _TYPE_OF_CONVERTER.get(id(t))
    if py is not None:
        if py is int:
            return isinstance(x, (int, np.integer))           # like Python: true and false are Ints too
        if py is float:
            return isinstance(x, (float, np.floating))
        return isinstance(x, py)
    if isinstance(t, StructType):
        if not isinstance(x, StructInstance):
            return False
        st = x.type
        while st is not None:
            if st is t or st.name == t.name:
                return True
            st = getattr(st, "base", None)
        return False
    if isinstance(t, type):
        return b.isinstance(x, t)
    raise RuntimeErr(f"isinstance(): {type_name(t)} is not a type")


def _setattr(obj, name, value):
    from .values import StructInstance, Record
    name = str(name)
    if name.startswith("_"):
        raise RuntimeErr(f"setattr(): attribute '{name}' is private")
    if isinstance(obj, StructInstance):
        if name not in obj.fields:
            raise RuntimeErr(f"{obj.type.name} has no field '{name}'")
        obj.fields[name] = value
    elif isinstance(obj, Record):
        obj.fields[name] = value
    else:
        _safe_setattr(obj, name, value)


def _safe_setattr(obj, name, value):
    """Setting a public attribute of an ordinary object (decimal's context.prec = 50, a dataclass field…).
    Modules, classes, functions and built-in objects stay read-only, so the sandbox cannot be bent."""
    import types
    if name.startswith("_"):
        raise RuntimeErr(f"cannot set the private attribute '{name}'")
    if isinstance(obj, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                        int, float, str, bytes, tuple, frozenset, bool, type(None))):
        raise RuntimeErr(f"cannot set attribute '{name}' on {type_name(obj)}")
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError) as e:
        raise RuntimeErr(f"cannot set attribute '{name}' on {type_name(obj)}: {e}")


def _py_type_name(x):
    """The name Python would give the type of x (int, float, str, list, dict, or the struct's name)."""
    from .values import StructInstance
    if isinstance(x, StructInstance):
        return x.type.name
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, (int, np.integer)):
        return "int"
    if isinstance(x, (float, np.floating)):
        return "float"
    if isinstance(x, np.ndarray):
        return "ndarray"
    if x is None:
        return "NoneType"
    return type(x).__name__


def _vars(obj):
    """The fields of a struct or record as a dict."""
    from .values import StructInstance, Record
    if isinstance(obj, (StructInstance, Record)):
        return dict(obj.fields)
    return {k: v for k, v in getattr(obj, "__dict__", {}).items() if not k.startswith("_")}


def _dir(obj=None):
    from .values import StructInstance, Record, Namespace
    if isinstance(obj, StructInstance):
        return sorted(list(obj.fields) + list(obj.type.methods))
    if isinstance(obj, Record):
        return sorted(obj.fields)
    if isinstance(obj, Namespace):
        return sorted(obj.members)
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def _hash(x):
    """A stable hash for numbers, strings, tuples and struct instances; tensors hash their bytes."""
    if isinstance(x, np.ndarray):
        import hashlib
        return int.from_bytes(hashlib.blake2b(x.tobytes() + str(x.shape).encode(), digest_size=8).digest(), "big", signed=True)
    if isinstance(x, list):
        return hash(tuple(_hash(v) for v in x))
    if isinstance(x, dict):
        return hash(tuple(sorted((_hash(k), _hash(v)) for k, v in x.items())))
    return hash(x)


def _repr(x):
    return fmt(x, False)


def _python_compat():
    """Python built-ins that are safe and commonly needed (also by `apsi convert`)."""
    import builtins as b
    return {"any": b.any, "all": b.all, "divmod": b.divmod, "pow": b.pow, "ord": b.ord, "chr": b.chr,
            "dict": b.dict, "set": b.set, "tuple": b.tuple, "repr": _repr, "hex": b.hex, "bin": b.bin,
            "oct": b.oct, "next": b.next, "iter": b.iter, "format": b.format, "callable": b.callable,
            "getattr": _getattr, "hasattr": _hasattr, "fsum": _fsum, "setattr": _setattr, "isinstance": _isinstance,
            "super": super, "property": property, "staticmethod": staticmethod, "classmethod": classmethod,
            "object": object,
            "hash": _hash, "id": b.id, "ascii": b.ascii, "bytes": b.bytes, "bytearray": b.bytearray,
            "complex": b.complex, "slice": b.slice, "vars": _vars, "dir": _dir, "issubclass": b.issubclass,
            "frozenset": b.frozenset, "py_type_name": _py_type_name}


def _fsum(x):
    """Correctly rounded sum (no accumulated rounding error), for lists and tensors of any size."""
    raw = ops.raw(x)
    if isinstance(x, Quantity):
        return Quantity(_fsum(x.v), x.dim)
    if isinstance(raw, np.ndarray):
        return math.fsum(raw.ravel().tolist())
    vals = list(x)
    if vals and isinstance(vals[0], Quantity):
        d = vals[0].dim
        for v in vals:
            if not isinstance(v, Quantity) or v.dim != d:
                raise UnitError("fsum(): all values need the same unit")
        return Quantity(math.fsum(float(v.v) for v in vals), d)
    return math.fsum(float(ops.raw(v)) for v in vals)


def _clock():
    return Quantity(_time.perf_counter(), parse_unit("s")[1])


def _dim(x):
    d = ops.dim_of(x)
    return dim_name(d).split(" [")[0] if d is not None else "None"


def _unit(x):
    return x.unit if isinstance(x, Quantity) else ""


def _magnitude(x):
    return x.magnitude if isinstance(x, Quantity) else x


def _pmap(f, xs, workers=None):
    """Parallel map on a thread pool (numpy/BLAS work releases the GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    items = list(xs)
    n = workers or min(len(items), os.cpu_count() or 1) or 1
    if n <= 1 or len(items) <= 1:
        return [f(x) for x in items]
    with ThreadPoolExecutor(max_workers=int(n)) as ex:
        return list(ex.map(f, items))


def _record(**kw):
    return Record(**kw)


def _map(f, xs):
    return [f(x) for x in xs]


def _filter(f, xs):
    return [x for x in xs if f(x)]


def _enumerate(xs, start=0):
    return list(enumerate(xs, start))


def _zip(*xs):
    return list(zip(*xs))


def _sorted(xs, key=None, reverse=False):
    return sorted(xs, key=key, reverse=reverse)


def _reversed(xs):
    return list(reversed(list(xs)))


def _unique(xs):
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _chars(s):
    return list(str(s))


def _join(xs, sep=""):
    return str(sep).join(fmt(x) for x in xs)


def _list(x=()):
    return list(x)


def _keys(d):
    return list(d.fields.keys()) if isinstance(d, Record) else list(d.keys())


def _param_count(tree):
    return int(sum(np.size(ops.raw(x)) for x in tree_leaves(tree) if ops.dim_of(x) is not None))


def _tree_map(f, *trees):
    return tree_map(f, *trees)


def _load_csv(path, skip_header=0):
    return np.loadtxt(path, delimiter=",", skiprows=int(skip_header), ndmin=1)


def _save_csv(path, data, header=""):
    np.savetxt(path, np.asarray(ops.raw(data)), delimiter=",", header=header, comments="")


def _exact(x):
    if isinstance(x, str):
        return Fraction(x)
    return Fraction(x).limit_denominator(10 ** 12) if isinstance(x, float) else Fraction(x)


def _to_float(x):
    return float(x) if isinstance(x, Fraction) else x


# =============================================================== nn module
def dense(n_in, n_out, seed=None, scale=None):
    """A dense layer's parameters: record(W, b) with He initialisation."""
    s = scale if scale is not None else math.sqrt(2.0 / n_in)
    rng = ops._rng(seed)
    return Record(W=rng.standard_normal((int(n_in), int(n_out))) * s, b=np.zeros(int(n_out)))


def mlp_init(sizes, seed=0):
    return [dense(a, b, seed=int(seed) + i) for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]


def mse(pred, target):
    d = pred - target
    return ops.mean(d * d)


def cross_entropy(logits, labels):
    """Mean cross-entropy of integer labels; labels [B] with logits [B, C], or labels [B, T] with logits
    [B, T, C] (sequence models: every position counts). Positions labelled -100 (any negative label) are left
    out, e.g. the prompt part of an instruction example."""
    labels = np.asarray(ops.raw(labels)).astype(int)
    if labels.ndim > 1:
        c = ops.shape(logits)[-1]
        logits = ops.reshape(logits, -1, c)
        labels = labels.reshape(-1)
    lp = ops.log_softmax(logits, axis=-1)
    if (labels < 0).any():                       # -100 (any negative label): not trained on, as in PyTorch
        keep = np.nonzero(labels >= 0)[0]
        if not len(keep):
            return ops.mean(lp[(np.arange(1), np.zeros(1, int))]) * 0.0
        return -ops.mean(lp[(keep, labels[keep])])
    n = labels.shape[0]
    picked = lp[(np.arange(n), labels)]
    return -ops.mean(picked)


def l1_loss(pred, target):
    """Mean absolute error."""
    return ops.mean(ops.abs_(pred - target))


def bce_with_logits(logits, target):
    """Binary cross-entropy on logits (numerically stable): mean(softplus(z) − y·z)."""
    return ops.mean(ops.softplus(logits) - target * logits)


def bce(prob, target):
    """Binary cross-entropy on probabilities (clamped like PyTorch: log ≥ −100)."""
    with np.errstate(divide="ignore"):
        lp = ops.log(prob)
        lq = ops.log(1.0 - prob)
    lp = ops.where(np.asarray(ops.raw(lp)) > -100.0, lp, -100.0)
    lq = ops.where(np.asarray(ops.raw(lq)) > -100.0, lq, -100.0)
    return -ops.mean(target * lp + (1.0 - target) * lq)


def nll_loss(log_probs, labels):
    """Negative log-likelihood of integer labels given log-probabilities [N, C]."""
    labels = np.asarray(ops.raw(labels)).astype(int)
    n = labels.shape[0]
    return -ops.mean(log_probs[(np.arange(n), labels)])


def huber_loss(pred, target, delta=1.0):
    """Huber / smooth-L1 loss (mean), quadratic below delta and linear above."""
    d = pred - target
    a = np.abs(np.asarray(ops.raw(d)))
    quad = 0.5 * d * d
    lin = delta * (ops.abs_(d) - 0.5 * delta)
    return ops.mean(ops.where(a <= delta, quad, lin))


class Loader:
    """Mini-batches of (X, y), reshuffled every time it is iterated (one pass = one epoch)."""

    def __init__(self, X, y, batch_size, shuffle=False, seed=None):
        self.X, self.y, self.size, self.shuffle = X, y, int(batch_size), bool(shuffle)
        self.n = np.shape(ops.raw(X))[0]
        self.rng = ops._rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for i in range(0, self.n, self.size):
            j = idx[i:i + self.size]
            yield (self.X[j], self.y[j])

    def __len__(self):
        return -(-self.n // self.size)


def loader(X, y, batch_size=32, shuffle=False, seed=None):
    """Iterable mini-batches (like a PyTorch DataLoader over a TensorDataset); reshuffled every epoch."""
    return Loader(X, y, batch_size, shuffle, seed)


def accuracy(logits, labels):
    pred = np.argmax(np.asarray(ops.raw(logits)), axis=-1)
    return float(np.mean(pred == np.asarray(ops.raw(labels))))


def one_hot(labels, k):
    labels = np.asarray(ops.raw(labels)).astype(int)
    out = np.zeros((labels.shape[0], int(k)))
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr, self.momentum, self.vel = lr, momentum, None

    def step(self, params, grads):
        if self.momentum:
            if self.vel is None:
                self.vel = tree_map(ops.zeros_like, grads)
            self.vel = tree_map(lambda v, g: self.momentum * v + g, self.vel, grads)
            grads = self.vel
        return tree_map(lambda p, g: p - self.lr * g, params, grads)


def _float_arr(g):
    """Gradient as a float array, keeping float32 when training in float32 (half the optimizer memory)."""
    a = np.asarray(ops.raw(g))
    return a if a.dtype.kind == "f" else a.astype(float)


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        g_raw = tree_map(_float_arr, grads)
        if self.m is None:
            self.m = tree_map(np.zeros_like, g_raw)
            self.v = tree_map(np.zeros_like, g_raw)
        self.t += 1
        b1, b2 = self.b1, self.b2
        self.m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, self.m, g_raw)
        self.v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, self.v, g_raw)
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t

        def upd(p, m, v):
            delta = self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
            if isinstance(p, Quantity):      # Adam steps are scale-free: update in SI units
                return Quantity(p.v - delta, p.dim)
            return p - delta
        return tree_map(upd, params, self.m, self.v)


def sgd(lr=0.01, momentum=0.0):
    return SGD(lr, momentum)


def adam(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    return Adam(lr, b1, b2, eps)


def spiral_data(n_per_class=100, classes=3, noise=0.2, seed=0):
    """Classic spiral toy dataset -> (X: Tensor[N, 2], y: Tensor[N] of class ids)."""
    rng = ops._rng(seed)
    n = int(n_per_class)
    X = np.zeros((n * classes, 2))
    y = np.zeros(n * classes, dtype=int)
    for c in range(int(classes)):
        r = np.linspace(0.0, 1.0, n)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n) + rng.standard_normal(n) * noise
        X[c * n:(c + 1) * n] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[c * n:(c + 1) * n] = c
    return (X, y)


def batches(X, y, size, seed=None):
    n = np.shape(ops.raw(X))[0]
    idx = ops._rng(seed).permutation(n)
    return [(X[idx[i:i + size]], y[idx[i:i + size]]) for i in range(0, n, int(size))]




# ---------------------------------------------------------------- more nn building blocks
def conv2d(x, w, b=None, stride=1, padding=0):
    out = ad.conv2d(x, w, int(stride), int(padding))
    if b is not None:
        out = out + ops.reshape(b, 1, -1, 1, 1)
    return out


def max_pool2d(x, k=2):
    return ad.max_pool2d(x, int(k))


def conv_init(c_in, c_out, k=3, seed=None):
    rng = ops._rng(seed)
    fan_in = c_in * k * k
    return Record(W=rng.standard_normal((int(c_out), int(c_in), int(k), int(k))) * math.sqrt(2.0 / fan_in),
                  b=np.zeros(int(c_out)))


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    m = ops.mean(x, axis=-1, keepdims=True)
    d = x - m
    v = ops.mean(d * d, axis=-1, keepdims=True)
    y = d / ops.sqrt(v + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def norm_init(d):
    return Record(gamma=np.ones(int(d)), beta=np.zeros(int(d)))


def dropout(x, p=0.1, train=True, seed=None):
    if not train or p <= 0:
        return x
    mask = (ops._rng(seed).random(np.shape(ops.raw(x))) >= p) / (1.0 - p)
    return x * mask


def embedding_init(vocab, d, seed=None):
    return ops._rng(seed).standard_normal((int(vocab), int(d))) * 0.02


def causal_mask(T):
    return np.triu(np.full((int(T), int(T)), -1e9), k=1)


def attention(q, k, v, mask=None):
    """Scaled dot-product attention over the last two axes."""
    d = np.shape(ops.raw(q))[-1]
    nd = len(np.shape(ops.raw(k)))
    axes = tuple(range(nd - 2)) + (nd - 1, nd - 2)
    s = (q @ ops.transpose(k, axes)) / math.sqrt(d)
    if mask is not None:
        s = s + mask
    return ops.softmax(s, axis=-1) @ v


def global_norm(grads):
    return math.sqrt(sum(float(np.sum(np.asarray(ops.raw(g), dtype=float) ** 2))
                         for g in tree_leaves(grads) if ops.dim_of(g) is not None))


def clip_grad_norm(grads, max_norm=1.0):
    n = global_norm(grads)
    if n <= max_norm or n == 0:
        return grads
    s = max_norm / n
    return tree_map(lambda g: g * s if ops.dim_of(g) is not None else g, grads)


class AdamW(Adam):
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr, b1, b2, eps)
        self.wd = weight_decay

    def step(self, params, grads):
        decayed = tree_map(lambda p: p * (1 - self.lr * self.wd)
                           if isinstance(ops.raw(p), np.ndarray) and np.ndim(ops.raw(p)) >= 2 else p, params)
        return super().step(decayed, grads)


def adamw(lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
    return AdamW(lr, b1, b2, eps, weight_decay)


def cosine_lr(step, total, base=1e-3, warmup=0, minimum=0.0):
    step, total = float(step), float(total)
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1.0, total - warmup))
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * t))


def set_lr(opt, lr):
    opt.lr = lr
    return opt


def save_params(path, params):
    leaves = [np.asarray(ops.raw(x)) for x in tree_leaves(params)]
    np.savez(path, *leaves)


def load_params(path, like):
    data = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
    it = iter([data[f"arr_{i}"] for i in range(len(data.files))])
    return tree_map(lambda x: next(it), like)


def numeric_grad(f, x, eps=1e-6):
    """Finite-difference gradient (for checking autodiff)."""
    x = np.asarray(ops.raw(x), dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (float(ops.raw(f(xp))) - float(ops.raw(f(xm)))) / (2 * eps)
    return g




# ---------------------------------------------------------------- recurrent nets, normalisation
def lstm_init(n_in, n_hidden, seed=None):
    """LSTM parameters: one fused weight for the 4 gates (input, forget, cell, output)."""
    rng = ops._rng(seed)
    k = 1.0 / math.sqrt(n_hidden)
    b = np.zeros(4 * int(n_hidden))
    b[int(n_hidden):2 * int(n_hidden)] = 1.0          # forget-gate bias = 1 (standard trick)
    return Record(W=rng.uniform(-k, k, (int(n_in) + int(n_hidden), 4 * int(n_hidden))), b=b)


def lstm_cell(p, x, h, c):
    """One LSTM step: returns (h, c). x: [B, n_in], h, c: [B, n_hidden]."""
    H = np.shape(ops.raw(h))[-1]
    z = ops.concat([x, h], axis=-1) @ p.W + p.b
    i = ops.sigmoid(z[:, 0:H])
    f = ops.sigmoid(z[:, H:2 * H])
    g = ops.tanh(z[:, 2 * H:3 * H])
    o = ops.sigmoid(z[:, 3 * H:4 * H])
    c2 = f * c + i * g
    return (o * ops.tanh(c2), c2)


def lstm(p, xs):
    """Run an LSTM over a sequence xs: [B, T, n_in] -> outputs [B, T, n_hidden]."""
    B, T = np.shape(ops.raw(xs))[:2]
    H = np.shape(ops.raw(p.b))[0] // 4
    h = np.zeros((B, H))
    c = np.zeros((B, H))
    outs = []
    for t in range(T):
        h, c = lstm_cell(p, xs[:, t, :], h, c)
        outs.append(h)
    return ops.stack(outs, axis=1)


def gru_init(n_in, n_hidden, seed=None):
    rng = ops._rng(seed)
    k = 1.0 / math.sqrt(n_hidden)
    n = int(n_in) + int(n_hidden)
    return Record(Wzr=rng.uniform(-k, k, (n, 2 * int(n_hidden))), bzr=np.zeros(2 * int(n_hidden)),
                  Wh=rng.uniform(-k, k, (n, int(n_hidden))), bh=np.zeros(int(n_hidden)))


def gru_cell(p, x, h):
    H = np.shape(ops.raw(h))[-1]
    zr = ops.sigmoid(ops.concat([x, h], axis=-1) @ p.Wzr + p.bzr)
    z, r = zr[:, 0:H], zr[:, H:2 * H]
    cand = ops.tanh(ops.concat([x, r * h], axis=-1) @ p.Wh + p.bh)
    return (1 - z) * h + z * cand


def gru(p, xs):
    B, T = np.shape(ops.raw(xs))[:2]
    H = np.shape(ops.raw(p.bh))[0]
    h = np.zeros((B, H))
    outs = []
    for t in range(T):
        h = gru_cell(p, xs[:, t, :], h)
        outs.append(h)
    return ops.stack(outs, axis=1)


def batch_norm(x, gamma=None, beta=None, eps=1e-5, axis=0):
    """Batch normalisation over `axis` (training-mode statistics)."""
    m = ops.mean(x, axis=axis, keepdims=True)
    d = x - m
    v = ops.mean(d * d, axis=axis, keepdims=True)
    y = d / ops.sqrt(v + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def rms_norm(x, gamma=None, eps=1e-6):
    y = x / ops.sqrt(ops.mean(x * x, axis=-1, keepdims=True) + eps)
    return y * gamma if gamma is not None else y


class RMSprop:
    def __init__(self, lr=1e-3, decay=0.9, eps=1e-8):
        self.lr, self.decay, self.eps, self.s = lr, decay, eps, None

    def step(self, params, grads):
        g_raw = tree_map(lambda g: np.asarray(ops.raw(g), dtype=float), grads)
        if self.s is None:
            self.s = tree_map(np.zeros_like, g_raw)
        self.s = tree_map(lambda s, g: self.decay * s + (1 - self.decay) * g * g, self.s, g_raw)

        def upd(p, g, s):
            delta = self.lr * g / (np.sqrt(s) + self.eps)
            return Quantity(p.v - delta, p.dim) if isinstance(p, Quantity) else p - delta
        return tree_map(upd, params, g_raw, self.s)


class Adagrad:
    def __init__(self, lr=1e-2, eps=1e-10):
        self.lr, self.eps, self.s = lr, eps, None

    def step(self, params, grads):
        g_raw = tree_map(lambda g: np.asarray(ops.raw(g), dtype=float), grads)
        if self.s is None:
            self.s = tree_map(np.zeros_like, g_raw)
        self.s = tree_map(lambda s, g: s + g * g, self.s, g_raw)

        def upd(p, g, s):
            delta = self.lr * g / (np.sqrt(s) + self.eps)
            return Quantity(p.v - delta, p.dim) if isinstance(p, Quantity) else p - delta
        return tree_map(upd, params, g_raw, self.s)


def rmsprop(lr=1e-3, decay=0.9, eps=1e-8):
    return RMSprop(lr, decay, eps)


def adagrad(lr=1e-2, eps=1e-10):
    return Adagrad(lr, eps)


def step_lr(step, base=1e-3, every=100, factor=0.5):
    return base * factor ** (int(step) // int(every))


def linear_warmup(step, base=1e-3, warmup=100):
    return base * min(1.0, (float(step) + 1) / float(warmup))


# ---------------------------------------------------------------- data
def make_moons(n=200, noise=0.1, seed=0):
    rng = ops._rng(seed)
    n1 = int(n) // 2
    n2 = int(n) - n1
    t1 = np.linspace(0, math.pi, n1)
    t2 = np.linspace(0, math.pi, n2)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    X += rng.standard_normal(X.shape) * noise
    y = np.r_[np.zeros(n1, dtype=int), np.ones(n2, dtype=int)]
    return (X, y)


def make_blobs(n=300, centers=3, dim=2, spread=0.5, seed=0):
    rng = ops._rng(seed)
    cs = rng.uniform(-5, 5, (int(centers), int(dim)))
    y = rng.integers(0, int(centers), int(n))
    X = cs[y] + rng.standard_normal((int(n), int(dim))) * spread
    return (X, y)


def make_regression(n=100, features=3, noise=0.1, seed=0):
    rng = ops._rng(seed)
    X = rng.standard_normal((int(n), int(features)))
    w = rng.standard_normal(int(features))
    y = X @ w + rng.standard_normal(int(n)) * noise
    return (X, y, w)


def train_test_split(X, y, test=0.2, seed=0):
    n = np.shape(ops.raw(X))[0]
    idx = ops._rng(seed).permutation(n)
    k = int(round(n * (1 - test)))
    a, b = idx[:k], idx[k:]
    return (X[a], X[b], y[a], y[b])


def standardize(X, axis=0):
    """Return (X_scaled, mean, std) so new data can be scaled the same way."""
    m = ops.mean(X, axis=axis, keepdims=True)
    s = ops.std(X, axis=axis)
    s = ops.reshape(s, *np.shape(ops.raw(m))) if np.ndim(ops.raw(m)) else s
    return ((X - m) / s, m, s)


def train(loss_fn, params, data, opt, epochs=10, batch_size=None, seed=0, verbose=True,
          clip=None, schedule=None, eval_fn=None):
    """Generic training loop.

    loss_fn(params, *batch) -> scalar loss;  data is a tuple of tensors (X, y, ...).
    Returns record(params, history) where history is a list of epoch losses.
    """
    data = tuple(data)
    n = np.shape(ops.raw(data[0]))[0]
    vg = value_and_grad(loss_fn)
    history = []
    step = 0
    rng = ops._rng(seed)
    for ep in range(int(epochs)):
        idx = rng.permutation(n) if batch_size else np.arange(n)
        bs = int(batch_size) if batch_size else n
        total, count = 0.0, 0
        for i in range(0, n, bs):
            sel = idx[i:i + bs]
            batch = tuple(d[sel] for d in data)
            if schedule is not None:
                opt.lr = schedule(step)
            loss, g = vg(params, *batch)
            if clip:
                g = clip_grad_norm(g, clip)
            params = opt.step(params, g)
            total += float(ops.raw(loss)) * len(sel)
            count += len(sel)
            step += 1
        history.append(total / count)
        if verbose and (ep % max(1, int(epochs) // 10) == 0 or ep == int(epochs) - 1):
            extra = f"  {eval_fn(params):.4f}" if eval_fn is not None else ""
            print(f"epoch {ep:4}  loss {history[-1]:.5f}{extra}")
    return Record(params=params, history=history)


# =============================================================== physics constants
def _q(value, unit):
    scale, dim = parse_unit(unit)
    return make(value * scale, dim)


CONSTANTS = {
    "c": _q(299792458.0, "m/s"),
    "G": _q(6.67430e-11, "m^3/(kg s^2)"),
    "h": _q(6.62607015e-34, "J s"),
    "hbar": _q(1.054571817e-34, "J s"),
    "k_B": _q(1.380649e-23, "J/K"),
    "N_A": _q(6.02214076e23, "1/mol"),
    "R_gas": _q(8.314462618, "J/(mol K)"),
    "q_e": _q(1.602176634e-19, "C"),
    "m_e": _q(9.1093837015e-31, "kg"),
    "m_p": _q(1.67262192369e-27, "kg"),
    "m_n": _q(1.67492749804e-27, "kg"),
    "epsilon_0": _q(8.8541878128e-12, "F/m"),
    "mu_0": _q(1.25663706212e-6, "N/A^2"),
    "g0": _q(9.80665, "m/s^2"),
    "sigma_SB": _q(5.670374419e-8, "W/(m^2 K^4)"),
    "a_0": _q(5.29177210903e-11, "m"),
    "alpha_fs": 7.2973525693e-3,
    "M_sun": _q(1.98847e30, "kg"),
    "M_earth": _q(5.9722e24, "kg"),
    "R_earth": _q(6.371e6, "m"),
    "AU": _q(1.495978707e11, "m"),
}


# =============================================================== native tier
def _native(*args, **kwargs):
    """@native / @native("fast") / @native("strict"): compile a function to machine code."""
    from .native import native
    return native(*args, **kwargs)


def _native_threads(n=None):
    """Number of CPU threads `parallel for` and large tensor kernels use (set it with an argument)."""
    from . import native as nat
    return nat.threads() if n is None else nat.set_threads(int(n))


def _gpu(*args, **kwargs):
    """@gpu: run a function's parallel for on the NVIDIA GPU(s); all CPU cores when there is none."""
    from .native.gpu import gpu
    return gpu(*args, **kwargs)


def _gpu_tensor(x, device="cuda"):
    """x on the card with Apsi's own kernels (products, arithmetic, activations, softmax, norms, sums stay there;
    any other function sees an ordinary tensor). On the processor when there is no card."""
    from .gpu_tensor import gpu_tensor
    return gpu_tensor(x, device)


def _gpu_device(device="cuda"):
    """The card gpu_tensor uses, or "cpu"."""
    from .gpu_tensor import gpu_device
    return gpu_device(device)


def _to_gpu(x, device=0):
    """Keep a tensor in GPU memory between @gpu calls: no copy per call. to_host(g) brings the values back."""
    from .native.gpu import to_gpu
    return to_gpu(x, device)


def _to_host(x):
    """The values of a GPU tensor (to_gpu) as an ordinary tensor."""
    from .native.gpu import to_host
    return to_host(x)


def _gpu_info():
    """The GPUs Apsi can use (needs the NVIDIA driver; no CUDA toolkit)."""
    from .native.gpu import info
    from .values import Record
    i = info()
    i["devices"] = [Record(**d) for d in i["devices"]]
    i.pop("skipped", None)
    return Record(**i)


def _native_info():
    """What the native compiler sees on this machine (CPU, SIMD, threads, cache)."""
    from . import native as nat
    from .values import Record
    return Record(**nat.info())


# =============================================================== registry
def _bitop(op):
    import operator as _o
    f = {"or": _o.or_, "and": _o.and_, "xor": _o.xor, "shl": _o.lshift, "shr": _o.rshift}[op]
    lf = {"or": np.logical_or, "and": np.logical_and, "xor": np.logical_xor}.get(op)

    def g(a, b):
        """Bitwise / set operator: ints bit by bit, sets union (or) / intersection (and) / symmetric difference
        (xor), dicts merged (or), booleans and boolean tensors element by element."""
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if lf is not None and (getattr(a, "dtype", None) == bool or getattr(b, "dtype", None) == bool):
                return lf(a, b)
            return f(np.asarray(a), np.asarray(b))
        if isinstance(a, (set, frozenset)) and isinstance(b, (list, tuple)):
            b = set(b)
        return f(a, b)
    g.__name__ = "b" + op
    return g


def _bnot(a):
    """Bitwise not of an integer (~a), or element-wise not of a boolean tensor."""
    if isinstance(a, np.ndarray) and a.dtype == bool:
        return np.logical_not(a)
    return ~a


def core_builtins():
    import builtins as _b
    for conv, py in ((_int, int), (_float, float), (_str, str), (_list, list), (_b.dict, dict), (_b.tuple, tuple),
                     (_b.set, set), (bool, bool), (_b.frozenset, frozenset), (_b.bytes, bytes), (_b.complex, complex)):
        _TYPE_OF_CONVERTER[id(conv)] = py
    return {
        "chain_lists": lambda xss: [x for xs in xss for x in xs],
        "write_out": _write_out, "write_err": _write_err, "flush_out": _flush_out,
        "set_recursion_limit": _set_recursion_limit, "platform": _platform,
        "bor": _bitop("or"), "band": _bitop("and"), "bxor": _bitop("xor"), "shl": _bitop("shl"),
        "shr": _bitop("shr"), "bnot": _bnot, "set": set, "frozenset": frozenset,
        # io & conversion
        "print": _print, "str": _str, "int": _int, "float": _float, "bool": bool,
        "len": _len, "range": _range, "type": type_name, "error": _error,
        "native": _native, "native_threads": _native_threads, "native_info": _native_info,
        "gpu": _gpu, "gpu_info": _gpu_info, "to_gpu": _to_gpu, "to_host": _to_host,
        "gpu_tensor": _gpu_tensor, "gpu_device": _gpu_device,
        "list": _list, "pmap": _pmap, "unique": _unique, "chars": _chars, "join": _join, "map": _map, "filter": _filter, "enumerate": _enumerate, "zip": _zip,
        "sorted": _sorted, "reversed": _reversed, "keys": _keys, "record": _record,
        **_python_compat(),
        "clock": _clock, "args": _args, "input": _input, "exit": _exit, "exact": _exact, "to_float": _to_float,
        "load_csv": _load_csv, "save_csv": _save_csv,
        # units
        "dim": _dim, "unit": _unit, "magnitude": _magnitude, "u": UnitNamespace(),
        "measure": ops.plusminus, "nominal": ops.nominal, "sigma": ops.sigma,
        "relative_uncertainty": ops.relative_uncertainty,
        # math
        "abs": ops.abs_, "sqrt": ops.sqrt, "cbrt": ops.cbrt, "exp": ops.exp, "log": ops.log,
        "log10": ops.log10, "log2": ops.log2, "sin": ops.sin, "cos": ops.cos, "tan": ops.tan,
        "asin": ops.asin, "acos": ops.acos, "atan": ops.atan, "atan2": ops.atan2,
        "sinh": ops.sinh, "cosh": ops.cosh, "tanh": ops.tanh, "floor": ops.floor,
        "ceil": ops.ceil, "round": ops.round_, "min": ops.min_, "max": ops.max_,
        "swapaxes": ops.swapaxes, "permute": ops.permute, "unsqueeze": ops.unsqueeze, "squeeze": ops.squeeze,
        "erf": ops.erf,
        "sum": ops.sum_, "mean": ops.mean, "std": ops.std, "var": ops.var, "prod": ops.prod,
        "cumsum": ops.cumsum, "argmax": ops.argmax, "argmin": ops.argmin, "approx": ops.approx,
        "pi": math.pi, "tau": math.tau, "e": math.e, "inf": math.inf, "nan": math.nan,
        # tensors
        "tensor": ops.tensor, "zeros": ops.zeros, "ones": ops.ones, "full": ops.full,
        "zeros_like": ops.zeros_like, "ones_like": ops.ones_like, "eye": ops.eye,
        "arange": ops.arange, "linspace": ops.linspace, "stack": ops.stack, "concat": ops.concat,
        "reshape": ops.reshape, "flatten": ops.flatten, "transpose": ops.transpose,
        "shape": ops.shape, "ndim": ops.ndim, "dot": ops.dot, "cross": ops.cross,
        "outer": ops.outer, "matmul": ops.matmul, "norm": ops.norm, "clip": ops.clip,
        "where": ops.where, "roll": ops.roll, "sort": ops.sort, "meshgrid": ops.meshgrid,
        "randn": ops.randn, "rand": ops.rand, "randint": ops.randint, "seed": ops.seed,
        "permutation": ops.permutation,
        # AI
        "relu": ops.relu, "sigmoid": ops.sigmoid, "softmax": ops.softmax,
        "log_softmax": ops.log_softmax, "gelu": ops.gelu, "softplus": ops.softplus,
        "grad": grad, "value_and_grad": value_and_grad, "derivative": derivative,
        "jacobian": jacobian, "hessian": hessian, "laplace": laplace, "stop_gradient": stop_gradient,
        "tree_map": _tree_map, "param_count": _param_count,
        # precision & decorators
        "f16": F16, "f32": F32, "f64": F64, "i32": I32, "i64": I64,
        "precision": precision, "memo": memo, "timed": timed,
    }


def modules(core):
    math_mod = {k: core[k] for k in ("abs", "sqrt", "cbrt", "exp", "log", "log10", "log2", "sin",
                                      "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
                                      "cosh", "tanh", "floor", "ceil", "round", "pi", "tau", "e",
                                      "inf", "nan")}
    fields = {"laplacian": ops.laplacian, "gradient": ops.gradient, "meshgrid": ops.meshgrid}
    nn = {"dense": dense, "mlp_init": mlp_init, "mse": mse, "cross_entropy": cross_entropy,
          "accuracy": accuracy, "one_hot": one_hot, "sgd": sgd, "adam": adam,
          "spiral_data": spiral_data, "batches": batches, "loader": loader, "l1_loss": l1_loss,
          "bce_with_logits": bce_with_logits, "bce": bce, "nll_loss": nll_loss, "huber_loss": huber_loss,
          "conv2d": conv2d, "max_pool2d": max_pool2d, "conv_init": conv_init,
          "layer_norm": layer_norm, "norm_init": norm_init, "dropout": dropout,
          "embedding_init": embedding_init, "causal_mask": causal_mask, "attention": attention,
          "global_norm": global_norm, "clip_grad_norm": clip_grad_norm, "adamw": adamw,
          "cosine_lr": cosine_lr, "set_lr": set_lr, "save_params": save_params,
          "load_params": load_params, "numeric_grad": numeric_grad,
          "lstm_init": lstm_init, "lstm_cell": lstm_cell, "lstm": lstm,
          "gru_init": gru_init, "gru_cell": gru_cell, "gru": gru,
          "batch_norm": batch_norm, "rms_norm": rms_norm, "rmsprop": rmsprop, "adagrad": adagrad,
          "step_lr": step_lr, "linear_warmup": linear_warmup,
          "make_moons": make_moons, "make_blobs": make_blobs, "make_regression": make_regression,
          "train_test_split": train_test_split, "standardize": standardize, "train": train,
          "relu": ops.relu, "gelu": ops.gelu, "sigmoid": ops.sigmoid, "tanh": ops.tanh,
          "softmax": ops.softmax, "log_softmax": ops.log_softmax}
    linalg = {"inv": ops.inv, "det": ops.det, "solve": ops.solve, "eigh": ops.eigh,
              "eig": ops.eig, "svd": ops.svd, "trace": ops.trace, "norm": ops.norm}
    rnd = {"seed": ops.seed, "randn": ops.randn, "rand": ops.rand, "randint": ops.randint,
           "permutation": ops.permutation}
    physics = dict(CONSTANTS)
    physics.update(fields)
    return {
        "math": math_mod,
        "physics": physics,
        "physics.units": {},
        "physics.constants": dict(CONSTANTS),
        "physics.fields": fields,
        "nn": nn,
        "linalg": linalg,
        "random": rnd,
    }
