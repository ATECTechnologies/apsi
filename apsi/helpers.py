"""Built-in helpers added in 2.4.4: numerically safe maths and everyday list utilities.

Numerically safe (no overflow to inf, no NaN where a finite answer exists):
    log1p(x)  expm1(x)          log(1 + x), e^x - 1 accurate for tiny x (1e-12 keeps every digit)
    hypot(a, b, ...)            √(a² + b² + …) without overflow or underflow in the squares
    logsumexp(xs)               log(Σ eˣ) for large or very negative xs (softmax denominators)
    safe_div(a, b, default=0)   a / b, or default when b is 0 (element-wise for tensors)
    isnan(x)  isfinite(x)  nan_to_num(x, nan=0, posinf=…, neginf=…)
    clamp(x, lo, hi)  lerp(a, b, t)

Everyday utilities:
    chunks(xs, n)               [[x0..xn-1], [xn..], …]
    windows(xs, n)              every run of n neighbours (sliding window)
    pairwise(xs)                [(x0, x1), (x1, x2), …]
    group_by(xs, key)           {key(x): [x, …]}
    count_by(xs, key=none)      {value: count}
    product(a, b, …)            every combination (Cartesian product)
    combinations(xs, k)  permutations(xs, k=none)
    argsort(xs, reverse=false)  positions that sort xs
    median(xs)  mode(xs)
    timeit(f, repeat=5)         record(best, median, mean, runs) in seconds
"""
import itertools
import math
import statistics
import time

import numpy as np

from . import ops
from .units import Quantity


def _arr(x):
    return np.asarray(ops.raw(x), dtype=float) if not isinstance(x, (int, float)) else x


def log1p(x):
    if isinstance(x, (int, float)):
        return math.log1p(x)
    return ops.py(np.log1p(np.asarray(ops.raw(x), dtype=float)))


def expm1(x):
    if isinstance(x, (int, float)):
        return math.expm1(x)
    return ops.py(np.expm1(np.asarray(ops.raw(x), dtype=float)))


def hypot(*xs):
    """√(Σ x²) without overflow/underflow; works with quantities of the same unit and with tensors."""
    if len(xs) == 1 and not isinstance(xs[0], (int, float, Quantity)):
        v = np.asarray(ops.raw(xs[0]), dtype=float)
        m = np.max(np.abs(v)) if v.size else 0.0
        return 0.0 if m == 0 else float(m * np.sqrt(np.sum((v / m) ** 2)))
    if any(isinstance(x, Quantity) for x in xs):
        dims = {x.dim for x in xs if isinstance(x, Quantity)}
        if len(dims) > 1 or not all(isinstance(x, Quantity) for x in xs):
            from .errors import UnitError
            raise UnitError("hypot() needs every argument in the same units")
        d = dims.pop()
        return Quantity(math.hypot(*[float(x.v) for x in xs]), d)
    if all(isinstance(x, (int, float)) for x in xs):
        return math.hypot(*xs)
    return ops.py(np.hypot.reduce(np.broadcast_arrays(*[np.asarray(ops.raw(x), float) for x in xs])))


def logsumexp(xs, axis=None):
    v = np.asarray(ops.raw(xs) if not isinstance(xs, list) else xs, dtype=float)
    m = np.max(v, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    r = np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)) + m
    return ops.py(np.squeeze(r, axis=axis) if axis is not None else r.reshape(()))


def safe_div(a, b, default=0.0):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return default if b == 0 else a / b
    a2, b2 = np.asarray(ops.raw(a), float), np.asarray(ops.raw(b), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b2 == 0, default, a2 / np.where(b2 == 0, 1.0, b2))
    return ops.py(r)


def isnan(x):
    if isinstance(x, (int, float)):
        return x != x
    return ops.py(np.isnan(np.asarray(ops.raw(x), float)))


def isfinite(x):
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return ops.py(np.isfinite(np.asarray(ops.raw(x), float)))


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    return ops.py(np.nan_to_num(np.asarray(ops.raw(x), float), nan=nan, posinf=posinf, neginf=neginf))


def clamp(x, lo, hi):
    if isinstance(x, (int, float)) and isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return lo if x < lo else hi if x > hi else x
    return ops.clip(x, lo, hi) if hasattr(ops, "clip") else ops.py(np.clip(ops.raw(x), lo, hi))


def lerp(a, b, t):
    return a + (b - a) * t


# ------------------------------------------------------------------------------------------------ lists
def _list(xs):
    if isinstance(xs, np.ndarray):
        return list(xs)
    return list(xs)


def chunks(xs, n):
    n = int(n)
    if n < 1:
        raise ValueError("chunks(): n must be at least 1")
    xs = _list(xs)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def windows(xs, n):
    n = int(n)
    xs = _list(xs)
    return [xs[i:i + n] for i in range(0, max(0, len(xs) - n + 1))]


def pairwise(xs):
    xs = _list(xs)
    return list(zip(xs, xs[1:]))


def group_by(xs, key):
    out = {}
    for x in _list(xs):
        out.setdefault(key(x), []).append(x)
    return out


def count_by(xs, key=None):
    out = {}
    for x in _list(xs):
        k = key(x) if key is not None else x
        out[k] = out.get(k, 0) + 1
    return out


def product(*lists, repeat=1):
    return [tuple(t) for t in itertools.product(*[_list(x) for x in lists], repeat=int(repeat))]


def combinations(xs, k):
    return [tuple(t) for t in itertools.combinations(_list(xs), int(k))]


def permutations(xs, k=None):
    return [tuple(t) for t in itertools.permutations(_list(xs), None if k is None else int(k))]


def argsort(xs, reverse=False):
    v = _list(xs)
    return sorted(range(len(v)), key=lambda i: v[i], reverse=bool(reverse))


def median(xs):
    if isinstance(xs, np.ndarray):
        return ops.py(np.median(xs))
    return statistics.median(_list(xs))


def mode(xs):
    return statistics.mode(_list(xs))


def timeit(f, repeat=5, warmup=1):
    """Run f() repeat times after warm-up runs; seconds as a record(best, median, mean, runs)."""
    from .values import Record
    for _ in range(int(warmup)):
        f()
    runs = []
    for _ in range(int(repeat)):
        t = time.perf_counter()
        f()
        runs.append(time.perf_counter() - t)
    return Record(best=min(runs), median=statistics.median(runs), mean=sum(runs) / len(runs), runs=runs)


def builtins():
    return {"log1p": log1p, "expm1": expm1, "hypot": hypot, "logsumexp": logsumexp, "safe_div": safe_div,
            "isnan": isnan, "isfinite": isfinite, "nan_to_num": nan_to_num, "clamp": clamp, "lerp": lerp,
            "chunks": chunks, "windows": windows, "pairwise": pairwise, "group_by": group_by,
            "count_by": count_by, "product": product, "combinations": combinations,
            "permutations": permutations, "argsort": argsort, "median": median, "mode": mode, "timeit": timeit}
