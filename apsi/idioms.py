"""Loop idioms that both engines replace with a library kernel.

Matrix multiplication written as three nested loops is the most common example: the loops are easy to read and
check, but even compiled to machine code they reach a few GFLOP/s, while a tuned BLAS kernel (cache blocking, SIMD,
all cores) reaches tens to hundreds. The compiler recognises these shapes of loop nest:

    for i in 0..M:                       for i in 0..M:                      for i in 0..M:
        for k in 0..K:                       for j in 0..N:                      for j in 0..N:
            let t = a[i, k]                      for k in 0..K:                      var s = 0.0
            for j in 0..N:                           c[i, j] += a[i, k] * b[k, j]        for k in 0..K:
                c[i, j] += t * b[k, j]                                                   s += a[i, k] * b[k, j]
                                                                                     c[i, j] = s

(any order of the three loops for the `+=` forms, `parallel for` on the outer loop, `range(X)` as well as `0..X`,
the two factors in either order) and runs `c[:M, :N] += a[:M, :K] @ b[:K, :N]` (or `=` for the last form) with
BLAS instead. The loops stay in the program as the fallback and run unchanged whenever the fast path does not
apply exactly: arrays that are not 2-D float64/float32 of one type, bounds outside the arrays (so the loop raises
the same error it always did), or `c` sharing memory with `a` or `b`. Results equal the loops up to the order in
which floating-point products are added (differences in the last bits, like any optimised BLAS).

APSI_IDIOMS=0 turns the replacement off.
"""
import os

from . import ast as A

ENABLED = os.environ.get("APSI_IDIOMS", "1") != "0"


class Gemm:
    __slots__ = ("a", "b", "c", "M", "N", "K", "mode")

    def __init__(self, a, b, c, M, N, K, mode):
        self.a, self.b, self.c, self.M, self.N, self.K, self.mode = a, b, c, M, N, K, mode

    def __repr__(self):
        op = "+=" if self.mode == 0 else "="
        return f"Gemm({self.c}[:M,:N] {op} {self.a}[:M,:K] @ {self.b}[:K,:N])"


def _loop(s, need_body_len=None):
    """(var, end expression, body) of `for v in 0..X` / `for v in range(X)` with nothing unusual, else None."""
    if not isinstance(s, A.For) or s.else_body or s.is_async or len(s.targets) != 1 \
            or not isinstance(s.targets[0], str):
        return None
    it = s.iter
    if isinstance(it, A.Range):
        if it.inclusive or not (isinstance(it.start, A.Num) and it.start.value == 0 and it.start.unit is None
                                and isinstance(it.start.value, int)):
            return None
        end = it.end
    elif isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "range" and len(it.args) == 1 \
            and not it.kwargs:
        end = it.args[0]
    else:
        return None
    if need_body_len is not None and len(s.body) != need_body_len:
        return None
    return s.targets[0], end, s.body


_PURE_CALLS = {"len", "min", "max", "int"}


def _names(e, out):
    """Collect the names an end expression reads; False when it could have side effects."""
    if isinstance(e, A.Name):
        out.add(e.id)
        return True
    if isinstance(e, A.Num):
        return e.unit is None
    if isinstance(e, A.Attr):
        return _names(e.obj, out)
    if isinstance(e, A.Index):
        return _names(e.obj, out) and _names(e.index, out)
    if isinstance(e, A.TupleLit):
        return all(_names(x, out) for x in e.items)
    if isinstance(e, A.Binary):
        return e.op in ("+", "-", "*", "//") and _names(e.left, out) and _names(e.right, out)
    if isinstance(e, A.Call):
        return isinstance(e.func, A.Name) and e.func.id in _PURE_CALLS and not e.kwargs \
            and all(_names(x, out) for x in e.args) and (out.add(e.func.id) or True)
    return False


def _idx2(e):
    """(array name, first index name, second index name) of `x[p, q]`, else None."""
    if isinstance(e, A.Index) and isinstance(e.obj, A.Name) and isinstance(e.index, A.TupleLit) \
            and len(e.index.items) == 2 and all(isinstance(x, A.Name) for x in e.index.items):
        return e.obj.id, e.index.items[0].id, e.index.items[1].id
    return None


def _product(e, hoisted=None):
    """The two factors of `x * y` as (a-ref, b-ref): each an _idx2 triple or ('hoist', name)."""
    if not (isinstance(e, A.Binary) and e.op == "*"):
        return None
    out = []
    for f in (e.left, e.right):
        if hoisted is not None and isinstance(f, A.Name) and f.id == hoisted:
            out.append(("hoist", f.id))
        else:
            r = _idx2(f)
            if r is None:
                return None
            out.append(r)
    return out


def _roles(c_ref, factors, hoist=None):
    """Assign a / b from the factors given c[r, col]: a[r, red] and b[red, col]. Returns (a, b, r, col, red)."""
    c, r, col = c_ref
    for fa, fb in (factors, factors[::-1]):
        if fa[0] == "hoist":
            if hoist is None:
                continue
            fa = hoist
        if fb[0] == "hoist":
            continue
        a, ar, ared = fa
        b, bred, bcol = fb
        if ar == r and bcol == col and ared == bred:
            return a, b, r, col, ared
    return None


def match_gemm(s):
    """A Gemm for a loop nest that is a matrix product, else None."""
    if not ENABLED:
        return None
    l1 = _loop(s, 1)
    if l1 is None:
        return None
    v1, e1, b1 = l1
    l2 = _loop(b1[0])
    if l2 is None:
        return None
    v2, e2, b2 = l2
    hoist = None
    mode = 0
    if len(b2) == 1:                                   # perfect nest: c[..] += x * y
        l3 = _loop(b2[0], 1)
        if l3 is None:
            return None
        v3, e3, b3 = l3
        st = b3[0]
        if not (isinstance(st, A.AugAssign) and st.op == "+"):
            return None
        c_ref = _idx2(st.target)
        factors = _product(st.value)
    elif len(b2) == 2 and isinstance(b2[0], A.Let) and len(b2[0].targets) == 1 \
            and isinstance(b2[0].targets[0], str) and b2[0].type is None:
        name = b2[0].targets[0]
        init = b2[0].value
        l3 = _loop(b2[1], 1) if isinstance(b2[1], A.For) else None
        if l3 is None:
            return None
        v3, e3, b3 = l3
        st = b3[0]
        if not (isinstance(st, A.AugAssign) and st.op == "+"):
            return None
        if not b2[0].mutable and _idx2(init) is not None:          # let t = a[i, k]; c[i, j] += t * b[k, j]
            hoist = _idx2(init)
            c_ref = _idx2(st.target)
            factors = _product(st.value, hoisted=name)
            if factors is None or not any(f[0] == "hoist" for f in factors):
                return None
            if {hoist[1], hoist[2]} != {v1, v2}:
                return None
        else:
            return None
    elif len(b2) == 3 and isinstance(b2[0], A.Let) and b2[0].mutable and len(b2[0].targets) == 1 \
            and isinstance(b2[0].targets[0], str) and (b2[0].type is None or (
                isinstance(b2[0].type, A.TName) and b2[0].type.name in ("f64", "f32", "f16", "bf16"))):
        # var s = 0.0; for k: s += a[i, k] * b[k, j]; c[i, j] = s
        name = b2[0].targets[0]
        init = b2[0].value
        if not (isinstance(init, A.Num) and init.value == 0 and init.unit is None):
            return None
        l3 = _loop(b2[1], 1)
        if l3 is None:
            return None
        v3, e3, b3 = l3
        st = b3[0]
        if not (isinstance(st, A.AugAssign) and st.op == "+" and isinstance(st.target, A.Name)
                and st.target.id == name):
            return None
        fin = b2[2]
        if not (isinstance(fin, A.Assign) and not fin.more and isinstance(fin.value, A.Name)
                and fin.value.id == name):
            return None
        c_ref = _idx2(fin.target)
        factors = _product(st.value)
        mode = 1
        if c_ref is None or {c_ref[1], c_ref[2]} != {v1, v2}:
            return None
    else:
        return None
    if c_ref is None or factors is None:
        return None
    roles = _roles(c_ref, factors, hoist)
    if roles is None:
        return None
    a, b, r, col, red = roles
    loops = {v1: e1, v2: e2, v3: e3}
    if len(loops) != 3 or {r, col, red} != set(loops):
        return None
    if mode == 1 and red != v3:
        return None
    if hoist is not None and v3 != col:
        return None
    c = c_ref[0]
    if c in (a, b) or len({v1, v2, v3, a, b, c}) < 5 + (a != b):
        return None
    locals_ = {v1, v2, v3}
    if hoist is not None or mode == 1:
        locals_.add(b2[0].targets[0])
    if locals_ & {a, b, c}:
        return None
    for e in (e1, e2, e3):
        used = set()
        if not _names(e, used) or used & locals_ or c in used:
            return None
    return Gemm(a, b, c, loops[r], loops[col], loops[red], mode)


def gemm(mode, a, b, c, M, N, K):
    """Run the matched product with BLAS. True when done; False means "run the loops" (nothing was changed)."""
    import numpy as np
    try:
        if not (type(a) is np.ndarray and type(b) is np.ndarray and type(c) is np.ndarray):
            return False
        if a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or not (a.dtype == b.dtype == c.dtype) \
                or a.dtype not in (np.float64, np.float32) or not c.flags.writeable:
            return False
        for v in (M, N, K):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                return False
        M, N, K = int(M), int(N), int(K)
        if M <= 0:
            return True                                   # the loops would not run at all
        if K > 0 and (M > a.shape[0] or K > a.shape[1]):
            return False                                  # the loops raise the index error themselves
        if N <= 0:
            return True
        if M > c.shape[0] or N > c.shape[1] or (K > 0 and (K > b.shape[0] or N > b.shape[1])):
            return False
        if np.may_share_memory(c, a) or np.may_share_memory(c, b):
            return False
        cv = c[:M, :N]
        if K <= 0:
            if mode == 1:
                cv[...] = 0
            return True
        if mode == 1:
            np.matmul(a[:M, :K], b[:K, :N], out=cv)
        else:
            cv += a[:M, :K] @ b[:K, :N]
        return True
    except Exception:                                     # noqa: BLE001 - anything odd: the loops decide
        return False
