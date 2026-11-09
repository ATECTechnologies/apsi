"""Rewrites that let ordinary list-style code reach the native compiler.

Python programmers write matrices as lists of lists:

    var c = [[0.0 for j in 0..n] for i in 0..n]
    for i in 0..n:
        for k in 0..n:
            c[i][j] += a[i][k] * b[k][j]

Lists hold separate boxed objects and cannot be compiled to machine code, so before 2.4.4 such a function quietly
ran on the bytecode engine (about 500× slower). Now:

* list (and nested-list) arguments made only of numbers are passed to native code as tensors, and the results
  are written back into the caller's lists, so the caller sees the same lists change as before;
* lists filled with one float constant — `[[0.0 for j in 0..m] for i in 0..n]`, `[0.0 for i in 0..n]`,
  `[0.0] * n`, `[[1.5] * m for i in 0..n]` — become `zeros` / `full` tensors inside the native function.

Indexing (`c[i][j]`, `c[i][j] += …`), `len()` and loops behave the same on both. The rewrite happens only on the
copy of the function that is compiled; the bytecode version keeps the original code.
"""
import copy

import numpy as np

from .. import ast as A


def _const(e):
    if isinstance(e, A.Num) and e.unit is None:
        return e.value
    if isinstance(e, A.Unary) and e.op == "-" and isinstance(e.operand, A.Num) and e.operand.unit is None:
        return -e.operand.value
    if isinstance(e, A.Bool):
        return bool(e.value)
    return None


def _length(it):
    """Number of items of `a..b`, `a..=b`, `range(n)`, `range(a, b)` as an expression (None if unknown)."""
    def num(v, node):
        return A.Num(v, str(v), None, line=node.line, col=node.col)

    def minus(b, a):
        if _const(a) == 0:
            return b
        return A.Binary("-", b, a, line=b.line, col=b.col)
    if isinstance(it, A.Range):
        n = minus(it.end, it.start)
        return A.Binary("+", n, num(1, it), line=it.line, col=it.col) if it.inclusive else n
    if isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "range" and not it.kwargs:
        if len(it.args) == 1:
            return it.args[0]
        if len(it.args) == 2:
            return minus(it.args[1], it.args[0])
    return None


def _uses(node, names):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, A.Name) and n.id in names:
            return True
        for f in getattr(n, "_fields", ()):
            v = getattr(n, f, None)
            if isinstance(v, A.AST):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(x for x in v if isinstance(x, A.AST))
    return False


def _filled(e):
    """(shape expressions, value) when `e` builds a list filled with one constant, else None."""
    if isinstance(e, A.ListComp) and not e.more and e.cond is None:
        n = _length(e.iter)
        if n is None or _uses(e.elem, set(e.targets)):
            return None
        c = _const(e.elem)
        if isinstance(c, float):                    # only float fills: full() makes float tensors
            return [n], c
        inner = _filled(e.elem)
        if inner is not None:
            return [n] + inner[0], inner[1]
        return None
    if isinstance(e, A.Binary) and e.op == "*":
        for lst, n in ((e.left, e.right), (e.right, e.left)):
            if isinstance(lst, A.ListLit) and len(lst.items) == 1:
                c = _const(lst.items[0])
                if isinstance(c, float):
                    return [n], c
    return None


def _make(shape, value, node):
    kw = dict(line=node.line, col=node.col)
    if isinstance(value, float) and value == 0.0:
        return A.Call(A.Name("zeros", **kw), list(shape), [], **kw)
    v = A.Num(float(value), repr(float(value)), None, **kw) if isinstance(value, float) else \
        A.Num(value, str(value), None, **kw) if not isinstance(value, bool) else A.Bool(value, **kw)
    return A.Call(A.Name("full", **kw), [A.TupleLit(list(shape), **kw), v], [], **kw)


def _rewrite(node):
    if isinstance(node, (A.ListComp, A.Binary)):
        f = _filled(node)
        if f is not None:
            return _make(f[0], f[1], node)
    for name in getattr(node, "_fields", ()):
        v = getattr(node, name, None)
        if isinstance(v, A.AST):
            setattr(node, name, _rewrite(v))
        elif isinstance(v, list):
            setattr(node, name, [_rewrite(x) if isinstance(x, A.AST) else
                                 [(_rewrite(y) if isinstance(y, A.AST) else y) for y in x] if isinstance(x, list) else x
                                 for x in v])
    return node


def lower(fndef):
    """A rewritten copy of a function definition for the native compiler (the original is left untouched)."""
    try:
        return _rewrite(copy.deepcopy(fndef))
    except Exception:                                   # noqa: BLE001 - never block a call because of this
        return fndef


# ------------------------------------------------------------------------------------------- list arguments
def as_array(v):
    """A tensor for a list (or nested lists) made only of numbers with a rectangular shape, else None."""
    if not isinstance(v, list) or not v:
        return None
    try:
        a = np.array(v)
    except (ValueError, TypeError):
        return None
    if a.dtype == object or a.dtype.kind not in "fiub" or a.ndim > 8:
        return None
    if a.dtype.kind == "b":
        return a
    if a.dtype.kind in "iu":
        if any(isinstance(x, float) for x in _leaves(v)):
            a = a.astype(np.float64)
        else:
            a = a.astype(np.int64)
    else:
        a = a.astype(np.float64)
    return a


def _leaves(v):
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(x)
        else:
            yield x


def write_back(lst, arr):
    """Copy a tensor's values back into the nested list it was made from (the caller's list changes in place)."""
    if arr.ndim == 1:
        conv = float if arr.dtype.kind == "f" else (bool if arr.dtype.kind == "b" else int)
        for i, x in enumerate(arr.tolist()):
            if lst[i] != x or type(lst[i]) is not type(x):
                lst[i] = conv(x)
        return
    for i in range(arr.shape[0]):
        write_back(lst[i], arr[i])
