"""Unroll-and-jam of independent outer loops, for exact vectorisation.

A loop such as the force loop of an n-body simulation

    for i in 0..n:
        var fx = 0.0
        for j in 0..n:
            fx += (x[j] - x[i]) * m[j] / ...
        ax[i] = fx

cannot be vectorised along j without changing the order of the additions into fx (and so the result). The
iterations of i, however, are independent. This pass rewrites the nest to work on four values of i at once:

    var i0 = 0
    while i0 + 4 <= n:
        var fx_0 = 0.0  ...  var fx_3 = 0.0
        for j in 0..n:
            fx_0 += ...   fx_1 += ...   fx_2 += ...   fx_3 += ...
        ax[i0] = fx_0  ...  ax[i0 + 3] = fx_3
        i0 += 4
    for i in i0..n:            # the last n % 4 iterations, unchanged
        ...

Each lane does exactly the same operations in the same order as before, so results are bit-for-bit identical,
but the four lanes are identical instruction streams on different data, which LLVM's SLP vectoriser packs into
SIMD instructions (4 doubles in an AVX2 register). This is the "outer-loop vectorisation" C compilers do.

The rewrite is applied only when it is provably safe: the outer loop counts by one; its body is straight-line code
around exactly one inner counting loop whose bounds do not depend on the outer variable; every tensor written in
the nest is accessed only at the outer index (`a[i]` or `a[i, …]`); every scalar assigned is declared inside the
nest; and only pure built-in functions are called. APSI_JAM=0 turns it off.
"""
import itertools
import os

from .. import ast as A

ENABLED = os.environ.get("APSI_JAM", "1") != "0"
LANES = 4
PURE = {"sqrt", "abs", "min", "max", "exp", "log", "log2", "log10", "sin", "cos", "tan", "tanh", "atan", "atan2",
        "floor", "ceil", "float", "int", "round", "len", "pow", "hypot", "sinh", "cosh", "asin", "acos", "erf"}
_EXPR = (A.Num, A.Name, A.Unary, A.Binary, A.Logic, A.IfExpr, A.Call, A.Index, A.Attr, A.TupleLit, A.Chain,
         A.Bool)
_cache = {}
_ids = itertools.count(1)


def jam(loop):
    """Replacement statements for a For node, or None when the rewrite does not apply. Cached per node."""
    if not ENABLED:
        return None
    key = id(loop)
    hit = _cache.get(key)
    if hit is not None and hit[0] is loop:
        return hit[1]
    try:
        out = _jam(loop)
    except Exception:                                   # noqa: BLE001 - anything unexpected: leave the loop alone
        out = None
    _cache[key] = (loop, out)
    return out


def _counting(loop):
    """(var, start, stop) of `for v in a..b` / `for v in range([a,] b)` counting by one, else None."""
    if not isinstance(loop, A.For) or loop.else_body or loop.is_async or len(loop.targets) != 1 \
            or not isinstance(loop.targets[0], str):
        return None
    it = loop.iter
    if isinstance(it, A.Range) and not it.inclusive:
        return loop.targets[0], it.start, it.end
    if isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "range" and not it.kwargs \
            and 1 <= len(it.args) <= 2:
        if len(it.args) == 1:
            return loop.targets[0], A.Num(0, "0", None, line=loop.line, col=loop.col), it.args[0]
        return loop.targets[0], it.args[0], it.args[1]
    return None


def _pure_expr(e, names):
    """True when e is a side-effect-free expression; collects the names it reads."""
    if e is None:
        return True
    if not isinstance(e, _EXPR):
        return False
    if isinstance(e, A.Name):
        names.add(e.id)
        return True
    if isinstance(e, A.Num):
        return True
    if isinstance(e, A.Call):
        if not (isinstance(e.func, A.Name) and e.func.id in PURE) or e.kwargs:
            return False
        return all(_pure_expr(a, names) for a in e.args)
    if isinstance(e, A.Chain):
        return all(_pure_expr(a, names) for a in e.operands)
    if isinstance(e, A.TupleLit):
        return all(_pure_expr(a, names) for a in e.items)
    if isinstance(e, A.Attr):
        return _pure_expr(e.obj, names)
    if isinstance(e, A.Index):
        return isinstance(e.obj, (A.Name, A.Attr)) and _pure_expr(e.obj, names) and _pure_expr(e.index, names)
    return all(_pure_expr(getattr(e, f), names) for f in e._fields if isinstance(getattr(e, f), A.AST))


def _stmts_ok(body, locals_, allow_if):
    """Straight-line statements (optionally with if/else of the same) with pure expressions."""
    for s in body:
        if isinstance(s, A.Let):
            if len(s.targets) != 1 or not isinstance(s.targets[0], str) or not _pure_expr(s.value, set()):
                return False
            locals_.add(s.targets[0])
        elif isinstance(s, (A.Assign, A.AugAssign)):
            if isinstance(s, A.Assign) and s.more:
                return False
            if not _pure_expr(s.value, set()):
                return False
            t = s.target
            if isinstance(t, A.Index):
                if not (isinstance(t.obj, A.Name) and _pure_expr(t.index, set())):
                    return False
            elif not isinstance(t, A.Name):
                return False
        elif isinstance(s, A.If) and allow_if:
            for cond, b in s.branches:
                if not _pure_expr(cond, set()) or not _stmts_ok(b, locals_, True):
                    return False
            if s.else_body and not _stmts_ok(s.else_body, locals_, True):
                return False
        elif isinstance(s, A.Pass):
            continue
        else:
            return False
    return True


def _walk(nodes):
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, A.AST):
            yield n
            stack.extend(getattr(n, f) for f in n._fields)
        elif isinstance(n, (list, tuple)):
            stack.extend(n)


def _first_index(idx):
    return idx.items[0] if isinstance(idx, A.TupleLit) and idx.items else idx


def _jam(loop):
    if loop.parallel:
        return None
    outer = _counting(loop)
    if outer is None:
        return None
    i, start, stop = outer
    body = loop.body
    fors = [k for k, s in enumerate(body) if isinstance(s, A.For)]
    if len(fors) != 1:
        return None
    k = fors[0]
    pre, inner, post = body[:k], body[k], body[k + 1:]
    if inner.parallel:
        return None
    inn = _counting(inner)
    if inn is None:
        return None
    j, jstart, jstop = inn
    locals_ = {i}
    if not _stmts_ok(pre, locals_, False) or not _stmts_ok(inner.body, locals_, True) \
            or not _stmts_ok(post, locals_, False):
        return None
    if j in locals_:
        return None
    front = list(pre) + list(inner.body)               # runs for all lanes before any lane's post statements
    lane_arrays, shared = set(), set()                 # tensors written at [i, ...]; state written only by post
    for part, is_post in ((front, False), (list(post), True)):
        for n in _walk(part):
            if isinstance(n, (A.NamedExpr, A.YieldExpr, A.Await, A.Lambda, A.ListComp, A.GenExp)):
                return None
            if not isinstance(n, (A.Assign, A.AugAssign)):
                continue
            t = n.target
            if isinstance(t, A.Name):
                if t.id in (i, j):
                    return None
                if t.id in locals_:
                    continue
                if not is_post:
                    return None
                shared.add(t.id)                        # e.g. a total over i: post runs lane by lane, in order
            else:
                first = _first_index(t.index)
                if isinstance(first, A.Name) and first.id == i:
                    lane_arrays.add(t.obj.id)
                elif is_post:
                    shared.add(t.obj.id)                # e.g. sums[cluster] += x[i], in the original order
                else:
                    return None
    if lane_arrays & shared:
        return None
    for n in _walk(front):                              # pre and inner never see what post statements change
        if isinstance(n, A.Name) and n.id in shared:
            return None
    for n in _walk(front + list(post)):                 # per-lane tensors are only touched at the outer index
        if isinstance(n, A.Index) and isinstance(n.obj, A.Name) and n.obj.id in lane_arrays:
            first = _first_index(n.index)
            if not (isinstance(first, A.Name) and first.id == i):
                return None
        if isinstance(n, A.Call):
            for x in n.args:
                if isinstance(x, A.Name) and (x.id in lane_arrays or x.id in shared):
                    return None
    written = lane_arrays | shared
    pre_vars = {t for st in pre if isinstance(st, A.Let) for t in st.targets}
    carried = any(isinstance(n, (A.Assign, A.AugAssign)) and isinstance(n.target, A.Name)
                  and n.target.id in pre_vars for n in _walk(list(inner.body)))
    if not carried:
        return None                                     # no chain of dependent updates along j: plain
                                                        # vectorisation of the inner loop is already best
    # loop bounds: pure, not changed by the nest, inner bounds independent of the outer variable
    bound_names = set()
    for e in (start, stop, jstart, jstop):
        if not _pure_expr(e, bound_names):
            return None
    if bound_names & locals_ or bound_names & set(written):
        return None
    L, C = loop.line, loop.col
    base = f"__jam{next(_ids)}"

    def nm(x):
        return A.Name(x, line=L, col=C)

    def num(v):
        return A.Num(v, str(v), None, line=L, col=C)

    def lane_names(kk):
        return {n: f"{n}__l{kk}" for n in locals_}

    out = [A.Let([base], None, start, True, line=L, col=C)]
    lanes = [lane_names(kk) for kk in range(LANES)]
    wbody = []
    for kk in range(LANES):
        wbody.append(A.Let([lanes[kk][i]], None, A.Binary("+", nm(base), num(kk), line=L, col=C) if kk else nm(base),
                           False, line=L, col=C))
    for s in pre:
        wbody += [_clone(s, lanes[kk]) for kk in range(LANES)]
    ibody = []
    for s in inner.body:
        ibody += [_clone(s, lanes[kk]) for kk in range(LANES)]
    wbody.append(A.For([j], inner.iter, ibody, False, None, None, line=inner.line, col=inner.col))
    for kk in range(LANES):                             # lane by lane: shared updates keep their original order
        wbody += [_clone(st, lanes[kk]) for st in post]
    wbody.append(A.Assign(nm(base), A.Binary("+", nm(base), num(LANES), line=L, col=C), None, line=L, col=C))
    cond = A.Binary("<=", A.Binary("+", nm(base), num(LANES), line=L, col=C), stop, line=L, col=C)
    out.append(A.While(cond, wbody, None, line=L, col=C))
    rest = A.For([i], A.Range(nm(base), stop, False, line=L, col=C), loop.body, False, None, None, line=L, col=C)
    _cache[id(rest)] = (rest, None)                     # the remainder loop stays as it is
    out.append(rest)
    return out


def _clone(node, ren):
    """A copy of a statement or expression with lane-local names renamed."""
    if isinstance(node, A.Name):
        return A.Name(ren.get(node.id, node.id), line=node.line, col=node.col)
    if isinstance(node, A.AST):
        vals = []
        for f in node._fields:
            v = getattr(node, f)
            if isinstance(node, A.Let) and f == "targets":
                v = [ren.get(t, t) if isinstance(t, str) else t for t in v]
            elif isinstance(node, A.For) and f == "targets":
                v = [ren.get(t, t) if isinstance(t, str) else t for t in v]
            else:
                v = _clone_val(v, ren)
            vals.append(v)
        return type(node)(*vals, line=node.line, col=node.col)
    return node


def _clone_val(v, ren):
    if isinstance(v, A.AST):
        return _clone(v, ren)
    if isinstance(v, list):
        return [_clone_val(x, ren) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone_val(x, ren) for x in v)
    return v
