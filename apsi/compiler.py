"""Closure compiler with *unit erasure*.

Hot loops (evolve bodies and the functions they call) are compiled from the AST
into nested Python closures. Units are verified once — by the static checker and
by the first, fully checked evaluation — and then erased: the compiled code runs
on raw SI magnitudes, so physical units cost nothing inside the loop.
"""
import operator

import numpy as np

from . import ast as A
from . import ops
from .autodiff import Node
from .units import Quantity
from .values import Function, StructInstance, Record, RangeVal, BoundMethod


class Unsupported(Exception):
    pass


def strip(v):
    return v.v if isinstance(v, Quantity) else v


_BIN = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
        "//": operator.floordiv, "%": operator.mod, "@": operator.matmul, "**": operator.pow,
        "==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le,
        ">": operator.gt, ">=": operator.ge, "~=": lambda a, b: ops.approx(a, b)}


class ErasedCompiler:
    def __init__(self, interp):
        self.interp = interp
        self.fn_cache = {}

    # ------------------------------------------------------------ functions
    def call(self, f, args, kwargs):
        if isinstance(f, Function):
            return self.call_fn(f, args, kwargs)
        if isinstance(f, BoundMethod):
            return self.call_fn(f.fn, [f.obj] + list(args), kwargs)
        return f(*args, **kwargs)

    def call_fn(self, fn, args, kwargs):
        key = id(fn.body)
        code = self.fn_cache.get(key)
        if code is None:
            code = self.fn_cache[key] = (self.expr(fn.body) if fn.is_expr else self.block(fn.body))
        from .interpreter import Env, ReturnSignal
        env = Env(fn.closure)
        params = fn.params
        for i, p in enumerate(params):
            if i < len(args):
                v = args[i]
            elif p.name in kwargs:
                v = kwargs[p.name]
            elif p.default is not None:
                v = strip(self.interp.eval(p.default, fn.closure))
            else:
                raise Unsupported("missing argument")
            env.vars[p.name] = v
        if fn.shape_params or any(_has_shape_syms(p.type) for p in params):
            raise Unsupported("shape-generic function")
        if fn.is_expr:
            return code(env)
        try:
            code(env)
        except ReturnSignal as r:
            return r.value
        return None

    # ------------------------------------------------------------ statements
    def block(self, stmts):
        codes = [self.stmt(s) for s in stmts]
        if len(codes) == 1:
            return codes[0]

        def run(env):
            for c in codes:
                c(env)
        return run

    def stmt(self, s):
        from .interpreter import Env, ReturnSignal, BreakSignal, ContinueSignal
        t = type(s)
        if t is A.ExprStmt:
            e = self.expr(s.expr)
            return lambda env: (e(env), None)[1]
        if t is A.Deriv:
            e = self.expr(s.expr)
            name, stack = s.name, self.interp.deriv_stack

            def run(env):
                stack[-1][name] = e(env)
            return run
        if t is A.Let:
            e = self.expr(s.value)
            if len(s.targets) == 1:
                n = s.targets[0]

                def run(env):
                    env.vars[n] = e(env)
                return run
            names = s.targets

            def run(env):
                for k, v in zip(names, e(env)):
                    env.vars[k] = v
            return run
        if t is A.Assign and isinstance(s.target, A.Name):
            e = self.expr(s.value)
            n = s.target.id

            def run(env):
                env.assign(n, e(env))
            return run
        if t is A.AugAssign and isinstance(s.target, A.Name):
            e = self.expr(s.value)
            f = _BIN[s.op]
            n = s.target.id

            def run(env):
                env.assign(n, f(env.lookup(n), e(env)))
            return run
        if t is A.Return:
            e = self.expr(s.value) if s.value is not None else (lambda env: None)

            def run(env):
                raise ReturnSignal(e(env))
            return run
        if t is A.If:
            branches = [(self.expr(c), self.block(b)) for c, b in s.branches]
            else_b = self.block(s.else_body) if s.else_body is not None else None

            def run(env):
                for c, b in branches:
                    if c(env):
                        b(Env(env))
                        return
                if else_b is not None:
                    else_b(Env(env))
            return run
        if t is A.For:
            it = self.expr(s.iter)
            body = self.block(s.body)
            names = s.targets

            def run(env):
                for item in it(env):
                    e2 = Env(env)
                    if len(names) == 1:
                        e2.vars[names[0]] = item
                    else:
                        for k, v in zip(names, item):
                            e2.vars[k] = v
                    try:
                        body(e2)
                    except BreakSignal:
                        break
                    except ContinueSignal:
                        continue
            return run
        if t is A.Pass:
            return lambda env: None
        raise Unsupported(t.__name__)

    # ------------------------------------------------------------ expressions
    def expr(self, n):
        t = type(n)
        if t is A.Num:
            if n.unit is not None:
                v = n.value * n.unit[1]
            else:
                v = n.value
            return lambda env: v
        if t in (A.Str, A.Bool):
            v = n.value
            return lambda env: v
        if t is A.NoneLit:
            return lambda env: None
        if t is A.Name:
            name = n.id

            def look(env):
                e = env
                while e is not None:
                    vs = e.vars
                    if name in vs:
                        v = vs[name]
                        return v.v if v.__class__ is Quantity else v
                    e = e.parent
                raise Unsupported(name)
            return look
        if t is A.Binary:
            f = _BIN.get(n.op)
            if f is None:
                raise Unsupported(n.op)
            a, b = self.expr(n.left), self.expr(n.right)
            return lambda env: f(a(env), b(env))
        if t is A.Unary:
            a = self.expr(n.operand)
            if n.op == "-":
                return lambda env: -a(env)
            if n.op == "+":
                return a
            return lambda env: not a(env)
        if t is A.Logic:
            a, b = self.expr(n.left), self.expr(n.right)
            if n.op == "and":
                return lambda env: b(env) if a(env) else False
            return lambda env: True if a(env) else b(env)
        if t is A.IfExpr:
            c, a, b = self.expr(n.cond), self.expr(n.a), self.expr(n.b)
            return lambda env: a(env) if c(env) else b(env)
        if t is A.Call:
            f = self.expr(n.func)
            args = [self.expr(x) for x in n.args]
            kws = [(k, self.expr(v)) for k, v in n.kwargs]
            call = self.call
            if not kws:
                return lambda env: call(f(env), [a(env) for a in args], {})
            return lambda env: call(f(env), [a(env) for a in args], {k: v(env) for k, v in kws})
        if t is A.Attr:
            o = self.expr(n.obj)
            name = n.name
            get = self.interp.get_attr
            return lambda env: strip(get(o(env), name))
        if t is A.Index:
            o = self.expr(n.obj)
            idx = self.index(n.index)
            return lambda env: o(env)[idx(env)]
        if t is A.ListLit:
            items = [self.expr(x) for x in n.items]
            return lambda env: [i(env) for i in items]
        if t is A.TupleLit:
            items = [self.expr(x) for x in n.items]
            return lambda env: tuple(i(env) for i in items)
        if t is A.As:
            e = self.expr(n.expr)
            scale, dim = n.unit[1], n.unit[2]
            if dim.dimensionless:
                return lambda env: e(env) / scale
            return e
        if t is A.Range:
            a, b = self.expr(n.start), self.expr(n.end)
            inc = n.inclusive
            return lambda env: RangeVal(a(env), b(env), inc)
        raise Unsupported(t.__name__)

    def index(self, n):
        if isinstance(n, A.Slice):
            parts = [self.expr(x) if x is not None else (lambda env: None) for x in (n.lo, n.hi, n.step)]
            return lambda env: slice(*(p(env) for p in parts))
        if isinstance(n, A.TupleLit):
            parts = [self.index(x) for x in n.items]
            return lambda env: tuple(p(env) for p in parts)
        return self.expr(n)


def _has_shape_syms(t):
    return isinstance(t, A.TName) and t.name in ("Tensor", "Vec", "Mat", "Matrix") and \
        any(isinstance(a, A.TName) and not a.args for a in t.args)
