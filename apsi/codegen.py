"""Apsi → Python bytecode compiler (the default execution engine).

The checked AST is translated into Python source, compiled with `compile()`
and executed. Compared with tree-walking this removes almost all interpretive
overhead. Key points:

* Lexical scoping is resolved at compile time; every Apsi declaration gets a
  unique Python name, and `nonlocal` / `global` are emitted where needed.
* Type annotations become specialised checker calls (fast paths for Int,
  Float and physical dimensions).
* Every function gets an *erased twin* (`f__e`) that runs on raw SI numbers
  with all unit checks removed. `evolve` validates units once with the normal
  code, then integrates with the erased code, and falls back to the checked
  code automatically if the erased probe disagrees.
* Runtime errors are mapped back to Apsi source lines.
"""
import keyword
import os
import sys
import traceback
from fractions import Fraction

import numpy as np

from . import ast as A
from . import ops
from .autodiff import Node
from .errors import ApsiError, RuntimeErr, UnitError, ShapeError, TypeCheckError, AssertionFailed
from .fmt import fmt, type_name
from .parser import parse
from .units import Quantity, DIMLESS, DIM_NAMES, dim_name, make
from .values import (StructType, StructInstance, Record, RangeVal, Namespace, BoundMethod,
                     UnitNamespace)

HELPER_PREFIX = "_A"


class _NoDefault:
    def __repr__(self):
        return "<no default>"


NO = _NoDefault()


# =========================================================================== compile-time scopes
class FuncCtx:
    def __init__(self, parent, kind, erased=False, exact=False, name="<module>"):
        self.parent = parent
        self.kind = kind            # "module" | "fn"
        self.erased = erased
        self.exact = exact
        self.name = name
        self.used = set()
        self.nonlocals = set()
        self.globals = set()


class Binding:
    __slots__ = ("py", "ctx", "kind", "erased_py", "mutable")

    def __init__(self, py, ctx, kind="var", erased_py=None, mutable=False):
        self.py, self.ctx, self.kind, self.erased_py, self.mutable = py, ctx, kind, erased_py, mutable


class Scope:
    def __init__(self, parent, ctx, loop=False):
        self.parent = parent
        self.ctx = ctx
        self.names = {}
        # inside a loop of the same python function -> closures must capture per iteration
        self.in_loop = loop or (parent is not None and parent.ctx is ctx and parent.in_loop)

    def lookup(self, name):
        s = self
        while s is not None:
            if name in s.names:
                return s.names[name]
            s = s.parent
        return None


# =========================================================================== code generator
class CodeGen:
    def __init__(self, rt, builtin_names, filename):
        self.rt = rt
        self.builtin_names = set(builtin_names)
        self.filename = filename
        self.lines = []          # (text, apsi_line)
        self.consts = {}         # py name -> object
        self.cur_line = (1, 1)
        self.indent = 0
        self.uid = 0
        self.fn_names = {}       # py fn name -> apsi name (for tracebacks)

    # ------------------------------------------------------------- helpers
    def emit(self, text):
        self.lines.append(("    " * self.indent + text, self.cur_line))

    def const(self, obj, tag="k"):
        self.uid += 1
        name = f"{HELPER_PREFIX}{tag}{self.uid}"
        self.consts[name] = obj
        return name

    def fresh(self, tag):
        self.uid += 1
        return f"{HELPER_PREFIX}{tag}{self.uid}"

    def reserved(self, ctx):
        return self.builtin_names | ctx.used

    def declare(self, scope, name, kind="var", mutable=False, is_param=False):
        ctx = scope.ctx
        base = name
        if keyword.iskeyword(base) or base in ("None", "True", "False") or base.startswith(HELPER_PREFIX):
            base = "u_" + base
        py = base
        taken = ctx.used if is_param else (ctx.used | self.builtin_names)
        k = 1
        while py in taken:
            k += 1
            py = f"{base}_{k}"
        ctx.used.add(py)
        b = Binding(py, ctx, kind, mutable=mutable)
        scope.names[name] = b
        return b

    # ------------------------------------------------------------- name access
    def load(self, name, scope):
        b = scope.lookup(name)
        ctx = scope.ctx
        if b is None:
            ref = name if name in self.builtin_names else name
            return f"{HELPER_PREFIX}S({ref})" if ctx.erased else ref
        if ctx.erased and not b.ctx.erased:
            return f"{HELPER_PREFIX}S({b.py})"
        return b.py

    def store_target(self, name, scope):
        b = scope.lookup(name)
        if b is None:
            raise RuntimeErr(f"'{name}' is not defined")
        ctx = scope.ctx
        if b.ctx is not ctx:
            if b.ctx.kind == "module":
                ctx.globals.add(b.py)
            else:
                ctx.nonlocals.add(b.py)
        return b.py

    # ------------------------------------------------------------- program
    def program(self, prog, module_scope):
        for s in prog.body:
            self.stmt(s, module_scope)

    def block(self, stmts, scope):
        if not stmts:
            self.emit("pass")
            return
        start = len(self.lines)
        for s in stmts:
            self.stmt(s, scope)
        if len(self.lines) == start:
            self.emit("pass")

    # ------------------------------------------------------------- statements
    def stmt(self, s, scope):
        self.cur_line = (s.line, s.col)
        getattr(self, "s_" + type(s).__name__)(s, scope)

    def s_ExprStmt(self, s, scope):
        self.emit(self.expr(s.expr, scope))

    def s_Pass(self, s, scope):
        self.emit("pass")

    def s_Break(self, s, scope):
        self.emit("break")

    def s_Continue(self, s, scope):
        self.emit("continue")

    def s_Let(self, s, scope):
        val = self.expr(s.value, scope)
        if len(s.targets) == 1:
            if s.type is not None and not scope.ctx.erased:
                chk = self.checker(s.type)
                val = f"{chk}({val}, None, {s.targets[0]!r})"
            b = self.declare(scope, s.targets[0], mutable=s.mutable)
            self.emit(f"{b.py} = {val}")
        else:
            tmp = self.fresh("t")
            self.emit(f"{tmp} = {HELPER_PREFIX}unpack({val}, {len(s.targets)})")
            names = [self.declare(scope, n, mutable=s.mutable).py for n in s.targets]
            self.emit(f"{', '.join(names)}, = {tmp}")

    def target(self, t, scope):
        """Return python code for an assignment target, or None for complex targets."""
        if isinstance(t, A.Name):
            return self.store_target(t.id, scope)
        if isinstance(t, A.TupleLit):
            return "(" + ", ".join(self.target(x, scope) for x in t.items) + ",)"
        return None

    def s_Assign(self, s, scope):
        val = self.expr(s.value, scope)
        t = s.target
        if isinstance(t, A.Attr):
            self.emit(f"{HELPER_PREFIX}setattr({self.expr(t.obj, scope)}, {t.name!r}, {val})")
        elif isinstance(t, A.Index):
            self.emit(f"{HELPER_PREFIX}setidx({self.expr(t.obj, scope)}, {self.index(t.index, scope)}, {val})")
        elif isinstance(t, A.TupleLit) and any(not isinstance(x, A.Name) for x in t.items):
            tmp = self.fresh("t")
            self.emit(f"{tmp} = {HELPER_PREFIX}unpack({val}, {len(t.items)})")
            for k, x in enumerate(t.items):
                self._assign_from(x, f"{tmp}[{k}]", scope)
        elif isinstance(t, A.TupleLit):
            tmp = self.fresh("t")
            self.emit(f"{tmp} = {HELPER_PREFIX}unpack({val}, {len(t.items)})")
            self.emit(f"{self.target(t, scope)} = {tmp}")
        else:
            self.emit(f"{self.target(t, scope)} = {val}")

    def _assign_from(self, t, code, scope):
        if isinstance(t, A.Attr):
            self.emit(f"{HELPER_PREFIX}setattr({self.expr(t.obj, scope)}, {t.name!r}, {code})")
        elif isinstance(t, A.Index):
            self.emit(f"{HELPER_PREFIX}setidx({self.expr(t.obj, scope)}, {self.index(t.index, scope)}, {code})")
        else:
            self.emit(f"{self.target(t, scope)} = {code}")

    def s_AugAssign(self, s, scope):
        op = self.binop(s.op)
        val = self.expr(s.value, scope)
        t = s.target
        if isinstance(t, A.Name):
            cur = self.load(t.id, scope)
            py = self.store_target(t.id, scope)
            self.emit(f"{py} = {op(cur, val)}")
        elif isinstance(t, A.Attr):
            o = self.fresh("o")
            self.emit(f"{o} = {self.expr(t.obj, scope)}")
            cur = f"{HELPER_PREFIX}attr({o}, {t.name!r})"
            self.emit(f"{HELPER_PREFIX}setattr({o}, {t.name!r}, {op(cur, val)})")
        else:
            o, i = self.fresh("o"), self.fresh("i")
            self.emit(f"{o} = {self.expr(t.obj, scope)}")
            self.emit(f"{i} = {self.index(t.index, scope)}")
            self.emit(f"{HELPER_PREFIX}setidx({o}, {i}, {op(f'{o}[{i}]', val)})")

    def s_If(self, s, scope):
        first = True
        for cond, body in s.branches:
            c = self.expr(cond, scope)
            self.emit(f"{'if' if first else 'elif'} {HELPER_PREFIX}truth({c}):")
            first = False
            self.indent += 1
            self.block(body, Scope(scope, scope.ctx))
            self.indent -= 1
        if s.else_body is not None:
            self.emit("else:")
            self.indent += 1
            self.block(s.else_body, Scope(scope, scope.ctx))
            self.indent -= 1

    def s_While(self, s, scope):
        self.emit(f"while {HELPER_PREFIX}truth({self.expr(s.cond, scope)}):")
        self.indent += 1
        self.block(s.body, Scope(scope, scope.ctx, loop=True))
        self.indent -= 1

    def s_For(self, s, scope):
        it = s.iter
        if isinstance(it, A.Range):
            src = f"{HELPER_PREFIX}irange({self.expr(it.start, scope)}, {self.expr(it.end, scope)}, {it.inclusive})"
        else:
            src = f"{HELPER_PREFIX}iterable({self.expr(it, scope)})"
        inner = Scope(scope, scope.ctx, loop=True)
        names = [self.declare(inner, n).py for n in s.targets]
        tgt = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
        self.emit(f"for {tgt} in {src}:")
        self.indent += 1
        self.block(s.body, inner)
        self.indent -= 1

    def s_Return(self, s, scope):
        val = self.expr(s.value, scope) if s.value is not None else "None"
        ret = getattr(scope.ctx, "ret_check", None)
        if ret and not scope.ctx.erased:
            val = f"{ret}({val}, {HELPER_PREFIX}b, {'return value of ' + scope.ctx.name + '()'!r})"
        self.emit(f"return {val}")

    def s_Assert(self, s, scope):
        c = s.cond
        msg = self.expr(s.msg, scope) if s.msg is not None else None
        if isinstance(c, A.Binary) and c.op in ("==", "!=", "<", "<=", ">", ">=", "~="):
            l, r = self.fresh("l"), self.fresh("r")
            self.emit(f"{l} = {self.expr(c.left, scope)}")
            self.emit(f"{r} = {self.expr(c.right, scope)}")
            test = self.binop(c.op)(l, r)
            detail = msg or f"'assertion failed: ' + {HELPER_PREFIX}fmt({l}) + ' {c.op} ' + {HELPER_PREFIX}fmt({r})"
        else:
            test = self.expr(c, scope)
            detail = msg or "'assertion failed'"
        self.emit(f"if not {HELPER_PREFIX}all({test}):")
        self.indent += 1
        self.emit(f"raise {HELPER_PREFIX}AssertionFailed({HELPER_PREFIX}fmt({detail}), {s.line}, {s.col})")
        self.indent -= 1

    def s_Use(self, s, scope):
        self.rt.interp.ensure_module(s.path, self.rt.base_dir)
        mods = self.rt.interp.modules
        if s.alias:
            b = self.declare(scope, s.alias)
            self.emit(f"{b.py} = {HELPER_PREFIX}namespace({s.path!r})")
            return
        for name in mods[s.path]:
            b = self.declare(scope, name)
            self.emit(f"{b.py} = {HELPER_PREFIX}mods[{s.path!r}][{name!r}]")
        if "." not in s.path and s.path not in mods[s.path] and scope.lookup(s.path) is None and s.path not in self.rt.interp.globals.vars:
            b = self.declare(scope, s.path)                            # `use mylib` also allows mylib.name(...)
            self.emit(f"{b.py} = {HELPER_PREFIX}namespace({s.path!r})")

    def s_Import(self, s, scope):
        path = s.path if os.path.isabs(s.path) else os.path.join(self.rt.base_dir, s.path)
        tmp = self.fresh("m")
        self.emit(f"{tmp} = {HELPER_PREFIX}import_file({path!r})")
        if s.alias:
            b = self.declare(scope, s.alias)
            self.emit(f"{b.py} = {HELPER_PREFIX}Namespace({os.path.basename(path)!r}, {tmp})")
            return
        with open(path, encoding="utf-8-sig") as fh:
            prog = parse(fh.read())
        for st in prog.body:
            names = st.targets if isinstance(st, A.Let) else (
                [st.name] if isinstance(st, (A.FnDef, A.StructDef)) else [])
            for n in names:
                if not n.startswith("_"):
                    b = self.declare(scope, n)
                    self.emit(f"{b.py} = {tmp}[{n!r}]")

    def s_Test(self, s, scope):
        fname = self.fresh("test")
        ctx = FuncCtx(scope.ctx, "fn", erased=False, exact=scope.ctx.exact, name=f"test {s.name}")
        self._function(fname, [], s.body, False, Scope(scope, ctx), ctx, None, [])
        self.emit(f"{HELPER_PREFIX}add_test({s.name!r}, {fname}, {s.line})")

    def s_Deriv(self, s, scope):
        self.emit(f"{HELPER_PREFIX}d[{s.name!r}] = {self.expr(s.expr, scope)}")

    # ------------------------------------------------------------- functions
    def _exact_deco(self, decos):
        for d in decos:
            if isinstance(d, A.Call) and isinstance(d.func, A.Name) and d.func.id == "precision" \
                    and d.args and isinstance(d.args[0], A.Name) and d.args[0].id == "exact":
                return True
        return False

    def s_FnDef(self, s, scope):
        exact = scope.ctx.exact or self._exact_deco(s.decorators)
        b = self.declare(scope, s.name, kind="fn")
        if scope.ctx.erased:
            ctx = FuncCtx(scope.ctx, "fn", erased=True, exact=exact, name=s.name)
            self._function(b.py, s.params, s.body, s.is_expr, Scope(scope, ctx), ctx, s.ret, s.shape_params)
            b.erased_py = b.py
        else:
            # normal version
            ctx = FuncCtx(scope.ctx, "fn", erased=False, exact=exact, name=s.name)
            self._function(b.py, s.params, s.body, s.is_expr, Scope(scope, ctx), ctx, s.ret, s.shape_params)
            # erased twin
            epy = b.py + "__e"
            scope.ctx.used.add(epy)
            b.erased_py = epy
            ectx = FuncCtx(scope.ctx, "fn", erased=True, exact=exact, name=s.name)
            self._function(epy, s.params, s.body, s.is_expr, Scope(scope, ectx), ectx, s.ret, s.shape_params)
            self.emit(f"{HELPER_PREFIX}twin({b.py}, {epy}, {s.name!r}, {self.const(s, 'D')})")
            for d in reversed(s.decorators):
                self.emit(f"{b.py} = {self.expr(d, scope)}({b.py})")
        self.fn_names[b.py] = s.name

    def _function(self, pyname, params, body, is_expr, fscope, ctx, ret, shape_params):
        header_at = len(self.lines)
        self.emit("")                                   # placeholder for def line
        self.indent += 1
        decl_at = len(self.lines)
        self.emit("")                                   # placeholder for global/nonlocal
        pnames = []
        for p in params:
            pb = self.declare(fscope, p.name, kind="param", is_param=True)
            pnames.append((p, pb.py))
        defaults = []
        for p, py in pnames:
            if p.default is not None:
                self.emit(f"if {py} is {HELPER_PREFIX}NO: {py} = {self.expr(p.default, fscope.parent)}")
        needs_b = any(p.type is not None for p in params) or ret is not None
        syms = list(shape_params)
        for p in params:
            _collect_syms(p.type, syms)
        if ret is not None:
            _collect_syms(ret, syms)
        if not ctx.erased:
            if needs_b:
                self.emit(f"{HELPER_PREFIX}b = {{}}")
            for p, py in pnames:
                if p.type is not None:
                    chk = self.checker(p.type)
                    self.emit(f"{py} = {chk}({py}, {HELPER_PREFIX}b, "
                              f"{'argument ' + repr(p.name) + ' of ' + ctx.name + '()'!r})")
            if ret is not None:
                ctx.ret_check = self.checker(ret)
        elif syms:
            pairs = ", ".join(f"({py}, {self.const(p.type, 'T')})" for p, py in pnames if p.type is not None)
            self.emit(f"{HELPER_PREFIX}b = {HELPER_PREFIX}shape_bind(({pairs},))")
        for sym in dict.fromkeys(syms):
            sb = self.declare(fscope, sym, kind="param")
            self.emit(f"{sb.py} = {HELPER_PREFIX}b.get({sym!r})")
        if is_expr:
            self.cur_line = (body.line, body.col)
            val = self.expr(body, fscope)
            if ret is not None and not ctx.erased:
                val = f"{ctx.ret_check}({val}, {HELPER_PREFIX}b, {'return value of ' + ctx.name + '()'!r})"
            self.emit(f"return {val}")
        else:
            self.block(body, fscope)
        self.indent -= 1
        sig = ", ".join(py + (f"={HELPER_PREFIX}NO" if p.default is not None else "") for p, py in pnames)
        pad = "    " * self.indent
        self.lines[header_at] = (f"{pad}def {pyname}({sig}):", self.lines[header_at][1])
        decls = []
        if ctx.globals:
            decls.append("global " + ", ".join(sorted(ctx.globals)))
        if ctx.nonlocals:
            decls.append("nonlocal " + ", ".join(sorted(ctx.nonlocals)))
        self.lines[decl_at] = (pad + "    " + ("; ".join(decls) if decls else "pass"), self.lines[decl_at][1])
        self.fn_names[pyname] = ctx.name

    def s_StructDef(self, s, scope):
        b = self.declare(scope, s.name, kind="struct")
        fields = []
        for f in s.fields:
            t = self.const(f.type, "T") if f.type is not None else "None"
            if f.default is not None:
                d = f"lambda: {self.expr(f.default, scope)}"
            else:
                d = "None"
            fields.append(f"({f.name!r}, {t}, {d})")
        self.emit(f"{b.py} = {HELPER_PREFIX}struct({s.name!r}, [{', '.join(fields)}])")
        for m in s.methods:
            mscope = Scope(scope, scope.ctx)
            self.s_FnDef(m, mscope)
            mb = mscope.names[m.name]
            self.emit(f"{b.py}.methods[{m.name!r}] = {mb.py}")

    # ------------------------------------------------------------- evolve
    def s_Evolve(self, s, scope):
        names = s.states
        base = self.fresh("ev")

        def deriv_fn(pyname, erased, only=None):
            ctx = FuncCtx(scope.ctx, "fn", erased=erased, exact=scope.ctx.exact, name="evolve")
            fs = Scope(scope, ctx)
            tb = self.declare(fs, s.tvar, kind="param", is_param=True)
            sbs = [self.declare(fs, n, kind="param", is_param=True) for n in names]
            body = [x for x in s.body if not (only and isinstance(x, A.Deriv) and x.name != only)]
            header_at = len(self.lines)
            self.emit("")
            self.indent += 1
            decl_at = len(self.lines)
            self.emit("pass")
            self.emit(f"{HELPER_PREFIX}d = {{}}")
            self.block(body, fs)
            self.emit(f"return {HELPER_PREFIX}d")
            self.indent -= 1
            pad = "    " * self.indent
            self.lines[header_at] = (f"{pad}def {pyname}({tb.py}, {', '.join(b.py for b in sbs)}):",
                                     self.lines[header_at][1])
            decls = []
            if ctx.globals:
                decls.append("global " + ", ".join(sorted(ctx.globals)))
            if ctx.nonlocals:
                decls.append("nonlocal " + ", ".join(sorted(ctx.nonlocals)))
            if decls:
                self.lines[decl_at] = (pad + "    " + "; ".join(decls), self.lines[decl_at][1])

        if scope.ctx.erased:
            deriv_fn(base, True)
            full, fast = base, base
            parts = "{" + ", ".join(f"{n!r}: {base}" for n in names) + "}"
        else:
            deriv_fn(base, False)
            deriv_fn(base + "e", True)
            pparts = []
            for n in names:
                pn = f"{base}e_{len(pparts)}"
                deriv_fn(pn, True, only=n)
                pparts.append(f"{n!r}: {pn}")
            full, fast = base, base + "e"
            parts = "{" + ", ".join(pparts) + "}"
        self.cur_line = (s.line, s.col)
        if isinstance(s.range, A.Range):
            t0, t1 = self.expr(s.range.start, scope), self.expr(s.range.end, scope)
        else:
            r = self.fresh("r")
            self.emit(f"{r} = {self.expr(s.range, scope)}")
            t0, t1 = f"{r}.start", f"{r}.end"
        every = self.expr(s.every, scope) if s.every is not None else "None"
        tol = self.expr(s.tol, scope) if s.tol is not None else "None"
        state = ", ".join(self.load(n, scope) for n in names)
        res = self.fresh("res")
        self.emit(f"{res} = {HELPER_PREFIX}evolve({tuple(names)!r}, ({state},), {t0}, {t1}, "
                  f"{self.expr(s.step, scope)}, {s.method!r}, {every}, {bool(s.into)}, "
                  f"{full}, {fast}, {parts}, {s.line}, {s.col}, {tol})")
        for k, n in enumerate(names):
            self.emit(f"{self.store_target(n, scope)} = {res}[0][{k}]")
        if s.into:
            b = self.declare(scope, s.into)
            self.emit(f"{b.py} = {res}[1]")

    # ------------------------------------------------------------- expressions
    def binop(self, op):
        simple = {"+": "+", "-": "-", "*": "*", "/": "/", "//": "//", "%": "%", "@": "@", "**": "**",
                  "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
                  "in": "in", "not in": "not in"}
        if op in simple:
            o = simple[op]
            return lambda a, b: f"({a} {o} {b})"
        if op == "~=":
            return lambda a, b: f"{HELPER_PREFIX}approx({a}, {b})"
        if op == "±":
            return lambda a, b: f"{HELPER_PREFIX}pm({a}, {b})"
        raise RuntimeErr(f"unknown operator {op}")

    def expr(self, e, scope):
        return getattr(self, "e_" + type(e).__name__)(e, scope)

    def e_Num(self, e, scope):
        ctx = scope.ctx
        if e.unit is not None:
            text, scale, dim = e.unit
            if ctx.erased or dim.dimensionless:
                return repr(float(e.value * scale))
            return self.const(Quantity(e.value * scale, dim, (text, scale)), "q")
        if ctx.exact and not ctx.erased:
            return self.const(Fraction(e.text), "x")
        return repr(e.value)

    def e_Str(self, e, scope):
        return repr(e.value)

    def e_Bool(self, e, scope):
        return "True" if e.value else "False"

    def e_NoneLit(self, e, scope):
        return "None"

    def e_Name(self, e, scope):
        return self.load(e.id, scope)

    def e_FStr(self, e, scope):
        parts = []
        for p in e.parts:
            if isinstance(p, str):
                parts.append(repr(p))
            else:
                ex, spec = p
                parts.append(f"{HELPER_PREFIX}fmt({self.expr(ex, scope)}, True, {spec!r})")
        return "(" + " + ".join(parts) + ")" if parts else "''"

    def e_Unary(self, e, scope):
        v = self.expr(e.operand, scope)
        if e.op == "not":
            return f"(not {HELPER_PREFIX}truth({v}))"
        return f"({e.op}{v})"

    def e_Binary(self, e, scope):
        return self.binop(e.op)(self.expr(e.left, scope), self.expr(e.right, scope))

    def e_Logic(self, e, scope):
        a, b = self.expr(e.left, scope), self.expr(e.right, scope)
        if e.op == "and":
            return f"({HELPER_PREFIX}T({a}) and {b})"
        return f"({HELPER_PREFIX}T({a}) or {b})"

    def e_IfExpr(self, e, scope):
        return f"({self.expr(e.a, scope)} if {HELPER_PREFIX}truth({self.expr(e.cond, scope)}) else {self.expr(e.b, scope)})"

    def e_Call(self, e, scope):
        args = [self.expr(a, scope) for a in e.args]
        kws = [(k, self.expr(v, scope)) for k, v in e.kwargs]
        f = e.func
        if scope.ctx.erased:
            fcode = None
            if isinstance(f, A.Name):
                b = scope.lookup(f.id)
                if b is not None and b.kind == "fn" and b.erased_py:
                    fcode = b.erased_py
            if fcode is None:
                fcode = f"{HELPER_PREFIX}E({self.expr(f, scope)})"
        else:
            fcode = self.expr(f, scope)
        parts = list(args)
        for k, v in kws:
            parts.append(f"{k}={v}" if k.isidentifier() and not keyword.iskeyword(k) else f"**{{{k!r}: {v}}}")
        return f"{fcode}({', '.join(parts)})"

    def e_Attr(self, e, scope):
        code = f"{HELPER_PREFIX}attr({self.expr(e.obj, scope)}, {e.name!r})"
        return f"{HELPER_PREFIX}S({code})" if scope.ctx.erased else code

    def index(self, n, scope):
        if isinstance(n, A.Slice):
            parts = [self.expr(x, scope) if x is not None else "None" for x in (n.lo, n.hi, n.step)]
            return f"slice({', '.join(parts)})"
        if isinstance(n, A.TupleLit):
            return "(" + ", ".join(self.index(x, scope) for x in n.items) + ",)"
        return f"{HELPER_PREFIX}idx({self.expr(n, scope)})"

    def e_Index(self, e, scope):
        return f"{self.expr(e.obj, scope)}[{self.index(e.index, scope)}]"

    def e_Slice(self, e, scope):
        return self.index(e, scope)

    def e_ListLit(self, e, scope):
        return "[" + ", ".join(self.expr(x, scope) for x in e.items) + "]"

    def e_TupleLit(self, e, scope):
        items = [self.expr(x, scope) for x in e.items]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"

    def e_DictLit(self, e, scope):
        return "{" + ", ".join(f"{self.expr(k, scope)}: {self.expr(v, scope)}" for k, v in e.pairs) + "}"

    def _comp_scope(self, targets, scope):
        inner = Scope(scope, scope.ctx, loop=True)
        names = [self.declare(inner, n).py for n in targets]
        tgt = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
        return inner, tgt

    def e_ListComp(self, e, scope):
        it = self.expr(e.iter, scope)
        inner, tgt = self._comp_scope(e.targets, scope)
        cond = f" if {HELPER_PREFIX}truth({self.expr(e.cond, inner)})" if e.cond is not None else ""
        return f"[{self.expr(e.elem, inner)} for {tgt} in {HELPER_PREFIX}iterable({it}){cond}]"

    def e_DictComp(self, e, scope):
        it = self.expr(e.iter, scope)
        inner, tgt = self._comp_scope(e.targets, scope)
        cond = f" if {HELPER_PREFIX}truth({self.expr(e.cond, inner)})" if e.cond is not None else ""
        return (f"{{{self.expr(e.key, inner)}: {self.expr(e.value, inner)} "
                f"for {tgt} in {HELPER_PREFIX}iterable({it}){cond}}}")

    def e_Lambda(self, e, scope):
        def build(erased):
            ctx = FuncCtx(scope.ctx, "fn", erased=erased, exact=scope.ctx.exact, name="<lambda>")
            # lambdas share the enclosing python function's namespace for closures
            ctx.used = set(scope.ctx.used)
            ls = Scope(scope, ctx)
            ps = []
            for p in e.params:
                b = self.declare(ls, p.name, kind="param", is_param=True)
                ps.append(b.py + (f"={self.expr(p.default, scope)}" if p.default is not None else ""))
            return f"(lambda {', '.join(ps)}: {self.expr(e.body, ls)})"
        code = build(True) if scope.ctx.erased else f"{HELPER_PREFIX}dual({build(False)}, {build(True)})"
        captured = self._loop_captures(scope)
        if captured:
            names = ", ".join(captured)
            code = f"(lambda {names}: {code})({names})"
        return code

    def _loop_captures(self, scope):
        """Immutable names declared inside loops: bind their current value (per-iteration closures)."""
        out = []
        sc = scope
        while sc is not None and sc.ctx is scope.ctx and sc.in_loop:
            for b in sc.names.values():
                if not b.mutable and b.kind != "fn" and b.py not in out:
                    out.append(b.py)
            sc = sc.parent
        return out

    def e_Range(self, e, scope):
        return f"{HELPER_PREFIX}Range({self.expr(e.start, scope)}, {self.expr(e.end, scope)}, {e.inclusive})"

    def e_As(self, e, scope):
        v = self.expr(e.expr, scope)
        text, scale, dim = e.unit
        if scope.ctx.erased:
            return f"({v} / {scale!r})" if dim.dimensionless else v
        return f"{HELPER_PREFIX}as_unit({v}, {self.const(e.unit, 'u')}, {e.line}, {e.col})"

    # ------------------------------------------------------------- type checkers
    def checker(self, t):
        return self.const(self.rt.make_checker(t), "C")

    # ------------------------------------------------------------- output
    def source(self):
        return "\n".join(text for text, _ in self.lines) + "\n"

    def linemap(self):
        return [line for _, line in self.lines]


def _collect_syms(t, out):
    if isinstance(t, A.TName) and t.name in ("Tensor", "Vec", "Mat", "Matrix"):
        for a in t.args:
            if isinstance(a, A.TName) and not a.args and a.name not in DIM_NAMES \
                    and a.name not in ("f16", "f32", "f64", "i32", "i64", "bool", "_", "bf16", "fp8_e4m3", "fp8_e5m2",
                                   "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"):
                if a.name not in out:
                    out.append(a.name)


# =========================================================================== runtime support
class Runtime:
    """Helpers referenced by generated code (all exposed under the `_A` prefix)."""

    def __init__(self, interp, filename, base_dir):
        self.interp = interp
        self.filename = filename
        self.base_dir = base_dir
        self.structs = {}

    # ---- type checks
    def make_checker(self, t):
        interp = self.interp
        structs = self.structs

        class _Env:
            def lookup(self, name):
                if name in structs:
                    return structs[name]
                raise KeyError(name)
        env = _Env()
        if isinstance(t, A.TName):
            n = t.name
            if n == "Int":
                def chk(v, b, what):
                    if v.__class__ is int or (isinstance(v, (int, np.integer)) and not isinstance(v, bool)):
                        return v
                    return interp.check_type(v, t, b if b is not None else {}, what, env)
                return chk
            if n in ("Float", "Number", "Real"):
                def chk(v, b, what):
                    c = v.__class__
                    if c is float or c is int or c is Node:
                        return v
                    return interp.check_type(v, t, b if b is not None else {}, what, env)
                return chk
            if n in DIM_NAMES and not t.args:
                want = DIM_NAMES[n]

                def chk(v, b, what):
                    if v.__class__ is Quantity and v.dim == want:
                        return v
                    return interp.check_type(v, t, b if b is not None else {}, what, env)
                return chk
            if n == "Any":
                return lambda v, b, what: v

        def chk(v, b, what):
            return interp.check_type(v, t, b if b is not None else {}, what, env)
        return chk

    def shape_bind(self, pairs):
        b = {}
        for v, t in pairs:
            if isinstance(t, A.TName) and t.name in ("Tensor", "Vec", "Mat", "Matrix"):
                specs = [a.name for a in t.args if isinstance(a, A.TName) and a.name not in DIM_NAMES
                         and a.name not in ("f16", "f32", "f64", "i32", "i64", "bool", "bf16", "fp8_e4m3",
                                            "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4",
                                            "fp8", "bf8", "fp4", "fp6")]
                shape = np.shape(ops.raw(v))
                for s, n in zip(specs, shape):
                    if s != "_":
                        b.setdefault(s, n)
        return b

    # ---- structs
    def struct(self, name, fields):
        rt = self

        class _Interp:
            def construct(self_, st, args, kwargs):
                return rt.construct(st, args, kwargs)
        st = StructType(_Interp(), name, fields, {}, None)
        self.structs[name] = st
        return st

    def construct(self, st, args, kwargs):
        if len(args) > len(st.fields):
            raise RuntimeErr(f"{st.name}() takes {len(st.fields)} field(s) but {len(args)} were given")
        kwargs = dict(kwargs)
        values, bindings = {}, {}
        for i, (name, t, default) in enumerate(st.fields):
            if i < len(args):
                v = args[i]
            elif name in kwargs:
                v = kwargs.pop(name)
            elif default is not None:
                v = default()
            else:
                raise RuntimeErr(f"{st.name}() is missing field '{name}'")
            if t is not None:
                v = self.make_checker(t)(v, bindings, f"field '{name}' of {st.name}")
            values[name] = v
        if kwargs:
            raise RuntimeErr(f"{st.name} has no field(s): {', '.join(kwargs)}")
        return StructInstance(st, values)

    # ---- evolve
    def evolve(self, names, state, t0, t1, dt, method, every, want_record, full, fast, parts, line, col,
               tol=None):
        tdim = ops.dim_of(dt)
        span = t1 - t0
        try:
            nsteps = span / dt
        except UnitError:
            raise UnitError(f"step has units {dim_name(ops.dim_of(dt))} but the time range has "
                            f"{dim_name(ops.dim_of(t0))}", line, col)
        if isinstance(nsteps, Quantity):
            raise UnitError("time range and step must have the same units", line, col)
        nsteps = int(round(float(nsteps)))
        if nsteps <= 0:
            raise RuntimeErr("evolve: the time range must be longer than one step", line, col)
        every = int(every) if every is not None else 1
        if method == "verlet" and len(names) != 2:
            raise RuntimeErr("verlet needs exactly two states: position, velocity", line, col)

        def collect(d):
            try:
                return [d[n] for n in names]
            except KeyError as k:
                raise RuntimeErr(f"evolve: no derivative for {k} (add a line {k.args[0]}' = ...)", line, col)

        # 1) full unit-checked evaluation
        first = collect(full(t0, *state))
        dims = [ops.dim_of(v) for v in state]
        for n, v, dv in zip(names, state, first):
            want = ops.dim_of(v) / tdim
            got = ops.dim_of(dv)
            if got is not None and got != want:
                raise UnitError(f"{n}' must have units of {dim_name(want)} ({n} per unit time), "
                                f"but got {dim_name(got)}", line, col)
        # 2) erased fast path, verified against the checked result
        use_fast = fast is not full and all(d is not None for d in dims) and tdim is not None
        if use_fast:
            raw_state = [ops.raw(v) if isinstance(v, Quantity) else v for v in state]
            try:
                probe = collect(fast(ops.raw(t0) if isinstance(t0, Quantity) else t0, *raw_state))
                for a, b in zip(probe, first):
                    if not np.allclose(np.asarray(ops.raw(a), dtype=float), np.asarray(ops.raw(b), dtype=float),
                                       rtol=1e-9, atol=1e-300, equal_nan=True):
                        use_fast = False
            except Exception:
                use_fast = False
        if use_fast:
            st = raw_state
            t_0 = t0.v if isinstance(t0, Quantity) else t0
            h = dt.v if isinstance(dt, Quantity) else dt

            def deriv(t, s):
                return collect(fast(t, *s))

            def one(t, s, i):
                return parts[names[i]](t, *s)[names[i]]
        else:
            st = list(state)
            t_0, h = t0, dt

            def deriv(t, s):
                return collect(full(t, *s))

            def one(t, s, i):
                return deriv(t, s)[i]

        rec_t, rec = [], [[] for _ in names]

        def record(t, s):
            rec_t.append(t)
            for k, v in enumerate(s):
                rec[k].append(v.copy() if isinstance(v, np.ndarray) else v)
        if want_record:
            record(t_0, st)
        if method == "rk45":
            from .integrators import rk45
            counter = [0]

            def on_step(t, cur):
                counter[0] += 1
                if want_record and counter[0] % every == 0:
                    record(t, cur)
            t_end = t_0 + nsteps * h
            st, _, _ = rk45(deriv, t_0, t_end, h, st, float(tol) if tol is not None else 1e-8, on_step)
            if want_record and counter[0] % every != 0:
                record(t_end, st)
            nsteps = 0
        for k in range(nsteps):
            t = t_0 + k * h
            if method == "euler":
                d = deriv(t, st)
                st = [x + h * dx for x, dx in zip(st, d)]
            elif method == "midpoint":
                k1 = deriv(t, st)
                mid = [x + (0.5 * h) * dx for x, dx in zip(st, k1)]
                k2 = deriv(t + 0.5 * h, mid)
                st = [x + h * dx for x, dx in zip(st, k2)]
            elif method == "rk4":
                k1 = deriv(t, st)
                k2 = deriv(t + 0.5 * h, [x + (0.5 * h) * a for x, a in zip(st, k1)])
                k3 = deriv(t + 0.5 * h, [x + (0.5 * h) * a for x, a in zip(st, k2)])
                k4 = deriv(t + h, [x + h * a for x, a in zip(st, k3)])
                st = [x + (h / 6.0) * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(st, k1, k2, k3, k4)]
            elif method == "symplectic":
                for i in reversed(range(len(st))):
                    st[i] = st[i] + h * one(t, st, i)
            elif method == "verlet":
                x, v = st
                vh = v + (0.5 * h) * one(t, [x, v], 1)
                x = x + h * one(t, [x, vh], 0)
                v = vh + (0.5 * h) * one(t + h, [x, vh], 1)
                st = [x, v]
            if want_record and ((k + 1) % every == 0 or k == nsteps - 1):
                record(t_0 + (k + 1) * h, st)
        if use_fast:
            st = [make(v, d) for v, d in zip(st, dims)]
        rec_out = None
        if want_record:
            ts = ops.stack(rec_t)
            fields = {"t": make(ts, tdim) if use_fast else ts}
            for k, n in enumerate(names):
                arr = ops.stack(rec[k])
                fields[n] = make(arr, dims[k]) if use_fast else arr
            rec_out = Record(**fields)
        return st, rec_out


# =========================================================================== helper namespace
def _truth(v):
    if isinstance(v, (np.ndarray, Node)) and np.size(ops.raw(v)) != 1:
        raise RuntimeErr("a tensor with several elements cannot be used as a condition")
    return v


def _all(v):
    if isinstance(v, (np.ndarray, Quantity, Node)):
        return bool(np.all(ops.raw(v)))
    return bool(v)


def _strip(v):
    return v.v if v.__class__ is Quantity else v


def _erased(f):
    e = getattr(f, "_apsi_e", None)
    if e is not None:
        return e
    if isinstance(f, BoundMethod):
        fe = getattr(f.fn, "_apsi_e", None)
        if fe is not None:
            obj = f.obj
            return lambda *a, **k: fe(obj, *a, **k)
    return f


def _twin(f, fe, name, node=None):
    f._apsi_e = fe
    f._apsi_def = node
    f.name = name
    fe.name = name
    return f


def _dual(f, fe):
    try:
        f._apsi_e = fe
        f.name = "<lambda>"
    except AttributeError:
        pass
    return f


def _irange(a, b, inclusive):
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return range(int(a), int(b) + (1 if inclusive else 0))
    return RangeVal(a, b, inclusive)


def _idx(v):
    if isinstance(v, list) and v and all(isinstance(x, (int, np.integer)) for x in v):
        return np.asarray(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _unpack(v, n):
    try:
        vals = tuple(v)
    except TypeError:
        raise RuntimeErr(f"cannot unpack {type_name(v)} into {n} names")
    if len(vals) != n:
        raise RuntimeErr(f"expected {n} values to unpack, got {len(vals)}")
    return vals


def _iterable(it):
    if isinstance(it, Record):
        return list(it.fields.items())
    if isinstance(it, (int, float)) and not isinstance(it, bool):
        raise RuntimeErr("cannot iterate over a number; use 0..n or range(n)")
    return it


def _as_unit(v, unit, line, col):
    text, scale, dim = unit
    have = ops.dim_of(v)
    if have is None:
        raise RuntimeErr(f"cannot convert {type_name(v)} to [{text}]", line, col)
    if have != dim:
        raise UnitError(f"cannot express {dim_name(have)} in [{text}] ({dim_name(dim)})", line, col)
    if dim.dimensionless:
        return v / scale
    return Quantity(v.v, v.dim, (text, scale))


class Compiled:
    """A compiled Apsi module ready to execute."""

    def __init__(self, interp, prog, filename, base_dir, tests):
        from .interpreter import wrap_python_error  # noqa: F401
        self.interp = interp
        self.filename = filename
        self.rt = Runtime(interp, filename, base_dir)
        builtins = dict(interp.globals.vars)
        gen = CodeGen(self.rt, builtins.keys(), filename)
        module_ctx = FuncCtx(None, "module")
        module_scope = Scope(None, module_ctx)
        gen.program(prog, module_scope)
        self.gen = gen
        self.source = gen.source()
        self.linemap = gen.linemap()
        self.module_scope = module_scope
        self.code = compile(self.source, f"<apsi:{filename}>", "exec")
        g = dict(builtins)
        g.update(gen.consts)
        rt = self.rt
        g.update({
            "__name__": "__apsi__",
            f"{HELPER_PREFIX}NO": NO, f"{HELPER_PREFIX}S": _strip, f"{HELPER_PREFIX}E": _erased,
            f"{HELPER_PREFIX}twin": _twin, f"{HELPER_PREFIX}dual": _dual,
            f"{HELPER_PREFIX}truth": _truth, f"{HELPER_PREFIX}T": _truth, f"{HELPER_PREFIX}all": _all,
            f"{HELPER_PREFIX}irange": _irange, f"{HELPER_PREFIX}iterable": _iterable,
            f"{HELPER_PREFIX}idx": _idx, f"{HELPER_PREFIX}unpack": _unpack,
            f"{HELPER_PREFIX}Range": RangeVal, f"{HELPER_PREFIX}as_unit": _as_unit,
            f"{HELPER_PREFIX}fmt": fmt, f"{HELPER_PREFIX}approx": ops.approx, f"{HELPER_PREFIX}pm": ops.plusminus,
            f"{HELPER_PREFIX}attr": interp.get_attr, f"{HELPER_PREFIX}setattr": interp.set_attr,
            f"{HELPER_PREFIX}setidx": _setidx(interp),
            f"{HELPER_PREFIX}AssertionFailed": AssertionFailed,
            f"{HELPER_PREFIX}mods": interp.modules,
            f"{HELPER_PREFIX}namespace": lambda p: Namespace(p, interp.modules[p]),
            f"{HELPER_PREFIX}Namespace": Namespace,
            f"{HELPER_PREFIX}import_file": lambda p: interp.import_compiled(p),
            f"{HELPER_PREFIX}struct": rt.struct, f"{HELPER_PREFIX}shape_bind": rt.shape_bind,
            f"{HELPER_PREFIX}evolve": rt.evolve,
            f"{HELPER_PREFIX}add_test": lambda name, fn, line: tests.append((name, fn, line)),
        })
        self.globals = g

    def run(self):
        try:
            exec(self.code, self.globals)
        except ApsiError as e:
            self._locate(e, sys.exc_info()[2])
            raise
        except RecursionError:
            err = RuntimeErr("maximum recursion depth exceeded")
            self._locate(err, sys.exc_info()[2])
            raise err from None
        except Exception as e:
            raise self.translate(e, sys.exc_info()[2]) from None
        return self

    def translate(self, e, tb):
        from .interpreter import wrap_python_error
        if isinstance(e, ApsiError):
            self._locate(e, tb)
            return e
        msg = str(e)
        if isinstance(e, TypeError) and ("positional argument" in msg or "unexpected keyword" in msg
                                         or "required positional" in msg):
            err = RuntimeErr(msg.replace("()", "()"))
        else:
            err = wrap_python_error(e)
        self._locate(err, tb)
        return err

    def _locate(self, err, tb):
        fname = f"<apsi:{self.filename}>"
        frames = [f for f in traceback.extract_tb(tb) if f.filename == fname]
        if not frames:
            return
        if err.line is None:
            ln = frames[-1].lineno
            if 0 < ln <= len(self.linemap):
                err.line, err.col = self.linemap[ln - 1]
        if not err.trace:
            for f in reversed(frames[1:]):
                name = self.gen.fn_names.get(f.name)
                if name and not f.name.startswith(HELPER_PREFIX):
                    err.trace.append(f"{name}()")

    def exports(self):
        out = {}
        for name, b in self.module_scope.names.items():
            if not name.startswith("_") and b.py in self.globals:
                out[name] = self.globals[b.py]
        return out


def _setidx(interp):
    def setidx(obj, idx, v):
        interp.set_index(obj, idx, v)
    return setidx
