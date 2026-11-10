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
import re
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
_HELPER_NAME = re.compile(r"^_A[a-z_]*\d*$")
# attribute names that reach interpreter internals: always routed through the checked accessor
_RISKY_ATTRS = {"gi_frame", "gi_code", "f_globals", "f_builtins", "f_locals", "f_back", "f_code", "tb_frame",
                "tb_next", "cr_frame", "cr_code", "ag_frame", "ag_code", "co_code", "func_globals", "mro",
                "tofile", "dump"}


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
        self.pyc = False         # `use pycompat`: Python's exact semantics (set by program())
        self.trusted = False     # --allow-python=all: _private attributes of library objects are reachable
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

    def declare(self, scope, name, kind="var", mutable=False, is_param=False, exact=False):
        ctx = scope.ctx
        if ctx.kind == "class":
            exact = True
        if self.pyc and not is_param and not exact:
            # Python scoping: one namespace per function — a name declared again in the same function is the
            # same variable (redefinitions, loop variables, names first assigned in a branch)
            s = scope
            root = scope
            while s is not None and s.ctx is ctx:
                b = s.names.get(name)
                if b is not None:
                    scope.names[name] = b
                    b.mutable = True
                    return b
                root = s
                if getattr(s, "is_comp", False):
                    break
                s = s.parent
            if not getattr(scope, "is_comp", False):
                scope = root                             # declare at function level: visible after the block
        base = name
        if keyword.iskeyword(base) or base in ("None", "True", "False") or (
                (_HELPER_NAME.match(base) or base == HELPER_PREFIX) if self.pyc else base.startswith(HELPER_PREFIX)):
            base = "u_" + base
        if self.pyc or exact:
            py = base
            k = 1
            while py in ctx.used and (not exact or is_param):
                k += 1
                py = f"{base}_{k}"
            ctx.used.add(py)
            b = Binding(py, ctx, kind, mutable=True if self.pyc else mutable)
            scope.names[name] = b
            return b
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
            ctx = scope.ctx
            if self.pyc and (name in ctx.globals or name in ctx.nonlocals):
                return name                        # declared global/nonlocal before the outer binding exists
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
        from .checker import is_python_mode
        self.pyc = is_python_mode(prog)
        body = prog.body
        first = next((x for x in body if not isinstance(x, A.Use)), None)
        if self.pyc and isinstance(first, A.ExprStmt) and isinstance(first.expr, A.Str):
            self.emit(f"__doc__ = {first.expr.value!r}")
        for s in body:
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
        if s.value is None:                      # Python mode `var x`: a declaration only
            for n in s.targets:
                self.pattern(n, scope)
            self.emit("pass")
            return
        val = self.expr(s.value, scope)
        if self.pyc or any(isinstance(n, tuple) for n in s.targets):
            if len(s.targets) == 1 and s.type is not None:
                val = f"{self.checker(s.type)}({val}, None, {s.targets[0]!r})"
            names = [self.pattern(n, scope) for n in s.targets]
            self.emit(f"{', '.join(names)}, = {val}" if len(names) > 1 else f"{names[0]} = {val}")
            return
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

    def pytarget(self, t, scope):
        """Python code for an assignment target (Python mode and the object model)."""
        if isinstance(t, A.Name):
            return self.store_target(t.id, scope)
        if isinstance(t, (A.TupleLit, A.ListLit)):
            inner = ", ".join(self.pytarget(x, scope) for x in t.items)
            return f"({inner}{',' if len(t.items) == 1 else ''})"
        if isinstance(t, A.Spread):
            return "*" + self.pytarget(t.value, scope)
        if isinstance(t, A.Attr):
            obj = self.expr(t.obj, scope)
            name = self._mangle(t.name, scope)
            if name.startswith("__") or name in _RISKY_ATTRS or keyword.iskeyword(name) or \
                    (name.startswith("_") and not self.trusted):
                return f"{HELPER_PREFIX}AttrSlot({obj}, {name!r}).value"
            return f"{obj}.{_pyattr(name)}"
        if isinstance(t, A.Index):
            return f"{self.expr(t.obj, scope)}[{self.index(t.index, scope)}]"
        raise RuntimeErr("cannot assign to this expression", t.line, t.col)

    def s_Assign(self, s, scope):
        val = self.expr(s.value, scope)
        t = s.target
        if self.pyc or s.more:
            targets = [t] + list(s.more or [])
            self.emit(" = ".join(self.pytarget(x, scope) for x in targets) + f" = {val}")
            return
        if isinstance(t, A.Attr):
            direct = self._self_attr(t, scope)
            if direct:
                self.emit(f"{direct} = {val}")
            else:
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
        val = self.expr(s.value, scope)
        t = s.target
        if self.pyc or s.op in ("|", "&", "<<", ">>", "^"):
            self.emit(f"{self.pytarget(t, scope)} {s.op}= {val}")
            return
        op = self.binop(s.op)
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

    def cond(self, c):
        return c if self.pyc else f"{HELPER_PREFIX}truth({c})"

    def fast_cond(self, c, scope):
        """A statement condition: plain bools (comparisons of numbers) skip the tensor check."""
        if self.pyc or scope.ctx.kind != "fn":
            return self.cond(c)
        v = f"{HELPER_PREFIX}cnd"
        return f"({v} if ({v} := {c}).__class__ is {HELPER_PREFIX}bool else {HELPER_PREFIX}truth({v}))"

    def s_If(self, s, scope):
        first = True
        for cond, body in s.branches:
            c = self.fast_cond(self.expr(cond, scope), scope)
            self.emit(f"{'if' if first else 'elif'} {c}:")
            first = False
            self.indent += 1
            self.block(body, Scope(scope, scope.ctx))
            self.indent -= 1
        if s.else_body is not None:
            self.emit("else:")
            self.indent += 1
            self.block(s.else_body, Scope(scope, scope.ctx))
            self.indent -= 1

    def _exc_class(self, k, scope):
        """Python code for an exception class named in `except` (Python mode)."""
        root = k.split(".")[0]
        if scope.lookup(root) is not None:
            code = self.load(root, scope)
            for part in k.split(".")[1:]:
                code += "." + _pyattr(part)
            return code
        return f"{HELPER_PREFIX}kind({k!r})"

    def s_Try(self, s, scope):
        self.emit("try:")
        self.indent += 1
        self.block(s.body, Scope(scope, scope.ctx))
        self.indent -= 1
        if s.handlers and self.pyc:
            for kinds, name, body in s.handlers:
                hs = Scope(scope, scope.ctx)
                head = "except*" if s.star else "except"
                if kinds:
                    cls = [self._exc_class(k, scope) for k in kinds]
                    head += " " + (cls[0] if len(cls) == 1 else "(" + ", ".join(cls) + ")")
                elif name:
                    head += f" {HELPER_PREFIX}Exception"
                if name:
                    head += f" as {self.declare(hs, name).py}"
                self.emit(head + ":")
                self.indent += 1
                self.block(body, hs)
                self.indent -= 1
        elif s.handlers:
            ev = self.fresh("ex")
            self.emit(f"except {HELPER_PREFIX}Exception as {ev}:")
            self.indent += 1
            first, catch_all = True, False
            for kinds, name, body in s.handlers:
                if kinds and not catch_all:
                    from .exceptions import KINDS, ALIASES
                    own = [self._exc_class(k, scope) for k in kinds
                           if ALIASES.get(k, k).split(".")[-1] not in KINDS and scope.lookup(k.split(".")[0]) is not None]
                    extra = f", ({', '.join(own)},)" if own else ""
                    self.emit(f"{'if' if first else 'elif'} {HELPER_PREFIX}exm({ev}, {tuple(kinds)!r}{extra}):")
                else:
                    self.emit("if True:" if first else "else:")
                    catch_all = True
                first = False
                self.indent += 1
                hs = Scope(scope, scope.ctx)
                if name:
                    hb = self.declare(hs, name)
                    self.emit(f"{hb.py} = {HELPER_PREFIX}exv({ev})")
                self.block(body, hs)
                self.indent -= 1
                if catch_all:
                    break
            if not catch_all:
                self.emit("else:")
                self.emit("    raise")
            self.indent -= 1
        if s.else_body is not None and s.handlers:
            self.emit("else:")
            self.indent += 1
            self.block(s.else_body, Scope(scope, scope.ctx))
            self.indent -= 1
        if s.final_body is not None:
            self.emit("finally:")
            self.indent += 1
            self.block(s.final_body, Scope(scope, scope.ctx))
            self.indent -= 1

    def s_Yield(self, s, scope):
        v = self.expr(s.value, scope) if s.value is not None else "None"
        self.emit(f"yield from {HELPER_PREFIX}iterable({v})" if s.from_ else f"yield {v}")

    def s_Raise(self, s, scope):
        if s.value is None:
            self.emit("raise")
            return
        v = self.expr(s.value, scope)
        if not self.pyc:
            v = f"{HELPER_PREFIX}raise_({v})"
        if s.cause is not None:
            c = self.expr(s.cause, scope)
            self.emit(f"raise {v} from {c if self.pyc else HELPER_PREFIX + 'raise_(' + c + ')'}")
        else:
            self.emit(f"raise {v}")

    def _loop_else(self, body, scope):
        if body:
            self.emit("else:")
            self.indent += 1
            self.block(body, Scope(scope, scope.ctx))
            self.indent -= 1

    def s_While(self, s, scope):
        self.emit(f"while {self.fast_cond(self.expr(s.cond, scope), scope)}:")
        self.indent += 1
        self.block(s.body, Scope(scope, scope.ctx, loop=True))
        self.indent -= 1
        self._loop_else(s.else_body, scope)

    def pattern(self, pat, scope):
        """Declare the names of a target pattern; return the Python target text."""
        if isinstance(pat, str):
            return self.declare(scope, pat).py
        if isinstance(pat, tuple) and pat[0] == "expr":
            return self.pytarget(pat[1], scope)
        if isinstance(pat, tuple):
            return "*" + self.declare(scope, pat[1]).py
        inner = ", ".join(self.pattern(p, scope) for p in pat)
        return f"({inner}{',' if len(pat) == 1 else ''})"

    def s_For(self, s, scope):
        if not self.pyc:
            from .idioms import match_gemm
            g = match_gemm(s)
            if g is not None:                        # a matrix product written as loops: BLAS, loops as fallback
                args = ", ".join([str(g.mode)] + [self.load(n, scope) for n in (g.a, g.b, g.c)] +
                                 [self.expr(e, scope) for e in (g.M, g.N, g.K)])
                self.emit(f"if not {HELPER_PREFIX}gemm({args}):")
                self.indent += 1
                self._for(s, scope)
                self.indent -= 1
                return
        self._for(s, scope)

    def _for(self, s, scope):
        it = s.iter
        if isinstance(it, A.Range):
            src = f"{HELPER_PREFIX}irange({self.expr(it.start, scope)}, {self.expr(it.end, scope)}, {it.inclusive})"
        elif self.pyc:
            src = self.expr(it, scope)
        else:
            src = f"{HELPER_PREFIX}iterable({self.expr(it, scope)})"
        inner = Scope(scope, scope.ctx, loop=True)
        pats = s.targets
        tgt = self.pattern(pats[0], inner) if len(pats) == 1 else self.pattern(list(pats), inner)
        self.emit(f"{'async ' if s.is_async else ''}for {tgt} in {src}:")
        self.indent += 1
        self.block(s.body, inner)
        self.indent -= 1
        self._loop_else(s.else_body, scope)

    # ------------------------------------------------------------- the Python object model
    def s_With(self, s, scope):
        parts = []
        inner = Scope(scope, scope.ctx)
        for e, target in s.items:
            code = self.expr(e, scope)
            if not self.pyc:
                code = f"{HELPER_PREFIX}ctx({code})"
            if target is not None:
                code += " as " + self.pattern(target if not isinstance(target, list) else target, inner)
            parts.append(code)
        self.emit(f"{'async ' if s.is_async else ''}with {', '.join(parts)}:")
        self.indent += 1
        self.block(s.body, inner)
        self.indent -= 1

    def s_Match(self, s, scope):
        """Python's match statement, emitted as Python (exact semantics; capture names are function variables)."""
        self.emit(f"match {self.expr(s.subject, scope)}:")
        self.indent += 1
        for pat, guard, body in s.cases:
            inner = Scope(scope, scope.ctx)
            text = self._pat(pat, scope, inner)
            g = f" if {self.cond(self.expr(guard, inner))}" if guard is not None else ""
            self.emit(f"case {text}{g}:")
            self.indent += 1
            self.block(body, inner)
            self.indent -= 1
        self.indent -= 1

    def _dotted_code(self, dotted, scope):
        parts = dotted.split(".")
        code = self.load(parts[0], scope)
        if not code.replace("_", "a").isidentifier():           # constants and helpers are not valid in patterns
            raise RuntimeErr(f"'{dotted}' cannot be used in a case pattern")
        return ".".join([code] + parts[1:])

    def _pat(self, p, scope, inner):
        k = p[0]
        if k == "wild":
            return "_"
        if k == "capture":
            return self.declare(inner, p[1]).py
        if k == "star":
            return "*" + (self.declare(inner, p[1]).py if p[1] else "_")
        if k == "as":
            return f"({self._pat(p[1], scope, inner)}) as {self.declare(inner, p[2]).py}"
        if k == "or":
            return " | ".join(f"({self._pat(x, scope, inner)})" for x in p[1])
        if k == "seq":
            return "[" + ", ".join(self._pat(x, scope, inner) for x in p[1]) + "]"
        if k == "value":
            return self._dotted_code(p[1], scope)
        if k == "map":
            items = [f"{self._pat_lit(key, scope)}: {self._pat(v, scope, inner)}" for key, v in p[1]]
            if p[2]:
                items.append("**" + self.declare(inner, p[2]).py)
            return "{" + ", ".join(items) + "}"
        if k == "class":
            args = [self._pat(x, scope, inner) for x in p[2]] + [f"{n}={self._pat(x, scope, inner)}" for n, x in p[3]]
            return f"{self._dotted_code(p[1], scope)}({', '.join(args)})"
        if k == "lit":
            return self._pat_lit(p[1], scope)
        raise RuntimeErr(f"unknown pattern {k}")

    def _pat_lit(self, node, scope):
        if isinstance(node, tuple) and node[0] == "value":
            return self._dotted_code(node[1], scope)
        if isinstance(node, A.Num) and not node.unit:
            return repr(node.value)
        if isinstance(node, A.Unary) and node.op == "-" and isinstance(node.operand, A.Num):
            return "-" + repr(node.operand.value)
        if isinstance(node, A.Str):
            return repr(node.value)
        if isinstance(node, A.Bytes):
            return repr(node.value)
        if isinstance(node, A.NoneLit):
            return "None"
        if isinstance(node, A.Bool):
            return "True" if node.value else "False"
        if isinstance(node, A.Binary) and node.op in ("+", "-"):        # complex literals: 1 + 2j
            return f"{self._pat_lit(node.left, scope)} {node.op} {self._pat_lit(node.right, scope)}"
        if isinstance(node, A.Name):
            return self._dotted_code(node.id, scope)
        raise RuntimeErr("a case pattern needs a literal, a dotted name, or a capture")

    @staticmethod
    def _mangle(name, scope):
        """Python's private-name mangling: `__x` written inside class C means `_C__x`."""
        if not name.startswith("__") or name.endswith("__"):
            return name
        ctx = scope.ctx
        while ctx is not None:
            owner = ctx.name if ctx.kind == "class" else getattr(ctx, "mangle_as", None)
            if owner is not None:
                cls = owner.lstrip("_")
                return f"_{cls}{name}" if cls else name
            ctx = ctx.parent
        return name

    def s_Del(self, s, scope):
        self.emit("del " + ", ".join(self.pytarget(t, scope) for t in s.targets))

    def s_Global(self, s, scope):
        for n in s.names:
            b = scope.lookup(n)
            if b is not None and b.ctx is not scope.ctx:
                (scope.ctx.globals if b.ctx.kind == "module" else scope.ctx.nonlocals).add(b.py)
            elif b is None and scope.ctx.kind == "fn":
                (scope.ctx.nonlocals if s.nonlocal_ else scope.ctx.globals).add(n)
        self.emit("pass")

    def s_FieldDef(self, s, scope):
        b = self.declare(scope, s.name, exact=True)
        ann, text = s.annotation
        self.emit(f"__annotations__[{s.name!r}] = {text!r}")
        if s.value is not None:
            self.emit(f"{b.py} = {self.expr(s.value, scope)}")

    def s_ClassDef(self, s, scope):
        b = self.declare(scope, s.name, kind="struct")
        bases = [self.expr(x, scope) for x in s.bases]
        for k, v in s.kwargs:
            bases.append(("**" + self.expr(v, scope)) if k is None else f"{k}={self.expr(v, scope)}")
        for d in s.decorators:
            self.emit(f"@{self.expr(d, scope)}")
        self.emit(f"class {b.py}({', '.join(bases)}):")
        self.indent += 1
        ctx = FuncCtx(scope.ctx, "class", erased=False, exact=scope.ctx.exact, name=s.name)
        ctx.class_of = b.py
        cscope = Scope(scope, ctx)
        cscope.is_class = True
        body = list(s.body)
        if body and isinstance(body[0], A.ExprStmt) and isinstance(body[0].expr, A.Str):
            self.emit(repr(body[0].expr.value))
            body = body[1:]
        if any(isinstance(x, A.FieldDef) for x in body):
            self.emit("__annotations__ = {}")
        if not body:
            self.emit("pass")
        for item in body:
            self.cur_line = (item.line, item.col)
            if isinstance(item, A.FnDef):
                self._method(item, cscope)
            elif isinstance(item, A.Let) and item.value is None:
                for n in item.targets:
                    self.declare(cscope, n, exact=True)
                self.emit("pass")
            elif isinstance(item, A.Let):
                val = self.expr(item.value, cscope)
                names = [self.declare(cscope, n, exact=True).py for n in item.targets]
                self.emit(f"{', '.join(names)} = {val}")
            elif isinstance(item, A.Assign) and isinstance(item.target, A.Name) and not item.more:
                val = self.expr(item.value, cscope)
                bb = cscope.names.get(item.target.id) or self.declare(cscope, item.target.id, exact=True)
                self.emit(f"{bb.py} = {val}")
            else:
                self.stmt(item, cscope)
        self.indent -= 1
        self.emit(f"{HELPER_PREFIX}own_class({b.py})")

    def _method(self, f, cscope):
        """A method: a plain Python function inside the class body (super() and the descriptor protocol work)."""
        b = self.declare(cscope, f.name, kind="fn", exact=True)
        ctx = FuncCtx(cscope.ctx.parent, "fn", erased=False, exact=cscope.ctx.exact, name=f.name)
        ctx.mangle_as = cscope.ctx.name
        if not f.decorators and f.params and f.params[0].name and f.params[0].star is None:
            ctx.self_param = f.params[0].name          # self.x in this method: a plain attribute of an instance
        # the body sees the scope around the class, not the class body (Python's rule)
        outer = cscope.parent
        self._pyfunction(b.py, f, Scope(outer, ctx), ctx)
        self.fn_names[b.py] = f.name

    def _pyfunction(self, pyname, f, fscope, ctx):
        """A function with Python's semantics: defaults evaluated once, *args a tuple, no type checks unless
        the parameters are annotated with Apsi types."""
        for d in f.decorators:
            self.emit(f"@{self.expr(d, fscope.parent)}")
        header_at = len(self.lines)
        self.emit("")
        self.indent += 1
        body = f.body
        if not f.is_expr and body and isinstance(body[0], A.ExprStmt) and isinstance(body[0].expr, A.Str):
            self.emit(repr(body[0].expr.value))
            body = body[1:]
        decl_at = len(self.lines)
        self.emit("")
        sig = []
        for p in f.params:
            if p.name is None:                       # bare * or /
                sig.append(p.star)
                continue
            pb = self.declare(fscope, p.name, kind="param", is_param=True)
            text = ("*" if p.star == "*" else "**" if p.star == "**" else "") + pb.py
            if p.default is not None:
                text += "=" + self.expr(p.default, fscope.parent)
            sig.append(text)
            if p.type is not None:
                if not any(l[0].strip() == f"{HELPER_PREFIX}b = {{}}" for l in self.lines[decl_at:]):
                    self.emit(f"{HELPER_PREFIX}b = {{}}")
                self.emit(f"{pb.py} = {self.checker(p.type)}({pb.py}, {HELPER_PREFIX}b, "
                          f"{'argument ' + repr(p.name) + ' of ' + ctx.name + '()'!r})")
        if f.ret is not None:
            ctx.ret_check = self.checker(f.ret)
            if not any(p.type is not None for p in f.params):
                self.emit(f"{HELPER_PREFIX}b = {{}}")
        if f.is_expr:
            self.cur_line = (body.line, body.col)
            val = self.expr(body, fscope)
            if f.ret is not None:
                val = f"{ctx.ret_check}({val}, {HELPER_PREFIX}b, {'return value of ' + ctx.name + '()'!r})"
            self.emit(f"return {val}")
        else:
            self.block(body, fscope)
        self.indent -= 1
        pad = "    " * self.indent
        self.lines[header_at] = (f"{pad}{'async ' if f.is_async else ''}def {pyname}({', '.join(sig)}):",
                                 self.lines[header_at][1])
        decls = []
        if ctx.globals:
            decls.append("global " + ", ".join(sorted(ctx.globals)))
        if ctx.nonlocals:
            decls.append("nonlocal " + ", ".join(sorted(ctx.nonlocals)))
        self.lines[decl_at] = (pad + "    " + ("; ".join(decls) if decls else "pass"), self.lines[decl_at][1])

    def s_Return(self, s, scope):
        val = self.expr(s.value, scope) if s.value is not None else "None"
        ret = getattr(scope.ctx, "ret_check", None)
        if ret and not scope.ctx.erased:
            val = self._ret(scope.ctx, val)
        self.emit(f"return {val}")

    def _ret(self, ctx, val):
        """The checked return value; Int/Float results of the expected class skip the checker call."""
        call = f"{ctx.ret_check}({{}}, {HELPER_PREFIX}b, {'return value of ' + ctx.name + '()'!r})"
        fast = getattr(ctx, "ret_fast", None)
        if fast and ctx.kind == "fn":
            r = f"{HELPER_PREFIX}ret"
            return f"({r} if ({r} := {val}).__class__ is {fast} else {call.format(r)})"
        return call.format(val)

    def s_Assert(self, s, scope):
        c = s.cond
        if self.pyc:                                   # Python's assert (kept even under python -O)
            self.emit(f"if not ({self.expr(c, scope)}):")
            msg = f"({self.expr(s.msg, scope)})" if s.msg is not None else ""
            self.emit(f"    raise AssertionError{msg if msg else '()'}" if not msg else f"    raise AssertionError{msg}")
            return
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
            b = self.declare(scope, s.path)                            # `use rl` also allows rl.ppo(...)
            self.emit(f"{b.py} = {HELPER_PREFIX}namespace({s.path!r})")

    def s_Import(self, s, scope):
        path = s.path if os.path.isabs(s.path) else os.path.join(self.rt.base_dir, s.path)
        tmp = self.fresh("m")
        if self.pyc and s.alias:                       # the module object itself (attributes, pickling)
            b = self.declare(scope, s.alias)
            self.emit(f"{b.py} = {HELPER_PREFIX}import_module({path!r})")
            return
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
        if self.pyc or s.is_async or any(p.name is None for p in s.params):
            b = self.declare(scope, s.name, kind="fn")
            ctx = FuncCtx(scope.ctx, "fn", erased=False, exact=scope.ctx.exact, name=s.name)
            self._pyfunction(b.py, s, Scope(scope, ctx), ctx)
            b.erased_py = None
            self.fn_names[b.py] = s.name
            return
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
            free = self._free_map(s, scope)
            self.emit(f"{HELPER_PREFIX}twin({b.py}, {epy}, {s.name!r}, {self.const(s, 'D')}, {self.const(free, 'F')})")
            for d in reversed(s.decorators):
                self.emit(f"{b.py} = {self.expr(d, scope)}({b.py})")
            if not s.decorators:
                self.emit(f"{b.py} = {HELPER_PREFIX}auto_native({b.py})")
            # unit-erased callers (evolve, hot loops) go through the native version when there is one
            self.emit(f"{epy} = {HELPER_PREFIX}native_e({b.py}, {epy})")
        self.fn_names[b.py] = s.name

    def _free_map(self, s, scope):
        """Apsi name -> (python name, mutable) for every outer name the function may read (for @native)."""
        body = [s.body] if s.is_expr else list(s.body)
        out = {}
        for n in _walk_ast(body + [p.default for p in s.params if p.default is not None]):
            if isinstance(n, A.Name) and n.id not in out:
                b = scope.lookup(n.id)
                if b is not None:
                    out[n.id] = (b.py, bool(b.mutable))
        return out

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
            if p.star == "*":
                self.emit(f"{py} = list({py})")
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
                # shape symbols are bound across parameters through one dict; scalar-only signatures need none
                shaped = bool(syms) or any(_has_shape(t) for t in [p.type for p in params] + [ret])
                self.emit(f"{HELPER_PREFIX}b = {{}}" if shaped else f"{HELPER_PREFIX}b = None")
            for p, py in pnames:
                if p.type is not None:
                    chk = self.checker(p.type)
                    call = f"{chk}({py}, {HELPER_PREFIX}b, {'argument ' + repr(p.name) + ' of ' + ctx.name + '()'!r})"
                    fast = _fast_class(p.type) if p.star is None else None
                    self.emit(f"{py} = {py} if {py}.__class__ is {fast} else {call}" if fast else f"{py} = {call}")
            if ret is not None:
                ctx.ret_check = self.checker(ret)
                ctx.ret_fast = _fast_class(ret)
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
                val = self._ret(ctx, val)
            self.emit(f"return {val}")
        else:
            self.block(body, fscope)
        self.indent -= 1
        sig = ", ".join(("*" if p.star == "*" else "**" if p.star == "**" else "") + py
                        + (f"={HELPER_PREFIX}NO" if p.default is not None else "") for p, py in pnames)
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
                  "in": "in", "not in": "not in", "is": "is", "is not": "is not", "|": "|", "&": "&", "^": "^",
                  "<<": "<<", ">>": ">>"}
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

    def e_Bytes(self, e, scope):
        return repr(e.value)

    def e_Chain(self, e, scope):
        parts = [self.expr(e.operands[0], scope)]
        for op, x in zip(e.ops, e.operands[1:]):
            parts.append(op)
            parts.append(self.expr(x, scope))
        return "(" + " ".join(parts) + ")"

    def e_SetLit(self, e, scope):
        if not e.items:
            return "set()"
        return "{" + ", ".join(self._item(x, scope) for x in e.items) + "}"

    def _item(self, x, scope):
        if isinstance(x, A.Spread):
            return ("**" if x.double else "*") + (self.expr(x.value, scope) if self.pyc
                                                  else f"{HELPER_PREFIX}iterable({self.expr(x.value, scope)})")
        return self.expr(x, scope)

    def e_SetComp(self, e, scope):
        inner, clauses = self._comp_clauses(e, scope)
        return f"{{{self.expr(e.elem, inner)} {clauses}}}"

    def e_GenExp(self, e, scope):
        inner, clauses = self._comp_clauses(e, scope)
        return f"({self.expr(e.elem, inner)} {clauses})"

    def e_NamedExpr(self, e, scope):
        b = scope.lookup(e.name)
        if b is None:
            s = scope
            while getattr(s, "is_comp", False) and s.parent is not None:
                s = s.parent
            b = self.declare(s, e.name, mutable=True)
            py = b.py
        else:
            py = self.store_target(e.name, scope)
        return f"({py} := {self.expr(e.value, scope)})"

    def e_AsyncIter(self, e, scope):
        return self.expr(e.value, scope)

    def e_Await(self, e, scope):
        return f"(await {self.expr(e.value, scope)})"

    def e_YieldExpr(self, e, scope):
        v = self.expr(e.value, scope) if e.value is not None else ""
        return f"(yield from {v})" if e.from_ else f"(yield {v})"

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
                continue
            ex, spec = p[0], p[1]
            conv = p[2] if len(p) > 2 else None
            v = self.expr(ex, scope)
            if conv:
                v = {"r": "repr", "s": "str", "a": "ascii"}[conv] + f"({v})" if self.pyc else \
                    f"{HELPER_PREFIX}conv({v}, {conv!r})"
            spec_code = self.expr(spec, scope) if isinstance(spec, A.FStr) else repr(spec)
            if self.pyc:
                parts.append(f"format({v}, {spec_code if spec is not None else repr('')})")
            else:
                parts.append(f"{HELPER_PREFIX}fmt({v}, True, {spec_code})")
        return "(" + " + ".join(parts) + ")" if parts else "''"

    def e_Unary(self, e, scope):
        v = self.expr(e.operand, scope)
        if e.op == "not":
            return f"(not {v})" if self.pyc else f"(not {HELPER_PREFIX}truth({v}))"
        return f"({e.op}{v})"

    def e_Binary(self, e, scope):
        return self.binop(e.op)(self.expr(e.left, scope), self.expr(e.right, scope))

    def e_Logic(self, e, scope):
        a, b = self.expr(e.left, scope), self.expr(e.right, scope)
        if self.pyc:
            return f"({a} {e.op} {b})"
        if e.op == "and":
            return f"({HELPER_PREFIX}T({a}) and {b})"
        return f"({HELPER_PREFIX}T({a}) or {b})"

    def e_IfExpr(self, e, scope):
        return f"({self.expr(e.a, scope)} if {self.cond(self.expr(e.cond, scope))} else {self.expr(e.b, scope)})"

    def e_Spread(self, e, scope):
        return ("**" if e.double else "*") + self.expr(e.value, scope)

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
            if k is None:
                parts.append(v)
                continue
            parts.append(f"{k}={v}" if k.isidentifier() and not keyword.iskeyword(k) else f"**{{{k!r}: {v}}}")
        return f"{fcode}({', '.join(parts)})"

    def e_Attr(self, e, scope):
        if self.pyc and not scope.ctx.erased:
            obj = self.expr(e.obj, scope)
            name = self._mangle(e.name, scope)
            if name.startswith("__") or name in _RISKY_ATTRS or keyword.iskeyword(name) or \
                    (name.startswith("_") and not self.trusted):
                return f"{HELPER_PREFIX}attr({obj}, {name!r})"
            if obj[:1].isdigit():
                obj = f"({obj})"                                 # 1024.to_bytes would lex as a float
            return f"{obj}.{_pyattr(name)}"
        direct = self._self_attr(e, scope)
        if direct:
            return direct
        code = f"{HELPER_PREFIX}attr({self.expr(e.obj, scope)}, {e.name!r})"
        return f"{HELPER_PREFIX}S({code})" if scope.ctx.erased else code

    def _self_attr(self, e, scope):
        """`self.name` inside a method of an Apsi class: Python's own attribute access (the instance is an
        ordinary object, the name public), with no helper call in between."""
        name = e.name
        if scope.ctx.erased or not isinstance(e.obj, A.Name) or getattr(scope.ctx, "self_param", None) != e.obj.id:
            return None
        if name.startswith("_") or name in _RISKY_ATTRS or name in ("shape", "tofile", "dump") or \
                keyword.iskeyword(name) or not name.isidentifier():
            return None
        b = scope.lookup(e.obj.id)
        if b is None or b.kind != "param" or b.ctx is not scope.ctx or b.mutable:
            return None
        return f"{b.py}.{name}"

    def index(self, n, scope):
        if self.pyc:
            if isinstance(n, A.Slice):
                parts = [self.expr(x, scope) if x is not None else "" for x in (n.lo, n.hi, n.step)]
                return ":".join(parts) if n.step is not None else ":".join(parts[:2])
            if isinstance(n, A.TupleLit):
                inner = ", ".join(self.index(x, scope) if not isinstance(x, A.Spread) else self._item(x, scope)
                                  for x in n.items)
                return inner + ("," if len(n.items) == 1 else "") if n.items else "()"
            return self.expr(n, scope)
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
        return "[" + ", ".join(self._item(x, scope) for x in e.items) + "]"

    def e_TupleLit(self, e, scope):
        items = [self._item(x, scope) for x in e.items]
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"

    def e_DictLit(self, e, scope):
        return "{" + ", ".join(("**" + self.expr(v.value, scope)) if k is None else
                               f"{self.expr(k, scope)}: {self.expr(v, scope)}" for k, v in e.pairs) + "}"

    def _comp_scope(self, targets, scope):
        inner = Scope(scope, scope.ctx, loop=True)
        inner.is_comp = True
        saved, self.pyc = self.pyc, False              # comprehension names are their own (Python 3)
        try:
            tgt = self.pattern(targets[0], inner) if len(targets) == 1 else self.pattern(list(targets), inner)
        finally:
            self.pyc = saved
        return inner, tgt

    def _comp_clauses(self, e, scope):
        parts = []
        inner = scope
        for targets, itx, cond in [(e.targets, e.iter, e.cond)] + list(e.more or []):
            it = self.expr(itx, inner)
            inner, tgt = self._comp_scope(targets, inner)
            c = f" if {self.cond(self.expr(cond, inner))}" if cond is not None else ""
            kw = "async for" if isinstance(itx, A.AsyncIter) else "for"
            parts.append(f"{kw} {tgt} in {it if self.pyc or kw != 'for' else HELPER_PREFIX + 'iterable(' + it + ')'}{c}")
        return inner, " ".join(parts)

    def e_ListComp(self, e, scope):
        inner, clauses = self._comp_clauses(e, scope)
        return f"[{self.expr(e.elem, inner)} {clauses}]"

    def e_DictComp(self, e, scope):
        inner, clauses = self._comp_clauses(e, scope)
        return f"{{{self.expr(e.key, inner)}: {self.expr(e.value, inner)} {clauses}}}"

    def e_Lambda(self, e, scope):
        if self.pyc:
            ctx = FuncCtx(scope.ctx, "fn", erased=False, exact=scope.ctx.exact, name="<lambda>")
            ctx.used = set(scope.ctx.used)
            ls = Scope(scope, ctx)
            ps = []
            for p in e.params:
                if p.name is None:
                    ps.append(p.star)
                    continue
                b = self.declare(ls, p.name, kind="param", is_param=True)
                ps.append((p.star or "") + b.py + (f"={self.expr(p.default, scope)}" if p.default is not None else ""))
            return f"(lambda {', '.join(ps)}: {self.expr(e.body, ls)})"

        def build(erased):
            ctx = FuncCtx(scope.ctx, "fn", erased=erased, exact=scope.ctx.exact, name="<lambda>")
            # lambdas share the enclosing python function's namespace for closures
            ctx.used = set(scope.ctx.used)
            ls = Scope(scope, ctx)
            ps = []
            for p in e.params:
                b = self.declare(ls, p.name, kind="param", is_param=True)
                ps.append((p.star or "") + b.py + (f"={self.expr(p.default, scope)}" if p.default is not None else ""))
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


def _pyattr(name):
    """An attribute name usable after '.' in Python (Apsi allows a few that are Python keywords)."""
    return name if not keyword.iskeyword(name) else name   # a keyword attribute cannot be written; kept as is


def _walk_ast(nodes):
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, A.AST):
            yield n
            for f in n._fields:
                stack.append(getattr(n, f))
        elif isinstance(n, (list, tuple)):
            stack.extend(n)


def _collect_syms(t, out):
    if isinstance(t, A.TName) and t.name in ("Tensor", "Vec", "Mat", "Matrix"):
        for a in t.args:
            if isinstance(a, A.TName) and not a.args and a.name not in DIM_NAMES \
                    and a.name not in ("f16", "f32", "f64", "i32", "i64", "bool", "_", "bf16", "fp8_e4m3", "fp8_e5m2",
                                   "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"):
                if a.name not in out:
                    out.append(a.name)


_SHAPED = {"Tensor", "Vec", "Mat", "Matrix"}


def _has_shape(t):
    """Whether a type annotation mentions a tensor type (whose shape symbols bind across parameters)."""
    if t is None:
        return False
    if isinstance(t, A.TName):
        return t.name in _SHAPED or any(_has_shape(a) for a in (t.args or []))
    return not isinstance(t, A.TUnit)


def _fast_class(t):
    """'int' / 'float' when values of exactly that class always pass the checker for t."""
    if isinstance(t, A.TName) and not t.args:
        if t.name == "Int":
            return f"{HELPER_PREFIX}int"                # the real Python classes: Apsi's own int/float shadow them
        if t.name in ("Float", "Number", "Real"):
            return f"{HELPER_PREFIX}float"
    return None


# =========================================================================== runtime support
class Runtime:
    """Helpers referenced by generated code (all exposed under the `_A` prefix)."""

    def __init__(self, interp, filename, base_dir):
        self.interp = interp
        self.filename = filename
        self.base_dir = base_dir
        self.structs = {}
        self._checkers = {}                 # id(type AST) -> (type AST, checker): one checker per annotation
        self._plans = {}                    # id(struct) -> (struct, [(field, checker, message)])
        structs = self.structs

        class _Env:
            def lookup(self, name):
                if name in structs:
                    return structs[name]
                raise KeyError(name)
        self._env = _Env()

    # ---- type checks
    def make_checker(self, t):
        hit = self._checkers.get(id(t))
        if hit is not None and hit[0] is t:
            return hit[1]
        chk = self._make_checker(t)
        self._checkers[id(t)] = (t, chk)
        return chk

    def _make_checker(self, t):
        interp = self.interp
        env = self._env
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
        if not kwargs and len(args) == len(st.fields):          # the common case: every field by position
            plan = self._plans.get(id(st))
            if plan is None or plan[0] is not st:
                plan = (st, [(name, None if t is None else self.make_checker(t), f"field '{name}' of {st.name}")
                             for name, t, _d in st.fields])
                self._plans[id(st)] = plan
            values, bindings = {}, {}
            for v, (name, chk, what) in zip(args, plan[1]):
                values[name] = v if chk is None else chk(v, bindings, what)
            return StructInstance(st, values)
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
_PLAIN_TRUTH = frozenset((bool, int, float, str, type(None), list, dict, tuple))


def _truth(v):
    if v.__class__ in _PLAIN_TRUTH:                  # conditions are nearly always plain values
        return v
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


def _twin(f, fe, name, node=None, free=None):
    f._apsi_e = fe
    f._apsi_def = node
    f._apsi_free = free
    f.name = name
    fe.name = name
    return f


_AUTO = None


def _exm(ex, kinds, classes=()):
    from .exceptions import matches
    return matches(ex, kinds, classes)


def _exv(ex):
    from .exceptions import value
    return value(ex)


def _raise(v):
    from .exceptions import to_raise
    return to_raise(v)


def _auto_native(f):
    """`apsi run --native` (APSI_NATIVE=auto): every function gets a native tier with silent fallback."""
    global _AUTO
    if _AUTO is None:
        _AUTO = os.environ.get("APSI_NATIVE", "").strip().lower() in ("1", "auto", "on", "true")
    if not _AUTO:
        return f
    from .native import auto
    return auto(f)


def _native_e(f, fe):
    e = getattr(f, "_apsi_native_e", None)
    return e if e is not None else fe


def _dual(f, fe):
    try:
        f._apsi_e = fe
        f.name = "<lambda>"
    except AttributeError:
        pass
    return f


from .idioms import gemm as _gemm  # noqa: E402


def _irange(a, b, inclusive):
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return range(int(a), int(b) + (1 if inclusive else 0))
    return RangeVal(a, b, inclusive)


def _idx(v):
    t = type(v)
    if t is int or t is str:
        return v
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
        from .pycompat import trusted as _trusted
        gen.trusted = _trusted(interp)
        module_ctx = FuncCtx(None, "module")
        module_scope = Scope(None, module_ctx)
        gen.program(prog, module_scope)
        self.gen = gen
        self.source = gen.source()
        self.linemap = gen.linemap()
        self.module_scope = module_scope
        from . import compile_cache as _cc
        from .debugger import register as _dbg_register
        _dbg_register(f"<apsi:{filename}>", filename, self.linemap, gen.fn_names,
                      module_ctx.used)
        self.code = _cc.load_code(self.source, f"<apsi:{filename}>")
        if self.code is None:
            try:
                self.code = compile(self.source, f"<apsi:{filename}>", "exec")
            except SyntaxError as e:                     # never a Python traceback: an Apsi error at the Apsi line
                ln = e.lineno or 0
                line, col = self.linemap[ln - 1] if 0 < ln <= len(self.linemap) else (None, None)
                raise RuntimeErr(f"this cannot run: {e.msg}", line, col) from None
            _cc.store_code(self.source, f"<apsi:{filename}>", self.code)
        import types as _types
        modname = getattr(interp, "_next_module_name", None) or (
            "__main__" if not filename.startswith("<") or gen.pyc else "__apsi__")
        interp._next_module_name = None
        self.pyc = gen.pyc
        self.module = _types.ModuleType(modname)
        self.module.__file__ = os.path.abspath(filename) if not filename.startswith("<") else filename
        g = self.module.__dict__
        g.update(builtins)
        g.update(gen.consts)
        rt = self.rt
        g.update({
            "__name__": modname, "__file__": self.module.__file__,
            f"{HELPER_PREFIX}ctx": _ctx, f"{HELPER_PREFIX}conv": _conv, f"{HELPER_PREFIX}kind": _kind,
            f"{HELPER_PREFIX}own_class": own_class,
            f"{HELPER_PREFIX}AttrSlot": _attr_slot(interp),
            f"{HELPER_PREFIX}import_module": lambda p: interp.import_module(p),
            f"{HELPER_PREFIX}NO": NO, f"{HELPER_PREFIX}S": _strip, f"{HELPER_PREFIX}E": _erased,
            f"{HELPER_PREFIX}twin": _twin, f"{HELPER_PREFIX}dual": _dual,
            f"{HELPER_PREFIX}auto_native": _auto_native, f"{HELPER_PREFIX}native_e": _native_e,
            f"{HELPER_PREFIX}truth": _truth, f"{HELPER_PREFIX}T": _truth, f"{HELPER_PREFIX}all": _all,
            f"{HELPER_PREFIX}irange": _irange, f"{HELPER_PREFIX}gemm": _gemm,
            f"{HELPER_PREFIX}int": int, f"{HELPER_PREFIX}float": float, f"{HELPER_PREFIX}bool": bool, f"{HELPER_PREFIX}iterable": _iterable,
            f"{HELPER_PREFIX}idx": _idx, f"{HELPER_PREFIX}unpack": _unpack,
            f"{HELPER_PREFIX}Range": RangeVal, f"{HELPER_PREFIX}as_unit": _as_unit,
            f"{HELPER_PREFIX}fmt": fmt, f"{HELPER_PREFIX}approx": ops.approx, f"{HELPER_PREFIX}pm": ops.plusminus,
            f"{HELPER_PREFIX}attr": interp.get_attr, f"{HELPER_PREFIX}setattr": interp.set_attr,
            f"{HELPER_PREFIX}setidx": _setidx(interp),
            f"{HELPER_PREFIX}AssertionFailed": AssertionFailed,
            f"{HELPER_PREFIX}exm": _exm, f"{HELPER_PREFIX}Exception": Exception, f"{HELPER_PREFIX}exv": _exv, f"{HELPER_PREFIX}raise_": _raise,
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
        if self.module.__name__ == "__main__":
            sys.modules["__main__"] = self.module          # as when Python runs a script (pickle finds its names)
            _tune_gc()
        elif self.module.__name__ != "__apsi__":
            sys.modules[self.module.__name__] = self.module  # during import already (dataclasses, typing need it)
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
        if self.pyc:                                   # Python-exact files report Python's own error type
            err = RuntimeErr(msg)
            err.kind = type(e).__name__
            err.original = e
            if isinstance(e, KeyError) and e.args:
                err.msg = repr(e.args[0])
        elif isinstance(e, TypeError) and ("positional argument" in msg or "unexpected keyword" in msg
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


def _tune_gc():
    """Before the main program runs: everything Apsi and the libraries loaded is frozen (the cycle collector never
    scans it again), and the first generation is collected every 50,000 allocations instead of 2,000. Programs that
    build many objects run 5–10 % faster; objects are still freed the moment their last reference goes (reference
    counting is untouched), only unreachable cycles wait a little longer. APSI_GC=default keeps Python's settings."""
    import gc
    if os.environ.get("APSI_GC", "").lower() in ("default", "python", "off"):
        return
    if gc.get_threshold()[0] == 2000 and gc.isenabled():
        gc.collect()
        gc.freeze()
        gc.set_threshold(50_000, 20, 20)


import weakref as _weakref
OWN_CLASSES = _weakref.WeakSet()       # classes defined by Apsi programs: their _private attributes are their own


def own_class(cls):
    try:
        OWN_CLASSES.add(cls)
    except TypeError:
        pass
    return cls


def is_own(obj):
    t = obj if isinstance(obj, type) else type(obj)
    try:
        return any(c in OWN_CLASSES for c in t.__mro__)
    except Exception:                      # noqa: BLE001
        return False


class _StructCtx:
    """`with` on an Apsi struct that has enter()/exit() methods (or __enter__/__exit__)."""

    def __init__(self, obj):
        self.obj = obj

    def _m(self, *names):
        for n in names:
            m = self.obj.type.methods.get(n)
            if m is not None:
                return m
        return None

    def __enter__(self):
        m = self._m("enter", "__enter__")
        return m(self.obj) if m is not None else self.obj

    def __exit__(self, et, ev, tb):
        m = self._m("exit", "__exit__")
        if m is None:
            return False
        try:
            return bool(m(self.obj, et, ev, tb))
        except TypeError:
            return bool(m(self.obj))


def _ctx(v):
    if isinstance(v, StructInstance):
        return _StructCtx(v)
    return v


def _conv(v, c):
    from .stdlib import _repr
    if c == "r":
        return _repr(v)
    if c == "a":
        return ascii(v)
    return fmt(v)


def _kind(k):
    from .exceptions import KINDS, ALIASES
    base = ALIASES.get(k, k).split(".")[-1]
    return KINDS.get(base, (Exception,))


def _attr_slot(interp):
    class AttrSlot:
        __slots__ = ("o", "n")

        def __init__(self, o, n):
            self.o, self.n = o, n

        @property
        def value(self):
            return interp.get_attr(self.o, self.n)

        @value.setter
        def value(self, v):
            interp.set_attr(self.o, self.n, v)

        @value.deleter
        def value(self):
            interp.del_attr(self.o, self.n)
    return AttrSlot


def _setidx(interp):
    set_index = interp.set_index

    def setidx(obj, idx, v):
        t = type(obj)
        if t is dict or t is list:                   # the common case without another call
            obj[idx] = v
        else:
            set_index(obj, idx, v)
    return setidx
