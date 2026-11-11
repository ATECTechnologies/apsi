"""Static checker for Apsi: names, mutability, arity, physical units, evolve rules.

Runs before execution. It is conservative: it only reports what it can prove
from literals, annotations and declared signatures, so valid programs pass.
"""
import os
from fractions import Fraction

from . import ast as A
from .errors import TypeCheckError, UnitError, NameErr, ParseError, LexError
from .units import Dim, DIMLESS, DIM_NAMES, dim_name
from .parser import parse
from . import ops

ZERO = "ZERO"          # literal 0: compatible with any unit in + and -

TENSOR_TYPES = {"Tensor", "Vec", "Mat", "Matrix", "Vec2", "Vec3", "Vec4"}
DTYPE_NAMES = {"f16", "f32", "f64", "i32", "i64", "bool", "bf16", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2",
               "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"}
PLAIN_TYPES = {"Bool", "Str", "List", "Tuple", "Dict", "Any", "Fn", "Record", "None"}
NUM_TYPES = {"Float", "Number", "Real", "Int", "f64", "f32", "f16", "bf16"}   # f32 …: a Float in that precision

DIMLESS_FNS = {"exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
               "sinh", "cosh", "tanh", "sigmoid", "softmax", "log_softmax", "gelu", "softplus"}
SAME_FNS = {"abs", "sum", "mean", "std", "norm", "relu", "clip", "roll", "floor", "ceil", "round",
            "cumsum", "sort", "flatten", "reshape", "transpose", "zeros_like", "ones_like"}


def isD(x):
    return isinstance(x, Dim)


class Info:
    __slots__ = ("dim", "mutable", "sig", "kind", "decl")

    def __init__(self, dim=None, mutable=False, sig=None, kind="var", decl=None):
        self.dim, self.mutable, self.sig, self.kind, self.decl = dim, mutable, sig, kind, decl


class Sig:
    def __init__(self, name, params, ret):
        self.name, self.params, self.ret = name, params, ret   # params: [(name, dim, has_default)]


class Scope:
    def __init__(self, parent=None):
        self.names = {}
        self.parent = parent

    def lookup(self, n):
        s = self
        while s is not None:
            if n in s.names:
                return s.names[n]
            s = s.parent
        return None


class Checker:
    def __init__(self, interp, base_dir):
        self.interp = interp
        self.base_dir = base_dir
        self.errors = []
        self.builtins = interp.globals.vars
        self.modules = interp.modules
        self.fn_stack = []
        self.evolve_stack = []
        self.deferred = []
        self.struct_names = set()
        self.refs = []           # (line, col, name, Info) for every resolved name — used by the LSP
        self.decls = []          # (line, col, name, Info) for declarations
        self.deps = []           # files read while checking (imports): a cached "checked clean" depends on them

    # ------------------------------------------------------------- utils
    def err(self, node, msg, cls=TypeCheckError):
        self.errors.append(cls(msg, node.line, node.col))

    def lookup(self, name, sc, node=None):
        info = sc.lookup(name)
        if info is not None:
            return info
        if name in ("__name__", "__file__", "__doc__"):
            return Info(None, False, None, "builtin")
        if name in self.builtins:
            v = self.builtins[name]
            return Info(ops.dim_of(v) if not callable(v) else None, False, None, "builtin")
        if node is not None and not self.pyc:
            # Python-mode programs keep Python's rule: an unknown name is a NameError when (and if) it runs
            from .errors import did_you_mean
            known = set(self.builtins)
            s_ = sc
            while s_ is not None:
                known.update(s_.names)
                s_ = s_.parent
            lit = {"True": "true", "False": "false", "None": "none", "null": "none", "nil": "none", "NULL": "none",
                   "undefined": "none", "this": "self"}.get(name)
            hint = f" (Apsi writes '{lit}')" if lit else did_you_mean(name, known)
            self.err(node, f"'{name}' is not defined" + hint, NameErr)
        return None

    def type_dim(self, t, sc, node):
        if t is None:
            return None
        if isinstance(t, A.TUnit):
            return t.dim
        n = t.name
        if n in DIM_NAMES:
            return DIM_NAMES[n]
        if n in NUM_TYPES:
            return DIMLESS
        if n in TENSOR_TYPES:
            dim = DIMLESS
            for a in t.args:
                if isinstance(a, A.TUnit):
                    dim = a.dim
                elif isinstance(a, A.TName) and a.name in DIM_NAMES:
                    dim = DIM_NAMES[a.name]
            return dim
        if n in PLAIN_TYPES:
            return None
        if n in self.struct_names or (sc.lookup(n) is not None and sc.lookup(n).kind == "struct"):
            return None
        self.err(t, f"unknown type '{n}'")
        return None

    def shape_symbols(self, t, out):
        if isinstance(t, A.TName) and t.name in TENSOR_TYPES:
            for a in t.args:
                if isinstance(a, A.TName) and not a.args and a.name not in DIM_NAMES \
                        and a.name not in DTYPE_NAMES and a.name != "_":
                    out.add(a.name)

    def const_num(self, e):
        if isinstance(e, A.Num) and e.unit is None:
            return Fraction(str(e.value))
        if isinstance(e, A.Unary) and e.op == "-":
            v = self.const_num(e.operand)
            return -v if v is not None else None
        if isinstance(e, A.Binary) and e.op == "/":
            a, b = self.const_num(e.left), self.const_num(e.right)
            if a is not None and b:
                return a / b
        return None

    # ------------------------------------------------------------- program
    def check(self, prog):
        self.pyc = is_python_mode(prog)
        for s in prog.body:
            if isinstance(s, (A.StructDef, A.ClassDef)):
                self.struct_names.add(s.name)
        top = Scope()
        self.block(prog.body, top, top_level=True)
        while self.deferred:
            item = self.deferred.pop(0)
            item()
        return self.errors

    def block(self, stmts, sc, top_level=False):
        for s in stmts:
            m = getattr(self, "c_" + type(s).__name__, None)
            if m:
                m(s, sc, top_level)

    # ------------------------------------------------------------- statements
    def c_ExprStmt(self, s, sc, top):
        self.infer(s.expr, sc)

    def c_Let(self, s, sc, top):
        d = self.infer(s.value, sc) if s.value is not None else None
        if len(s.targets) == 1:
            declared = self.type_dim(s.type, sc, s) if s.type is not None else None
            if isD(declared) and isD(d) and d != declared:
                self.err(s, f"'{s.targets[0]}' is declared as {dim_name(declared)} but the value is "
                            f"{dim_name(d)}", UnitError)
            dim = declared if declared is not None else (None if d == ZERO else d)
            self._decl(sc, s.targets[0], Info(dim, s.mutable), s)
        else:
            for n in pattern_names(s.targets):
                self._decl(sc, n, Info(None, s.mutable), s)

    def _assign_name(self, node, name, d, sc):
        info = self.lookup(name, sc, None)
        if info is None:
            self.err(node, f"'{name}' is not defined; declare it first with let or var", NameErr)
            return None
        if not info.mutable and not self.pyc:
            what = "a parameter" if info.kind == "param" else (
                "a function" if info.kind in ("fn", "builtin") else "declared with let")
            self.err(node, f"cannot assign to '{name}': it is {what} (use var for mutable variables)")
            return None
        return info

    def c_Assign(self, s, sc, top):
        d = self.infer(s.value, sc)
        self._target(s.target, d, sc, s)
        for t in s.more or ():
            self._target(t, d, sc, s)

    def _target(self, t, d, sc, s):
        if isinstance(t, A.Name):
            info = self._assign_name(t, t.id, d, sc)
            if info and isD(info.dim) and isD(d) and info.dim != d:
                self.err(t, f"'{t.id}' holds {dim_name(info.dim)}; cannot assign {dim_name(d)}", UnitError)
        elif isinstance(t, (A.TupleLit, A.ListLit)):
            for x in t.items:
                self._target(x, None, sc, s)
        elif isinstance(t, A.Spread):
            self._target(t.value, None, sc, s)
        elif isinstance(t, A.Attr):
            self.infer(t.obj, sc)
        elif isinstance(t, A.Index):
            self.infer(t.obj, sc)
            self.infer_index(t.index, sc)

    def c_AugAssign(self, s, sc, top):
        d = self.infer(s.value, sc)
        t = s.target
        if isinstance(t, A.Name):
            info = self._assign_name(t, t.id, d, sc)
            if info and isD(info.dim) and isD(d):
                if s.op in ("+", "-") and info.dim != d:
                    self.err(s, f"cannot {'add' if s.op == '+' else 'subtract'} {dim_name(d)} "
                                f"{'to' if s.op == '+' else 'from'} '{t.id}' ({dim_name(info.dim)})", UnitError)
                if s.op in ("*", "/") and d != DIMLESS:
                    self.err(s, f"'{t.id} {s.op}= ...' would change the units of '{t.id}'", UnitError)
        else:
            self._target(t, None, sc, s)

    def _sig(self, s, sc):
        params = [(p.name, self.type_dim(p.type, sc, p) if p.type is not None else None,
                   p.default is not None) for p in s.params if not p.star]
        if any(p.star in ("*", "/") and p.name is None for p in s.params):
            sig = Sig(s.name, params, None)                    # keyword-only / positional-only markers:
            sig.varargs = sig.varkw = True                     # arity is checked when the call runs
            return sig
        ret = self.type_dim(s.ret, sc, s) if s.ret is not None else None
        sig = Sig(s.name, params, ret)
        sig.varargs = any(p.star == "*" for p in s.params)
        sig.varkw = any(p.star == "**" for p in s.params)
        return sig

    def c_FnDef(self, s, sc, top):
        for d in s.decorators:
            self.infer(d, sc)
        sig = self._sig(s, sc)
        if s.decorators and self.pyc:
            sig = None                                         # a decorator may change the signature
        self._decl(sc, s.name, Info(None, self.pyc, sig, "fn"), s)
        if sig is None:
            sig = self._sig(s, sc)
        if top or self.pyc:
            self.deferred.append(lambda: self.fn_body(s, sc, sig))
        else:
            self.fn_body(s, sc, sig)

    def fn_body(self, s, sc, sig, extra=None):
        fsc = Scope(sc)
        fsc.fn_root = True
        for k, v in (extra or {}).items():
            fsc.names[k] = v
        syms = set(s.shape_params)
        for p in s.params:
            self.shape_symbols(p.type, syms)
        if s.ret is not None:
            self.shape_symbols(s.ret, syms)
        for sym in syms:
            fsc.names[sym] = Info(DIMLESS, False, kind="param")
        for p in s.params:
            if p.star and p.name is not None:
                self._decl(fsc, p.name, Info(None, self.pyc, kind="param"), p)
        for p, (_, d, _) in zip([q for q in s.params if not q.star], self._sig_plain(s, sc)):
            if p.default is not None:
                pd = self.infer(p.default, sc)
                if isD(d) and isD(pd) and pd != d:
                    self.err(p, f"default value of '{p.name}' is {dim_name(pd)}, expected {dim_name(d)}", UnitError)
            self._decl(fsc, p.name, Info(d, self.pyc, kind="param"), p)
        self.fn_stack.append(sig)
        if s.is_expr:
            self.check_ret(s.body, self.infer(s.body, fsc), sig)
        else:
            self.block(s.body, fsc)
        self.fn_stack.pop()

    def _sig_plain(self, s, sc):
        return [(p.name, self.type_dim(p.type, sc, p) if p.type is not None else None, p.default is not None)
                for p in s.params if not p.star]

    def check_ret(self, node, d, sig):
        if isD(sig.ret) and isD(d) and d != sig.ret:
            self.err(node, f"{sig.name}() should return {dim_name(sig.ret)} but this returns "
                           f"{dim_name(d)}", UnitError)

    def c_Return(self, s, sc, top):
        if not self.fn_stack:
            self.err(s, "'return' outside of a function")
            return
        if self.fn_stack[-1] is None:
            if s.value is not None:
                self.infer(s.value, sc)
            return
        d = self.infer(s.value, sc) if s.value is not None else None
        self.check_ret(s, d, self.fn_stack[-1])

    def c_StructDef(self, s, sc, top):
        self._decl(sc, s.name, Info(None, False, None, "struct"), s)
        for f in s.fields:
            self.type_dim(f.type, sc, f)
            if f.default is not None:
                self.infer(f.default, sc)
        for m in s.methods:
            sig = self._sig(m, sc)
            extra = {}
            self.deferred.append(lambda m=m, sig=sig: self.fn_body(m, sc, sig))

    # ---- the Python object model: classes, with, del, global
    def c_ClassDef(self, s, sc, top):
        for d in s.decorators:
            self.infer(d, sc)
        for b in s.bases:
            self.infer(b, sc)
        for _k, v in s.kwargs:
            self.infer(v, sc)
        self._decl(sc, s.name, Info(None, self.pyc, None, "struct"), s)
        csc = Scope(sc)
        csc.is_class = True
        for item in s.body:
            if isinstance(item, A.FieldDef):
                if item.value is not None:
                    self.infer(item.value, csc)
                self._decl(csc, item.name, Info(None, True), item)
            elif isinstance(item, A.FnDef):
                for d in item.decorators:
                    self.infer(d, csc)
                sig = self._sig(item, sc)
                csc.names[item.name] = Info(None, True, None, "fn")
                # methods see the enclosing scope, not the class body (Python's rule)
                self.deferred.append(lambda m=item, sig=sig: self.fn_body(m, sc, sig))
            else:
                m = getattr(self, "c_" + type(item).__name__, None)
                if m:
                    m(item, csc, False)

    def c_FieldDef(self, s, sc, top):
        if s.value is not None:
            self.infer(s.value, sc)
        self._decl(sc, s.name, Info(None, True), s)

    def c_With(self, s, sc, top):
        ws = Scope(sc) if not self.pyc else sc
        for e, target in s.items:
            self.infer(e, sc)
            if target is not None:
                for n in pattern_names(target):
                    if self.pyc and _local_lookup(sc, n) is not None:
                        continue
                    self._decl(ws, n, Info(None, True), s)
        self.block(s.body, ws if not self.pyc else Scope(sc))

    def c_Match(self, s, sc, top):
        self.infer(s.subject, sc)
        for pat, guard, body in s.cases:
            inner = Scope(sc)
            self._pattern(pat, sc, inner, s)
            if guard is not None:
                self.infer(guard, inner)
            self.block(body, inner)

    def _pattern(self, p, sc, inner, node):
        k = p[0]
        if k in ("capture",):
            self._decl(inner, p[1], Info(None, True), node)
        elif k == "star" and p[1]:
            self._decl(inner, p[1], Info(None, True), node)
        elif k == "as":
            self._pattern(p[1], sc, inner, node)
            self._decl(inner, p[2], Info(None, True), node)
        elif k in ("or", "seq"):
            for x in p[1]:
                self._pattern(x, sc, inner, node)
        elif k == "map":
            for key, v in p[1]:
                if not isinstance(key, tuple):
                    self.infer(key, sc)
                self._pattern(v, sc, inner, node)
            if p[2]:
                self._decl(inner, p[2], Info(None, True), node)
        elif k == "class":
            self.lookup(p[1].split(".")[0], sc, node)
            for x in p[2]:
                self._pattern(x, sc, inner, node)
            for _n, x in p[3]:
                self._pattern(x, sc, inner, node)
        elif k == "value":
            self.lookup(p[1].split(".")[0], sc, node)
        elif k == "lit" and not isinstance(p[1], tuple):
            self.infer(p[1], sc)

    def i_AsyncIter(self, e, sc):
        self.infer(e.value, sc)
        return None

    def c_Del(self, s, sc, top):
        for t in s.targets:
            if isinstance(t, A.Name):
                self.lookup(t.id, sc, t)
            else:
                self.infer(t, sc)

    def c_Global(self, s, sc, top):
        for n in s.names:
            if self.lookup(n, sc, None) is None:
                self.err(s, f"'{n}' is not defined in an enclosing scope (declare it there with var)", NameErr)

    def c_If(self, s, sc, top):
        for cond, body in s.branches:
            self.infer(cond, sc)
            self.block(body, Scope(sc))
        if s.else_body:
            self.block(s.else_body, Scope(sc))

    def c_Try(self, s, sc, top):
        from .exceptions import KINDS, ALIASES
        self.block(s.body, Scope(sc))
        for kinds, name, body in s.handlers:
            for k in kinds:
                base = ALIASES.get(k, k)
                if base not in KINDS and self.lookup(base.split(".")[0], sc, None) is None \
                        and not (self.pyc and base in _PY_EXC):
                    self.err(s, f"unknown error kind '{k}' (use a kind such as ValueError, KeyError, IndexError, "
                                f"ZeroDivisionError, UnitError, OSError, or `except as e` for every error)", NameErr)
            hs = Scope(sc)
            if name:
                self._decl(hs, name, Info(None, False), s)
            self.block(body, hs)
        if s.else_body:
            self.block(s.else_body, Scope(sc))
        if s.final_body:
            self.block(s.final_body, Scope(sc))

    def c_Yield(self, s, sc, top):
        if not self.fn_stack:
            self.err(s, "'yield' is only allowed inside a function", TypeCheckError)
        if s.value is not None:
            self.infer(s.value, sc)

    def c_Raise(self, s, sc, top):
        if s.value is not None:
            self.infer(s.value, sc)
        if s.cause is not None:
            self.infer(s.cause, sc)

    def c_While(self, s, sc, top):
        self.infer(s.cond, sc)
        self.block(s.body, Scope(sc))
        if s.else_body:
            self.block(s.else_body, Scope(sc))

    def c_For(self, s, sc, top):
        self.infer(s.iter, sc)
        fs = Scope(sc)
        for n in pattern_names(s.targets):
            if self.pyc and _local_lookup(sc, n) is not None:
                continue                                       # Python: the loop assigns the existing variable
            self._decl(fs, n, Info(None, self.pyc), s)
        self.block(s.body, fs)
        if s.else_body:
            self.block(s.else_body, Scope(sc))

    def c_Use(self, s, sc, top):
        if s.path not in self.modules:
            try:
                self.interp.ensure_module(s.path, self.base_dir)
            except Exception as e:                       # noqa: BLE001 - report library load failures as errors
                self.err(s, f"library '{s.path}' failed to load: {getattr(e, 'msg', e)}", NameErr)
                return
        if s.path not in self.modules:
            self.err(s, f"unknown module '{s.path}'. Available: {', '.join(sorted(self.modules))}. Install a library with: "
                            f"apsi install {s.path}", NameErr)
            return
        if s.alias:
            sc.names[s.alias] = Info(None, False, kind="module")
        else:
            members = self.modules[s.path]
            for k, v in members.items():
                sc.names[k] = Info(ops.dim_of(v) if not callable(v) else None, False, None, "builtin")
            if "." not in s.path and s.path not in members and sc.lookup(s.path) is None and s.path not in self.builtins:
                sc.names[s.path] = Info(None, False, kind="module")      # `use rl` also allows rl.ppo(...)

    def c_Import(self, s, sc, top):
        path = s.path if os.path.isabs(s.path) else os.path.join(self.base_dir, s.path)
        if not os.path.exists(path):
            self.err(s, f"cannot find file '{s.path}'", NameErr)
            return
        if s.alias:
            self._decl(sc, s.alias, Info(None, False, kind="module"), s)
            return
        try:
            self.deps.append(path)
            with open(path, encoding="utf-8-sig") as fh:
                prog = parse(fh.read(), cache=True)
        except (ParseError, LexError) as e:
            self.err(s, f"cannot parse '{s.path}': {e.msg}")
            return
        for st in prog.body:
            names = []
            if isinstance(st, A.Let):
                names = st.targets
            elif isinstance(st, (A.FnDef, A.StructDef)):
                names = [st.name]
            for n in names:
                if not n.startswith("_"):
                    sc.names[n] = Info(None, False, None, "struct" if isinstance(st, A.StructDef) else "builtin")

    def c_Test(self, s, sc, top):
        self.deferred.append(lambda: self.block(s.body, Scope(sc)))

    def c_Assert(self, s, sc, top):
        self.infer(s.cond, sc)
        if s.msg is not None:
            self.infer(s.msg, sc)

    def c_Evolve(self, s, sc, top):
        state_dims = {}
        for n in s.states:
            info = self.lookup(n, sc, None)
            if info is None:
                self.err(s, f"evolve state '{n}' is not defined", NameErr)
            elif not info.mutable:
                self.err(s, f"evolve state '{n}' must be declared with var")
            state_dims[n] = info.dim if info else None
        tdim = None
        if isinstance(s.range, A.Range):
            a, b = self.infer(s.range.start, sc), self.infer(s.range.end, sc)
            if isD(a) and isD(b) and a != b:
                self.err(s.range, f"time range mixes {dim_name(a)} and {dim_name(b)}", UnitError)
            tdim = a if isD(a) else (b if isD(b) else None)
        else:
            self.infer(s.range, sc)
        sd = self.infer(s.step, sc)
        if isD(tdim) and isD(sd) and tdim != sd:
            self.err(s.step, f"step is {dim_name(sd)} but the time range is {dim_name(tdim)}", UnitError)
        tdim = tdim if isD(tdim) else (sd if isD(sd) else None)
        if s.every is not None:
            self.infer(s.every, sc)
        if s.tol is not None:
            td = self.infer(s.tol, sc)
            if isD(td) and td != DIMLESS:
                self.err(s.tol, "tol is a relative tolerance and must be dimensionless", UnitError)
        if s.method == "verlet" and len(s.states) != 2:
            self.err(s, "verlet needs exactly two states: position, velocity")
        esc = Scope(sc)
        esc.names[s.tvar] = Info(tdim, False)
        for n in s.states:
            esc.names[n] = Info(state_dims[n], False)
        ctx = {"states": state_dims, "tdim": tdim, "seen": set()}
        self.evolve_stack.append(ctx)
        self.block(s.body, esc)
        self.evolve_stack.pop()
        for n in s.states:
            if n not in ctx["seen"]:
                self.err(s, f"evolve has no derivative for '{n}' (add a line: {n}' = ...)")
        if s.into:
            self._decl(sc, s.into, Info(None, False), s)

    def c_Deriv(self, s, sc, top):
        d = self.infer(s.expr, sc)
        if not self.evolve_stack:
            self.err(s, f"{s.name}' = ... is only allowed inside an evolve block")
            return
        ctx = self.evolve_stack[-1]
        if s.name not in ctx["states"]:
            self.err(s, f"'{s.name}' is not a state of this evolve")
            return
        ctx["seen"].add(s.name)
        sd, td = ctx["states"][s.name], ctx["tdim"]
        if isD(sd) and isD(td) and isD(d):
            want = sd / td
            if d != want:
                self.err(s, f"{s.name}' must have units of {dim_name(want)} ({s.name} per unit time), "
                            f"but this is {dim_name(d)}", UnitError)

    # ------------------------------------------------------------- expressions
    def infer(self, e, sc):
        m = getattr(self, "i_" + type(e).__name__, None)
        return m(e, sc) if m else None

    def infer_index(self, e, sc):
        if isinstance(e, A.Slice):
            for x in (e.lo, e.hi, e.step):
                if x is not None:
                    self.infer(x, sc)
        elif isinstance(e, A.TupleLit):
            for x in e.items:
                self.infer_index(x, sc)
        else:
            self.infer(e, sc)

    def i_Num(self, e, sc):
        if e.unit is not None:
            return e.unit[2]
        return ZERO if e.value == 0 else DIMLESS

    def i_Name(self, e, sc):
        info = self.lookup(e.id, sc, e)
        if info is not None:
            self.refs.append((e.line, e.col, e.id, info))
        return info.dim if info else None

    def _decl(self, sc, name, info, node):
        info.decl = (node.line, node.col)
        if self.pyc:
            # Python: blocks are not scopes — a name belongs to the whole function (or class / module)
            while sc.parent is not None and not (getattr(sc, "fn_root", False) or getattr(sc, "is_class", False)
                                                 or getattr(sc, "is_comp", False)):
                sc = sc.parent
            info.mutable = True
        sc.names[name] = info
        self.decls.append((node.line, node.col, name, info))
        return info

    def i_Unary(self, e, sc):
        d = self.infer(e.operand, sc)
        if e.op == "not":
            return None
        return DIMLESS if d == ZERO else d

    def i_Binary(self, e, sc):
        l, r = self.infer(e.left, sc), self.infer(e.right, sc)
        op = e.op
        if op == "±":
            if isD(l) and isD(r) and l != r:
                self.err(e, f"value and uncertainty must have the same units: {dim_name(l)} ± {dim_name(r)}", UnitError)
            return l if isD(l) else (r if isD(r) else None)
        if op in ("+", "-", "%", "//"):
            if isD(l) and isD(r) and l != r:
                verb = {"+": "add", "-": "subtract", "%": "take modulo of", "//": "floor-divide"}[op]
                hint = ""
                if DIMLESS in (l, r):
                    unit = dim_name(r if l == DIMLESS else l).split("[")[-1].rstrip("]") if "[" in \
                        dim_name(r if l == DIMLESS else l) else ""
                    hint = f" (give the number a unit too, e.g. 1[{unit}])" if unit else ""
                self.err(e, f"cannot {verb} {dim_name(l)} and {dim_name(r)}{hint}", UnitError)
            if op == "//":
                return DIMLESS if isD(l) and isD(r) else None
            if isD(l):
                return l
            if isD(r):
                return r
            return ZERO if l == ZERO and r == ZERO else None
        L = DIMLESS if l == ZERO else l
        R = DIMLESS if r == ZERO else r
        if op in ("*", "@"):
            return L * R if isD(L) and isD(R) else None
        if op == "/":
            return L / R if isD(L) and isD(R) else None
        if op == "**":
            if isD(R) and R != DIMLESS:
                self.err(e, f"exponent must be dimensionless, got {dim_name(R)}", UnitError)
            if L == DIMLESS:
                return DIMLESS
            k = self.const_num(e.right)
            if isD(L) and k is not None:
                return L ** k
            return None
        if op in ("==", "!=", "<", "<=", ">", ">=", "~="):
            if isD(l) and isD(r) and l != r:
                self.err(e, f"cannot compare {dim_name(l)} and {dim_name(r)}", UnitError)
        return None

    def i_Logic(self, e, sc):
        self.infer(e.left, sc)
        self.infer(e.right, sc)
        return None

    def i_IfExpr(self, e, sc):
        self.infer(e.cond, sc)
        a, b = self.infer(e.a, sc), self.infer(e.b, sc)
        if isD(a) and isD(b) and a != b:
            self.err(e, f"both branches must have the same units: {dim_name(a)} vs {dim_name(b)}", UnitError)
        return a if isD(a) else (b if isD(b) else None)

    def i_Spread(self, e, sc):
        self.infer(e.value, sc)
        return None

    def i_Call(self, e, sc):
        args = [self.infer(a, sc) for a in e.args]
        kws = {k: self.infer(v, sc) for k, v in e.kwargs if k is not None}
        for k, v in e.kwargs:
            if k is None:
                self.infer(v, sc)
        f = e.func
        if any(isinstance(a, A.Spread) for a in e.args) or any(k is None for k, _ in e.kwargs):
            if not isinstance(f, A.Name):
                self.infer(f, sc)
            else:
                self.lookup(f.id, sc, f)
            return None
        if not isinstance(f, A.Name):
            self.infer(f, sc)
            return None
        info = self.lookup(f.id, sc, f)
        if info is None:
            return None
        if info.sig is not None:
            return self.check_call(e, info.sig, args, kws)
        if info.kind == "builtin":
            return self.builtin_rule(f.id, e, args)
        return None

    def check_call(self, e, sig, args, kws):
        ps = sig.params
        if len(args) > len(ps) and not getattr(sig, "varargs", False):
            self.err(e, f"{sig.name}() takes {len(ps)} argument(s) but {len(args)} were given")
            return sig.ret
        names = {p[0] for p in ps}
        for k in kws:
            if k not in names and not getattr(sig, "varkw", False):
                self.err(e, f"{sig.name}() has no parameter '{k}'")
        for i, (pname, pd, has_def) in enumerate(ps):
            if i < len(args):
                ad = args[i]
            elif pname in kws:
                ad = kws[pname]
            else:
                if not has_def:
                    self.err(e, f"{sig.name}() is missing argument '{pname}'")
                continue
            if isD(pd) and isD(ad) and ad != pd:
                self.err(e, f"argument '{pname}' of {sig.name}() must be {dim_name(pd)}, got {dim_name(ad)}",
                         UnitError)
        return sig.ret

    def builtin_rule(self, name, e, args):
        a0 = args[0] if args else None
        a0 = DIMLESS if a0 == ZERO else a0
        if name == "sqrt":
            return a0 ** Fraction(1, 2) if isD(a0) else None
        if name == "cbrt":
            return a0 ** Fraction(1, 3) if isD(a0) else None
        if name in DIMLESS_FNS:
            if isD(a0) and a0 != DIMLESS:
                self.err(e, f"{name}() needs a dimensionless argument, got {dim_name(a0)} "
                            f"(divide by a reference value first)", UnitError)
            return DIMLESS
        if name in SAME_FNS:
            return a0 if isD(a0) else None
        if name in ("max", "min") and len(args) > 1:
            ds = [a for a in args if isD(a)]
            if len(set(ds)) > 1:
                self.err(e, f"{name}() arguments have different units")
            return ds[0] if ds else None
        if name in ("dot", "cross", "outer", "matmul") and len(args) >= 2:
            a1 = DIMLESS if args[1] == ZERO else args[1]
            return a0 * a1 if isD(a0) and isD(a1) else None
        if name == "laplacian" and len(args) >= 2:
            a1 = args[1]
            return a0 / (a1 * a1) if isD(a0) and isD(a1) else None
        if name == "gradient" and len(args) >= 2:
            a1 = args[1]
            return a0 / a1 if isD(a0) and isD(a1) else None
        if name == "atan2" and len(args) == 2:
            if isD(args[0]) and isD(args[1]) and args[0] != args[1]:
                self.err(e, "atan2() needs both arguments in the same units", UnitError)
            return DIMLESS
        return None

    def i_Index(self, e, sc):
        o = self.infer(e.obj, sc)
        self.infer_index(e.index, sc)
        return o if isD(o) and o != DIMLESS else None

    def i_Slice(self, e, sc):
        self.infer_index(e, sc)
        return None

    def i_Attr(self, e, sc):
        self.infer(e.obj, sc)
        return None

    def i_ListLit(self, e, sc):
        for x in e.items:
            self.infer(x, sc)
        return None

    i_TupleLit = i_ListLit

    def i_DictLit(self, e, sc):
        for k, v in e.pairs:
            if k is not None:
                self.infer(k, sc)
            self.infer(v, sc)
        return None

    def _comp_scope(self, e, sc):
        cs = sc
        for targets, itx, cond in [(e.targets, e.iter, e.cond)] + list(e.more or []):
            self.infer(itx, cs)
            cs = Scope(cs)
            cs.is_comp = True
            for n in pattern_names(targets):
                cs.names[n] = Info(None, False)
            if cond is not None:
                self.infer(cond, cs)
        return cs

    def i_ListComp(self, e, sc):
        cs = self._comp_scope(e, sc)
        self.infer(e.elem, cs)
        return None

    i_SetComp = i_GenExp = i_ListComp

    def i_SetLit(self, e, sc):
        for x in e.items:
            self.infer(x, sc)
        return None

    def i_Bytes(self, e, sc):
        return None

    def i_Chain(self, e, sc):
        ds = [self.infer(x, sc) for x in e.operands]
        for op, a, b, node in zip(e.ops, ds, ds[1:], e.operands[1:]):
            if op in ("<", "<=", ">", ">=", "==", "!=") and isD(a) and isD(b) and a != b:
                self.err(node, f"cannot compare {dim_name(a)} and {dim_name(b)}", UnitError)
        return None

    def i_NamedExpr(self, e, sc):
        self.infer(e.value, sc)
        if sc.lookup(e.name) is None:
            # Python binds the name in the enclosing function (not in a comprehension's own scope)
            target = sc
            while getattr(target, "is_comp", False) and target.parent is not None:
                target = target.parent
            target.names[e.name] = Info(None, True)
        return None

    def i_Await(self, e, sc):
        self.infer(e.value, sc)
        return None

    def i_YieldExpr(self, e, sc):
        if e.value is not None:
            self.infer(e.value, sc)
        return None

    def i_DictComp(self, e, sc):
        cs = self._comp_scope(e, sc)
        self.infer(e.key, cs)
        self.infer(e.value, cs)
        return None

    def i_Lambda(self, e, sc):
        ls = Scope(sc)
        for p in e.params:
            if p.default is not None:
                self.infer(p.default, sc)
            if p.name is not None:
                ls.names[p.name] = Info(self.type_dim(p.type, sc, p) if p.type else None, False, kind="param")
        self.fn_stack.append(Sig("<lambda>", [], None))
        self.infer(e.body, ls)
        self.fn_stack.pop()
        return None

    def i_Range(self, e, sc):
        a, b = self.infer(e.start, sc), self.infer(e.end, sc)
        if isD(a) and isD(b) and a != b:
            self.err(e, f"range mixes {dim_name(a)} and {dim_name(b)}", UnitError)
        return None

    def i_As(self, e, sc):
        d = self.infer(e.expr, sc)
        text, scale, dim = e.unit
        if isD(d) and d != dim:
            self.err(e, f"cannot express {dim_name(d)} in [{text}] ({dim_name(dim)})", UnitError)
        return DIMLESS if dim.dimensionless else dim

    def i_FStr(self, e, sc):
        for p in e.parts:
            if not isinstance(p, str):
                self.infer(p[0], sc)
                if isinstance(p[1], A.FStr):
                    self.infer(p[1], sc)
        return None


def pattern_names(pats):
    """Every name a target pattern binds (see ast.For)."""
    if isinstance(pats, str):
        return [pats]
    if isinstance(pats, tuple):                    # ("*", name) or ("expr", attribute/index target)
        return [] if pats[0] == "expr" else [pats[1]]
    out = []
    for p in pats:
        out += pattern_names(p)
    return out


def _local_lookup(sc, name):
    """A binding of `name` in the same function (Python scoping: one namespace per function)."""
    s = sc
    while s is not None:
        if name in s.names:
            return s.names[name]
        if getattr(s, "fn_root", False) or getattr(s, "is_class", False):
            return None
        s = s.parent
    return None


def is_python_mode(prog):
    """`use pycompat` at the top of a file: Python's exact semantics (converted programs)."""
    for s in prog.body[:40]:
        if isinstance(s, A.Use) and s.path == "pycompat" and s.alias is None:
            return True
    return False


import builtins as _bi  # noqa: E402
_PY_EXC = {k for k, v in vars(_bi).items() if isinstance(v, type) and issubclass(v, BaseException)}
