"""Static checker for Apsi: names, mutability, arity, physical units, evolve rules.

Runs before execution. It is conservative: it only reports what it can prove
from literals, annotations and declared signatures, so valid programs pass.
"""
import os
from fractions import Fraction

from . import ast as A
from .errors import TypeCheckError, UnitError, NameErr, ParseError, LexError
from .units import Dim, DIMLESS, DIM_NAMES, dim_name
from .parser import parse, EVOLVE_METHODS
from . import ops

ZERO = "ZERO"          # literal 0: compatible with any unit in + and -

TENSOR_TYPES = {"Tensor", "Vec", "Mat", "Matrix", "Vec2", "Vec3", "Vec4"}
DTYPE_NAMES = {"f16", "f32", "f64", "i32", "i64", "bool", "bf16", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2",
               "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"}
PLAIN_TYPES = {"Bool", "Str", "List", "Tuple", "Dict", "Any", "Fn", "Record", "None"}
NUM_TYPES = {"Float", "Number", "Real", "Int"}

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

    # ------------------------------------------------------------- utils
    def err(self, node, msg, cls=TypeCheckError):
        self.errors.append(cls(msg, node.line, node.col))

    def lookup(self, name, sc, node=None):
        info = sc.lookup(name)
        if info is not None:
            return info
        if name in self.builtins:
            v = self.builtins[name]
            return Info(ops.dim_of(v) if not callable(v) else None, False, None, "builtin")
        if node is not None:
            self.err(node, f"'{name}' is not defined", NameErr)
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
        for s in prog.body:
            if isinstance(s, A.StructDef):
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
        d = self.infer(s.value, sc)
        if len(s.targets) == 1:
            declared = self.type_dim(s.type, sc, s) if s.type is not None else None
            if isD(declared) and isD(d) and d != declared:
                self.err(s, f"'{s.targets[0]}' is declared as {dim_name(declared)} but the value is "
                            f"{dim_name(d)}", UnitError)
            dim = declared if declared is not None else (None if d == ZERO else d)
            self._decl(sc, s.targets[0], Info(dim, s.mutable), s)
        else:
            for n in s.targets:
                self._decl(sc, n, Info(None, s.mutable), s)

    def _assign_name(self, node, name, d, sc):
        info = self.lookup(name, sc, None)
        if info is None:
            self.err(node, f"'{name}' is not defined; declare it first with let or var", NameErr)
            return None
        if not info.mutable:
            what = "a parameter" if info.kind == "param" else (
                "a function" if info.kind in ("fn", "builtin") else "declared with let")
            self.err(node, f"cannot assign to '{name}': it is {what} (use var for mutable variables)")
            return None
        return info

    def c_Assign(self, s, sc, top):
        d = self.infer(s.value, sc)
        self._target(s.target, d, sc, s)

    def _target(self, t, d, sc, s):
        if isinstance(t, A.Name):
            info = self._assign_name(t, t.id, d, sc)
            if info and isD(info.dim) and isD(d) and info.dim != d:
                self.err(t, f"'{t.id}' holds {dim_name(info.dim)}; cannot assign {dim_name(d)}", UnitError)
        elif isinstance(t, A.TupleLit):
            for x in t.items:
                self._target(x, None, sc, s)
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
                   p.default is not None) for p in s.params]
        ret = self.type_dim(s.ret, sc, s) if s.ret is not None else None
        return Sig(s.name, params, ret)

    def c_FnDef(self, s, sc, top):
        for d in s.decorators:
            self.infer(d, sc)
        sig = self._sig(s, sc)
        self._decl(sc, s.name, Info(None, False, sig, "fn"), s)
        if top:
            self.deferred.append(lambda: self.fn_body(s, sc, sig))
        else:
            self.fn_body(s, sc, sig)

    def fn_body(self, s, sc, sig, extra=None):
        fsc = Scope(sc)
        for k, v in (extra or {}).items():
            fsc.names[k] = v
        syms = set(s.shape_params)
        for p in s.params:
            self.shape_symbols(p.type, syms)
        if s.ret is not None:
            self.shape_symbols(s.ret, syms)
        for sym in syms:
            fsc.names[sym] = Info(DIMLESS, False, kind="param")
        for p, (_, d, _) in zip(s.params, sig.params):
            if p.default is not None:
                pd = self.infer(p.default, sc)
                if isD(d) and isD(pd) and pd != d:
                    self.err(p, f"default value of '{p.name}' is {dim_name(pd)}, expected {dim_name(d)}", UnitError)
            self._decl(fsc, p.name, Info(d, False, kind="param"), p)
        self.fn_stack.append(sig)
        if s.is_expr:
            self.check_ret(s.body, self.infer(s.body, fsc), sig)
        else:
            self.block(s.body, fsc)
        self.fn_stack.pop()

    def check_ret(self, node, d, sig):
        if isD(sig.ret) and isD(d) and d != sig.ret:
            self.err(node, f"{sig.name}() should return {dim_name(sig.ret)} but this returns "
                           f"{dim_name(d)}", UnitError)

    def c_Return(self, s, sc, top):
        if not self.fn_stack:
            self.err(s, "'return' outside of a function")
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

    def c_If(self, s, sc, top):
        for cond, body in s.branches:
            self.infer(cond, sc)
            self.block(body, Scope(sc))
        if s.else_body:
            self.block(s.else_body, Scope(sc))

    def c_While(self, s, sc, top):
        self.infer(s.cond, sc)
        self.block(s.body, Scope(sc))

    def c_For(self, s, sc, top):
        self.infer(s.iter, sc)
        fs = Scope(sc)
        for n in s.targets:
            self._decl(fs, n, Info(None, False), s)
        self.block(s.body, fs)

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
                sc.names[s.path] = Info(None, False, kind="module")      # `use mylib` also allows mylib.name(...)

    def c_Import(self, s, sc, top):
        path = s.path if os.path.isabs(s.path) else os.path.join(self.base_dir, s.path)
        if not os.path.exists(path):
            self.err(s, f"cannot find file '{s.path}'", NameErr)
            return
        if s.alias:
            sc.names[s.alias] = Info(None, False, kind="module")
            return
        try:
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
                self.err(e, f"cannot {verb} {dim_name(l)} and {dim_name(r)}", UnitError)
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

    def i_Call(self, e, sc):
        args = [self.infer(a, sc) for a in e.args]
        kws = {k: self.infer(v, sc) for k, v in e.kwargs}
        f = e.func
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
        if len(args) > len(ps):
            self.err(e, f"{sig.name}() takes {len(ps)} argument(s) but {len(args)} were given")
            return sig.ret
        names = {p[0] for p in ps}
        for k in kws:
            if k not in names:
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
            self.infer(k, sc)
            self.infer(v, sc)
        return None

    def i_ListComp(self, e, sc):
        self.infer(e.iter, sc)
        cs = Scope(sc)
        for n in e.targets:
            cs.names[n] = Info(None, False)
        if e.cond is not None:
            self.infer(e.cond, cs)
        self.infer(e.elem, cs)
        return None

    def i_DictComp(self, e, sc):
        self.infer(e.iter, sc)
        cs = Scope(sc)
        for n in e.targets:
            cs.names[n] = Info(None, False)
        if e.cond is not None:
            self.infer(e.cond, cs)
        self.infer(e.key, cs)
        self.infer(e.value, cs)
        return None

    def i_Lambda(self, e, sc):
        ls = Scope(sc)
        for p in e.params:
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
        return None
