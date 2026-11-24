"""Apsi native backend: the syntax tree of an Apsi program to C, linked with the C runtime (rt/apsi_rt.c) into a
machine-code executable. Nothing of Python is left in the result: the values, memory, exceptions and built-ins are
the runtime's.

Every value the program holds lives in a slot of its function's frame (locals first, then temporaries), and the
frames are chained for the collector, so collection can run at any safe point (function entry, loop back edges).
Variables captured by inner functions live in cells. Exceptions are setjmp / longjmp handlers that also undo
frames and the runtime's temporary roots."""
import os
import re

from .. import ast as A


class NativeError(Exception):
    def __init__(self, msg, node=None):
        line = getattr(node, "line", 0) or 0
        super().__init__(f"line {line}: {msg}" if line else msg)
        self.line = line


_RT = os.path.join(os.path.dirname(__file__), "rt")


def _builtin_names():
    src = open(os.path.join(_RT, "apsi_rt.c"), encoding="utf-8").read()
    names = set(re.findall(r'\{"([A-Za-z_0-9]+)", b_[A-Za-z_0-9]+\}', src))
    exc = re.search(r"a_exc_names\[A_NEXC\] = \{(.*?)\};", src, re.S).group(1)
    names |= set(re.findall(r'"([A-Za-z]+)"', exc))
    return names | {"pi", "e", "inf", "nan", "Error", "IOError", "EnvironmentError"}


BUILTINS = _builtin_names()
PRELUDE = os.path.join(os.path.dirname(__file__), "prelude.apsi")


def _prelude_names():
    from ..parser import parse, static_public
    with open(PRELUDE, encoding="utf-8") as fh:
        return set(static_public(parse(fh.read()))) - BUILTINS


_PRELUDE_NAMES = None


def prelude_names():
    global _PRELUDE_NAMES
    if _PRELUDE_NAMES is None:
        _PRELUDE_NAMES = _prelude_names()
    return _PRELUDE_NAMES

def _same(*names):
    return {n: n for n in names}


# the standard modules the native runtime has: module member -> built-in function
NATIVE_MODULES = {
    "math": _same("abs", "acos", "asin", "atan", "atan2", "cbrt", "ceil", "comb", "cos", "cosh", "e", "erf", "erfc",
                  "exp", "factorial", "floor", "gamma", "inf", "lgamma", "log", "log10", "log2", "nan", "pi",
                  "round", "sin", "sinh", "sqrt", "tan", "tanh", "tau", "beta", "hypot", "expm1", "log1p"),
    "random": _same("seed", "rand", "randint", "randn", "permutation"),
    "io": _same("exists", "read_text", "write_text", "json_parse", "json_str", "load_json", "save_json"),
    "sys": _same("cpu_threads"),
    "collections": _same("first", "last", "fold", "scan", "partition", "flat_map", "take_while", "drop_while",
                         "max_by", "min_by", "sum_by", "frequencies", "distinct_by", "interleave", "zip_longest",
                         "compose", "partial"),
}

BINOPS = {"+": "OP_ADD", "-": "OP_SUB", "*": "OP_MUL", "/": "OP_DIV", "//": "OP_FLOORDIV", "%": "OP_MOD",
          "**": "OP_POW", "@": "OP_MATMUL", "&": "OP_AND", "|": "OP_OR", "^": "OP_XOR", "<<": "OP_SHL",
          ">>": "OP_SHR"}
CMPOPS = {"==": "OP_EQ", "!=": "OP_NE", "<": "OP_LT", "<=": "OP_LE", ">": "OP_GT", ">=": "OP_GE", "in": "OP_IN",
          "not in": "OP_NOTIN", "is": "OP_IS", "is not": "OP_ISNOT"}
FAST_CMP = {"<": "a_lt", "<=": "a_le", ">": "a_gt", ">=": "a_ge", "==": "a_eqf"}
FAST_BIN = {"+": "a_add", "-": "a_sub", "*": "a_mul"}


def cstr(s):
    b = s.encode("utf-8")
    out = []
    for ch in b:
        c = chr(ch)
        if c in '"\\':
            out.append("\\" + c)
        elif 32 <= ch < 127 and c != "?":
            out.append(c)
        else:
            out.append(f"\\{ch:03o}")
    return '"' + "".join(out) + '"'


# ======================================================================= scopes
class Scope:
    """One function (or the module): which names it binds, which it reads, and how each name is reached."""

    def __init__(self, node, parent, kind):
        self.node, self.parent, self.kind = node, parent, kind      # kind: module | fn | class
        self.bound = []          # names bound here, in first-binding order
        self.used = set()
        self.globals_ = set()    # `global x`
        self.nonlocals = set()
        self.children = []
        self.cells = set()       # bound names an inner function captures
        self.free = []           # names this function reads from an enclosing function (its cells, in order)
        self.loop_names = []     # while walking: the immutable names declared by the loops around (a stack of sets)
        self.snap = set()        # a lambda's names captured by value (per-iteration closures)

    def bind(self, name):
        if name not in self.bound:
            self.bound.append(name)

    def fn_scope(self):
        s = self
        while s.kind == "class":
            s = s.parent
        return s


class Analyzer:
    """Walks the tree once: finds each function's bound and used names, renames comprehension variables apart,
    then works out cells and free variables."""

    def __init__(self, use_names=None):
        self.scopes = {}
        self.comp_n = 0
        self.use_names = use_names or (lambda s: [])

    def run(self, prog):
        mod = Scope(prog, None, "module")
        self.scopes[id(prog)] = mod
        self.block(prog.body, mod)
        self.resolve(mod)
        return mod

    # ---- binding
    def bind_pattern(self, p, sc):
        if isinstance(p, str):
            sc.bind(p)
        elif isinstance(p, tuple) and len(p) == 2 and p[0] == "*":
            sc.bind(p[1])
        elif isinstance(p, list):
            for x in p:
                self.bind_pattern(x, sc)

    def bind_target(self, t, sc):
        if isinstance(t, A.Name):
            sc.bind(t.id)
        elif isinstance(t, (A.TupleLit, A.ListLit)):
            for x in t.items:
                self.bind_target(x, sc)
        elif isinstance(t, A.Spread):
            self.bind_target(t.value, sc)
        else:
            self.expr(t, sc)

    def block(self, body, sc):
        for s in body:
            self.stmt(s, sc)

    def stmt(self, s, sc):
        k = type(s).__name__
        if k == "Let":
            if s.value is not None:
                self.expr(s.value, sc)
            for t in s.targets:
                self.bind_pattern(t, sc)
                if not s.mutable and sc.loop_names:
                    self.loop_declare(t, sc)
        elif k == "Declare":
            for n in s.names:
                sc.bind(n)
        elif k == "Assign":
            self.expr(s.value, sc)
            for t in [s.target] + list(s.more or []):
                self.bind_target(t, sc)
        elif k == "AugAssign":
            self.expr(s.value, sc)
            if isinstance(s.target, A.Name):
                sc.used.add(s.target.id)
                sc.bind(s.target.id)
            else:
                self.expr(s.target, sc)
        elif k == "ExprStmt":
            self.expr(s.expr, sc)
        elif k in ("FnDef",):
            for d in s.decorators or []:
                self.expr(d, sc)
            sc.bind(s.name)
            self.function(s, s.params, s.body if not s.is_expr else None, s.body if s.is_expr else None, sc)
        elif k in ("StructDef", "ClassDef"):
            sc.bind(s.name)
            for b in getattr(s, "bases", None) or []:
                self.expr(b, sc)
            for d in getattr(s, "decorators", None) or []:
                self.expr(d, sc)
            cs = Scope(s, sc, "class")
            self.scopes[id(s)] = cs
            if k == "StructDef":
                for f in s.fields:
                    if f.default is not None:
                        self.expr(f.default, sc)
                for m in s.methods:
                    self.function(m, m.params, m.body if not m.is_expr else None, m.body if m.is_expr else None, sc)
            else:
                for item in s.body:
                    if isinstance(item, A.FnDef):
                        for d in item.decorators or []:
                            self.expr(d, sc)
                        self.function(item, item.params, item.body if not item.is_expr else None,
                                      item.body if item.is_expr else None, sc)
                    elif isinstance(item, A.FieldDef):
                        if item.value is not None:
                            self.expr(item.value, sc)
                    elif isinstance(item, A.Let):
                        if item.value is not None:
                            self.expr(item.value, sc)
                    elif isinstance(item, A.Assign):
                        self.expr(item.value, sc)
                    elif isinstance(item, (A.Pass, A.ExprStmt)):
                        pass
                    else:
                        raise NativeError(f"a {type(item).__name__} in a class body is not supported natively", item)
        elif k == "If":
            for c, b in s.branches:
                self.expr(c, sc)
                self.block(b, sc)
            self.block(s.else_body or [], sc)
        elif k == "While":
            self.expr(s.cond, sc)
            sc.loop_names.append(set())
            self.block(s.body, sc)
            sc.loop_names.pop()
            self.block(s.else_body or [], sc)
        elif k == "For":
            self.expr(s.iter, sc)
            sc.loop_names.append(set())
            for t in s.targets:
                self.bind_pattern(t, sc)
                self.loop_declare(t, sc)
            self.block(s.body, sc)
            sc.loop_names.pop()
            self.block(s.else_body or [], sc)
        elif k == "Return":
            if s.value is not None:
                self.expr(s.value, sc)
        elif k in ("Break", "Continue", "Pass", "Test"):
            pass
        elif k == "Try":
            self.block(s.body, sc)
            for kinds, name, body in s.handlers:
                if kinds is not None:
                    for x in (kinds if isinstance(kinds, list) else [kinds]):
                        if isinstance(x, str):
                            sc.used.add(x)
                        else:
                            self.expr(x, sc)
                if name:
                    sc.bind(name)
                self.block(body, sc)
            self.block(s.else_body or [], sc)
            self.block(s.final_body or [], sc)
        elif k == "Raise":
            if s.value is not None:
                self.expr(s.value, sc)
        elif k == "Assert":
            self.expr(s.cond, sc)
            if s.msg is not None:
                self.expr(s.msg, sc)
        elif k == "Global":
            (sc.nonlocals if s.nonlocal_ else sc.globals_).update(s.names)
        elif k == "Del":
            for t in s.targets:
                self.expr(t, sc)
        elif k == "With":
            for e, t in s.items:
                self.expr(e, sc)
                if t is not None:
                    self.bind_pattern(t, sc) if not isinstance(t, A.AST) else self.bind_target(t, sc)
            self.block(s.body, sc)
        elif k == "Use":
            for n in self.use_names(s):
                sc.bind(n)
        else:
            raise NativeError(f"'{k}' is not supported by the native backend yet", s)

    def loop_declare(self, p, sc):
        if isinstance(p, str):
            sc.loop_names[-1].add(p)
        elif isinstance(p, tuple) and len(p) == 2 and p[0] == "*":
            sc.loop_names[-1].add(p[1])
        elif isinstance(p, list):
            for x in p:
                self.loop_declare(x, sc)

    def function(self, node, params, body, expr_body, sc):
        fs = Scope(node, sc, "fn")
        self.scopes[id(node)] = fs
        sc.fn_scope().children.append(fs) if sc.kind != "class" else sc.parent.fn_scope().children.append(fs)
        for p in params:
            if p.default is not None:
                self.expr(p.default, sc)
            fs.bind(p.name)
        if body is not None:
            self.block(body, fs)
        else:
            self.expr(expr_body, fs)
        return fs

    # ---- expressions
    def comp(self, e, sc):
        """Rename the comprehension's variables apart: they are the enclosing function's slots afterwards."""
        clauses = [(e.targets, e.iter, e.cond)] + list(e.more or [])
        ren = {}
        sc.fn_scope().loop_names.append(set())
        try:
            self._comp(e, sc, clauses, ren)
        finally:
            sc.fn_scope().loop_names.pop()

    def _comp(self, e, sc, clauses, ren):
        for i, (targets, it, cond) in enumerate(clauses):
            self.expr(it, sc) if not ren else self.expr(self.rename(it, ren), sc)
            new_t = [self.rename_pat(t, ren) for t in targets]
            for t in new_t:
                self.bind_pattern(t, sc)
                self.loop_declare(t, sc.fn_scope())
            clauses[i] = (new_t, self.rename(it, ren) if i else it, cond)
            clauses[i] = (new_t, clauses[i][1], self.rename(cond, ren) if cond is not None else None)
        e.targets, e.iter, e.cond = clauses[0]
        e.more = clauses[1:]
        for cl in e.more:
            if cl[2] is not None:
                self.expr(cl[2], sc)
        if e.cond is not None:
            self.expr(e.cond, sc)
        for f in ("elem", "key", "value"):
            if hasattr(e, f) and getattr(e, f) is not None:
                setattr(e, f, self.rename(getattr(e, f), ren))
                self.expr(getattr(e, f), sc)

    def rename_pat(self, p, ren):
        if isinstance(p, str):
            self.comp_n += 1
            ren[p] = f"{p}${self.comp_n}"
            return ren[p]
        if isinstance(p, tuple) and len(p) == 2 and p[0] == "*":
            return ("*", self.rename_pat(p[1], ren))
        return [self.rename_pat(x, ren) for x in p]

    def rename(self, e, ren):
        """A copy-free rename: Name nodes inside are rewritten in place (lambdas keep their own parameters)."""
        if not ren or e is None:
            return e
        if isinstance(e, A.Name):
            if e.id in ren:
                e.id = ren[e.id]
            return e
        if isinstance(e, A.Lambda):
            inner = {k: v for k, v in ren.items() if k not in {p.name for p in e.params}}
            self.rename(e.body, inner)
            return e
        if isinstance(e, A.AST):
            for f in e._fields:
                v = getattr(e, f)
                if isinstance(v, A.AST):
                    self.rename(v, ren)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, A.AST):
                            self.rename(x, ren)
                        elif isinstance(x, tuple):
                            for y in x:
                                if isinstance(y, A.AST):
                                    self.rename(y, ren)
        return e

    def expr(self, e, sc):
        if e is None:
            return
        k = type(e).__name__
        if k == "Name":
            sc.used.add(e.id)
        elif k in ("ListComp", "DictComp", "SetComp", "GenExp"):
            self.comp(e, sc)
        elif k == "Lambda":
            fs = self.function(e, e.params, None, e.body, sc)
            for names in sc.fn_scope().loop_names:
                fs.snap |= names
        elif k == "NamedExpr":
            sc.bind(e.name)
            self.expr(e.value, sc)
        elif k in ("YieldExpr", "Await"):
            raise NativeError("generators and async are not supported by the native backend yet", e)
        elif k == "FStr":
            for p in e.parts:
                if isinstance(p, tuple):
                    self.expr(p[0], sc)
                    if isinstance(p[1], A.AST):
                        self.expr(p[1], sc)
        elif k == "Call":
            self.expr(e.func, sc)
            for a in e.args:
                self.expr(a, sc)
            for _, v in e.kwargs:
                self.expr(v, sc)
        elif k == "DictLit":
            for kk, v in e.pairs:
                self.expr(kk, sc)
                self.expr(v, sc)
        elif isinstance(e, A.AST):
            for f in e._fields:
                v = getattr(e, f)
                if isinstance(v, A.AST):
                    self.expr(v, sc)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, A.AST):
                            self.expr(x, sc)

    # ---- cells and free variables
    def resolve(self, mod):
        def visit(fs):
            for c in fs.children:
                visit(c)
            # names read by inner functions and not bound by them are free there; walk up to the binder
            if fs.kind == "fn":
                for name in sorted(self.reads(fs)):
                    if name in fs.snap and name not in fs.bound:
                        if name not in fs.free:
                            fs.free.append(name)          # its value now, in a cell of its own
                        continue
                    self.capture(fs, name)

        visit(mod)

    def reads(self, fs):
        own = set(fs.bound) - fs.globals_ - fs.nonlocals
        r = (fs.used - own) | fs.nonlocals
        for c in fs.children:
            r |= set(c.free) - own
        return r - fs.globals_

    def capture(self, fs, name):
        p = fs.parent.fn_scope() if fs.parent else None
        while p is not None and p.kind != "module":
            if name in p.bound and name not in p.globals_ and name not in p.nonlocals:
                p.cells.add(name)
                if name not in fs.free:
                    fs.free.append(name)
                return
            p = p.parent.fn_scope() if p.parent else None
        # a global or a built-in: nothing to capture


# ======================================================================= code generation
class Fn:
    def __init__(self, cname, scope, name):
        self.cname, self.scope, self.name = cname, scope, name
        self.lines = []
        self.ind = 1
        self.slots = {}          # local name -> slot
        self.nlocal = 1          # slot 0: the function object itself
        self.t = 0               # temporaries in use (above the locals)
        self.tmax = 0
        self.loops = []          # [(break_label_flag, depth_of_try_stack)]
        self.trys = []           # active try contexts: (handler var, finally body or None, kind)
        self.uid = 0

    def local(self, name):
        if name not in self.slots:
            self.slots[name] = self.nlocal
            self.nlocal += 1
        return self.slots[name]


class CGen:
    def __init__(self, filename="<program>", source=""):
        self.filename = filename
        self.source = source
        self.fns = []
        self.consts = {}         # (kind, value) -> index in K
        self.const_init = []
        self.globals = {}        # module name -> index in G
        self.fn = None
        self.fn_n = 0
        self.mod_scope = None
        self.analyzer = None
        self.mod_id = 0              # which file's globals: 0 the program, 1.. its modules
        self.base_dir = os.path.dirname(os.path.abspath(filename)) if filename else "."
        self.local_mods = {}         # path -> (id, init function, public names)
        self.compiling = set()       # module files being compiled now
        self.n_mods = 0

    # ---- output helpers
    def emit(self, s):
        self.fn.lines.append("    " * self.fn.ind + s)

    def tmp(self):
        f = self.fn
        k = f.t
        f.t += 1
        f.tmax = max(f.tmax, f.t)
        return f"S[T+{k}]"

    def label(self, p="L"):
        self.fn.uid += 1
        return f"{p}{self.fn.uid}"

    def k_str(self, s):
        key = ("s", s)
        if key not in self.consts:
            self.consts[key] = len(self.consts)
            self.const_init.append(f"K[{self.consts[key]}] = a_strn({cstr(s)}, {len(s.encode('utf-8'))});")
        return f"K[{self.consts[key]}]"

    def k_builtin(self, name):
        key = ("b", name)
        if key not in self.consts:
            self.consts[key] = len(self.consts)
            self.const_init.append(f"K[{self.consts[key]}] = a_global_builtin({cstr(name)});")
        return f"K[{self.consts[key]}]"

    def gslot(self, name, mod=None):
        key = (self.mod_id if mod is None else mod, name)
        if key not in self.globals:
            self.globals[key] = len(self.globals)
        return self.globals[key]

    # ---- modules
    def resolve_use(self, s):
        """('native', module, {name: builtin}) | ('local', path)"""
        from ..apsil import local_module
        if s.path in NATIVE_MODULES and not s.py:
            return ("native", s.path, NATIVE_MODULES[s.path])
        local = local_module(s.path, self.base_dir)
        if local is not None:
            return ("local", local)
        if s.path in NATIVE_MODULES:
            return ("native", s.path, NATIVE_MODULES[s.path])
        raise NativeError(f"module '{s.path}' is not available to the native backend yet "
                          f"(native modules: {', '.join(sorted(NATIVE_MODULES))}, and the program's own files)", s)

    def use_names(self, s):
        r = self.resolve_use(s)
        if s.names is not None:
            return [a or n for n, a in s.names]
        if s.alias:
            return [s.alias]
        if r[0] == "native":
            return list(r[2]) + ([s.path] if "." not in s.path else [])
        from ..parser import parse, static_public
        with open(r[1], encoding="utf-8-sig") as fh:
            names = list(static_public(parse(fh.read())))
        return names + ([s.path] if "." not in s.path and s.path not in names else [])

    def local_module(self, path):
        if path in self.local_mods:
            return self.local_mods[path]
        from ..parser import parse, static_public
        with open(path, encoding="utf-8-sig") as fh:
            prog = parse(fh.read())
        self.n_mods += 1
        mid = self.n_mods
        cname = f"mod{mid}_body"
        pub = list(static_public(prog))
        self.local_mods[path] = (mid, cname, pub)
        self.compiling.add(path)
        saved = (self.fn, self.mod_scope, self.analyzer, self.mod_id, self.base_dir)
        self.fn = None
        self.mod_id = mid
        self.base_dir = os.path.dirname(path)
        self.analyzer = Analyzer(self.use_names)
        self.mod_scope = self.analyzer.run(prog)
        self.function_body(cname, self.mod_scope, f"<module {os.path.basename(path)}>", None, prog.body, None)
        self.fn, self.mod_scope, self.analyzer, self.mod_id, self.base_dir = saved
        self.compiling.discard(path)
        return self.local_mods[path]

    def s_Use(self, s):
        r = self.resolve_use(s)
        if r[0] == "native":
            members = r[2]
            if s.names is not None:
                for n, a in s.names:
                    if n not in members:
                        raise NativeError(f"module '{s.path}' has no '{n}' natively", s)
                    self.store(a or n, self.k_builtin(members[n]))
                return
            if s.alias or "." not in s.path:
                mod = self.module_obj(s.path, [(n, self.k_builtin(b)) for n, b in members.items()])
                self.store(s.alias or s.path, mod)
                if s.alias:
                    return
            for n, b in members.items():
                self.store(n, self.k_builtin(b))
            return
        mid, cname, pub = self.local_module(r[1])
        self.emit(f"if (!M[{mid}]) {{ M[{mid}] = 1; {cname}(NULL, 0, NULL, NULL); }}")
        vals = [(n, f"a_gload(G[{self.gslot(n, mid)}], {cstr(n)})") for n in pub]
        if s.names is not None:
            for n, a in s.names:
                self.store(a or n, f"a_gload(G[{self.gslot(n, mid)}], {cstr(n)})")
            return
        if s.alias:
            self.store(s.alias, self.module_obj(s.path, vals))
            return
        for n, v in vals:
            self.store(n, v)
        if "." not in s.path and s.path not in pub:
            self.store(s.path, self.module_obj(s.path, vals))

    def module_obj(self, name, items):
        base = self.block_of([v for _, v in items]) if items else "S[0]"
        arr = ", ".join(cstr(n) for n, _ in items) or "NULL"
        t = self.tmp()
        self.emit(f"{{ static const char *nm[] = {{{arr}}}; {t} = a_module({cstr(name)}, {len(items)}, nm, &{base}); }}")
        return t

    # ---- name access
    def where(self, name):
        """('local', slot) | ('cell', slot) | ('free', index) | ('global', index) | ('builtin', K)"""
        f = self.fn
        sc = f.scope
        if sc.kind == "fn" and name not in sc.globals_:
            if name in sc.nonlocals:
                return ("free", sc.free.index(name))
            if name in sc.bound:
                return ("cell" if name in sc.cells else "local", f.local(name))
            if name in sc.free:
                return ("free", sc.free.index(name))
        if name in self.mod_scope.bound or name in sc.globals_:
            return ("global", self.gslot(name))
        if name in BUILTINS:
            return ("builtin", self.k_builtin(name))
        if name in prelude_names() and os.path.abspath(PRELUDE) not in self.compiling:
            mid = self.local_module(os.path.abspath(PRELUDE))[0]
            return ("prelude", mid, self.gslot(name, mid))
        return ("global", self.gslot(name))      # bound at run time later (or a NameError then)

    def load(self, name, node=None):
        w = self.where(name)
        if w[0] == "local":
            return f"S[{w[1]}]"
        if w[0] == "cell":
            return f"((ACell *)S[{w[1]}].o)->v"
        if w[0] == "free":
            return f"self->cells[{w[1]}]->v"
        if w[0] == "global":
            return f"a_gload(G[{w[1]}], {cstr(name)})"
        if w[0] == "prelude":
            self.emit(f"if (!M[{w[1]}]) {{ M[{w[1]}] = 1; mod{w[1]}_body(NULL, 0, NULL, NULL); }}")
            return f"G[{w[2]}]"
        return w[1]

    def store(self, name, val):
        w = self.where(name)
        if w[0] == "local":
            self.emit(f"S[{w[1]}] = {val};")
        elif w[0] == "cell":
            self.emit(f"((ACell *)S[{w[1]}].o)->v = {val};")
        elif w[0] == "free":
            self.emit(f"self->cells[{w[1]}]->v = {val};")
        elif w[0] == "global":
            self.emit(f"G[{w[1]}] = {val};")
        else:
            self.emit(f"G[{self.gslot(name)}] = {val};")

    # ---- program
    def program(self, prog):
        self.analyzer = Analyzer(self.use_names)
        self.mod_scope = self.analyzer.run(prog)
        main = self.function_body("apsi_main", self.mod_scope, "<module>", None, prog.body, None)
        out = ['#include "apsi_rt.h"', "#include <stdio.h>", ""]
        out.append(f"static AV G[{max(1, len(self.globals))}];")
        out.append(f"static AV K[{max(1, len(self.consts))}];")
        out.append(f"static char M[{self.n_mods + 1}];")
        for fn in self.fns:
            out.append(f"static AV {fn.cname}(AFunc *self, int n, AV *args, ADict *kw);")
        out.append("")
        for fn in self.fns:
            out.extend(fn.text)
            out.append("")
        out.append("int main(int argc, char **argv) {")
        out.append("    a_init(argc, argv);")
        out.append(f"    a_file = {cstr(self.filename)};")
        out.append(f"    a_source = {cstr(self.source)};")
        from ..units import _DISPLAY_MAP
        names = [f"{{{self.dim_c(d)}, {cstr(n)}}}" for d, n in _DISPLAY_MAP.items()]
        out.append(f"    static const AUnitName un[] = {{{', '.join(names)}}};")
        out.append(f"    a_unit_names(un, {len(names)});")
        out.append(f"    for (int i = 0; i < {max(1, len(self.globals))}; i++) {{ G[i].t = 255; G[i].i = 0; }}")
        out.append(f"    a_root(G, {max(1, len(self.globals))});")
        out.append(f"    a_root(K, {max(1, len(self.consts))});")
        for line in self.const_init:
            out.append("    " + line)
        out.append(f"    {main.cname}(NULL, 0, NULL, NULL);")
        out.append("    fflush(stdout);")
        out.append("    return 0;")
        out.append("}")
        return "\n".join(out) + "\n"

    def function_body(self, cname, scope, name, params, body, expr_body):
        """Compile one function into self.fns; returns its Fn."""
        outer = self.fn
        fn = Fn(cname, scope, name)
        self.fn = fn
        self.fns.append(fn)
        self.params(params or [])
        if scope.kind == "fn":
            for nm in scope.bound:          # cells exist from the start, so closures made early see later values
                if nm in scope.cells and nm not in {p.name for p in (params or [])}:
                    self.emit(f"S[{fn.local(nm)}] = a_obj(a_cell(A_NONE));")
        self.emit("a_gc_safe();")
        if expr_body is not None:
            v = self.expr(expr_body)
            self.ret(v)
        else:
            self.block(body)
        self.emit("a_frame = fr.prev;")
        self.emit("return A_NONE;")
        nslots = fn.nlocal + fn.tmax
        head = [f"static AV {cname}(AFunc *self, int n, AV *args, ADict *kw) {{",
                f"    AV S[{nslots}]; enum {{ T = {fn.nlocal} }};",
                f"    for (int i = 0; i < {nslots}; i++) S[i] = A_NONE;",
                "    if (self) S[0] = a_obj(self);",
                f"    AFrame fr = {{a_frame, {nslots}, S, {cstr(name)}, 0}}; a_frame = &fr;",
                "    (void)n; (void)args; (void)kw;"]
        fn.text = head + fn.lines + ["}"]
        self.fn = outer
        return fn

    def params(self, params):
        if not params:
            return
        fn = self.fn
        sc = fn.scope
        npos = 0
        for p in params:
            if p.star:
                break
            npos += 1
        has_star = any(p.star == "*" for p in params)
        if not has_star:
            self.emit(f"if (n > {npos}) a_error(\"TypeError\", \"{fn.name}() takes {npos} positional "
                      f"argument{'s' if npos != 1 else ''} but %d were given\", n);")
        di = 0
        pos = 0
        kwonly = False
        names = [p.name for p in params if not p.star]
        for p in params:
            slot = fn.local(p.name)
            if p.star == "*":
                self.emit(f"S[{slot}] = a_rest_args(n, args, {npos});")
                kwonly = True
            elif p.star == "**":
                arr = ", ".join(cstr(x) for x in names) or "NULL"
                self.emit(f"{{ static const char *nm[] = {{{arr}}}; S[{slot}] = a_obj(a_rest_kw(kw, {len(names)}, nm)); }}")
            else:
                key = self.k_str(p.name)
                pos_part = f"if (n > {pos}) S[{slot}] = args[{pos}]; else " if not kwonly else ""
                if p.default is not None:
                    tail = f"S[{slot}] = self->defaults[{di}];"
                    di += 1
                else:
                    tail = f"a_error(\"TypeError\", \"{fn.name}() missing argument '{p.name}'\");"
                self.emit(f"{pos_part}if (!(kw && a_dict_get(kw, {key}, &S[{slot}]))) {tail}")
                if not kwonly:
                    pos += 1
            if p.name in sc.cells:
                self.emit(f"S[{slot}] = a_obj(a_cell(S[{slot}]));")

    # ---- making function objects
    def make_function(self, node, params, body, expr_body, name):
        scope = self.analyzer.scopes[id(node)]
        self.fn_n += 1
        cname = f"f{self.fn_n}_{re.sub(r'[^A-Za-z0-9_]', '_', name)[:40]}"
        defaults = [p.default for p in params if p.default is not None]
        dvals = []
        for d in defaults:                      # defaults are evaluated where the function is defined
            v = self.expr(d)
            t = self.tmp()
            self.emit(f"{t} = {v};")
            dvals.append(t)
        self.function_body(cname, scope, name, params, body, expr_body)
        t = self.tmp()
        self.emit(f"{t} = a_func({cstr(name)}, {cname}, {len(scope.free)}, {len(defaults)});")
        for i, nm in enumerate(scope.free):
            if nm in scope.snap:
                self.emit(f"((AFunc *){t}.o)->cells[{i}] = a_cell({self.load(nm)});")
            else:
                self.emit(f"((AFunc *){t}.o)->cells[{i}] = {self.cell_ref(nm)};")
        for i, d in enumerate(dvals):
            self.emit(f"((AFunc *){t}.o)->defaults[{i}] = {d};")
        return t

    def cell_ref(self, name):
        w = self.where(name)
        if w[0] == "cell":
            return f"(ACell *)S[{w[1]}].o"
        if w[0] == "free":
            return f"self->cells[{w[1]}]"
        raise NativeError(f"internal: '{name}' is not a cell here")

    # ---- statements
    def block(self, body):
        for s in body:
            self.stmt(s)

    def stmt(self, s):
        t0 = self.fn.t
        if getattr(s, "line", 0):
            self.emit(f"fr.pos = {(s.line << 12) | min(getattr(s, 'col', 0) or 0, 4095)};")
        getattr(self, "s_" + type(s).__name__, self.s_unknown)(s)
        self.fn.t = t0

    def s_unknown(self, s):
        raise NativeError(f"'{type(s).__name__}' is not supported by the native backend yet", s)

    def s_Pass(self, s):
        pass

    def s_Test(self, s):
        pass                                 # test blocks run under `apsi test`, not in a built program

    def s_ExprStmt(self, s):
        v = self.expr(s.expr)
        self.emit(f"(void)({v});")

    def s_Declare(self, s):
        pass

    def s_Global(self, s):
        pass

    def s_Let(self, s):
        if s.value is None:
            return
        if len(s.targets) == 1:
            self.assign_pattern(s.targets[0], self.expr(s.value))
            return
        self.assign_many(s.targets, s.value)

    def assign_many(self, targets, value):
        if isinstance(value, A.TupleLit) and len(value.items) == len(targets) and \
                not any(isinstance(x, A.Spread) for x in value.items):
            temps = []
            for x in value.items:
                v = self.expr(x)
                t = self.tmp()
                self.emit(f"{t} = {v};")
                temps.append(t)
            for tg, t in zip(targets, temps):
                self.assign_pattern(tg, t)
            return
        self.assign_pattern(list(targets), self.expr(value))

    def assign_pattern(self, p, val):
        """p: a name, a list of patterns, or ('*', name)."""
        if isinstance(p, str):
            self.store(p, val)
            return
        if isinstance(p, A.AST):
            self.assign_target(p, val)
            return
        items = list(p)
        stars = [i for i, x in enumerate(items) if (isinstance(x, tuple) and len(x) == 2 and x[0] == "*") or
                 isinstance(x, A.Spread)]
        f = self.fn
        base = f.t
        f.t += len(items)
        f.tmax = max(f.tmax, f.t)
        src = self.tmp()
        self.emit(f"{src} = {val};")
        if stars:
            i = stars[0]
            self.emit(f"a_unpack_star({src}, {i}, {len(items) - i - 1}, &S[T+{base}]);")
        else:
            self.emit(f"a_unpack({src}, {len(items)}, &S[T+{base}]);")
        for i, x in enumerate(items):
            if isinstance(x, tuple) and len(x) == 2 and x[0] == "*":
                x = x[1]
            elif isinstance(x, A.Spread):
                x = x.value
            self.assign_pattern(x, f"S[T+{base + i}]")

    def assign_target(self, t, val):
        if isinstance(t, A.Name):
            self.store(t.id, val)
        elif isinstance(t, (A.TupleLit, A.ListLit)):
            self.assign_pattern(list(t.items), val)
        elif isinstance(t, A.Attr):
            v = self.hold(val)
            o = self.expr(t.obj)
            self.emit(f"a_setattrk({o}, {self.k_str(t.name)}, {v});")
        elif isinstance(t, A.Index):
            v = self.hold(val)
            o = self.hold(self.expr(t.obj))
            i = self.index(t.index)
            self.emit(f"a_setitem({o}, {i}, {v});")
        else:
            raise NativeError("cannot assign to this expression", t)

    def hold(self, v):
        """Keep a value in a temporary slot (values are only safe in slots across calls)."""
        if re.fullmatch(r"S\[(T\+)?\d+\]|K\[\d+\]|a_int\(-?\d+\)|A_NONE|a_bool\([01]\)", v):
            return v
        t = self.tmp()
        self.emit(f"{t} = {v};")
        return t

    def s_Assign(self, s):
        targets = [s.target] + list(s.more or [])
        if len(targets) == 1 and isinstance(s.target, A.TupleLit) and isinstance(s.value, A.TupleLit) and \
                len(s.target.items) == len(s.value.items) and \
                not any(isinstance(x, A.Spread) for x in s.target.items + s.value.items):
            self.assign_many(s.target.items, s.value)
            return
        v = self.expr(s.value)
        if len(targets) > 1:
            v = self.hold(v)
        for t in targets:
            self.assign_target(t, v)

    def s_AugAssign(self, s):
        op = BINOPS[s.op.rstrip("=")] if s.op.endswith("=") and s.op != "==" else BINOPS[s.op]
        t = s.target
        if isinstance(t, A.Name):
            cur = self.load(t.id)
            v = self.expr(s.value)
            self.store(t.id, f"a_inplace({op}, {cur}, {v})")
        elif isinstance(t, A.Attr):
            o = self.hold(self.expr(t.obj))
            k = self.k_str(t.name)
            cur = self.hold(f"a_attrk({o}, {k})")
            v = self.expr(s.value)
            self.emit(f"a_setattrk({o}, {k}, a_inplace({op}, {cur}, {v}));")
        elif isinstance(t, A.Index):
            o = self.hold(self.expr(t.obj))
            i = self.hold(self.index(t.index))
            cur = self.hold(f"a_getitem({o}, {i})")
            v = self.expr(s.value)
            r = self.hold(f"a_inplace({op}, {cur}, {v})")
            self.emit(f"a_setitem({o}, {i}, {r});")
        else:
            raise NativeError("cannot assign to this expression", t)

    def s_If(self, s):
        first = True
        closes = 0
        for cond, body in s.branches:
            t0 = self.fn.t
            c = self.cond(cond)
            self.emit(f"if ({c}) {{")
            self.fn.ind += 1
            self.block(body)
            self.fn.ind -= 1
            self.fn.t = t0
            if s.else_body or cond is not s.branches[-1][0]:
                self.emit("} else {")
                self.fn.ind += 1
                closes += 1
            else:
                self.emit("}")
            first = False
        if s.else_body:
            self.block(s.else_body)
        for _ in range(closes):
            self.fn.ind -= 1
            self.emit("}")
        del first

    def loop(self, head_lines, body, else_body, pre_body=None):
        f = self.fn
        flag = None
        if else_body:
            flag = self.label("brk")
            self.emit(f"int {flag} = 0;")
        for h in head_lines:
            self.emit(h)
        self.fn.ind += 1
        f.loops.append((flag, len(f.trys)))
        if pre_body:
            pre_body()
        self.block(body)
        f.loops.pop()
        self.fn.ind -= 1
        self.emit("}")
        if else_body:
            self.emit(f"if (!{flag}) {{")
            self.fn.ind += 1
            self.block(else_body)
            self.fn.ind -= 1
            self.emit("}")

    def s_While(self, s):
        def pre():
            t0 = self.fn.t
            c = self.cond(s.cond)
            self.emit(f"if (!({c})) break;")
            self.fn.t = t0
        self.loop(["for (;;) {", "    a_gc_safe();"], s.body, s.else_body, pre)

    def s_For(self, s):
        if s.parallel:
            pass                                  # a parallel for runs in order natively
        targets = s.targets
        it = s.iter
        if isinstance(it, A.Range):
            a = self.hold(self.expr(it.start))
            b = self.hold(self.expr(it.end))
            st = self.tmp()
            self.emit(f"{st} = (({a}.t == T_INT || {a}.t == T_BOOL) && ({b}.t == T_INT || {b}.t == T_BOOL)) ? A_NONE "
                      f": a_iter(a_range_val({a}, {b}, {int(bool(it.inclusive))}));")
            c = self.label("i")
            e = self.label("e")
            cur = self.tmp()
            self.emit(f"int64_t {c} = {a}.i, {e} = {b}.i + {int(bool(it.inclusive))};")

            def pre():
                self.emit(f"if ({st}.t == T_NONE) {{ if ({c} >= {e}) break; {cur} = a_int({c}++); }}")
                self.emit(f"else if (!a_next({st}, &{cur})) break;")
                self.bind_for(targets, cur)
            self.loop(["for (;;) {", "    a_gc_safe();"], s.body, s.else_body, pre)
            return
        src = self.hold(self.expr(it))
        itv = self.tmp()
        self.emit(f"{itv} = a_iter({src});")
        cur = self.tmp()

        def pre():
            self.emit(f"if (!a_next({itv}, &{cur})) break;")
            self.bind_for(targets, cur)
        self.loop(["for (;;) {", "    a_gc_safe();"], s.body, s.else_body, pre)

    def bind_for(self, targets, cur):
        if len(targets) == 1:
            self.assign_pattern(targets[0], cur)
        else:
            self.assign_pattern(list(targets), cur)

    def unwind(self, depth):
        """Leave the try blocks above depth (innermost first): pop handlers, run finally bodies."""
        f = self.fn
        saved = list(f.trys)
        for i in range(len(f.trys) - 1, depth - 1, -1):
            h, final, kind = f.trys[i]
            self.emit(f"a_handler = {h}.prev;")
            if final is not None and kind == "final":
                f.trys = saved[:i]
                self.block(final)
                f.trys = saved
        f.trys = saved

    def s_Break(self, s):
        f = self.fn
        if not f.loops:
            raise NativeError("break outside a loop", s)
        flag, depth = f.loops[-1]
        self.unwind(depth)
        self.emit(f"{flag + ' = 1; ' if flag else ''}break;")

    def s_Continue(self, s):
        f = self.fn
        if not f.loops:
            raise NativeError("continue outside a loop", s)
        self.unwind(f.loops[-1][1])
        self.emit("continue;")

    def ret(self, v):
        v = self.hold(v)
        self.unwind(0)
        self.emit(f"a_frame = fr.prev; return {v};")

    def s_Return(self, s):
        self.ret(self.expr(s.value) if s.value is not None else "A_NONE")

    def s_FnDef(self, s):
        params = s.params
        t = self.make_function(s, params, None if s.is_expr else s.body, s.body if s.is_expr else None, s.name)
        for d in reversed(s.decorators or []):
            dv = self.expr(d)
            t2 = self.tmp()
            self.emit(f"{t2} = a_call({dv}, 1, &{t}, NULL);")
            t = t2
        self.store(s.name, t)

    def s_StructDef(self, s):
        fields = [f.name for f in s.fields]
        arr = ", ".join(cstr(x) for x in fields) or "NULL"
        c = self.tmp()
        self.emit(f"{{ static const char *fl[] = {{{arr}}}; {c} = a_class({cstr(s.name)}, A_NONE, 1, {len(fields)}, fl); }}")
        for f in s.fields:
            if f.default is not None:
                v = self.expr(f.default)
                self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(f.name)}, {v});")
        self.store(s.name, c)                   # methods see the struct by its name
        for m in s.methods:
            t0 = self.fn.t
            fv = self.make_function(m, m.params, None if m.is_expr else m.body, m.body if m.is_expr else None,
                                    f"{s.name}.{m.name}")
            self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(m.name)}, {fv});")
            self.fn.t = t0

    def s_ClassDef(self, s):
        if len(s.bases) > 1:
            raise NativeError("a class with several bases is not supported by the native backend yet", s)
        base = self.hold(self.expr(s.bases[0])) if s.bases else "A_NONE"
        fields = [x.name for x in s.body if isinstance(x, A.FieldDef)]
        has_init = any(isinstance(x, A.FnDef) and x.name == "__init__" for x in s.body)
        is_struct = int(bool(fields) and not has_init)
        arr = ", ".join(cstr(x) for x in fields) if is_struct else "NULL"
        c = self.tmp()
        self.emit(f"{{ static const char *fl[] = {{{arr or 'NULL'}}}; {c} = a_class({cstr(s.name)}, {base}, "
                  f"{is_struct}, {len(fields) if is_struct else 0}, fl); }}")
        self.store(s.name, c)
        for item in s.body:
            t0 = self.fn.t
            if isinstance(item, A.FnDef):
                fv = self.make_function(item, item.params, None if item.is_expr else item.body,
                                        item.body if item.is_expr else None, f"{s.name}.{item.name}")
                for d in reversed(item.decorators or []):
                    dname = d.id if isinstance(d, A.Name) else None
                    if dname in ("staticmethod",):
                        continue
                    dv = self.expr(d)
                    t2 = self.tmp()
                    self.emit(f"{t2} = a_call({dv}, 1, &{fv}, NULL);")
                    fv = t2
                self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(item.name)}, {fv});")
            elif isinstance(item, A.FieldDef):
                if item.value is not None:
                    v = self.expr(item.value)
                    self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(item.name)}, {v});")
            elif isinstance(item, A.Let) and item.value is not None and len(item.targets) == 1 and \
                    isinstance(item.targets[0], str):
                v = self.expr(item.value)
                self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(item.targets[0])}, {v});")
            elif isinstance(item, A.Assign) and isinstance(item.target, A.Name):
                v = self.expr(item.value)
                self.emit(f"a_dict_set(((AClass *){c}.o)->attrs, {self.k_str(item.target.id)}, {v});")
            self.fn.t = t0
        for d in reversed(s.decorators or []):
            dv = self.expr(d)
            t2 = self.tmp()
            self.emit(f"{t2} = a_call({dv}, 1, &{c}, NULL);")
            self.store(s.name, t2)

    def s_Raise(self, s):
        if s.value is None:
            self.emit("a_raise(a_exc);")
            return
        v = self.expr(s.value)
        self.emit(f"a_raise({v});")

    def s_Assert(self, s):
        c = self.cond(s.cond)
        if s.msg is not None:
            self.emit(f"if (!({c})) {{")
            self.fn.ind += 1
            m = self.expr(s.msg)
            self.emit(f"a_raise(a_call(a_global_builtin(\"AssertionError\"), 1, (AV[]){{a_to_str({m})}}, NULL));")
            self.fn.ind -= 1
            self.emit("}")
        else:
            self.emit(f"if (!({c})) a_error(\"AssertionError\", \"assertion failed\");")

    def s_Del(self, s):
        for t in s.targets:
            if isinstance(t, A.Index):
                o = self.hold(self.expr(t.obj))
                i = self.index(t.index)
                self.emit(f"a_delitem({o}, {i});")
            elif isinstance(t, A.Name):
                self.store(t.id, "A_NONE")
            else:
                raise NativeError("cannot delete this", t)

    def s_Try(self, s):
        f = self.fn
        final = s.final_body or None
        if final:
            hf = self.label("hf")
            saved = self.tmp()
            self.emit(f"AHandler {hf};")
            self.emit(f"if (!A_TRY({hf})) {{")
            f.ind += 1
            f.trys.append((hf, final, "final"))
            self.try_except(s)
            f.trys.pop()
            self.emit(f"a_handler = {hf}.prev;")
            f.ind -= 1
            self.emit("} else {")
            f.ind += 1
            self.emit(f"{saved} = a_exc;")
            self.block(final)
            self.emit(f"a_raise({saved});")
            f.ind -= 1
            self.emit("}")
            self.block(final)
        else:
            self.try_except(s)

    def try_except(self, s):
        f = self.fn
        if not s.handlers:
            self.block(s.body)
            self.block(s.else_body or [])
            return
        h = self.label("h")
        self.emit(f"AHandler {h};")
        self.emit(f"if (!A_TRY({h})) {{")
        f.ind += 1
        f.trys.append((h, None, "except"))
        self.block(s.body)
        f.trys.pop()
        self.emit(f"a_handler = {h}.prev;")
        self.block(s.else_body or [])
        f.ind -= 1
        self.emit("} else {")
        f.ind += 1
        exc = self.tmp()
        self.emit(f"{exc} = a_exc;")
        first = True
        for kinds, name, body in s.handlers:
            if kinds is None:
                cond = "1"
            else:
                ks = kinds if isinstance(kinds, list) else [kinds]
                parts = []
                for k in ks:
                    kv = self.load(k) if isinstance(k, str) else self.expr(k)
                    parts.append(f"a_exc_matches({exc}, {kv})")
                cond = " || ".join(parts)
            self.emit(f"{'if' if first else '} else if'} ({cond}) {{")
            f.ind += 1
            if name:
                self.store(name, exc)
            self.emit(f"a_exc = {exc};")
            self.block(body)
            f.ind -= 1
            first = False
        self.emit("} else {")
        self.emit(f"    a_raise({exc});")
        self.emit("}")
        f.ind -= 1
        self.emit("}")

    def s_With(self, s):
        if len(s.items) != 1:
            raise NativeError("with several items is not supported natively yet", s)
        e, target = s.items[0]
        mgr = self.hold(self.expr(e))
        val = self.tmp()
        self.emit(f"{val} = a_call(a_attrk({mgr}, {self.k_str('__enter__')}), 0, NULL, NULL);")
        if target is not None:
            self.assign_pattern(target, val) if not isinstance(target, A.AST) else self.assign_target(target, val)
        exit_call = A.ExprStmt(A.Call(A.Attr(A.Name("$mgr"), "__exit__"), [A.NoneLit(), A.NoneLit(), A.NoneLit()], []))
        # the manager is reached through a hidden local
        self.fn.scope.bind("$mgr")
        self.store("$mgr", mgr)
        self.s_Try(A.Try(s.body, [], None, [exit_call], None))

    # ---- expressions: each returns a C expression; anything allocated is first put in a slot
    def expr(self, e):
        return getattr(self, "e_" + type(e).__name__, self.e_unknown)(e)

    def e_unknown(self, e):
        raise NativeError(f"'{type(e).__name__}' is not supported by the native backend yet", e)

    def e_Num(self, e):
        v = e.value
        if e.unit is not None:
            text, scale, dim = e.unit
            if dim.dimensionless:
                return f"a_float({float(v * scale)!r})"
            key = ("q", float(v * scale), self.dim_c(dim), text, float(scale))
            if key not in self.consts:
                self.consts[key] = len(self.consts)
                self.const_init.append(f"K[{self.consts[key]}] = a_qty({float(v * scale)!r}, "
                                       f"(const int16_t[]){self.dim_c(dim)}, {cstr(text)}, {float(scale)!r});")
            return f"K[{self.consts[key]}]"
        if isinstance(v, bool):
            return f"a_bool({int(v)})"
        if isinstance(v, int):
            if not -(2 ** 63) <= v < 2 ** 63:
                raise NativeError("integer constant does not fit in 64 bits", e)
            return f"a_int({v}LL)" if abs(v) > 2 ** 31 else f"a_int({v})"
        f = float(v)
        if f != f:
            return "a_float(NAN)"
        if f in (float("inf"), float("-inf")):
            return "a_float(INFINITY)" if f > 0 else "a_float(-INFINITY)"
        return f"a_float({f!r})"

    @staticmethod
    def dim_c(dim):
        out = []
        for x in dim:
            y = x * 60
            if y.denominator != 1:
                raise NativeError("a unit exponent must be a multiple of 1/60 natively")
            out.append(str(int(y)))
        return "{" + ", ".join(out) + "}"

    def e_As(self, e):
        v = self.expr(e.expr)
        text, scale, dim = e.unit
        return self.hold(f"a_as_unit({v}, (const int16_t[]){self.dim_c(dim)}, {cstr(text)}, {float(scale)!r})")

    def e_Str(self, e):
        return self.k_str(e.value)

    def e_Bool(self, e):
        return f"a_bool({int(e.value)})"

    def e_NoneLit(self, e):
        return "A_NONE"

    def e_Name(self, e):
        return self.load(e.id, e)

    def e_FStr(self, e):
        parts = []
        for p in e.parts:
            if isinstance(p, str):
                parts.append(self.k_str(p))
                continue
            ex, spec = p[0], p[1]
            conv = p[2] if len(p) > 2 else None
            v = self.hold(self.expr(ex))
            if conv == "r":
                v = self.hold(f"a_repr({v})")
            elif conv == "s":
                v = self.hold(f"a_to_str({v})")
            if isinstance(spec, A.FStr):
                sv = self.hold(self.expr(spec))
                v = self.hold(f"a_fmt_spec({v}, {sv})")
            elif spec:
                v = self.hold(f"a_format({v}, {cstr(spec)})")
            parts.append(v)
        if not parts:
            return self.k_str("")
        base = self.block_of(parts)
        t = self.tmp()
        self.emit(f"{t} = a_concat_strs({len(parts)}, &{base});")
        return t

    def block_of(self, vals):
        """Copy values into consecutive slots; returns the first slot."""
        f = self.fn
        base = f.t
        f.t += len(vals)
        f.tmax = max(f.tmax, f.t)
        for i, v in enumerate(vals):
            self.emit(f"S[T+{base + i}] = {v};")
        return f"S[T+{base}]"

    def e_Unary(self, e):
        v = self.expr(e.operand)
        if e.op == "not":
            return f"a_bool(!a_test({v}))"
        op = {"-": "OP_NEG", "+": "OP_POS", "~": "OP_INV"}[e.op]
        if e.op == "-" and isinstance(e.operand, A.Num) and e.operand.unit is None:
            return self.e_Num(A.Num(-e.operand.value, None, None))
        return self.hold(f"a_unop({op}, {v})")

    def e_Binary(self, e):
        a = self.hold(self.expr(e.left))
        b = self.expr(e.right)
        if e.op in FAST_BIN:
            return self.hold(f"{FAST_BIN[e.op]}({a}, {b})")
        if e.op in CMPOPS:
            return self.hold(f"a_bool({self.cmp_code(e.op, a, b)})")
        if e.op not in BINOPS:
            raise NativeError(f"operator {e.op} is not supported natively", e)
        return self.hold(f"a_binop({BINOPS[e.op]}, {a}, {b})")

    def cmp_code(self, op, a, b):
        if op in FAST_CMP:
            return f"{FAST_CMP[op]}({a}, {b})"
        return f"a_compare({CMPOPS[op]}, {a}, {b})"

    def e_Chain(self, e):
        r = self.tmp()
        end = self.label("ch")
        self.emit(f"{r} = a_bool(0);")
        self.emit("do {")
        self.fn.ind += 1
        a = self.hold(self.expr(e.operands[0]))
        for op, x in zip(e.ops, e.operands[1:]):
            b = self.hold(self.expr(x))
            self.emit(f"if (!{self.cmp_code(op, a, b)}) break;")
            a = b
        self.emit(f"{r} = a_bool(1);")
        self.fn.ind -= 1
        self.emit("} while (0);")
        del end
        return r

    def e_Logic(self, e):
        r = self.tmp()
        a = self.expr(e.left)
        self.emit(f"{r} = {a};")
        self.emit(f"if ({'' if e.op == 'and' else '!'}a_test({r})) {{")
        self.fn.ind += 1
        t0 = self.fn.t
        b = self.expr(e.right)
        self.emit(f"{r} = {b};")
        self.fn.t = t0
        self.fn.ind -= 1
        self.emit("}")
        return r

    def e_IfExpr(self, e):
        r = self.tmp()
        c = self.cond(e.cond)
        self.emit(f"if ({c}) {{")
        self.fn.ind += 1
        t0 = self.fn.t
        self.emit(f"{r} = {self.expr(e.a)};")
        self.fn.t = t0
        self.fn.ind -= 1
        self.emit("} else {")
        self.fn.ind += 1
        self.emit(f"{r} = {self.expr(e.b)};")
        self.fn.t = t0
        self.fn.ind -= 1
        self.emit("}")
        return r

    def cond(self, e):
        """A C int expression for a condition."""
        if isinstance(e, A.Binary) and e.op in CMPOPS:
            a = self.hold(self.expr(e.left))
            b = self.expr(e.right)
            return self.cmp_code(e.op, a, b)
        if isinstance(e, A.Unary) and e.op == "not":
            return f"!({self.cond(e.operand)})"
        if isinstance(e, A.Bool):
            return str(int(e.value))
        return f"a_test({self.expr(e)})"

    def e_NamedExpr(self, e):
        v = self.hold(self.expr(e.value))
        self.store(e.name, v)
        return v

    def e_ListLit(self, e):
        return self.seq(e.items, "a_list")

    def e_TupleLit(self, e):
        return self.seq(e.items, "a_tuple")

    def seq(self, items, ctor):
        if any(isinstance(x, A.Spread) for x in items):
            t = self.tmp()
            self.emit(f"{t} = a_list(0, NULL);")
            for x in items:
                if isinstance(x, A.Spread):
                    v = self.expr(x.value)
                    self.emit(f"a_inplace(OP_ADD, {t}, {v});")
                else:
                    v = self.expr(x)
                    self.emit(f"a_list_append((AList *){t}.o, {v});")
            if ctor == "a_tuple":
                self.emit(f"{t}.o->kind = O_TUPLE;")
            return t
        if not items:
            return self.hold(f"{ctor}(0, NULL)")
        vals = [self.hold(self.expr(x)) for x in items]
        base = self.block_of(vals)
        return self.hold(f"{ctor}({len(items)}, &{base})")

    def e_SetLit(self, e):
        t = self.tmp()
        self.emit(f"{t} = a_set();")
        for x in e.items:
            if isinstance(x, A.Spread):
                v = self.hold(self.expr(x.value))
                it = self.tmp()
                cur = self.tmp()
                self.emit(f"{it} = a_iter({v}); while (a_next({it}, &{cur})) a_dict_set((ADict *){t}.o, {cur}, A_NONE);")
            else:
                v = self.expr(x)
                self.emit(f"a_dict_set((ADict *){t}.o, {v}, A_NONE);")
        return t

    def e_DictLit(self, e):
        t = self.tmp()
        self.emit(f"{t} = a_dict();")
        for k, v in e.pairs:
            if k is None or isinstance(k, A.Spread):          # **other
                src = v if k is None else k.value
                if isinstance(src, A.Spread):
                    src = src.value
                sv = self.expr(src)
                self.emit(f"a_inplace(OP_OR, {t}, {sv});")
                continue
            kv = self.hold(self.expr(k))
            vv = self.expr(v)
            self.emit(f"a_dict_set((ADict *){t}.o, {kv}, {vv});")
        return t

    def comp_loops(self, e, inner):
        """Emit the nested loops of a comprehension; inner() emits the innermost body."""
        clauses = [(e.targets, e.iter, e.cond)] + list(e.more or [])
        opened = 0
        for targets, it, cond in clauses:
            src = self.hold(self.expr(it))
            itv = self.tmp()
            cur = self.tmp()
            self.emit(f"{itv} = a_iter({src});")
            self.emit("for (;;) {")
            self.fn.ind += 1
            opened += 1
            self.emit(f"if (!a_next({itv}, &{cur})) break;")
            self.bind_for(targets, cur)
            if cond is not None:
                c = self.cond(cond)
                self.emit(f"if (!({c})) continue;")
        t0 = self.fn.t
        inner()
        self.fn.t = t0
        for _ in range(opened):
            self.fn.ind -= 1
            self.emit("}")

    def e_ListComp(self, e):
        r = self.tmp()
        self.emit(f"{r} = a_list(0, NULL);")
        self.comp_loops(e, lambda: self.emit(f"a_list_append((AList *){r}.o, {self.expr(e.elem)});"))
        return r

    def e_GenExp(self, e):
        return self.e_ListComp(e)             # natively a generator expression is evaluated eagerly

    def e_SetComp(self, e):
        r = self.tmp()
        self.emit(f"{r} = a_set();")
        self.comp_loops(e, lambda: self.emit(f"a_dict_set((ADict *){r}.o, {self.expr(e.elem)}, A_NONE);"))
        return r

    def e_DictComp(self, e):
        r = self.tmp()
        self.emit(f"{r} = a_dict();")

        def inner():
            k = self.hold(self.expr(e.key))
            v = self.expr(e.value)
            self.emit(f"a_dict_set((ADict *){r}.o, {k}, {v});")
        self.comp_loops(e, inner)
        return r

    def e_Lambda(self, e):
        return self.make_function(e, e.params, None, e.body, "<lambda>")

    def e_Range(self, e):
        a = self.hold(self.expr(e.start))
        b = self.expr(e.end)
        return self.hold(f"a_range_val({a}, {b}, {int(bool(e.inclusive))})")

    def e_Attr(self, e):
        o = self.expr(e.obj)
        return self.hold(f"a_attrk({o}, {self.k_str(e.name)})")

    def index(self, ix):
        if isinstance(ix, A.Slice):
            lo = self.hold(self.expr(ix.lo) if ix.lo is not None else "A_NONE")
            hi = self.hold(self.expr(ix.hi) if ix.hi is not None else "A_NONE")
            st = self.expr(ix.step) if ix.step is not None else "A_NONE"
            return self.hold(f"a_slice({lo}, {hi}, {st})")
        if isinstance(ix, A.TupleLit):
            return self.e_TupleLit(ix)
        return self.expr(ix)

    def e_Index(self, e):
        o = self.hold(self.expr(e.obj))
        i = self.index(e.index)
        return self.hold(f"a_getitem({o}, {i})")

    def e_Slice(self, e):
        return self.index(e)

    def e_Call(self, e):
        f = e.func
        # super().name(...) inside a method
        if isinstance(f, A.Attr) and isinstance(f.obj, A.Call) and isinstance(f.obj.func, A.Name) and \
                f.obj.func.id == "super" and not f.obj.args:
            cls_name, self_name = self.method_owner(e)
            fv = self.hold(f"a_super_attr({self.load(self_name)}, {self.load(cls_name)}, {cstr(f.name)})")
        else:
            fv = self.hold(self.expr(f))
        dspread = [a.value for a in e.args if isinstance(a, A.Spread) and a.double]
        pargs = [a for a in e.args if not (isinstance(a, A.Spread) and a.double)]
        spread = any(isinstance(a, A.Spread) for a in pargs) or any(k is None for k, _ in e.kwargs) or dspread
        if spread:
            lst = self.seq(pargs, "a_tuple") if pargs else self.hold("a_tuple(0, NULL)")
            kw = "A_NONE"
            if e.kwargs or dspread:
                pairs = [(None if k is None else A.Str(k), v) for k, v in e.kwargs] + [(None, d) for d in dspread]
                kw = self.e_DictLit(A.DictLit(pairs))
            return self.hold(f"a_call_spread({fv}, {lst}, {kw})")
        kw = "NULL"
        if e.kwargs:
            d = self.e_DictLit(A.DictLit([(A.Str(k), v) for k, v in e.kwargs]))
            kw = f"(ADict *){d}.o"
        if not e.args:
            return self.hold(f"a_call({fv}, 0, NULL, {kw})")
        f_ = self.fn
        base = f_.t
        f_.t += len(e.args)
        f_.tmax = max(f_.tmax, f_.t)
        for i, a in enumerate(e.args):
            v = self.expr(a)
            self.emit(f"S[T+{base + i}] = {v};")
        return self.hold(f"a_call({fv}, {len(e.args)}, &S[T+{base}], {kw})")

    def method_owner(self, node):
        """For super(): the class being defined and the method's first parameter."""
        sc = self.fn.scope
        n = sc.node
        owner = None
        for s in self.analyzer.scopes.values():
            if s.kind == "class":
                body = s.node.body if isinstance(s.node, A.ClassDef) else s.node.methods
                if any(x is n for x in body):
                    owner = s.node.name
        if owner is None or not getattr(n, "params", None):
            raise NativeError("super() outside a method", node)
        return owner, n.params[0].name

    def e_Spread(self, e):
        raise NativeError("*x is only allowed in calls and literals", e)


def compile_c(src, filename="<program>"):
    from ..parser import parse
    prog = parse(src)
    return CGen(filename, src).program(prog)
