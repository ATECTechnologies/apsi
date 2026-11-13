"""`apsi convert` — Python → Apsi with Python's exact semantics.

The result starts with `use pycompat`, which gives the file Python's object model and built-ins (see
pycompat.py), so the converted program does exactly what the Python program did — classes that inherit from
library classes (torch.nn.Module, threading.Thread, Exception …), `with`, generators, decorators, closures,
`global`/`nonlocal`, star-unpacking, f-strings with conversions and nested specs, every operator — while the
text is ordinary Apsi: `fn`, `let`/`var`, `none`/`true`/`false`, `import "x.apsi"`, `pyimport("torch")`.

Rules that keep it exact:
* Names follow Python's scoping. A name is declared (`let`/`var`) where Python first binds it; names read before
  their first binding in a function are declared `var name` (bound later, exactly like Python) at the top of that function.
* Modules next to the program (x.py, packages with __init__.py) become `import "x.apsi"`; everything else is
  loaded with pyimport()/pyfrom() — pure standard modules need no permission, others need --allow-python.
* Annotations of variables and parameters are dropped (Python does not enforce them either); class field
  annotations are kept (dataclasses need them).
* Comments, docstrings, blank lines and the order of everything are kept.

What has no exact Apsi form (match statements, `except*`) is reported with its line number and kept as a comment.
"""
from . import __version__
import ast
import io
import os
import re
import tokenize

from .lexer import KEYWORDS

# Python identifiers that are Apsi keywords (Python's own keywords cannot be identifiers anyway)
_CLASH = {k for k in KEYWORDS if k not in {"return", "if", "elif", "else", "while", "for", "in", "break", "continue",
                                           "and", "or", "not", "import", "assert", "as", "pass"}}

# precedence levels (higher binds tighter)
P_LAMBDA, P_IF, P_OR, P_AND, P_NOT, P_CMP, P_BOR, P_BXOR, P_BAND, P_SHIFT, P_ADD, P_MUL, P_UNARY, P_POW, \
    P_AWAIT, P_ATOM = range(1, 17)
BINOP = {ast.Add: ("+", P_ADD), ast.Sub: ("-", P_ADD), ast.Mult: ("*", P_MUL), ast.Div: ("/", P_MUL),
         ast.FloorDiv: ("//", P_MUL), ast.Mod: ("%", P_MUL), ast.MatMult: ("@", P_MUL), ast.Pow: ("**", P_POW),
         ast.LShift: ("<<", P_SHIFT), ast.RShift: (">>", P_SHIFT), ast.BitOr: ("|", P_BOR),
         ast.BitXor: ("^", P_BXOR), ast.BitAnd: ("&", P_BAND)}
CMPOP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.In: "in",
         ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not"}


class Unsupported(Exception):
    pass


def q(s):
    """An Apsi string literal with Python's exact value."""
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 32 or o == 127:
            out.append(f"\\x{o:02x}")
        elif 0x80 <= o < 0xA0 or ch in "  ﻿" or 0xD800 <= o <= 0xDFFF or o in (0xAD,) or \
                0x200B <= o <= 0x200F or 0x202A <= o <= 0x202E or 0x2060 <= o <= 0x206F or 0xFFF0 <= o:
            out.append(f"\\u{o:04x}" if o <= 0xFFFF else f"\\U{o:08x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def qb(b):
    out = ['b"']
    for o in b:
        ch = chr(o)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif 32 <= o < 127:
            out.append(ch)
        else:
            out.append(f"\\x{o:02x}")
    out.append('"')
    return "".join(out)


# ------------------------------------------------------------------------------------------------ scope analysis
class FnScope:
    """The names one Python scope binds, and the order of first binding."""

    def __init__(self, kind, parent=None, params=()):
        self.kind = kind                    # module | function | class
        self.parent = parent
        self.params = set(params)
        self.locals = set()
        self.globals = set()
        self.nonlocals = set()
        self.declared = set(params)         # declared so far while emitting
        self.hoist = set()                  # declared at the top (read before first binding)


def _bound_names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            yield from _bound_names(e)
    elif isinstance(target, ast.Starred):
        yield from _bound_names(target.value)


class _Binder(ast.NodeVisitor):
    """Collect names bound and first-load-before-bind facts for one scope (not entering nested scopes)."""

    def __init__(self):
        self.bound = []                     # in textual order (first occurrence)
        self.loads_before = set()
        self.globals = set()
        self.nonlocals = set()
        self._seen = set()
        self.count = {}
        self.loop_depth = 0

    def bind(self, name):
        self.count[name] = self.count.get(name, 0) + 1
        if name not in self._seen:
            self._seen.add(name)
            self.bound.append(name)

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load):
            if n.id not in self._seen:
                self.loads_before.add(n.id)
        else:
            self.bind(n.id)

    def visit_Global(self, n):
        self.globals.update(n.names)

    def visit_Nonlocal(self, n):
        self.nonlocals.update(n.names)

    def _fn(self, n):
        for d in n.decorator_list:
            self.visit(d)
        a = n.args
        for d in list(a.defaults) + [x for x in a.kw_defaults if x is not None]:
            self.visit(d)
        self.bind(n.name)

    visit_FunctionDef = visit_AsyncFunctionDef = _fn

    def visit_ClassDef(self, n):
        for d in n.decorator_list:
            self.visit(d)
        for b in n.bases:
            self.visit(b)
        for k in n.keywords:
            self.visit(k.value)
        self.bind(n.name)

    def visit_Lambda(self, n):
        for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x is not None]:
            self.visit(d)

    def _comp(self, n):
        # the first iterable is evaluated in this scope; walrus targets bind here
        self.visit(n.generators[0].iter)
        for node in ast.walk(n):
            if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                self.bind(node.target.id)

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = _comp

    def visit_Import(self, n):
        for a in n.names:
            self.bind(a.asname or a.name.split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            if a.name != "*":
                self.bind(a.asname or a.name)

    def visit_ExceptHandler(self, n):
        if n.type is not None:
            self.visit(n.type)
        if n.name:
            self.bind(n.name)
        for s in n.body:
            self.visit(s)

    def visit_AugAssign(self, n):
        if isinstance(n.target, ast.Name) and n.target.id not in self._seen:
            self.loads_before.add(n.target.id)
        self.visit(n.value)
        if isinstance(n.target, ast.Name):
            self.bind(n.target.id)
        else:
            self.visit(n.target)

    def visit_Assign(self, n):
        self.visit(n.value)
        for t in n.targets:
            self.visit(t)

    def visit_AnnAssign(self, n):
        if n.value is not None:
            self.visit(n.value)
            self.visit(n.target)

    def visit_For(self, n):
        self.visit(n.iter)
        self.loop_depth += 1
        self.visit(n.target)
        for s in n.body:
            self.visit(s)
        self.loop_depth -= 1
        for s in n.orelse:
            self.visit(s)

    visit_AsyncFor = visit_For

    def visit_While(self, n):
        self.visit(n.test)
        self.loop_depth += 1
        for s in n.body:
            self.visit(s)
        self.loop_depth -= 1
        for s in n.orelse:
            self.visit(s)

    def visit_NamedExpr(self, n):
        self.visit(n.value)
        self.bind(n.target.id)


def analyse(body, params=()):
    b = _Binder()
    for s in body:
        b.visit(s)
    # names a nested function rebinds (nonlocal / global): never constant
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Nonlocal, ast.Global)):
                for name in sub.names:
                    b.count[name] = b.count.get(name, 0) + 2
    return b


# ------------------------------------------------------------------------------------------------ converter
class Converter:
    def __init__(self, source, filename="<python>", base_dir=None, packages=None, out_path=None, vendored=False):
        self.vendored = vendored            # a module of a converted library: absolute imports never mean a sibling
        self.packages = packages            # a Packages: pure-Python libraries are converted with the program
        self.out_path = out_path            # where this module's .apsi is written (paths to converted libraries)
        self.source = source
        self.filename = filename
        self.base_dir = base_dir or (os.path.dirname(os.path.abspath(filename)) if filename and
                                     not filename.startswith("<") else os.getcwd())
        self.tree = ast.parse(source, filename)
        self.lines = []
        self.warnings = []
        self.scope = None
        self.tmp = 0
        self.comments = self._read_comments()
        self.used_comments = set()
        self.src_lines = source.splitlines()
        self.needs_python = False
        self.in_loop = 0
        self.local_modules = set()
        self.kept_local = set()                     # modules next to the program kept as Python (convert --only)
        self.pkg_sources = {}               # .apsi path of a converted library -> its Python source
        self.renames = {}

    # -------------------------------------------------------------------------------- output helpers
    def warn(self, node, msg):
        line = getattr(node, "lineno", 0)
        if (line, msg) not in self.warnings:
            self.warnings.append((line, msg))

    def emit(self, text, ind):
        self.lines.append("    " * ind + text if text else "")

    def fresh(self, base="t"):
        self.tmp += 1
        return f"_{base}{self.tmp}"

    def _read_comments(self):
        out = {}
        try:
            lines = self.source.splitlines()
            for tok in tokenize.generate_tokens(io.StringIO(self.source).readline):
                if tok.type == tokenize.COMMENT:
                    line = tok.start[0]
                    before = lines[line - 1][:tok.start[1]]
                    out[line] = (tok.string[1:], not before.strip())
        except (tokenize.TokenError, IndentationError, SyntaxError):
            pass
        return out

    def _comments_before(self, upto, ind):
        for line in sorted(k for k in self.comments if k < upto and k not in self.used_comments):
            text, own = self.comments[line]
            self.used_comments.add(line)
            if own:
                if line > 1 and self.lines and self.lines[-1] != "" and line - 1 not in self.comments and \
                        line - 2 >= 0 and self.src_lines[line - 2].strip() == "":
                    self.emit("", 0)
                if line == 1 and text.startswith("!"):
                    continue
                self.emit("#" + text.rstrip(), ind)

    def name(self, n):
        """A Python identifier as an Apsi name."""
        if n in _CLASH:
            return f"`{n}`"                                      # the same name, written so Apsi reads it as a name
        return n

    # -------------------------------------------------------------------------------- entry
    def convert(self):
        mod_scope = FnScope("module")
        info = analyse(self.tree.body)
        self.scope = mod_scope
        mod_scope.locals = set(info.bound)
        mod_scope.counts = info.count
        mod_scope.hoist = {n for n in info.loads_before if n in mod_scope.locals}
        # a module-level global set only from functions: declare it at the top
        assigned_in_fns = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Global):
                assigned_in_fns.update(node.names)
        mod_scope.hoist |= {n for n in assigned_in_fns if n not in info.bound}
        body = self.tree.body
        out_body = []
        saved, self.lines = self.lines, out_body
        for n in sorted(mod_scope.hoist):
            self.emit(f"var {self.name(n)}", 0)
            mod_scope.declared.add(n)
        self.block(body, 0, top=True)
        self._comments_before(10 ** 9, 0)
        self.lines = saved
        head = [f"# Converted from {os.path.basename(self.filename)} by `apsi convert` {__version__} (Python's exact semantics).",
                "use pycompat"]
        text = "\n".join(head + out_body).rstrip() + "\n"
        return re.sub(r"\n{4,}", "\n\n\n", text)

    # -------------------------------------------------------------------------------- statements
    def block(self, stmts, ind, top=False):
        start = len(self.lines)
        prev_end = None
        for s in stmts:
            if prev_end is not None and s.lineno - 1 > prev_end:
                gap = self.src_lines[prev_end:s.lineno - 1]
                if any(not g.strip() for g in gap) and self.lines and self.lines[-1] != "":
                    self.emit("", 0)
            self._comments_before(s.lineno, ind)
            before = len(self.lines)
            try:
                self.stmt(s, ind)
            except Unsupported as e:
                self.warn(s, str(e))
                self.emit(f"# NOTE: not converted — {e}", ind)
                for line in (ast.get_source_segment(self.source, s) or "").splitlines():
                    self.emit("#     " + line, ind)
                self.emit("pass", ind)
            c = self.comments.get(s.lineno)
            if c and not c[1] and s.lineno not in self.used_comments and len(self.lines) > before:
                self.used_comments.add(s.lineno)
                k = before
                while k < len(self.lines) - 1 and (not self.lines[k].strip() or self.lines[k].lstrip().startswith("#")):
                    k += 1
                if '"""' not in self.lines[k]:
                    self.lines[k] += "  #" + c[0].rstrip()
            prev_end = getattr(s, "end_lineno", s.lineno)
        if not any(l.strip() and not l.strip().startswith("#") for l in self.lines[start:]) and not top:
            self.emit("pass", ind)

    def stmt(self, s, ind):
        m = getattr(self, "s_" + type(s).__name__, None)
        if m is None:
            raise Unsupported(f"{type(s).__name__} statements")
        m(s, ind)

    def s_Expr(self, s, ind):
        v = s.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            self.emit(self._docstring(v.value, ind), ind)
            return
        if isinstance(v, (ast.Yield, ast.YieldFrom)):
            self.emit(self.yield_text(v), ind)
            return
        self.emit(self.ex(v), ind)

    def _docstring(self, text, ind):
        if "\n" in text and '"""' not in text and "\\" not in text and not text.endswith('"') and \
                all(ch == "\n" or ch == "\t" or 32 <= ord(ch) for ch in text):
            return '"""' + text + '"""'
        return q(text)

    def s_Pass(self, s, ind):
        self.emit("pass", ind)

    def s_Break(self, s, ind):
        self.emit("break", ind)

    def s_Continue(self, s, ind):
        self.emit("continue", ind)

    def s_Return(self, s, ind):
        self.emit("return" + (" " + self.ex(s.value, top=True) if s.value is not None else ""), ind)

    def s_Global(self, s, ind):
        self.emit("global " + ", ".join(self.name(n) for n in s.names), ind)

    def s_Nonlocal(self, s, ind):
        self.emit("nonlocal " + ", ".join(self.name(n) for n in s.names), ind)

    def s_Delete(self, s, ind):
        self.emit("del " + ", ".join(self.ex(t) for t in s.targets), ind)

    def s_Assert(self, s, ind):
        self.emit("assert " + self.ex(s.test) + (", " + self.ex(s.msg) if s.msg is not None else ""), ind)

    def s_Raise(self, s, ind):
        if s.exc is None:
            self.emit("raise", ind)
            return
        self.emit("raise " + self.ex(s.exc) + (" from " + self.ex(s.cause) if s.cause is not None else ""), ind)

    # ---- binding
    def _is_declared(self, name):
        return name in self.scope.declared

    def _declare_names(self, names, ind):
        """Declare names that are about to be bound by a statement that cannot declare them itself."""
        for n in names:
            if n in self.scope.globals or n in self.scope.nonlocals:
                continue
            if not self._is_declared(n):
                self.emit(f"var {self.name(n)}", ind)
                self.scope.declared.add(n)

    def _decl_kw(self, name):
        counts = getattr(self.scope, "counts", {})
        return "let" if counts.get(name, 2) == 1 and not self.in_loop else "var"

    def s_Assign(self, s, ind):
        value = self.rhs(s.value)
        targets = s.targets
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            n = targets[0].id
            if self._binds_new(n):
                self.scope.declared.add(n)
                self.emit(f"{self._decl_kw(n)} {self.name(n)} = {value}", ind)
                return
            self.emit(f"{self.name(n)} = {value}", ind)
            return
        if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)) and \
                all(isinstance(e, ast.Name) or isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
                    for e in targets[0].elts) and len(targets[0].elts) > 1:
            names = [e.id if isinstance(e, ast.Name) else e.value.id for e in targets[0].elts]
            if all(self._binds_new(n) for n in names) and len(set(names)) == len(names):
                for n in names:
                    self.scope.declared.add(n)
                parts = [self.name(e.id) if isinstance(e, ast.Name) else "*" + self.name(e.value.id)
                         for e in targets[0].elts]
                self.emit(f"var {', '.join(parts)}{',' if len(parts) == 1 else ''} = {value}", ind)
                return
        names = [n for t in targets for n in _bound_names(t)]
        self._declare_names([n for n in dict.fromkeys(names) if self._binds_new(n)], ind)
        self.emit(" = ".join(self.target(t) for t in targets) + f" = {value}", ind)

    def _binds_new(self, n):
        sc = self.scope
        if n in sc.globals or n in sc.nonlocals:
            return False
        return n not in sc.declared

    def target(self, t):
        if isinstance(t, ast.Name):
            return self.name(t.id)
        if isinstance(t, (ast.Tuple, ast.List)):
            inner = ", ".join(self.target(e) for e in t.elts)
            return f"({inner}{',' if len(t.elts) == 1 else ''})"
        if isinstance(t, ast.Starred):
            return "*" + self.target(t.value)
        return self.ex(t)

    def s_AugAssign(self, s, ind):
        op = BINOP[type(s.op)][0]
        if isinstance(s.target, ast.Name):
            self._declare_names([s.target.id] if self._binds_new(s.target.id) else [], ind)
        self.emit(f"{self.target(s.target)} {op}= {self.rhs(s.value)}", ind)

    def s_AnnAssign(self, s, ind):
        if self.scope.kind == "class" and isinstance(s.target, ast.Name):
            ann = self.annotation(s.annotation)
            n = s.target.id
            self.scope.declared.add(n)
            self.emit(f"{self.name(n)}: {ann}" + (f" = {self.rhs(s.value)}" if s.value is not None else ""), ind)
            return
        if s.value is None:
            return
        self.s_Assign(ast.Assign(targets=[s.target], value=s.value, lineno=s.lineno), ind)

    def annotation(self, a):
        try:
            text = self.ex(a)
        except Unsupported:
            text = q(ast.unparse(a))
        return text

    def rhs(self, v):
        if isinstance(v, (ast.Yield, ast.YieldFrom)):
            return self.yield_text(v)
        if isinstance(v, ast.Await):
            return self.ex(v, top=True)
        return self.ex(v, top=True)

    def yield_text(self, v):
        if isinstance(v, ast.YieldFrom):
            return "yield from " + self.ex(v.value, top=True)
        return "yield" + (" " + self.ex(v.value, top=True) if v.value is not None else "")

    # ---- control flow
    def s_If(self, s, ind):
        if self._is_type_checking(s.test) and not s.orelse:
            self.emit("if false:  # typing.TYPE_CHECKING", ind)
            self.block(s.body, ind + 1)
            return
        self.emit(f"if {self.ex(s.test, top=True)}:", ind)
        self.block(s.body, ind + 1)
        orelse = s.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If) and self._is_elif(orelse[0]):
            e = orelse[0]
            self._comments_before(e.lineno, ind)
            self.emit(f"elif {self.ex(e.test, top=True)}:", ind)
            self.block(e.body, ind + 1)
            orelse = e.orelse
        if orelse:
            self.emit("else:", ind)
            self.block(orelse, ind + 1)

    def _is_type_checking(self, t):
        return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")

    def _is_elif(self, node):
        line = self.src_lines[node.lineno - 1] if node.lineno - 1 < len(self.src_lines) else ""
        return line.lstrip().startswith("elif")

    def s_While(self, s, ind):
        self.emit(f"while {self.ex(s.test, top=True)}:", ind)
        self.in_loop += 1
        self.block(s.body, ind + 1)
        self.in_loop -= 1
        if s.orelse:
            self.emit("else:", ind)
            self.block(s.orelse, ind + 1)

    def _loop_target(self, t, allow_expr=False):
        """Apsi loop target text, or None when it must go through a temporary (attribute/index targets)."""
        if isinstance(t, ast.Name):
            return self.name(t.id)
        if isinstance(t, ast.Starred) and isinstance(t.value, ast.Name):
            return "*" + self.name(t.value.id)
        if allow_expr and isinstance(t, (ast.Attribute, ast.Subscript)) and isinstance(
                t.value if isinstance(t, ast.Attribute) else t.value, ast.Name):
            return self.ex(t)
        if isinstance(t, (ast.Tuple, ast.List)):
            parts = [self._loop_target(e, allow_expr) for e in t.elts]
            if any(p is None for p in parts):
                return None
            return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
        return None

    def s_For(self, s, ind, kw="for"):
        tgt = self._loop_target(s.target)
        pre = []
        if tgt is None:
            tmp = self.fresh("item")
            tgt = tmp
            self._declare_names([n for n in dict.fromkeys(_bound_names(s.target)) if self._binds_new(n)], ind)
            pre = [f"{self.target(s.target)} = {tmp}"]
        else:
            for n in _bound_names(s.target):
                if self._binds_new(n):
                    self.scope.declared.add(n)
        if tgt.startswith("(") and tgt.endswith(")") and ", " in tgt:
            tgt = tgt[1:-1]
        self.emit(f"{kw} {tgt} in {self.ex(s.iter, top=True)}:", ind)
        for p in pre:
            self.emit(p, ind + 1)
        self.in_loop += 1
        self.block(s.body, ind + 1)
        self.in_loop -= 1
        if s.orelse:
            self.emit("else:", ind)
            self.block(s.orelse, ind + 1)

    def s_AsyncFor(self, s, ind):
        self.s_For(s, ind, "async for")

    def s_With(self, s, ind, kw="with"):
        parts = []
        post = []
        for it in s.items:
            text = self.ex(it.context_expr)
            if it.optional_vars is not None:
                t = self._loop_target(it.optional_vars)
                if t is None:
                    tmp = self.fresh("ctx")
                    post.append(f"{self.target(it.optional_vars)} = {tmp}")
                    t = tmp
                else:
                    for n in _bound_names(it.optional_vars):
                        if self._binds_new(n):
                            self.scope.declared.add(n)
                text += " as " + t
            parts.append(text)
        self.emit(f"{kw} {', '.join(parts)}:", ind)
        for p in post:
            self.emit(p, ind + 1)
        self.block(s.body, ind + 1)

    def s_AsyncWith(self, s, ind):
        self.s_With(s, ind, "async with")

    def s_Try(self, s, ind, star=False):
        self.emit("try:", ind)
        self.block(s.body, ind + 1)
        computed = []
        for h in s.handlers:
            self._comments_before(h.lineno, ind)
            head = "except*" if star else "except"
            kinds = []
            if h.type is not None:
                elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
                kinds = [self._dotted(t) for t in elts]
            if h.type is not None and (None in kinds or computed) and not star:
                # error kinds computed by an expression: catch, test with isinstance, re-raise the others
                computed.append(h)
                continue
            if kinds:
                head += " " + ", ".join(kinds)
            if h.name:
                head += " as " + self.name(h.name)
                self.scope.declared.add(h.name)
            self.emit(head + ":", ind)
            self.block(h.body, ind + 1)
        if computed:
            tmp = self.fresh("err")
            self.emit(f"except BaseException as {tmp}:", ind)
            for i, h in enumerate(computed):
                self.emit(f"{'if' if i == 0 else 'elif'} isinstance({tmp}, {self.ex(h.type, P_LAMBDA)}):", ind + 1)
                if h.name:
                    self._bind(h.name, tmp, ind + 2)
                self.block(h.body, ind + 2)
            self.emit("else:", ind + 1)
            self.emit("raise", ind + 2)
        if s.orelse:
            self.emit("else:", ind)
            self.block(s.orelse, ind + 1)
        if s.finalbody:
            self.emit("finally:", ind)
            self.block(s.finalbody, ind + 1)

    def s_TryStar(self, s, ind):
        self.s_Try(s, ind, star=True)

    def s_Match(self, s, ind):
        self.emit(f"match {self.ex(s.subject, top=True)}:", ind)
        for c in s.cases:
            self._comments_before(c.pattern.lineno, ind + 1)
            for n in _pattern_names(c.pattern):
                self.scope.declared.add(n)
            g = f" if {self.ex(c.guard, P_LAMBDA)}" if c.guard is not None else ""
            self.emit(f"case {self.pat(c.pattern, top=True)}{g}:", ind + 1)
            self.block(c.body, ind + 2)

    def pat(self, p, top=False):
        if isinstance(p, ast.MatchValue):
            return self.ex(p.value)
        if isinstance(p, ast.MatchSingleton):
            return {None: "none", True: "true", False: "false"}[p.value]
        if isinstance(p, ast.MatchSequence):
            return "[" + ", ".join(self.pat(x) for x in p.patterns) + "]"
        if isinstance(p, ast.MatchStar):
            return "*" + (self.name(p.name) if p.name else "_")
        if isinstance(p, ast.MatchMapping):
            items = [f"{self.ex(k)}: {self.pat(v)}" for k, v in zip(p.keys, p.patterns)]
            if p.rest:
                items.append("**" + self.name(p.rest))
            return "{" + ", ".join(items) + "}"
        if isinstance(p, ast.MatchClass):
            args = [self.pat(x) for x in p.patterns] + [f"{k}={self.pat(v)}" for k, v in zip(p.kwd_attrs,
                                                                                          p.kwd_patterns)]
            return f"{self.ex(p.cls)}({', '.join(args)})"
        if isinstance(p, ast.MatchAs):
            if p.pattern is None:
                return self.name(p.name) if p.name else "_"
            inner = self.pat(p.pattern)
            return f"{inner if top else '(' + inner + ')'} as {self.name(p.name)}"
        if isinstance(p, ast.MatchOr):
            text = " | ".join(self.pat(x) for x in p.patterns)
            return text if top else f"({text})"
        raise Unsupported(f"the pattern {type(p).__name__}")

    def s_TypeAlias(self, s, ind):
        name = s.name.id
        self._declare_type_params(s.type_params, ind)
        self._bind(name, f'pyfrom("typing", "TypeAliasType")({q(name)}, {self.ex(s.value, P_LAMBDA)})', ind)

    def _declare_type_params(self, params, ind):
        """PEP 695 type parameters: real TypeVar / ParamSpec / TypeVarTuple objects, as Python creates them."""
        out = []
        for tp in params or ():
            kind = {"TypeVar": "TypeVar", "ParamSpec": "ParamSpec", "TypeVarTuple": "TypeVarTuple"}[type(tp).__name__]
            extra = ""
            bound = getattr(tp, "bound", None)
            if bound is not None:
                if isinstance(bound, ast.Tuple):
                    extra = ", " + ", ".join(self.ex(b, P_LAMBDA) for b in bound.elts)
                else:
                    extra = f", bound={self.ex(bound, P_LAMBDA)}"
            self._bind(tp.name, f'pyfrom("typing", {q(kind)})({q(tp.name)}{extra})', ind)
            out.append(tp)
        return out

    def _dotted(self, t):
        parts = []
        while isinstance(t, ast.Attribute):
            parts.append(t.attr)
            t = t.value
        if isinstance(t, ast.Name):
            parts.append(self.name(t.id))
            return ".".join(reversed(parts))
        return None

    # ---- imports
    def _local(self, module, level=0):
        """The .apsi path of a module that sits next to the program (converted with it), or None."""
        if not module and not level:
            return None
        if self.vendored and not level:
            return None
        base = self.base_dir
        for _ in range(max(0, level - 1)):
            base = os.path.dirname(base)
        rel = (module or "").replace(".", os.sep)
        for cand in (rel + ".py", os.path.join(rel, "__init__.py")):
            if rel and os.path.exists(os.path.join(base, cand)):
                keep = getattr(self, "keep_local", None)
                if keep is not None and keep(os.path.abspath(os.path.join(base, cand))):
                    if not level:
                        self.kept_local.add(module)
                        return None                 # stays a Python import: pyimport() finds it next to the program
                apsi = cand[:-3] + ".apsi"
                self.local_modules.add(os.path.join(base, cand))
                return os.path.relpath(os.path.join(base, apsi), self.base_dir).replace(os.sep, "/")
        return None

    def _package(self, module, node=None):
        """The .apsi path (from this module) of an installed pure-Python library that is converted with the
        program, or None (standard library, compiled libraries, not found)."""
        if not module or self.packages is None or self.out_path is None:
            return None
        hit = self.packages.resolve(module)
        if hit is None:
            top = module.split(".")[0]
            kind = self.packages.kind(top)
            if kind == "compiled" and node is not None:
                self.warn(node, f"'{top}' is a compiled (C/C++) library: used as is through "
                                f"Python{self.packages.hint(top)}")
            elif kind is None and node is not None and top not in self.kept_local and not self._local(top):
                self.warn(node, f"'{top}' is not installed in the program's .venv or the converting Python, so it "
                                f"was not converted; it stays a Python import{self.packages.hint(top)}")
            return None
        src, dst = hit
        self.local_modules.add((src, dst))
        rel = os.path.relpath(dst, os.path.dirname(os.path.abspath(self.out_path))).replace(os.sep, "/")
        self.pkg_sources[rel] = src
        return rel

    def _bind(self, name, value, ind):
        if self._binds_new(name):
            self.scope.declared.add(name)
            self.emit(f"{self._decl_kw(name)} {self.name(name)} = {value}", ind)
        else:
            self.emit(f"{self.name(name)} = {value}", ind)

    def s_Import(self, s, ind):
        for a in s.names:
            local = self._local(a.name)
            if local is None and (a.asname or "." not in a.name):
                local = self._package(a.name, s)
            if local is not None:
                alias = a.asname or a.name.split(".")[0]
                if a.asname or "." not in a.name:
                    if self._binds_new(alias):
                        self.scope.declared.add(alias)
                        self.emit(f'import "{local}" as {self.name(alias)}', ind)
                    else:
                        tmp = self.fresh("mod")
                        self.emit(f'import "{local}" as {tmp}', ind)
                        self.emit(f"{self.name(alias)} = {tmp}", ind)
                    continue
            if a.asname:
                self._bind(a.asname, f"pyimport({q(a.name)})", ind)
            elif "." in a.name:
                self.emit(f"pyimport({q(a.name)})", ind)
                root = a.name.split(".")[0]
                self._bind(root, f"pyimport({q(root)})", ind)
            else:
                self._bind(a.name, f"pyimport({q(a.name)})", ind)

    def s_ImportFrom(self, s, ind):
        if s.module == "__future__":
            return
        local = self._local(s.module, s.level) if (s.level or s.module) else None
        pkg_mode = False
        if local is None and not s.level and s.module:
            local = self._package(s.module, s)
            pkg_mode = local is not None
        if s.level and local is None and s.module is None:
            # from . import x  → each x is a module next to this one
            for a in s.names:
                path = self._local(a.name, s.level)
                if path is None:
                    raise Unsupported(f"relative import of '{a.name}' (no {a.name}.py next to this file)")
                tmp = self.fresh("mod")
                self.emit(f'import "{path}" as {tmp}', ind)
                self._bind(a.asname or a.name, tmp, ind)
            return
        if local is not None:
            tmp = self.fresh("mod")
            self.emit(f'import "{local}" as {tmp}', ind)
            for a in s.names:
                if a.name == "*":
                    for n in self._public_names_local(local):
                        self._bind(n, f"{tmp}.{n}", ind)
                    continue
                sub = (self._package(f"{s.module}.{a.name}") if pkg_mode
                       else self._local(f"{s.module}.{a.name}", s.level) if s.module else None)
                if sub is not None and not self._defines(local, a.name):
                    self.emit(f'import "{sub}" as {self.name(a.asname or a.name)}', ind)
                    self.scope.declared.add(a.asname or a.name)
                    continue
                self._bind(a.asname or a.name, f"{tmp}.{a.name}", ind)
            return
        if s.level:
            raise Unsupported(f"relative import from '{s.module}' (no such file next to this one)")
        for a in s.names:
            if a.name == "*":
                self._star_external(s.module, ind)
                continue
            self._bind(a.asname or a.name, f"pyfrom({q(s.module)}, {q(a.name)})", ind)

    def _star_external(self, module, ind):
        """`from module import *` for an installed module: the names it exports today, bound in one statement."""
        import importlib
        import importlib.util
        names = None
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        origin = getattr(spec, "origin", None) or ""
        if origin.endswith(".py"):
            # read the names from the source: converting never runs the module's code (`this`, `antigravity` …)
            names = _public_names_file(origin)
        if names is None:
            if spec is None or not (origin in ("built-in", "frozen") or origin.endswith((".so", ".pyd"))):
                raise Unsupported(f"'from {module} import *' — the module could not be found to list its names")
            try:
                mod = importlib.import_module(module)        # compiled modules: no Python code to run
            except Exception:
                raise Unsupported(f"'from {module} import *' — the module could not be loaded to list its names")
            names = getattr(mod, "__all__", None)
            if names is None:
                names = [n for n in dir(mod) if not n.startswith("_")]
        names = [n for n in names if n.isidentifier()]
        for i in range(0, len(names), 6):
            chunk = names[i:i + 6]
            self._declare_names([n for n in chunk if self._binds_new(n)], ind)
            targets = ", ".join(self.name(n) for n in chunk) + ("," if len(chunk) == 1 else "")
            self.emit(f"{targets} = pystar({q(module)}, {', '.join(q(n) for n in chunk)})", ind)

    def _public_names_local(self, apsi_rel):
        if apsi_rel in self.pkg_sources:
            return _public_names_file(self.pkg_sources[apsi_rel]) or []
        py = os.path.join(self.base_dir, apsi_rel[:-5] + ".py")
        if os.path.basename(py) == "__init__.py" or not os.path.exists(py):
            py = os.path.join(self.base_dir, apsi_rel[:-5], "__init__.py") if not os.path.exists(py) else py
        return _public_names_file(py) or []

    def _defines(self, apsi_rel, name):
        py = self.pkg_sources.get(apsi_rel) or os.path.join(self.base_dir, apsi_rel[:-5] + ".py")
        try:
            tree = ast.parse(open(py, encoding="utf-8").read())
        except (OSError, SyntaxError):
            return True
        for s in tree.body:
            if isinstance(s, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and s.name == name:
                return True
            if isinstance(s, ast.Assign) and any(name in _bound_names(t) for t in s.targets):
                return True
        return False

    # ---- functions and classes
    def params(self, args):
        out = []
        pos = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
        for i, (a, d) in enumerate(zip(pos, defaults)):
            out.append(self.name(a.arg) + (f"={self.ex(d)}" if d is not None else ""))
            if args.posonlyargs and i == len(args.posonlyargs) - 1:
                out.append("/")
        if args.vararg:
            out.append("*" + self.name(args.vararg.arg))
        elif args.kwonlyargs:
            out.append("*")
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            out.append(self.name(a.arg) + (f"={self.ex(d)}" if d is not None else ""))
        if args.kwarg:
            out.append("**" + self.name(args.kwarg.arg))
        return out

    def _param_names(self, args):
        names = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
        if args.vararg:
            names.append(args.vararg.arg)
        if args.kwarg:
            names.append(args.kwarg.arg)
        return names

    def s_FunctionDef(self, s, ind, kw="fn"):
        for d in s.decorator_list:
            self.emit("@" + self.ex(d), ind)
        self._declare_type_params(getattr(s, "type_params", None), ind)
        params = self.params(s.args)
        name = s.name
        if self.scope.kind != "class":
            self.scope.declared.add(name)
        self.emit(f"{kw} {self.name(name)}({', '.join(params)}):", ind)
        self.function_body(s.body, self._param_names(s.args), ind + 1)

    def s_AsyncFunctionDef(self, s, ind):
        self.s_FunctionDef(s, ind, "async fn")

    def function_body(self, body, params, ind):
        info = analyse(body, params)
        outer = self.scope
        sc = FnScope("function", outer, params)
        sc.globals = info.globals
        sc.nonlocals = info.nonlocals
        sc.locals = {n for n in info.bound if n not in info.globals and n not in info.nonlocals}
        sc.counts = info.count
        self.scope = sc
        saved_loop, self.in_loop = self.in_loop, 0
        try:
            start = 0
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                self._comments_before(body[0].lineno, ind)
                self.emit(self._docstring(body[0].value.value, ind), ind)
                start = 1
            for n in sorted(sc.locals & info.loads_before - set(params)):
                self.emit(f"var {self.name(n)}", ind)
                sc.declared.add(n)
            rest = body[start:]
            if rest:
                self.block(rest, ind)
            elif start == 0:
                self.emit("pass", ind)
        finally:
            self.scope = outer
            self.in_loop = saved_loop

    def s_ClassDef(self, s, ind):
        for d in s.decorator_list:
            self.emit("@" + self.ex(d), ind)
        tps = self._declare_type_params(getattr(s, "type_params", None), ind)
        args = [self.ex(b) for b in s.bases]
        if tps:
            args.append('pyfrom("typing", "Generic")[' + ", ".join(
                f'pyfrom("typing", "Unpack")[{self.name(t.name)}]' if isinstance(t, ast.TypeVarTuple)
                else self.name(t.name) for t in tps) + "]")
        args += [(f"{k.arg}={self.ex(k.value)}" if k.arg else "**" + self.ex(k.value)) for k in s.keywords]
        self.scope.declared.add(s.name)
        self.emit(f"class {self.name(s.name)}" + (f"({', '.join(args)})" if args else "") + ":", ind)
        outer = self.scope
        sc = FnScope("class", outer)
        sc.counts = analyse(s.body).count
        self.scope = sc
        try:
            body = s.body
            if not body:
                self.emit("pass", ind + 1)
            self.block(body, ind + 1)
        finally:
            self.scope = outer

    # -------------------------------------------------------------------------------- expressions
    def ex(self, n, prec=0, top=False):
        m = getattr(self, "e_" + type(n).__name__, None)
        if m is None:
            raise Unsupported(f"{type(n).__name__} expressions")
        text, p = m(n)
        return f"({text})" if p < prec else text

    def e_Constant(self, n):
        v = n.value
        if v is None:
            return "none", P_ATOM
        if v is True:
            return "true", P_ATOM
        if v is False:
            return "false", P_ATOM
        if v is Ellipsis:
            return "Ellipsis", P_ATOM
        if isinstance(v, str):
            return q(v), P_ATOM
        if isinstance(v, bytes):
            return qb(v), P_ATOM
        if isinstance(v, bool):
            return ("true" if v else "false"), P_ATOM
        if isinstance(v, int):
            return str(v), P_ATOM
        if isinstance(v, float):
            if v != v:
                return 'float("nan")', P_ATOM
            if v in (float("inf"), float("-inf")):
                return ('float("inf")' if v > 0 else '-float("inf")'), P_ATOM if v > 0 else P_UNARY
            r = repr(v)
            return r, P_ATOM
        if isinstance(v, complex):
            return f"complex({v.real!r}, {v.imag!r})", P_ATOM
        raise Unsupported(f"the constant {v!r}")

    def e_Name(self, n):
        return self.name(n.id), P_ATOM

    def e_Attribute(self, n):
        v = self.ex(n.value, P_ATOM)
        if v[:1].isdigit():
            v = f"({v})"
        return f"{v}.{n.attr}", P_ATOM

    def e_Subscript(self, n):
        return f"{self.ex(n.value, P_ATOM)}[{self.subscript(n.slice)}]", P_ATOM

    def subscript(self, s):
        if isinstance(s, ast.Slice):
            lo = self.ex(s.lower) if s.lower is not None else ""
            hi = self.ex(s.upper) if s.upper is not None else ""
            if s.step is not None:
                return f"{lo}:{hi}:{self.ex(s.step)}"
            return f"{lo}:{hi}"
        if isinstance(s, ast.Tuple) and s.elts:
            parts = [self.subscript(e) for e in s.elts]
            return ", ".join(parts) + ("," if len(parts) == 1 else "")
        return self.ex(s, top=True)

    def e_Starred(self, n):
        return "*" + self.ex(n.value, P_BOR), P_ATOM

    def e_BinOp(self, n):
        op, p = BINOP[type(n.op)]
        if op == "**":
            return f"{self.ex(n.left, P_POW + 1)} ** {self.ex(n.right, P_UNARY)}", P_POW
        return f"{self.ex(n.left, p)} {op} {self.ex(n.right, p + 1)}", p

    def e_UnaryOp(self, n):
        if isinstance(n.op, ast.Not):
            return f"not {self.ex(n.operand, P_NOT)}", P_NOT
        op = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}[type(n.op)]
        return f"{op}{self.ex(n.operand, P_UNARY)}", P_UNARY

    def e_BoolOp(self, n):
        op = "and" if isinstance(n.op, ast.And) else "or"
        p = P_AND if op == "and" else P_OR
        return f" {op} ".join(self.ex(v, p + 1) for v in n.values), p

    def e_Compare(self, n):
        parts = [self.ex(n.left, P_CMP + 1)]
        for op, c in zip(n.ops, n.comparators):
            parts.append(CMPOP[type(op)])
            parts.append(self.ex(c, P_CMP + 1))
        return " ".join(parts), P_CMP

    def e_IfExp(self, n):
        return f"{self.ex(n.body, P_OR)} if {self.ex(n.test, P_OR)} else {self.ex(n.orelse, P_IF)}", P_IF

    def e_Lambda(self, n):
        return f"fn({', '.join(self.params(n.args))}) => {self.ex(n.body, P_LAMBDA)}", P_LAMBDA

    def e_NamedExpr(self, n):
        return f"({self.name(n.target.id)} := {self.ex(n.value, P_LAMBDA)})", P_ATOM

    def e_Await(self, n):
        return f"await {self.ex(n.value, P_UNARY)}", P_UNARY

    def e_Yield(self, n):
        return f"({self.yield_text(n)})", P_ATOM

    e_YieldFrom = e_Yield

    def e_Call(self, n):
        args = [self.ex(a, P_LAMBDA) if not isinstance(a, ast.GeneratorExp) or len(n.args) + len(n.keywords) > 1
                else self._comp_inner(a) for a in n.args]
        for k in n.keywords:
            args.append(f"{k.arg}={self.ex(k.value, P_LAMBDA)}" if k.arg else "**" + self.ex(k.value, P_BOR))
        return f"{self.ex(n.func, P_ATOM)}({', '.join(args)})", P_ATOM

    def e_List(self, n):
        return "[" + ", ".join(self.ex(e, P_LAMBDA) for e in n.elts) + "]", P_ATOM

    def e_Tuple(self, n):
        if not n.elts:
            return "()", P_ATOM
        inner = ", ".join(self.ex(e, P_LAMBDA) for e in n.elts)
        return "(" + inner + ("," if len(n.elts) == 1 else "") + ")", P_ATOM

    def e_Set(self, n):
        return "{" + ", ".join(self.ex(e, P_LAMBDA) for e in n.elts) + "}", P_ATOM

    def e_Dict(self, n):
        parts = []
        for k, v in zip(n.keys, n.values):
            if k is None:
                parts.append("**" + self.ex(v, P_BOR))
            else:
                parts.append(f"{self.ex(k, P_LAMBDA)}: {self.ex(v, P_LAMBDA)}")
        return "{" + ", ".join(parts) + "}", P_ATOM

    def comp_clauses(self, gens):
        out = []
        for g in gens:
            tgt = self._loop_target(g.target, allow_expr=True)
            if tgt.startswith("(") and tgt.endswith(")") and ", " in tgt:
                tgt = tgt[1:-1]
            part = f"{'async for' if g.is_async else 'for'} {tgt} in {self.ex(g.iter, P_OR)}"
            for c in g.ifs:
                part += f" if {self.ex(c, P_OR)}"
            out.append(part)
        return " ".join(out)

    def _comp_inner(self, n):
        return f"{self.ex(n.elt, P_LAMBDA)} {self.comp_clauses(n.generators)}"

    def e_ListComp(self, n):
        return f"[{self._comp_inner(n)}]", P_ATOM

    def e_SetComp(self, n):
        return "{" + self._comp_inner(n) + "}", P_ATOM

    def e_GeneratorExp(self, n):
        return f"({self._comp_inner(n)})", P_ATOM

    def e_DictComp(self, n):
        return "{" + f"{self.ex(n.key, P_LAMBDA)}: {self.ex(n.value, P_LAMBDA)} " + \
            self.comp_clauses(n.generators) + "}", P_ATOM

    def e_Slice(self, n):
        return self.subscript(n), P_ATOM

    def e_JoinedStr(self, n):
        """An f-string: an Apsi f-string when every piece can be written inside one, else format() calls."""
        try:
            return self._fstring(n), P_ATOM
        except Unsupported:
            pass
        parts = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                parts.append(q(v.value))
            else:
                parts.append(self._format_call(v))
        if not parts:
            return '""', P_ATOM
        if len(parts) == 1:
            return (parts[0] if parts[0].startswith("format(") else f"str({parts[0]})"), P_ATOM
        return " + ".join(parts), P_ADD

    def _format_call(self, v):
        val = self.ex(v.value, P_LAMBDA)
        if v.conversion != -1:
            val = {115: "str", 114: "repr", 97: "ascii"}[v.conversion] + f"({val})"
        if v.format_spec is None:
            return f"format({val})"
        spec = self.ex(v.format_spec) if not (len(v.format_spec.values) == 1 and isinstance(
            v.format_spec.values[0], ast.Constant)) else q(v.format_spec.values[0].value)
        return f"format({val}, {spec})"

    def _fstring(self, n):
        out = ['f"']
        for v in n.values:
            if isinstance(v, ast.Constant):
                out.append(q(v.value)[1:-1].replace("{", "{{").replace("}", "}}"))
                continue
            expr = self.ex(v.value, P_LAMBDA)
            if any(c in expr for c in '"{}\\\n') or "!" in expr and "!=" not in expr or ":" in expr \
                    or expr.rstrip().endswith("="):
                raise Unsupported("f-string piece")
            piece = "{" + expr
            if v.conversion != -1:
                piece += "!" + chr(v.conversion)
            if v.format_spec is not None:
                spec = []
                for sv in v.format_spec.values:
                    if isinstance(sv, ast.Constant):
                        if any(c in sv.value for c in '{}"\\\n'):
                            raise Unsupported("f-string spec")
                        spec.append(sv.value)
                    else:
                        inner = self.ex(sv.value, P_LAMBDA)
                        if any(c in inner for c in '"{}\\:!') or sv.conversion != -1 or sv.format_spec is not None:
                            raise Unsupported("f-string spec")
                        spec.append("{" + inner + "}")
                piece += ":" + "".join(spec)
            out.append(piece + "}")
        out.append('"')
        return "".join(out)

    def e_FormattedValue(self, n):
        return self._format_call(n), P_ATOM


def _public_names_file(py):
    """The names `from module import *` binds, read from a module's source without running it."""
    try:
        with open(py, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    names = []
    for st in tree.body:
        if isinstance(st, (ast.Assign, ast.AugAssign)) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in (st.targets if isinstance(st, ast.Assign)
                                                                           else [st.target])):
            try:
                return [n for n in ast.literal_eval(st.value) if isinstance(n, str)]
            except ValueError:
                return None                  # computed __all__: let the caller decide
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(st.name)
        elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                names += _bound_names(t)
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            names += [(a.asname or a.name).split(".")[0] for a in st.names if a.name != "*"]
    return [n for n in dict.fromkeys(names) if not n.startswith("_")]


def _pattern_names(p):
    out = []
    for node in ast.walk(p):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            out.append(node.name)
        if isinstance(node, ast.MatchMapping) and node.rest:
            out.append(node.rest)
    return out


def convert_source(source, filename="<python>", base_dir=None, packages=None, out_path=None, vendored=False,
                   keep_local=None):
    """Convert Python source; returns (apsi_text, warnings, local_modules). local_modules holds the paths of
    modules next to the program, and (source, destination) pairs for converted libraries. keep_local: a function
    path -> bool; a module next to the program for which it says true stays a Python import (convert --only)."""
    c = Converter(source, filename, base_dir, packages, out_path, vendored)
    c.keep_local = keep_local
    text = c.convert()
    if c.kept_local:
        c.warnings.append((1, "modules of the program kept as Python imports (--only): "
                              + ", ".join(sorted(c.kept_local)) + "; run with --allow-python"))
    return text, c.warnings, c.local_modules


# ------------------------------------------------------------------------------------------------ installed libraries
# Libraries written in Python are converted to Apsi with the program (into apsi_packages/ next to it), so the result
# runs without the Python environment it came from. Libraries with compiled code (C, C++, Rust: numpy, pyarrow,
# torch …) cannot be converted; they stay pyimport() and the conversion notes say so, with Apsi's own tool for the
# job when there is one.
_EQUIVALENT = {
    "numpy": "Apsi tensors (`apsi convert --idiomatic` maps small numeric scripts onto them)",
    "torch": "Apsi layers and fit() (`apsi convert` turns torch.nn models into Apsi models)",
    "tensorflow": "Apsi layers and fit()",
    "jax": "Apsi tensors and grad()",
    "pandas": "`use dataframe`",
    "polars": "`use dataframe`",
    "pyarrow": "`use dataframe` (CSV, JSON, Parquet)",
    "datasets": "`use hub` (hub.download(repo, file, kind=\"dataset\")) and `use data`",
    "transformers": "`use hub` (hub.model(...): models from Hugging Face without Python)",
    "huggingface_hub": "`use hub` (hub.download, hub.model: Hugging Face without Python)",
    "hf_transfer": "`use hub` (parallel, resumable downloads)",
    "safetensors": "`use models` (safetensors files)",
    "scipy": "Apsi's `sci` built-ins (integrate, minimize, fft, curve_fit)",
    "sklearn": "Apsi layers and `use data`",
    "requests": "`use web` (http_get / http_post_json)",
    "aiohttp": "`use web`",
    "PIL": "`use video` / image functions",
    "cv2": "`use video`",
}


class Packages:
    """Finds the installed libraries a program imports: in the program's virtual environment (any Python version)
    and in the Python running the conversion."""

    def __init__(self, program_dir, vendor_dir):
        import sys
        import sysconfig
        self.vendor_dir = vendor_dir
        roots = []
        for cand in (".venv", "venv", "env"):
            v = os.path.join(program_dir, cand)
            if os.path.isfile(os.path.join(v, "pyvenv.cfg")):
                import glob
                roots += glob.glob(os.path.join(v, "Lib", "site-packages"))
                roots += glob.glob(os.path.join(v, "lib", "python3*", "site-packages"))
        roots += [p for p in sys.path if p.rstrip("\\/").endswith(("site-packages", "dist-packages"))]
        self.roots = [r for i, r in enumerate(roots) if os.path.isdir(r) and r not in roots[:i]]
        std = getattr(sys, "stdlib_module_names", None) or set()
        # names of newer Python versions' standard library count as standard too (a 3.14 program on Apsi's 3.12)
        self.stdlib = set(std) | {"__future__", "pycompat", "annotationlib", "compression", "_typeshed", "tomllib",
                                  "zoneinfo", "graphlib"}
        self.stdlib_dir = os.path.normcase(sysconfig.get_paths().get("stdlib", ""))
        self._kind = {}
        self.used = set()                   # top-level libraries converted

    def copy_data(self):
        """The data files of the converted libraries (certificates, tables, templates …) and their .dist-info
        (versions for importlib.metadata) next to the converted modules."""
        import shutil
        n = 0
        for top in sorted(self.used):
            root, path = self._find(top)
            if os.path.isdir(path):
                for d, dirs, files in os.walk(path):
                    dirs[:] = [x for x in dirs if x not in ("__pycache__", "tests", "test")]
                    for f in files:
                        if f.endswith((".py", ".pyc", ".pyo", ".pyi")):
                            continue
                        src = os.path.join(d, f)
                        dst = os.path.join(self.vendor_dir, os.path.relpath(src, root))
                        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(src, dst)
                            n += 1
            for entry in os.listdir(root):
                if entry.endswith(".dist-info") and entry.split("-")[0].lower().replace("_", "-") == \
                        top.lower().replace("_", "-"):
                    dst = os.path.join(self.vendor_dir, entry)
                    if not os.path.exists(dst):
                        shutil.copytree(os.path.join(root, entry), dst, ignore=shutil.ignore_patterns("RECORD"))
        return n

    def _find(self, top):
        """(root, path of the package folder or module file) for an importable top-level name, or None."""
        for r in self.roots:
            d = os.path.join(r, top)
            if os.path.isfile(os.path.join(d, "__init__.py")):
                return r, d
            f = os.path.join(r, top + ".py")
            if os.path.isfile(f):
                return r, f
            if os.path.isdir(d) and any(n.endswith(".py") for n in os.listdir(d)):   # namespace package
                return r, d
        return None

    def kind(self, top):
        """'pure', 'compiled', 'stdlib' or None (not installed where the program's code could find it)."""
        if top in self._kind:
            return self._kind[top]
        k = None
        if top in self.stdlib:
            k = "stdlib"
        else:
            hit = self._find(top)
            if hit is None:
                for r in self.roots:                                    # a compiled single module (top.pyd / .so)
                    if any(n.startswith(top + ".") and n.endswith((".pyd", ".so")) for n in os.listdir(r)):
                        k = "compiled"
                        break
            elif os.path.isfile(hit[1]):
                k = "pure"
            else:
                k = "pure"
                for _root, _dirs, files in os.walk(hit[1]):
                    if any(f.endswith((".pyd", ".so", ".dll", ".dylib")) for f in files):
                        k = "compiled"
                        break
        self._kind[top] = k
        return k

    def hint(self, top):
        eq = _EQUIVALENT.get(top)
        return f"; Apsi's own tool for this: {eq}" if eq else ""

    def resolve(self, module):
        """(source .py, destination .apsi) for a module of a pure-Python library, or None."""
        top = module.split(".")[0]
        if self.kind(top) != "pure":
            return None
        root, path = self._find(top)
        self.used.add(top)
        rest = module.split(".")[1:]
        base = os.path.join(root, *([top] + rest))
        for cand in (base + ".py", os.path.join(base, "__init__.py")):
            if os.path.isfile(cand):
                rel = os.path.relpath(cand, root)
                return cand, os.path.join(self.vendor_dir, os.path.splitext(rel)[0] + ".apsi")
        return None
