"""`apsi convert` — translate Python programs into Apsi.

    apsi convert script.py                 writes script.apsi next to it and checks it
    apsi convert src/ -o apsi_src/         a whole folder (keeps the folder layout)
    apsi convert script.py --stdout        print the result instead of writing a file
    apsi convert script.py --run           convert, check and run

What is translated
  * the whole language core: variables (let/var chosen automatically, block-scoped names hoisted),
    functions (defaults, keyword arguments, *args / **kwargs, nested functions, closures with nonlocal,
    lambdas, decorators such as @lru_cache → @memo), classes (fields, methods, __init__, inheritance inside
    the file, class attributes, @dataclass with field(default_factory=…), @staticmethod, special methods
    __add__/__eq__/__lt__/__len__/__getitem__/__str__ … → struct methods that operators use), generators
    (yield / yield from: lazy, endless ones work), if/elif/else, for/while with else, break/continue,
    try/except/else/finally and raise (same error kinds, own exception classes), comprehensions (several
    for clauses), f-strings, % and .format, slicing, unpacking (also nested and *rest), assert,
    with open(...), `if __name__ == "__main__":`, bitwise and set operators, docstrings and comments
  * libraries, mapped onto Apsi's own built-ins and modules: numpy (arrays, maths, random, linalg, fft),
    math, random, statistics, json, time, os.path, sys.argv/exit, PyTorch tensors and torch.nn layers
    (→ `use layers`), torch.nn.functional and scipy.integrate/optimize basics
  * pure standard modules (collections, heapq, itertools, functools, bisect, re, string, copy, datetime,
    fractions, decimal …) are used as they are, without --allow-python
  * whole PyTorch programs (nn.Module classes, losses, optimizers, data loaders, training loops) become native
    Apsi that runs without Python or PyTorch — see convert_torch.py
  * every other library still works: it is loaded with pyimport() (the program then needs
    --allow-python) and the converter says so

Anything that cannot be translated exactly is kept as close as possible and reported with its line
number, so you can review just those places. `apsi check` runs on the result automatically.
"""
import ast
import io
import os
import re
import sys
import tokenize

APSI_RESERVED = {"let", "var", "fn", "struct", "use", "evolve", "none", "true", "false", "test", "over",
                 "step", "using", "tol", "record", "every", "into"}
TYPE_MAP = {"int": "Int", "float": "Float", "str": "Str", "bool": "Bool", "list": "List", "dict": "Dict",
            "tuple": "Tuple", "List": "List", "Dict": "Dict", "Tuple": "Tuple", "Any": "Any",
            "ndarray": "Tensor", "Tensor": "Tensor", "Callable": "Fn"}
DTYPES = {"float32": "f32", "float64": "f64", "float16": "f16", "int32": "i32", "int64": "i64",
          "float": "f64", "int": "i64", "bfloat16": "bf16", "half": "f16", "double": "f64", "long": "i64"}
ISINSTANCE = {"int": '"Int"', "float": '"Float"', "str": '"Str"', "bool": '"Bool"', "list": '"List"',
              "dict": '"Dict"', "tuple": '"Tuple"'}

# python dotted name -> Apsi name (simple renames); functions get the same arguments
SIMPLE = {}
for _np in ("numpy", "torch"):
    SIMPLE.update({f"{_np}.{k}": v for k, v in {
        "array": "tensor", "asarray": "tensor", "tensor": "tensor", "as_tensor": "tensor", "from_numpy": "tensor",
        "zeros": "zeros", "ones": "ones", "full": "full", "eye": "eye", "identity": "eye",
        "zeros_like": "zeros_like", "ones_like": "ones_like", "arange": "arange", "linspace": "linspace",
        "sum": "sum", "mean": "mean", "std": "std", "var": "var", "prod": "prod", "max": "max", "amax": "max",
        "min": "min", "amin": "min", "argmax": "argmax", "argmin": "argmin", "cumsum": "cumsum",
        "abs": "abs", "absolute": "abs", "sqrt": "sqrt", "cbrt": "cbrt", "exp": "exp", "log": "log",
        "log10": "log10", "log2": "log2", "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin",
        "arccos": "acos", "arctan": "atan", "arctan2": "atan2", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "floor": "floor", "ceil": "ceil",
        "round": "round", "around": "round", "clip": "clip", "clamp": "clip", "where": "where", "dot": "dot",
        "matmul": "matmul", "mm": "matmul", "outer": "outer", "cross": "cross", "transpose": "transpose",
        "concatenate": "concat", "cat": "concat", "stack": "stack", "meshgrid": "meshgrid", "sort": "sort",
        "roll": "roll", "einsum": "einsum", "flatten": "flatten", "unique": "unique", "isclose": "approx",
        "allclose": "approx", "sigmoid": "sigmoid", "relu": "relu", "softmax": "softmax",
        "log_softmax": "log_softmax", "pi": "pi", "e": "e", "inf": "inf", "nan": "nan", "shape": "shape",
        "ndim": "ndim", "reshape": "reshape", "norm": "norm",
        "random.randn": "randn", "random.rand": "rand", "random.seed": "seed", "random.permutation": "permutation",
        "random.randint": "randint", "manual_seed": "seed", "randn": "randn", "rand": "rand", "randint": "randint",
        "randperm": "permutation",
        "linalg.inv": "linalg.inv", "linalg.det": "linalg.det", "linalg.solve": "linalg.solve",
        "linalg.eig": "linalg.eig", "linalg.eigh": "linalg.eigh", "linalg.svd": "linalg.svd",
        "linalg.norm": "norm", "trace": "linalg.trace",
        "fft.fft": "sci.fft", "fft.ifft": "sci.ifft", "fft.rfft": "sci.rfft", "fft.irfft": "sci.irfft",
        "fft.fftfreq": "sci.fftfreq", "fft.rfftfreq": "sci.rfftfreq", "trapz": "sci.trapz",
        "trapezoid": "sci.trapz", "interp": "sci.interp", "real": "sci.real", "imag": "sci.imag",
        "conj": "sci.conj",
    }.items()})
SIMPLE.update({
    "math.sqrt": "sqrt", "math.exp": "exp", "math.log10": "log10", "math.log2": "log2", "math.sin": "sin",
    "math.cos": "cos", "math.tan": "tan", "math.asin": "asin", "math.acos": "acos", "math.atan": "atan",
    "math.atan2": "atan2", "math.sinh": "sinh", "math.cosh": "cosh", "math.tanh": "tanh", "math.floor": "floor",
    "math.ceil": "ceil", "math.fabs": "abs", "math.pi": "pi", "math.e": "e", "math.tau": "tau", "math.inf": "inf",
    "math.nan": "nan", "math.fsum": "fsum", "math.pow": "pow", "math.isclose": "approx", "math.trunc": "int",
    "random.random": "rand", "random.seed": "seed",
    "json.loads": "json_parse", "json.dumps": "json_str",
    "time.time": "clock", "time.perf_counter": "clock", "time.monotonic": "clock", "time.process_time": "clock",
    "os.path.exists": "exists", "os.path.isfile": "exists", "sys.exit": "exit", "exit": "exit", "quit": "exit",
    "statistics.mean": "mean", "statistics.fmean": "mean", "statistics.pstdev": "std",
    "functools.reduce": None,
    "torch.nn.functional.relu": "relu", "torch.nn.functional.gelu": "gelu", "torch.nn.functional.sigmoid": "sigmoid",
    "torch.nn.functional.tanh": "tanh", "torch.nn.functional.softmax": "softmax",
    "torch.nn.functional.log_softmax": "log_softmax", "torch.nn.functional.mse_loss": "nn.mse",
    "torch.nn.functional.cross_entropy": "nn.cross_entropy", "torch.nn.functional.dropout": "nn.dropout",
    "torch.nn.functional.layer_norm": "nn.layer_norm", "torch.sigmoid": "sigmoid",
    "scipy.integrate.quad": None, "scipy.optimize.minimize": None,
})
SIMPLE = {k: v for k, v in SIMPLE.items() if v is not None}
NATIVE_MODULES = {"numpy", "math", "random", "json", "time", "os", "os.path", "sys", "statistics", "torch",
                  "torch.nn", "torch.nn.functional", "functools", "typing", "dataclasses", "__future__",
                  "collections.abc", "abc", "scipy.integrate", "scipy.optimize"}
TORCH_LAYERS = {"Linear": "layers.linear", "ReLU": None, "GELU": None, "Tanh": None, "Sigmoid": None,
                "SiLU": None, "Softmax": None, "Dropout": "layers.dropout_layer", "LayerNorm": "layers.layer_norm_layer",
                "Embedding": "layers.embedding", "Conv2d": "layers.conv2d_layer", "MaxPool2d": "layers.max_pool",
                "Flatten": "layers.flatten_layer", "LSTM": "layers.lstm_layer", "GRU": "layers.gru_layer",
                "Sequential": "layers.sequential", "MultiheadAttention": "layers.multi_head_attention",
                "BatchNorm1d": "layers.batch_norm_layer", "BatchNorm2d": "layers.batch_norm_layer"}
TORCH_ACTS = {"ReLU": "relu", "GELU": "gelu", "Tanh": "tanh", "Sigmoid": "sigmoid", "SiLU": "silu",
              "Softmax": "softmax", "LeakyReLU": "leaky_relu", "ELU": "elu", "Identity": "identity"}
OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
       ast.Pow: "**", ast.MatMult: "@"}
PREC = {"lambda": 1, "ifexp": 2, "or": 3, "and": 4, "not": 5, "cmp": 6, "+": 8, "-": 8, "*": 9, "/": 9,
        "//": 9, "%": 9, "@": 9, "unary": 10, "**": 12, "atom": 20}
CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.In: "in",
       ast.NotIn: "not in", ast.Is: "==", ast.IsNot: "!="}


# Python special methods -> the Apsi struct method names that operators and built-ins use (see values.py)
DUNDER = {"__str__": "to_str", "__repr__": "to_str", "__add__": "add", "__radd__": "radd", "__sub__": "sub",
          "__rsub__": "rsub", "__mul__": "mul", "__rmul__": "rmul", "__truediv__": "div", "__rtruediv__": "rdiv",
          "__floordiv__": "floordiv", "__mod__": "mod", "__pow__": "pow", "__matmul__": "matmul", "__neg__": "neg",
          "__eq__": "eq", "__lt__": "lt", "__le__": "le", "__gt__": "gt", "__ge__": "ge", "__len__": "len",
          "__getitem__": "get", "__setitem__": "set", "__contains__": "contains", "__iter__": "iter",
          "__call__": "call", "__hash__": "hash", "__bool__": "bool"}


def full_is_lib(full):
    return bool(full) and "." in full


class Unsupported(Exception):
    pass


def _q(s):
    """An Apsi string literal."""
    out = []
    for ch in s:
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
        elif ch == "\0":
            out.append("\\0")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


# ----------------------------------------------------------------------------------- scope analysis
class ScopeInfo:
    def __init__(self, params=()):
        self.params = set(params)
        self.assign = {}          # name -> number of assignments
        self.nested = set()       # assigned inside a nested block
        self.aug = set()
        self.globals = set()
        self.declared = set(params)
        self.reassigned_params = set()


def _targets(t):
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)
    elif isinstance(t, ast.Starred):
        yield from _targets(t.value)


def analyse(body, params=()):
    info = ScopeInfo(params)

    first_depth = {}

    def note(name, depth, aug=False):
        info.assign[name] = info.assign.get(name, 0) + 1
        first_depth.setdefault(name, depth)
        if first_depth[name] > 0:          # first assigned inside a block: must be declared before it
            info.nested.add(name)
        if aug:
            info.aug.add(name)

    def walk(stmts, depth):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                note(s.name, depth)
                continue
            if isinstance(s, (ast.Global, ast.Nonlocal)):
                info.globals.update(s.names)
                continue
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    for n in _targets(t):
                        note(n, depth)
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                for n in _targets(s.target):
                    note(n, depth)
            elif isinstance(s, ast.AugAssign):
                for n in _targets(s.target):
                    note(n, depth, aug=True)
            elif isinstance(s, ast.With):
                for it in s.items:
                    if it.optional_vars is not None and not _is_open(it.context_expr):
                        for n in _targets(it.optional_vars):
                            note(n, depth + 1)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(s, field, None)
                if sub and isinstance(sub, list) and not isinstance(s, (ast.FunctionDef, ast.ClassDef)):
                    walk(sub, depth + 1)
            if isinstance(s, ast.Try):
                for h in s.handlers:
                    walk(h.body, depth + 1)
            if isinstance(s, ast.If) and _is_main_guard(s):
                pass
    # the body of `if __name__ == "__main__":` behaves like top level
    flat = []
    for s in body:
        if isinstance(s, ast.If) and _is_main_guard(s):
            flat.extend(s.body)
        else:
            flat.append(s)
    walk(flat, 0)
    info.reassigned_params = {p for p in info.params if p in info.assign}
    return info


def _is_main_guard(s):
    t = s.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
            and t.comparators[0].value == "__main__")


def _is_open(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"


def _has_yield(fn):
    for n in ast.walk(fn):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            # ignore yields of nested functions
            return True
    return False


# ----------------------------------------------------------------------------------- converter
class Converter:
    def __init__(self, source, filename="<python>"):
        self.source = source
        self.filename = filename
        self.tree = ast.parse(source, filename)
        self.lines = []
        self.warnings = []                     # (line, message)
        self.uses = set()
        self.pyimports = {}                    # local alias -> python module
        self.alias = {}                        # local name -> dotted python module/object it refers to
        self.classes = {}                      # name -> dict(info)
        self.scopes = []
        self.tmp = 0
        self.files = {}                        # with-open file variable -> (path expr, mode)
        self.comments = self._read_comments()
        self.used_comments = set()
        self.gen_stack = []                    # generator result variables
        self.loop_flags = []                   # per enclosing loop: the `broke` flag of a for/while … else
        self.cur_class = None
        self.pre = []                          # statements to emit before the current statement
        self.lazy_py = {}                      # python modules imported but only loaded when used
        self.in_comprehension = False
        # PyTorch programs: models, losses, optimizers and training steps become native Apsi first
        from .convert_torch import rewrite, APSI
        self.torch_used, _ = rewrite(self.tree, self.warn)
        self.alias[APSI] = APSI

    # ------------------------------------------------------------ comments
    def _read_comments(self):
        out = {}
        try:
            for tok in tokenize.generate_tokens(io.StringIO(self.source).readline):
                if tok.type == tokenize.COMMENT:
                    line = tok.start[0]
                    before = self.source.splitlines()[line - 1][:tok.start[1]]
                    out[line] = (tok.string[1:].strip(), not before.strip())
        except (tokenize.TokenError, IndentationError):
            pass
        return out

    def _flush_comments(self, upto, ind):
        for line in sorted(self.comments):
            if line >= upto:
                break
            if line in self.used_comments:
                continue
            text, own_line = self.comments[line]
            self.used_comments.add(line)
            if own_line and not text.startswith("!"):
                self.emit(f"# {text}" if text else "#", ind)

    def warn(self, node, msg):
        line = getattr(node, "lineno", 0)
        if (line, msg) not in self.warnings:
            self.warnings.append((line, msg))

    def emit(self, text, ind):
        self.lines.append("    " * ind + text)

    def new_tmp(self, base="t"):
        self.tmp += 1
        return f"_{base}{self.tmp}"

    # ------------------------------------------------------------ entry
    def convert(self):
        self._collect_imports(self.tree.body)
        self._collect_classes(self.tree.body)
        body_lines = []
        saved, self.lines = self.lines, body_lines
        self.block(self.tree.body, 0, analyse(self.tree.body), top=True)
        self._flush_comments(10 ** 9, 0)
        self.lines = saved
        head = [f"# Converted from {os.path.basename(self.filename)} by `apsi convert`."]
        if self.warnings:
            head.append(f"# {len(self.warnings)} place(s) need a look — see the NOTE comments below.")
        for u in sorted(self.uses):
            head.append(f"use {u} as {u}" if u in ("linalg", "sci", "layers", "parallel", "data", "nn", "models", "torch")
                        else f"use {u}")
        for alias, mod in sorted(self.pyimports.items()):
            from .interop import PURE_MODULES
            head.append(f'let {alias} = pyimport("{mod}")' + ("" if mod in PURE_MODULES else "   # needs --allow-python"))
        if len(head) > 1 or self.uses or self.pyimports:
            head.append("")
        return "\n".join(head + body_lines).rstrip() + "\n"

    # ------------------------------------------------------------ imports
    def _collect_imports(self, body):
        for s in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(s, ast.Import):
                for a in s.names:
                    local = a.asname or a.name.split(".")[0]
                    self.alias[local] = a.name if a.asname else a.name.split(".")[0]
                    if not self._native(a.name):
                        self.lazy_py[local] = (a.name if a.asname else a.name.split(".")[0], s)
            elif isinstance(s, ast.ImportFrom) and s.module:
                for a in s.names:
                    if a.name == "*":
                        self.warn(s, f"'from {s.module} import *' — names are resolved one by one")
                        continue
                    local = a.asname or a.name
                    full = f"{s.module}.{a.name}"
                    self.alias[local] = full
                    if not self._native(s.module) and not self._native(full):
                        mod_alias = self._safe(s.module.replace(".", "_"))
                        self.alias[local] = f"@py:{mod_alias}.{a.name}"
                        self.lazy_py[mod_alias] = (s.module, s)

    def _native(self, mod):
        return mod in NATIVE_MODULES or mod.split(".")[0] in ("numpy", "math", "random", "json", "time",
                                                              "typing", "dataclasses", "functools", "__future__")

    def _safe(self, name):
        return name + "_" if name in APSI_RESERVED else name

    # ------------------------------------------------------------ classes
    def _collect_classes(self, body):
        for s in body:
            if isinstance(s, ast.ClassDef):
                fields, methods, init, statics = [], {}, None, set()
                is_dc = any(dotted(d.func if isinstance(d, ast.Call) else d) in ("dataclass", "dataclasses.dataclass")
                            for d in s.decorator_list)
                for item in s.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            init = item
                        methods[item.name] = item
                        if any(dotted(d) == "staticmethod" for d in item.decorator_list):
                            statics.add(item.name)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        fields.append((item.target.id, item.annotation, item.value))
                    elif isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                fields.append((t.id, None, item.value))
                if init is not None:
                    for n in ast.walk(init):
                        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                            ts = n.targets if isinstance(n, ast.Assign) else [n.target]
                            for t in ts:
                                for sub in ast.walk(t):
                                    if (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                                            and sub.value.id == "self" and sub.attr not in [f[0] for f in fields]):
                                        fields.append((sub.attr, None, None))
                bases = [dotted(b) for b in s.bases]
                # class attributes written or read as ClassName.attr become a module-level variable ClassName_attr
                assigned = {t.id: item.value for item in s.body if isinstance(item, ast.Assign)
                            for t in item.targets if isinstance(t, ast.Name)}
                assigned.update({item.target.id: item.value for item in s.body if isinstance(item, ast.AnnAssign)
                                 and isinstance(item.target, ast.Name) and item.value is not None and not is_dc})
                classvars = {}
                for node in ast.walk(self.tree):
                    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                            and node.value.id == s.name and node.attr in assigned):
                        classvars[node.attr] = assigned[node.attr]
                from .exceptions import KINDS
                is_exc = any(b.split(".")[-1] in KINDS or b in ("Exception", "BaseException")
                             or self.classes.get(b, {}).get("exception") for b in bases)
                if is_exc and "message" not in [f[0] for f in fields]:
                    fields.insert(0, ("message", None, ast.Constant(value="")))
                self.classes[s.name] = dict(node=s, fields=fields, methods=methods, init=init, dataclass=is_dc,
                                            bases=bases, statics=statics, classvars=classvars, exception=is_exc)

    def _subclasses(self, name):
        out = []
        for k, c in self.classes.items():
            if name in c["bases"] and k != name:
                out += [k] + self._subclasses(k)
        return out

    def _class_fields(self, name, seen=None):
        c = self.classes[name]
        out = []
        for b in c["bases"]:
            if b in self.classes and b != name:
                out += self._class_fields(b)
        names = [f[0] for f in out]
        for f in c["fields"]:
            if f[0] not in names:
                out.append(f)
                names.append(f[0])
        return out

    # ------------------------------------------------------------ statements
    def block(self, stmts, ind, scope, top=False, prologue=()):
        self.scopes.append(scope)
        for line in prologue:
            self.emit(line, ind)
        # hoist names that are first assigned inside a nested block
        hoisted = sorted(n for n in scope.nested if n not in scope.params and n not in scope.globals
                         and n not in self.classes and not self._is_outer_declared(n))
        for n in hoisted:
            self.emit(f"var {self._safe(n)} = none", ind)
            scope.declared.add(n)
        for p in sorted(scope.reassigned_params):
            self.emit(f"var {self._safe(p)} = {self._safe(p)}", ind)
        count = 0
        block_start = len(self.lines)
        for i, s in enumerate(stmts):
            if i == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                for line in s.value.value.strip().splitlines():
                    self.emit(f"# {line.strip()}".rstrip(), ind)
                count += 1
                continue
            if ind == 0 and isinstance(s, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and self.lines \
                    and self.lines[-1].strip():
                self.emit("", 0)
            self._flush_comments(s.lineno, ind)
            before = len(self.lines)
            try:
                self.stmt(s, ind)
            except Unsupported as e:
                self.warn(s, str(e))
                self.emit(f"# NOTE: not converted — {e}", ind)
                for line in ast.get_source_segment(self.source, s).splitlines() if ast.get_source_segment(self.source, s) else []:
                    self.emit(f"#     {line}", ind)
            # inline comment of the statement's first line
            c = self.comments.get(s.lineno)
            if c and not c[1] and s.lineno not in self.used_comments and len(self.lines) > before:
                self.used_comments.add(s.lineno)
                self.lines[before] += f"  # {c[0]}"
            count += 1
        if not top and not any(l.strip() and not l.strip().startswith("#") for l in self.lines[block_start:]):
            self.emit("pass", ind)
        self.scopes.pop()

    def _is_outer_declared(self, name):
        return False

    @property
    def scope(self):
        return self.scopes[-1]

    def body(self, stmts, ind, scope=None):
        """A nested block (same scope); guarantees at least one statement."""
        start = len(self.lines)
        if scope is not None:
            self.block(stmts, ind, scope)
        else:
            for s in stmts:
                self._flush_comments(s.lineno, ind)
                before = len(self.lines)
                try:
                    self.stmt(s, ind)
                except Unsupported as e:
                    self.warn(s, str(e))
                    self.emit(f"# NOTE: not converted — {e}", ind)
                    seg = ast.get_source_segment(self.source, s) or ""
                    for line in seg.splitlines():
                        self.emit(f"#     {line}", ind)
                c = self.comments.get(s.lineno)
                if c and not c[1] and s.lineno not in self.used_comments and len(self.lines) > before:
                    self.used_comments.add(s.lineno)
                    self.lines[before] += f"  # {c[0]}"
        if not any(l.strip() and not l.strip().startswith("#") for l in self.lines[start:]):
            self.emit("pass", ind)

    def flush_pre(self, ind):
        for line in self.pre:
            self.emit(line, ind)
        self.pre = []

    def stmt(self, s, ind):
        m = getattr(self, "s_" + type(s).__name__, None)
        if m is None:
            raise Unsupported(f"{type(s).__name__} statements have no Apsi equivalent")
        start = len(self.lines)
        outer, self.pre = self.pre, []
        try:
            m(s, ind)
            mine = self.pre
        finally:
            self.pre = outer
        if mine:
            self.lines[start:start] = ["    " * ind + l for l in mine]

    def s_Expr(self, s, ind):
        v = s.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            for line in v.value.strip().splitlines():
                self.emit(f"# {line.strip()}".rstrip(), ind)
            return
        if isinstance(v, (ast.Yield, ast.YieldFrom)):
            if isinstance(v, ast.Yield):
                self.emit("yield" + (f" {self.ex(v.value, top=True)}" if v.value is not None else ""), ind)
            else:
                self.emit(f"yield from {self.ex(v.value, top=True)}", ind)
            return
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and self._file_of(v.func.value) \
                and v.func.attr in ("write", "writelines"):
            fv = v.func.value.id
            arg = self.ex(v.args[0])
            self.emit(f"{fv}_text += " + (f"str({arg})" if v.func.attr == "write" else f'join({arg}, "")'), ind)
            return
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print":
            fkw = next((k for k in v.keywords if k.arg == "file"), None)
            if fkw is not None and self._file_of(fkw.value):
                fv = fkw.value.id
                sep = next((self.ex(k.value) for k in v.keywords if k.arg == "sep"), '" "')
                end = next((self.ex(k.value) for k in v.keywords if k.arg == "end"), '"\\n"')
                items = ", ".join(f"str({self.ex(a)})" for a in v.args)
                self.emit(f"{fv}_text += join([{items}], {sep}) + {end}", ind)
                return
            if fkw is not None:
                self.warn(s, "print(file=…) to something other than an open file prints to the screen")
                v = ast.Call(func=v.func, args=v.args, keywords=[k for k in v.keywords if k.arg != "file"])
        self.emit(self.ex(v), ind)

    def s_Pass(self, s, ind):
        self.emit("pass", ind)

    def s_Break(self, s, ind):
        if self.loop_flags and self.loop_flags[-1]:
            self.emit(f"{self.loop_flags[-1]} = true", ind)
        self.emit("break", ind)

    def s_Continue(self, s, ind):
        self.emit("continue", ind)

    def s_Return(self, s, ind):
        if self.gen_stack:
            self.emit(f"return {self.gen_stack[-1]}", ind)
            return
        self.emit("return" + (f" {self.ex(s.value, top=True)}" if s.value is not None else ""), ind)

    def s_Global(self, s, ind):
        pass

    s_Nonlocal = s_Global

    def s_Import(self, s, ind):
        pass

    s_ImportFrom = s_Import

    def s_Delete(self, s, ind):
        for t in s.targets:
            if isinstance(t, ast.Subscript):
                self.emit(f"{self.ex(t.value)}.pop({self.ex(t.slice)})", ind)
            else:
                self.warn(s, "del of a variable is not needed in Apsi (dropped)")

    def s_Assert(self, s, ind):
        self.emit(f"assert {self.ex(s.test)}" + (f", {self.ex(s.msg)}" if s.msg else ""), ind)

    def s_Raise(self, s, ind):
        if s.exc is None:
            self.emit("raise", ind)
            return
        e = s.exc
        name = dotted(e.func if isinstance(e, ast.Call) else e) or ""
        short = name.split(".")[-1]
        from .exceptions import KINDS
        if short in KINDS or short in self.classes or short.endswith(("Error", "Exception", "Warning")):
            if short not in KINDS and short not in self.classes:
                # an exception class from a library: keep its name as the kind
                arg = self.ex(e.args[0]) if isinstance(e, ast.Call) and e.args else _q(short)
                self.emit(f"raise {arg}", ind)
                return
            if short in self.classes:
                self.emit(f"raise {self.ex(e)}", ind)
                return
            arg = ", ".join(self.ex(x) for x in e.args) if isinstance(e, ast.Call) else ""
            self.emit(f"raise {short}({arg})", ind)
            return
        self.emit(f"raise {self.ex(e)}", ind)

    def _decl(self, name):
        """'let ', 'var ' or '' for the first/next assignment of `name` in the current scope."""
        sc = self.scope
        if name in sc.globals or name in sc.declared:
            return ""
        sc.declared.add(name)
        mutable = sc.assign.get(name, 0) > 1 or name in sc.aug
        if self._global_written(name):
            mutable = True
        return "var " if mutable else "let "

    def _global_written(self, name):
        if len(self.scopes) != 1:
            # a variable of a function that an inner function rebinds with `nonlocal`
            return any(isinstance(n, ast.Nonlocal) and name in n.names for n in ast.walk(self.tree))
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
                return True
            if isinstance(n, ast.FunctionDef):
                for sub in ast.walk(n):
                    if isinstance(sub, ast.Attribute) and False:
                        pass
        return self._mutated_in_function(name)

    def _mutated_in_function(self, name):
        # a module-level list/dict mutated inside functions is fine as `let`; only rebinding matters
        return False

    def s_Assign(self, s, ind):
        if len(s.targets) > 1:
            tmp = self.new_tmp("v")
            self.emit(f"let {tmp} = {self.ex(s.value, top=True)}", ind)
            for t in s.targets:
                self.assign(t, tmp, ind, s)
            return
        self.assign(s.targets[0], self.ex(s.value, top=True), ind, s)

    def s_AnnAssign(self, s, ind):
        if s.value is None:
            if isinstance(s.target, ast.Name) and s.target.id not in self.scope.declared:
                self.scope.declared.add(s.target.id)
                self.emit(f"var {self._safe(s.target.id)} = none", ind)
            return
        typ = self.typ(s.annotation)
        if isinstance(s.target, ast.Name) and s.target.id not in self.scope.declared and typ and typ in (
                "Int", "Float", "Str", "Bool", "Tensor"):
            d = self._decl(s.target.id)
            self.emit(f"{d}{self._safe(s.target.id)}: {typ} = {self.ex(s.value, top=True)}" if d
                      else f"{self._safe(s.target.id)} = {self.ex(s.value, top=True)}", ind)
            return
        self.assign(s.target, self.ex(s.value, top=True), ind, s)

    def assign(self, t, value, ind, node):
        if isinstance(t, ast.Name):
            self.emit(f"{self._decl(t.id)}{self._safe(t.id)} = {value}", ind)
        elif isinstance(t, (ast.Tuple, ast.List)):
            if any(isinstance(e, ast.Starred) for e in t.elts):
                tmp = self.new_tmp("v")
                self.emit(f"let {tmp} = list({value})", ind)
                star = [i for i, e in enumerate(t.elts) if isinstance(e, ast.Starred)][0]
                n_after = len(t.elts) - star - 1
                for i, e in enumerate(t.elts):
                    if i < star:
                        self.assign(e, f"{tmp}[{i}]", ind, node)
                    elif i == star:
                        end = f"len({tmp}) - {n_after}" if n_after else f"len({tmp})"
                        self.assign(e.value, f"{tmp}[{star}:{end}]", ind, node)
                    else:
                        self.assign(e, f"{tmp}[len({tmp}) - {len(t.elts) - i}]", ind, node)
                return
            simple = all(isinstance(e, ast.Name) for e in t.elts)
            if simple:
                names = [e.id for e in t.elts]
                fresh = [n for n in names if n not in self.scope.declared and n not in self.scope.globals]
                if fresh and len(fresh) == len(names):
                    kinds = {self._decl(n) for n in names}
                    kw = "var " if "var " in kinds else "let "
                    self.emit(f"{kw}{', '.join(self._safe(n) for n in names)} = {value}", ind)
                    return
                for n in fresh:
                    self.scope.declared.add(n)
                    self.emit(f"var {self._safe(n)} = none", ind)
                self.emit(f"{', '.join(self._safe(n) for n in names)} = {value}", ind)
                return
            tmp = self.new_tmp("v")
            self.emit(f"let {tmp} = {value}", ind)
            for i, e in enumerate(t.elts):
                self.assign(e, f"{tmp}[{i}]", ind, node)
        elif isinstance(t, (ast.Subscript, ast.Attribute)):
            self.emit(f"{self.ex(t)} = {value}", ind)
        else:
            raise Unsupported(f"assignment to {type(t).__name__}")

    def s_AugAssign(self, s, ind):
        op = OPS.get(type(s.op))
        if op is None:
            raise Unsupported(f"the operator {type(s.op).__name__}= (bitwise) is not available in Apsi")
        if isinstance(s.target, ast.Name) and s.target.id not in self.scope.declared and s.target.id not in self.scope.globals \
                and len(self.scopes) == 1:
            self.scope.declared.add(s.target.id)
        self.emit(f"{self.ex(s.target)} {op}= {self.ex(s.value, top=True)}", ind)

    def s_If(self, s, ind):
        if _is_main_guard(s):
            self.emit("# --- main program", ind)
            self.body(s.body, ind)
            return
        self.emit(f"if {self.ex(s.test, top=True)}:", ind)
        self.body(s.body, ind + 1)
        orelse = s.orelse
        while len(orelse) == 1 and isinstance(orelse[0], ast.If) and not _is_main_guard(orelse[0]):
            e = orelse[0]
            self._flush_comments(e.lineno, ind)
            self.emit(f"elif {self.ex(e.test, top=True)}:", ind)
            self.body(e.body, ind + 1)
            orelse = e.orelse
        if orelse:
            self.emit("else:", ind)
            self.body(orelse, ind + 1)

    def _loop_enter(self, s, ind):
        flag = None
        if s.orelse:
            flag = self.new_tmp("broke")
            self.emit(f"var {flag} = false", ind)
        self.loop_flags.append(flag)
        return flag

    def _loop_exit(self, s, ind, flag):
        self.loop_flags.pop()
        if s.orelse:
            self.emit(f"if not {flag}:", ind)
            self.body(s.orelse, ind + 1)

    def s_While(self, s, ind):
        flag = self._loop_enter(s, ind)
        try:
            self._while(s, ind)
        finally:
            self._loop_exit(s, ind, flag)

    def _while(self, s, ind):
        has_walrus = any(isinstance(x, ast.NamedExpr) for x in ast.walk(s.test))
        if has_walrus:
            # while (x := f()): …  →  re-evaluate the assignment at the top of every iteration
            self.emit("while true:", ind)
            saved = self.pre
            self.pre = []
            cond = self.ex(s.test, top=True)
            for line in self.pre:
                if line.startswith("var "):
                    name = line[4:].split(" =", 1)[0]
                    self.lines.insert(len(self.lines) - 1, "    " * ind + f"var {name} = none")
                    line = line[4:]
                self.emit(line, ind + 1)
            self.pre = saved
            self.emit(f"if not ({cond}):", ind + 1)
            self.emit("break", ind + 2)
        else:
            self.emit(f"while {self.ex(s.test, top=True)}:", ind)
        self.body(s.body, ind + 1)

    def s_For(self, s, ind):
        flag = self._loop_enter(s, ind)
        try:
            self._for(s, ind)
        finally:
            self._loop_exit(s, ind, flag)

    def _for(self, s, ind):
        it = self.iter_expr(s.iter)
        t = s.target
        unpack = []
        if isinstance(t, ast.Name):
            target = self._safe(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Name) for e in t.elts):
            target = ", ".join(self._safe(e.id) for e in t.elts)
        elif isinstance(t, (ast.Tuple, ast.List)):
            parts = []
            for e in t.elts:
                if isinstance(e, ast.Name):
                    parts.append(self._safe(e.id))
                else:
                    tmp = self.new_tmp("item")
                    parts.append(tmp)
                    unpack.append((e, tmp))
            target = ", ".join(parts)
        else:
            raise Unsupported("this loop target")
        body_assigned = analyse(s.body).assign
        rebinds = []
        for tn in [x.strip() for x in target.split(",")]:
            raw = tn[:-1] if tn.endswith("_") and tn[:-1] in APSI_RESERVED else tn
            if raw in body_assigned or raw in self.scope.declared:
                tmp = self.new_tmp("it")
                rebinds.append((raw, tmp))
                target = ", ".join(tmp if x.strip() == tn else x.strip() for x in target.split(","))
        self.emit(f"for {target} in {it}:", ind)
        for raw, tmp in rebinds:
            if raw in self.scope.declared:
                self.emit(f"{self._safe(raw)} = {tmp}", ind + 1)
            else:
                self.emit(f"var {self._safe(raw)} = {tmp}", ind + 1)
        for e, tmp in unpack:
            self._unpack_into(e, tmp, ind + 1)
        self.body(s.body, ind + 1)

    def _unpack_into(self, e, value, ind):
        if isinstance(e, ast.Name):
            self.emit(f"let {self._safe(e.id)} = {value}", ind)
        elif isinstance(e, (ast.Tuple, ast.List)):
            for i, sub in enumerate(e.elts):
                self._unpack_into(sub, f"{value}[{i}]", ind)
        else:
            raise Unsupported("this unpacking target")

    def iter_expr(self, node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" \
                and not node.keywords and self.alias.get("range") is None:
            a = node.args
            if len(a) == 1:
                return f"0..{self.ex(a[0], prec=PREC['+'] + 1)}"
            if len(a) == 2:
                return f"{self.ex(a[0], prec=PREC['+'] + 1)}..{self.ex(a[1], prec=PREC['+'] + 1)}"
        name = self._file_of(node)
        if name:
            return f"read_text({self.files[name][0]}).splitlines()"
        return self.ex(node, top=True)

    def s_With(self, s, ind):
        handled = True
        for it in s.items:
            if _is_open(it.context_expr) and isinstance(it.optional_vars, ast.Name):
                args = it.context_expr.args
                mode = "r"
                if len(args) > 1 and isinstance(args[1], ast.Constant):
                    mode = args[1].value
                for kw in it.context_expr.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                path_tmp = self.new_tmp("path")
                self.emit(f"let {path_tmp} = {self.ex(args[0])}", ind)
                self.files[it.optional_vars.id] = (path_tmp, mode)
                if "a" in mode:
                    self.warn(s, "files opened for appending are rewritten with read_text + write_text")
                    self.emit(f"var {it.optional_vars.id}_text = read_text({path_tmp}) if exists({path_tmp}) else \"\"", ind)
                elif "w" in mode:
                    self.emit(f"var {it.optional_vars.id}_text = \"\"", ind)
            else:
                handled = False
                if it.optional_vars is not None:
                    self.assign(it.optional_vars, self.ex(it.context_expr), ind, s)
                else:
                    self.emit(self.ex(it.context_expr), ind)
                self.warn(s, f"'with {ast.get_source_segment(self.source, it.context_expr)}': the block runs "
                             "without entering/exiting the context")
        self.body(s.body, ind)
        for it in s.items:
            if _is_open(it.context_expr) and isinstance(it.optional_vars, ast.Name):
                path, mode = self.files.pop(it.optional_vars.id)
                if "w" in mode or "a" in mode:
                    self.emit(f"write_text({path}, {it.optional_vars.id}_text)", ind)
        _ = handled

    def s_Try(self, s, ind):
        self.emit("try:", ind)
        self.body(s.body, ind + 1)
        for h in s.handlers:
            kinds = []
            if h.type is not None:
                elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
                for t in elts:
                    k = dotted(t) or "Error"
                    if k in ("Exception", "BaseException"):
                        k = "Error"
                    if k in self.classes:
                        kinds += [x for x in [k] + self._subclasses(k) if x not in kinds]
                    else:
                        kinds.append(k.split(".")[-1] if not k.startswith("json.") else k)
            head = "except"
            if kinds:
                head += " " + ", ".join(kinds)
            if h.name:
                head += f" as {self._safe(h.name)}"
            self.emit(head + ":", ind)
            self.body(h.body, ind + 1)
        if s.orelse:
            self.emit("else:", ind)
            self.body(s.orelse, ind + 1)
        if s.finalbody:
            self.emit("finally:", ind)
            self.body(s.finalbody, ind + 1)

    s_TryStar = s_Try

    def s_Match(self, s, ind):
        raise Unsupported("match statements — rewrite as if/elif")

    # ------------------------------------------------------------ functions
    def _params(self, args, skip_first=False):
        out = []
        pos = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
        for i, (a, d) in enumerate(zip(pos, defaults)):
            if skip_first and i == 0:
                continue
            out.append(self._param(a, d))
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            out.append(self._param(a, d))
        if args.vararg:
            out.append(f"*{self._safe(args.vararg.arg)}")
        if args.kwarg:
            out.append(f"**{self._safe(args.kwarg.arg)}")
        return out

    def _param(self, a, default):
        s = self._safe(a.arg)
        typ = self.typ(a.annotation) if a.annotation is not None else None
        if typ in ("Int", "Float", "Str", "Bool", "Tensor", "List", "Dict", "Fn"):
            s += f": {typ}"
        if default is not None:
            s += f" = {self.ex(default)}"
        return s

    def typ(self, ann):
        if ann is None:
            return None
        name = dotted(ann) if not isinstance(ann, ast.Subscript) else dotted(ann.value)
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            name = ann.value
        if not name:
            return None
        return TYPE_MAP.get(name.split(".")[-1])

    def s_FunctionDef(self, s, ind, method_of=None):
        decos = []
        for d in s.decorator_list:
            name = dotted(d.func if isinstance(d, ast.Call) else d)
            if name in ("functools.lru_cache", "lru_cache", "functools.cache", "cache"):
                decos.append("@memo")
            elif name in ("staticmethod", "classmethod", "property", "abstractmethod", "abc.abstractmethod"):
                continue
            elif name and name.split(".")[0] in self.classes or (name and self._defined(name.split(".")[0])):
                decos.append("@" + self.ex(d))
            else:
                self.warn(s, f"decorator @{name or '?'} has no Apsi equivalent (dropped)")
        for d in decos:
            self.emit(d, ind)
        skip = method_of is not None and s.args.args and s.args.args[0].arg in ("cls",)
        params = self._params(s.args, skip_first=skip)
        name = self._safe(s.name)
        if method_of is None:
            self.scope.declared.add(s.name)
        ret = self.typ(s.returns)
        ret_s = f" -> {ret}" if ret in ("Int", "Float", "Str", "Bool", "Tensor") else ""
        param_names = [a.arg for a in list(s.args.posonlyargs) + list(s.args.args) + list(s.args.kwonlyargs)]
        if s.args.vararg:
            param_names.append(s.args.vararg.arg)
        if s.args.kwarg:
            param_names.append(s.args.kwarg.arg)
        info = analyse(s.body, param_names)
        body = s.body
        gen = _has_yield(s)
        # one-line function
        if (not gen and len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None
                and not info.reassigned_params):
            self.scopes.append(info)
            expr = self.ex(body[0].value, top=True)
            self.scopes.pop()
            if not self.pre:
                self.emit(f"fn {name}({', '.join(params)}){ret_s} = {expr}", ind)
                return
            self.pre = []
        self.emit(f"fn {name}({', '.join(params)}){ret_s}:", ind)
        prologue = []
        saved_files = dict(self.files)
        doc_skip = body
        self.block(doc_skip, ind + 1, info, prologue=prologue)
        self.files = saved_files

    s_AsyncFunctionDef = s_FunctionDef

    def _defined(self, name):
        return any(name in sc.declared for sc in self.scopes)

    def s_ClassDef(self, s, ind):
        c = self.classes.get(s.name)
        if c is None:                      # nested class
            self._collect_classes([s])
            c = self.classes[s.name]
        for b in c["bases"]:
            if b and b not in self.classes and b not in ("object",):
                if b.endswith("Module"):
                    self.warn(s, f"class {s.name}({b}): the model is converted as a plain struct; "
                                 f"consider rewriting it with `use layers` (sequential/dag)")
                elif b.split(".")[-1] in ("Dataset", "IterableDataset"):
                    c["dataset"] = True
                elif b not in ("Exception", "ValueError", "RuntimeError"):
                    self.warn(s, f"base class {b} is not defined in this file; its behaviour is not inherited")
        fields = self._class_fields(s.name)
        self.scope.declared.add(s.name)
        doc = ast.get_docstring(s)
        if doc:
            for line in doc.splitlines():
                self.emit(f"# {line}".rstrip(), ind)
        self.emit(f"struct {s.name}:", ind)
        wrote = False
        for fname, ann, default in fields:
            t = self.typ(ann) if ann is not None else None
            t = t if t in ("Int", "Float", "Str", "Bool", "List", "Dict", "Tensor") else "Any"
            if (isinstance(default, ast.Call) and dotted(default.func) in ("field", "dataclasses.field")):
                kw = {k.arg: k.value for k in default.keywords}
                if "default_factory" in kw:
                    fac = dotted(kw["default_factory"])
                    default = {"list": ast.List(elts=[], ctx=ast.Load()), "dict": ast.Dict(keys=[], values=[]),
                               }.get(fac) or ast.Call(func=kw["default_factory"], args=[], keywords=[])
                else:
                    default = kw.get("default")
            d = self.ex(default) if default is not None else ("none" if not c["dataclass"] or ann is None else None)
            if c["dataclass"] and default is None and ann is not None:
                self.emit(f"{fname}: {t}", ind + 1)
            else:
                self.emit(f"{fname}: {t} = {d if d is not None else 'none'}", ind + 1)
            wrote = True
        # methods (own and inherited from classes in this file)
        methods = {}
        for b in c["bases"]:
            if b in self.classes and b != s.name:
                methods.update(self.classes[b]["methods"])
        methods.update(c["methods"])
        self.cur_class = s.name
        for mname, m in methods.items():
            if mname == "__init__" or mname in c["statics"]:
                continue
            if mname == "__repr__" and "__str__" in methods:
                continue
            if mname.startswith("__") and mname.endswith("__"):
                if c.get("dataset") and mname in ("__len__", "__getitem__"):
                    pass                                   # used by torch.DataLoader
                elif mname in DUNDER:
                    m = ast.FunctionDef(**{**m.__dict__, "name": DUNDER[mname]})
                else:
                    self.warn(m, f"special method {mname} is not used by Apsi operators (kept as a normal method)")
            info_method = m
            self.scopes.append(analyse([], ()))
            self.s_FunctionDef(info_method, ind + 1, method_of=s.name)
            self.scopes.pop()
            wrote = True
        self.cur_class = None
        if not wrote:
            self.emit("pass", ind + 1)
        for cv, val in c.get("classvars", {}).items():
            self.emit(f"var {s.name}_{cv} = {self.ex(val)}", ind)
        # static methods become functions Class_name
        for mname in c["statics"]:
            m = c["methods"][mname]
            self.s_FunctionDef(ast.FunctionDef(**{**m.__dict__, "name": f"{s.name}_{mname}"}), ind)
        # constructor
        init = c["init"] or next((self.classes[b]["init"] for b in c["bases"]
                                  if b in self.classes and self.classes[b]["init"] is not None), None)
        if init is not None:
            self._emit_init(s.name, init, ind)
        self.emit("", ind)

    def _emit_init(self, cls, init, ind):
        params = self._params(init.args, skip_first=True)
        names = [self._safe(a.arg) for a in list(init.args.posonlyargs) + list(init.args.args)][1:]
        names += [self._safe(a.arg) for a in init.args.kwonlyargs]
        param_names = [a.arg for a in list(init.args.posonlyargs) + list(init.args.args) + list(init.args.kwonlyargs)]
        self.cur_class = cls
        self.emit(f"fn {cls}_init({', '.join(['self'] + params)}):", ind)
        info = analyse(init.body, param_names)
        self.block(init.body, ind + 1, info)
        self.emit(f"fn {cls}_new({', '.join(params)}):", ind)
        self.emit(f"var obj = {cls}()", ind + 1)
        self.emit(f"{cls}_init(obj, {', '.join(f'{n}={n}' for n in names)})" if names else f"{cls}_init(obj)", ind + 1)
        self.emit("return obj", ind + 1)
        self.cur_class = None

    # ------------------------------------------------------------ expressions
    def ex(self, n, prec=0, top=False):
        m = getattr(self, "e_" + type(n).__name__, None)
        if m is None:
            raise Unsupported(f"{type(n).__name__} expressions")
        text, p = m(n)
        return f"({text})" if p < prec else text

    def e_Constant(self, n):
        v = n.value
        if v is None:
            return "none", PREC["atom"]
        if v is True:
            return "true", PREC["atom"]
        if v is False:
            return "false", PREC["atom"]
        if isinstance(v, str):
            return _q(v), PREC["atom"]
        if isinstance(v, bytes):
            self.warn(n, "bytes literal converted to a string")
            return _q(v.decode("latin-1")), PREC["atom"]
        if isinstance(v, complex):
            raise Unsupported("complex numbers (use sci.fft/real/imag for spectra)")
        if v is Ellipsis:
            raise Unsupported("... (Ellipsis)")
        r = repr(v)
        if r in ("inf", "nan"):
            return r, PREC["atom"]
        return (r, PREC["atom"]) if not r.startswith("-") else (r, PREC["unary"])

    def e_Name(self, n):
        name = n.id
        if name in self.alias and not self._defined(name):
            return self.resolve(self.alias[name], n), PREC["atom"]
        if name == "self" and self.cur_class:
            return "self", PREC["atom"]
        return self._safe(name), PREC["atom"]

    def resolve(self, full, node):
        """Map a python dotted name (module.attr...) to Apsi."""
        if full.startswith("__apsi__."):                   # produced by the PyTorch rewriter: already Apsi
            target = full[len("__apsi__."):]
            mod = target.split(".")[0] if "." in target else None
            if mod in ("linalg", "sci", "nn", "layers", "models", "torch"):
                self.uses.add(mod)
            return target
        if full.startswith("@py:"):
            self._use_py(full[4:].split(".")[0])
            return full[4:]
        if full in SIMPLE:
            target = SIMPLE[full]
            mod = target.split(".")[0] if "." in target else None
            if mod in ("linalg", "sci", "nn", "layers"):
                self.uses.add(mod)
            return target
        parts = full.split(".")
        if parts[0] in ("numpy", "torch") and len(parts) > 1 and parts[-1] in DTYPES:
            return DTYPES[parts[-1]]
        if parts[0] == "numpy" and parts[1:] == ["newaxis"]:
            return "none"
        if full in ("sys.argv",):
            return '([""] + args())'
        # fall back to Python through pyimport
        root = parts[0]
        if self._safe(root) in self.lazy_py or root in self.lazy_py:
            self._use_py(root)
            return ".".join([self._safe(root)] + parts[1:])
        if root in self.alias and self.alias[root] != root and not full.startswith(self.alias[root]):
            root_mod = self.alias[root]
        else:
            root_mod = root
        alias = self._safe(root_mod.replace(".", "_")) + ("_py" if root_mod in ("numpy", "math", "random", "torch",
                                                                                "json", "time", "os", "sys") else "")
        self.pyimports[alias] = root_mod
        self.warn(node, f"{full} has no Apsi equivalent: it is called through pyimport(\"{root_mod}\") "
                        f"(run with --allow-python)")
        return ".".join([alias] + parts[1:])

    def _use_py(self, alias):
        key = alias if alias in self.lazy_py else self._safe(alias)
        if key in self.lazy_py and self._safe(alias) not in self.pyimports:
            mod, node = self.lazy_py[key]
            self.pyimports[self._safe(alias)] = mod
            from .interop import PURE_MODULES
            if mod in PURE_MODULES:
                return
            self.warn(node, f"'{mod}' has no Apsi equivalent: it is loaded with pyimport() — run with "
                            f"--allow-python (libraries that reach the network or the system need --allow-python=all)")

    def e_Attribute(self, n):
        d = dotted(n)
        if d:
            root = d.split(".")[0]
            if root in self.alias and not self._defined(root):
                full = self.alias[root] + d[len(root):]
                if full.startswith("@py:"):
                    self._use_py(full[4:].split(".")[0])
                    return full[4:], PREC["atom"]
                return self.resolve(full, n), PREC["atom"]
            if root in self.classes and (n.attr in self.classes[root]["statics"]
                                         or n.attr in self.classes[root].get("classvars", {})):
                return f"{root}_{n.attr}", PREC["atom"]
        fv = self._file_of(n.value)
        if fv:
            raise Unsupported(f"file method .{n.attr} outside a call")
        if n.attr.startswith("__"):
            raise Unsupported(f"special attribute {n.attr}")
        return f"{self.ex(n.value, PREC['atom'])}.{n.attr}", PREC["atom"]

    def _file_of(self, node):
        if isinstance(node, ast.Name) and node.id in self.files:
            return node.id
        return None

    def e_BinOp(self, n):
        op = OPS.get(type(n.op))
        if op is None:
            fn = {ast.BitAnd: "band", ast.BitOr: "bor", ast.BitXor: "bxor", ast.LShift: "shl",
                  ast.RShift: "shr"}[type(n.op)]
            # ints bit by bit, sets union/intersection, dicts merged, boolean tensors element-wise — like Python
            return f"{fn}({self.ex(n.left)}, {self.ex(n.right)})", PREC["atom"]
        if op == "%" and isinstance(n.left, ast.Constant) and isinstance(n.left.value, str):
            return f"{self.ex(n.left, PREC['%'])} % {self.ex(n.right, PREC['%'] + 1)}", PREC["%"]
        p = PREC[op]
        if op == "**":
            return f"{self.ex(n.left, p + 1)} ** {self.ex(n.right, p)}", p
        return f"{self.ex(n.left, p)} {op} {self.ex(n.right, p + 1)}", p

    def e_UnaryOp(self, n):
        if isinstance(n.op, ast.Not):
            return f"not {self.ex(n.operand, PREC['not'])}", PREC["not"]
        if isinstance(n.op, ast.USub):
            return f"-{self.ex(n.operand, PREC['unary'])}", PREC["unary"]
        if isinstance(n.op, ast.UAdd):
            return f"+{self.ex(n.operand, PREC['unary'])}", PREC["unary"]
        raise Unsupported("the bitwise ~ operator")

    def e_BoolOp(self, n):
        op = "and" if isinstance(n.op, ast.And) else "or"
        p = PREC[op]
        return f" {op} ".join(self.ex(v, p + 1) for v in n.values), p

    def e_Compare(self, n):
        parts = [self.ex(n.left, PREC["cmp"] + 1)]
        for op, c in zip(n.ops, n.comparators):
            parts.append(CMP[type(op)])
            parts.append(self.ex(c, PREC["cmp"] + 1))
        return " ".join(parts), PREC["cmp"]

    def e_IfExp(self, n):
        return (f"{self.ex(n.body, PREC['ifexp'] + 1)} if {self.ex(n.test, PREC['ifexp'] + 1)} "
                f"else {self.ex(n.orelse, PREC['ifexp'])}"), PREC["ifexp"]

    def e_Lambda(self, n):
        params = self._params(n.args)
        names = [a.arg for a in n.args.args]
        self.scopes.append(analyse([], names))
        body = self.ex(n.body)
        self.scopes.pop()
        return f"fn({', '.join(params)}) => {body}", PREC["lambda"]

    def e_List(self, n):
        return "[" + ", ".join(self.ex(e) for e in n.elts) + "]", PREC["atom"]

    def e_Tuple(self, n):
        if len(n.elts) == 1:
            return f"({self.ex(n.elts[0])},)", PREC["atom"]
        return "(" + ", ".join(self.ex(e) for e in n.elts) + ")", PREC["atom"]

    def e_Set(self, n):
        return "set([" + ", ".join(self.ex(e) for e in n.elts) + "])", PREC["atom"]

    def e_Dict(self, n):
        items = []
        for k, v in zip(n.keys, n.values):
            if k is None:
                raise Unsupported("**dict unpacking")
            items.append(f"{self.ex(k)}: {self.ex(v)}")
        return "{" + ", ".join(items) + "}", PREC["atom"]

    def _gens(self, gens):
        out = []
        names = []
        for g in gens:
            if g.is_async:
                raise Unsupported("async comprehensions")
            t = g.target
            if isinstance(t, ast.Name):
                tt = self._safe(t.id)
                names.append(t.id)
            elif isinstance(t, ast.Tuple) and all(isinstance(e, ast.Name) for e in t.elts):
                tt = ", ".join(self._safe(e.id) for e in t.elts)
                names += [e.id for e in t.elts]
            else:
                raise Unsupported("nested unpacking inside a comprehension")
            out.append(f"for {tt} in {self.iter_expr(g.iter)}")
            for cond in g.ifs:
                out.append(f"if {self.ex(cond, PREC['ifexp'] + 1)}")
        return " ".join(out)

    def e_ListComp(self, n):
        saved, self.in_comprehension = self.in_comprehension, True
        try:
            if len(n.generators) > 1:
                # [e for a in A for b in B if c]  →  chain_lists([[e for b in B if c] for a in A])
                inner = ast.ListComp(elt=n.elt, generators=n.generators[1:])
                return f"chain_lists([{self.e_ListComp(inner)[0]} {self._gens(n.generators[:1])}])", PREC["atom"]
            return f"[{self.ex(n.elt)} {self._gens(n.generators)}]", PREC["atom"]
        finally:
            self.in_comprehension = saved

    def e_GeneratorExp(self, n):
        return self.e_ListComp(n)

    def e_SetComp(self, n):
        return f"set({self.e_ListComp(ast.ListComp(elt=n.elt, generators=n.generators))[0]})", PREC["atom"]

    def e_DictComp(self, n):
        if len(n.generators) > 1:
            pairs = self.e_ListComp(ast.ListComp(elt=ast.Tuple(elts=[n.key, n.value], ctx=ast.Load()),
                                                 generators=n.generators))[0]
            return f"dict({pairs})", PREC["atom"]
        return f"{{{self.ex(n.key)}: {self.ex(n.value)} {self._gens(n.generators)}}}", PREC["atom"]

    def e_Subscript(self, n):
        return f"{self.ex(n.value, PREC['atom'])}[{self.slice(n.slice)}]", PREC["atom"]

    def slice(self, s):
        if isinstance(s, ast.Slice):
            a = self.ex(s.lower) if s.lower is not None else ""
            b = self.ex(s.upper) if s.upper is not None else ""
            text = f"{a}:{b}"
            if s.step is not None:
                text += f":{self.ex(s.step)}"
            return text
        if isinstance(s, ast.Tuple):
            return ", ".join(self.slice(e) for e in s.elts)
        if isinstance(s, ast.Constant) and s.value is None:
            raise Unsupported("indexing with None/np.newaxis — use reshape()")
        return self.ex(s)

    def e_Starred(self, n):
        return "*" + self.ex(n.value, PREC["atom"]), PREC["atom"]

    def e_NamedExpr(self, n):
        if self.in_comprehension:
            raise Unsupported("the := operator inside a comprehension")
        name = n.target.id
        value = self.ex(n.value)
        if name not in self.scope.declared and name not in self.scope.globals:
            self.scope.declared.add(name)
            self.pre.append(f"var {self._safe(name)} = {value}")
        else:
            self.pre.append(f"{self._safe(name)} = {value}")
        return self._safe(name), PREC["atom"]

    def e_Await(self, n):
        raise Unsupported("await")

    def e_Yield(self, n):
        raise Unsupported("yield used as a value")

    def e_JoinedStr(self, n):
        parts = []
        simple = True
        for v in n.values:
            if isinstance(v, ast.Constant):
                parts.append(("s", str(v.value)))
            else:
                expr = self.ex(v.value)
                spec = None
                if v.format_spec is not None:
                    spec = "".join(str(x.value) if isinstance(x, ast.Constant) else "{" + self.ex(x.value) + "}"
                                   for x in v.format_spec.values)
                    if "{" in spec:
                        simple = False
                if '"' in expr or "{" in expr or "}" in expr:
                    simple = False
                parts.append(("e", expr, spec))
        if simple:
            out = []
            for p in parts:
                if p[0] == "s":
                    out.append(_q(p[1])[1:-1].replace("{", "{{").replace("}", "}}"))
                else:
                    out.append("{" + p[1] + (":" + p[2] if p[2] else "") + "}")
            return 'f"' + "".join(out) + '"', PREC["atom"]
        pieces = []
        for p in parts:
            if p[0] == "s":
                pieces.append(_q(p[1]))
            elif p[2]:
                pieces.append(f"format({p[1]}, {_q(p[2])})")
            else:
                pieces.append(f"str({p[1]})")
        return " + ".join(pieces) if pieces else '""', PREC["+"]

    def e_FormattedValue(self, n):
        return "str(" + self.ex(n.value) + ")", PREC["atom"]

    def e_Slice(self, n):
        return self.slice(n), PREC["atom"]

    # ------------------------------------------------------------ calls
    def e_Call(self, n):
        if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
            args = [self.ex(a) for a in n.args]
            kws = [f"**{self.ex(k.value, PREC['atom'])}" if k.arg is None else f"{k.arg}={self.ex(k.value)}"
                   for k in n.keywords]
            return f"{self.ex(n.func, PREC['atom'])}({', '.join(args + kws)})", PREC["atom"]
        fname = dotted(n.func)
        full = None
        if fname:
            root = fname.split(".")[0]
            if root in self.alias and not self._defined(root):
                full = self.alias[root] + fname[len(root):]
            elif root not in self.classes and not self._defined(root):
                full = fname
        args = [self.ex(a) for a in n.args]
        kws = [(k.arg, self.ex(k.value)) for k in n.keywords]
        special = self._special_call(n, fname, full, args, kws)
        if special is not None:
            return special, PREC["atom"]
        if fname and fname in self.classes:
            c = self.classes[fname]
            if c["init"] is not None or any(self.classes.get(b, {}).get("init") for b in c["bases"]):
                return self._call(f"{fname}_new", args, kws), PREC["atom"]
            return self._call(fname, args, kws), PREC["atom"]
        if isinstance(n.func, ast.Attribute):
            fv = self._file_of(n.func.value)
            if fv:
                return self._file_call(n, fv, n.func.attr, args), PREC["atom"]
            if isinstance(n.func.value, ast.Call) and isinstance(n.func.value.func, ast.Name) \
                    and n.func.value.func.id == "super" and self.cur_class:
                c = self.classes[self.cur_class]
                base = next((b for b in c["bases"] if b in self.classes), None)
                if n.func.attr == "__init__" and base:
                    return self._call(f"{base}_init", ["self"] + args, kws), PREC["atom"]
                if n.func.attr == "__init__" and c.get("exception"):
                    # super().__init__(message) of an exception class: keep the message
                    return (f"self.message = str({args[0]})" if args else "self.message = \"\""), PREC["atom"]
                raise Unsupported("super() calls other than super().__init__")
        if full and (full in SIMPLE or full.split(".")[0] in ("numpy", "math", "random", "torch", "json", "time",
                                                              "os", "sys", "statistics", "functools")):
            target = self.resolve(full, n)
            return self._call(target, args, self._kw_map(target, kws)), PREC["atom"]
        func = self.ex(n.func, PREC["atom"])
        return self._call(func, args, kws), PREC["atom"]

    def _kw_map(self, target, kws):
        out = []
        for k, v in kws:
            if k == "dtype":
                v = DTYPES.get(v.split(".")[-1], v)
            if k == "keepdim":
                k = "keepdims"
            if k == "dim":
                k = "axis"
            if k in ("device", "requires_grad", "out", "order", "copy"):
                continue
            out.append((k, v))
        return out

    def _call(self, func, args, kws):
        return f"{func}(" + ", ".join(list(args) + [f"{k}={v}" for k, v in kws]) + ")"

    def _file_call(self, n, fv, meth, args):
        path, mode = self.files[fv]
        if meth == "read":
            return f"read_text({path})"
        if meth == "readlines":
            return f"[l + \"\\n\" for l in read_text({path}).splitlines()]"
        if meth == "readline":
            raise Unsupported("file.readline() — read the whole file with read_text()")
        raise Unsupported(f"file method .{meth}()")

    def _special_call(self, n, fname, full, args, kws):
        kwd = dict(kws)
        name = full or fname
        # builtins that differ
        if fname == "range" and not self._defined("range"):
            if len(args) == 1 and not kws:
                return f"list(0..{args[0]})" if False else f"range({args[0]})"
            return None
        if fname == "print":
            return None
        if fname in ("max", "min") and "key" in kwd and not self._defined(fname):
            seq = args[0] if len(args) == 1 else "[" + ", ".join(args) + "]"
            return f"sorted({seq}, key={kwd['key']})[{-1 if fname == 'max' else 0}]"
        if fname == "isinstance" and len(n.args) == 2:
            t = n.args[1]
            names = [dotted(e) for e in t.elts] if isinstance(t, ast.Tuple) else [dotted(t)]
            tests = []
            for tn in names:
                if tn in ISINSTANCE:
                    tests.append(f"type({args[0]}) == {ISINSTANCE[tn]}")
                elif tn in self.classes:
                    subs = [tn] + self._subclasses(tn)
                    tests.append(f"type({args[0]}) == {_q(tn)}" if len(subs) == 1
                                 else f"type({args[0]}) in [{', '.join(_q(x) for x in subs)}]")
                elif tn in ("np.ndarray", "numpy.ndarray", "torch.Tensor"):
                    tests.append(f"type({args[0]}) == \"Tensor\"")
                else:
                    self.warn(n, f"isinstance(…, {tn}) cannot be checked in Apsi (treated as false)")
                    tests.append("false")
            if fname == "isinstance" and names and any(x == "float" for x in names) and "int" not in names:
                tests.append(f"type({args[0]}) == \"Int\"") if False else None
            return "(" + " or ".join(tests) + ")"
        if fname == "sum" and not full_is_lib(full) and len(n.args) == 2 and not kws:
            return f"({args[1]} + sum({args[0]}))"
        if fname == "filter" and len(n.args) == 2 and isinstance(n.args[0], ast.Constant) and n.args[0].value is None:
            return f"filter(bool, {args[1]})"
        if fname == "open":
            raise Unsupported("open() outside `with open(...) as f:` — use read_text()/write_text()")
        if fname == "type" and len(args) == 1:
            return f"type({args[0]})"
        if fname == "len" or fname == "abs" or fname == "sum" and not kws:
            return None
        if fname in ("super",):
            raise Unsupported("super()")
        if not name:
            return None
        # library calls with custom translation
        if name in ("math.log",) and len(args) == 2:
            return f"(log({args[0]}) / log({args[1]}))"
        if name in ("math.hypot",):
            return "sqrt(" + " + ".join(f"({a}) ** 2" for a in args) + ")"
        if name in ("math.radians",):
            return f"({args[0]}) * pi / 180"
        if name in ("math.degrees",):
            return f"({args[0]}) * 180 / pi"
        if name in ("math.factorial",):
            return f"prod([float(_k) for _k in 1..=int({args[0]})]) if int({args[0]}) > 0 else 1"
        if name in ("math.isnan",):
            return f"({args[0]} != {args[0]})"
        if name in ("math.isinf",):
            return f"(abs({args[0]}) == inf)"
        if name in ("numpy.maximum", "torch.maximum"):
            return f"where({args[0]} > {args[1]}, {args[0]}, {args[1]})"
        if name in ("numpy.minimum", "torch.minimum"):
            return f"where({args[0]} < {args[1]}, {args[0]}, {args[1]})"
        if name in ("numpy.square", "torch.square"):
            return f"({args[0]}) ** 2"
        if name in ("numpy.sign", "torch.sign"):
            return f"where({args[0]} > 0, 1.0, where({args[0]} < 0, -1.0, 0.0))"
        if name in ("numpy.reshape", "torch.reshape") and len(args) == 2:
            return f"reshape({args[0]}, {args[1]})"
        if name in ("numpy.vstack",):
            return f"concat({args[0]}, axis=0)"
        if name in ("numpy.hstack",):
            return f"concat({args[0]}, axis=-1)"
        if name in ("numpy.size",):
            return f"len(flatten({args[0]}))"
        if name in ("numpy.random.normal", "numpy.random.uniform", "random.uniform", "random.gauss",
                    "random.normalvariate"):
            loc = kwd.get("loc", args[0] if args else "0.0") if "normal" in name or "gauss" in name else \
                kwd.get("low", args[0] if args else "0.0")
            sc = kwd.get("scale", args[1] if len(args) > 1 else "1.0") if "normal" in name or "gauss" in name else \
                kwd.get("high", args[1] if len(args) > 1 else "1.0")
            size = kwd.get("size", args[2] if len(args) > 2 else None)
            draw = ("randn" if "normal" in name or "gauss" in name else "rand")
            shape = f"*{size}" if size and size.startswith("(") else (size or "")
            r = f"{draw}({shape})" if size else f"{draw}()"
            if "uniform" in name:
                return f"({loc} + ({sc} - {loc}) * {r})"
            return f"({loc} + {sc} * {r})"
        if name in ("numpy.random.random", "numpy.random.random_sample"):
            return f"rand({'*' + args[0] if args and args[0].startswith('(') else (args[0] if args else '')})"
        if name == "random.randint":
            return f"randint({args[0]}, {args[1]} + 1)"
        if name == "random.randrange":
            return f"randint({args[0]}, {args[1]})" if len(args) > 1 else f"randint(0, {args[0]})"
        if name == "random.choice":
            return f"{args[0]}[randint(0, len({args[0]}))]"
        if name in ("random.shuffle",):
            raise Unsupported("random.shuffle() changes a list in place — use permutation(len(xs))")
        if name in ("numpy.random.choice",):
            return f"{args[0]}[randint(0, len({args[0]}))]" if len(args) == 1 else None
        if name in ("json.load",) and n.args and self._file_of(n.args[0]):
            return f"load_json({self.files[self._file_of(n.args[0])][0]})"
        if name in ("json.dump",) and len(n.args) > 1 and self._file_of(n.args[1]):
            return f"save_json({self.files[self._file_of(n.args[1])][0]}, {args[0]})"
        if name in ("os.path.join",):
            return " + \"/\" + ".join(args)
        if name in ("time.sleep",):
            self.uses.add("parallel")
            return f"parallel.sleep({args[0]})"
        if name in ("statistics.median", "numpy.median"):
            return f"sorted(list({args[0]}))[len(list({args[0]})) // 2]"
        if name == "functools.reduce":
            raise Unsupported("functools.reduce — write the loop out")
        if name.startswith("torch.nn.") and name.split(".")[-1] in TORCH_ACTS:
            self.uses.add("layers")
            return f"layers.activation({_q(TORCH_ACTS[name.split('.')[-1]])})"
        if name.startswith("torch.nn.") and name.split(".")[-1] in TORCH_LAYERS:
            self.uses.add("layers")
            target = TORCH_LAYERS[name.split(".")[-1]]
            if name.endswith("Sequential"):
                return f"{target}([{', '.join(args)}])"
            if name.endswith("Dropout"):
                return f"{target}({args[0] if args else kwd.get('p', '0.5')})"
            kept = [(k, v) for k, v in kws if k not in ("batch_first", "bias", "padding_mode")]
            return self._call(target, args, kept)
        if name in ("torch.no_grad", "torch.enable_grad"):
            return "none"
        return None



def _postprocess(text):
    return text


def convert_source(source, filename="<python>"):
    c = Converter(source, filename)
    out = c.convert()
    return _postprocess(out), sorted(c.warnings)


# ----------------------------------------------------------------------------------- CLI
def _check(text, path):
    from .parser import parse
    from .checker import Checker
    from .interpreter import Interpreter
    from .errors import ApsiError
    try:
        prog = parse(text)
        errs = Checker(Interpreter(), os.path.dirname(os.path.abspath(path))).check(prog)
        return [e.format(text, path) for e in errs]
    except ApsiError as e:
        return [e.format(text, path)]


def cmd_convert(args):
    import argparse  # noqa: F401
    paths = args.paths
    jobs = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", ".venv",
                                                                                   "node_modules", "build", "dist")]
                for f in sorted(files):
                    if f.endswith(".py"):
                        src = os.path.join(root, f)
                        rel = os.path.relpath(src, p)
                        out = os.path.join(args.output, rel) if args.output else src
                        jobs.append((src, os.path.splitext(out)[0] + ".apsi"))
        elif os.path.isfile(p):
            out = args.output if args.output and not os.path.isdir(args.output) else (
                os.path.join(args.output, os.path.basename(p)) if args.output else p)
            jobs.append((p, os.path.splitext(out)[0] + ".apsi"))
        else:
            print(f"apsi convert: '{p}' does not exist", file=sys.stderr)
            return 2
    if not jobs:
        print("apsi convert: no .py files found", file=sys.stderr)
        return 2
    status = 0
    total_warn = 0
    for src, dst in jobs:
        try:
            with open(src, encoding="utf-8-sig") as fh:
                code = fh.read()
            text, warns = convert_source(code, src)
        except SyntaxError as e:
            print(f"✗ {src}:{e.lineno}: Python syntax error: {e.msg}", file=sys.stderr)
            status = 1
            continue
        total_warn += len(warns)
        notes = {}
        for line, msg in warns:
            notes.setdefault(line, []).append(msg)
        if args.stdout:
            sys.stdout.write(text)
        else:
            if os.path.exists(dst) and not args.force and not os.path.samefile(src, dst):
                with open(dst, encoding="utf-8") as fh:
                    if not fh.read().startswith("# Converted from"):
                        print(f"✗ {dst} exists and was not written by apsi convert (use --force)", file=sys.stderr)
                        status = 1
                        continue
            os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        problems = [] if args.no_check else _check(text, dst)
        mark = "✓" if not problems else "!"
        if not args.stdout or len(jobs) > 1:
            print(f"{mark} {src} → {dst if not args.stdout else '(stdout)'}  "
                  f"({len(warns)} note{'s' if len(warns) != 1 else ''}"
                  f"{', ' + str(len(problems)) + ' check error' + ('s' if len(problems) != 1 else '') if problems else ''})",
                  file=sys.stderr if args.stdout else sys.stdout)
        for line, msgs in sorted(notes.items()):
            for m in msgs:
                print(f"    line {line}: {m}", file=sys.stderr if args.stdout else sys.stdout)
        for p in problems[:20]:
            print("    " + p.replace("\n", "\n    "), file=sys.stderr if args.stdout else sys.stdout)
        if problems:
            status = 1
        if args.run and not problems and len(jobs) == 1:
            from .cli import main as apsi_main
            run_args = ["run", dst] + (["--allow-python"] if "pyimport(" in text else [])
            return apsi_main(run_args)
    if len(jobs) > 1:
        print(f"\nconverted {len(jobs)} files, {total_warn} notes")
    return status


def add_parser(sub):
    p = sub.add_parser("convert", help="translate Python files or folders into Apsi (numpy, math, torch, … mapped)")
    p.add_argument("paths", nargs="+", help=".py files or folders")
    p.add_argument("-o", "--output", help="output file or folder (default: next to each .py file)")
    p.add_argument("--stdout", action="store_true", help="print the Apsi code instead of writing files")
    p.add_argument("--run", action="store_true", help="run the result (single file) when it checks cleanly")
    p.add_argument("--no-check", action="store_true", help="do not run the Apsi checker on the result")
    p.add_argument("--force", action="store_true", help="overwrite .apsi files that were not made by apsi convert")
    p.set_defaults(fn=cmd_convert)
