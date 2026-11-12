"""Tree-walking reference interpreter for Apsi."""
import re
import operator
import os
import sys
import time
from fractions import Fraction

import numpy as np

from . import ast as A
from . import ops
from . import stdlib
from .autodiff import Node
from .errors import (ApsiError, RuntimeErr, NameErr, TypeCheckError, UnitError, ShapeError,
                     AssertionFailed, CheckFailed)
from .fmt import fmt, type_name
from .parser import parse
from .units import Quantity, DIM_NAMES, dim_name, make
from .values import (Function, BoundMethod, StructType, StructInstance, Record, RangeVal,
                     Namespace, UnitNamespace)

np.seterr(all="ignore")
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


class ControlSignal(BaseException):
    pass


class ReturnSignal(ControlSignal):
    def __init__(self, value):
        self.value = value


class BreakSignal(ControlSignal):
    pass


class ContinueSignal(ControlSignal):
    pass


class LazyModules(dict):
    """The module table; heavy modules (web, db, dataframe, rt …) are imported the first time a program uses them."""

    def __init__(self, *a):
        super().__init__(*a)
        self._lazy = {}

    def lazy(self, name, factory):
        self._lazy[name] = factory

    def _load(self, name):
        f = self._lazy.pop(name, None)
        if f is not None:
            v = f()
            if not dict.__contains__(self, name):
                dict.__setitem__(self, name, v)

    def __getitem__(self, k):
        if k in self._lazy:
            self._load(k)
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        if k in self._lazy:
            self._load(k)
        return dict.get(self, k, default)

    def __contains__(self, k):
        return k in self._lazy or dict.__contains__(self, k)

    def _all(self):
        for k in list(self._lazy):
            self._load(k)

    def keys(self):
        return list(dict.keys(self)) + list(self._lazy)

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        self._all()
        return dict.items(self)

    def values(self):
        self._all()
        return dict.values(self)

    def __len__(self):
        return dict.__len__(self) + len(self._lazy)


class Env:
    __slots__ = ("vars", "consts", "parent", "module")

    def __init__(self, parent=None):
        self.vars = {}
        self.consts = set()
        self.parent = parent
        self.module = None

    def lookup(self, name):
        e = self
        while e is not None:
            v = e.vars
            if name in v:
                return v[name]
            e = e.parent
        raise KeyError(name)

    def define(self, name, value, const=False):
        self.vars[name] = value
        if const:
            self.consts.add(name)
        else:
            self.consts.discard(name)

    def assign(self, name, value):
        e = self
        while e is not None:
            if name in e.vars:
                if name in e.consts:
                    raise RuntimeErr(f"cannot assign to '{name}': it is a constant "
                                     f"(declared with let or a parameter); use var")
                e.vars[name] = value
                return
            e = e.parent
        raise NameErr(f"'{name}' is not defined; declare it first with let or var")


def _eq(a, b):
    r = a == b
    return r


BINOPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "//": operator.floordiv, "%": operator.mod, "@": operator.matmul, "**": operator.pow,
    "==": _eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt,
    ">=": operator.ge, "~=": lambda a, b: ops.approx(a, b),
    "in": lambda a, b: a in b, "not in": lambda a, b: a not in b,
    "±": lambda a, b: ops.plusminus(a, b),
    "is": operator.is_, "is not": operator.is_not, "|": operator.or_, "&": operator.and_,
    "<<": operator.lshift, ">>": operator.rshift, "^": operator.xor,
}
IOPS = {"+": operator.iadd, "-": operator.isub, "*": operator.imul, "/": operator.itruediv,
        "//": operator.ifloordiv, "%": operator.imod, "@": operator.imatmul, "**": operator.ipow,
        "|": operator.ior, "&": operator.iand, "<<": operator.ilshift, ">>": operator.irshift, "^": operator.ixor}


def bind_pattern(pat, value, define):
    """Assign a value to a target pattern (see ast.For): names, nested tuples and one *name."""
    if isinstance(pat, str):
        define(pat, value)
        return
    if isinstance(pat, tuple):
        define(pat[1], list(value))
        return
    vals = list(value)
    star = next((i for i, p in enumerate(pat) if isinstance(p, tuple)), None)
    if star is None:
        if len(vals) != len(pat):
            raise ValueError(f"expected {len(pat)} values to unpack, got {len(vals)}")
        for p, v in zip(pat, vals):
            bind_pattern(p, v, define)
        return
    after = len(pat) - star - 1
    if len(vals) < len(pat) - 1:
        raise ValueError(f"expected at least {len(pat) - 1} values to unpack, got {len(vals)}")
    for p, v in zip(pat[:star], vals[:star]):
        bind_pattern(p, v, define)
    define(pat[star][1], vals[star:len(vals) - after])
    for p, v in zip(pat[star + 1:], vals[len(vals) - after:]):
        bind_pattern(p, v, define)

TENSOR_TYPES = {"Tensor", "Vec", "Mat", "Matrix", "Vec2", "Vec3", "Vec4"}
DTYPE_NAMES = {"f16", "f32", "f64", "i32", "i64", "bool", "bf16", "fp8_e4m3", "fp8_e5m2", "fp6_e3m2",
               "fp6_e2m3", "fp4_e2m1", "int8", "int4", "fp8", "bf8", "fp4", "fp6"}


_PLAIN_GET = set()       # classes whose public attributes are read with plain getattr (learned on first use)
_PLAIN_SET = set()       # classes whose public attributes may be set with plain setattr (after _safe_setattr allowed it)
_FRAME_ATTRS = {"gi_frame", "gi_code", "f_globals", "f_builtins", "f_locals", "f_back", "f_code", "tb_frame",
                "tb_next", "cr_frame", "cr_code", "ag_frame", "ag_code", "co_code", "func_globals", "mro"}
_SLOW_NAMES = _FRAME_ATTRS | {"shape", "tofile", "dump"}
_NOT_PLAIN = (StructInstance, Record, Namespace, UnitNamespace, np.ndarray)
_PUBLIC_UNDERSCORE = {"_replace", "_asdict", "_fields", "_make", "_field_defaults", "_value_", "_name_",
                      "_missing_", "_member_names_", "_member_map_", "_value2member_map_", "_ignore_", "_order_",
                      "_generate_next_value_"}


class _GenStop(BaseException):
    """Ends a generator's thread when its consumer stops early."""


def is_generator(body):
    """True when a function body contains `yield` (outside nested functions)."""
    def walk(n):
        if isinstance(n, A.Yield):
            return True
        if isinstance(n, (A.FnDef, A.Lambda, A.StructDef)):
            return False
        if isinstance(n, (list, tuple)):
            return any(walk(x) for x in n)
        if isinstance(n, A.AST):
            return any(walk(getattr(n, f, None)) for f in n._fields)
        return False
    return walk(body)


def _no_attr(obj, name, msg=None, names=None):
    """'X has no attribute y' with the closest real name, the caret on '.y'."""
    from .errors import did_you_mean
    if names is None:
        try:
            names = [n for n in dir(obj) if not n.startswith("_")]
        except Exception:
            names = []
    err = RuntimeErr((msg or f"{type_name(obj)} has no attribute '{name}'") + did_you_mean(name, names))
    err.focus = "." + name
    return err


_APSI_TYPES = {"list": "List", "tuple": "Tuple", "string": "Str", "str": "Str", "range object": "Range",
               "int": "Int", "float": "Float", "bool": "Bool", "NoneType": "None", "dict": "Dict"}


def _type_hint(msg):
    """A plain-words fix for the most common Python type errors seen through Apsi."""
    if "can only concatenate str" in msg or ("unsupported operand" in msg and "'str'" in msg):
        return "; to join text and a number write str(x) (or an f-string: f\"total {x}\")"
    if "takes no arguments" in msg:
        return "; to take arguments, give the class an fn __init__(self, ...)"
    if "object is not callable" in msg:
        return "; a value, not a function, is being called (check for a missing operator or a name reused)"
    if "has no len()" in msg:
        return "; len() works on text, lists, tuples, dicts, sets and tensors"
    if "does not support item assignment" in msg and "tuple" in msg:
        return "; tuples cannot change: use a list ([...]) to change items"
    if "not subscriptable" in msg:
        return "; only lists, tuples, text, dicts and tensors can be indexed with [...]"
    if "unhashable" in msg:
        return "; dict keys and set items must not change: use a tuple instead of a list"
    return ""


def wrap_python_error(e):
    err = _wrap_python_error(e)
    err.original = e                     # `except ValueError` still recognises the error after wrapping
    return err


def _wrap_python_error(e):
    if isinstance(e, ZeroDivisionError):
        err = RuntimeErr("division by zero")
        err.focus = "/" if "modulo" not in str(e) else "%"
        return err
    msg = str(e)
    if isinstance(e, ValueError) and any(k in msg for k in ("broadcast", "shape", "matmul", "aligned", "dimension")):
        return ShapeError(f"shape mismatch: {msg}")
    if isinstance(e, ValueError) and "truth value" in msg:
        return RuntimeErr("a tensor with several elements cannot be used as true/false; "
                          "use any(...) / all(...) style reductions")
    if isinstance(e, AttributeError) and getattr(e, "name", None) and hasattr(e, "obj"):
        return _no_attr(e.obj, e.name)
    if isinstance(e, IndexError):
        what = msg.split(" index")[0] if " index out of range" in msg else ""
        err = RuntimeErr(f"index out of range{' for this ' + _APSI_TYPES.get(what, what) if what else ''}"
                         f" (positions start at 0; [-1] is the last element; len() gives the size)"
                         if "out of range" in msg else f"index out of range: {msg}")
        err.focus = "["
        return err
    if isinstance(e, KeyError):
        key = e.args[0] if e.args else msg
        err = RuntimeErr(f"key not found: {key!r} (use d.get(key, default) when a key may be missing, "
                         f"or `key in d` to check first)")
        if isinstance(key, (str, int)):
            err.focus = repr(key) if isinstance(key, int) else None
            if isinstance(key, str):
                err.focus = f'"{key}"'
        return err
    if isinstance(e, TypeError):
        m = re.match(r'can only concatenate str \(not "(\w+)"\) to str', msg)
        if m:
            err = RuntimeErr(f"cannot join Str and {_APSI_TYPES.get(m.group(1), m.group(1).capitalize())} with '+'"
                             + _type_hint(msg))
            err.focus = "+"
            return err
        err = RuntimeErr(f"type error: {msg}" + _type_hint(msg))
        if "has no len()" in msg:
            err.focus = "len("
        elif "item assignment" in msg:
            err.focus = "["
        return err
    if isinstance(e, ValueError) and ("values to unpack" in msg):
        return RuntimeErr(f"{msg} (the number of names must match the items; for a dict's keys and values "
                          f"write: for k, v in d.items())")
    if isinstance(e, ValueError):
        m = re.match(r"invalid literal for (int|float)\(\)(?: with base \d+)?: (.*)", msg)
        if m:
            kind = "whole number" if m.group(1) == "int" else "number"
            sample = "'42'" if m.group(1) == "int" else "'3.5'"
            err = RuntimeErr(f"{m.group(2)} is not a {kind}: {m.group(1)}() reads text such as {sample}")
            err.focus = m.group(1) + "("
            return err
        if "could not convert string to float" in msg:
            err = RuntimeErr(f"{msg.split(': ', 1)[-1]} is not a number: float() reads text such as '3.5'")
            err.focus = "float("
            return err
    if isinstance(e, RecursionError):
        return RuntimeErr("maximum recursion depth exceeded")
    return RuntimeErr(f"{type(e).__name__}: {msg}")


class Interpreter:
    def __init__(self, permissions=None, policy=None):
        from .security import Policy
        self.policy = policy if policy is not None else Policy.from_set(
            set(permissions) if permissions is not None else {"fs"})
        self.testing = False
        self.globals = Env()
        core = stdlib.core_builtins()
        self.globals.vars.update(core)
        self.globals.vars["testing"] = lambda: self.testing
        from .helpers import builtins as _helper_builtins
        for _k, _v in _helper_builtins().items():
            self.globals.vars.setdefault(_k, _v)
        from .exceptions import builtins_table
        for _k, _v in builtins_table().items():
            self.globals.vars.setdefault(_k, _v)
        self.modules = LazyModules(stdlib.modules(core))
        from . import interop, crypto, hardware, sci, precision, data, concurrency, models
        interop.install(self)
        models.install(self)

        def _web():
            from . import web as _w
            _w.install(self)                               # sets modules["web"] (asyncio loads only when used)
            return dict.__getitem__(self.modules, "web")
        self.modules.lazy("web", _web)
        data.install(self)
        concurrency.install(self)
        self.modules["sci"] = sci.module()
        from . import layers
        self.modules["layers"] = layers.module(self)
        self.modules["precision"] = precision.module()
        for k, v in self.modules["precision"].items():
            if k not in ("f16", "f32", "f64"):
                self.globals.vars.setdefault(k, v)
        self.globals.vars["einsum"] = sci.einsum
        self.modules["crypto"] = crypto.module()
        from . import hp as _hp
        self.modules["hp"] = _hp.module()
        self.modules.lazy("rt", lambda: __import__("apsi.rt", fromlist=["module"]).module(self))
        self.modules.lazy("hub", lambda: __import__("apsi.hub", fromlist=["module"]).module(self))
        self.modules.lazy("tokens", lambda: __import__("apsi.tokens", fromlist=["module"]).module(self))
        self.modules.lazy("memory", lambda: __import__("apsi.memory", fromlist=["module"]).module(self))
        self.modules.lazy("log", lambda: __import__("apsi.log", fromlist=["module"]).module(self))
        self.modules.lazy("config", lambda: __import__("apsi.config", fromlist=["module"]).module(self))
        from . import structures as _st
        self.modules.lazy("collections", _st.module)
        for k, v in _st.FUNCTIONS.items():
            self.globals.vars.setdefault(k, v)
        self.modules.lazy("db", lambda: __import__("apsi.db", fromlist=["module"]).module(self))
        self.modules.lazy("env", lambda: __import__("apsi.envfile", fromlist=["module"]).module(self))
        self.modules.lazy("dataframe", lambda: __import__("apsi.dataframe", fromlist=["module"]).module(self))
        from . import pycompat as _pyc
        self.modules["pycompat"] = _pyc.module(self)
        self.globals.vars.setdefault("open", self.modules["pycompat"]["open"])   # checked by --allow-read/write
        self.modules["sys"] = hardware.module()
        self.modules["gpu"] = self.modules["sys"]
        from . import scale
        self.modules["scale"] = scale.module()
        # science, engineering and medicine (numpy only; see docs/SCIENCE.md)
        from . import special, stats, dsp, mechanics, pde, quantum, physics, medical, lab
        self.modules["special"] = special.module()
        self.modules["stats"] = stats.module()
        self.modules["signal"] = dsp.module()
        self.modules["mechanics"] = mechanics.module()
        self.modules["pde"] = pde.module()
        self.modules["quantum"] = quantum.module()
        self.modules["medical"] = dict(medical.module(), **medical.install(self))
        self.modules["lab"] = dict(lab.module(), **lab.install(self))
        from . import torch_compat, rl
        self.modules["torch"] = torch_compat.module()
        self.modules["rl"] = rl.module()
        formulas = physics.module()
        self.modules["physics"].update(formulas)
        self.modules["physics.formulas"] = formulas
        for k in ("gamma", "lgamma", "erf", "erfc", "erfinv", "factorial", "comb", "beta", "besselj", "legendre"):
            self.modules["math"][k] = self.modules["special"][k]
        self.exact = False
        self.deriv_stack = []
        self.tests = []
        self.module_cache = {}
        self.importing = []          # files being imported right now (detects import cycles)
        self._handling = []          # errors being handled by `except` blocks (for a bare `raise`)
        self.dir_stack = [os.getcwd()]
        self._ev = {}
        self._ex = {}
        for cls in A.ALL:
            m = getattr(self, "ev_" + cls.__name__, None)
            if m:
                self._ev[cls] = m
            m = getattr(self, "ex_" + cls.__name__, None)
            if m:
                self._ex[cls] = m

    # ================================================================ entry
    engine = "compile"

    @property
    def permissions(self):
        return self.policy.as_set()

    def run(self, source, filename="<main>", check=True, env=None, engine=None):
        prog = parse(source, cache=not filename.startswith("<"))
        if getattr(self, "_main_dir", None) is None:
            self._main_dir = os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else \
                os.getcwd()
        engine = engine or self.engine
        from .checker import is_python_mode
        if is_python_mode(prog) and not filename.startswith("<repl"):
            env = None                                     # Python-exact files always compile
            engine = "compile"
        if env is None and engine == "compile":
            self._src_for_check = source
            return self._run_compiled(prog, filename, check)
        if check:
            from .checker import Checker
            errs = Checker(self, os.path.dirname(os.path.abspath(filename))
                           if not filename.startswith("<") else os.getcwd()).check(prog)
            if errs:
                raise CheckFailed(errs)
        env = env or Env(self.globals)
        base = os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd()
        self.dir_stack.append(base)
        try:
            self.exec_block(prog.body, env)
        except ReturnSignal:
            raise RuntimeErr("'return' outside of a function")
        except (BreakSignal, ContinueSignal):
            raise RuntimeErr("'break'/'continue' outside of a loop")
        finally:
            self.dir_stack.pop()
        return env

    def _check(self, prog, filename):
        from .checker import Checker
        from . import check_cache
        base = os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd()
        src, self._src_for_check = getattr(self, "_src_for_check", None), None
        if src is not None and check_cache.known_clean(src, base, self):
            return base
        ch = Checker(self, base)
        errs = ch.check(prog)
        if errs:
            raise CheckFailed(errs)
        if src is not None:
            check_cache.remember(src, base, self, ch.deps)
        return base

    def _run_compiled(self, prog, filename, check):
        from .codegen import Compiled
        base = self._check(prog, filename) if check else (
            os.path.dirname(os.path.abspath(filename)) if not filename.startswith("<") else os.getcwd())
        self.dir_stack.append(base)
        try:
            comp = Compiled(self, prog, filename, base, self.tests)
            self.last_compiled = comp
            comp.run()
        finally:
            self.dir_stack.pop()
        env = Env(self.globals)
        env.vars.update(comp.exports())
        env.module = comp.module
        return env

    def _enter_import(self, path):
        if path in self.importing:
            chain = self.importing[self.importing.index(path):] + [path]
            raise NameErr("circular import: " + " -> ".join(os.path.basename(p) for p in chain)
                          + "; move the shared code into a third file that both import")
        self.importing.append(path)

    def import_module(self, path):
        """`import "x.apsi" as m` in a Python-exact file: the module object itself (m._private works, objects
        pickle by reference, the module is in sys.modules under its file name)."""
        self.import_compiled(path)
        env = self.module_cache[os.path.normpath(path)]
        mod = getattr(env, "module", None)
        if mod is None:
            from .values import Namespace
            return Namespace(os.path.basename(path), dict(env.vars))
        return mod

    def _shadow_name(self, dotted, path):
        """A module whose Python name is already taken (a local json/ next to the program, say) lives under a
        private top-level package, so `pkg.sub` attributes and pickling still work."""
        import sys as _sys
        import types as _types
        if not all(p.isidentifier() for p in dotted.split(".")):
            return f"apsi_{abs(hash(path)):x}"
        root = f"apsi_{abs(hash(getattr(self, '_main_dir', '') or '')):x}"
        if root not in _sys.modules:
            pkg = _types.ModuleType(root)
            pkg.__path__ = []
            _sys.modules[root] = pkg
        parts = dotted.split(".")
        for i in range(1, len(parts)):                    # parents that are not modules of their own yet
            pname = root + "." + ".".join(parts[:i])
            if pname not in _sys.modules:
                m = _types.ModuleType(pname)
                m.__path__ = []
                _sys.modules[pname] = m
        return root + "." + dotted

    def _dotted_name(self, path):
        """Python's module name for a file: `pkg/sub.apsi` is `pkg.sub` when pkg is a package (has __init__),
        `pkg/__init__.apsi` is `pkg`; the main program's own folder is never part of the name."""
        path = os.path.abspath(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        top = getattr(self, "_main_dir", None)
        d = os.path.dirname(path)
        parts = [] if stem == "__init__" else [stem]
        while d != top and any(os.path.exists(os.path.join(d, f"__init__.{e}")) for e in ("apsi", "py")):
            parts.insert(0, os.path.basename(d))
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
        return ".".join(parts) or stem

    def _library_finder(self, path):
        """Python's own imports (__import__, importlib.import_module) of a library converted into apsi_packages/
        load its .apsi modules, so code that imports by name at run time finds the converted library."""
        parts = path.split(os.sep)
        if "apsi_packages" not in parts:
            return
        root = os.sep.join(parts[:len(parts) - parts[::-1].index("apsi_packages")])
        import sys as _sys
        if any(getattr(f, "apsi_root", None) == root for f in _sys.meta_path):
            return
        interp = self

        class _Finder:
            apsi_root = root

            def find_spec(self, name, target_path=None, target=None):
                import importlib.machinery
                base = os.path.join(root, *name.split("."))
                for cand in (base + ".apsi", os.path.join(base, "__init__.apsi")):
                    if os.path.isfile(cand):
                        return importlib.machinery.ModuleSpec(name, _Loader(cand), origin=cand,
                                                              is_package=cand.endswith("__init__.apsi"))
                return None

        class _Loader:
            def __init__(self, file):
                self.file = file

            def create_module(self, spec):
                interp.import_compiled(self.file)
                return _sys.modules.get(spec.name)

            def exec_module(self, module):
                pass

        _sys.meta_path.insert(0, _Finder())

    def import_compiled(self, path):
        path = os.path.normpath(path)
        self._library_finder(os.path.abspath(path))
        if path not in self.module_cache:
            import sys as _sys
            stem = os.path.splitext(os.path.basename(path))[0]
            dotted = self._dotted_name(path)
            taken = _sys.modules.get(dotted)
            name = dotted if all(p.isidentifier() for p in dotted.split(".")) and (
                taken is None or os.path.normpath(getattr(taken, "__file__", "") or "") == os.path.abspath(path)) \
                else self._shadow_name(dotted, path)
            self._next_module_name = name
            if not os.path.exists(path):
                raise NameErr(f"cannot find file '{path}'")
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
            saved = self.tests
            self.tests = []
            self._enter_import(path)
            try:
                menv = self.run(src, path, check=True)
            except ApsiError as e:
                if e.filename is None:
                    e.filename, e.source = path, src
                raise
            finally:
                self.tests = saved
                self.importing.pop()
            self.module_cache[path] = menv
            mod = getattr(menv, "module", None)
            if mod is not None:
                _sys.modules.setdefault(mod.__name__, mod)
                parent, _, leaf = mod.__name__.rpartition(".")
                if parent and parent in _sys.modules:
                    setattr(_sys.modules[parent], leaf, mod)   # importing pkg.sub makes it pkg's attribute
        return {k: v for k, v in self.module_cache[path].vars.items() if not k.startswith("_")}

    def run_tests(self, report=print):
        passed, failed = 0, []
        for item in self.tests:
            if len(item) == 3:
                name, fn, line = item
                t0 = time.perf_counter()
                try:
                    try:
                        fn()
                    except ApsiError:
                        raise
                    except Exception as e:
                        raise self.last_compiled.translate(e, sys.exc_info()[2]) from None
                    report(f"  ✓ {name}  ({(time.perf_counter() - t0) * 1000:.1f} ms)")
                    passed += 1
                except ApsiError as e:
                    report(f"  ✗ {name}\n      {e.kind}: {e.msg}" + (f" (line {e.line})" if e.line else ""))
                    failed.append((name, e))
                continue
            name, body, env, line = item
            t0 = time.perf_counter()
            try:
                self.exec_block(body, Env(env))
                dt = (time.perf_counter() - t0) * 1000
                report(f"  ✓ {name}  ({dt:.1f} ms)")
                passed += 1
            except ApsiError as e:
                report(f"  ✗ {name}\n      {e.kind}: {e.msg}" + (f" (line {e.line})" if e.line else ""))
                failed.append((name, e))
        return passed, failed

    # ================================================================ statements
    def exec_block(self, stmts, env):
        ex = self._ex
        for s in stmts:
            try:
                ex[s.__class__](s, env)
            except ControlSignal:
                raise
            except ApsiError as e:
                if e.line is None:
                    e.line, e.col = s.line, s.col
                raise
            except Exception as e:
                err = wrap_python_error(e)
                err.line, err.col = s.line, s.col
                raise err from None

    def ex_ExprStmt(self, s, env):
        v = self.eval(s.expr, env)
        if getattr(self, "repl_echo", False) and v is not None and env.parent is self.globals:
            print(fmt(v))
            env.define("_", v)

    def ex_Let(self, s, env):
        if s.value is None:
            return None
        v = self.eval(s.value, env)
        if len(s.targets) == 1:
            if s.type is not None:
                v = self.check_type(v, s.type, {}, f"'{s.targets[0]}'", env)
            env.define(s.targets[0], v, const=not s.mutable)
        elif any(isinstance(n, tuple) for n in s.targets):
            bind_pattern(list(s.targets), v, lambda n, x: env.define(n, x, const=not s.mutable))
        else:
            vals = self._unpack(v, len(s.targets))
            for n, x in zip(s.targets, vals):
                env.define(n, x, const=not s.mutable)

    def _unpack(self, v, n):
        try:
            vals = list(v)
        except TypeError:
            raise RuntimeErr(f"cannot unpack {type_name(v)} into {n} names")
        if len(vals) != n:
            raise RuntimeErr(f"expected {n} values to unpack, got {len(vals)}")
        return vals

    def ex_Assign(self, s, env):
        v = self.eval(s.value, env)
        self.assign(s.target, v, env)
        for t in s.more or ():
            self.assign(t, v, env)

    def assign(self, t, v, env):
        if isinstance(t, A.Name):
            env.assign(t.id, v)
        elif isinstance(t, (A.TupleLit, A.ListLit)) and any(isinstance(x, A.Spread) for x in t.items):
            vals = list(v)
            star = next(i for i, x in enumerate(t.items) if isinstance(x, A.Spread))
            after = len(t.items) - star - 1
            if len(vals) < len(t.items) - 1:
                raise RuntimeErr(f"expected at least {len(t.items) - 1} values to unpack, got {len(vals)}")
            for sub, x in zip(t.items[:star], vals[:star]):
                self.assign(sub, x, env)
            self.assign(t.items[star].value, vals[star:len(vals) - after], env)
            for sub, x in zip(t.items[star + 1:], vals[len(vals) - after:]):
                self.assign(sub, x, env)
        elif isinstance(t, (A.TupleLit, A.ListLit)):
            vals = self._unpack(v, len(t.items))
            for sub, x in zip(t.items, vals):
                self.assign(sub, x, env)
        elif isinstance(t, A.Attr):
            obj = self.eval(t.obj, env)
            self.set_attr(obj, t.name, v)
        elif isinstance(t, A.Index):
            obj = self.eval(t.obj, env)
            idx = self.eval_index(t.index, env)
            self.set_index(obj, idx, v)
        else:
            raise RuntimeErr("invalid assignment target")

    def set_index(self, obj, idx, v):
        t = type(obj)
        if t is dict or t is list:
            obj[idx] = v
            return
        if isinstance(obj, np.ndarray) and isinstance(v, Quantity):
            raise UnitError(f"cannot store {dim_name(v.dim)} into a dimensionless tensor")
        if isinstance(obj, np.ndarray) and isinstance(v, Fraction):
            v = float(v)
        obj[idx] = v

    @staticmethod
    def _own_object(obj, name=None):
        """An instance (or class) defined by an Apsi program: its _private attributes are its own business. Also the
        public underscore APIs of the standard library (namedtuple._replace, Enum._value_ …)."""
        if name is not None and (name in _PUBLIC_UNDERSCORE or (len(name) > 2 and name.endswith("_")
                                                                 and not name.startswith("__"))):
            return True
        from .codegen import is_own
        return is_own(obj)

    def del_attr(self, obj, name):
        from .pycompat import check_attr, trusted
        if name.startswith("_") and not (name.startswith("__") or trusted(self) or self._own_object(obj, name)):
            raise RuntimeErr(f"cannot delete the private attribute '{name}'")
        check_attr(self, name)
        delattr(obj, name)

    def set_attr(self, obj, name, v):
        if obj.__class__ in _PLAIN_SET and name[0] != "_":
            try:
                setattr(obj, name, v)
                return
            except (AttributeError, TypeError) as e:
                raise RuntimeErr(f"cannot set attribute '{name}' on {type_name(obj)}: {e}") from None
        if name.startswith("_") and not isinstance(obj, (StructInstance, Record)):
            from .pycompat import check_attr, trusted
            check_attr(self, name)
            if trusted(self) or self._own_object(obj, name):
                if name in ("__class__", "__code__", "__globals__") and not trusted(self):
                    raise RuntimeErr(f"setting '{name}' needs --allow-python=all")
                setattr(obj, name, v)
                return
        if isinstance(obj, StructInstance):
            if name not in obj.fields:
                raise RuntimeErr(f"{obj.type.name} has no field '{name}'")
            obj.fields[name] = v
        elif isinstance(obj, Record):
            obj.fields[name] = v
        else:
            from .stdlib import _safe_setattr
            _safe_setattr(obj, name, v)                     # refuses modules, classes, functions, built-ins
            if not name.startswith("_"):
                _PLAIN_SET.add(obj.__class__)

    def ex_AugAssign(self, s, env):
        fn = IOPS[s.op] if s.op in ("|", "&", "<<", ">>", "^") else BINOPS[s.op]
        t = s.target
        rhs = self.eval(s.value, env)
        if isinstance(t, A.Name):
            env.assign(t.id, fn(self.lookup(t.id, env, t), rhs))
        elif isinstance(t, A.Attr):
            obj = self.eval(t.obj, env)
            self.set_attr(obj, t.name, fn(self.get_attr(obj, t.name), rhs))
        else:
            obj = self.eval(t.obj, env)
            idx = self.eval_index(t.index, env)
            self.set_index(obj, idx, fn(obj[idx], rhs))

    def ex_FnDef(self, s, env):
        f = Function(self, s.name, s.params, s.ret, s.body, s.is_expr, env, s.shape_params)
        for d in reversed(s.decorators):
            deco = self.eval(d, env)
            f = self.call_value(deco, [f], {})
        env.define(s.name, f)

    def ex_StructDef(self, s, env):
        st = StructType(self, s.name, [(p.name, p.type, p.default) for p in s.fields], {}, env)
        env.define(s.name, st)
        for m in s.methods:
            f = Function(self, m.name, m.params, m.ret, m.body, m.is_expr, env, m.shape_params)
            for d in reversed(m.decorators):
                f = self.call_value(self.eval(d, env), [f], {})
            st.methods[m.name] = f

    def ex_If(self, s, env):
        for cond, body in s.branches:
            if self.truthy(self.eval(cond, env)):
                self.exec_block(body, Env(env))
                return
        if s.else_body is not None:
            self.exec_block(s.else_body, Env(env))

    def truthy(self, v):
        if isinstance(v, (np.ndarray, Node)) and np.size(ops.raw(v)) != 1:
            raise RuntimeErr("a tensor with several elements cannot be used as a condition")
        return bool(v)

    def ex_Try(self, s, env):
        from . import exceptions as X
        try:
            try:
                self.exec_block(s.body, Env(env))
            except ControlSignal:
                raise
            except Exception as ex:                    # noqa: BLE001 - matched against the except clauses below
                for kinds, name, body in s.handlers:
                    classes = ()
                    if kinds:
                        cl = []
                        for k in kinds:
                            if X.ALIASES.get(k, k).split(".")[-1] in X.KINDS:
                                continue
                            try:
                                v = self.lookup(k.split(".")[0], env, s)
                                for part in k.split(".")[1:]:
                                    v = getattr(v, part)
                                if isinstance(v, type) and issubclass(v, BaseException):
                                    cl.append(v)
                            except Exception:          # noqa: BLE001 - not a class: matched by name
                                pass
                        classes = tuple(cl)
                    if not kinds or X.matches(getattr(ex, "original", None) or ex, kinds, classes) \
                            or X.matches(ex, kinds, classes):
                        henv = Env(env)
                        if name:
                            henv.vars[name] = X.value(getattr(ex, "original", None) or ex)
                        self._handling.append(ex)
                        try:
                            self.exec_block(body, henv)
                        finally:
                            self._handling.pop()
                        break
                else:
                    raise
            else:
                if s.else_body is not None:
                    self.exec_block(s.else_body, Env(env))
        finally:
            if s.final_body is not None:
                self.exec_block(s.final_body, Env(env))

    def ex_Raise(self, s, env):
        from . import exceptions as X
        if s.value is None:
            if not self._handling:
                raise RuntimeErr("'raise' without a value is only allowed inside an 'except' block")
            raise self._handling[-1]
        ex = X.to_raise(self.eval(s.value, env))
        if s.cause is not None:
            raise ex from X.to_raise(self.eval(s.cause, env))
        raise ex

    def ex_While(self, s, env):
        while self.truthy(self.eval(s.cond, env)):
            try:
                self.exec_block(s.body, Env(env))
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        else:
            if s.else_body:
                self.exec_block(s.else_body, Env(env))

    def ex_With(self, s, env):
        from contextlib import ExitStack
        from .codegen import _ctx
        wenv = Env(env)
        with ExitStack() as stack:
            for e, target in s.items:
                v = stack.enter_context(_ctx(self.eval(e, env)))
                if target is not None:
                    bind_pattern(target, v, lambda n, x: wenv.define(n, x))
            self.exec_block(s.body, wenv)

    def ex_Del(self, s, env):
        for t in s.targets:
            if isinstance(t, A.Name):
                e = env
                while e is not None and t.id not in e.vars:
                    e = e.parent
                if e is None:
                    raise NameErr(f"'{t.id}' is not defined", t.line, t.col)
                del e.vars[t.id]
            elif isinstance(t, A.Attr):
                self.del_attr(self.eval(t.obj, env), t.name)
            elif isinstance(t, A.Index):
                del self.eval(t.obj, env)[self.eval_index(t.index, env)]

    def ex_Global(self, s, env):
        pass

    def ex_ClassDef(self, s, env):
        """A real Python class whose methods are Apsi functions."""
        bases = tuple(self.eval(b, env) for b in s.bases)
        kw = {k: self.eval(v, env) for k, v in s.kwargs if k is not None}
        ns = {"__module__": "__apsi__", "__qualname__": s.name}
        cenv = Env(env)
        menv_all = []
        annotations = {}
        interp = self
        for item in s.body:
            if isinstance(item, A.FnDef):
                menv = Env(env)                            # holds __class__ for zero-argument super()
                if item.params and item.params[0].name:
                    menv.vars["__selfname__"] = item.params[0].name
                menv_all.append(menv)
                f = Function(self, item.name, item.params, item.ret, item.body, item.is_expr, menv, item.shape_params)

                def method(*a, _f=f, **k):
                    return interp.call_value(_f, list(a), k)
                method.__name__ = method.__qualname__ = item.name
                m = method
                for d in reversed(item.decorators):
                    m = self.call_value(self.eval(d, cenv), [m], {})
                ns[item.name] = m
                cenv.vars[item.name] = m
            elif isinstance(item, A.FieldDef):
                annotations[item.name] = item.annotation[1]
                if item.value is not None:
                    ns[item.name] = cenv.vars[item.name] = self.eval(item.value, cenv)
            elif isinstance(item, A.ExprStmt) and isinstance(item.expr, A.Str) and "__doc__" not in ns:
                ns["__doc__"] = item.expr.value
            else:
                self.exec_stmt(item, cenv) if hasattr(self, "exec_stmt") else self.exec_block([item], cenv)
                ns.update({k: v for k, v in cenv.vars.items() if k not in ns})
        if annotations:
            ns["__annotations__"] = annotations
        meta = kw.pop("metaclass", type(bases[0]) if bases else type)
        cls = meta(s.name, bases or (object,), ns, **kw)
        for menv in menv_all:
            menv.vars["__class__"] = cls
        for d in reversed(s.decorators):
            cls = self.call_value(self.eval(d, env), [cls], {})
        from .codegen import own_class
        own_class(cls)
        env.define(s.name, cls)

    def ex_For(self, s, env):
        it = self.eval(s.iter, env)
        if isinstance(it, (int, float)) and not isinstance(it, bool):
            raise RuntimeErr("cannot iterate over a number; use 0..n or range(n)")
        if isinstance(it, Record):
            it = list(it.fields.items())
        pat = s.targets[0] if len(s.targets) == 1 else list(s.targets)
        for item in it:
            loop_env = Env(env)
            bind_pattern(pat, item, lambda n, x: loop_env.define(n, x, const=True))
            try:
                self.exec_block(s.body, loop_env)
            except BreakSignal:
                break
            except ContinueSignal:
                continue
        else:
            if s.else_body:
                self.exec_block(s.else_body, Env(env))

    def ex_Return(self, s, env):
        raise ReturnSignal(self.eval(s.value, env) if s.value is not None else None)

    def ex_Break(self, s, env):
        raise BreakSignal()

    def ex_Continue(self, s, env):
        raise ContinueSignal()

    def ex_Pass(self, s, env):
        pass

    def ensure_module(self, name, base=None):
        """Load an installed .apsil library the first time `use NAME` names it. True when NAME is available."""
        if name in self.modules:
            return True
        from . import apsil
        try:
            entry = apsil.resolve_use(name, base or self.dir_stack[-1])
        except apsil.PackageError as e:
            raise NameErr(str(e))
        if not entry:
            return False
        with open(entry, encoding="utf-8-sig") as fh:
            src = fh.read()
        saved = self.tests
        self.tests = []
        self._enter_import(entry)
        try:
            menv = self.run(src, entry, check=True, env=Env(self.globals))
        except ApsiError as e:
            if e.filename is None:
                e.filename, e.source = entry, src
            raise
        finally:
            self.importing.pop()
            self.tests = saved
        self.modules[name] = {k: v for k, v in menv.vars.items() if not k.startswith("_")}
        return True

    def ex_Use(self, s, env):
        if s.path not in self.modules and not self.ensure_module(s.path):
            raise NameErr(f"unknown module '{s.path}'. Available: {', '.join(sorted(self.modules))}")
        members = self.modules[s.path]
        if s.alias:
            env.define(s.alias, Namespace(s.path, members))
        else:
            for k, v in members.items():
                env.define(k, v)
            if "." not in s.path and s.path not in members and s.path not in env.vars and s.path not in self.globals.vars:
                env.define(s.path, Namespace(s.path, members))      # `use rl` also allows rl.ppo(...)

    def ex_Import(self, s, env):
        path = s.path
        if not os.path.isabs(path):
            path = os.path.join(self.dir_stack[-1], path)
        path = os.path.normpath(path)
        if path not in self.module_cache:
            if not os.path.exists(path):
                raise NameErr(f"cannot find file '{s.path}'")
            with open(path, encoding="utf-8-sig") as fh:
                src = fh.read()
            self._enter_import(path)
            try:
                menv = self.run(src, path, check=True, env=Env(self.globals))
            except ApsiError as e:
                if e.filename is None:
                    e.filename, e.source = path, src
                raise
            finally:
                self.importing.pop()
            self.module_cache[path] = menv
        menv = self.module_cache[path]
        public = {k: v for k, v in menv.vars.items() if not k.startswith("_")}
        if s.alias:
            env.define(s.alias, Namespace(os.path.basename(path), public))
        else:
            for k, v in public.items():
                env.define(k, v)

    def ex_Test(self, s, env):
        self.tests.append((s.name, s.body, env, s.line))

    def ex_Assert(self, s, env):
        v = self.eval(s.cond, env)
        if isinstance(v, (np.ndarray, Quantity, Node)):
            ok = bool(np.all(ops.raw(v)))
        else:
            ok = bool(v)
        if not ok:
            msg = fmt(self.eval(s.msg, env)) if s.msg is not None else self._assert_detail(s.cond, env)
            raise AssertionFailed(msg, s.line, s.col)

    def _assert_detail(self, cond, env):
        if isinstance(cond, A.Binary) and cond.op in ("==", "!=", "<", "<=", ">", ">=", "~="):
            try:
                a, b = self.eval(cond.left, env), self.eval(cond.right, env)
                return f"assertion failed: {fmt(a)} {cond.op} {fmt(b)}"
            except Exception:
                pass
        return "assertion failed"

    def ex_Deriv(self, s, env):
        if not self.deriv_stack:
            raise RuntimeErr(f"{s.name}' = ... is only allowed inside an evolve block")
        self.deriv_stack[-1][s.name] = self.eval(s.expr, env)

    # ---------------------------------------------------------------- evolve
    def ex_Evolve(self, s, env):
        names = s.states
        state = [self.lookup(n, env, s) for n in names]
        rng = self.eval(s.range, env)
        if not isinstance(rng, RangeVal):
            raise RuntimeErr("evolve needs a time range like 0[s]..10[s]")
        dt = self.eval(s.step, env)
        t0, t1 = rng.start, rng.end
        span = t1 - t0
        try:
            nsteps = span / dt
        except UnitError:
            raise UnitError(f"step has units {dim_name(ops.dim_of(dt))} but the time range has "
                            f"{dim_name(ops.dim_of(t0))}")
        if isinstance(nsteps, Quantity):
            raise UnitError("time range and step must have the same units")
        nsteps = int(round(float(nsteps)))
        if nsteps <= 0:
            raise RuntimeErr("evolve: the time range must be longer than one step")
        every = int(self.eval(s.every, env)) if s.every is not None else 1
        tdim = ops.dim_of(dt)

        body, tvar = s.body, s.tvar
        checked = [False]

        def deriv(t, st):
            e = Env(env)
            e.vars[tvar] = t
            e.consts.add(tvar)
            for n, v in zip(names, st):
                e.vars[n] = v
                e.consts.add(n)
            d = {}
            self.deriv_stack.append(d)
            try:
                self.exec_block(body, e)
            finally:
                self.deriv_stack.pop()
            out = []
            for n in names:
                if n not in d:
                    raise RuntimeErr(f"evolve: no derivative for '{n}' (add a line {n}' = ...)", s.line, s.col)
                out.append(d[n])
            if not checked[0]:
                checked[0] = True
                for n, v, dv in zip(names, st, out):
                    want = ops.dim_of(v) / tdim
                    got = ops.dim_of(dv)
                    if got is not None and got != want:
                        raise UnitError(f"{n}' must have units of {dim_name(want)} "
                                        f"({n} per unit time), but got {dim_name(got)}", s.line, s.col)
            return out

        # ---- fast path: units checked on the first evaluation, then erased
        state_dims = [ops.dim_of(v) for v in state]
        fast = None
        if all(d is not None for d in state_dims) and tdim is not None:
            from .compiler import ErasedCompiler, Unsupported
            try:
                comp = ErasedCompiler(self)
                code = comp.block(body)
                # one compiled block per state that skips the other states' derivatives
                partial = {n: comp.block([b for b in body if not (isinstance(b, A.Deriv) and b.name != n)])
                           for n in names}
                deriv(t0, state)          # full, unit-checked evaluation

                def fast_deriv(t, st, only=None):
                    e = Env(env)
                    e.vars[tvar] = t
                    for n, v in zip(names, st):
                        e.vars[n] = v
                    d = {}
                    self.deriv_stack.append(d)
                    try:
                        (code if only is None else partial[only])(e)
                    finally:
                        self.deriv_stack.pop()
                    if only is not None:
                        return d[only]
                    return [d[n] for n in names]
                raw_state = [ops.raw(v) if not isinstance(v, Quantity) else v.v for v in state]
                fast_deriv(ops.raw(t0) if isinstance(t0, Quantity) else t0, raw_state)   # probe
                fast = fast_deriv
            except Unsupported:
                fast = None
            except ApsiError:
                raise
        if fast is not None:
            deriv = fast
            st_wrap = state_dims
            state = [v.v if isinstance(v, Quantity) else v for v in state]
            t0 = t0.v if isinstance(t0, Quantity) else t0
            dt = dt.v if isinstance(dt, Quantity) else dt
        else:
            st_wrap = None

        def axpy(xs, ds, h):
            return [x + h * d for x, d in zip(xs, ds)]

        rec_t, rec = [], {n: [] for n in names}

        def record(t, st):
            rec_t.append(t)
            for n, v in zip(names, st):
                rec[n].append(v.copy() if isinstance(v, np.ndarray) else
                              (Quantity(v.v.copy(), v.dim) if isinstance(v, Quantity) and isinstance(v.v, np.ndarray) else v))

        def one(t, st, i):
            if fast is not None:
                return deriv(t, st, names[i])
            return deriv(t, st)[i]

        method = s.method
        if method == "verlet" and len(names) != 2:
            raise RuntimeErr("verlet needs exactly two states: position, velocity")
        st = list(state)
        if s.into:
            record(t0, st)
        if method == "rk45":
            from .integrators import rk45
            tol = float(self.eval(s.tol, env)) if s.tol is not None else 1e-8
            counter = [0]

            def on_step(t, cur):
                counter[0] += 1
                if s.into and counter[0] % every == 0:
                    record(t, cur)
            t_end = t0 + nsteps * dt
            st, _, _ = rk45(deriv, t0, t_end, dt, st, tol, on_step)
            if s.into and counter[0] % every != 0:
                record(t_end, st)
            nsteps = 0
        for k in range(nsteps):
            t = t0 + k * dt
            if method == "euler":
                st = axpy(st, deriv(t, st), dt)
            elif method == "midpoint":
                k1 = deriv(t, st)
                st = axpy(st, deriv(t + 0.5 * dt, axpy(st, k1, 0.5 * dt)), dt)
            elif method == "rk4":
                k1 = deriv(t, st)
                k2 = deriv(t + 0.5 * dt, axpy(st, k1, 0.5 * dt))
                k3 = deriv(t + 0.5 * dt, axpy(st, k2, 0.5 * dt))
                k4 = deriv(t + dt, axpy(st, k3, dt))
                st = [x + (dt / 6.0) * (a + 2 * b + 2 * c + d)
                      for x, a, b, c, d in zip(st, k1, k2, k3, k4)]
            elif method == "symplectic":
                for i in reversed(range(len(st))):
                    st[i] = st[i] + dt * one(t, st, i)
            elif method == "verlet":
                x, v = st
                a0 = one(t, [x, v], 1)
                vh = v + (0.5 * dt) * a0
                x = x + dt * one(t, [x, vh], 0)
                v = vh + (0.5 * dt) * one(t + dt, [x, vh], 1)
                st = [x, v]
            if s.into and ((k + 1) % every == 0 or k == nsteps - 1):
                record(t0 + (k + 1) * dt, st)
        if st_wrap is not None:
            st = [make(v, d) for v, d in zip(st, st_wrap)]
        for n, v in zip(names, st):
            env.assign(n, v)
        if s.into:
            ts = ops.stack(rec_t)
            fields = {"t": make(ts, tdim) if st_wrap is not None else ts}
            for n, d in zip(names, state_dims):
                arr = ops.stack(rec[n])
                fields[n] = make(arr, d) if st_wrap is not None else arr
            env.define(s.into, Record(**fields))

    # ================================================================ expressions
    def eval(self, node, env):
        return self._ev[node.__class__](node, env)

    def ev_Num(self, n, env):
        if n.unit is not None:
            text, scale, dim = n.unit
            if dim.dimensionless:
                return n.value * scale
            return Quantity(n.value * scale, dim, (text, scale))
        if self.exact:
            return Fraction(n.text)
        return n.value

    def ev_Str(self, n, env):
        return n.value

    def ev_FStr(self, n, env):
        from .codegen import _conv
        out = []
        for p in n.parts:
            if isinstance(p, str):
                out.append(p)
            else:
                e, spec = p[0], p[1]
                v = self.eval(e, env)
                if len(p) > 2:
                    v = _conv(v, p[2])
                if isinstance(spec, A.FStr):
                    spec = self.ev_FStr(spec, env)
                out.append(fmt(v, True, spec))
        return "".join(out)

    def ev_Bytes(self, n, env):
        return n.value

    def ev_SetLit(self, n, env):
        out = set()
        for x in n.items:
            if isinstance(x, A.Spread):
                out.update(self.eval(x.value, env))
            else:
                out.add(self.eval(x, env))
        return out

    def ev_SetComp(self, n, env):
        return {self.eval(n.elem, e) for e in self._comp_envs(n, env)}

    def ev_GenExp(self, n, env):
        return (self.eval(n.elem, e) for e in self._comp_envs(n, env))

    def ev_Chain(self, n, env):
        a = self.eval(n.operands[0], env)
        for op, x in zip(n.ops, n.operands[1:]):
            b = self.eval(x, env)
            if not self.truthy(BINOPS[op](a, b)):
                return False
            a = b
        return True

    def ev_NamedExpr(self, n, env):
        v = self.eval(n.value, env)
        try:
            env.assign(n.name, v)
        except Exception:                                  # noqa: BLE001 - first use: a new variable
            env.define(n.name, v)
        return v

    def ev_Bool(self, n, env):
        return n.value

    def ev_NoneLit(self, n, env):
        return None

    def lookup(self, name, env, node):
        try:
            return env.lookup(name)
        except KeyError:
            raise NameErr(f"'{name}' is not defined", node.line, node.col)

    def ev_Name(self, n, env):
        e = env
        name = n.id
        while e is not None:
            v = e.vars
            if name in v:
                return v[name]
            e = e.parent
        raise NameErr(f"'{name}' is not defined", n.line, n.col)

    def ev_Unary(self, n, env):
        v = self.eval(n.operand, env)
        if n.op == "-":
            return -v
        if n.op == "+":
            return +v
        if n.op == "~":
            return ~v
        return not self.truthy(v)

    def ev_Binary(self, n, env):
        a = self.eval(n.left, env)
        b = self.eval(n.right, env)
        try:
            if n.op == "**" and isinstance(a, Fraction) and isinstance(b, Fraction) and b.denominator != 1:
                return float(a) ** float(b)
            return BINOPS[n.op](a, b)
        except ControlSignal:
            raise
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            if isinstance(e, TypeError):
                err = RuntimeErr(f"cannot apply '{n.op}' to {type_name(a)} and {type_name(b)}")
            err.line, err.col = n.line, n.col
            raise err from None

    def ev_Logic(self, n, env):
        a = self.eval(n.left, env)
        if n.op == "and":
            return self.eval(n.right, env) if self.truthy(a) else a
        return a if self.truthy(a) else self.eval(n.right, env)

    def ev_IfExpr(self, n, env):
        return self.eval(n.a, env) if self.truthy(self.eval(n.cond, env)) else self.eval(n.b, env)

    def ev_Call(self, n, env):
        if isinstance(n.func, A.Name) and n.func.id == "super" and not n.args and not n.kwargs:
            e = env                                        # zero-argument super() inside a method
            while e is not None and "__class__" not in e.vars:
                e = e.parent
            if e is not None:
                first = e.vars.get("__selfname__")
                se = env
                while se is not None and first not in se.vars:
                    se = se.parent
                if se is not None:
                    return super(e.vars["__class__"], se.vars[first])
        f = self.eval(n.func, env)
        args = []
        for a in n.args:
            if isinstance(a, A.Spread):
                args.extend(self.eval(a.value, env))
            else:
                args.append(self.eval(a, env))
        kwargs = {}
        for k, v in n.kwargs:
            if k is None:
                kwargs.update(self.eval(v.value, env))
            else:
                kwargs[k] = self.eval(v, env)
        try:
            return self.call_value(f, args, kwargs)
        except ControlSignal:
            raise
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            err.line, err.col = n.line, n.col
            raise err from None

    def call_value(self, f, args, kwargs):
        if isinstance(f, Function):
            return self.call_function(f, args, kwargs)
        if callable(f):
            return f(*args, **kwargs)
        raise RuntimeErr(f"{type_name(f)} is not callable")

    def call_function(self, fn, args, kwargs):
        if not fn.is_expr and is_generator(fn.body):
            return self._generator(fn, args, kwargs)
        return self._call_function(fn, args, kwargs)

    def _generator(self, fn, args, kwargs):
        """A function containing `yield`: values are produced lazily (infinite generators work)."""
        import queue
        import threading
        q = queue.Queue(maxsize=1)
        DONE = object()
        interp = self
        gen_self = {"stop": False}

        def run():
            interp._yield_q = getattr(interp, "_yield_q", {})
            interp._yield_q[threading.get_ident()] = (q, gen_self)
            try:
                interp._call_function(fn, args, kwargs)
                q.put((DONE, None))
            except BaseException as e:               # noqa: BLE001 - handed to the consumer
                q.put((DONE, e))

        def gen():
            t = threading.Thread(target=run, daemon=True)
            t.start()
            try:
                while True:
                    tag, v = q.get()
                    if tag is DONE:
                        if v is not None and not isinstance(v, _GenStop):
                            raise v
                        return
                    yield v
                    ack.put(True)
            finally:
                gen_self["stop"] = True
                try:
                    ack.put_nowait(False)
                except queue.Full:
                    pass
        ack = queue.Queue(maxsize=1)
        gen_self["ack"] = ack
        return gen()

    def ex_Yield(self, s, env):
        import threading
        slot = getattr(self, "_yield_q", {}).get(threading.get_ident())
        if slot is None:
            raise RuntimeErr("'yield' outside a generator")
        q, st = slot
        vals = self.eval(s.value, env) if s.value is not None else None
        for v in (vals if s.from_ else [vals]):
            q.put(("v", v))
            if not st["ack"].get() or st["stop"]:
                raise _GenStop()

    def _call_function(self, fn, args, kwargs):
        env = Env(fn.closure)
        bindings = {}
        params = fn.params
        rest = kwrest = None
        if params and params[-1].star == "**":
            kwrest, params = params[-1], params[:-1]
        if params and params[-1].star == "*":
            rest, params = params[-1], params[:-1]
        if len(args) > len(params) and rest is None:
            raise RuntimeErr(f"{fn.name}() takes {len(params)} argument(s) but {len(args)} were given")
        kwargs = dict(kwargs)
        prec = fn.precision
        if rest is not None:
            env.vars[rest.name] = list(args[len(params):])
            args = args[:len(params)]
        for i, p in enumerate(params):
            if i < len(args):
                v = args[i]
            elif p.name in kwargs:
                v = kwargs.pop(p.name)
            elif p.default is not None:
                v = self.eval(p.default, fn.closure)
            else:
                raise RuntimeErr(f"{fn.name}() is missing argument '{p.name}'")
            if prec in ("f16", "f32", "f64"):
                v = _cast(v, prec)
            if p.type is not None:
                v = self.check_type(v, p.type, bindings, f"argument '{p.name}' of {fn.name}()", fn.closure)
            env.vars[p.name] = v
            env.consts.add(p.name)
        if kwrest is not None:
            env.vars[kwrest.name] = dict(kwargs)
            kwargs = {}
        if kwargs:
            raise RuntimeErr(f"{fn.name}() got unexpected argument(s): {', '.join(kwargs)}")
        for sym, val in bindings.items():
            if sym not in env.vars:
                env.vars[sym] = val
                env.consts.add(sym)
        old_exact = self.exact
        if prec == "exact":
            self.exact = True
        elif prec is not None:
            self.exact = False
        try:
            if fn.is_expr:
                result = self.eval(fn.body, env)
            else:
                try:
                    self.exec_block(fn.body, env)
                    result = None
                except ReturnSignal as r:
                    result = r.value
        except ApsiError as e:
            e.trace.append(f"{fn.name}()")
            raise
        finally:
            self.exact = old_exact
        if prec in ("f16", "f32", "f64"):
            result = _cast(result, prec)
        if fn.ret is not None:
            result = self.check_type(result, fn.ret, bindings, f"return value of {fn.name}()", fn.closure)
        return result

    def construct(self, st, args, kwargs):
        if len(args) > len(st.fields):
            raise RuntimeErr(f"{st.name}() takes {len(st.fields)} field(s) but {len(args)} were given")
        kwargs = dict(kwargs)
        values = {}
        bindings = {}
        for i, (name, t, default) in enumerate(st.fields):
            if i < len(args):
                v = args[i]
            elif name in kwargs:
                v = kwargs.pop(name)
            elif default is not None:
                v = self.eval(default, st.env)
            else:
                raise RuntimeErr(f"{st.name}() is missing field '{name}'")
            if t is not None:
                v = self.check_type(v, t, bindings, f"field '{name}' of {st.name}", st.env)
            values[name] = v
        if kwargs:
            raise RuntimeErr(f"{st.name} has no field(s): {', '.join(kwargs)}")
        return StructInstance(st, values)

    def ev_Index(self, n, env):
        obj = self.eval(n.obj, env)
        idx = self.eval_index(n.index, env)
        try:
            if isinstance(obj, (list, tuple, str)) and isinstance(idx, float) and idx.is_integer():
                idx = int(idx)
            return obj[idx]
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise
        except Exception as e:
            err = wrap_python_error(e)
            err.line, err.col = n.line, n.col
            raise err from None

    def eval_index(self, node, env):
        if isinstance(node, A.Slice):
            return slice(*(self.eval(x, env) if x is not None else None
                           for x in (node.lo, node.hi, node.step)))
        if isinstance(node, A.TupleLit):
            return tuple(self.eval_index(x, env) for x in node.items)
        v = self.eval(node, env)
        if isinstance(v, list) and v and all(isinstance(x, (int, np.integer)) for x in v):
            return np.asarray(v)
        return v

    def ev_Slice(self, n, env):
        return self.eval_index(n, env)

    def ev_Attr(self, n, env):
        obj = self.eval(n.obj, env)
        try:
            return self.get_attr(obj, n.name)
        except ApsiError as e:
            if e.line is None:
                e.line, e.col = n.line, n.col
            raise

    def get_attr(self, obj, name):
        c = obj.__class__
        if c is Record:                                           # the commonest: a record's field
            f = obj.fields
            if name in f:
                return f[name]
        if (c is str or c is list or c is dict or c in _PLAIN_GET) and name[0] != "_" and name not in _SLOW_NAMES:
            try:                                                  # text, lists, dicts, class instances …
                return getattr(obj, name)
            except AttributeError:
                raise _no_attr(obj, name) from None
        if c is StructInstance:                                   # fast path for structs
            f = obj.fields
            if name in f:
                return f[name]
        if isinstance(obj, StructInstance):
            if name in obj.fields:
                return obj.fields[name]
            m = obj.type.methods.get(name)
            if m is not None:
                return BoundMethod(m, obj)
            raise _no_attr(obj, name, f"{obj.type.name} has no field or method '{name}'",
                           list(obj.fields) + list(obj.type.methods))
        if isinstance(obj, Record):
            if name in obj.fields:
                return obj.fields[name]
            raise _no_attr(obj, name, f"record has no field '{name}' (fields: {', '.join(obj.fields)})",
                           list(obj.fields))
        if isinstance(obj, Namespace):
            if name in obj.members:
                return obj.members[name]
            from .errors import did_you_mean
            err = NameErr(f"module '{obj.name}' has no member '{name}'" + did_you_mean(name, obj.members))
            err.focus = "." + name
            raise err
        if isinstance(obj, UnitNamespace):
            return obj.get(name)
        if name.startswith("_"):
            from .pycompat import check_attr, trusted
            if name.startswith("__") and name.endswith("__"):
                check_attr(self, name, obj)                # dunders: allowed except interpreter internals
            elif not (trusted(self) or self._own_object(obj, name)):
                raise RuntimeErr(f"cannot access private attribute '{name}'")
            try:
                return getattr(obj, name)
            except AttributeError:
                raise _no_attr(obj, name) from None
        if isinstance(obj, np.ndarray) and name == "shape":
            return tuple(obj.shape)
        if name in _FRAME_ATTRS:
            from .pycompat import check_attr
            check_attr(self, name, obj)
        if name in ("tofile", "dump") and isinstance(obj, np.ndarray):
            from .pycompat import trusted
            if not trusted(self):
                raise RuntimeErr(f"array.{name}() writes a file without Apsi's permission checks; use "
                                 f"numpy.save() (checked by --allow-write) or run with --allow-python=all")
        try:
            v = getattr(obj, name)
        except AttributeError:
            raise _no_attr(obj, name) from None
        if not isinstance(obj, _NOT_PLAIN):
            _PLAIN_GET.add(c)                               # next time: straight to getattr (same checks apply)
        return v

    def ev_ListLit(self, n, env):
        out = []
        for x in n.items:
            if isinstance(x, A.Spread):
                out.extend(self.eval(x.value, env))
            else:
                out.append(self.eval(x, env))
        return out

    def ev_TupleLit(self, n, env):
        return tuple(self.eval(x, env) for x in n.items)

    def ev_DictLit(self, n, env):
        out = {}
        for k, v in n.pairs:
            if k is None:
                out.update(self.eval(v.value, env))
            else:
                out[self.eval(k, env)] = self.eval(v, env)
        return out

    def _comp_envs(self, n, env):
        """Every environment of a comprehension's for-clauses, in Python's order."""
        clauses = [(n.targets, n.iter, n.cond)] + list(n.more or [])

        def walk(i, env):
            if i == len(clauses):
                yield env
                return
            targets, itx, cond = clauses[i]
            pat = targets[0] if len(targets) == 1 else list(targets)
            for item in self.eval(itx, env):
                e = Env(env)
                bind_pattern(pat, item, e.vars.__setitem__)
                if cond is None or self.truthy(self.eval(cond, e)):
                    yield from walk(i + 1, e)
        return walk(0, env)

    def ev_ListComp(self, n, env):
        return [self.eval(n.elem, e) for e in self._comp_envs(n, env)]

    def ev_DictComp(self, n, env):
        out = {}
        for e in self._comp_envs(n, env):
            out[self.eval(n.key, e)] = self.eval(n.value, e)
        return out

    def ev_Lambda(self, n, env):
        return Function(self, "<lambda>", n.params, None, n.body, True, env)

    def ev_Range(self, n, env):
        return RangeVal(self.eval(n.start, env), self.eval(n.end, env), n.inclusive)

    def ev_As(self, n, env):
        v = self.eval(n.expr, env)
        text, scale, dim = n.unit
        have = ops.dim_of(v)
        if have is None:
            raise RuntimeErr(f"cannot convert {type_name(v)} to [{text}]")
        if have != dim:
            raise UnitError(f"cannot express {dim_name(have)} in [{text}] ({dim_name(dim)})",
                            n.line, n.col)
        if dim.dimensionless:
            return v / scale
        return Quantity(v.v, v.dim, (text, scale))

    # ================================================================ types
    def check_type(self, v, t, bindings, what, env):
        if isinstance(t, A.TUnit):
            have = ops.dim_of(v)
            if have != t.dim:
                raise UnitError(f"{what} must be in [{t.text}] ({dim_name(t.dim)}), "
                                f"got {self._describe(v)}")
            return v
        name = t.name
        if name == "Any":
            return v
        if name in ("Float", "Number", "Real"):
            if isinstance(v, Quantity):
                raise UnitError(f"{what} must be a dimensionless Float, got {self._describe(v)}")
            if isinstance(v, (int, float, Fraction, np.number, Node)) and not isinstance(v, bool):
                return v
            raise TypeCheckError(f"{what} must be a Float, got {type_name(v)}")
        if name in ("f64", "f32", "f16", "bf16"):              # a Float kept in that precision (f16/bf16: f32 math)
            if isinstance(v, Quantity):
                raise UnitError(f"{what} must be a dimensionless Float, got {self._describe(v)}")
            if isinstance(v, (int, float, Fraction, np.number)) and not isinstance(v, bool):
                return float(v) if name == "f64" else np.float32(v)
            if isinstance(v, Node):
                return v
            raise TypeCheckError(f"{what} must be a Float, got {type_name(v)}")
        if name == "Int":
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                return v
            raise TypeCheckError(f"{what} must be an Int, got {type_name(v)}")
        if name == "Bool":
            if isinstance(v, (bool, np.bool_)):
                return v
            raise TypeCheckError(f"{what} must be a Bool, got {type_name(v)}")
        if name == "Str":
            if isinstance(v, str):
                return v
            raise TypeCheckError(f"{what} must be a Str, got {type_name(v)}")
        simple = {"List": list, "Tuple": tuple, "Dict": dict, "Record": Record}
        if name in simple:
            if isinstance(v, simple[name]):
                return v
            raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
        if name == "Fn":
            if callable(v):
                return v
            raise TypeCheckError(f"{what} must be a function, got {type_name(v)}")
        if name in DIM_NAMES:
            have = ops.dim_of(v)
            if have is None:
                raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
            if have != DIM_NAMES[name]:
                raise UnitError(f"{what} must be {dim_name(DIM_NAMES[name])}, got {self._describe(v)}")
            return v
        if name in TENSOR_TYPES:
            return self._check_tensor(v, t, bindings, what)
        try:
            st = env.lookup(name)
        except KeyError:
            st = None
        if isinstance(st, StructType):
            if isinstance(v, StructInstance) and v.type is st:
                return v
            raise TypeCheckError(f"{what} must be a {name}, got {type_name(v)}")
        raise TypeCheckError(f"unknown type '{name}'")

    def _describe(self, v):
        d = ops.dim_of(v)
        if isinstance(v, Quantity) and np.ndim(ops.raw(v)) == 0:
            return f"{dim_name(d)} ({fmt(v)})"
        return dim_name(d) if d is not None else type_name(v)

    def _check_tensor(self, v, t, bindings, what):
        specs, dim, dtype = [], None, None
        if t.name in ("Vec2", "Vec3", "Vec4"):
            specs.append(int(t.name[3]))
        for a in t.args:
            if isinstance(a, int):
                specs.append(a)
            elif isinstance(a, A.TUnit):
                dim = a.dim
            elif a.name in DIM_NAMES:
                dim = DIM_NAMES[a.name]
            elif a.name in DTYPE_NAMES:
                dtype = a.name
            else:
                specs.append(a.name)
        r = ops.raw(v)
        if not isinstance(r, np.ndarray):
            if isinstance(v, list):
                raise TypeCheckError(f"{what} must be a Tensor, got a List; wrap it with tensor([...])")
            if not (np.ndim(r) == 0 and not specs and isinstance(r, (int, float, np.number))):
                raise TypeCheckError(f"{what} must be a Tensor, got {type_name(v)}")
        shape = tuple(np.shape(r))
        if specs:
            if len(shape) != len(specs):
                want = ", ".join(str(s) for s in specs)
                raise ShapeError(f"{what} must have shape [{want}] (rank {len(specs)}), got shape {shape}")
            for s, actual in zip(specs, shape):
                if isinstance(s, int):
                    if s != actual:
                        raise ShapeError(f"{what}: expected size {s}, got {actual} (shape {shape})")
                elif s == "_":
                    continue
                elif s in bindings:
                    if bindings[s] != actual:
                        raise ShapeError(f"{what}: dimension {s} is {actual}, but {s} was already "
                                         f"bound to {bindings[s]} (shape {shape})")
                else:
                    bindings[s] = actual
        if dim is not None:
            have = ops.dim_of(v)
            if have != dim:
                raise UnitError(f"{what} must hold {dim_name(dim)}, got {dim_name(have)}")
        elif isinstance(v, Quantity) and t.name not in ("Vec2", "Vec3", "Vec4", "Vec"):
            pass
        if dtype is not None and isinstance(r, np.ndarray):
            v = _cast(v, dtype)
        return v


def _cast(v, prec):
    if prec not in ("f16", "f32", "f64", "i32", "i64", "bool"):
        from .precision import round_to
        return round_to(v, prec) if np.ndim(ops.raw(v)) else v
    npd = {"f16": np.float16, "f32": np.float32, "f64": np.float64,
           "i32": np.int32, "i64": np.int64, "bool": np.bool_}[prec]
    if isinstance(v, np.ndarray):
        return v.astype(npd) if v.dtype != npd else v
    if isinstance(v, Quantity) and isinstance(v.v, np.ndarray):
        return Quantity(v.v.astype(npd), v.dim, v.disp)
    return v


def run_source(source, filename="<main>", check=True):
    return Interpreter().run(source, filename, check)


def run_file(path, check=True):
    with open(path, encoding="utf-8-sig") as fh:
        return Interpreter().run(fh.read(), path, check)
