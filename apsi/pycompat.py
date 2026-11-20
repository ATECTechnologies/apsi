"""`use pycompat` — Python's exact semantics for a file (what `apsi convert` writes at the top).

A file that starts with `use pycompat` keeps Apsi's syntax but behaves exactly like the Python it came from:

* print / str / repr / format / f-strings give Python's text (True, None, 'quotes' inside lists)
* the built-ins are Python's own: range, map, zip, sorted, len, int, isinstance, super, property, open …
* classes are real Python classes (inheritance from any library class, super(), properties, dunder methods)
* `with`, `del`, `is`, chained comparisons, `|`/`&`/`<<`/`>>`, loop `else`, generators, sets, walrus …
* names follow Python's scoping (one namespace per function), default values are evaluated once,
  *args is a tuple, `a += b` updates lists in place, attribute access is direct (fast)

Apsi's permissions still apply: open() checks --allow-read/--allow-write, pyimport() needs --allow-python
(pure standard modules excepted), and eval/exec/compile/__import__ need --allow-python=all. Introspection that
reaches interpreter internals (__globals__, __subclasses__, frame objects …) is refused unless the program was
started with --allow-python=all.
"""
import builtins as _B

from .errors import RuntimeErr

# dunder attributes that lead to interpreter internals (sandbox escapes) — refused unless --allow-python=all
_DENY = {"__globals__", "__builtins__", "__subclasses__", "__code__", "__closure__", "__loader__", "__spec__",
         "__import__", "__reduce__", "__reduce_ex__", "__self__", "__objclass__", "__getattribute__",
         "gi_frame", "gi_code", "f_globals", "f_builtins", "f_locals", "f_back", "f_code", "tb_frame", "tb_next",
         "cr_frame", "cr_code", "ag_frame", "ag_code", "co_code", "func_globals"}

_PLAIN = ["abs", "aiter", "all", "anext", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
          "classmethod", "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
          "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "memoryview", "min",
          "next", "object", "oct", "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set",
          "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "zip", "NotImplemented",
          "Ellipsis"]


_PURE_SUB = {"urllib.parse", "html.parser", "concurrent.futures.thread", "email.utils", "xml.sax.saxutils"}



class _Unavailable:
    """A name a star import could not give under Apsi's permissions; using it raises the permission error."""

    def __init__(self, name, message):
        self._name, self._message = name, message

    def __call__(self, *a, **k):
        raise AttributeError(self._message)

    def __getattr__(self, k):
        raise AttributeError(self.__dict__.get("_message", "not available"))

    def __repr__(self):
        return f"<{self._name}: not available under Apsi's permissions>"

def _safe(interp):
    st = getattr(interp, "_pysafe", None)
    if st is None:
        from .pysafe import make
        st = interp._pysafe = make(interp)
    return st


def trusted(interp):
    pol = getattr(interp, "policy", None)
    return pol is not None and getattr(pol, "python", False) == "all"


def check_attr(interp, name, obj=None):
    if name in _DENY and not trusted(interp):
        raise RuntimeErr(f"the attribute '{name}' reaches interpreter internals; it needs --allow-python=all")
    if name == "__dict__" and isinstance(obj, type) and not trusted(interp):
        from .codegen import is_own
        if not is_own(obj):                        # a library class's __dict__ leads to every class in the process
            raise RuntimeErr("the __dict__ of a library class needs --allow-python=all")


def safe_attrgetter(interp):
    import operator

    def attrgetter(*names):
        for n in names:
            for part in str(n).split("."):
                check_attr(interp, part)
                if part.startswith("_") and not part.startswith("__") and not trusted(interp):
                    raise RuntimeErr(f"attrgetter('{n}'): private attributes need --allow-python=all")
        return operator.attrgetter(*names)

    def methodcaller(name, *a, **k):
        check_attr(interp, str(name))
        if str(name).startswith("_") and not trusted(interp):
            raise RuntimeErr(f"methodcaller('{name}'): private methods need --allow-python=all")
        return operator.methodcaller(name, *a, **k)
    return attrgetter, methodcaller


def module(interp):
    from .security import check_path, need, PermissionErr
    m = {k: getattr(_B, k) for k in _PLAIN if hasattr(_B, k)}
    for k, v in vars(_B).items():                           # every exception and warning class
        if isinstance(v, type) and issubclass(v, BaseException):
            m[k] = v

    def _priv(obj, name):
        if name.startswith("_") and not name.startswith("__") and not trusted(interp):
            if not interp._own_object(obj, name):
                raise RuntimeErr(f"cannot access private attribute '{name}'")

    def getattr_(obj, name, *default):
        check_attr(interp, str(name), obj)
        _priv(obj, str(name))
        return _B.getattr(obj, name, *default)

    def hasattr_(obj, name):
        check_attr(interp, str(name), obj)
        return _B.hasattr(obj, name)

    def setattr_(obj, name, value):
        check_attr(interp, str(name))
        if str(name) in ("__class__", "__dict__", "__code__") and not trusted(interp):
            raise RuntimeErr(f"setting '{name}' needs --allow-python=all")
        _B.setattr(obj, name, value)

    def delattr_(obj, name):
        check_attr(interp, str(name))
        _B.delattr(obj, name)

    def vars_(*obj):
        if not obj:
            raise RuntimeErr("vars() without an argument is not available; use vars(obj)")
        import types
        if isinstance(obj[0], (types.FunctionType, types.CodeType, types.FrameType, types.ModuleType)) \
                and not trusted(interp):
            raise RuntimeErr("vars() of a function or module needs --allow-python=all")
        check_attr(interp, "__dict__", obj[0])
        return _B.vars(obj[0])

    def dir_(*obj):
        return _B.dir(*obj) if obj else []

    def open_(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            need(interp, "python", "open() of a file descriptor")
            return _B.open(file, mode, *args, **kwargs)
        import os
        path = os.fspath(file)
        kind = "rw" if "+" in mode else ("write" if any(c in mode for c in "wax") else "read")
        real = check_path(interp, path if isinstance(path, str) else path.decode(), kind, "open()")
        return _B.open(real, mode, *args, **kwargs)

    def _gated(name):
        f = getattr(_B, name)

        def gated(*a, **k):
            if not trusted(interp):
                raise PermissionErr(f"{name}() runs arbitrary code; it needs --allow-python=all")
            if name in ("globals", "locals"):              # the caller's namespace, not this wrapper's
                import sys
                fr = sys._getframe(1)
                return fr.f_globals if name == "globals" else fr.f_locals
            if name in ("eval", "exec") and len(a) == 1 and "globals" not in k and "locals" not in k:
                import sys
                fr = sys._getframe(1)
                return f(a[0], fr.f_globals, fr.f_locals)
            return f(*a, **k)
        gated.__name__ = name
        return gated

    def input_(prompt=""):
        return _B.input(prompt)

    def exit_(code=None):
        raise SystemExit(code)

    def pyimport_(name):
        """pyimport() for Python-exact files: pure standard modules (and their submodules) need no permission;
        with --allow-python=all every module is the real module object."""
        import importlib
        from .interop import PURE_MODULES, pure_module
        name = str(name)
        if trusted(interp):
            import os
            import sys
            cwd = interp.dir_stack[-1] if getattr(interp, "dir_stack", None) else os.getcwd()
            if cwd not in sys.path:
                sys.path.append(cwd)
            try:
                return importlib.import_module(name)
            except ImportError as e:
                from .interop import PyModuleMissing
                err = PyModuleMissing(f"Python module '{name}' is not installed ({e})")
                err.name = name
                raise err from None
        safe = _safe(interp)
        if name in safe:
            return safe[name]
        if name.startswith("numpy."):
            from .pysafe import _NP_SUBMODULES_OK
            if name in _NP_SUBMODULES_OK:
                return importlib.import_module(name)
        if name == "operator":
            mod = pure_module(name)
            ag, mc = safe_attrgetter(interp)
            from .values import Namespace
            return Namespace("operator", {**mod.members, "attrgetter": ag, "methodcaller": mc})
        if name.split(".")[0] in PURE_MODULES or name in _PURE_SUB:
            return pure_module(name)
        return interp.globals.vars["pyimport"](name)

    def pyfrom(module, name):
        """`from module import name`: an attribute of the module, or its submodule."""
        mod = pyimport_(module)
        try:
            return _B.getattr(mod, name)
        except AttributeError:
            return pyimport_(f"{module}.{name}")

    def pystar(module, *names):
        """`from module import *`: the public names, in one go."""
        mod = pyimport_(module)
        out = []
        for n in names:
            try:
                out.append(_B.getattr(mod, n))
            except AttributeError as e:
                if "permissions" not in str(e):
                    raise
                out.append(_Unavailable(n, str(e)))       # the error comes when the name is used, not at import
        return tuple(out)

    m.update(pyimport=pyimport_, pyfrom=pyfrom, pystar=pystar, getattr=getattr_, hasattr=hasattr_, setattr=setattr_, delattr=delattr_, vars=vars_, dir=dir_,
             open=open_, input=input_, exit=exit_, quit=exit_)
    for name in ("eval", "exec", "compile", "__import__", "globals", "locals", "breakpoint"):
        m[name] = _gated(name)
    return m
