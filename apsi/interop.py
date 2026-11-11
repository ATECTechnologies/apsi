"""Interop with other languages and the outside world, gated by permissions.

  python : pyimport("numpy"), to_torch(x), from_torch(t), to_py(x), from_py(v)
  ffi    : cload("libm.so") -> lib.fn("cos", [f64], f64)  (C / C++ extern "C" / Rust ...)
  net    : http_get(url), http_post_json(url, data)
  fs     : read_text, write_text, load_csv, save_csv, load_json, save_json
"""
import ctypes
import json
import os

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity, make, parse_unit
from .values import Record, StructInstance, Dtype


from .security import PermissionErr, need as _need, check_path, check_net, check_env, resolve_checked  # noqa: F401


# ------------------------------------------------------------------ values <-> Python
def to_py(x):
    """Apsi value -> plain Python/JSON-friendly value (units become {value, unit})."""
    if isinstance(x, Quantity):
        v = x.magnitude
        v = v.tolist() if isinstance(v, np.ndarray) else float(v)
        return {"value": v, "unit": x.unit}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_py(i) for i in x]
    if isinstance(x, dict):
        return {str(k): to_py(v) for k, v in x.items()}
    if isinstance(x, Record):
        return {k: to_py(v) for k, v in x.fields.items()}
    if isinstance(x, StructInstance):
        return {k: to_py(v) for k, v in x.fields.items()}
    from fractions import Fraction
    if isinstance(x, Fraction):
        return float(x)
    return x


def from_py(x):
    """Python/JSON value -> Apsi value ({value, unit} dicts become Quantities)."""
    if isinstance(x, dict):
        if set(x) == {"value", "unit"}:
            scale, dim = parse_unit(x["unit"])
            v = np.asarray(x["value"], dtype=float) if isinstance(x["value"], list) else float(x["value"])
            return make(v * scale, dim, (x["unit"], scale))
        return {k: from_py(v) for k, v in x.items()}
    if isinstance(x, list):
        if x and all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in x):
            return np.asarray(x)
        if x and all(isinstance(i, list) for i in x):
            try:
                a = np.asarray(x)
                if a.dtype != object:
                    return a
            except ValueError:
                pass
        return [from_py(i) for i in x]
    return x


def _to_torch(x, device=None):
    import torch
    r = ops.raw(x)
    t = torch.as_tensor(np.asarray(r))
    return t.to(device) if device else t


def _from_torch(t):
    return t.detach().cpu().numpy()


# ------------------------------------------------------------------ C / C++ FFI
_CTYPES = {"f64": ctypes.c_double, "f32": ctypes.c_float, "i32": ctypes.c_int32,
           "i64": ctypes.c_int64, "void": None, "str": ctypes.c_char_p,
           "ptr_f64": ctypes.POINTER(ctypes.c_double), "ptr_f32": ctypes.POINTER(ctypes.c_float),
           "ptr_i32": ctypes.POINTER(ctypes.c_int32), "ptr_i64": ctypes.POINTER(ctypes.c_int64)}
_NP_FOR_PTR = {"ptr_f64": np.float64, "ptr_f32": np.float32, "ptr_i32": np.int32, "ptr_i64": np.int64}


def _tname(t):
    if isinstance(t, Dtype):
        return t.name
    return str(t)


class CFunction:
    def __init__(self, cfn, name, argtypes, restype):
        self.cfn, self.name, self.argtypes, self.restype = cfn, name, argtypes, restype

    def __call__(self, *args):
        if len(args) != len(self.argtypes):
            raise RuntimeErr(f"C function {self.name} takes {len(self.argtypes)} argument(s), got {len(args)}")
        conv = []
        for a, t in zip(args, self.argtypes):
            if isinstance(a, Quantity):
                raise RuntimeErr(f"pass plain numbers to C ({self.name}); use x.magnitude")
            if t in _NP_FOR_PTR:
                arr = np.ascontiguousarray(a, dtype=_NP_FOR_PTR[t])
                if isinstance(a, np.ndarray) and arr is not a:
                    raise RuntimeErr(f"argument for {t} must be a contiguous {t[4:]} tensor "
                                     f"(use tensor(x, dtype={t[4:]}))")
                conv.append(arr.ctypes.data_as(_CTYPES[t]))
            elif t == "str":
                conv.append(str(a).encode())
            else:
                conv.append(a)
        r = self.cfn(*conv)
        if self.restype == "str" and r is not None:
            return r.decode()
        return r


class CLibrary:
    def __init__(self, path):
        self.path = path
        self.lib = ctypes.CDLL(path)

    def fn(self, name, argtypes=(), restype="f64"):
        """Declare a C function signature: lib.fn("dot", [ptr_f64, ptr_f64, i64], f64)."""
        args = [_tname(a) for a in argtypes]
        res = _tname(restype)
        for t in args + [res]:
            if t not in _CTYPES:
                raise RuntimeErr(f"unknown C type '{t}'; use one of {', '.join(_CTYPES)}")
        cfn = getattr(self.lib, name)
        cfn.argtypes = [_CTYPES[t] for t in args]
        cfn.restype = _CTYPES[res]
        return CFunction(cfn, name, args, res)


def library_candidates(path):
    """vecmath -> [vecmath.dll, libvecmath.dll] on Windows, [libvecmath.so, vecmath.so] on Linux ..."""
    import sys
    d, base = os.path.split(path)
    if os.path.splitext(base)[1].lower() in (".dll", ".so", ".dylib") or ".so." in base:
        return [path]
    if sys.platform.startswith("win"):
        names = [base + ".dll", "lib" + base + ".dll"]
    elif sys.platform == "darwin":
        names = ["lib" + base + ".dylib", base + ".dylib", "lib" + base + ".so"]
    else:
        names = ["lib" + base + ".so", base + ".so"]
    if base.startswith("lib"):
        names.insert(0, base + (".dll" if sys.platform.startswith("win") else
                                ".dylib" if sys.platform == "darwin" else ".so"))
    return [os.path.join(d, n) for n in names]


def resolve_library(path, base_dir, must_exist=True):
    cands = library_candidates(path)
    for c in cands:
        for full in ([c] if os.path.isabs(c) else [os.path.join(base_dir, c), c]):
            if os.path.exists(full):
                return full
    if not must_exist:
        c = cands[0]
        return c if os.path.isabs(c) else os.path.join(base_dir, c)
    # fall back to the system loader search path (e.g. "m", "kernel32")
    import ctypes.util
    sysname = ctypes.util.find_library(os.path.basename(path))
    return sysname or cands[0]


# ------------------------------------------------------------------ pure Python standard modules
# These modules only compute: they cannot read files, open connections or start programs, so programs may use
# them without --allow-python. Only their public functions, classes and constants are exposed (no submodules), so
# nothing reaches os/sys through them. `apsi convert` relies on this to keep collections, heapq, itertools, re …
PURE_MODULES = {"collections", "heapq", "bisect", "itertools", "functools", "copy", "string", "re", "textwrap",
                "fractions", "decimal", "datetime", "calendar", "enum", "statistics", "cmath", "math", "random",
                "json", "typing", "dataclasses", "array", "numbers", "difflib", "unicodedata", "hashlib", "hmac",
                "base64", "binascii", "zlib", "struct", "secrets", "graphlib", "colorsys", "uuid", "keyword",
                "html", "abc", "contextlib", "pprint", "time", "argparse", "warnings", "operator", "queue",
                "threading", "ipaddress", "shlex", "weakref", "numbers", "zoneinfo", "textwrap"}
_PURE_DROP = {"functools": {"_lru_cache_wrapper"}, "pprint": {"PrettyPrinter"}, "operator": {"attrgetter",
                                                                                              "methodcaller"},
              "contextlib": {"redirect_stdout", "redirect_stderr", "chdir"}, "random": {"SystemRandom"},
              "uuid": {"getnode", "uuid1"}}
_pure_cache = {}


class PyModuleMissing(RuntimeErr, ModuleNotFoundError):
    """A missing Python module: an Apsi error that `except ImportError` in converted code also catches."""


def pure_module(name):
    if name not in _pure_cache:
        import importlib
        import types
        from .values import Namespace
        mod = importlib.import_module(name)
        drop = _PURE_DROP.get(name, set())
        members = {}
        for k in dir(mod):
            if k.startswith("_") or k in drop:
                continue
            v = getattr(mod, k)
            if isinstance(v, types.ModuleType):
                continue
            members[k] = v
        _pure_cache[name] = Namespace(name, members)
    return _pure_cache[name]


# ------------------------------------------------------------------ install
def install(interp):
    g = interp.globals.vars

    def pyimport(name):
        if str(name) in PURE_MODULES:
            return pure_module(str(name))
        _need(interp, "python", f"pyimport(\"{name}\")")
        from .guard import check_pyimport
        check_pyimport(interp.policy, str(name))
        import importlib, sys as _sys
        # launchers run Python isolated (-I); local modules next to the program are found
        # only for programs that were explicitly granted --allow-python
        cwd = os.getcwd()
        added = cwd not in _sys.path
        if added:
            _sys.path.append(cwd)
        try:
            return importlib.import_module(name)
        except ImportError as e:
            err = PyModuleMissing(f"Python module '{name}' is not installed ({e})")
            err.name = name
            raise err from None
        finally:
            if added and cwd in _sys.path:
                _sys.path.remove(cwd)

    def to_torch(x, device=None):
        _need(interp, "python", "to_torch()")
        try:
            return _to_torch(x, device)
        except ImportError:
            raise RuntimeErr("PyTorch is not installed (pip install torch)")

    def from_torch(t):
        _need(interp, "python", "from_torch()")
        return _from_torch(t)

    def cload(path):
        _need(interp, "ffi", f"cload(\"{path}\")")
        found = resolve_library(path, interp.dir_stack[-1])
        from .guard import check_native
        found = check_native(interp.policy, found) if os.path.exists(found) else found
        try:
            return CLibrary(found)
        except OSError as e:
            raise RuntimeErr(f"cannot load C library '{found}': {e}")

    def library_path(path):
        """The actual file cload() would load (e.g. vecmath -> vecmath.dll on Windows)."""
        return resolve_library(path, interp.dir_stack[-1], must_exist=False)

    def _rpath(p):
        return check_path(interp, p, "read")

    def _wpath(p):
        full = check_path(interp, p, "write")
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        return full

    def exists(path):
        return os.path.exists(_rpath(path))

    def read_text(path):
        with open(_rpath(path), encoding="utf-8-sig") as fh:
            return fh.read()

    def write_text(path, text):
        with open(_wpath(path), "w", encoding="utf-8") as fh:
            fh.write(str(text))

    def load_json(path):
        with open(_rpath(path), encoding="utf-8-sig") as fh:
            return from_py(json.load(fh))

    def save_json(path, value):
        with open(_wpath(path), "w", encoding="utf-8") as fh:
            json.dump(to_py(value), fh, ensure_ascii=False, indent=2)

    def load_csv(path, skip_header=0):
        return np.loadtxt(_rpath(path), delimiter=",", skiprows=int(skip_header), ndmin=1)

    def save_csv(path, data, header=""):
        np.savetxt(_wpath(path), np.asarray(ops.raw(data)), delimiter=",", header=header, comments="")

    def _check_url(url, what):
        from urllib.parse import urlparse
        u = urlparse(str(url))
        if u.scheme not in ("http", "https"):
            raise PermissionErr(f"{what}: only http:// and https:// URLs are allowed")
        if u.username or u.password:
            raise PermissionErr(f"{what}: credentials inside URLs are not allowed (use a header)")
        port = u.port or (443 if u.scheme == "https" else 80)
        check_net(interp, u.hostname or "", port, what)
        resolve_checked(interp, u.hostname or "", port, what)

    MAX_BODY = int(os.environ.get("APSI_HTTP_MAX_BYTES", 64 * 1024 * 1024))

    def _opener(what):
        import urllib.request

        class Guard(urllib.request.HTTPRedirectHandler):
            max_redirections = 5

            def redirect_request(self, req, fp, code, msg, headers, newurl):
                _check_url(newurl, what + " (redirect)")   # every hop must pass the same policy
                return super().redirect_request(req, fp, code, msg, headers, newurl)
        return urllib.request.build_opener(Guard)

    def _read(resp, what):
        data = resp.read(MAX_BODY + 1)
        if len(data) > MAX_BODY:
            raise RuntimeErr(f"{what}: response larger than {MAX_BODY} bytes (set APSI_HTTP_MAX_BYTES)")
        return data

    def http_get(url, timeout=30):
        _check_url(url, "http_get()")
        import urllib.request
        req = urllib.request.Request(str(url), headers={"User-Agent": f"apsi/{__import__('apsi').__version__}"})
        with _opener("http_get()").open(req, timeout=timeout) as r:
            return _read(r, "http_get()").decode("utf-8", "replace")

    def http_post_json(url, data, timeout=30):
        _check_url(url, "http_post_json()")
        import urllib.request
        body = json.dumps(to_py(data)).encode()
        req = urllib.request.Request(str(url), body, {"Content-Type": "application/json",
                                                      "User-Agent": f"apsi/{__import__('apsi').__version__}"})
        with _opener("http_post_json()").open(req, timeout=timeout) as r:
            return from_py(json.loads(_read(r, "http_post_json()").decode()))

    def json_parse(text):
        return from_py(json.loads(text))

    def json_str(value, indent=None, sort_keys=False, ensure_ascii=False, separators=None, default=None):
        return json.dumps(to_py(value), ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys,
                          separators=separators, default=default)

    def permissions():
        return sorted(interp.permissions)

    def getenv(name, default=None):
        """Read an environment variable (needs --allow-env); the place for secrets like API keys."""
        check_env(interp, str(name))
        return os.environ.get(str(name), default)

    def security_policy():
        from .values import Record
        return Record(**interp.policy.describe())

    funcs = dict(exists=exists, library_path=library_path, pyimport=pyimport, to_torch=to_torch, from_torch=from_torch, to_py=to_py,
                 from_py=from_py, cload=cload, read_text=read_text, write_text=write_text,
                 load_json=load_json, save_json=save_json, load_csv=load_csv, save_csv=save_csv,
                 http_get=http_get, http_post_json=http_post_json, json_parse=json_parse,
                 json_str=json_str, permissions=permissions, getenv=getenv,
                 security_policy=security_policy)
    g.update(funcs)
    for t in ("ptr_f64", "ptr_f32", "ptr_i32", "ptr_i64", "void"):
        g[t] = Dtype(t, None)
    g["cstr"] = Dtype("str", None)
    interp.modules["io"] = {k: funcs[k] for k in ("exists", "read_text", "write_text", "load_json", "save_json",
                                                 "load_csv", "save_csv", "json_parse", "json_str")}
    interp.modules["ffi"] = {"cload": cload, "ptr_f64": g["ptr_f64"], "ptr_f32": g["ptr_f32"],
                             "ptr_i32": g["ptr_i32"], "ptr_i64": g["ptr_i64"], "void": g["void"]}
    interp.modules["python"] = {"pyimport": pyimport, "to_torch": to_torch, "from_torch": from_torch,
                                "to_py": to_py, "from_py": from_py}
    interp.modules["net"] = {"http_get": http_get, "http_post_json": http_post_json}
