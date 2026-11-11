"""Permission-checked stand-ins for sys, os, os.path and pathlib (Python-exact files without --allow-python=all).

Converted Python programs import sys, os and pathlib almost always. Under Apsi's permissions these modules would
give the whole computer to the program, so `pyimport("os")` normally needs --allow-python=all. In a `use pycompat`
file the import instead returns a stand-in with the same names and behaviour for everything a program normally does
(paths, files, folders, argv, stdout, environment variables …), where every touch of the file system, the
environment or other processes goes through Apsi's permission checks (--allow-read, --allow-write, --allow-env).
Anything that would escape the sandbox (os.system, subprocess, exec*, fork, kill, sys.modules, ctypes …) is simply
not there and says which permission brings the real module.
"""
import os as _os
import pathlib as _pl
import sys as _sys

from .errors import RuntimeErr


class _Missing:
    def __init__(self, module):
        self.module = module

    def __call__(self, name):
        raise AttributeError(f"{self.module}.{name} is not available under Apsi's permissions; run with "
                             f"--allow-python=all to use the real {self.module} module")


class _Proxy:
    """A module-like object with a fixed set of members."""

    def __init__(self, name, members):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_members", members)

    def __getattr__(self, k):
        m = object.__getattribute__(self, "_members")
        if k in m:
            return m[k]
        _Missing(object.__getattribute__(self, "_name"))(k)

    def __setattr__(self, k, v):
        m = object.__getattribute__(self, "_members")
        if k in m and k in ("stdout", "stderr", "stdin", "argv", "path", "set_printoptions"):
            m[k] = v
            return
        raise AttributeError(f"cannot set {object.__getattribute__(self, '_name')}.{k}")

    def __dir__(self):
        return sorted(object.__getattribute__(self, "_members"))

    def __repr__(self):
        return f"<module '{object.__getattribute__(self, '_name')}' (Apsi permissions)>"


def make(interp):
    from .security import check_path, check_env

    def rd(p, what):
        return check_path(interp, _os.fspath(p), "read", what)

    def wr(p, what):
        return check_path(interp, _os.fspath(p), "write", what)

    # ---------------------------------------------------------------- os.path
    P = _os.path
    path_members = {k: getattr(P, k) for k in (
        "join", "basename", "dirname", "split", "splitext", "splitdrive", "normpath", "normcase", "isabs",
        "commonpath", "commonprefix", "relpath", "abspath", "expanduser", "sep", "extsep", "altsep", "pathsep",
        "curdir", "pardir", "devnull", "supports_unicode_filenames") if hasattr(P, k)}

    def _q(fn, name):
        def f(p, *a):
            try:
                rd(p, f"os.path.{name}()")
            except RuntimeErr:
                raise
            return fn(p, *a)
        f.__name__ = name
        return f
    for k in ("exists", "lexists", "isfile", "isdir", "islink", "getsize", "getmtime", "getatime", "getctime",
              "realpath", "samefile"):
        if hasattr(P, k):
            path_members[k] = _q(getattr(P, k), k)
    ospath = _Proxy("os.path", path_members)

    # ---------------------------------------------------------------- os
    def listdir(p="."):
        return _os.listdir(rd(p, "os.listdir()"))

    def scandir(p="."):
        return _os.scandir(rd(p, "os.scandir()"))

    def walk(top, *a, **k):
        rd(top, "os.walk()")
        return _os.walk(top, *a, **k)

    def stat(p, *a, **k):
        return _os.stat(rd(p, "os.stat()"), *a, **k)

    def makedirs(p, mode=0o777, exist_ok=False):
        return _os.makedirs(wr(p, "os.makedirs()"), mode, exist_ok)

    def mkdir(p, mode=0o777):
        return _os.mkdir(wr(p, "os.mkdir()"), mode)

    def remove(p):
        return _os.remove(wr(p, "os.remove()"))

    def rmdir(p):
        return _os.rmdir(wr(p, "os.rmdir()"))

    def rename(a, b):
        return _os.rename(wr(a, "os.rename()"), wr(b, "os.rename()"))

    def replace(a, b):
        return _os.replace(wr(a, "os.replace()"), wr(b, "os.replace()"))

    def fsync(fd):
        return _os.fsync(fd)

    def getenv(k, default=None):
        check_env(interp, k)
        return _os.environ.get(k, default)

    class _Environ(dict):
        def __getitem__(self, k):
            check_env(interp, k)
            return _os.environ[k]

        def get(self, k, default=None):
            check_env(interp, k)
            return _os.environ.get(k, default)

        def __contains__(self, k):
            check_env(interp, k)
            return k in _os.environ

        def __setitem__(self, k, v):
            check_env(interp, k)
            _os.environ[k] = v

        def setdefault(self, k, v=None):
            check_env(interp, k)
            return _os.environ.setdefault(k, v)

        def pop(self, k, *d):
            check_env(interp, k)
            return _os.environ.pop(k, *d)

        def __iter__(self):
            check_env(interp, "*")
            return iter(_os.environ)

    os_members = {k: getattr(_os, k) for k in (
        "sep", "linesep", "pathsep", "curdir", "pardir", "extsep", "altsep", "devnull", "name", "fspath",
        "fsencode", "fsdecode", "cpu_count", "getpid", "urandom", "get_terminal_size", "PathLike", "SEEK_SET",
        "SEEK_CUR", "SEEK_END", "O_RDONLY", "O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC", "F_OK",
        "R_OK", "W_OK", "X_OK", "getcwd", "terminal_size", "error", "strerror", "times") if hasattr(_os, k)}
    os_members.update(path=ospath, listdir=listdir, scandir=scandir, walk=walk, stat=stat, makedirs=makedirs,
                      mkdir=mkdir, remove=remove, unlink=remove, rmdir=rmdir, rename=rename, replace=replace,
                      fsync=fsync, getenv=getenv, environ=_Environ())
    if hasattr(_os, "sched_getaffinity"):
        os_members["sched_getaffinity"] = _os.sched_getaffinity
    os_mod = _Proxy("os", os_members)

    # ---------------------------------------------------------------- sys
    sys_members = {k: getattr(_sys, k) for k in (
        "maxsize", "float_info", "int_info", "hash_info", "maxunicode", "byteorder", "version", "version_info",
        "hexversion", "implementation", "platform", "float_repr_style", "getrecursionlimit", "getsizeof",
        "intern", "getdefaultencoding", "getfilesystemencoding", "exc_info", "exception", "stdout", "stderr",
        "stdin", "executable", "flags", "api_version", "copyright", "is_finalizing", "getswitchinterval",
        "dont_write_bytecode") if hasattr(_sys, k)}

    def setrecursionlimit(n):
        _sys.setrecursionlimit(max(100, min(int(n), 200000)))

    def exit(code=None):
        raise SystemExit(code)
    sys_members.update(argv=_sys.argv, path=list(_sys.path), setrecursionlimit=setrecursionlimit, exit=exit,
                       modules={}, builtin_module_names=_sys.builtin_module_names)
    sys_mod = _Proxy("sys", sys_members)

    # ---------------------------------------------------------------- pathlib
    base = type(_pl.Path())

    def _chk(kind):
        def deco(meth):
            def f(self, *a, **k):
                check_path(interp, str(self), kind, f"Path.{meth.__name__}()")
                return meth(self, *a, **k)
            f.__name__ = meth.__name__
            return f
        return deco

    ns = {}
    for name in ("read_text", "read_bytes", "exists", "is_file", "is_dir", "is_symlink", "iterdir", "glob", "rglob",
                 "stat", "lstat", "owner", "group", "samefile", "readlink", "walk"):
        if hasattr(base, name):
            ns[name] = _chk("read")(getattr(base, name))
    for name in ("write_text", "write_bytes", "mkdir", "unlink", "rmdir", "touch", "chmod", "symlink_to",
                 "hardlink_to"):
        if hasattr(base, name):
            ns[name] = _chk("write")(getattr(base, name))

    def open_(self, mode="r", *a, **k):
        kind = "rw" if "+" in mode else ("write" if any(c in mode for c in "wax") else "read")
        check_path(interp, str(self), kind, "Path.open()")
        return base.open(self, mode, *a, **k)

    def rename_(self, target):
        check_path(interp, str(self), "write", "Path.rename()")
        check_path(interp, str(target), "write", "Path.rename()")
        return base.rename(self, target)
    ns.update(open=open_, rename=rename_, replace=rename_)
    SafePath = type("Path", (base,), ns)
    pl_members = {"Path": SafePath, "PurePath": _pl.PurePath, "PurePosixPath": _pl.PurePosixPath,
                  "PureWindowsPath": _pl.PureWindowsPath}
    pathlib_mod = _Proxy("pathlib", pl_members)
    return {"sys": sys_mod, "os": os_mod, "os.path": ospath, "pathlib": pathlib_mod, "numpy": _numpy(interp)}


_NP_BLOCK = {"ctypeslib", "f2py", "distutils", "testing", "show_config", "show_runtime", "DataSource", "lib",
             "_core", "core"}
_NP_SUBMODULES_OK = {"numpy.linalg", "numpy.fft", "numpy.random", "numpy.polynomial", "numpy.ma", "numpy.char",
                     "numpy.emath", "numpy.typing", "numpy.lib.stride_tricks", "numpy.exceptions", "numpy.dtypes"}


def _numpy(interp):
    """numpy with its file functions checked against --allow-read/--allow-write (and np.load never unpickles)."""
    import numpy as np
    from .security import check_path

    def rd(p, what):
        return check_path(interp, _os.fspath(p), "read", what) if isinstance(p, (str, bytes, _os.PathLike)) else p

    def wr(p, what):
        return check_path(interp, _os.fspath(p), "write", what) if isinstance(p, (str, bytes, _os.PathLike)) else p

    def _file_arg(fn, kind, name):
        def f(file, *a, **k):
            if not hasattr(file, "read") and not hasattr(file, "write"):
                file = (rd if kind == "read" else wr)(file, f"numpy.{name}()")
            return fn(file, *a, **k)
        f.__name__ = name
        return f

    def load(file, mmap_mode=None, allow_pickle=False, **k):
        if allow_pickle:
            raise RuntimeErr("numpy.load(allow_pickle=True) can run code from the file; it needs --allow-python=all")
        if not hasattr(file, "read"):
            file = rd(file, "numpy.load()")
        return np.load(file, mmap_mode=mmap_mode, allow_pickle=False, **k)

    members = {k: getattr(np, k) for k in dir(np) if not k.startswith("_") and k not in _NP_BLOCK}
    for name in ("save", "savez", "savez_compressed", "savetxt"):
        members[name] = _file_arg(getattr(np, name), "write", name)
    for name in ("loadtxt", "genfromtxt", "fromfile", "fromregex"):
        if hasattr(np, name):
            members[name] = _file_arg(getattr(np, name), "read", name)

    def memmap(filename, *a, mode="r+", **k):
        kind = "read" if mode == "r" else "rw"
        check_path(interp, _os.fspath(filename), kind, "numpy.memmap()")
        return np.memmap(filename, *a, mode=mode, **k)
    members.update(load=load, memmap=memmap, __version__=np.__version__)
    import numpy.lib.stride_tricks as st
    members["lib"] = _Proxy("numpy.lib", {"stride_tricks": st})
    return _Proxy("numpy", members)
