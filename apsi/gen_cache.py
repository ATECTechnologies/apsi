"""Cache of generated code per file: the Python source and bytecode the code generator makes for an .apsi file,
with what it needs around it (line map, constants, module names). A program of hundreds of files then starts by
reading these instead of generating every file again.

An entry is keyed by the source, the file's place, the code generator's own files, the built-in names, and how
every `use`/`import` in the file resolves right now (with the size and time of each local file it names), so a
change to any of them makes a new entry. Entries are signed with this user's key: one copied from elsewhere is
ignored. APSI_NO_CACHE=1 turns it off."""
import hashlib
import hmac
import marshal
import os
import pickle
import sys

from . import ast as A

_FP = None
_VERSION = 1


def _fingerprint():
    global _FP
    if _FP is None:
        from . import __version__
        h = hashlib.sha256(f"{__version__}\0{sys.version}\0{_VERSION}".encode())
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("codegen.py", "parser.py", "lexer.py", "ast.py", "gen_cache.py", "interpreter.py",
                     "pycompat.py", "units.py"):
            try:
                st = os.stat(os.path.join(here, name))
                h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                h.update(name.encode())
        _FP = h.hexdigest()
    return _FP


def node_paths(prog):
    """id(node) -> its path from the program root: a tuple of field names and list indexes."""
    out = {}
    stack = [(prog, ())]
    while stack:
        n, path = stack.pop()
        if isinstance(n, A.AST):
            out[id(n)] = path
            for f in n._fields:
                stack.append((getattr(n, f), path + (f,)))
        elif isinstance(n, (list, tuple)):
            for i, x in enumerate(n):
                if isinstance(x, (A.AST, list, tuple)):
                    stack.append((x, path + (i,)))
    return out


def follow(prog, path):
    n = prog
    for step in path:
        n = n[step] if isinstance(step, int) else getattr(n, step)
    return n


def _stamp(path):
    try:
        st = os.stat(path)
        return f"{path}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{path}:-"


def _imports_signature(prog, interp, base_dir):
    """How each use/import of the program resolves now (this also loads the libraries it names, as generating
    the code would)."""
    parts = []
    for n in _walk(prog):
        if isinstance(n, A.Use):
            kind, where = interp.import_target(n.path, base_dir, n.names, n.py)
            parts.append(f"U{n.path}|{kind}|{where}")
            if kind == "local":
                parts.append(_stamp(where))
            elif kind == "module" and n.path in getattr(interp, "_libraries", ()):
                parts.append(",".join(sorted(interp.modules[n.path])))
        elif isinstance(n, A.Import):
            p = n.path if os.path.isabs(n.path) else os.path.join(base_dir, n.path)
            parts.append("I" + _stamp(os.path.abspath(p)))
    return "\n".join(parts)


_BODIES = ("body", "else_body", "final_body", "branches", "handlers", "cases", "methods")


def _walk(prog):
    """The statements of a program, nested blocks included (imports are statements; expressions are skipped)."""
    stack = [prog]
    while stack:
        n = stack.pop()
        if isinstance(n, A.AST):
            yield n
            for f in _BODIES:
                v = getattr(n, f, None)
                if isinstance(v, (list, tuple)):
                    stack.append(v)
        elif isinstance(n, (list, tuple)):
            for x in n:
                if isinstance(x, (A.AST, list, tuple)):
                    stack.append(x)


def key(source, filename, base_dir, interp, prog, trusted):
    from .compile_cache import enabled
    if not enabled() or filename.startswith("<"):
        return None
    try:
        h = hashlib.sha256()
        for part in (_fingerprint(), os.path.abspath(filename), base_dir, str(bool(trusted)),
                     os.environ.get("APSI_NATIVE", ""), "\0".join(sorted(interp.globals.vars)),
                     _imports_signature(prog, interp, base_dir), source):
            h.update(part.encode("utf-8", "surrogatepass"))
            h.update(b"\0")
        return h.hexdigest()
    except Exception:                                  # noqa: BLE001 - no key: generate as usual
        return None


def _path(k):
    from .compile_cache import cache_dir
    return os.path.join(cache_dir(), "gen", k[:2], k + ".gen")


def load(k, prog, rt):
    """The cached generation for key k, rebuilt against this run's AST and runtime, or None."""
    if k is None:
        return None
    from .compile_cache import _secret
    path = _path(k)
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    sig, body = blob[:32], blob[32:]
    if not hmac.compare_digest(sig, hmac.new(_secret(), body, hashlib.sha256).digest()):
        return None
    try:
        rec = pickle.loads(body)
        consts = {}
        for name, (kind, v) in rec["consts"].items():
            if kind == "ast":
                consts[name] = follow(prog, v)
            elif kind == "chk":
                consts[name] = rt.make_checker(follow(prog, v))
            else:
                consts[name] = v
        rec["consts"] = consts
        rec["code"] = marshal.loads(rec["code"])
        return rec
    except Exception:                                  # noqa: BLE001 - unusable entry: generate as usual
        return None


def store(k, prog, gen, code, used, scope_names):
    """Save a generation; skipped when a constant cannot be described (it is then generated every time)."""
    if k is None:
        return
    from .compile_cache import _secret
    try:
        ids = None
        consts = {}
        for name, obj in gen.consts.items():
            src = gen.const_src.get(name)
            if (src is not None or isinstance(obj, A.AST)) and ids is None:
                ids = node_paths(prog)
            if src is not None:
                consts[name] = ("chk", ids[id(src)])
            elif isinstance(obj, A.AST):
                consts[name] = ("ast", ids[id(obj)])
            else:
                pickle.dumps(obj)
                consts[name] = ("val", obj)
        rec = {"source": gen.source(), "linemap": gen.linemap(), "fn_names": gen.fn_names, "pyc": gen.pyc,
               "used": set(used), "names": scope_names, "consts": consts, "code": marshal.dumps(code)}
        body = pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL)
        sig = hmac.new(_secret(), body, hashlib.sha256).digest()
        path = _path(k)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(sig + body)
        os.replace(tmp, path)
    except Exception:                                  # noqa: BLE001 - the cache is an optimisation only
        pass
