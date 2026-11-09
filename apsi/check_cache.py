"""A signed "checked clean" marker per (Apsi version, checker, source, folder): a large program that passed the
checker is not checked again until it, a file it imports, or Apsi itself changes. Markers live next to the compile
cache and carry an HMAC made with this user's key, so a marker copied from elsewhere is ignored."""
import hashlib
import hmac
import json
import os

MIN_SIZE = 5000            # small programs check faster than a disk lookup
_FP = None


def _fingerprint(interp):
    global _FP
    if _FP is None:
        from . import __version__
        h = hashlib.sha256(__version__.encode())
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("checker.py", "parser.py", "lexer.py", "ast.py", "units.py"):
            try:
                st = os.stat(os.path.join(here, name))
                h.update(f"{name}:{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                h.update(name.encode())
        h.update("\0".join(sorted(interp.globals.vars)).encode())
        _FP = h.hexdigest()
    return _FP


def _path(src, base, interp):
    from .compile_cache import cache_dir
    key = hashlib.sha256((_fingerprint(interp) + "\0" + base + "\0" + src).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir(), "checked", key[:2], key + ".ok"), key


def _sign(key, deps):
    from .compile_cache import _secret
    return hmac.new(_secret(), (key + json.dumps(deps)).encode(), hashlib.sha256).hexdigest()


def _stamp(path):
    st = os.stat(path)
    return [path, st.st_size, st.st_mtime_ns]


def known_clean(src, base, interp):
    from .compile_cache import enabled
    if len(src) < MIN_SIZE or not enabled():
        return False
    try:
        path, key = _path(src, base, interp)
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if not hmac.compare_digest(rec["sig"], _sign(key, rec["deps"])):
            return False
        return all(_stamp(d[0]) == d for d in rec["deps"])
    except Exception:                      # noqa: BLE001 - no usable marker: check normally
        return False


def remember(src, base, interp, deps):
    from .compile_cache import enabled
    if len(src) < MIN_SIZE or not enabled():
        return
    try:
        path, key = _path(src, base, interp)
        stamps = [_stamp(d) for d in dict.fromkeys(deps)]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"sig": _sign(key, stamps), "deps": stamps}, fh)
        os.replace(tmp, path)
    except Exception:                      # noqa: BLE001 - the marker is an optimisation only
        pass
