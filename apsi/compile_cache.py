"""Compile cache: parsed programs are kept so that a program is parsed once, not on every run.

* In memory: the same source is parsed once per process (the Guard scan, the checker, imports and the run share it).
* On disk (``~/.apsi/cache/compiled``): parse results of program files survive between runs. Every entry is a
  pickle prefixed with an HMAC-SHA256 signature made with this user's secret key (``~/.apsi/cache.key``); an entry
  whose signature does not verify is never unpickled. Entries are keyed by the SHA-256 of the source and by a
  fingerprint of the Apsi front end (version, lexer, parser, AST), so an updated Apsi never reuses stale results.
* ``apsi compile PATHS`` fills the cache ahead of time; ``apsi compile --clear`` empties it.
* ``APSI_NO_CACHE=1`` turns the disk cache off.
"""
import hashlib
import hmac
import os
import pickle
from collections import OrderedDict

MAX_ENTRIES = 2000           # oldest entries are removed beyond this
_MEMO = OrderedDict()
_MEMO_SIZE = 128
_FINGERPRINT = None


def cache_dir():
    from .security import config_dir
    return os.path.join(config_dir(), "cache", "compiled")


def enabled():
    return os.environ.get("APSI_NO_CACHE", "").strip() not in ("1", "true", "yes")


def _fingerprint():
    """Changes whenever the front end that produces the cached objects changes."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        from . import __version__
        h = hashlib.sha256(__version__.encode())
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ("lexer.py", "parser.py", "ast.py"):
            try:
                st = os.stat(os.path.join(here, name))
                h.update(f"{name}:{st.st_size}:{int(st.st_mtime)}".encode())
            except OSError:                       # inside a zip (apsi build): the version pins it
                h.update(name.encode())
        _FINGERPRINT = h.hexdigest()[:16]
    return _FINGERPRINT


def _key(source):
    return hashlib.sha256((_fingerprint() + "\0" + source).encode("utf-8")).hexdigest()


def _secret():
    from .data import _cache_key
    return _cache_key()


def _remember(key, prog):
    _MEMO[key] = prog
    _MEMO.move_to_end(key)
    while len(_MEMO) > _MEMO_SIZE:
        _MEMO.popitem(last=False)


def load(source, disk=True):
    key = _key(source)
    prog = _MEMO.get(key)
    if prog is not None:
        _MEMO.move_to_end(key)
        return prog
    if not (disk and enabled()):
        return None
    path = os.path.join(cache_dir(), key[:2], key + ".ast")
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return None
    sig, body = blob[:32], blob[32:]
    try:
        if not hmac.compare_digest(sig, hmac.new(_secret(), body, hashlib.sha256).digest()):
            _drop(path)                           # tampered or from another machine: never unpickled
            return None
        prog = pickle.loads(body)
    except Exception:
        _drop(path)
        return None
    _remember(key, prog)
    return prog


def store(source, prog, disk=True):
    key = _key(source)
    _remember(key, prog)
    if not (disk and enabled()):
        return
    folder = os.path.join(cache_dir(), key[:2])
    path = os.path.join(folder, key + ".ast")
    try:
        if os.path.exists(path):
            return
        body = pickle.dumps(prog, protocol=pickle.HIGHEST_PROTOCOL)
        sig = hmac.new(_secret(), body, hashlib.sha256).digest()
        os.makedirs(folder, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(sig + body)
        os.replace(tmp, path)
        _prune()
    except (OSError, pickle.PicklingError, RecursionError):
        pass                                      # the cache is an optimisation, never a failure


def preload(entries):
    """Trusted parse results shipped inside a program built with `apsi build` (source -> program)."""
    for source, prog in entries:
        _remember(_key(source), prog)


def _drop(path):
    try:
        os.remove(path)
    except OSError:
        pass


_writes = 0


def _prune():
    global _writes
    _writes += 1
    if _writes % 50:
        return
    files = []
    for root, _, names in os.walk(cache_dir()):
        for n in names:
            if n.endswith(".ast"):
                p = os.path.join(root, n)
                try:
                    files.append((os.path.getmtime(p), p))
                except OSError:
                    pass
    if len(files) > MAX_ENTRIES:
        files.sort()
        for _, p in files[:len(files) - MAX_ENTRIES]:
            _drop(p)


def stats():
    n = size = 0
    for root, _, names in os.walk(cache_dir()):
        for f in names:
            if f.endswith(".ast"):
                n += 1
                try:
                    size += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return n, size


def clear():
    import shutil
    n, _ = stats()
    shutil.rmtree(cache_dir(), ignore_errors=True)
    _MEMO.clear()
    return n
