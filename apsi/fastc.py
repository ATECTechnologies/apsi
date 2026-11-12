"""Small C kernels compiled on first use (gcc/clang/cc), cached per user; everything works without them."""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading

_LOCK = threading.Lock()
_LIBS = {}


def _cache_dir():
    base = os.environ.get("APSI_HOME") or os.path.join(os.path.expanduser("~"), ".apsi")
    d = os.path.join(base, "native-c")
    os.makedirs(d, exist_ok=True)
    return d


def load(name):
    """The compiled library for apsi/csrc/<name>.c, or None (no compiler, or it failed)."""
    with _LOCK:
        if name in _LIBS:
            return _LIBS[name]
        lib = None
        if os.environ.get("APSI_NO_C") != "1":
            try:
                lib = _build(name)
            except Exception:                                  # noqa: BLE001 - fall back to NumPy/Python
                lib = None
        _LIBS[name] = lib
        return lib


def _build(name):
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", name + ".c")
    with open(src, "rb") as fh:
        code = fh.read()
    tag = hashlib.sha256(code + sys.platform.encode()).hexdigest()[:16]
    ext = ".dll" if sys.platform.startswith("win") else ".so"
    out = os.path.join(_cache_dir(), f"{name}-{tag}{ext}")
    if not os.path.exists(out):
        cc = next((c for c in (os.environ.get("CC"), "cc", "gcc", "clang") if c and shutil.which(c)), None)
        if cc is None:
            return None
        tmp = out + f".{os.getpid()}.tmp"
        flags = ["-O3", "-shared"] + ([] if sys.platform.startswith("win") else ["-fPIC"])
        r = subprocess.run([cc, *flags, "-o", tmp, src], capture_output=True, timeout=120)
        if r.returncode != 0:
            return None
        os.replace(tmp, out)
    return ctypes.CDLL(out)
