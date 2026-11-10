"""`use memory` — RAM, SSD, CPU and GPU managed for you.

* status()            what the machine has right now: RAM, swap, disk space for the cache, cores, load, GPU memory.
* budget(gb)          the RAM this program may use for big arrays (default: 70% of what is free at start).
* array(shape, dtype) a NumPy array in RAM while it fits the budget, otherwise backed by a file on the SSD
                      (memory-mapped: same code, no MemoryError). free(a) releases either kind.
* store(name)         a key/value cache with two tiers: a RAM tier bounded in bytes (least recently used values
                      leave first) and an SSD tier bounded in GB. Arrays are stored as .npy and come back
                      memory-mapped; other values are pickled and signed so a tampered file is never loaded.
* cached(fn)          memoise a function through a store: results survive restarts.
* chunks(a, rows)     iterate a big array (or .npy file) in row blocks, reading the next block from the SSD while
                      the current one is processed.
* map(fn, a, rows)    fn over row blocks on all cores (NumPy releases the GIL), results written into a RAM or
                      SSD-backed output array.
* fit_batch(fn, n)    call fn(batch) and halve the batch on out-of-memory (CPU MemoryError, CUDA OOM) until it
                      fits; returns (result, batch) so the next call starts from a size that works.
* devices()           CPU and GPUs with free memory; best_device() picks the GPU with the most free memory or
                      the CPU.

Files go to ~/.apsi/spill and ~/.apsi/store (or APSI_SPILL_DIR); those folders belong to Apsi, so no
permission is needed. A store at a path you choose needs --allow-write for that path.
"""
import collections
import hashlib
import hmac
import os
import pickle
import shutil
import sys
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_BUDGET = None          # bytes of RAM for array(); None = decided on first use
_USED = 0               # bytes handed out by array() in RAM
_LOCK = threading.Lock()


# ------------------------------------------------------------------ machine status
def _meminfo():
    """(total, available, swap_total, swap_free) in bytes."""
    try:
        vals = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                vals[k] = int(v.split()[0]) * 1024
        return vals["MemTotal"], vals.get("MemAvailable", vals["MemFree"]), vals.get("SwapTotal", 0), \
            vals.get("SwapFree", 0)
    except (OSError, KeyError, ValueError):
        pass
    if sys.platform.startswith("win"):
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys, m.ullAvailPhys, m.ullTotalPageFile - m.ullTotalPhys, \
            max(0, m.ullAvailPageFile - m.ullAvailPhys)
    if sys.platform == "darwin":
        return _mac_meminfo()
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        return total, avail, 0, 0
    except (ValueError, OSError, AttributeError):
        return 0, 0, 0, 0


_MAC = [0.0, None]


def _mac_meminfo():
    """macOS: total from sysctl hw.memsize; available = free + inactive + purgeable + speculative pages (vm_stat).
    Cached for one second (allocation decisions ask often)."""
    import subprocess
    import time as _t
    if _MAC[1] is not None and _t.monotonic() - _MAC[0] < 1.0:
        return _MAC[1]
    total = avail = swt = swf = 0
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                   timeout=5).stdout.strip() or 0)
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        first = out.splitlines()[0] if out else ""
        if "page size of" in first:
            page = int(first.split("page size of")[1].split()[0])
        pages = {}
        for line in out.splitlines()[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    pages[k.strip()] = int(v)
        avail = page * sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages purgeable",
                                                     "Pages speculative"))
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        def mb(tag):
            part = sw.split(tag + " = ")[1].split()[0] if tag + " = " in sw else "0M"
            unit = {"K": 2**10, "M": 2**20, "G": 2**30}.get(part[-1], 1)
            return int(float(part.rstrip("KMG")) * unit)
        swt, swf = mb("total"), mb("free")
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        pass
    _MAC[0], _MAC[1] = _t.monotonic(), (total, avail, swt, swf)
    return _MAC[1]


def _rss():
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024
    except Exception:                     # noqa: BLE001
        return 0


def _gpus():
    out = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                out.append({"index": i, "name": torch.cuda.get_device_name(i), "free_gb": free / 2**30,
                            "total_gb": total / 2**30, "via": "torch"})
            return out
    except Exception:                     # noqa: BLE001 - no torch / no CUDA
        pass
    try:
        import cupy
        for i in range(cupy.cuda.runtime.getDeviceCount()):
            with cupy.cuda.Device(i):
                free, total = cupy.cuda.runtime.memGetInfo()
            out.append({"index": i, "name": f"cuda:{i}", "free_gb": free / 2**30, "total_gb": total / 2**30,
                        "via": "cupy"})
    except Exception:                     # noqa: BLE001
        pass
    return out


def spill_dir():
    d = os.environ.get("APSI_SPILL_DIR") or os.path.join(_config_dir(), "spill")
    os.makedirs(d, exist_ok=True)
    return d


def _config_dir():
    from .security import config_dir
    return config_dir()


class Status:
    """Readable as text, fields as attributes (s.ram_free_gb), or as_dict()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, k):
        return self.__dict__[k]

    def as_dict(self):
        return dict(self.__dict__)

    def __str__(self):
        g = "; ".join(f"GPU {x['index']} {x['free_gb']:.1f}/{x['total_gb']:.1f} GB free" for x in self["gpus"]) \
            or "no GPU"
        return (f"RAM {self['ram_free_gb']:.1f}/{self['ram_total_gb']:.1f} GB free (this program "
                f"{self['process_gb']:.2f} GB, budget {self['budget_gb']:.1f} GB) | swap "
                f"{self['swap_free_gb']:.1f} GB | SSD {self['disk_free_gb']:.0f} GB free at {self['spill_dir']} | "
                f"{self['cpus']} cores, load {self['load']:.2f} | {g}")

    __repr__ = __str__


def status():
    total, avail, swt, swf = _meminfo()
    d = spill_dir()
    du = shutil.disk_usage(d)
    try:
        load = os.getloadavg()[0]
    except (OSError, AttributeError):
        load = 0.0
    return Status(ram_total_gb=total / 2**30, ram_free_gb=avail / 2**30, swap_total_gb=swt / 2**30,
                  swap_free_gb=swf / 2**30, process_gb=_rss() / 2**30, budget_gb=_budget() / 2**30,
                  in_ram_gb=_USED / 2**30, disk_free_gb=du.free / 2**30, spill_dir=d, cpus=os.cpu_count() or 1,
                  load=load, gpus=_gpus())


def devices():
    out = [{"name": "cpu", "cores": os.cpu_count() or 1, "free_gb": _meminfo()[1] / 2**30}]
    out += [dict(g, name=f"cuda:{g['index']}") for g in _gpus()]
    return out


def best_device(need_gb=0.0):
    """The GPU with the most free memory (if it has need_gb free), otherwise "cpu"."""
    gpus = sorted(_gpus(), key=lambda g: -g["free_gb"])
    if gpus and gpus[0]["free_gb"] >= float(need_gb):
        return f"cuda:{gpus[0]['index']}"
    return "cpu"


# ------------------------------------------------------------------ RAM budget and spilling arrays
def _budget():
    global _BUDGET
    if _BUDGET is None:
        env = os.environ.get("APSI_RAM_BUDGET")
        if env:
            from .security import parse_size
            _BUDGET = parse_size(env)
        else:
            _BUDGET = int(_meminfo()[1] * 0.7) or 2**31
    return _BUDGET


def budget(gb=None):
    """Set (or read) the RAM budget for array(), in GB."""
    global _BUDGET
    if gb is not None:
        _BUDGET = int(float(gb) * 2**30)
    return _budget() / 2**30


_SPILLED = {}           # id(array) -> path, removed when the array is freed or collected


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


def array(shape, dtype="float32", fill=None, ram=None):
    """An array that lives in RAM while it fits the budget, else in a memory-mapped file on the SSD.
    ram=true/false forces the choice."""
    global _USED
    shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
    dt = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    with _LOCK:
        in_ram = ram if ram is not None else (_USED + nbytes <= _budget() and nbytes <= _meminfo()[1] * 0.9)
        if in_ram:
            _USED += nbytes
    if in_ram:
        a = np.zeros(shape, dt) if fill is None else np.full(shape, fill, dt)   # zeros: lazy pages, like a file
        weakref.finalize(a, _release, nbytes)
        return a
    fd, path = tempfile.mkstemp(suffix=".npy", dir=spill_dir())
    os.close(fd)
    a = np.lib.format.open_memmap(path, mode="w+", dtype=dt, shape=shape)
    if fill is not None and fill != 0:
        a[...] = fill
    weakref.finalize(a, _cleanup, path)
    _SPILLED[id(a)] = path
    return a


def _release(n):
    global _USED
    with _LOCK:
        _USED = max(0, _USED - n)


def on_disk(a):
    return isinstance(a, np.memmap) or id(a) in _SPILLED


def free(a):
    """Release an array from array() now (RAM is returned, a spill file is deleted)."""
    path = _SPILLED.pop(id(a), None)
    if path is not None:
        mm = getattr(a, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
        _cleanup(path)


# ------------------------------------------------------------------ out-of-core iteration
def _open(a):
    if isinstance(a, str):
        return np.load(a, mmap_mode="r")
    return a


def chunks(a, rows=65536):
    """Row blocks of a (memory-mapped) array; the next block is read ahead on another thread."""
    a = _open(a)
    n = len(a)
    rows = max(1, int(rows))
    if n == 0:
        return
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(lambda s: np.ascontiguousarray(a[s:s + rows]), 0)
        for s in range(0, n, rows):
            cur = nxt.result()
            if s + rows < n:
                nxt = ex.submit(lambda s2: np.ascontiguousarray(a[s2:s2 + rows]), s + rows)
            yield cur


def map(fn, a, rows=65536, out=None, workers=None):
    """out[i:j] = fn(a[i:j]) for row blocks, on all cores. out may be omitted (allocated with array())."""
    a = _open(a)
    n = len(a)
    rows = max(1, int(rows))
    workers = int(workers or os.cpu_count() or 1)
    starts = list(range(0, n, rows))
    if not starts:
        return out
    first = np.asarray(fn(np.ascontiguousarray(a[0:min(rows, n)])))
    if out is None:
        out = array((n,) + first.shape[1:], first.dtype)
    out[0:len(first)] = first

    def job(s):
        out[s:s + rows] = fn(np.ascontiguousarray(a[s:s + rows]))
    if len(starts) > 1:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, starts[1:]))
    return out


# ------------------------------------------------------------------ out-of-memory safe batches
def _is_oom(e):
    if isinstance(e, MemoryError):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cudaerrormemoryallocation" in msg \
        or type(e).__name__ in ("OutOfMemoryError", "OutOfMemory")


def fit_batch(fn, batch, minimum=1):
    """fn(batch); on out-of-memory the batch is halved (GPU cache emptied) and tried again."""
    b = int(batch)
    while True:
        try:
            return fn(b), b
        except Exception as e:             # noqa: BLE001 - only out-of-memory is retried
            if not _is_oom(e) or b <= int(minimum):
                raise
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:              # noqa: BLE001
                pass
            import gc
            gc.collect()
            b = max(int(minimum), b // 2)


# ------------------------------------------------------------------ two-tier store
def _key_of(k):
    if isinstance(k, str) and len(k) <= 120 and all(c.isalnum() or c in "-_." for c in k):
        return k
    return hashlib.sha256(pickle.dumps(k, protocol=4)).hexdigest()


def _size_of(v):
    if isinstance(v, np.ndarray):
        return v.nbytes
    if isinstance(v, (bytes, bytearray, str)):
        return len(v)
    try:
        return len(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:                      # noqa: BLE001
        return sys.getsizeof(v)


class Store:
    """RAM tier (LRU, ram_mb) in front of an SSD tier (LRU by file time, disk_gb)."""

    def __init__(self, path, ram_mb=256, disk_gb=10):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.ram_limit = int(float(ram_mb) * 2**20)
        self.disk_limit = int(float(disk_gb) * 2**30)
        self.ram = collections.OrderedDict()
        self.ram_bytes = 0
        self.hits = self.misses = self.disk_hits = 0
        self.lock = threading.RLock()
        self._key = None

    # signatures for pickled entries
    def _secret(self):
        if self._key is None:
            from .compile_cache import _secret
            self._key = _secret()
        return self._key

    def _file(self, key, ext):
        return os.path.join(self.path, key[:2], key + ext)

    def _ram_put(self, key, v, size):
        if size > self.ram_limit:
            return
        old = self.ram.pop(key, None)
        if old is not None:
            self.ram_bytes -= old[1]
        self.ram[key] = (v, size)
        self.ram_bytes += size
        while self.ram_bytes > self.ram_limit and self.ram:
            _, (_, s) = self.ram.popitem(last=False)
            self.ram_bytes -= s

    def set(self, k, v):
        key = _key_of(k)
        size = _size_of(v)
        with self.lock:
            self._ram_put(key, v, size)
            folder = os.path.join(self.path, key[:2])
            os.makedirs(folder, exist_ok=True)
            if isinstance(v, np.ndarray) and v.dtype != object:
                dst = self._file(key, ".npy")
                tmp = dst + f".{os.getpid()}.tmp.npy"
                np.save(tmp, np.ascontiguousarray(v))
                os.replace(tmp, dst)
                _cleanup(self._file(key, ".pkl"))
            else:
                body = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                sig = hmac.new(self._secret(), body, hashlib.sha256).digest()
                dst = self._file(key, ".pkl")
                tmp = dst + f".{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(sig + body)
                os.replace(tmp, dst)
                _cleanup(self._file(key, ".npy"))
            self._trim()
        return v

    def get(self, k, default=None):
        key = _key_of(k)
        with self.lock:
            hit = self.ram.get(key)
            if hit is not None:
                self.ram.move_to_end(key)
                self.hits += 1
                return hit[0]
            npy = self._file(key, ".npy")
            if os.path.exists(npy):
                v = np.load(npy, mmap_mode="r")
                os.utime(npy)
                self.disk_hits += 1
                return v
            pkl = self._file(key, ".pkl")
            try:
                with open(pkl, "rb") as fh:
                    blob = fh.read()
            except OSError:
                self.misses += 1
                return default
            sig, body = blob[:32], blob[32:]
            if not hmac.compare_digest(sig, hmac.new(self._secret(), body, hashlib.sha256).digest()):
                _cleanup(pkl)                  # changed outside Apsi: never unpickled
                self.misses += 1
                return default
            v = pickle.loads(body)
            os.utime(pkl)
            self._ram_put(key, v, len(body))
            self.disk_hits += 1
            return v

    def __contains__(self, k):
        key = _key_of(k)
        return key in self.ram or os.path.exists(self._file(key, ".npy")) or os.path.exists(self._file(key, ".pkl"))

    __getitem__ = lambda self, k: self._must(k)      # noqa: E731
    __setitem__ = set

    def _must(self, k):
        miss = object()
        v = self.get(k, miss)
        if v is miss:
            raise KeyError(k)
        return v

    def delete(self, k):
        key = _key_of(k)
        with self.lock:
            old = self.ram.pop(key, None)
            if old is not None:
                self.ram_bytes -= old[1]
            _cleanup(self._file(key, ".npy"))
            _cleanup(self._file(key, ".pkl"))

    def _files(self):
        out = []
        for root, _, names in os.walk(self.path):
            for n in names:
                if n.endswith((".npy", ".pkl")) and ".tmp" not in n:
                    p = os.path.join(root, n)
                    try:
                        st = os.stat(p)
                    except OSError:
                        continue
                    out.append((st.st_mtime, st.st_size, p))
        return out

    def _trim(self):
        files = self._files()
        total = sum(f[1] for f in files)
        if total <= self.disk_limit:
            return
        for _, size, p in sorted(files):
            _cleanup(p)
            total -= size
            if total <= self.disk_limit:
                break

    def clear(self):
        with self.lock:
            self.ram.clear()
            self.ram_bytes = 0
            for _, _, p in self._files():
                _cleanup(p)

    def stats(self):
        files = self._files()
        return {"ram_items": len(self.ram), "ram_mb": self.ram_bytes / 2**20, "disk_items": len(files),
                "disk_mb": sum(f[1] for f in files) / 2**20, "hits": self.hits, "disk_hits": self.disk_hits,
                "misses": self.misses}

    def __repr__(self):
        s = self.stats()
        return (f"<store {self.path}: {s['ram_items']} in RAM ({s['ram_mb']:.1f} MB), {s['disk_items']} on SSD "
                f"({s['disk_mb']:.1f} MB)>")


def _store_path(name):
    if os.path.isabs(name) or os.sep in name or "/" in name:
        return name, True
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "default"
    return os.path.join(os.environ.get("APSI_SPILL_DIR") or _config_dir(), "store", safe), False


def module(interp=None):
    def store(name="default", ram_mb=256, disk_gb=10):
        path, custom = _store_path(str(name))
        if custom and interp is not None:
            from .security import check_path
            check_path(interp, path, "write", "memory.store()")
        return Store(path, ram_mb, disk_gb)

    def cached(fn=None, name=None, ram_mb=128, disk_gb=5):
        """Memoise fn in a two-tier store (RAM, then SSD). Use as @memory.cached or memory.cached(fn)."""
        def wrap(f):
            st = store(name or f"fn-{getattr(f, '__module__', '')}.{getattr(f, '__qualname__', 'f')}", ram_mb,
                       disk_gb)
            miss = object()

            def inner(*args, **kwargs):
                k = (args, tuple(sorted(kwargs.items())))
                v = st.get(k, miss)
                if v is miss:
                    v = st.set(k, f(*args, **kwargs))
                return v
            inner.store = st
            inner.__name__ = getattr(f, "__name__", "cached")
            inner.__wrapped__ = f
            return inner
        return wrap(fn) if fn is not None else wrap

    def _readable(a):
        if isinstance(a, str) and interp is not None:
            from .security import check_path
            check_path(interp, a, "read", "memory.chunks()/map()")
        return a

    def chunks_(a, rows=65536):
        return chunks(_readable(a), rows)

    def map_(fn, a, rows=65536, out=None, workers=None):
        return map(fn, _readable(a), rows, out, workers)

    return {"status": status, "budget": budget, "array": array, "free": free, "on_disk": on_disk,
            "chunks": chunks_, "map": map_, "fit_batch": fit_batch, "store": store, "cached": cached,
            "devices": devices, "best_device": best_device, "spill_dir": spill_dir}


_ = time
