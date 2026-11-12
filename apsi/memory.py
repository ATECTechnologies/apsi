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
* limits()            the memory this process may really use: the machine's, or the container's (cgroup v1/v2)
                      limit when it is smaller - every decision here uses the smaller one.
* pressure()          ok / moderate / high / critical, from free memory and the kernel's pressure stall
                      information (Linux PSI). Under high pressure arrays go to the SSD, stores and pools shrink
                      their RAM tiers by themselves; relieve() does it now; on_pressure(fn) calls you.
* pool()              reusable buffers: take(shape, dtype) / give(a) hand out memory that is already mapped, so
                      loops that allocate the same big arrays again and again skip the page faults.
* report()            every array made by array(): shape, size, RAM or SSD, file.

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
_CG = [0.0, None]
CGROUP_FILE = "/proc/self/cgroup"      # module constants so tests can point them at a fake hierarchy
CGROUP_ROOT = "/sys/fs/cgroup"


def _read_int(path):
    try:
        with open(path, encoding="ascii") as fh:
            v = fh.read().strip()
    except OSError:
        return None
    if v in ("", "max"):
        return None
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n < 2**60 else None


def _stat_value(path, key):
    try:
        with open(path, encoding="ascii") as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                if k == key:
                    return int(v)
    except (OSError, ValueError):
        pass
    return 0


def _cgroup():
    """(limit, usage, reclaimable, pressure file or None, version) of this process's memory cgroup when it limits
    memory, else None. Containers (Docker, Kubernetes, systemd slices) show the host's RAM in /proc/meminfo but
    kill the process at the cgroup limit, so the limit is what counts. Cached for half a second."""
    now = time.monotonic()
    if _CG[1] is not None and now - _CG[0] < 0.5:
        return _CG[1] or None
    found = False
    try:
        with open(CGROUP_FILE, encoding="ascii") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrl, path = parts
        if hid == "0" and ctrl == "":                                   # cgroup v2
            base = CGROUP_ROOT
            d = os.path.normpath(base + path)
            best = None
            while d.startswith(base):
                lim = _read_int(os.path.join(d, "memory.max"))
                if lim is not None and (best is None or lim < best[0]):
                    best = (lim, d)
                if d == base:
                    break
                d = os.path.dirname(d)
            if best is not None:
                lim, d = best
                use = _read_int(os.path.join(d, "memory.current")) or 0
                rec = _stat_value(os.path.join(d, "memory.stat"), "inactive_file")
                pf = os.path.join(d, "memory.pressure")
                _CG[1] = (lim, use, rec, pf if os.path.exists(pf) else None, 2)
                found = True
                break
        elif "memory" in ctrl.split(","):                               # cgroup v1
            d = os.path.join(CGROUP_ROOT, "memory", path.lstrip("/"))
            lim = _read_int(os.path.join(d, "memory.limit_in_bytes"))
            if lim is None:
                d = os.path.join(CGROUP_ROOT, "memory")
                lim = _read_int(os.path.join(d, "memory.limit_in_bytes"))
            if lim is not None:
                use = _read_int(os.path.join(d, "memory.usage_in_bytes")) or 0
                rec = _stat_value(os.path.join(d, "memory.stat"), "total_inactive_file")
                _CG[1] = (lim, use, rec, None, 1)
                found = True
                break
    if not found:
        _CG[1] = ()
    _CG[0] = now
    return _CG[1] or None


def _meminfo():
    """(total, available, swap_total, swap_free) in bytes that this process can really use: the machine's
    memory, or the container's cgroup limit when that is smaller."""
    total, avail, swt, swf = _host_meminfo()
    cg = _cgroup()
    if cg is not None:
        lim, use, rec = cg[0], cg[1], cg[2]
        if lim < total or not total:
            total = lim
        avail = max(0, min(avail, lim - use + rec)) if avail else max(0, lim - use + rec)
    return total, avail, swt, swf


def _host_meminfo():
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


def limits():
    """The memory this process may use and where the limit comes from."""
    host = _host_meminfo()
    cg = _cgroup()
    total, avail, _, _ = _meminfo()
    src = f"container (cgroup v{cg[4]})" if cg is not None and cg[0] <= host[0] else "machine"
    return Status(ram_total_gb=total / 2**30, ram_free_gb=avail / 2**30, machine_ram_gb=host[0] / 2**30,
                  cgroup_limit_gb=(cg[0] / 2**30) if cg else None, cgroup_used_gb=(cg[1] / 2**30) if cg else None,
                  source=src, budget_gb=_budget() / 2**30, _text=lambda: (
                      f"{total / 2**30:.1f} GB usable ({src}), {avail / 2**30:.1f} GB free, budget "
                      f"{_budget() / 2**30:.1f} GB"))


LEVELS = {"ok": 0, "moderate": 1, "high": 2, "critical": 3}


def _psi(path=None):
    """(some avg10, full avg10) in % from Linux pressure stall information, or (0, 0)."""
    path = path or "/proc/pressure/memory"
    some = full = 0.0
    try:
        with open(path, encoding="ascii") as fh:
            for line in fh:
                kind, _, rest = line.partition(" ")
                for item in rest.split():
                    if item.startswith("avg10="):
                        v = float(item[6:])
                        if kind == "some":
                            some = v
                        elif kind == "full":
                            full = v
    except (OSError, ValueError):
        pass
    return some, full


def pressure():
    """How hard the system is pressed for memory: level ok / moderate / high / critical, with the numbers."""
    total, avail, _, _ = _meminfo()
    frac = avail / total if total else 1.0
    cg = _cgroup()
    some, full = _psi(cg[3] if cg is not None and cg[3] else None)
    if frac < 0.05 or full >= 10:
        level = "critical"
    elif frac < 0.10 or some >= 20 or full >= 2:
        level = "high"
    elif frac < 0.20 or some >= 5:
        level = "moderate"
    else:
        level = "ok"
    _LEVEL[0], _LEVEL[1] = time.monotonic(), level
    return Status(level=level, ram_free_fraction=frac, ram_free_gb=avail / 2**30, psi_some=some, psi_full=full,
                  _text=lambda: f"{level}: {avail / 2**30:.1f} GB free ({frac * 100:.0f} %), stalls {some:.1f} % "
                                f"(some) / {full:.1f} % (full)")


_LEVEL = [0.0, "ok"]
_SHRINKABLE = weakref.WeakSet()       # stores and pools that give RAM back under pressure


def _level(max_age=0.5):
    """The pressure level, re-measured at most every max_age seconds (cheap enough for hot paths)."""
    if time.monotonic() - _LEVEL[0] >= max_age:
        pressure()
    return _LEVEL[1]


def relieve(level="high"):
    """Give RAM back now: stores drop (part of) their RAM tier, pools their spare buffers, then a garbage
    collection. Nothing is lost - store values stay on the SSD. Returns the bytes released from Apsi's caches."""
    import gc
    hard = LEVELS.get(str(level), 2) >= LEVELS["critical"]
    freed = 0
    for obj in list(_SHRINKABLE):
        try:
            freed += obj._shrink(hard)
        except Exception:                 # noqa: BLE001 - relieving memory must never fail
            pass
    gc.collect()
    return freed


def _react():
    """Called on allocation paths: under high pressure shrink the caches (at most twice a second)."""
    lv = _level()
    if LEVELS[lv] >= LEVELS["high"] and time.monotonic() - _REACTED[0] >= 0.5:
        _REACTED[0] = time.monotonic()
        relieve(lv)
    return lv


_REACTED = [0.0]


class Watcher:
    """A background thread that calls fn(pressure) when the level reaches `level` (again every `repeat`
    seconds while it stays there). stop() ends it."""

    def __init__(self, fn, level="high", every=1.0, repeat=10.0, auto_relieve=True):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        self.fn, self.level, self.every, self.repeat, self.auto = fn, level, float(every), float(repeat), \
            auto_relieve
        self.calls = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="apsi-memory-watch", daemon=True)
        self._t.start()

    def _run(self):
        last = None
        while not self._stop.wait(self.every):
            p = pressure()
            if LEVELS[p.level] >= LEVELS[self.level]:
                now = time.monotonic()
                if last is None or now - last >= self.repeat:
                    last = now
                    if self.auto:
                        relieve(p.level)
                    self.calls += 1
                    if self.fn is not None:
                        try:
                            self.fn(p)
                        except Exception as e:     # noqa: BLE001 - a failing callback must not kill the watch
                            sys.stderr.write(f"apsi: memory.on_pressure callback failed: {e}\n")
            else:
                last = None

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def __repr__(self):
        return f"<memory watcher: level {self.level}, every {self.every:g} s, {self.calls} calls>"


def on_pressure(fn=None, level="high", every=1.0, repeat=10.0):
    """Watch memory pressure in the background; see Watcher."""
    return Watcher(fn, level, every, repeat)


class Pool:
    """Reusable buffers. take(shape, dtype) returns an array whose memory is already mapped (from an earlier
    give, when one of the same size is spare), so a loop that makes the same big temporary arrays skips the page
    faults and zeroing of fresh memory. give(a) hands the memory back - do not use `a` afterwards. Spare buffers
    are bounded (max_gb) and dropped first under memory pressure."""

    def __init__(self, max_gb=None):
        self.limit = int(float(max_gb) * 2**30) if max_gb is not None else None
        self.spare = collections.OrderedDict()        # nbytes -> [buffers]
        self.spare_bytes = 0
        self.mine = {}                                # id(array taken) -> (buffer, nbytes)
        self.taken = self.reused = 0
        self.lock = threading.Lock()
        _SHRINKABLE.add(self)

    def _limit(self):
        return self.limit if self.limit is not None else max(64 * 2**20, _budget() // 10)

    def take(self, shape, dtype="float64", zero=False):
        shape = (int(shape),) if np.isscalar(shape) else tuple(int(x) for x in shape)
        dt = np.dtype(dtype)
        n = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        if LEVELS[_level()] >= LEVELS["high"]:
            self._shrink(True)
        buf = None
        with self.lock:
            self.taken += 1
            lst = self.spare.get(n)
            if lst:
                buf = lst.pop()
                if not lst:
                    del self.spare[n]
                self.spare_bytes -= n
                self.reused += 1
        if buf is None:
            buf = np.empty(max(n, 1), np.uint8)
        a = buf[:n].view(dt).reshape(shape)
        self.mine[id(a)] = (buf, n)
        weakref.finalize(a, self.mine.pop, id(a), None)    # dropped without give(): its memory is just freed
        if zero:
            a.fill(0)
        return a

    def give(self, a):
        hit = self.mine.pop(id(a), None)
        if hit is None:
            raise ValueError("pool.give(): this array was not taken from this pool")
        buf, n = hit
        with self.lock:
            if n > self._limit():
                return
            self.spare.setdefault(n, []).append(buf)
            self.spare.move_to_end(n)
            self.spare_bytes += n
            while self.spare_bytes > self._limit() and self.spare:
                k, lst = next(iter(self.spare.items()))
                lst.pop()
                self.spare_bytes -= k
                if not lst:
                    del self.spare[k]

    def _shrink(self, hard=True):
        with self.lock:
            freed = self.spare_bytes
            self.spare.clear()
            self.spare_bytes = 0
        return freed

    def clear(self):
        self._shrink(True)

    def stats(self):
        return {"taken": self.taken, "reused": self.reused, "spare_mb": self.spare_bytes / 2**20,
                "limit_mb": self._limit() / 2**20}

    def __repr__(self):
        s = self.stats()
        return f"<pool: {s['taken']} taken, {s['reused']} reused, {s['spare_mb']:.1f} MB spare>"


_POOL = []


def pool(max_gb=None):
    """A new buffer pool (max_gb of spare buffers; default a tenth of the RAM budget)."""
    return Pool(max_gb)


_TRACKED = {}          # id(array) -> (weakref, shape, dtype, nbytes, path or None, created)


def report():
    """Arrays made by array() that are still alive: where each one lives and how big it is."""
    rows = []
    for k, (ref, shape, dt, nbytes, path, t0) in list(_TRACKED.items()):
        if ref() is None:
            _TRACKED.pop(k, None)
            continue
        rows.append(Status(shape=list(shape), dtype=dt, gb=nbytes / 2**30, where="SSD" if path else "RAM",
                           file=path, age_s=time.monotonic() - t0,
                           _text=lambda shape=shape, dt=dt, nbytes=nbytes, path=path: (
                               f"{list(shape)} {dt} {nbytes / 2**30:.3f} GB {'SSD ' + path if path else 'RAM'}")))
    return rows


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
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __str__(self):
        if "_text" in self.__dict__:
            return self.__dict__["_text"]()
        g = "; ".join(f"GPU {x['index']} {x['free_gb']:.1f}/{x['total_gb']:.1f} GB free" for x in self["gpus"]) \
            or "no GPU"
        return (f"RAM {self['ram_free_gb']:.1f}/{self['ram_total_gb']:.1f} GB free (this program "
                f"{self['process_gb']:.2f} GB, budget {self['budget_gb']:.1f} GB) | swap "
                f"{self['swap_free_gb']:.1f} GB | pressure {self['pressure']}"
                f"{' (container limit)' if self['limit_source'] == 'container' else ''} | SSD {self['disk_free_gb']:.0f} GB free at {self['spill_dir']} | "
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
                  load=load, gpus=_gpus(), pressure=pressure().level,
                  limit_source=("container" if _cgroup() is not None and _cgroup()[0] <= _host_meminfo()[0]
                                else "machine"))


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
    lv = _react() if ram is None and nbytes >= 2**20 else "ok"
    with _LOCK:
        in_ram = ram if ram is not None else (_USED + nbytes <= _budget() and nbytes <= _meminfo()[1] * 0.9
                                              and LEVELS[lv] < LEVELS["high"])
        if in_ram:
            _USED += nbytes
    if in_ram:
        a = np.zeros(shape, dt) if fill is None else np.full(shape, fill, dt)   # zeros: lazy pages, like a file
        weakref.finalize(a, _release, nbytes)
        _track(a, shape, dt, nbytes, None)
        return a
    fd, path = tempfile.mkstemp(suffix=".npy", dir=spill_dir())
    os.close(fd)
    a = np.lib.format.open_memmap(path, mode="w+", dtype=dt, shape=shape)
    if fill is not None and fill != 0:
        a[...] = fill
    weakref.finalize(a, _cleanup, path)
    _SPILLED[id(a)] = path
    _track(a, shape, dt, nbytes, path)
    return a


def _track(a, shape, dt, nbytes, path):
    k = id(a)
    try:
        ref = weakref.ref(a, lambda _r, k=k: _TRACKED.pop(k, None))
    except TypeError:
        return
    _TRACKED[k] = (ref, shape, str(dt), nbytes, path, time.monotonic())


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


def _madvise(a, start_row, stop_row, advice):
    """madvise() on the pages of rows [start_row, stop_row) of a memory-mapped array (Linux / macOS)."""
    if not isinstance(a, np.memmap) or getattr(a, "mode", "r") == "c" or not len(a) or sys.platform == "win32":
        return
    try:
        import ctypes
        import mmap as _mm
        libc = _LIBC[0]
        if libc is None:
            libc = _LIBC[0] = ctypes.CDLL(None, use_errno=True)
        row = a.strides[0]
        page = _mm.PAGESIZE
        lo = a.ctypes.data + start_row * row
        hi = a.ctypes.data + min(stop_row, len(a)) * row
        lo_al = lo - lo % page
        if advice == "dontneed":                     # whole pages only: never drop a page the next rows share
            lo_al = lo if lo % page == 0 else lo - lo % page + page
            hi = hi - hi % page
        n = hi - lo_al
        if n <= 0:
            return
        flag = {"dontneed": getattr(_mm, "MADV_DONTNEED", 4), "willneed": getattr(_mm, "MADV_WILLNEED", 3),
                "sequential": getattr(_mm, "MADV_SEQUENTIAL", 2)}[advice]
        libc.madvise(ctypes.c_void_p(lo_al), ctypes.c_size_t(n), ctypes.c_int(flag))
    except Exception:                                # noqa: BLE001 - advice only; never fails the caller
        pass


_LIBC = [None]


def chunks(a, rows=65536):
    """Row blocks of an array or a .npy file, as views (no copy). For a file on the SSD the kernel is asked to
    read the next block ahead while this one is used, and each finished block leaves this process's memory, so
    a file much larger than RAM streams with flat memory use (pages are read again if an old block is touched)."""
    a = _open(a)
    n = len(a)
    rows = max(1, int(rows))
    if n == 0:
        return
    shared = isinstance(a, np.memmap) and sys.platform != "win32" and getattr(a, "mode", "r") in ("r", "r+", "w+")
    if not shared:
        for s in range(0, n, rows):
            yield a[s:s + rows]
        return
    _madvise(a, 0, n, "sequential")
    _madvise(a, 0, rows, "willneed")
    for s in range(0, n, rows):
        if s + rows < n:
            _madvise(a, s + rows, s + 2 * rows, "willneed")     # the kernel reads it while this block is used
        yield a[s:s + rows]
        _madvise(a, s, s + rows, "dontneed")


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
        _madvise(a, s, s + rows, "dontneed")            # memory-mapped input / output: keep memory use flat
        _madvise(out, s, s + rows, "dontneed")
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
        self.disk_bytes = None          # bytes on the SSD tier, counted once and then kept up to date
        self._ram_scale = 1.0           # < 1 while memory is under pressure
        _SHRINKABLE.add(self)

    def _shrink(self, hard=False):
        """Memory pressure: drop the RAM tier to half (all when hard). Values stay on the SSD."""
        with self.lock:
            before = self.ram_bytes
            target = 0 if hard else self.ram_bytes // 2
            while self.ram_bytes > target and self.ram:
                _, (_, s) = self.ram.popitem(last=False)
                self.ram_bytes -= s
            self._ram_scale = 0.25 if hard else 0.5
            return before - self.ram_bytes

    def _disk_now(self):
        if self.disk_bytes is None:
            self.disk_bytes = sum(f[1] for f in self._files())
        return self.disk_bytes

    def _disk_add(self, path, old):
        try:
            new = os.path.getsize(path)
        except OSError:
            new = 0
        if self.disk_bytes is not None:
            self.disk_bytes += new - old

    # signatures for pickled entries
    def _secret(self):
        if self._key is None:
            from .compile_cache import _secret
            self._key = _secret()
        return self._key

    def _file(self, key, ext):
        return os.path.join(self.path, key[:2], key + ext)

    def _size_on_disk(self, key):
        n = 0
        for ext in (".npy", ".pkl"):
            try:
                n += os.path.getsize(self._file(key, ext))
            except OSError:
                pass
        return n

    def _ram_put(self, key, v, size):
        lv = _level()
        if LEVELS[lv] >= LEVELS["high"]:
            if time.monotonic() - _REACTED[0] >= 0.5:
                _REACTED[0] = time.monotonic()
                relieve(lv)
        elif self._ram_scale < 1.0 and LEVELS[lv] == LEVELS["ok"]:
            self._ram_scale = 1.0
        limit = int(self.ram_limit * self._ram_scale)
        if size > limit:
            return
        old = self.ram.pop(key, None)
        if old is not None:
            self.ram_bytes -= old[1]
        self.ram[key] = (v, size)
        self.ram_bytes += size
        while self.ram_bytes > limit and self.ram:
            _, (_, s) = self.ram.popitem(last=False)
            self.ram_bytes -= s

    def set(self, k, v):
        key = _key_of(k)
        size = _size_of(v)
        with self.lock:
            self._ram_put(key, v, size)
            folder = os.path.join(self.path, key[:2])
            os.makedirs(folder, exist_ok=True)
            self._disk_now()
            old = self._size_on_disk(key)
            if isinstance(v, np.ndarray) and v.dtype != object:
                dst = self._file(key, ".npy")
                tmp = dst + f".{os.getpid()}.tmp.npy"
                np.save(tmp, np.ascontiguousarray(v))
                os.replace(tmp, dst)
                _cleanup(self._file(key, ".pkl"))
                self._disk_add(dst, old)
            else:
                body = pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
                sig = hmac.new(self._secret(), body, hashlib.sha256).digest()
                dst = self._file(key, ".pkl")
                tmp = dst + f".{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    fh.write(sig + body)
                os.replace(tmp, dst)
                _cleanup(self._file(key, ".npy"))
                self._disk_add(dst, old)
            if self.disk_bytes > self.disk_limit:
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
            gone = self._size_on_disk(key)
            _cleanup(self._file(key, ".npy"))
            _cleanup(self._file(key, ".pkl"))
            if self.disk_bytes is not None:
                self.disk_bytes = max(0, self.disk_bytes - gone)

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
        """Over the SSD limit: remove the least recently used files down to 90 % of the limit (so the next sets
        do not trim again at once). The only full scan of the folder, which also corrects the running total
        when other processes share the store."""
        files = self._files()
        total = sum(f[1] for f in files)
        if total > self.disk_limit:
            goal = int(self.disk_limit * 0.9)
            for _, size, p in sorted(files):
                _cleanup(p)
                total -= size
                if total <= goal:
                    break
        self.disk_bytes = total

    def clear(self):
        with self.lock:
            self.ram.clear()
            self.ram_bytes = 0
            for _, _, p in self._files():
                _cleanup(p)
            self.disk_bytes = 0

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
            "devices": devices, "best_device": best_device, "spill_dir": spill_dir, "limits": limits,
            "pressure": pressure, "relieve": relieve, "on_pressure": on_pressure, "pool": pool, "report": report}


_ = time
