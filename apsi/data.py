"""`use data` — streaming data pipelines and out-of-core storage.

    let ds = from_tensors(X, y) |> shuffle(4096) |> map(augment, workers=8) |> batch(256) |> prefetch(4)
    for xb, yb in ds: ...

* Lazy and streaming: nothing is materialised unless you ask (`cache`, `to_list`).
* Parallel `map` on a worker pool with a bounded in-flight window (order preserved).
* `prefetch` overlaps data loading with compute on a background thread.
* `cache(dir)` stores the processed stream on SSD as memory-mapped shards; later epochs
  read straight from disk at memory-map speed with no re-processing.
* Memory-mapped tensors (`open_tensor`, `load_tensor(mmap=true)`) let you work on data
  larger than RAM; `disk_cache` memoises expensive results on disk.
"""
import hashlib
import io
import os
import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import ops
from .errors import RuntimeErr
from .units import Quantity


class LengthUnknown(RuntimeErr, TypeError):
    pass


class Pipeline:
    """A re-iterable, lazy stream. Each stage wraps a factory that builds a fresh iterator."""

    def __init__(self, factory, length=None, desc="source"):
        self._factory = factory
        self._len = length
        self.desc = desc

    def __iter__(self):
        return iter(self._factory())

    def __len__(self):
        if self._len is None:
            raise LengthUnknown("length of this pipeline is unknown (it has a filter or a streaming source)")
        return self._len

    def __repr__(self):
        return f"<pipeline {self.desc}>"

    # chainable methods (also available as functions for |>)
    def map(self, f, workers=None, ordered=True):
        return map_(self, f, workers, ordered)

    def filter(self, f):
        return filter_(self, f)

    def batch(self, size, drop_last=False):
        return batch(self, size, drop_last)

    def unbatch(self):
        return unbatch(self)

    def shuffle(self, buffer=1024, seed=None):
        return shuffle(self, buffer, seed)

    def prefetch(self, n=2):
        return prefetch(self, n)

    def take(self, n):
        return take(self, n)

    def skip(self, n):
        return skip(self, n)

    def repeat(self, times=None):
        return repeat(self, times)

    def to_list(self):
        return [x for x in self]


# ------------------------------------------------------------------ sources
def from_tensors(*arrays, batch=None, shuffle=False, seed=None, drop_last=False):
    """Rows of one or more equal-length tensors: from_tensors(X, y) yields (x_i, y_i).

    With batch=N it yields whole slices instead (vectorised, much faster for big data);
    shuffle=true reshuffles the row order every epoch."""
    if not arrays:
        raise RuntimeErr("from_tensors needs at least one tensor")
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise RuntimeErr("from_tensors: all tensors must have the same number of rows")
    if batch:
        size = int(batch)
        state = {"epoch": 0}

        def run_batches():
            if shuffle:
                rng = np.random.default_rng(None if seed is None else int(seed) + state["epoch"])
                order = rng.permutation(n)
            else:
                order = None
            state["epoch"] += 1
            stop = n - (n % size) if drop_last else n
            for i in range(0, stop, size):
                if order is None:
                    parts = tuple(a[i:i + size] for a in arrays)
                else:
                    idx = np.sort(order[i:i + size])        # sorted reads are SSD/mmap friendly
                    parts = tuple(a[idx] for a in arrays)
                yield parts[0] if len(parts) == 1 else parts
        count = n // size if drop_last else -(-n // size)
        return Pipeline(run_batches, count, f"tensors(batch={size})")
    if len(arrays) == 1:
        a = arrays[0]
        return Pipeline(lambda: (a[i] for i in range(n)), n, "tensor")
    return Pipeline(lambda: (tuple(a[i] for a in arrays) for i in range(n)), n, "tensors")


def from_list(xs):
    xs = list(xs)
    return Pipeline(lambda: iter(xs), len(xs), "list")


def from_range(n):
    n = int(n)
    return Pipeline(lambda: iter(range(n)), n, "range")


def from_generator(f):
    """f() is called at the start of every epoch and must return an iterable."""
    return Pipeline(lambda: iter(f()), None, "generator")


def _csv_rows(path, skip_header, chunk, delimiter):
    with open(path, encoding="utf-8-sig") as fh:
        for _ in range(int(skip_header)):
            next(fh, None)
        buf = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            if len(buf) >= chunk:
                arr = np.loadtxt(io.StringIO("\n".join(buf)), delimiter=delimiter, ndmin=2)
                yield from arr
                buf = []
        if buf:
            arr = np.loadtxt(io.StringIO("\n".join(buf)), delimiter=delimiter, ndmin=2)
            yield from arr


def _jsonl_rows(path):
    import json
    from .interop import from_py
    with open(path, encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    yield from_py(json.loads(line))
                except ValueError as e:
                    raise RuntimeErr(f"{os.path.basename(path)} line {n}: not valid JSON ({e})")


def _parquet_batches(path, columns, rows):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeErr("from_parquet() needs pyarrow — install it with: apsi pkg install data")
    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=int(rows), columns=list(columns) if columns else None):
        yield {name: np.asarray(b.column(i).to_numpy(zero_copy_only=False))
               for i, name in enumerate(b.schema.names)}


def interleave(pipelines, cycle=True):
    """Alternate items from several pipelines (round robin) until all are exhausted."""
    srcs = [_as_pipeline(p) for p in pipelines]

    def run():
        its = [iter(s) for s in srcs]
        while its:
            alive = []
            for it in its:
                try:
                    yield next(it)
                    alive.append(it)
                except StopIteration:
                    pass
            its = alive
    total = None
    try:
        total = sum(len(s) for s in srcs)
    except Exception:
        total = None
    return Pipeline(run, total, f"interleave({len(srcs)})")


def window(p, size, shift=1, drop_last=True):
    """Sliding windows over a stream: each item is a tensor of `size` consecutive items (time series)."""
    src = _as_pipeline(p)
    size, shift = int(size), int(shift)

    def run():
        from collections import deque
        buf = deque(maxlen=size)
        k = 0
        for x in src:
            buf.append(x)
            if len(buf) == size and k % shift == 0:
                yield np.stack([np.asarray(ops.raw(v)) for v in buf])
            if len(buf) == size:
                k += 1
    return Pipeline(run, None, f"{src.desc} -> window({size}, {shift})")


def shard(p, count, index):
    """Every `count`-th item starting at `index` — split one dataset across workers or machines."""
    src = _as_pipeline(p)
    count, index = int(count), int(index)
    if not 0 <= index < count:
        raise RuntimeErr("shard(): index must be in 0..count-1")
    return Pipeline(lambda: (x for i, x in enumerate(src) if i % count == index), None,
                    f"{src.desc} -> shard({index}/{count})")


def from_npy(path):
    arr = np.load(path, mmap_mode="r")
    return Pipeline(lambda: (np.asarray(arr[i]) for i in range(arr.shape[0])), int(arr.shape[0]), "npy")


def from_files(pattern):
    import glob
    files = sorted(glob.glob(pattern, recursive=True))
    return Pipeline(lambda: iter(files), len(files), "files")


# ------------------------------------------------------------------ stages
def map_(p, f, workers=None, ordered=True):
    """Apply f to every item; workers > 1 runs f in parallel (numpy work releases the GIL)."""
    if callable(p) and not isinstance(p, Pipeline) and not callable(f):
        return [p(x) for x in f]                      # plain map(f, list) still works
    src = _as_pipeline(p)
    w = int(workers) if workers else 1

    def run():
        if w <= 1:
            for x in src:
                yield f(x)
            return
        with ThreadPoolExecutor(max_workers=w) as ex:
            window = []
            limit = 4 * w
            for x in src:
                window.append(ex.submit(f, x))
                if len(window) >= limit:
                    if ordered:
                        yield window.pop(0).result()
                    else:
                        done = next((k for k, fu in enumerate(window) if fu.done()), 0)
                        yield window.pop(done).result()
            for fu in window:
                yield fu.result()
    return Pipeline(run, src._len, f"{src.desc} -> map")


def filter_(p, f):
    if callable(p) and not isinstance(p, Pipeline) and not callable(f):
        return [x for x in f if p(x)]
    src = _as_pipeline(p)
    return Pipeline(lambda: (x for x in src if f(x)), None, f"{src.desc} -> filter")


def _collate(items):
    first = items[0]
    if isinstance(first, tuple):
        return tuple(_collate([it[k] for it in items]) for k in range(len(first)))
    if isinstance(first, dict):
        return {k: _collate([it[k] for it in items]) for k in first}
    if isinstance(first, (np.ndarray, Quantity, int, float, np.number)):
        return ops.stack(items)
    return list(items)


def batch(p, size, drop_last=False):
    src = _as_pipeline(p)
    size = int(size)

    def run():
        buf = []
        for x in src:
            buf.append(x)
            if len(buf) == size:
                yield _collate(buf)
                buf = []
        if buf and not drop_last:
            yield _collate(buf)
    n = None
    if src._len is not None:
        n = src._len // size if drop_last else -(-src._len // size)
    return Pipeline(run, n, f"{src.desc} -> batch({size})")


def unbatch(p):
    src = _as_pipeline(p)

    def run():
        for b in src:
            if isinstance(b, tuple):
                for row in zip(*b):
                    yield tuple(row)
            else:
                yield from b
    return Pipeline(run, None, f"{src.desc} -> unbatch")


def shuffle(p, buffer=1024, seed=None):
    """Streaming shuffle with a bounded buffer (a full shuffle when buffer >= dataset size)."""
    src = _as_pipeline(p)
    state = {"epoch": 0}

    def run():
        rng = np.random.default_rng(None if seed is None else int(seed) + state["epoch"])
        state["epoch"] += 1
        buf = []
        for x in src:
            if len(buf) < buffer:
                buf.append(x)
                continue
            k = int(rng.integers(len(buf)))
            yield buf[k]
            buf[k] = x
        rng.shuffle(buf)
        yield from buf
    return Pipeline(run, src._len, f"{src.desc} -> shuffle")


_END = object()


def prefetch(p, n=2):
    """Produce up to n items ahead on a background thread (overlaps loading and compute)."""
    src = _as_pipeline(p)

    def run():
        q = queue.Queue(maxsize=max(1, int(n)))
        err = []
        stop = threading.Event()

        def worker():
            try:
                for x in src:
                    while not stop.is_set():
                        try:
                            q.put(x, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
            except BaseException as e:      # surface errors in the consumer
                err.append(e)
            finally:
                q.put(_END)
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while True:
                x = q.get()
                if x is _END:
                    if err:
                        raise err[0]
                    return
                yield x
        finally:
            stop.set()
    return Pipeline(run, src._len, f"{src.desc} -> prefetch({n})")


def take(p, n):
    src = _as_pipeline(p)

    def run():
        for k, x in enumerate(src):
            if k >= n:
                return
            yield x
    return Pipeline(run, None if src._len is None else min(src._len, int(n)), f"{src.desc} -> take")


def skip(p, n):
    src = _as_pipeline(p)

    def run():
        for k, x in enumerate(src):
            if k >= n:
                yield x
    return Pipeline(run, None if src._len is None else max(0, src._len - int(n)), f"{src.desc} -> skip")


def repeat(p, times=None):
    src = _as_pipeline(p)

    def run():
        k = 0
        while times is None or k < times:
            yield from src
            k += 1
    return Pipeline(run, None if times is None or src._len is None else src._len * int(times),
                    f"{src.desc} -> repeat")


def _as_pipeline(p):
    if isinstance(p, Pipeline):
        return p
    if isinstance(p, (list, tuple, range)):
        return from_list(p)
    if isinstance(p, np.ndarray):
        return from_tensors(p)
    raise RuntimeErr(f"expected a pipeline, got {type(p).__name__}")


def throughput(p, limit=None):
    """Measure items per second of a pipeline (optionally only the first `limit` items)."""
    from .values import Record
    t0 = time.perf_counter()
    n = 0
    for _ in (take(p, limit) if limit else p):
        n += 1
    dt = time.perf_counter() - t0
    return Record(items=n, seconds=dt, per_second=n / dt if dt > 0 else float("inf"))


# ------------------------------------------------------------------ on-disk storage (needs fs)
def install(interp):
    from .security import check_path

    def _rpath(p):
        return check_path(interp, p, "read", "reading data")

    def _wpath(p):
        return check_path(interp, p, "write", "writing data")

    def _rwpath(p):
        return check_path(interp, p, "rw", "disk cache")

    def from_csv(path, skip_header=0, chunk=10000, delimiter=","):
        full = _rpath(path)
        return Pipeline(lambda: _csv_rows(full, skip_header, int(chunk), delimiter), None, "csv")

    def from_npy_(path):
        return from_npy(_rpath(path))

    def save_tensor(path, x):
        full = _wpath(path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        np.save(full, np.asarray(ops.raw(x)))
        return full if full.endswith(".npy") else full + ".npy"

    def load_tensor(path, mmap=False):
        full = _rpath(path)
        if not full.endswith(".npy") and not os.path.exists(full):
            full += ".npy"
        return np.load(full, mmap_mode="r" if mmap else None)

    def open_tensor(path, shape, dtype="f32", mode="w+"):
        """A memory-mapped tensor on disk (SSD): larger than RAM, written/read lazily."""
        full = _rpath(path) if mode == "r" else _rwpath(path)
        if not full.endswith(".npy"):
            full += ".npy"
        npd = {"f64": np.float64, "f32": np.float32, "f16": np.float16, "i64": np.int64,
               "i32": np.int32, "u8": np.uint8, "int8": np.int8, "uint8": np.uint8}[getattr(dtype, "name", dtype)]
        if mode == "r" or (mode == "r+" and os.path.exists(full)):
            return np.lib.format.open_memmap(full, mode=mode)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        shp = tuple(int(s) for s in (shape if isinstance(shape, (list, tuple)) else (shape,)))
        return np.lib.format.open_memmap(full, mode="w+", dtype=npd, shape=shp)

    def flush(x):
        if hasattr(x, "flush"):
            x.flush()

    def cache(p, directory=None, shard=4096):
        """Cache a pipeline. directory=none -> RAM; otherwise memory-mapped shards on disk."""
        src = _as_pipeline(p)
        if directory is None:
            store = []
            done = [False]

            def run_mem():
                if done[0]:
                    yield from store
                    return
                store.clear()
                for x in src:
                    store.append(x)
                    yield x
                done[0] = True
            return Pipeline(run_mem, src._len, f"{src.desc} -> cache(ram)")
        root = _rwpath(directory)

        def run_disk():
            marker = os.path.join(root, "_complete")
            if os.path.exists(marker):
                shards = []
                while os.path.exists(os.path.join(root, f"shard_{len(shards):06d}.pkl")):
                    shards.append(os.path.join(root, f"shard_{len(shards):06d}.pkl"))
                if not all(_cache_ok(p) for p in shards):
                    _discard(root, "pipeline cache")
            if os.path.exists(marker):
                k = 0
                while os.path.exists(os.path.join(root, f"shard_{k:06d}.pkl")):
                    yield from _signed_load(os.path.join(root, f"shard_{k:06d}.pkl"))
                    k += 1
                return
            os.makedirs(root, exist_ok=True)
            buf, k = [], 0
            for x in src:
                buf.append(x)
                yield x
                if len(buf) >= shard:
                    _write_shard(root, k, buf)
                    buf, k = [], k + 1
            if buf:
                _write_shard(root, k, buf)
            open(marker, "w").close()
        return Pipeline(run_disk, src._len, f"{src.desc} -> cache({directory})")

    def disk_cache(directory="cache"):
        """@disk_cache("cache"): memoise a function's results on disk across runs."""
        root = _rwpath(directory)

        def deco(fn):
            def wrapper(*args, **kwargs):
                key = _hash((getattr(fn, "name", ""), args, sorted(kwargs.items())))
                path = os.path.join(root, key + ".pkl")
                if os.path.exists(path):
                    if _cache_ok(path):
                        return _signed_load(path)
                    _discard(path, "cache file")
                r = fn(*args, **kwargs)
                os.makedirs(root, exist_ok=True)
                _signed_dump(path, r)
                return r
            for attr in ("name", "_apsi_def"):
                if hasattr(fn, attr):
                    setattr(wrapper, attr, getattr(fn, attr))
            return wrapper
        return deco

    def from_jsonl(path):
        full = _rpath(path)
        return Pipeline(lambda: _jsonl_rows(full), None, "jsonl")

    def from_parquet(path, columns=None, rows=65536):
        """Stream a Parquet file in record batches: each item is a dict of column -> tensor."""
        full = _rpath(path)
        return Pipeline(lambda: _parquet_batches(full, columns, rows), None, "parquet")

    mod = {
        "from_jsonl": from_jsonl, "from_parquet": from_parquet, "interleave": interleave, "window": window,
        "shard": shard,
        "from_tensors": from_tensors, "from_list": from_list, "from_range": from_range,
        "from_generator": from_generator, "from_csv": from_csv, "from_npy": from_npy_,
        "from_files": lambda pattern: from_files(_rpath(pattern)),
        "map": map_, "filter": filter_, "batch": batch, "unbatch": unbatch, "shuffle": shuffle,
        "prefetch": prefetch, "take": take, "skip": skip, "repeat": repeat, "cache": cache,
        "throughput": throughput, "save_tensor": save_tensor, "load_tensor": load_tensor,
        "open_tensor": open_tensor, "flush": flush, "disk_cache": disk_cache,
    }
    interp.modules["data"] = mod


# ------------------------------------------------------------------ signed cache files
# Cache files are pickles; to make tampering impossible each file is prefixed with an
# HMAC-SHA256 computed with a per-user secret key (~/.apsi/cache.key). Files whose
# signature does not verify are never unpickled.
def _cache_key():
    import hmac  # noqa: F401
    import secrets
    home = os.environ.get("APSI_HOME") or os.path.join(os.path.expanduser("~"), ".apsi")
    path = os.path.join(home, "cache.key")
    try:
        with open(path, "rb") as fh:
            k = fh.read()
            if len(k) == 32:
                return k
    except OSError:
        pass
    os.makedirs(home, exist_ok=True)
    k = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(k)
    return k


def _signed_dump(path, obj):
    import hmac
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sig = hmac.new(_cache_key(), body, hashlib.sha256).digest()
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(sig + body)
    os.replace(tmp, path)


class CacheInvalid(RuntimeErr):
    pass


def _cache_ok(path):
    import hmac
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        return False
    sig, body = blob[:32], blob[32:]
    return hmac.compare_digest(sig, hmac.new(_cache_key(), body, hashlib.sha256).digest())


def _discard(path, what):
    """A cache entry with a wrong signature is never unpickled: it is deleted and rebuilt."""
    import sys as _sys
    print(f"apsi: discarded {what} '{path}' (signature does not match this machine's key: copied from "
          f"another computer or modified); rebuilding it", file=_sys.stderr)
    try:
        if os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError:
        pass


def _signed_load(path):
    import hmac
    with open(path, "rb") as fh:
        blob = fh.read()
    sig, body = blob[:32], blob[32:]
    if not hmac.compare_digest(sig, hmac.new(_cache_key(), body, hashlib.sha256).digest()):
        raise CacheInvalid(f"cache file '{path}' failed its integrity check (modified or from another "
                         f"machine); delete the cache folder to rebuild it")
    return pickle.loads(body)


def _write_shard(root, k, items):
    _signed_dump(os.path.join(root, f"shard_{k:06d}.pkl"), items)


def _hash(obj):
    h = hashlib.sha256()

    def feed(o):
        if isinstance(o, np.ndarray):
            h.update(str(o.dtype).encode() + str(o.shape).encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, Quantity):
            h.update(repr(o.dim).encode())
            feed(np.asarray(o.v))
        elif isinstance(o, (list, tuple)):
            h.update(b"(")
            for x in o:
                feed(x)
            h.update(b")")
        elif isinstance(o, dict):
            feed(sorted(o.items()))
        else:
            h.update(repr(o).encode())
    feed(obj)
    return h.hexdigest()[:32]
