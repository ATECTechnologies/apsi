"""`use parallel` — tasks, channels, locks, parallel loops, thread control and benchmarking.

numpy/BLAS kernels release Python's GIL, so threads give real multi-core speed-ups for
tensor work; BLAS itself already runs matrix products on all cores.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _fwait, FIRST_COMPLETED

import numpy as np

from .errors import RuntimeErr
from .units import Quantity
from .values import Record

_POOL = {"pool": None, "size": None}


def _pool():
    n = _POOL["size"] or (os.cpu_count() or 1)
    if _POOL["pool"] is None:
        _POOL["pool"] = ThreadPoolExecutor(max_workers=max(2, n), thread_name_prefix="apsi")
    return _POOL["pool"]


def _seconds(t):
    if isinstance(t, Quantity):
        return float(t.v)
    return float(t)


class Task:
    def __init__(self, fut, name):
        self._fut = fut
        self.name = name

    def wait(self, timeout=None):
        """Block until finished and return the result (re-raises the task's error)."""
        return self._fut.result(timeout=None if timeout is None else _seconds(timeout))

    def done(self):
        return self._fut.done()

    def cancel(self):
        return self._fut.cancel()

    def __repr__(self):
        return f"<task {self.name} {'done' if self.done() else 'running'}>"


def spawn(f, *args, **kwargs):
    """Run f(*args) concurrently; returns a Task (use task.wait() for the result)."""
    return Task(_pool().submit(f, *args, **kwargs), getattr(f, "name", getattr(f, "__name__", "task")))


def wait_all(tasks, timeout=None):
    return [t.wait(timeout) for t in tasks]


def wait_any(tasks, timeout=None):
    """Return (index, result) of the first task to finish."""
    futs = [t._fut for t in tasks]
    done, _ = _fwait(futs, timeout=None if timeout is None else _seconds(timeout), return_when=FIRST_COMPLETED)
    if not done:
        raise RuntimeErr("wait_any: timed out")
    f = next(iter(done))
    return (futs.index(f), f.result())


class Channel:
    """A thread-safe FIFO for producer/consumer pipelines. Iterating stops after close()."""
    _CLOSED = object()

    def __init__(self, capacity=0):
        import queue
        self._q = queue.Queue(maxsize=int(capacity))
        self._closed = False

    def send(self, x, timeout=None):
        if self._closed:
            raise RuntimeErr("send on a closed channel")
        self._q.put(x, timeout=None if timeout is None else _seconds(timeout))

    def recv(self, timeout=None):
        import queue
        try:
            x = self._q.get(timeout=None if timeout is None else _seconds(timeout))
        except queue.Empty:
            raise RuntimeErr("recv: timed out")
        if x is Channel._CLOSED:
            self._q.put(Channel._CLOSED)
            raise RuntimeErr("recv on a closed, empty channel")
        return x

    def close(self):
        if not self._closed:
            self._closed = True
            self._q.put(Channel._CLOSED)

    def __iter__(self):
        while True:
            x = self._q.get()
            if x is Channel._CLOSED:
                self._q.put(Channel._CLOSED)
                return
            yield x

    def __len__(self):
        return self._q.qsize()


class Lock:
    def __init__(self):
        self._l = threading.RLock()

    def acquire(self):
        self._l.acquire()

    def release(self):
        self._l.release()

    def run(self, f, *args):
        """Run f(*args) while holding the lock."""
        with self._l:
            return f(*args)


class Atomic:
    """A thread-safe number: counter.add(1), counter.get()."""

    def __init__(self, value=0):
        self._v = value
        self._l = threading.Lock()

    def add(self, d=1):
        with self._l:
            self._v = self._v + d
            return self._v

    def set(self, v):
        with self._l:
            self._v = v

    def get(self):
        return self._v


def parallel_for(items, f, workers=None):
    """Call f(item) for every item on a thread pool; returns when all are done."""
    items = list(items)
    n = int(workers) if workers else min(len(items), os.cpu_count() or 1)
    if n <= 1:
        for x in items:
            f(x)
        return None
    with ThreadPoolExecutor(max_workers=n) as ex:
        for fu in [ex.submit(f, x) for x in items]:
            fu.result()
    return None


def parallel_reduce(items, f, combine, init, workers=None):
    """Map f over items in parallel, then fold the results with combine(acc, x)."""
    items = list(items)
    n = int(workers) if workers else min(len(items), os.cpu_count() or 1) or 1
    with ThreadPoolExecutor(max_workers=max(1, n)) as ex:
        results = list(ex.map(f, items))
    acc = init
    for r in results:
        acc = combine(acc, r)
    return acc


def set_threads(n):
    """Limit compute threads (BLAS and Apsi's worker pool). Returns the active setting."""
    n = int(n)
    _POOL["size"] = n
    if _POOL["pool"] is not None:
        _POOL["pool"].shutdown(wait=False)
        _POOL["pool"] = None
    applied = "pool"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
        applied = "pool+blas"
    except Exception:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(n)
    return Record(threads=n, applied=applied)


def cpu_count():
    return os.cpu_count() or 1


def sleep(t):
    time.sleep(_seconds(t))


def bench(f, repeat=5, warmup=1):
    """Time f() -> record(best, mean, runs) in seconds (as Time quantities)."""
    from .units import parse_unit
    sdim = parse_unit("s")[1]
    for _ in range(int(warmup)):
        f()
    ts = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return Record(best=Quantity(min(ts), sdim), mean=Quantity(sum(ts) / len(ts), sdim), runs=len(ts))


def install(interp):
    mod = {"spawn": spawn, "wait_all": wait_all, "wait_any": wait_any,
           "channel": lambda capacity=0: Channel(capacity), "lock": lambda: Lock(),
           "atomic": lambda value=0: Atomic(value), "parallel_for": parallel_for,
           "parallel_reduce": parallel_reduce, "set_threads": set_threads, "cpu_count": cpu_count,
           "sleep": sleep, "bench": bench}
    interp.modules["parallel"] = mod
