"""`use rt` — real-time loops at 1000–2000 Hz and video pipelines from 640×480 to 1920×1080.

One scheduler instead of a flood of commands: you describe *what* runs and *how often*, and one precise loop
calls it on time. Waiting is a coarse sleep followed by a short spin on the high-resolution clock, so ticks land
within microseconds of their deadline while the CPU stays mostly idle.

    use rt
    fn control(tick, t):                       # called 1000 times per second
        ...
    let stats = rt.loop(1000, control, seconds=5)
    print(stats.hz, stats.jitter_p99_us, stats.missed)

    for tick in rt.ticks(2000, seconds=1):     # the same timing as a for loop (2 kHz)
        ...

    let s = rt.scheduler()                     # many tasks, one loop, each at its own rate
    s.every(1ms, read_sensor)
    s.every(10ms, update_model)
    s.every(100ms, draw)
    let report = s.run(seconds=10)

Video (one command runs the whole pipeline — capture thread, newest-frame-only hand-off, your function,
timing, adaptive resolution under load, error isolation):

    let cam = rt.camera(0, width=1920, height=1080, fps=60)     # or rt.synthetic(1920, 1080, fps=240)
    fn process(frame, info):
        let m = rt.motion(info.prev_gray, rt.gray(frame))
        ...
    let report = rt.video(cam, process, fps=120, seconds=10)
    print(report)

Frame helpers (OpenCV when installed — SIMD and multithreaded — otherwise NumPy; `rt.backend("gpu")` moves the
frame helpers to CUDA through PyTorch or CuPy when present):
    rt.gray  rt.resize  rt.downsample  rt.diff  rt.motion  rt.ema  rt.box_blur  rt.edges  rt.threshold
    rt.fps_meter
"""
import gc
import os
import sys
import threading
import time

import numpy as np

from .units import Quantity

_SPIN_NS = 1_500_000            # last 1.5 ms of each wait: busy-wait on the clock for microsecond accuracy

class _LazyCV:
    """OpenCV, imported on first use (it takes ~20 ms to load); falsy when it is not installed."""
    _mod = None
    _tried = False

    def _load(self):
        if not _LazyCV._tried:
            _LazyCV._tried = True
            try:
                import cv2
                cv2.setUseOptimized(True)
                _LazyCV._mod = cv2
            except Exception:                              # noqa: BLE001
                _LazyCV._mod = None
        return _LazyCV._mod

    def __bool__(self):
        return self._load() is not None

    def __repr__(self):
        return "true" if self else "false"

    def __getattr__(self, k):
        m = self._load()
        if m is None:
            raise AttributeError(k)
        return getattr(m, k)


_cv = _LazyCV()                    # OpenCV is optional; everything works without it


def _seconds(v):
    """A period from a time quantity (1ms), a number of seconds, or none."""
    if v is None:
        return None
    if isinstance(v, Quantity):
        return float(v.v)
    return float(v)


def _record(**kw):
    from .values import Record
    return Record(**kw)


# ------------------------------------------------------------------------------------------- timing
class _Realtime:
    """Everything that makes a loop steadier while it runs, undone afterwards:
    1 ms timer resolution on Windows, the garbage collector paused (no surprise pauses mid-frame), and a higher
    thread priority where the OS allows it without admin rights."""

    def __enter__(self):
        self.undo = []
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)
                self.undo.append(lambda: ctypes.windll.winmm.timeEndPeriod(1))
                k = ctypes.windll.kernel32
                th = k.GetCurrentThread()
                old = k.GetThreadPriority(th)
                if k.SetThreadPriority(th, 2):              # THREAD_PRIORITY_HIGHEST
                    self.undo.append(lambda: k.SetThreadPriority(th, old))
            except Exception:                               # noqa: BLE001
                pass
        elif sys.platform.startswith("linux"):
            try:                                            # timers wake on time, not up to 50 µs late
                import ctypes
                libc = ctypes.CDLL(None, use_errno=True)
                libc.prctl(29, 1, 0, 0, 0)                  # PR_SET_TIMERSLACK = 1 ns
                self.undo.append(lambda: libc.prctl(29, 0, 0, 0, 0))
            except Exception:                               # noqa: BLE001
                pass
            # (SCHED_FIFO is not used: a spinning real-time thread is throttled by the kernel and stalls for ~50 ms)
        self.gc_was = gc.isenabled()
        if self.gc_was:
            gc.collect(0)
            gc.disable()
            self.undo.append(gc.enable)
        return self

    def __exit__(self, *exc):
        for f in reversed(self.undo):
            try:
                f()
            except Exception:                               # noqa: BLE001
                pass
        return False


def _wait_until(deadline_ns, spin_ns=_SPIN_NS):
    now = time.perf_counter_ns()
    remaining = deadline_ns - now
    if remaining > spin_ns:
        time.sleep((remaining - spin_ns) / 1e9)
    while time.perf_counter_ns() < deadline_ns:
        pass


class Stats:
    """What a real-time run achieved."""

    def __init__(self, hz_target, lateness_ns, missed, busy_ns, total_ns, errors=0):
        lat = np.asarray(lateness_ns, dtype=np.float64) / 1000.0
        self.target_hz = hz_target
        self.ticks = int(lat.size)
        self.hz = self.ticks / (total_ns / 1e9) if total_ns else 0.0
        self.missed = int(missed)
        self.errors = int(errors)
        self.jitter_p50_us = float(np.percentile(lat, 50)) if lat.size else 0.0
        self.jitter_p99_us = float(np.percentile(lat, 99)) if lat.size else 0.0
        self.max_late_us = float(lat.max()) if lat.size else 0.0
        self.busy = busy_ns / total_ns if total_ns else 0.0

    def __str__(self):
        s = (f"{self.ticks} ticks at {self.hz:.1f} Hz (target {self.target_hz:g} Hz), lateness p50 "
             f"{self.jitter_p50_us:.1f} µs / p99 {self.jitter_p99_us:.1f} µs / max {self.max_late_us:.1f} µs, "
             f"missed {self.missed}, work {self.busy * 100:.1f} % of the time")
        return s + (f", {self.errors} errors" if self.errors else "")

    __repr__ = __str__


def loop(hz, step, seconds=None, steps=None, catch_up=False, on_error="raise"):
    """Call step(tick, t) `hz` times per second for `seconds` (or `steps` ticks). A tick that cannot start on time
    is counted as missed and skipped (catch_up=true runs it late instead), so the rhythm never drifts.
    on_error="skip" keeps the loop alive when one tick raises (the error is counted and the first one printed)."""
    hz = float(hz)
    if hz <= 0 or hz > 100_000:
        raise ValueError("rt.loop: hz must be between 0 and 100000")
    period = int(round(1e9 / hz))
    n = int(steps) if steps is not None else int(round(_seconds(seconds) * hz)) if seconds is not None else None
    if n is None:
        raise ValueError("rt.loop needs seconds= or steps=")
    lateness = np.zeros(n, dtype=np.int64)               # preallocated: no list growth inside the loop
    li = missed = busy = errors = 0
    skip = on_error == "skip"
    with _Realtime():
        t0 = time.perf_counter_ns() + period
        k = 0
        while k < n:
            deadline = t0 + k * period
            _wait_until(deadline)
            start = time.perf_counter_ns()
            lateness[li] = start - deadline
            li += 1
            try:
                step(k, (start - t0) / 1e9)
            except Exception as e:                         # noqa: BLE001
                if not skip:
                    raise
                errors += 1
                if errors == 1:
                    print(f"apsi: rt.loop tick {k} failed ({type(e).__name__}: {e}); continuing", file=sys.stderr)
            end = time.perf_counter_ns()
            busy += end - start
            k += 1
            if not catch_up:
                behind = (end - t0) // period - k + 1
                if behind > 0:                           # the work took longer than a period: skip, keep rhythm
                    missed += int(behind)
                    k += int(behind)
        total = time.perf_counter_ns() - t0 + period
    return Stats(hz, lateness[:li], missed, busy, total, errors)


def ticks(hz, seconds=None, steps=None):
    """A for-loop that runs at `hz`: `for tick in rt.ticks(1000, seconds=2):`. Yields the tick number."""
    hz = float(hz)
    period = int(round(1e9 / hz))
    n = int(steps) if steps is not None else int(round(_seconds(seconds) * hz))
    with _Realtime():
        t0 = time.perf_counter_ns() + period
        for k in range(n):
            _wait_until(t0 + k * period)
            yield k


class Scheduler:
    """Many periodic tasks, one precise loop. Each task: every(period, fn) — fn(t) with t in seconds."""

    def __init__(self):
        self.tasks = []                   # [period_ns, next_ns, fn, name, runs, late_ns_max, errors]

    def every(self, period, fn, name=None):
        p = _seconds(period)
        if not p or p <= 0:
            raise ValueError("every(): the period must be positive (e.g. 1ms)")
        self.tasks.append([int(p * 1e9), 0, fn, name or getattr(fn, "name", None) or getattr(fn, "__name__", "task"),
                           0, 0, 0])
        return self

    def hz(self, rate, fn, name=None):
        return self.every(1.0 / float(rate), fn, name)

    def run(self, seconds=None, on_error="raise"):
        end = time.perf_counter_ns() + int(_seconds(seconds) * 1e9) if seconds is not None else None
        start = time.perf_counter_ns()
        for t in self.tasks:
            t[1] = start + t[0]
        with _Realtime():
            while self.tasks:
                nxt = min(t[1] for t in self.tasks)
                if end is not None and nxt > end:
                    break
                _wait_until(nxt)
                now = time.perf_counter_ns()
                for t in self.tasks:
                    if t[1] <= now:
                        t[5] = max(t[5], now - t[1])
                        try:
                            t[2]((now - start) / 1e9)
                        except Exception as e:             # noqa: BLE001
                            if on_error != "skip":
                                raise
                            t[6] += 1
                            if t[6] == 1:
                                print(f"apsi: task {t[3]} failed ({type(e).__name__}: {e}); continuing",
                                      file=sys.stderr)
                        t[4] += 1
                        t[1] += t[0]
                        while t[1] <= now:                  # fell behind: skip, keep the rhythm
                            t[1] += t[0]
        total = (time.perf_counter_ns() - start) / 1e9
        return [_record(name=t[3], runs=t[4], rate_hz=t[4] / total if total else 0.0, max_late_us=t[5] / 1000,
                        errors=t[6]) for t in self.tasks]


def scheduler():
    return Scheduler()


# ------------------------------------------------------------------------------------------- backends
_BACKEND = {"name": "cpu"}


def _gpu_lib():
    try:
        import torch
        if torch.cuda.is_available():
            return "torch", torch
    except Exception:                                      # noqa: BLE001
        pass
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return "cupy", cupy
    except Exception:                                      # noqa: BLE001
        pass
    return None, None


def backend(name=None):
    """rt.backend() → the current frame backend; rt.backend("cpu" | "gpu" | "auto").
    "auto" picks the GPU only for frames large enough to beat the copy to the card (≥ 1600×900); "gpu" without a
    usable card stays on the CPU and says so once."""
    if name is None:
        return _BACKEND["name"]
    name = str(name).lower()
    if name not in ("cpu", "gpu", "auto"):
        raise ValueError('rt.backend: use "cpu", "gpu" or "auto"')
    if name in ("gpu", "auto"):
        kind, lib = _gpu_lib()
        if kind is None:
            if name == "gpu":
                print("apsi: note: no CUDA device (PyTorch or CuPy) found; rt stays on the CPU", file=sys.stderr)
            _BACKEND.update(name="cpu", kind=None, lib=None)
            return "cpu"
        _BACKEND.update(name=name, kind=kind, lib=lib)
        return name
    _BACKEND.update(name="cpu", kind=None, lib=None)
    return "cpu"


def _use_gpu(a):
    n = _BACKEND["name"]
    return n == "gpu" or (n == "auto" and a.shape[0] * a.shape[1] >= 1600 * 900)


def _to_gpu(a):
    kind, lib = _BACKEND["kind"], _BACKEND["lib"]
    if kind == "torch":
        return lib.from_numpy(np.ascontiguousarray(a)).cuda(non_blocking=True)
    return lib.asarray(a)


def _to_cpu(x):
    kind = _BACKEND.get("kind")
    if kind == "torch":
        return x.cpu().numpy()
    if kind == "cupy":
        return x.get()
    return x


def _a(x):
    if isinstance(x, Quantity):
        x = x.v
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _u8(a):
    return a.dtype == np.uint8


# ------------------------------------------------------------------------------------------- frames
def gray(rgb):
    """Luminance of an RGB(A) image (BT.601). uint8 in → uint8 out (fast path); other types → float32."""
    a = _a(rgb)
    if a.ndim == 2:
        return a if _u8(a) else a.astype(np.float32)
    if _u8(a):
        if _cv:
            return _cv.cvtColor(a, _cv.COLOR_RGBA2GRAY if a.shape[2] == 4 else _cv.COLOR_RGB2GRAY)
        # integer weights 77/150/29 (= .299/.587/.114 × 256) in uint16: exact to ±1 and 5× faster than floats
        r = a[..., 0].astype(np.uint16)
        r *= 77
        g = a[..., 1].astype(np.uint16)
        g *= 150
        r += g
        g = a[..., 2].astype(np.uint16)
        g *= 29
        r += g
        r += 128
        r >>= 8
        return r.astype(np.uint8)
    if _BACKEND["name"] != "cpu" and _use_gpu(a):
        x = _to_gpu(a.astype(np.float32))
        return _to_cpu(x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)
    a = a.astype(np.float32, copy=False)
    return a[..., 0] * np.float32(0.299) + a[..., 1] * np.float32(0.587) + a[..., 2] * np.float32(0.114)


def resize(img, width, height=None):
    """Resize to width×height (area averaging when shrinking — no aliasing; smooth when enlarging).
    resize(img, 0.5) scales by a factor."""
    a = _a(img)
    if height is None:
        f = float(width)
        width, height = max(1, int(round(a.shape[1] * f))), max(1, int(round(a.shape[0] * f)))
    width, height = int(width), int(height)
    if (a.shape[1], a.shape[0]) == (width, height):
        return a
    if _cv and a.dtype in (np.uint8, np.float32, np.uint16, np.float64):
        shrink = width < a.shape[1]
        return _cv.resize(a, (width, height), interpolation=_cv.INTER_AREA if shrink else _cv.INTER_LINEAR)
    ys = (np.arange(height) * (a.shape[0] / height)).astype(np.intp)
    xs = (np.arange(width) * (a.shape[1] / width)).astype(np.intp)
    return a[ys][:, xs]


def downsample(img, k=2):
    """Average k×k blocks (fast, no aliasing from skipping pixels)."""
    a = _a(img)
    k = int(k)
    if k <= 1:
        return a
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    if _cv and a.dtype in (np.uint8, np.float32, np.uint16, np.float64):
        return _cv.resize(a[:h, :w], (w // k, h // k), interpolation=_cv.INTER_AREA)
    a = a[:h, :w].astype(np.float32)
    return a.reshape(h // k, k, w // k, k, *a.shape[2:]).mean(axis=(1, 3))


def _absdiff(a, b):
    if a.shape != b.shape:
        raise ValueError(f"rt.diff: frame sizes differ ({a.shape} vs {b.shape})")
    if _u8(a) and _u8(b):
        if _cv:
            return _cv.absdiff(a, b)
        return np.maximum(a, b) - np.minimum(a, b)             # stays uint8: no overflow, no float temporaries
    return np.abs(a.astype(np.float32) - b.astype(np.float32))


def diff(a, b, threshold=0.0):
    """|a - b| per pixel (largest channel for colour); with a threshold, a boolean mask of pixels that changed
    more than it."""
    d = _absdiff(_a(a), _a(b))
    if d.ndim == 3:
        d = d.max(axis=2)
    return d > float(threshold) if threshold else d


def motion(prev, cur, threshold=25.0, step=None):
    """Did something move? record(moved, fraction, cx, cy, box): changed share of pixels, the centre of the change
    and its bounding box [x0, y0, x1, y1] in full-frame pixels. Works on gray or colour frames of any size; above
    640×480 it measures on a 2×-subsampled grid (step=1 forces full resolution) — same answer, 4× less work."""
    p, c = _a(prev), _a(cur)
    if p.ndim == 3:
        p, c = gray(p), gray(c)
    if step is None:
        step = 2 if p.shape[0] * p.shape[1] > 640 * 480 else 1
    step = max(1, int(step))
    if step > 1:
        p, c = p[::step, ::step], c[::step, ::step]
    d = _absdiff(p, c)
    m = d > float(threshold)
    n = int(np.count_nonzero(m))
    if n == 0:
        return _record(moved=False, fraction=0.0, cx=-1.0, cy=-1.0, box=[-1, -1, -1, -1])
    cols = np.count_nonzero(m, axis=0)
    rows = np.count_nonzero(m, axis=1)
    xs = np.flatnonzero(cols)
    ys = np.flatnonzero(rows)
    cx = float(cols @ np.arange(cols.size)) / n
    cy = float(rows @ np.arange(rows.size)) / n
    frac = n / m.size
    return _record(moved=frac > 0.001, fraction=frac, cx=cx * step, cy=cy * step,
                   box=[int(xs[0] * step), int(ys[0] * step), int(xs[-1] * step + step - 1),
                        int(ys[-1] * step + step - 1)])


def ema(prev, new, alpha=0.2):
    """Exponential moving average (smooths noise while staying responsive)."""
    a = float(alpha)
    p, n = _a(prev), _a(new)
    if _cv and p.shape == n.shape and p.dtype == n.dtype and p.dtype in (np.uint8, np.float32):
        return _cv.addWeighted(p, 1.0 - a, n, a, 0.0)
    return p * (1.0 - a) + n * a


def box_blur(img, r=1):
    """Mean filter of radius r (cost independent of r)."""
    a = _a(img)
    r = int(r)
    if r <= 0:
        return a
    k = 2 * r + 1
    if _cv and a.dtype in (np.uint8, np.float32, np.uint16):
        return _cv.blur(a, (k, k), borderType=_cv.BORDER_REPLICATE)
    a = a.astype(np.float64)
    p = np.pad(a, [(r + 1, r)] + [(r + 1, r)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    s = p.cumsum(0).cumsum(1)
    out = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return (out / (k * k)).astype(np.float32)


def threshold(img, t, high=255):
    """Pixels above t become `high`, the rest 0 (uint8)."""
    a = _a(img)
    if a.ndim == 3:
        a = gray(a)
    if _cv and a.dtype in (np.uint8, np.float32):
        return _cv.threshold(a, float(t), float(high), _cv.THRESH_BINARY)[1].astype(np.uint8, copy=False)
    return np.where(a > t, np.uint8(high), np.uint8(0))


def edges(img, low=50, high=150):
    """Edge map (Canny with OpenCV; gradient magnitude above `high` otherwise), uint8 0/255."""
    a = _a(img)
    if a.ndim == 3:
        a = gray(a)
    if _cv:
        return _cv.Canny(a.astype(np.uint8, copy=False), float(low), float(high))
    f = a.astype(np.float32)
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[:, 1:-1] = f[:, 2:] - f[:, :-2]
    gy[1:-1] = f[2:] - f[:-2]
    return np.where(np.hypot(gx, gy) > high, np.uint8(255), np.uint8(0))


class FpsMeter:
    def __init__(self):
        self.t = []

    def tick(self):
        self.t.append(time.perf_counter())
        if len(self.t) > 240:
            del self.t[:120]
        return self.fps()

    def fps(self):
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / (self.t[-1] - self.t[0])


def fps_meter():
    return FpsMeter()


# ------------------------------------------------------------------------------------------- sources
class _Latest:
    """Single-slot hand-off between a capture thread and the processing loop: the producer always overwrites, the
    consumer always gets the newest frame. Old frames are dropped (counted), never queued — queues are what make
    video lag grow over time."""

    def __init__(self):
        self.cv = threading.Condition()
        self.frame = None
        self.t = 0
        self.seq = 0
        self.taken = 0
        self.dropped = 0
        self.closed = False
        self.error = None

    def put(self, frame, t_ns):
        with self.cv:
            if self.frame is not None and self.seq != self.taken:
                self.dropped += 1
            self.frame, self.t = frame, t_ns
            self.seq += 1
            self.cv.notify()

    def get(self, timeout):
        with self.cv:
            if self.seq == self.taken and not self.closed:
                self.cv.wait(timeout)
            if self.error is not None:
                raise self.error
            if self.seq == self.taken:
                return None, 0, self.seq
            self.taken = self.seq
            return self.frame, self.t, self.seq

    def close(self, error=None):
        with self.cv:
            self.closed = True
            self.error = error
            self.cv.notify_all()


class Source:
    """Base for frame sources. A source runs its own capture thread once started."""

    width = height = 0
    fps = 0.0
    name = "source"

    def start(self):
        self.slot = _Latest()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"apsi-rt-{self.name}", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if getattr(self, "_stop", None) is not None:
            self._stop.set()
            self.thread.join(2.0)
        self.close()

    def close(self):
        pass

    def _run(self):
        try:
            self._produce()
            self.slot.close()
        except Exception as e:                              # noqa: BLE001
            self.slot.close(RuntimeError(f"{self.name}: {type(e).__name__}: {e}"))

    def read(self):
        """One frame without the pipeline (blocking)."""
        raise NotImplementedError

    def __str__(self):
        return f"<{self.name} {self.width}x{self.height} @ {self.fps:g} fps>"

    __repr__ = __str__


class Synthetic(Source):
    """A test source: a textured background with a moving square, at an exact frame rate (or as fast as possible
    with fps=0). Use it to measure a pipeline without a camera."""

    def __init__(self, width=640, height=480, fps=120.0, channels=3, noise=False):
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.name = "synthetic"
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        base = ((xx * 255 // max(1, self.width - 1)) ^ (yy * 255 // max(1, self.height - 1))).astype(np.uint8)
        self.bg = np.repeat(base[..., None], channels, axis=2) if channels > 1 else base
        if noise:
            self.bg = (self.bg.astype(np.int16) + rng.integers(-6, 7, self.bg.shape)).clip(0, 255).astype(np.uint8)
        self.side = max(8, self.height // 6)
        self.bufs = [self.bg.copy() for _ in range(3)]       # a small ring: no allocation per frame
        self.k = 0

    def _frame(self, k):
        buf = self.bufs[k % 3]
        np.copyto(buf, self.bg)
        span = max(1, self.width - self.side)
        x = (k * 7) % (2 * span)
        x = x if x < span else 2 * span - x
        y = (self.height - self.side) // 2
        buf[y:y + self.side, x:x + self.side] = 255
        return buf

    def read(self):
        f = self._frame(self.k)
        self.k += 1
        return f

    def _produce(self):
        period = int(1e9 / self.fps) if self.fps > 0 else 0
        t0 = time.perf_counter_ns()
        k = 0
        while not self._stop.is_set():
            if period:                                  # plain sleep: a spinning producer would steal the GIL
                d = t0 + k * period - time.perf_counter_ns()
                if d > 0:
                    time.sleep(d / 1e9)
            self.slot.put(self._frame(k), time.perf_counter_ns())
            k += 1


SIZES = {                                  # named frame sizes (any width x height works too)
    "qvga": (320, 240), "vga": (640, 480), "svga": (800, 600), "wvga": (854, 480), "xga": (1024, 768),
    "hd": (1280, 720), "720p": (1280, 720), "wxga": (1366, 768), "hd+": (1600, 900), "900p": (1600, 900),
    "fhd": (1920, 1080), "1080p": (1920, 1080), "qhd": (2560, 1440), "1440p": (2560, 1440),
    "4k": (3840, 2160), "uhd": (3840, 2160), "2160p": (3840, 2160),
}


def size(width, height=None):
    """(width, height) from 1600, 900 / "1600x900" / "fhd" / "1080p" / (w, h)."""
    if isinstance(width, (tuple, list)):
        width, height = width
    if isinstance(width, str):
        key = width.strip().lower()
        if key in SIZES:
            return SIZES[key]
        w, sep, h = key.replace("*", "x").partition("x")
        if not sep:
            raise ValueError(f"unknown frame size '{width}': use WIDTHxHEIGHT or one of {', '.join(SIZES)}")
        return int(w), int(h)
    if height is None:
        raise ValueError("give both width and height, or a size such as \"1600x900\" or \"fhd\"")
    w, h = int(width), int(height)
    if w < 16 or h < 16 or w > 16384 or h > 16384:
        raise ValueError(f"frame size {w}x{h} is outside 16..16384")
    return w, h


def synthetic(width=640, height=None, fps=120.0, noise=False):
    if height is None and not isinstance(width, (str, tuple, list)):
        height = 480 if int(width) == 640 else int(width) * 9 // 16
    w, h = size(width, height)
    return Synthetic(w, h, fps, noise=bool(noise))


class Camera(Source):
    """A camera through OpenCV with the settings that matter for latency: a one-frame driver buffer, MJPG at high
    resolutions (USB cameras only reach 60–120 fps at 720p/1080p in MJPG), frames read into reused buffers."""

    def __init__(self, index=0, width=None, height=None, fps=None, fourcc=None):
        if not _cv:
            raise RuntimeError("rt.camera needs OpenCV: apsi pkg install opencv-python")
        self.name = f"camera {index}"
        api = _cv.CAP_DSHOW if sys.platform.startswith("win") else (_cv.CAP_V4L2 if sys.platform.startswith("linux")
                                                                    else _cv.CAP_ANY)
        self.cap = _cv.VideoCapture(index, api) if isinstance(index, int) else _cv.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap = _cv.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        want_mjpg = fourcc or ("MJPG" if (width or 0) >= 1280 or (fps or 0) > 60 else None)
        if want_mjpg:
            self.cap.set(_cv.CAP_PROP_FOURCC, _cv.VideoWriter_fourcc(*str(want_mjpg)[:4]))
        for prop, v in ((_cv.CAP_PROP_FRAME_WIDTH, width), (_cv.CAP_PROP_FRAME_HEIGHT, height),
                        (_cv.CAP_PROP_FPS, fps)):
            if v:
                self.cap.set(prop, float(v))
        self.cap.set(_cv.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(_cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(_cv.CAP_PROP_FPS) or 0.0)
        if width and height and (self.width, self.height) != (int(width), int(height)):
            print(f"apsi: note: the camera gave {self.width}x{self.height} instead of {width}x{height}",
                  file=sys.stderr)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("the camera returned no frame")
        return _cv.cvtColor(frame, _cv.COLOR_BGR2RGB)

    def _produce(self):
        bufs = [None, None, None]
        k = 0
        fails = 0
        while not self._stop.is_set():
            ok, bufs[k % 3] = self.cap.read(bufs[k % 3])
            t = time.perf_counter_ns()
            if not ok:
                fails += 1
                if fails > 50:
                    raise RuntimeError("the camera stopped sending frames")
                time.sleep(0.002)
                continue
            fails = 0
            rgb = _cv.cvtColor(bufs[k % 3], _cv.COLOR_BGR2RGB)
            self.slot.put(rgb, t)
            k += 1

    def close(self):
        self.cap.release()


class _ArraySource(Source):
    """Frames from a list/tensor of images (or a generator) — replays them at `fps`."""

    def __init__(self, frames, fps=0.0):
        self.frames = frames
        self.fps = float(fps)
        self.name = "frames"
        first = frames[0] if hasattr(frames, "__getitem__") else None
        if first is not None:
            self.height, self.width = _a(first).shape[:2]

    def read(self):
        return _a(self.frames[0])

    def _produce(self):
        period = int(1e9 / self.fps) if self.fps > 0 else 0
        t0 = time.perf_counter_ns()
        for k, f in enumerate(self.frames):
            if self._stop.is_set():
                break
            if period:
                d = t0 + k * period - time.perf_counter_ns()
                if d > 0:
                    time.sleep(d / 1e9)
            self.slot.put(_a(f), time.perf_counter_ns())


# ------------------------------------------------------------------------------------------- video pipeline
_SCALES = (1.0, 0.75, 0.5, 0.375, 0.25)


class VideoReport:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __str__(self):
        s = (f"{self.frames} frames in {self.seconds:.2f} s = {self.fps:.1f} fps "
             f"(source {self.source}, target {self.target_fps:g} fps)\n"
             f"  work per frame   p50 {self.work_p50_ms:.2f} ms  p99 {self.work_p99_ms:.2f} ms  max {self.work_max_ms:.2f} ms"
             f"  (budget {self.budget_ms:.2f} ms, headroom {self.headroom * 100:.0f} %)\n"
             f"  latency          p50 {self.latency_p50_ms:.2f} ms  p99 {self.latency_p99_ms:.2f} ms"
             f"  (camera timestamp → your function finished)\n"
             f"  frame interval   p50 {self.interval_p50_ms:.2f} ms  p99 {self.interval_p99_ms:.2f} ms"
             f"  jitter (p99 - p50) {self.jitter_ms:.2f} ms\n"
             f"  dropped stale {self.dropped}, late {self.late}, errors {self.errors}, "
             f"scale {self.scale_min:g}–{self.scale_max:g} ({self.scale_changes} changes)")
        return s

    __repr__ = __str__


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def video(source, process, fps=None, seconds=None, frames=None, adapt=False, min_scale=0.25, on_error="skip",
          gray_prev=True, warmup=5):
    """Run `process(frame, info)` on the newest frame from `source` until `seconds` or `frames` is reached.

    - The source captures on its own thread; the pipeline always takes the newest frame, so latency never builds up
      (frames that arrive while you work are dropped and counted, not queued).
    - fps= caps the processing rate (None = as fast as frames arrive).
    - adapt=true: when the work no longer fits the frame budget it lowers the working resolution in steps
      (1, 0.75, 0.5 …, never below min_scale) and raises it again when there is room — one slow frame never
      triggers it, a sustained overload does (p95 of the last 30 frames).
    - on_error="skip": an exception in your function is counted and the pipeline keeps going (first one printed);
      "raise" stops it.
    - info: record(index, t, latency_ms, scale, prev, prev_gray, fps) — prev/prev_gray are the previous frame and
      its gray image, ready for rt.motion.
    Returns a report with fps, work time, latency, jitter, dropped frames, errors and the scales used."""
    if isinstance(source, (list, tuple)) or (isinstance(source, np.ndarray) and source.ndim >= 3):
        source = _ArraySource(source)
    if not isinstance(source, Source):
        raise TypeError("rt.video: the source must be rt.camera(...), rt.synthetic(...) or a list of frames")
    cap_fps = float(fps) if fps else 0.0
    budget = 1.0 / cap_fps if cap_fps else (1.0 / source.fps if source.fps else 1 / 60)
    min_period = int(1e9 / cap_fps) if cap_fps else 0
    max_frames = int(frames) if frames is not None else None
    src_period = 1.0 / source.fps if source.fps else 0.0
    fresh = bool(cap_fps and source.fps and source.fps >= 1.5 * cap_fps)
    end_ns = None
    work, lat, inter = [], [], []
    errors = late = changes = 0
    scale_i = 0
    smin = smax = 1.0
    prev = prev_g = None
    from .values import Record
    source.start()
    try:
        with _Realtime():
            t_start = time.perf_counter_ns()
            if seconds is not None:
                end_ns = t_start + int(_seconds(seconds) * 1e9)
            last_done = None
            k = 0
            next_ok = t_start
            idle_collect = 0
            while True:
                if max_frames is not None and k >= max_frames + warmup:
                    break
                now = time.perf_counter_ns()
                if end_ns is not None and now >= end_ns:
                    break
                waited_from = now
                if min_period and now < next_ok:
                    _wait_until(next_ok, 300_000)
                    waited_from = next_ok - (src_period * 1e9 * 0.25 if fresh else 0)
                frame, t_cap, _seq = source.slot.get(0.25)
                if fresh and frame is not None and t_cap < waited_from:
                    # the source is much faster than the cap: skip the frame that waited in the slot and take the
                    # next one, so what you process is at most a fraction of a source period old
                    f2, t2, _s2 = source.slot.get(src_period * 1.5)
                    if f2 is not None:
                        frame, t_cap = f2, t2
                if frame is None:
                    if source.slot.closed:
                        break
                    continue
                t0 = time.perf_counter_ns()
                scale = _SCALES[scale_i]
                f = frame if scale == 1.0 else resize(frame, scale)
                g = gray(f) if gray_prev else None
                if prev is not None and prev.shape != f.shape:    # the working scale changed: start a new pair
                    prev = prev_g = None
                info = Record(index=k, t=(t0 - t_start) / 1e9, latency_ms=(t0 - t_cap) / 1e6, scale=scale,
                              prev=prev, prev_gray=prev_g if prev_g is not None else g, gray=g,
                              fps=(1e9 / inter[-1]) if inter else 0.0)
                try:
                    process(f, info)
                except Exception as e:                      # noqa: BLE001
                    if on_error == "raise":
                        raise
                    errors += 1
                    if errors == 1:
                        print(f"apsi: rt.video frame {k} failed ({type(e).__name__}: {e}); continuing",
                              file=sys.stderr)
                t1 = time.perf_counter_ns()
                prev, prev_g = (f.copy() if f is frame else f), (g.copy() if g is not None and g is f else g)
                if k >= warmup:
                    work.append(t1 - t0)
                    lat.append(t1 - t_cap)
                    if last_done is not None:
                        inter.append(t1 - last_done)
                        if (t1 - last_done) > 1.5e9 * budget:
                            late += 1
                last_done = t1
                if min_period:
                    next_ok = max(next_ok + min_period, t1 - min_period // 4)
                k += 1
                if adapt and len(work) >= 30 and k % 10 == 0:
                    recent = np.asarray(work[-30:]) / 1e9
                    p95 = float(np.percentile(recent, 95))
                    if p95 > 0.85 * budget and _SCALES[min(scale_i + 1, len(_SCALES) - 1)] >= min_scale \
                            and scale_i < len(_SCALES) - 1:
                        scale_i += 1
                        changes += 1
                    elif scale_i > 0 and len(work) >= 120 and \
                            float(np.percentile(np.asarray(work[-120:]) / 1e9, 95)) < 0.45 * budget:
                        scale_i -= 1
                        changes += 1
                    smin, smax = min(smin, _SCALES[scale_i]), max(smax, _SCALES[scale_i])
                # spare time this frame: a cheap young-generation collection keeps memory flat without pauses
                idle_collect += 1
                if idle_collect >= 120 and (time.perf_counter_ns() - t1) < budget * 2e8:
                    gc.collect(0)
                    idle_collect = 0
            t_end = time.perf_counter_ns()
    finally:
        source.stop()
    n = len(work)
    w = np.asarray(work, dtype=np.float64) / 1e6
    lt = np.asarray(lat, dtype=np.float64) / 1e6
    it = np.asarray(inter, dtype=np.float64) / 1e6
    secs = (t_end - t_start) / 1e9
    measured = (it.sum() / 1e3) if it.size else secs
    return VideoReport(
        frames=n, seconds=secs, fps=(len(it) / measured) if it.size and measured else 0.0, target_fps=cap_fps,
        source=f"{source.width}x{source.height}" + (f" @ {source.fps:g} fps" if source.fps else ""),
        budget_ms=budget * 1e3, work_p50_ms=_pct(w, 50), work_p99_ms=_pct(w, 99),
        work_max_ms=float(w.max()) if n else 0.0, headroom=max(0.0, 1 - _pct(w, 50) / (budget * 1e3)),
        latency_p50_ms=_pct(lt, 50), latency_p99_ms=_pct(lt, 99), interval_p50_ms=_pct(it, 50),
        interval_p99_ms=_pct(it, 99), jitter_ms=_pct(it, 99) - _pct(it, 50), dropped=source.slot.dropped,
        late=late, errors=errors, scale_min=smin, scale_max=smax, scale_changes=changes)


def module(interp=None):
    m = {"loop": loop, "ticks": ticks, "scheduler": scheduler, "gray": gray, "resize": resize,
         "downsample": downsample, "diff": diff, "motion": motion, "ema": ema, "box_blur": box_blur,
         "threshold": threshold, "edges": edges, "fps_meter": fps_meter, "synthetic": synthetic, "video": video,
         "backend": backend, "has_opencv": _cv, "size": size, "sizes": dict(SIZES)}
    if interp is not None:
        def camera(index=0, width=None, height=None, fps=None, fourcc=None):
            """A camera through OpenCV (pip install opencv-python); needs --allow-python. `width` may be a size
            name or "WxH" ("fhd", "1600x900")."""
            from .security import need
            need(interp, "python", "rt.camera()")
            if isinstance(width, (str, tuple, list)):
                width, height = size(width)
            return Camera(index, width, height, fps, fourcc)
        m["camera"] = camera
    return m


_ = os
