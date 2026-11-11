"""GPU temperature control: long work on the card (training, answering, serving) stays near a target temperature
instead of running at the card's own limit (often 83-90 C on desktop cards).

The temperature comes from the NVIDIA driver's management library (NVML: nvml.dll / libnvidia-ml.so.1, installed
with the display driver). While the card is above the band, the work is paced: after each piece of work (a training
step, a generated token) Apsi waits a fraction of the time that piece took, so the card does less per second and
cools; below the band the waiting goes away. The fraction moves a little every half second, so the temperature
settles in the band rather than swinging. A cool card is never slowed.

    APSI_GPU_TEMP=77        the target in degrees Celsius (default 77: the band is 75-78)
    APSI_GPU_TEMP=off       no control (the card's own limits only)

hub.gpu_status() (thermal.status()) -> {"temperature_c", "target_c", "paced_fraction", "max_c", "paced_s"} for each GPU in use.
"""
import ctypes
import os
import sys
import threading
import time

_lock = threading.Lock()
_nvml = None
_nvml_failed = None
_govs = {}


def _target():
    v = os.environ.get("APSI_GPU_TEMP", "77").strip().lower()
    if v in ("", "0", "off", "no", "false", "none"):
        return None
    try:
        t = float(v)
    except ValueError:
        return 77.0
    return min(95.0, max(50.0, t))


def _load_nvml():
    global _nvml, _nvml_failed
    if _nvml is not None or _nvml_failed is not None:
        return _nvml
    names = []
    if sys.platform.startswith("win"):
        sysroot = os.environ.get("SystemRoot", r"C:\Windows")
        names = [os.path.join(sysroot, "System32", "nvml.dll"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI",
                              "nvml.dll"), "nvml.dll"]
    else:
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            if lib.nvmlInit_v2() == 0:
                _nvml = lib
                return lib
        except (OSError, AttributeError):
            continue
    _nvml_failed = "the NVIDIA management library (NVML) was not found"
    return None


class Governor:
    """Paces the work of one GPU to keep it in [target - 2, target + 1] degrees Celsius."""
    PERIOD = 0.5                       # seconds between temperature readings and adjustments

    def __init__(self, read_temp, target):
        self.read_temp, self.target = read_temp, float(target)
        self.low, self.high = self.target - 2.0, self.target + 1.0
        self.fraction = 0.0            # of each piece of work, how much to wait after it (0 = full speed)
        self.temp = None
        self.max_temp = None
        self.paced = 0.0
        self.last_read = 0.0
        self.last_tick = None
        self.lock = threading.Lock()

    def _adjust(self, now):
        t = self.read_temp()
        if t is None:
            return
        self.temp = t
        self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        if t > self.high:              # too warm: pace more, harder the further above the band
            self.fraction = min(0.85, self.fraction + 0.04 * (t - self.high + 1.0))
        elif t < self.low:             # cool: give speed back, faster when well below
            self.fraction = max(0.0, self.fraction - 0.03 * (self.low - t + 1.0))
        self.last_read = now

    def tick(self, sync=None):
        """Call after each piece of work. Waits when the card should cool down; `sync` (optional) is called first so
        that the waiting is real time off for the card, not time while queued work still runs."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_read >= self.PERIOD:
                self._adjust(now)
            work = 0.0 if self.last_tick is None else min(now - self.last_tick, 2.0)
            frac = self.fraction
        if frac > 0.0 and work > 0.0:
            if sync is not None:
                sync()
            wait = work * frac / (1.0 - frac)
            time.sleep(wait)
            self.paced += wait
        self.last_tick = time.monotonic()

    def status(self):
        return {"temperature_c": self.temp, "target_c": self.target, "band_c": [self.low, self.high],
                "paced_fraction": round(self.fraction, 3), "max_c": self.max_temp, "paced_s": round(self.paced, 1)}


class _Null:
    def tick(self, sync=None):
        pass

    def status(self):
        return None


NULL = _Null()


def _pci_bus_id(dev):
    try:
        buf = ctypes.create_string_buffer(64)
        dev.drv.lib.cuDeviceGetPCIBusId(buf, 64, dev.handle)
        return buf.value
    except Exception:
        return None


def governor(dev):
    """The governor of a native/cuda.Device (or NULL when the temperature cannot be read or control is off)."""
    target = _target()
    if target is None or dev is None:
        return NULL
    key = getattr(dev, "index", 0)
    with _lock:
        g = _govs.get(key)
        if g is not None:
            g.target = target
            g.low, g.high = target - 2.0, target + 1.0
            return g
        lib = _load_nvml()
        if lib is None:
            _govs[key] = NULL
            return NULL
        h = ctypes.c_void_p()
        bus = _pci_bus_id(dev)
        ok = bus is not None and lib.nvmlDeviceGetHandleByPciBusId_v2(ctypes.c_char_p(bus), ctypes.byref(h)) == 0
        if not ok and lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(key), ctypes.byref(h)) != 0:
            _govs[key] = NULL
            return NULL

        def read():
            t = ctypes.c_uint()
            return float(t.value) if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)) == 0 else None
        if read() is None:
            _govs[key] = NULL
            return NULL
        g = _govs[key] = Governor(read, target)
        return g


def status():
    return {k: g.status() for k, g in _govs.items() if g is not NULL}


def temperature(dev=None):
    """The GPU's temperature in degrees Celsius now, or None."""
    g = governor(dev) if dev is not None else next((x for x in _govs.values() if x is not NULL), None)
    if g is None or g is NULL:
        return None
    return g.read_temp()
