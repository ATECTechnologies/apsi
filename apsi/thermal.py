"""GPU temperature control: long work on the card (training, answering, serving) stays near a target temperature
instead of running at the card's own limit (often 83-90 C on desktop cards).

The temperature comes from the NVIDIA driver's management library (NVML: nvml.dll / libnvidia-ml.so.1, installed
with the display driver). While the card is above the band, the work is paced: after each piece of work (a training
step, a generated token) Apsi waits a fraction of the time that piece took, so the card does less per second and
cools; below the band the waiting goes away. The fraction moves a little every half second, so the temperature
settles in the band rather than swinging. A cool card is never slowed.

    APSI_GPU_TEMP=77        the target in degrees Celsius (default 77: the band is 75-78)
    APSI_GPU_TEMP=off       no control (the card's own limits only)
    APSI_CPU_TEMP=77        the same for the processor during training and answering on it (Linux sensors; on
                            Windows the ACPI thermal zone, which needs an administrator)

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
        self.integral = 0.0
        self.temp = None
        self.max_temp = None
        self.paced = 0.0
        self.last_read = 0.0
        self.last_tick = None
        self.lock = threading.Lock()

    def _adjust(self, now):
        """A PI controller on the temperature: the pause fraction follows the error smoothly (integral part) and
        reacts at once to a change (proportional part), so the card settles at the target instead of swinging
        between too hot and paused."""
        t = self.read_temp()
        if t is None:
            return
        dt = min(2.0, max(0.05, now - self.last_read)) if self.last_read else self.PERIOD
        self.temp = t
        self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        err = t - self.target
        if self.fraction > 0.0 or err > -1.0:           # cool and not pacing: nothing to wind up
            self.integral = min(0.85, max(0.0, self.integral + 0.012 * err * dt))
        self.fraction = min(0.85, max(0.0, self.integral + 0.035 * err))
        self.last_read = now

    def tick(self, sync=None):
        """Call after each piece of work. Waits when the card should cool down; `sync` (optional) first waits for
        the work queued on the card, so the pause is real time off for the card and its length matches the work."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_read >= self.PERIOD:
                self._adjust(now)
            frac = self.fraction
        if frac > 0.0:
            if sync is not None:
                sync()
                now = time.monotonic()
            work = 0.0 if self.last_tick is None else min(now - self.last_tick, 2.0)
            if work > 0.0:
                wait = work * frac / (1.0 - frac)
                time.sleep(wait)
                self.paced += wait
        self.last_tick = time.monotonic()

    def status(self):
        out = {"temperature_c": self.temp, "target_c": self.target, "band_c": [self.low, self.high],
               "paced_fraction": round(self.fraction, 3), "max_c": self.max_temp, "paced_s": round(self.paced, 1)}
        if getattr(self, "read_power", None):
            out["power_w"] = self.read_power()
        if getattr(self, "power_note", None):
            out["power_limit"] = self.power_note
        return out


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

        def watts():
            mw = ctypes.c_uint()
            return round(mw.value / 1000.0, 1) if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0 else None
        g.read_power = watts
        g.power_note = _power_limit(lib, h)
        return g


# ---------------------------------------------------------------------------------------------------- processor
def _cpu_reader():
    """A function giving the processor's temperature in C, or None where it cannot be read.
    Linux: the hwmon sensors (k10temp / zenpower for AMD, coretemp for Intel) or the package thermal zone.
    Windows: the ACPI thermal zone through WMI, which Windows only shows to administrators (and on some boards it
    is the motherboard's sensor); it is read in the background every 3 s."""
    if sys.platform.startswith("linux"):
        import glob
        best = None
        for d in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                name = open(os.path.join(d, "name")).read().strip()
            except OSError:
                continue
            if name in ("k10temp", "zenpower", "coretemp", "cpu_thermal", "soc_thermal"):
                f = os.path.join(d, "temp1_input")
                if os.path.exists(f):
                    best = f
                    break
        if best is None:
            for z in glob.glob("/sys/class/thermal/thermal_zone*"):
                try:
                    if open(os.path.join(z, "type")).read().strip() in ("x86_pkg_temp", "cpu-thermal", "acpitz"):
                        best = os.path.join(z, "temp")
                        break
                except OSError:
                    continue
        if best is None:
            return None

        def read():
            try:
                return int(open(best).read().strip()) / 1000.0
            except (OSError, ValueError):
                return None
        return read if read() is not None else None
    if sys.platform.startswith("win"):
        import subprocess
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "(Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop | "
               "Measure-Object CurrentTemperature -Maximum).Maximum"]

        def once():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=8,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
                v = float(out) / 10.0 - 273.15
                return v if 0.0 < v < 125.0 else None
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
        first = once()
        if first is None:
            return None
        box = {"t": first}

        def poll():
            while True:
                time.sleep(3.0)
                v = once()
                if v is not None:
                    box["t"] = v
        threading.Thread(target=poll, daemon=True, name="apsi-cpu-temp").start()
        return lambda: box["t"]
    return None


def cpu_governor():
    """The processor's governor (APSI_CPU_TEMP, default 77 C), or NULL when its temperature cannot be read."""
    v = os.environ.get("APSI_CPU_TEMP", "77").strip().lower()
    if v in ("", "0", "off", "no", "false", "none"):
        return NULL
    try:
        target = min(100.0, max(50.0, float(v)))
    except ValueError:
        target = 77.0
    with _lock:
        g = _govs.get("cpu")
        if g is None:
            read = _cpu_reader()
            g = _govs["cpu"] = Governor(read, target) if read is not None else NULL
        elif g is not NULL:
            g.target, g.low, g.high = target, target - 2.0, target + 1.0
        return g


def _power_limit(lib, h):
    """APSI_GPU_POWER=80 (percent of the card's default limit) or =130W: a lower power limit for as long as this
    program runs (restored at exit). The card then runs at lower clocks and voltage, where it does much more work per
    watt: about -20 % power costs about -5 % speed, and the card runs several degrees cooler without any pauses.
    Windows and Linux allow it only to an administrator (root); otherwise nothing changes and the reason is noted."""
    want = os.environ.get("APSI_GPU_POWER", "").strip().lower()
    if not want:
        return None
    cur, dflt, lo, hi = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    if lib.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(cur)) or \
            lib.nvmlDeviceGetPowerManagementDefaultLimit(h, ctypes.byref(dflt)) or \
            lib.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(lo), ctypes.byref(hi)):
        return "this card does not report its power limit"
    try:
        mw = int(float(want[:-1]) * 1000) if want.endswith("w") else int(dflt.value * float(want.rstrip("%")) / 100)
    except ValueError:
        return f"APSI_GPU_POWER={want}: give a percentage (80) or watts (130W)"
    mw = max(lo.value, min(hi.value, mw))
    r = lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(mw))
    if r == 4:
        return "a lower power limit needs an administrator (Windows) or root (Linux); temperature pacing only"
    if r != 0:
        return f"the driver refused the power limit (NVML error {r})"
    import atexit
    old = cur.value
    atexit.register(lambda: lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(old)))
    return f"power limit {mw / 1000:.0f} W (default {dflt.value / 1000:.0f} W) while Apsi runs"


def status():
    """{"gpu0": {...}, "cpu": {...}}: temperature, target, highest seen, time spent pacing."""
    return {(f"gpu{k}" if k != "cpu" else k): g.status() for k, g in _govs.items() if g is not NULL}


def temperature(dev=None):
    """The GPU's temperature in degrees Celsius now, or None."""
    g = governor(dev) if dev is not None else next((x for x in _govs.values() if x is not NULL), None)
    if g is None or g is NULL:
        return None
    return g.read_temp()
