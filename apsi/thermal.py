"""GPU temperature control: long work on the card (training, answering, serving) stays near a target temperature
instead of running at the card's own limit (often 83-90 C on desktop cards).

The temperature comes from the NVIDIA driver's management library (NVML: nvml.dll / libnvidia-ml.so.1, installed
with the display driver). While the card is above the band, the work is paced: after each piece of work (a training
step, a generated token) Apsi waits a fraction of the time that piece took, so the card does less per second and
cools; below the band the waiting goes away. The fraction moves a little every half second, so the temperature
settles in the band rather than swinging. A cool card is never slowed.

    APSI_GPU_TEMP=79        the ceiling in degrees Celsius (default 79): the card is held just under it (78-79), as much
                            power as that temperature allows and never more heat
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
    v = os.environ.get("APSI_GPU_TEMP", "79").strip().lower()
    if v in ("", "0", "off", "no", "false", "none"):
        return None
    try:
        t = float(v)
    except ValueError:
        return 79.0
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
    """Holds one GPU just under a ceiling (APSI_GPU_TEMP, default 79 C): the smoothed temperature is steered to
    ceiling - 0.5, so the readings stay in [ceiling - 1, ceiling] and the card gets all the power that temperature
    allows (a lower, wider band would leave power unused)."""
    PERIOD = 0.5                       # seconds between temperature readings and adjustments

    def __init__(self, read_temp, target):
        self.read_temp = read_temp
        self.retarget(target)

    def retarget(self, ceiling):
        """A new ceiling (APSI_GPU_TEMP read again by the next run): steered to ceiling - 0.5, band [-1, 0]."""
        self.ceiling = float(ceiling)
        self.target = self.ceiling - 0.5
        self.low, self.high = self.ceiling - 1.0, self.ceiling
        self.fraction = 0.0            # of each piece of work, how much to wait after it (0 = full speed)
        self.integral = 0.0
        self.temp = None
        self.max_temp = None
        self.paced = 0.0
        self.last_read = 0.0
        self.last_tick = None
        self.lock = threading.Lock()

    KP, KI = 0.02, 0.004               # pause fraction per degree; per degree-second

    def _adjust(self, now):
        """A PI controller on the smoothed temperature. The die of a graphics card jumps several degrees within a
        fraction of a second whenever the load changes, and the sensor reads whole degrees, so the raw reading (and
        above all its rate of change) is mostly noise: reacting to it made the pauses swing between none and most of
        the time (the card between 62 and 77 C, the speed halved). Smoothed over about two seconds with a half-degree
        dead band, the pause fraction moves slowly to the one value that holds the target, so the temperature and
        the speed stay level. Above target + 1 the integral moves three times faster (no lasting overshoot)."""
        t = self.read_temp()
        if t is None:
            return
        dt = min(2.0, max(0.05, now - self.last_read)) if self.last_read else self.PERIOD
        self.temp = t
        self.max_temp = t if self.max_temp is None else max(self.max_temp, t)
        sm = getattr(self, "smooth", None)
        self.smooth = t if sm is None else sm + min(1.0, dt / 2.0) * (t - sm)
        # the slope of the smoothed temperature (degrees per second), smoothed again: how fast the card heats now
        sl = 0.0 if sm is None else (self.smooth - sm) / dt
        self.slope = sl if getattr(self, "slope", None) is None else self.slope + min(1.0, dt / 2.0) * (sl - self.slope)
        err = self.smooth - self.target
        e = 0.0 if abs(err) < 0.25 else err - (0.25 if err > 0 else -0.25)
        self.last_read = now
        fan = getattr(self, "fan", None)
        power = getattr(self, "power", None)
        ceiling = getattr(self, "ceiling", self.target + 0.5)
        if fan is not None and fan.ok:
            # fixed power, the fans hold the temperature; the power limit only as the last resort (fans at the top of their range)
            draw = None
            if getattr(self, "read_power", None):
                try:
                    draw = self.read_power()
                except Exception:                       # noqa: BLE001
                    draw = None
            run = float(getattr(power, "run", getattr(power, "orig", 0.0))) if power is not None else 0.0
            pinned = power is not None and power.limit < run - 500.0
            room = fan.control(self.smooth, self.slope, self.target, ceiling, dt, draw=draw, pinned=pinned)
            if fan.ok:
                self.fraction = 0.0
                if power is not None:
                    # the last resort, slow and small so it never swings. The fans come first both ways: the watts
                    # go down only with the fans at the top and the card still warm, and come back (slowly, up to
                    # the run's limit) while the fans stay at the top, before the fans slow down again. A reading
                    # at the ceiling + 0.5 itself: 4 % at once.
                    lim = power.limit
                    if t >= ceiling + 0.5:
                        lim *= 0.96
                    elif not room and self.smooth > self.target:
                        lim -= 600.0 * (self.smooth - self.target + 0.25) * dt
                    elif self.smooth < self.target - 0.5 and self.slope < 0.05:
                        lim += 300.0 * dt
                    lim = min(run, max(float(power.floor), lim))
                    if lim != power.limit:
                        power.limit = lim
                        applied = getattr(power, "applied", None)
                        if applied is None or abs(lim - applied) >= 1000.0 or lim == run != applied:
                            power.set(int(lim))
                            power.applied = lim
                return
        draw = None
        if power is not None and getattr(self, "read_power", None):
            try:
                draw = self.read_power()
            except Exception:                           # noqa: BLE001
                draw = None
        if power is not None and not power.step(e, dt, raw=t, slope=self.slope, draw=draw,
                                                ceiling=getattr(self, "ceiling", None)) and self.integral == 0.0:
            self.fraction = 0.0                         # the power limit holds it: no pauses, steady speed
            return
        if self.fraction > 0.0 or e > 0.0:              # a cool card that is not paced winds nothing up
            gain = self.KI * (8.0 if e > 2.0 else 3.0 if e > 1.0 else 1.0)    # far above: catch up fast
            self.integral = min(0.85, max(0.0, self.integral + gain * e * dt))
        self.fraction = min(0.85, max(0.0, self.integral + self.KP * max(e, -5.0)))

    def tick(self, sync=None):
        """Call after each piece of work. Waits when the card should cool down; `sync` (optional) first waits for
        the work queued on the card, so the pause is real time off for the card and its length matches the work."""
        now = time.monotonic()
        with self.lock:
            if now - self.last_read >= self.PERIOD:
                self._adjust(now)
            frac = self.fraction
        work = 0.0 if self.last_tick is None else min(now - self.last_tick, 2.0)
        self.last_tick = now
        if frac <= 0.0:
            self.debt = 0.0
            return
        # the pauses are gathered into pieces of at least 20 ms (one wait for the card each), so short steps are not
        # each followed by a wait for the card, which by itself would leave the card idle while the next is queued
        self.debt = getattr(self, "debt", 0.0) + work * frac / (1.0 - frac)
        if self.debt >= 0.02:
            t0 = time.monotonic()
            if sync is not None:
                sync()
            # the wait for the card is its queued work (several steps of a large model): work done, so it earns its
            # share of pause too (counted into the next one), or the pauses fall short and the card runs hot
            busy = min(time.monotonic() - t0, 2.0)
            time.sleep(self.debt)
            self.paced += self.debt
            self.debt = busy * frac / (1.0 - frac)
            self.last_tick = time.monotonic()

    def status(self):
        out = {"temperature_c": self.temp, "ceiling_c": getattr(self, "ceiling", None), "target_c": self.target,
               "band_c": [self.low, self.high],
               "paced_fraction": round(self.fraction, 3), "max_c": self.max_temp, "paced_s": round(self.paced, 1)}
        if getattr(self, "read_power", None):
            out["power_w"] = self.read_power()
        if getattr(self, "power", None) is not None:
            out["power_limit_w"] = round(self.power.limit / 1000.0, 1)
        if getattr(self, "power_note", None):
            out["power_limit"] = self.power_note
        fan = getattr(self, "fan", None)
        if fan is not None and fan.ok:
            out["control"] = (f"fixed power {round(getattr(self.power, 'run', self.power.limit) / 1000.0)} W, "
                              f"fans {round(getattr(fan, 'low', fan.LOW))}-{round(getattr(fan, 'top', fan.TOP))} % "
                              f"hold the temperature" + (f" (now {fan.percent} %)" if fan.percent else
                                                         " (driver's curve now)"))
        elif getattr(self, "power", None) is not None:
            out["control"] = ("the power limit holds the temperature (the fans could not be set: "
                              + (getattr(fan, "why", None) or "no fan control") + ")")
        return out


class _Null:
    def tick(self, sync=None):
        pass

    def retarget(self, ceiling):
        pass

    def status(self):
        return None


NULL = _Null()


def _max_mode():
    return os.environ.get("APSI_GPU_TEMP", "").strip().lower() in ("max", "full", "card")


def _max_governor(lib, h, key, read):
    """APSI_GPU_TEMP=max: the card as PyTorch and games run it: its full default power limit (170 W on an RTX 3060,
    put back at exit), its own fan curve and its own temperature limits (the driver slows the clocks near 83 C and
    protects the card above that). Apsi does nothing during the run but report."""
    g = Governor(read, 150.0)                          # (no ceiling of Apsi's own: never paces)
    g.retarget = lambda ceiling: None                  # (a later run reusing it keeps it so)

    def watts():
        mw = ctypes.c_uint()
        return round(mw.value / 1000.0, 1) if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0 else None
    g.read_power = watts
    pc = _PowerControl.try_make(lib, h, key)
    if pc is not None:
        pc.set(int(pc.top))
        g.max_power = pc                               # (kept: its restore runs at exit)
        g.power_note = (f"maximum: the card's default power limit {pc.top / 1000:.0f} W set {pc.via}, its own fan "
                        f"curve and temperature limits")
    else:
        g.power_note = ("maximum: the card's own limits (the power limit could not be raised here: administrator, "
                        "or apsi gpu setup)")
    return g


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
    if (target is None and not _max_mode()) or dev is None:
        return NULL
    key = getattr(dev, "index", 0)
    with _lock:
        g = _govs.get(key)
        if g is not None:
            g.retarget(target)                      # the same steering as a new governor (78.5, band 78-79)
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
        if _max_mode():
            return _govs.setdefault(key, _max_governor(lib, h, key, read))
        g = _govs[key] = Governor(read, target)

        def watts():
            mw = ctypes.c_uint()
            return round(mw.value / 1000.0, 1) if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0 else None
        g.read_power = watts
        g.power_note = _power_limit(lib, h)
        if g.power_note is None:
            g.power = _PowerControl.try_make(lib, h, key)  # steer the power limit, not pauses (when allowed)
            if g.power is not None:                     # faster fans first, with the same rights
                prof = _profile(lib, h, key)            # this card's own settings (apsi gpu calibrate), if measured
                g.fan = None if prof and prof.get("fans") is False else \
                    _FanControl.try_make(lib, h, key, getattr(g.power, "via", "") == "directly", prof)
                g.power.run = float(g.power.limit)
                if g.fan is not None:                   # the fans hold the temperature: the power can go up
                    want = _train_power(g.power, prof)
                    if want > g.power.limit + 500.0:
                        g.power.set(int(want))
                        g.power.limit = g.power.applied = g.power.run = float(want)
            if g.power is not None:
                g.power_note = (f"fixed power limit set {g.power.via}; the fans hold the temperature"
                                if getattr(g, "fan", None) is not None else
                                f"temperature held by the power limit, {g.power.via} (no pauses; the fans could "
                                f"not be set)")
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
            g.retarget(target)
        return g


class _PowerControl:
    """Holds the temperature by moving the card's power limit (between 55 % and 100 % of its default) instead of
    pausing the work: a card at a lower limit runs at lower clocks and voltage, so it loses much less speed per
    degree than pauses do, and the speed stays steady. Needs an administrator (Windows) or root (Linux); the
    original limit is put back at exit."""

    @staticmethod
    def try_make(lib, h, index=0):
        """Directly (this program is an administrator), or through the GPU helper (apsi gpu setup, once)."""
        if os.environ.get("APSI_GPU_POWER_CONTROL", "1").strip().lower() in ("0", "off", "no", "false"):
            return None
        cur, dflt, lo, hi = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        try:
            if lib.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(cur)) or \
                    lib.nvmlDeviceGetPowerManagementDefaultLimit(h, ctypes.byref(dflt)) or \
                    lib.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(lo), ctypes.byref(hi)):
                return None
            direct = lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(cur.value)) == 0
        except AttributeError:
            return None
        pc = _PowerControl()
        if direct:
            pc.set = lambda mw: lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(int(mw)))
            pc.via = "directly"
        else:
            from . import gpu_helper
            st = gpu_helper.call("status", gpu=index)
            if not st or not st.get("ok"):
                return None
            pc.set = lambda mw: gpu_helper.call("set", gpu=index, mw=int(mw))
            pc.via = "through the GPU helper"
        # the limit to put back: the one found before ANY Apsi program changed it. A run that was killed (its window
        # closed) never put its lowered limit back; the next run would then take that low limit for the card's own
        # and keep the card slow for good. So the first change records the original in a small file, every run
        # restores from it, and the file goes when the limit is back.
        state = _power_state_file(index)
        orig = _read_power_state(state)
        if orig is None:
            orig = cur.value
            _write_power_state(state, orig)
        elif orig != cur.value:
            pc_set = (lambda mw: lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(int(mw)))) if direct else \
                (lambda mw: __import__("apsi.gpu_helper", fromlist=["x"]).call("set", gpu=index, mw=int(mw)))
            pc_set(orig)                               # a killed run's limit: back to the card's own first
        pc.orig = orig
        pc.top = dflt.value
        pc.floor = max(lo.value, int(dflt.value * 0.55))
        pc.limit = float(orig)
        pc.state = state
        import atexit

        def _restore():
            pc.set(pc.orig)
            try:
                os.remove(state)
            except OSError:
                pass
        pc.restore = _restore
        atexit.register(_restore)
        return pc

    def step(self, err, dt, raw=None, slope=0.0, draw=None, ceiling=None):
        """The power limit that holds the card just under its ceiling, steadily and through sudden changes of load:

        * looking ahead: the error is taken 4 s ahead along the temperature's slope, so a card heating up quickly is
          given less power before it gets there, not after (no overshoot when a heavy phase starts);
        * fast down, slow up: too hot cuts the limit at once (about 4 W per degree and second, plus a step), too
          cool gives it back at most 2 W per second, so the limit does not swing;
        * at the ceiling itself (a raw reading at or above it): 6 % less at once, the guard against 80 C;
        * no wind-up: while the card draws less than its limit (a light phase), the limit is not raised beyond what
          it draws plus 8 W, so the next heavy phase does not start from an unused high limit."""
        e = err + 4.0 * slope
        if ceiling is not None and raw is not None and raw >= ceiling + 0.5:
            new = self.limit * 0.94                    # at the ceiling: at once
        elif e > 0:
            new = self.limit - (4000.0 * e * dt + 1500.0 * e)
        else:
            new = self.limit + min(2000.0 * dt, -3000.0 * e * dt)
            if draw is not None and draw > 0:
                new = min(new, max(self.limit, draw * 1000.0 + 8000.0))
        new = min(self.top, max(self.floor, new))
        applied = getattr(self, "applied", self.limit)  # what the card has now (changes below 1 W are gathered)
        # each change of the limit makes the driver re-plan the clocks (on Windows a short stall of the card), so
        # small changes are gathered: at least 1 W and 1 s apart, except at the ceiling itself
        urgent = ceiling is not None and raw is not None and raw >= ceiling + 0.5
        tnow = time.monotonic()
        if (abs(new - applied) >= 1000.0 or (new in (self.top, self.floor) and int(new) != int(applied))) and \
                (urgent or tnow - getattr(self, "set_at", -1e9) >= 1.0):
            self.set(int(new))
            self.applied, self.set_at = new, tnow
        self.limit = new
        return new <= self.floor + 1.0                 # at the floor: pauses may still be needed


def _fan_range(prof=None):
    """APSI_GPU_FAN_MIN / _MAX, else the card's measured range (apsi gpu calibrate), else 70-82 %."""
    def pct(name, d):
        try:
            return max(0.0, min(100.0, float(os.environ.get(name, "").strip().rstrip("%") or d)))
        except ValueError:
            return float(d)
    prof = prof or {}
    lo, hi = pct("APSI_GPU_FAN_MIN", prof.get("fan_min", 70)), pct("APSI_GPU_FAN_MAX", prof.get("fan_max", 82))
    return min(lo, hi), max(lo, hi)


class _FanControl:
    """The temperature is held by the fans, the power stays fixed. While a run trains, the card runs at a fixed power
    limit (APSI_GPU_TRAIN_POWER, default 145 W, never below the card's own limit or above its default) and its fans
    are steered between 70 and 82 % (APSI_GPU_FAN_MIN / APSI_GPU_FAN_MAX) to hold it at the ceiling - 0.5 (78.5 C
    for the default 79): looking ahead along the
    temperature's slope, faster at once when it rises, slower gently when it falls. The power limit is touched only
    if the fans are at the top of their range and the card still reaches the ceiling (a blocked cooler, a hot room): then a few watts
    at a time, and given back as soon as the fans have room again. Back to the driver's own fan curve when the card
    is well below the ceiling and at exit. Needs the rights of the power limit (administrator or the GPU helper);
    APSI_GPU_FAN=0 leaves the fans alone (the power limit holds the temperature instead)."""

    LOW, TOP = 70.0, 82.0               # the range it steers in (APSI_GPU_FAN_MIN / APSI_GPU_FAN_MAX)

    @staticmethod
    def try_make(lib, h, index, direct, prof=None):
        if os.environ.get("APSI_GPU_FAN", "1").strip().lower() in ("0", "off", "no", "false"):
            return None
        fc = _FanControl()
        fc.percent, fc.level, fc.ok = None, None, True
        fc.low, fc.top = _fan_range(prof)
        if direct:
            if not hasattr(lib, "nvmlDeviceSetFanSpeed_v2") and os.name != "nt":
                return None
            from . import gpu_helper
            fc._set = lambda pct: gpu_helper._fan(lib, h, pct, index).get("ok")
        else:
            from . import gpu_helper
            fc._set = lambda pct: bool((gpu_helper.call("fan_auto" if pct is None else "fan", gpu=index,
                                                        percent=pct) or {}).get("ok"))
        # a probe: handing the fans back to the driver's curve changes nothing, and tells whether this driver lets
        # Apsi set them at all (NVML, or NvAPI on Windows); if not, the power limit holds the temperature
        try:
            if not fc._set(None):
                return None
        except Exception:                               # noqa: BLE001
            return None
        import atexit
        atexit.register(fc.auto)
        return fc

    def control(self, smooth, slope, target, ceiling, dt, draw=None, pinned=False):
        """One step of the fans: returns True while they still have room (below the top of their range). Three terms, the way a
        spacecraft's thermal loop is built: the error now, the error 4 s ahead along the temperature's slope, and a
        feed-forward from the power the card draws: heat that has gone in but not yet shown on the sensor (the die
        takes about 20 s to settle), so the fans answer a heavy phase as it starts, not after the degrees come."""
        if not self.ok:
            return False
        e = (smooth - target) + 4.0 * slope            # 4 s ahead
        if draw is not None and draw > 0:
            ps = getattr(self, "p_smooth", None)
            self.p_smooth = draw if ps is None else ps + min(1.0, dt / 20.0) * (draw - ps)
            # watts above the settled level -> degrees still to come (half of the about 0.3 C per W of a desktop card: the loop adds the rest)
            if smooth > target - 3.0:
                e += 0.15 * max(0.0, draw - self.p_smooth)
        low, top = getattr(self, "low", self.LOW), getattr(self, "top", self.TOP)
        if pinned:                                     # the power is below the run's limit: fans stay at the top
            self.level = top
            self._apply(int(round(top)))
            return False
        if self.level is None:
            if smooth < ceiling - 10.0 and e < 0:
                return True                            # cool: the driver's own curve
            self.level = low                           # taking over: at the bottom of the range
        if e > 0:
            self.level += 3.0 * e * dt + 1.0 * e       # up fast
        else:
            self.level += 0.75 * e * dt                # down gently (no swinging)
        self.level = max(low, min(top, self.level))
        if smooth < ceiling - 12.0 and self.level <= low + 0.5:
            self.level = None                          # well below: back to the driver's curve
            self._apply(None)
            return True
        self._apply(int(round(self.level)))
        return self.level < top - 0.5

    def _apply(self, pct):
        edge = pct is not None and pct in (int(round(getattr(self, "low", self.LOW))), int(round(getattr(self, "top", self.TOP))))
        if pct == self.percent or (pct is not None and self.percent is not None and abs(pct - self.percent) < 2
                                   and not edge):
            return
        if self._set(pct):
            self.percent = pct
        else:
            self.ok = False                            # refused: the power limit holds the temperature instead
            self.why = "the driver refused the fan speed"

    def auto(self):
        if self.percent is not None:
            self._set(None)
            self.percent = None
        self.level = None


def _profile(lib, h, index):
    try:
        name = ctypes.create_string_buffer(96)
        if lib.nvmlDeviceGetName(h, name, ctypes.c_uint(96)) != 0:
            return None
        from . import gpu_calibrate
        return gpu_calibrate.load_profile(gpu_calibrate.card_key(name.value.decode(errors="replace"), index))
    except Exception:                                   # noqa: BLE001
        return None


def _train_power(pc, prof=None):
    """The fixed power limit of a training run while the fans hold the temperature, in mW: APSI_GPU_TRAIN_POWER
    (watts like 145W, or a share of the card's default like 85%), else what apsi gpu calibrate measured for this
    card, else 85 % of its default (an RTX 3060: 145 of 170 W); never below the limit the card has and never above
    its default. off: the card's limit as it is."""
    v = os.environ.get("APSI_GPU_TRAIN_POWER", "").strip().lower()
    if not v:                                       # the card's measured setting, else 85 % of its default
        v = f"{prof['power_w']}w" if prof and prof.get("power_w") else "85%"
    cur = float(pc.limit)
    if v in ("", "off", "0", "no", "none"):
        return cur
    try:
        mw = float(v[:-1]) / 100.0 * pc.top if v.endswith("%") else float(v.rstrip("w")) * 1000.0
    except ValueError:
        return cur
    return min(float(pc.top), max(cur, mw))


def _power_state_file(index):
    import tempfile
    return os.path.join(tempfile.gettempdir(), f"apsi-gpu{int(index)}-power-limit.txt")


def _read_power_state(path):
    try:
        with open(path) as fh:
            v = int(fh.read().strip())
        return v if v > 0 else None
    except (OSError, ValueError):
        return None


def _write_power_state(path, mw):
    try:
        with open(path, "w") as fh:
            fh.write(str(int(mw)))
    except OSError:
        pass


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


# ---------------------------------------------------------------------------------------------------- apsi gpu
def _nvml_handles():
    lib = _load_nvml()
    if lib is None:
        return None, []
    n = ctypes.c_uint()
    if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
        return lib, []
    hs = []
    for i in range(n.value):
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(i), ctypes.byref(h)) == 0:
            hs.append((i, h))
    return lib, hs


def gpu_report():
    """Each card now: name, temperature, power draw and limits, clocks, fan, load."""
    lib, hs = _nvml_handles()
    out = []
    for i, h in hs:
        name = ctypes.create_string_buffer(96)
        lib.nvmlDeviceGetName(h, name, 96)
        u = ctypes.c_uint()
        lo, hi = ctypes.c_uint(), ctypes.c_uint()

        def get(fn, *a):
            return u.value if fn(h, *a, ctypes.byref(u)) == 0 else None
        r = {"index": i, "name": name.value.decode(errors="replace"),
             "temperature_c": get(lib.nvmlDeviceGetTemperature, 0),
             "power_w": (lambda v: v and round(v / 1000.0, 1))(get(lib.nvmlDeviceGetPowerUsage)),
             "power_limit_w": (lambda v: v and round(v / 1000.0, 1))(get(lib.nvmlDeviceGetPowerManagementLimit)),
             "default_limit_w": (lambda v: v and round(v / 1000.0, 1))(get(lib.nvmlDeviceGetPowerManagementDefaultLimit)),
             "graphics_mhz": get(lib.nvmlDeviceGetClockInfo, 0), "memory_mhz": get(lib.nvmlDeviceGetClockInfo, 2),
             "fan_percent": get(lib.nvmlDeviceGetFanSpeed)}
        if lib.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(lo), ctypes.byref(hi)) == 0:
            r["limit_range_w"] = [round(lo.value / 1000.0), round(hi.value / 1000.0)]

        class Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
        ut = Util()
        if lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(ut)) == 0:
            r["load_percent"] = ut.gpu
        out.append(r)
    return out


def set_power(spec, index=None):
    """Sets the power limit of the card(s) until the next restart of the computer or the driver: '80%' (of the
    default), '130W', or 'default'. Needs an administrator / root."""
    lib, hs = _nvml_handles()
    if lib is None or not hs:
        return ["no NVIDIA card (or no NVML)"]
    msgs = []
    for i, h in hs:
        if index is not None and i != index:
            continue
        dflt, lo, hi = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        if lib.nvmlDeviceGetPowerManagementDefaultLimit(h, ctypes.byref(dflt)) or \
                lib.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(lo), ctypes.byref(hi)):
            msgs.append(f"GPU {i}: does not report its power limits")
            continue
        sp = str(spec).strip().lower()
        if sp == "default":
            mw = dflt.value
        elif sp.endswith("w"):
            mw = int(float(sp[:-1]) * 1000)
        else:
            mw = int(dflt.value * float(sp.rstrip("%")) / 100.0)
        mw = max(lo.value, min(hi.value, mw))
        r = lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(mw))
        if r == 4:                                      # not an administrator: the GPU helper, when set up
            from . import gpu_helper
            hr = gpu_helper.call("default" if sp == "default" else "set", gpu=i, mw=mw)
            if hr and hr.get("ok"):
                msgs.append(f"GPU {i}: power limit {hr['mw'] / 1000:.0f} W (through the GPU helper)")
                continue
        if r == 0:
            msgs.append(f"GPU {i}: power limit {mw / 1000:.0f} W (default {dflt.value / 1000:.0f} W) until the next "
                        "restart")
        elif r == 4:
            msgs.append(f"GPU {i}: needs an administrator once: 'apsi gpu setup' from a terminal opened as "
                        "administrator (or with sudo) installs the GPU helper, and from then on no program needs "
                        "administrator rights for this")
        else:
            msgs.append(f"GPU {i}: the driver refused (NVML error {r})")
    return msgs


def hold(target=79.0, index=0, seconds=None, quiet=False):
    """apsi gpu hold 79: keeps the card near the temperature by its power limit for ANY program running on it (a
    PyTorch job, a game, a renderer) until stopped (Ctrl+C) — the same control Apsi's own training uses, so other
    programs can be measured under the same conditions. Needs an administrator or the GPU helper."""
    os.environ["APSI_GPU_TEMP"] = str(target)
    from .native import cuda
    drv = cuda.driver()
    devs = [d for d in (drv.devices if drv else []) if d.index == index]
    if not devs:
        print("no NVIDIA card")
        return 1
    g = governor(devs[0])
    if g is NULL or getattr(g, "power", None) is None:
        print("the power limit cannot be set from here: run as administrator, or apsi gpu setup once")
        return 1
    print(f"holding GPU {index} at {target:g} C at most by its power limit (Ctrl+C to stop)", flush=True)
    t0 = last = time.monotonic()
    try:
        stop = os.environ.get("APSI_HOLD_STOP")           # a file whose appearance ends the hold (from another program)
        while (seconds is None or time.monotonic() - t0 < seconds) and not (stop and os.path.exists(stop)):
            g._adjust(time.monotonic())
            g.fraction = 0.0                              # never pauses: only the power limit (the program is not ours)
            g.integral = 0.0
            if not quiet and time.monotonic() - last >= 5:
                last = time.monotonic()
                print(f"  {g.temp} C, power limit {round(g.power.limit / 1000)} W", flush=True)
            time.sleep(g.PERIOD)
    except KeyboardInterrupt:
        pass
    finally:
        (getattr(g.power, "restore", None) or (lambda: g.power.set(g.power.orig)))()
    return 0


def _calibrate_all(index=None):
    from . import gpu_calibrate
    lib = _load_nvml()
    n = ctypes.c_uint(0)
    if lib is None or lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0 or not n.value:
        print("no NVIDIA card to calibrate")
        return 1
    rc = 0
    for i in ([index] if index is not None else range(n.value)):
        try:
            for m in gpu_calibrate.calibrate(i):
                print(m)
        except Exception as e:                          # noqa: BLE001
            print(f"GPU {i}: calibration stopped ({e}); the defaults stay (85 % power, fans 70-82 %)")
            rc = 1
    return rc


def add_parser(sub):
    p = sub.add_parser("gpu", help="the NVIDIA cards now: temperature, power, clocks; set a power limit")
    p.add_argument("what", nargs="?", default="status",
                   choices=["status", "power", "watch", "setup", "remove", "hold", "bench", "calibrate"])
    p.add_argument("value", nargs="?", default=None,
                   help="calibrate: measures each card's power and fan settings for training (setup does it too); "
                        "for power: 80%% (of the default), 130W or default; for hold: the ceiling (79); for bench: "
                        "127m (default), 11m, modern, check (every kernel against the reference) or cards")
    p.add_argument("--precision", default="auto", help="bench: auto (bf16), fp32, fp8, fp4, ...")
    p.add_argument("--steps", type=int, default=40, help="bench: training steps to time")
    p.add_argument("--gpu", type=int, default=None)
    p.set_defaults(fn=_cmd)


def _cmd(args):
    if args.what in ("setup", "remove"):
        from . import gpu_helper
        for m in (gpu_helper.setup(args.value) if args.what == "setup" else gpu_helper.remove()):
            print(m)
        if args.what == "setup" and os.environ.get("APSI_GPU_CALIBRATE", "1").strip().lower() not in ("0", "off", "no"):
            _calibrate_all(args.gpu)                    # this card's own power and fan settings, measured once
        return 0
    if args.what == "calibrate":
        return _calibrate_all(args.gpu)
    if args.what == "hold":
        return hold(float(args.value or 79), args.gpu or 0)
    if args.what == "bench":
        from . import gpu_bench
        return gpu_bench.main(args.value or "127m", args.precision, args.steps)
    if args.what == "power":
        if not args.value:
            print("apsi gpu power 80%   (or 130W, or default)")
            return 2
        for m in set_power(args.value, args.gpu):
            print(m)
        return 0
    rounds = 10 ** 9 if args.what == "watch" else 1
    for k in range(rounds):
        rep = gpu_report()
        if not rep:
            print("no NVIDIA card found (or the driver's NVML library is missing)")
            return 1
        for r in rep:
            print(f"GPU {r['index']} {r['name']}: {r['temperature_c']} C, {r['power_w']} W of {r['power_limit_w']} W "
                  f"(default {r['default_limit_w']} W, range {r.get('limit_range_w')}), {r['graphics_mhz']} MHz, "
                  f"memory {r['memory_mhz']} MHz, fan {r['fan_percent']} %, load {r.get('load_percent')} %")
        if k + 1 < rounds:
            time.sleep(1.0)
    return 0
