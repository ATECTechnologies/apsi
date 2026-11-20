"""apsi gpu calibrate: the power limit and fan range each card trains at, measured on the card itself (once, at
install time by `apsi gpu setup`, or whenever the cooling changes).

The card is loaded with matrix products at a trial power limit and its fans are set to the top of the range, then
to the bottom; the temperature curve of each phase is fitted to a first-order model (T(t) = T_end + (T_start -
T_end) e^(-t/tau)), which gives the temperature the card would settle at without waiting for it. From the two
settled temperatures and the idle temperature come the degrees per watt at each fan speed, and from those the
highest power the fans can hold at the ceiling - 1.5 C, and the fan range that holds it with the least noise. The
result is saved in ~/.apsi/gpu-profile.json and read by every training run (thermal.governor); APSI_GPU_TRAIN_POWER,
APSI_GPU_FAN_MIN and APSI_GPU_FAN_MAX still override it. Safety: a reading at the ceiling + 1 cuts the trial power by
10 % at once, the load stops at the ceiling + 3, and the card's own fan curve and power limit come back at the end
whatever happens."""
import json
import math
import os
import threading
import time

PROFILE = "gpu-profile.json"


def _path():
    from .security import config_dir
    return os.path.join(config_dir(), PROFILE)


def load_profile(key):
    """The saved settings of a card (key: its name and index, see card_key), or None."""
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
        p = data.get(str(key))
        return p if isinstance(p, dict) and p.get("power_w") else None
    except (OSError, ValueError):
        return None


def save_profile(key, prof):
    path = _path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[str(key)] = prof
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def card_key(name, index):
    return f"{index}:{name}"


def fit(ts, temps):
    """(settled temperature, time constant) of T(t) = T_end + (T_start - T_end) e^(-t/tau), least squares over tau
    (5 to 120 s) with T_start and T_end solved exactly for each tau."""
    if len(ts) < 6:
        return (temps[-1] if temps else None), None
    t0 = ts[0]
    best = None
    for tau in [5.0 * 1.08 ** k for k in range(40)]:
        if tau > 120.0:
            break
        e = [math.exp(-(t - t0) / tau) for t in ts]
        # T = a (1 - e) + b e   (a: settled, b: start)
        s11 = sum((1 - x) ** 2 for x in e)
        s12 = sum((1 - x) * x for x in e)
        s22 = sum(x * x for x in e)
        y1 = sum((1 - x) * y for x, y in zip(e, temps))
        y2 = sum(x * y for x, y in zip(e, temps))
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-12:
            continue
        a = (y1 * s22 - y2 * s12) / det
        b = (s11 * y2 - s12 * y1) / det
        err = sum((a * (1 - x) + b * x - y) ** 2 for x, y in zip(e, temps))
        if best is None or err < best[0]:
            best = (err, a, tau)
    return (best[1], best[2]) if best else (temps[-1], None)


def plan(idle, settled_hi, watts_hi, settled_lo, watts_lo, top_w, own_w, ceiling=79.0, fan_lo=70.0, fan_hi=82.0):
    """The settings from the measurements: degrees per watt with the fans at the top (r_hi) and at the bottom (r_lo)
    of the range; the power the fans hold at the ceiling - 1.5 (never below the card's own limit, never above its
    default); the fan range: the bottom raised when even the bottom cannot hold that power, the top raised (up to
    100 %) when the top of the range would hold less than the card's own limit."""
    hold = ceiling - 1.5
    r_hi = max(1e-3, (settled_hi - idle) / max(1.0, watts_hi))
    r_lo = max(r_hi, (settled_lo - idle) / max(1.0, watts_lo))
    lo, hi = fan_lo, fan_hi
    watts = (hold - idle) / r_hi
    if watts < own_w and hi < 100.0:
        # the slope per fan percent, from the two phases, extends the range upwards
        per = (r_lo - r_hi) / max(1.0, hi - lo)
        while watts < own_w and hi < 100.0:
            hi += 1.0
            watts = (hold - idle) / max(1e-3, r_hi - per * (hi - fan_hi))
    watts = max(float(own_w), min(float(top_w), 0.97 * watts))
    # the bottom of the range: the slowest speed that still holds that power with 2 C to spare
    per = (r_lo - r_hi) / max(1.0, fan_hi - fan_lo)
    while lo < hi - 4.0 and idle + (r_hi + per * (fan_hi - lo)) * watts > hold - 2.0:
        lo += 2.0
    return {"power_w": round(watts, 1), "fan_min": round(lo), "fan_max": round(hi), "c_per_w_fan_max": round(r_hi, 4),
            "c_per_w_fan_min": round(r_lo, 4), "idle_c": round(idle, 1), "ceiling_c": ceiling}


def measure(read_temp, read_power, set_power, set_fan, start_load, stop_load, top_w, own_w, ceiling=79.0,
            fan_lo=70.0, fan_hi=82.0, phase_s=70.0, sleep=time.sleep, now=time.monotonic, say=print):
    """The two phases on the card (or a simulated one in the tests); returns plan(...). A phase whose trial power
    had to come down starts its curve again at the new power (the fit needs one constant power). When the fans
    cannot be set, one phase on the driver's own fan curve: the profile then holds only the power."""
    trial = min(float(top_w), max(float(own_w), 0.85 * float(top_w)))
    idle = read_temp()
    say(f"  idle {idle:.0f} C; load at {trial:.0f} W")
    set_power(trial)
    fans_ok = set_fan(fan_hi) is not False
    if not fans_ok:
        say("  the fans cannot be set on this card/driver: measuring on the driver's own fan curve")
    start_load()
    out = []
    try:
        for fan in ((fan_hi, fan_lo) if fans_ok else (None,)):
            if fan is not None and set_fan(fan) is False:
                raise RuntimeError("the fans stopped taking their speed")
            ts, temps, ws = [], [], []
            t0 = now()
            deadline = t0 + 2.5 * phase_s
            while now() - t0 < phase_s and now() < deadline:
                t = read_temp()
                w = read_power()
                if t >= ceiling + 3.0:
                    raise RuntimeError(f"the card reached {t:.0f} C even with the safety cuts: calibration stopped")
                if t >= ceiling + 1.0:
                    trial *= 0.9
                    set_power(trial)
                    say(f"  {t:.0f} C: trial power down to {trial:.0f} W")
                    ts, temps, ws, t0 = [], [], [], now()     # a new curve at the new power
                else:
                    ts.append(now())
                    temps.append(t)
                    if w:
                        ws.append(w)
                sleep(0.5)
            settled, tau = fit(ts, temps)
            watts = sum(ws[len(ws) // 2:]) / max(1, len(ws) - len(ws) // 2) if ws else trial
            say(f"  fans {'driver curve' if fan is None else f'{fan:.0f} %'}: settles at {settled:.1f} C "
                f"({watts:.0f} W" + (f", time constant {tau:.0f} s)" if tau else ")"))
            out.append((settled, watts))
    finally:
        stop_load()
        set_fan(None)
    if not fans_ok:
        s_, w_ = out[0]
        r = max(1e-3, (s_ - idle) / max(1.0, w_))
        watts = max(0.55 * float(top_w), min(float(top_w), 0.97 * (ceiling - 1.5 - idle) / r))
        return {"power_w": round(watts, 1), "fans": False, "c_per_w": round(r, 4), "idle_c": round(idle, 1),
                "ceiling_c": ceiling}
    (s_hi, w_hi), (s_lo, w_lo) = out
    return plan(idle, s_hi, w_hi, s_lo, w_lo, top_w, own_w, ceiling, fan_lo, fan_hi)


def _load(ops):
    """Matrix products back to back on the card until stopped (the heaviest steady load training makes)."""
    import numpy as np
    n = 4096
    A, B, C = ops.alloc(n * n), ops.alloc(n * n), ops.alloc(n * n)
    ops.put(A, np.random.default_rng(0).standard_normal(n * n).astype(np.float32) * 0.01)
    ops.put(B, np.random.default_rng(1).standard_normal(n * n).astype(np.float32) * 0.01)
    stop = threading.Event()

    def run():
        while not stop.is_set():
            for _ in range(8):
                ops.gemm((A, 0), (B, 0), (C, 0), n, n, n, n, 1, n, 1, n)
            ops.sync()
    th = threading.Thread(target=run, daemon=True)
    return (lambda: th.start()), (lambda: (stop.set(), th.join(timeout=10)))


def calibrate(index=0, ceiling=None, say=print):
    """apsi gpu calibrate: measures this card (about 2.5 minutes under full load) and saves its settings."""
    import ctypes
    from . import thermal, gpu_train
    lib = thermal._load_nvml()
    if lib is None:
        return ["no NVIDIA driver library (NVML): nothing to calibrate"]
    h = ctypes.c_void_p()
    if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h)) != 0:
        return [f"no GPU {index}"]
    name = ctypes.create_string_buffer(96)
    lib.nvmlDeviceGetName(h, name, ctypes.c_uint(96))
    name = name.value.decode(errors="replace")
    pc = thermal._PowerControl.try_make(lib, h, index)
    if pc is None:
        return [f"GPU {index}: the power limit cannot be set from here (administrator, or apsi gpu setup first)"]
    fc = thermal._FanControl.try_make(lib, h, index, getattr(pc, "via", "") == "directly")
    if fc is None:                                  # (APSI_GPU_FAN=0, or no way to set fans): measured without
        fc = type("NoFans", (), {"_set": staticmethod(lambda pct: False if pct is not None else True)})()
    ops, why = gpu_train.device(f"cuda:{index}")
    if ops is None:
        return [f"GPU {index}: {why}"]
    ceiling = float(ceiling or os.environ.get("APSI_GPU_TEMP", "79") or 79)

    def temp():
        t = ctypes.c_uint()
        return float(t.value) if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(t)) == 0 else ceiling

    def watts():
        mw = ctypes.c_uint()
        return mw.value / 1000.0 if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(mw)) == 0 else None

    start, stop = _load(ops)
    say(f"GPU {index} {name}: calibrating (about 2.5 minutes at full load; the fans will be loud for a while)")
    try:
        prof = measure(temp, watts, lambda w: pc.set(int(w * 1000)), lambda f: bool(fc._set(f)), start, stop,
                       pc.top / 1000.0,
                       pc.orig / 1000.0, ceiling, say=say)
    finally:
        pc.restore()
    prof.update(name=name, measured=time.strftime("%Y-%m-%d"), v=3)
    save_profile(card_key(name, index), prof)
    if prof.get("fans") is False:
        return [f"GPU {index} {name}: the fans cannot be set here; about {prof['power_w']:.0f} W holds "
                f"{ceiling - 1.5:.1f} C on the driver's fan curve (saved; the power limit holds the temperature)"]
    return [f"GPU {index} {name}: trains at {prof['power_w']:.0f} W, fans {prof['fan_min']}-{prof['fan_max']} %, "
            f"held at {ceiling - 0.5:.1f} C (saved; apsi gpu calibrate again after changing the cooling)"]
