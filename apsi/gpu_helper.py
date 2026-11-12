"""The GPU helper: a tiny program that runs with administrator rights (started at logon by Windows' Task Scheduler,
or by systemd on Linux) and does one thing for Apsi programs that run without them: set the graphics card's power
limit. So the temperature can be held by the power limit (steady speed, no pauses) without ever running Apsi as an
administrator.

Set up once, from a terminal opened as administrator (or with sudo):  apsi gpu setup      (apsi gpu remove undoes it)

Safety: it listens on 127.0.0.1 only; every request must carry the key in ~/.apsi/gpu-helper.json, which only your
account can read; it can only set a limit between the card's minimum and its *default* (never above: no
overclocking), read the card's state, or put the default back. It uses nothing but the driver's NVML library.
This file is self-contained (Python standard library + ctypes), so it keeps working whatever Apsi version is
installed; `apsi gpu setup` copies it to ~/.apsi/gpu_helper.py.
"""
import ctypes
import hmac
import json
import os
import socket
import sys
import threading
import time

PORT = 47731


def config_path():
    return os.path.join(os.path.expanduser("~"), ".apsi", "gpu-helper.json")


def _nvml():
    names = ([os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvml.dll"),
              os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI",
                           "nvml.dll")] if sys.platform.startswith("win") else ["libnvidia-ml.so.1", "libnvidia-ml.so"])
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            if lib.nvmlInit_v2() == 0:
                return lib
        except (OSError, AttributeError):
            continue
    return None


def _handle(lib, index):
    h = ctypes.c_void_p()
    return h if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(index)), ctypes.byref(h)) == 0 else None


def _limits(lib, h):
    cur, dflt, lo, hi = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
    if lib.nvmlDeviceGetPowerManagementLimit(h, ctypes.byref(cur)) or \
            lib.nvmlDeviceGetPowerManagementDefaultLimit(h, ctypes.byref(dflt)) or \
            lib.nvmlDeviceGetPowerManagementLimitConstraints(h, ctypes.byref(lo), ctypes.byref(hi)):
        return None
    return {"current_mw": cur.value, "default_mw": dflt.value, "min_mw": lo.value, "max_mw": hi.value}


def handle(lib, req, key):
    if not hmac.compare_digest(str(req.get("key", "")), key):
        return {"ok": False, "error": "wrong key"}
    op = req.get("op")
    h = _handle(lib, req.get("gpu", 0))
    if h is None:
        return {"ok": False, "error": "no such GPU"}
    lim = _limits(lib, h)
    if lim is None:
        return {"ok": False, "error": "this card does not report power limits"}
    if op == "status":
        return {"ok": True, **lim}
    if op in ("set", "default"):
        mw = lim["default_mw"] if op == "default" else int(req.get("mw", lim["default_mw"]))
        mw = max(lim["min_mw"], min(lim["default_mw"], mw))          # never above the default
        r = lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(mw))
        return {"ok": r == 0, "mw": mw, "error": None if r == 0 else f"NVML error {r}"}
    return {"ok": False, "error": f"unknown request '{op}'"}


def serve(start_limit=None):
    lib = _nvml()
    if lib is None:
        print("apsi gpu helper: no NVIDIA driver (NVML)", file=sys.stderr)
        return 1
    with open(config_path(), encoding="utf-8") as fh:
        cfg = json.load(fh)
    key, port = cfg["key"], int(cfg.get("port", PORT))
    if start_limit or cfg.get("start_limit"):
        spec = str(start_limit or cfg["start_limit"]).lower()
        n = ctypes.c_uint()
        lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        for i in range(n.value):
            h = _handle(lib, i)
            lim = h is not None and _limits(lib, h)
            if lim:
                mw = int(float(spec[:-1]) * 1000) if spec.endswith("w") else int(lim["default_mw"] * float(spec.rstrip("%")) / 100)
                lib.nvmlDeviceSetPowerManagementLimit(h, ctypes.c_uint(max(lim["min_mw"], min(lim["default_mw"], mw))))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)
    lock = threading.Lock()

    def client(conn):
        with conn:
            conn.settimeout(10)
            buf = b""
            while not buf.endswith(b"\n") and len(buf) < 4096:
                chunk = conn.recv(1024)
                if not chunk:
                    break
                buf += chunk
            try:
                req = json.loads(buf.decode() or "{}")
                with lock:
                    out = handle(lib, req, key)
            except Exception as e:                       # noqa: BLE001
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall((json.dumps(out) + "\n").encode())
            except OSError:
                pass
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=client, args=(conn,), daemon=True).start()


def call(op, timeout=2.0, **kw):
    """From any Apsi program: a request to the helper; None when there is no helper."""
    try:
        with open(config_path(), encoding="utf-8") as fh:
            cfg = json.load(fh)
        with socket.create_connection(("127.0.0.1", int(cfg.get("port", PORT))), timeout=timeout) as s:
            s.sendall((json.dumps(dict(kw, op=op, key=cfg["key"])) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
        return json.loads(buf.decode())
    except (OSError, ValueError, KeyError):
        return None


# ------------------------------------------------------------------------------------------------- set up / remove
TASK = "Apsi GPU helper"


def _is_admin():
    if sys.platform.startswith("win"):
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0


def setup(start_limit=None):
    """Installs the helper (needs an administrator / root once). Returns the lines to show."""
    import secrets
    import shutil
    import subprocess
    if not _is_admin():
        return ["apsi gpu setup needs an administrator once: right-click the terminal (or gpu-setup.bat) -> Run as "
                "administrator" if sys.platform.startswith("win") else "run it once with sudo"]
    home = os.path.join(os.path.expanduser("~"), ".apsi")
    os.makedirs(home, exist_ok=True)
    script = os.path.join(home, "gpu_helper.py")
    shutil.copyfile(os.path.abspath(__file__), script)
    cfg = {"key": secrets.token_hex(24), "port": PORT, "start_limit": start_limit}
    with open(config_path(), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    try:
        os.chmod(config_path(), 0o600)
    except OSError:
        pass
    exe = sys.executable
    if sys.platform.startswith("win"):
        w = os.path.join(os.path.dirname(exe), "pythonw.exe")
        exe = w if os.path.exists(w) else exe
        user = f"{os.environ.get('USERDOMAIN', '')}\\{os.environ.get('USERNAME', '')}".strip("\\")
        cmd = ["schtasks", "/Create", "/TN", TASK, "/TR", f'"{exe}" "{script}" serve', "/SC", "ONLOGON",
               "/RL", "HIGHEST", "/RU", user, "/IT", "/F"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return [f"Task Scheduler refused: {(r.stderr or r.stdout).strip()}"]
        subprocess.run(["schtasks", "/Run", "/TN", TASK], capture_output=True, text=True)
        where = f"Task Scheduler task '{TASK}' (at every logon, highest privileges)"
    else:
        unit = "/etc/systemd/system/apsi-gpu-helper.service"
        user_home = os.path.expanduser("~")
        with open(unit, "w") as fh:
            fh.write(f"[Unit]\nDescription=Apsi GPU helper (power limit)\nAfter=multi-user.target\n\n[Service]\n"
                     f"Environment=HOME={user_home}\nExecStart={exe} {script} serve\nRestart=on-failure\n\n"
                     f"[Install]\nWantedBy=multi-user.target\n")
        subprocess.run(["systemctl", "daemon-reload"])
        subprocess.run(["systemctl", "enable", "--now", "apsi-gpu-helper"])
        where = "systemd service apsi-gpu-helper"
    for _ in range(20):
        time.sleep(0.25)
        if call("status") is not None:
            return [f"the GPU helper runs ({where}); Apsi now holds the temperature by the power limit without "
                    "administrator rights" + (f"; start limit {start_limit}" if start_limit else "")]
    return [f"installed ({where}); it starts at the next logon"]


def remove():
    import subprocess
    if not _is_admin():
        return ["apsi gpu remove needs an administrator / root once"]
    call("default")
    if sys.platform.startswith("win"):
        subprocess.run(["schtasks", "/End", "/TN", TASK], capture_output=True)
        subprocess.run(["schtasks", "/Delete", "/TN", TASK, "/F"], capture_output=True)
    else:
        subprocess.run(["systemctl", "disable", "--now", "apsi-gpu-helper"])
        try:
            os.remove("/etc/systemd/system/apsi-gpu-helper.service")
        except OSError:
            pass
    for f in (config_path(), os.path.join(os.path.expanduser("~"), ".apsi", "gpu_helper.py")):
        try:
            os.remove(f)
        except OSError:
            pass
    return ["the GPU helper is removed and the card's default power limit is back"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.exit(serve(sys.argv[2] if len(sys.argv) > 2 else None))
    print(__doc__)
