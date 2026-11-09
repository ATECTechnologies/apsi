"""Apsi security model: capability permissions with path/host scoping, resource limits,
security profiles, static capability audit and installation integrity checks.

Permissions (deny by default except what the profile grants):
  read   : read files            --allow-read[=DIR,DIR]
  write  : create/modify files   --allow-write[=DIR,DIR]
  net    : outbound + listening  --allow-net[=HOST,HOST:PORT,:PORT]
  env    : environment variables --allow-env[=NAME,NAME]
  python : import Python modules --allow-python
  ffi    : load native libraries --allow-ffi

Profiles (default from ~/.apsi/config.json, override with --profile):
  sandbox   : nothing at all
  strict    : read/write only inside the program's own folder
  standard  : read/write files anywhere (default, matches earlier releases)
  developer : everything (same as --allow-all)
"""
import hashlib
import json
import os
import sys
import threading

from .errors import ApsiError

ALL = None          # scope value meaning "no restriction"
PROFILES = ("sandbox", "strict", "standard", "developer")


class PermissionErr(ApsiError):
    kind = "PermissionError"


class Policy:
    def __init__(self):
        self.read = []          # [] = denied, ALL = anywhere, [dirs] = only there
        self.write = []
        self.net = []
        self.env = []
        self.python = False
        self.ffi = False
        self.block_private = False   # wildcard --allow-net may not reach private/loopback addresses
        self.unguarded = False       # True only for internal tooling, never from the command line
        self.ffi_pins = {}           # library name or path -> sha256 (from --allow-ffi=lib@sha256)

    # -- legacy set view: {"fs", "net", "python", "ffi", "env"} for older code and permissions()
    def as_set(self):
        s = set()
        if self.read != [] or self.write != []:
            s.add("fs")
        if self.net != []:
            s.add("net")
        if self.env != []:
            s.add("env")
        if self.python:
            s.add("python")
        if self.ffi:
            s.add("ffi")
        return s

    @classmethod
    def from_set(cls, perms):
        p = cls()
        perms = set(perms or ())
        if "fs" in perms:
            p.read = ALL
            p.write = ALL
        if "net" in perms:
            p.net = ALL
        if "env" in perms:
            p.env = ALL
        p.python = "all" if "python" in perms else False
        p.ffi = "ffi" in perms
        return p

    @classmethod
    def profile(cls, name, base_dir=None):
        p = cls()
        if name == "sandbox":
            return p
        if name == "strict":
            root = os.path.realpath(base_dir or os.getcwd())
            p.read = [root]
            p.write = [root]
            p.block_private = True
            return p
        if name == "developer":
            return cls.from_set({"fs", "net", "env", "python", "ffi"})
        p.read = ALL
        p.write = ALL
        return p

    def describe(self):
        def scope(v):
            return "anywhere" if v is ALL else ("denied" if v == [] else ", ".join(v))
        return {"read": scope(self.read), "write": scope(self.write), "net": scope(self.net),
                "env": scope(self.env), "python": self.python, "ffi": self.ffi}


# ------------------------------------------------------------------ checks
def _policy(interp):
    pol = getattr(interp, "policy", None)
    if pol is None:
        pol = Policy.from_set(getattr(interp, "permissions", {"fs"}))
        interp.policy = pol
    return pol


def _within(path, roots):
    rp = os.path.realpath(path)
    for r in roots:
        r = os.path.realpath(r)
        if rp == r or rp.startswith(r.rstrip(os.sep) + os.sep):
            return True
    return False


# ------------------------------------------------------------------ guarded locations
# Writes here are refused for every profile: a program must never be able to raise its own
# permissions (config.json), tamper with the runtime, or plant start-up code.
def protected_write_roots():
    home = os.path.expanduser("~")
    roots = [config_dir(), os.path.dirname(os.path.abspath(__file__)), sys.prefix, sys.exec_prefix,
             os.path.join(home, ".ssh"), os.path.join(home, ".gnupg"),
             os.path.join(home, ".bashrc"), os.path.join(home, ".profile"), os.path.join(home, ".zshrc"),
             os.path.join(home, ".bash_profile"), os.path.join(home, ".config", "autostart"),
             os.path.join(home, ".local", "bin")]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        roots += [os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup"),
                  os.environ.get("SystemRoot", r"C:\Windows"),
                  os.environ.get("ProgramFiles", r"C:\Program Files"),
                  os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    else:
        roots += ["/etc", "/bin", "/sbin", "/usr", "/boot", "/lib", "/lib64", "/proc", "/sys", "/dev"]
    extra = os.environ.get("APSI_PROTECT")
    if extra:
        roots += _split(extra.replace(os.pathsep, ","))
    return [r for r in roots if r]


# Reads here are refused unless the folder is listed explicitly (--allow-read=~/.ssh):
# a blanket "read anywhere" grant does not include credentials.
def secret_read_roots():
    home = os.path.expanduser("~")
    names = [".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".netrc", ".pgpass", ".git-credentials",
             os.path.join(".config", "gcloud"), os.path.join(".config", "gh"),
             os.path.join(".mozilla", "firefox"), os.path.join(".config", "google-chrome"),
             os.path.join(".config", "chromium"), os.path.join(".password-store")]
    roots = [os.path.join(home, n) for n in names]
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        appdata = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        roots += [os.path.join(local, "Google", "Chrome", "User Data"),
                  os.path.join(local, "Microsoft", "Edge", "User Data"),
                  os.path.join(appdata, "Mozilla", "Firefox"),
                  os.path.join(appdata, "Microsoft", "Credentials"),
                  os.path.join(local, "Microsoft", "Credentials"),
                  os.path.join(appdata, "Microsoft", "Protect")]
    else:
        roots += ["/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/root"]
    roots.append(os.path.join(config_dir(), "secrets"))
    return roots


_BAD_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def _sanity(path, what):
    if not isinstance(path, str) or not path:
        raise PermissionErr(f"{what}: path must be a non-empty string")
    if "\x00" in path:
        raise PermissionErr(f"{what}: path contains a NUL byte")
    if os.name == "nt":
        p = path.replace("/", "\\")
        if p.startswith("\\\\?\\") or p.startswith("\\\\.\\"):
            raise PermissionErr(f"{what}: device and \\\\?\\ paths are not allowed")
        stem = os.path.basename(p).split(".")[0].upper()
        if stem in _BAD_NAMES:
            raise PermissionErr(f"{what}: '{path}' is a reserved device name")
        if ":" in p[2:]:
            raise PermissionErr(f"{what}: alternate data streams are not allowed ('{path}')")


def check_path(interp, path, mode, what="file access"):
    """Resolve `path` against the running program's folder and enforce read/write scopes.

    Symlinks are resolved before the check, so a link inside an allowed folder cannot
    point outside it. Guarded locations are refused even when the scope is "anywhere"."""
    pol = _policy(interp)
    _sanity(path, what)
    base = interp.dir_stack[-1] if getattr(interp, "dir_stack", None) else os.getcwd()
    full = path if os.path.isabs(path) else os.path.join(base, path)
    scopes = [pol.read] if mode == "read" else ([pol.write] if mode == "write" else [pol.read, pol.write])
    for sc in scopes:
        if sc == []:
            flag = "--allow-read" if sc is pol.read else "--allow-write"
            if mode == "rw":
                flag = "--allow-read and --allow-write"
            raise PermissionErr(f"{what} ('{path}') needs {mode} permission; run with {flag} "
                                f"(or a profile that allows it)")
        if sc is not ALL and not _within(full, sc):
            raise PermissionErr(f"{what}: '{path}' is outside the allowed folders ({', '.join(sc)})")
    if mode in ("write", "rw") and interp is not None and not getattr(pol, "unguarded", False):
        from .guard import monitor_for
        monitor_for(interp).destructive(os.path.realpath(full), what)
    if mode in ("write", "rw") and not getattr(pol, "unguarded", False) and _within(full, protected_write_roots()):
        _record(interp, "denied-write", full)
        raise PermissionErr(f"{what}: '{path}' is a protected location (Apsi settings, the runtime, "
                            f"system folders, shell start-up files); programs can never write there")
    if mode in ("read", "rw") and pol.read is ALL and _within(full, secret_read_roots()):
        _record(interp, "denied-read", full)
        raise PermissionErr(f"{what}: '{path}' holds credentials or browser data; allow it explicitly "
                            f"with --allow-read={os.path.dirname(os.path.realpath(full))}")
    return os.path.realpath(full)


def _record(interp, kind, detail):
    """Append a security event to the audit log (~/.apsi/audit.log) when logging is enabled."""
    if not (os.environ.get("APSI_AUDIT_LOG") or load_config().get("audit_log")):
        return
    try:
        import time
        os.makedirs(config_dir(), exist_ok=True)
        log = os.path.join(config_dir(), "audit.log")
        if os.path.exists(log) and os.path.getsize(log) > 5 * 1024 * 1024:
            os.replace(log, log + ".1")          # keep one 5 MB generation
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, "detail": str(detail),
                                 "pid": os.getpid()}) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------ network guard (SSRF)
def _ip_blocked(ip):
    """Cloud metadata, link-local, multicast and unspecified addresses are never reachable
    through a wildcard grant; loopback and private ranges only with an explicit host."""
    import ipaddress
    a = ipaddress.ip_address(ip)
    if getattr(a, "ipv4_mapped", None):
        a = a.ipv4_mapped
    if a.is_link_local or a.is_multicast or a.is_unspecified or a.is_reserved:
        return "link-local/metadata"
    if a.is_loopback or a.is_private:
        return "private"
    return None


def resolve_checked(interp, host, port, what="network access"):
    """Resolve `host` and refuse addresses that a wildcard --allow-net must not reach
    (defeats DNS tricks such as evil.example -> 169.254.169.254). Returns the IPs."""
    import socket
    pol = _policy(interp)
    explicit = pol.net is not ALL and any(e.lower().split(":")[0] == (host or "").lower() for e in pol.net if e)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise PermissionErr(f"{what}: cannot resolve '{host}' ({e})")
    ips = sorted({i[4][0] for i in infos})
    for ip in ips:
        why = _ip_blocked(ip.split("%")[0])
        if why == "link-local/metadata" and not explicit:
            _record(interp, "denied-net", f"{host} -> {ip}")
            raise PermissionErr(f"{what}: '{host}' resolves to {ip} ({why}); this address is blocked "
                                f"unless the host is listed exactly in --allow-net")
        if why == "private" and not explicit and pol.net is ALL and getattr(pol, "block_private", False):
            _record(interp, "denied-net", f"{host} -> {ip}")
            raise PermissionErr(f"{what}: '{host}' resolves to private address {ip}; list it in --allow-net "
                                f"to reach internal services")
    return ips


def check_net(interp, host, port=None, what="network access"):
    pol = _policy(interp)
    if pol.net == []:
        raise PermissionErr(f"{what} needs permission 'net'; run with --allow-net or --allow-net={host}")
    if pol.net is ALL:
        return
    host = (host or "").lower()
    for entry in pol.net:
        e = entry.lower()
        h, _, p = e.rpartition(":") if e.count(":") == 1 else (e, "", "")
        if e.startswith(":"):
            if port is not None and str(port) == e[1:]:
                return
            continue
        if (h or e) in (host, "*") and (not p or port is None or str(port) == p):
            return
        if (h or e).startswith("*.") and host.endswith((h or e)[1:]) and (not p or port is None or str(port) == p):
            return
    _record(interp, "denied-net", f"{host}:{port}")
    raise PermissionErr(f"{what}: '{host}{':' + str(port) if port else ''}' is not in --allow-net "
                        f"({', '.join(pol.net)})")


def check_env(interp, name):
    pol = _policy(interp)
    if pol.env == []:
        raise PermissionErr(f"reading environment variable '{name}' needs --allow-env")
    if pol.env is not ALL and name not in pol.env:
        raise PermissionErr(f"environment variable '{name}' is not in --allow-env ({', '.join(pol.env)})")


def need(interp, perm, what):
    """Coarse check used by older call sites."""
    pol = _policy(interp)
    ok = {"fs": pol.read != [] or pol.write != [], "net": pol.net != [], "env": pol.env != [],
          "python": pol.python, "ffi": pol.ffi}.get(perm, False)
    if not ok:
        if perm == "fs":
            raise PermissionErr(f"{what} is not allowed in sandbox mode")
        flag = {"net": "--allow-net", "env": "--allow-env", "python": "--allow-python", "ffi": "--allow-ffi"}[perm]
        raise PermissionErr(f"{what} needs permission '{perm}'; run with {flag} or --allow-all")


# ------------------------------------------------------------------ config & profiles
def config_dir():
    return os.environ.get("APSI_HOME") or os.path.join(os.path.expanduser("~"), ".apsi")


def load_config():
    path = os.path.join(config_dir(), "config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(config_dir(), exist_ok=True)
    path = os.path.join(config_dir(), "config.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, path)
    return path


def _split(v):
    return [x.strip() for x in str(v).split(",") if x.strip()]


def policy_from_args(args, program_path=None):
    """Build the policy for a CLI run: profile (config or --profile) + explicit flags."""
    cfg = load_config()
    base = os.path.dirname(os.path.abspath(program_path)) if program_path else os.getcwd()
    name = getattr(args, "profile", None) or cfg.get("profile", "standard")
    if name not in PROFILES:
        name = "standard"
    if getattr(args, "sandbox", False):
        name = "sandbox"
    pol = Policy.profile(name, base)
    if getattr(args, "allow_all", False):
        return Policy.profile("developer")

    def apply(attr, value):
        if value is None:
            return
        if value == "*":
            setattr(pol, attr, ALL)
        else:
            cur = getattr(pol, attr)
            items = [os.path.realpath(os.path.join(base, x)) if attr in ("read", "write") else x
                     for x in _split(value)]
            setattr(pol, attr, items if cur is ALL or cur == [] else cur + items)
    apply("read", getattr(args, "allow_read", None))
    apply("write", getattr(args, "allow_write", None))
    apply("net", getattr(args, "allow_net", None))
    apply("env", getattr(args, "allow_env", None))
    ap = getattr(args, "allow_python", None)
    if ap:
        pol.python = "all" if str(ap).lower() in ("all", "unsafe") else "safe"
    af = getattr(args, "allow_ffi", None)
    if af:
        pol.ffi = True
        if af not in (True, "*"):
            for item in _split(af):
                name, _, h = item.partition("@")
                if h:
                    pol.ffi_pins[name] = h.lower()
    return pol


# ------------------------------------------------------------------ resource limits
def parse_size(text):
    t = str(text).strip().upper().replace(" ", "")
    mult = 1
    for suf, m in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10), ("T", 1 << 40),
                   ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10), ("B", 1)):
        if t.endswith(suf):
            mult, t = m, t[:-len(suf)]
            break
    return int(float(t) * mult)


def parse_seconds(text):
    t = str(text).strip().lower()
    for suf, m in (("ms", 0.001), ("s", 1), ("min", 60), ("m", 60), ("h", 3600)):
        if t.endswith(suf):
            return float(t[:-len(suf)]) * m
    return float(t)


def apply_limits(max_memory=None, timeout=None):
    """--max-memory caps the address space (Linux) / job memory (Windows); --timeout kills runaway programs."""
    notes = []
    if max_memory:
        limit = parse_size(max_memory)
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard if hard != resource.RLIM_INFINITY and hard < limit
                                                    else hard))
        elif sys.platform.startswith("win"):
            if not _windows_job_limit(limit):
                notes.append("could not apply --max-memory on this Windows system")
    if timeout:
        secs = parse_seconds(timeout)

        def kill():
            sys.stderr.write(f"\napsi: time limit of {secs:g} s exceeded — program stopped\n")
            sys.stderr.flush()
            os._exit(124)
        t = threading.Timer(secs, kill)
        t.daemon = True
        t.start()
    return notes


def _windows_job_limit(limit):
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic.LimitFlags = 0x100          # JOB_OBJECT_LIMIT_PROCESS_MEMORY
        info.ProcessMemoryLimit = limit
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        return bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    except Exception:
        return False


# ------------------------------------------------------------------ static capability audit
CAPABILITIES = {
    "pyimport": ("python", "imports Python modules"), "to_torch": ("python", "uses PyTorch"),
    "from_torch": ("python", "uses PyTorch"), "onnx_model": ("python", "runs ONNX models"),
    "hf_pipeline": ("python", "runs Hugging Face models"), "torch_load": ("python", "loads PyTorch files"),
    "cload": ("ffi", "loads native libraries"),
    "http_get": ("net", "makes HTTP requests"), "http_post_json": ("net", "makes HTTP requests"),
    "run": ("net", "may start a network server (app.run)"),
    "getenv": ("env", "reads environment variables"),
    "write_text": ("write", "writes files"), "save_json": ("write", "writes files"),
    "save_csv": ("write", "writes files"), "save_tensor": ("write", "writes files"),
    "open_tensor": ("write", "creates memory-mapped files"), "save_safetensors": ("write", "writes model files"),
    "disk_cache": ("write", "writes a disk cache"), "cache": ("write", "may write a disk cache"),
    "save_params": ("write", "writes files"), "read_text": ("read", "reads files"),
    "load_json": ("read", "reads files"), "load_csv": ("read", "reads files"),
    "load_tensor": ("read", "reads files"), "from_csv": ("read", "reads files"),
    "from_npy": ("read", "reads files"), "from_files": ("read", "lists files"),
    "load_safetensors": ("read", "reads model files"),
    "save_model": ("write", "writes model files"), "save_gguf": ("write", "writes model files"),
    "load_model": ("read", "reads model files"), "load_gguf": ("read", "reads model files"),
    "model_info": ("read", "reads model files"), "gguf_info": ("read", "reads model files"),
    "read_dicom": ("read", "reads medical images"), "read_dicom_series": ("read", "reads medical images"),
    "load_runs": ("read", "reads experiment logs"), "experiment": ("write", "writes experiment logs"), "load_params": ("read", "reads files"),
    "static": ("read", "serves files (app.static)"), "exists": ("read", "checks for files"),
}


def audit(prog):
    """Return [(line, name, permission, description)] for every capability a program uses."""
    from . import ast as A
    found = []

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
            return
        if isinstance(n, tuple):
            for x in n:
                walk(x)
            return
        if not isinstance(n, A.AST):
            return
        if isinstance(n, A.Call):
            f = n.func
            name = f.id if isinstance(f, A.Name) else (f.name if isinstance(f, A.Attr) else None)
            if name in CAPABILITIES:
                perm, desc = CAPABILITIES[name]
                found.append((n.line, name, perm, desc))
        if isinstance(n, A.Use) and n.path in ("python", "ffi", "net"):
            found.append((n.line, f"use {n.path}", n.path, f"imports the {n.path} module"))
        for fld in n._fields:
            walk(getattr(n, fld))
    walk(prog.body)
    return found


# ------------------------------------------------------------------ installation integrity
def package_dir():
    return os.path.dirname(os.path.abspath(__file__))


def compute_manifest():
    """SHA-256 of every .py file of the runtime, subpackages included (key: path relative to the package)."""
    out = {}
    root = package_dir()
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        rel = os.path.relpath(base, root)
        for name in sorted(files):
            if name.endswith(".py"):
                key = name if rel == "." else rel.replace(os.sep, "/") + "/" + name
                with open(os.path.join(base, name), "rb") as fh:
                    out[key] = hashlib.sha256(fh.read()).hexdigest()
    return out


def write_manifest():
    from . import __version__
    path = os.path.join(package_dir(), "MANIFEST.json")
    data = {"version": __version__, "files": compute_manifest()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return path


def verify_installation():
    """Compare every source file of the installed runtime with the release manifest."""
    path = os.path.join(package_dir(), "MANIFEST.json")
    if not os.path.exists(path):
        return None, ["no MANIFEST.json (development checkout)"]
    with open(path, encoding="utf-8") as fh:
        expected = json.load(fh)["files"]
    actual = compute_manifest()
    problems = []
    for name, digest in expected.items():
        if name not in actual:
            problems.append(f"missing: {name}")
        elif actual[name] != digest:
            problems.append(f"modified: {name}")
    for name in actual:
        if name not in expected:
            problems.append(f"unexpected file: {name}")
    return not problems, problems
