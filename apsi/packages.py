"""`apsi pkg` — install, list, remove and lock the Python libraries Apsi programs use.

Packages are installed into the Python runtime that runs Apsi (the private runtime of an
installed Apsi, or the active environment for a pip install). Defaults are chosen for safety:

  * binary wheels only (no package build scripts run on your machine) unless --allow-source
  * HTTPS index only; --index-url must be https://
  * `apsi pkg lock` writes exact versions + sha256 hashes; `apsi pkg install -r apsi.lock`
    installs with --require-hashes so a tampered download is rejected
  * `apsi pkg audit` reports broken requirements (pip check) and known-vulnerable versions
    (pip-audit, when installed)
"""
import json
import os
import subprocess
import sys

# bundle name -> (description, pip requirements, extra pip arguments)
BUNDLES = {
    "gpu": ("NVIDIA GPU arrays (CUDA 12, GTX 1660 / compute 7.5 and newer)", ["cupy-cuda12x>=13"], []),
    "onnx": ("ONNX Runtime (CPU)", ["onnxruntime>=1.18"], []),
    "onnx-gpu": ("ONNX Runtime with CUDA", ["onnxruntime-gpu>=1.18"], []),
    # PyTorch builds come from PyTorch's own index (--index-url, not --extra-index-url: otherwise pip may take a
    # newer CPU-only wheel from PyPI). A CUDA build also runs on the CPU, so one build is enough.
    "torch": ("PyTorch, CPU only (smaller download)", ["torch>=2.3"],
              ["--index-url", "https://download.pytorch.org/whl/cpu"]),
    "torch-cuda": ("PyTorch + CUDA 12.6 (GTX 16 / RTX 20, 30, 40; NVIDIA driver 560+)", ["torch>=2.6"],
                   ["--index-url", "https://download.pytorch.org/whl/cu126"]),
    "torch-cu128": ("PyTorch + CUDA 12.8 (also RTX 50 series; driver 570+)", ["torch>=2.7"],
                    ["--index-url", "https://download.pytorch.org/whl/cu128"]),
    "torch-cu124": ("PyTorch + CUDA 12.4 (driver 550+)", ["torch>=2.3,<2.7"],
                    ["--index-url", "https://download.pytorch.org/whl/cu124"]),
    "torch-cu118": ("PyTorch + CUDA 11.8 (older drivers, GTX 10 series; driver 520+)", ["torch>=2.3"],
                    ["--index-url", "https://download.pytorch.org/whl/cu118"]),
    "transformers": ("Hugging Face Transformers + tokenizers + safetensors",
                     ["transformers>=4.40", "tokenizers", "safetensors", "huggingface_hub"], []),
    "sci": ("SciPy (sparse, special functions, optimisation)", ["scipy>=1.11"], []),
    "data": ("pandas + pyarrow (Parquet, Arrow, CSV at scale)", ["pandas>=2.1", "pyarrow>=15"], []),
    "plot": ("matplotlib for figures", ["matplotlib>=3.8"], []),
    "crypto": ("cryptography (AES-GCM, Ed25519, X.509)", ["cryptography>=42"], []),
    "audit": ("pip-audit vulnerability scanner", ["pip-audit>=2.7"], []),
}
TORCH_BUILDS = ("torch", "torch-cuda", "torch-cu128", "torch-cu124", "torch-cu118", "torch-rocm")
BUNDLES["torch-rocm"] = ("PyTorch + ROCm 6.2 (AMD GPUs on Linux)", ["torch>=2.5"],
                         ["--index-url", "https://download.pytorch.org/whl/rocm6.2"])
BUNDLES["torch-auto"] = ("PyTorch, the right build for this computer (CUDA for NVIDIA, ROCm for AMD, else CPU)",
                         ["torch>=2.3"], [])

GROUPS = {
    "ai": ["onnx", "torch-auto", "transformers"],
    "ai-gpu": ["gpu", "onnx-gpu", "torch-cuda", "transformers"],          # torch-cuda = CUDA 12.6
    "science": ["sci", "data", "plot"],
    "all": ["onnx", "torch", "transformers", "sci", "data", "plot", "crypto"],
}


def _pip(args, capture=False):
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-input"] + args
    if args and args[0] == "install":
        # PyTorch CUDA wheels are 2–3 GB: allow slow connections instead of pip's 15 s read timeout
        cmd += ["--timeout", "120", "--retries", "8"]
    env = dict(os.environ, PIP_NO_INPUT="1", PYTHONNOUSERSITE="1")
    if capture:
        return subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("$ " + " ".join(cmd[2:]))
    return subprocess.run(cmd, env=env)


def detect_torch_build():
    """The PyTorch bundle that fits this computer, and why.

    NVIDIA: picked from the driver version (nvidia-smi) — 570+ CUDA 12.8, 560+ CUDA 12.6, 550+ CUDA 12.4,
    520+ CUDA 11.8. AMD on Linux (rocminfo present): ROCm. Otherwise the CPU build (on macOS it includes MPS)."""
    import platform
    import shutil
    if platform.system() == "Darwin":
        return "torch", "macOS: the standard build (CPU + Apple MPS)"
    try:
        from .hardware import _find_smi
        smi = _find_smi()
    except Exception:
        smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=15)
            line = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
            name, _, drv = line.partition(",")
            major = int(drv.strip().split(".")[0])
            for need, bundle in ((570, "torch-cu128"), (560, "torch-cuda"), (550, "torch-cu124"), (520, "torch-cu118")):
                if major >= need:
                    return bundle, f"{name.strip()}, NVIDIA driver {drv.strip()}"
            return "torch", f"{name.strip()}: NVIDIA driver {drv.strip()} is too old for current CUDA builds " \
                            f"(update the driver to use the GPU)"
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    if platform.system() == "Linux" and shutil.which("rocminfo"):
        return "torch-rocm", "AMD GPU with ROCm"
    return "torch", "no supported GPU found"


def expand(names):
    reqs, extra, bundles = [], [], []
    for n in names:
        key = n.lower()
        if key == "torch-auto":
            key, why = detect_torch_build()
            print(f"PyTorch build for this computer: {key} ({why})")
        if key in GROUPS:
            for b in GROUPS[key]:
                r, e, bb = expand([b])
                reqs += r
                extra += [x for x in e if x not in extra]
                bundles += bb
        elif key in BUNDLES:
            reqs += BUNDLES[key][1]
            for x in BUNDLES[key][2]:
                extra.append(x)
            bundles.append(key)
        else:
            if any(c in n for c in ";|&`$<>") or n.startswith("-"):
                raise SystemExit(f"apsi pkg: refusing suspicious package name '{n}'")
            reqs.append(n)
    return reqs, extra, bundles


def _state_path():
    from .security import config_dir
    return os.path.join(config_dir(), "packages.json")


def _record(bundles, reqs, removed=False):
    try:
        path = _state_path()
        st = {}
        if os.path.exists(path):
            st = json.load(open(path, encoding="utf-8"))
        cur = set(st.get("installed", []))
        cur = (cur - set(bundles + reqs)) if removed else (cur | set(bundles or reqs))
        st["installed"] = sorted(cur)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(st, open(path, "w", encoding="utf-8"), indent=2)
    except (OSError, ValueError):
        pass


def _installed(dist):
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def cmd_list(args):
    print("Bundles (apsi pkg install NAME):")
    for k, (d, reqs, _) in BUNDLES.items():
        name = reqs[0].split(">")[0].split("=")[0]
        v = _installed(name)
        mark = f"installed {v}" if v else "-"
        print(f"  {k:<13} {mark:<20} {d}")
    print("\nGroups: " + ", ".join(f"{g} ({' '.join(b)})" for g, b in GROUPS.items()))
    print(f"\nRuntime: {sys.executable} (Python {sys.version.split()[0]})")
    return 0


def cmd_install(args):
    pip_args = ["install"]
    if args.requirement:
        pip_args += ["-r", args.requirement]
        if not args.no_hashes:
            pip_args.append("--require-hashes")
        reqs, extra, bundles = [], [], []
    else:
        if not args.names:
            print("apsi pkg install: name a bundle (see 'apsi pkg list') or a Python package")
            return 2
        reqs, extra, bundles = expand(args.names)
        if "--index-url" in extra:            # a PyTorch build: its own pip call, the rest from PyPI
            return _install_split(args, bundles, [n for n in args.names if n.lower() not in BUNDLES
                                                  and n.lower() not in GROUPS])
        pip_args += reqs + extra
    if not args.allow_source:
        pip_args.append("--only-binary=:all:")
    if args.index_url:
        if not args.index_url.startswith("https://"):
            print("apsi pkg: --index-url must use https://")
            return 2
        pip_args += ["--index-url", args.index_url]
    if args.upgrade:
        pip_args.append("--upgrade")
    if args.dry_run:
        pip_args.append("--dry-run")
    if os.environ.get("VIRTUAL_ENV") is None and _externally_managed():
        pip_args.append("--user")
    r = _pip(pip_args)
    if r.returncode == 0 and not args.dry_run:
        _record(bundles, reqs)
        print("Done. Programs that import these need --allow-python (or the developer profile).")
    return r.returncode


def _torch_build():
    """'2.6.0+cu126' / '2.6.0+cpu' / '2.6.0' (PyPI build) of the installed torch, or None."""
    return _installed("torch")


def _install_split(args, bundles, plain):
    """Install bundles with a private index (PyTorch builds) separately from the PyPI ones."""
    from types import SimpleNamespace
    torch_bundles = [b for b in bundles if b in TORCH_BUILDS]
    others = [b for b in bundles if b not in TORCH_BUILDS]
    if len(set(torch_bundles)) > 1:
        print(f"apsi pkg: choose one PyTorch build, not {', '.join(sorted(set(torch_bundles)))}")
        return 2
    rc = 0
    if torch_bundles:
        b = torch_bundles[0]
        reqs, extra = BUNDLES[b][1], BUNDLES[b][2]
        want = extra[extra.index("--index-url") + 1].rstrip("/").rsplit("/", 1)[-1]       # cpu / cu126 / …
        have = _torch_build()
        if have and have.endswith("+" + want) and not args.upgrade:
            print(f"PyTorch {have} is already installed — nothing to download.")
        else:
            if have and not args.dry_run:
                # pip would call another build of the same version "already satisfied"; remove it first so the
                # wanted build and its own dependencies (e.g. the nvidia-* libraries on Linux) are installed
                print(f"PyTorch {have} is installed; replacing it with the {want} build.")
                rc = _pip(["uninstall", "-y", "torch"]).returncode
            sub = SimpleNamespace(**dict(vars(args), names=[], requirement=None, index_url=None))
            rc = rc or _run_pip(sub, reqs + extra)
            if rc == 0:
                _record([b], reqs)
    if rc == 0 and (others or plain):
        reqs = expand(others)[0] + list(plain)
        extra = expand(others)[1]
        rc = _run_pip(args, reqs + extra)
        if rc == 0:
            _record(others, reqs)
    if rc == 0 and not args.dry_run:
        print("Done. Programs that import these need --allow-python (or the developer profile).")
    return rc


def _run_pip(args, items):
    pip_args = ["install"] + items
    if not args.allow_source:
        pip_args.append("--only-binary=:all:")
    if args.upgrade:
        pip_args.append("--upgrade")
    if args.dry_run:
        pip_args.append("--dry-run")
    if os.environ.get("VIRTUAL_ENV") is None and _externally_managed():
        pip_args.append("--user")
    return _pip(pip_args).returncode


def _externally_managed():
    import sysconfig
    stdlib = sysconfig.get_paths().get("stdlib", "")
    return os.path.exists(os.path.join(stdlib, "EXTERNALLY-MANAGED"))


def cmd_remove(args):
    reqs, _, bundles = expand(args.names)
    names = [r.split(">")[0].split("=")[0].split("<")[0] for r in reqs]
    r = _pip(["uninstall", "-y"] + names)
    if r.returncode == 0:
        _record(bundles, reqs, removed=True)
    return r.returncode


def cmd_lock(args):
    """Write exact versions with sha256 hashes of the wheels for this platform."""
    r = _pip(["freeze", "--exclude-editable"], capture=True)
    if r.returncode:
        print(r.stderr)
        return r.returncode
    pins = [l for l in r.stdout.splitlines() if l and not l.startswith(("#", "-e", "apsi=="))]
    lines = ["# apsi.lock — install with: apsi pkg install -r apsi.lock", ""]
    if args.no_hashes:
        lines += pins
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = _pip(["download", "--no-deps", "--only-binary=:all:", "-d", tmp] + pins, capture=True)
            if d.returncode:
                print(d.stderr[-2000:])
                return d.returncode
            import hashlib
            files = {f.lower(): os.path.join(tmp, f) for f in os.listdir(tmp)}
            for pin in pins:
                name = pin.split("==")[0].lower().replace("-", "_")
                hashes = [hashlib.sha256(open(p, "rb").read()).hexdigest()
                          for f, p in files.items() if f.replace("-", "_").startswith(name + "-")]
                lines.append(pin + "".join(f" \\\n    --hash=sha256:{h}" for h in hashes))
    out = args.output or "apsi.lock"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(pins)} packages)")
    return 0


def cmd_audit(args):
    rc = 0
    r = _pip(["check"], capture=True)
    print("Dependency check:", "ok" if r.returncode == 0 else "problems")
    if r.returncode:
        print(r.stdout)
        rc = 1
    try:
        import pip_audit  # noqa: F401
        a = subprocess.run([sys.executable, "-m", "pip_audit", "--progress-spinner", "off"])
        rc = rc or a.returncode
    except ImportError:
        print("Vulnerability scan: install the scanner with 'apsi pkg install audit'")
    return rc


def add_parser(sub):
    p = sub.add_parser("pkg", help="install, list, remove and lock Python libraries (GPU, ONNX, PyTorch, …)")
    ps = p.add_subparsers(dest="pkg_cmd")
    q = ps.add_parser("list", help="show bundles and what is installed")
    q.set_defaults(fn=cmd_list)
    q = ps.add_parser("install", help="install bundles (gpu, onnx, torch, transformers, sci, data, …) or packages")
    q.add_argument("names", nargs="*")
    q.add_argument("-r", "--requirement", help="install from a lock/requirements file (hashes required)")
    q.add_argument("--no-hashes", action="store_true", help="with -r: do not require hashes")
    q.add_argument("--allow-source", action="store_true", help="also allow source packages (runs their build scripts)")
    q.add_argument("--index-url", help="alternative https package index")
    q.add_argument("-U", "--upgrade", action="store_true")
    q.add_argument("--dry-run", action="store_true", help="show what would be installed")
    q.set_defaults(fn=cmd_install)
    q = ps.add_parser("remove", help="uninstall bundles or packages")
    q.add_argument("names", nargs="+")
    q.set_defaults(fn=cmd_remove)
    q = ps.add_parser("lock", help="write apsi.lock with exact versions and sha256 hashes")
    q.add_argument("-o", "--output")
    q.add_argument("--no-hashes", action="store_true")
    q.set_defaults(fn=cmd_lock)
    q = ps.add_parser("audit", help="check installed packages for conflicts and known vulnerabilities")
    q.set_defaults(fn=cmd_audit)
    p.set_defaults(fn=lambda a: cmd_list(a) if not getattr(a, "pkg_cmd", None) else 0)
