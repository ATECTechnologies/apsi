"""Hardware detection and Apsi's hardware targets (`apsi doctor`, `use sys`).

Target tier:
  CPU  : x86-64-v3 (AVX2 + FMA) — Intel 7th gen Core i5 (i5-7500) or newer, AMD Ryzen or newer.
         Apple Silicon / ARMv8.2+ NEON is also accepted.
  GPU  : NVIDIA compute capability >= 7.5 (Turing: GTX 1660 / RTX 20xx) or newer.
"""
import os
import platform
import shutil
import subprocess

import numpy as np

from .values import Record

CPU_REQUIRED = ("AVX2", "FMA3")
GPU_MIN_CC = 7.5


def _cpu_features():
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as f
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as f
        except ImportError:
            f = {}
    return {k for k, v in f.items() if v}


def _cpu_name():
    name = platform.processor() or ""
    if platform.system() == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            return name
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or name
        except Exception:
            pass
    return name or platform.machine()


def _ram_gb():
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys / 1e9
        except Exception:
            return None
    try:
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    return None


def _find_smi():
    smi = shutil.which("nvidia-smi")
    if smi:
        return smi
    if platform.system() == "Windows":
        for p in (r"C:\Windows\System32\nvidia-smi.exe",
                  r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
            if os.path.exists(p):
                return p
    return None


def os_info():
    """(description, meets_target) — targets: Windows 10+ (build >= 10240), Linux kernel >= 4.15."""
    system = platform.system()
    if system == "Windows":
        try:
            build = __import__("sys").getwindowsversion().build
        except Exception:
            build = 0
        name = "Windows 11" if build >= 22000 else ("Windows 10" if build >= 10240 else f"Windows ({platform.release()})")
        return f"{name} (build {build})", build >= 10240
    if system == "Linux":
        rel = platform.release()
        try:
            major, minor = (int(x) for x in rel.split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        distro = ""
        try:
            with open("/etc/os-release", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PRETTY_NAME="):
                        distro = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        libc = " ".join(platform.libc_ver())
        desc = f"{distro or 'Linux'} (kernel {rel}{', ' + libc if libc.strip() else ''})"
        return desc, (major, minor) >= (4, 15)
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]}", True
    return f"{system} {platform.release()}", False


def _gpus():
    out = []
    smi = _find_smi()
    if not smi:
        return out
    try:
        r = subprocess.run([smi, "--query-gpu=name,compute_cap,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    cc = float(parts[1])
                except ValueError:
                    cc = 0.0
                out.append({"name": parts[0], "compute_capability": cc, "driver": parts[2],
                            "memory": parts[3], "meets_target": cc >= GPU_MIN_CC})
    except Exception:
        pass
    return out


def _cupy_ok():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def info():
    feats = _cpu_features()
    arch = platform.machine().lower()
    is_arm = arch in ("arm64", "aarch64")
    cpu_ok = is_arm and "ASIMD" in feats or all(f in feats for f in CPU_REQUIRED)
    gpus = _gpus()
    os_desc, os_ok = os_info()
    return {
        "os": os_desc, "os_meets_target": os_ok,
        "python": platform.python_version(),
        "cpu": _cpu_name(), "arch": arch, "threads": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "simd": sorted(f for f in feats if f.startswith(("AVX", "FMA", "SSE4", "ASIMD", "NEON"))),
        "cpu_meets_target": bool(cpu_ok),
        "gpus": gpus, "gpu_meets_target": any(g["meets_target"] for g in gpus),
        "cupy": _cupy_ok(),
        "numpy": np.__version__,
    }


def doctor(verbose=True):
    i = info()
    ok = lambda b: "✓" if b else "✗"
    print("Apsi doctor — hardware check")
    print(f"  OS       : {i['os']}")
    print(f"  Python   : {i['python']}   numpy {i['numpy']}")
    print(f"  {ok(i['os_meets_target'])} OS target   : Windows 10/11 (build 10240+) or Linux kernel 4.15+")
    print(f"  CPU      : {i['cpu']}  ({i['arch']}, {i['threads']} threads)")
    if i["ram_gb"]:
        print(f"  RAM      : {i['ram_gb']:.1f} GB")
    print(f"  SIMD     : {', '.join(i['simd']) or 'unknown'}")
    print(f"  {ok(i['cpu_meets_target'])} CPU target  : x86-64-v3 (AVX2+FMA, i5-7500 class or newer) / ARM NEON")
    if i["gpus"]:
        for g in i["gpus"]:
            print(f"  GPU      : {g['name']}  cc {g['compute_capability']}  driver {g['driver']}  {g['memory']}")
        print(f"  {ok(i['gpu_meets_target'])} GPU target  : NVIDIA compute capability >= {GPU_MIN_CC} (GTX 1660 class or newer)")
        print(f"  {'✓' if i['cupy'] else '-'} CuPy        : {'found' if i['cupy'] else 'not installed; optional, only for use gpu tensors (apsi pkg install gpu)'}")
    else:
        print("  - GPU      : no NVIDIA GPU detected (CPU backend will be used)")
    ready = i["cpu_meets_target"] and i["os_meets_target"]
    try:                                              # the native tier (machine code) and @gpu kernels
        from . import native as _nat
        from .native.gpu import info as _ginfo
        n = _nat.info()
        if n["available"]:
            print(f"  {ok(True)} Native      : LLVM {n['llvm']} for {n['cpu']} ({', '.join(n['simd'][-4:]) or 'scalar'}),"
                  f" {n['threads']} threads for parallel for")
        else:
            print(f"  {ok(False)} Native      : {n.get('reason')} (functions run on the bytecode engine)")
        g = _ginfo()
        if g["available"]:
            for d in g["devices"]:
                print(f"  {ok(True)} @gpu        : GPU {d['index']} {d['name']} (sm {d['compute_capability']}, "
                      f"{d['memory_gb']} GB) - driver PTX JIT, no toolkit needed")
            print(f"  {ok(True)} use hub     : language models run on GPU {g['devices'][0]['index']} by themselves"
                  " (nothing to install)")
        else:
            print(f"  - @gpu        : {g['reason']}; @gpu functions use all CPU cores")
    except Exception as e:                            # never let diagnostics fail
        print(f"  - Native      : could not be checked ({e})")
    print(f"\n  Status: {'ready' if ready else 'below target tier'}")
    return 0 if ready else 1


# ------------------------------------------------------------------ GPU tensors
def gpu_available():
    i = info()
    return i["gpu_meets_target"] and i["cupy"]


def to_gpu(x):
    import cupy
    from . import ops
    from .units import Quantity
    if isinstance(x, Quantity):
        return Quantity(cupy.asarray(ops.raw(x)), x.dim, x.disp)
    return cupy.asarray(ops.raw(x))


def to_cpu(x):
    from .units import Quantity
    v = x.v if isinstance(x, Quantity) else x
    if hasattr(v, "get"):
        v = v.get()
    return Quantity(v, x.dim, x.disp) if isinstance(x, Quantity) else v


def device():
    """"cuda:0" when a GPU meeting Apsi's target tier is usable through CuPy, otherwise "cpu"."""
    return "cuda:0" if gpu_available() else "cpu"


def to_device(x, dev=None):
    dev = dev or device()
    return to_gpu(x) if str(dev).startswith("cuda") else to_cpu(x)


def gpu_info():
    """GPU 0: name, compute capability, memory. Works with CuPy, or with just the NVIDIA driver."""
    from .values import Record
    try:
        import cupy
    except ImportError:
        cupy = None
    if cupy is None:
        from .native import cuda
        devs = cuda.devices()
        if not devs:
            from .errors import RuntimeErr
            raise RuntimeErr(f"no NVIDIA GPU is available ({cuda.unavailable_reason()})")
        d = devs[0]
        return Record(name=d["name"], compute_capability=d["compute_capability"], memory_gb=d["memory_gb"],
                      free_gb=None, multiprocessors=d["multiprocessors"])
    p = cupy.cuda.runtime.getDeviceProperties(0)
    free, total = cupy.cuda.runtime.memGetInfo()
    return Record(name=p["name"].decode() if isinstance(p["name"], bytes) else p["name"],
                  compute_capability=f"{p['major']}.{p['minor']}", memory_gb=total / 1e9,
                  free_gb=free / 1e9, multiprocessors=p["multiProcessorCount"])


def module():
    def hardware():
        i = info()
        i = dict(i)
        i["gpus"] = [Record(**g) for g in i["gpus"]]
        return Record(**i)
    return {"hardware": hardware, "gpu_available": gpu_available, "to_gpu": to_gpu,
            "to_cpu": to_cpu, "cpu_threads": lambda: os.cpu_count(), "device": device,
            "to_device": to_device, "gpu_info": gpu_info}
