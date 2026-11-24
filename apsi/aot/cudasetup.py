"""`apsi cuda`: find the NVIDIA driver and the CUDA libraries Apsi can use, and install cuBLAS when it is missing.

Native programs and `apsi run` load the driver (nvcuda.dll / libcuda.so) at run time and use Apsi's own kernels;
when cuBLAS is found (any CUDA 11, 12 or 13) the matrix products go to it, with Tensor Cores on request
(APSI_GPU_MATH=tf32 / bf16 / f16). cuBLAS comes from a CUDA Toolkit install (CUDA_PATH / CUDA_HOME), from the
NVIDIA wheels in a virtual environment (.venv next to the program, or the active one), or from the system path.

    apsi cuda                     what is found, and what each program will use
    apsi cuda install             cuBLAS for the installed driver into ./.venv (made when missing)
    apsi cuda install --cuda 12.6 a given CUDA line (cu11 / cu12 / cu13 wheels)
    apsi cuda install --user      into ~/.apsi/cuda instead (found by every program of this user)
"""
import glob
import os
import shutil
import subprocess
import sys

WHEELS = {"11": "nvidia-cublas-cu11", "12": "nvidia-cublas-cu12", "13": "nvidia-cublas-cu13"}


def _driver():
    import ctypes
    names = ["nvcuda.dll"] if os.name == "nt" else ["libcuda.so.1", "libcuda.so"]
    for n in names:
        try:
            lib = ctypes.CDLL(n)
        except OSError:
            continue
        v = ctypes.c_int(0)
        try:
            if lib.cuDriverGetVersion(ctypes.byref(v)) == 0:
                return n, v.value
        except AttributeError:
            pass
        return n, 0
    return None, 0


def _cublas_candidates(venv=None):
    pats = []
    win = os.name == "nt"
    lib = "cublas64_*.dll" if win else "libcublas.so*"
    for var in ("APSI_CUBLAS_PATH",):
        if os.environ.get(var):
            pats.append(os.environ[var])
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        r = os.environ.get(var)
        if r:
            pats += [os.path.join(r, "bin", lib), os.path.join(r, "bin", "x64", lib), os.path.join(r, "lib64", lib),
                     os.path.join(r, "lib", lib)]
    for v in [venv, os.environ.get("VIRTUAL_ENV"), ".venv", os.path.join(os.path.expanduser("~"), ".apsi", "cuda")]:
        if not v:
            continue
        if win:
            pats += [os.path.join(v, "Lib", "site-packages", "nvidia", "cublas", "bin", lib),
                     os.path.join(v, "Lib", "site-packages", "nvidia", "cu13", "bin", lib)]
        else:
            pats += [os.path.join(v, "lib", "python3.*", "site-packages", "nvidia", "cublas", "lib", lib),
                     os.path.join(v, "lib", "python3.*", "site-packages", "nvidia", "cu13", "lib", lib)]
    if win:
        pats.append(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\cublas64_*.dll")
    else:
        pats += ["/usr/local/cuda*/lib64/libcublas.so*", "/usr/lib/x86_64-linux-gnu/libcublas.so*"]
    found = []
    for p in pats:
        for f in sorted(glob.glob(p)):
            if f not in found and "Lt" not in os.path.basename(f):
                found.append(f)
    return found


def _line_for(driver_version):
    """The newest CUDA line the driver runs (CUDA 13 needs driver 580+, 12 needs 525+, 11 needs 450+)."""
    major = driver_version // 1000
    if major >= 13:
        return "13"
    if major >= 12:
        return "12"
    return "11"


def status(venv=None):
    name, ver = _driver()
    if name:
        v = f"CUDA {ver // 1000}.{(ver % 1000) // 10}" if ver else "version unknown"
        print(f"driver   {name}  ({v} or older programs run)")
    else:
        print("driver   none found: install the NVIDIA driver (https://www.nvidia.com/drivers); without it programs "
              "run on the CPU")
    libs = _cublas_candidates(venv)
    if libs:
        print("cuBLAS   " + libs[0] + (f"  (+{len(libs) - 1} more)" if len(libs) > 1 else ""))
        print("         matrix products use it; APSI_GPU_MATH=tf32 / bf16 / f16 turns on Tensor Cores")
    else:
        print("cuBLAS   none found: Apsi's own kernels run (fine); `apsi cuda install` adds cuBLAS for the fastest "
              "matrix products")
    return 0


def _python():
    """A Python to run pip with (only for installing the wheels; the programs never need it)."""
    for c in (sys.executable, shutil.which("python3"), shutil.which("python"), shutil.which("py")):
        if c:
            return c
    return None


def install(line=None, venv=".venv", user=False):
    name, ver = _driver()
    if line is None:
        line = _line_for(ver) if ver else "12"
    line = str(line).split(".")[0].lstrip("cu")
    if line not in WHEELS:
        print(f"apsi cuda: CUDA {line} is not one of 11, 12, 13", file=sys.stderr)
        return 1
    py = _python()
    if py is None:
        print("apsi cuda: pip needs a Python to run; install Python, or install the CUDA Toolkit from "
              "https://developer.nvidia.com/cuda-downloads and set CUDA_PATH", file=sys.stderr)
        return 1
    if user:
        target = os.path.join(os.path.expanduser("~"), ".apsi", "cuda")
        os.makedirs(target, exist_ok=True)
        site = os.path.join(target, "Lib", "site-packages") if os.name == "nt" else \
            os.path.join(target, "lib", f"python3.{sys.version_info.minor}", "site-packages")
        cmd = [py, "-m", "pip", "install", "--upgrade", "--target", site, WHEELS[line]]
    else:
        if not os.path.isdir(venv):
            print(f"making {venv}")
            r = subprocess.run([py, "-m", "venv", venv])
            if r.returncode:
                return r.returncode
        vpy = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
        cmd = [vpy, "-m", "pip", "install", "--upgrade", WHEELS[line]]
    print("running: " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode == 0:
        print(f"cuBLAS for CUDA {line} is installed; programs started from this folder"
              + (" (or anywhere, for this user)" if user else "") + " find it by themselves.")
        status(None if user else venv)
    return r.returncode


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="apsi cuda", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", default="status", choices=["status", "install"])
    p.add_argument("--cuda", help="the CUDA line: 11, 12 or 13 (12.6 means 12); default: what the driver runs")
    p.add_argument("--venv", default=".venv", help="the virtual environment to install into (default .venv)")
    p.add_argument("--user", action="store_true", help="install into ~/.apsi/cuda for every program of this user")
    a = p.parse_args(argv)
    if a.action == "install":
        return install(a.cuda, a.venv, a.user)
    return status(a.venv)


if __name__ == "__main__":
    sys.exit(main())
