"""C and C++ source files next to an Apsi program: compiled with Apsi's own toolchain.

`cload("vecmath.c")` in `apsi run` builds the file into a shared library once (cached by the hash of the source
and the compiler) and loads it; `apsi build --native` compiles the same files into the program itself."""
import hashlib
import os
import shutil
import subprocess
import sys

C_SUFFIXES = (".c",)
CXX_SUFFIXES = (".cc", ".cpp", ".cxx", ".c++")
SOURCE_SUFFIXES = C_SUFFIXES + CXX_SUFFIXES


def is_source(path):
    return str(path).lower().endswith(SOURCE_SUFFIXES)


def cxx_of(cc):
    """The C++ driver next to a C compiler (clang -> clang++, gcc -> g++)."""
    d, name = os.path.split(cc)
    for a, b in (("clang", "clang++"), ("gcc", "g++"), ("cc", "c++")):
        if os.path.splitext(name)[0].endswith(a):
            cand = os.path.join(d, name.replace(a, b, 1)) if d else name.replace(a, b, 1)
            if os.path.isfile(cand) or shutil.which(cand):
                return cand
    return cc


def lib_suffix():
    return ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")


def compile_shared(src, cache_dir=None):
    """Build one C/C++ source file into a shared library (cached); returns the library's path."""
    from .toolchain import find
    src = os.path.abspath(src)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    cc = find()
    cxx = src.lower().endswith(CXX_SUFFIXES)
    tool = cxx_of(cc) if cxx else cc
    with open(src, "rb") as fh:
        key = hashlib.sha256(fh.read() + tool.encode() + b"v1").hexdigest()[:24]
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".apsi", "cache", "c")
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(src))[0]}-{key}{lib_suffix()}")
    if os.path.isfile(out):
        return out
    cmd = [tool, "-O3", "-shared", "-I", os.path.dirname(src), src, "-o", out + ".part"]
    if os.name != "nt":
        cmd[1:1] = ["-fPIC"]
    else:
        cmd += ["-static-libgcc"] + (["-static-libstdc++"] if cxx else [])
    cmd[1:1] = ["-std=c++17"] if cxx else ["-std=gnu11"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compiling {os.path.basename(src)} failed:\n{r.stderr[-3000:]}")
    os.replace(out + ".part", out)
    return out
