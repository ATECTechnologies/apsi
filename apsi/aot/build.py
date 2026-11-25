"""Build an Apsi program into a native executable: Apsi -> C (cgen) -> machine code (clang, or gcc)."""
import os
import shutil
import subprocess
import sys
import tempfile

from .cgen import compile_c, NativeError

RT = os.path.join(os.path.dirname(__file__), "rt")


def find_cc():
    from .toolchain import find, ToolchainError
    try:
        return find()
    except ToolchainError as e:
        raise NativeError(str(e))


class CheckError(NativeError):
    def __init__(self, err, src, path):
        super().__init__(str(err))
        self.err, self.src, self.path = err, src, path


def check(src, path):
    """The checker runs first, as for apsi run: a built program has no check errors."""
    from ..checker import Checker
    from ..errors import ApsiError
    from ..interpreter import Interpreter, CheckFailed
    from ..parser import parse
    try:
        prog = parse(src)
        errs = Checker(Interpreter(), os.path.dirname(os.path.abspath(path))).check(prog)
    except ApsiError as e:
        raise CheckError(e, src, path)
    if errs:
        raise CheckError(CheckFailed(errs), src, path)


def _cache_root():
    base = os.environ.get("APSI_CACHE") or os.path.join(os.path.expanduser("~"), ".apsi", "cache")
    return os.path.join(base, "rt")


def runtime_objects(cc, flags, names, target):
    """The runtime compiled once per compiler, flags and runtime source, then kept: a program's build compiles only
    the program. The objects are built side by side in parallel and moved into place whole (no half-written cache)."""
    import hashlib
    h = hashlib.sha256()
    try:
        st = os.stat(shutil.which(cc) or cc)
        h.update(f"{cc}|{st.st_size}|{st.st_mtime_ns}".encode())
    except OSError:
        h.update(cc.encode())
    h.update(repr((flags, names, target)).encode())
    for f in sorted(os.listdir(RT)):
        if f.endswith((".c", ".h")):
            with open(os.path.join(RT, f), "rb") as fh:
                h.update(f.encode() + b"\0" + fh.read())
    key = h.hexdigest()[:24]
    root = _cache_root()
    final = os.path.join(root, key)
    objs = [os.path.join(final, n[:-2] + ".o") for n in names]
    if all(os.path.isfile(o) for o in objs):
        return objs
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=key + "-", dir=root)
    procs = []
    for n in names:
        o = os.path.join(tmp, n[:-2] + ".o")
        procs.append((n, subprocess.Popen([cc] + flags + ["-c", os.path.join(RT, n), "-o", o],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    errs = []
    for n, pr in procs:
        _, err = pr.communicate()
        if pr.returncode != 0:
            errs.append(f"{n}:\n{err[-3000:]}")
    if errs:
        shutil.rmtree(tmp, ignore_errors=True)
        raise NativeError("compiling the runtime failed:\n" + "\n".join(errs))
    try:
        os.replace(tmp, final)
    except OSError:                      # another build put it there first
        shutil.rmtree(tmp, ignore_errors=True)
    if not all(os.path.isfile(o) for o in objs):
        raise NativeError("the runtime cache in " + root + " is not writable; set APSI_CACHE to another folder")
    return objs


def build(path, out=None, opt="-O3", keep_c=False, target=None):
    src = open(path, encoding="utf-8").read()
    check(src, path)
    extra = []
    c = compile_c(src, path, extra)
    if out is None:
        out = os.path.splitext(path)[0] + (".exe" if os.name == "nt" or (target or "").endswith("windows-msvc")
                                           else "")
    if (os.name == "nt" or "windows" in (target or "") or "mingw" in (target or "")) and not out.lower().endswith(".exe"):
        out += ".exe"                       # Windows runs programs by their .exe name
    work = tempfile.mkdtemp(prefix="apsi-native-")
    cfile = os.path.join(work, "program.c")
    with open(cfile, "w", encoding="utf-8") as f:
        f.write(c)
    if keep_c:
        shutil.copy(cfile, os.path.splitext(out)[0] + ".c")
    cc = find_cc()
    rt_src = ["apsi_rt.c", "apsi_tensor.c", "apsi_gpu.c", "apsi_gemm.c"]
    cmd = [cc, opt, "-std=gnu11", "-w", "-D_CRT_SECURE_NO_WARNINGS", "-I", RT, cfile, "-o", out]
    if os.environ.get("APSI_GPU_EMU_BUILD"):        # tests: the card emulated on host threads
        rt_src.append("kernels.c")
        cmd[1:1] = ["-DAPSI_GPU_EMU_BUILD", "-DAPSI_EMU", "-DAPSI_CHECK_DEV"]
    if target:
        cmd[1:1] = ["-target", target]
    elif not os.environ.get("APSI_PORTABLE"):
        cmd[1:1] = ["-march=native", "-mprefer-vector-width=512"]   # built where it runs: every vector unit it has
    cmd[1:1] = ["-fno-math-errno"]
    flags = [x for x in cmd[1:] if x not in (cfile, "-o", out)]
    objs_rt = runtime_objects(cc, flags, rt_src, target)
    windows = os.name == "nt" or "windows" in (target or "") or "mingw" in (target or "")
    if not windows:
        cmd += ["-lm", "-lpthread", "-ldl"]
    elif "msvc" not in (target or "") and not cc.lower().endswith("cl.exe"):
        cmd += ["-static", "-lm", "-lws2_32", "-lbcrypt"]               # MinGW: one file with nothing to install beside it
    if extra:                                   # C / C++ files the program cload()s: built into it, symbols exported
        from .csource import CXX_SUFFIXES, cxx_of
        objs, any_cxx = [], False
        for i, f in enumerate(extra):
            if f.lower().endswith(CXX_SUFFIXES):
                any_cxx = True
                o = os.path.join(work, f"cxx{i}.o")
                rc = subprocess.run([cxx_of(cc), opt, "-std=c++17", "-I", os.path.dirname(f), "-c", f, "-o", o]
                                    + (["-fPIC"] if os.name != "nt" else []), capture_output=True, text=True)
                if rc.returncode != 0:
                    shutil.rmtree(work, ignore_errors=True)
                    raise NativeError(f"compiling {os.path.basename(f)} failed:\n" + rc.stderr[-3000:])
                objs.append(o)
            else:
                cmd.insert(cmd.index("-o"), f)
        if any_cxx:
            cmd[0] = cxx_of(cc)
            first = cmd.index(cfile)
            cmd[first:first] = ["-x", "c"]
            cmd.insert(cmd.index("-o"), "-x")
            cmd.insert(cmd.index("-o"), "none")
            cmd[cmd.index("-std=gnu11")] = "-std=gnu11"
        for o in objs:
            cmd.insert(cmd.index("-o"), o)
        windows_t = os.name == "nt" or "windows" in (target or "") or "mingw" in (target or "")
        cmd += ["-Wl,--export-all-symbols"] if windows_t else (["-rdynamic"] if sys.platform != "darwin" else [])
    at = cmd.index("-o")                       # the runtime's objects: after any `-x c` (read as objects, not as C)
    cmd[at:at] = (["-x", "none"] if "-x" in cmd[:at] and cmd[:at][::-1][cmd[:at][::-1].index("-x") - 1] != "none"
                  else []) + objs_rt
    r = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(work, ignore_errors=True)
    if r.returncode != 0:
        raise NativeError("C compiler failed:\n" + r.stderr[-4000:])
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="apsi build")
    p.add_argument("file")
    p.add_argument("-o", "--out")
    p.add_argument("--emit-c", action="store_true", help="also write the generated C next to the output")
    p.add_argument("--target", help="a clang target triple, e.g. x86_64-pc-windows-msvc")
    p.add_argument("-O", dest="opt", default="3")
    a = p.parse_args(argv)
    try:
        out = build(a.file, a.out, "-O" + a.opt, a.emit_c, a.target)
    except CheckError as e:
        from ..cli import _report
        _report(e.err, e.src, e.path)
        return 1
    except NativeError as e:
        print(f"apsi build: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
