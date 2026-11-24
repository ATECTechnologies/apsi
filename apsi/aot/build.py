"""Build an Apsi program into a native executable: Apsi -> C (cgen) -> machine code (clang, or gcc)."""
import os
import shutil
import subprocess
import sys
import tempfile

from .cgen import compile_c, NativeError

RT = os.path.join(os.path.dirname(__file__), "rt")


def find_cc():
    for cc in (os.environ.get("APSI_CC"), "clang", "gcc", "cc"):
        if cc and shutil.which(cc):
            return cc
    raise NativeError("no C compiler found (clang or gcc); set APSI_CC")


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


def build(path, out=None, opt="-O2", keep_c=False, target=None):
    src = open(path, encoding="utf-8").read()
    check(src, path)
    c = compile_c(src, path)
    if out is None:
        out = os.path.splitext(path)[0] + (".exe" if os.name == "nt" or (target or "").endswith("windows-msvc")
                                           else "")
    work = tempfile.mkdtemp(prefix="apsi-native-")
    cfile = os.path.join(work, "program.c")
    with open(cfile, "w", encoding="utf-8") as f:
        f.write(c)
    if keep_c:
        shutil.copy(cfile, os.path.splitext(out)[0] + ".c")
    cc = find_cc()
    cmd = [cc, opt, "-std=gnu11", "-w", "-I", RT, cfile, os.path.join(RT, "apsi_rt.c"),
           os.path.join(RT, "apsi_tensor.c"), os.path.join(RT, "apsi_gpu.c"), "-o", out]
    if os.environ.get("APSI_GPU_EMU_BUILD"):        # tests: the card emulated on host threads
        cmd[-2:-2] = [os.path.join(RT, "kernels.c")]
        cmd[1:1] = ["-DAPSI_GPU_EMU_BUILD", "-DAPSI_EMU", "-DAPSI_CHECK_DEV"]
    if target:
        cmd[1:1] = ["-target", target]
    if not (target or "").endswith("windows-msvc") and os.name != "nt":
        cmd += ["-lm", "-lpthread", "-ldl"]
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
    p.add_argument("-O", dest="opt", default="2")
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
