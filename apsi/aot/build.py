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


def build(path, out=None, opt="-O2", keep_c=False, target=None):
    src = open(path, encoding="utf-8").read()
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
    cmd = [cc, opt, "-std=gnu11", "-w", "-I", RT, cfile, os.path.join(RT, "apsi_rt.c"), "-o", out]
    if target:
        cmd[1:1] = ["-target", target]
    if not (target or "").endswith("windows-msvc") and os.name != "nt":
        cmd.append("-lm")
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
    except NativeError as e:
        print(f"apsi build: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
