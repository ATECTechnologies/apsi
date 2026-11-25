"""The C toolchain the native backend builds with.

On Windows Apsi carries its own: the installer puts a trimmed llvm-mingw (clang + lld + the MinGW-w64 headers and
libraries, x86-64 only) in <install>/toolchain/llvm-mingw. A source checkout gets the same toolchain downloaded once,
the first time it is needed, into %LOCALAPPDATA%/apsi/toolchain. Elsewhere the system's clang or gcc is used.

    apsi toolchain            where the compiler is (and fetch it when missing)"""
import os
import shutil
import sys
import urllib.request
import zipfile

LLVM_MINGW = "20260519"
ZIP = f"llvm-mingw-{LLVM_MINGW}-ucrt-x86_64.zip"
URL = f"https://github.com/mstorsjo/llvm-mingw/releases/download/{LLVM_MINGW}/{ZIP}"

# what a build needs from the llvm-mingw tree (everything else: other targets, lldb, python, docs, is left out)
KEEP_BIN = ("clang", "clang-", "ld.lld", "lld", "x86_64-w64-mingw32-", "libc++", "libunwind", "libLLVM", "libclang",
            "llvm-ar", "llvm-ranlib")
KEEP_DIRS = ("include", "x86_64-w64-mingw32", "lib/clang", "generic-w64-mingw32")


class ToolchainError(Exception):
    pass


def _apsi_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bundled_dirs():
    """<install>/toolchain/llvm-mingw, looked for above the apsi package (the installer's layout)."""
    d = _apsi_root()
    for _ in range(7):
        yield os.path.join(d, "toolchain", "llvm-mingw")
        d = os.path.dirname(d)
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    yield os.path.join(local, "apsi", "toolchain", "llvm-mingw")


def _clang_in(d):
    p = os.path.join(d, "bin", "clang.exe" if os.name == "nt" else "clang")
    return p if os.path.isfile(p) else None


def trim(tree):
    """Remove from an unpacked llvm-mingw what an x86-64 Windows build never uses."""
    for name in os.listdir(tree):
        full = os.path.join(tree, name)
        if name == "bin":
            for f in os.listdir(full):
                if f.endswith(".cfg"):
                    continue                      # clang's configuration (target, runtime libraries, lld)
                if not f.startswith(KEEP_BIN) or f.startswith(("clang-tidy", "clang-format", "clangd", "clang-scan", "lldb")):
                    p = os.path.join(full, f)
                    if os.path.isfile(p) and not f.endswith(".dll"):
                        os.remove(p)
        elif name == "lib":
            for f in os.listdir(full):
                if f != "clang":
                    p = os.path.join(full, f)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        elif name not in ("include", "x86_64-w64-mingw32", "generic-w64-mingw32", "LICENSE.TXT"):
            shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)


def fetch(dest_parent, archive=None, quiet=False):
    """Download (or take archive) llvm-mingw, unpack it trimmed into dest_parent/llvm-mingw."""
    os.makedirs(dest_parent, exist_ok=True)
    if archive is None:
        archive = os.path.join(dest_parent, ZIP)
        if not os.path.isfile(archive):
            if not quiet:
                print(f"apsi: getting the C toolchain once ({URL}, about 150 MB)...", file=sys.stderr, flush=True)
            tmp = archive + ".part"
            try:
                urllib.request.urlretrieve(URL, tmp)
            except OSError as e:
                raise ToolchainError(f"could not download the toolchain from {URL}: {e}")
            os.replace(tmp, archive)
    with zipfile.ZipFile(archive) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(dest_parent)
    dest = os.path.join(dest_parent, "llvm-mingw")
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(os.path.join(dest_parent, top), dest)
    trim(dest)
    if os.path.isfile(os.path.join(dest_parent, ZIP)):
        os.remove(os.path.join(dest_parent, ZIP))
    if not _clang_in(dest):
        raise ToolchainError(f"the toolchain in {dest} has no clang")
    return dest


def find(auto_fetch=True):
    """The C compiler to build with: APSI_CC, Apsi's own toolchain, then the system's."""
    env = os.environ.get("APSI_CC")
    if env:
        return shutil.which(env) or env
    for d in bundled_dirs():
        c = _clang_in(d)
        if c:
            return c
    for cc in ("clang", "gcc", "cc"):
        p = shutil.which(cc)
        if p:
            return p
    if os.name == "nt":
        if not auto_fetch:
            raise ToolchainError("Apsi's C toolchain is missing; run: apsi toolchain")
        local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return _clang_in(fetch(os.path.join(local, "apsi", "toolchain")))
    raise ToolchainError("no C compiler: install clang or gcc (Debian/Ubuntu: sudo apt install clang; "
                         "Fedora: sudo dnf install clang; macOS: xcode-select --install)")


def main(argv=None):
    try:
        cc = find()
    except ToolchainError as e:
        print(f"apsi toolchain: {e}", file=sys.stderr)
        return 1
    print(cc)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def link_flags(cc, target=None):
    """What llvm-mingw's configuration files say, given on the command line: the target, compiler-rt, libunwind,
    libc++ and lld. Without them clang for Windows looks for libgcc and libstdc++, which llvm-mingw does not have
    ('unable to find library -lgcc', 'cmath' not found), and for a triple-named `ld` that Windows cannot start."""
    name = os.path.basename(cc).lower()
    windows = os.name == "nt" or "windows" in (target or "") or "mingw" in (target or "")
    if not (windows and "clang" in name) or "msvc" in (target or ""):
        return []
    flags = ["-rtlib=compiler-rt", "-unwindlib=libunwind", "-stdlib=libc++", "-fuse-ld=lld",
             "-Wno-unused-command-line-argument"]
    if not target:
        flags.insert(0, "--target=x86_64-w64-mingw32")
    return flags
