"""Build the self-contained Apsi payload for one platform.

The payload is what every installer ships: a relocatable CPython 3.12, numpy, the Apsi
runtime (with its signed release manifest), examples, docs, icons and the VS Code extension.

    python packaging/build_payload.py --platform windows --out build/payload-windows
    python packaging/build_payload.py --platform linux   --out build/payload-linux

Runs on Windows or Linux. Downloads are cached in packaging/.cache.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "packaging", ".cache")
PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/20250612/"
RUNTIMES = {
    "windows": ("cpython-3.12.11+20250612-x86_64-pc-windows-msvc-install_only.tar.gz", "win_amd64",
                os.path.join("python", "Lib", "site-packages")),
    "linux": ("cpython-3.12.11+20250612-x86_64-unknown-linux-gnu-install_only.tar.gz", "manylinux2014_x86_64",
              os.path.join("python", "lib", "python3.12", "site-packages")),
}
JUNK = ["Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "tcl", "lib/python3.12/test",
        "lib/python3.12/idlelib", "lib/python3.12/tkinter", "lib/python3.12/turtledemo", "include", "share",
        "lib/Tix8.4.3", "lib/itcl4.2.4", "lib/thread2.8.9", "lib/tcl8.6", "lib/tk8.6", "lib/tcl8",
        "lib/pkgconfig", "lib/python3.12/config-3.12-x86_64-linux-gnu", "lib/python3.12/ensurepip"]


def strip_binaries(rt):
    """Release builds of the Linux runtime ship with debug symbols (~300 MB); strip them."""
    strip = shutil.which("strip")
    if not strip:
        log("strip not found: keeping debug symbols (larger payload)")
        return
    for root, _, files in os.walk(rt):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                continue
            if f.endswith(".so") or ".so." in f:          # never the (static) python executable
                with open(fp, "rb") as fh:
                    if fh.read(4) != b"\x7fELF":
                        continue
                subprocess.call([strip, "--strip-debug", fp], stderr=subprocess.DEVNULL)


def log(msg):
    print(f"[payload] {msg}", flush=True)


def version():
    sys.path.insert(0, ROOT)
    import apsi
    return apsi.__version__


def build_wheel():
    sys.path.insert(0, ROOT)
    from apsi.security import write_manifest
    write_manifest()
    out = os.path.join(ROOT, "build", "wheel")
    shutil.rmtree(out, ignore_errors=True)
    # standard library only: no setuptools, no network, and no build/ or *.egg-info left in the source folder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_wheel
    return make_wheel.build(out)


def fetch(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        log(f"downloading {name}")
        tmp = path + ".part"
        urllib.request.urlretrieve(PBS + name, tmp)
        os.replace(tmp, path)
    return path


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", choices=sorted(RUNTIMES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-archive", help="use this python-build-standalone archive instead of downloading")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    archive, pip_platform, site = RUNTIMES[a.platform]
    v = version()
    log(f"Apsi {v} for {a.platform} -> {out}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    whl = build_wheel()
    log(f"wheel {os.path.basename(whl)}")

    rt = os.path.join(out, "runtime")
    os.makedirs(rt)
    with tarfile.open(a.runtime_archive or fetch(archive)) as tf:
        try:
            tf.extractall(rt, filter="tar")
        except TypeError:                       # Python < 3.12
            tf.extractall(rt)
    for j in JUNK:
        shutil.rmtree(os.path.join(rt, "python", j), ignore_errors=True)
    for root, _, files in os.walk(rt):
        for f in files:
            if f.endswith(".pdb") or f.endswith(".a"):
                os.remove(os.path.join(root, f))
    if a.platform == "linux":
        # the python executable is statically linked; the shared libpython is only for embedding
        for f in os.listdir(os.path.join(rt, "python", "lib")):
            if f.startswith("libpython3"):
                os.remove(os.path.join(rt, "python", "lib", f))
        strip_binaries(rt)
    log("installing numpy + llvmlite (native compiler) + apsi into the runtime")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-compile", "--upgrade",
                           "--target", os.path.join(rt, site), "--platform", pip_platform,
                           "--python-version", "3.12", "--implementation", "cp", "--only-binary=:all:",
                           "numpy>=1.26", "llvmlite>=0.44", whl] + (["pyreadline3>=3.4"] if a.platform == "windows" else []))

    for d in ("examples", "docs"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(out, d),
                        ignore=shutil.ignore_patterns("*.so", "*.dll", "__pycache__", "*_store", "_tmp_*"))
    for f in ("README.md", "CHANGELOG.md"):
        shutil.copy(os.path.join(ROOT, f), out)
    if os.path.exists(os.path.join(ROOT, "LICENSE")):
        shutil.copy(os.path.join(ROOT, "LICENSE"), out)
    shutil.copytree(os.path.join(ROOT, "assets", "brand"), os.path.join(out, "brand"),
                    ignore=shutil.ignore_patterns("logo-source.png", "*.bmp", "*banner*"))
    vsix = os.path.join(ROOT, "dist", f"apsi-language-{v}.vsix")
    os.makedirs(os.path.join(out, "vscode"), exist_ok=True)
    if os.path.exists(vsix):
        shutil.copy(vsix, os.path.join(out, "vscode"))
    # every editor integration (VS Code family, Sublime Text, Vim/Neovim, Notepad++, JetBrains, Helix, Zed, Emacs)
    shutil.copytree(os.path.join(ROOT, "editors"), os.path.join(out, "editors"),
                    ignore=shutil.ignore_patterns("node_modules", "*.vsix", ".vscode-test", "__pycache__"))

    info = {
        "name": "Apsi", "version": v, "platform": a.platform,
        "python": "3.12.11",
        "components": {
            "runtime": dir_size(rt), "examples": dir_size(os.path.join(out, "examples")),
            "docs": dir_size(os.path.join(out, "docs")), "editors": dir_size(os.path.join(out, "vscode")) + dir_size(os.path.join(out, "editors")),
        },
    }
    # per-file sha256 list: the installer re-hashes every file before copying it, so a payload
    # that was modified after the build (or corrupted while downloading) is refused
    files = {}
    for root, _, names in sorted(os.walk(out)):
        for f in sorted(names):
            full = os.path.join(root, f)
            if os.path.islink(full):
                continue
            rel = os.path.relpath(full, out).replace(os.sep, "/")
            if rel in ("payload.json", "payload-files.json"):
                continue
            h = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = h.hexdigest()
    blob = json.dumps({"algorithm": "sha256", "files": files}, sort_keys=True, separators=(",", ":")).encode()
    with open(os.path.join(out, "payload-files.json"), "wb") as fh:
        fh.write(blob)
    info["files"] = len(files)
    info["sha256"] = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(out, "payload.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    total = sum(info["components"].values())
    log(f"done: {total / 1e6:.1f} MB ({', '.join(f'{k} {s / 1e6:.1f} MB' for k, s in info['components'].items())})")


if __name__ == "__main__":
    main()
