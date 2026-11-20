"""Check release files before they are published: every Apsi copy inside them has the shipped GPU kernels (PTX) of
this very source, the version of this source, and nothing that must never ship.

    python tools/check_release.py dist/linux/*.whl dist/linux/*.deb dist/linux/*.tar.gz installer/payload

Exit code 0: all good; 1: a problem (each one printed).
"""
import hashlib
import io
import os
import re
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = re.compile(r"(^|/)(\.env)$", re.I)


def _expected():
    with open(os.path.join(ROOT, "apsi", "gpu_train_kernels.py"), "rb") as f:
        src16 = hashlib.sha256(f.read()).hexdigest()[:16]
    names = sorted(f for f in os.listdir(os.path.join(ROOT, "apsi", "ptx")) if f.endswith(f"-{src16}.ptx.gz"))
    version = re.search(r'__version__ = "([^"]+)"', open(os.path.join(ROOT, "apsi", "__init__.py")).read()).group(1)
    return src16, names, version


def _members(path):
    """(name, read()) of every file inside a wheel, .deb, .tar.gz, .zip or folder."""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                p = os.path.join(root, f)
                yield os.path.relpath(p, path).replace(os.sep, "/"), (lambda p=p: open(p, "rb").read())
    elif path.endswith((".whl", ".zip", ".vsix")):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            yield n, (lambda n=n: z.read(n))
    elif path.endswith((".tar.gz", ".tgz")):
        t = tarfile.open(path)
        for m in t.getmembers():
            if m.isfile():
                yield m.name, (lambda m=m: t.extractfile(m).read())
    elif path.endswith(".deb"):
        data = open(path, "rb").read()
        pos = 8
        while pos < len(data):
            name = data[pos:pos + 16].decode().strip().rstrip("/")
            size = int(data[pos + 48:pos + 58].decode().strip())
            body = data[pos + 60:pos + 60 + size]
            pos += 60 + size + (size & 1)
            if name.startswith("data.tar"):
                if name.endswith(".zst"):                   # zstd: dpkg-deb unpacks it
                    import shutil
                    import subprocess
                    import tempfile
                    if not shutil.which("dpkg-deb"):
                        raise RuntimeError("a zstd .deb: unpack it with dpkg-deb -x and pass the folder")
                    tmp = tempfile.mkdtemp()
                    subprocess.check_call(["dpkg-deb", "-x", path, tmp])
                    yield from _members(tmp)
                    shutil.rmtree(tmp, ignore_errors=True)
                    continue
                t = tarfile.open(fileobj=io.BytesIO(body))
                for m in t.getmembers():
                    if m.isfile():
                        yield m.name, (lambda m=m, t=t: t.extractfile(m).read())
    else:
        raise RuntimeError(f"{path}: not a wheel, .deb, .tar.gz, .zip or folder")


def check(path, src16, names, version):
    problems, apsi_dirs, ptx, init_version = [], set(), {}, {}
    for name, read in _members(path):
        if FORBIDDEN.search(name):
            problems.append(f"{name}: must not ship")
        m = re.search(r"(.*?)(?:^|/)apsi/ptx/([^/]+\.ptx\.gz)$", name)
        if m:
            ptx.setdefault(m.group(1), set()).add(m.group(2))
        m = re.search(r"(.*?)(?:^|/)apsi/gpu_train_kernels\.py$", name)
        if m:
            apsi_dirs.add(m.group(1))
            if hashlib.sha256(read()).hexdigest()[:16] != src16:
                problems.append(f"{name}: not this source's kernels")
        m = re.search(r"(.*?)(?:^|/)apsi/__init__\.py$", name)
        if m:
            v = re.search(rb'__version__ = "([^"]+)"', read())
            init_version[m.group(1)] = v.group(1).decode() if v else "?"
    for d in sorted(apsi_dirs):
        have = ptx.get(d, set())
        missing = [n for n in names if n not in have]
        if missing:
            problems.append(f"{d or '.'}/apsi/ptx: {len(missing)} of {len(names)} kernel files missing")
        stale = [n for n in have if src16 not in n]
        if stale:
            problems.append(f"{d or '.'}/apsi/ptx: {len(stale)} files of another source")
        if init_version.get(d) not in (None, version):
            problems.append(f"{d or '.'}/apsi: version {init_version[d]}, the source is {version}")
    if not apsi_dirs and not path.endswith(".vsix"):
        problems.append("no Apsi package inside")
    return problems, len(apsi_dirs)


def main(paths):
    src16, names, version = _expected()
    if not names:
        print(f"apsi/ptx has no kernels for this source ({src16}): run python tools/build_ptx.py")
        return 1
    bad = 0
    for p in paths:
        try:
            problems, n = check(p, src16, names, version)
        except Exception as e:                                  # noqa: BLE001
            problems, n = [str(e)], 0
        print(f"{'ok  ' if not problems else 'FAIL'} {p}" + (f"  ({n} Apsi copies, {len(names)} kernel files each)"
                                                             if not problems else ""))
        for q in problems:
            print(f"     {q}")
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
