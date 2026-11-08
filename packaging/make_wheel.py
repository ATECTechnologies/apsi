"""Build the Apsi wheel with the standard library only (no setuptools/build needed, works offline).

    python packaging/make_wheel.py [OUT_DIR]        -> OUT_DIR/apsi-<version>-py3-none-any.whl  (default: dist/)
"""
import base64
import hashlib
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project():
    text = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    get = lambda key: re.search(rf'^{key}\s*=\s*"(.*)"', text, re.M).group(1)
    # TOML basic strings may contain escaped quotes: "pyreadline3>=3.4; sys_platform == \"win32\""
    block = re.search(r"^dependencies\s*=\s*\[(.*?)\]\s*$", text, re.M | re.S).group(1)
    deps = [re.sub(r'\\(.)', r'\1', d) for d in re.findall(r'"((?:[^"\\]|\\.)*)"', block)]
    return get("name"), get("version"), get("description"), get("requires-python"), deps


def _record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def build(out_dir):
    name, version, summary, requires, deps = _project()
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}", f"Summary: {summary}",
            f"Requires-Python: {requires}"] + [f"Requires-Dist: {d}" for d in deps] + \
           ["Provides-Extra: gpu", 'Requires-Dist: cupy-cuda12x; extra == "gpu"',
            "Provides-Extra: torch", 'Requires-Dist: torch; extra == "torch"',
            "Description-Content-Type: text/markdown", "", readme]
    info = f"{name}-{version}.dist-info"
    files = {}
    pkg = os.path.join(ROOT, "apsi")
    for f in sorted(os.listdir(pkg)):
        if f.endswith(".py") or f == "MANIFEST.json":
            with open(os.path.join(pkg, f), "rb") as fh:
                files[f"apsi/{f}"] = fh.read()
    files[f"{info}/METADATA"] = "\n".join(meta).encode("utf-8")
    files[f"{info}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: apsi make_wheel\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{info}/entry_points.txt"] = b"[console_scripts]\napsi = apsi.cli:main\n"
    files[f"{info}/top_level.txt"] = b"apsi\n"
    record = [_record_line(n, d) for n, d in files.items()] + [f"{info}/RECORD,,"]
    files[f"{info}/RECORD"] = ("\n".join(record) + "\n").encode()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}-{version}-py3-none-any.whl")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            zi = zipfile.ZipInfo(n, date_time=(2026, 11, 8, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, d)
    return path


if __name__ == "__main__":
    print(build(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dist"))))
