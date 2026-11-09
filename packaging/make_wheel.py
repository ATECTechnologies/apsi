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


def _toml_strings(chunk):
    """Decode every TOML string literal in `chunk` ("basic" with escapes, or 'literal')."""
    import json
    out = []
    for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"|\'([^\'\n]*)\'', chunk):
        out.append(json.loads('"' + m.group(1) + '"') if m.group(1) is not None else m.group(2))
    return out


def _project():
    path = os.path.join(ROOT, "pyproject.toml")
    try:                                         # Python 3.11+: the real TOML parser
        import tomllib
        with open(path, "rb") as fh:
            p = tomllib.load(fh)["project"]
        return p["name"], p["version"], p["description"], p["requires-python"], list(p["dependencies"])
    except ImportError:
        pass
    text = open(path, encoding="utf-8").read()
    get = lambda key: _toml_strings(re.search(rf"^{key}\s*=\s*(.*)$", text, re.M).group(1))[0]
    deps = _toml_strings(re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.M | re.S).group(1))
    return get("name"), get("version"), get("description"), get("requires-python"), deps


def _check_requirement(req):
    """Refuse to write metadata pip would reject (invalid metadata once broke a Windows payload)."""
    if "\\" in req or req.count('"') % 2 or req.count("'") % 2:
        raise SystemExit(f"make_wheel: malformed requirement in pyproject.toml: {req!r}")
    return req


def _record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def build(out_dir):
    name, version, summary, requires, deps = _project()
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}", f"Summary: {summary}",
            f"Requires-Python: {requires}", "Author: ATEC", "License: Apsi Non-Commercial License 1.0"] + [f"Requires-Dist: {_check_requirement(d)}" for d in deps] + \
           ["Provides-Extra: gpu", 'Requires-Dist: cupy-cuda12x; extra == "gpu"',
            "Provides-Extra: torch", 'Requires-Dist: torch; extra == "torch"',
            "Description-Content-Type: text/markdown", "", readme]
    info = f"{name}-{version}.dist-info"
    files = {}
    pkg = os.path.join(ROOT, "apsi")
    for root, dirs, names in os.walk(pkg):                 # subpackages too (apsi/native)
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        rel = os.path.relpath(root, pkg)
        arc = "apsi" if rel == "." else "apsi/" + rel.replace(os.sep, "/")
        for f in sorted(names):
            if f.endswith(".py") or (f == "MANIFEST.json" and rel == "."):
                with open(os.path.join(root, f), "rb") as fh:
                    files[f"{arc}/{f}"] = fh.read()
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
