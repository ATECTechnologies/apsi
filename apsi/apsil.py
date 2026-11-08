"""Apsi libraries: the `.apsil` package format, the registry client and `apsi install / pack / publish`.

    apsi install NAME                 newest version from the registry (also NAME@1.2.3, NAME@^1.2, "NAME>=1,<2")
    apsi install                      everything listed in apsi.toml [dependencies], then write apsi.lock
    apsi install ./dist/x-1.0.0.apsil a local file
    apsi uninstall NAME
    apsi pack                         check the library and build dist/NAME-VERSION.apsil
    apsi publish [FILE] --token T     upload to the registry (token from the website: Settings → Developer)
    apsi search WORDS / apsi lib info NAME / apsi lib list / apsi lib login / apsi lib registry URL

A library is used like a standard module: `use NAME` (dashes become underscores).

The .apsil format (version 1) is a ZIP archive written deterministically (sorted entries, fixed timestamps), so
the same sources always give the same bytes:

    apsil.json        manifest: name, version, summary, authors, license, entry, dependencies, apsi version range,
                      the SHA-256 and size of every other file
    src/…             the library's .apsi files (entry = the module `use NAME` loads)
    README.md, LICENSE, examples/, data/ …   anything else listed in [package] include

Installing checks the registry's SHA-256 of the whole file, every file's SHA-256 from the manifest, refuses paths
that leave the package folder, and runs Apsi Guard on the sources before the library can be used.
"""
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

FORMAT_VERSION = 1
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
CHUNK = 1 << 20
DEFAULT_REGISTRY = "https://apsi.dev"
# Uploads larger than this are sent in resumable pieces (Cloudflare and most proxies limit a request to 100 MB).
SINGLE_UPLOAD_MAX = int(os.environ.get("APSI_UPLOAD_SINGLE_MAX", 64 << 20))


class PackageError(Exception):
    pass


# =============================================================================== versions
def parse_version(v):
    m = VERSION_RE.match(str(v).strip())
    if not m:
        raise PackageError(f"'{v}' is not a version like 1.2.3 (semantic versioning)")
    pre = m.group(4)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), pre)


def _key(v):
    a, b, c, pre = parse_version(v)
    pk = (1,) if pre is None else (0,) + tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split("."))
    return (a, b, c, pk)


def _parse_spec(spec):
    """'^1.2', '~1.2.3', '>=1,<2', '==1.0.0', '1.2.3', '*' -> list of (op, version-tuple-key, prerelease)."""
    spec = (spec or "*").strip()
    if spec in ("*", "", "latest"):
        return []
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        m = re.match(r"^(\^|~|>=|<=|==|!=|>|<|=)?\s*v?(\d+)(?:\.(\d+|\*|x))?(?:\.(\d+|\*|x))?(-[0-9A-Za-z.-]+)?$", part)
        if not m:
            raise PackageError(f"cannot read the version requirement '{part}'")
        op = m.group(1) or "="
        maj = int(m.group(2))
        mi = None if m.group(3) in (None, "*", "x") else int(m.group(3))
        pa = None if m.group(4) in (None, "*", "x") else int(m.group(4))
        pre = m.group(5)[1:] if m.group(5) else None
        base = f"{maj}.{mi or 0}.{pa or 0}" + (f"-{pre}" if pre else "")
        if op == "^":
            upper = f"{maj + 1}.0.0" if maj > 0 else (f"0.{(mi or 0) + 1}.0" if mi else "1.0.0")
            if maj == 0 and mi == 0 and pa is not None:
                upper = f"0.0.{pa + 1}"
            out += [(">=", base), ("<", upper + "-0")]
        elif op == "~":
            upper = f"{maj}.{(mi or 0) + 1}.0" if mi is not None else f"{maj + 1}.0.0"
            out += [(">=", base), ("<", upper + "-0")]
        elif op in ("=", "==") and (mi is None or pa is None):
            upper = f"{maj + 1}.0.0" if mi is None else f"{maj}.{mi + 1}.0"
            out += [(">=", base), ("<", upper + "-0")]
        else:
            out.append(("==" if op == "=" else op, base))
    return out


def satisfies(version, spec):
    k = _key(version)
    cons = _parse_spec(spec)
    if parse_version(version)[3] is not None and not any(parse_version(v)[3] for _, v in cons):
        return False                       # pre-releases only when asked for
    for op, v in cons:
        o = _key(v)
        if not {"==": k == o, "!=": k != o, ">=": k >= o, "<=": k <= o, ">": k > o, "<": k < o}[op]:
            return False
    return True


def best_version(versions, spec="*"):
    ok = [v for v in versions if satisfies(v, spec)]
    return max(ok, key=_key) if ok else None


def split_requirement(req):
    """'name@^1.2' / 'name>=1,<2' / 'name' -> (name, spec)."""
    req = req.strip()
    m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*)\s*(?:@\s*(.+)|((?:\^|~|>=|<=|==|!=|>|<|=).+))?$", req)
    if not m:
        raise PackageError(f"cannot read the requirement '{req}'")
    return m.group(1).lower(), (m.group(2) or m.group(3) or "*").strip()


def module_name(pkg):
    return pkg.replace("-", "_")


# =============================================================================== configuration and storage
def _config_dir():
    from .security import config_dir
    return config_dir()


def _cred_path():
    return os.path.join(_config_dir(), "registry.json")


def _load_cred():
    try:
        with open(_cred_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_cred(d):
    os.makedirs(_config_dir(), exist_ok=True)
    p = _cred_path()
    tmp = p + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, p)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def registry_url(explicit=None):
    url = (explicit or os.environ.get("APSI_REGISTRY") or _load_cred().get("registry") or DEFAULT_REGISTRY).rstrip("/")
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("https", "http") or not p.netloc:
        raise PackageError(f"'{url}' is not a web address")
    if p.scheme == "http" and p.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise PackageError("the registry must use https:// (http:// is only allowed for localhost)")
    return url


def stored_token(registry):
    return os.environ.get("APSI_TOKEN") or _load_cred().get("tokens", {}).get(registry)


def package_roots(project_dir=None, include_global=True):
    roots = []
    if project_dir:
        roots.append(os.path.join(project_dir, "apsi_packages"))
    if include_global:
        roots.append(os.path.join(_config_dir(), "packages"))
    return roots


def _project_dir(start=None):
    from .project import find_project
    return find_project(start)


def installed(project_dir=None, include_global=True):
    """{name: (version, folder)} of the active versions, project packages first."""
    out = {}
    for root in reversed(package_roots(project_dir, include_global or not project_dir)):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            cur = os.path.join(root, name, "current")
            try:
                with open(cur, encoding="utf-8") as fh:
                    v = fh.read().strip()
            except OSError:
                continue
            folder = os.path.join(root, name, v)
            if os.path.isfile(os.path.join(folder, "apsil.json")):
                out[name] = (v, folder)
    return out


def resolve_use(name, start_dir=None):
    """Entry file for `use NAME` when NAME is an installed library, else None."""
    proj = _project_dir(start_dir)
    want = name.replace("_", "-")
    for cand in (name, want):
        hit = installed(proj).get(cand)
        if hit:
            folder = hit[1]
            with open(os.path.join(folder, "apsil.json"), encoding="utf-8") as fh:
                man = json.load(fh)
            entry = os.path.normpath(os.path.join(folder, man["entry"]))
            if not entry.startswith(os.path.normpath(folder) + os.sep):
                raise PackageError(f"library {cand}: its entry file is outside the package")
            return entry
    return None


# =============================================================================== building (.apsil)
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_config(root):
    from .project import read_toml
    cfg = read_toml(os.path.join(root, "apsi.toml"))
    pkg = dict(cfg.get("package", {}))
    proj = cfg.get("project", {})
    for k in ("name", "version"):
        pkg.setdefault(k, proj.get(k))
    pkg.setdefault("summary", proj.get("description", ""))
    deps = dict(cfg.get("dependencies", {}))
    return pkg, deps, cfg


def _collect(root, pkg):
    include = pkg.get("include") or ["src/**", "README*", "LICENSE*", "CHANGELOG*", "examples/**"]
    exclude = list(pkg.get("exclude") or []) + ["dist/**", "apsi_packages/**", ".git/**", "**/__pycache__/**",
                                                "**/.DS_Store", "build/**", "*.apsil"]
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") or d == ".")
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if os.path.islink(full):
                continue
            if any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.rstrip("/*") + "/*") for p in exclude):
                continue
            if any(fnmatch.fnmatch(rel, p) or (p.endswith("/**") and rel.startswith(p[:-3] + "/")) for p in include):
                files.append(rel)
    return sorted(set(files))


def pack(root=".", out_dir=None, check=True, quiet=False):
    """Build dist/NAME-VERSION.apsil from the library in `root`. Returns the file path."""
    root = os.path.abspath(root)
    if not os.path.isfile(os.path.join(root, "apsi.toml")):
        raise PackageError(f"{root} has no apsi.toml (create a library with: apsi new NAME --template lib)")
    pkg, deps, _ = _package_config(root)
    name, version = str(pkg.get("name") or ""), str(pkg.get("version") or "")
    if not NAME_RE.match(name):
        raise PackageError(f"package name '{name}' must be 2–64 characters: lowercase letters, digits, '-' or '_', "
                           f"starting with a letter")
    parse_version(version)
    entry = str(pkg.get("entry") or f"src/{module_name(name)}.apsi")
    if not os.path.isfile(os.path.join(root, entry)):
        alt = next((a for a in ("src/lib.apsi", "src/main.apsi") if os.path.isfile(os.path.join(root, a))), None)
        if alt and not pkg.get("entry"):
            entry = alt
        else:
            raise PackageError(f"the entry file {entry} does not exist (set [package] entry = \"src/…\")")
    for dname, spec in deps.items():
        if not NAME_RE.match(dname):
            raise PackageError(f"dependency name '{dname}' is not valid")
        _parse_spec(str(spec))
    files = _collect(root, pkg)
    if entry not in files:
        files.append(entry)
    if check:
        _check_sources(root, [f for f in files if f.endswith(".apsi")])
    manifest = {
        "format": "apsil", "format_version": FORMAT_VERSION, "name": name, "version": version,
        "summary": str(pkg.get("summary") or ""), "description_file": next((f for f in files
                                                                           if f.lower().startswith("readme")), None),
        "authors": list(pkg.get("authors") or []), "license": str(pkg.get("license") or ""),
        "homepage": str(pkg.get("homepage") or ""), "repository": str(pkg.get("repository") or ""),
        "keywords": list(pkg.get("keywords") or []), "apsi": str(pkg.get("apsi") or _default_apsi_range()),
        "entry": entry, "dependencies": {k: str(v) for k, v in sorted(deps.items())},
        "files": [{"path": f, "sha256": _sha256_file(os.path.join(root, f)),
                   "size": os.path.getsize(os.path.join(root, f))} for f in sorted(files)],
    }
    from . import __version__
    manifest["built_with"] = f"apsi {__version__}"
    out_dir = out_dir or os.path.join(root, "dist")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"{name}-{version}.apsil")
    tmp = target + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        info = zipfile.ZipInfo("apsil.json", ZIP_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, json.dumps(manifest, indent=1, sort_keys=True))
        for f in manifest["files"]:
            info = zipfile.ZipInfo(f["path"], ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(os.path.join(root, f["path"]), "rb") as src, z.open(info, "w", force_zip64=f["size"] >= 1 << 31) as dst:
                shutil.copyfileobj(src, dst, CHUNK)
    os.replace(tmp, target)
    if not quiet:
        print(f"built {os.path.relpath(target)}  ({_human(os.path.getsize(target))}, {len(manifest['files'])} files, "
              f"sha256 {_sha256_file(target)[:16]}…)")
    return target


def _check_sources(root, rels):
    from .checker import Checker
    from .interpreter import Interpreter
    from .parser import parse

    def check_source(src, path):
        return Checker(Interpreter(), os.path.dirname(path)).check(parse(src))
    problems = []
    for rel in rels:
        path = os.path.join(root, rel)
        with open(path, encoding="utf-8-sig") as fh:
            src = fh.read()
        try:
            errs = check_source(src, path)
        except Exception as e:                     # noqa: BLE001 - report every kind of failure the same way
            errs = [e]
        for e in errs or []:
            problems.append(f"  {rel}: {e}")
    if problems:
        raise PackageError("the library has errors; fix them before packing:\n" + "\n".join(problems[:30]))


def _default_apsi_range():
    from . import __version__
    a, b, _, _ = parse_version(__version__.split("+")[0])
    return f">={a}.{b}.0"


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def read_manifest(path):
    """Validate an .apsil file's structure and return its manifest (does not extract anything)."""
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise PackageError(f"{os.path.basename(path)} is not an .apsil file ({e})")
    with z:
        try:
            raw = z.read("apsil.json")
        except KeyError:
            raise PackageError("the .apsil file has no apsil.json manifest")
        if len(raw) > 4 << 20:
            raise PackageError("apsil.json is too large")
        try:
            man = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise PackageError("apsil.json is not valid JSON")
        validate_manifest(man)
        names = {i.filename for i in z.infolist() if not i.is_dir()} - {"apsil.json"}
        listed = {f["path"] for f in man["files"]}
        if names != listed:
            raise PackageError("the archive's files do not match the manifest")
        for i in z.infolist():
            f = next((x for x in man["files"] if x["path"] == i.filename), None)
            if f is not None and i.file_size != f["size"]:
                raise PackageError(f"{i.filename}: size differs from the manifest")
        return man


def validate_manifest(man):
    if not isinstance(man, dict) or man.get("format") != "apsil":
        raise PackageError("not an Apsi library manifest")
    if int(man.get("format_version", 0)) > FORMAT_VERSION:
        raise PackageError(f"this .apsil needs a newer Apsi (format {man.get('format_version')})")
    if not NAME_RE.match(str(man.get("name", ""))):
        raise PackageError("invalid package name in the manifest")
    parse_version(man.get("version", ""))
    files = man.get("files")
    if not isinstance(files, list):
        raise PackageError("manifest has no file list")
    for f in files:
        p = f.get("path", "") if isinstance(f, dict) else ""
        if not p or p.startswith(("/", "\\")) or ".." in p.replace("\\", "/").split("/") or ":" in p or "\0" in p:
            raise PackageError(f"unsafe path in the package: {p!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))) or not isinstance(f.get("size"), int):
            raise PackageError(f"bad file entry for {p}")
    entry = man.get("entry")
    if entry not in {f["path"] for f in files}:
        raise PackageError("the manifest's entry file is not in the package")
    if not isinstance(man.get("dependencies", {}), dict):
        raise PackageError("dependencies must be a table")
    return man


# =============================================================================== installing
def _extract(path, man, dest):
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with zipfile.ZipFile(path) as z:
        for f in man["files"]:
            target = os.path.normpath(os.path.join(tmp, f["path"]))
            if not target.startswith(os.path.normpath(tmp) + os.sep):
                raise PackageError(f"unsafe path {f['path']!r}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            h = hashlib.sha256()
            n = 0
            with z.open(f["path"]) as src, open(target, "wb") as out:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    n += len(chunk)
                    if n > f["size"]:
                        raise PackageError(f"{f['path']} is larger than the manifest says")
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != f["sha256"] or n != f["size"]:
                raise PackageError(f"{f['path']} does not match its SHA-256 in the manifest (damaged or modified)")
        with open(os.path.join(tmp, "apsil.json"), "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)


def _guard(folder, quiet):
    """Scan an installed library with Apsi Guard. Critical findings remove it again; high ones are shown."""
    from . import guard
    results, _ = guard.scan_paths([folder])
    crit, high = [], []
    for f, _digest, found in results:
        for x in found or []:
            (crit if x.severity == "critical" else high if x.severity == "high" else []).append((f, x))
    if crit:
        shutil.rmtree(folder, ignore_errors=True)
        f, x = crit[0]
        raise PackageError(f"Apsi Guard blocked this library: {os.path.relpath(f, os.path.dirname(folder))}:"
                           f"{x.line}: {x.message}")
    if high and not quiet:
        print(f"  Apsi Guard: {len(high)} finding(s) worth a look:")
        for f, x in high[:5]:
            print(f"    {os.path.basename(f)}:{x.line}: {x.message}")


def install_file(path, project_dir=None, global_=False, quiet=False, expected_sha=None):
    man = read_manifest(path)
    if expected_sha and _sha256_file(path) != expected_sha:
        raise PackageError(f"{os.path.basename(path)} does not match the registry's SHA-256 (download damaged)")
    from . import __version__
    if man.get("apsi") and not satisfies(__version__, man["apsi"]):
        raise PackageError(f"{man['name']} {man['version']} needs Apsi {man['apsi']} (this is {__version__})")
    roots = package_roots(None if global_ else project_dir, include_global=True)
    root = roots[0]
    dest = os.path.join(root, man["name"], man["version"])
    _extract(path, man, dest)
    with open(os.path.join(root, man["name"], "current"), "w", encoding="utf-8") as fh:
        fh.write(man["version"])
    _guard(dest, quiet)
    if not quiet:
        print(f"installed {man['name']} {man['version']}  →  use {module_name(man['name'])}")
    return man


def _http(url, data=None, headers=None, method=None, timeout=60):
    from . import __version__
    h = {"User-Agent": f"apsi/{__version__}", "Accept": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8", "replace"))
            msg = body.get("error") or body.get("message") or str(body)
        except Exception:                          # noqa: BLE001
            msg = e.reason
        raise PackageError(f"registry: {e.code} {msg}")
    except urllib.error.URLError as e:
        raise PackageError(f"cannot reach the registry {url.split('/api/')[0]}: {e.reason}")


def package_info(name, registry=None):
    url = registry_url(registry)
    with _http(f"{url}/api/v1/packages/{urllib.parse.quote(name)}") as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url, dest, size=None, quiet=False):
    with _http(url, headers={"Accept": "application/octet-stream"}, timeout=120) as r, open(dest, "wb") as out:
        total = int(r.headers.get("Content-Length") or size or 0)
        got = 0
        t0 = time.time()
        for chunk in iter(lambda: r.read(CHUNK), b""):
            out.write(chunk)
            got += len(chunk)
            if not quiet and total > 8 * CHUNK and sys.stderr.isatty():
                sys.stderr.write(f"\r  {_human(got)} / {_human(total)}  ({got / max(time.time() - t0, 1e-3) / 1e6:.1f} MB/s)")
        if not quiet and total > 8 * CHUNK and sys.stderr.isatty():
            sys.stderr.write("\n")


def install(requirements, project_dir=None, global_=False, registry=None, upgrade=False, quiet=False, _seen=None):
    """Install requirements ('name', 'name@^1.2', local .apsil paths) and their dependencies."""
    seen = {} if _seen is None else _seen
    have = installed(None if global_ else project_dir, include_global=global_ or not project_dir)
    done = []
    for req in requirements:
        if req.endswith(".apsil") and os.path.isfile(req):
            man = install_file(req, project_dir, global_, quiet)
            seen[man["name"]] = man["version"]
            done.append(man)
            done += install([f"{k}@{v}" for k, v in man.get("dependencies", {}).items()], project_dir, global_,
                            registry, False, quiet, seen)
            continue
        name, spec = split_requirement(req)
        if name in seen:
            if not satisfies(seen[name], spec):
                raise PackageError(f"version conflict: {name} {seen[name]} is already chosen but {spec} is required")
            continue
        if name in have and not upgrade and satisfies(have[name][0], spec):
            seen[name] = have[name][0]
            if not quiet:
                print(f"{name} {have[name][0]} is already installed")
            continue
        info = package_info(name, registry)
        versions = {v["version"]: v for v in info.get("versions", []) if not v.get("yanked")}
        v = best_version(list(versions), spec)
        if v is None:
            raise PackageError(f"no version of {name} matches {spec} (available: "
                               f"{', '.join(sorted(versions, key=_key)[-8:]) or 'none'})")
        meta = versions[v]
        base = registry_url(registry)
        dl = meta.get("url") or f"/api/v1/packages/{name}/{v}/download"
        if dl.startswith("/"):
            dl = base + dl
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, f"{name}-{v}.apsil")
            if not quiet:
                print(f"downloading {name} {v} ({_human(int(meta.get('size') or 0))})")
            _download(dl, f, meta.get("size"), quiet)
            man = install_file(f, project_dir, global_, quiet, expected_sha=meta.get("sha256"))
        seen[name] = v
        done.append(man)
        done += install([f"{k}@{s}" for k, s in man.get("dependencies", {}).items()], project_dir, global_,
                        registry, False, quiet, seen)
    return done


def uninstall(name, project_dir=None, global_=False):
    removed = False
    for root in package_roots(None if global_ else project_dir, include_global=global_ or not project_dir):
        d = os.path.join(root, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
            removed = True
    if not removed:
        raise PackageError(f"{name} is not installed")
    return True


def write_lock(project_dir):
    pk = installed(project_dir)
    data = {"apsi_lock": 1, "packages": {}}
    for name, (v, folder) in sorted(pk.items()):
        with open(os.path.join(folder, "apsil.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        data["packages"][name] = {"version": v, "dependencies": man.get("dependencies", {}),
                                  "files_sha256": hashlib.sha256(json.dumps(man["files"], sort_keys=True)
                                                                 .encode()).hexdigest()}
    path = os.path.join(project_dir, "apsi-packages.lock")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return path


# =============================================================================== publishing
def _multipart(fields, file_field, file_path):
    boundary = "----apsi" + hashlib.sha256(os.urandom(16)).hexdigest()[:24]
    head = io.BytesIO()
    for k, v in fields.items():
        head.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    fname = os.path.basename(file_path).replace('"', "")
    head.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{fname}\"\r\n"
               f"Content-Type: application/vnd.apsi.library+zip\r\n\r\n".encode())
    tail = f"\r\n--{boundary}--\r\n".encode()
    size = len(head.getvalue()) + os.path.getsize(file_path) + len(tail)

    class Body(io.RawIOBase):
        def __init__(self):
            self.parts = [io.BytesIO(head.getvalue()), open(file_path, "rb"), io.BytesIO(tail)]

        def readable(self):
            return True

        def readinto(self, b):
            while self.parts:
                n = self.parts[0].readinto(b)
                if n:
                    return n
                self.parts.pop(0).close()
            return 0
    return Body(), boundary, size


def _send(req, timeout=3600):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error")
        except Exception:                           # noqa: BLE001
            msg = e.reason
        err = PackageError(f"upload refused: {e.code} {msg}")
        err.status = e.code
        raise err
    except urllib.error.URLError as e:
        err = PackageError(f"cannot reach the registry: {e.reason}")
        err.status = 0
        raise err


def _chunked_upload(url, auth, path, size, sha, quiet, purpose="library"):
    """Send a large file in resumable pieces; returns the upload id."""
    start = _send(urllib.request.Request(f"{url}/api/v1/uploads", method="POST", headers={**auth, "Content-Type": "application/json"},
                                         data=json.dumps({"purpose": purpose, "size": size, "sha256": sha,
                                                          "filename": os.path.basename(path)}).encode()))
    uid, chunk = start["id"], int(start.get("chunk_size") or (32 << 20))
    offset, tries, t0 = 0, 0, time.time()
    with open(path, "rb") as fh:
        while offset < size:
            fh.seek(offset)
            data = fh.read(min(chunk, size - offset))
            end = offset + len(data) - 1
            req = urllib.request.Request(f"{url}/api/v1/uploads/{uid}", data=data, method="PUT", headers={
                **auth, "Content-Type": "application/octet-stream", "Content-Range": f"bytes {offset}-{end}/{size}"})
            try:
                offset = int(_send(req, timeout=600)["received"])
                tries = 0
            except PackageError as e:
                tries += 1
                if tries > 6 or 400 <= getattr(e, "status", 0) < 500 and e.status not in (409, 429):
                    raise
                time.sleep(min(30, 2 ** tries))
                st = _send(urllib.request.Request(f"{url}/api/v1/uploads/{uid}", headers=auth))
                offset = int(st["received"])
            if not quiet and sys.stderr.isatty():
                sys.stderr.write(f"\r  {_human(offset)} / {_human(size)}  ({offset / max(time.time() - t0, 1e-3) / 1e6:.1f} MB/s)")
    if not quiet and sys.stderr.isatty():
        sys.stderr.write("\n")
    return uid


def publish(path=None, token=None, registry=None, quiet=False):
    """Upload an .apsil file. Returns the registry's answer (name, version, url)."""
    if path is None:
        cands = sorted((os.path.join("dist", f) for f in os.listdir("dist") if f.endswith(".apsil")),
                       key=os.path.getmtime) if os.path.isdir("dist") else []
        if not cands:
            raise PackageError("nothing to publish: run `apsi pack` first (it writes dist/NAME-VERSION.apsil)")
        path = cands[-1]
    man = read_manifest(path)
    url = registry_url(registry)
    tok = token or stored_token(url)
    if not tok:
        raise PackageError("no API token: create one on the site (Settings → Developer → New token) and pass "
                           "--token, set APSI_TOKEN, or store it with `apsi lib login`")
    sha = _sha256_file(path)
    size = os.path.getsize(path)
    auth = {"Authorization": f"Bearer {tok}", "User-Agent": "apsi", "Accept": "application/json"}
    # check the token before sending the file: a refused upload is then reported clearly on every OS, and a large
    # file is not sent for nothing (registries without this check answer 404 and are used as before)
    try:
        _send(urllib.request.Request(f"{url}/api/v1/token", headers=auth), timeout=60)
    except PackageError as e:
        if getattr(e, "status", 0) != 404:
            raise
    if not quiet:
        print(f"uploading {man['name']} {man['version']} ({_human(size)}) to {url}")
    if size <= SINGLE_UPLOAD_MAX:
        body, boundary, total = _multipart({"name": man["name"], "version": man["version"], "sha256": sha}, "file", path)
        req = urllib.request.Request(f"{url}/api/v1/upload", data=body, method="POST", headers={
            **auth, "Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(total)})
        res = _send(req)
    else:
        upload_id = _chunked_upload(url, auth, path, size, sha, quiet)
        req = urllib.request.Request(f"{url}/api/v1/upload", method="POST", headers={**auth, "Content-Type": "application/json"},
                                     data=json.dumps({"upload_id": upload_id, "name": man["name"],
                                                      "version": man["version"], "sha256": sha}).encode())
        res = _send(req)
    if not quiet:
        print(f"published {res.get('name')} {res.get('version')}  →  {res.get('url', url)}")
    return res


# =============================================================================== command line
def _err(e):
    print(f"error: {e}", file=sys.stderr)
    return 1


def cmd_install(args):
    try:
        proj = None if args.global_ else _project_dir()
        reqs = list(args.packages)
        if not reqs:
            if not proj:
                return _err("give a library name, or run inside a project with [dependencies] in apsi.toml")
            _, deps, _ = _package_config(proj)
            reqs = [f"{k}@{v}" for k, v in deps.items()]
            if not reqs:
                print("apsi.toml has no [dependencies]")
                return 0
        install(reqs, proj, args.global_, args.registry, args.upgrade)
        if proj and not args.global_:
            write_lock(proj)
        return 0
    except PackageError as e:
        return _err(e)


def cmd_uninstall(args):
    try:
        for n in args.packages:
            uninstall(n.lower(), None if args.global_ else _project_dir(), args.global_)
            print(f"removed {n}")
        return 0
    except PackageError as e:
        return _err(e)


def cmd_pack(args):
    try:
        pack(args.path, args.out, check=not args.no_check)
        return 0
    except PackageError as e:
        return _err(e)


def cmd_publish(args):
    try:
        publish(args.file, args.token, args.registry)
        return 0
    except PackageError as e:
        return _err(e)


def cmd_search(args):
    try:
        url = registry_url(args.registry)
        with _http(f"{url}/api/v1/search?q={urllib.parse.quote(' '.join(args.words))}") as r:
            res = json.loads(r.read().decode("utf-8"))
        items = res.get("results", [])
        if not items:
            print("no libraries found")
        for it in items:
            print(f"{it['name']:<28} {it.get('latest', ''):<10} {it.get('summary', '')[:70]}")
        return 0
    except PackageError as e:
        return _err(e)


def cmd_lib(args):
    try:
        a = args.lib_cmd
        if a == "list":
            pk = installed(_project_dir())
            if not pk:
                print("no libraries installed")
            for n, (v, folder) in sorted(pk.items()):
                where = "project" if "apsi_packages" in folder else "user"
                print(f"{n:<28} {v:<10} {where}")
        elif a == "info":
            info = package_info(args.name, args.registry)
            print(f"{info['name']} — {info.get('summary', '')}")
            print(f"latest: {info.get('latest')}   downloads: {info.get('downloads', 0)}")
            for v in info.get("versions", [])[:20]:
                print(f"  {v['version']:<12} {_human(int(v.get('size') or 0)):>9}  {v.get('published_at', '')}"
                      f"{'  (yanked)' if v.get('yanked') else ''}")
        elif a == "registry":
            d = _load_cred()
            if args.url:
                d["registry"] = registry_url(args.url)
                _save_cred(d)
            print(f"registry: {d.get('registry') or DEFAULT_REGISTRY + ' (default)'}")
        elif a == "login":
            url = registry_url(args.registry)
            tok = args.token or input("API token: ").strip()
            if not re.fullmatch(r"apsi_[A-Za-z0-9_-]{20,200}", tok):
                return _err("that does not look like an Apsi API token (apsi_…)")
            d = _load_cred()
            d.setdefault("tokens", {})[url] = tok
            _save_cred(d)
            print(f"token stored for {url} (file readable only by you)")
        elif a == "logout":
            url = registry_url(args.registry)
            d = _load_cred()
            d.get("tokens", {}).pop(url, None)
            _save_cred(d)
            print("token removed")
        return 0
    except PackageError as e:
        return _err(e)


def add_parser(sub):
    p = sub.add_parser("install", help="install Apsi libraries (.apsil) from the registry or a file")
    p.add_argument("packages", nargs="*", help="NAME, NAME@^1.2, \"NAME>=1,<2\" or a path to an .apsil file")
    p.add_argument("-g", "--global", dest="global_", action="store_true", help="install for your user, not the project")
    p.add_argument("-U", "--upgrade", action="store_true", help="install the newest matching version")
    p.add_argument("--registry", help="registry address (default: https://apsi.dev, or `apsi lib registry`)")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("uninstall", help="remove installed Apsi libraries")
    p.add_argument("packages", nargs="+")
    p.add_argument("-g", "--global", dest="global_", action="store_true")
    p.set_defaults(fn=cmd_uninstall)
    p = sub.add_parser("pack", help="check a library and build dist/NAME-VERSION.apsil")
    p.add_argument("path", nargs="?", default=".")
    p.add_argument("--out", help="output folder (default: dist)")
    p.add_argument("--no-check", action="store_true", help="skip checking the sources")
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("publish", help="upload an .apsil library to the registry")
    p.add_argument("file", nargs="?", help="default: the newest file in dist/")
    p.add_argument("--token", help="API token (or APSI_TOKEN, or `apsi lib login`)")
    p.add_argument("--registry")
    p.set_defaults(fn=cmd_publish)
    p = sub.add_parser("search", help="search the library registry")
    p.add_argument("words", nargs="+")
    p.add_argument("--registry")
    p.set_defaults(fn=cmd_search)
    p = sub.add_parser("lib", help="libraries: list, info, registry, login, logout")
    ls = p.add_subparsers(dest="lib_cmd", required=True)
    ls.add_parser("list", help="installed libraries")
    q = ls.add_parser("info", help="versions of a library in the registry")
    q.add_argument("name")
    q.add_argument("--registry")
    q = ls.add_parser("registry", help="show or set the registry address")
    q.add_argument("url", nargs="?")
    q = ls.add_parser("login", help="store an API token for publishing")
    q.add_argument("--token")
    q.add_argument("--registry")
    q = ls.add_parser("logout", help="forget the stored API token")
    q.add_argument("--registry")
    p.set_defaults(fn=cmd_lib)
