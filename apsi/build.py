"""`apsi build` — turn a program into an executable, and `apsi compile` — fill the compile cache ahead of time.

    apsi build app.apsi                       # app.exe (Windows) / app (Linux) next to the program
    apsi build app.apsi -o dist --allow-python --name trainer
    apsi build app.apsi --standalone          # dist folder with its own Python runtime: runs where Apsi is not installed
    apsi compile src/ tests/                  # parse, check and compile every program now; later runs start from the cache

What goes into the executable (one file):
  * a small native launcher (the same one pip uses for console scripts), the path of the Python runtime
  * a zip archive with: the Apsi runtime compiled to bytecode, the program and every file it imports,
    the already parsed and checked program, and the permissions given at build time
The program is checked when it is built; an executable never contains a program with check errors.
"""
import hashlib
import io
import json
import os
import pickle
import platform
import py_compile
import re
import shutil
import sys
import tempfile
import time
import zipfile

from . import __version__

MAIN = '''# Apsi executable (made by `apsi build`)
import json, os, sys, zipfile
def _main():
    exe = os.path.abspath(sys.argv[0])
    with zipfile.ZipFile(exe) as z:
        meta = json.loads(z.read("build.json"))
        from apsi.build import run_built
        return run_built(exe, z, meta, sys.argv[1:])
sys.exit(_main())
'''

PERM_FLAGS = ("allow_read", "allow_write", "allow_net", "allow_env", "allow_python", "allow_ffi")


# =============================================================== collecting the program
def _imports(prog):
    from . import ast as A
    out = []

    def walk(node):
        if isinstance(node, A.Import):
            out.append(node.path)
        if isinstance(node, A.AST):
            for f in node._fields:
                walk(getattr(node, f, None))
        elif isinstance(node, (list, tuple)):
            for x in node:
                walk(x)
    walk(prog)
    return out


def collect(entry):
    """The entry file and every .apsi file it imports, recursively: {absolute path: source}."""
    from .parser import parse
    files, todo, notes = {}, [os.path.abspath(entry)], []
    while todo:
        path = os.path.normpath(todo.pop())
        if path in files:
            continue
        with open(path, encoding="utf-8-sig") as fh:
            src = fh.read()
        files[path] = src
        try:
            prog = parse(src, cache=True)
        except Exception as e:
            e.build_file, e.build_source = path, src
            raise
        for imp in _imports(prog):
            if os.path.isabs(imp):
                notes.append(f"{os.path.basename(path)}: import \"{imp}\" is an absolute path; it is read from "
                             f"that location when the program runs")
                continue
            target = os.path.join(os.path.dirname(path), imp)
            if os.path.exists(target):
                todo.append(target)
    return files, notes


def _pycs(zf):
    """Add the Apsi runtime to the archive as bytecode (no compilation at start-up)."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix="apsi-build-")
    try:
        for name in sorted(os.listdir(pkg)):
            if name.endswith(".py"):
                out = os.path.join(tmp, name + "c")
                py_compile.compile(os.path.join(pkg, name), cfile=out, dfile=f"apsi/{name}", doraise=True,
                                   invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
                zf.write(out, f"apsi/{name}c")
            elif name == "MANIFEST.json":
                zf.write(os.path.join(pkg, name), f"apsi/{name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _flags(args):
    out = []
    for attr in PERM_FLAGS:
        v = getattr(args, attr, None)
        if v:
            flag = "--" + attr.replace("_", "-")
            out.append(flag if v in (True, "*", "safe") else f"{flag}={v}")
    for attr in ("allow_all", "sandbox", "trust"):
        if getattr(args, attr, False):
            out.append("--" + attr.replace("_", "-"))
    if getattr(args, "profile", None):
        out.append(f"--profile={args.profile}")
    if getattr(args, "engine", None) and args.engine != "compile":
        out.append(f"--engine={args.engine}")
    return out


# =============================================================== launcher
def _launcher(windowed):
    try:
        import pip._vendor.distlib as distlib
    except ImportError:
        raise SystemExit("apsi build: the Windows launcher comes with pip, which is missing from this Python "
                         "(python -m ensurepip)")
    arch = "-arm" if platform.machine().lower() in ("arm64", "aarch64") else ""
    path = os.path.join(os.path.dirname(distlib.__file__), f"{'w' if windowed else 't'}64{arch}.exe")
    if not os.path.exists(path):
        raise SystemExit(f"apsi build: launcher not found at {path}")
    with open(path, "rb") as fh:
        return fh.read()


def _shebang(python, windows, relative):
    if windows:
        exe = f"<launcher_dir>\\{relative}" if relative else f'"{python}"'
        return f"#!{exe} -I -X utf8\n".encode("utf-8")          # the format pip uses for its .exe launchers
    return f"#!{python} -IXutf8\n".encode("utf-8")      # Linux passes everything after the path as one argument


# =============================================================== standalone runtime
_STDLIB_SKIP = {"site-packages", "test", "tests", "idlelib", "ensurepip", "lib2to3", "turtledemo", "tkinter"}


def _needed_python_packages(sources, include):
    names = {"numpy"} | {x.strip() for x in include if x.strip()}
    text = "\n".join(sources)
    names |= set(re.findall(r'pyimport\(\s*"([A-Za-z0-9_\.]+)"', text))
    if re.search(r'backend\s*=\s*"(torch|auto)"|torch_module\(|check_torch\(|to_torch\(', text):
        names.add("torch")
    return {n.split(".")[0] for n in names}


def _distributions(top_names):
    from importlib import metadata
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    by_pkg = metadata.packages_distributions()
    todo = [d for n in top_names for d in by_pkg.get(n, [])]
    missing = sorted(n for n in top_names if n not in by_pkg)
    seen = {}
    while todo:
        name = todo.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[key] = dist
        for req in dist.requires or []:
            if Requirement is not None:
                try:
                    r = Requirement(req)
                except Exception:
                    continue
                if r.marker is not None and not r.marker.evaluate({"extra": ""}):
                    continue
                todo.append(r.name)
            elif "extra ==" not in req:
                todo.append(re.split(r"[ ;<>=!~\[]", req, 1)[0])
    return list(seen.values()), missing


def _copy_runtime(dest, sources, include, say):
    base = sys.base_prefix
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(base):
        src = os.path.join(base, name)
        if os.path.isfile(src) and name.lower().endswith((".exe", ".dll", ".pyd", "._pth", ".zip")):
            shutil.copy2(src, os.path.join(dest, name))
    if os.path.isdir(os.path.join(base, "DLLs")):
        shutil.copytree(os.path.join(base, "DLLs"), os.path.join(dest, "DLLs"), dirs_exist_ok=True)
    lib = os.path.join(base, "Lib")
    for name in os.listdir(lib):
        if name in _STDLIB_SKIP:
            continue
        src = os.path.join(lib, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, "Lib", name), dirs_exist_ok=True)
        else:
            os.makedirs(os.path.join(dest, "Lib"), exist_ok=True)
            shutil.copy2(src, os.path.join(dest, "Lib", name))
    site = next((p for p in sys.path if p.rstrip("\\/").endswith("site-packages") and os.path.isdir(p)), None)
    dists, missing = _distributions(_needed_python_packages(sources, include))
    for n in missing:
        say(f"  note: Python package '{n}' is used by the program but not installed here; it is not bundled")
    out_site = os.path.join(dest, "Lib", "site-packages")
    os.makedirs(out_site, exist_ok=True)
    for dist in dists:
        for f in dist.files or []:
            src = os.path.join(site, str(f)) if site else str(dist.locate_file(f))
            if not os.path.isfile(src):
                continue
            target = os.path.normpath(os.path.join(out_site, str(f)))
            if not target.startswith(os.path.normpath(out_site)):          # RECORD entries outside site-packages
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(src, target)
    say(f"  bundled Python {platform.python_version()} with {', '.join(sorted(d.metadata['Name'] for d in dists))}")


# =============================================================== apsi build
def cmd_build(args):
    from .checker import Checker
    from .errors import ApsiError
    from .interpreter import Interpreter
    from .parser import parse
    say = (lambda *a: None) if args.quiet else print
    entry = os.path.abspath(args.file)
    if not os.path.isfile(entry):
        print(f"apsi build: no such file: {args.file}", file=sys.stderr)
        return 2
    windows = os.name == "nt"
    if args.standalone and not windows:
        print("apsi build: --standalone makes a Windows folder with its own Python; on Linux the executable uses "
              "the installed Apsi runtime (build without --standalone)", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    from .cli import _report
    try:
        files, notes = collect(entry)
    except ApsiError as e:
        _report(e, getattr(e, "build_source", ""), getattr(e, "build_file", args.file))
        return 1
    except OSError as e:
        print(f"apsi build: {e}", file=sys.stderr)
        return 1
    # check everything first: an executable never contains a program with errors
    it = Interpreter()
    parsed = []
    for path, src in files.items():
        try:
            prog = parse(src, cache=True)
            errs = Checker(it, os.path.dirname(path)).check(prog)
        except ApsiError as e:
            _report(e, src, path)
            return 1
        if errs:
            from .interpreter import CheckFailed
            _report(CheckFailed(errs), src, path)
            return 1
        parsed.append((src, prog))
    root = os.path.commonpath([os.path.dirname(p) for p in files])
    extra = []
    for item in args.add or []:
        p = os.path.abspath(item)
        if not os.path.exists(p):
            print(f"apsi build: --add {item}: not found", file=sys.stderr)
            return 2
        if os.path.commonpath([root, p]) != root:
            print(f"apsi build: --add {item}: must be inside {root}", file=sys.stderr)
            return 2
        extra += [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs] if os.path.isdir(p) else [p]
    name = args.name or os.path.splitext(os.path.basename(entry))[0]
    out_dir = os.path.abspath(args.output or os.path.dirname(entry))
    if args.standalone:
        out_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    digest = hashlib.sha256()
    for p in sorted(files):
        digest.update(os.path.relpath(p, root).encode() + b"\0" + files[p].encode())
    meta = {"name": name, "entry": os.path.relpath(entry, root).replace("\\", "/"), "flags": _flags(args),
            "apsi": __version__, "python": platform.python_version(), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "id": None}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("__main__.py", MAIN)
        _pycs(zf)
        for p, src in files.items():
            zf.writestr("app/" + os.path.relpath(p, root).replace("\\", "/"), src.encode("utf-8"))
        for p in extra:
            with open(p, "rb") as fh:
                data = fh.read()
            digest.update(data)
            zf.writestr("app/" + os.path.relpath(p, root).replace("\\", "/"), data)
        zf.writestr("parsed.pickle", pickle.dumps(parsed, protocol=pickle.HIGHEST_PROTOCOL))
        digest.update(__version__.encode() + json.dumps(meta["flags"]).encode())
        meta["id"] = digest.hexdigest()[:16]
        zf.writestr("build.json", json.dumps(meta, indent=1))
    if args.standalone:
        say(f"copying the Python runtime into {out_dir}\\python …")
        _copy_runtime(os.path.join(out_dir, "python"), list(files.values()), args.include or [], say)
        python_rel = "python\\pythonw.exe" if args.windowed else "python\\python.exe"
    else:
        python_rel = None
    py = sys.executable
    if windows and args.windowed and not args.standalone:
        cand = os.path.join(os.path.dirname(py), "pythonw.exe")
        py = cand if os.path.exists(cand) else py
    head = (_launcher(args.windowed) if windows else b"") + _shebang(py, windows, python_rel)
    target = os.path.join(out_dir, name + (".exe" if windows else ""))
    tmp = target + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(head + buf.getvalue())
    os.replace(tmp, target)
    if not windows:
        os.chmod(target, 0o755)
    size = os.path.getsize(target)
    for n in notes:
        say(f"  note: {n}")
    say(f"✓ built {target}  ({size / 1024:.0f} KB, {len(files)} file{'s' if len(files) != 1 else ''}, "
        f"{time.perf_counter() - t0:.1f} s)")
    if meta["flags"]:
        say(f"  permissions: {' '.join(meta['flags'])}")
    if not args.standalone:
        say(f"  runs with the Apsi runtime at {os.path.dirname(py)}" if windows else f"  runs with {py}")
    return 0


# =============================================================== running a built program
def _apps_dir():
    from .security import config_dir
    return os.path.join(config_dir(), "apps")


def run_built(exe, z, meta, argv):
    """Entry point of an executable made by `apsi build`."""
    from . import cli, compile_cache, runner
    root = os.path.join(_apps_dir(), f"{meta['name']}-{meta['id']}")
    if not os.path.exists(os.path.join(root, ".complete")):
        tmp = f"{root}.{os.getpid()}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        for info in z.infolist():
            if info.filename.startswith("app/") and not info.is_dir():
                rel = info.filename[4:]
                dest = os.path.normpath(os.path.join(tmp, rel))
                if not dest.startswith(os.path.normpath(tmp) + os.sep):
                    continue                               # never write outside the app folder
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as fh:
                    fh.write(z.read(info))
        open(os.path.join(tmp, ".complete"), "w").close()
        try:
            os.replace(tmp, root)
        except OSError:                                    # another copy of the program got there first
            shutil.rmtree(tmp, ignore_errors=True)
    compile_cache.preload(pickle.loads(z.read("parsed.pickle")))      # trusted: it is part of this executable
    cli._utf8()
    ap, sub = cli._parser()
    ns = sub.choices["run"].parse_args([os.path.join(root, *meta["entry"].split("/"))] + meta["flags"])
    ns.args = list(argv)
    ns._project_applied = True          # the permissions were fixed at build time; an apsi.toml nearby has no say
    return runner.call(cli.cmd_run, ns) or 0


# =============================================================== apsi compile
def cmd_compile(args):
    from .cli import _collect, _read, _report
    from .checker import Checker
    from .errors import ApsiError
    from .interpreter import Interpreter, CheckFailed
    from .parser import parse
    from . import compile_cache
    if args.clear:
        n = compile_cache.clear()
        print(f"compile cache cleared ({n} entries)")
        return 0
    if args.stats:
        n, size = compile_cache.stats()
        print(f"compile cache: {n} programs, {size / 1024:.0f} KB in {compile_cache.cache_dir()}")
        return 0
    if not compile_cache.enabled():
        print("apsi compile: the compile cache is turned off (APSI_NO_CACHE)", file=sys.stderr)
        return 2
    files = _collect(args.paths or ["."])
    if not files:
        print("apsi compile: no .apsi files found", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    status = ok = 0
    it = Interpreter()
    for f in files:
        src = _read(f)
        try:
            prog = parse(src, cache=True)
            errs = Checker(it, os.path.dirname(os.path.abspath(f))).check(prog)
            if errs:
                raise CheckFailed(errs)
            from .codegen import Compiled
            Compiled(Interpreter(), prog, f, os.path.dirname(os.path.abspath(f)), [])
            ok += 1
        except (ApsiError, CheckFailed) as e:
            _report(e, src, f)
            status = 1
    n, size = compile_cache.stats()
    print(f"compiled {ok}/{len(files)} file{'s' if len(files) != 1 else ''} in {time.perf_counter() - t0:.2f} s · "
          f"cache: {n} programs, {size / 1024:.0f} KB")
    if args.runtime:
        import compileall
        pkg = os.path.dirname(os.path.abspath(__file__))
        quiet = compileall.compile_dir(pkg, quiet=1)
        print("Apsi runtime bytecode: " + ("up to date" if quiet else "some files could not be compiled"))
    return status


def register(sub, perm_flags):
    p = sub.add_parser("build", help="make an executable from a program (app.exe on Windows)")
    p.add_argument("file", help="the program's main .apsi file")
    p.add_argument("-o", "--output", help="output folder (default: next to the program)")
    p.add_argument("--name", help="name of the executable (default: the file name)")
    p.add_argument("--standalone", action="store_true",
                   help="Windows: a folder with its own Python runtime, for computers without Apsi")
    p.add_argument("--include", action="append", metavar="PKG",
                   help="with --standalone: also bundle this Python package (pyimport/torch use is detected)")
    p.add_argument("--add", action="append", metavar="PATH", help="also pack this data file or folder")
    p.add_argument("--windowed", action="store_true", help="Windows: no console window (GUI / server programs)")
    p.add_argument("-q", "--quiet", action="store_true")
    perm_flags(p)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("compile", help="parse, check and compile programs now so later runs start from the cache")
    p.add_argument("paths", nargs="*", help="files or folders (default: current folder)")
    p.add_argument("--runtime", action="store_true", help="also compile the Apsi runtime itself to bytecode")
    p.add_argument("--stats", action="store_true", help="show the size of the compile cache")
    p.add_argument("--clear", action="store_true", help="empty the compile cache")
    p.set_defaults(fn=cmd_compile)
