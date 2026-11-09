"""Command-line interface for the Apsi toolchain.

    apsi                      interactive REPL (or run a program piped on stdin)
    apsi FILE.apsi [ARGS]     run a program (same as `apsi run`)
    apsi -e "CODE"            evaluate code and print the result
    apsi TASK                 run a task from apsi.toml when TASK is not a command

`apsi help` / `apsi COMMAND --help` describe every command.
"""
import argparse
import contextlib
import io
import os
import sys
import time

from . import __version__
from .errors import ApsiError, CheckFailed

PERMS = ("python", "ffi", "net", "fs", "env")


def _sigpipe():
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `apsi run x | head` exits quietly
    except (AttributeError, ValueError):
        pass


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _read(path):
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return fh.read()
    except OSError as e:
        print(f"apsi: cannot open '{path}': {e.strerror}", file=sys.stderr)
        raise SystemExit(2)


_COLOR = None


def _red(s):
    global _COLOR
    if _COLOR is None:
        from .repl import color_supported
        _COLOR = color_supported(sys.stderr)
    if not _COLOR:
        return s
    head, nl, rest = s.partition("\n")
    return f"\x1b[31m{head}\x1b[0m{nl}{rest}"


def _report(err, source, filename):
    if isinstance(err, CheckFailed):
        for e in err.errors:
            print(_red(e.format(source, filename)), file=sys.stderr)
        n = len(err.errors)
        print(f"\n{n} error{'s' if n != 1 else ''} found by the checker; nothing was run.", file=sys.stderr)
    else:
        print(_red(err.format(source, filename)), file=sys.stderr)


def _project_defaults(args):
    """Fill --profile / --allow-* from apsi.toml [run] when the command line does not set them."""
    if getattr(args, "_project_applied", False):
        return
    args._project_applied = True
    try:
        from .project import load_project, project_run_flags
        root, cfg = load_project()
    except SystemExit:
        return
    if not root:
        return
    profile, allow = project_run_flags(cfg)
    if profile and not getattr(args, "profile", None) and hasattr(args, "profile"):
        args.profile = profile
    for perm in allow:
        attr = "allow_" + perm.replace("-", "_")
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, "safe" if perm == "python" else ("*" if perm != "ffi" else True))


def _make_interp(args, program=None):
    from .interpreter import Interpreter
    from .security import policy_from_args
    if program and program != "-":
        _project_defaults(args)
    it = Interpreter(policy=policy_from_args(args, program if program != "-" else None))
    it.engine = getattr(args, "engine", None) or "compile"
    return it


def _limits(args):
    from .security import apply_limits
    for note in apply_limits(getattr(args, "max_memory", None), getattr(args, "timeout", None)):
        print(f"apsi: warning: {note}", file=sys.stderr)


def _profile_name(args):
    from .security import load_config
    if getattr(args, "allow_all", False):
        return "developer"
    if getattr(args, "sandbox", False):
        return "sandbox"
    return getattr(args, "profile", None) or load_config().get("profile", "standard")


def _guard(args, src, path):
    """Apsi Guard pre-run scan; returns False when the program must not run."""
    from .parser import parse
    from .guard import gate_program
    try:
        prog = parse(src, cache=path != "<stdin>")
    except ApsiError:
        return True          # syntax errors are reported by the normal run path
    return gate_program(prog, src, path, _profile_name(args), trust=getattr(args, "trust", False))


def cmd_run(args):
    if not args.file:
        from .project import project_main
        root, main = project_main()
        if not main:
            print("apsi run: no file given and no apsi.toml with [project] main here "
                  "(usage: apsi run FILE.apsi, or create a project with `apsi new NAME`)", file=sys.stderr)
            return 2
        args.file = os.path.relpath(main)
    if args.watch:
        from .tools import cmd_watch
        import argparse
        return cmd_watch(argparse.Namespace(paths=[args.file], do="run", extra=None, ext=None,
                                            interval=0.3, no_clear=False))
    from . import stdlib
    stdlib.PROGRAM_ARGS[:] = list(args.args or [])
    src = _read(args.file)
    name = "<stdin>" if args.file == "-" else args.file
    if not _guard(args, src, name):
        return 3
    _limits(args)
    if getattr(args, "native", False):
        os.environ["APSI_NATIVE"] = "auto"          # every function gets a native tier (silent fallback)
    if getattr(args, "threads", None):
        from . import native as _nat
        _nat.set_threads(args.threads)
    it = _make_interp(args, args.file)
    t0 = time.perf_counter()
    try:
        it.run(src, name, check=not args.no_check)
    except (ApsiError, CheckFailed) as e:
        _report(e, src, name)
        return 1
    except KeyboardInterrupt:
        print("\napsi: interrupted", file=sys.stderr)
        return 130
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    finally:
        if args.time:
            print(f"apsi: {name} ran in {time.perf_counter() - t0:.3f}s", file=sys.stderr)
    return 0


def cmd_native(args):
    """apsi native info | explain FILE | ir FILE FN | asm FILE FN | clear-cache"""
    import json
    from . import native as nat
    action = args.action
    if action == "info":
        from .native.gpu import info as ginfo
        out = nat.info()
        out["gpu"] = ginfo()
        if args.json:
            print(json.dumps(out, indent=1, default=str))
            return 0
        if not out["available"]:
            print(f"native compiler: unavailable ({out.get('reason')})")
            return 1
        print(f"native compiler : LLVM {out['llvm']}, target {out['triple']}, CPU {out['cpu']}")
        print(f"SIMD            : {', '.join(out['simd']) or 'none'}")
        print(f"threads         : {out['threads']} (APSI_THREADS or native_threads(n) to change)")
        print(f"machine-code cache: {out['cache_dir']}")
        g = out["gpu"]
        if g["available"]:
            for d in g["devices"]:
                print(f"GPU {d['index']}           : {d['name']}, compute {d['compute_capability']}, "
                      f"{d['multiprocessors']} SMs, {d['memory_gb']} GB")
        else:
            print(f"GPU             : none ({g['reason']}); @gpu functions run on all CPU cores")
        return 0
    if action == "clear-cache":
        import shutil
        from .security import config_dir
        d = os.path.join(config_dir(), "cache", "native")
        shutil.rmtree(d, ignore_errors=True)
        print(f"removed {d}")
        return 0
    if not args.file:
        print(f"apsi native {action}: give the program file", file=sys.stderr)
        return 2
    os.environ["APSI_NATIVE_QUIET"] = "1"
    if args.all:
        os.environ["APSI_NATIVE"] = "auto"
    src = _read(args.file)
    if not _guard(args, src, args.file):
        return 3
    it = _make_interp(args, args.file)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            it.run(src, args.file)
    except (ApsiError, CheckFailed) as e:
        _report(e, src, args.file)
        return 1
    fns = [f for f in nat.REGISTRY if not f.erased]
    if action == "explain":
        if not fns:
            print("no @native / @gpu functions were called (use --all to try every function)")
            return 0
        for f in fns:
            done = [k for k, v in f.specs.items() if not isinstance(v, nat._Fallback)]
            fell = [v.reason for v in f.specs.values() if isinstance(v, nat._Fallback)]
            fell += [v.reason for v in f._apsi_e.specs.values() if isinstance(v, nat._Fallback)]
            if done:
                sig = "; ".join("(" + ", ".join(_tn(t, d) for t, d in k[0]) + ")" for k in done)
                print(f"  native   {f.name}{'' if not fell else ' (partly)'}  {sig}")
            if fell:
                print(f"  bytecode {f.name}: {fell[0]}")
            if not done and not fell and f.py_calls:
                print(f"  bytecode {f.name}: called with values native code does not take (lists, records, "
                      f"structs, autodiff values ...)")
            elif not done and not fell:
                print(f"  -        {f.name}: not called")
            for name, msg in nat.reports():
                if name == f.name and msg not in fell:
                    print(f"           {msg}")
        return 0
    want = args.function
    match = [f for f in fns if f.name == want] if want else fns[:1]
    if not match:
        print(f"apsi native {action}: no native function named {want!r} was called", file=sys.stderr)
        return 1
    f = match[0]
    specs = [v for v in f.specs.values() if not isinstance(v, nat._Fallback)]
    if not specs:
        print(f"apsi native {action}: {f.name}() did not compile natively", file=sys.stderr)
        return 1
    text = specs[0].ir_text
    if action == "ir":
        print(text)
    else:
        print(nat._eng.engine().assembly(text)[1])
    return 0


def _tn(t, d):
    from .native.compiler import tname
    s = tname(t)
    return s if d.dimensionless else f"{s}[{d}]"


def cmd_check(args):
    from .interpreter import Interpreter
    from .checker import Checker
    from .parser import parse
    status = 0
    files = _collect(args.files or ["."])
    if not files:
        print("apsi check: no .apsi files found", file=sys.stderr)
        return 2
    for f in files:
        src = _read(f)
        try:
            prog = parse(src)
            errs = Checker(Interpreter(), os.path.dirname(os.path.abspath(f))).check(prog)
        except ApsiError as e:
            _report(e, src, f)
            status = 1
            continue
        if errs:
            _report(CheckFailed(errs), src, f)
            status = 1
        elif not args.quiet:
            print(f"✓ {f}: no errors")
    if args.quiet and status == 0:
        print(f"✓ {len(files)} file{'s' if len(files) != 1 else ''}: no errors")
    return status


def _collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in
                                 ("node_modules", "__pycache__", "dist", "build", "venv"))
                out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".apsi")]
        else:
            out.append(p)
    return out


def cmd_test(args):
    paths = args.paths
    if not paths:
        from .project import find_project
        root = find_project()
        paths = [os.path.relpath(os.path.join(root, "tests"))] if root and os.path.isdir(
            os.path.join(root, "tests")) else ["tests"]
    files = _collect(paths)
    if not files:
        print(f"apsi test: no .apsi files in {', '.join(paths)}", file=sys.stderr)
        return 2
    total_pass, total_fail, skipped = 0, 0, 0
    t0 = time.perf_counter()
    for f in files:
        src = _read(f)
        it = _make_interp(args, f)
        it.testing = True
        print(f"● {f}")
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                it.run(src, f)
        except (ApsiError, CheckFailed) as e:
            if getattr(e, "kind", "") == "PermissionError":
                print(f"  - skipped: {e.msg}")
                skipped += 1
                continue
            print(f"  ✗ file failed to load")
            _report(e, src, f)
            total_fail += 1
            continue
        p, failed = it.run_tests()
        total_pass += p
        total_fail += len(failed)
        if not it.tests:
            print("  (no tests)")
    dt = time.perf_counter() - t0
    extra = f", {skipped} file(s) skipped (need permissions)" if skipped else ""
    print(f"\n{total_pass} passed, {total_fail} failed{extra} in {dt:.2f}s")
    return 1 if total_fail else 0


def cmd_repl(args):
    from .repl import run_repl
    return run_repl(args, _make_interp)


def cmd_ast(args):
    from .parser import parse
    from . import ast as A
    src = _read(args.file)
    try:
        print(A.dump(parse(src).body))
    except ApsiError as e:
        _report(e, src, args.file)
        return 1
    return 0


def cmd_tokens(args):
    from .lexer import tokenize
    src = _read(args.file)
    try:
        for t in tokenize(src):
            print(t)
    except ApsiError as e:
        _report(e, src, args.file)
        return 1
    return 0


def cmd_fmt(args):
    from .formatter import format_checked
    status = 0
    for f in _collect(args.files):
        src = _read(f)
        try:
            new, changed = format_checked(src)
        except ApsiError as e:
            _report(e, src, f)
            status = 1
            continue
        if not changed:
            continue
        if args.check:
            print(f"would reformat {f}")
            status = 1
        else:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(new)
            print(f"formatted {f}")
    return status


def cmd_lsp(args):
    from .lsp import main as lsp_main
    return lsp_main()


def cmd_scan(args):
    from .guard import scan_paths, report_json, SEVERITY, worst
    for p in args.paths:
        if not os.path.exists(p):
            print(f"apsi scan: '{p}' does not exist")
            return 2
    results, av = scan_paths(args.paths, with_av=args.av)
    if args.json:
        print(report_json(results, av))
    else:
        icons = {"critical": "✖", "high": "▲", "medium": "●", "low": "·", "info": " "}
        shown = 0
        for f, digest, found in results:
            real = [x for x in found if x.severity != "info"]
            if not real:
                continue
            print(f"\n{f}")
            for x in sorted(real, key=lambda x: (-SEVERITY[x.severity], x.line)):
                print(f"  {icons[x.severity]} {x.severity:<8} line {x.line:<5} {x.rule}: {x.message}")
                shown += 1
        clean = sum(1 for _, _, fs in results if not [x for x in fs if x.severity != "info"])
        print(f"\nScanned {len(results)} files: {clean} clean, {len(results) - clean} with findings ({shown} total).")
        if av:
            eng, ok, detail = av
            if eng:
                print(f"{eng}: {'no threats found' if ok else 'THREATS FOUND'}")
                for d in detail:
                    print("  " + d)
            else:
                print(detail[0])
    level = max((worst(fs) for _, _, fs in results), default=0)
    if av and av[1] is False:
        return 1
    return 1 if level >= SEVERITY[args.fail_on] else 0


def cmd_audit(args):
    from .parser import parse
    from .security import audit
    status = 0
    for f in _collect(args.files):
        src = _read(f)
        try:
            found = audit(parse(src))
        except ApsiError as e:
            _report(e, src, f)
            status = 1
            continue
        print(f"● {f}")
        if not found:
            print("  pure computation: runs fine with --sandbox")
            continue
        perms = {}
        for line, name, perm, desc in found:
            print(f"  line {line:4}: {name:18} {perm:7} {desc}")
            perms.setdefault(perm, True)
        flags = []
        for perm in ("read", "write", "net", "env", "python", "ffi"):
            if perm in perms:
                flags.append(f"--allow-{perm}")
        print("  minimal flags: " + " ".join(flags))
    return status


def cmd_verify(args):
    from .security import verify_installation
    from . import __version__
    ok, problems = verify_installation()
    if ok is None:
        print(f"Apsi {__version__}: {problems[0]}; nothing to verify")
        return 0
    if ok:
        print(f"✓ Apsi {__version__}: all runtime files match the release manifest")
        return 0
    print(f"✗ Apsi {__version__}: installation does not match the release manifest:")
    for p in problems:
        print("   " + p)
    return 1


def cmd_config(args):
    from .security import load_config, save_config, PROFILES, Policy, config_dir
    cfg = load_config()
    changed = False
    if args.profile:
        cfg["profile"] = args.profile
        changed = True
    if changed:
        path = save_config(cfg)
        print(f"saved {path}")
    prof = cfg.get("profile", "standard")
    print(f"config folder : {config_dir()}")
    print(f"profile       : {prof}   (choices: {', '.join(PROFILES)})")
    for k, v in Policy.profile(prof).describe().items():
        print(f"  {k:7}: {v}")
    return 0


def cmd_doctor(args):
    from .hardware import doctor, info
    if getattr(args, "json", False):
        import json
        print(json.dumps(info(), default=str))
        return 0
    return doctor(verbose=True)


def cmd_serve(args):
    from .server import serve
    src = _read(args.file)
    if not _guard(args, src, args.file):
        return 3
    _limits(args)
    it = _make_interp(args, args.file)
    try:
        env = it.run(src, args.file)
    except (ApsiError, CheckFailed) as e:
        _report(e, src, args.file)
        return 1
    return serve(it, env, args.host, args.port, args.only)


def cmd_version(args):
    print(f"Apsi {__version__}")
    return 0


_OPTIONAL_VALUE_FLAGS = {"--allow-read": "*", "--allow-write": "*", "--allow-net": "*", "--allow-env": "*",
                         "--allow-python": "safe"}


def _glue_flag_values(argv):
    """`--allow-python app.apsi` must not take `app.apsi` as the flag's value: a following word is used as the value
    only when it can be one (DIRS / HOSTS / NAMES, or all|safe for --allow-python); otherwise the default applies."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            return out + argv[i:]
        if tok in _OPTIONAL_VALUE_FLAGS and i + 1 < len(argv):
            nxt = argv[i + 1]
            if tok == "--allow-python":
                is_value = nxt in ("all", "safe", "unsafe")
            else:
                is_value = not nxt.startswith("-") and not nxt.lower().endswith(".apsi") and nxt not in ("-",)
            if not is_value:
                out.append(f"{tok}={_OPTIONAL_VALUE_FLAGS[tok]}")
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def _parser():
    """The argument parser of every command (also used by programs built with `apsi build`)."""
    ap = argparse.ArgumentParser(
        prog="apsi", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="The Apsi language toolchain.",
        epilog="shortcuts:\n"
               "  apsi                     start the REPL (or run a program piped on stdin)\n"
               "  apsi FILE.apsi [ARGS]    run a program\n"
               "  apsi -e \"CODE\"           evaluate code and print the result\n"
               "  apsi TASK                run a task from apsi.toml\n"
               "  apsi editors install     add Apsi to VS Code, Cursor, Windsurf, Sublime, Vim/Neovim, Notepad++\n"
               "  apsi convert app.py      translate Python (numpy, math, torch, …) into Apsi\n\n"
               "docs: docs/LANGUAGE.md — the complete language reference (chapter 24: every command)")
    ap.add_argument("-V", "--version", action="version", version=f"Apsi {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")

    def perm_flags(p):
        g = p.add_argument_group("security")
        g.add_argument("--profile", choices=["sandbox", "strict", "standard", "developer"],
                       help="permission profile (default: from 'apsi config', else standard)")
        g.add_argument("--allow-read", nargs="?", const="*", metavar="DIRS", help="allow reading files (optionally only in DIRS)")
        g.add_argument("--allow-write", nargs="?", const="*", metavar="DIRS", help="allow writing files (optionally only in DIRS)")
        g.add_argument("--allow-net", nargs="?", const="*", metavar="HOSTS", help="allow network access (optionally only HOST[:PORT],:PORT)")
        g.add_argument("--allow-env", nargs="?", const="*", metavar="NAMES", help="allow reading environment variables")
        g.add_argument("--allow-python", nargs="?", const="safe", metavar="all",
                       help="allow importing Python libraries (numpy, torch, ...); =all also allows system modules "
                            "such as subprocess and os that bypass Apsi's permissions")
        g.add_argument("--allow-ffi", nargs="?", const=True, metavar="LIB@SHA256",
                       help="allow loading C/C++ shared libraries (optionally pinned: vecmath.so@<sha256>,...)")
        g.add_argument("--trust", action="store_true", help="skip the Apsi Guard pre-run threat scan")
        g.add_argument("--allow-all", action="store_true", help="allow everything (developer profile)")
        g.add_argument("--sandbox", action="store_true", help="deny everything, including files")
        g.add_argument("--max-memory", metavar="SIZE", help="memory cap, e.g. 2GB")
        g.add_argument("--timeout", metavar="TIME", help="stop the program after e.g. 30s or 5min")
        p.add_argument("--engine", choices=["compile", "interp"], default="compile",
                       help="compile (default, fast) or interp (reference tree-walker)")

    p = sub.add_parser("run", help="run a .apsi program (default: the project's main file)")
    p.add_argument("file", nargs="?", help="program to run; '-' reads it from stdin")
    p.add_argument("args", nargs="*", help="arguments for the program (read them with args())")
    p.add_argument("--no-check", action="store_true", help="skip the static checker")
    p.add_argument("--time", action="store_true", help="print how long the program ran")
    p.add_argument("-w", "--watch", action="store_true", help="run again every time the file changes")
    p.add_argument("--native", action="store_true",
                   help="compile every function to machine code when possible (as if each had @native)")
    p.add_argument("--threads", type=int, metavar="N", help="CPU threads for parallel for (default: all cores)")
    perm_flags(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("native", help="native compiler: info, explain FILE, ir FILE [FN], asm FILE [FN], clear-cache")
    p.add_argument("action", choices=["info", "explain", "ir", "asm", "clear-cache"])
    p.add_argument("file", nargs="?", help="program to run (explain / ir / asm)")
    p.add_argument("function", nargs="?", help="function name (ir / asm; default: the first one)")
    p.add_argument("--all", action="store_true", help="try every function, as apsi run --native does")
    p.add_argument("--json", action="store_true", help="machine-readable output (info)")
    perm_flags(p)
    p.set_defaults(fn=cmd_native)

    p = sub.add_parser("check", help="statically check files or folders without running them")
    p.add_argument("files", nargs="*", help="files or folders (default: current folder)")
    p.add_argument("-q", "--quiet", action="store_true", help="print only errors and a summary")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("test", help="run test blocks in files or folders")
    p.add_argument("paths", nargs="*")
    perm_flags(p)
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("repl", help="interactive prompt")
    perm_flags(p)
    p.set_defaults(fn=cmd_repl)

    p = sub.add_parser("fmt", help="format .apsi files in place (safe: syntax tree must stay identical)")
    p.add_argument("files", nargs="*", default=["."], help="files or folders (default: current folder)")
    p.add_argument("--check", action="store_true", help="only report files that would change")
    p.set_defaults(fn=cmd_fmt)

    p = sub.add_parser("audit", help="list the capabilities a program uses and the minimal flags it needs")
    p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("scan", help="scan programs, folders and companion files for malware patterns")
    p.add_argument("paths", nargs="+")
    p.add_argument("--av", action="store_true", help="also run Microsoft Defender or ClamAV")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("--fail-on", choices=["critical", "high", "medium", "low"], default="high",
                   help="exit with status 1 at or above this severity (default high)")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("verify", help="check the installed runtime against its release manifest")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("config", help="show or change settings (security profile)")
    p.add_argument("--profile", choices=["sandbox", "strict", "standard", "developer"])
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("lsp", help="run the language server (stdio) for editors")
    p.set_defaults(fn=cmd_lsp)

    p = sub.add_parser("doctor", help="check this machine against Apsi's hardware targets")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("serve", help="expose a program's functions as a JSON HTTP API")
    p.add_argument("file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--only", nargs="*", help="only expose these function names")
    perm_flags(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("ast", help="print the syntax tree of a file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_ast)

    p = sub.add_parser("tokens", help="print the tokens of a file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_tokens)

    from . import packages as _packages, editors as _editors, convert as _convert
    from . import scale as _scale
    _scale.add_parser(sub)
    from . import modelio as _modelio
    _modelio.add_parser(sub)
    _packages.add_parser(sub)
    from . import apsil as _apsil
    _apsil.add_parser(sub)
    from . import remote as _remote
    _remote.add_parser(sub)
    _editors.add_parser(sub)
    _convert.add_parser(sub)

    p = sub.add_parser("version", help="print the version")
    p.set_defaults(fn=cmd_version)

    # ------------------------------------------------------------ project & developer commands
    from . import project as _project, tools as _tools

    p = sub.add_parser("new", help="create a new project folder (apsi new NAME --template app|lib|physics|ai|server)")
    p.add_argument("name")
    p.add_argument("-t", "--template", choices=list(_project.TEMPLATES), default="app")
    p.add_argument("--no-vscode", action="store_true", help="do not write .vscode/ tasks and settings")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(fn=_project.cmd_new)

    p = sub.add_parser("init", help="turn the current (or given) folder into an Apsi project")
    p.add_argument("path", nargs="?")
    p.add_argument("--name")
    p.add_argument("-t", "--template", choices=list(_project.TEMPLATES), default="app")
    p.add_argument("--no-vscode", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=_project.cmd_init)

    p = sub.add_parser("task", help="list or run the [tasks] of apsi.toml")
    p.add_argument("name", nargs="?")
    p.add_argument("args", nargs=argparse.REMAINDER, help="extra arguments passed to the task's last step")
    p.set_defaults(fn=_project.cmd_task)

    p = sub.add_parser("eval", help="evaluate code from the command line and print the result")
    p.add_argument("code", nargs="+", help='code, e.g. "sqrt(2) * 3m"   (use ; or \\n for several statements)')
    perm_flags(p)
    p.set_defaults(fn=lambda a: _tools.cmd_eval(a, _make_interp, _report))

    p = sub.add_parser("watch", help="re-run a program, tests or the checker whenever files change")
    p.add_argument("paths", nargs="*", help="files or folders to watch (default: current folder)")
    p.add_argument("--do", choices=["run", "test", "check"], default="run", help="what to run (default: run)")
    p.add_argument("--ext", help="comma-separated extensions to watch (default .apsi,.toml,.csv,.json)")
    p.add_argument("--interval", type=float, default=0.3, help="seconds between checks")
    p.add_argument("--no-clear", action="store_true", help="do not clear the screen between runs")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra flags, e.g. -- --allow-net")
    p.set_defaults(fn=_tools.cmd_watch)

    p = sub.add_parser("bench", help="time a program over several runs (output suppressed)")
    p.add_argument("file")
    p.add_argument("-n", "--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    perm_flags(p)
    p.set_defaults(fn=lambda a: _tools.cmd_bench(a, _make_interp, _read, _report))

    p = sub.add_parser("doc", help="documentation for built-ins and modules (apsi doc nn, apsi doc sqrt)")
    p.add_argument("name", nargs="?")
    p.add_argument("filter", nargs="?", help="only list names containing this text")
    p.set_defaults(fn=_tools.cmd_doc)

    p = sub.add_parser("completion", help="print a shell completion script (bash, zsh, fish, powershell)")
    p.add_argument("shell", choices=["bash", "zsh", "fish", "powershell"])
    p.set_defaults(fn=lambda a: _tools.cmd_completion(a, ap))

    p = sub.add_parser("help", help="show help for a command (apsi help run)")
    p.add_argument("topic", nargs="?")
    p.set_defaults(fn=lambda a: (sub.choices[a.topic].print_help() if a.topic in sub.choices
                                 else ap.print_help()) or 0)
    from . import build as _build
    _build.register(sub, perm_flags)
    return ap, sub


def main(argv=None):
    _utf8()
    _sigpipe()
    argv = _glue_flag_values(list(sys.argv[1:] if argv is None else argv))
    ap, sub = _parser()
    argv = _shortcut(argv, set(sub.choices), ap)
    if argv is None:
        return 2
    if argv and argv[0] == "run" and not any(a in ("-h", "--help") for a in argv[1:]):
        # flags may come before or after the program arguments: apsi run app.apsi a b --allow-net
        args = sub.choices["run"].parse_intermixed_args(argv[1:])
        args.cmd = "run"
    else:
        args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 0
    try:
        if args.cmd in ("run", "test", "eval", "bench"):
            from . import runner
            return runner.call(args.fn, args) or 0
        return args.fn(args) or 0
    except KeyboardInterrupt:
        print("\napsi: interrupted", file=sys.stderr)
        return 130


def _shortcut(argv, commands, ap):
    """Map the short forms (`apsi`, `apsi file.apsi`, `apsi -e`, `apsi TASK`) onto real commands."""
    if not argv:
        return ["repl"] if sys.stdin.isatty() else ["run", "-"]
    first = argv[0]
    if first in ("-e", "-c", "--eval"):
        return ["eval"] + argv[1:]
    if first in ("-i", "--interactive"):
        return ["repl"] + argv[1:]
    if first == "-":
        return ["run"] + argv
    if first.startswith(("--allow-", "--profile", "--sandbox", "--trust", "--engine")):
        # permission flags before -e CODE or a file: `apsi --allow-python app.apsi` = `apsi run --allow-python app.apsi`
        for flag in ("-e", "-c", "--eval"):
            if flag in argv:
                i = argv.index(flag)
                return ["eval"] + argv[:i] + argv[i + 1:]
        if any(t.lower().endswith(".apsi") for t in argv if not t.startswith("-")):
            return ["run"] + argv
    if first.startswith("-") or first in commands:
        return argv
    if first.endswith(".apsi") or os.path.isfile(first):
        return ["run"] + argv
    from .project import load_project, tasks_of
    try:
        root, cfg = load_project()
    except SystemExit:
        root, cfg = None, {}
    if root and first in tasks_of(cfg):
        return ["task"] + argv
    import difflib
    close = difflib.get_close_matches(first, sorted(commands), n=3, cutoff=0.5)
    msg = f"apsi: '{first}' is not a command, a .apsi file or a project task"
    if close:
        msg += f"\n  did you mean: {', '.join(close)}"
    print(msg + "\n  see: apsi --help", file=sys.stderr)
    return None
