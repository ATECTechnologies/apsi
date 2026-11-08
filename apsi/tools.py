"""Developer commands: `apsi eval`, `apsi watch`, `apsi bench`, `apsi doc`, `apsi completion`."""
import io
import contextlib
import os
import subprocess
import sys
import time

from .errors import ApsiError, CheckFailed


# ---------------------------------------------------------------------------- eval
def cmd_eval(args, make_interp, report):
    """Run code given on the command line; the value of a final expression is printed."""
    src = " ".join(args.code) if isinstance(args.code, list) else args.code
    src = src.replace("\\n", "\n") if "\n" not in src and "\\n" in src else src
    from .interpreter import Env
    it = make_interp(args)
    it.repl_echo = True
    env = Env(it.globals)
    try:
        it.run(src, "<eval>", check=False, env=env, engine="interp")
    except (ApsiError, CheckFailed) as e:
        report(e, src, "<eval>")
        return 1
    return 0


# ---------------------------------------------------------------------------- watch
SKIP_DIRS = {"node_modules", "__pycache__", ".git", ".apsi_cache", "dist", "build", ".venv", "venv"}


def _snapshot(paths, exts):
    out = {}
    for p in paths:
        if os.path.isfile(p):
            try:
                out[p] = os.stat(p).st_mtime_ns
            except OSError:
                pass
            continue
        for root, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for f in files:
                if f.endswith(exts):
                    fp = os.path.join(root, f)
                    try:
                        out[fp] = os.stat(fp).st_mtime_ns
                    except OSError:
                        pass
    return out


def cmd_watch(args):
    """Re-run a command whenever a watched .apsi file changes."""
    paths = args.paths or ["."]
    for p in paths:
        if not os.path.exists(p):
            print(f"apsi watch: '{p}' does not exist", file=sys.stderr)
            return 2
    action = args.do
    files = [p for p in paths if os.path.isfile(p)]
    if action == "run" and len(files) != 1:
        from .project import project_main
        _, main = project_main()
        if not main and not files:
            print("apsi watch: give the file to run (apsi watch app.apsi), or use --do test/check", file=sys.stderr)
            return 2
        target = [files[0] if files else main]
    elif action == "run":
        target = files
    else:
        target = paths
    cmd = [sys.executable, "-m", "apsi", action] + target + list(args.extra or [])
    exts = tuple(args.ext.split(",")) if args.ext else (".apsi", ".toml", ".csv", ".json")
    tty = sys.stdout.isatty()

    def go(reason):
        if tty and not args.no_clear:
            sys.stdout.write("\x1b[2J\x1b[H")
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {reason} → {' '.join(['apsi', action] + target)}", flush=True)
        t0 = time.perf_counter()
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            rc = 130
        ok = "\x1b[32m✓\x1b[0m" if tty else "ok"
        bad = "\x1b[31m✗\x1b[0m" if tty else "failed"
        print(f"\n{ok if rc == 0 else bad} exit {rc} in {time.perf_counter() - t0:.2f}s — watching "
              f"{', '.join(paths)} (Ctrl+C to stop)", flush=True)

    before = _snapshot(paths, exts)
    go("start")
    try:
        while True:
            time.sleep(args.interval)
            now = _snapshot(paths, exts)
            if now != before:
                changed = sorted(set(now.items()) ^ set(before.items()))
                names = sorted({os.path.relpath(f) for f, _ in changed})
                before = now
                time.sleep(0.05)             # let editors finish writing
                before = _snapshot(paths, exts)
                go("changed: " + ", ".join(names[:3]) + (" …" if len(names) > 3 else ""))
    except KeyboardInterrupt:
        print()
        return 0


# ---------------------------------------------------------------------------- bench
def _fmt_t(s):
    if s < 1e-3:
        return f"{s * 1e6:8.1f} µs"
    if s < 1:
        return f"{s * 1e3:8.2f} ms"
    return f"{s:8.3f} s "


def cmd_bench(args, make_interp, read, report):
    src = read(args.file)
    times = []
    for i in range(args.warmup + args.runs):
        it = make_interp(args, args.file)
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                it.run(src, args.file, check=(i == 0))
        except (ApsiError, CheckFailed) as e:
            report(e, src, args.file)
            return 1
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)
        if args.runs >= 5 and sys.stderr.isatty():
            print(f"\r  run {i + 1}/{args.warmup + args.runs}", end="", file=sys.stderr, flush=True)
    if sys.stderr.isatty():
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr)
    times.sort()
    n = len(times)
    mean = sum(times) / n
    sd = (sum((t - mean) ** 2 for t in times) / max(n - 1, 1)) ** 0.5
    print(f"{args.file}  ({n} runs, engine {getattr(args, 'engine', 'compile')})")
    print(f"  best   {_fmt_t(times[0])}")
    print(f"  median {_fmt_t(times[n // 2])}")
    print(f"  mean   {_fmt_t(mean)}  ± {_fmt_t(sd).strip()}")
    print(f"  worst  {_fmt_t(times[-1])}")
    return 0


# ---------------------------------------------------------------------------- doc
def _kind(v):
    from .fmt import type_name
    try:
        return type_name(v)
    except Exception:
        return type(v).__name__


def cmd_doc(args):
    from .interpreter import Interpreter
    from .repl import _signature, _docstring
    it = Interpreter()
    mods = it.modules
    name = args.name
    if not name:
        print("Modules (use NAME to import with `use NAME`):")
        for m in sorted(mods):
            print(f"  {m:<20} {len(mods[m]) if hasattr(mods[m], '__len__') else '':>4} names")
        print(f"\nBuilt-ins: {len([k for k in it.globals.vars if not k.startswith('_')])} names "
              f"(apsi doc builtins).  Details: apsi doc NAME   e.g. apsi doc nn.adam, apsi doc sqrt")
        print("Full language reference: docs/LANGUAGE.md")
        return 0
    if name == "builtins":
        _list({k: v for k, v in it.globals.vars.items() if not k.startswith("_")}, args.filter)
        return 0
    if name in mods:
        print(f"module {name}  —  `use {name}`")
        _list(mods[name], args.filter)
        return 0
    obj, owner = None, None
    if "." in name:
        mod, _, attr = name.rpartition(".")
        if mod in mods and attr in mods[mod]:
            obj, owner = mods[mod][attr], mod
    if obj is None and name in it.globals.vars:
        obj = it.globals.vars[name]
    if obj is None:
        hits = sorted(f"{m}.{k}" for m, d in mods.items() if hasattr(d, "keys") for k in d if k == name)
        if hits:
            obj = mods[hits[0].rpartition(".")[0]][name]
            owner = hits[0].rpartition(".")[0]
    if obj is None:
        import difflib
        pool = list(it.globals.vars) + [f"{m}.{k}" for m, d in mods.items() if hasattr(d, "keys") for k in d]
        close = difflib.get_close_matches(name, pool, n=5)
        from .lexer import KEYWORDS
        if name in KEYWORDS:
            print(f"'{name}' is a keyword — see the 'Statements' chapter of docs/LANGUAGE.md")
            return 0
        print(f"apsi doc: no built-in or module member named '{name}'" +
              (f"\n  did you mean: {', '.join(close)}" if close else ""), file=sys.stderr)
        return 1
    print(f"{name} : {_kind(obj)}" + (f"   (module {owner} — `use {owner}`)" if owner else ""))
    sig = _signature(obj)
    if sig:
        print(f"  {sig}")
    doc = _docstring(obj)
    if doc:
        print()
        for line in doc.strip().splitlines():
            print("  " + line.strip())
    return 0


def _list(d, flt=None):
    items = sorted((k, v) for k, v in d.items() if not k.startswith("_") and (not flt or flt in k))
    if not items:
        print("  (nothing)")
        return
    w = min(max(len(k) for k, _ in items), 28)
    for k, v in items:
        print(f"  {k.ljust(w)}  {_kind(v)}")


# ---------------------------------------------------------------------------- shell completion
def _commands(parser):
    import argparse
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return {name: sorted({o for x in p._actions for o in x.option_strings})
                    for name, p in a.choices.items()}
    return {}


def cmd_completion(args, parser):
    cmds = _commands(parser)
    names = " ".join(sorted(cmds))
    shell = args.shell
    if shell == "bash":
        cases = "\n".join(f'        {c}) opts="{" ".join(o)}";;' for c, o in sorted(cmds.items()))
        print(f'''# Apsi bash completion — add to ~/.bashrc:  eval "$(apsi completion bash)"
_apsi() {{
    local cur cmd opts
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    cmd="${{COMP_WORDS[1]}}"
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "{names}" -- "$cur") $(compgen -f -X '!*.apsi' -- "$cur") )
        return
    fi
    case "$cmd" in
{cases}
        *) opts="";;
    esac
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
    else
        COMPREPLY=( $(compgen -f -- "$cur") )
    fi
}}
complete -o filenames -F _apsi apsi''')
    elif shell == "zsh":
        print(f'''# Apsi zsh completion — add to ~/.zshrc:  eval "$(apsi completion zsh)"
_apsi() {{
    if (( CURRENT == 2 )); then
        compadd {names}
        _files -g '*.apsi'
    else
        _files
    fi
}}
compdef _apsi apsi''')
    elif shell == "fish":
        lines = ["# Apsi fish completion — save as ~/.config/fish/completions/apsi.fish"]
        for c in sorted(cmds):
            lines.append(f"complete -c apsi -n '__fish_use_subcommand' -a {c}")
            for o in cmds[c]:
                if o.startswith("--"):
                    lines.append(f"complete -c apsi -n '__fish_seen_subcommand_from {c}' -l {o[2:]}")
        print("\n".join(lines))
    else:
        table = "\n".join(f"        '{c}' = @({', '.join(repr(o) for o in opts) or ''})"
                          for c, opts in sorted(cmds.items()))
        print(f'''# Apsi PowerShell completion — add to your profile:  apsi completion powershell | Out-String | Invoke-Expression
Register-ArgumentCompleter -Native -CommandName apsi -ScriptBlock {{
    param($wordToComplete, $commandAst, $cursorPosition)
    $opts = @{{
{table}
    }}
    $words = $commandAst.CommandElements | ForEach-Object {{ $_.ToString() }}
    if ($words.Count -le 1 -or ($words.Count -eq 2 -and $wordToComplete)) {{
        $opts.Keys | Where-Object {{ $_ -like "$wordToComplete*" }} | Sort-Object | ForEach-Object {{
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_) }}
        Get-ChildItem -Filter "$wordToComplete*.apsi" -ErrorAction SilentlyContinue | ForEach-Object {{
            [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name) }}
    }} elseif ($wordToComplete -like '-*' -and $opts.ContainsKey($words[1])) {{
        $opts[$words[1]] | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterName', $_) }}
    }}
}}''')
    return 0
