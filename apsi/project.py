"""Projects: `apsi new`, `apsi init`, `apsi task` and the apsi.toml file.

apsi.toml
---------
    [project]
    name = "orbit"
    version = "0.1.0"
    main = "src/main.apsi"          # what `apsi run` runs when no file is given

    [run]
    profile = "standard"            # sandbox | strict | standard | developer
    allow = ["read", "write"]       # extra permissions for `apsi run` / `apsi test` in this project

    [tasks]                         # `apsi task NAME` (or just `apsi NAME` when NAME is not a command)
    start = "apsi run src/main.apsi"
    ci = ["apsi check src", "apsi test tests"]   # a list runs in order and stops at the first failure

Commands that start with `apsi ` run with the same Python and Apsi that started the task, so tasks
work even when `apsi` is not on PATH.
"""
import os
import re
import shlex
import subprocess
import sys
import time

CONFIG = "apsi.toml"


# ---------------------------------------------------------------------------- tiny TOML reader
def _parse_value(raw, path, lineno):
    raw = raw.strip()
    if raw.startswith('"""') or raw.startswith("'''"):
        raise ValueError(f"{path}:{lineno}: multi-line strings are not supported in apsi.toml")
    if raw[:1] in ('"', "'"):
        q = raw[0]
        end = raw.find(q, 1)
        while q == '"' and end > 0 and raw[end - 1] == "\\":
            end = raw.find(q, end + 1)
        if end < 0:
            raise ValueError(f"{path}:{lineno}: unterminated string")
        s = raw[1:end]
        return (bytes(s, "utf-8").decode("unicode_escape") if q == '"' and "\\" in s else s), raw[end + 1:]
    if raw.startswith("["):
        items, rest = [], raw[1:].lstrip()
        while True:
            if rest.startswith("]"):
                return items, rest[1:]
            v, rest = _parse_value(rest, path, lineno)
            items.append(v)
            rest = rest.lstrip()
            if rest.startswith(","):
                rest = rest[1:].lstrip()
            elif not rest.startswith("]"):
                raise ValueError(f"{path}:{lineno}: expected ',' or ']' in array")
    m = re.match(r"(true|false|[+-]?\d[\d_]*(\.\d+)?([eE][+-]?\d+)?)", raw)
    if not m:
        raise ValueError(f"{path}:{lineno}: cannot read value '{raw}'")
    tok = m.group(1)
    if tok in ("true", "false"):
        v = tok == "true"
    elif "." in tok or "e" in tok.lower():
        v = float(tok.replace("_", ""))
    else:
        v = int(tok.replace("_", ""))
    return v, raw[m.end():]


def _strip_comment(rest):
    rest = rest.strip()
    if rest and not rest.startswith("#"):
        return rest
    return ""


def read_toml(path):
    try:
        import tomllib                     # Python 3.11+
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ImportError:
        pass
    data, table = {}, None
    table = data
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            name = line.strip("[] \t").split("#")[0].strip()
            table = data
            for part in name.split("."):
                table = table.setdefault(part.strip().strip('"'), {})
            continue
        key, eq, raw = line.partition("=")
        if not eq:
            raise ValueError(f"{path}:{i}: expected 'key = value'")
        # arrays may continue over several lines
        while raw.count("[") > raw.count("]") and i < len(lines):
            raw += " " + lines[i].split("#")[0]
            i += 1
        v, rest = _parse_value(raw, path, i)
        if _strip_comment(rest):
            raise ValueError(f"{path}:{i}: unexpected text after value: {rest.strip()}")
        table[key.strip().strip('"')] = v
    return data


def find_project(start=None):
    """Folder that holds apsi.toml, searching upward from `start` (default: cwd)."""
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(d, CONFIG)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_project(start=None):
    root = find_project(start)
    if not root:
        return None, {}
    try:
        return root, read_toml(os.path.join(root, CONFIG))
    except (OSError, ValueError) as e:
        print(f"apsi: cannot read {os.path.join(root, CONFIG)}: {e}", file=sys.stderr)
        raise SystemExit(2)


# ---------------------------------------------------------------------------- templates
GITIGNORE = """__pycache__/
*.pyc
.apsi_cache/
*_store/
dist/
build/
.env
"""

EDITORCONFIG = """root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.apsi]
indent_style = space
indent_size = 4
"""

TEMPLATES = {
    "app": {
        "desc": "a program with a main file and tests (default)",
        "main": '''# {name} — entry point. Run it with:  apsi run   (or: apsi task start)

fn greet(who: Str) -> Str = f"Hello, {{who}}!"

fn main():
    print(greet("{name}"))
    let g = 9.81[m/s^2]
    let t = 2s
    print(f"a stone falls {{0.5 * g * t ** 2}} in {{t}}")

main()
''',
        "test": '''import "../src/main.apsi"

test "greet says hello":
    assert greet("Apsi") == "Hello, Apsi!"
''',
    },
    "lib": {
        "desc": "a reusable library that other files import",
        "main": '''# {name} — library. Import it with:  import "src/main.apsi" as {ident}

fn clamp(x, lo, hi) = min(max(x, lo), hi)

fn lerp(a, b, t: Float) = a + (b - a) * t

fn mean_of(xs: List) = sum(xs) / len(xs)
''',
        "test": '''import "../src/main.apsi"

test "clamp keeps values in range":
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0

test "lerp works with units":
    assert lerp(0m, 10m, 0.25) ~= 2.5m
''',
    },
    "physics": {
        "desc": "a simulation with units and an ODE solver",
        "main": '''# {name} — damped spring simulated with evolve (units checked before running)

let k = 40[N/m]
let m = 2kg
let c = 1.5[N*s/m]

fn simulate(x0: Length, t_end: Time):
    var x = x0
    var v = 0[m/s]
    evolve x, v over t in 0s..t_end step 1[ms] using rk4 record every 100 into tr:
        x' = v
        v' = -(k * x + c * v) / m
    return x, tr

let x_end, traj = simulate(10[cm], 5s)
print(f"position after 5 s: {{x_end as [mm]}}")
print(f"recorded {{len(traj.t)}} samples")
''',
        "test": '''import "../src/main.apsi"

test "the spring loses energy":
    let x, tr = simulate(10[cm], 2s)
    assert abs(x) < 10[cm]
''',
    },
    "ai": {
        "desc": "a small neural network trained with autodiff",
        "main": '''# {name} — train a small neural network on a 3-class spiral
use nn

let X, y = spiral_data(n_per_class=100, classes=3, noise=0.2, seed=0)
var params = mlp_init([2, 64, 3], seed=42)

fn forward(params, x):
    var h = x
    for i, layer in enumerate(params):
        h = h @ layer.W + layer.b
        if i < len(params) - 1:
            h = relu(h)
    return h

fn loss_fn(params, x, labels) = cross_entropy(forward(params, x), labels)

let opt = adam(lr=0.01)
for step in 0..300:
    let loss, g = value_and_grad(loss_fn)(params, X, y)
    params = opt.step(params, g)
    if step % 100 == 0:
        print(f"step {{step:4}}  loss {{loss:.4f}}")

let acc = accuracy(forward(params, X), y)
print(f"accuracy: {{acc:.3f}}")
''',
        "test": '''import "../src/main.apsi"

test "the network learns":
    assert acc > 0.8
''',
    },
    "server": {
        "desc": "a JSON HTTP API built with `use web`",
        "main": '''# {name} — HTTP API. Start it with:  apsi run --allow-net
use web

let app = server(name="{name}")
app.health()

@app.get("/hello/{{who}}")
fn hello(req):
    let who = req.params["who"]
    return {{"message": f"Hello, {{who}}!"}}

@app.post("/kinetic")
fn kinetic(req):
    let body = req.json()
    let m = body["m"] * u.kg
    let v = body["v"] * u.m / u.s
    return {{"energy": 0.5 * m * v ** 2}}

if not testing():
    app.run(host="127.0.0.1", port=8080)
''',
        "test": None,
    },
}


def _vscode_files(main):
    tasks = {
        "version": "2.0.0",
        "tasks": [
            {"label": "apsi: run", "type": "shell", "command": "apsi", "args": ["run", main],
             "group": {"kind": "build", "isDefault": True}, "problemMatcher": "$apsi"},
            {"label": "apsi: test", "type": "shell", "command": "apsi", "args": ["test"],
             "group": {"kind": "test", "isDefault": True}, "problemMatcher": "$apsi"},
            {"label": "apsi: check", "type": "shell", "command": "apsi", "args": ["check", "src"],
             "problemMatcher": "$apsi"},
            {"label": "apsi: format", "type": "shell", "command": "apsi", "args": ["fmt", "src", "tests"],
             "problemMatcher": []},
            {"label": "apsi: watch", "type": "shell", "command": "apsi", "args": ["watch", main],
             "isBackground": True, "problemMatcher": []},
        ],
    }
    settings = {"files.associations": {"*.apsi": "apsi"}, "[apsi]": {"editor.tabSize": 4,
                "editor.insertSpaces": True, "editor.formatOnSave": True}}
    ext = {"recommendations": ["apsi.apsi-language"]}
    import json
    return {
        ".vscode/tasks.json": json.dumps(tasks, indent=2) + "\n",
        ".vscode/settings.json": json.dumps(settings, indent=2) + "\n",
        ".vscode/extensions.json": json.dumps(ext, indent=2) + "\n",
    }


def _toml(name, main, template):
    allow = '\nallow = ["net"]' if template == "server" else ""
    return f'''[project]
name = "{name}"
version = "0.1.0"
description = ""
main = "{main}"

[run]
profile = "standard"{allow}

[tasks]
start = "apsi run {main}"
test = "apsi test tests"
check = "apsi check src"
fmt = "apsi fmt src tests"
ci = ["apsi check src", "apsi fmt --check src tests", "apsi test tests"]
'''


def scaffold(root, name, template="app", vscode=True, force=False):
    if template not in TEMPLATES:
        raise ValueError(f"unknown template '{template}' (choose from {', '.join(TEMPLATES)})")
    t = TEMPLATES[template]
    ident = re.sub(r"\W", "_", name).strip("_") or "app"
    main = "src/main.apsi"
    files = {
        CONFIG: _toml(name, main, template),
        main: t["main"].format(name=name, ident=ident),
        "README.md": f"# {name}\n\nAn [Apsi](https://github.com/apsi-lang/apsi) project.\n\n"
                     "```\napsi run            # run src/main.apsi\napsi test           # run tests/\n"
                     "apsi task           # list project tasks\napsi watch src      # re-run on every save\n```\n",
        ".gitignore": GITIGNORE,
        ".editorconfig": EDITORCONFIG,
    }
    if t["test"]:
        files["tests/main_test.apsi"] = t["test"]
    else:
        files["tests/main_test.apsi"] = 'import "../src/main.apsi"\n\ntest "the program loads":\n    assert true\n'
    if vscode:
        files.update(_vscode_files(main))
    existing = [f for f in files if os.path.exists(os.path.join(root, f))]
    if existing and not force:
        raise FileExistsError("these files already exist (use --force to overwrite): " + ", ".join(existing))
    for rel, text in files.items():
        p = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return sorted(files)


def cmd_new(args):
    name = args.name
    root = os.path.abspath(name)
    if os.path.exists(root) and os.listdir(root) and not args.force:
        print(f"apsi new: folder '{name}' already exists and is not empty (use --force)", file=sys.stderr)
        return 1
    os.makedirs(root, exist_ok=True)
    return _create(root, os.path.basename(root), args)


def cmd_init(args):
    root = os.path.abspath(args.path or ".")
    os.makedirs(root, exist_ok=True)
    return _create(root, args.name or os.path.basename(root), args)


def _create(root, name, args):
    try:
        files = scaffold(root, name, args.template, vscode=not args.no_vscode, force=args.force)
    except (ValueError, FileExistsError) as e:
        print(f"apsi: {e}", file=sys.stderr)
        return 1
    print(f"Created Apsi project '{name}' ({args.template}) in {root}")
    for f in files:
        print(f"  + {f}")
    rel = os.path.relpath(root)
    print("\nNext:")
    if rel != ".":
        print(f"  cd {rel}")
    print("  apsi run        # run src/main.apsi")
    print("  apsi test       # run the tests")
    print("  apsi task       # list the project's tasks")
    return 0


# ---------------------------------------------------------------------------- tasks
def _apsi_argv():
    return [sys.executable, "-m", "apsi"]


def _expand(cmd):
    """Run `apsi ...` with this interpreter; everything else goes to the system shell."""
    s = cmd.strip()
    if s == "apsi" or s.startswith("apsi "):
        parts = shlex.split(s, posix=(os.name != "nt"))
        return _apsi_argv() + parts[1:], False
    return s, True


def run_command(cmd, cwd, extra=()):
    argv, shell = _expand(cmd)
    if shell and extra:
        argv = argv + " " + " ".join(shlex.quote(x) for x in extra)
    elif not shell:
        argv = argv + list(extra)
    try:
        return subprocess.call(argv, cwd=cwd, shell=shell)
    except OSError as e:
        print(f"apsi task: cannot run '{cmd}': {e}", file=sys.stderr)
        return 127


def tasks_of(cfg):
    t = cfg.get("tasks", {})
    return {k: v for k, v in t.items() if isinstance(v, (str, list))}


def cmd_task(args):
    root, cfg = load_project()
    if not root:
        print("apsi task: no apsi.toml found here or in any parent folder "
              "(create one with `apsi init`)", file=sys.stderr)
        return 2
    tasks = tasks_of(cfg)
    if not args.name:
        if not tasks:
            print(f"{os.path.join(root, CONFIG)} has no [tasks]")
            return 0
        print(f"Tasks in {os.path.join(root, CONFIG)}:")
        w = max(len(k) for k in tasks)
        for k, v in tasks.items():
            shown = v if isinstance(v, str) else "  →  ".join(v)
            print(f"  {k.ljust(w)}   {shown}")
        print("\nRun one with: apsi task NAME [args...]")
        return 0
    if args.name not in tasks:
        import difflib
        close = difflib.get_close_matches(args.name, tasks, n=3)
        print(f"apsi task: no task '{args.name}'" + (f" — did you mean {', '.join(close)}?" if close else
                                                     f" (tasks: {', '.join(tasks) or 'none'})"), file=sys.stderr)
        return 2
    steps = tasks[args.name]
    steps = [steps] if isinstance(steps, str) else steps
    t0 = time.perf_counter()
    for i, step in enumerate(steps):
        extra = args.args if i == len(steps) - 1 else ()
        print(f"\x1b[2m> {step}{(' ' + ' '.join(extra)) if extra else ''}\x1b[0m" if sys.stdout.isatty()
              else f"> {step}", flush=True)
        rc = run_command(step, root, extra)
        if rc:
            print(f"apsi task: '{args.name}' failed at step {i + 1} (exit status {rc})", file=sys.stderr)
            return rc
    if len(steps) > 1:
        print(f"task '{args.name}' finished in {time.perf_counter() - t0:.2f}s")
    return 0


def project_main():
    """(root, main file) of the project around the current folder, or (None, None)."""
    root, cfg = load_project()
    if not root:
        return None, None
    main = (cfg.get("project") or {}).get("main")
    return root, (os.path.join(root, main) if main else None)


def project_run_flags(cfg):
    run = cfg.get("run") or {}
    return run.get("profile"), [a for a in run.get("allow", []) if isinstance(a, str)]
