"""Build and run the heavy CPU benchmarks in Apsi, C, C++, Rust, Go and Mojo with one command; check that every
language computed the same answer; write results.json and results.md.

    python3 benchmarks/heavy/run.py                      everything that is installed, live table in the terminal
    python3 benchmarks/heavy/run.py --skip Mojo Go       leave languages out
    python3 benchmarks/heavy/run.py --only Apsi C        just these
    python3 benchmarks/heavy/run.py --publish 2.0.0      also update apsi.dev/benchmarks live while it runs
    python3 benchmarks/heavy/run.py --apsi-root ../apsi-1.10.0 --label "Apsi 1.10.0"
                                                         measure another Apsi version (no @native: one timed run)
    python3 benchmarks/heavy/run.py --merge old.json     keep languages from an earlier results.json you skip now

The table redraws as each program finishes, so you see every number the moment it exists. With --publish the
website receives the partial results after every program (needs an API token with the `release` permission:
`apsi lib login --token apsi_…`), and visitors of /benchmarks see the numbers appear in real time.

Needs whatever compilers you want on PATH: gcc, g++, rustc, go, mojo (pip install mojo). Missing ones are skipped
and reported. All programs are single-threaded and built with the usual release flags for the machine.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(os.path.dirname(HERE))
LINE = re.compile(r"^(.+?): ([0-9.]+) ms\s+\[check (.+?)\s*\]$")
ORDER = ["nbody-dense 2000x10", "jacobi-2d 1024x200", "matmul 1024", "lu-pivot 800", "ising 512x50",
         "levenshtein 10k", "ode-rk4 20M", "kmeans 200k k16 x20"]
TTY = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


def env_for(root):
    # APSI_IDIOMS=0: every language runs the same loops (Apsi would otherwise run the matmul loops with BLAS;
    # benchmarks/matmul measures that separately)
    return dict(os.environ, APSI_IDIOMS=os.environ.get("APSI_IDIOMS", "0"),
                PYTHONPATH=os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))


def langs(tmp, apsi_root, apsi_src):
    py = [sys.executable, "-m", "apsi"]
    return [
        ("Apsi", None, py + ["run", apsi_src], "apsi run (@native, strict IEEE)", apsi_root),
        ("C", ["gcc", "-O3", "-march=native", "-o", f"{tmp}/c", f"{HERE}/heavy.c", "-lm"], [f"{tmp}/c"],
         "gcc -O3 -march=native", None),
        ("C++", ["g++", "-std=c++17", "-O3", "-march=native", "-o", f"{tmp}/cpp", f"{HERE}/heavy.cpp"], [f"{tmp}/cpp"],
         "g++ -O3 -march=native", None),
        ("Rust", ["rustc", "-C", "opt-level=3", "-C", "target-cpu=native", "-o", f"{tmp}/rs", f"{HERE}/heavy.rs"],
         [f"{tmp}/rs"], "rustc -C opt-level=3 -C target-cpu=native", None),
        ("Go", ["go", "build", "-o", f"{tmp}/go", f"{HERE}/heavy.go"], [f"{tmp}/go"], "go build", None),
        ("Mojo", ["mojo", "build", "-o", f"{tmp}/mojo", f"{HERE}/heavy.mojo"], [f"{tmp}/mojo"], "mojo build", None),
        ("Java", ["javac", "-d", f"{tmp}/java", f"{HERE}/Heavy.java"], ["java", "-cp", f"{tmp}/java", "Heavy"],
         "javac + java (HotSpot JIT)", None),
        ("JavaScript", None, ["node", f"{HERE}/heavy.js"], "node (V8 JIT)", None),
        ("C#", ["dotnet", "build", _csproj(tmp), "-c", "Release", "-o", f"{tmp}/cs", "-nologo", "-v", "q"],
         ["dotnet", f"{tmp}/cs/heavy.dll"], "dotnet build -c Release (RyuJIT)", None),
        ("Julia", None, ["julia", "-O3", f"{HERE}/heavy.jl"], "julia -O3", None),
    ]


def _csproj(tmp):
    d = os.path.join(tmp, "csproj")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(HERE, "heavy.cs"), os.path.join(d, "heavy.cs"))
    path = os.path.join(d, "heavy.csproj")
    with open(path, "w") as fh:
        fh.write('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType>'
                 '<TargetFramework>net8.0</TargetFramework><Optimize>true</Optimize><TieredPGO>true</TieredPGO>'
                 '<InvariantGlobalization>true</InvariantGlobalization><AssemblyName>heavy</AssemblyName>'
                 '</PropertyGroup></Project>')
    return path


def version(name, root):
    cmd = {"Apsi": [sys.executable, "-m", "apsi", "version"], "C": ["gcc", "--version"], "C++": ["g++", "--version"],
           "Rust": ["rustc", "--version"], "Go": ["go", "version"], "Mojo": ["mojo", "--version"],
           "Java": ["java", "-version"], "JavaScript": ["node", "--version"], "C#": ["dotnet", "--version"],
           "Julia": ["julia", "--version"]}[name]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env_for(root or SRC_ROOT), timeout=60)
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "JAVA_TOOL_OPTIONS" not in l]
        return (("Node.js " if name == "JavaScript" else ".NET " if name == "C#" else "") + lines[0].strip())
    except Exception:                       # noqa: BLE001
        return "?"


def interpreted_copy(tmp):
    """heavy.apsi for Apsi versions without @native: same programs, one timed run each (they take minutes)."""
    with open(os.path.join(HERE, "heavy.apsi"), encoding="utf-8") as fh:
        src = fh.read()
    src = re.sub(r"(?m)^@native\s*$", "", src)
    i = src.index("fn timeit(")
    j = src.index('timeit("')
    src = src[:i] + ('fn timeit(name, f):\n    let t = clock()\n    let r = f()\n'
                     '    print(f"{name}: {magnitude(clock() - t) * 1000:.1f} ms  [check {r}]")\n\n') + src[j:]
    path = os.path.join(tmp, "heavy_interpreted.apsi")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


# ------------------------------------------------------------------------------------------------ live table
class Board:
    def __init__(self, names):
        self.names = names
        self.res = {}
        self.status = {n: "waiting" for n in names}
        self.started = {}
        self.lines = 0
        self.lock = threading.Lock()

    def set(self, lang, bench, ms, check):
        with self.lock:
            self.res.setdefault(bench, {})[lang] = {"ms": ms, "check": check}

    def draw(self, final=False):
        if not TTY and not final:
            return
        with self.lock:
            benches = [b for b in ORDER if b in self.res] + [b for b in self.res if b not in ORDER]
            for b in ORDER:
                if b not in benches:
                    benches.append(b)
            w = max(len(b) for b in benches) + 2
            out = [f"{'program':<{w}}" + "".join(f"{n:>11}" for n in self.names)]
            out.append("-" * len(out[0]))
            for b in benches:
                per = self.res.get(b, {})
                best = min((v["ms"] for v in per.values()), default=None)
                row = f"{b:<{w}}"
                for n in self.names:
                    v = per.get(n)
                    if v is None:
                        cell = "…" if self.status[n] == "running" else "·"
                        row += f"{cell:>11}"
                    else:
                        txt = f"{v['ms']:.0f}"
                        if TTY and v["ms"] == best:
                            row += " " * (11 - len(txt)) + f"\033[1;32m{txt}\033[0m"
                        else:
                            row += f"{txt:>11}"
                out.append(row)
            st = []
            for n in self.names:
                s = self.status[n]
                if s == "running":
                    s = f"running {time.time() - self.started.get(n, time.time()):.0f}s"
                st.append(f"{n}: {s}")
            out.append("")
            out.append("  ".join(st))
            text = "\n".join(out)
            if TTY:
                if self.lines:
                    sys.stdout.write(f"\033[{self.lines}F\033[J")
                sys.stdout.write(text + "\n")
                sys.stdout.flush()
                self.lines = text.count("\n") + 1
            else:
                print(text, flush=True)


def ticker(board, stop):
    while not stop.wait(1.0):
        board.draw()


# ------------------------------------------------------------------------------------------------ publishing
def publish(res, version_label, registry, final):
    try:
        sys.path.insert(0, SRC_ROOT)
        from apsi.remote import _auth, _json
        url, auth = _auth(registry, None)
        data = dict(res)
        data["state"] = "final" if final else "running"
        _json(f"{url}/api/v1/benchmarks/{version_label}", auth, "PUT", data, timeout=60)
        return None
    except Exception as e:                  # noqa: BLE001 - publishing must never stop the measurement
        return str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--apsi-root", help="source folder of the Apsi version to measure (default: this one)")
    ap.add_argument("--label", help="name shown for Apsi, e.g. 'Apsi 1.10.0'")
    ap.add_argument("--interpreted", action="store_true",
                    help="run heavy.apsi without @native (automatic for versions that have no native compiler)")
    ap.add_argument("--publish", metavar="VERSION", help="send the results to the website as they arrive")
    ap.add_argument("--registry", help="website address (default https://apsi.dev)")
    ap.add_argument("--merge", help="an earlier results.json: languages not measured now are kept from it")
    ap.add_argument("--out", default=HERE, help="folder for results.json and results.md")
    args = ap.parse_args()

    apsi_root = os.path.abspath(args.apsi_root) if args.apsi_root else SRC_ROOT
    has_native = os.path.isdir(os.path.join(apsi_root, "apsi", "native"))
    interpreted = args.interpreted or not has_native
    res = {"machine": {"cpu": _cpu(), "cores_used": 1, "os": platform.platform(), "python": platform.python_version()},
           "languages": {}, "benchmarks": {}, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.merge:
        with open(args.merge, encoding="utf-8") as fh:
            old = json.load(fh)
    else:
        old = None
    with tempfile.TemporaryDirectory() as tmp:
        apsi_src = interpreted_copy(tmp) if interpreted else os.path.join(HERE, "heavy.apsi")
        todo = []
        for name, build, exe, flags, root in langs(tmp, apsi_root, apsi_src):
            if name in args.skip or (args.only and name not in args.only):
                continue
            tool = (build or exe)[0]
            if tool != sys.executable and not shutil.which(tool):
                print(f"skip {name}: {tool} not found")
                continue
            if name == "Apsi" and interpreted:
                flags = "apsi run (no native compiler in this version: bytecode engine, one timed run)"
            todo.append((name, build, exe, flags, root))
        names = [t[0] for t in todo]
        if old:
            for n, meta in old.get("languages", {}).items():
                if n not in names:
                    res["languages"][n] = dict(meta, note="from an earlier run on the same machine")
                    for b, per in old.get("benchmarks", {}).items():
                        if n in per:
                            res["benchmarks"].setdefault(b, {})[n] = per[n]
        board = Board(names + [n for n in res["languages"] if n not in names])
        for b, per in res["benchmarks"].items():
            for n, v in per.items():
                board.set(n, b, v["ms"], v["check"])
                board.status[n] = "earlier run"
        stop = threading.Event()
        threading.Thread(target=ticker, args=(board, stop), daemon=True).start()
        pub_err = None
        try:
            for name, build, exe, flags, root in todo:
                label = args.label if (name == "Apsi" and args.label) else None
                res["languages"][name] = {"version": label or version(name, root), "flags": flags}
                board.status[name] = "building" if build else "running"
                board.draw()
                if build:
                    b_ = subprocess.run(build, capture_output=True, text=True)
                    if b_.returncode:                       # one compiler failing must not stop the others
                        print(f"skip {name}: build failed\n{(b_.stdout + b_.stderr)[-2000:]}", flush=True)
                        res["languages"].pop(name, None)
                        board.status[name] = "build failed"
                        continue
                board.status[name] = "running"
                board.started[name] = time.time()
                proc = subprocess.Popen(exe, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                        env=env_for(root or SRC_ROOT), bufsize=1)
                for line in proc.stdout:
                    m = LINE.match(line.strip())
                    if not m:
                        continue
                    ms, check = float(m.group(2)), m.group(3)
                    res["benchmarks"].setdefault(m.group(1), {})[name] = {"ms": ms, "check": check}
                    board.set(name, m.group(1), ms, check)
                    board.draw()
                    if args.publish:
                        pub_err = publish(res, args.publish, args.registry, False) or pub_err
                err = proc.stderr.read()
                if proc.wait() != 0:
                    board.status[name] = "failed"
                    print(f"\n{name} failed:\n{err[-2000:]}", file=sys.stderr)
                else:
                    board.status[name] = f"done in {time.time() - board.started[name]:.0f}s"
                board.draw()
        finally:
            stop.set()
    bad = []
    for bname, per in res["benchmarks"].items():
        vals = {k: v["check"] for k, v in per.items()}
        ref = vals.get("C") or next(iter(vals.values()))
        for k, v in vals.items():
            try:
                same = abs(float(v) - float(ref)) <= 1e-9 * max(1.0, abs(float(ref)))
            except ValueError:
                same = v == ref
            if not same:
                bad.append(f"{bname}: {k} gave {v}, C gave {ref}")
    res["all_checks_agree"] = not bad
    res["benchmarks"] = {b: res["benchmarks"][b] for b in ORDER if b in res["benchmarks"]}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    names = list(res["languages"])
    md = ["| Benchmark | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for bname, per in res["benchmarks"].items():
        best = min(v["ms"] for v in per.values())
        cells = []
        for n in names:
            v = per.get(n)
            cells.append("—" if v is None else (f"**{v['ms']:.0f}**" if v["ms"] == best else f"{v['ms']:.0f}"))
        md.append(f"| {bname} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.out, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    if not TTY:
        print("\n".join(md))
    print("checksums agree across languages" if not bad else "CHECKSUM MISMATCH:\n" + "\n".join(bad))
    if args.publish:
        e = publish(res, args.publish, args.registry, True)
        print(f"published to the website as {args.publish}" if not e else f"could not publish: {e or pub_err}")
    return 0 if not bad else 1


def _cpu():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    sys.exit(main())
