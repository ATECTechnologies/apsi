"""Measure several Apsi source trees (for example one per commit) on the same machine, to find which change made
what slower. Every round runs every tree once, in rotating order, so drift of the machine (heat, neighbours on a
shared host) spreads evenly over all trees; the table keeps the best of all rounds.

    python3 benchmarks/commits/measure.py --tree base=../apsi-old --tree head=. [--rounds 3] [--out DIR]
    python3 benchmarks/commits/measure.py --commits fb784c9 7810294 HEAD      # git worktrees of these commits

What it measures, with the programs of the newest tree for every tree (so all trees run the same code):

* heavy — benchmarks/heavy/heavy.apsi, the eight @native programs (each best of its own runs)
* programs — benchmarks/versions/programs.apsi: loops, @native, calls, objects, dictionaries, strings
* start-up — `apsi run` of a one-line program (best of 5 per round)
* import — `python -c "import apsi"` (best of 5 per round)
* cold heavy — heavy.apsi with an empty native cache: compile time plus run time, whole process
* check — `apsi check` of every example program with an empty cache

Writes commits.json and commits.md. Exit status 0; regressions are listed at the end of the table (a row more than
--tolerance slower than the first tree).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LINE = re.compile(r"^(.+?): ([0-9.]+) ms\s*\[check (.+?)\s*\]$")


def run(root, args, home, timeout=3600, extra=None):
    env = dict(os.environ, PYTHONPATH=root, APSI_NATIVE_QUIET="1", HOME=home, USERPROFILE=home,
               APSI_IDIOMS=os.environ.get("APSI_IDIOMS", "0"), **(extra or {}))
    t = time.perf_counter()
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=timeout, cwd=root)
    return r, (time.perf_counter() - t) * 1000


def lines(out):
    res = {}
    for line in out.splitlines():
        m = LINE.match(line.strip())
        if m:
            res[m.group(1)] = (float(m.group(2)), m.group(3))
    return res


def measure(root, home, progs, examples, quick):
    out = {}
    for label, src in (("heavy", progs["heavy"]), ("programs", progs["programs"])):
        r, _ = run(root, ["-m", "apsi", "run", src], home)
        if r.returncode != 0:
            sys.stderr.write(f"{root}: {src} failed\n{r.stderr[-1500:]}\n")
        for name, (ms, check) in lines(r.stdout).items():
            out[f"{label}: {name}"] = (ms, check)
    with tempfile.TemporaryDirectory() as d:
        hello = os.path.join(d, "hello.apsi")
        open(hello, "w").write('print("hello")\n')
        run(root, ["-m", "apsi", "run", hello], home)
        out["start-up: apsi run hello.apsi"] = (min(run(root, ["-m", "apsi", "run", hello], home)[1]
                                                    for _ in range(5)), "")
        out["import apsi"] = (min(run(root, ["-c", "import apsi"], home)[1] for _ in range(5)), "")
        if not quick:
            cold = os.path.join(d, "cold")
            os.makedirs(cold)
            r, ms = run(root, ["-m", "apsi", "run", progs["heavy"]], cold)
            out["cold heavy.apsi: compile + run, whole process"] = (ms, "")
            ms = 0.0
            for f in examples:
                ms += run(root, ["-m", "apsi", "check", f], cold)[1]
            out[f"apsi check, {len(examples)} examples, empty cache"] = (ms, "")
    return out


def worktrees(commits, base):
    trees = []
    for c in commits:
        sha = subprocess.run(["git", "rev-parse", "--short", c], capture_output=True, text=True, cwd=ROOT,
                             check=True).stdout.strip()
        path = os.path.join(base, sha)
        if not os.path.isdir(path):
            subprocess.run(["git", "worktree", "add", "--detach", path, sha], cwd=ROOT, check=True,
                           capture_output=True)
        trees.append((sha, path))
    return trees


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", action="append", default=[], help="name=path")
    ap.add_argument("--commits", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="skip the cold compile and check rows")
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="apsi-commits-")
    trees = [tuple(t.split("=", 1)) for t in a.tree] + worktrees(a.commits, os.path.join(tmp, "trees"))
    trees = [(n, os.path.abspath(p)) for n, p in trees]
    progs = {"heavy": os.path.join(ROOT, "benchmarks", "heavy", "heavy.apsi"),
             "programs": os.path.join(ROOT, "benchmarks", "versions", "programs.apsi")}
    exdir = os.path.join(ROOT, "examples")
    examples = sorted(os.path.join(exdir, f) for f in os.listdir(exdir) if f.endswith(".apsi"))
    homes = {n: os.path.join(tmp, "home-" + n) for n, _ in trees}
    for h in homes.values():
        os.makedirs(h, exist_ok=True)
    best, checks = {}, {}
    for rnd in range(a.rounds):
        order = trees[rnd % len(trees):] + trees[:rnd % len(trees)]
        for name, path in order:
            t = time.perf_counter()
            for row, (ms, check) in measure(path, homes[name], progs, examples, a.quick or rnd > 0).items():
                key = (row, name)
                best[key] = min(best.get(key, 1e18), ms)
                checks.setdefault(row, {})[name] = check
            print(f"round {rnd + 1}/{a.rounds} {name}: {time.perf_counter() - t:.0f} s", flush=True)
    names = [n for n, _ in trees]
    rows = list(dict.fromkeys(r for r, _ in best))
    res = {"machine": f"{_cpu()}, {os.cpu_count()} threads, Python {sys.version.split()[0]}", "trees": names,
           "rounds": a.rounds, "rows": []}
    md = [f"Machine: {res['machine']}; best of {a.rounds} rounds; times in ms (lower is better); "
          f"ratio = first tree / this tree.\n", "| Row | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    worse = []
    for row in rows:
        vals = {n: best.get((row, n)) for n in names}
        base = vals[names[0]]
        cells = []
        for n in names:
            v = vals[n]
            if v is None:
                cells.append("—")
                continue
            ratio = base / v if base else 1.0
            cells.append(f"{v:,.1f} ({ratio:.2f}×)")
            if n == names[-1] and ratio < 1 - a.tolerance:
                worse.append((row, ratio))
        same = len(set(c for c in checks[row].values() if c)) <= 1
        md.append(f"| {row}{'' if same else ' (checks differ)'} | " + " | ".join(cells) + " |")
        res["rows"].append({"name": row, "ms": vals, "checks": checks[row]})
    md.append("")
    md.append("Slower in the last tree than in the first: " +
              (", ".join(f"{r} ({x:.2f}×)" for r, x in worse) if worse else "none") + "\n")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "commits.json"), "w"), indent=1)
    open(os.path.join(a.out, "commits.md"), "w").write("\n".join(md))
    print("\n".join(md))
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=ROOT, capture_output=True)


def _cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()
