"""Measure two Apsi versions on the same machine with the same programs, and write versions.json / versions.md.

    python3 benchmarks/versions/compare.py --old ../apsi-2.0.0 --new . [--web] [--out DIR]

* programs.apsi — array loops, @native code, calls, objects, dictionaries, strings, integer loops (best of 3)
* start-up — `apsi run` of a one-line program, best of 5 (process start to exit)
* check — `apsi check` of every example program with an empty cache (a fresh HOME), best of 3
* web (--web) — requests per second of the same route (benchmarks/web/srv.apsi) at 64 and 256 connections,
  measured with the Go load generator in benchmarks/web (needs `go`)

Both versions run with the Python that runs this script. Checks printed by the programs must agree between the
versions; a row whose checks differ is reported and left out of the summary.
"""
import argparse
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LINE = re.compile(r"^(.+?): ([0-9.]+) ms \[check (.+?)\]$")


def version_of(root):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', open(os.path.join(root, "apsi", "__init__.py")).read())
    return m.group(1) if m else "?"


def run(root, args, env_extra=None, timeout=3600, cwd=None):
    env = dict(os.environ, PYTHONPATH=root, APSI_NATIVE_QUIET="1", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "apsi", *args], capture_output=True, text=True, env=env,
                          timeout=timeout, cwd=cwd or root)


def programs(root):
    r = run(root, ["run", os.path.join(HERE, "programs.apsi")])
    out = {}
    for line in r.stdout.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group(1)] = {"ms": float(m.group(2)), "check": m.group(3)}
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
    return out


def startup(root):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hello.apsi")
        with open(p, "w") as fh:
            fh.write('print("hello")\n')
        run(root, ["run", p])
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            run(root, ["run", p])
            best = min(best, time.perf_counter() - t)
    return best * 1000


def check_cold(root):
    files = sorted(os.path.join(ROOT, "examples", f) for f in os.listdir(os.path.join(ROOT, "examples"))
                   if f.endswith(".apsi"))
    lines = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in files)
    best = 1e9
    for _ in range(3):
        with tempfile.TemporaryDirectory() as home:
            t = time.perf_counter()
            for f in files:
                run(root, ["check", f], {"HOME": home, "USERPROFILE": home})
            best = min(best, time.perf_counter() - t)
    return best * 1000, len(files), lines


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def web(root, load):
    port = free_port()
    with tempfile.TemporaryDirectory() as d:
        src = open(os.path.join(ROOT, "benchmarks", "web", "srv.apsi")).read().replace("58090", str(port))
        p = os.path.join(d, "srv.apsi")
        with open(p, "w") as fh:
            fh.write(src)
        env = dict(os.environ, PYTHONPATH=root)
        srv = subprocess.Popen([sys.executable, "-m", "apsi", "run", p, "--allow-net"], env=env, cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", port), 0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            res = {}
            for conns in (64, 256):
                subprocess.run([load, f"127.0.0.1:{port}", "/hello", str(conns), "2"], capture_output=True)
                o = subprocess.run([load, f"127.0.0.1:{port}", "/hello", str(conns), "5"], capture_output=True,
                                   text=True).stdout
                m = re.search(r"= ([0-9]+) req/s, p50 ([0-9]+) us, p99 ([0-9]+) us, errors ([0-9]+)", o)
                if m:
                    res[conns] = {"rps": int(m.group(1)), "p50_us": int(m.group(2)), "p99_us": int(m.group(3)),
                                  "errors": int(m.group(4))}
            return res
        finally:
            srv.terminate()
            srv.wait(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default=ROOT)
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    old, new = os.path.abspath(a.old), os.path.abspath(a.new)
    vo, vn = version_of(old), version_of(new)
    try:
        import llvmlite  # noqa: F401
        native = True
    except ImportError:
        native = False
    res = {"old": vo, "new": vn, "machine": f"{platform.processor() or platform.machine()}, {os.cpu_count()} threads, "
           f"{platform.system()}, Python {platform.python_version()}", "native": native, "rows": []}

    def row(name, o, n, unit="ms", higher=False, note=""):
        if o is None or n is None:
            return
        ratio = (n / o) if higher else (o / n)
        res["rows"].append({"name": name, "old": round(o, 2), "new": round(n, 2), "unit": unit,
                            "higher": higher, "speedup": round(ratio, 2), "note": note})
        print(f"{name:45} {vo} {o:10.2f} {unit}   {vn} {n:10.2f} {unit}   {ratio:7.2f}x", flush=True)

    po, pn = programs(old), programs(new)
    for name in pn:
        if name not in po:
            continue
        same = po[name]["check"] == pn[name]["check"] or _close(po[name]["check"], pn[name]["check"])
        if not same:
            print(f"{name}: results differ ({po[name]['check']} vs {pn[name]['check']}) - left out", flush=True)
            continue
        note = "" if native or "@native" not in name else "llvmlite not installed: both versions ran it without @native"
        row(name, po[name]["ms"], pn[name]["ms"], note=note)
    row("start-up: apsi run hello.apsi", startup(old), startup(new))
    (co, nf, nl), (cn, _, _) = check_cold(old), check_cold(new)
    row(f"apsi check, {nf} example programs ({nl:,} lines), empty cache", co, cn)
    if a.web:
        load = os.path.join(tempfile.gettempdir(), "apsi-load")
        subprocess.run(["go", "build", "-o", load, "."], cwd=os.path.join(ROOT, "benchmarks", "web"), check=True)
        wo, wn = web(old, load), web(new, load)
        for c in (64, 256):
            if c in wo and c in wn:
                row(f"web server, {c} connections", wo[c]["rps"], wn[c]["rps"], "req/s", higher=True)
    sp = [r["speedup"] for r in res["rows"]]
    res["geomean"] = round(math.exp(sum(math.log(x) for x in sp) / len(sp)), 2) if sp else None
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "versions.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, "versions.md"), "w") as fh:
        fh.write(f"# Apsi {vn} vs {vo}\n\n{res['machine']}\n\n| Program | {vo} | {vn} | |\n|---|---:|---:|---:|\n")
        for r in res["rows"]:
            fh.write(f"| {r['name']} | {r['old']:,} {r['unit']} | {r['new']:,} {r['unit']} | {r['speedup']}× |\n")
        fh.write(f"\nGeometric mean: {res['geomean']}×\n")
    print(f"geometric mean {res['geomean']}x")


def _close(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9)
    except ValueError:
        return False


if __name__ == "__main__":
    main()
