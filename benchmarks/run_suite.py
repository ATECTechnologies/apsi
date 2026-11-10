# Every benchmark category of the website (apsi.dev/benchmarks) on one machine, several rounds each, raw output kept:
#   python3 benchmarks/run_suite.py [OUT_DIR]        (default: benchmarks-out/)
# Needs gcc, Go and Node.js 22 for the comparison rows and Docker for PostgreSQL / Redis; missing tools only skip rows.
# The website shows the best round of each measurement (the CPU tab comes from benchmarks/heavy/run.py).
import os, subprocess, sys, time, glob, shutil, json
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "benchmarks-out")
os.makedirs(OUT, exist_ok=True)
ENV = {**os.environ, "PYTHONPATH": W, "APSI_NATIVE_QUIET": "1"}
node22 = sorted(glob.glob("/opt/hostedtoolcache/node/22*/x64/bin"))
if node22:
    ENV["PATH"] = node22[-1] + ":" + ENV["PATH"]


def sh(name, cmd, cwd=W, env=None, timeout=1800):
    """Run a shell command, append its output to OUT/name.txt."""
    t = time.time()
    try:
        r = subprocess.run(cmd, shell=True, cwd=cwd, env=env or ENV, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = f"TIMEOUT {e}"
    with open(f"{OUT}/{name}.txt", "a") as fh:
        fh.write(f"### {cmd}  ({time.time() - t:.1f}s)\n{out}\n")
    print(f"[{time.time() - t:6.1f}s] {name}: {cmd[:90]}", flush=True)
    return out


def bg(cmd, cwd=W, env=None):
    return subprocess.Popen(cmd, shell=True, cwd=cwd, env=env or ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def wait_port(p, s=20):
    import socket
    for _ in range(s * 10):
        try:
            socket.create_connection(("127.0.0.1", p), 0.2).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def stop(p):
    import signal
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(10)
    except Exception:
        pass


sh("machine", "lscpu | grep -E 'Model name|^CPU\\(s\\)'; free -g | head -2; python --version; node --version; go version; gcc --version | head -1")
B = W + "/benchmarks"

# ---------------------------------------------------------------- Python, converted
for f in ("nbody", "objs", "strs"):
    sh("setup", f"cd {B}/python && python -m apsi convert {f}.py -o {f}.apsi || python -m apsi convert {f}.py")
for rnd in range(5):
    for f in ("nbody", "objs", "strs"):
        sh(f"python_{f}_cpython", f"python {f}.py", cwd=B + "/python")
        sh(f"python_{f}_apsi", f"python -m apsi run {f}.apsi", cwd=B + "/python")
sh("python_stdlib", "python -m pytest -q tests/test_stdlib_convert.py 2>&1 | tail -3", timeout=3600)

# ---------------------------------------------------------------- matrix multiply
sh("setup", f"gcc -O3 -march=native {B}/matmul/mm.c -o /tmp/mm -lm")
for rnd in range(3):
    sh("matmul_apsi", f"python -m apsi run {B}/matmul/mm.apsi")
    sh("matmul_c", "/tmp/mm")
    sh("matmul_py", f"python {B}/matmul/mm.py")

# ---------------------------------------------------------------- tables
for rnd in range(3):
    sh("dataframe", f"python -I {B}/dataframe/run.py")
    sh("dataframe_csv", f"python -I {B}/dataframe/csv.py")

# ---------------------------------------------------------------- web server
sh("setup", f"cd {B}/web && go build -o /tmp/load .")
open("/tmp/srv_auto.apsi", "w").write(open(f"{B}/web/srv.apsi").read().replace("app.run(port=58090)", 'app.run(port=58090, processes="auto")'))
cases = [("web_asyncio", f"python -m apsi run {B}/web/srv.apsi --allow-net", {"APSI_UVLOOP": "0"}, 58090),
         ("web_uvloop", f"python -m apsi run {B}/web/srv.apsi --allow-net", {}, 58090),
         ("web_auto", "python -m apsi run /tmp/srv_auto.apsi --allow-net", {}, 58090),
         ("web_node", f"node {B}/web/node_server.js", {}, 58092)]
for rnd in range(3):
    for name, cmd, extra, port in cases:
        p = bg(cmd, env={**ENV, **extra})
        if wait_port(port):
            time.sleep(1)
            sh("setup", f"/tmp/load 127.0.0.1:{port} /hello 256 2")
            for conns in (64, 256):
                sh(f"{name}_{conns}", f"/tmp/load 127.0.0.1:{port} /hello {conns} 5")
        stop(p)
        time.sleep(1)
p = bg(f"python -m apsi run {B}/web/srv2.apsi --allow-net")
if wait_port(58091):
    time.sleep(1)
    for rnd in range(3):
        sh("web_blocking", f"(/tmp/load 127.0.0.1:58091 /slow 64 6 > /tmp/slow.txt &) ; sleep 0.5; /tmp/load 127.0.0.1:58091 /hello 64 5; sleep 1; cat /tmp/slow.txt")
    sh("web_1000", "/tmp/load 127.0.0.1:58091 /hello 1000 5")
stop(p)

# ---------------------------------------------------------------- real time
for rnd in range(2):
    sh("rt_sizes", f"python -m apsi run {B}/rt/sizes.apsi", timeout=900)
for rnd in range(3):
    sh("rt_loop", f"python -m apsi run {B}/rt/loop.apsi")

# ---------------------------------------------------------------- AI
for rnd in range(3):
    sh("ai_learn", f"python -m apsi run {B}/ai/learn.apsi --allow-python", timeout=1200)

# ---------------------------------------------------------------- services
p = bg(f"python -m apsi run {B}/websocket/server.apsi --allow-net")
if wait_port(58080):
    time.sleep(1)
    for rnd in range(3):
        sh("ws_client", f"node {B}/websocket/client.mjs")
stop(p)
sh("setup", "docker run -d --name pg -e POSTGRES_USER=apsi -e POSTGRES_PASSWORD=secretpw -p 55432:5432 postgres:16 && "
            "docker run -d --name rd -p 56379:6379 redis:7 redis-server --requirepass redispw")
for _ in range(60):
    if "accepting" in subprocess.run("docker exec pg pg_isready -U apsi", shell=True, capture_output=True, text=True).stdout:
        break
    time.sleep(1)
time.sleep(3)
for rnd in range(3):
    sh("db", f"python -m apsi run {B}/db/run.apsi --allow-net")

sh("setup", "docker rm -f pg rd")
print("raw results in", OUT)
