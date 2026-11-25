"""The native backend (apsi build --native): each program prints exactly what `apsi run` prints."""
import glob
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
PROGS = sorted(glob.glob(os.path.join(HERE, "native_aot", "*.apsi")))
def _cc():
    try:
        from apsi.aot.build import find_cc
        return find_cc()                 # the bundled / fetched toolchain on Windows, clang or gcc elsewhere
    except Exception:
        return shutil.which("clang") or shutil.which("gcc")


CC = _cc()


@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize("prog", PROGS, ids=[os.path.basename(p) for p in PROGS])
def test_native_matches_run(prog, tmp_path):
    from apsi.aot.build import build
    exe = build(prog, str(tmp_path / "prog"))
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    want = subprocess.run([sys.executable, "-m", "apsi", "run", prog], capture_output=True, text=True,
                          cwd=str(tmp_path), env=env)                 # files a program writes stay in tmp_path
    got = subprocess.run([exe], capture_output=True, text=True, cwd=str(tmp_path))
    assert got.stdout == want.stdout
    assert got.returncode == want.returncode


@pytest.mark.skipif(CC is None or os.name == "nt", reason="the emulated card runs on POSIX threads (a real card is "
                                                         "tested by tests/native_win/gpu_check.apsi)")
def test_gpu_kernels_emulated(tmp_path, monkeypatch):
    """The card's kernels and the runtime's GPU path, run on host threads (APSI_GPU_EMU)."""
    from apsi.aot.build import build
    monkeypatch.setenv("APSI_GPU_EMU_BUILD", "1")
    exe = build(os.path.join(HERE, "native_aot_gpu.apsi"), str(tmp_path / "gpu"))
    got = subprocess.run([exe], capture_output=True, text=True, env=dict(os.environ, APSI_GPU_EMU="1"))
    assert got.returncode == 0, got.stderr
    lines = got.stdout.splitlines()
    assert lines[0] == "true" and lines[1].startswith("gpu [[6., 6.],")
    assert lines[-1] == "true"
    assert lines[-2].startswith("0.0002282") and lines[-2].endswith("gpu")


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_c_and_cpp_sources(tmp_path):
    """cload("x.c") / cload("y.cpp"): built into a native program, and built once for apsi run."""
    from apsi.aot.build import build
    prog = os.path.join(HERE, "native_ffi", "app.apsi")
    exe = build(prog, str(tmp_path / "app"))
    want = subprocess.run([sys.executable, "-m", "apsi", "run", prog, "--allow-ffi"], capture_output=True, text=True,
                          cwd=os.path.dirname(HERE))
    got = subprocess.run([exe], capture_output=True, text=True)
    assert got.stdout == want.stdout == "32.0\n[10., 20., 30.]\n5\n3.0\n5.0\n"


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_async_event_loop(tmp_path):
    """async fn / await with the native asyncio: the same output as Python's asyncio under apsi run."""
    from apsi.aot.build import build
    exe = build(os.path.join(HERE, "native_async", "tasks.apsi"), str(tmp_path / "tasks"))
    got = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    assert got.stdout == ('done b\ndone c\ndone a\n["a!", "b!", "c!"]\ndone d\nd!\ncaught bad\ntimeout\n3\n42\n')


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_http_server_native(tmp_path):
    """The native HTTP server: routing, JSON, pipelining, limits and the header timeout."""
    import socket
    import time
    from apsi.aot.build import build
    src = tmp_path / "srv.apsi"
    src.write_text('use web\nlet app = server(log=false, header_timeout=1.0)\n'
                   'app.get("/hi/{n}", fn(req) => {"hi": req.params["n"], "q": req.query})\n'
                   'app.post("/sum", fn(req) => {"sum": sum(req.json()["xs"])})\n'
                   'app.run(port=int(getenv("PORT")))\n')
    exe = build(str(src), str(tmp_path / "srv"))
    with socket.socket() as s0:
        s0.bind(("127.0.0.1", 0))
        port = s0.getsockname()[1]
    proc = subprocess.Popen([exe], env=dict(os.environ, PORT=str(port)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            try:
                c = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                time.sleep(0.05)

        def ask(raw):
            c.sendall(raw)
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += c.recv(65536)
            head, _, body = buf.partition(b"\r\n\r\n")
            n = int([h for h in head.split(b"\r\n") if h.lower().startswith(b"content-length")][0].split(b":")[1])
            while len(body) < n:
                body += c.recv(65536)
            return head.split(b"\r\n")[0], body
        line, body = ask(b"GET /hi/x%20y?a=1 HTTP/1.1\r\nHost: t\r\n\r\n")
        assert line == b"HTTP/1.1 200 OK" and body == b'{"hi": "x y", "q": {"a": "1"}}'
        line, body = ask(b"POST /sum HTTP/1.1\r\nHost: t\r\nContent-Length: 14\r\n\r\n{\"xs\":[1,2,3]}")
        assert body == b'{"sum": 6}'
        line, _ = ask(b"POST /sum HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n")
        assert line.startswith(b"HTTP/1.1 501")
        c.close()
        slow = socket.create_connection(("127.0.0.1", port), timeout=5)
        slow.sendall(b"GET / HTTP/1.1\r\n")
        assert slow.recv(100).startswith(b"HTTP/1.1 408")
        slow.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
