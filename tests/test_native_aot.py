"""The native backend (apsi build --native): each program prints exactly what `apsi run` prints."""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
    expected = os.path.splitext(prog)[0] + ".expected"      # what a program must print, on both backends
    if os.path.exists(expected):
        with open(expected, encoding="utf-8") as f:
            assert got.stdout == f.read()


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
def test_ffi_module(tmp_path):
    """use ffi: ffi.cload and ffi.f64 / ffi.ptr_f64 / ... are the built-in C interface."""
    from apsi.aot.build import build
    prog = os.path.join(HERE, "native_ffi", "app_module.apsi")
    exe = build(prog, str(tmp_path / "app_module"))
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


def test_runtime_version_matches():
    import re
    import apsi
    src = (Path(apsi.__file__).parent / "aot" / "rt" / "apsi_lib.c").read_text(encoding="utf-8")
    assert re.search(r'#define APSI_VERSION "([^"]+)"', src).group(1) == apsi.__version__


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_http_client_native(tmp_path):
    """http_get / http_post_json / redirects / errors against a local server (HTTPS is checked by the Windows test)."""
    import http.server
    import json
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, extra=()):
            self.send_response(code)
            for k, v in extra:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/redir":
                return self._send(302, b"", [("Location", "/hello?x=1")])
            if self.path.startswith("/hello"):
                return self._send(200, ("héllo " + self.path).encode())
            return self._send(404, b"no")

        def do_POST(self):
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, json.dumps({"got": d}).encode())

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    src = tmp_path / "client.apsi"
    src.write_text(f'''let base = "http://127.0.0.1:{port}"
print(http_get(base + "/hello"))
print(http_get(base + "/redir"))
print(http_post_json(base + "/p", {{"a": [1, 2]}}))
try:
    http_get(base + "/missing")
except Exception as e:
    print("error:", e)
''', encoding="utf-8")
    from apsi.aot.build import build
    exe = build(str(src), str(tmp_path / "client"))
    got = subprocess.run([exe], capture_output=True, text=True, encoding="utf-8", timeout=60)
    srv.shutdown()
    assert got.returncode == 0, got.stderr
    assert got.stdout.splitlines() == ["héllo /hello", "héllo /hello?x=1", '{"got": {"a": [1, 2]}}',
                                       "error: HTTP Error 404: Not Found"]


def _native_matches_host(tmp_path, src_text, name):
    from apsi.aot.build import build
    src = tmp_path / f"{name}.apsi"
    src.write_text(src_text, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    want = subprocess.run([sys.executable, "-m", "apsi", "run", "--allow-net", "--allow-env", str(src)], capture_output=True,
                          text=True, cwd=str(tmp_path), env=env, timeout=120)
    exe = build(str(src), str(tmp_path / name))
    got = subprocess.run([exe], capture_output=True, text=True, cwd=str(tmp_path), timeout=120)
    assert got.stdout == want.stdout, (got.stdout, got.stderr, want.stderr)


@pytest.mark.skipif(CC is None or shutil.which("redis-server") is None, reason="needs redis-server")
def test_redis_native(tmp_path):
    import socket
    import time
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no", "--requirepass", "sekret"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        prog = open(os.path.join(HERE, "native_net", "redis.apsi"), encoding="utf-8").read().replace("PORT", str(port))
        _native_matches_host(tmp_path, prog, "redis_client")
    finally:
        srv.terminate()


@pytest.mark.skipif(CC is None or not os.environ.get("APSI_TEST_PG_URL"), reason="set APSI_TEST_PG_URL to a PostgreSQL server")
def test_postgres_native(tmp_path):
    prog = open(os.path.join(HERE, "native_net", "postgres.apsi"), encoding="utf-8").read().replace("PG_URL", os.environ["APSI_TEST_PG_URL"])
    _native_matches_host(tmp_path, prog, "pg_client")


def _dicom_bytes(rows, cols, pixels, z=0.0, inst=1, signed=True):
    """A small uncompressed DICOM file (explicit VR little endian), as scanners write them."""
    import struct

    def el(g, e, vr, val):
        if isinstance(val, str):
            val = val.encode("latin-1")
            if len(val) % 2:
                val += b" "
        if vr in (b"OB", b"OW", b"SQ", b"UN", b"UT"):
            return struct.pack("<HH", g, e) + vr + b"\0\0" + struct.pack("<I", len(val)) + val
        return struct.pack("<HH", g, e) + vr + struct.pack("<H", len(val)) + val
    ts = el(0x0002, 0x0010, b"UI", "1.2.840.10008.1.2.1\0")
    meta = el(0x0002, 0x0000, b"UL", struct.pack("<I", len(ts))) + ts
    body = (el(0x0008, 0x0060, b"CS", "CT") + el(0x0010, 0x0010, b"PN", "Doe^Jane") +
            el(0x0018, 0x0050, b"DS", "2.5") + el(0x0020, 0x0013, b"IS", str(inst)) +
            el(0x0020, 0x0032, b"DS", f"0\\0\\{z}") + el(0x0020, 0x0037, b"DS", "1\\0\\0\\0\\1\\0") +
            el(0x0028, 0x0002, b"US", struct.pack("<H", 1)) + el(0x0028, 0x0010, b"US", struct.pack("<H", rows)) +
            el(0x0028, 0x0011, b"US", struct.pack("<H", cols)) + el(0x0028, 0x0030, b"DS", "0.7\\0.7") +
            el(0x0028, 0x0100, b"US", struct.pack("<H", 16)) + el(0x0028, 0x0101, b"US", struct.pack("<H", 12)) +
            el(0x0028, 0x0103, b"US", struct.pack("<H", 1 if signed else 0)) +
            el(0x0028, 0x1050, b"DS", "40") + el(0x0028, 0x1051, b"DS", "400") +
            el(0x0028, 0x1052, b"DS", "-1024") + el(0x0028, 0x1053, b"DS", "1") +
            el(0x7FE0, 0x0010, b"OW", struct.pack("<%dh" % len(pixels), *pixels)))
    return b"\0" * 128 + b"DICM" + meta + body


@pytest.mark.skipif(CC is None, reason="no C compiler")
def test_dicom_native(tmp_path):
    (tmp_path / "one.dcm").write_bytes(_dicom_bytes(3, 4, [0, 100, 2047, -2048, 5, 6, 7, 8, 1000, 1100, 1200, 4095]))
    series = tmp_path / "series"
    series.mkdir()
    for k, z in enumerate([5.0, 0.0, 2.5]):
        (series / f"s{k}.dcm").write_bytes(_dicom_bytes(2, 2, [k, k + 1, k + 2, k + 3], z=z, inst=k + 1))
    prog = f'''use medical as m
let d = m.read_dicom("{(tmp_path / 'one.dcm').as_posix()}")
print(d.rows, d.columns, d.modality, d.pixel_spacing, d.slice_thickness, d.window_center, d.rescale_intercept, d.bits_stored)
print(d.pixels)
let d2 = m.read_dicom("{(tmp_path / 'one.dcm').as_posix()}", include_identifiers=true)
print(d2.patient_name)
let s = m.read_dicom_series("{series.as_posix()}")
print(s.count, s.spacing, s.positions, s.volume[:, 0, 0])
print(m.window_level(d.pixels, 40, 400))
'''
    _native_matches_host(tmp_path, prog, "dicom")


@pytest.mark.skipif(CC is None, reason="no C compiler")
@pytest.mark.parametrize("name", ["classes", "stdlib", "program", "text", "random_exact", "modules", "regex", "speed"])
def test_converted_python_native(tmp_path, name):
    """A Python program, converted by apsi convert (use pycompat) and built natively, prints what Python prints."""
    import shutil
    from apsi.aot.build import build
    src = os.path.join(HERE, "native_py", name + ".py")
    shutil.copy(src, tmp_path / (name + ".py"))
    conv = subprocess.run([sys.executable, "-m", "apsi", "convert", str(tmp_path / (name + ".py")), "-o",
                           str(tmp_path / (name + ".apsi"))], capture_output=True, text=True, cwd=os.path.dirname(HERE))
    assert conv.returncode == 0, conv.stderr
    want = subprocess.run([sys.executable, src], capture_output=True, text=True, cwd=tmp_path)
    exe = build(str(tmp_path / (name + ".apsi")), str(tmp_path / name))
    got = subprocess.run([exe], capture_output=True, text=True, cwd=tmp_path)
    assert got.stdout == want.stdout, got.stderr
