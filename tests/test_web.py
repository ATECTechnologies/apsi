"""Network-level tests of the Apsi web server: real sockets, malicious inputs, concurrency."""
import asyncio
import os
import socket
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apsi import Interpreter  # noqa: E402

SRC = """
use web
let app = server(name="t", log=false, max_body=1000, header_timeout=1[s], handler_timeout=2[s])
app.health()
app.static("/files", STATIC)
@app.get("/mut")
fn mut(req):
    let seen = req.headers.get("x-seen") != none
    req.headers["x-seen"] = "1"
    return {"a": req.headers["x-a"], "seen": seen}
@app.get("/inject")
fn inject(req) = response("ok", headers={"X-Ok": "yes", "X-Bad": "a\\r\\nEvil: 1"})
@app.post("/echo")
fn echo(req) = {"got": req.json()}
app.compress(min_size=200)
let v1 = app.group("/v1")
@v1.get("/big")
fn big(req) = {"text": "apsi " * 400}
use parallel
@app.get("/slow")
fn slow(req):
    sleep(4[s])
    return "late"
"""


class WebServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(cls.tmp, "www"))
        with open(os.path.join(cls.tmp, "www", "index.html"), "w") as fh:
            fh.write("<h1>hi</h1>")
        with open(os.path.join(cls.tmp, "secret.txt"), "w") as fh:
            fh.write("secret")
        it = Interpreter(permissions={"fs", "net"})
        src = SRC.replace("STATIC", repr(os.path.join(cls.tmp, "www")).replace("'", '"'))
        env = it.run(src, "<web-test>")
        cls.app = env.vars["app"]
        cls.bg = cls.app.run_background(port=0)
        cls.port = cls.bg.port

    @classmethod
    def tearDownClass(cls):
        cls.app.stop()

    def raw(self, data, timeout=5):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        s.sendall(data)
        out = b""
        try:
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                out += chunk
        except socket.timeout:
            pass
        s.close()
        return out

    def status(self, data, timeout=5):
        r = self.raw(data, timeout)
        return int(r.split(b" ", 2)[1]) if r else None

    def test_gzip_and_groups_over_the_wire(self):
        import gzip
        import json
        r = self.raw(b"GET /v1/big HTTP/1.1\r\nHost: x\r\nAccept-Encoding: gzip\r\nConnection: close\r\n\r\n")
        head, _, body = r.partition(b"\r\n\r\n")
        self.assertIn(b"Content-Encoding: gzip", head)
        length = int(next(ln.split(b":")[1] for ln in head.split(b"\r\n") if ln.lower().startswith(b"content-length")))
        self.assertEqual(length, len(body))
        self.assertLess(length, 400)
        self.assertEqual(json.loads(gzip.decompress(body))["text"], "apsi " * 400)
        plain = self.raw(b"GET /v1/big HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        self.assertNotIn(b"Content-Encoding", plain.partition(b"\r\n\r\n")[0])

    def test_basic_and_keepalive(self):
        r = self.raw(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\nGET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        self.assertEqual(r.count(b"HTTP/1.1 200"), 2)
        self.assertIn(b"X-Content-Type-Options: nosniff", r)

    def test_repeated_header_block_is_parsed_once_but_isolated(self):
        req = b"GET /mut HTTP/1.1\r\nHost: x\r\nX-A: 1\r\n\r\n"
        r = self.raw(req * 3 + b"GET /mut HTTP/1.1\r\nHost: x\r\nX-A: 2\r\nConnection: close\r\n\r\n")
        self.assertEqual(r.count(b'{"a":"1","seen":false}'), 3)     # the handler's change does not leak
        self.assertEqual(r.count(b'{"a":"2","seen":false}'), 1)

    def test_header_injection_dropped(self):
        r = self.raw(b"GET /inject HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        head = r.partition(b"\r\n\r\n")[0]
        self.assertNotIn(b"Evil", head)
        self.assertIn(b"X-Ok: yes", head)

    def test_json_echo(self):
        body = b'{"a": [1, 2, 3]}'
        r = self.raw(b"POST /echo HTTP/1.1\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
        self.assertIn(b'"got":{"a":[1,2,3]}', r)

    def test_body_too_large(self):
        self.assertEqual(self.status(b"POST /echo HTTP/1.1\r\nContent-Length: 5000\r\n\r\n" + b"x" * 5000), 413)

    def test_request_smuggling_rejected(self):
        self.assertEqual(self.status(b"POST /echo HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nab"), 400)
        self.assertEqual(self.status(b"POST /echo HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"), 411)

    def test_garbage_request_line(self):
        self.assertEqual(self.status(b"HELLO\r\n\r\n"), 400)
        self.assertEqual(self.status(b"GET /health HTTP/1.1\r\nBad Header\r\n\r\n"), 400)

    def test_header_flood(self):
        big = b"GET /health HTTP/1.1\r\n" + b"".join(b"X-%d: %s\r\n" % (i, b"a" * 200) for i in range(200)) + b"\r\n"
        self.assertEqual(self.status(big), 431)

    def test_slowloris_is_cut_off(self):
        t0 = time.time()
        r = self.raw(b"GET /health HTTP/1.1\r\nHost: x\r\n", timeout=5)  # never finishes headers
        self.assertLess(time.time() - t0, 4)
        self.assertIn(b"408", r)

    def test_path_traversal_blocked(self):
        for p in (b"/files/../secret.txt", b"/files/%2e%2e/secret.txt", b"/files/..%2fsecret.txt"):
            r = self.raw(b"GET " + p + b" HTTP/1.1\r\nConnection: close\r\n\r\n")
            self.assertNotIn(b"secret", r.split(b"\r\n\r\n", 1)[-1], p)
        ok = self.raw(b"GET /files/ HTTP/1.1\r\nConnection: close\r\n\r\n")
        self.assertIn(b"<h1>hi</h1>", ok)

    def test_handler_timeout(self):
        self.assertEqual(self.status(b"GET /slow HTTP/1.1\r\nConnection: close\r\n\r\n", timeout=10), 504)

    def test_many_concurrent_clients(self):
        async def one():
            r, w = await asyncio.open_connection("127.0.0.1", self.port)
            ok = 0
            for _ in range(20):
                w.write(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                await w.drain()
                head = await r.readuntil(b"\r\n\r\n")
                n = int([l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
                await r.readexactly(n)
                ok += head.startswith(b"HTTP/1.1 200")
            w.close()
            return ok

        async def main():
            return await asyncio.gather(*[one() for _ in range(300)])
        res = asyncio.run(main())
        self.assertEqual(sum(res), 300 * 20)


if __name__ == "__main__":
    unittest.main()


class Proxies(unittest.TestCase):
    """trust_proxy / allowed_hosts / force_https / production (Cloudflare and local tunnels)."""

    def make(self, **kw):
        from apsi import web
        app = web.App(log=False)
        app.get("/ip", lambda r: {"client": r.client, "scheme": r.scheme, "peer": r.peer})
        app.production(**kw)
        return app

    def call(self, app, headers, peer):
        return app.handle("GET", "/ip", headers=headers, client=peer)

    def test_cloudflare_edge_is_believed(self):
        app = self.make(domain="example.com", cloudflare=True)
        r = self.call(app, {"Host": "example.com", "CF-Connecting-IP": "1.2.3.4", "X-Forwarded-Proto": "https"},
                      "173.245.48.5")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.body["client"], "1.2.3.4")
        self.assertIn("Strict-Transport-Security", r.headers)

    def test_spoofed_headers_from_anyone_else_are_ignored(self):
        app = self.make(domain="example.com", cloudflare=True)
        r = self.call(app, {"Host": "example.com", "CF-Connecting-IP": "1.2.3.4", "X-Forwarded-Proto": "https"},
                      "8.8.8.8")
        self.assertEqual(r.status, 308)                     # still plain http: sent to https
        self.assertEqual(r.headers["Location"], "https://example.com/ip")
        app2 = self.make(cloudflare=True)
        r = self.call(app2, {"Host": "x", "X-Forwarded-For": "1.2.3.4"}, "8.8.8.8")
        self.assertEqual(r.body["client"], "8.8.8.8")

    def test_local_tunnel_forwarded_for(self):
        app = self.make(domain="example.com", cloudflare=True)
        r = self.call(app, {"Host": "www.example.com", "X-Forwarded-For": "9.9.9.9, 127.0.0.1",
                            "X-Forwarded-Proto": "https"}, "127.0.0.1")
        self.assertEqual((r.status, r.body["client"], r.body["scheme"]), (200, "9.9.9.9", "https"))

    def test_other_host_names_refused(self):
        app = self.make(domain="example.com")
        self.assertEqual(self.call(app, {"Host": "evil.com"}, "127.0.0.1").status, 421)
        self.assertEqual(self.call(app, {"Host": "localhost:8080"}, "127.0.0.1").status, 200)

    def test_rate_limit_counts_the_visitor_not_the_proxy(self):
        app = self.make(cloudflare=True, rate=2)
        codes = [self.call(app, {"Host": "x", "CF-Connecting-IP": f"1.1.1.{i % 2}"}, "173.245.48.5").status
                 for i in range(8)]
        self.assertEqual(codes.count(429), 0)               # two visitors, bursts of 4 each
        codes = [self.call(app, {"Host": "x", "CF-Connecting-IP": "2.2.2.2"}, "173.245.48.5").status
                 for _ in range(8)]
        self.assertIn(429, codes)

    def test_tunnel_command(self):
        from apsi.tunnel import tunnel_command
        self.assertEqual(tunnel_command("cf", 8080), ["cf", "tunnel", "--no-autoupdate", "--url",
                                                       "http://127.0.0.1:8080"])
        self.assertEqual(tunnel_command("cf", 8080, "site")[-2:], ["run", "site"])
