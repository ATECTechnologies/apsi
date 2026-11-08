"""`use web` — a hardened, high-concurrency HTTP server framework for Apsi.

    use web
    let app = server()

    @app.get("/hello/{name}")
    fn hello(req) = {"message": "hi " + req.params["name"]}

    app.run(port=8080)

Architecture: an asyncio event loop accepts and parses connections (thousands of
concurrent keep-alive clients at low memory cost); handlers run on a worker thread
pool so numpy/AI work runs in parallel without blocking the loop. On Linux,
`processes=N` forks N copies that share the port (SO_REUSEPORT) to use every core.

Secure by default: request size limits, header/body timeouts (slowloris protection),
per-client rate limiting, connection caps, strict path matching, security headers,
no stack traces leaked to clients, optional TLS 1.2+, JWT (HS256) and API-key helpers,
path-traversal-safe static files, and graceful shutdown.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import signal
import socket
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, parse_qs

from .errors import ApsiError, RuntimeErr
from .values import Record

STATUS = {200: "OK", 201: "Created", 202: "Accepted", 204: "No Content", 301: "Moved Permanently",
          302: "Found", 304: "Not Modified", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
          404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout", 409: "Conflict",
          413: "Payload Too Large", 415: "Unsupported Media Type", 422: "Unprocessable Entity",
          429: "Too Many Requests", 431: "Request Header Fields Too Large", 500: "Internal Server Error",
          503: "Service Unavailable", 504: "Gateway Timeout"}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


# =========================================================================== request / response
class Request:
    __slots__ = ("method", "path", "query", "headers", "body", "params", "client", "id", "started", "user")

    def __init__(self, method, path, query, headers, body, client):
        self.method, self.path, self.query, self.headers = method, path, query, headers
        self.body, self.client = body, client
        self.params = {}
        self.id = uuid.uuid4().hex[:16]
        self.started = time.perf_counter()
        self.user = None

    @property
    def text(self):
        return self.body.decode("utf-8", "replace")

    def json(self):
        from .interop import from_py
        if not self.body:
            return None
        try:
            return from_py(json.loads(self.body.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            raise HTTPError(400, "request body is not valid JSON")

    def header(self, name, default=None):
        return self.headers.get(name.lower(), default)

    def __repr__(self):
        return f"<request {self.method} {self.path}>"


class Response:
    __slots__ = ("status", "headers", "body")

    def __init__(self, body=b"", status=200, headers=None, content_type=None):
        self.status = int(status)
        self.headers = dict(headers or {})
        if isinstance(body, str):
            body = body.encode("utf-8")
            self.headers.setdefault("Content-Type", content_type or "text/plain; charset=utf-8")
        elif content_type:
            self.headers.setdefault("Content-Type", content_type)
        self.body = body or b""


class HTTPError(BaseException):
    # BaseException so it travels through user code untouched in both engines
    def __init__(self, status, message, headers=None):
        super().__init__(message)
        self.status, self.message, self.headers = status, message, headers or {}


def response(body="", status=200, headers=None, type=None):
    """Build a response explicitly: response("created", status=201, headers={"X-A": "1"})."""
    if isinstance(body, (dict, list, Record)) or not isinstance(body, (str, bytes)):
        return _json_response(body, status, headers)
    return Response(body, status, headers, type)


def redirect(location, status=302):
    return Response(b"", status, {"Location": str(location)})


def abort(status, message="error"):
    """Abort the current request with an HTTP error: abort(404, "no such user")."""
    raise HTTPError(int(status), str(message))


def _clean(o):
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None                                   # JSON has no NaN/Infinity
    if isinstance(o, list):
        return [_clean(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    return o


def _json_response(obj, status=200, headers=None):
    from .interop import to_py
    body = json.dumps(_clean(to_py(obj)), ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    h = {"Content-Type": "application/json; charset=utf-8"}
    h.update(headers or {})
    return Response(body, status, h)


def _to_response(r):
    if isinstance(r, Response):
        return r
    if r is None:
        return Response(b"", 204)
    if isinstance(r, bytes):
        return Response(r, 200, content_type="application/octet-stream")
    if isinstance(r, str):
        return Response(r)
    return _json_response(r)


# =========================================================================== routing
_PARAM = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::(int|float|str|path))?\}")
_CONVERT = {"int": (r"-?\d+", int), "float": (r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", float),
            "str": (r"[^/]+", str), "path": (r".+", str)}


class Route:
    def __init__(self, method, pattern, handler):
        self.method, self.pattern, self.handler = method, pattern, handler
        self.converters = {}
        regex, pos = "", 0
        for m in _PARAM.finditer(pattern):
            regex += re.escape(pattern[pos:m.start()])
            name, kind = m.group(1), m.group(2) or "str"
            rx, conv = _CONVERT[kind]
            regex += f"(?P<{name}>{rx})"
            self.converters[name] = conv
            pos = m.end()
        regex += re.escape(pattern[pos:])
        self.regex = re.compile("^" + regex + "$")

    def match(self, path):
        m = self.regex.match(path)
        if not m:
            return None
        try:
            return {k: self.converters[k](unquote(v)) for k, v in m.groupdict().items()}
        except ValueError:
            return None


class RateLimiter:
    """Token bucket per client: `rate` requests per `per` seconds, bursts up to `burst`."""

    def __init__(self, rate, per=1.0, burst=None):
        self.rate = float(rate) / float(per)
        self.burst = float(burst if burst is not None else rate)
        self.buckets = {}
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self.buckets[key] = (tokens, now)
                return False, (1.0 - tokens) / self.rate
            self.buckets[key] = (tokens - 1.0, now)
            if len(self.buckets) > 100_000:            # bound memory under attack
                cutoff = now - 600
                self.buckets = {k: v for k, v in self.buckets.items() if v[1] > cutoff}
            return True, 0.0


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.by_status = {}
        self.latencies = []
        self.started = time.time()
        self.active = 0

    def record(self, status, seconds):
        with self.lock:
            self.requests += 1
            if status >= 500:
                self.errors += 1
            self.by_status[status] = self.by_status.get(status, 0) + 1
            self.latencies.append(seconds)
            if len(self.latencies) > 10000:
                self.latencies = self.latencies[-5000:]

    def snapshot(self):
        with self.lock:
            lat = sorted(self.latencies)

            def pct(p):
                return round(lat[min(len(lat) - 1, int(p * len(lat)))] * 1000, 3) if lat else 0.0
            up = time.time() - self.started
            return {"requests": self.requests, "errors": self.errors,
                    "status": {str(k): v for k, v in sorted(self.by_status.items())},
                    "latency_ms": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99)},
                    "active_connections": self.active, "uptime_s": round(up, 1),
                    "requests_per_s": round(self.requests / up, 1) if up > 0 else 0.0}


# =========================================================================== application
class App:
    def __init__(self, name="apsi", max_body=1 << 20, max_header=16 << 10, header_timeout=10.0,
                 body_timeout=30.0, handler_timeout=60.0, keepalive_timeout=15.0, max_connections=10000,
                 workers=None, security_headers=True, log=True, interp=None):
        self.name = name
        self.routes = []
        self.before_hooks = []
        self.after_hooks = []
        self.static_dirs = []
        self.limiter = None
        self.cors_origins = None
        self.max_body = int(max_body)
        self.max_header = int(max_header)
        self.header_timeout = float(header_timeout)
        self.body_timeout = float(body_timeout)
        self.handler_timeout = float(handler_timeout)
        self.keepalive_timeout = float(keepalive_timeout)
        self.max_connections = int(max_connections)
        self.workers = int(workers) if workers else max(4, (os.cpu_count() or 1) * 4)
        self.security_headers = bool(security_headers)
        self.log = bool(log)
        self.metrics = Metrics()
        self.interp = interp
        self._executor = None
        self._server = None
        self._loop = None
        self._stopping = False
        self.tls = False

    # ------------------------------------------------------------- registration
    def route(self, method, path, handler=None):
        def add(h):
            self.routes.append(Route(method.upper(), str(path), h))
            return h
        return add(handler) if handler is not None else add

    def get(self, path, handler=None):
        return self.route("GET", path, handler)

    def post(self, path, handler=None):
        return self.route("POST", path, handler)

    def put(self, path, handler=None):
        return self.route("PUT", path, handler)

    def patch(self, path, handler=None):
        return self.route("PATCH", path, handler)

    def delete(self, path, handler=None):
        return self.route("DELETE", path, handler)

    def before(self, hook):
        """hook(req) runs before every handler; return a response to short-circuit (e.g. auth)."""
        self.before_hooks.append(hook)
        return hook

    def after(self, hook):
        """hook(req, resp) runs after every handler (logging, headers)."""
        self.after_hooks.append(hook)
        return hook

    def rate_limit(self, rate, per=1.0, burst=None):
        """Limit each client IP to `rate` requests per `per` seconds (429 + Retry-After otherwise)."""
        per = float(per.v) if hasattr(per, "v") else float(per)
        self.limiter = RateLimiter(rate, per, burst)
        return self

    def cors(self, origins="*"):
        """Allow cross-origin browser calls from the given origins ("*" or a list)."""
        self.cors_origins = origins if origins == "*" else list(origins)
        return self

    def static(self, prefix, directory):
        """Serve files under `directory` at URL `prefix` (path traversal is impossible)."""
        if self.interp is not None:
            from .security import check_path
            directory = check_path(self.interp, directory, "read", "serving static files")
        self.static_dirs.append(("/" + str(prefix).strip("/"), os.path.realpath(directory)))
        return self

    def health(self, path="/health"):
        self.get(path, lambda req: {"status": "ok", "name": self.name})
        return self

    def expose_metrics(self, path="/metrics"):
        self.get(path, lambda req: self.metrics.snapshot())
        return self

    def stats(self):
        return self.metrics.snapshot()

    # ------------------------------------------------------------- dispatch (thread)
    def dispatch(self, req):
        """Route a parsed request to its handler; always returns a Response."""
        try:
            if self.limiter is not None:
                ok, wait = self.limiter.allow(req.client)
                if not ok:
                    raise HTTPError(429, "rate limit exceeded", {"Retry-After": str(max(1, int(wait + 0.999)))})
            if req.method == "OPTIONS" and self.cors_origins is not None:
                return self._finish(req, Response(b"", 204))
            for hook in self.before_hooks:
                r = hook(req)
                if r is not None:
                    return self._finish(req, _to_response(r))
            allowed = []
            for route in self.routes:
                params = route.match(req.path)
                if params is None:
                    continue
                if route.method != req.method and not (route.method == "GET" and req.method == "HEAD"):
                    allowed.append(route.method)
                    continue
                req.params = params
                resp = _to_response(route.handler(req))
                return self._finish(req, resp)
            static = self._static(req)
            if static is not None:
                return self._finish(req, static)
            if allowed:
                raise HTTPError(405, "method not allowed", {"Allow": ", ".join(sorted(set(allowed)))})
            raise HTTPError(404, "not found")
        except HTTPError as e:
            return self._finish(req, _json_response({"error": e.message, "status": e.status}, e.status, e.headers))
        except ApsiError as e:
            if e.kind in ("UnitError", "ShapeError", "TypeError"):      # bad input from the client
                return self._finish(req, _json_response({"error": e.msg, "kind": e.kind, "status": 400}, 400))
            self._log_internal(req, e)
            return self._finish(req, _json_response({"error": "internal server error", "id": req.id}, 500))
        except Exception as e:
            err = e
            comp = getattr(self.interp, "last_compiled", None) if self.interp is not None else None
            if comp is not None:
                try:
                    err = comp.translate(e, sys.exc_info()[2])
                    if err.filename is None:
                        err.filename = comp.filename
                except Exception:
                    err = e
            self._log_internal(req, err)
            return self._finish(req, _json_response({"error": "internal server error", "id": req.id}, 500))

    def _log_internal(self, req, err):
        msg = err.format() if isinstance(err, ApsiError) else "".join(
            traceback.format_exception(type(err), err, err.__traceback__))
        sys.stderr.write(f"[{self.name}] internal error in {req.method} {req.path} (request id {req.id}):\n"
                         f"  {msg}\n")

    def _static(self, req):
        if req.method not in ("GET", "HEAD"):
            return None
        for prefix, root in self.static_dirs:
            if req.path == prefix or req.path.startswith(prefix + "/"):
                rel = req.path[len(prefix):].lstrip("/") or "index.html"
                full = os.path.realpath(os.path.join(root, unquote(rel)))
                if not (full == root or full.startswith(root + os.sep)):
                    raise HTTPError(403, "forbidden")
                if os.path.isdir(full):
                    full = os.path.join(full, "index.html")
                if not os.path.isfile(full):
                    return None
                ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
                with open(full, "rb") as fh:
                    data = fh.read()
                etag = '"' + hashlib.sha256(data).hexdigest()[:16] + '"'
                if req.headers.get("if-none-match") == etag:
                    return Response(b"", 304, {"ETag": etag})
                h = {"Content-Type": ctype, "ETag": etag, "Cache-Control": "public, max-age=300",
                     "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'"}
                return Response(data, 200, h)
        return None

    def _finish(self, req, resp):
        if self.cors_origins is not None:
            origin = req.headers.get("origin")
            if self.cors_origins == "*":
                resp.headers.setdefault("Access-Control-Allow-Origin", "*")
            elif origin in self.cors_origins:
                resp.headers.setdefault("Access-Control-Allow-Origin", origin)
                resp.headers.setdefault("Vary", "Origin")
            resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
            resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if self.security_headers:
            for k, v in SECURITY_HEADERS.items():
                resp.headers.setdefault(k, v)
            if self.tls:
                resp.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
        resp.headers.setdefault("X-Request-Id", req.id)
        for hook in self.after_hooks:
            try:
                hook(req, resp)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        dt = time.perf_counter() - req.started
        self.metrics.record(resp.status, dt)
        if self.log:
            sys.stdout.write(f"{time.strftime('%H:%M:%S')} {req.client} {req.method} {req.path} "
                             f"{resp.status} {dt * 1000:.1f}ms\n")
        return resp

    # in-process testing (no network needed)
    def handle(self, method, path, body=None, headers=None, client="127.0.0.1"):
        """Call the app directly — handy for tests: app.handle("GET", "/hello/x")."""
        if body is not None and not isinstance(body, (bytes, str)):
            from .interop import to_py
            body = json.dumps(to_py(body))
        if isinstance(body, str):
            body = body.encode("utf-8")
        path, _, qs = str(path).partition("?")
        req = Request(method.upper(), path, _query(qs),
                      {k.lower(): str(v) for k, v in (headers or {}).items()}, body or b"", client)
        resp = self.dispatch(req)
        ctype = resp.headers.get("Content-Type", "")
        data = resp.body
        if ctype.startswith("application/json") and data:
            from .interop import from_py
            data = from_py(json.loads(data.decode("utf-8")))
        elif ctype.startswith("text/"):
            data = data.decode("utf-8")
        return Record(status=resp.status, body=data, headers=dict(resp.headers))

    # ------------------------------------------------------------- network server
    def run(self, host="127.0.0.1", port=8080, tls_cert=None, tls_key=None, processes=1, workers=None):
        """Start serving (blocks). processes > 1 forks workers sharing the port (Linux)."""
        if self.interp is not None:
            from .security import check_net
            check_net(self.interp, host, int(port), "app.run() (listening)")
        if workers:
            self.workers = int(workers)
        ssl_ctx = None
        if tls_cert:
            import ssl
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_ctx.load_cert_chain(tls_cert, tls_key)
            ssl_ctx.set_alpn_protocols(["http/1.1"])
            self.tls = True
        processes = int(processes or 1)
        sock = _bind(host, int(port), reuse_port=processes > 1)
        scheme = "https" if ssl_ctx else "http"
        print(f"{self.name} listening on {scheme}://{host}:{sock.getsockname()[1]}  "
              f"({processes} process{'es' if processes > 1 else ''} × {self.workers} worker threads)")
        if processes > 1 and hasattr(os, "fork"):
            children = []
            for _ in range(processes - 1):
                pid = os.fork()
                if pid == 0:
                    self._serve(sock, ssl_ctx)
                    os._exit(0)
                children.append(pid)
            try:
                self._serve(sock, ssl_ctx)
            finally:
                for pid in children:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                for pid in children:
                    try:
                        os.waitpid(pid, 0)
                    except OSError:
                        pass
        else:
            self._serve(sock, ssl_ctx)
        return 0

    def run_background(self, host="127.0.0.1", port=0):
        """Start the server on a background thread; returns record(port, stop)."""
        sock = _bind(host, int(port))
        ready = threading.Event()

        def target():
            self._serve(sock, None, ready)
        t = threading.Thread(target=target, daemon=True)
        t.start()
        ready.wait(10)
        return Record(port=sock.getsockname()[1], stop=self.stop, url=f"http://{host}:{sock.getsockname()[1]}")

    def stop(self):
        self._stopping = True
        if self._loop is not None and self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)

    def _serve(self, sock, ssl_ctx, ready=None):
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=f"{self.name}-w")
        loop.set_default_executor(self._executor)

        async def main():
            self._server = await asyncio.start_server(self._client, sock=sock, ssl=ssl_ctx,
                                                      limit=self.max_header + 1024, backlog=2048)
            if threading.current_thread().name == "apsi-program":
                # running under apsi.runner: Ctrl+C / SIGTERM arrive in the main thread and call stop()
                from . import runner
                runner.add_stop_hook(self.stop)
                hooked.append(True)
            elif threading.current_thread() is threading.main_thread():
                for sig in ("SIGINT", "SIGTERM"):
                    if hasattr(signal, sig):
                        try:
                            loop.add_signal_handler(getattr(signal, sig), self.stop)
                        except (NotImplementedError, RuntimeError):
                            pass
            if ready is not None:
                ready.set()
            async with self._server:
                try:
                    await self._server.serve_forever()
                except asyncio.CancelledError:
                    pass
        hooked = []
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            pass
        finally:
            if hooked:
                from . import runner
                runner.remove_stop_hook(self.stop)
            # graceful: let in-flight handlers finish
            self._executor.shutdown(wait=True)
            loop.close()

    async def _client(self, reader, writer):
        peer = writer.get_extra_info("peername")
        client = peer[0] if isinstance(peer, tuple) else "unknown"
        if self.metrics.active >= self.max_connections:
            writer.close()
            return
        self.metrics.active += 1
        loop = asyncio.get_running_loop()
        try:
            first = True
            while not self._stopping:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"),
                                                  self.header_timeout if first else self.keepalive_timeout)
                except asyncio.TimeoutError:
                    if first:
                        await self._send(writer, Response(b'{"error":"request timeout"}', 408,
                                                          {"Content-Type": "application/json"}), False)
                    break
                except asyncio.LimitOverrunError:
                    await self._send(writer, Response(b'{"error":"headers too large"}', 431,
                                                      {"Content-Type": "application/json"}), False)
                    break
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                first = False
                try:
                    method, target, version, headers = _parse_head(head, self.max_header)
                except HTTPError as e:
                    await self._send(writer, _json_response({"error": e.message}, e.status), False)
                    break
                keep = version == "HTTP/1.1" and headers.get("connection", "").lower() != "close"
                if headers.get("transfer-encoding", "").lower() not in ("", "identity"):
                    await self._send(writer, _json_response({"error": "chunked request bodies are not supported"}, 411), False)
                    break
                try:
                    n = int(headers.get("content-length", "0") or 0)
                except ValueError:
                    n = -1
                if n < 0:
                    await self._send(writer, _json_response({"error": "bad Content-Length"}, 400), False)
                    break
                if n > self.max_body:
                    await self._send(writer, _json_response({"error": "payload too large"}, 413), False)
                    break
                try:
                    body = await asyncio.wait_for(reader.readexactly(n), self.body_timeout) if n else b""
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    await self._send(writer, _json_response({"error": "request timeout"}, 408), False)
                    break
                path, _, qs = target.partition("?")
                if not path.startswith("/") or "\x00" in path:
                    await self._send(writer, _json_response({"error": "bad request target"}, 400), False)
                    break
                req = Request(method, unquote(path) if "%2f" not in path.lower() else path, _query(qs),
                              headers, body, client)
                try:
                    resp = await asyncio.wait_for(loop.run_in_executor(None, self.dispatch, req),
                                                  self.handler_timeout)
                except asyncio.TimeoutError:
                    resp = _json_response({"error": "handler timed out", "id": req.id}, 504)
                await self._send(writer, resp, keep, head_only=method == "HEAD")
                if not keep:
                    break
        finally:
            self.metrics.active -= 1
            try:
                writer.close()
            except Exception:
                pass

    async def _send(self, writer, resp, keep, head_only=False):
        status = resp.status
        lines = [f"HTTP/1.1 {status} {STATUS.get(status, 'Status')}"]
        h = resp.headers
        h["Content-Length"] = str(len(resp.body))
        h["Connection"] = "keep-alive" if keep else "close"
        h.setdefault("Server", "apsi")
        for k, v in h.items():
            if "\r" in str(v) or "\n" in str(v):        # header injection guard
                continue
            lines.append(f"{k}: {v}")
        data = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace")
        try:
            writer.write(data if head_only else data + resp.body)
            await writer.drain()
        except (ConnectionError, RuntimeError):
            pass


def _bind(host, port, reuse_port=False):
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    if os.name != "nt":
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    else:
        s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR), 1)
    if reuse_port and hasattr(socket, "SO_REUSEPORT"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind((host, port))
    s.listen(2048)
    s.setblocking(False)
    return s


_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _parse_head(head, limit):
    if len(head) > limit:
        raise HTTPError(431, "headers too large")
    try:
        text = head.decode("latin-1")
    except UnicodeDecodeError:
        raise HTTPError(400, "bad header encoding")
    lines = text.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1.") or not _TOKEN.match(parts[0]):
        raise HTTPError(400, "bad request line")
    method, target, version = parts
    headers = {}
    for line in lines[1:]:
        if not line:
            continue
        if line[0] in " \t":
            raise HTTPError(400, "obsolete header folding is not allowed")
        k, sep, v = line.partition(":")
        if not sep or not _TOKEN.match(k):
            raise HTTPError(400, "bad header")
        k = k.lower()
        if k in headers and k in ("content-length", "host", "transfer-encoding"):
            raise HTTPError(400, f"duplicate {k} header")     # request-smuggling guard
        headers[k] = v.strip()
    if len(headers) > 100:
        raise HTTPError(431, "too many headers")
    return method.upper(), target, version, headers


def _query(qs):
    if not qs:
        return {}
    return {k: (v[0] if len(v) == 1 else v) for k, v in parse_qs(qs, keep_blank_values=True).items()}


# =========================================================================== auth helpers
def _b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64u_dec(s):
    s = str(s)
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def jwt_sign(claims, secret, expires_in=3600):
    """Create an HS256 JSON Web Token. expires_in: seconds (number or Time quantity)."""
    from .interop import to_py
    exp = float(expires_in.v) if hasattr(expires_in, "v") else float(expires_in)
    payload = dict(to_py(claims))
    now = int(time.time())
    payload.setdefault("iat", now)
    if exp != 0:
        payload.setdefault("exp", now + int(exp))
    h = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    p = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(str(secret).encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64u(sig)}"


def jwt_verify(token, secret, leeway=30):
    """Return the claims of a valid token, or none if the signature/expiry/algorithm is wrong."""
    from .interop import from_py
    try:
        h, p, s = str(token).split(".")
        header = json.loads(_b64u_dec(h))
        if header.get("alg") != "HS256":                 # blocks alg=none / key-confusion attacks
            return None
        expect = hmac.new(str(secret).encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expect, _b64u_dec(s)):
            return None
        claims = json.loads(_b64u_dec(p))
        now = time.time()
        if "exp" in claims and now > float(claims["exp"]) + float(leeway):
            return None
        if "nbf" in claims and now + float(leeway) < float(claims["nbf"]):
            return None
        return from_py(claims)
    except (ValueError, TypeError, KeyError):
        return None


def bearer_token(req):
    auth = req.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return None


def require_jwt(secret, exempt=()):
    """A before-hook: rejects requests without a valid Bearer JWT; sets req.user to its claims."""
    exempt = tuple(exempt)

    def hook(req):
        if req.path in exempt or req.method == "OPTIONS":
            return None
        claims = jwt_verify(bearer_token(req) or "", secret)
        if claims is None:
            return Response(b'{"error":"unauthorized","status":401}', 401,
                            {"Content-Type": "application/json", "WWW-Authenticate": "Bearer"})
        req.user = claims
        return None
    return hook


def require_api_key(keys, header="x-api-key", exempt=()):
    """A before-hook accepting any of the given API keys (constant-time comparison)."""
    keys = [str(k).encode() for k in keys]
    exempt = tuple(exempt)

    def hook(req):
        if req.path in exempt or req.method == "OPTIONS":
            return None
        got = (req.headers.get(header.lower()) or "").encode()
        if not any(hmac.compare_digest(got, k) for k in keys):
            return Response(b'{"error":"unauthorized","status":401}', 401, {"Content-Type": "application/json"})
        return None
    return hook


def install(interp):
    def server(name="apsi", max_body=1 << 20, workers=None, max_connections=10000, log=True,
               handler_timeout=60.0, header_timeout=10.0, body_timeout=30.0, keepalive_timeout=15.0,
               security_headers=True):
        sec = lambda t: float(t.v) if hasattr(t, "v") else float(t)
        return App(name=name, max_body=max_body, workers=workers, max_connections=max_connections,
                   log=log, handler_timeout=sec(handler_timeout), header_timeout=sec(header_timeout),
                   body_timeout=sec(body_timeout), keepalive_timeout=sec(keepalive_timeout),
                   security_headers=security_headers, interp=interp)
    interp.modules["web"] = {
        "server": server, "response": response, "redirect": redirect, "abort": abort,
        "jwt_sign": jwt_sign, "jwt_verify": jwt_verify, "bearer_token": bearer_token,
        "require_jwt": require_jwt, "require_api_key": require_api_key,
    }
