# Writing servers with Apsi (`use web`)

```apsi
use web
let app = server(name="my-api")

@app.get("/hello/{name}")
fn hello(req) = {"message": "Hello " + req.params["name"]}

@app.post("/energy")
fn energy(req):
    let b = req.json()                         # {"mass": {"value": 2, "unit": "kg"}, ...}
    return {"energy": 0.5 * b["mass"] * b["speed"] ** 2}   # units are checked on every request

app.run(host="0.0.0.0", port=8080)            # run with --allow-net
```

## Architecture

| Part | How it works |
|---|---|
| Accepting connections | one asyncio event loop holds tens of thousands of keep-alive clients with little memory |
| Processing | handlers run on a worker thread pool (`workers=`), so numpy/AI work runs truly in parallel and never blocks accepting requests. A route that proves fast (under 0.5 ms, measured by CPU time so a busy server does not mislead it) runs directly on the event loop, without a thread hand-off; a route that is slow or blocks on I/O stays on the workers |
| Several cores | `app.run(processes="auto")` (one per CPU) or `processes=N` forks processes on the same port (SO_REUSEPORT, Linux and macOS) and the kernel spreads connections between them. Each process has its own memory: keep shared state in a database or `memory.store()` |
| Event loop | uvloop (libuv) when it is installed — `pip install "apsi[web]"` or `pip install uvloop` — otherwise asyncio's loop; `APSI_UVLOOP=0` turns uvloop off. The start-up line names the loop in use |
| Per-request cost | request ids from a counter (no system call), the security headers merged in one step, routes without parameters found by a dictionary lookup, the header block validated by one regular expression, one timer per connection instead of one per request, and fast routes timed by the wall clock only |
| Shutdown | on SIGINT/SIGTERM the server stops accepting new connections and lets running requests finish (graceful shutdown) |

**Measured — a synthetic benchmark** (a route that returns a fixed 25-byte JSON object, keep-alive, the Go load
generator in `benchmarks/web` on the same machine):

| Machine | Apsi, 1 process | Apsi, `processes="auto"` | Node.js 22 bare `http` (1 process) |
|---|---:|---:|---:|
| GitHub runner, AMD EPYC 7763, 4 vCPU | 14,600–15,700 req/s | **38,800–41,500 req/s** | 23,400–24,100 req/s |
| 2-vCPU cloud VM (load generator on the same 2 cores) | 22,100–23,000 req/s (26,200 with uvloop) | **43,400–50,800 req/s** | 39,600–55,700 req/s |

Per process, the work per request went down by about 16 % in this release (19,200 → 22,400 req/s on the VM). With
one process per core Apsi answers more requests than a single Node.js process; Node.js itself also needs its
`cluster` module to use more than one core. With 50 to 1,000 open connections the rate stays the same and no request
fails; while 64 connections keep a blocking 10 ms route busy, a fast route on the same server still answers about
15,000 requests/s with a p99 of 1.0 ms (the slow route runs on worker threads, the fast one on the event loop).

**What these numbers are and are not.** This is a synthetic test of the server's fixed cost per request: no database,
no request body to parse, no TLS, no real network (loopback), and the load generator shares the cores with the
server. It shows how much work the server itself adds. It does not predict a real application, where a request
usually spends milliseconds in the database, in other services or in the handler's own work; there, the per-request
overhead measured here (tens of microseconds) is a small part of the total, and the worker threads (which run
handlers in parallel) and `processes="auto"` matter more. Measure your own routes with `tools/loadtest.py` or
`benchmarks/web/`.

```bash
python tools/loadtest.py http://127.0.0.1:8080/health --requests 20000 --concurrency 500
```

## Security (on by default)

| Threat | Protection |
|---|---|
| Slowloris (slow headers) | `header_timeout` (10 s), `body_timeout` (30 s), `keepalive_timeout` (15 s) |
| Oversized body/headers | `max_body` (1 MB) → 413; headers over 16 KB or over 100 lines → 431 |
| Request smuggling | duplicate `Content-Length`, chunked requests and obs-fold headers are rejected |
| Header injection | headers containing line breaks are dropped |
| Brute force / DoS | `app.rate_limit(rate, per, burst)` — a token bucket per IP → 429 with `Retry-After`; connection limit (`max_connections`) |
| Stuck handlers | `handler_timeout` → 504 |
| Path traversal | `app.static()` checks the real path, so nothing outside the folder can be served (including `%2e%2e`) |
| Information leaks | clients never see a stack trace (only a `request id`); details go to the server log |
| Clickjacking / MIME sniffing | `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, CSP, `Referrer-Policy`, `Permissions-Policy`, HSTS with TLS |
| Authentication | `jwt_sign` / `jwt_verify` (HS256, `alg=none` refused, `exp`/`nbf` checked), `require_jwt`, `require_api_key` (constant-time compare) |
| Passwords | `use crypto` → `password_hash` (scrypt, or PBKDF2 with 600k rounds) |
| TLS | `app.run(tls_cert="cert.pem", tls_key="key.pem")` — TLS 1.2 or newer |
| Program permissions | opening a port needs `--allow-net`; serving files needs the `fs` permission |

Socket-level tests for all of this live in `tests/test_web.py`.

## API

```apsi
let app = server(name="api", max_body=1048576, workers=64, max_connections=10000, log=true,
                 handler_timeout=60[s], header_timeout=10[s], body_timeout=30[s],
                 keepalive_timeout=15[s], security_headers=true)

app.get(path, handler)  /  @app.get(path)          # also post, put, patch, delete, route(method, path)
app.before(fn(req) => none | response)             # middleware before the handler
app.after(fn(req, resp) => ...)                    # middleware after the handler
app.rate_limit(100, per=1[s], burst=200)
app.cors(["https://example.com"])                  # or "*"
app.static("/assets", "public")
app.health()            # GET /health
app.expose_metrics()    # GET /metrics: requests, errors, status codes, p50/p90/p99, req/s
app.stats()             # the same numbers inside your program
app.handle("GET", "/path", body, headers)          # call a route directly without the network (for tests)
app.run(host, port, tls_cert, tls_key, processes, workers)
app.run_background(port=0)                         # run on a separate thread; returns record(port, url, stop)
```

**Path patterns:** `/users/{id:int}`, `/files/{rest:path}`, `/x/{v:float}`, `/name/{s}`

**Request:** `req.method`, `req.path`, `req.params`, `req.query`, `req.headers`, `req.header(name)`,
`req.body` (bytes), `req.text`, `req.json()`, `req.client`, `req.id`, `req.user` (set by `require_jwt`)

**Handler return values:** dict / list / tensor / quantity → JSON; `Str` → text;
`response(body, status, headers, type)`; `redirect(url)`; `abort(status, message)` to stop with an error; `none` → 204

**Errors:** a UnitError/ShapeError/TypeError caused by the client's data → 400; any other error → 500

## WebSocket

```apsi
use web
let app = web.server()

@app.websocket("/echo")
fn echo(ws, msg):                      # message mode: called once per message; no thread waits per connection,
    ws.send(msg)                       # so tens of thousands of idle connections cost almost nothing

@app.websocket("/chat", origins=["https://example.com"], max_message=1_000_000, ping=20)
fn chat(ws):                           # session mode: one conversation per connection
    ws.join("lobby")
    for msg in ws:                     # ends when the client closes
        app.room("lobby").send({"from": ws.id, "text": msg})    # dicts and lists are sent as JSON

app.run(port=8080)
```

* `app.websocket(path, handler, origins=, max_message=, ping=, on_open=, on_close=, subprotocols=)` — the handler
  takes `(ws, msg)` for message mode or `(ws)` for session mode.
* `ws.send(text | bytes | dict | list)`, `ws.receive(timeout=)`, `ws.close(code, reason)`, `ws.join(room)`,
  `ws.leave(room)`, `ws.receive_json()`, `ws.id`, `ws.request` (the upgrade request: path, headers, query).
* `app.room(name).send(msg, exclude=ws)` broadcasts to every member; `len(app.room(name))` counts them.
* Messages of one connection are handled in order; slow receivers get back-pressure instead of unbounded buffers.
* RFC 6455 in full: text and binary, fragmentation, ping/pong keep-alive (dead peers are dropped), the closing
  handshake, UTF-8 checking of text, client masking enforced, a size limit per message (16 MB by default) and Origin
  checking.

Client (needs `--allow-net` for the host):

```apsi
let c = web.websocket_connect("ws://localhost:8080/echo")
c.send("hi")
print(c.receive())
c.close()
```

Measured on a 2-core VM with a Node.js 22 client: a round trip takes 206 µs; a burst of 20,000 messages is echoed
at 36,800 messages per second, in order; 1,000 concurrent connections all answered. Reproduce with
`benchmarks/websocket/`.

## Talking to Next.js / JavaScript / Java

See [`INTEROP.md`](INTEROP.md). Values with units always travel as `{"value": 9.0, "unit": "J"}`.
