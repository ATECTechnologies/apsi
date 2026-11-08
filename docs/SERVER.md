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
| Processing | handlers run on a worker thread pool (`workers=`), so numpy/AI work runs truly in parallel and never blocks accepting requests |
| Several cores | on Linux, `app.run(processes=N)` forks N processes on the same port (SO_REUSEPORT) and the kernel spreads connections between them |
| Shutdown | on SIGINT/SIGTERM the server stops accepting new connections and lets running requests finish (graceful shutdown) |

**Measured** (2 vCPU machine, load generator on the same machine): **7,700 requests/s** at 500 concurrent
connections, p99 82 ms, 0 errors out of 20,000 requests.

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

## Talking to Next.js / JavaScript / Java

See [`INTEROP.md`](INTEROP.md). Values with units always travel as `{"value": 9.0, "unit": "J"}`.
