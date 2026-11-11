"""Serve a language model over HTTP with the OpenAI API, so any client library or tool that speaks it works as is.

    apsi serve-model Qwen/Qwen2.5-3B-Instruct --port 8000          # from the command line
    hub.serve(lm, port=8000)                                        # from a program

Endpoints:
  POST /v1/chat/completions    {"messages": [...], "max_tokens", "temperature", "top_p", "top_k", "seed", "stop",
                                "stream", "repetition_penalty"}  (stream=true: server-sent events)
  POST /v1/completions         {"prompt": "...", ...the same options}
  GET  /v1/models              the model served
  GET  /health                 {"status": "ok", "queue": n, "device": ...}
  GET  /metrics                requests, tokens, tokens/s, waiting and latency percentiles, errors

The model computes one request at a time, in arrival order (first come, first served); the others wait in a queue
of at most `max_queue` (more get 429 with Retry-After). A client that disconnects stops its generation at the next
token, so the model is free for the next one at once. Each request gets a time limit (`timeout`). With an API key
(api_key= or APSI_API_KEY), requests need "Authorization: Bearer <key>". Errors are JSON in the OpenAI shape.
"""
import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: F401

from .hub_models import StopGeneration


class _Gate:
    """First come, first served: a ticket lock with a bounded queue."""

    def __init__(self, max_queue):
        self.cv = threading.Condition()
        self.next_ticket = 0
        self.serving = 0
        self.max_queue = max_queue
        self.skip = set()                         # tickets of requests that timed out while waiting

    def waiting(self):
        with self.cv:
            return self.next_ticket - self.serving

    def enter(self, deadline):
        with self.cv:
            if self.next_ticket - self.serving > self.max_queue:
                return None
            t = self.next_ticket
            self.next_ticket += 1
            while self.serving != t:
                left = deadline - time.monotonic()
                if left <= 0:                     # gave up while waiting: the turn passes on when it comes
                    self.skip.add(t)
                    return False
                self.cv.wait(min(left, 1.0))
            return True

    def leave(self):
        with self.cv:
            self.serving += 1
            while self.serving in self.skip:
                self.skip.discard(self.serving)
                self.serving += 1
            self.cv.notify_all()


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = self.ok = self.errors = self.rejected = self.cancelled = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.busy = 0.0
        self.wait = deque(maxlen=2000)
        self.latency = deque(maxlen=2000)
        self.ttft = deque(maxlen=2000)
        self.tps = deque(maxlen=2000)

    def snapshot(self):
        def pct(d, q):
            if not d:
                return None
            s = sorted(d)
            return round(s[min(len(s) - 1, int(q * len(s)))], 4)
        with self.lock:
            up = time.time() - self.started
            return {"uptime_s": round(up, 1), "requests": self.requests, "ok": self.ok, "errors": self.errors,
                    "rejected": self.rejected, "cancelled": self.cancelled,
                    "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                    "busy_fraction": round(self.busy / up, 4) if up else 0.0,
                    "tokens_per_s_median": pct(self.tps, 0.5),
                    "wait_s": {"p50": pct(self.wait, 0.5), "p95": pct(self.wait, 0.95), "max": pct(self.wait, 1.0)},
                    "first_token_s": {"p50": pct(self.ttft, 0.5), "p95": pct(self.ttft, 0.95)},
                    "latency_s": {"p50": pct(self.latency, 0.5), "p95": pct(self.latency, 0.95),
                                  "max": pct(self.latency, 1.0)}}


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def handle_error(self, request, client_address):
        import sys
        e = sys.exc_info()[1]
        if isinstance(e, OSError):                # a client that went away: nothing to report
            return
        print(f"apsi serve-model: {type(e).__name__}: {e} (request from {client_address[0]})", file=sys.stderr)


class _ApiError(Exception):
    def __init__(self, status, message, kind="invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


def _num(body, key, default, lo, hi, cast=float):
    v = body.get(key, default)
    if v is None:
        return default
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise _ApiError(400, f"'{key}' must be a number")
    if not lo <= v <= hi:
        raise _ApiError(400, f"'{key}' must be between {lo} and {hi}")
    return v


class Server:
    """The model server; serve() runs it in the foreground, start() in a background thread (tests, programs)."""

    def __init__(self, model, host="127.0.0.1", port=8000, api_key=None, name=None, max_queue=64,
                 max_tokens=4096, timeout=600.0, log=True):
        self.model = model
        self.name = name or os.path.basename(str(getattr(model, "name", "model")).rstrip("/\\")) or "model"
        self.api_key = api_key if api_key is not None else (os.environ.get("APSI_API_KEY") or None)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.log = log
        self.gate = _Gate(int(max_queue))
        self.stats = _Stats()
        self.created = int(time.time())
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "apsi-ai"

            def log_message(self, fmt, *args):
                pass

            def _json(self, code, obj, extra=None):
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _error(self, e, extra=None):
                self._json(e.status, {"error": {"message": e.message, "type": e.kind, "code": e.status}}, extra)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _authorized(self):
                if not server.api_key:
                    return True
                import hmac
                got = self.headers.get("Authorization", "")
                return hmac.compare_digest(got.encode(), f"Bearer {server.api_key}".encode())

            def do_GET(self):
                path = self.path.split("?")[0].rstrip("/") or "/"
                if path == "/health":
                    return self._json(200, {"status": "ok", "model": server.name, "queue": server.gate.waiting(),
                                            "device": str(getattr(server.model, "device", "?"))})
                if not self._authorized():
                    return self._error(_ApiError(401, "missing or wrong API key", "authentication_error"))
                if path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [
                        {"id": server.name, "object": "model", "created": server.created, "owned_by": "apsi"}]})
                if path == "/metrics":
                    return self._json(200, server.stats.snapshot())
                if path == "/":
                    return self._json(200, {"model": server.name, "endpoints": [
                        "/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/metrics"]})
                self._error(_ApiError(404, f"no endpoint {path}", "not_found"))

            def do_POST(self):
                path = self.path.split("?")[0].rstrip("/")
                try:
                    if not self._authorized():
                        raise _ApiError(401, "missing or wrong API key", "authentication_error")
                    if path not in ("/v1/chat/completions", "/v1/completions"):
                        raise _ApiError(404, f"no endpoint {path}", "not_found")
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > 8 << 20:
                        raise _ApiError(413, "request body over 8 MB")
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except ValueError:
                        raise _ApiError(400, "the body is not valid JSON")
                    if not isinstance(body, dict):
                        raise _ApiError(400, "the body must be a JSON object")
                    server._complete(self, body, chat=path.endswith("chat/completions"))
                except _ApiError as e:
                    with server.stats.lock:
                        server.stats.requests += 1
                        if e.status == 429:
                            server.stats.rejected += 1
                        else:
                            server.stats.errors += 1
                    try:
                        self._error(e, {"Retry-After": "2"} if e.status == 429 else None)
                    except OSError:
                        pass
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:                      # noqa: BLE001 - the model failed: report, keep serving
                    with server.stats.lock:
                        server.stats.requests += 1
                        server.stats.errors += 1
                    msg = f"{type(e).__name__}: {e}"
                    print(f"apsi serve-model: request failed: {msg}", flush=True)
                    try:
                        if getattr(self, "_streaming", False):
                            self.wfile.write(b"data: " + json.dumps({"error": {"message": msg, "type": "server_error",
                                                                     "code": 500}}).encode() + b"\n\n")
                            self.wfile.flush()
                        else:
                            self._error(_ApiError(500, msg, "server_error"))
                    except OSError:
                        pass

        self.httpd = _HTTPServer((host, int(port)), Handler)
        self.port = self.httpd.server_address[1]
        self.host = host
        self.thread = None

    # ------------------------------------------------------------------ one request
    def _options(self, body, chat):
        o = {"max_new_tokens": _num(body, "max_tokens", body.get("max_completion_tokens") or 512, 1, self.max_tokens, int),
             "temperature": _num(body, "temperature", 0.7 if chat else 0.0, 0.0, 5.0),
             "top_p": _num(body, "top_p", 0.9 if chat else 1.0, 0.0, 1.0),
             "top_k": _num(body, "top_k", 0, 0, 100000, int),
             "repetition_penalty": _num(body, "repetition_penalty", 1.1 if chat else 1.0, 0.5, 3.0)}
        seed = body.get("seed")
        o["seed"] = int(seed) if seed is not None else None
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        if len(stop) > 8 or not all(isinstance(s, str) and s for s in stop):
            raise _ApiError(400, "'stop' must be a string or up to 8 strings")
        return o, stop

    def _complete(self, h, body, chat):
        t0 = time.monotonic()
        o, stop = self._options(body, chat)
        if chat:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise _ApiError(400, "'messages' must be a non-empty list")
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or \
                        not isinstance(m.get("content", ""), (str, list)):
                    raise _ApiError(400, "each message needs a role (system, user, assistant) and text content")
            msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
                     "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))} for m in msgs]
        else:
            prompt = body.get("prompt")
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            if not isinstance(prompt, str):
                raise _ApiError(400, "'prompt' must be a string")
        stream = bool(body.get("stream"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        kind = "chat.completion" if chat else "text_completion"
        deadline = t0 + self.timeout
        got = self.gate.enter(deadline)
        if got is None:
            raise _ApiError(429, f"the server is busy ({self.gate.max_queue} requests waiting); try again shortly",
                            "rate_limit_error")
        if got is False:
            raise _ApiError(503, "timed out waiting for the model", "timeout")
        t_start = time.monotonic()
        text = []
        state = {"first": None, "sent": 0, "finish": "stop", "gone": False}
        try:
            if stream:
                h._streaming = True
                h.send_response(200)
                h.send_header("Content-Type", "text/event-stream; charset=utf-8")
                h.send_header("Cache-Control", "no-cache")
                h.send_header("Connection", "close")
                h.send_header("Access-Control-Allow-Origin", "*")
                h.end_headers()
                h.close_connection = True
                if chat:
                    self._event(h, rid, kind, {"role": "assistant", "content": ""}, None)

            def piece(s):
                if state["first"] is None:
                    state["first"] = time.monotonic()
                text.append(s)
                whole = "".join(text)
                cut = None
                for st in stop:
                    i = whole.find(st)
                    if i >= 0 and (cut is None or i < cut):
                        cut = i
                safe = len(whole) if cut is None else cut
                if cut is None and stop:                         # hold back what may be the start of a stop string
                    hold = max((k for st in stop for k in range(1, len(st)) if whole.endswith(st[:k])), default=0)
                    safe -= hold
                if stream and safe > state["sent"]:
                    try:
                        self._event(h, rid, kind, whole[state["sent"]:safe], None, chat)
                    except OSError:
                        state["gone"] = True
                        raise StopGeneration()
                    state["sent"] = safe
                if cut is not None:
                    text[:] = [whole[:cut]]
                    raise StopGeneration()
                if time.monotonic() > deadline:
                    state["finish"] = "length"
                    raise StopGeneration()
            m = self.model
            if chat:
                reply = m.chat(msgs, stream=piece, **o)
            else:
                reply = m.generate(prompt, stream=piece, **{k: v for k, v in o.items()})
            out_tokens = int(getattr(m, "last_tokens", 0) or 0)
            in_tokens = int(getattr(m, "last_prompt_tokens", 0) or 0)
        finally:
            self.gate.leave()
        t_end = time.monotonic()
        raw = "".join(text)                                       # exactly what the stream has seen
        full = reply if not (stop or state["gone"]) else (raw.strip() if chat else raw)
        if out_tokens >= o["max_new_tokens"] and state["finish"] == "stop" and not any(full.endswith(s) for s in stop):
            state["finish"] = "length"
        usage = {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
        gen_s = t_end - t_start
        with self.stats.lock:
            st = self.stats
            st.requests += 1
            st.ok += 0 if state["gone"] else 1
            st.cancelled += 1 if state["gone"] else 0
            st.prompt_tokens += in_tokens
            st.completion_tokens += out_tokens
            st.busy += gen_s
            st.wait.append(t_start - t0)
            st.latency.append(t_end - t0)
            if state["first"] is not None:
                st.ttft.append(state["first"] - t0)
            if out_tokens > 1 and gen_s > 0:
                st.tps.append(out_tokens / gen_s)
        if self.log:
            print(f"[{time.strftime('%H:%M:%S')}] {'chat' if chat else 'completion'} {in_tokens}+{out_tokens} tokens "
                  f"wait {t_start - t0:.2f}s run {gen_s:.2f}s"
                  f"{' (client left)' if state['gone'] else ''}", flush=True)
        if state["gone"]:
            return
        if stream:
            try:
                if state["sent"] < len(raw):                    # text held back for a stop string that never came
                    self._event(h, rid, kind, raw[state["sent"]:], None, chat)
                self._event(h, rid, kind, None, state["finish"], chat, usage)
                h.wfile.write(b"data: [DONE]\n\n")
                h.wfile.flush()
            except OSError:
                pass
            return
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": state["finish"]}
        else:
            choice = {"index": 0, "text": full, "finish_reason": state["finish"], "logprobs": None}
        h._json(200, {"id": rid, "object": kind, "created": int(time.time()), "model": self.name,
                      "choices": [choice], "usage": usage})

    def _event(self, h, rid, kind, content, finish, chat=True, usage=None):
        if chat:
            delta = content if isinstance(content, dict) else ({"content": content} if content is not None else {})
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
            obj = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()), "model": self.name,
                   "choices": [choice]}
        else:
            obj = {"id": rid, "object": "text_completion", "created": int(time.time()), "model": self.name,
                   "choices": [{"index": 0, "text": content or "", "finish_reason": finish, "logprobs": None}]}
        if usage is not None:
            obj["usage"] = usage
        h.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
        h.wfile.flush()

    # ------------------------------------------------------------------ running
    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        """Serve in a background thread; returns self (url, stop())."""
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
        self.thread.start()
        return self

    def serve(self):
        print(f"serving {self.name} on {self.url}  (OpenAI API: {self.url}/v1, device {getattr(self.model, 'device', '?')}"
              f"{', API key required' if self.api_key else ''}); Ctrl+C stops", flush=True)
        try:
            self.httpd.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
        return 0

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def serve(model, host="127.0.0.1", port=8000, api_key=None, name=None, max_queue=64, max_tokens=4096, timeout=600.0,
          background=False, log=True):
    s = Server(model, host, port, api_key, name, max_queue, max_tokens, timeout, log)
    return s.start() if background else s.serve()


def add_parser(sub):
    p = sub.add_parser("serve-model", help="serve a language model with the OpenAI API (/v1/chat/completions)")
    p.add_argument("model", help="a Hugging Face name (Qwen/Qwen2.5-3B-Instruct) or a local folder")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept other machines (set --api-key then)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None, help="require 'Authorization: Bearer KEY' (or APSI_API_KEY)")
    p.add_argument("--name", default=None, help="the model id clients see")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N")
    p.add_argument("--gpu-weights", default=None, help="auto, bf16, int8, fp8, fp6, fp6_e2m3, int4, fp4")
    p.add_argument("--max-queue", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=4096, help="the largest max_tokens a request may ask")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per request, waiting included")
    p.add_argument("--quiet", action="store_true", help="no line per request")
    p.set_defaults(fn=_cmd)


def _cmd(args):
    from . import hub
    if args.gpu_weights:
        os.environ["APSI_GPU_WEIGHTS"] = args.gpu_weights
    src = args.model
    if os.path.isdir(src):
        lm = hub.Model(src, device=args.device)
    else:
        lm = hub.Hub().model(src, device=args.device)
    return serve(lm, args.host, args.port, args.api_key, args.name, args.max_queue, args.max_tokens, args.timeout,
                 log=not args.quiet)
