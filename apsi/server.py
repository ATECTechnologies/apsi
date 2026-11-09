"""`apsi serve app.apsi` — expose top-level Apsi functions as a JSON HTTP API.

Any language can call Apsi this way (JS / Next.js / Java / Go / C# / Python ...):

  GET  /functions            -> list of functions and their parameters
  POST /call/<name>          body: {"args": [...], "kwargs": {...}}  -> {"result": ...}

Quantities travel as {"value": 9.8, "unit": "m/s^2"} and are unit-checked on arrival.
"""
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from .errors import ApsiError
from .interop import to_py, from_py
from .values import Function
from . import ast as A


def _type_str(t):
    if t is None:
        return None
    if isinstance(t, A.TUnit):
        return f"[{t.text}]"
    args = ", ".join(str(a) if isinstance(a, int) else _type_str(a) for a in t.args)
    return f"{t.name}[{args}]" if args else t.name


def describe(fn):
    node = fn if isinstance(fn, Function) else getattr(fn, "_apsi_def", None)
    params = node.params if node is not None else []
    ret = node.ret if node is not None else None
    return {"name": getattr(fn, "name", "?"),
            "params": [{"name": p.name, "type": _type_str(p.type), "has_default": p.default is not None}
                       for p in params],
            "returns": _type_str(ret)}


def _is_apsi_fn(v):
    return isinstance(v, Function) or getattr(v, "_apsi_def", None) is not None


def serve(interp, env, host="127.0.0.1", port=8787, only=None, max_body=10 * 1024 * 1024):
    fns = {k: v for k, v in env.vars.items()
           if _is_apsi_fn(v) and not k.startswith("_") and (not only or k in only)}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path in ("/", "/functions"):
                self._send(200, {"language": "apsi", "functions": [describe(f) for f in fns.values()]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/call/"):
                return self._send(404, {"error": "not found"})
            name = self.path[len("/call/"):]
            if name not in fns:
                return self._send(404, {"error": f"no exposed function '{name}'"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > max_body:
                return self._send(413, {"error": "request too large"})
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                args = [from_py(a) for a in payload.get("args", [])]
                kwargs = {k: from_py(v) for k, v in payload.get("kwargs", {}).items()}
                result = fns[name](*args, **kwargs)
                self._send(200, {"result": to_py(result)})
            except ApsiError as e:
                self._send(400, {"error": e.msg, "kind": e.kind})
            except (ValueError, TypeError) as e:
                self._send(400, {"error": f"bad request: {e}"})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": "internal error"})

        def log_message(self, fmt, *args):
            print(f"[apsi serve] {self.address_string()} {fmt % args}")

    httpd = HTTPServer((host, port), Handler)
    print(f"Apsi serving {len(fns)} function(s) on http://{host}:{port}")
    for k in fns:
        print(f"  POST /call/{k}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
