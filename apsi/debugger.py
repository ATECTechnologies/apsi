"""The Apsi debugger: breakpoints, stepping, call stack, variables and tensors — in VS Code (F5) or the terminal.

    apsi debug app.apsi [-- args]       a terminal debugger (b LINE, c, n, s, o, p EXPR, bt, l, q)
    apsi dap                            the Debug Adapter Protocol server VS Code starts (stdio)

Works on the compiled engine: Apsi lines are mapped from the generated code, so breakpoints, the stack and
`step` follow your source, also through imported .apsi files. Conditional breakpoints (`x > 10`), hit counts,
log points, "stop on uncaught error", and readable tensors (shape, dtype, min/max/mean, the first values) are
supported. Variables can be expanded (lists, dicts, records, objects) and changed while paused.
"""
import json
import os
import sys
import threading
import traceback

import numpy as np

# "<apsi:FILE>" (the code object's filename) → (apsi file, linemap, function names)
COMPILED = {}


USER_NAMES = {}


def register(code_filename, apsi_file, linemap, fn_names, user_names=()):
    COMPILED[code_filename] = (os.path.abspath(apsi_file) if not apsi_file.startswith("<") else apsi_file,
                               linemap, fn_names)
    USER_NAMES[code_filename] = set(user_names)


def visible(frame):
    """The variables worth showing for a frame (module level: only what the program declared)."""
    if frame.f_locals is frame.f_globals:
        users = USER_NAMES.get(frame.f_code.co_filename, set())
        return {k: v for k, v in frame.f_globals.items() if k in users and not k.endswith("__e")}
    return {k: v for k, v in frame.f_locals.items() if not _hidden(k, v) and not k.endswith("__e")}


def _hidden(name, value=None):
    if name.startswith("_A") or name.startswith("__") or name in ("u_None",):
        return True
    if name in _BUILTIN_NAMES():
        try:
            from .interpreter import Interpreter  # noqa: F401
            return value is _BUILTIN_VALUES().get(name, object()) or callable(value) or value is None
        except Exception:                                      # noqa: BLE001
            return True
    import types
    return isinstance(value, types.ModuleType)


def describe(v):
    """A short text for a value (tensors get shape/dtype/stats instead of thousands of numbers)."""
    try:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return f"tensor {v.shape} {v.dtype} (empty)"
            if v.dtype.kind in "fiub" and v.size > 1:
                a = v.astype(np.float64)
                head = np.array2string(v.ravel()[:6], precision=4, separator=", ")
                return (f"tensor {tuple(v.shape)} {v.dtype}  min {np.nanmin(a):.4g} max {np.nanmax(a):.4g} "
                        f"mean {np.nanmean(a):.4g}  {head[:-1]}{', …' if v.size > 6 else ''}]")
            return f"tensor {tuple(v.shape)} {v.dtype} {np.array2string(v, threshold=8)}"
        from .fmt import fmt
        s = fmt(v, False) if not hasattr(v, "__dict__") or type(v).__repr__ is not object.__repr__ else repr(v)
    except Exception:                                          # noqa: BLE001
        s = object.__repr__(v)
    return s if len(s) <= 300 else s[:297] + "…"


def _children(v):
    """(name, value) pairs for expandable values."""
    if isinstance(v, dict):
        return [(repr(k), x) for k, x in list(v.items())[:500]]
    if isinstance(v, (list, tuple)):
        return [(f"[{i}]", x) for i, x in enumerate(v[:500])]
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.size > 1:
        return [(f"[{i}]", v[i]) for i in range(min(len(v), 200))]
    fields = getattr(v, "fields", None)
    if isinstance(fields, dict):
        return list(fields.items())
    d = getattr(v, "__dict__", None)
    if isinstance(d, dict):
        return [(k, x) for k, x in d.items() if not k.startswith("__")]
    slots = getattr(type(v), "__slots__", ())
    if slots and not isinstance(v, (str, bytes, int, float)):
        return [(s, getattr(v, s)) for s in slots if hasattr(v, s)]
    return []


def _expandable(v):
    return bool(isinstance(v, (dict, list, tuple)) and v) or (isinstance(v, np.ndarray) and v.ndim >= 1
                                                               and v.size > 1) or bool(_children(v))


class Debugger:
    """The engine: a sys.settrace hook that pauses the program thread and answers questions about it."""

    def __init__(self, on_stop=None, on_output=None):
        self.breakpoints = {}              # abs file -> {line: {"condition", "hit", "log", "count"}}
        self.on_stop = on_stop or (lambda reason, info: None)
        self.on_output = on_output or (lambda text: None)
        self.resume = threading.Event()
        self.mode = "run"                  # run | step | next | out | pause
        self.base_depth = 0
        self.stopped_frame = None
        self.frames = []
        self.refs = {}
        self.next_ref = 1
        self.stop_on_error = True
        self.last_line = {}
        self.thread = None

    # ---------------------------------------------------------------- mapping
    @staticmethod
    def apsi_pos(frame):
        info = COMPILED.get(frame.f_code.co_filename)
        if info is None:
            return None
        f, linemap, names = info
        ln = frame.f_lineno
        if 0 < ln <= len(linemap):
            return f, linemap[ln - 1][0], names.get(frame.f_code.co_name, frame.f_code.co_name)
        return None

    def _depth(self, frame):
        d = 0
        while frame is not None:
            if frame.f_code.co_filename in COMPILED:
                d += 1
            frame = frame.f_back
        return d

    # ---------------------------------------------------------------- tracing
    def trace(self, frame, event, arg):
        if frame.f_code.co_filename not in COMPILED:
            return None
        if event == "line":
            self._line(frame)
        elif event == "exception" and self.stop_on_error:
            exc = arg[1]
            if not getattr(exc, "_apsi_dbg_seen", False) and not self._handled_below(frame):
                try:
                    exc._apsi_dbg_seen = True
                except Exception:                              # noqa: BLE001
                    pass
                if not self._caught_somewhere(frame):
                    self._pause(frame, "exception", f"{type(exc).__name__}: {exc}")
        return self.trace

    @staticmethod
    def _handled_below(frame):
        return False

    @staticmethod
    def _caught_somewhere(frame):
        """Is there an enclosing try in Apsi code? (conservative: look for an exception table entry)."""
        f = frame
        while f is not None:
            if f.f_code.co_filename in COMPILED:
                try:
                    import dis
                    for e in dis.Bytecode(f.f_code).exception_entries:
                        if e.start <= f.f_lasti < e.end and not e.lasti:
                            return True
                except Exception:                              # noqa: BLE001
                    return False
            f = f.f_back
        return False

    def _line(self, frame):
        pos = self.apsi_pos(frame)
        if pos is None:
            return
        f, line, _name = pos
        code = frame.f_code
        if code.co_name != "<module>":
            info = COMPILED[code.co_filename][1]
            fl = code.co_firstlineno
            if 0 < fl <= len(info) and info[fl - 1][0] == line and frame.f_lineno != fl:
                return                                         # the function header's bookkeeping lines
        key = (id(frame), )
        if self.last_line.get(key) == line:
            return                                             # several generated lines for one Apsi line
        self.last_line[key] = line
        bp = self.breakpoints.get(f, {}).get(line)
        if bp is not None:
            bp["count"] = bp.get("count", 0) + 1
            ok = True
            if bp.get("condition"):
                try:
                    ok = bool(self.evaluate(bp["condition"], frame))
                except Exception:                              # noqa: BLE001
                    ok = True
            if ok and bp.get("hit"):
                try:
                    ok = bp["count"] >= int(str(bp["hit"]).lstrip(">= "))
                except ValueError:
                    pass
            if ok and bp.get("log"):
                self.on_output(self._interpolate(bp["log"], frame) + "\n")
                ok = False
            if ok:
                self._pause(frame, "breakpoint")
                return
        if self.mode == "step":
            self._pause(frame, "step")
        elif self.mode == "next" and self._depth(frame) <= self.base_depth:
            self._pause(frame, "step")
        elif self.mode == "out" and self._depth(frame) < self.base_depth:
            self._pause(frame, "step")
        elif self.mode == "pause":
            self._pause(frame, "pause")

    def _interpolate(self, text, frame):
        out, i = [], 0
        while i < len(text):
            j = text.find("{", i)
            if j < 0:
                out.append(text[i:])
                break
            k = text.find("}", j)
            out.append(text[i:j])
            try:
                out.append(describe(self.evaluate(text[j + 1:k], frame)))
            except Exception as e:                             # noqa: BLE001
                out.append(f"<{e}>")
            i = k + 1
        return "".join(out)

    def _pause(self, frame, reason, text=None):
        self.stopped_frame = frame
        self.frames = []
        f = frame
        while f is not None:
            if f.f_code.co_filename in COMPILED and self.apsi_pos(f) is not None:
                self.frames.append(f)
            f = f.f_back
        self.refs = {}
        self.next_ref = 1
        self.resume.clear()
        self.on_stop(reason, text)
        self.resume.wait()
        self.stopped_frame = None
        self.last_line.clear()

    # ---------------------------------------------------------------- control (from the client thread)
    def cont(self):
        self.mode = "run"
        self.resume.set()

    def step_in(self):
        self.mode = "step"
        self.resume.set()

    def step_over(self):
        self.mode = "next"
        self.base_depth = self._depth(self.stopped_frame) if self.stopped_frame else 0
        self.resume.set()

    def step_out(self):
        self.mode = "out"
        self.base_depth = self._depth(self.stopped_frame) if self.stopped_frame else 0
        self.resume.set()

    def pause(self):
        self.mode = "pause"

    # ---------------------------------------------------------------- inspection
    def stack(self):
        out = []
        for i, f in enumerate(self.frames):
            file, line, name = self.apsi_pos(f)
            out.append({"id": i + 1, "name": "<module>" if name == "<module>" else f"{name}()",
                        "file": file, "line": line})
        return out

    def _ref(self, value):
        r = self.next_ref
        self.next_ref += 1
        self.refs[r] = value
        return r

    def variables(self, ref):
        v = self.refs.get(ref)
        items = list(v.items()) if isinstance(v, dict) and ref in self.refs and _is_scope(v) else _children(v)
        out = []
        for name, x in items:
            out.append({"name": str(name), "value": describe(x), "type": _type_name(x),
                        "variablesReference": self._ref(x) if _expandable(x) else 0})
        return out

    def set_variable(self, frame_id, name, text):
        f = self.frames[frame_id - 1]
        val = self.evaluate(text, f)
        f.f_locals[name] = val
        try:
            import ctypes
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(f), ctypes.c_int(0))
        except Exception:                                      # noqa: BLE001
            pass
        return val

    def evaluate(self, expr, frame=None, frame_id=None):
        """Evaluate an Apsi expression in the paused frame (names, attributes, indexing, arithmetic, calls)."""
        if frame is None:
            frame = self.frames[(frame_id or 1) - 1]
        from .parser import Parser
        from .lexer import tokenize
        node = Parser(tokenize(expr)).expr()
        return _Eval(frame).ev(node)


class _ScopeDict(dict):
    pass


def _is_scope(v):
    return isinstance(v, _ScopeDict)


def _type_name(x):
    from .fmt import type_name
    try:
        return type_name(x)
    except Exception:                                          # noqa: BLE001
        return type(x).__name__


class _Eval:
    """A small evaluator for watch expressions and conditions (no side effects beyond calls you write)."""

    def __init__(self, frame):
        self.f = frame

    def name(self, n):
        for scope in (self.f.f_locals, self.f.f_globals):
            if n in scope:
                return scope[n]
            for k, v in scope.items():                         # renamed by the compiler: x_2 …
                if k.startswith(n + "_") and k[len(n) + 1:].isdigit():
                    return v
        b = self.f.f_globals.get("__builtins__")
        if isinstance(b, dict) and n in b:
            return b[n]
        raise NameError(f"'{n}' is not defined here")

    def ev(self, e):
        from . import ast as A
        import operator as op
        if isinstance(e, A.Num):
            return e.value
        if isinstance(e, A.Str):
            return e.value
        if isinstance(e, A.Bool):
            return e.value
        if isinstance(e, A.NoneLit):
            return None
        if isinstance(e, A.Name):
            return self.name(e.id)
        if isinstance(e, A.Attr):
            if e.name.startswith("__"):
                raise NameError("dunder attributes are not available in watch expressions")
            return getattr(self.ev(e.obj), e.name)
        if isinstance(e, A.Index):
            idx = e.index
            if isinstance(idx, A.Slice):
                return self.ev(e.obj)[slice(*(self.ev(x) if x is not None else None for x in (idx.lo, idx.hi, idx.step)))]
            return self.ev(e.obj)[self.ev(idx)]
        if isinstance(e, A.Unary):
            v = self.ev(e.operand)
            return {"-": op.neg, "+": op.pos, "not": op.not_, "~": op.invert}[e.op](v)
        if isinstance(e, A.Binary):
            ops = {"+": op.add, "-": op.sub, "*": op.mul, "/": op.truediv, "//": op.floordiv, "%": op.mod,
                   "**": op.pow, "@": op.matmul, "==": op.eq, "!=": op.ne, "<": op.lt, "<=": op.le, ">": op.gt,
                   ">=": op.ge, "in": lambda a, b: a in b, "not in": lambda a, b: a not in b, "is": op.is_,
                   "is not": op.is_not, "&": op.and_, "|": op.or_}
            return ops[e.op](self.ev(e.left), self.ev(e.right))
        if isinstance(e, A.Chain):
            vals = [self.ev(x) for x in e.operands]
            return all(self.ev(A.Binary(o, A.Num(a, "", None), A.Num(b, "", None))) for o, a, b in
                       zip(e.ops, vals, vals[1:]))
        if isinstance(e, A.Logic):
            a = self.ev(e.left)
            if e.op == "and":
                return self.ev(e.right) if a else a
            return a if a else self.ev(e.right)
        if isinstance(e, A.Call):
            f = self.ev(e.func)
            return f(*[self.ev(a) for a in e.args], **{k: self.ev(v) for k, v in e.kwargs if k})
        if isinstance(e, A.ListLit):
            return [self.ev(x) for x in e.items]
        if isinstance(e, A.TupleLit):
            return tuple(self.ev(x) for x in e.items)
        if isinstance(e, A.IfExpr):
            return self.ev(e.a) if self.ev(e.cond) else self.ev(e.b)
        raise NameError(f"{type(e).__name__} is not supported in watch expressions")


# ======================================================================================== running a program
def run_program(path, args, dbg, policy_args=None):
    """Run an Apsi file under the debugger (in the calling thread)."""
    from .interpreter import Interpreter
    from .security import Policy
    from . import stdlib
    stdlib.PROGRAM_ARGS[:] = list(args)
    sys.argv[:] = [path] + list(args)
    it = Interpreter(policy=policy_args or Policy.from_set({"fs", "net", "env", "python"}))
    with open(path, encoding="utf-8-sig") as fh:
        src = fh.read()
    sys.settrace(dbg.trace)
    threading.settrace(dbg.trace)
    try:
        it.run(src, path)
    finally:
        sys.settrace(None)
        threading.settrace(None)


# ======================================================================================== DAP server
class DAP:
    """Debug Adapter Protocol over stdio (what VS Code speaks)."""

    def __init__(self, rin=None, wout=None):
        self.rin = rin or sys.stdin.buffer
        self.wout = wout or sys.stdout.buffer
        self.seq = 1
        self.lock = threading.Lock()
        self.dbg = Debugger(on_stop=self._stopped, on_output=lambda t: self.event("output", {"category": "console",
                                                                                              "output": t}))
        self.program = None
        self.args = []
        self.stop_on_entry = False
        self.done = threading.Event()

    def send(self, msg):
        msg["seq"] = self.seq
        self.seq += 1
        body = json.dumps(msg).encode()
        with self.lock:
            self.wout.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
            self.wout.flush()

    def event(self, name, body=None):
        self.send({"type": "event", "event": name, "body": body or {}})

    def reply(self, req, body=None, success=True, message=None):
        m = {"type": "response", "request_seq": req["seq"], "command": req["command"], "success": success,
             "body": body or {}}
        if message:
            m["message"] = message
        self.send(m)

    def read(self):
        headers = {}
        while True:
            line = self.rin.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            k, _, v = line.decode().partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", 0))
        return json.loads(self.rin.read(n))

    def _stopped(self, reason, text):
        body = {"reason": reason, "threadId": 1, "allThreadsStopped": True}
        if text:
            body["text"] = text
            body["description"] = text
        self.event("stopped", body)

    def serve(self):
        # the program's print() goes to the debug console, never into the protocol stream
        sys.stdout = _EventStream(self, "stdout")
        sys.stderr = _EventStream(self, "stderr")
        while True:
            req = self.read()
            if req is None:
                break
            cmd = req.get("command")
            try:
                getattr(self, "c_" + cmd, self.c_unknown)(req)
            except Exception as e:                             # noqa: BLE001
                self.reply(req, success=False, message=f"{type(e).__name__}: {e}")
            if cmd == "disconnect":
                break

    def c_unknown(self, req):
        self.reply(req)

    def c_initialize(self, req):
        self.reply(req, {"supportsConfigurationDoneRequest": True, "supportsConditionalBreakpoints": True,
                         "supportsHitConditionalBreakpoints": True, "supportsLogPoints": True,
                         "supportsEvaluateForHovers": True, "supportsSetVariable": True,
                         "exceptionBreakpointFilters": [{"filter": "uncaught", "label": "Uncaught errors",
                                                         "default": True}]})
        self.event("initialized")

    def c_launch(self, req):
        a = req.get("arguments", {})
        self.program = os.path.abspath(a["program"])
        self.args = list(a.get("args", []))
        self.stop_on_entry = bool(a.get("stopOnEntry"))
        if a.get("cwd"):
            os.chdir(a["cwd"])
        self.reply(req)

    def c_setBreakpoints(self, req):
        a = req["arguments"]
        f = os.path.abspath(a["source"]["path"])
        bps, out = {}, []
        for b in a.get("breakpoints", []):
            bps[b["line"]] = {"condition": b.get("condition"), "hit": b.get("hitCondition"),
                              "log": b.get("logMessage")}
            out.append({"verified": True, "line": b["line"]})
        self.dbg.breakpoints[f] = bps
        self.reply(req, {"breakpoints": out})

    def c_setExceptionBreakpoints(self, req):
        self.dbg.stop_on_error = "uncaught" in req["arguments"].get("filters", [])
        self.reply(req)

    def c_configurationDone(self, req):
        self.reply(req)
        if self.stop_on_entry:
            self.dbg.mode = "step"
        t = threading.Thread(target=self._run, name="apsi-program", daemon=True)
        t.start()

    def _run(self):
        code = 0
        try:
            run_program(self.program, self.args, self.dbg)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException as e:                             # noqa: BLE001
            text = e.format() if hasattr(e, "format") else "".join(traceback.format_exception_only(e))
            self.event("output", {"category": "stderr", "output": text + "\n"})
            code = 1
        self.event("exited", {"exitCode": code})
        self.event("terminated")
        self.done.set()

    def c_threads(self, req):
        self.reply(req, {"threads": [{"id": 1, "name": "main"}]})

    def c_stackTrace(self, req):
        frames = [{"id": s["id"], "name": s["name"], "line": s["line"], "column": 1,
                   "source": {"path": s["file"], "name": os.path.basename(s["file"])}} for s in self.dbg.stack()]
        self.reply(req, {"stackFrames": frames, "totalFrames": len(frames)})

    def c_scopes(self, req):
        fid = req["arguments"]["frameId"]
        f = self.dbg.frames[fid - 1]
        loc = _ScopeDict(visible(f).items())
        scopes = [{"name": "Locals", "variablesReference": self.dbg._ref(loc), "expensive": False}]
        if f.f_globals is not f.f_locals:
            users = USER_NAMES.get(f.f_code.co_filename, set())
            glob = _ScopeDict((k, v) for k, v in f.f_globals.items() if k in users and not k.endswith("__e"))
            scopes.append({"name": "Globals", "variablesReference": self.dbg._ref(glob), "expensive": False})
        self.reply(req, {"scopes": scopes})

    def c_variables(self, req):
        self.reply(req, {"variables": self.dbg.variables(req["arguments"]["variablesReference"])})

    def c_setVariable(self, req):
        a = req["arguments"]
        v = self.dbg.set_variable(1, a["name"], a["value"])
        self.reply(req, {"value": describe(v)})

    def c_evaluate(self, req):
        a = req["arguments"]
        try:
            v = self.dbg.evaluate(a["expression"], frame_id=a.get("frameId") or 1)
            self.reply(req, {"result": describe(v), "variablesReference": self.dbg._ref(v) if _expandable(v) else 0})
        except Exception as e:                                 # noqa: BLE001
            self.reply(req, success=False, message=str(e))

    def c_continue(self, req):
        self.reply(req, {"allThreadsContinued": True})
        self.dbg.cont()

    def c_next(self, req):
        self.reply(req)
        self.dbg.step_over()

    def c_stepIn(self, req):
        self.reply(req)
        self.dbg.step_in()

    def c_stepOut(self, req):
        self.reply(req)
        self.dbg.step_out()

    def c_pause(self, req):
        self.reply(req)
        self.dbg.pause()

    def c_disconnect(self, req):
        self.reply(req)
        self.dbg.cont()
        os._exit(0) if not self.done.is_set() and not getattr(self, "keep_alive", False) else None


class _EventStream:
    def __init__(self, dap, category):
        self.dap, self.category = dap, category

    def write(self, text):
        if text:
            self.dap.event("output", {"category": self.category, "output": text})
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


_BN = []


def _BUILTIN_VALUES():
    if not _BN:
        from .interpreter import Interpreter
        it = Interpreter()
        vals = dict(it.globals.vars)
        for m in it.modules.values():
            if isinstance(m, dict):
                for k, v in m.items():
                    vals.setdefault(k, v)
        _BN.append(vals)
    return _BN[0]


def _BUILTIN_NAMES():
    return _BUILTIN_VALUES()


# ======================================================================================== terminal debugger
def cmd_debug(args):
    path = args.file
    dbg = Debugger(on_output=lambda t: sys.stdout.write(t))
    state = {"q": False}

    def show(frame):
        f, line, name = Debugger.apsi_pos(frame)
        try:
            src = open(f, encoding="utf-8").read().splitlines()
            print(f"→ {os.path.basename(f)}:{line} in {name}   {src[line - 1].strip()}")
        except OSError:
            print(f"→ {f}:{line}")

    def on_stop(reason, text):
        if state["q"]:
            dbg.resume.set()
            return
        if text:
            print(f"■ stopped: {text}")
        show(dbg.stopped_frame)
        while True:
            try:
                cmd = input("(apsi-dbg) ").strip()
            except EOFError:
                cmd = "c"
            if cmd in ("c", "continue"):
                dbg.mode = "run"
                return
            if cmd in ("s", "step"):
                dbg.mode = "step"
                return
            if cmd in ("n", "next"):
                dbg.mode = "next"
                dbg.base_depth = dbg._depth(dbg.stopped_frame)
                return
            if cmd in ("o", "out"):
                dbg.mode = "out"
                dbg.base_depth = dbg._depth(dbg.stopped_frame)
                return
            if cmd in ("q", "quit"):
                state["q"] = True
                dbg.mode = "run"
                dbg.breakpoints.clear()
                os._exit(0)
            if cmd in ("bt", "where"):
                for s in dbg.stack():
                    print(f"  {s['name']} at {os.path.basename(s['file'])}:{s['line']}")
                continue
            if cmd in ("l", "locals"):
                for k, v in visible(dbg.stopped_frame).items():
                    print(f"  {k} = {describe(v)}")
                continue
            if cmd.startswith("b "):
                _add_bp(dbg, path, cmd[2:])
                continue
            if cmd.startswith("p "):
                try:
                    print("  " + describe(dbg.evaluate(cmd[2:], dbg.stopped_frame)))
                except Exception as e:                         # noqa: BLE001
                    print(f"  error: {e}")
                continue
            print("  c continue · n next · s step in · o step out · b LINE [if COND] · p EXPR · l locals · "
                  "bt stack · q quit")

    def resume_after(reason, text):
        on_stop(reason, text)
        dbg.resume.set()

    dbg.on_stop = resume_after
    for b in args.breakpoints or []:
        _add_bp(dbg, path, b)
    if not args.breakpoints:
        dbg.mode = "step"                                      # stop at the first line
    try:
        run_program(path, args.args or [], dbg)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except BaseException as e:                                 # noqa: BLE001
        print(e.format() if hasattr(e, "format") else e, file=sys.stderr)
        return 1
    return 0


def _add_bp(dbg, default_file, spec):
    spec = spec.strip()
    cond = None
    if " if " in spec:
        spec, cond = spec.split(" if ", 1)
    if ":" in spec:
        f, line = spec.rsplit(":", 1)
    else:
        f, line = default_file, spec
    dbg.breakpoints.setdefault(os.path.abspath(f), {})[int(line)] = {"condition": cond}
    print(f"  breakpoint at {os.path.basename(f)}:{line}" + (f" if {cond}" if cond else ""))


def cmd_dap(args):
    DAP().serve()
    return 0
