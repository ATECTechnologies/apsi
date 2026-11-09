"""`apsi lsp` — Language Server Protocol server (stdio).

Works with VS Code (editors/vscode), Neovim, Helix, Zed, Sublime LSP, Emacs eglot …
Features: live diagnostics from the parser + checker (units, shapes, names),
hover with physical dimensions and signatures, go-to-definition, completion,
document symbols and formatting.
"""
import json
import os
import sys
import traceback
from urllib.parse import unquote, urlparse

from . import ast as A
from .errors import ApsiError
from .lexer import KEYWORDS
from .units import dim_name, UNITS, DIM_NAMES

SEVERITY_ERROR = 1


def _uri_to_path(uri):
    p = urlparse(uri)
    path = unquote(p.path)
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return path


def _type_str(t):
    if t is None:
        return None
    if isinstance(t, A.TUnit):
        return f"[{t.text}]"
    args = ", ".join(str(a) if isinstance(a, int) else _type_str(a) for a in t.args)
    return f"{t.name}[{args}]" if args else t.name


class Document:
    def __init__(self, uri, text):
        self.uri = uri
        self.text = text
        self.checker = None
        self.prog = None


class Server:
    def __init__(self, inp=None, out=None):
        from .interpreter import Interpreter
        self.inp = inp or sys.stdin.buffer
        self.out = out or sys.stdout.buffer
        self.docs = {}
        self.interp = Interpreter()
        self.running = True

    # ------------------------------------------------------------- transport
    def read(self):
        headers = {}
        while True:
            line = self.inp.readline()
            if not line:
                return None
            line = line.decode("ascii", "replace").strip()
            if not line:
                break
            k, _, v = line.partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", 0))
        body = self.inp.read(n)
        return json.loads(body.decode("utf-8"))

    def send(self, msg):
        data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        self.out.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
        self.out.flush()

    def reply(self, mid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": mid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self.send(msg)

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    # ------------------------------------------------------------- main loop
    def serve(self):
        while self.running:
            msg = self.read()
            if msg is None:
                break
            method = msg.get("method")
            mid = msg.get("id")
            try:
                handler = getattr(self, "on_" + (method or "").replace("/", "_").replace("$", "S"), None)
                if handler is None:
                    if mid is not None:
                        self.reply(mid, error={"code": -32601, "message": f"unsupported: {method}"})
                    continue
                result = handler(msg.get("params") or {})
                if mid is not None:
                    self.reply(mid, result)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                if mid is not None:
                    self.reply(mid, error={"code": -32603, "message": str(e)})
        return 0

    # ------------------------------------------------------------- lifecycle
    def on_initialize(self, p):
        return {
            "capabilities": {
                "textDocumentSync": 1,
                "hoverProvider": True,
                "definitionProvider": True,
                "completionProvider": {"triggerCharacters": [".", "["]},
                "documentFormattingProvider": True,
                "documentSymbolProvider": True,
            },
            "serverInfo": {"name": "apsi-lsp", "version": __import__("apsi").__version__},
        }

    def on_initialized(self, p):
        return None

    def on_shutdown(self, p):
        return None

    def on_exit(self, p):
        self.running = False

    # ------------------------------------------------------------- documents
    def on_textDocument_didOpen(self, p):
        d = p["textDocument"]
        self.docs[d["uri"]] = Document(d["uri"], d["text"])
        self.analyse(d["uri"])

    def on_textDocument_didChange(self, p):
        uri = p["textDocument"]["uri"]
        changes = p.get("contentChanges") or []
        if changes:
            self.docs[uri] = Document(uri, changes[-1]["text"])
            self.analyse(uri)

    def on_textDocument_didSave(self, p):
        self.analyse(p["textDocument"]["uri"])

    def on_textDocument_didClose(self, p):
        uri = p["textDocument"]["uri"]
        self.docs.pop(uri, None)
        self.notify("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": []})

    def analyse(self, uri):
        from .checker import Checker
        from .parser import parse
        doc = self.docs.get(uri)
        if doc is None:
            return
        diags = []
        try:
            doc.prog = parse(doc.text)
            path = _uri_to_path(uri)
            base = os.path.dirname(path) if path else os.getcwd()
            ch = Checker(self.interp, base)
            errs = ch.check(doc.prog)
            doc.checker = ch
            for e in errs:
                diags.append(self._diag(e, doc.text))
        except ApsiError as e:
            diags.append(self._diag(e, doc.text))
        self.notify("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": diags})

    def _diag(self, e, text):
        line = max((e.line or 1) - 1, 0)
        col = max((e.col or 1) - 1, 0)
        lines = text.splitlines()
        end = col + 1
        if line < len(lines):
            src = lines[line]
            j = col
            while j < len(src) and (src[j].isalnum() or src[j] in "_[]./^"):
                j += 1
            end = max(j, col + 1)
        return {"range": {"start": {"line": line, "character": col}, "end": {"line": line, "character": end}},
                "severity": SEVERITY_ERROR, "source": "apsi", "message": f"{e.kind}: {e.msg}"}

    # ------------------------------------------------------------- queries
    def _ref_at(self, doc, pos):
        if doc.checker is None:
            return None
        line, ch = pos["line"] + 1, pos["character"] + 1
        best = None
        for (l, c, name, info) in doc.checker.refs + doc.checker.decls:
            if l == line and c <= ch < c + len(name):
                best = (l, c, name, info)
        if best is None:
            # fall back to the word under the cursor
            lines = doc.text.splitlines()
            if pos["line"] < len(lines):
                src = lines[pos["line"]]
                i = pos["character"]
                s = i
                while s > 0 and (src[s - 1].isalnum() or src[s - 1] == "_"):
                    s -= 1
                e = i
                while e < len(src) and (src[e].isalnum() or src[e] == "_"):
                    e += 1
                word = src[s:e]
                for (l, c, name, info) in doc.checker.decls:
                    if name == word:
                        best = (l, c, name, info)
                if best is None and word:
                    return (line, s + 1, word, None)
        return best

    def on_textDocument_hover(self, p):
        doc = self.docs.get(p["textDocument"]["uri"])
        if doc is None:
            return None
        ref = self._ref_at(doc, p["position"])
        if ref is None:
            return None
        _, _, name, info = ref
        text = self._describe(name, info)
        return {"contents": {"kind": "markdown", "value": text}} if text else None

    def _describe(self, name, info):
        if info is not None and info.sig is not None:
            node = self._find_def(name)
            if node is not None:
                ps = ", ".join(p.name + (f": {_type_str(p.type)}" if p.type else "") for p in node.params)
                ret = f" -> {_type_str(node.ret)}" if node.ret else ""
                return f"```apsi\nfn {name}({ps}){ret}\n```"
        if info is not None and info.kind in ("var", "param") and info.dim is not None:
            kw = "var" if info.mutable else ("param" if info.kind == "param" else "let")
            return f"```apsi\n{kw} {name}: {dim_name(info.dim)}\n```"
        if info is not None and info.kind == "struct":
            return f"```apsi\nstruct {name}\n```"
        v = self.interp.globals.vars.get(name)
        if v is None:
            for mod in self.interp.modules.values():
                if name in mod:
                    v = mod[name]
                    break
        if v is not None:
            from .fmt import fmt
            doc = (getattr(v, "__doc__", None) or "").strip().split("\n")[0] if callable(v) else ""
            if callable(v):
                return f"**{name}** (builtin)" + (f"\n\n{doc}" if doc else "")
            return f"```apsi\n{name} = {fmt(v)}\n```"
        if name in DIM_NAMES:
            return f"physical type **{name}** = {dim_name(DIM_NAMES[name])}"
        if info is not None:
            kw = "var" if info.mutable else "let"
            return f"```apsi\n{kw} {name}\n```"
        return None

    def _find_def(self, name):
        for doc in self.docs.values():
            if doc.prog is None:
                continue
            stack = list(doc.prog.body)
            while stack:
                n = stack.pop()
                if isinstance(n, A.FnDef) and n.name == name:
                    return n
                if isinstance(n, A.StructDef):
                    stack.extend(n.methods)
        return None

    def on_textDocument_definition(self, p):
        uri = p["textDocument"]["uri"]
        doc = self.docs.get(uri)
        if doc is None:
            return None
        ref = self._ref_at(doc, p["position"])
        if ref is None or ref[3] is None or ref[3].decl is None:
            return None
        l, c = ref[3].decl
        return {"uri": uri, "range": {"start": {"line": l - 1, "character": c - 1},
                                      "end": {"line": l - 1, "character": c - 1}}}

    def on_textDocument_completion(self, p):
        doc = self.docs.get(p["textDocument"]["uri"])
        items = []
        line_text = ""
        if doc is not None:
            lines = doc.text.splitlines()
            if p["position"]["line"] < len(lines):
                line_text = lines[p["position"]["line"]][:p["position"]["character"]]
        if line_text.rstrip().endswith("[") or ("[" in line_text and "]" not in line_text.split("[")[-1]):
            return [{"label": u, "kind": 12, "detail": dim_name(UNITS[u][1])} for u in sorted(UNITS)]
        seen = set()

        def add(label, kind, detail=""):
            if label not in seen:
                seen.add(label)
                items.append({"label": label, "kind": kind, "detail": detail})
        if doc is not None and doc.checker is not None:
            for (_, _, name, info) in doc.checker.decls:
                add(name, 3 if info.sig else (22 if info.kind == "struct" else 6),
                    dim_name(info.dim) if info.dim is not None else "")
        for k in sorted(KEYWORDS | {"test", "over", "step", "using", "record", "every", "into", "tol"}):
            add(k, 14)
        for k, v in sorted(self.interp.globals.vars.items()):
            add(k, 3 if callable(v) else 21)
        for m in sorted(self.interp.modules):
            add(m, 9, "module")
        for t in sorted(DIM_NAMES):
            add(t, 7, dim_name(DIM_NAMES[t]))
        return items

    def on_textDocument_formatting(self, p):
        from .formatter import format_checked
        doc = self.docs.get(p["textDocument"]["uri"])
        if doc is None:
            return []
        try:
            new, changed = format_checked(doc.text)
        except ApsiError:
            return []
        if not changed:
            return []
        n = len(doc.text.splitlines()) + 1
        return [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": n, "character": 0}},
                 "newText": new}]

    def on_textDocument_documentSymbol(self, p):
        doc = self.docs.get(p["textDocument"]["uri"])
        if doc is None or doc.prog is None:
            return []
        out = []
        for s in doc.prog.body:
            kind, name = None, None
            if isinstance(s, A.FnDef):
                kind, name = 12, s.name
            elif isinstance(s, A.StructDef):
                kind, name = 23, s.name
            elif isinstance(s, A.Let):
                kind, name = (13 if s.mutable else 14), ", ".join(s.targets)
            elif isinstance(s, A.Test):
                kind, name = 24, f'test "{s.name}"'
            if kind:
                r = {"start": {"line": s.line - 1, "character": 0}, "end": {"line": s.line - 1, "character": 0}}
                out.append({"name": name, "kind": kind, "range": r, "selectionRange": r})
        return out


def main():
    return Server().serve()
