"""API documentation for Apsi source: `apsi doc FILE.apsi|FOLDER [--out api.md]`.

Reads the files without running them. Public names (not starting with `_`, or the ones listed in `__all__`) are
documented with their signature as written, the docstring (a string as the first statement of the body) or the
`#` comment lines right above the definition."""
import os

from . import ast as A
from .parser import parse, static_public

_SKIP_DIRS = {".git", "dist", "apsi_packages", "__pycache__", "node_modules", ".vscode", "build"}


def _files(target):
    if os.path.isfile(target):
        return [os.path.abspath(target)]
    root = os.path.abspath(target)
    src = os.path.join(root, "src")
    if os.path.isdir(src):
        root = src
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in _SKIP_DIRS and not x.startswith(".") and x != "tests")
        for f in sorted(files):
            if f.endswith(".apsi") and not f.endswith("_test.apsi"):
                out.append(os.path.join(d, f))
    return out


def _header(lines, line):
    """The definition's header as written: from `fn`/`def`/`class`/`struct` to the ':' that opens the body or the
    '=' of an expression body."""
    i = line - 1
    text, depth = "", 0
    while i < len(lines):
        part = lines[i].strip()
        for k, ch in enumerate(part):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0 and ch == ":" and (k == len(part) - 1 or part[k + 1] in " #"):
                return (text + part[:k]).strip()
            elif depth == 0 and ch == "=" and part[k - 1:k] not in ("=", "!", "<", ">") \
                    and part[k + 1:k + 2] not in ("=", ">"):
                return (text + part[:k]).strip()
        text += part + " "
        i += 1
        if i - line > 30:
            break
    return lines[line - 1].strip()


def _comment_above(lines, line):
    out = []
    i = line - 2
    while i >= 0 and lines[i].strip().startswith("#") and not lines[i].strip().startswith("#!"):
        out.append(lines[i].strip().lstrip("#").strip())
        i -= 1
    while i >= 0 and lines[i].strip().startswith("@"):          # decorators sit between the comment and the def
        i -= 1
    return "\n".join(reversed(out))


def _docstring(body):
    if isinstance(body, list) and body and isinstance(body[0], A.ExprStmt) and isinstance(body[0].expr, A.Str):
        return body[0].expr.value.strip()
    return ""


def _describe(node, lines):
    return _docstring(node.body if hasattr(node, "body") else None) or _comment_above(lines, node.line)


def document(target):
    """Markdown documentation of the public API in a file or a project/library folder."""
    files = _files(target)
    out = []
    base = os.path.dirname(files[0]) if len(files) == 1 else os.path.abspath(target)
    title = os.path.basename(os.path.abspath(target)).removesuffix(".apsi")
    out.append(f"# {title} — API\n")
    total = 0
    for path in files:
        with open(path, encoding="utf-8-sig") as fh:
            src = fh.read()
        try:
            prog = parse(src)
        except Exception as e:                                     # noqa: BLE001
            out.append(f"## {os.path.relpath(path, base)}\n\n(could not be read: {getattr(e, 'msg', e)})\n")
            continue
        lines = src.splitlines()
        public = static_public(prog)
        mod_doc = _docstring(prog.body)
        items = []
        for st in prog.body:
            if isinstance(st, (A.FnDef, A.StructDef, A.ClassDef)) and st.name in public:
                items.append(st)
            elif isinstance(st, A.Let) and any(isinstance(t, str) and t in public for t in st.targets):
                items.append(st)
            elif isinstance(st, A.Assign) and isinstance(st.target, A.Name) and st.target.id in public:
                items.append(st)
        if not items and not mod_doc:
            continue
        rel = os.path.relpath(path, base)
        out.append(f"## {rel}\n")
        if mod_doc:
            out.append(mod_doc + "\n")
        for st in items:
            total += 1
            if isinstance(st, A.FnDef):
                out.append(f"### `{st.name}`\n\n```apsi\n{_header(lines, st.line)}\n```\n")
                d = _describe(st, lines)
                if d:
                    out.append(d + "\n")
            elif isinstance(st, (A.StructDef, A.ClassDef)):
                kind = "struct" if isinstance(st, A.StructDef) else "class"
                out.append(f"### {kind} `{st.name}`\n\n```apsi\n{_header(lines, st.line)}\n```\n")
                d = (_docstring(st.body) if isinstance(st, A.ClassDef) else "") or _comment_above(lines, st.line)
                if d:
                    out.append(d + "\n")
                if isinstance(st, A.StructDef):
                    if st.fields:
                        out.append("| field | written as |\n|---|---|")
                        for f in st.fields:
                            out.append(f"| `{f.name}` | `{lines[f.line - 1].strip()}` |")
                        out.append("")
                    methods = st.methods
                else:
                    fields = [x for x in st.body if isinstance(x, A.FieldDef)]
                    if fields:
                        out.append("| field | written as |\n|---|---|")
                        for f in fields:
                            out.append(f"| `{f.name}` | `{lines[f.line - 1].strip()}` |")
                        out.append("")
                    methods = [x for x in st.body if isinstance(x, A.FnDef)]
                for m in methods:
                    if m.name.startswith("_") and m.name != "__init__":
                        continue
                    out.append(f"- `{_header(lines, m.line)}`")
                    d = _describe(m, lines)
                    if d:
                        out.append("  " + d.replace("\n", "\n  "))
                out.append("")
            else:
                out.append(f"### `{lines[st.line - 1].strip()}`\n")
                d = _comment_above(lines, st.line)
                if d:
                    out.append(d + "\n")
    out.append(f"\n_{total} public names in {len(files)} file(s)._\n")
    return "\n".join(out)
