"""`apsi fmt` — canonical formatting for .apsi files.

Safe by construction: the formatted text must parse to exactly the same syntax
tree as the original, otherwise the file is left untouched.
"""
from . import ast as A
from .errors import ApsiError
from .parser import parse

SPACED = ["**=", "//=", "<<=", ">>=", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "@=", "|=", "&=",
          "^=", ":=", "->", "=>", "|>", "~=", "+/-", "±", "="]


def _tree(src):
    def norm(n):
        if isinstance(n, list):
            return [norm(x) for x in n]
        if isinstance(n, tuple):
            return tuple(norm(x) for x in n)
        if isinstance(n, A.AST):
            return (type(n).__name__,) + tuple(norm(getattr(n, f)) for f in n._fields)
        return n
    return norm(parse(src).body)


def _format_line(code, depth=0):
    """Normalise spacing inside one physical line (no leading indentation)."""
    out = []
    i, n = 0, len(code)
    unit_depth = 0
    while i < n:
        c = code[i]
        if c == "#":
            text = code[i:].rstrip()
            if len(text) > 1 and text[1] not in " !":
                text = "# " + text[1:]
            while out and out[-1] == " ":
                out.pop()
            if "".join(out).strip():
                gap = len(code[:i]) - len(code[:i].rstrip(" "))
                out.append(" " * max(2, gap))               # aligned comments keep their column
            out.append(text)
            break
        if c == '"' or (c == "f" and i + 1 < n and code[i + 1] == '"' and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_"))):
            j = i + (2 if c == "f" else 1)
            triple = code.startswith('"""', j - 1)
            if triple:
                end = code.find('"""', j + 2)
                j = (end + 3) if end >= 0 else n
            else:
                while j < n and code[j] != '"':
                    j += 2 if code[j] == "\\" else 1
                j += 1
            out.append(code[i:j])
            i = j
            continue
        if c == "[" and (unit_depth or (i > 0 and (code[i - 1].isdigit() or code[i - 1] == "."))
                         or code[:i].rstrip().endswith(" as") or code[:i].rstrip() == "as"):
            j = code.find("]", i)
            if j > 0:
                out.append(code[i:j + 1])
                i = j + 1
                continue
        if c in "([{":
            depth += 1
            out.append(c)
            i += 1
            while i < n and code[i] == " ":
                i += 1
            continue
        elif c in ")]}":
            depth = max(0, depth - 1)
            while out and out[-1] == " ":
                out.pop()
            if out and out[-1] == ", ":
                out[-1] = ","
        if c == "=" and depth > 0 and not any(code.startswith(o, i) for o in SPACED if o != "=") \
                and not (i > 0 and code[i - 1] in "=!<>+-*/%@~|&^:"):
            while out and out[-1] == " ":
                out.pop()
            out.append("=")
            i += 1
            while i < n and code[i] == " ":
                i += 1
            continue
        if c == ",":
            while out and out[-1] == " ":
                out.pop()
            out.append(", ")
            i += 1
            while i < n and code[i] == " ":
                i += 1
            continue
        if code.startswith("..=", i) or code.startswith("..", i):
            k = 3 if code.startswith("..=", i) else 2
            out.append(code[i:i + k])
            i += k
            continue
        op = next((o for o in SPACED if code.startswith(o, i)), None)
        if op and not (op == "=" and depth > 0):
            if op in ("=", "->", "=>", ":=") or depth == 0 or op in ("|>", "~=", "±", "+/-"):
                while out and out[-1] == " ":
                    out.pop()
                if out and not "".join(out).endswith(" "):
                    out.append(" ")
                out.append(op)
                out.append(" ")
                i += len(op)
                while i < n and code[i] == " ":
                    i += 1
                continue
        if c == " ":
            if out and out[-1] in (" ", ", "):
                i += 1
                continue
            if out and out[-1].endswith(" "):
                i += 1
                continue
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out).rstrip()


def format_source(src):
    lines = src.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    out = []
    indents = [0]
    depth = 0
    blank = 0
    cont_base = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if depth == 0:
                blank += 1
            continue
        width = len(raw) - len(raw.lstrip(" "))
        if depth > 0:
            # continuation line inside brackets: keep relative indentation, normalised
            if stripped[0] in ")]}" and width <= cont_base:
                level_pad = cont_base                       # a closing bracket back at the statement's level
            else:
                level_pad = width if width > cont_base else cont_base + 4   # hand-aligned lines stay aligned
            text = " " * level_pad + _format_line(stripped, depth)
        else:
            if stripped.startswith("#"):
                if width > indents[-1]:
                    level = len(indents)            # comment opening a deeper block
                else:
                    level = len(indents) - 1
                    while level > 0 and width < indents[level]:
                        level -= 1
            else:
                if width > indents[-1]:
                    indents.append(width)
                else:
                    while width < indents[-1]:
                        indents.pop()
                level = len(indents) - 1
            limit = 2 if level == 0 else 1
            if out and blank:
                out.extend([""] * min(blank, limit))
            blank = 0
            text = "    " * level + _format_line(stripped)
            cont_base = 4 * level
        out.append(text)
        depth += _bracket_delta(stripped)
        depth = max(depth, 0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def _bracket_delta(line):
    d, i, n = 0, 0, len(line)
    while i < n:
        c = line[i]
        if c == "#":
            break
        if c == '"':
            j = i + 1
            while j < n and line[j] != '"':
                j += 2 if line[j] == "\\" else 1
            i = j + 1
            continue
        if c in "([{":
            d += 1
        elif c in ")]}":
            d -= 1
        i += 1
    return d


def format_checked(src):
    """Return (formatted, changed). Raises ApsiError if the input doesn't parse."""
    before = _tree(src)
    new = format_source(src)
    try:
        after = _tree(new)
    except ApsiError:
        return src, False
    if after != before:
        return src, False
    return new, new != src
