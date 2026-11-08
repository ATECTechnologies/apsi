"""Tokenizer for Apsi: indentation-sensitive, Unicode identifiers, unit literals."""
from .errors import LexError, UnitError
from .units import parse_unit

NUMBER, STRING, FSTRING, NAME, KW, OP, UNIT, NEWLINE, INDENT, DEDENT, EOF = (
    "NUMBER", "STRING", "FSTRING", "NAME", "KW", "OP", "UNIT", "NEWLINE", "INDENT", "DEDENT", "EOF")

KEYWORDS = {"let", "var", "fn", "return", "if", "elif", "else", "while", "for", "in", "break",
            "continue", "struct", "true", "false", "none", "and", "or", "not", "use", "import",
            "assert", "evolve", "as", "pass"}

OPS3 = ["..=", "**=", "//="]
OPS2 = ["..", "**", "//", "==", "!=", "<=", ">=", "->", "=>", "|>", "+=", "-=", "*=", "/=",
        "~=", "%=", "@="]
OPS1 = set("+-*/%@^<>=()[]{},.:'")
UNICODE_OPS = {"±": "±", "×": "*", "·": "*", "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=", "→": "->",
               "−": "-"}
UNIT_CHARS = "µμΩÅ"
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0"}


class Token:
    __slots__ = ("type", "value", "line", "col", "unit")

    def __init__(self, type_, value, line, col, unit=None):
        self.type, self.value, self.line, self.col, self.unit = type_, value, line, col, unit

    def __repr__(self):
        u = f" [{self.unit}]" if self.unit else ""
        return f"{self.type}({self.value!r}{u})@{self.line}:{self.col}"


def _ident_char(ch):
    return ("a" + ch).isidentifier()


def tokenize(src):
    if src.startswith("\ufeff"):
        src = src[1:]                       # UTF-8 BOM written by Windows Notepad
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    toks = []
    indents = [0]
    depth = 0
    i, n = 0, len(src)
    line, line_start = 1, 0
    at_line_start = True

    def col():
        return i - line_start + 1

    def add(t, v, unit=None, c=None):
        toks.append(Token(t, v, line, c if c is not None else col(), unit))

    while i < n:
        if at_line_start and depth == 0:
            j, width = i, 0
            while j < n and src[j] in " \t":
                width += 4 if src[j] == "\t" else 1
                j += 1
            if j >= n:
                i = j
                break
            if src[j] in "\r\n#":
                while j < n and src[j] != "\n":
                    j += 1
                i = j + 1
                line += 1
                line_start = i
                continue
            i = j
            if src.startswith("|>", j) and toks and toks[-1].type == NEWLINE:
                toks.pop()                  # a line starting with |> continues the expression above
                at_line_start = False
                continue
            if width > indents[-1]:
                indents.append(width)
                add(INDENT, width)
            else:
                while width < indents[-1]:
                    indents.pop()
                    add(DEDENT, width)
                if width != indents[-1]:
                    raise LexError("indentation does not match any outer block", line, col())
            at_line_start = False
            continue

        c = src[i]
        if c == "\n":
            if depth == 0 and toks and toks[-1].type not in (NEWLINE, INDENT, DEDENT):
                add(NEWLINE, "\n")
            i += 1
            line += 1
            line_start = i
            at_line_start = depth == 0
            continue
        if c in " \t\r":
            i += 1
            continue
        if c == "\\" and i + 1 < n and src[i + 1] == "\n":   # line continuation
            i += 2
            line += 1
            line_start = i
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == ";":
            if depth == 0 and toks and toks[-1].type != NEWLINE:
                add(NEWLINE, ";")
            i += 1
            continue

        start_col = col()
        # numbers -------------------------------------------------------
        if c.isdigit():
            s = i
            while i < n and (src[i].isdigit() or src[i] == "_"):
                i += 1
            if i + 1 < n and src[i] == "." and src[i + 1].isdigit():
                i += 1
                while i < n and (src[i].isdigit() or src[i] == "_"):
                    i += 1
            if i < n and src[i] in "eE":
                k = i + 1
                if k < n and src[k] in "+-":
                    k += 1
                if k < n and src[k].isdigit():
                    i = k
                    while i < n and src[i].isdigit():
                        i += 1
            text = src[s:i].replace("_", "")
            unit = None
            if i < n and src[i] == "[":
                j = src.find("]", i)
                if j < 0 or "\n" in src[i:j]:
                    raise LexError("unclosed unit bracket", line, col())
                unit = src[i + 1:j].strip()
                i = j + 1
            elif i < n and (src[i].isalpha() or src[i] in UNIT_CHARS):
                j = i
                while j < n and (src[j].isalnum() or src[j] in UNIT_CHARS):
                    j += 1
                unit = src[i:j]
                i = j
            if unit is not None:
                try:
                    parse_unit(unit)
                except UnitError as e:
                    raise LexError(e.msg, line, start_col)
            add(NUMBER, text, unit, start_col)
            continue
        # strings -------------------------------------------------------
        if c == '"' or (c == "f" and i + 1 < n and src[i + 1] == '"'):
            is_f = c == "f"
            if is_f:
                i += 1
            triple = src.startswith('"""', i)
            q = '"""' if triple else '"'
            i += len(q)
            buf = []
            while True:
                if i >= n:
                    raise LexError("unterminated string", line, start_col)
                if src.startswith(q, i):
                    i += len(q)
                    break
                ch = src[i]
                if ch == "\n":
                    if not triple:
                        raise LexError("unterminated string", line, start_col)
                    line += 1
                    line_start = i + 1
                if ch == "\\" and i + 1 < n:
                    e = src[i + 1]
                    if e in ESCAPES:
                        buf.append(ESCAPES[e])
                        i += 2
                        continue
                buf.append(ch)
                i += 1
            add(FSTRING if is_f else STRING, "".join(buf), c=start_col)
            continue
        # identifiers ---------------------------------------------------
        if _ident_char(c) and not c.isdigit():
            s = i
            while i < n and _ident_char(src[i]):
                i += 1
            word = src[s:i]
            if word in KEYWORDS:
                add(KW, word, c=start_col)
                if word == "as":     # 'as [unit]' -> raw UNIT token
                    k = i
                    while k < n and src[k] in " \t":
                        k += 1
                    if k < n and src[k] == "[":
                        j = src.find("]", k)
                        if j < 0:
                            raise LexError("unclosed unit bracket", line, col())
                        unit = src[k + 1:j].strip()
                        try:
                            parse_unit(unit)
                        except UnitError as e:
                            raise LexError(e.msg, line, k - line_start + 1)
                        add(UNIT, unit, c=k - line_start + 1)
                        i = j + 1
            else:
                add(NAME, word, c=start_col)
            continue
        # operators -----------------------------------------------------
        if c in UNICODE_OPS:
            add(OP, UNICODE_OPS[c], c=start_col)
            i += 1
            continue
        if src.startswith("+/-", i):
            add(OP, "±", c=start_col)
            i += 3
            continue
        for group in (OPS3, OPS2):
            m = next((o for o in group if src.startswith(o, i)), None)
            if m:
                add(OP, m, c=start_col)
                i += len(m)
                break
        else:
            if c in OPS1:
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth = max(depth - 1, 0)
                add(OP, c, c=start_col)
                i += 1
            else:
                raise LexError(f"unexpected character {c!r}", line, start_col)

    if toks and toks[-1].type not in (NEWLINE, DEDENT, INDENT):
        add(NEWLINE, "\n")
    while len(indents) > 1:
        indents.pop()
        add(DEDENT, 0)
    add(EOF, None)
    return toks
