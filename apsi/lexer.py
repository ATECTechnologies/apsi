"""Tokenizer for Apsi: indentation-sensitive, Unicode identifiers, unit literals."""
import re
from .errors import LexError, UnitError
from .units import parse_unit

NUMBER, STRING, FSTRING, NAME, KW, OP, UNIT, NEWLINE, INDENT, DEDENT, EOF, BYTES = (
    "NUMBER", "STRING", "FSTRING", "NAME", "KW", "OP", "UNIT", "NEWLINE", "INDENT", "DEDENT", "EOF", "BYTES")

KEYWORDS = {"let", "var", "fn", "return", "if", "elif", "else", "while", "for", "in", "break",
            "continue", "struct", "true", "false", "none", "and", "or", "not", "use", "import",
            "assert", "evolve", "as", "pass"}

OPS3 = ["..=", "**=", "//=", "<<=", ">>="]
OPS2 = ["..", "**", "//", "==", "!=", "<=", ">=", "->", "=>", "|>", "+=", "-=", "*=", "/=",
        "~=", "%=", "@=", "<<", ">>", "|=", "&=", ":=", "^=", "??", "?.", "?["]
_OPS3, _OPS2 = frozenset(OPS3), frozenset(OPS2)
OPS1 = set("+-*/%@^<>=()[]{},.:'|&~")
UNICODE_OPS = {"±": "±", "×": "*", "·": "*", "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~=", "→": "->",
               "−": "-"}
UNIT_CHARS = "µμΩÅ"
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "0": "\0", "'": "'", "a": "\a", "b": "\b",
           "f": "\f", "v": "\v"}
_HEX_ESC = {"x": 2, "u": 4, "U": 8}


class Token:
    __slots__ = ("type", "value", "line", "col", "unit")

    def __init__(self, type_, value, line, col, unit=None):
        self.type, self.value, self.line, self.col, self.unit = type_, value, line, col, unit

    def __repr__(self):
        u = f" [{self.unit}]" if self.unit else ""
        return f"{self.type}({self.value!r}{u})@{self.line}:{self.col}"


_ASCII_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]*")


def _ident_char(ch):
    return ("a" + ch).isidentifier()


_TOK_MEMO = {}



def _single_quote_string(src, i, prefixed=False):
    """Is the ' at i the start of a '…' string? After a name or a closing bracket it is the derivative mark of
    `evolve` (y' = …); otherwise it starts a string when the same line closes it."""
    j = i - 1 if not prefixed else i - 2
    if j >= 0 and (src[j].isalnum() or src[j] in "_)]}'"):
        return False
    if src.startswith("\'\'\'", i):
        return True
    end = src.find("\n", i + 1)
    line_ = src[i + 1:end if end >= 0 else len(src)]
    k = 0
    while k < len(line_):
        if line_[k] == "\\":
            k += 2
            continue
        if line_[k] == "'":
            if prefixed and line_[k + 1:].lstrip().startswith("=") and not line_[k + 1:].lstrip().startswith("=="):
                return False
            return True
        k += 1
    return False


def _skip_string(src, i):
    """The index after the string literal that starts at i (one or three quotes), or -1."""
    qc = src[i]
    q = qc * 3 if src.startswith(qc * 3, i) else qc
    k = i + len(q)
    while k < len(src):
        if src[k] == "\\":
            k += 2
            continue
        if src.startswith(q, k):
            return k + len(q)
        if len(q) == 1 and src[k] == "\n":
            return -1
        k += 1
    return -1

def tokenize(src):
    """Tokens of a source text. The last few results are remembered (the Guard scan and the parser read the
    same file); the caller gets its own list."""
    if len(src) < 2000:
        return _tokenize(src)
    toks = _TOK_MEMO.get(src)
    if toks is None:
        toks = _tokenize(src)
        if len(_TOK_MEMO) >= 4:
            _TOK_MEMO.pop(next(iter(_TOK_MEMO)))
        _TOK_MEMO[src] = toks
    return list(toks)


def _tokenize(src):
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
            i = src.find("\n", i)
            if i < 0:
                i = n
            continue
        if c == ";":
            if depth == 0 and toks and toks[-1].type != NEWLINE:
                add(NEWLINE, ";")
            i += 1
            continue

        start_col = col()
        # numbers -------------------------------------------------------
        if c == "0" and i + 1 < n and src[i + 1] in "xXbBoO":     # 0x1F, 0b1010, 0o755 (and _ between digits)
            base = {"x": 16, "b": 2, "o": 8}[src[i + 1].lower()]
            ok = {16: "0123456789abcdefABCDEF", 2: "01", 8: "01234567"}[base]
            j = i + 2
            while j < n and (src[j] in ok or (src[j] == "_" and j + 1 < n and src[j + 1] in ok)):
                j += 1
            digits = src[i + 2:j].replace("_", "")
            if not digits:
                raise LexError(f"{src[i:i + 2]} needs digits after it", line, start_col)
            if j < n and (src[j].isalnum() or src[j] == "."):
                raise LexError(f"'{src[j]}' in a base-{base} number", line, col() + (j - i))
            add(NUMBER, str(int(digits, base)), None, start_col)
            i = j
            continue
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
        # raw strings: r#"..."# (any number of #): no escapes, may span lines, end at the quote + the same #s
        if c == "r" and i + 1 < n and src[i + 1] == "#":
            j = i + 1
            while j < n and src[j] == "#":
                j += 1
            if j < n and src[j] == '"':
                close = '"' + "#" * (j - i - 1)
                end = src.find(close, j + 1)
                if end < 0:
                    raise LexError("unterminated raw string", line, start_col)
                text = src[j + 1:end]
                add(STRING, text, c=start_col)
                line += text.count("\n")
                if "\n" in text:
                    line_start = j + 1 + text.rfind("\n") + 1
                i = end + len(close)
                continue
        # Python raw strings: r"…" r'…' rb"…" br"…" (and triple-quoted): backslashes stay as they are
        if c in "rRbB" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            k = i + 1
            if k < n and src[k] in "rRbB" and src[k].lower() != c.lower():
                k += 1
            pre = src[i:k].lower()
            if "r" in pre and k < n and (src[k] == '"' or (src[k] == "'" and _single_quote_string(src, k, True))):
                qc = src[k]
                q = qc * 3 if src.startswith(qc * 3, k) else qc
                j = k + len(q)
                while j < n and not src.startswith(q, j):
                    if src[j] == "\\":
                        j += 1
                    elif src[j] == "\n" and len(q) == 1:
                        raise LexError("unterminated string", line, start_col)
                    j += 1
                if j >= n:
                    raise LexError("unterminated string", line, start_col)
                text = src[k + len(q):j]
                if "b" in pre:
                    add(BYTES, text.encode("latin-1", "backslashreplace"), c=start_col)
                else:
                    add(STRING, text, c=start_col)
                line += text.count("\n")
                if "\n" in text:
                    line_start = k + len(q) + text.rfind("\n") + 1
                i = j + len(q)
                continue
        # strings -------------------------------------------------------
        if c == '"' or (c in "fb" and i + 1 < n and src[i + 1] == '"') or \
                (c == "'" and _single_quote_string(src, i)) or \
                (c in "fb" and i + 1 < n and src[i + 1] == "'" and _single_quote_string(src, i + 1, prefixed=True)):
            is_f = c == "f"
            is_b = c == "b"
            if is_f or is_b:
                i += 1
            qc = src[i]                                    # " or ' (Python's two quote characters)
            triple = src.startswith(qc * 3, i)
            q = qc * 3 if triple else qc
            i += len(q)
            buf = []
            fdepth = 0                                      # f-strings: inside {…}, quotes belong to the expression
            while True:
                if i >= n:
                    raise LexError("unterminated string", line, start_col)
                if is_f and fdepth > 0 and src[i] in "\"'":
                    k = _skip_string(src, i)               # a string inside a replacement field: f"{"x"!r}"
                    if k < 0:
                        raise LexError("unterminated string", line, start_col)
                    buf.append(src[i:k])
                    i = k
                    continue
                if fdepth == 0 and src.startswith(q, i):
                    i += len(q)
                    break
                ch = src[i]
                if is_f:
                    if ch == "{":
                        if fdepth == 0 and src.startswith("{{", i):
                            buf.append("{{")
                            i += 2
                            continue
                        fdepth += 1
                    elif ch == "}" and fdepth > 0:
                        fdepth -= 1
                if ch == "\n":
                    if not triple:
                        raise LexError("unterminated string", line, start_col)
                    line += 1
                    line_start = i + 1
                if ch == "\\" and i + 1 < n:
                    e = src[i + 1]
                    if e in ESCAPES and not (is_f and e in "{}"):
                        buf.append(ESCAPES[e])
                        i += 2
                        continue
                    if e in _HEX_ESC and not (is_b and e != "x"):
                        k = _HEX_ESC[e]
                        digits = src[i + 2:i + 2 + k]
                        if len(digits) == k and all(d in "0123456789abcdefABCDEF" for d in digits):
                            buf.append(chr(int(digits, 16)))
                            i += 2 + k
                            continue
                    if e == "\n":                      # backslash-newline: line continuation inside a string
                        line += 1
                        line_start = i + 2
                        i += 2
                        continue
                buf.append(ch)
                i += 1
            if is_b:
                try:
                    add(BYTES, "".join(buf).encode("latin-1"), c=start_col)
                except UnicodeEncodeError:
                    raise LexError("bytes literals hold only characters up to \\xff", line, start_col)
                continue
            add(FSTRING if is_f else STRING, "".join(buf), c=start_col)
            continue
        # identifiers ---------------------------------------------------
        if (c in _ASCII_START or not c.isascii() and _ident_char(c)) and not c.isdigit():
            s = i
            i = _ASCII_WORD.match(src, i).end()          # fast path for the ASCII part of a name
            while i < n and not src[i].isascii() and _ident_char(src[i]):
                i = _ASCII_WORD.match(src, i + 1).end()
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
        # `name`: any identifier, also one that is an Apsi keyword (converted Python: `var`, `fn` …)
        if c == "`":
            j = src.find("`", i + 1)
            word = src[i + 1:j] if j > 0 else ""
            if j < 0 or not word.isidentifier():
                raise LexError("expected an identifier between backticks, e.g. `var`", line, col())
            add(NAME, word, c=col())
            i = j + 1
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
        m = src[i:i + 3]
        if m not in _OPS3:
            m = m[:2]
            if m not in _OPS2:
                m = None
        if m:
            if m == "?[":
                depth += 1
            add(OP, m, c=start_col)
            i += len(m)
        else:
            if c in OPS1:
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth = max(depth - 1, 0)
                add(OP, c, c=start_col)
                i += 1
            else:
                hint = {"&": " (for 'both' write: and)", "|": " (for 'either' write: or)",
                        "!": " (for 'not' write: not; for 'not equal' write: !=)"}.get(c, "")
                raise LexError(f"unexpected character {c!r}{hint}", line, start_col)

    if toks and toks[-1].type not in (NEWLINE, DEDENT, INDENT):
        add(NEWLINE, "\n")
    while len(indents) > 1:
        indents.pop()
        add(DEDENT, 0)
    add(EOF, None)
    return toks
