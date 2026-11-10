"""Error types for Apsi. Every error carries an optional source location."""


class ApsiError(Exception):
    kind = "Error"

    def __init__(self, msg, line=None, col=None):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col
        self.trace = []        # Apsi call frames, innermost first
        self.filename = None
        self.source = None
        self.focus = None      # text the error is about (".uper", "[", "+"): the caret points at it on its line

    def format(self, source=None, filename=None):
        source = self.source or source
        filename = self.filename or filename or "<input>"
        loc = f"{filename}:{self.line}:{self.col}" if self.line else filename
        out = f"{loc}: {self.kind}: {self.msg}"
        if source and self.line:
            lines = source.splitlines()
            if 0 < self.line <= len(lines):
                src = lines[self.line - 1]
                col = self.col or 1
                if self.focus:                    # the line is known, the expression not: point at what failed
                    at = src.find(self.focus, max(col - 1, 0))
                    if at < 0:
                        at = src.find(self.focus)
                    if at >= 0:
                        col = at + 1 + (1 if self.focus[0] == "." and len(self.focus) > 1 else 0)
                        out = f"{filename}:{self.line}:{col}: {self.kind}: {self.msg}"
                out += f"\n    {src}\n    " + " " * max(col - 1, 0) + "^"
        out += _format_trace(self.trace)
        return out

    def __str__(self):
        return self.format()


def did_you_mean(name, candidates, n=1):
    """ " (did you mean 'x'?)" for the closest of the candidates, or ""."""
    import difflib
    try:
        name = str(name)
        pool = sorted({str(c) for c in candidates if isinstance(c, str) and c and not c.startswith("_")})
    except TypeError:
        return ""
    if not name or len(pool) > 20000:
        return ""
    close = difflib.get_close_matches(name, pool, n=n, cutoff=0.7)
    if not close:                                  # a different case: Print -> print
        close = [c for c in pool if c.lower() == name.lower()][:n]
    if not close:
        return ""
    return " (did you mean " + " or ".join(f"'{c}'" for c in close) + "?)"


def _format_trace(frames, keep=8):
    """Call frames, innermost first; repeated frames (deep recursion) are folded."""
    folded = []
    for f in frames:
        if folded and folded[-1][0] == f:
            folded[-1][1] += 1
        else:
            folded.append([f, 1])
    lines = []
    for f, n in folded:
        lines.append(f"\n  in {f}" + (f"  [{n} nested calls]" if n > 1 else ""))
    if len(lines) > 2 * keep + 1:
        hidden = len(lines) - 2 * keep
        lines = lines[:keep] + [f"\n  ... {hidden} more frames ..."] + lines[-keep:]
    return "".join(lines)


class LexError(ApsiError):
    kind = "SyntaxError"


class ParseError(ApsiError):
    kind = "SyntaxError"


class TypeCheckError(ApsiError):
    kind = "TypeError"


class NameErr(ApsiError):
    kind = "NameError"


class UnitError(ApsiError):
    kind = "UnitError"


class ShapeError(ApsiError):
    kind = "ShapeError"


class RuntimeErr(ApsiError):
    kind = "RuntimeError"


class AssertionFailed(ApsiError):
    kind = "AssertionError"


class CheckFailed(Exception):
    """Raised when the static checker finds one or more errors."""

    def __init__(self, errors):
        super().__init__(f"{len(errors)} error(s)")
        self.errors = errors
