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

    def format(self, source=None, filename=None):
        source = self.source or source
        filename = self.filename or filename or "<input>"
        loc = f"{filename}:{self.line}:{self.col}" if self.line else filename
        out = f"{loc}: {self.kind}: {self.msg}"
        if source and self.line:
            lines = source.splitlines()
            if 0 < self.line <= len(lines):
                src = lines[self.line - 1]
                out += f"\n    {src}\n    " + " " * max((self.col or 1) - 1, 0) + "^"
        out += _format_trace(self.trace)
        return out

    def __str__(self):
        return self.format()


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
