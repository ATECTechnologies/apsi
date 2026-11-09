"""AST node definitions for Apsi."""


class AST:
    __slots__ = ("line", "col")
    _fields = ()

    def __init__(self, *args, line=0, col=0):
        for f, a in zip(self._fields, args):
            setattr(self, f, a)
        for f in self._fields[len(args):]:          # optional trailing fields default to None
            setattr(self, f, None)
        self.line, self.col = line, col

    def __repr__(self):
        inner = ", ".join(f"{f}={getattr(self, f)!r}" for f in self._fields)
        return f"{type(self).__name__}({inner})"


ALL = []


def _node(name, *fields):
    cls = type(name, (AST,), {"__slots__": fields, "_fields": fields})
    ALL.append(cls)
    return cls


# expressions
Num = _node("Num", "value", "text", "unit")          # unit: None | (text, scale, dim)
Str = _node("Str", "value")
FStr = _node("FStr", "parts")                        # parts: str | (expr, spec)
Bool = _node("Bool", "value")
NoneLit = _node("NoneLit")
Name = _node("Name", "id")
Unary = _node("Unary", "op", "operand")
Binary = _node("Binary", "op", "left", "right")
Logic = _node("Logic", "op", "left", "right")
IfExpr = _node("IfExpr", "cond", "a", "b")
Call = _node("Call", "func", "args", "kwargs")       # kwargs: list[(name, expr)]
Index = _node("Index", "obj", "index")
Slice = _node("Slice", "lo", "hi", "step")
Attr = _node("Attr", "obj", "name")
ListLit = _node("ListLit", "items")
TupleLit = _node("TupleLit", "items")
DictLit = _node("DictLit", "pairs")
ListComp = _node("ListComp", "elem", "targets", "iter", "cond")
DictComp = _node("DictComp", "key", "value", "targets", "iter", "cond")
Lambda = _node("Lambda", "params", "body")
Range = _node("Range", "start", "end", "inclusive")
As = _node("As", "expr", "unit")                     # unit: (text, scale, dim)

# types
TName = _node("TName", "name", "args")
TUnit = _node("TUnit", "text", "scale", "dim")
Param = _node("Param", "name", "type", "default", "star")     # star: None | "*" | "**"
Spread = _node("Spread", "value", "double")       # *xs / **d in a call

# statements
Let = _node("Let", "targets", "type", "value", "mutable")
Assign = _node("Assign", "target", "value")
AugAssign = _node("AugAssign", "target", "op", "value")
ExprStmt = _node("ExprStmt", "expr")
FnDef = _node("FnDef", "name", "shape_params", "params", "ret", "body", "is_expr", "decorators")
StructDef = _node("StructDef", "name", "fields", "methods")
If = _node("If", "branches", "else_body")
While = _node("While", "cond", "body")
For = _node("For", "targets", "iter", "body", "parallel")   # parallel: True for `parallel for`
Return = _node("Return", "value")
Break = _node("Break")
Continue = _node("Continue")
Pass = _node("Pass")
Use = _node("Use", "path", "alias")
Import = _node("Import", "path", "alias")
Test = _node("Test", "name", "body")
Assert = _node("Assert", "cond", "msg")
Evolve = _node("Evolve", "states", "tvar", "range", "step", "method", "every", "into", "body", "tol")
Deriv = _node("Deriv", "name", "expr")
Try = _node("Try", "body", "handlers", "else_body", "final_body")   # handlers: [(kinds, name, body)]
Raise = _node("Raise", "value")
Yield = _node("Yield", "value", "from_")          # `yield x` / `yield from xs`: the function becomes a generator
Program = _node("Program", "body")


def dump(node, indent=0):
    pad = "  " * indent
    if isinstance(node, list):
        if not node:
            return pad + "[]"
        return "\n".join(dump(n, indent) for n in node)
    if not isinstance(node, AST):
        return pad + repr(node)
    simple = [f for f in node._fields if not isinstance(getattr(node, f), (AST, list))]
    head = pad + type(node).__name__ + (
        "(" + ", ".join(f"{f}={_short(getattr(node, f))}" for f in simple) + ")" if simple else "")
    lines = [head]
    for f in node._fields:
        v = getattr(node, f)
        if isinstance(v, AST) or (isinstance(v, list) and v and
                                  any(isinstance(x, (AST, tuple)) for x in v)):
            lines.append(pad + "  ." + f + ":")
            items = v if isinstance(v, list) else [v]
            for it in items:
                if isinstance(it, tuple):
                    for sub in it:
                        lines.append(dump(sub, indent + 2) if isinstance(sub, (AST, list))
                                     else "  " * (indent + 2) + repr(sub))
                else:
                    lines.append(dump(it, indent + 2))
    return "\n".join(lines)


def _short(v):
    if isinstance(v, tuple) and len(v) == 3 and isinstance(v[0], str):
        return f"[{v[0]}]"
    return repr(v)
