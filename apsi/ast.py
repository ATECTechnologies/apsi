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
ListComp = _node("ListComp", "elem", "targets", "iter", "cond", "more")   # more: [(targets, iter, cond)] of later for-clauses
DictComp = _node("DictComp", "key", "value", "targets", "iter", "cond", "more")
Lambda = _node("Lambda", "params", "body")
Range = _node("Range", "start", "end", "inclusive")
As = _node("As", "expr", "unit")                     # unit: (text, scale, dim)
Bytes = _node("Bytes", "value")                      # b"..."
SetLit = _node("SetLit", "items")                    # {a, b}
SetComp = _node("SetComp", "elem", "targets", "iter", "cond", "more")
GenExp = _node("GenExp", "elem", "targets", "iter", "cond", "more")      # (x for x in xs): lazy
Chain = _node("Chain", "ops", "operands")             # a < b <= c: every operand evaluated once
NamedExpr = _node("NamedExpr", "name", "value")       # (n := expr)
Await = _node("Await", "value")
YieldExpr = _node("YieldExpr", "value", "from_")      # x = yield v

# types
TName = _node("TName", "name", "args")
TUnit = _node("TUnit", "text", "scale", "dim")
Param = _node("Param", "name", "type", "default", "star")     # star: None | "*" | "**"
Spread = _node("Spread", "value", "double")       # *xs / **d in a call

# statements
Let = _node("Let", "targets", "type", "value", "mutable")
Assign = _node("Assign", "target", "value", "more")     # more: extra targets of a = b = value
AugAssign = _node("AugAssign", "target", "op", "value")
ExprStmt = _node("ExprStmt", "expr")
FnDef = _node("FnDef", "name", "shape_params", "params", "ret", "body", "is_expr", "decorators", "is_async")
StructDef = _node("StructDef", "name", "fields", "methods")
If = _node("If", "branches", "else_body")
While = _node("While", "cond", "body", "else_body")
For = _node("For", "targets", "iter", "body", "parallel", "else_body", "is_async")
# For/comprehension targets: a list of patterns; a pattern is a name (str), a list of patterns (nested
# unpacking) or ("*", name) for a starred name
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
Try = _node("Try", "body", "handlers", "else_body", "final_body", "star")   # handlers: [(kinds, name, body)]; star: except*
Raise = _node("Raise", "value", "cause")
Yield = _node("Yield", "value", "from_")          # `yield x` / `yield from xs`: the function becomes a generator
ClassDef = _node("ClassDef", "name", "bases", "kwargs", "body", "decorators")   # a real class (Python object model)
FieldDef = _node("FieldDef", "name", "annotation", "value")      # `name: Type [= value]` inside a class body
With = _node("With", "items", "body", "is_async")    # items: [(expr, target pattern | None)]
Del = _node("Del", "targets")
Global = _node("Global", "names", "nonlocal_")
Match = _node("Match", "subject", "cases")         # cases: [(pattern, guard | None, body)]
AsyncIter = _node("AsyncIter", "value")             # `async for` inside a comprehension
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
