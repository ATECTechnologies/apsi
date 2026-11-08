"""Recursive-descent parser for Apsi."""
from fractions import Fraction

from . import ast as A
from .errors import ParseError, UnitError
from .lexer import (tokenize, NUMBER, STRING, FSTRING, NAME, KW, OP, UNIT, NEWLINE, INDENT,
                    DEDENT, EOF)
from .units import parse_unit

COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">=", "~="}
AUG_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "@="}
EVOLVE_METHODS = {"euler", "midpoint", "rk4", "rk45", "symplectic", "verlet"}


def _unit_info(text, line, col):
    try:
        scale, dim = parse_unit(text)
    except UnitError as e:
        raise ParseError(e.msg, line, col)
    return (text, scale, dim)


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    # -- token helpers ---------------------------------------------------
    def peek(self, k=0):
        j = min(self.i + k, len(self.t) - 1)
        return self.t[j]

    def next(self):
        tok = self.t[self.i]
        if self.i < len(self.t) - 1:
            self.i += 1
        return tok

    def at(self, type_, value=None, k=0):
        tok = self.peek(k)
        return tok.type == type_ and (value is None or tok.value == value)

    def at_op(self, v, k=0):
        return self.at(OP, v, k)

    def at_kw(self, v, k=0):
        return self.at(KW, v, k)

    def at_word(self, v, k=0):
        return self.at(NAME, v, k)

    def accept_op(self, v):
        if self.at_op(v):
            return self.next()
        return None

    def accept_kw(self, v):
        if self.at_kw(v):
            return self.next()
        return None

    def error(self, msg, tok=None):
        tok = tok or self.peek()
        raise ParseError(msg, tok.line, tok.col)

    def _desc(self, tok):
        if tok.type == EOF:
            return "end of file"
        if tok.type == NEWLINE:
            return "end of line"
        if tok.type == INDENT:
            return "indentation"
        if tok.type == DEDENT:
            return "end of block"
        return f"'{tok.value}'"

    def expect_op(self, v, ctx=""):
        if not self.at_op(v):
            self.error(f"expected '{v}'{ctx}, found {self._desc(self.peek())}")
        return self.next()

    def expect_kw(self, v, ctx=""):
        if not self.at_kw(v):
            self.error(f"expected '{v}'{ctx}, found {self._desc(self.peek())}")
        return self.next()

    def expect_word(self, v, ctx=""):
        if not self.at_word(v):
            self.error(f"expected '{v}'{ctx}, found {self._desc(self.peek())}")
        return self.next()

    def expect_name(self, ctx=""):
        if self.peek().type != NAME:
            self.error(f"expected a name{ctx}, found {self._desc(self.peek())}")
        return self.next()

    def expect_eol(self):
        tok = self.peek()
        if tok.type == NEWLINE:
            self.next()
        elif tok.type not in (DEDENT, EOF):
            self.error(f"expected end of line, found {self._desc(tok)}")

    def pos(self, tok=None):
        tok = tok or self.peek()
        return {"line": tok.line, "col": tok.col}

    # -- program ---------------------------------------------------------
    def parse_program(self):
        body = []
        while self.peek().type != EOF:
            if self.peek().type == NEWLINE:
                self.next()
                continue
            if self.peek().type == INDENT:
                self.error("unexpected indentation")
            body.append(self.statement())
        return A.Program(body, line=1, col=1)

    def block(self):
        self.expect_op(":", " to start a block")
        if self.peek().type == NEWLINE:
            self.next()
            if self.peek().type != INDENT:
                self.error("expected an indented block")
            self.next()
            body = []
            while self.peek().type not in (DEDENT, EOF):
                if self.peek().type == NEWLINE:
                    self.next()
                    continue
                body.append(self.statement())
            if self.peek().type == DEDENT:
                self.next()
            return body
        return [self.statement()]

    # -- statements ------------------------------------------------------
    def statement(self):
        tok = self.peek()
        if tok.type == OP and tok.value == "@":
            return self.decorated()
        if tok.type == KW:
            v = tok.value
            if v in ("let", "var"):
                return self.let_stmt()
            if v == "fn" and self.at(NAME, k=1):
                return self.fn_def([])
            if v == "struct":
                return self.struct_def()
            if v == "if":
                return self.if_stmt()
            if v == "while":
                self.next()
                cond = self.expr()
                return A.While(cond, self.block(), **self.pos(tok))
            if v == "for":
                return self.for_stmt()
            if v == "return":
                self.next()
                val = None
                if self.peek().type not in (NEWLINE, DEDENT, EOF):
                    val = self.tuple_expr()
                self.expect_eol()
                return A.Return(val, **self.pos(tok))
            if v in ("break", "continue", "pass"):
                self.next()
                self.expect_eol()
                return {"break": A.Break, "continue": A.Continue, "pass": A.Pass}[v](**self.pos(tok))
            if v == "use":
                return self.use_stmt()
            if v == "import":
                self.next()
                if self.peek().type != STRING:
                    self.error('expected a file path string after import, e.g. import "lib.apsi"')
                path = self.next().value
                alias = None
                if self.accept_kw("as"):
                    alias = self.expect_name(" after 'as'").value
                self.expect_eol()
                return A.Import(path, alias, **self.pos(tok))
            if v == "assert":
                self.next()
                cond = self.expr()
                msg = self.expr() if self.accept_op(",") else None
                self.expect_eol()
                return A.Assert(cond, msg, **self.pos(tok))
            if v == "evolve":
                return self.evolve_stmt()
        if tok.type == NAME and tok.value == "test" and self.at(STRING, k=1):
            self.next()
            name = self.next().value
            return A.Test(name, self.block(), **self.pos(tok))
        if tok.type == NAME and self.at_op("'", 1):
            self.next()
            self.next()
            self.expect_op("=", " in derivative statement (x' = ...)")
            e = self.expr()
            self.expect_eol()
            return A.Deriv(tok.value, e, **self.pos(tok))
        return self.simple_stmt()

    def simple_stmt(self):
        tok = self.peek()
        lhs = self.tuple_expr()
        if self.at_op("="):
            self.next()
            self._check_target(lhs)
            val = self.tuple_expr()
            self.expect_eol()
            return A.Assign(lhs, val, **self.pos(tok))
        if self.peek().type == OP and self.peek().value in AUG_OPS:
            op = self.next().value[:-1]
            if not isinstance(lhs, (A.Name, A.Attr, A.Index)):
                self.error("invalid target for augmented assignment", tok)
            val = self.expr()
            self.expect_eol()
            return A.AugAssign(lhs, op, val, **self.pos(tok))
        self.expect_eol()
        return A.ExprStmt(lhs, **self.pos(tok))

    def _check_target(self, t):
        if isinstance(t, A.TupleLit):
            for x in t.items:
                self._check_target(x)
        elif not isinstance(t, (A.Name, A.Attr, A.Index)):
            raise ParseError("cannot assign to this expression", t.line, t.col)

    def let_stmt(self):
        tok = self.next()
        mutable = tok.value == "var"
        names = [self.expect_name(f" after '{tok.value}'").value]
        while self.accept_op(","):
            names.append(self.expect_name().value)
        type_ = None
        if self.accept_op(":"):
            if len(names) > 1:
                self.error("a type annotation needs a single variable")
            type_ = self.type_expr()
        self.expect_op("=", f" in '{tok.value}' declaration")
        val = self.tuple_expr()
        self.expect_eol()
        return A.Let(names, type_, val, mutable, **self.pos(tok))

    def decorated(self):
        decos = []
        while self.at_op("@"):
            self.next()
            decos.append(self.expr())
            self.expect_eol()
        if not self.at_kw("fn"):
            self.error("decorators must be followed by a function definition")
        return self.fn_def(decos)

    def params(self):
        self.expect_op("(")
        ps = []
        while not self.at_op(")"):
            tok = self.expect_name(" for parameter")
            t = self.type_expr() if self.accept_op(":") else None
            d = self.expr() if self.accept_op("=") else None
            ps.append(A.Param(tok.value, t, d, **self.pos(tok)))
            if not self.accept_op(","):
                break
        self.expect_op(")", " to close parameter list")
        return ps

    def fn_def(self, decos):
        tok = self.next()          # 'fn'
        name = self.expect_name(" after 'fn'").value
        shape_params = []
        if self.accept_op("["):
            while not self.at_op("]"):
                shape_params.append(self.expect_name(" for shape parameter").value)
                if not self.accept_op(","):
                    break
            self.expect_op("]")
        ps = self.params()
        ret = self.type_expr() if self.accept_op("->") else None
        if self.accept_op("="):
            if self.peek().type == NEWLINE and self.peek(1).type == INDENT:
                self.next()
                self.next()
                body = self.expr()
                self.expect_eol()
                while self.peek().type == NEWLINE:
                    self.next()
                if self.peek().type != DEDENT:
                    self.error("a '=' function body must be a single expression; use ':' for a block")
                self.next()
            else:
                body = self.expr()
                self.expect_eol()
            return A.FnDef(name, shape_params, ps, ret, body, True, decos, **self.pos(tok))
        body = self.block()
        return A.FnDef(name, shape_params, ps, ret, body, False, decos, **self.pos(tok))

    def struct_def(self):
        tok = self.next()
        name = self.expect_name(" after 'struct'").value
        self.expect_op(":")
        self.expect_eol()
        if self.peek().type != INDENT:
            self.error("expected an indented struct body")
        self.next()
        fields, methods = [], []
        while self.peek().type not in (DEDENT, EOF):
            if self.peek().type == NEWLINE:
                self.next()
                continue
            if self.at_kw("fn") or self.at_op("@"):
                methods.append(self.decorated() if self.at_op("@") else self.fn_def([]))
                continue
            if self.accept_kw("pass"):
                self.expect_eol()
                continue
            ftok = self.expect_name(" for struct field")
            self.expect_op(":", " after field name")
            ft = self.type_expr()
            d = self.expr() if self.accept_op("=") else None
            self.expect_eol()
            fields.append(A.Param(ftok.value, ft, d, **self.pos(ftok)))
        if self.peek().type == DEDENT:
            self.next()
        return A.StructDef(name, fields, methods, **self.pos(tok))

    def if_stmt(self):
        tok = self.next()
        branches = [(self.expr(), self.block())]
        else_body = None
        while True:
            if self.at_kw("elif"):
                self.next()
                branches.append((self.expr(), self.block()))
            elif self.at_kw("else"):
                self.next()
                else_body = self.block()
                break
            else:
                break
        return A.If(branches, else_body, **self.pos(tok))

    def targets(self):
        paren = self.accept_op("(")
        names = [self.expect_name(" as loop variable").value]
        while self.accept_op(","):
            names.append(self.expect_name().value)
        if paren:
            self.expect_op(")")
        return names

    def for_stmt(self):
        tok = self.next()
        names = self.targets()
        self.expect_kw("in", " in for loop")
        it = self.expr()
        return A.For(names, it, self.block(), **self.pos(tok))

    def use_stmt(self):
        tok = self.next()
        parts = [self.expect_name(" after 'use'").value]
        while self.accept_op("."):
            parts.append(self.expect_name().value)
        alias = None
        if self.accept_kw("as"):
            alias = self.expect_name(" after 'as'").value
        self.expect_eol()
        return A.Use(".".join(parts), alias, **self.pos(tok))

    def evolve_stmt(self):
        tok = self.next()
        states = [self.expect_name(" (state variable) after 'evolve'").value]
        while self.accept_op(","):
            states.append(self.expect_name().value)
        self.expect_word("over", " in evolve (evolve x, v over t in 0[s]..1[s] step 1[ms])")
        tvar = self.expect_name(" for the time variable").value
        self.expect_kw("in", " after the time variable")
        rng = self.expr()
        self.expect_word("step", " in evolve")
        step = self.expr()
        method, every, into = "rk4", None, None
        if self.at_word("using"):
            self.next()
            mtok = self.expect_name(" (integration method)")
            if mtok.value not in EVOLVE_METHODS:
                self.error(f"unknown method '{mtok.value}'; choose one of "
                           f"{', '.join(sorted(EVOLVE_METHODS))}", mtok)
            method = mtok.value
        tol = None
        if self.at_word("tol"):
            if method != "rk45":
                self.error("'tol' is only used with the adaptive method rk45")
            self.next()
            tol = self.expr()
        if self.at_word("record"):
            self.next()
            if self.at_word("every"):
                self.next()
                every = self.expr()
            self.expect_word("into", " in record clause")
            into = self.expect_name().value
        body = self.block()
        return A.Evolve(states, tvar, rng, step, method, every, into, body, tol, **self.pos(tok))

    # -- types -----------------------------------------------------------
    def type_expr(self):
        tok = self.peek()
        if self.at_op("["):
            self.next()
            parts = []
            while not self.at_op("]"):
                t = self.next()
                if t.type in (NEWLINE, EOF):
                    self.error("unclosed unit type")
                parts.append(t.value + (t.unit or ""))
            self.next()
            text = " ".join(parts)
            u = _unit_info(text, tok.line, tok.col)
            return A.TUnit(text, u[1], u[2], **self.pos(tok))
        name = self.expect_name(" for type").value
        args = []
        if self.accept_op("["):
            while not self.at_op("]"):
                if self.peek().type == NUMBER:
                    args.append(int(self.next().value))
                else:
                    args.append(self.type_expr())
                if not self.accept_op(","):
                    break
            self.expect_op("]")
        return A.TName(name, args, **self.pos(tok))

    # -- expressions -----------------------------------------------------
    def tuple_expr(self):
        tok = self.peek()
        e = self.expr()
        if self.at_op(","):
            items = [e]
            while self.accept_op(","):
                if self.peek().type in (NEWLINE, EOF, DEDENT) or self.at_op("="):
                    break
                items.append(self.expr())
            return A.TupleLit(items, **self.pos(tok))
        return e

    def expr(self):
        if self.at_kw("fn") and self.at_op("(", 1):
            tok = self.next()
            ps = self.params()
            self.expect_op("=>", " in lambda (fn(x) => expr)")
            return A.Lambda(ps, self.expr(), **self.pos(tok))
        return self.pipe()

    def pipe(self):
        e = self.ternary()
        while self.at_op("|>"):
            tok = self.next()
            r = self.ternary()
            if isinstance(r, A.Call):
                e = A.Call(r.func, [e] + r.args, r.kwargs, **self.pos(tok))
            else:
                e = A.Call(r, [e], [], **self.pos(tok))
        return e

    def ternary(self):
        e = self.or_expr()
        if self.at_kw("if"):
            tok = self.next()
            cond = self.or_expr()
            self.expect_kw("else", " in conditional expression (a if cond else b)")
            b = self.ternary()
            return A.IfExpr(cond, e, b, **self.pos(tok))
        return e

    def or_expr(self):
        e = self.and_expr()
        while self.at_kw("or"):
            tok = self.next()
            e = A.Logic("or", e, self.and_expr(), **self.pos(tok))
        return e

    def and_expr(self):
        e = self.not_expr()
        while self.at_kw("and"):
            tok = self.next()
            e = A.Logic("and", e, self.not_expr(), **self.pos(tok))
        return e

    def not_expr(self):
        if self.at_kw("not"):
            tok = self.next()
            return A.Unary("not", self.not_expr(), **self.pos(tok))
        return self.comparison()

    def comparison(self):
        e = self.range_expr()
        parts = []
        while True:
            tok = self.peek()
            if tok.type == OP and tok.value in COMPARE_OPS:
                op = self.next().value
            elif self.at_kw("in"):
                self.next()
                op = "in"
            elif self.at_kw("not") and self.at_kw("in", 1):
                self.next()
                self.next()
                op = "not in"
            else:
                break
            right = self.range_expr()
            parts.append(A.Binary(op, e if not parts else parts[-1].right, right, **self.pos(tok)))
            e = right
        if not parts:
            return e
        out = parts[0]
        for p in parts[1:]:
            out = A.Logic("and", out, p, line=p.line, col=p.col)
        return out

    def range_expr(self):
        e = self.additive()
        if self.at_op("..") or self.at_op("..="):
            tok = self.next()
            e = A.Range(e, self.additive(), tok.value == "..=", **self.pos(tok))
        while self.at_kw("as"):
            tok = self.next()
            if self.peek().type != UNIT:
                self.error("expected a unit in brackets after 'as', e.g. x as [km]")
            ut = self.next()
            e = A.As(e, _unit_info(ut.value, ut.line, ut.col), **self.pos(tok))
        return e

    def additive(self):
        e = self.term()
        while self.peek().type == OP and self.peek().value in ("+", "-"):
            tok = self.next()
            e = A.Binary(tok.value, e, self.term(), **self.pos(tok))
        return e

    def term(self):
        e = self.unary()
        while self.peek().type == OP and self.peek().value in ("*", "/", "//", "%", "@"):
            tok = self.next()
            e = A.Binary(tok.value, e, self.unary(), **self.pos(tok))
        return e

    def unary(self):
        if self.peek().type == OP and self.peek().value in ("-", "+"):
            tok = self.next()
            return A.Unary(tok.value, self.unary(), **self.pos(tok))
        return self.plusminus()

    def plusminus(self):
        e = self.power()
        if self.at_op("±"):
            tok = self.next()
            return A.Binary("±", e, self.power(), **self.pos(tok))
        return e

    def power(self):
        e = self.postfix()
        if self.peek().type == OP and self.peek().value in ("**", "^"):
            tok = self.next()
            return A.Binary("**", e, self.unary(), **self.pos(tok))
        return e

    def postfix(self):
        e = self.primary()
        while True:
            tok = self.peek()
            if self.at_op("("):
                self.next()
                args, kwargs = [], []
                while not self.at_op(")"):
                    if self.peek().type == NAME and self.at_op("=", 1):
                        k = self.next().value
                        self.next()
                        kwargs.append((k, self.expr()))
                    else:
                        if kwargs:
                            self.error("positional argument after keyword argument")
                        args.append(self.expr())
                    if not self.accept_op(","):
                        break
                self.expect_op(")", " to close call")
                e = A.Call(e, args, kwargs, **self.pos(tok))
            elif self.at_op("["):
                self.next()
                idx = [self.subscript()]
                while self.accept_op(","):
                    if self.at_op("]"):
                        break
                    idx.append(self.subscript())
                self.expect_op("]", " to close index")
                index = idx[0] if len(idx) == 1 else A.TupleLit(idx, **self.pos(tok))
                e = A.Index(e, index, **self.pos(tok))
            elif self.at_op("."):
                self.next()
                nt = self.peek()
                if nt.type not in (NAME, KW):
                    self.error("expected attribute name after '.'")
                self.next()
                e = A.Attr(e, nt.value, **self.pos(tok))
            else:
                return e

    def subscript(self):
        tok = self.peek()
        lo = hi = step = None
        if not self.at_op(":"):
            lo = self.expr()
            if not self.at_op(":"):
                return lo
        self.next()
        if not (self.at_op(":") or self.at_op("]") or self.at_op(",")):
            hi = self.expr()
        if self.accept_op(":"):
            if not (self.at_op("]") or self.at_op(",")):
                step = self.expr()
        return A.Slice(lo, hi, step, **self.pos(tok))

    def primary(self):
        tok = self.peek()
        p = self.pos(tok)
        if tok.type == NUMBER:
            self.next()
            text = tok.value
            is_float = any(ch in text for ch in ".eE")
            value = float(text) if is_float else int(text)
            unit = _unit_info(tok.unit, tok.line, tok.col) if tok.unit else None
            return A.Num(value, text, unit, **p)
        if tok.type == STRING:
            self.next()
            return A.Str(tok.value, **p)
        if tok.type == FSTRING:
            self.next()
            return A.FStr(parse_fstring(tok.value, tok.line, tok.col), **p)
        if tok.type == NAME:
            self.next()
            return A.Name(tok.value, **p)
        if tok.type == KW:
            if tok.value in ("true", "false"):
                self.next()
                return A.Bool(tok.value == "true", **p)
            if tok.value == "none":
                self.next()
                return A.NoneLit(**p)
            if tok.value == "fn":
                return self.expr()
        if tok.type == OP:
            if tok.value == "(":
                self.next()
                if self.accept_op(")"):
                    return A.TupleLit([], **p)
                e = self.expr()
                if self.at_op(","):
                    items = [e]
                    while self.accept_op(","):
                        if self.at_op(")"):
                            break
                        items.append(self.expr())
                    self.expect_op(")")
                    return A.TupleLit(items, **p)
                self.expect_op(")", " to close parenthesis")
                return e
            if tok.value == "[":
                self.next()
                if self.accept_op("]"):
                    return A.ListLit([], **p)
                first = self.expr()
                if self.at_kw("for"):
                    self.next()
                    names = self.targets()
                    self.expect_kw("in", " in list comprehension")
                    it = self.or_expr()
                    cond = self.or_expr() if self.accept_kw("if") else None
                    self.expect_op("]")
                    return A.ListComp(first, names, it, cond, **p)
                items = [first]
                while self.accept_op(","):
                    if self.at_op("]"):
                        break
                    items.append(self.expr())
                self.expect_op("]", " to close list")
                return A.ListLit(items, **p)
            if tok.value == "{":
                self.next()
                pairs = []
                while not self.at_op("}"):
                    k = self.expr()
                    self.expect_op(":", " in dict literal")
                    val = self.expr()
                    if not pairs and self.at_kw("for"):
                        self.next()
                        names = self.targets()
                        self.expect_kw("in", " in dict comprehension")
                        it = self.or_expr()
                        cond = self.or_expr() if self.accept_kw("if") else None
                        self.expect_op("}", " to close dict comprehension")
                        return A.DictComp(k, val, names, it, cond, **p)
                    pairs.append((k, val))
                    if not self.accept_op(","):
                        break
                self.expect_op("}", " to close dict")
                return A.DictLit(pairs, **p)
        self.error(f"unexpected {self._desc(tok)}")


def parse_fstring(raw, line, col):
    parts, buf, i = [], [], 0
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            if raw[i + 1:i + 2] == "{":
                buf.append("{")
                i += 2
                continue
            depth, j, colon = 0, i + 1, None
            while j < len(raw):
                c = raw[j]
                if c in "([{":
                    depth += 1
                elif c in ")]":
                    depth -= 1
                elif c == "}":
                    if depth == 0:
                        break
                    depth -= 1
                elif c == ":" and depth == 0 and colon is None:
                    colon = j
                j += 1
            if j >= len(raw):
                raise ParseError("unclosed '{' in f-string", line, col)
            expr_text = raw[i + 1:(colon if colon is not None else j)]
            spec = raw[colon + 1:j] if colon is not None else None
            try:
                sub = Parser(tokenize(expr_text))
                e = sub.expr()
            except ParseError as err:
                raise ParseError(f"in f-string expression '{expr_text}': {err.msg}", line, col)
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append((e, spec))
            i = j + 1
        elif ch == "}" and raw[i + 1:i + 2] == "}":
            buf.append("}")
            i += 2
        else:
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def parse(source, cache=False):
    """Parse Apsi source. The same source is parsed once per process; with cache=True (program files) the
    result is also kept in the signed on-disk compile cache, so later runs skip parsing."""
    from . import compile_cache
    prog = compile_cache.load(source, disk=cache)
    if prog is None:
        prog = Parser(tokenize(source)).parse_program()
        compile_cache.store(source, prog, disk=cache)
    elif cache:
        compile_cache.store(source, prog, disk=True)      # parsed earlier in this process: keep it on disk too
    return prog
