"""Recursive-descent parser for Apsi."""
import itertools
from fractions import Fraction

from . import ast as A
from .errors import ParseError, UnitError
from .lexer import (tokenize, NUMBER, STRING, FSTRING, NAME, KW, OP, UNIT, NEWLINE, INDENT,
                    DEDENT, EOF, BYTES)
from .units import parse_unit

COMPARE_OPS = {"==", "!=", "<", "<=", ">", ">=", "~="}
AUG_OPS = {"+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=", "|=", "&=", "<<=", ">>=", "^="}
# a soft keyword starts a statement only when the next token cannot continue an expression that uses it as a name
_NOT_STMT_AFTER = {"=", ".", "[", ",", ")", "(", ":", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "@=", "|=", "&=",
                   "<<=", ">>=", "==", "!=", "<", ">", "<=", ">=", "+", "-", "*", "/", "//", "%", "|>", ":="}
EVOLVE_METHODS = {"euler", "midpoint", "rk4", "rk45", "symplectic", "verlet"}


def _unit_info(text, line, col):
    try:
        scale, dim = parse_unit(text)
    except UnitError as e:
        raise ParseError(e.msg, line, col)
    return (text, scale, dim)


_QIDS = itertools.count(1)          # hidden names for `??` and `?.` (each value is computed once)


class Parser:
    def __init__(self, tokens):
        self.last = len(tokens) - 1
        self.t = tokens + [tokens[-1]] * 8 if tokens else tokens   # EOF padding: peek(k) needs no bounds check
        self.i = 0
        self.pyc = False             # after `use pycompat`: `^` is Python's exclusive or

    # -- token helpers ---------------------------------------------------
    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j <= self.last + 8 else self.t[-1]

    def next(self):
        tok = self.t[self.i]
        if self.i < self.last:
            self.i += 1
        return tok

    def at(self, type_, value=None, k=0):
        tok = self.t[self.i + k] if k < 8 else self.peek(k)
        return tok.type == type_ and (value is None or tok.value == value)

    def at_op(self, v, k=0):
        tok = self.t[self.i + k] if k < 8 else self.peek(k)
        return tok.value == v and tok.type == OP

    def at_kw(self, v, k=0):
        tok = self.t[self.i + k] if k < 8 else self.peek(k)
        return tok.value == v and tok.type == KW

    def at_word(self, v, k=0):
        tok = self.t[self.i + k] if k < 8 else self.peek(k)
        return tok.value == v and tok.type == NAME

    def accept_op(self, v):
        tok = self.t[self.i]
        if tok.value == v and tok.type == OP:
            if self.i < self.last:
                self.i += 1
            return tok
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
    def _soft(self, word):
        """A soft keyword (class, with, del, global, …) at the start of a statement."""
        tok = self.peek()
        if tok.type != NAME or tok.value != word:
            return False
        nxt = self.peek(1)
        if nxt.type == OP and nxt.value in _NOT_STMT_AFTER:
            return False
        return nxt.type not in (NEWLINE, EOF, DEDENT) or word in ("pass",)

    def statement(self):
        tok = self.peek()
        if tok.type == OP and tok.value == "@":
            return self.decorated()
        if tok.type == NAME and tok.value == "async" and (self.at_kw("fn", 1) or self.at_word("with", 1)
                                                          or self.at_kw("for", 1)):
            self.next()
            st = self.statement()
            st.is_async = True
            return st
        if self._soft("class") and self.at(NAME, k=1):
            return self.class_def([])
        if self._soft("with"):
            return self.with_stmt()
        if tok.type == NAME and tok.value == "match" and not (self.peek(1).type == OP and self.peek(1).value in
                                                               _NOT_STMT_AFTER - {"(", "["}):
            m = self.try_match()
            if m is not None:
                return m
        if self._soft("del"):
            self.next()
            targets = [self.postfix_target()]
            while self.accept_op(","):
                targets.append(self.postfix_target())
            self.expect_eol()
            return A.Del(targets, **self.pos(tok))
        if (self._soft("global") or self._soft("nonlocal")) and self.at(NAME, k=1):
            word = self.next().value
            names = [self.expect_name().value]
            while self.accept_op(","):
                names.append(self.expect_name().value)
            self.expect_eol()
            return A.Global(names, word == "nonlocal", **self.pos(tok))
        if tok.type == NAME and tok.value == "parallel" and self.at_kw("for", k=1):
            self.next()                              # `parallel for`: a soft keyword
            loop = self.for_stmt()
            loop.parallel = True
            return loop
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
                body = self.block()
                return A.While(cond, body, self.loop_else(), **self.pos(tok))
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
        if tok.type == NAME and tok.value == "try" and self.at_op(":", 1):
            return self.try_stmt()
        if tok.type == NAME and tok.value == "raise" and not (self.at_op("=", 1) or self.at_op(".", 1)
                                                               or self.at_op("[", 1) or self.at_op(",", 1)):
            self.next()
            val = cause = None
            if self.peek().type not in (NEWLINE, DEDENT, EOF):
                val = self.expr()
                if self.peek().type == NAME and self.peek().value == "from":   # `raise X from Y`
                    self.next()
                    cause = self.expr()
            self.expect_eol()
            return A.Raise(val, cause, **self.pos(tok))
        if tok.type == NAME and tok.value == "yield" and not (self.at_op("=", 1) or self.at_op(".", 1)
                                                               or self.at_op("[", 1) or self.at_op("(", 1)
                                                               and False):
            self.next()
            frm = False
            if self.peek().type == NAME and self.peek().value == "from":
                self.next()
                frm = True
            val = None if self.peek().type in (NEWLINE, DEDENT, EOF) else self.tuple_expr()
            self.expect_eol()
            return A.Yield(val, frm, **self.pos(tok))
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

    def try_stmt(self):
        tok = self.next()                              # 'try' (a soft keyword)
        body = self.block()
        handlers = []

        def skip_nl():
            while self.peek().type == NEWLINE:
                self.next()
        skip_nl()
        star = False
        while self.peek().type == NAME and self.peek().value == "except":
            self.next()
            if self.accept_op("*"):                             # except* (exception groups)
                star = True
            kinds, name = [], None
            if not self.at_op(":"):
                if not self.at_kw("as"):
                    paren = self.accept_op("(")
                    while True:
                        part = self.expect_name(" (an error kind such as ValueError)").value
                        while self.accept_op("."):
                            part += "." + self.expect_name().value
                        kinds.append(part)
                        if not self.accept_op(","):
                            break
                        if paren and self.at_op(")"):
                            break
                    if paren:
                        self.expect_op(")")
                if self.accept_kw("as"):
                    name = self.expect_name(" after 'as'").value
            handlers.append((kinds, name, self.block()))
            skip_nl()
        else_body = final_body = None
        if handlers and self.at_kw("else"):
            self.next()
            else_body = self.block()
            skip_nl()
        if self.peek().type == NAME and self.peek().value == "finally" and self.at_op(":", 1):
            self.next()
            final_body = self.block()
        if not handlers and final_body is None:
            self.error("'try' needs at least one 'except' or a 'finally' block", tok)
        return A.Try(body, handlers, else_body, final_body, star or None, **self.pos(tok))

    def rhs(self):
        """The right side of an assignment: a tuple expression, or `yield …` used as a value."""
        if self.at_word("yield") and not self.at_op("=", 1):
            tok = self.next()
            frm = False
            if self.at_word("from"):
                self.next()
                frm = True
            val = None if self.peek().type in (NEWLINE, DEDENT, EOF) else self.tuple_expr()
            return A.YieldExpr(val, frm, **self.pos(tok))
        return self.tuple_expr()

    def simple_stmt(self):
        tok = self.peek()
        lhs = self.tuple_expr()
        if self.at_op("="):
            self.next()
            targets = [self._target(lhs)]
            val = self.rhs()
            while self.accept_op("="):                          # a = b = value
                targets.append(self._target(val))
                val = self.rhs()
            self.expect_eol()
            return A.Assign(targets[0], val, targets[1:] or None, **self.pos(tok))
        if self.peek().type == OP and self.peek().value in AUG_OPS:
            op = self.next().value[:-1]
            if op == "^" and not self.pyc:
                self.error("'^=' is only available after `use pycompat` (in Apsi '^' is a power)", tok)
            if not isinstance(lhs, (A.Name, A.Attr, A.Index)):
                self.error("invalid target for augmented assignment", tok)
            val = self.rhs()
            self.expect_eol()
            return A.AugAssign(lhs, op, val, **self.pos(tok))
        self.expect_eol()
        return A.ExprStmt(lhs, **self.pos(tok))

    def _target(self, t):
        if isinstance(t, A.ListLit):                           # [a, b] = … is the same as a, b = …
            t = A.TupleLit(t.items, line=t.line, col=t.col)
        self._check_target(t)
        return t

    def _check_target(self, t, star_ok=False):
        if isinstance(t, (A.TupleLit, A.ListLit)):
            stars = sum(isinstance(x, A.Spread) for x in t.items)
            if stars > 1:
                raise ParseError("only one starred name is allowed in an assignment", t.line, t.col)
            for x in t.items:
                self._check_target(x, True)
        elif isinstance(t, A.Spread) and star_ok and not t.double:
            self._check_target(t.value)
        elif not isinstance(t, (A.Name, A.Attr, A.Index)):
            raise ParseError("cannot assign to this expression", t.line, t.col)

    def postfix_target(self):
        t = self.postfix()
        if not isinstance(t, (A.Name, A.Attr, A.Index)):
            if isinstance(t, A.TupleLit):
                return t
            raise ParseError("cannot delete this expression", t.line, t.col)
        return t

    # ------------------------------------------------------------------ match / case (Python's structural patterns)
    def try_match(self):
        save = self.i
        tok = self.next()
        try:
            subject = self.tuple_expr()
            if not self.at_op(":"):
                raise ParseError("not a match", tok.line, tok.col)
            self.next()
            if self.peek().type != NEWLINE:
                raise ParseError("not a match", tok.line, tok.col)
            self.next()
            while self.peek().type == NEWLINE:
                self.next()
            if self.peek().type != INDENT or not (self.peek(1).type == NAME and self.peek(1).value == "case"):
                raise ParseError("not a match", tok.line, tok.col)
        except ParseError:
            self.i = save
            return None
        self.next()                                              # INDENT
        cases = []
        while self.peek().type == NAME and self.peek().value == "case":
            self.next()
            pat = self.pattern_top()
            guard = None
            if self.accept_kw("if"):
                guard = self.expr()
            cases.append((pat, guard, self.block()))
            while self.peek().type == NEWLINE:
                self.next()
        if self.peek().type == DEDENT:
            self.next()
        return A.Match(subject, cases, **self.pos(tok))

    def pattern_top(self):
        p = self.pattern_as()
        if self.at_op(","):                                      # case a, b: an open sequence
            items = [p]
            while self.accept_op(","):
                if self.at_op(":") or self.at_kw("if"):
                    break
                items.append(self.pattern_as())
            return ("seq", items)
        return p

    def pattern_as(self):
        p = self.pattern_or()
        if self.accept_kw("as"):
            return ("as", p, self.expect_name(" after 'as'").value)
        return p

    def pattern_or(self):
        alts = [self.pattern_one()]
        while self.accept_op("|"):
            alts.append(self.pattern_one())
        return alts[0] if len(alts) == 1 else ("or", alts)

    def pattern_one(self):
        tok = self.peek()
        if tok.type == OP and tok.value == "*":
            self.next()
            n = self.expect_name(" after '*'").value
            return ("star", None if n == "_" else n)
        if tok.type == OP and tok.value in ("(", "["):
            close = ")" if self.next().value == "(" else "]"
            items, trailing = [], False
            while not self.at_op(close):
                items.append(self.pattern_as())
                trailing = self.accept_op(",")
                if not trailing:
                    break
            self.expect_op(close)
            if close == ")" and len(items) == 1 and not trailing:
                return items[0]                                  # a group
            return ("seq", items)
        if tok.type == OP and tok.value == "{":
            self.next()
            pairs, rest = [], None
            while not self.at_op("}"):
                if self.accept_op("**"):
                    rest = self.expect_name().value
                else:
                    key = self.pattern_literal_or_value()
                    self.expect_op(":")
                    pairs.append((key, self.pattern_as()))
                if not self.accept_op(","):
                    break
            self.expect_op("}")
            return ("map", pairs, rest)
        if tok.type == NAME:
            name = self.next().value
            dotted = name
            while self.at_op(".") and self.peek(1).type in (NAME, KW):
                self.next()
                dotted += "." + self.next().value
            if self.at_op("("):                                  # a class pattern
                self.next()
                pos, kws = [], []
                while not self.at_op(")"):
                    if self.peek().type == NAME and self.at_op("=", 1):
                        k = self.next().value
                        self.next()
                        kws.append((k, self.pattern_as()))
                    else:
                        pos.append(self.pattern_as())
                    if not self.accept_op(","):
                        break
                self.expect_op(")")
                return ("class", dotted, pos, kws)
            if "." in dotted:
                return ("value", dotted)
            return ("wild",) if name == "_" else ("capture", name)
        return ("lit", self.pattern_literal_or_value())

    def pattern_literal_or_value(self):
        tok = self.peek()
        if tok.type == NAME:
            dotted = self.next().value
            while self.accept_op("."):
                dotted += "." + self.next().value
            return A.Name(dotted, **self.pos(tok)) if "." not in dotted else ("value", dotted)
        return self.unary()

    def loop_else(self):
        """The `else:` of a for/while loop (runs when the loop ended without break)."""
        save = self.i
        while self.peek().type == NEWLINE:
            self.next()
        if self.at_kw("else"):
            self.next()
            return self.block()
        self.i = save
        return None

    def with_stmt(self):
        tok = self.next()
        items = []
        while True:
            e = self.expr()
            target = None
            if self.accept_kw("as"):
                target = self.target_pattern()
            items.append((e, target))
            if not self.accept_op(","):
                break
        return A.With(items, self.block(), None, **self.pos(tok))

    def target_pattern(self):
        """A name, `*name`, or a (possibly nested) tuple of them — the targets of for/with/comprehensions."""
        if self.at_op("*"):
            self.next()
            return ("*", self.expect_name(" after '*'").value)
        if self.at_op("(") or self.at_op("["):
            close = ")" if self.next().value == "(" else "]"
            items = [self.target_pattern()]
            while self.accept_op(","):
                if self.at_op(close):
                    break
                items.append(self.target_pattern())
            self.expect_op(close)
            return items
        if self.peek().type == NAME and (self.at_op(".", 1) or self.at_op("[", 1)):
            t = self.postfix()                                   # for self.x in …, for a[i] in …
            if not isinstance(t, (A.Attr, A.Index)):
                self.error("expected a name, an attribute or an index as a target")
            return ("expr", t)
        return self.expect_name(" as a target").value

    def class_def(self, decos):
        tok = self.next()                                       # 'class'
        name = self.expect_name(" after 'class'").value
        bases, kwargs = [], []
        if self.accept_op("("):
            while not self.at_op(")"):
                if self.at_op("*") or self.at_op("**"):
                    st = self.next()
                    sp = A.Spread(self.expr(), st.value == "**", **self.pos(st))
                    (kwargs.append((None, sp)) if sp.double else bases.append(sp))
                elif self.peek().type in (NAME, KW) and self.at_op("=", 1):
                    k = self.next().value
                    self.next()
                    kwargs.append((k, self.expr()))
                else:
                    bases.append(self.expr())
                if not self.accept_op(","):
                    break
            self.expect_op(")", " to close the base classes")
        self.expect_op(":", " after the class header")
        body = []
        if self.peek().type == NEWLINE:
            self.next()
            if self.peek().type != INDENT:
                self.error("expected an indented class body")
            self.next()
            while self.peek().type not in (DEDENT, EOF):
                if self.peek().type == NEWLINE:
                    self.next()
                    continue
                body.append(self.class_item())
            if self.peek().type == DEDENT:
                self.next()
        else:
            body.append(self.class_item())
        return A.ClassDef(name, bases, kwargs, body, decos, **self.pos(tok))

    def class_item(self):
        tok = self.peek()
        if tok.type == NAME and self.at_op(":", 1) and tok.value not in ("try", "finally", "except", "with"):
            self.next()
            self.next()
            start = self.i
            ann = self.expr()
            text = _tokens_text(self.t[start:self.i])
            val = self.rhs() if self.accept_op("=") else None
            self.expect_eol()
            return A.FieldDef(tok.value, (ann, text), val, **self.pos(tok))
        return self.statement()

    def let_stmt(self):
        tok = self.next()
        mutable = tok.value == "var"

        def one():
            if self.accept_op("*"):
                return ("*", self.expect_name(" after '*'").value)
            return self.expect_name(f" after '{tok.value}'").value
        names = [one()]
        while self.accept_op(","):
            names.append(one())
        type_ = None
        if self.accept_op(":"):
            if len(names) > 1:
                self.error("a type annotation needs a single variable")
            type_ = self.type_expr()
        if self.pyc and mutable and type_ is None and self.at(NEWLINE) or self.pyc and mutable and self.at(EOF):
            self.expect_eol()                    # `var x`: declared, not yet bound (reads raise like Python)
            return A.Let(names, None, None, True, **self.pos(tok))
        self.expect_op("=", f" in '{tok.value}' declaration")
        val = self.rhs()
        self.expect_eol()
        return A.Let(names, type_, val, mutable, **self.pos(tok))

    def decorated(self):
        decos = []
        while self.at_op("@"):
            self.next()
            decos.append(self.expr())
            self.expect_eol()
            while self.peek().type == NEWLINE:
                self.next()
        if self._soft("class") and self.at(NAME, k=1):
            return self.class_def(decos)
        is_async = False
        if self.at_word("async") and self.at_kw("fn", 1):
            self.next()
            is_async = True
        if not self.at_kw("fn"):
            self.error("decorators must be followed by a function or class definition")
        f = self.fn_def(decos)
        f.is_async = is_async
        return f

    def params(self):
        self.expect_op("(")
        ps = []
        while not self.at_op(")"):
            star = None
            if self.at_op("/") and (self.at_op(",", 1) or self.at_op(")", 1)):
                tok = self.next()                               # positional-only marker
                ps.append(A.Param(None, None, None, "/", **self.pos(tok)))
                if not self.accept_op(","):
                    break
                continue
            if self.at_op("*") and (self.at_op(",", 1) or self.at_op(")", 1)):
                tok = self.next()                               # keyword-only marker
                ps.append(A.Param(None, None, None, "*", **self.pos(tok)))
                if not self.accept_op(","):
                    break
                continue
            if self.at_op("*") or self.at_op("**"):
                star = self.next().value
                if any(p.star == star or p.star == "**" for p in ps if p.name is not None):
                    self.error(f"only one '{star}' parameter is allowed, after the others")
            tok = self.expect_name(" for parameter")
            t = self.type_expr() if self.accept_op(":") else None
            d = None if star else (self.expr() if self.accept_op("=") else None)
            ps.append(A.Param(tok.value, t, d, star, **self.pos(tok)))
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
        """Loop / comprehension targets: a list of patterns (see ast.For)."""
        if self.at_op("(") or self.at_op("["):
            p = self.target_pattern()
            if not self.at_op(","):
                # (a, b) alone is the same as a, b; a single parenthesised name is just that name
                return p if isinstance(p, list) else [p]
            pats = [p]
        else:
            pats = [self.target_pattern()]
        while self.accept_op(","):
            if self.at_kw("in"):
                break
            pats.append(self.target_pattern())
        return pats

    def for_stmt(self):
        tok = self.next()
        names = self.targets()
        self.expect_kw("in", " in for loop")
        it = self.tuple_expr()
        body = self.block()
        return A.For(names, it, body, None, self.loop_else(), **self.pos(tok))

    def use_stmt(self):
        tok = self.next()
        parts = [self.expect_name(" after 'use'").value]
        while self.accept_op("."):
            parts.append(self.expect_name().value)
        alias = None
        if self.accept_kw("as"):
            alias = self.expect_name(" after 'as'").value
        self.expect_eol()
        if parts == ["pycompat"] and alias is None:
            self.pyc = True
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
        e = self.list_item()
        if self.at_op(",") or isinstance(e, A.Spread):
            items = [e]
            while self.accept_op(","):
                if self.peek().type in (NEWLINE, EOF, DEDENT) or self.at_op("=") or self.at_op(":") \
                        or (self.peek().type == OP and self.peek().value in AUG_OPS):
                    break
                items.append(self.list_item())
            return A.TupleLit(items, **self.pos(tok))
        return e

    def expr(self):
        if self.at_kw("fn") and self.at_op("(", 1):
            tok = self.next()
            ps = self.params()
            self.expect_op("=>", " in lambda (fn(x) => expr)")
            return A.Lambda(ps, self.expr(), **self.pos(tok))
        if self.peek().type == NAME and self.at_op(":=", 1):     # (name := value)
            tok = self.next()
            self.next()
            return A.NamedExpr(tok.value, self.expr(), **self.pos(tok))
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
        e = self.coalesce()
        if self.at_kw("if"):
            tok = self.next()
            cond = self.coalesce()
            self.expect_kw("else", " in conditional expression (a if cond else b)")
            b = self.ternary()
            return A.IfExpr(cond, e, b, **self.pos(tok))
        return e

    def coalesce(self):
        """a ?? b: a unless it is none, then b (b is evaluated only when needed; a only once)."""
        e = self.or_expr()
        while self.at_op("??"):
            tok = self.next()
            b = self.or_expr()
            if isinstance(e, A.NoneLit):
                e = b
                continue
            if isinstance(e, (A.Num, A.Str, A.FStr, A.Bool)):     # a literal is never none
                continue
            ref, test = self._once(e)
            e = A.IfExpr(A.Binary("is", test, A.NoneLit(**self.pos(tok)), **self.pos(tok)), b, ref,
                         **self.pos(tok))
        return e

    def _once(self, e):
        """(an expression that reads e's value again, the expression that computes it): plain names are read
        twice, anything else is computed once into a hidden variable."""
        if isinstance(e, A.Name):
            return e, e
        tmp = f"__q{next(_QIDS)}"
        return A.Name(tmp, line=e.line, col=e.col), A.NamedExpr(tmp, e, line=e.line, col=e.col)

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
        first = e = self.bit_or()
        ops, operands = [], [e]
        tok0 = self.peek()
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
            elif self.at_word("is"):
                self.next()
                op = "is not" if self.accept_kw("not") else "is"
            else:
                break
            ops.append(op)
            operands.append(self.bit_or())
        if not ops:
            return first
        if len(ops) == 1:
            return A.Binary(ops[0], operands[0], operands[1], **self.pos(tok0))
        return A.Chain(ops, operands, **self.pos(tok0))       # a < b < c: b is evaluated once

    def bit_or(self):
        e = self.bit_xor()
        while self.at_op("|"):
            tok = self.next()
            e = A.Binary("|", e, self.bit_xor(), **self.pos(tok))
        return e

    def bit_xor(self):
        e = self.bit_and()
        while self.pyc and self.at_op("^"):
            tok = self.next()
            e = A.Binary("^", e, self.bit_and(), **self.pos(tok))
        return e

    def bit_and(self):
        e = self.shift()
        while self.at_op("&"):
            tok = self.next()
            e = A.Binary("&", e, self.shift(), **self.pos(tok))
        return e

    def shift(self):
        e = self.range_expr()
        while self.at_op("<<") or self.at_op(">>"):
            tok = self.next()
            e = A.Binary(tok.value, e, self.range_expr(), **self.pos(tok))
        return e

    def range_expr(self):
        e = self.additive()
        if self.at_op("..") or self.at_op("..="):
            tok = self.next()
            e = A.Range(e, self.additive(), tok.value == "..=", **self.pos(tok))
        while self.at_kw("as") and self.peek(1).type == UNIT:
            tok = self.next()
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
        if self.peek().type == OP and self.peek().value in ("-", "+", "~"):
            tok = self.next()
            return A.Unary(tok.value, self.unary(), **self.pos(tok))
        if self.at_word("await") and self.peek(1).type in (NAME, NUMBER, STRING, FSTRING, KW) \
                or self.at_word("await") and self.at_op("(", 1):
            tok = self.next()
            return A.Await(self.unary(), **self.pos(tok))
        return self.plusminus()

    def plusminus(self):
        e = self.power()
        if self.at_op("±"):
            tok = self.next()
            return A.Binary("±", e, self.power(), **self.pos(tok))
        return e

    def power(self):
        e = self.postfix()
        if self.peek().type == OP and (self.peek().value == "**" or self.peek().value == "^" and not self.pyc):
            tok = self.next()
            return A.Binary("**", e, self.unary(), **self.pos(tok))
        return e

    def postfix(self):
        return self.postfix_rest(self.primary())

    def postfix_rest(self, e):
        while True:
            tok = self.peek()
            if self.at_op("?.") or self.at_op("?["):
                # a?.b, a?[i], a?.f(x): none when a is none, and the rest of the chain is skipped too
                if isinstance(e, (A.Num, A.Str, A.FStr, A.Bool)):  # a literal is never none: a plain . or [
                    self.next()
                    if tok.value == "?.":
                        nt = self.next()
                        e = A.Attr(e, nt.value, **self.pos(tok))
                    else:
                        e = self._index_after_bracket(e, tok)
                    continue
                ref, test = self._once(e)
                self.next()
                if tok.value == "?.":
                    nt = self.peek()
                    if nt.type not in (NAME, KW):
                        self.error("expected attribute name after '?.'")
                    self.next()
                    inner = A.Attr(ref, nt.value, **self.pos(tok))
                else:
                    inner = self._index_after_bracket(ref, tok)
                rest = self.postfix_rest(inner)
                return A.IfExpr(A.Binary("is", test, A.NoneLit(**self.pos(tok)), **self.pos(tok)),
                                A.NoneLit(**self.pos(tok)), rest, **self.pos(tok))
            if self.at_op("("):
                self.next()
                args, kwargs = [], []
                while not self.at_op(")"):
                    if self.at_op("*") or self.at_op("**"):
                        st = self.next()
                        sp = A.Spread(self.expr(), st.value == "**", **self.pos(st))
                        (kwargs.append((None, sp)) if sp.double else args.append(sp))
                    elif self.peek().type in (NAME, KW) and self.at_op("=", 1):
                        k = self.next().value
                        self.next()
                        kwargs.append((k, self.expr()))
                    else:
                        if kwargs and any(k is not None for k, _ in kwargs):
                            self.error("positional argument after keyword argument")
                        a = self.expr()
                        if self._comp_for() and not args:
                            cl = self.comp_clauses("generator expression")
                            (names, it, cond), more = cl[0], cl[1:]
                            a = A.GenExp(a, names, it, cond, more or None, line=a.line, col=a.col)
                        args.append(a)
                    if not self.accept_op(","):
                        break
                self.expect_op(")", " to close call")
                e = A.Call(e, args, kwargs, **self.pos(tok))
            elif self.at_op("["):
                self.next()
                e = self._index_after_bracket(e, tok)
            elif self.at_op("."):
                self.next()
                nt = self.peek()
                if nt.type not in (NAME, KW):
                    self.error("expected attribute name after '.'")
                self.next()
                e = A.Attr(e, nt.value, **self.pos(tok))
            else:
                return e

    def _index_after_bracket(self, e, tok):
        idx = [self.subscript()]
        comma = False
        while self.accept_op(","):
            comma = True
            if self.at_op("]"):
                break
            idx.append(self.subscript())
        self.expect_op("]", " to close index")
        index = idx[0] if len(idx) == 1 and not (comma and self.pyc) and not isinstance(idx[0], A.Spread) \
            else A.TupleLit(idx, **self.pos(tok))
        return A.Index(e, index, **self.pos(tok))

    def list_item(self):
        """An element of a list literal: an expression or `*xs` (all items of xs spread in place)."""
        if self.at_op("*"):
            tok = self.next()
            return A.Spread(self.expr(), False, **self.pos(tok))
        return self.expr()

    def _comp_for(self):
        return self.at_kw("for") or (self.at_word("async") and self.at_kw("for", 1))

    def comp_clauses(self, what):
        """`for a in A [if c]... [for b in B [if d]...]...` — Python's comprehension clauses, in order."""
        out = []
        while self.at_kw("for") or (self.at_word("async") and self.at_kw("for", 1)):
            is_async = self.at_word("async")
            if is_async:
                self.next()
            self.next()
            names = self.targets()
            self.expect_kw("in", f" in {what}")
            it = self.or_expr()
            if is_async:
                it = A.AsyncIter(it, line=it.line, col=it.col)
            cond = None
            while self.at_kw("if"):
                tok = self.next()
                c = self.or_expr()
                cond = c if cond is None else A.Logic("and", cond, c, **self.pos(tok))
            out.append((names, it, cond))
        return out

    def subscript(self):
        tok = self.peek()
        if self.pyc and self.at_op("*"):
            self.next()
            return A.Spread(self.or_expr(), False, **self.pos(tok))
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
        if tok.type == BYTES:
            self.next()
            return A.Bytes(tok.value, **p)
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
                if self.at_word("yield"):
                    e = self.rhs()
                    self.expect_op(")", " to close parenthesis")
                    return e
                e = self.list_item()
                if self._comp_for():                      # (x for x in xs): a lazy generator expression
                    cl = self.comp_clauses("generator expression")
                    self.expect_op(")", " to close the generator expression")
                    (names, it, cond), more = cl[0], cl[1:]
                    return A.GenExp(e, names, it, cond, more or None, **p)
                if self.at_op(",") or isinstance(e, A.Spread):
                    items = [e]
                    while self.accept_op(","):
                        if self.at_op(")"):
                            break
                        items.append(self.list_item())
                    self.expect_op(")")
                    return A.TupleLit(items, **p)
                self.expect_op(")", " to close parenthesis")
                return e
            if tok.value == "[":
                self.next()
                if self.accept_op("]"):
                    return A.ListLit([], **p)
                first = self.list_item()
                if self._comp_for():
                    clauses = self.comp_clauses("list comprehension")
                    self.expect_op("]")
                    (names, it, cond), more = clauses[0], clauses[1:]
                    return A.ListComp(first, names, it, cond, more or None, **p)
                items = [first]
                while self.accept_op(","):
                    if self.at_op("]"):
                        break
                    items.append(self.list_item())
                self.expect_op("]", " to close list")
                return A.ListLit(items, **p)
            if tok.value == "{":
                self.next()
                pairs = []
                items = None                                    # a set literal {a, b} once known
                while not self.at_op("}"):
                    if self.at_op("**"):
                        st = self.next()
                        pairs.append((None, A.Spread(self.bit_or(), True, **self.pos(st))))
                    elif self.at_op("*"):
                        items = (items or []) + [self.list_item()]
                    else:
                        k = self.expr()
                        if items is not None or not self.at_op(":"):
                            if pairs:
                                self.error("expected ':' in dict literal")
                            if items is None and self._comp_for():
                                clauses = self.comp_clauses("set comprehension")
                                self.expect_op("}", " to close set comprehension")
                                (names, it, cond), more = clauses[0], clauses[1:]
                                return A.SetComp(k, names, it, cond, more or None, **p)
                            items = (items or []) + [k]
                        else:
                            self.expect_op(":", " in dict literal")
                            val = self.expr()
                            if not pairs and self._comp_for():
                                clauses = self.comp_clauses("dict comprehension")
                                self.expect_op("}", " to close dict comprehension")
                                (names, it, cond), more = clauses[0], clauses[1:]
                                return A.DictComp(k, val, names, it, cond, more or None, **p)
                            pairs.append((k, val))
                    if not self.accept_op(","):
                        break
                self.expect_op("}", " to close dict")
                if items is not None:
                    return A.SetLit(items, **p)
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
            conv = None
            if len(expr_text) >= 2 and expr_text[-2] == "!" and expr_text[-1] in "rsa" \
                    and not expr_text.endswith("!=!r"):
                conv = expr_text[-1]
                expr_text = expr_text[:-2]
            debug = expr_text.rstrip().endswith("=") and not expr_text.rstrip().endswith(("==", "!=", "<=", ">="))
            if debug:                                     # f"{x=}" prints "x=" and the value's repr
                buf.append(expr_text)
                expr_text = expr_text.rstrip()[:-1]
                if conv is None and spec is None:
                    conv = "r"
            if spec is not None and "{" in spec:          # a nested format spec: f"{x:{width}.{prec}f}"
                spec = A.FStr(parse_fstring(spec, line, col), line=line, col=col)
            try:
                sub = Parser(tokenize(expr_text))
                e = sub.expr()
            except ParseError as err:
                raise ParseError(f"in f-string expression '{expr_text}': {err.msg}", line, col)
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append((e, spec) if conv is None else (e, spec, conv))
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


def _tokens_text(toks):
    """Source-like text of a token run (annotations are kept as text, as `from __future__ import annotations`
    does)."""
    out = []
    for t in toks:
        if t.type in (NEWLINE, INDENT, DEDENT, EOF):
            continue
        v = repr(t.value) if t.type == STRING else {"none": "None", "true": "True", "false": "False"}.get(
            str(t.value), str(t.value)) if t.type == KW else str(t.value)
        if out and v not in ".,[]()" and out[-1] not in ".[(":
            out.append(" ")
        out.append(v)
    return "".join(out).replace(",", ", ")


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
