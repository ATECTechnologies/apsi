"""Exceptions in Apsi: `try / except / else / finally` and `raise`.

    try:
        let x = parse_number(text)
    except ValueError, KeyError as e:        # several kinds; `(ValueError, KeyError)` works too
        print(f"bad input: {e.message}")
    except as e:                             # anything else (also written `except Error as e`)
        raise                                # re-raise the error being handled
    else:
        print("no error")
    finally:
        close_everything()

    raise ValueError("speed must be positive")
    raise "plain message"                    # a RuntimeError with this message

Kinds are the familiar Python names, so converted Python code keeps its exact error handling. Errors Apsi raises
itself are matched too: a division by zero is a ZeroDivisionError, a missing dictionary key a KeyError, an index
outside a list an IndexError, a unit mismatch a UnitError (and a ValueError), a wrong type a TypeError.
The value bound with `as e` has .kind, .message and .line; str(e) is the message.
"""
import builtins

from .errors import (ApsiError, AssertionFailed, NameErr, RuntimeErr, ShapeError, TypeCheckError, UnitError)

_B = builtins
# kind name -> Python exception classes it matches (Apsi's own errors are classified by _apsi_kinds below)
KINDS = {
    "Error": (Exception,), "Exception": (Exception,), "BaseException": (Exception,),
    "ValueError": (_B.ValueError,), "TypeError": (_B.TypeError,), "KeyError": (_B.KeyError,),
    "IndexError": (_B.IndexError,), "LookupError": (_B.LookupError,), "ZeroDivisionError": (_B.ZeroDivisionError,),
    "ArithmeticError": (_B.ArithmeticError,), "OverflowError": (_B.OverflowError,),
    "FloatingPointError": (_B.FloatingPointError,), "AttributeError": (_B.AttributeError,),
    "NameError": (_B.NameError,), "RuntimeError": (_B.RuntimeError,), "RecursionError": (_B.RecursionError,),
    "NotImplementedError": (_B.NotImplementedError,), "AssertionError": (_B.AssertionError,),
    "StopIteration": (_B.StopIteration,), "OSError": (_B.OSError,), "IOError": (_B.OSError,),
    "EnvironmentError": (_B.OSError,), "FileNotFoundError": (_B.FileNotFoundError,),
    "FileExistsError": (_B.FileExistsError,), "IsADirectoryError": (_B.IsADirectoryError,),
    "PermissionError": (_B.PermissionError,), "TimeoutError": (_B.TimeoutError,),
    "ConnectionError": (_B.ConnectionError,), "UnicodeError": (_B.UnicodeError,),
    "UnicodeDecodeError": (_B.UnicodeDecodeError,), "UnicodeEncodeError": (_B.UnicodeEncodeError,),
    "MemoryError": (_B.MemoryError,), "EOFError": (_B.EOFError,), "ImportError": (_B.ImportError,),
    "ModuleNotFoundError": (_B.ModuleNotFoundError,),
    "UnitError": (UnitError,), "ShapeError": (ShapeError,),
}
ALIASES = {"json.JSONDecodeError": "ValueError", "JSONDecodeError": "ValueError", "subprocess.CalledProcessError":
           "RuntimeError", "socket.timeout": "TimeoutError", "decimal.InvalidOperation": "ArithmeticError"}
# classes that `raise NAME(...)` can create (the same objects Python uses, so they match Python's own errors)
RAISABLE = {k: v[0] for k, v in KINDS.items() if k not in ("Error", "Exception", "BaseException", "UnitError",
                                                          "ShapeError")}
_PARENTS = {"ZeroDivisionError": "ArithmeticError", "OverflowError": "ArithmeticError",
            "FloatingPointError": "ArithmeticError", "KeyError": "LookupError", "IndexError": "LookupError",
            "FileNotFoundError": "OSError", "FileExistsError": "OSError", "PermissionError": "OSError",
            "IsADirectoryError": "OSError", "TimeoutError": "OSError", "ConnectionError": "OSError",
            "RecursionError": "RuntimeError", "NotImplementedError": "RuntimeError",
            "UnicodeDecodeError": "ValueError", "UnicodeEncodeError": "ValueError", "UnicodeError": "ValueError",
            "UnitError": "ValueError", "ShapeError": "ValueError", "ModuleNotFoundError": "ImportError"}


class UserError(Exception):
    """An error raised from Apsi code with `raise "message"` or `raise Error("message")`."""


RAISABLE["Error"] = UserError
RAISABLE["Exception"] = UserError


def _apsi_kinds(ex):
    """The Python-style kinds an error Apsi raised itself belongs to."""
    msg = str(getattr(ex, "msg", ex)).lower()
    out = set()
    if isinstance(ex, UnitError):
        out.add("UnitError")
    if isinstance(ex, ShapeError):
        out.add("ShapeError")
    if isinstance(ex, TypeCheckError):
        out.add("TypeError")
    if isinstance(ex, NameErr):
        out.add("NameError")
    if isinstance(ex, AssertionFailed):
        out.add("AssertionError")
    if type(ex).__name__ == "PermissionErr" or "permission" in msg and "needs" in msg:
        out.add("PermissionError")
    if isinstance(ex, RuntimeErr):
        if "division by zero" in msg or "divide by zero" in msg or "modulo by zero" in msg:
            out.add("ZeroDivisionError")
        elif "out of range" in msg and "index" in msg:
            out.add("IndexError")
        elif "key" in msg and ("not found" in msg or "missing" in msg or "no key" in msg):
            out.add("KeyError")
        elif "cannot find file" in msg or "no such file" in msg:
            out.add("FileNotFoundError")
        elif "not a number" in msg or "invalid literal" in msg or "could not convert" in msg:
            out.add("ValueError")
        elif "has no attribute" in msg or "no field" in msg:
            out.add("AttributeError")
        elif "is not callable" in msg or "unsupported" in msg and "type" in msg:
            out.add("TypeError")
        else:
            out.add("RuntimeError")
    return out


def kinds_of(ex):
    if isinstance(ex, ErrorValue):
        ex = ex.error
    ks = set()
    if isinstance(ex, ApsiError):
        ks = _apsi_kinds(ex)
    elif isinstance(ex, UserError):
        ks = {getattr(ex, "kind", None) or "RuntimeError"}
    else:
        for k, classes in KINDS.items():
            if k not in ("Error", "Exception", "BaseException") and isinstance(ex, classes):
                ks.add(k)
    for k in list(ks):
        p = _PARENTS.get(k)
        while p:
            ks.add(p)
            p = _PARENTS.get(p)
    return ks


def matches(ex, kinds, classes=()):
    """True when the exception `ex` is one of `kinds` (names as written after `except`) or an instance of one of
    `classes` (exception classes the program defined or imported)."""
    if isinstance(ex, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        return False
    if classes:
        orig = getattr(ex, "original", None)
        if isinstance(ex, classes) or (orig is not None and isinstance(orig, classes)):
            return True
    have = None
    for k in kinds:
        k = ALIASES.get(k, k).split(".")[-1]
        if k in ("Error", "Exception", "BaseException"):
            return True
        if have is None:
            have = kinds_of(ex)
        if k in have:
            return True
        # a class defined by the program (struct used as an exception) is matched by name
        if getattr(ex, "kind", None) == k:
            return True
    return False


class ErrorValue:
    """What `except ... as e` binds: e.kind, e.message, e.line; str(e) gives the message."""
    __slots__ = ("error", "kind", "message", "line")

    def __init__(self, error):
        self.error = error
        ks = kinds_of(error)
        order = ["ZeroDivisionError", "KeyError", "IndexError", "FileNotFoundError", "PermissionError", "UnitError",
                 "ShapeError", "TypeError", "ValueError", "NameError", "AttributeError", "AssertionError",
                 "StopIteration", "TimeoutError", "OSError", "RuntimeError"]
        self.kind = getattr(error, "kind", None) or next((k for k in order if k in ks), type(error).__name__)
        if isinstance(error, ApsiError):
            self.message = str(error.msg) if hasattr(error, "msg") else str(error)
        elif isinstance(error, KeyError) and error.args:
            self.message = str(error.args[0])
        else:
            self.message = str(error)
        self.line = getattr(error, "line", None)

    def __getattr__(self, name):
        # fields of an error struct raised by the program (raise InsufficientFunds("…", amount=30))
        err = object.__getattribute__(self, "error")
        payload = getattr(err, "payload", None)
        if payload is not None and name in getattr(payload, "fields", {}):
            return payload.fields[name]
        if not name.startswith("_") and name in getattr(err, "__dict__", {}):
            return err.__dict__[name]                  # e.code of a database error, e.status of an HTTP error …
        raise AttributeError(name)

    @property
    def args(self):
        return [self.message]

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message


def value(ex):
    """What `except … as e` binds. Python's own exceptions (and classes the program defined) are bound as they are —
    isinstance(), e.args, e.__cause__, raise e all behave as in Python — with .kind/.message/.line added; Apsi's
    own errors are wrapped in an ErrorValue."""
    if isinstance(ex, ErrorValue):
        return ex
    orig = getattr(ex, "original", None)
    real = orig if isinstance(orig, BaseException) and not isinstance(orig, ApsiError) else ex
    if not isinstance(real, (ApsiError, UserError)):
        try:
            ev = ErrorValue(real)
            if "kind" not in getattr(real, "__dict__", {}):
                real.kind = ev.kind
            if "message" not in getattr(real, "__dict__", {}):
                real.message = ev.message
            if "line" not in getattr(real, "__dict__", {}):
                real.line = getattr(ex, "line", None)
            return real
        except (AttributeError, TypeError):
            pass
    return ErrorValue(ex)


def to_raise(v):
    """Turn the value after `raise` into a Python exception to raise."""
    if v is None:
        return None
    if isinstance(v, ErrorValue):
        return v.error
    if isinstance(v, BaseException):
        return v
    if isinstance(v, type) and issubclass(v, BaseException):
        return v()
    if isinstance(v, str):
        e = UserError(v)
        e.kind = "RuntimeError"
        return e
    # a struct instance used as an error: its type name is the kind, a `message` field (if any) the text
    tname = getattr(getattr(v, "type", None), "name", None)
    if tname:
        fields = getattr(v, "fields", {}) or {}
        e = UserError(str(fields.get("message", fields.get("msg", tname))))
        e.kind = tname
        e.payload = v
        return e
    e = UserError(str(v))
    e.kind = "RuntimeError"
    return e


def builtins_table():
    """Names added to Apsi's global scope: the raisable error classes."""
    return dict(RAISABLE)
