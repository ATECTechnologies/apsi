"""Runtime value types of Apsi (functions, structs, records, ranges, modules)."""
import numpy as np

from .units import Quantity, _is_zero


class Function:
    """A user-defined Apsi function or lambda (a closure)."""

    def __init__(self, interp, name, params, ret, body, is_expr, closure, shape_params=()):
        self.interp = interp
        self.name = name
        self.params = params
        self.ret = ret
        self.body = body
        self.is_expr = is_expr
        self.closure = closure
        self.shape_params = shape_params
        self.precision = None

    def __call__(self, *args, **kwargs):
        return self.interp.call_function(self, list(args), kwargs)

    def __repr__(self):
        return f"<fn {self.name}>"


class BoundMethod:
    def __init__(self, fn, obj):
        self.fn = fn
        self.obj = obj
        self.name = f"{obj.type.name}.{fn.name}"

    def __call__(self, *args, **kwargs):
        return self.fn(self.obj, *args, **kwargs)


class StructType:
    def __init__(self, interp, name, fields, methods, env):
        self.interp = interp
        self.name = name
        self.fields = fields          # list of (name, type, default_expr)
        self.methods = methods        # name -> Function
        self.env = env

    def __call__(self, *args, **kwargs):
        return self.interp.construct(self, list(args), kwargs)


class StructInstance:
    """An instance of a struct. Methods with the names below make it work with operators and built-ins, the way
    Python's special methods do (so converted Python classes keep their behaviour):

        add sub mul div mod pow neg   →  + - * / % ** unary -        eq lt le gt ge  →  == < <= > >=
        len get set contains iter     →  len(x) x[i] x[i]=v  in  for     to_str      →  print / str / f-strings
        call hash bool                →  x(...)  dict keys  if x
    """
    __slots__ = ("type", "fields")

    def __init__(self, type_, fields):
        self.type = type_
        self.fields = fields

    def _m(self, name):
        return self.type.methods.get(name) if hasattr(self.type, "methods") else None

    def _call(self, name, *args):
        m = self._m(name)
        if m is None:
            return NotImplemented
        return m(self, *args)

    def __add__(self, o): return self._call("add", o)
    def __radd__(self, o): return self._call("radd", o) if self._m("radd") else self._call("add", o)
    def __sub__(self, o): return self._call("sub", o)
    def __rsub__(self, o): return self._call("rsub", o)
    def __mul__(self, o): return self._call("mul", o)
    def __rmul__(self, o): return self._call("rmul", o) if self._m("rmul") else self._call("mul", o)
    def __truediv__(self, o): return self._call("div", o)
    def __rtruediv__(self, o): return self._call("rdiv", o)
    def __floordiv__(self, o): return self._call("floordiv", o)
    def __mod__(self, o): return self._call("mod", o)
    def __pow__(self, o): return self._call("pow", o)
    def __matmul__(self, o): return self._call("matmul", o)
    def __lt__(self, o): return self._call("lt", o)
    def __le__(self, o): return self._call("le", o)
    def __gt__(self, o): return self._call("gt", o)
    def __ge__(self, o): return self._call("ge", o)

    def __neg__(self):
        r = self._call("neg")
        if r is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'neg' method for unary -")
        return r

    def __eq__(self, o):
        if self._m("eq") is not None:
            return bool(self._call("eq", o))
        if isinstance(o, StructInstance):
            return o is self or (o.type is self.type and _fields_equal(self.fields, o.fields))
        return False

    def __ne__(self, o):
        r = self.__eq__(o)
        return not r

    def __hash__(self):
        if self._m("hash") is not None:
            return hash(self._call("hash"))
        return id(self) >> 4

    def __len__(self):
        r = self._call("len")
        if r is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'len' method")
        return int(r)

    def __getitem__(self, k):
        r = self._call("get", k)
        if r is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'get' method for indexing")
        return r

    def __setitem__(self, k, v):
        if self._call("set", k, v) is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'set' method for index assignment")

    def __contains__(self, k):
        r = self._call("contains", k)
        if r is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'contains' method for 'in'")
        return bool(r)

    def __iter__(self):
        r = self._call("iter")
        if r is NotImplemented:
            raise TypeError(f"{self.type.name} has no 'iter' method for 'for'")
        return iter(r)

    def __bool__(self):
        if self._m("bool") is not None:
            return bool(self._call("bool"))
        if self._m("len") is not None:
            return len(self) > 0
        return True

    def __call__(self, *args, **kwargs):
        m = self._m("call")
        if m is None:
            raise TypeError(f"{self.type.name} is not callable (define a 'call' method)")
        return m(self, *args, **kwargs)

    def __str__(self):
        if self._m("to_str") is not None:
            return str(self._call("to_str"))
        from .fmt import fmt
        return fmt(self)


def _fields_equal(a, b):
    try:
        return all(bool(np.all(a[k] == b.get(k))) for k in a) and a.keys() == b.keys()
    except Exception:                                   # noqa: BLE001
        return False


class Record:
    """A lightweight named-field container: record(W=..., b=...)."""
    __slots__ = ("fields",)

    def __init__(self, **kw):
        self.fields = dict(kw)

    def __getitem__(self, k):
        return self.fields[k]

    def __getattr__(self, k):            # lets Python-side library code write p.W
        if k == "fields":
            raise AttributeError(k)
        try:
            return self.fields[k]
        except KeyError:
            raise AttributeError(k) from None


class RangeVal:
    def __init__(self, start, end, inclusive=False):
        self.start, self.end, self.inclusive = start, end, inclusive

    def _ints(self):
        return all(isinstance(x, (int, np.integer)) and not isinstance(x, bool)
                   for x in (self.start, self.end))

    def __iter__(self):
        if self._ints():
            return iter(range(self.start, self.end + (1 if self.inclusive else 0)))
        if isinstance(self.start, Quantity) or isinstance(self.end, Quantity):
            raise TypeError("cannot iterate over a range of quantities; use linspace()")
        x = self.start
        out = []
        while x < self.end or (self.inclusive and x == self.end):
            out.append(x)
            x = x + 1
        return iter(out)

    def __len__(self):
        return len(list(iter(self)))

    def __contains__(self, x):
        return self.start <= x and (x <= self.end if self.inclusive else x < self.end)


class Namespace:
    def __init__(self, name, members):
        self.members = dict(members)
        # members as instance attributes: `math.sqrt` is a plain dict lookup, not a __getattr__ call per access
        self.__dict__.update((k, v) for k, v in self.members.items()
                             if isinstance(k, str) and k != "members" and not k.startswith("__"))
        self.name = name

    def __getattr__(self, k):            # direct attribute access (Python-exact files, library code)
        members = self.__dict__.get("members")
        if members is not None and k in members:
            return members[k]
        raise AttributeError(f"module '{self.__dict__.get('name')}' has no attribute '{k}'")

    def __dir__(self):
        return list(self.members)

    def __repr__(self):
        return f"<module '{self.name}'>"


class Dtype:
    def __init__(self, name, np_dtype):
        self.name = name
        self.np = np_dtype

    def __repr__(self):
        return self.name


class UnitNamespace:
    """`u.km`, `u.eV` ... -> one of that unit as a Quantity."""
    name = "u"

    def get(self, attr):
        from .units import parse_unit, make
        scale, dim = parse_unit(attr)
        return make(scale, dim, (attr, scale)) if not dim.dimensionless else scale


# ---------------------------------------------------------------- pytrees
def tree_map(f, *trees):
    t0 = trees[0]
    if isinstance(t0, list):
        return [tree_map(f, *xs) for xs in zip(*trees)]
    if isinstance(t0, tuple):
        return tuple(tree_map(f, *xs) for xs in zip(*trees))
    if isinstance(t0, dict):
        return {k: tree_map(f, *(t[k] for t in trees)) for k in t0}
    if isinstance(t0, Record):
        return Record(**{k: tree_map(f, *(t.fields[k] for t in trees)) for k in t0.fields})
    if isinstance(t0, StructInstance):
        return StructInstance(t0.type, {k: tree_map(f, *(t.fields[k] for t in trees))
                                        for k in t0.fields})
    return f(*trees)


def tree_leaves(t):
    out = []
    tree_map(lambda x: out.append(x), t)
    return out
