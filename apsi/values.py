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
    __slots__ = ("type", "fields")

    def __init__(self, type_, fields):
        self.type = type_
        self.fields = fields


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
        self.name = name
        self.members = dict(members)


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
