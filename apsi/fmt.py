"""Human-friendly formatting of Apsi values."""
from fractions import Fraction

import numpy as np

from .autodiff import Node
from .units import Quantity, unit_string
from .uncertain import Uncertain, format_uncertain
from .values import (Function, StructType, StructInstance, Record, RangeVal, Namespace,
                     BoundMethod, Dtype)


def fmt_num(x, spec=None):
    if spec:
        if isinstance(x, Fraction):
            x = float(x)
        return format(x, spec)
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, Fraction):
        return str(x)
    if isinstance(x, (complex, np.complexfloating)):
        return f"{fmt_num(x.real)}{'+' if x.imag >= 0 else '-'}{fmt_num(abs(x.imag))}i"
    x = float(x)
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    s = format(x, ".12g")
    if "e" not in s and "." not in s:
        s += ".0"
    return s


def _short(v):
    s = format(float(v), ".6g")
    if "e" not in s and "." not in s and "n" not in s:
        s += "."
    return s


def fmt_array(a, spec=None):
    a = np.asarray(a)
    if a.ndim == 0:
        return fmt_num(a.item(), spec)
    if a.dtype == bool:
        formatter = {"bool": lambda v: "true" if v else "false"}
    elif spec:
        formatter = {"float_kind": lambda v: format(v, spec), "int": lambda v: format(v, spec)}
    else:
        formatter = {"float_kind": _short}
    return np.array2string(a, separator=", ", threshold=40, edgeitems=3,
                           max_line_width=100, formatter=formatter)


def fmt_quantity(q, spec=None):
    if isinstance(q.v, Uncertain):
        name, scale = q.disp if q.disp else (unit_string(q.dim), 1.0)
        return f"({format_uncertain(q.v.x, q.v.sigma, scale)}) {name}"
    if q.disp:
        name, scale = q.disp
        val = q.v / scale
    else:
        name, val = unit_string(q.dim), q.v
    if isinstance(val, Node):
        val = val.value
    s = fmt_array(val, spec) if np.ndim(val) else fmt_num(val, spec)
    return f"{s} {name}"


def shape_str(shape):
    return "x".join(str(s) for s in shape)


def fmt(v, top=True, spec=None):
    if v is None:
        return "none"
    if isinstance(v, str):
        return (format(v, spec) if spec else v) if top else '"' + v.replace('"', '\\"') + '"'
    if isinstance(v, (bool, np.bool_, int, float, Fraction, np.number, complex)):
        return fmt_num(v, spec)
    if isinstance(v, Quantity):
        if not top and np.ndim(getattr(v.v, "value", v.v)) and np.size(getattr(v.v, "value", v.v)) > 8:
            return f"Tensor[{shape_str(v.shape)}] [{v.unit}]"
        return fmt_quantity(v, spec)
    if isinstance(v, Node):
        return fmt(v.value, top, spec)
    if isinstance(v, Uncertain):
        return format_uncertain(v.x, v.sigma)
    if isinstance(v, np.ndarray):
        if not top and v.size > 8:
            return f"Tensor[{shape_str(v.shape)}]"
        return fmt_array(v, spec)
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x, False) for x in v) + "]"
    if isinstance(v, tuple):
        inner = ", ".join(fmt(x, False) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{fmt(k, False)}: {fmt(x, False)}" for k, x in v.items()) + "}"
    if isinstance(v, StructInstance):
        inner = ", ".join(f"{k}={fmt(x, False)}" for k, x in v.fields.items())
        return f"{v.type.name}({inner})"
    if isinstance(v, Record):
        inner = ", ".join(f"{k}={fmt(x, False)}" for k, x in v.fields.items())
        return f"record({inner})"
    if isinstance(v, Function):
        return f"<fn {v.name}>"
    if isinstance(v, BoundMethod):
        return f"<method {v.name}>"
    if isinstance(v, StructType):
        return f"<struct {v.name}>"
    if isinstance(v, RangeVal):
        return f"{fmt(v.start, False)}{'..=' if v.inclusive else '..'}{fmt(v.end, False)}"
    if isinstance(v, Namespace):
        return f"<module {v.name}>"
    if isinstance(v, Dtype):
        return v.name
    if type(v).__name__ == "QTensor":
        return repr(v)
    if callable(v) and type(v).__repr__ is object.__repr__:
        return f"<builtin {getattr(v, '__name__', 'fn')}>"
    return repr(v) if not isinstance(v, str) else v


def type_name(v):
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return "Bool"
    if isinstance(v, (int, np.integer)):
        return "Int"
    if isinstance(v, (float, np.floating)):
        return "Float"
    if isinstance(v, Fraction):
        return "Exact"
    if isinstance(v, Uncertain):
        return "Uncertain"
    if isinstance(v, str):
        return "Str"
    if isinstance(v, Quantity):
        from .units import dim_name
        d = dim_name(v.dim).split(" ")[0]
        return f"Tensor[{shape_str(v.shape)}, {d}]" if v.ndim else d
    if isinstance(v, Node):
        return "Traced[" + type_name(v.value) + "]"
    if isinstance(v, np.ndarray):
        return f"Tensor[{shape_str(v.shape)}, {v.dtype}]"
    if isinstance(v, list):
        return "List"
    if isinstance(v, tuple):
        return "Tuple"
    if isinstance(v, dict):
        return "Dict"
    if isinstance(v, StructInstance):
        return v.type.name
    if isinstance(v, Record):
        return "Record"
    if isinstance(v, RangeVal):
        return "Range"
    if isinstance(v, Namespace):
        return "Module"
    if callable(v):
        return "Fn"
    return type(v).__name__
