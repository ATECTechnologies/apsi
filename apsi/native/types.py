"""Native types: which Python values native code takes, and how they are described.

Deliberately free of LLVM imports: deciding whether a call can go native must not load the compiler.
"""
import numpy as np

from ..units import Quantity, DIMLESS

I64, F64, F32, BOOL, NONE, ANY = ("i",), ("f",), ("s",), ("b",), ("n",), ("?",)
SCALARS = (I64, F64, F32, BOOL)
# f2 = IEEE half (float16), h2 = bfloat16: stored in 16 bits, read and computed as f32 (16-bit storage, 32-bit math)
ELEMS = {"float64": "f8", "float32": "f4", "float16": "f2", "bfloat16": "h2", "float8_e4m3fn": "e4",
         "float8_e5m2": "e5", "int64": "i8", "int32": "i4",
         "int16": "i2", "int8": "i1", "uint8": "u1", "bool": "b1"}
ELEM_DTYPE = {v: k for k, v in ELEMS.items()}
ELEM_SCALAR = {"f8": F64, "f4": F32, "f2": F32, "h2": F32, "e4": F32, "e5": F32, "i8": I64, "i4": I64, "i2": I64, "i1": I64, "u1": I64,
               "b1": BOOL}
ELEM_SIZE = {"f8": 8, "f4": 4, "f2": 2, "h2": 2, "e4": 1, "e5": 1, "i8": 8, "i4": 4, "i2": 2, "i1": 1, "u1": 1, "b1": 1}
FLOAT_ELEMS = ("f8", "f4", "f2", "h2", "e4", "e5")


def np_dtype(elem):
    """The NumPy dtype of an element kind (bfloat16 comes from the ml_dtypes package)."""
    if elem in ("h2", "e4", "e5"):
        import ml_dtypes
        return np.dtype({"h2": ml_dtypes.bfloat16, "e4": ml_dtypes.float8_e4m3fn, "e5": ml_dtypes.float8_e5m2}[elem])
    return np.dtype(ELEM_DTYPE[elem])
SCALAR_ELEM = {F64: "f8", F32: "f4", I64: "i8", BOOL: "b1"}


def ARR(elem, ndim, contig=True):
    return ("a", elem, ndim, bool(contig))


def TUP(items):
    return ("t", tuple(items))


def is_arr(t):
    return t[0] == "a"


def is_lazy(t):
    return t[0] == "x"


def is_arrish(t):
    return t[0] in ("a", "x")


def tname(t):
    return {"i": "Int", "f": "Float", "s": "f32", "b": "Bool", "n": "None", "?": "?"}.get(t[0]) or (
        f"Tensor[{ELEM_DTYPE[t[1]]}, rank {t[2]}]" if t[0] in ("a", "x") else
        "(" + ", ".join(tname(x) for x in t[1]) + ")")


# error codes shared with the Python side (apsi.native.__init__)
E_INDEX, E_OVERFLOW, E_ZERODIV, E_ASSERT, E_NEGEXP, E_SHAPE, E_NOMEM, E_NEGSIZE, E_NORETURN, E_EMPTY, \
    E_CONVERT, E_COND = range(1, 13)


def type_of_value(v):
    """(type, dim) of a Python value, or None when it cannot be passed to native code."""
    d = DIMLESS
    if isinstance(v, Quantity):
        d = v.dim
        v = v.v
    if isinstance(v, (bool, np.bool_)):
        return BOOL, d
    if isinstance(v, (int, np.integer)):
        if isinstance(v, int) and not (-(1 << 63) <= v < (1 << 63)):
            return None
        return I64, d
    if isinstance(v, np.float32):
        return F32, d
    if isinstance(v, (float, np.floating)):
        return F64, d
    if isinstance(v, np.ndarray) and type(v) is np.ndarray:
        e = ELEMS.get(v.dtype.name)
        if e is None or v.ndim == 0 or v.ndim > 8:
            return None
        if any(s % v.itemsize for s in v.strides):
            return None
        return ARR(e, v.ndim, v.flags.c_contiguous), d
    return None


