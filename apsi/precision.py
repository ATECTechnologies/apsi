"""Numeric formats: FP64, FP32, FP16, BF16, FP8 (E4M3 / E5M2), FP6 (E3M2 / E2M3), FP4 (E2M1),
INT8, INT4, UINT8 — with exact round-to-nearest-even conversion, scaled quantisation
(per-tensor or per-channel), bit packing (two 4-bit values per byte) and
straight-through gradients so quantisation-aware training works with `grad`.

Formats that numpy has natively (f64, f32, f16) use real hardware arithmetic.
BF16 / FP8 / FP6 / FP4 / INT4 are stored as compact integer codes and decoded to
FP32 for arithmetic — bit-exact with the formats used by modern GPUs
(OCP MX / NVIDIA FP8 E4M3 & E5M2), so you get the real accuracy and the real
memory footprint on any machine.
"""
import math

import numpy as np

from . import autodiff as ad
from .autodiff import Node
from .errors import RuntimeErr
from .units import Quantity


# ------------------------------------------------------------------ minifloat tables
def _minifloat_values(exp_bits, man_bits, bias, finite_only, nan_top):
    """All non-negative values of a sign-magnitude minifloat, indexed by code."""
    vals = []
    for code in range(1 << (exp_bits + man_bits)):
        e = code >> man_bits
        m = code & ((1 << man_bits) - 1)
        if e == 0:
            v = (m / (1 << man_bits)) * 2.0 ** (1 - bias)
        else:
            v = (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
        top_e = (1 << exp_bits) - 1
        if not finite_only and e == top_e:
            v = math.inf if m == 0 else math.nan
        if nan_top and e == top_e and m == (1 << man_bits) - 1:
            v = math.nan
        vals.append(v)
    return np.array(vals)


class Format:
    def __init__(self, name, bits, kind, **kw):
        self.name, self.bits, self.kind = name, bits, kind
        self.__dict__.update(kw)
        if kind == "minifloat":
            self.table = _minifloat_values(self.exp_bits, self.man_bits, self.bias,
                                           self.finite_only, self.nan_top)
            finite = np.isfinite(self.table)
            self.max = float(np.max(self.table[finite]))
            order = np.argsort(np.where(finite, self.table, np.inf))
            self.sorted_codes = order[finite[order]]
            self.sorted_vals = self.table[self.sorted_codes]
        elif kind in ("int", "uint"):
            self.max = float(self.qmax)
        elif kind == "bf16":
            self.max = 3.3895313892515355e38
        else:
            self.max = float(np.finfo(self.np_dtype).max)

    def __repr__(self):
        return self.name


FORMATS = {
    "f64": Format("f64", 64, "native", np_dtype=np.float64),
    "f32": Format("f32", 32, "native", np_dtype=np.float32),
    "f16": Format("f16", 16, "native", np_dtype=np.float16),
    "bf16": Format("bf16", 16, "bf16"),
    # OCP 8-bit floats (as on NVIDIA Hopper/Blackwell, AMD MI300)
    "fp8_e4m3": Format("fp8_e4m3", 8, "minifloat", exp_bits=4, man_bits=3, bias=7, finite_only=True, nan_top=True),
    "fp8_e5m2": Format("fp8_e5m2", 8, "minifloat", exp_bits=5, man_bits=2, bias=15, finite_only=False, nan_top=False),
    # OCP MX 6-bit and 4-bit floats
    "fp6_e3m2": Format("fp6_e3m2", 6, "minifloat", exp_bits=3, man_bits=2, bias=3, finite_only=True, nan_top=False),
    "fp6_e2m3": Format("fp6_e2m3", 6, "minifloat", exp_bits=2, man_bits=3, bias=1, finite_only=True, nan_top=False),
    "fp4_e2m1": Format("fp4_e2m1", 4, "minifloat", exp_bits=2, man_bits=1, bias=1, finite_only=True, nan_top=False),
    "int8": Format("int8", 8, "int", qmax=127),
    "int4": Format("int4", 4, "int", qmax=7),
    "uint8": Format("uint8", 8, "uint", qmax=255),
}
ALIASES = {"fp64": "f64", "fp32": "f32", "fp16": "f16", "fp8": "fp8_e4m3", "bf8": "fp8_e5m2",
           "e4m3": "fp8_e4m3", "e5m2": "fp8_e5m2", "fp6": "fp6_e3m2", "fp4": "fp4_e2m1"}


def get_format(f):
    name = getattr(f, "name", f)
    name = ALIASES.get(name, name)
    if name not in FORMATS:
        raise RuntimeErr(f"unknown numeric format '{name}'; choose one of {', '.join(FORMATS)}")
    return FORMATS[name]


# ------------------------------------------------------------------ element conversion
def _bf16_round(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        r = np.where(nan, np.uint16(0x7FC0), r)
    return r


def _bf16_decode(codes):
    return (codes.astype(np.uint32) << 16).view(np.float32)


def _minifloat_encode(fmt, x):
    """Round-to-nearest-even onto the format grid, saturating at ±max. Returns uint8 codes."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.signbit(x)
    a = np.abs(x)
    vals, codes = fmt.sorted_vals, fmt.sorted_codes
    a_c = np.minimum(a, fmt.max)
    idx = np.searchsorted(vals, a_c)
    idx = np.clip(idx, 1, len(vals) - 1)
    lo, hi = vals[idx - 1], vals[idx]
    pick_hi = (a_c - lo) > (hi - a_c)
    tie = (a_c - lo) == (hi - a_c)
    even_hi = (codes[idx] & 1) == 0
    pick_hi = pick_hi | (tie & even_hi)
    code = np.where(pick_hi, codes[idx], codes[idx - 1]).astype(np.uint8)
    nbits_mag = fmt.exp_bits + fmt.man_bits
    code = code | (sign.astype(np.uint8) << nbits_mag)
    nan = np.isnan(x)
    if nan.any():
        nan_code = np.nanargmax(np.isnan(fmt.table)) if np.isnan(fmt.table).any() else (1 << nbits_mag) - 1
        code = np.where(nan, np.uint8(nan_code), code)
    return code


def _minifloat_decode(fmt, codes):
    nbits_mag = fmt.exp_bits + fmt.man_bits
    codes = codes.astype(np.int64)
    mag = fmt.table[codes & ((1 << nbits_mag) - 1)]
    sign = np.where((codes >> nbits_mag) & 1, -1.0, 1.0)
    return (mag * sign).astype(np.float32)


# ------------------------------------------------------------------ quantised tensors
class QTensor:
    """A tensor stored in a compact format: codes (+ packed nibbles) and a scale."""
    __slots__ = ("codes", "scale", "fmt", "shape", "axis", "dim", "packed")

    def __init__(self, codes, scale, fmt, shape, axis, dim=None, packed=False):
        self.codes, self.scale, self.fmt, self.shape = codes, scale, fmt, tuple(shape)
        self.axis, self.dim, self.packed = axis, dim, packed

    @property
    def nbytes(self):
        n = self.codes.nbytes
        if isinstance(self.scale, np.ndarray):
            n += self.scale.nbytes
        return n

    @property
    def ndim(self):
        return len(self.shape)

    def dequantize(self):
        return dequantize(self)

    def __repr__(self):
        return f"QTensor[{'x'.join(map(str, self.shape))}, {self.fmt.name}, {self.nbytes} bytes]"


def _pack4(codes):
    flat = codes.ravel().astype(np.uint8) & 0x0F
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, np.uint8)])
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def _unpack4(packed, n):
    out = np.empty(packed.size * 2, np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out[:n]


def _pack6(codes):
    flat = codes.ravel().astype(np.uint32) & 0x3F
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.uint32)])
    g = flat.reshape(-1, 4)
    word = g[:, 0] | (g[:, 1] << 6) | (g[:, 2] << 12) | (g[:, 3] << 18)
    out = np.empty((word.size, 3), np.uint8)
    out[:, 0] = word & 0xFF
    out[:, 1] = (word >> 8) & 0xFF
    out[:, 2] = (word >> 16) & 0xFF
    return out.ravel()


def _unpack6(packed, n):
    b = packed.reshape(-1, 3).astype(np.uint32)
    word = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    out = np.empty((word.size, 4), np.uint8)
    for k in range(4):
        out[:, k] = (word >> (6 * k)) & 0x3F
    return out.ravel()[:n]


def _scale_for(x, fmt, axis):
    a = np.abs(x)
    if axis is None:
        m = float(np.max(a)) if a.size else 0.0
    else:
        red = tuple(i for i in range(x.ndim) if i != axis % x.ndim)
        m = np.max(a, axis=red, keepdims=True)
    s = np.where(np.asarray(m) > 0, np.asarray(m) / fmt.max, 1.0)
    return float(s) if axis is None else s.astype(np.float32)


def quantize(x, fmt, scale="auto", axis=None):
    """Quantise a tensor. scale="auto" maps max|x| to the format's largest value
    (per tensor, or per channel along `axis`); scale=1.0 keeps raw values (no scaling)."""
    fmt = get_format(fmt)
    dim = x.dim if isinstance(x, Quantity) else None
    xv = np.asarray(x.v if isinstance(x, Quantity) else (x.value if isinstance(x, Node) else x), dtype=np.float64)
    if fmt.kind == "native":
        return x.astype(fmt.np_dtype) if isinstance(x, np.ndarray) else np.asarray(xv, fmt.np_dtype)
    if fmt.kind == "bf16":
        s = 1.0 if scale == "auto" else float(scale)
        return QTensor(_bf16_round(xv / s), s, fmt, xv.shape, None, dim)
    s = _scale_for(xv, fmt, axis) if scale == "auto" else float(scale)
    y = xv / s
    if fmt.kind == "minifloat":
        codes = _minifloat_encode(fmt, y)
    elif fmt.kind == "int":
        codes = (np.clip(np.rint(y), -fmt.qmax, fmt.qmax).astype(np.int8)).view(np.uint8)
    else:  # uint8: asymmetric [min, max] -> [0, 255]
        lo = float(np.min(xv)) if xv.size else 0.0
        hi = float(np.max(xv)) if xv.size else 1.0
        s = (hi - lo) / 255.0 if hi > lo else 1.0
        codes = np.clip(np.rint((xv - lo) / s), 0, 255).astype(np.uint8)
        q = QTensor(codes, s, fmt, xv.shape, None, dim)
        q.axis = ("zero", lo)
        return q
    packed = fmt.bits in (4, 6)
    if fmt.bits == 4:
        codes = _pack4(codes)
    elif fmt.bits == 6:
        codes = _pack6(codes)
    return QTensor(codes, s, fmt, xv.shape, axis, dim, packed)


def dequantize(q):
    if not isinstance(q, QTensor):
        return q
    fmt = q.fmt
    n = int(np.prod(q.shape)) if q.shape else 1
    if q.packed:
        codes = (_unpack4 if fmt.bits == 4 else _unpack6)(q.codes, n).reshape(q.shape)
    else:
        codes = q.codes.reshape(q.shape)
    if fmt.kind == "bf16":
        v = _bf16_decode(codes) * np.float32(q.scale)
    elif fmt.kind == "minifloat":
        v = _minifloat_decode(fmt, codes) * np.asarray(q.scale, dtype=np.float32)
    elif fmt.kind == "int":
        c = codes.astype(np.uint8)
        if fmt.bits == 4:
            c = np.where(c > 7, c.astype(np.int16) - 16, c.astype(np.int16))
        else:
            c = c.view(np.int8)
        v = c.astype(np.float32) * np.asarray(q.scale, dtype=np.float32)
    else:
        v = codes.astype(np.float32) * np.float32(q.scale) + np.float32(q.axis[1])
    v = v.reshape(q.shape) if q.shape else v.reshape(())
    if q.dim is not None:
        return Quantity(v, q.dim)
    return v


# ------------------------------------------------------------------ fake quantisation
def round_to(x, fmt, scale="auto", axis=None):
    """Round x to what the format can represent and return it as a normal tensor
    ("fake quantisation"). Gradients pass straight through, so this works inside training."""
    f = get_format(fmt)
    if isinstance(x, Quantity):
        return Quantity(round_to(x.v, f, scale, axis), x.dim, x.disp)
    if isinstance(x, Node):
        rounded = np.asarray(round_to(x.value, f, scale, axis), dtype=np.float64)
        return ad._op(lambda a: rounded, lambda g, out, need, a: (g,), [x])   # straight-through
    if f.kind == "native":
        return np.asarray(x).astype(f.np_dtype).astype(np.float64) if np.ndim(x) else float(np.asarray(x, f.np_dtype))
    r = dequantize(quantize(x, f, scale, axis))
    r = np.asarray(r, dtype=np.float64)
    return r if r.ndim else float(r)


def qmatmul(a, b):
    """Matrix multiply with quantised operands: decode to FP32, multiply, accumulate in FP32
    (the same numerics as low-precision tensor-core GEMMs)."""
    av = dequantize(a) if isinstance(a, QTensor) else np.asarray(a, dtype=np.float32)
    bv = dequantize(b) if isinstance(b, QTensor) else np.asarray(b, dtype=np.float32)
    return av @ bv


def nbytes(x):
    if isinstance(x, QTensor):
        return x.nbytes
    if isinstance(x, Quantity):
        x = x.v
    return int(np.asarray(x).nbytes)


def quant_error(x, fmt, axis=None):
    """Relative RMS error introduced by storing x in `fmt`."""
    xv = np.asarray(x.v if isinstance(x, Quantity) else x, dtype=np.float64)
    r = np.asarray(round_to(xv, fmt, "auto", axis), dtype=np.float64)
    denom = math.sqrt(float(np.mean(xv * xv))) or 1.0
    return math.sqrt(float(np.mean((r - xv) ** 2))) / denom


def format_info(fmt):
    from .values import Record
    f = get_format(fmt)
    if f.kind == "native":
        fi = np.finfo(f.np_dtype)
        return Record(name=f.name, bits=f.bits, max=float(fi.max), eps=float(fi.eps), tiny=float(fi.tiny))
    if f.kind == "bf16":
        return Record(name="bf16", bits=16, max=3.3895313892515355e38, eps=2.0 ** -7, tiny=2.0 ** -126)
    if f.kind == "minifloat":
        pos = f.table[np.isfinite(f.table) & (f.table > 0)]
        return Record(name=f.name, bits=f.bits, max=f.max, eps=2.0 ** -f.man_bits, tiny=float(np.min(pos)),
                      values=int(np.isfinite(f.table).sum() * 2 - 1))
    return Record(name=f.name, bits=f.bits, max=f.max, eps=1.0, tiny=1.0)


def module():
    from .values import Dtype
    out = {"quantize": quantize, "dequantize": dequantize, "round_to": round_to, "qmatmul": qmatmul,
           "nbytes": nbytes, "quant_error": quant_error, "format_info": format_info}
    for name in list(FORMATS) + list(ALIASES):
        out[name] = Dtype(name, FORMATS[ALIASES.get(name, name)].__dict__.get("np_dtype"))
    return out
