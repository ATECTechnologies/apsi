"""Apsi native compiler: checked Apsi AST -> typed, unit-erased LLVM IR.

A function is *specialised* on the types of the arguments it is called with (Int, Float, f32, Bool, tensors of a
given dtype / rank / layout) and on their physical dimensions. Everything else is decided at compile time:

* **Types** are inferred by abstract execution: the body is walked without emitting code until every variable's
  type stops changing, then emitted for real.
* **Units** are checked statically along the way (adding metres to seconds is a UnitError at compile time) and
  then erased: the machine code works on plain SI numbers, and the result gets its unit back at the boundary.
* **Safety** (default): integer overflow, division by zero, out-of-range indices and failed asserts become normal
  Apsi errors with the source line. Index checks whose loop range can be proven safe up front are hoisted out of
  inner loops (the loop is versioned), so the hot loop has no checks and can be vectorised.
* **Semantics follow Apsi**: Python-style floor division and modulo, negative indices, `min`/`max` order,
  round-half-even, IEEE floats (x/0.0 -> inf, as for tensors). Strict IEEE arithmetic by default, so results match
  the bytecode engine bit for bit; `@native(fast)` allows reassociation/FMA and removes the checks.
* **Tensors**: element access, row/slice views, and whole-array expressions (`a * b + sin(c)`), which are *fused*
  into one loop with no temporaries, broadcast like NumPy, and run on all cores when large. Reductions (`sum`,
  `mean`, `min`, `max`, `dot`, `norm`, ...) fuse with the expression they reduce.
* **`parallel for`** runs a loop's iterations on all cores; `+=`, `-=`, `*=`, `x = min(x, ..)` and
  `x = max(x, ..)` on outer variables become deterministic parallel reductions.
"""
import math
from fractions import Fraction

import numpy as np

from .. import ast as A
from ..errors import UnitError, TypeCheckError
from ..units import Quantity, DIMLESS, DIM_NAMES, dim_name
from .engine import NCHUNKS

try:
    import llvmlite.ir as ir
except ImportError:                                    # pragma: no cover
    ir = None


class Unsupported(Exception):
    """The function uses something the native compiler does not handle; it runs on the bytecode engine."""

    def __init__(self, msg, node=None):
        super().__init__(msg)
        self.msg = msg
        self.line = getattr(node, "line", None)
        self.col = getattr(node, "col", None)

    def __str__(self):
        return f"{self.msg} (line {self.line})" if self.line else self.msg


# ============================================================================= types
from .types import (I64, F64, F32, BOOL, NONE, ANY, SCALARS, ELEMS, ELEM_DTYPE, ELEM_SCALAR, SCALAR_ELEM, ELEM_SIZE, np_dtype,  # noqa: F401
                    ARR, TUP, is_arr, is_lazy, is_arrish, tname, type_of_value,
                    E_INDEX, E_OVERFLOW, E_ZERODIV, E_ASSERT, E_NEGEXP, E_SHAPE, E_NOMEM, E_NEGSIZE, E_NORETURN,
                    E_EMPTY, E_CONVERT, E_COND)


class TV:
    """A value's type, physical dimension, compile-time constant (if known) and provenance.

    npv: is this value a NumPy scalar in the bytecode engine (it came from a tensor element or a reduction)? Then
    NumPy's rules apply - x/0.0 -> inf, Int // 0 -> 0, Int overflow wraps - instead of Python's errors.
    True / False when known at compile time; when it depends on the path taken, an i1 run-time flag (DYN while
    inferring types), so native code follows exactly the rule the bytecode engine would apply on that path."""
    __slots__ = ("t", "d", "const", "weak", "npv")

    def __init__(self, t, d=DIMLESS, const=None, weak=False, npv=False):
        self.t, self.d, self.const, self.weak, self.npv = t, d, const, weak, npv


DYN = "dyn"                    # provenance decided at run time (inference stage)


class Val(TV):
    __slots__ = ("v",)

    def __init__(self, t, v, d=DIMLESS, const=None, weak=False, npv=False):
        super().__init__(t, d, const, weak, npv)
        self.v = v


# ============================================================================= helpers
def _dim_of(v):
    return v.dim if isinstance(v, Quantity) else DIMLESS


def unify(a, b, node=None):
    """The type a variable must have to hold both a and b."""
    if a is None or a == ANY:
        return b
    if b is None or b == ANY or a == b:
        return a
    num = {I64: 0, BOOL: 0, F32: 1, F64: 2}
    if a in num and b in num:
        if a == BOOL or b == BOOL:
            other = b if a == BOOL else a
            return I64 if other == I64 else other
        return a if num[a] >= num[b] else b
    if is_arr(a) and is_arr(b) and a[1] == b[1] and a[2] == b[2]:
        return ARR(a[1], a[2], a[3] and b[3])
    if a[0] == "t" and b[0] == "t" and len(a[1]) == len(b[1]):
        return TUP(unify(x, y, node) for x, y in zip(a[1], b[1]))
    raise Unsupported(f"a variable holds both {tname(a)} and {tname(b)}", node)


def arith(a, b, wa=False, wb=False):
    """Result type of +,-,* on two scalars (NumPy 2 'weak' literal rules for f32)."""
    if a == ANY:
        return b if b != BOOL else I64
    if b == ANY:
        return a if a != BOOL else I64
    a = I64 if a == BOOL else a
    b = I64 if b == BOOL else b
    if a == b:
        return a
    if F32 in (a, b):
        other = b if a == F32 else a
        if other == I64 or (other == F64 and (wb if a == F32 else wa)):
            return F32
        return F64
    return F64


class Spec:
    """One specialisation of an Apsi function for given argument types."""

    def __init__(self, mc, fndef, args, resolver, name):
        self.mc, self.fndef, self.args, self.resolver, self.name = mc, fndef, args, resolver, name
        self.slots = {}           # slot key -> [type, dim]
        self.ret = None           # TV
        self.ret_seen = False
        self.state = "new"
        self.fn = None
        self.may_fail = False
        self.notes = []
        self.globals = {}         # array globals referenced: name -> slot index
        self.callees = set()      # specs this one calls (for the may-fail fixed point)
        self.ret_np = None        # (must, may): are returned scalars NumPy scalars on every / some path
        self.fast = resolver.options.get("fast", False)


# ============================================================================= module context
class ModuleCtx:
    def __init__(self, options):
        self.options = options
        self.module = ir.Module(name="apsi") if ir else None
        self.specs = {}
        self.order = []
        self.gslots = []          # global arrays: (resolver, name, type)
        self.gindex = {}
        self.print_sites = []     # (parts, interp)
        self.assert_msgs = []
        self.fn_names = []
        self.uid = 0
        self._decl = {}
        self.uses_parallel = False
        self.gpu = bool(options.get("gpu"))      # generating a CUDA kernel (NVPTX)
        self.gpu_reds = []                       # (slot key, type) reduced across GPU threads

    def fresh(self, tag):
        self.uid += 1
        return f"{tag}.{self.uid}"

    # ------------------------------------------------------------- specs
    def spec(self, fndef, args, resolver, node=None):
        key = (id(fndef), tuple((a.t, a.d, a.npv if isinstance(a.npv, bool) else DYN) for a in args), id(resolver))
        s = self.specs.get(key)
        if s is not None:
            if s.state == "inferring":
                s.recursive = True
            return s
        name = f"apsi.{_safe(fndef.name)}.{len(self.specs)}"
        s = Spec(self, fndef, args, resolver, name)
        s.recursive = False
        if fndef.shape_params:
            pass
        self.specs[key] = s
        self.fn_names.append(fndef.name)
        s.index = len(self.fn_names) - 1
        s.state = "inferring"
        for _ in range(12):
            before = (repr(sorted((k, tuple(v[0]), v[2], v[3]) for k, v in s.slots.items())), s.ret and s.ret.t,
                      s.ret_np)
            FnEmitter(self, s, dry=True).run()
            after = (repr(sorted((k, tuple(v[0]), v[2], v[3]) for k, v in s.slots.items())), s.ret and s.ret.t,
                     s.ret_np)
            if before == after and not (s.recursive and s.ret is None):
                break
        else:
            raise Unsupported(f"could not settle the types in {fndef.name}()", fndef)
        if s.ret is None:
            s.ret = TV(NONE)
        must, may = s.ret_np or (False, False)
        s.ret.npv = True if must else (DYN if may and s.ret.t in SCALARS else False)
        s.state = "inferred"
        self.order.append(s)
        return s

    def emit_all(self):
        # a function can fail if it has a check itself or calls one that can (fixed point over the call graph)
        changed = True
        while changed:
            changed = False
            for s in self.order:
                if not s.may_fail and any(c.may_fail for c in s.callees):
                    s.may_fail = changed = True
        for s in self.order:
            self.declare_fn(s)
        for s in self.order:
            FnEmitter(self, s, dry=False).run()
        self.seal()

    def seal(self):
        """Blocks nothing can reach (after return/break) still need a terminator."""
        for f in self.module.functions:
            for bb in f.blocks:
                if not bb.is_terminated:
                    ir.IRBuilder(bb).unreachable()

    def declare_fn(self, s):
        if s.fn is not None:
            return s.fn
        params = []
        for a in s.args:
            params.append(lltype(a.t))
        params = [lltype(a.t) for a in s.args] + [i1t() for a in s.args if a.npv is DYN]
        fnty = ir.FunctionType(ret_ll(s), params)
        s.fn = ir.Function(self.module, fnty, name=s.name)
        s.fn.linkage = "internal"
        s.fn.attributes.add("nounwind")
        return s.fn

    # ------------------------------------------------------------- globals
    def global_array(self, resolver, name, t):
        key = (id(resolver), name)
        idx = self.gindex.get(key)
        if idx is None:
            idx = len(self.gslots)
            self.gindex[key] = idx
            self.gslots.append((resolver, name, t))
        return idx

    def decl(self, name, ret, args, varargs=False):
        f = self._decl.get(name)
        if f is None:
            f = self.module.globals.get(name)
            if f is None:
                f = ir.Function(self.module, ir.FunctionType(ret, args, var_arg=varargs), name=name)
            self._decl[name] = f
        return f

    def err_global(self):
        g = self.module.globals.get("apsi_err")
        if g is None:
            if self.gpu:                         # lives in GPU global memory; the host reads it back
                g = ir.GlobalVariable(self.module, ir.ArrayType(i64t(), 8), "apsi_err", addrspace=1)
                g.initializer = ir.Constant(ir.ArrayType(i64t(), 8), None)
            else:
                g = ir.GlobalVariable(self.module, ir.ArrayType(i64t(), 8), "apsi_err")
                g.linkage = "external"
        return g

    def gpu_global(self, name, n):
        g = self.module.globals.get(name)
        if g is None:
            g = ir.GlobalVariable(self.module, ir.ArrayType(i64t(), n), name, addrspace=1)
            g.initializer = ir.Constant(ir.ArrayType(i64t(), n), None)
        return g

    def gtable(self):
        """Table of pointers to global tensor descriptors (filled in by the Python side)."""
        g = self.module.globals.get("apsi_gtab_HASH")
        if g is None:
            t = ir.ArrayType(i8p(), 64)
            g = ir.GlobalVariable(self.module, t, "apsi_gtab_HASH")
            g.initializer = ir.Constant(t, None)
        if len(self.gslots) > 64:
            raise Unsupported("more than 64 global tensors in one native function")
        return g

    def site_base(self):
        """Index of this module's first print site (set by the Python side after loading)."""
        g = self.module.globals.get("apsi_sites_HASH")
        if g is None:
            g = ir.GlobalVariable(self.module, i64t(), "apsi_sites_HASH")
            g.initializer = ir.Constant(i64t(), 0)
        return g


def ret_ll(s):
    if s.ret.t == NONE:
        return ir.VoidType()
    if s.ret.npv is DYN:
        return ir.LiteralStructType([lltype(s.ret.t), i1t()])
    return lltype(s.ret.t)


def _safe(name):
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ----------------------------------------------------------------------------- LLVM types
def i64t():
    return ir.IntType(64)


def i1t():
    return ir.IntType(1)


def i8p():
    return ir.IntType(8).as_pointer()


def elem_ll(e):
    return {"f8": ir.DoubleType(), "f4": ir.FloatType(), "f2": ir.IntType(16), "h2": ir.IntType(16),
            "e4": ir.IntType(8), "e5": ir.IntType(8),
            "i8": ir.IntType(64), "i4": ir.IntType(32), "i2": ir.IntType(16), "i1": ir.IntType(8),
            "u1": ir.IntType(8), "b1": ir.IntType(8)}[e]


# 16-bit floats are kept as raw 16-bit integers and converted with integer operations, so the same code works on
# every CPU (no F16C or libcalls needed) and in GPU kernels; conversions to 16 bits round to nearest, ties to even.
def _u32(v):
    return ir.Constant(ir.IntType(32), v)


def half_to_f32(bd, raw):
    i32, f32 = ir.IntType(32), ir.FloatType()
    h = bd.zext(raw, i32)
    sign = bd.shl(bd.and_(h, _u32(0x8000)), _u32(16))
    ex = bd.and_(bd.lshr(h, _u32(10)), _u32(0x1F))
    man = bd.and_(h, _u32(0x3FF))
    normal = bd.or_(sign, bd.or_(bd.shl(bd.add(ex, _u32(112)), _u32(23)), bd.shl(man, _u32(13))))
    special = bd.or_(sign, bd.or_(_u32(0x7F800000), bd.shl(man, _u32(13))))
    bits = bd.select(bd.icmp_unsigned("==", ex, _u32(0x1F)), special, normal)
    as_f = bd.bitcast(bits, f32)
    sub = bd.fmul(bd.uitofp(man, f32), ir.Constant(f32, 2.0 ** -24))          # zero and subnormals (exact)
    sub = bd.select(bd.icmp_unsigned("!=", sign, _u32(0)), bd.fneg(sub), sub)
    return bd.select(bd.icmp_unsigned("==", ex, _u32(0)), sub, as_f)


def f32_to_half(bd, f):
    i32, i16, f32 = ir.IntType(32), ir.IntType(16), ir.FloatType()
    u = bd.bitcast(f, i32)
    sign = bd.and_(u, _u32(0x80000000))
    a = bd.xor(u, sign)
    big = bd.select(bd.icmp_unsigned(">", a, _u32(0x7F800000)), _u32(0x7E00), _u32(0x7C00))   # NaN / overflow
    magic = (127 - 15 + 23 - 10 + 1) << 23
    small = bd.sub(bd.bitcast(bd.fadd(bd.bitcast(a, f32), ir.Constant(f32, struct_float(magic))), i32),
                   _u32(magic))
    odd = bd.and_(bd.lshr(a, _u32(13)), _u32(1))
    norm = bd.lshr(bd.add(bd.add(a, _u32(((15 - 127) << 23) + 0xFFF)), odd), _u32(13))
    o = bd.select(bd.icmp_unsigned(">=", a, _u32((127 + 16) << 23)), big,
                  bd.select(bd.icmp_unsigned("<", a, _u32(113 << 23)), small, norm))
    return bd.trunc(bd.or_(o, bd.lshr(sign, _u32(16))), i16)


def bf16_to_f32(bd, raw):
    return bd.bitcast(bd.shl(bd.zext(raw, ir.IntType(32)), _u32(16)), ir.FloatType())


def f32_to_bf16(bd, f):
    i32 = ir.IntType(32)
    u = bd.bitcast(f, i32)
    rounded = bd.lshr(bd.add(u, bd.add(_u32(0x7FFF), bd.and_(bd.lshr(u, _u32(16)), _u32(1)))), _u32(16))
    nan = bd.or_(bd.lshr(u, _u32(16)), _u32(0x0040))                     # keep NaN a (quiet) NaN
    is_nan = bd.icmp_unsigned(">", bd.and_(u, _u32(0x7FFFFFFF)), _u32(0x7F800000))
    return bd.trunc(bd.select(is_nan, nan, rounded), ir.IntType(16))


# 8-bit floats (OCP FP8, as in ml_dtypes): (exponent bits, mantissa bits, bias, has infinity)
FP8 = {"e4": (4, 3, 7, False), "e5": (5, 2, 15, True)}       # float8_e4m3fn (max 448, no inf), float8_e5m2


def fp8_to_f32(bd, raw, kind):
    E, M, b, has_inf = FP8[kind]
    i32, f32 = ir.IntType(32), ir.FloatType()
    c = bd.zext(raw, i32)
    sign = bd.shl(bd.and_(c, _u32(0x80)), _u32(24))
    ex = bd.and_(bd.lshr(c, _u32(M)), _u32((1 << E) - 1))
    man = bd.and_(c, _u32((1 << M) - 1))
    normal = bd.or_(sign, bd.or_(bd.shl(bd.add(ex, _u32((127 - b) & 0xFFFFFFFF)), _u32(23)),
                                 bd.shl(man, _u32(23 - M))))
    top = bd.icmp_unsigned("==", ex, _u32((1 << E) - 1))
    if has_inf:
        special = bd.or_(sign, bd.select(bd.icmp_unsigned("==", man, _u32(0)), _u32(0x7F800000), _u32(0x7FC00000)))
        bits = bd.select(top, special, normal)
    else:
        is_nan = bd.and_(top, bd.icmp_unsigned("==", man, _u32((1 << M) - 1)))
        bits = bd.select(is_nan, bd.or_(sign, _u32(0x7FC00000)), normal)
    as_f = bd.bitcast(bits, f32)
    sub = bd.fmul(bd.uitofp(man, f32), ir.Constant(f32, 2.0 ** (1 - b - M)))
    sub = bd.select(bd.icmp_unsigned("!=", sign, _u32(0)), bd.fneg(sub), sub)
    return bd.select(bd.icmp_unsigned("==", ex, _u32(0)), sub, as_f)


def f32_to_fp8(bd, f, kind):
    """Round to nearest, ties to even. e5m2 overflows to infinity; e4m3fn (no infinity) to NaN, like ml_dtypes."""
    E, M, b, has_inf = FP8[kind]
    i32, i8, f32 = ir.IntType(32), ir.IntType(8), ir.FloatType()
    u = bd.bitcast(f, i32)
    sign = bd.and_(u, _u32(0x80000000))
    a = bd.xor(u, sign)
    magic = (127 + 1 - b - M + 23) << 23
    small = bd.sub(bd.bitcast(bd.fadd(bd.bitcast(a, f32), ir.Constant(f32, struct_float(magic))), i32), _u32(magic))
    sh = 23 - M
    rebased = bd.add(a, _u32(((b - 127) << 23) & 0xFFFFFFFF))
    norm = bd.lshr(bd.add(bd.add(rebased, _u32((1 << (sh - 1)) - 1)), bd.and_(bd.lshr(a, _u32(sh)), _u32(1))),
                   _u32(sh))
    code = bd.select(bd.icmp_unsigned("<", a, _u32((127 + 1 - b) << 23)), small, norm)
    if has_inf:
        code = bd.select(bd.icmp_unsigned(">=", code, _u32(0x7C)), _u32(0x7C), code)
        code = bd.select(bd.icmp_unsigned(">", a, _u32(0x7F800000)), _u32(0x7E), code)       # NaN stays NaN
    else:
        bad = bd.or_(bd.icmp_unsigned(">", code, _u32(0x7E)), bd.icmp_unsigned(">", a, _u32(0x7F800000)))
        code = bd.select(bad, _u32(0x7F), code)
    return bd.trunc(bd.or_(code, bd.lshr(sign, _u32(24))), i8)


def struct_float(bits):
    import struct
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def arr_ll(e, nd):
    return ir.LiteralStructType([elem_ll(e).as_pointer(), i8p(), ir.ArrayType(i64t(), nd),
                                 ir.ArrayType(i64t(), nd)])


def lltype(t):
    if t == I64:
        return i64t()
    if t == F64:
        return ir.DoubleType()
    if t == F32:
        return ir.FloatType()
    if t == BOOL:
        return i1t()
    if t[0] == "a":
        return arr_ll(t[1], t[2])
    if t[0] == "t":
        return ir.LiteralStructType([lltype(x) for x in t[1]])
    raise Unsupported(f"no machine type for {tname(t)}")


# ============================================================================= resolver of free names
NATIVE_BUILTINS = {"sqrt", "cbrt", "exp", "log", "log10", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
                   "atan2", "sinh", "cosh", "tanh", "abs", "floor", "ceil", "round", "min", "max", "sum", "mean",
                   "prod", "len", "float", "int", "bool", "pow", "zeros", "ones", "full", "zeros_like",
                   "ones_like", "range", "print", "dot", "norm", "clip", "where", "relu", "sigmoid", "shape",
                   "ndim", "argmin", "argmax", "enumerate", "pi", "e", "tau", "inf", "nan", "magnitude"}


class Resolver:
    """Looks up the names a function uses but does not define (globals, closures, builtins)."""

    def __init__(self, pyfn, freemap, options):
        self.pyfn = pyfn
        self.freemap = freemap or {}
        self.options = options
        code = getattr(pyfn, "__code__", None)
        self.cells = {}
        if code is not None and pyfn.__closure__:
            self.cells = dict(zip(code.co_freevars, pyfn.__closure__))
        self.g = getattr(pyfn, "__globals__", {})

    def lookup(self, name):
        """-> ('value', obj, mutable) | ('builtin', name) | None"""
        ent = self.freemap.get(name)
        if ent is None:
            if name in NATIVE_BUILTINS:
                return ("builtin", name)
            if name in self.g:
                return ("value", self.g[name], False)
            return None
        py, mutable = ent
        if py in self.cells:
            try:
                return ("value", self.cells[py].cell_contents, mutable)
            except ValueError:
                return None
        if py in self.g:
            return ("value", self.g[py], mutable)
        return None

    def attr(self, obj, name):
        get = self.g.get("_Aattr")
        if get is None:
            return getattr(obj, name)
        return get(obj, name)

    def builtin_obj(self, name):
        return self.g.get(name)


def apsi_function(obj):
    """(fndef, resolver-pyfn, freemap, options) for an Apsi function value usable natively, else None."""
    from . import NativeFunction
    opts = {}
    if isinstance(obj, NativeFunction):
        opts = obj.options
        obj = obj.fn
    fd = getattr(obj, "_apsi_def", None)
    if fd is None:
        return None
    pyfn = getattr(obj, "_apsi_e", None) or obj
    return fd, pyfn, getattr(obj, "_apsi_free", None), opts


# ============================================================================= lazy tensor expressions (fusion)
class Lazy:
    """A tensor expression not yet written to memory: leaves combined element by element."""

    def __init__(self, kind, elem, ndim, **kw):
        self.kind, self.elem, self.ndim = kind, elem, ndim
        self.__dict__.update(kw)

    def leaves(self):
        if self.kind == "leaf":
            return [self]
        out = []
        for c in getattr(self, "kids", ()):
            if isinstance(c, Lazy):
                out += c.leaves()
        return out


# ============================================================================= function emitter
class Frame:
    def __init__(self, fn, builder):
        self.fn = fn
        self.b = builder
        self.ptrs = {}            # slot key -> alloca
        self.loops = []           # (continue_bb, break_bb, iter_arena)
        self.arena = None
        self.err_bb = None
        self.entry_b = None
        self.reductions = {}      # slot key -> op (inside a parallel chunk)
        self.in_parallel = False


def new_frame(fn):
    """Frame for `fn`: an 'entry' block that only holds allocas, then the code from 'body'."""
    entry = fn.append_basic_block("entry")
    body = fn.append_basic_block("body")
    eb = ir.IRBuilder(entry)
    term = eb.branch(body)
    eb.position_before(term)
    f = Frame(fn, ir.IRBuilder(body))
    f.entry_b = eb
    f.arena = eb.alloca(i8p(), name="arena")
    eb.store(ir.Constant(i8p(), None), f.arena)
    return f


class FnEmitter:
    def __init__(self, mc, spec, dry):
        self.mc, self.spec, self.dry = mc, spec, dry
        self.r = spec.resolver
        self.fast = spec.fast
        self.erased = bool(spec.resolver.options.get("erased"))   # called from unit-erased code: no units
        self.scopes = []
        self.proven = set()
        self.f = None

    # ------------------------------------------------------------------ driver
    def run(self):
        s = self.spec
        fd = s.fndef
        if not self.dry:
            fn = s.fn
            fn.attributes.add("nounwind")
            self.f = new_frame(fn)
        self.scopes = [{}]
        nflag = len(s.args)
        for i, (p, a) in enumerate(zip(fd.params, s.args)):
            key = ("param", i)
            self._set_slot(key, a.t, a.d, p, a.npv)
            self.scopes[-1][p.name] = key
            if not self.dry:
                ptr = self._alloca(key)
                self.f.b.store(s.fn.args[i], ptr)
                if a.npv is DYN:
                    self.store_np(key, s.fn.args[nflag])
                    nflag += 1
                else:
                    self.store_np(key, a.npv)
        if fd.is_expr:
            v = self.expr(fd.body)
            self.do_return(v, fd.body)
        else:
            self.block(fd.body, new_scope=False)
            if not self.dry and not self.f.b.block.is_terminated:
                if s.ret.t == NONE:
                    self.free_arena(self.f.arena)
                    self.f.b.ret_void()
                else:
                    self.fail(E_NORETURN, fd)
                    self.f.b.unreachable()
            elif self.dry and s.ret is not None and s.ret.t != NONE and not _always_returns(fd.body):
                raise Unsupported(f"{fd.name}() can end without returning a value", fd)

    # ------------------------------------------------------------------ provenance flags
    def np_i1(self, x):
        return ir.Constant(i1t(), int(x)) if isinstance(x, bool) else x

    def np_or(self, a, b):
        """Provenance of a result computed from a and b: NumPy if either operand is NumPy."""
        if a is True or b is True:
            return True
        if a is False:
            return b
        if b is False:
            return a
        return DYN if self.dry else self.f.b.or_(a, b)

    def np_pick(self, cond, a, b):
        """Provenance of `b if cond else a` (min/max/select return one of the objects)."""
        if isinstance(a, bool) and a is b:
            return a
        if self.dry:
            return DYN
        return self.f.b.select(cond, self.np_i1(b), self.np_i1(a))

    def py_check(self, npv, ok, code, node, a=None, b=None):
        """A check that only Python numbers fail (NumPy scalars follow IEEE / wrap-around instead)."""
        if npv is True:
            return
        if self.dry or ok is None:
            self.spec.may_fail = True
            return
        if npv is not False:
            ok = self.f.b.or_(npv, ok)
        self.check(ok, code, node, a, b)

    def slot_dyn(self, key):
        t, d, must, may = self.spec.slots[key]
        return may and not must

    def _np_alloca(self, key):
        p = self.f.__dict__.setdefault("nps", {}).get(key)
        if p is None:
            p = self.f.entry_b.alloca(i1t(), name="npflag")
            self.f.nps[key] = p
        return p

    def store_np(self, key, npv):
        if not self.dry and self.slot_dyn(key):
            self.f.b.store(self.np_i1(npv if npv is not DYN else False), self._np_alloca(key))

    # ------------------------------------------------------------------ slots
    def _set_slot(self, key, t, d, node, npv=False):
        cur = self.spec.slots.get(key)
        if is_lazy(t):
            t = ARR(t[1], t[2], True)
        if cur is None:
            self.spec.slots[key] = [t, d, npv is True and t in SCALARS, npv is not False and t in SCALARS]
            return t
        if cur[1] != d:
            raise UnitError(f"a variable holds both {dim_name(cur[1])} and {dim_name(d)}",
                            getattr(node, "line", None), getattr(node, "col", None))
        nt = unify(cur[0], t, node)
        cur[0] = nt
        cur[2] = bool(cur[2] and npv is True and t in SCALARS)   # must: NumPy on every assignment
        cur[3] = bool(cur[3] or (npv is not False and t in SCALARS))   # may: NumPy on some assignment
        return nt

    def _alloca(self, key):
        ptr = self.f.ptrs.get(key)
        if ptr is None:
            t = self.spec.slots[key][0]
            ptr = self.f.entry_b.alloca(lltype(t), name=f"v{len(self.f.ptrs)}")
            self.f.ptrs[key] = ptr
        return ptr

    def declare(self, name, key, val, node):
        t = self._set_slot(key, val.t, val.d, node, val.npv)
        self.scopes[-1][name] = key
        if not self.dry:
            v = self.materialize(val, node)
            self.f.b.store(self.convert(v, t, node).v, self._alloca(key))
            self.store_np(key, val.npv)

    def find(self, name):
        for sc in reversed(self.scopes):
            if name in sc:
                return sc[name]
        return None

    # ------------------------------------------------------------------ errors
    def global_ptr(self, g):
        """A generic pointer to module global `g` (GPU globals live in address space 1)."""
        if not self.mc.gpu:
            return g
        cache = self.f.__dict__.setdefault("gptrs", {})
        p = cache.get(g.name)
        if p is None:
            p = cache[g.name] = self.f.entry_b.addrspacecast(g, g.value_type.as_pointer())
        return p

    def fail(self, code, node, a=None, b=None, msg_index=None):
        """Record an error (code, line, col, a, b) and leave the function."""
        self.spec.may_fail = True
        if self.dry:
            return
        bd = self.f.b
        g = self.global_ptr(self.mc.err_global())
        vals = [ir.Constant(i64t(), code), ir.Constant(i64t(), getattr(node, "line", 0) or 0),
                ir.Constant(i64t(), getattr(node, "col", 0) or 0),
                a if a is not None else ir.Constant(i64t(), 0), b if b is not None else ir.Constant(i64t(), 0),
                ir.Constant(i64t(), self.spec.index),
                ir.Constant(i64t(), -1 if msg_index is None else msg_index)]
        for k in range(len(vals) - 1, -1, -1):        # code last, so a reader sees complete data
            p = bd.gep(g, [ir.Constant(i64t(), 0), ir.Constant(i64t(), k)])
            if k == 0:
                bd.store_atomic(vals[k], p, "release", 8)
            else:
                bd.store(vals[k], p)
        bd.branch(self.err_block())

    def err_block(self):
        if self.f.err_bb is None:
            cur = self.f.b.block
            bb = self.f.fn.append_basic_block("error")
            self.f.err_bb = bb
            self.f.b.position_at_end(bb)
            if self.f.arena is not None:
                self.free_arena(self.f.arena)
            for (_, _, it) in self.f.loops:
                if it is not None:
                    self.free_arena(it)
            rt = self.f.fn.function_type.return_type
            if isinstance(rt, ir.VoidType):
                self.f.b.ret_void()
            else:
                self.f.b.ret(ir.Constant(rt, ir.Undefined))
            self.f.b.position_at_end(cur)
        return self.f.err_bb

    def check(self, ok, code, node, a=None, b=None, msg_index=None):
        """Continue if `ok` (i1) is true, otherwise fail."""
        self.spec.may_fail = True
        if self.dry:
            return
        bd = self.f.b
        good = self.f.fn.append_basic_block("ok")
        bad = self.f.fn.append_basic_block("fail")
        bd.cbranch(ok, good, bad)
        bd.position_at_end(bad)
        self.fail(code, node, a, b, msg_index)
        bd.position_at_end(good)

    def check_err_flag(self):
        if self.dry:
            return
        bd = self.f.b
        g = self.global_ptr(self.mc.err_global())
        p = bd.gep(g, [ir.Constant(i64t(), 0), ir.Constant(i64t(), 0)])
        e = bd.load(p)            # a plain load: it cannot move across the call that may have set it
        good = self.f.fn.append_basic_block("ok")
        bd.cbranch(bd.icmp_signed("!=", e, ir.Constant(i64t(), 0)), self.err_block(), good)
        bd.position_at_end(good)

    # ------------------------------------------------------------------ memory
    def alloc_array(self, elem, shape, node, zero=True):
        """A new contiguous array (registered in the frame's arena); shape: list of i64 values."""
        nd = len(shape)
        t = ARR(elem, nd, True)
        if self.mc.gpu:
            raise Unsupported("creating tensors inside a GPU kernel (pass the output tensor in)", node)
        if self.dry:
            return Val(t, None)
        bd = self.f.b
        total = ir.Constant(i64t(), 1)
        for s_ in shape:
            if not self.fast:
                self.check(bd.icmp_signed(">=", s_, ir.Constant(i64t(), 0)), E_NEGSIZE, node, s_)
            total = bd.mul(total, s_)
        esz = ELEM_SIZE[elem]
        nbytes = bd.add(bd.mul(total, ir.Constant(i64t(), esz)), ir.Constant(i64t(), 64))
        calloc = self.mc.decl("calloc", i8p(), [i64t(), i64t()])
        blk = bd.call(calloc, [ir.Constant(i64t(), 1), nbytes])
        self.check(bd.icmp_unsigned("!=", blk, ir.Constant(i8p(), None)), E_NOMEM, node, nbytes)
        arena = self.cur_arena()
        head = bd.load(arena)
        bd.store(head, bd.bitcast(blk, i8p().as_pointer()))
        bd.store(blk, arena)
        data = bd.bitcast(bd.gep(blk, [ir.Constant(i64t(), 64)]), elem_ll(elem).as_pointer())
        st = ir.Constant(arr_ll(elem, nd), ir.Undefined)
        st = bd.insert_value(st, data, 0)
        st = bd.insert_value(st, blk, 1)
        stride = ir.Constant(i64t(), 1)
        strides = [None] * nd
        for k in range(nd - 1, -1, -1):
            strides[k] = stride
            stride = bd.mul(stride, shape[k])
        for k in range(nd):
            st = bd.insert_value(st, shape[k], [2, k])
            st = bd.insert_value(st, strides[k], [3, k])
        return Val(t, st)

    def iter_arena(self, loop):
        if self.mc.gpu:
            return None
        return self._iter_arena(loop)

    def _iter_arena(self, loop):
        """A per-iteration arena for `loop` when no tensor made in an iteration can outlive it.

        A tensor escapes an iteration only by being assigned to a variable declared outside the loop body
        (returns keep what they return). Then the loop's temporaries go to the enclosing arena instead."""
        if self.dry or not _allocates(loop.body):
            return None
        inner = _declared_names(loop.body) | set(getattr(loop, "targets", ()) or ())
        for name in _assigned_names(loop.body) - inner:
            key = self.find(name)
            if key is not None and _holds_array(self.spec.slots[key][0]):
                return None
        a = self.f.entry_b.alloca(i8p(), name="iter.arena")
        self.f.entry_b.store(ir.Constant(i8p(), None), a)
        return a

    def cur_arena(self):
        for (_, _, it) in reversed(self.f.loops):
            if it is not None:
                return it
        return self.f.arena

    def free_arena(self, arena, keep=()):
        """Free every block in `arena` except those in `keep` (i8* values)."""
        if self.dry or arena is None or self.mc.gpu:
            return
        bd = self.f.b
        free = self.mc.decl("free", ir.VoidType(), [i8p()])
        head_bb = self.f.fn.append_basic_block("free.loop")
        body_bb = self.f.fn.append_basic_block("free.body")
        out_bb = self.f.fn.append_basic_block("free.done")
        start = bd.load(arena)
        pre = bd.block
        bd.branch(head_bb)
        bd.position_at_end(head_bb)
        cur = bd.phi(i8p())
        cur.add_incoming(start, pre)
        bd.cbranch(bd.icmp_unsigned("==", cur, ir.Constant(i8p(), None)), out_bb, body_bb)
        bd.position_at_end(body_bb)
        nxt = bd.load(bd.bitcast(cur, i8p().as_pointer()))
        skip = ir.Constant(i1t(), 0)
        for k in keep:
            skip = bd.or_(skip, bd.icmp_unsigned("==", cur, k))
        do_free = self.f.fn.append_basic_block("free.one")
        after = self.f.fn.append_basic_block("free.next")
        bd.cbranch(skip, after, do_free)
        bd.position_at_end(do_free)
        bd.call(free, [cur])
        bd.branch(after)
        bd.position_at_end(after)
        cur.add_incoming(nxt, after)
        bd.branch(head_bb)
        bd.position_at_end(out_bb)
        bd.store(ir.Constant(i8p(), None), arena)

    def adopt(self, base):
        """Take ownership of a block returned by a callee (push it on this frame's arena)."""
        if self.dry:
            return
        bd = self.f.b
        has = self.f.fn.append_basic_block("adopt")
        done = self.f.fn.append_basic_block("adopted")
        bd.cbranch(bd.icmp_unsigned("!=", base, ir.Constant(i8p(), None)), has, done)
        bd.position_at_end(has)
        arena = self.cur_arena()
        bd.store(bd.load(arena), bd.bitcast(base, i8p().as_pointer()))
        bd.store(base, arena)
        bd.branch(done)
        bd.position_at_end(done)

    # ------------------------------------------------------------------ statements
    def block(self, stmts, new_scope=True):
        if new_scope:
            self.scopes.append({})
        try:
            for st in stmts:
                if not self.dry and self.f.b.block.is_terminated:
                    break                                      # unreachable code after return/break
                self.stmt(st)
        finally:
            if new_scope:
                self.scopes.pop()

    def stmt(self, s):
        m = getattr(self, "s_" + type(s).__name__, None)
        if m is None:
            raise Unsupported(f"'{type(s).__name__.lower()}' statements", s)
        m(s)

    def s_Pass(self, s):
        pass

    def s_ExprStmt(self, s):
        e = s.expr
        if isinstance(e, A.Call) and isinstance(e.func, A.Name) and e.func.id == "print" \
                and self.find("print") is None and self.r.lookup("print") == ("builtin", "print"):
            self.do_print(e)
            return
        self.expr(e)

    def s_Let(self, s):
        v = self.expr(s.value)
        if s.type is not None:
            v = self.annotate(v, s.type, s)
        if len(s.targets) == 1:
            self.declare(s.targets[0], ("let", id(s), 0), v, s)
            return
        items = self.unpack(v, len(s.targets), s)
        for k, (n, it) in enumerate(zip(s.targets, items)):
            self.declare(n, ("let", id(s), k), it, s)

    def unpack(self, v, n, node):
        if v.t[0] != "t" or len(v.t[1]) != n:
            raise Unsupported(f"unpacking {tname(v.t)} into {n} names", node)
        out = []
        for k, t in enumerate(v.t[1]):
            d = v.d[k] if isinstance(v.d, tuple) else DIMLESS
            out.append(Val(t, None if self.dry else self.f.b.extract_value(v.v, k), d))
        return out

    def annotate(self, v, t, node):
        if isinstance(t, A.TName) and not (self.erased and t.name in DIM_NAMES):
            n = t.name
            if n == "Int":
                if v.t not in (I64, BOOL, ANY):
                    raise TypeCheckError(f"expected Int, got {tname(v.t)}", node.line, node.col)
                return v
            if n in ("Float", "Number", "Real", "Any"):
                return v
            if n in ("f32", "f16", "bf16", "f64"):              # f16/bf16 scalars compute in f32
                want = F64 if n == "f64" else F32
                if v.t not in (I64, BOOL, F64, F32, ANY):
                    raise TypeCheckError(f"expected a Float, got {tname(v.t)}", node.line, node.col)
                if self.dry:
                    return Val(want, None, v.d, const=v.const)
                return self.convert(v, want, node)
            if n in DIM_NAMES and not t.args:
                if v.d != DIM_NAMES[n] and v.t != ANY:
                    raise UnitError(f"expected {n}, got {dim_name(v.d)}", node.line, node.col)
                return v
        return v

    def assign_name(self, name, v, node):
        key = self.find(name)
        if key is None:
            res = self.r.lookup(name)
            if res is not None:
                raise Unsupported(f"assigning to the outer variable '{name}' (pass it in and return it)", node)
            raise Unsupported(f"'{name}' is not defined", node)
        if self.f is not None and key in self.f.reductions and not self.dry:
            pass
        t = self._set_slot(key, v.t, v.d, node, v.npv)
        if not self.dry:
            npv = v.npv
            v = self.materialize(v, node)
            self.f.b.store(self.convert(v, t, node).v, self._alloca(key))
            self.store_np(key, npv)

    def load_slot(self, key, node):
        t, d, must, may = self.spec.slots[key]
        npv = True if must else (False if not may else DYN)
        if self.dry:
            return Val(t, None, d, npv=npv)
        if npv is DYN:
            npv = self.f.b.load(self._np_alloca(key))
        return Val(t, self.f.b.load(self._alloca(key)), d, npv=npv)

    def s_Assign(self, s):
        tgt = s.target
        if isinstance(tgt, A.Name):
            self.assign_name(tgt.id, self.expr(s.value), s)
        elif isinstance(tgt, A.Index):
            self.store_index(tgt, self.expr(s.value), s)
        elif isinstance(tgt, A.TupleLit) and all(isinstance(x, A.Name) for x in tgt.items):
            v = self.expr(s.value)
            items = self.unpack(v, len(tgt.items), s)
            for x, it in zip(tgt.items, items):
                self.assign_name(x.id, it, s)
        else:
            raise Unsupported("this kind of assignment target", s)

    def s_AugAssign(self, s):
        tgt = s.target
        if isinstance(tgt, A.Name):
            cur = self.expr(tgt)
            self.assign_name(tgt.id, self.binop(s.op, cur, self.expr(s.value), s), s)
        elif isinstance(tgt, A.Index):
            arr, idxs = self.index_parts(tgt, s)
            cur = self.index_load(arr, idxs, tgt)
            self.store_to(arr, idxs, self.binop(s.op, cur, self.expr(s.value), s), tgt)
        else:
            raise Unsupported("this kind of assignment target", s)

    def s_Return(self, s):
        if self.f is not None and self.f.in_parallel:
            raise Unsupported("return inside parallel for", s)
        v = self.expr(s.value) if s.value is not None else Val(NONE, None)
        self.do_return(v, s)

    def do_return(self, v, node):
        sp = self.spec
        if is_lazy(v.t):
            v = self.materialize(v, node)
        if self.dry:
            if v.t == NONE:
                if sp.ret is not None and sp.ret.t != NONE:
                    raise Unsupported("a function that returns a value on some paths and nothing on others", node)
                sp.ret = TV(NONE)
                return
            if v.t in SCALARS or v.t == ANY:
                must, may = sp.ret_np if sp.ret_np is not None else (True, False)
                if v.t != ANY:
                    sp.ret_np = (must and v.npv is True, may or v.npv is not False)
            else:
                sp.ret_np = (False, False)
            if sp.ret is None or sp.ret.t == NONE and not sp.ret_seen:
                sp.ret = TV(v.t, v.d)
            else:
                if sp.ret.t == NONE:
                    raise Unsupported("a function that returns a value on some paths and nothing on others", node)
                if v.t != ANY and sp.ret.d != v.d:
                    raise UnitError(f"{sp.fndef.name}() returns both {_dims(sp.ret.d)} and {_dims(v.d)}",
                                    node.line, node.col)
                sp.ret = TV(unify(sp.ret.t, v.t, node), sp.ret.d if v.t == ANY else v.d)
            sp.ret_seen = True
            fd = sp.fndef
            if not self.erased and fd.ret is not None and isinstance(fd.ret, A.TName) and fd.ret.name in DIM_NAMES \
                    and v.t != ANY and v.d != DIM_NAMES[fd.ret.name]:
                raise UnitError(f"return value of {fd.name}() should be {fd.ret.name}, got {dim_name(v.d)}",
                                node.line, node.col)
            return
        bd = self.f.b
        keep = []
        if v.t == NONE:
            self.free_arena(self.f.arena)
            bd.ret_void()
            return
        npv = v.npv
        v = self.convert(v, sp.ret.t, node)
        for part_t, part in _array_parts(v.t, v.v, bd):
            keep.append(bd.extract_value(part, 1))
        for (_, _, it) in self.f.loops:
            if it is not None:
                self.free_arena(it, keep)
        self.free_arena(self.f.arena, keep)
        if sp.ret.npv is DYN:                       # the result's origin travels with it
            st = ir.Constant(ret_ll(sp), ir.Undefined)
            st = bd.insert_value(st, v.v, 0)
            st = bd.insert_value(st, self.np_i1(npv if npv is not DYN else False), 1)
            bd.ret(st)
            return
        bd.ret(v.v)

    def s_If(self, s):
        if self.dry:
            for c, body in s.branches:
                self.truth(self.expr(c), c)
                self.block(body)
            if s.else_body is not None:
                self.block(s.else_body)
            return
        bd = self.f.b
        end = self.f.fn.append_basic_block("endif")
        for c, body in s.branches:
            cond = self.truth(self.expr(c), c)
            then = self.f.fn.append_basic_block("then")
            nxt = self.f.fn.append_basic_block("else")
            bd.cbranch(cond, then, nxt)
            bd.position_at_end(then)
            self.block(body)
            if not bd.block.is_terminated:
                bd.branch(end)
            bd.position_at_end(nxt)
        if s.else_body is not None:
            self.block(s.else_body)
        if not bd.block.is_terminated:
            bd.branch(end)
        bd.position_at_end(end)

    def s_While(self, s):
        if self.dry:
            self.truth(self.expr(s.cond), s.cond)
            self.loop_body(s.body, None, None)
            return
        bd = self.f.b
        head = self.f.fn.append_basic_block("while")
        body = self.f.fn.append_basic_block("while.body")
        end = self.f.fn.append_basic_block("while.end")
        bd.branch(head)
        bd.position_at_end(head)
        bd.cbranch(self.truth(self.expr(s.cond), s.cond), body, end)
        bd.position_at_end(body)
        it_arena = self.iter_arena(s)
        latch = self.f.fn.append_basic_block("while.next")
        self.loop_body(s.body, latch, end, it_arena)
        if not bd.block.is_terminated:
            bd.branch(latch)
        bd.position_at_end(latch)
        if it_arena is not None:
            self.free_arena(it_arena)
        bd.branch(head)
        bd.position_at_end(end)

    def loop_body(self, stmts, cont, brk, iter_arena=None):
        if self.f is not None:
            self.f.loops.append((cont, brk, iter_arena))
        try:
            self.block(stmts)
        finally:
            if self.f is not None:
                self.f.loops.pop()

    def s_Break(self, s):
        if self.f is not None and self.f.in_parallel and not self.f.loops:
            raise Unsupported("break inside parallel for", s)
        if self.dry:
            return
        if not self.f.loops:
            raise Unsupported("break outside a loop", s)
        if self.f.loops[-1][2] is not None:
            self.free_arena(self.f.loops[-1][2])
        self.f.b.branch(self.f.loops[-1][1])

    def s_Continue(self, s):
        if self.dry:
            return
        if not self.f.loops:
            raise Unsupported("continue outside a loop", s)
        self.f.b.branch(self.f.loops[-1][0])

    def s_Assert(self, s):
        c = self.truth(self.expr(s.cond), s.cond)
        idx = None
        if isinstance(s.msg, A.Str):
            idx = len(self.mc.assert_msgs)
            self.mc.assert_msgs.append(s.msg.value)
        elif s.msg is not None:
            raise Unsupported("assert messages that are not plain strings", s)
        self.check(c, E_ASSERT, s, msg_index=idx)

    # ------------------------------------------------------------------ for loops
    def range_of(self, it, node):
        """(start, stop, step-constant-or-None, step) for a counting loop, or None."""
        if isinstance(it, A.Range):
            a, b = self.expr(it.start), self.expr(it.end)
            if a.t not in (I64, BOOL) or b.t not in (I64, BOOL):
                raise Unsupported("ranges over non-integers in native code", node)
            stop = b
            if it.inclusive:
                stop = self.binop("+", b, Val(I64, ir.Constant(i64t(), 1) if not self.dry else None, const=1),
                                  node)
            return self.as_int(a), self.as_int(stop), 1, None
        if isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "range" \
                and self.find("range") is None and not it.kwargs and 1 <= len(it.args) <= 3:
            vals = [self.expr(x) for x in it.args]
            for v in vals:
                if v.t not in (I64, BOOL):
                    raise Unsupported("range() over non-integers in native code", node)
            vals = [self.as_int(v) for v in vals]
            if len(vals) == 1:
                return Val(I64, ir.Constant(i64t(), 0) if not self.dry else None, const=0), vals[0], 1, None
            if len(vals) == 2:
                return vals[0], vals[1], 1, None
            st = vals[2]
            if st.const is not None:
                if st.const == 0:
                    raise Unsupported("range() step must not be zero", node)
                return vals[0], vals[1], st.const, st
            return vals[0], vals[1], None, st
        return None

    def s_For(self, s):
        if not self.dry and not self.mc.gpu:
            from ..idioms import match_gemm
            g = match_gemm(s)
            done = self.gemm_call(g, s) if g is not None else None
            if done is not None:                     # a matrix product written as loops: BLAS, loops as fallback
                bd = self.f.b
                slow = self.f.fn.append_basic_block("gemm.loops")
                join = self.f.fn.append_basic_block("gemm.done")
                bd.cbranch(done, join, slow)
                bd.position_at_end(slow)
                self.for_loop(s)
                if not bd.block.is_terminated:
                    bd.branch(join)
                bd.position_at_end(join)
                return
        if not self.mc.gpu:
            from ..idioms import match_gemm
            from .jam import jam
            rep = jam(s) if match_gemm(s) is None else None
            if rep is not None:                      # four independent outer iterations at once (see jam.py)
                for st in rep:
                    self.stmt(st)
                return None
        return self.for_loop(s)

    def gemm_call(self, g, s):
        """Emit apsi_gemm(...) for a matched loop nest; an i1 that is true when BLAS did the work, or None when the
        operand types rule it out at compile time (then only the loops are compiled)."""
        try:
            arrs = [self.expr(A.Name(n, line=s.line, col=s.col)) for n in (g.a, g.b, g.c)]
            if not all(is_arr(v.t) and v.t[2] == 2 for v in arrs):
                return None
            elem = arrs[0].t[1]
            if elem not in ("f8", "f4") or any(v.t[1] != elem for v in arrs):
                return None
            dims = []
            for e in (g.M, g.N, g.K):
                v = self.expr(e)
                if v.t not in (I64, BOOL):
                    return None
                dims.append(self.as_int(v).v)
        except Exception:                                  # noqa: BLE001 - the loops alone decide
            return None
        bd = self.f.b
        descs = []
        for v in arrs:
            p = self._entry_alloca(lltype(v.t), "gemm")
            bd.store(v.v, p)
            descs.append(bd.bitcast(p, i8p()))
        code = (8 if elem == "f8" else 4) * 100 + g.mode
        fn = self.mc.decl("apsi_gemm", i64t(), [i64t(), i8p(), i8p(), i8p(), i64t(), i64t(), i64t()])
        r = bd.call(fn, [ir.Constant(i64t(), code)] + descs + dims)
        return bd.icmp_signed("==", r, ir.Constant(i64t(), 0))

    def for_loop(self, s):
        rng = self.range_of(s.iter, s)
        if rng is not None:
            if len(s.targets) != 1:
                raise Unsupported("unpacking a number", s)
            if s.parallel:
                why = self.parallel_problem(s)
                if why is None:
                    return self.parallel_for(s, rng)
                if self.mc.gpu:
                    raise Unsupported(f"this parallel for cannot run on the GPU: {why}", s)
                if self.dry and why not in self.spec.notes:
                    self.spec.notes.append(f"line {s.line}: parallel for runs sequentially: {why}")
            return self.range_loop(s, rng)
        it = self.expr(s.iter) if not _is_enumerate(s.iter, self) else None
        if it is None:                                            # enumerate(tensor)
            arr = self.materialize(self.expr(s.iter.args[0]), s)
            if not is_arr(arr.t) or len(s.targets) != 2:
                raise Unsupported("enumerate() over something that is not a tensor", s)
            return self.tensor_loop(s, arr, s.targets[1], s.targets[0])
        if is_arrish(it.t):
            if len(s.targets) != 1:
                raise Unsupported("unpacking tensor elements", s)
            return self.tensor_loop(s, self.materialize(it, s), s.targets[0], None)
        raise Unsupported(f"loops over {tname(it.t)}", s)

    def tensor_loop(self, s, arr, name, ivar):
        n = self.dim_size(arr, 0)
        zero = Val(I64, None if self.dry else ir.Constant(i64t(), 0), const=0)

        def body(i):
            if ivar is not None:
                self.declare(ivar, ("for", id(s), 1), i, s)
            el = self.index_load(arr, [i], s, proven=True)
            self.declare(name, ("for", id(s), 0), el, s)
            self.block(s.body, new_scope=False)
        self.counted_loop(s, zero, n, 1, None, body)

    def range_loop(self, s, rng):
        start, stop, step_c, step = rng
        name = s.targets[0]
        key = ("for", id(s), 0)
        if self.dry:
            self.scopes.append({})
            self.declare(name, key, Val(I64, None), s)
            self.loop_body(s.body, None, None)
            self.scopes.pop()
            return
        plan = self.bounds_plan(s, name, start, stop, step_c)

        def body(i):
            self.declare(name, key, i, s)
            self.block(s.body, new_scope=False)
        if plan is None:
            return self.counted_loop(s, start, stop, step_c, step, body)
        ok, proven = plan
        bd = self.f.b
        fast_bb = self.f.fn.append_basic_block("for.fast")
        slow_bb = self.f.fn.append_basic_block("for.checked")
        join = self.f.fn.append_basic_block("for.join")
        bd.cbranch(ok, fast_bb, slow_bb)
        bd.position_at_end(fast_bb)
        saved = self.proven
        self.proven = saved | proven
        self.counted_loop(s, start, stop, step_c, step, body)
        self.proven = saved
        if not bd.block.is_terminated:
            bd.branch(join)
        bd.position_at_end(slow_bb)
        self.counted_loop(s, start, stop, step_c, step, body)
        if not bd.block.is_terminated:
            bd.branch(join)
        bd.position_at_end(join)

    def counted_loop(self, s, start, stop, step_c, step, body):
        """for i = start; i < stop (or > for negative steps); i += step: body(i)."""
        self.scopes.append({})
        try:
            if self.dry:
                body(Val(I64, None))
                return
            bd = self.f.b
            fn = self.f.fn
            pre = bd.block
            head = fn.append_basic_block("for")
            bodyb = fn.append_basic_block("for.body")
            latch = fn.append_basic_block("for.next")
            end = fn.append_basic_block("for.end")
            if step_c is None:                     # runtime step: count iterations
                stv = step.v
                self.check(bd.icmp_signed("!=", stv, ir.Constant(i64t(), 0)), E_ZERODIV, s)
                pre = bd.block
            bd.branch(head)
            bd.position_at_end(head)
            i = bd.phi(i64t(), name="i")
            i.add_incoming(start.v, pre)
            if step_c is not None:
                cond = bd.icmp_signed("<" if step_c > 0 else ">", i, stop.v)
            else:
                pos = bd.icmp_signed(">", stv, ir.Constant(i64t(), 0))
                cond = bd.select(pos, bd.icmp_signed("<", i, stop.v), bd.icmp_signed(">", i, stop.v))
            bd.cbranch(cond, bodyb, end)
            bd.position_at_end(bodyb)
            it_arena = self.iter_arena(s)
            self.f.loops.append((latch, end, it_arena))
            try:
                body(Val(I64, i))
            finally:
                self.f.loops.pop()
            if not bd.block.is_terminated:
                bd.branch(latch)
            bd.position_at_end(latch)
            if it_arena is not None:
                self.free_arena(it_arena)             # temporaries of this iteration
            inc = ir.Constant(i64t(), step_c) if step_c is not None else stv
            nxt = bd.add(i, inc, name="i.next")
            # i never overflows: the loop exits before i passes stop
            i.add_incoming(nxt, bd.block)
            bd.branch(head)
            bd.position_at_end(end)
        finally:
            self.scopes.pop()

    # ---- bounds-check hoisting
    def bounds_plan(self, s, var, start, stop, step_c):
        """For an inner loop, prove every `a[i + c]` / `a[k]` access in range before the loop starts.

        Returns (i1 'all in range', set of proven (node id, dim)) or None. When the test fails at run time the
        checked copy of the loop runs instead, so behaviour (negative indices, errors) is unchanged."""
        if self.fast or step_c != 1 or _has_loop(s.body):
            return None
        assigned = _assigned_names(s.body)
        if var in assigned:
            return None
        accesses = []
        for node in _walk(s.body):
            if isinstance(node, A.Index) and isinstance(node.obj, A.Name):
                key = self.find(node.obj.id)
                if node.obj.id in assigned:
                    continue
                idxs = node.index.items if isinstance(node.index, A.TupleLit) else [node.index]
                accesses.append((node, key, node.obj.id, idxs))
        if not accesses:
            return None
        bd = self.f.b
        ok = ir.Constant(i1t(), 1)
        proven = set()
        lo, hi = start.v, stop.v
        nonempty = bd.icmp_signed("<", lo, hi)
        last = bd.sub(hi, ir.Constant(i64t(), 1))
        for node, key, name, idxs in accesses:
            try:
                arr = self.expr(node.obj)
            except (Unsupported, UnitError):
                continue
            if not is_arr(arr.t) or len(idxs) > arr.t[2]:
                continue
            for k, ix in enumerate(idxs):
                if isinstance(ix, A.Slice):
                    break
                aff = _affine(ix, var)
                n = bd.extract_value(arr.v, [2, k])
                if aff is not None:
                    c = ir.Constant(i64t(), aff)
                    a0 = bd.add(lo, c)
                    a1 = bd.add(last, c)
                    good = bd.and_(bd.icmp_signed(">=", a0, ir.Constant(i64t(), 0)), bd.icmp_signed("<", a1, n))
                elif _invariant(ix, assigned | {var}, self):
                    try:
                        v = self.expr(ix)
                    except (Unsupported, UnitError):
                        continue
                    if v.t not in (I64, BOOL):
                        continue
                    v = self.as_int(v)
                    good = bd.and_(bd.icmp_signed(">=", v.v, ir.Constant(i64t(), 0)), bd.icmp_signed("<", v.v, n))
                else:
                    continue
                ok = bd.and_(ok, good)
                proven.add((id(node), k))
        if not proven:
            return None
        return bd.or_(bd.not_(nonempty), ok), proven

    # ------------------------------------------------------------------ parallel for
    def parallel_problem(self, s):
        if self.f is not None and self.f.in_parallel:
            return "it is nested inside another parallel for"
        for node in _walk_no_nested_loops(s.body):
            if isinstance(node, A.Return):
                return "it contains return"
            if isinstance(node, A.Break):
                return "it contains break"
            if isinstance(node, A.ExprStmt) and isinstance(node.expr, A.Call) and \
                    isinstance(node.expr.func, A.Name) and node.expr.func.id == "print":
                return "it prints (output order would be random)"
        for node in _walk(s.body):
            if isinstance(node, A.Return):
                return "it contains return"
        local = _declared_names(s.body) | set(s.targets)
        for name, kinds in _writes(s.body).items():
            if name in local:
                continue
            red = _reduction_kind(kinds)
            if red is None:
                return f"'{name}' is assigned in every iteration (use +=, -=, *=, x = min(x, ..) or max)"
            key = self.find(name)
            if key is None:
                return f"'{name}' is not a local variable"
            t = self.spec.slots[key][0]
            if t not in (I64, F64, F32):
                return f"'{name}' is a {tname(t)}, only numbers can be reduced"
        return None

    def gpu_parallel_for(self, s, rng):
        """One GPU thread per iteration: k = global thread id + krange[0], skip when k >= krange[1].

        `+=` / `-=` reductions accumulate in a register and are added atomically to apsi_red[j] at the end."""
        if s is not getattr(self.mc, "gpu_loop", None):
            raise Unsupported("a GPU kernel has exactly one parallel for (not in called functions)", s)
        if not (s.iter and (isinstance(s.iter, A.Range) or isinstance(s.iter, A.Call))):
            raise Unsupported("the GPU parallel for must count over a range", s)
        start, stop, step_c, step = rng
        name = s.targets[0]
        key = ("for", id(s), 0)
        local = _declared_names(s.body) | set(s.targets)
        reds = []
        for nm, kinds in _writes(s.body).items():
            if nm not in local:
                if _reduction_kind(kinds) != "+":
                    raise Unsupported(f"GPU kernels reduce with += / -= only ('{nm}')", s)
                reds.append(self.find(nm))
        if self.dry:
            self.scopes.append({})
            self.declare(name, key, Val(I64, None), s)
            self.block(s.body)
            self.scopes.pop()
            return
        self.mc.gpu_reds = [(k, self.spec.slots[k][0]) for k in reds]
        bd = self.f.b

        def sreg(n):
            f = self.mc.decl(f"llvm.nvvm.read.ptx.sreg.{n}", ir.IntType(32), [])
            return bd.zext(bd.call(f, []), i64t())
        kr = self.global_ptr(self.mc.gpu_global("apsi_krange", 2))
        klo = bd.load(bd.gep(kr, [ir.Constant(i64t(), 0), ir.Constant(i64t(), 0)]))
        khi = bd.load(bd.gep(kr, [ir.Constant(i64t(), 0), ir.Constant(i64t(), 1)]))
        tid = bd.add(bd.mul(sreg("ctaid.x"), sreg("ntid.x")), sreg("tid.x"))
        k = bd.add(klo, tid)
        run = self.f.fn.append_basic_block("gpu.iter")
        nxt = self.f.fn.append_basic_block("gpu.next")
        done = self.f.fn.append_basic_block("gpu.done")
        bd.cbranch(bd.icmp_signed("<", k, khi), run, done)
        bd.position_at_end(run)
        sv = ir.Constant(i64t(), step_c) if step_c is not None else step.v
        i = bd.add(start.v, bd.mul(k, sv))
        saved = {}
        for rk in reds:
            t = self.spec.slots[rk][0]
            loc = self.f.entry_b.alloca(lltype(t), name="red")
            saved[rk] = self.f.ptrs.get(rk)
            self.f.ptrs[rk] = loc
            bd.store(_identity("+", t), loc)
        self.scopes.append({})
        self.f.loops.append((nxt, None, None))
        try:
            self.declare(name, key, Val(I64, i), s)
            self.block(s.body, new_scope=False)
        finally:
            self.f.loops.pop()
            self.scopes.pop()
        if not bd.block.is_terminated:
            bd.branch(nxt)
        bd.position_at_end(nxt)
        red = self.global_ptr(self.mc.gpu_global("apsi_red", 16))
        for j, rk in enumerate(reds):
            t = self.spec.slots[rk][0]
            v = bd.load(self.f.ptrs[rk])
            slot = bd.gep(red, [ir.Constant(i64t(), 0), ir.Constant(i64t(), j)])
            if t == I64:
                bd.atomic_rmw("add", slot, v, "monotonic")
            else:
                v = v if t == F64 else bd.fpext(v, ir.DoubleType())
                bd.atomic_rmw("fadd", bd.bitcast(slot, ir.DoubleType().as_pointer()), v, "monotonic")
            self.f.ptrs[rk] = saved[rk] if saved[rk] is not None else self._alloca(rk)
        bd.branch(done)
        bd.position_at_end(done)

    def parallel_for(self, s, rng):
        if self.mc.gpu:
            return self.gpu_parallel_for(s, rng)
        start, stop, step_c, step = rng
        name = s.targets[0]
        key = ("for", id(s), 0)
        local = _declared_names(s.body) | set(s.targets)
        reds = {}
        for nm, kinds in _writes(s.body).items():
            if nm not in local:
                reds[self.find(nm)] = (_reduction_kind(kinds), nm)
        self.mc.uses_parallel = True
        if self.dry:
            self.scopes.append({})
            self.declare(name, key, Val(I64, None), s)
            self.block(s.body)
            self.scopes.pop()
            return
        bd = self.f.b
        sv = step.v if step_c is None else ir.Constant(i64t(), step_c)
        if step_c is None:
            self.check(bd.icmp_signed("!=", sv, ir.Constant(i64t(), 0)), E_ZERODIV, s)
        # trip count
        span = bd.sub(stop.v, start.v)
        if step_c == 1:
            trips = span
        else:
            adj = bd.select(bd.icmp_signed(">", sv, ir.Constant(i64t(), 0)),
                            bd.sub(sv, ir.Constant(i64t(), 1)), bd.add(sv, ir.Constant(i64t(), 1)))
            trips = bd.sdiv(bd.add(span, adj), sv)
        trips = bd.select(bd.icmp_signed(">", trips, ir.Constant(i64t(), 0)), trips, ir.Constant(i64t(), 0))
        nch = bd.select(bd.icmp_signed("<", trips, ir.Constant(i64t(), NCHUNKS)), trips,
                        ir.Constant(i64t(), NCHUNKS))
        # captured outer slots: everything visible in the enclosing scopes
        outer = []
        seen = set()
        for sc in self.scopes:
            for nm, k in sc.items():
                if k not in seen and k in self.spec.slots:
                    seen.add(k)
                    outer.append(k)
        red_keys = [k for k in outer if k in reds]
        partials = {}
        for k in red_keys:
            t = self.spec.slots[k][0]
            buf = self._entry_alloca(ir.ArrayType(lltype(t), NCHUNKS), "partials")
            partials[k] = buf
        dyn_keys = [k for k in outer if self.slot_dyn(k)]           # run-time provenance flags of outer values
        ptr_types = [self._alloca(k).type for k in outer] + [partials[k].type for k in red_keys] + \
            [self._np_alloca(k).type for k in dyn_keys]
        ctx_t = ir.LiteralStructType(ptr_types + [i64t(), i64t()])
        ctx = self._entry_alloca(ctx_t, "pctx")
        for j, k in enumerate(outer):
            bd.store(self._alloca(k), bd.gep(ctx, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), j)]))
        for j, k in enumerate(red_keys):
            bd.store(partials[k], bd.gep(ctx, [ir.Constant(ir.IntType(32), 0),
                                               ir.Constant(ir.IntType(32), len(outer) + j)]))
        for j, k in enumerate(dyn_keys):
            bd.store(self._np_alloca(k), bd.gep(ctx, [ir.Constant(ir.IntType(32), 0),
                                                      ir.Constant(ir.IntType(32), len(outer) + len(red_keys) + j)]))
        bd.store(start.v, bd.gep(ctx, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), len(ptr_types))]))
        bd.store(sv, bd.gep(ctx, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), len(ptr_types) + 1)]))
        # ---- the chunk function
        chunk = ir.Function(self.mc.module, ir.FunctionType(ir.VoidType(), [i8p(), i64t(), i64t(), i64t()]),
                            name=self.mc.fresh(self.spec.name + ".chunk"))
        chunk.linkage = "internal"
        chunk.attributes.add("nounwind")
        saved_f = self.f
        nf = new_frame(chunk)
        cb = nf.b
        nf.in_parallel = True
        cctx = cb.bitcast(chunk.args[0], ctx_t.as_pointer())
        for j, k in enumerate(outer):
            p = cb.load(cb.gep(cctx, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), j)]))
            if k in reds:
                t = self.spec.slots[k][0]
                loc = cb.alloca(lltype(t), name="red")
                cb.store(_identity(reds[k][0], t), loc)
                nf.ptrs[k] = loc
                nf.reductions[k] = reds[k][0]
            else:
                # read-only inside the loop (parallel_problem guarantees it): a private copy lets LLVM keep it
                # in registers instead of re-loading it after every store
                t = self.spec.slots[k][0]
                loc = nf.entry_b.alloca(lltype(t), name="cap")
                cb.store(cb.load(p), loc)
                nf.ptrs[k] = loc
        for j, k in enumerate(dyn_keys):
            fp = cb.load(cb.gep(cctx, [ir.Constant(ir.IntType(32), 0),
                                       ir.Constant(ir.IntType(32), len(outer) + len(red_keys) + j)]))
            loc = nf.entry_b.alloca(i1t(), name="npflag")
            cb.store(cb.load(fp), loc)
            nf.__dict__.setdefault("nps", {})[k] = loc
        cstart = cb.load(cb.gep(cctx, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), len(ptr_types))]))
        cstep = cb.load(cb.gep(cctx, [ir.Constant(ir.IntType(32), 0),
                                      ir.Constant(ir.IntType(32), len(ptr_types) + 1)]))
        self.f = nf
        try:
            lo_k, hi_k = chunk.args[1], chunk.args[2]
            if step_c == 1:
                lo_i = Val(I64, cb.add(cstart, lo_k))
                hi_i = Val(I64, cb.add(cstart, hi_k))

                def body(i):
                    self.declare(name, key, i, s)
                    self.block(s.body, new_scope=False)
                self.scopes.append({})
                plan = self.bounds_plan(s, name, lo_i, hi_i, 1)
                if plan is None:
                    self.counted_loop(s, lo_i, hi_i, 1, None, body)
                else:
                    ok, proven = plan
                    fb = chunk.append_basic_block("for.fast")
                    sb = chunk.append_basic_block("for.checked")
                    jb = chunk.append_basic_block("for.join")
                    cb.cbranch(ok, fb, sb)
                    cb.position_at_end(fb)
                    sp = self.proven
                    self.proven = sp | proven
                    self.counted_loop(s, lo_i, hi_i, 1, None, body)
                    self.proven = sp
                    if not cb.block.is_terminated:
                        cb.branch(jb)
                    cb.position_at_end(sb)
                    self.counted_loop(s, lo_i, hi_i, 1, None, body)
                    if not cb.block.is_terminated:
                        cb.branch(jb)
                    cb.position_at_end(jb)
                self.scopes.pop()
            else:
                def body(kv):
                    i = Val(I64, cb.add(cstart, cb.mul(kv.v, cstep)))
                    self.declare(name, key, i, s)
                    self.block(s.body, new_scope=False)
                self.counted_loop(s, Val(I64, lo_k), Val(I64, hi_k), 1, None, body)
            if not cb.block.is_terminated:
                for j, k in enumerate(red_keys):
                    buf = cb.load(cb.gep(cctx, [ir.Constant(ir.IntType(32), 0),
                                                ir.Constant(ir.IntType(32), len(outer) + j)]))
                    cb.store(cb.load(nf.ptrs[k]), cb.gep(buf, [ir.Constant(i64t(), 0), chunk.args[3]]))
                self.free_arena(nf.arena)
                cb.ret_void()
        finally:
            self.f = saved_f
        # ---- dispatch and combine
        bd = self.f.b
        disp = self.mc.decl("apsi_parallel", ir.VoidType(), [i8p(), i8p(), i64t(), i64t(), i64t()])
        any_trips = self.f.fn.append_basic_block("par.run")
        after = self.f.fn.append_basic_block("par.done")
        bd.cbranch(bd.icmp_signed(">", trips, ir.Constant(i64t(), 0)), any_trips, after)
        bd.position_at_end(any_trips)
        bd.call(disp, [bd.bitcast(chunk, i8p()), bd.bitcast(ctx, i8p()), ir.Constant(i64t(), 0), trips, nch])
        self.check_err_flag()
        for k in red_keys:
            op = reds[k][0]
            t = self.spec.slots[k][0]
            ptr = self._alloca(k)
            pre = bd.block
            head = self.f.fn.append_basic_block("red")
            body_bb = self.f.fn.append_basic_block("red.body")
            done = self.f.fn.append_basic_block("red.done")
            bd.branch(head)
            bd.position_at_end(head)
            c = bd.phi(i64t())
            c.add_incoming(ir.Constant(i64t(), 0), pre)
            bd.cbranch(bd.icmp_signed("<", c, nch), body_bb, done)
            bd.position_at_end(body_bb)
            part = Val(t, bd.load(bd.gep(partials[k], [ir.Constant(i64t(), 0), c])), self.spec.slots[k][1])
            cur = Val(t, bd.load(ptr), self.spec.slots[k][1])
            if op in ("min", "max"):
                nv = self.minmax(op, cur, part, s)
            else:
                nv = self.binop(op, cur, part, s)
            bd.store(self.convert(nv, t, s).v, ptr)
            c.add_incoming(bd.add(c, ir.Constant(i64t(), 1)), bd.block)
            bd.branch(head)
            bd.position_at_end(done)
        bd.branch(after)
        bd.position_at_end(after)

    def _entry_alloca(self, t, name):
        return self.f.entry_b.alloca(t, name=name)

    # ------------------------------------------------------------------ expressions
    def expr(self, e):
        m = getattr(self, "e_" + type(e).__name__, None)
        if m is None:
            raise Unsupported(_describe(e), e)
        return m(e)

    def const_val(self, x, d=DIMLESS, node=None, weak=False):
        if isinstance(x, (bool, np.bool_)):
            return Val(BOOL, None if self.dry else ir.Constant(i1t(), int(bool(x))), d, const=bool(x))
        if isinstance(x, (int, np.integer)):
            if not -(1 << 63) <= int(x) < (1 << 63):
                raise Unsupported("an integer constant that does not fit in 64 bits", node)
            return Val(I64, None if self.dry else ir.Constant(i64t(), int(x)), d, const=int(x), weak=weak)
        if isinstance(x, np.float32):
            return Val(F32, None if self.dry else ir.Constant(ir.FloatType(), float(x)), d, const=float(x))
        if isinstance(x, (float, np.floating, Fraction)):
            return Val(F64, None if self.dry else ir.Constant(ir.DoubleType(), float(x)), d, const=float(x),
                       weak=weak)
        raise Unsupported(f"a constant of type {type(x).__name__}", node)

    def e_Num(self, e):
        if e.unit is not None:
            text, scale, dim = e.unit
            if self.erased:
                dim = DIMLESS
            return self.const_val(float(e.value * scale), dim, e, weak=dim.dimensionless)
        return self.const_val(e.value, DIMLESS, e, weak=True)

    def e_Bool(self, e):
        return self.const_val(bool(e.value), node=e)

    def e_Name(self, e):
        key = self.find(e.id)
        if key is not None:
            return self.load_slot(key, e)
        res = self.r.lookup(e.id)
        if res is None:
            raise Unsupported(f"'{e.id}' is not defined", e)
        if res[0] == "builtin":
            c = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan}.get(res[1])
            if c is not None:
                return self.const_val(c, node=e)
            raise Unsupported(f"using the built-in '{e.id}' as a value", e)
        _, obj, mutable = res
        return self.global_value(e.id, obj, mutable, e)

    def global_value(self, name, obj, mutable, node):
        tv = type_of_value(obj)
        if tv is None:
            if apsi_function(obj) is not None:
                raise Unsupported(f"using the function '{name}' as a value", node)
            raise Unsupported(f"the global '{name}' ({type(obj).__name__})", node)
        t, d = tv
        if self.erased:
            d = DIMLESS
        if t in SCALARS:
            if mutable:
                raise Unsupported(f"reading the mutable global '{name}' (pass it as an argument)", node)
            raw = obj.v if isinstance(obj, Quantity) else obj
            v = self.const_val(raw, d, node)
            v.npv = isinstance(raw, np.generic)
            return v
        idx = self.mc.global_array(self.r, name, t)
        self.spec.globals[name] = idx
        if self.dry:
            return Val(t, None, d)
        bd = self.f.b
        tab = self.mc.gtable()
        p = bd.load(bd.gep(tab, [ir.Constant(i64t(), 0), ir.Constant(i64t(), idx)]))
        return Val(t, bd.load(bd.bitcast(p, lltype(t).as_pointer())), d)

    def e_Unary(self, e):
        v = self.expr(e.operand)
        if e.op == "not":
            c = self.truth(v, e)
            return Val(BOOL, None if self.dry else self.f.b.not_(c))
        if e.op == "+":
            return v
        if e.op == "-":
            if is_arrish(v.t):
                return self.lazy_un("neg", v, e)
            if v.const is not None and not isinstance(v.const, bool):
                return self.const_val(-v.const, v.d, e, weak=v.weak)
            if v.t in (BOOL, I64):
                v2 = self.as_int(v)
                r = self.int_arith("-", self.const_val(0, v2.d), v2, e, None, v.npv)
                r.npv = v.npv
                return r
            if self.dry:
                return Val(v.t, None, v.d, npv=v.npv)
            return Val(v.t, self.f.b.fneg(v.v), v.d, npv=v.npv)
        raise Unsupported(f"unary '{e.op}'", e)

    def e_Logic(self, e):
        a = self.expr(e.left)
        b = self.expr(e.right)
        if a.t != b.t or a.t not in SCALARS:
            if a.t == BOOL or b.t == BOOL or a.t != b.t:
                raise Unsupported(f"'{e.op}' between {tname(a.t)} and {tname(b.t)}", e)
        if self.dry:
            return Val(a.t, None, a.d)
        # both sides are evaluated eagerly only when the right side is free of effects
        if _pure(e.right):
            ca = self.truth(a, e)
            pick = self.f.b.select(ca, b.v, a.v) if e.op == "and" else self.f.b.select(ca, a.v, b.v)
            return Val(a.t, pick, a.d)
        raise Unsupported(f"'{e.op}' with a right side that has side effects", e)

    def e_IfExpr(self, e):
        c = self.truth(self.expr(e.cond), e.cond)
        if self.dry:
            a, b = self.expr(e.a), self.expr(e.b)
            if a.d != b.d and a.t != ANY and b.t != ANY:
                raise UnitError(f"the two branches have different units ({_dims(a.d)} and {_dims(b.d)})",
                                e.line, e.col)
            t = unify(a.t, b.t, e) if not (is_lazy(a.t) or is_lazy(b.t)) else None
            if t is None:
                raise Unsupported("tensor expressions in if-expressions", e)
            return Val(t, None, a.d, npv=a.npv if (isinstance(a.npv, bool) and a.npv is b.npv) else DYN)
        bd = self.f.b
        ta = self.f.fn.append_basic_block("ifx.a")
        tb = self.f.fn.append_basic_block("ifx.b")
        end = self.f.fn.append_basic_block("ifx.end")
        bd.cbranch(c, ta, tb)
        bd.position_at_end(ta)
        a = self.materialize(self.expr(e.a), e)
        a_end = bd.block                         # evaluating a branch may have created blocks (checks)
        bd.position_at_end(tb)
        b = self.materialize(self.expr(e.b), e)
        b_end = bd.block
        t = unify(a.t, b.t, e)
        bd.position_at_end(a_end)
        av = self.convert(a, t, e).v
        afl = self.np_i1(a.npv)
        ta_end = bd.block
        bd.branch(end)
        bd.position_at_end(b_end)
        bv = self.convert(b, t, e).v
        bfl = self.np_i1(b.npv)
        tb_end = bd.block
        bd.branch(end)
        bd.position_at_end(end)
        phi = bd.phi(lltype(t))
        phi.add_incoming(av, ta_end)
        phi.add_incoming(bv, tb_end)
        if isinstance(a.npv, bool) and a.npv is b.npv:
            return Val(t, phi, a.d, npv=a.npv)
        flag = bd.phi(i1t())
        flag.add_incoming(afl, ta_end)
        flag.add_incoming(bfl, tb_end)
        return Val(t, phi, a.d, npv=flag)

    def e_TupleLit(self, e):
        items = [self.materialize(self.expr(x), e) for x in e.items]
        t = TUP(x.t for x in items)
        d = tuple(x.d for x in items)
        if self.dry:
            return Val(t, None, d)
        st = ir.Constant(lltype(t), ir.Undefined)
        for k, x in enumerate(items):
            st = self.f.b.insert_value(st, x.v, k)
        return Val(t, st, d)

    def e_As(self, e):
        v = self.expr(e.expr)
        text, scale, dim = e.unit
        if self.erased:
            return self.binop("/", v, self.const_val(float(scale)), e) if dim.dimensionless and scale != 1 else v
        if v.d != dim:
            raise UnitError(f"cannot express {dim_name(v.d)} in [{text}] ({dim_name(dim)})", e.line, e.col)
        if dim.dimensionless and scale != 1:
            return self.binop("/", v, self.const_val(float(scale)), e)
        return v

    def e_Attr(self, e):
        if isinstance(e.obj, A.Name) and self.find(e.obj.id) is None:
            res = self.r.lookup(e.obj.id)
            if res is not None and res[0] == "value" and type_of_value(res[1]) is None:
                try:
                    obj = self.r.attr(res[1], e.name)
                except Exception:
                    raise Unsupported(f"'{e.obj.id}.{e.name}'", e)
                return self.global_value(f"{e.obj.id}.{e.name}", obj, False, e)
        v = self.expr(e.obj)
        if is_arrish(v.t):
            v = self.materialize(v, e)
            if e.name == "shape":
                return self.shape_tuple(v)
            if e.name == "ndim":
                return self.const_val(v.t[2])
            if e.name == "size":
                n = self.const_val(1)
                for k in range(v.t[2]):
                    n = self.int_arith("*", n, self.dim_size(v, k), e)
                return n
            if e.name == "T" and v.t[2] == 2:
                return self.transpose(v)
        raise Unsupported(f"the attribute '.{e.name}' here", e)

    def shape_tuple(self, v):
        items = [self.dim_size(v, k) for k in range(v.t[2])]
        t = TUP([I64] * len(items))
        if self.dry:
            return Val(t, None, tuple(DIMLESS for _ in items))
        st = ir.Constant(lltype(t), ir.Undefined)
        for k, x in enumerate(items):
            st = self.f.b.insert_value(st, x.v, k)
        return Val(t, st, tuple(DIMLESS for _ in items))

    def transpose(self, v):
        t = ARR(v.t[1], 2, False)
        if self.dry:
            return Val(t, None, v.d)
        bd = self.f.b
        st = v.v
        s0, s1 = bd.extract_value(st, [2, 0]), bd.extract_value(st, [2, 1])
        r0, r1 = bd.extract_value(st, [3, 0]), bd.extract_value(st, [3, 1])
        if v.t[3]:
            r1 = ir.Constant(i64t(), 1)
        st = bd.insert_value(st, s1, [2, 0])
        st = bd.insert_value(st, s0, [2, 1])
        st = bd.insert_value(st, r1, [3, 0])
        st = bd.insert_value(st, r0, [3, 1])
        return Val(t, st, v.d)

    def dim_size(self, arr, k):
        if self.dry:
            return Val(I64, None)
        return Val(I64, self.f.b.extract_value(arr.v, [2, k]))

    # ------------------------------------------------------------------ indexing
    def index_parts(self, e, node):
        """(array value, [index exprs]) for a[i], a[i, j], a[i][j] chains."""
        idxs = []
        cur = e
        while isinstance(cur, A.Index):
            ix = cur.index.items if isinstance(cur.index, A.TupleLit) else [cur.index]
            idxs = [(cur, k, x) for k, x in enumerate(ix)] + idxs
            cur = cur.obj
            if isinstance(cur, A.Index) and any(isinstance(x, A.Slice) for x in ix):
                break
        base = self.expr(cur)
        if base.t[0] == "t":
            if len(idxs) != 1:
                raise Unsupported("indexing a tuple twice", node)
            return base, idxs
        if not is_arrish(base.t):
            raise Unsupported(f"indexing {tname(base.t)}", node)
        base = self.materialize(base, node)
        if len(idxs) > base.t[2]:
            raise Unsupported(f"{len(idxs)} indices for a rank-{base.t[2]} tensor", node)
        return base, idxs

    def e_Index(self, e):
        base, idxs = self.index_parts(e, e)
        return self.index_load(base, idxs, e)

    def index_load(self, base, idxs, node, proven=False):
        if base.t[0] == "t":
            (_, _, ix), = idxs
            k = self.expr(ix) if isinstance(ix, A.AST) else ix
            if k.const is None:
                raise Unsupported("indexing a tuple with a non-constant", node)
            n = len(base.t[1])
            kk = k.const + n if k.const < 0 else k.const
            if not 0 <= kk < n:
                raise Unsupported("tuple index out of range", node)
            return Val(base.t[1][kk], None if self.dry else self.f.b.extract_value(base.v, kk),
                       base.d[kk] if isinstance(base.d, tuple) else DIMLESS)
        view, rest = self.view(base, idxs, node, proven)
        if rest == 0:
            return view
        return view

    def view(self, base, idxs, node, proven=False):
        """Apply indices/slices along the leading axes; returns (element or sub-view, remaining rank)."""
        t = base.t
        elem, nd, contig = t[1], t[2], t[3]
        if self.dry:
            out_nd = nd
            for item in idxs:
                ix = item[2] if isinstance(item, tuple) else None
                if isinstance(ix, A.Slice):
                    for p in (ix.lo, ix.hi, ix.step):
                        if p is not None:
                            pv = self.expr(p)
                            if pv.t not in (I64, BOOL):
                                raise Unsupported("slices with non-integer bounds", node)
                    if ix.step is not None:
                        contig = False
                    continue
                if isinstance(item, tuple):
                    v = self.expr(ix)
                    if is_arrish(v.t):
                        raise Unsupported("indexing with a tensor (fancy indexing)", node)
                    if v.t not in (I64, BOOL):
                        raise Unsupported(f"an index of type {tname(v.t)}", node)
                out_nd -= 1
            if out_nd == 0:
                return Val(ELEM_SCALAR[elem], None, base.d, npv=True), 0
            return Val(ARR(elem, out_nd, contig), None, base.d), out_nd
        bd = self.f.b
        st = base.v
        ptr = bd.extract_value(st, 0)
        shapes = [bd.extract_value(st, [2, k]) for k in range(nd)]
        strides = [bd.extract_value(st, [3, k]) for k in range(nd)]
        if contig:
            strides[-1] = ir.Constant(i64t(), 1)
        off = ir.Constant(i64t(), 0)
        new_shape, new_strides = [], []
        axis = 0
        for item in idxs:
            if isinstance(item, tuple):
                owner, k, ix = item
            else:
                owner, k, ix = None, axis, item
            n = shapes[axis]
            if isinstance(ix, A.Slice):
                lo, hi, step = self.slice_bounds(ix, n, node)
                cnt = bd.sdiv(bd.add(bd.sub(hi, lo), bd.sub(step, ir.Constant(i64t(), 1))), step)
                cnt = bd.select(bd.icmp_signed(">", cnt, ir.Constant(i64t(), 0)), cnt, ir.Constant(i64t(), 0))
                off = bd.add(off, bd.mul(lo, strides[axis]))
                new_shape.append(cnt)
                new_strides.append(bd.mul(strides[axis], step))
                if ix.step is not None:
                    contig = False
                axis += 1
                continue
            if isinstance(ix, A.AST):
                iv = self.as_int(self.expr(ix))
            else:
                iv = ix
            i = iv.v
            known = proven or (owner is not None and (id(owner), k) in self.proven) or \
                (iv.const is not None and iv.const >= 0 and False)
            if not known and not self.fast:
                neg = bd.icmp_signed("<", i, ir.Constant(i64t(), 0))
                i = bd.select(neg, bd.add(i, n), i)
                self.check(bd.icmp_unsigned("<", i, n), E_INDEX, node, iv.v, n)
            off = bd.add(off, bd.mul(i, strides[axis]))
            axis += 1
        # the remaining axes are untouched
        for a in range(axis, nd):
            new_shape.append(shapes[a])
            new_strides.append(strides[a])
        p = bd.gep(ptr, [off], inbounds=True)
        if not new_shape:
            raw = bd.load(p)
            return self.from_elem(raw, elem, base.d), 0
        if len(new_shape) != nd - sum(1 for it in idxs if not isinstance(it[2] if isinstance(it, tuple) else it,
                                                                           A.Slice)):
            pass
        out_t = ARR(elem, len(new_shape), contig and (new_strides[-1] is strides[-1] or contig))
        vst = ir.Constant(arr_ll(elem, len(new_shape)), ir.Undefined)
        vst = bd.insert_value(vst, p, 0)
        vst = bd.insert_value(vst, ir.Constant(i8p(), None), 1)      # a view never owns memory
        for k in range(len(new_shape)):
            vst = bd.insert_value(vst, new_shape[k], [2, k])
            vst = bd.insert_value(vst, new_strides[k], [3, k])
        return Val(out_t, vst, base.d), len(new_shape)

    def slice_bounds(self, sl, n, node):
        """Python slice semantics (clamped, negative bounds count from the end)."""
        bd = self.f.b
        zero = ir.Constant(i64t(), 0)
        if sl.step is not None:
            sv = self.as_int(self.expr(sl.step))
            if sv.const is None or sv.const <= 0:
                raise Unsupported("slices with a non-constant or negative step", node)
            step = sv.v
        else:
            step = ir.Constant(i64t(), 1)

        def norm(x, default):
            if x is None:
                return default
            v = self.as_int(self.expr(x)).v
            v = bd.select(bd.icmp_signed("<", v, zero), bd.add(v, n), v)
            v = bd.select(bd.icmp_signed("<", v, zero), zero, v)
            return bd.select(bd.icmp_signed(">", v, n), n, v)
        lo = norm(sl.lo, zero)
        hi = norm(sl.hi, n)
        hi = bd.select(bd.icmp_signed("<", hi, lo), lo, hi)
        return lo, hi, step

    def from_elem(self, raw, elem, d):
        bd = self.f.b
        if elem == "f2":
            return Val(F32, half_to_f32(bd, raw), d, npv=True)
        if elem == "h2":
            return Val(F32, bf16_to_f32(bd, raw), d, npv=True)
        if elem in FP8:
            return Val(F32, fp8_to_f32(bd, raw, elem), d, npv=True)
        if elem in ("f8", "f4", "i8"):
            return Val(ELEM_SCALAR[elem], raw, d, npv=True)
        if elem in ("i4", "i2", "i1"):
            return Val(I64, bd.sext(raw, i64t()), d, npv=True)
        if elem == "u1":
            return Val(I64, bd.zext(raw, i64t()), d, npv=True)
        return Val(BOOL, bd.icmp_unsigned("!=", raw, ir.Constant(ir.IntType(8), 0)), d, npv=True)

    def to_elem(self, v, elem, node):
        bd = self.f.b
        st = ELEM_SCALAR[elem]
        if elem == "b1":
            return bd.zext(self.truth(v, node), ir.IntType(8))
        if elem == "f2":
            return f32_to_half(bd, self.convert(v, F32, node).v)
        if elem == "h2":
            return f32_to_bf16(bd, self.convert(v, F32, node).v)
        if elem in FP8:
            return f32_to_fp8(bd, self.convert(v, F32, node).v, elem)
        if st in (F64, F32):
            return self.convert(v, st, node).v
        iv = self.convert(v, I64, node, store=True).v
        if elem == "i8":
            return iv
        return bd.trunc(iv, elem_ll(elem))

    def store_index(self, tgt, v, node):
        arr, idxs = self.index_parts(tgt, node)
        if arr.t[0] == "t":
            raise Unsupported("assigning into a tuple", node)
        self.store_to(arr, idxs, v, tgt)

    def store_to(self, arr, idxs, v, node):
        if arr.d != v.d and v.t != ANY and not (v.const == 0 and v.d.dimensionless):
            raise UnitError(f"cannot store {_dims(v.d)} in a tensor of {_dims(arr.d)}", node.line, node.col)
        if self.dry:
            self.view(arr, idxs, node)
            return
        bd = self.f.b
        target, rest = self.view_ptr(arr, idxs, node)
        if rest is None:                                   # a single element
            bd.store(self.to_elem(self.materialize(v, node) if is_lazy(v.t) else v, arr.t[1], node), target)
            return
        self.write_into(rest, v, node)                     # a whole sub-tensor

    def view_ptr(self, arr, idxs, node):
        """Element pointer (and None) for full indexing, or (None, view) for a sub-tensor target."""
        bd = self.f.b
        full = len(idxs) == arr.t[2] and not any(isinstance(x[2], A.Slice) for x in idxs)
        if not full:
            v, _ = self.view(arr, idxs, node)
            return None, v
        st = arr.v
        ptr = bd.extract_value(st, 0)
        off = ir.Constant(i64t(), 0)
        for axis, (owner, k, ix) in enumerate(idxs):
            n = bd.extract_value(st, [2, axis])
            stride = ir.Constant(i64t(), 1) if (arr.t[3] and axis == arr.t[2] - 1) else bd.extract_value(st, [3, axis])
            iv = self.as_int(self.expr(ix))
            i = iv.v
            if (id(owner), k) not in self.proven and not self.fast:
                neg = bd.icmp_signed("<", i, ir.Constant(i64t(), 0))
                i = bd.select(neg, bd.add(i, n), i)
                self.check(bd.icmp_unsigned("<", i, n), E_INDEX, node, iv.v, n)
            off = bd.add(off, bd.mul(i, stride))
        return bd.gep(ptr, [off], inbounds=True), None

    # ------------------------------------------------------------------ conversions
    def as_int(self, v):
        if v.t == I64:
            return v
        if v.t == BOOL:
            return Val(I64, None if self.dry else self.f.b.zext(v.v, i64t()), v.d,
                       const=None if v.const is None else int(v.const), npv=v.npv)
        if v.t == ANY:
            return Val(I64, None)
        raise Unsupported(f"an integer was expected, got {tname(v.t)}")

    def convert(self, v, t, node, store=False):
        if v.t == t or t == ANY:
            return v
        if self.dry:
            return Val(t, None, v.d)
        bd = self.f.b
        if v.t == ANY:
            return Val(t, ir.Constant(lltype(t), ir.Undefined), v.d)
        if t == F64:
            if v.t == I64:
                return Val(t, bd.sitofp(v.v, ir.DoubleType()), v.d)
            if v.t == BOOL:
                return Val(t, bd.uitofp(v.v, ir.DoubleType()), v.d)
            if v.t == F32:
                return Val(t, bd.fpext(v.v, ir.DoubleType()), v.d)
        if t == F32:
            if v.t == I64:
                return Val(t, bd.sitofp(v.v, ir.FloatType()), v.d)
            if v.t == BOOL:
                return Val(t, bd.uitofp(v.v, ir.FloatType()), v.d)
            if v.t == F64:
                return Val(t, bd.fptrunc(v.v, ir.FloatType()), v.d)
        if t == I64:
            if v.t == BOOL:
                return Val(t, bd.zext(v.v, i64t()), v.d)
            if v.t in (F64, F32) and store:                     # storing a float into an integer tensor
                ft = lltype(v.t)
                fn = self.mc.decl(f"llvm.fptosi.sat.i64.{'f64' if v.t == F64 else 'f32'}", i64t(), [ft])
                return Val(t, bd.call(fn, [v.v]), v.d)
        if is_arr(t) and is_arr(v.t) and t[1] == v.t[1] and t[2] == v.t[2]:
            return Val(t, v.v, v.d)                              # contiguity only weakens
        if t[0] == "t" and v.t[0] == "t" and len(t[1]) == len(v.t[1]):
            st = ir.Constant(lltype(t), ir.Undefined)
            for k, (a, b) in enumerate(zip(v.t[1], t[1])):
                x = self.convert(Val(a, bd.extract_value(v.v, k)), b, node, store)
                st = bd.insert_value(st, x.v, k)
            return Val(t, st, v.d)
        raise Unsupported(f"converting {tname(v.t)} to {tname(t)}", node)

    def truth(self, v, node):
        if v.t == BOOL:
            return v.v
        if is_arrish(v.t):
            raise Unsupported("a tensor used as a condition", node)
        if self.dry:
            return None
        bd = self.f.b
        if v.t == I64:
            return bd.icmp_signed("!=", v.v, ir.Constant(i64t(), 0))
        if v.t in (F64, F32):
            return bd.fcmp_unordered("!=", v.v, ir.Constant(lltype(v.t), 0.0))
        if v.t == ANY:
            return ir.Constant(i1t(), 0)
        raise Unsupported(f"{tname(v.t)} used as a condition", node)

    # ------------------------------------------------------------------ arithmetic
    def e_Binary(self, e):
        a = self.expr(e.left)
        b = self.expr(e.right)
        return self.binop(e.op, a, b, e)

    def binop(self, op, a, b, node):
        if op in ("==", "!=", "<", "<=", ">", ">="):
            return self.compare(op, a, b, node)
        if op == "~=":
            return self.approx(a, b, node)
        if op == "@":
            return self.matmul(a, b, node)
        if op not in ("+", "-", "*", "/", "//", "%", "**"):
            raise Unsupported(f"the operator '{op}'", node)
        if is_arrish(a.t) or is_arrish(b.t):
            return self.lazy_bin(op, a, b, node)
        if a.t[0] == "t" or b.t[0] == "t" or NONE in (a.t, b.t):
            raise Unsupported(f"'{op}' on {tname(a.t)} and {tname(b.t)}", node)
        d = self.result_dim(op, a, b, node)
        npv = self.np_or(a.npv, b.npv)
        r = self._scalar_binop(op, a, b, d, node, npv)
        r.npv = npv
        return r

    def _scalar_binop(self, op, a, b, d, node, npv):
        if a.const is not None and b.const is not None and op in ("+", "-", "*") \
                and not isinstance(a.const, bool) and not isinstance(b.const, bool):
            r = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y}[op](a.const, b.const)
            if isinstance(r, float) or -(1 << 63) <= r < (1 << 63):
                rt = arith(a.t, b.t, a.weak, b.weak)
                if rt == I64:
                    return self.const_val(int(r), d, node, weak=a.weak and b.weak)
                if rt == F64:
                    return self.const_val(float(r), d, node, weak=a.weak and b.weak)
        if op == "/":
            t = arith(a.t, b.t, a.weak, b.weak)
            t = F64 if t == I64 else t
            av, bv = self.convert(a, t, node), self.convert(b, t, node)
            if npv is not True and not self.fast and not (b.const is not None and b.const != 0):
                # Python numbers: x / 0 is ZeroDivisionError (NumPy scalars give inf/nan instead)
                nz = None if self.dry else self.f.b.fcmp_unordered("!=", bv.v, ir.Constant(lltype(t), 0.0))
                self.py_check(npv, nz, E_ZERODIV, node)
            return self.fop("/", av, bv, t, d)
        if op == "**":
            return self.power(a, b, d, node, npv)
        t = arith(a.t, b.t, a.weak, b.weak)
        if t == I64:
            return self.int_arith(op, self.as_int(a), self.as_int(b), node, d, npv)
        if op in ("//", "%"):
            return self.float_divmod(op, self.convert(a, t, node), self.convert(b, t, node), t, d, node, npv)
        return self.fop(op, self.convert(a, t, node), self.convert(b, t, node), t, d)

    def matmul(self, a, b, node):
        """a @ b for vectors and matrices: the optimised multi-threaded BLAS (the same kernels NumPy uses)."""
        if not (is_arrish(a.t) and is_arrish(b.t)):
            raise Unsupported("@ needs two tensors", node)
        if self.mc.gpu:
            raise Unsupported("@ inside a GPU kernel (write the loops, or use matmul() outside)", node)
        a, b = self.materialize(a, node), self.materialize(b, node)
        if a.t[1] != b.t[1] or a.t[1] not in ("f8", "f4"):
            raise Unsupported("@ in native code needs two f64 or two f32 tensors", node)
        na, nb = a.t[2], b.t[2]
        if na not in (1, 2) or nb not in (1, 2):
            raise Unsupported("@ of tensors with more than 2 axes in native code", node)
        d = a.d * b.d
        if na == 1 and nb == 1:
            r = self.reduce("sum", self.lazy_bin("*", Val(a.t, a.v, DIMLESS), Val(b.t, b.v, DIMLESS), node), node)
            return Val(r.t, r.v, d, npv=True)
        out_nd = 2 if na == 2 and nb == 2 else 1
        self.spec.may_fail = True
        if self.dry:
            return Val(ARR(a.t[1], out_nd, True), None, d)
        bd = self.f.b
        ka = bd.extract_value(a.v, [2, na - 1])
        kb = bd.extract_value(b.v, [2, 0])
        self.check(bd.icmp_signed("==", ka, kb), E_SHAPE, node, ka, kb)
        if na == 2 and nb == 2:
            shape = [bd.extract_value(a.v, [2, 0]), bd.extract_value(b.v, [2, 1])]
        elif na == 2:
            shape = [bd.extract_value(a.v, [2, 0])]
        else:
            shape = [bd.extract_value(b.v, [2, 1])]
        out = self.alloc_array(a.t[1], shape, node)
        descs = []
        for v in (a, b, out):
            p = self._entry_alloca(lltype(v.t), "mm")
            bd.store(v.v, p)
            descs.append(bd.bitcast(p, i8p()))
        code = (8 if a.t[1] == "f8" else 4) * 100 + na * 10 + nb
        fn = self.mc.decl("apsi_matmul", ir.VoidType(), [i64t(), i8p(), i8p(), i8p()])
        bd.call(fn, [ir.Constant(i64t(), code)] + descs)
        self.check_err_flag()
        out.d = d
        return out

    def result_dim(self, op, a, b, node):
        da, db = a.d, b.d
        if a.t == ANY:
            da = db if op in ("+", "-", "%", "//") else da
        if b.t == ANY:
            db = da if op in ("+", "-", "%", "//") else db
        if op in ("+", "-", "%", "//"):
            if da != db:
                if a.const == 0 and da.dimensionless:
                    return db
                if b.const == 0 and db.dimensionless:
                    return da
                verb = {"+": "add", "-": "subtract", "%": "take the remainder of", "//": "floor-divide"}[op]
                raise UnitError(f"cannot {verb} {dim_name(da)} and {dim_name(db)}", node.line, node.col)
            return DIMLESS if op == "//" else da
        if op == "*":
            return da * db
        if op == "/":
            return da / db
        if op == "**":
            if not db.dimensionless:
                raise UnitError("an exponent must be dimensionless", node.line, node.col)
            if da.dimensionless:
                return DIMLESS
            if b.const is None:
                raise Unsupported("a variable exponent on a quantity with a unit", node)
            return da ** b.const
        return DIMLESS

    def fop(self, op, a, b, t, d):
        if self.dry:
            return Val(t, None, d)
        bd = self.f.b
        f = {"+": bd.fadd, "-": bd.fsub, "*": bd.fmul, "/": bd.fdiv}[op]
        r = f(a.v, b.v)
        if self.fast:
            r.flags.append("fast")
        return Val(t, r, d)

    def int_arith(self, op, a, b, node, d=None, npv=False):
        """Int + - * // %: Python semantics (errors on overflow and division by zero) for Python numbers, NumPy
        semantics (wrap-around, x // 0 == x % 0 == 0) for NumPy scalars, decided at run time when npv is a flag."""
        d = a.d if d is None else d
        if self.dry:
            if npv is not True and not (self.fast and op in ("+", "-", "*")):
                self.spec.may_fail = True
            return Val(I64, None, d, npv=npv)
        bd = self.f.b
        python_only = npv is False
        if op in ("+", "-", "*"):
            if npv is True:
                return Val(I64, {"+": bd.add, "-": bd.sub, "*": bd.mul}[op](a.v, b.v), d, npv=True)
            if self.fast:                          # like C: signed overflow is undefined -> nsw
                r = {"+": bd.add, "-": bd.sub, "*": bd.mul}[op](a.v, b.v, flags=["nsw"] if python_only else [])
                return Val(I64, r, d, npv=npv)
            f = {"+": bd.sadd_with_overflow, "-": bd.ssub_with_overflow, "*": bd.smul_with_overflow}[op]
            res = f(a.v, b.v)
            self.py_check(npv, bd.not_(bd.extract_value(res, 1)), E_OVERFLOW, node, a.v, b.v)
            return Val(I64, bd.extract_value(res, 0), d, npv=npv)
        if op not in ("//", "%"):
            raise Unsupported(f"integer '{op}'", node)
        zero = ir.Constant(i64t(), 0)
        one = ir.Constant(i64t(), 1)
        imin = ir.Constant(i64t(), -(1 << 63))
        bz = bd.icmp_signed("==", b.v, zero)
        ovf = bd.and_(bd.icmp_signed("==", a.v, imin), bd.icmp_signed("==", b.v, ir.Constant(i64t(), -1)))
        if npv is not True:
            self.py_check(npv, bd.not_(bz), E_ZERODIV, node, a.v)
            if not self.fast:
                self.py_check(npv, bd.not_(ovf), E_OVERFLOW, node, a.v, b.v)
        if python_only:                             # the checks above guarantee a safe divisor
            bs = b.v
        else:
            bs = bd.select(bd.or_(bz, ovf), one, b.v)
        q = bd.sdiv(a.v, bs)
        r = bd.srem(a.v, bs)
        adjust = bd.and_(bd.icmp_signed("!=", r, zero),
                         bd.xor(bd.icmp_signed("<", r, zero), bd.icmp_signed("<", bs, zero)))
        if op == "//":
            v = bd.select(adjust, bd.sub(q, one), q)
            if not python_only:
                v = bd.select(ovf, imin, v)
        else:
            v = bd.select(adjust, bd.add(r, bs), r)
            if not python_only:
                v = bd.select(ovf, zero, v)
        if not python_only:
            v = bd.select(bz, zero, v)
        return Val(I64, v, d, npv=npv)

    def float_divmod(self, op, a, b, t, d, node, npv=False):
        """CPython's float // and % (floor semantics, sign of the divisor, ZeroDivisionError).

        For NumPy scalars a zero divisor gives a/b (//) or nan (%) instead of an error. npv=None: no check."""
        if self.dry:
            if npv is not True and npv is not None:
                self.spec.may_fail = True
            return Val(t, None, d)
        bd = self.f.b
        ft = lltype(t)
        z = ir.Constant(ft, 0.0)
        if npv is not False and npv is not None:    # NumPy, or decided at run time
            if npv is not True:
                self.py_check(npv, bd.fcmp_unordered("!=", b.v, z), E_ZERODIV, node)
            r = self.float_divmod(op, a, Val(t, bd.select(bd.fcmp_ordered("==", b.v, z), ir.Constant(ft, 1.0), b.v)),
                                  t, d, node, npv=None)
            special = bd.fdiv(a.v, b.v) if op == "//" else ir.Constant(ft, float("nan"))
            return Val(t, bd.select(bd.fcmp_ordered("==", b.v, z), special, r.v), d)
        if npv is not None:
            self.check(bd.fcmp_unordered("!=", b.v, z), E_ZERODIV, node)
        sfx = "f64" if t == F64 else "f32"
        # `frem` is C's fmod with IEEE semantics fixed by the IR (NaN in -> NaN out); a call to the libm function
        # lets the optimiser assume things about NaN arguments that are not true for Python/NumPy values
        mod = bd.frem(a.v, b.v)
        if op == "%":
            # CPython tests `if (mod)`: NaN counts as true, hence the unordered comparisons
            fix = bd.and_(bd.fcmp_unordered("!=", mod, z),
                          bd.xor(bd.fcmp_ordered("<", b.v, z), bd.fcmp_ordered("<", mod, z)))
            r = bd.select(fix, bd.fadd(mod, b.v), mod)
            cps = self.mc.decl(f"llvm.copysign.{sfx}", ft, [ft, ft])
            r = bd.select(bd.fcmp_ordered("==", r, z), bd.call(cps, [z, b.v]), r)
            return Val(t, r, d)
        div = bd.fdiv(bd.fsub(a.v, mod), b.v)
        fix = bd.and_(bd.fcmp_unordered("!=", mod, z),
                      bd.xor(bd.fcmp_ordered("<", b.v, z), bd.fcmp_ordered("<", mod, z)))
        div = bd.select(fix, bd.fsub(div, ir.Constant(ft, 1.0)), div)
        floor = self.mc.decl(f"llvm.floor.{sfx}", ft, [ft])
        fd = bd.call(floor, [div])
        fd = bd.select(bd.fcmp_ordered(">", bd.fsub(div, fd), ir.Constant(ft, 0.5)), bd.fadd(fd, ir.Constant(ft, 1.0)),
                       fd)
        cps = self.mc.decl(f"llvm.copysign.{sfx}", ft, [ft, ft])
        q = bd.select(bd.fcmp_unordered("!=", div, z), fd, bd.call(cps, [z, bd.fdiv(a.v, b.v)]))
        return Val(t, q, d)

    def power(self, a, b, d, node, npv=False):
        if a.t in (I64, BOOL) and b.t in (I64, BOOL):
            if b.const is not None and b.const < 0:
                return self.power(self.convert(a, F64, node), self.convert(b, F64, node), d, node, npv)
            return self.int_pow(self.as_int(a), self.as_int(b), node, d, npv)
        if npv is not True and not self.fast and not (b.const is not None and b.const >= 0) and \
                not (a.const is not None and a.const != 0):
            # Python: 0.0 ** (negative) is ZeroDivisionError (NumPy scalars give inf)
            if self.dry:
                self.spec.may_fail = True
            else:
                bd = self.f.b
                t0 = arith(a.t, b.t, a.weak, b.weak)
                t0 = F64 if t0 == I64 else t0
                av, bv = self.convert(a, t0, node).v, self.convert(b, t0, node).v
                bad = bd.and_(bd.fcmp_ordered("==", av, ir.Constant(lltype(t0), 0.0)),
                              bd.fcmp_ordered("<", bv, ir.Constant(lltype(t0), 0.0)))
                self.py_check(npv, bd.not_(bad), E_ZERODIV, node)
        t = arith(a.t, b.t, a.weak, b.weak)
        if t == I64:
            t = F64
        av = self.convert(a, t, node)
        if b.const is not None and not self.dry:
            c = b.const
            if c == 2:
                return self.fop("*", av, av, t, d)
            if c == 1:
                return Val(t, av.v, d)
            if self.fast and isinstance(c, (int, float)) and float(c).is_integer() and 0 <= c <= 16:
                return self.ipow_float(av, int(c), t, d)
            if c == 0.5 and self.fast:
                return Val(t, self.f.b.call(self.mc.decl(f"llvm.sqrt.{_sfx(t)}", lltype(t), [lltype(t)]), [av.v]), d)
        if self.dry:
            return Val(t, None, d)
        bv = self.convert(b, t, node)
        fn = self.mc.decl(f"llvm.pow.{_sfx(t)}", lltype(t), [lltype(t), lltype(t)])
        return Val(t, self.f.b.call(fn, [av.v, bv.v]), d)

    def ipow_float(self, x, n, t, d):
        result = None
        base = x.v
        while n:
            if n & 1:
                result = base if result is None else self._fmul(result, base)
            n >>= 1
            if n:
                base = self._fmul(base, base)
        if result is None:
            result = ir.Constant(lltype(t), 1.0)
        return Val(t, result, d)

    def _fmul(self, a, b):
        r = self.f.b.fmul(a, b)
        if self.fast:
            r.flags.append("fast")
        return r

    def int_pow(self, a, b, node, d, npv=False):
        if self.dry:
            self.spec.may_fail = True
            return Val(I64, None, d)
        if b.const is not None and 0 <= b.const <= 64:
            n = b.const
            result = self.const_val(1, d)
            base = a
            while n:
                if n & 1:
                    result = self.int_arith("*", result, base, node, d, npv) if result.const != 1 else base
                n >>= 1
                if n:
                    base = self.int_arith("*", base, base, node, d, npv)
            return Val(I64, result.v, d)
        bd = self.f.b
        self.check(bd.icmp_signed(">=", b.v, ir.Constant(i64t(), 0)), E_NEGEXP, node, b.v)
        pre = bd.block
        head = self.f.fn.append_basic_block("ipow")
        body = self.f.fn.append_basic_block("ipow.body")
        end = self.f.fn.append_basic_block("ipow.end")
        bd.branch(head)
        bd.position_at_end(head)
        acc = bd.phi(i64t())
        base = bd.phi(i64t())
        e = bd.phi(i64t())
        acc.add_incoming(ir.Constant(i64t(), 1), pre)
        base.add_incoming(a.v, pre)
        e.add_incoming(b.v, pre)
        bd.cbranch(bd.icmp_signed(">", e, ir.Constant(i64t(), 0)), body, end)
        bd.position_at_end(body)
        odd = bd.icmp_signed("!=", bd.and_(e, ir.Constant(i64t(), 1)), ir.Constant(i64t(), 0))
        m = bd.smul_with_overflow(acc, base)
        ovf1 = bd.and_(odd, bd.extract_value(m, 1))
        acc2 = bd.select(odd, bd.extract_value(m, 0), acc)
        e2 = bd.ashr(e, ir.Constant(i64t(), 1))
        sq = bd.smul_with_overflow(base, base)
        ovf2 = bd.and_(bd.icmp_signed(">", e2, ir.Constant(i64t(), 0)), bd.extract_value(sq, 1))
        self.py_check(npv, bd.not_(bd.or_(ovf1, ovf2)), E_OVERFLOW, node, a.v, b.v)
        cur = bd.block
        acc.add_incoming(acc2, cur)
        base.add_incoming(bd.extract_value(sq, 0), cur)
        e.add_incoming(e2, cur)
        bd.branch(head)
        bd.position_at_end(end)
        return Val(I64, acc, d)

    def compare(self, op, a, b, node):
        if is_arrish(a.t) or is_arrish(b.t):
            return self.lazy_bin(op, a, b, node)
        if a.d != b.d and ANY not in (a.t, b.t) and not (a.const == 0 or b.const == 0):
            raise UnitError(f"cannot compare {dim_name(a.d)} and {dim_name(b.d)}", node.line, node.col)
        if a.t[0] == "t" or b.t[0] == "t" or NONE in (a.t, b.t):
            raise Unsupported(f"comparing {tname(a.t)} and {tname(b.t)}", node)
        if self.dry:
            return Val(BOOL, None)
        bd = self.f.b
        t = arith(a.t, b.t, a.weak, b.weak)
        if t == I64:
            return Val(BOOL, bd.icmp_signed(op, self.as_int(a).v, self.as_int(b).v))
        if t == F32 and F64 in (a.t, b.t):
            t = F64
        av, bv = self.convert(a, t, node).v, self.convert(b, t, node).v
        if op == "!=":
            return Val(BOOL, bd.fcmp_unordered("!=", av, bv))
        return Val(BOOL, bd.fcmp_ordered(op, av, bv))

    def approx(self, a, b, node):
        """a ~= b: |a - b| <= max(1e-6 * max(|a|, |b|), 1e-12) (the same tolerance as the bytecode engine)."""
        if is_arrish(a.t) or is_arrish(b.t):
            raise Unsupported("~= on tensors in native code", node)
        if a.d != b.d and not (a.const == 0 or b.const == 0):
            raise UnitError(f"cannot compare {dim_name(a.d)} and {dim_name(b.d)}", node.line, node.col)
        if self.dry:
            return Val(BOOL, None)
        bd = self.f.b
        av, bv = self.convert(a, F64, node).v, self.convert(b, F64, node).v
        fabs = self.mc.decl("llvm.fabs.f64", ir.DoubleType(), [ir.DoubleType()])
        fmax = self.mc.decl("llvm.maxnum.f64", ir.DoubleType(), [ir.DoubleType(), ir.DoubleType()])
        diff = bd.call(fabs, [bd.fsub(av, bv)])
        scale = bd.call(fmax, [bd.call(fabs, [av]), bd.call(fabs, [bv])])
        tol = bd.call(fmax, [bd.fmul(scale, ir.Constant(ir.DoubleType(), 1e-6)), ir.Constant(ir.DoubleType(), 1e-12)])
        eq = bd.fcmp_ordered("==", av, bv)
        return Val(BOOL, bd.or_(eq, bd.fcmp_ordered("<=", diff, tol)))

    def minmax(self, op, a, b, node):
        """Python's min/max for two scalars: keeps the first unless the second is strictly better."""
        if a.d != b.d and ANY not in (a.t, b.t):
            raise UnitError(f"{op}() of {dim_name(a.d)} and {dim_name(b.d)}", node.line, node.col)
        t = arith(a.t, b.t, a.weak, b.weak)
        if self.dry:
            return Val(t, None, a.d, npv=self.np_pick(None, a.npv, b.npv))
        av, bv = self.convert(a, t, node), self.convert(b, t, node)
        better = self.compare("<" if op == "min" else ">", bv, av, node).v
        return Val(t, self.f.b.select(better, bv.v, av.v), a.d, npv=self.np_pick(better, a.npv, b.npv))

    # ------------------------------------------------------------------ calls
    def e_Call(self, e):
        f = e.func
        if isinstance(f, A.Name) and self.find(f.id) is None:
            res = self.r.lookup(f.id)
            if res is None:
                raise Unsupported(f"'{f.id}' is not defined", e)
            if res[0] == "builtin":
                return self.builtin(res[1], e)
            obj = res[1]
            info = apsi_function(obj)
            if info is not None:
                return self.call_apsi(info, e)
            raise Unsupported(f"calling '{f.id}' (a {type(obj).__name__}) from native code", e)
        if isinstance(f, A.Attr) and isinstance(f.obj, A.Name) and self.find(f.obj.id) is None:
            res = self.r.lookup(f.obj.id)
            if res is not None and res[0] == "value":
                try:
                    obj = self.r.attr(res[1], f.name)
                except Exception:
                    obj = None
                info = apsi_function(obj) if obj is not None else None
                if info is not None:
                    return self.call_apsi(info, e)
                name = f.name
                if name in NATIVE_BUILTINS and obj is not None and obj is self.r.builtin_obj(name):
                    return self.builtin(name, e)
        raise Unsupported("this kind of call", e)

    def call_apsi(self, info, e):
        fd, pyfn, free, opts = info
        if e.kwargs:
            raise Unsupported("keyword arguments in native calls", e)
        args = [self.materialize(self.expr(a), e) for a in e.args]
        if len(args) > len(fd.params):
            raise Unsupported(f"{fd.name}() takes {len(fd.params)} arguments", e)
        for p in fd.params[len(args):]:
            if p.default is None:
                raise Unsupported(f"{fd.name}() is missing the argument '{p.name}'", e)
            dv = self.const_default(p.default, e)
            args.append(dv)
        for a in args:
            if a.t not in SCALARS and not is_arr(a.t):
                raise Unsupported(f"passing {tname(a.t)} to {fd.name}()", e)
        for p, a in zip(fd.params, args):
            if not self.erased and isinstance(p.type, A.TName) and p.type.name in DIM_NAMES and a.d != DIM_NAMES[p.type.name] \
                    and a.t != ANY:
                raise UnitError(f"argument '{p.name}' of {fd.name}() should be {p.type.name}, got "
                                f"{dim_name(a.d)}", e.line, e.col)
        targs = [TV(a.t, a.d, npv=(a.npv if isinstance(a.npv, bool) else DYN) if a.t in SCALARS else False)
                 for a in args]
        if any(a.t == ANY for a in args):
            if self.dry:
                return Val(ANY, None)
        sub_opts = dict(self.r.options)
        sub_opts.update(opts)
        resolver = self.mc_resolver(pyfn, free, sub_opts)
        spec = self.mc.spec(fd, targs, resolver, e)
        self.spec.callees.add(spec)
        if spec.may_fail:
            self.spec.may_fail = True
        if spec.ret is None:
            if self.dry:
                return Val(ANY, None)
            raise Unsupported(f"could not infer what {fd.name}() returns", e)
        if self.dry:
            return Val(spec.ret.t, None, spec.ret.d, npv=spec.ret.npv)
        bd = self.f.b
        fn = self.mc.declare_fn(spec)
        flags = [a.npv for a, ta in zip(args, targs) if ta.npv is DYN]
        r = bd.call(fn, [a.v for a in args] + flags)
        if spec.may_fail:
            self.check_err_flag()
        if spec.ret.t == NONE:
            return Val(NONE, None)
        rnp = spec.ret.npv
        if rnp is DYN:
            r, rnp = bd.extract_value(r, 0), bd.extract_value(r, 1)
        for _, part in _array_parts(spec.ret.t, r, bd):
            self.adopt(bd.extract_value(part, 1))
        return Val(spec.ret.t, r, spec.ret.d, npv=rnp)

    def mc_resolver(self, pyfn, free, opts):
        cache = self.mc.__dict__.setdefault("_resolvers", {})
        key = (id(pyfn), tuple(sorted(opts.items())))
        r = cache.get(key)
        if r is None:
            r = cache[key] = Resolver(pyfn, free, opts)
        return r

    def const_default(self, d, node):
        if isinstance(d, A.Num):
            return self.e_Num(d)
        if isinstance(d, A.Bool):
            return self.e_Bool(d)
        if isinstance(d, A.Unary) and d.op == "-" and isinstance(d.operand, A.Num):
            return self.e_Unary(d)
        raise Unsupported("a default argument that is not a number", node)

    def builtin(self, name, e):
        args = e.args
        kw = dict(e.kwargs)
        bd = None if self.dry else self.f.b

        def one():
            if len(args) != 1 or kw:
                raise Unsupported(f"{name}() with {len(args)} arguments", e)
            return self.expr(args[0])
        unary = {"sqrt": "sqrt", "exp": "exp", "log": "log", "log10": "log10", "log2": "log2", "sin": "sin",
                 "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan", "sinh": "sinh",
                 "cosh": "cosh", "tanh": "tanh", "floor": "floor", "ceil": "ceil", "cbrt": "cbrt", "abs": "abs",
                 "relu": "relu", "sigmoid": "sigmoid"}
        if name in unary:
            v = one()
            if is_arrish(v.t):
                return self.lazy_un(name, v, e)
            return self.math1(name, v, e)
        if name in ("min", "max"):
            if kw:
                raise Unsupported(f"{name}() with keyword arguments", e)
            vals = [self.expr(a) for a in args]
            if len(vals) == 1:
                if is_arrish(vals[0].t):
                    return self.reduce(name, vals[0], e)
                raise Unsupported(f"{name}() of {tname(vals[0].t)}", e)
            if any(is_arrish(v.t) for v in vals):
                raise Unsupported(f"element-wise {name}() of tensors (use where())", e)
            cur = vals[0]
            for v in vals[1:]:
                cur = self.minmax(name, cur, v, e)
            return cur
        if name in ("sum", "mean", "prod", "argmin", "argmax"):
            if kw or len(args) != 1:
                raise Unsupported(f"{name}() with axis or several arguments", e)
            v = self.expr(args[0])
            if not is_arrish(v.t):
                raise Unsupported(f"{name}() of {tname(v.t)}", e)
            return self.reduce(name, v, e)
        if name == "norm":
            v = one()
            if not is_arrish(v.t):
                return self.math1("abs", v, e)
            sq = self.lazy_bin("*", v, v, e)
            s = self.reduce("sum", sq, e)
            r = self.math1("sqrt", self.convert(s, F64 if s.t != F32 else F32, e), e)
            return Val(r.t, r.v, v.d)
        if name == "dot":
            if len(args) != 2 or kw:
                raise Unsupported("dot() needs two arguments", e)
            a, b = self.expr(args[0]), self.expr(args[1])
            if not (is_arrish(a.t) and is_arrish(b.t) and a.t[2] == 1 and b.t[2] == 1):
                raise Unsupported("dot() of anything but two vectors", e)
            return self.reduce("sum", self.lazy_bin("*", a, b, e, same_shape=True), e)
        if name == "len":
            v = one()
            if is_arrish(v.t):
                v = self.materialize(v, e)
                return self.dim_size(v, 0)
            if v.t[0] == "t":
                return self.const_val(len(v.t[1]))
            raise Unsupported(f"len() of {tname(v.t)}", e)
        if name == "shape":
            v = self.materialize(one(), e)
            if not is_arr(v.t):
                raise Unsupported("shape() of a non-tensor", e)
            return self.shape_tuple(v)
        if name == "ndim":
            v = one()
            if not is_arrish(v.t):
                raise Unsupported("ndim() of a non-tensor", e)
            return self.const_val(v.t[2])
        if name == "float":
            v = one()
            if v.t not in SCALARS:
                raise Unsupported(f"float() of {tname(v.t)}", e)
            return self.convert(v, F64, e)
        if name == "int":
            v = one()
            if v.t in (I64, BOOL):
                return self.as_int(v)
            if v.t in (F64, F32):
                if self.dry:
                    self.spec.may_fail = True
                    return Val(I64, None, v.d)
                ft = lltype(v.t)
                fv = v.v
                lim = ir.Constant(ft, 9.223372036854775807e18)
                ok = bd.and_(bd.fcmp_ordered("<", fv, lim), bd.fcmp_ordered(">=", fv, ir.Constant(ft, -lim.constant)))
                self.check(ok, E_CONVERT, e)
                return Val(I64, bd.fptosi(fv, i64t()), v.d)
            raise Unsupported(f"int() of {tname(v.t)}", e)
        if name == "bool":
            v = one()
            return Val(BOOL, self.truth(v, e), DIMLESS)
        if name == "round":
            if len(args) != 1 or kw:
                raise Unsupported("round() with digits", e)
            v = self.expr(args[0])
            if v.t in (I64, BOOL):
                return self.as_int(v)
            if v.t not in (F64, F32):
                raise Unsupported(f"round() of {tname(v.t)}", e)
            if self.dry:
                self.spec.may_fail = True
                return Val(I64, None, v.d)
            ft = lltype(v.t)
            rv = bd.call(self.mc.decl(f"llvm.roundeven.{_sfx(v.t)}", ft, [ft]), [v.v])
            lim = ir.Constant(ft, 9.223372036854775807e18)
            ok = bd.and_(bd.fcmp_ordered("<", rv, lim), bd.fcmp_ordered(">=", rv, ir.Constant(ft, -lim.constant)))
            self.check(ok, E_CONVERT, e)
            return Val(I64, bd.fptosi(rv, i64t()), v.d)
        if name == "pow":
            if len(args) != 2 or kw:
                raise Unsupported("pow() with a modulus", e)
            return self.binop("**", self.expr(args[0]), self.expr(args[1]), e)
        if name == "atan2":
            if len(args) != 2 or kw:
                raise Unsupported("atan2() needs two arguments", e)
            y, x = self.expr(args[0]), self.expr(args[1])
            if y.d != x.d:
                raise UnitError(f"atan2() of {dim_name(y.d)} and {dim_name(x.d)}", e.line, e.col)
            if is_arrish(y.t) or is_arrish(x.t):
                raise Unsupported("atan2() of tensors in native code", e)
            t = F64 if arith(y.t, x.t) != F32 else F32
            if self.dry:
                return Val(t, None)
            ft = lltype(t)
            fn = self.mc.decl("atan2" if t == F64 else "atan2f", ft, [ft, ft])
            return Val(t, bd.call(fn, [self.convert(y, t, e).v, self.convert(x, t, e).v]))
        if name == "magnitude":
            v = one()
            return Val(v.t, v.v, DIMLESS)
        if name == "clip":
            if len(args) != 3 or kw:
                raise Unsupported("clip() needs three arguments", e)
            x, lo, hi = (self.expr(a) for a in args)
            if is_arrish(x.t):
                return self.lazy_clip(x, lo, hi, e)
            return self.minmax("min", self.minmax("max", x, lo, e), hi, e)
        if name == "where":
            if len(args) != 3 or kw:
                raise Unsupported("where() needs three arguments", e)
            c, a, b = (self.expr(x) for x in args)
            if any(is_arrish(v.t) for v in (c, a, b)):
                return self.lazy_where(c, a, b, e)
            if a.d != b.d:
                raise UnitError(f"where() of {dim_name(a.d)} and {dim_name(b.d)}", e.line, e.col)
            t = arith(a.t, b.t, a.weak, b.weak)
            if self.dry:
                return Val(t, None, a.d)
            return Val(t, bd.select(self.truth(c, e), self.convert(a, t, e).v, self.convert(b, t, e).v), a.d)
        if name in ("zeros", "ones", "full", "zeros_like", "ones_like"):
            return self.alloc_builtin(name, e)
        if name == "print":
            raise Unsupported("print() used as a value", e)
        raise Unsupported(f"the built-in '{name}()'", e)

    def math1(self, name, v, e, elementwise=False):
        if v.t not in SCALARS:
            raise Unsupported(f"{name}() of {tname(v.t)}", e)
        if name == "abs":
            if v.t in (I64, BOOL):
                v = self.as_int(v)
                if self.dry:
                    self.spec.may_fail = True
                    return Val(I64, None, v.d, npv=v.npv)
                bd = self.f.b
                if not self.fast:
                    self.py_check(v.npv, bd.icmp_signed("!=", v.v, ir.Constant(i64t(), -(1 << 63))), E_OVERFLOW, e, v.v)
                neg = bd.icmp_signed("<", v.v, ir.Constant(i64t(), 0))
                return Val(I64, bd.select(neg, bd.sub(ir.Constant(i64t(), 0), v.v), v.v), v.d, npv=v.npv)
            if self.dry:
                return Val(v.t, None, v.d, npv=v.npv)
            ft = lltype(v.t)
            return Val(v.t, self.f.b.call(self.mc.decl(f"llvm.fabs.{_sfx(v.t)}", ft, [ft]), [v.v]), v.d, npv=v.npv)
        if name == "relu":
            return self.minmax("max", v, self.const_val(0.0 if v.t != I64 else 0, v.d, weak=True), e)
        d = DIMLESS
        if name == "sqrt":
            d = v.d ** Fraction(1, 2)
        elif name == "cbrt":
            d = v.d ** Fraction(1, 3)
        elif name in ("floor", "ceil"):
            d = v.d
        elif not v.d.dimensionless:
            raise UnitError(f"{name}() needs a dimensionless argument, got {dim_name(v.d)}", e.line, e.col)
        t = F32 if v.t == F32 else F64
        if name in ("floor", "ceil") and v.t in (I64, BOOL):
            return self.as_int(v)
        if name in ("floor", "ceil") and not elementwise:
            # Apsi's floor/ceil of a number return an Int (math.floor); element-wise on tensors they stay Float
            if self.dry:
                self.spec.may_fail = True
                return Val(I64, None, d)
            bd = self.f.b
            ft = lltype(t)
            fv = bd.call(self.mc.decl(f"llvm.{name}.{_sfx(t)}", ft, [ft]), [self.convert(v, t, e).v])
            lim = ir.Constant(ft, 9.223372036854775807e18)
            ok = bd.and_(bd.fcmp_ordered("<", fv, lim), bd.fcmp_ordered(">=", fv, ir.Constant(ft, -lim.constant)))
            self.check(ok, E_CONVERT, e)
            return Val(I64, bd.fptosi(fv, i64t()), d)
        if self.dry:
            return Val(t, None, d)
        x = self.convert(v, t, e).v
        bd = self.f.b
        ft = lltype(t)
        if name == "sigmoid":
            ex = self.math1("exp", Val(t, bd.fneg(x)), e).v
            return Val(t, bd.fdiv(ir.Constant(ft, 1.0), bd.fadd(ir.Constant(ft, 1.0), ex)), d)
        intr = {"sqrt", "exp", "log", "log10", "log2", "sin", "cos", "floor", "ceil"}
        if name in intr:
            fn = self.mc.decl(f"llvm.{name}.{_sfx(t)}", ft, [ft])
        else:
            fn = self.mc.decl(name if t == F64 else name + "f", ft, [ft])
        r = bd.call(fn, [x])
        if name in ("floor", "ceil"):                 # Apsi's floor/ceil of a float return a float
            return Val(t, r, d)
        return Val(t, r, d)

    # ------------------------------------------------------------------ allocation
    def alloc_builtin(self, name, e):
        if name in ("zeros_like", "ones_like"):
            if len(e.args) != 1 or e.kwargs:
                raise Unsupported(f"{name}() with options", e)
            src = self.materialize(self.expr(e.args[0]), e)
            if not is_arr(src.t):
                raise Unsupported(f"{name}() of {tname(src.t)}", e)
            shape = [] if self.dry else [self.f.b.extract_value(src.v, [2, k]) for k in range(src.t[2])]
            if self.dry:
                shape = [None] * src.t[2]
            out = self.alloc_array("f8", shape, e)
            out.d = src.d
            if name == "ones_like":
                self.fill(out, self.const_val(1.0), e)
            return out
        kw = dict(e.kwargs)
        elem = "f8"
        if "dtype" in kw:
            dt = kw.pop("dtype")
            dn = dt.id if isinstance(dt, A.Name) else (dt.value if isinstance(dt, A.Str) else None)
            elem = {"f64": "f8", "fp64": "f8", "float": "f8", "f32": "f4", "fp32": "f4", "f16": "f2", "fp16": "f2",
                    "bf16": "h2", "bfloat16": "h2", "fp8_e4m3": "e4", "fp8": "e4", "fp8_e5m2": "e5",
                    "i64": "i8",
                    "int": "i8", "i32": "i4", "bool": "b1", "uint8": "u1", "int8": "i1"}.get(dn)
            if elem is None:
                raise Unsupported(f"{name}() with dtype {dn}", e)
        if kw:
            raise Unsupported(f"{name}() with options", e)
        fill = None
        args = list(e.args)
        if name == "full":
            if len(args) != 2:
                raise Unsupported("full() needs a shape and a value", e)
            fill = self.expr(args[1])
            if fill.t not in SCALARS:
                raise Unsupported("full() with a non-number fill value", e)
            args = [args[0]]
        dims = []
        for a in args:
            v = self.expr(a)
            if v.t[0] == "t":
                if self.dry:
                    dims += [Val(I64, None)] * len(v.t[1])
                else:
                    dims += [Val(x, self.f.b.extract_value(v.v, k)) for k, x in enumerate(v.t[1])]
            elif v.t in (I64, BOOL):
                dims.append(self.as_int(v))
            else:
                raise Unsupported(f"{name}() with a size of type {tname(v.t)}", e)
        if not dims:
            raise Unsupported(f"{name}() with no size", e)
        out = self.alloc_array(elem, [d.v for d in dims], e)
        if name == "ones":
            self.fill(out, self.const_val(1), e)
        if fill is not None:
            out.d = fill.d
            self.fill(out, fill, e)
        return out

    def fill(self, arr, v, e):
        if self.dry:
            return
        self.write_into(arr, v, e)

    # ------------------------------------------------------------------ tensor expressions
    def leaf(self, v, node):
        if isinstance(v.v, Lazy) or is_lazy(v.t):
            return v.v
        if is_arr(v.t):
            return Lazy("leaf", v.t[1], v.t[2], val=v, scalar=ELEM_SCALAR[v.t[1]])
        if v.t in SCALARS:
            return Lazy("scalar", None, 0, val=v, scalar=v.t)
        raise Unsupported(f"{tname(v.t)} in a tensor expression", node)

    def lazy_bin(self, op, a, b, node, same_shape=False):
        if op in ("//", "%", "**") and (is_arrish(a.t) or is_arrish(b.t)):
            if op == "**" and b.t in SCALARS and b.const == 2:
                return self.lazy_bin("*", a, a, node)
            raise Unsupported(f"'{op}' on tensors in native code", node)
        la, lb = self.leaf(a, node), self.leaf(b, node)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            if a.d != b.d and not (a.const == 0 or b.const == 0):
                raise UnitError(f"cannot compare {dim_name(a.d)} and {dim_name(b.d)}", node.line, node.col)
            st, d = BOOL, DIMLESS
        else:
            d = self.result_dim(op, a, b, node)
            st = arith(la.scalar, lb.scalar, a.weak, b.weak)
            if op == "/" and st == I64:
                st = F64
        nd = max(la.ndim, lb.ndim)
        lz = Lazy("bin", SCALAR_ELEM[st], nd, op=op, kids=[la, lb], scalar=st, node=node)
        return Val(("x", SCALAR_ELEM[st], nd), lz, d)

    def lazy_un(self, name, v, node):
        lv = self.leaf(v, node)
        d = v.d
        if name == "sqrt":
            d = v.d ** Fraction(1, 2)
        elif name == "cbrt":
            d = v.d ** Fraction(1, 3)
        elif name not in ("neg", "abs", "floor", "ceil", "relu") and not v.d.dimensionless:
            raise UnitError(f"{name}() needs a dimensionless argument, got {dim_name(v.d)}", node.line, node.col)
        st = lv.scalar
        if name not in ("neg", "abs", "relu", "floor", "ceil") or st == BOOL:
            st = F32 if st == F32 else (F64 if name not in ("neg", "abs", "relu") else I64)
        if st == BOOL:
            st = I64
        lz = Lazy("un", SCALAR_ELEM[st], lv.ndim, fn=name, kids=[lv], scalar=st, node=node)
        return Val(("x", SCALAR_ELEM[st], lv.ndim), lz, d)

    def lazy_where(self, c, a, b, node):
        if a.d != b.d:
            raise UnitError(f"where() of {dim_name(a.d)} and {dim_name(b.d)}", node.line, node.col)
        lc, la, lb = self.leaf(c, node), self.leaf(a, node), self.leaf(b, node)
        st = arith(la.scalar, lb.scalar, a.weak, b.weak)
        nd = max(lc.ndim, la.ndim, lb.ndim)
        lz = Lazy("where", SCALAR_ELEM[st], nd, kids=[lc, la, lb], scalar=st, node=node)
        return Val(("x", SCALAR_ELEM[st], nd), lz, a.d)

    def lazy_clip(self, x, lo, hi, node):
        lx, ll, lh = self.leaf(x, node), self.leaf(lo, node), self.leaf(hi, node)
        if not (x.d == lo.d == hi.d):
            raise UnitError("clip() bounds must have the same unit as the value", node.line, node.col)
        st = arith(arith(lx.scalar, ll.scalar, x.weak, lo.weak), lh.scalar, False, hi.weak)
        lz = Lazy("clip", SCALAR_ELEM[st], lx.ndim, kids=[lx, ll, lh], scalar=st, node=node)
        return Val(("x", SCALAR_ELEM[st], lx.ndim), lz, x.d)

    def lazy_elem(self, lz, get_leaf):
        """Scalar value of the expression at the current position (get_leaf(leaf) -> scalar Val)."""
        if lz.kind == "leaf":
            return get_leaf(lz)
        if lz.kind == "scalar":
            return lz.val
        kids = [self.lazy_elem(k, get_leaf) for k in lz.kids]
        node = lz.node
        if lz.kind == "bin":
            a, b = kids
            a = Val(a.t, a.v, DIMLESS, a.const, a.weak)
            b = Val(b.t, b.v, DIMLESS, b.const, b.weak)
            return self.binop(lz.op, a, b, node)
        if lz.kind == "un":
            (a,) = kids
            a = Val(a.t, a.v, DIMLESS)
            if lz.fn == "neg":
                if a.t in (I64, BOOL):
                    return self.int_arith("-", self.const_val(0), self.as_int(a), node)
                return Val(a.t, self.f.b.fneg(a.v))
            return self.math1(lz.fn, a, node, elementwise=True)
        if lz.kind == "where":
            c, a, b = kids
            t = lz.scalar
            return Val(t, self.f.b.select(self.truth(c, node), self.convert(a, t, node).v,
                                          self.convert(b, t, node).v))
        if lz.kind == "clip":
            x, lo, hi = [Val(k.t, k.v) for k in kids]
            return self.minmax("min", self.minmax("max", x, lo, node), hi, node)
        raise Unsupported("tensor expression")

    def broadcast(self, leaves, nd, node, target_shape=None):
        """Output shape and, per leaf, effective strides (0 on broadcast axes). Checks compatibility."""
        bd = self.f.b
        one = ir.Constant(i64t(), 1)
        out = [one] * nd if target_shape is None else list(target_shape)
        info = []
        for lf in leaves:
            st = lf.val.v
            lnd = lf.ndim
            shp = [bd.extract_value(st, [2, k]) for k in range(lnd)]
            strd = [bd.extract_value(st, [3, k]) for k in range(lnd)]
            if lf.val.t[3]:
                strd[-1] = one
            info.append((lf, shp, strd))
            if target_shape is None:
                for k in range(lnd):
                    j = nd - lnd + k
                    out[j] = bd.select(bd.icmp_signed("!=", shp[k], one), shp[k], out[j])
        eff = {}
        for lf, shp, strd in info:
            es = [ir.Constant(i64t(), 0)] * nd
            for k in range(lf.ndim):
                j = nd - lf.ndim + k
                if not self.fast or target_shape is not None:
                    good = bd.or_(bd.icmp_signed("==", shp[k], out[j]), bd.icmp_signed("==", shp[k], one))
                    self.check(good, E_SHAPE, node, shp[k], out[j])
                es[j] = bd.select(bd.icmp_signed("==", shp[k], out[j]), strd[k], ir.Constant(i64t(), 0))
            eff[id(lf)] = (bd.extract_value(lf.val.v, 0), es, shp)
        return out, eff

    def flat_ok(self, leaves, out, node):
        """i1: every leaf has exactly the output shape and is contiguous (so one flat loop covers it)."""
        bd = self.f.b
        ok = ir.Constant(i1t(), 1)
        for lf in leaves:
            if not lf.val.t[3] or lf.ndim != len(out):
                return None
            for k in range(lf.ndim):
                ok = bd.and_(ok, bd.icmp_signed("==", bd.extract_value(lf.val.v, [2, k]), out[k]))
        return ok

    def materialize(self, v, node):
        if not is_lazy(v.t):
            return v
        lz = v.v
        if self.dry:
            return Val(ARR(lz.elem, lz.ndim, True), None, v.d)
        leaves = lz.leaves()
        out_shape, _ = self.broadcast(leaves, lz.ndim, node)
        out = self.alloc_array(lz.elem, out_shape, node, zero=False)
        out.d = v.d
        self.write_into(out, Val(v.t, lz, v.d), node)
        return out

    def write_into(self, target, v, node):
        """target[...] = v (scalar or tensor expression), broadcasting v to target's shape."""
        if self.dry:
            return
        bd = self.f.b
        lz = self.leaf(v, node)
        tnd = target.t[2]
        if lz.ndim > tnd:
            raise Unsupported("assigning a higher-rank tensor into a lower-rank one", node)
        tshape = [bd.extract_value(target.v, [2, k]) for k in range(tnd)]
        tstr = [bd.extract_value(target.v, [3, k]) for k in range(tnd)]
        if target.t[3]:
            tstr[-1] = ir.Constant(i64t(), 1)
        leaves = lz.leaves()
        _, eff = self.broadcast(leaves, tnd, node, target_shape=tshape)
        tptr = bd.extract_value(target.v, 0)
        elem = target.t[1]
        total = ir.Constant(i64t(), 1)
        for s_ in tshape:
            total = bd.mul(total, s_)
        flat = self.flat_ok(leaves, tshape, node) if target.t[3] else None

        def flat_body(k):
            def get_leaf(lf):
                p, _, _ = eff[id(lf)]
                return self.from_elem(bd.load(bd.gep(p, [k], inbounds=True)), lf.elem, DIMLESS)
            val = self.lazy_elem(lz, get_leaf)
            bd.store(self.to_elem(val, elem, node), bd.gep(tptr, [k], inbounds=True))

        def nested_body(idx):
            def get_leaf(lf):
                p, es, _ = eff[id(lf)]
                off = ir.Constant(i64t(), 0)
                for j in range(tnd):
                    off = bd.add(off, bd.mul(idx[j], es[j]))
                return self.from_elem(bd.load(bd.gep(p, [off], inbounds=True)), lf.elem, DIMLESS)
            val = self.lazy_elem(lz, get_leaf)
            off = ir.Constant(i64t(), 0)
            for j in range(tnd):
                off = bd.add(off, bd.mul(idx[j], tstr[j]))
            bd.store(self.to_elem(val, elem, node), bd.gep(tptr, [off], inbounds=True))

        if flat is not None:
            fb = self.f.fn.append_basic_block("fuse.flat")
            nb = self.f.fn.append_basic_block("fuse.nd")
            jb = self.f.fn.append_basic_block("fuse.done")
            bd.cbranch(flat, fb, nb)
            bd.position_at_end(fb)
            if self.can_split():
                pb = self.f.fn.append_basic_block("fuse.par")
                sb = self.f.fn.append_basic_block("fuse.serial")
                bd.cbranch(bd.icmp_signed(">=", total, ir.Constant(i64t(), self.PAR_MIN)), pb, sb)
                bd.position_at_end(pb)

                def par_elem(k, get_leaf, ex):
                    val = self.lazy_elem(lz, get_leaf)
                    self.f.b.store(self.to_elem(val, elem, node), self.f.b.gep(ex[0], [k], inbounds=True))
                self.flat_parallel(total, lz, eff, leaves, par_elem, extra=(tptr,))
                bd.branch(jb)
                bd.position_at_end(sb)
            self.simple_loop(ir.Constant(i64t(), 0), total, flat_body)
            bd.branch(jb)
            bd.position_at_end(nb)
            self.nested_loops(tshape, nested_body)
            bd.branch(jb)
            bd.position_at_end(jb)
        else:
            self.nested_loops(tshape, nested_body)

    PAR_MIN = 1 << 17           # elements: below this a fused kernel runs on one core (thread wake-up costs more)

    def can_split(self):
        return not self.dry and not self.mc.gpu and not self.f.in_parallel

    def flat_parallel(self, total, lz, eff, leaves, elem, extra=(), red=None):
        """for k in [0, total): elem(k, get_leaf, extra) on all cores, in the fixed 256 chunks of `parallel for`.

        red = (type, identity, combine(bd, a, b)): elem returns the value to accumulate; the per-chunk partials are
        combined in chunk order (deterministic) and the result is returned."""
        self.mc.uses_parallel = True
        bd = self.f.b
        scal = [x for x in _lazy_walk(lz) if x.kind == "scalar"]
        caps = [eff[id(lf)][0] for lf in leaves] + [x.val.v for x in scal] + list(extra)
        nch = bd.select(bd.icmp_signed("<", total, ir.Constant(i64t(), NCHUNKS)), total,
                        ir.Constant(i64t(), NCHUNKS))
        parts = None
        types = [c.type for c in caps]
        if red is not None:
            parts = self._entry_alloca(ir.ArrayType(red[0], NCHUNKS), "fpart")
            types.append(parts.type)
        ctx_t = ir.LiteralStructType(types or [i64t()])
        ctx = self._entry_alloca(ctx_t, "fctx")
        i32 = ir.IntType(32)
        for j, c in enumerate(caps + ([parts] if parts is not None else [])):
            bd.store(c, bd.gep(ctx, [ir.Constant(i32, 0), ir.Constant(i32, j)]))
        chunk = ir.Function(self.mc.module, ir.FunctionType(ir.VoidType(), [i8p(), i64t(), i64t(), i64t()]),
                            name=self.mc.fresh(self.spec.name + ".kernel"))
        chunk.linkage = "internal"
        chunk.attributes.add("nounwind")
        saved_f, saved_vals = self.f, [x.val for x in scal]
        nf = new_frame(chunk)
        nf.in_parallel = True
        self.f = nf
        try:
            cb = nf.b
            cctx = cb.bitcast(chunk.args[0], ctx_t.as_pointer())
            got = [cb.load(cb.gep(cctx, [ir.Constant(i32, 0), ir.Constant(i32, j)])) for j in range(len(types))]
            n_l = len(leaves)
            local_eff = {id(lf): (got[j], None, None) for j, lf in enumerate(leaves)}
            for j, x in enumerate(scal):
                old = x.val
                # inside a tensor kernel every value combines with a tensor element, so it is NumPy anyway
                x.val = Val(old.t, got[n_l + j], old.d, old.const, old.weak,
                            old.npv if isinstance(old.npv, bool) else True)
            ex = got[n_l + len(scal):n_l + len(scal) + len(extra)]
            acc = None
            if red is not None:
                acc = nf.entry_b.alloca(red[0], name="facc")
                nf.entry_b.store(red[1], acc)

            def get_leaf(lf, k):
                return self.from_elem(cb.load(cb.gep(local_eff[id(lf)][0], [k], inbounds=True)), lf.elem, DIMLESS)

            def body(k):
                v = elem(k, lambda lf: get_leaf(lf, k), ex)
                if red is not None:
                    cb.store(red[2](cb, cb.load(acc), v), acc)
            self.simple_loop(chunk.args[1], chunk.args[2], body)
            if red is not None:
                cb.store(cb.load(acc), cb.gep(got[-1], [ir.Constant(i64t(), 0), chunk.args[3]]))
            cb.ret_void()
        finally:
            self.f = saved_f
            for x, v in zip(scal, saved_vals):
                x.val = v
        bd = self.f.b
        disp = self.mc.decl("apsi_parallel", ir.VoidType(), [i8p(), i8p(), i64t(), i64t(), i64t()])
        bd.call(disp, [bd.bitcast(chunk, i8p()), bd.bitcast(ctx, i8p()), ir.Constant(i64t(), 0), total, nch])
        self.check_err_flag()
        if red is None:
            return None
        res = self._entry_alloca(red[0], "fres")
        bd.store(red[1], res)
        self.simple_loop(ir.Constant(i64t(), 0), nch, lambda c: bd.store(
            red[2](bd, bd.load(res), bd.load(bd.gep(parts, [ir.Constant(i64t(), 0), c]))), res))
        return bd.load(res)

    def simple_loop(self, lo, hi, body):
        bd = self.f.b
        pre = bd.block
        head = self.f.fn.append_basic_block("loop")
        bb = self.f.fn.append_basic_block("loop.body")
        end = self.f.fn.append_basic_block("loop.end")
        bd.branch(head)
        bd.position_at_end(head)
        i = bd.phi(i64t())
        i.add_incoming(lo, pre)
        bd.cbranch(bd.icmp_signed("<", i, hi), bb, end)
        bd.position_at_end(bb)
        body(i)
        nxt = bd.add(i, ir.Constant(i64t(), 1))
        i.add_incoming(nxt, bd.block)
        bd.branch(head)
        bd.position_at_end(end)

    def nested_loops(self, shape, body, idx=()):
        if len(idx) == len(shape):
            body(list(idx))
            return
        self.simple_loop(ir.Constant(i64t(), 0), shape[len(idx)],
                         lambda i: self.nested_loops(shape, body, idx + (i,)))

    def reduce(self, name, v, node):
        lz = self.leaf(v, node)
        st = lz.scalar
        if name in ("argmin", "argmax"):
            rt, d = I64, DIMLESS
        elif name == "mean":
            rt, d = (F32 if st == F32 else F64), v.d
        elif name == "prod":
            rt, d = (I64 if st in (I64, BOOL) else st), v.d ** lz.ndim if False else v.d
            if not v.d.dimensionless:
                raise Unsupported("prod() of a tensor with a unit", node)
        else:
            rt, d = (I64 if st in (I64, BOOL) else st), v.d
        if self.dry:
            if name in ("min", "max", "argmin", "argmax"):
                self.spec.may_fail = True
            return Val(rt, None, d, npv=True)
        bd = self.f.b
        if lz.kind != "leaf" or True:
            leaves = lz.leaves()
            shape, eff = self.broadcast(leaves, lz.ndim, node)
        total = ir.Constant(i64t(), 1)
        for s_ in shape:
            total = bd.mul(total, s_)
        acc = self._entry_alloca(lltype(rt), "acc")
        pos = self._entry_alloca(i64t(), "argpos")
        counter = self._entry_alloca(i64t(), "k")
        bd.store(ir.Constant(i64t(), 0), counter)
        bd.store(ir.Constant(i64t(), 0), pos)
        if name in ("sum", "mean"):
            bd.store(ir.Constant(lltype(rt), 0 if rt == I64 else 0.0), acc)
        elif name == "prod":
            bd.store(ir.Constant(lltype(rt), 1 if rt == I64 else 1.0), acc)
        else:
            self.check(bd.icmp_signed(">", total, ir.Constant(i64t(), 0)), E_EMPTY, node)
        nd = lz.ndim

        def visit(val):
            val = self.convert(Val(val.t, val.v), rt, node) if name not in ("argmin", "argmax") else val
            k = bd.load(counter)
            if name in ("sum", "mean"):
                cur = Val(rt, bd.load(acc))
                nv = self.binop("+", cur, val, node) if rt == I64 else self.fop("+", cur, val, rt, DIMLESS)
                bd.store(nv.v, acc)
            elif name == "prod":
                cur = Val(rt, bd.load(acc))
                nv = self.binop("*", cur, val, node) if rt == I64 else self.fop("*", cur, val, rt, DIMLESS)
                bd.store(nv.v, acc)
            else:
                first = bd.icmp_signed("==", k, ir.Constant(i64t(), 0))
                if name in ("argmin", "argmax"):
                    ct = st if st != BOOL else I64
                    val = self.convert(val, ct, node)
                    best_ptr = self._acc2(ct)
                    cur = Val(ct, bd.load(best_ptr))
                    better = self.compare("<" if name == "argmin" else ">", val, cur, node).v
                    if ct in (F64, F32):              # NumPy: the first NaN wins
                        isnan = bd.fcmp_unordered("uno", val.v, val.v)
                        curnan = bd.fcmp_unordered("uno", cur.v, cur.v)
                        better = bd.or_(bd.and_(isnan, bd.not_(curnan)), bd.and_(better, bd.not_(curnan)))
                    take = bd.or_(first, better)
                    bd.store(bd.select(take, val.v, cur.v), best_ptr)
                    bd.store(bd.select(take, k, bd.load(pos)), pos)
                else:
                    cur = Val(rt, bd.load(acc))
                    if rt in (F64, F32):              # NumPy: NaN propagates
                        fn = self.mc.decl(f"llvm.{'minimum' if name == 'min' else 'maximum'}.{_sfx(rt)}",
                                          lltype(rt), [lltype(rt), lltype(rt)])
                        nv = bd.call(fn, [cur.v, val.v])
                    else:
                        nv = bd.select(bd.icmp_signed("<" if name == "min" else ">", val.v, cur.v), val.v, cur.v)
                    bd.store(bd.select(first, val.v, nv), acc)
            bd.store(bd.add(k, ir.Constant(i64t(), 1)), counter)

        flat = self.flat_ok(leaves, shape, node)

        def flat_body(k):
            def get_leaf(lf):
                p, _, _ = eff[id(lf)]
                return self.from_elem(bd.load(bd.gep(p, [k], inbounds=True)), lf.elem, DIMLESS)
            visit(self.lazy_elem(lz, get_leaf))

        def nested_body(idx):
            def get_leaf(lf):
                p, es, _ = eff[id(lf)]
                off = ir.Constant(i64t(), 0)
                for j in range(nd):
                    off = bd.add(off, bd.mul(idx[j], es[j]))
                return self.from_elem(bd.load(bd.gep(p, [off], inbounds=True)), lf.elem, DIMLESS)
            visit(self.lazy_elem(lz, get_leaf))
        if flat is not None:
            fb = self.f.fn.append_basic_block("red.flat")
            nb = self.f.fn.append_basic_block("red.nd")
            jb = self.f.fn.append_basic_block("red.done")
            bd.cbranch(flat, fb, nb)
            bd.position_at_end(fb)
            red = self._par_reduction(name, rt) if self.can_split() else None
            if red is not None:
                pb = self.f.fn.append_basic_block("red.par")
                sb = self.f.fn.append_basic_block("red.serial")
                bd.cbranch(bd.icmp_signed(">=", total, ir.Constant(i64t(), self.PAR_MIN)), pb, sb)
                bd.position_at_end(pb)

                def par_elem(k, get_leaf, ex):
                    v = self.lazy_elem(lz, get_leaf)
                    return self.convert(Val(v.t, v.v), rt, node).v
                r = self.flat_parallel(total, lz, eff, leaves, par_elem, red=red)
                bd.store(r, acc)
                bd.branch(jb)
                bd.position_at_end(sb)
            self.simple_loop(ir.Constant(i64t(), 0), total, flat_body)
            bd.branch(jb)
            bd.position_at_end(nb)
            self.nested_loops(shape, nested_body)
            bd.branch(jb)
            bd.position_at_end(jb)
        else:
            self.nested_loops(shape, nested_body)
        if name in ("argmin", "argmax"):
            return Val(I64, bd.load(pos), DIMLESS, npv=True)
        res = bd.load(acc)
        if name == "mean":
            n = bd.sitofp(total, lltype(rt))
            return Val(rt, bd.fdiv(res, n), d, npv=True)
        return Val(rt, res, d, npv=True)

    def _par_reduction(self, name, rt):
        """(type, identity, combine) for reductions that can be split across cores, else None."""
        if name not in ("sum", "mean", "prod", "min", "max") or rt not in (F64, F32, I64):
            return None
        lt = lltype(rt)
        fl = rt != I64
        fast = self.fast
        if name in ("sum", "mean"):
            def comb(b, x, y):
                r = b.fadd(x, y) if fl else b.add(x, y)
                if fl and fast:
                    r.flags.append("fast")
                return r
            return lt, ir.Constant(lt, 0.0 if fl else 0), comb
        if name == "prod":
            return lt, ir.Constant(lt, 1.0 if fl else 1), (lambda b, x, y: b.fmul(x, y) if fl else b.mul(x, y))
        if fl:
            fn = self.mc.decl(f"llvm.{'minimum' if name == 'min' else 'maximum'}.{_sfx(rt)}", lt, [lt, lt])
            ident = ir.Constant(lt, math.inf if name == "min" else -math.inf)
            return lt, ident, (lambda b, x, y: b.call(fn, [x, y]))
        ident = ir.Constant(lt, (1 << 63) - 1 if name == "min" else -(1 << 63))
        op = "<" if name == "min" else ">"
        return lt, ident, (lambda b, x, y: b.select(b.icmp_signed(op, y, x), y, x))

    def _acc2(self, t):
        return self._entry_alloca(lltype(t), "best")

    # ------------------------------------------------------------------ print
    def do_print(self, e):
        """print(...) from native code: values are handed to Apsi's own print (same formatting)."""
        if self.mc.gpu:
            raise Unsupported("print() inside a GPU kernel", e)
        parts = []
        vals = []
        sep, end = " ", "\n"
        for k, v in e.kwargs:
            if k in ("sep", "end") and isinstance(v, A.Str):
                if k == "sep":
                    sep = v.value
                else:
                    end = v.value
            else:
                raise Unsupported("print() with this keyword", e)

        def add_value(x, spec=None):
            v = self.materialize(self.expr(x), e)
            if v.t not in SCALARS:
                raise Unsupported(f"printing {tname(v.t)} from native code", e)
            parts.append(("v", len(vals), v.t, v.d, spec))
            vals.append(v)
        items = []
        for a in e.args:
            if isinstance(a, A.Str):
                items.append([("s", a.value)])
            elif isinstance(a, A.FStr):
                start = len(parts)
                for p in a.parts:
                    if isinstance(p, str):
                        parts.append(("s", p))
                    else:
                        add_value(p[0], p[1])
                items.append(parts[start:])
                del parts[start:]
                continue
            else:
                add_value(a)
                items.append([parts.pop()])
        if self.dry:
            return
        bd = self.f.b
        site = len(self.mc.print_sites)
        self.mc.print_sites.append((items, sep, end, self.r))
        buf = self._entry_alloca(ir.ArrayType(i64t(), max(1, len(vals))), "pbuf")
        for k, v in enumerate(vals):
            if v.t in (F64, F32):
                raw = bd.bitcast(self.convert(v, F64, e).v, i64t())
            else:
                raw = self.as_int(v).v
            bd.store(raw, bd.gep(buf, [ir.Constant(i64t(), 0), ir.Constant(i64t(), k)]))
        fn = self.mc.decl("apsi_print", ir.VoidType(), [i64t(), i8p()])
        base = bd.load(self.mc.site_base())
        bd.call(fn, [bd.add(base, ir.Constant(i64t(), site)), bd.bitcast(buf, i8p())])


# ============================================================================= AST utilities
def _sfx(t):
    return "f32" if t == F32 else "f64"


def _dims(d):
    return dim_name(d) if not isinstance(d, tuple) or isinstance(d, type(DIMLESS)) else str(d)


def _identity(op, t):
    if op in ("+", "-"):
        return ir.Constant(lltype(t), 0 if t == I64 else 0.0)
    if op == "*":
        return ir.Constant(lltype(t), 1 if t == I64 else 1.0)
    if op == "min":
        return ir.Constant(lltype(t), (1 << 63) - 1 if t == I64 else math.inf)
    if op == "max":
        return ir.Constant(lltype(t), -(1 << 63) if t == I64 else -math.inf)
    raise Unsupported(f"reduction {op}")


def _array_parts(t, v, bd):
    if is_arr(t):
        return [(t, v)]
    if t[0] == "t":
        out = []
        for k, x in enumerate(t[1]):
            if is_arr(x):
                out.append((x, bd.extract_value(v, k)))
        return out
    return []


def _children(n):
    for f in n._fields:
        v = getattr(n, f)
        if isinstance(v, A.AST):
            yield v
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, A.AST):
                    yield x
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        if isinstance(y, A.AST):
                            yield y
                        elif isinstance(y, list):
                            for z in y:
                                if isinstance(z, A.AST):
                                    yield z


def _walk(nodes):
    stack = list(nodes) if isinstance(nodes, list) else [nodes]
    while stack:
        n = stack.pop()
        if isinstance(n, A.AST):
            yield n
            stack.extend(_children(n))


def _walk_no_nested_loops(nodes):
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, A.AST):
            yield n
            if isinstance(n, (A.For, A.While)):
                continue
            stack.extend(_children(n))


def _allocates(body):
    """Could this code create a tensor? (allocation built-ins, calls, or tensor arithmetic)"""
    for n in _walk(body):
        if isinstance(n, A.Call):
            return True
        if isinstance(n, A.Binary) and n.op not in ("==", "!=", "<", "<=", ">", ">=", "~="):
            return True
    return False


def _holds_array(t):
    return t[0] == "a" or (t[0] == "t" and any(_holds_array(x) for x in t[1]))


def _lazy_walk(lz):
    out = [lz]
    for k in getattr(lz, "kids", ()):
        if isinstance(k, Lazy):
            out += _lazy_walk(k)
    return out


def _has_loop(body):
    return any(isinstance(n, (A.For, A.While, A.Evolve)) for n in _walk(body))


def _assigned_names(body):
    out = set()
    for n in _walk(body):
        if isinstance(n, (A.Assign, A.AugAssign)):
            t = n.target
            if isinstance(t, A.Name):
                out.add(t.id)
            elif isinstance(t, A.TupleLit):
                out |= {x.id for x in t.items if isinstance(x, A.Name)}
        elif isinstance(n, A.Let):
            out |= set(n.targets)
        elif isinstance(n, A.For):
            out |= set(n.targets)
    return out


def _declared_names(body):
    out = set()
    for n in _walk(body):
        if isinstance(n, A.Let):
            out |= set(n.targets)
        elif isinstance(n, A.For):
            out |= set(n.targets)
    return out


def _writes(body):
    """name -> list of write kinds ('+', '-', '*', 'min', 'max', '=') for plain-name targets."""
    out = {}
    for n in _walk(body):
        if isinstance(n, A.AugAssign) and isinstance(n.target, A.Name):
            out.setdefault(n.target.id, []).append(n.op)
        elif isinstance(n, A.Assign):
            t = n.target
            if isinstance(t, A.Name):
                kind = "="
                v = n.value
                if isinstance(v, A.Call) and isinstance(v.func, A.Name) and v.func.id in ("min", "max") \
                        and len(v.args) == 2 and not v.kwargs and \
                        any(isinstance(x, A.Name) and x.id == t.id for x in v.args):
                    other = v.args[1] if isinstance(v.args[0], A.Name) and v.args[0].id == t.id else v.args[0]
                    if not any(isinstance(m, A.Name) and m.id == t.id for m in _walk([other])):
                        kind = v.func.id
                out.setdefault(t.id, []).append(kind)
            elif isinstance(t, A.TupleLit):
                for x in t.items:
                    if isinstance(x, A.Name):
                        out.setdefault(x.id, []).append("=")
    return out


def _reduction_kind(kinds):
    ks = set(kinds)
    if ks <= {"+", "-"}:
        return "+"
    if ks == {"*"}:
        return "*"
    if ks == {"min"}:
        return "min"
    if ks == {"max"}:
        return "max"
    return None


def _affine(e, var):
    """c when e is `var + c`, `var - c`, `c + var` or `var` (c an integer literal), else None."""
    if isinstance(e, A.Name) and e.id == var:
        return 0
    if isinstance(e, A.Binary) and e.op in ("+", "-"):
        l, r = e.left, e.right
        if isinstance(l, A.Name) and l.id == var and isinstance(r, A.Num) and isinstance(r.value, int) \
                and r.unit is None:
            return r.value if e.op == "+" else -r.value
        if e.op == "+" and isinstance(r, A.Name) and r.id == var and isinstance(l, A.Num) \
                and isinstance(l.value, int) and l.unit is None:
            return l.value
    return None


def _invariant(e, assigned, em):
    """True when e only reads loop-invariant locals and constants (no calls, no tensor reads)."""
    for n in _walk([e]):
        if isinstance(n, A.Name):
            if n.id in assigned:
                return False
            if em.find(n.id) is None:
                return False
        elif isinstance(n, A.Num):
            if n.unit is not None:
                return False
        elif isinstance(n, A.Binary):
            if n.op not in ("+", "-", "*", "//"):
                return False
        else:
            return False
    return True


def _pure(e):
    return not any(isinstance(n, A.Call) for n in _walk([e]))


def _always_returns(body):
    if not body:
        return False
    last = body[-1]
    if isinstance(last, A.Return):
        return True
    if isinstance(last, A.If):
        return last.else_body is not None and all(_always_returns(b) for _, b in last.branches) \
            and _always_returns(last.else_body)
    if isinstance(last, A.While) and isinstance(last.cond, A.Bool) and last.cond.value:
        return not any(isinstance(n, A.Break) for n in _walk_no_nested_loops(last.body))
    return False


def _is_enumerate(it, em):
    return isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "enumerate" \
        and em.find("enumerate") is None and len(it.args) == 1 and not it.kwargs


def _describe(e):
    return {"Str": "strings", "FStr": "f-strings", "ListLit": "lists", "DictLit": "dictionaries",
            "ListComp": "list comprehensions", "DictComp": "dict comprehensions", "Lambda": "lambdas",
            "NoneLit": "none", "Range": "a range used as a value", "Slice": "slices here"}.get(
        type(e).__name__, f"'{type(e).__name__}' expressions")
