"""Apsi native tier: Apsi functions compiled to machine code.

    @native                      # safe: overflow / index / zero-division checks, strict IEEE arithmetic
    fn energy(x: Tensor) -> Float: ...

    @native("fast")              # checks off, fast-math (reassociation, FMA) - like C with -O3 -ffast-math
    @native("strict")            # fail instead of falling back to the bytecode engine

A `NativeFunction` compiles one specialisation per combination of argument types and units, the first time it is
called with them, then calls straight into machine code. When a function uses something the native compiler does
not support, it says why (once) and keeps running on the bytecode engine, so adding `@native` never breaks a
program. Calls from autodiff (`grad`), with lists, structs or other Python objects also take the bytecode path.
"""
import ctypes
import hashlib
import os
import sys
import threading
import weakref

import numpy as np

from .. import ast as A
from ..errors import RuntimeErr, AssertionFailed, ApsiError
from ..fmt import type_name
from ..units import Quantity, DIMLESS, DIM_NAMES
from . import engine as _eng
from .types import type_of_value as _type_of_value

_compile_lock = threading.RLock()
OPTION_NAMES = {"fast", "strict", "quiet", "auto"}
STATS = {"compiled": 0, "fallbacks": 0, "calls": 0}
_REPORTS = []          # (function name, message) for `apsi native explain`
REGISTRY = []          # every NativeFunction made in this process (for explain/ir)
_FREE = None


def available():
    return _eng.available()


def set_threads(n):
    _eng.POOL.set_threads(n)
    return _eng.POOL.threads


def threads():
    return _eng.POOL.threads


def info():
    """A record describing the native tier on this machine."""
    out = {"available": False, "threads": _eng.POOL.threads}
    try:
        e = _eng.engine()
    except _eng.NativeUnavailable as err:
        out["reason"] = str(err)
        return out
    feats = [f[1:] for f in e.features.split(",") if f.startswith("+")]
    simd = [f for f in ("sse4.2", "avx", "avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512fp16",
                        "avx512bf16", "amx-tile", "neon", "sve") if f in feats]
    out.update({"available": True, "cpu": e.cpu, "triple": e.triple, "llvm": e.llvm_version,
                "simd": simd, "cache_hits": e.cache_hits, "cache_misses": e.cache_misses,
                "compile_seconds": round(e.compile_seconds, 3), "cache_dir": e.cache_dir(),
                "functions_compiled": STATS["compiled"], "fallbacks": STATS["fallbacks"]})
    return out


def reports():
    return list(_REPORTS)


# ----------------------------------------------------------------------------- the decorator
def native(*args, **kwargs):
    """@native, @native("fast"), @native("strict"), @native(fast=true)."""
    if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], str):
        return NativeFunction(args[0], {})
    opts = {}
    for a in args:
        if not isinstance(a, str) or a not in OPTION_NAMES:
            raise RuntimeErr(f"@native options are {', '.join(sorted(OPTION_NAMES))}; got {a!r}")
        opts[a] = True
    for k, v in kwargs.items():
        if k not in OPTION_NAMES:
            raise RuntimeErr(f"unknown @native option {k!r}")
        opts[k] = bool(v)

    def deco(f):
        return NativeFunction(f, opts)
    return deco


def _plain_function(fd):
    """Generators and functions with *args / **kwargs stay on the bytecode engine."""
    from ..interpreter import is_generator
    return not any(getattr(p, "star", None) for p in fd.params) and not (not fd.is_expr and is_generator(fd.body))


def auto(f):
    """Used by `apsi run --native`: compile when possible, silently use the bytecode engine otherwise."""
    if isinstance(f, NativeFunction) or getattr(f, "_apsi_def", None) is None:
        return f
    if not _plain_function(f._apsi_def):
        return f
    return NativeFunction(f, {"auto": True, "quiet": True})


class _Fallback:
    __slots__ = ("reason", "permanent")

    def __init__(self, reason, permanent=False):
        self.reason = reason
        self.permanent = permanent          # the function body itself is not supported (any argument types)


class NativeFunction:
    """An Apsi function with machine-code specialisations."""

    def __init__(self, fn, options, erased=False):
        from ..values import Function
        if isinstance(fn, NativeFunction):
            fn = fn.fn
        if getattr(fn, "_apsi_def", None) is None:
            if isinstance(fn, Function):
                raise RuntimeErr("@native works on functions in compiled programs (apsi run), not in the "
                                 "tree-walking interpreter")
            raise RuntimeErr("@native can only decorate an Apsi function")
        if not _plain_function(fn._apsi_def):
            raise RuntimeErr(f"@native cannot compile {fn._apsi_def.name}(): generators and *args / **kwargs "
                             "parameters run on the bytecode engine — remove @native")
        self.fn = fn
        self.options = dict(options)
        if erased:
            self.options["erased"] = True
        self.erased = erased
        from .lower import lower
        self.fndef = lower(fn._apsi_def)          # list-style code → tensors (only in the compiled copy)
        self.name = getattr(fn, "name", self.fndef.name)
        self._apsi_def = fn._apsi_def
        self._apsi_free = getattr(fn, "_apsi_free", None)
        self.specs = {}
        self.warned = False
        self.auto = bool(self.options.get("auto"))
        self.fast = {}                              # cheap argument signature -> direct machine-code call
        self.retired = False
        self.py_calls = 0
        self._py_fallback = getattr(fn, "_apsi_e", fn) if erased else fn
        self.__name__ = self.name
        if not erased:
            self._apsi_e = NativeFunction(fn, options, erased=True)
            self._apsi_native_e = self._apsi_e
            REGISTRY.append(self)

    def __repr__(self):
        return f"<native fn {self.name}>"

    # ---------------------------------------------------------------- call
    def __call__(self, *args, **kwargs):
        if self.retired:                            # auto mode gave up on this function: no overhead at all
            return self._py_fallback(*args, **kwargs)
        if not kwargs and self.fast:
            try:
                sig = tuple([_fast_sig(a) for a in args])
                fc = self.fast.get(sig)
            except TypeError:
                fc = None
            if fc is not None:
                return fc(args)
        fd = self.fndef
        if kwargs or len(args) != len(fd.params):
            bound = self._bind(args, kwargs)
            if bound is None:
                return self._py_fallback(*args, **kwargs)
            args = bound
        key = []
        mask = 0
        orig = args
        lists = None
        for j, a in enumerate(args):
            if type(a) is list:                     # numbers in (nested) lists → a tensor for native code
                from .lower import as_array
                arr = as_array(a)
                if arr is not None:
                    if lists is None:
                        lists, args = [], list(args)
                    lists.append((a, arr))
                    args[j] = arr
        for j, a in enumerate(args):
            if isinstance(a.v if isinstance(a, Quantity) else a, np.generic):
                mask |= 1 << j                      # a NumPy scalar: NumPy's arithmetic rules apply
            tv = _key_of(a)
            if tv is None:
                self.py_calls += 1
                why = (f"argument {j + 1} ('{fd.params[j].name}') is a {type_name(a)}; native code takes numbers, "
                       "tensors and lists of numbers")
                if self.options.get("strict"):
                    raise RuntimeErr(f"@native(strict): {self.name}() cannot run natively: {why}")
                if not self.warned and not self.options.get("quiet") and not self.auto and \
                        os.environ.get("APSI_NATIVE_QUIET", "") not in ("1", "true"):
                    sys.stderr.write(f"apsi: note: {self.name}() runs on the bytecode engine: {why}\n")
                    self.warned = True
                if self.auto and self.py_calls >= 8:
                    self.retired = True             # called with records/strings/...: stay on the bytecode path
                return self._py_fallback(*orig)
            key.append(tv)
        key = (tuple(key), mask)
        for _ in range(3):
            spec = self.specs.get(key)
            if spec is None:
                spec = self._compile(key, args)
            if isinstance(spec, _Fallback):
                if self.auto and spec.permanent:
                    self.retired = True
                return self._py_fallback(*orig)
            STATS["calls"] += 1
            if spec.fast_call is not None and not kwargs and len(args) == len(fd.params):
                sig = tuple(_fast_sig(a) for a in args)
                if None not in sig:
                    self.fast[sig] = spec.fast_call
            try:
                result = spec.call(args)
            except _Retry:                         # a global tensor changed type: specialise again
                continue
            if lists:
                from .lower import write_back
                for lst, arr in lists:              # the caller's lists see every change, as before
                    write_back(lst, arr)
            return result
        return self._py_fallback(*orig)

    def _bind(self, args, kwargs):
        """Positional arguments in parameter order, filling simple defaults; None -> use the bytecode engine."""
        fd = self.fndef
        if len(args) > len(fd.params):
            return None
        out = list(args)
        for p in fd.params[len(args):]:
            if p.name in kwargs:
                out.append(kwargs[p.name])
            elif p.default is not None:
                v = _const_default(p.default)
                if v is None:
                    return None
                out.append(v)
            else:
                return None
        if set(kwargs) - {p.name for p in fd.params[len(args):]}:
            return None
        return out

    def _boundary_ok(self, key):
        """Do the arguments satisfy the parameter annotations? (if not, the checked path reports it)."""
        for p, (t, d) in zip(self.fndef.params, key):
            ann = p.type
            if ann is None or not isinstance(ann, A.TName):
                if ann is not None and not isinstance(ann, A.TName):
                    return False
                continue
            n = ann.name
            if n == "Int":
                if t[0] not in ("i", "b") or (not d.dimensionless and not self.erased):
                    return False
            elif n in ("Float", "Number", "Real"):
                if t[0] not in ("i", "b", "f", "s") or (not d.dimensionless and not self.erased):
                    return False
            elif n == "Bool":
                if t[0] != "b":
                    return False
            elif n in DIM_NAMES and not ann.args:
                if t[0] not in ("i", "f", "s", "a"):
                    return False
                if not self.erased and d != DIM_NAMES[n]:
                    return False
            elif n in ("Tensor", "Vec", "Mat", "Matrix", "Vec2", "Vec3", "Vec4"):
                if t[0] != "a":
                    return False
                if ann.args:
                    return False            # shape-generic checks stay on the bytecode path for now
            elif n == "Any":
                continue
            else:
                return False
        return True

    def _compile(self, key, args):
        with _compile_lock:
            spec = self.specs.get(key)
            if spec is not None:
                return spec
            if not self._boundary_ok(key[0]):
                spec = _Fallback("argument types do not match the annotations")
                self.specs[key] = spec
                return spec
            try:
                if self.fndef.shape_params:
                    from .compiler import Unsupported
                    raise Unsupported("shape-generic functions (fn f[N](...))", self.fndef)
                spec = Specialization(self, key)
                STATS["compiled"] += 1
            except ApsiError:
                raise
            except Exception as e:              # Unsupported, NativeUnavailable, LLVM errors
                from .compiler import Unsupported
                STATS["fallbacks"] += 1
                msg = str(e) if isinstance(e, (Unsupported, _eng.NativeUnavailable)) else \
                    f"internal compiler error: {type(e).__name__}: {e}"
                if self.options.get("strict"):
                    err = RuntimeErr(f"@native(strict): {self.name}() cannot be compiled: {msg}",
                                     getattr(e, "line", None), getattr(e, "col", None))
                    raise err from None
                _REPORTS.append((self.name, msg))
                if not self.warned and not self.options.get("quiet") and \
                        os.environ.get("APSI_NATIVE_QUIET", "") not in ("1", "true"):
                    sys.stderr.write(f"apsi: note: {self.name}() runs on the bytecode engine: {msg}\n")
                self.warned = True
                spec = _Fallback(msg, permanent=isinstance(e, Unsupported) and not _type_dependent(msg))
            self.specs[key] = spec
            return spec

    # ---------------------------------------------------------------- inspection
    def llvm_ir(self, *args):
        mask = sum(1 << j for j, a in enumerate(args) if isinstance(a.v if isinstance(a, Quantity) else a, np.generic))
        key = (tuple(_key_of(a) for a in args), mask)
        spec = self.specs.get(key) or self._compile(key, args)
        if isinstance(spec, _Fallback):
            raise RuntimeErr(f"{self.name}() is not compiled natively: {spec.reason}")
        return spec.ir_text

    def assembly(self, *args):
        ir_text = self.llvm_ir(*args)
        return _eng.engine().assembly(ir_text)[1]


def _const_default(d):
    if isinstance(d, A.Num) and d.unit is None:
        return d.value
    if isinstance(d, A.Num):
        text, scale, dim = d.unit
        return Quantity(d.value * scale, dim) if not dim.dimensionless else d.value * scale
    if isinstance(d, A.Bool):
        return bool(d.value)
    if isinstance(d, A.Unary) and d.op == "-" and isinstance(d.operand, A.Num) and d.operand.unit is None:
        return -d.operand.value
    return None


_TYPE_OF = None
_NP_LAYOUT = []


def np_layout():
    """Byte offsets of data, dimensions and strides in NumPy's array struct (PyArrayObject_fields), measured on this
    interpreter and checked against .ctypes; None when they cannot be established (the slower path is used).

    Only memory inside the array object is read, and a pointer is followed only after the struct's documented
    order (data, nd, dimensions, strides) has been confirmed."""
    if _NP_LAYOUT:
        return _NP_LAYOUT[0]
    lay = None
    try:
        a = np.zeros((3, 5))
        nwords = type(a).__basicsize__ // 8
        words = (ctypes.c_int64 * nwords).from_address(id(a))
        data = [i for i in range(nwords - 3) if words[i] == a.ctypes.data and words[i + 1] == 2]
        if len(data) == 1:
            i = data[0]
            cand = (i * 8, (i + 2) * 8, (i + 3) * 8)
            ok = True
            for b in (a, np.arange(7.0), np.ones((2, 3, 4), dtype=np.float32), np.zeros((4, 6))[1:3]):
                w = (ctypes.c_int64 * nwords).from_address(id(b))
                nd = b.ndim
                ok = ok and w[i] == b.ctypes.data and w[i + 1] == nd and w[i + 2] != 0 and w[i + 3] != 0
                if ok:
                    ok = tuple((ctypes.c_int64 * nd).from_address(w[i + 2])) == b.shape and \
                        tuple((ctypes.c_int64 * nd).from_address(w[i + 3])) == b.strides
            lay = cand if ok else None
    except Exception:
        lay = None
    _NP_LAYOUT.append(lay)
    return lay


def _fast_sig(a):
    """A cheap description of an argument for the call cache, or None (then the full path is used)."""
    c = a.__class__
    if c is float or c is int or c is bool or c is np.float64 or c is np.int64 or c is np.float32 or c is np.bool_:
        return c
    if c is np.ndarray and a.flags.c_contiguous:
        return (a.dtype.num, a.ndim)
    return None


def _key_of(v):
    return _type_of_value(v)


def _type_dependent(msg):
    """Fallback reasons that may not hold for other argument types."""
    return any(w in msg for w in ("holds both", "Tensor[", "of type", "Int was expected", "an integer was expected",
                                  "converting", "passing", "indexing", "rank-"))


# ----------------------------------------------------------------------------- one specialisation
class Specialization:
    def __init__(self, nf, full_key):
        from . import compiler as C
        if C.ir is None:
            raise _eng.NativeUnavailable("the native compiler needs llvmlite (pip install llvmlite)")
        eng = _eng.engine()
        opts = dict(nf.options)
        self.nf = nf
        self.key = full_key
        key, mask = full_key
        mc = C.ModuleCtx(opts)
        resolver = C.Resolver(getattr(nf.fn, "_apsi_e", None) or nf.fn, nf._apsi_free, opts)
        targs = [C.TV(t, DIMLESS if nf.erased else d, npv=bool(mask >> j & 1) and t in C.SCALARS)
                 for j, (t, d) in enumerate(key)]
        root = mc.spec(nf.fndef, targs, resolver)
        fd = nf.fndef
        if isinstance(fd.ret, A.TName) and fd.ret.name == "Int" and root.ret.t not in (C.I64, C.BOOL):
            raise C.Unsupported(f"{fd.name}() is declared -> Int but returns {C.tname(root.ret.t)}", fd)
        mc.emit_all()
        self.ret = root.ret
        self.root = root
        self.notes = [n for s in mc.order for n in s.notes]
        for n in self.notes:
            _REPORTS.append((nf.name, n))
            if not nf.options.get("quiet") and os.environ.get("APSI_NATIVE_QUIET", "") not in ("1", "true"):
                sys.stderr.write(f"apsi: note: {nf.name}(): {n}\n")
        self.arg_layout = [(t, d) for t, d in key]
        self._build_entry(mc, root)
        # make the module's text unique to everything that is not in the IR itself
        info = repr((mc.print_sites and [(i, s, e) for i, s, e, _ in mc.print_sites], mc.assert_msgs,
                     mc.fn_names)).encode()
        tag = hashlib.sha256(str(mc.module).encode() + info).hexdigest()[:20]
        text = str(mc.module).replace("apsi_direct_HASH", f"apsi_direct_{tag}") \
            .replace("apsi_entry_HASH", f"apsi_entry_{tag}") \
            .replace("apsi_gtab_HASH", f"apsi_gtab_{tag}").replace("apsi_sites_HASH", f"apsi_sites_{tag}")
        text += f'\n@"apsi_info_{tag}" = private constant [{len(info)} x i8] c"{_llstr(info)}"\n'
        self.ir_text = text
        with _compile_lock:
            prior = _LOADED.get(tag)
            if prior is None:
                names = [f"apsi_entry_{tag}"] + ([f"apsi_direct_{tag}"] if self.direct_types else [])
                addrs = eng.compile(text, names)
                gtab = eng.ee.get_global_value_address(f"apsi_gtab_{tag}")
                sites = eng.ee.get_global_value_address(f"apsi_sites_{tag}")
                base = len(_eng.PRINT_SITES)
                for items, sep, end, res in mc.print_sites:
                    _eng.PRINT_SITES.append(_make_printer(items, sep, end, res))
                ctypes.c_int64.from_address(sites).value = base
                prior = _LOADED[tag] = (addrs[f"apsi_entry_{tag}"], gtab, addrs.get(f"apsi_direct_{tag}"))
        self.entry_addr, gtab, direct_addr = prior
        self.entry = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(self.entry_addr)
        self.fn_names = mc.fn_names
        self.assert_msgs = mc.assert_msgs
        # global tensors: one descriptor buffer per slot, re-filled whenever the global is rebound
        self.gslots = []
        for idx, (res, name, t) in enumerate(mc.gslots):
            nd = t[2]
            buf = (ctypes.c_int64 * (2 + 2 * nd))()
            ctypes.c_void_p.from_address(gtab + 8 * idx).value = ctypes.addressof(buf)
            self.gslots.append([res, name, t, buf, None, None])
        self.nargs = sum(_slot_count(t) for t, _ in key)
        self.nret = (_slot_count(root.ret.t) + (1 if root.ret.npv is C.DYN else 0)) if root.ret.t != C.NONE else 0
        self.lock = threading.Lock()
        self.fast_call = self._make_fast_call(direct_addr) if direct_addr and self.direct_types else None

    def _build_entry(self, mc, root):
        """apsi_entry(i64* args, i64* ret): unpack the argument slots, call the function, pack the result."""
        from . import compiler as C
        ir = C.ir
        m = mc.module
        mc.gtable()
        mc.site_base()
        fnty = ir.FunctionType(ir.VoidType(), [C.i64t().as_pointer(), C.i64t().as_pointer()])
        fn = ir.Function(m, fnty, name="apsi_entry_HASH")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        argp, retp = fn.args
        pos = [0]

        def slot():
            p = b.gep(argp, [ir.Constant(C.i64t(), pos[0])])
            pos[0] += 1
            return b.load(p)
        vals = [_unpack(b, t, slot, C) for t, _ in self.arg_layout]
        r = b.call(root.fn, vals)
        if root.ret.t != C.NONE:
            out = [0]

            def put(v):
                b.store(v, b.gep(retp, [ir.Constant(C.i64t(), out[0])]))
                out[0] += 1
            if root.ret.npv is C.DYN:                # value, then 1 if it is a NumPy scalar on this path
                _pack(b, root.ret.t, b.extract_value(r, 0), put, C)
                put(b.zext(b.extract_value(r, 1), C.i64t()))
            else:
                _pack(b, root.ret.t, r, put, C)
        b.ret_void()
        self._build_direct(mc, root)

    def _build_direct(self, mc, root):
        """apsi_direct(args...) -> scalar: arguments in registers, tensors as the NumPy object itself."""
        from . import compiler as C
        ir = C.ir
        self.direct_types = None
        rt = root.ret.t
        if rt not in (C.I64, C.F64, C.F32, C.BOOL, C.NONE) or root.ret.npv is C.DYN or mc.gslots:
            return
        lay = np_layout() if any(t[0] == "a" for t, _ in self.arg_layout) else (0, 0, 0)
        if lay is None:
            return
        params, kinds = [], []
        for t, d in self.arg_layout:
            if t in (C.I64, C.BOOL):
                params.append(C.i64t()); kinds.append("i")
            elif t in (C.F64, C.F32):
                params.append(ir.DoubleType()); kinds.append("f")
            elif t[0] == "a" and t[3]:
                params.append(C.i8p()); kinds.append("a")
            else:
                return
        ret_ll = {C.I64: C.i64t(), C.BOOL: C.i64t(), C.F64: ir.DoubleType(), C.F32: ir.DoubleType(),
                  C.NONE: ir.VoidType()}[rt]
        fn = ir.Function(mc.module, ir.FunctionType(ret_ll, params), name="apsi_direct_HASH")
        b = ir.IRBuilder(fn.append_basic_block("entry"))
        vals = []
        for (t, d), arg in zip(self.arg_layout, fn.args):
            if t == C.BOOL:
                vals.append(b.trunc(arg, ir.IntType(1)))
            elif t == C.F32:
                vals.append(b.fptrunc(arg, ir.FloatType()))
            elif t[0] == "a":
                nd = t[2]
                esz = {"f8": 8, "f4": 4, "i8": 8, "i4": 4, "i2": 2, "i1": 1, "u1": 1, "b1": 1}[t[1]]

                def word(off):
                    return b.load(b.bitcast(b.gep(arg, [ir.Constant(C.i64t(), off)]), C.i64t().as_pointer()))
                data = b.inttoptr(word(lay[0]), C.elem_ll(t[1]).as_pointer())
                dims = b.inttoptr(word(lay[1]), C.i64t().as_pointer())
                strd = b.inttoptr(word(lay[2]), C.i64t().as_pointer())
                st = ir.Constant(C.lltype(t), ir.Undefined)
                st = b.insert_value(st, data, 0)
                st = b.insert_value(st, ir.Constant(C.i8p(), None), 1)
                for k in range(nd):
                    st = b.insert_value(st, b.load(b.gep(dims, [ir.Constant(C.i64t(), k)])), [2, k])
                    sb = b.load(b.gep(strd, [ir.Constant(C.i64t(), k)]))
                    st = b.insert_value(st, b.sdiv(sb, ir.Constant(C.i64t(), esz), flags=["exact"]), [3, k])
                vals.append(st)
            else:
                vals.append(arg)
        r = b.call(root.fn, vals)
        if rt == C.NONE:
            b.ret_void()
        elif rt == C.BOOL:
            b.ret(b.zext(r, C.i64t()))
        elif rt == C.F32:
            b.ret(b.fpext(r, ir.DoubleType()))
        else:
            b.ret(r)
        self.direct_types = (kinds, rt)

    # ---------------------------------------------------------------- calling
    def call(self, args):
        from . import compiler as C
        buf = (ctypes.c_int64 * max(1, self.nargs))()
        keep = []
        k = 0
        for (t, d), a in zip(self.arg_layout, args):
            k = _fill(buf, k, t, a, keep)
        ret = (ctypes.c_int64 * max(1, self.nret))()
        for gs in self.gslots:
            self._refresh_global(gs)
        err = _eng.error_slots()
        err[0] = 0
        self.entry(buf, ret)
        if err[0]:
            raise self._error(err)
        if self.ret.t == C.NONE:
            return None
        out = _unpack_ret(self.ret.t, self.ret.d, ret, [0], args, self, C)
        is_np = bool(ret[self.nret - 1]) if self.ret.npv is C.DYN else self.ret.npv is True
        if is_np:
            raw = out.v if isinstance(out, Quantity) else out
            raw = {int: np.int64, float: np.float64, bool: np.bool_}.get(type(raw), lambda x: x)(raw)
            out = Quantity(raw, out.dim) if isinstance(out, Quantity) else raw
        return out

    def _make_fast_call(self, addr):
        from . import compiler as C
        kinds, rt = self.direct_types
        ct = {"i": ctypes.c_int64, "f": ctypes.c_double, "a": ctypes.c_void_p}
        res = {C.I64: ctypes.c_int64, C.BOOL: ctypes.c_int64, C.F64: ctypes.c_double, C.F32: ctypes.c_double,
               C.NONE: None}[rt]
        fn = ctypes.CFUNCTYPE(res, *[ct[k] for k in kinds])(addr)
        self._direct = fn
        err = _eng.error_slots()
        arrays = tuple(i for i, k in enumerate(kinds) if k == "a")
        d = self.ret.d
        wrap = None
        if rt != C.NONE:
            conv = {C.I64: int, C.BOOL: bool, C.F64: float, C.F32: np.float32}[rt]
            if self.ret.npv is True:
                conv = {C.I64: np.int64, C.BOOL: np.bool_, C.F64: np.float64, C.F32: np.float32}[rt]
            if not d.dimensionless and not self.nf.erased:
                def wrap(r, conv=conv, d=d):
                    return Quantity(conv(r), d)
            elif conv not in (int, float):
                wrap = conv
        spec = self

        ints = tuple(i for i, k in enumerate(kinds) if k == "i")
        lo, hi = -(1 << 63), 1 << 63
        nf = self.nf

        def fast(args):
            for i in ints:                        # ctypes would silently truncate an Int beyond 64 bits
                v = args[i]
                if v.__class__ is int and not lo <= v < hi:
                    return nf._py_fallback(*args)
            if arrays:
                args = list(args)
                for i in arrays:
                    args[i] = id(args[i])
            err[0] = 0
            r = fn(*args)
            if err[0]:
                raise spec._error(err)
            return wrap(r) if wrap is not None else r
        return fast

    def _refresh_global(self, gs):
        res, name, t, buf, last_id, holder = gs
        type_of_value = _type_of_value
        if "." in name:
            base, attr = name.split(".", 1)
            r = res.lookup(base)
            obj = res.attr(r[1], attr) if r else None
        else:
            r = res.lookup(name)
            obj = r[1] if r else None
        if obj is None:
            raise RuntimeErr(f"native: the global '{name}' disappeared")
        if id(obj) == last_id:
            return
        tv = type_of_value(obj)
        if tv is None or tv[0][:3] != t[:3] or (t[3] and not tv[0][3]):
            # the global changed type: drop this specialisation and compile again next call
            self.nf.specs.pop(self.key, None)
            raise _Retry()
        raw = obj.v if isinstance(obj, Quantity) else obj
        nd = t[2]
        buf[0] = raw.ctypes.data
        buf[1] = 0
        for j in range(nd):
            buf[2 + j] = raw.shape[j]
            buf[2 + nd + j] = raw.strides[j] // raw.itemsize
        gs[4] = id(obj)
        gs[5] = obj

    def _error(self, err):
        from . import compiler as C
        code, line, col, a, b, fidx, msg = (int(err[i]) for i in range(7))
        err[0] = 0
        where = self.fn_names[fidx] if 0 <= fidx < len(self.fn_names) else self.nf.name
        if code == C.E_INDEX:
            e = RuntimeErr(f"index {a} is out of range for an axis of length {b}")
        elif code == C.E_OVERFLOW:
            e = RuntimeErr("integer overflow: the result does not fit in 64 bits (native code uses 64-bit Int;"
                           " use Float or the bytecode engine for bigger integers)")
        elif code == C.E_ZERODIV:
            e = RuntimeErr("division by zero")
        elif code == C.E_ASSERT:
            text = self.assert_msgs[msg] if 0 <= msg < len(self.assert_msgs) else "assertion failed"
            e = AssertionFailed(text, line, col)
        elif code == C.E_NEGEXP:
            e = RuntimeErr(f"negative exponent {a} for an Int power in native code (write it as a Float)")
        elif code == C.E_SHAPE:
            e = RuntimeErr(f"shapes do not match: an axis of length {a} cannot broadcast to {b}")
        elif code == C.E_NOMEM:
            e = RuntimeErr(f"out of memory allocating {a} bytes")
        elif code == C.E_NEGSIZE:
            e = RuntimeErr(f"negative tensor size {a}")
        elif code == C.E_NORETURN:
            e = RuntimeErr(f"{where}() ended without returning a value")
        elif code == C.E_EMPTY:
            e = RuntimeErr("min/max/argmin/argmax of an empty tensor")
        elif code == C.E_CONVERT:
            e = RuntimeErr("cannot convert NaN or infinity (or a too large number) to Int")
        else:
            e = RuntimeErr(f"native runtime error {code}")
        e.line, e.col = line or None, col or None
        if where != self.nf.name:
            e.trace.append(f"{where}()")
        return e


class _Retry(Exception):
    pass


_LOADED = {}


def _llstr(b):
    return "".join(chr(c) if 32 <= c < 127 and c not in (34, 92) else f"\\{c:02X}" for c in b)


# ----------------------------------------------------------------------------- marshalling
def _slot_count(t):
    if t[0] == "a":
        return 2 + 2 * t[2]
    if t[0] == "t":
        return sum(_slot_count(x) for x in t[1])
    return 1


def _unpack(b, t, slot, C):
    ir = C.ir
    if t == C.I64:
        return slot()
    if t == C.BOOL:
        return b.trunc(slot(), ir.IntType(1))
    if t == C.F64:
        return b.bitcast(slot(), ir.DoubleType())
    if t == C.F32:
        return b.fptrunc(b.bitcast(slot(), ir.DoubleType()), ir.FloatType())
    if t[0] == "a":
        st = ir.Constant(C.lltype(t), ir.Undefined)
        st = b.insert_value(st, b.inttoptr(slot(), C.elem_ll(t[1]).as_pointer()), 0)
        st = b.insert_value(st, b.inttoptr(slot(), C.i8p()), 1)
        for k in range(t[2]):
            st = b.insert_value(st, slot(), [2, k])
        for k in range(t[2]):
            st = b.insert_value(st, slot(), [3, k])
        return st
    raise C.Unsupported(f"passing {C.tname(t)} from Python")


def _pack(b, t, v, put, C):
    ir = C.ir
    if t == C.I64:
        put(v)
    elif t == C.BOOL:
        put(b.zext(v, C.i64t()))
    elif t == C.F64:
        put(b.bitcast(v, C.i64t()))
    elif t == C.F32:
        put(b.bitcast(b.fpext(v, ir.DoubleType()), C.i64t()))
    elif t[0] == "a":
        put(b.ptrtoint(b.extract_value(v, 0), C.i64t()))
        put(b.ptrtoint(b.extract_value(v, 1), C.i64t()))
        for k in range(t[2]):
            put(b.extract_value(v, [2, k]))
        for k in range(t[2]):
            put(b.extract_value(v, [3, k]))
    elif t[0] == "t":
        for k, x in enumerate(t[1]):
            _pack(b, x, b.extract_value(v, k), put, C)
    else:
        raise C.Unsupported(f"returning {C.tname(t)} to Python")


_F64 = ctypes.c_double


def _fill(buf, k, t, a, keep):
    raw = a.v if isinstance(a, Quantity) else a
    c = t[0]
    if c == "f" or c == "s":
        buf[k] = ctypes.c_int64.from_buffer(ctypes.c_double(float(raw))).value
        return k + 1
    if c == "i" or c == "b":
        buf[k] = int(raw)
        return k + 1
    nd = t[2]
    buf[k] = raw.ctypes.data
    buf[k + 1] = 0
    for j in range(nd):
        buf[k + 2 + j] = raw.shape[j]
        buf[k + 2 + nd + j] = raw.strides[j] // raw.itemsize
    keep.append(raw)
    return k + 2 + 2 * nd


def _free_fn():
    global _FREE
    if _FREE is None:
        import ctypes.util
        lib = None
        if sys.platform == "win32":
            for n in ("ucrtbase", "msvcrt"):
                try:
                    lib = ctypes.CDLL(n)
                    break
                except OSError:
                    pass
        else:
            lib = ctypes.CDLL(ctypes.util.find_library("c") or None)
        _FREE = lib.free
        _FREE.argtypes = [ctypes.c_void_p]
        _FREE.restype = None
    return _FREE


def _unpack_ret(t, d, ret, pos, args, spec, C):
    c = t[0]
    if c == "t":
        out = []
        for k, x in enumerate(t[1]):
            out.append(_unpack_ret(x, d[k] if isinstance(d, tuple) else DIMLESS, ret, pos, args, spec, C))
        return tuple(out)
    k = pos[0]
    if c in ("i", "b", "f", "s"):
        pos[0] += 1
        raw = ret[k]
        if c == "i":
            v = int(raw)
        elif c == "b":
            v = bool(raw)
        else:
            v = ctypes.c_double.from_buffer(ctypes.c_int64(raw)).value
            if c == "s":
                v = np.float32(v)
        return Quantity(v, d) if not d.dimensionless and not spec.nf.erased else v
    nd = t[2]
    pos[0] += 2 + 2 * nd
    data, base = ret[k] & ((1 << 64) - 1), ret[k + 1] & ((1 << 64) - 1)
    shape = tuple(int(ret[k + 2 + j]) for j in range(nd))
    strides = tuple(int(ret[k + 2 + nd + j]) for j in range(nd))
    dtype = np.dtype(C.ELEM_DTYPE[t[1]])
    if base:
        total = 1
        for s in shape:
            total *= s
        nbytes = max(1, total) * dtype.itemsize
        holder = (ctypes.c_char * nbytes).from_address(data)
        weakref.finalize(holder, _free_fn(), ctypes.c_void_p(base))
        arr = np.frombuffer(holder, dtype=dtype, count=total).reshape(shape)
    else:
        arr = _view_of_input(data, shape, strides, dtype, args, spec)
    return Quantity(arr, d) if not d.dimensionless and not spec.nf.erased else arr


def _view_of_input(data, shape, strides, dtype, args, spec):
    """A returned view into an argument or global tensor: give back a real NumPy view of it."""
    cands = [a.v if isinstance(a, Quantity) else a for a in args]
    cands += [gs[5].v if isinstance(gs[5], Quantity) else gs[5] for gs in spec.gslots if gs[5] is not None]
    for a in cands:
        if not isinstance(a, np.ndarray) or a.dtype != dtype:
            continue
        lo = a.ctypes.data
        span = (a.size or 1) * a.itemsize
        if lo <= data < lo + span + 1 and a.flags.c_contiguous:
            byte_strides = tuple(s * dtype.itemsize for s in strides)
            return np.ndarray(shape, dtype=dtype, buffer=a, offset=data - lo, strides=byte_strides)
    # unknown owner: copy it now rather than ever alias memory that may be freed
    span = 1
    for n, st in zip(shape, strides):
        span += (n - 1) * st if n else 0
    raw = (ctypes.c_char * max(1, span * dtype.itemsize)).from_address(data)
    view = np.ndarray(shape, dtype=dtype, buffer=raw, strides=tuple(s * dtype.itemsize for s in strides))
    return view.copy()


def _make_printer(items, sep, end, resolver):
    from ..fmt import fmt
    from ..units import Quantity as Q
    out_fn = resolver.builtin_obj("print")

    def run(slots):
        arr = ctypes.cast(slots, ctypes.POINTER(ctypes.c_int64))
        pieces = []
        for item in items:
            text = []
            whole = None
            for part in item:
                if part[0] == "s":
                    text.append(part[1])
                    continue
                _, k, t, d, spec = part
                raw = arr[k]
                if t[0] in ("f", "s"):
                    v = ctypes.c_double.from_buffer(ctypes.c_int64(raw)).value
                elif t[0] == "b":
                    v = bool(raw)
                else:
                    v = int(raw)
                if not d.dimensionless:
                    v = Q(v, d)
                if len(item) == 1 and spec is None:
                    whole = v
                else:
                    text.append(fmt(v, True, spec))
            pieces.append(whole if whole is not None else "".join(text))
        if out_fn is not None:
            try:
                out_fn(*pieces, sep=sep, end=end)
                return
            except TypeError:
                pass
        sys.stdout.write(sep.join(fmt(p, True) if not isinstance(p, str) else p for p in pieces) + end)
        sys.stdout.flush()
    return run
