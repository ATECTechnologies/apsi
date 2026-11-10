"""Hub language models on the processor, from bfloat16 weights.

A token of a language model reads every weight once; on the processor that reading is the cost. NumPy needs float32
weights (twice the bytes of the bfloat16 the checkpoints are published in), so this module compiles, with Apsi's
native compiler (LLVM, already part of Apsi), a matrix product that reads bfloat16 directly and runs it on all cores:
half the memory and about half the time per token. The sums are float32, as before.

Used by hub_models.Decoder when the native compiler is available and the checkpoint is bfloat16; otherwise the
NumPy path runs. APSI_HUB_NATIVE=0 turns it off.
"""
import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_lock = threading.Lock()
_fn = None
_why = None
_pool = None
MANY = 24                                               # from this many rows on: decode + BLAS


def _build():
    """y[t, o] = sum_i bf16(W[o, i]) * x[t, i] for o in [o0, o1): rows of one thread's share."""
    import llvmlite.ir as ir
    from .native import engine as _eng
    eng = _eng.engine()
    i16, i32, i64, f32 = ir.IntType(16), ir.IntType(32), ir.IntType(64), ir.FloatType()
    m = ir.Module(name="apsi_hub_cpu")
    fma = ir.Function(m, ir.FunctionType(f32, [f32, f32, f32]), name="llvm.fma.f32")
    fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [i16.as_pointer(), f32.as_pointer(), f32.as_pointer(),
                                                       i64, i64, i64, i64, i64]), name="apsi_hub_bf16_rows")
    W, X, Y, o0, o1, I, T, O = fn.args
    for a in (W, X, Y):
        a.add_attribute("noalias")
    fast = ("fast",)
    entry = fn.append_basic_block("entry")
    b = ir.IRBuilder(entry)

    def loop(b, lo, hi, body):
        pre = b.block
        head = b.append_basic_block("h")
        bod = b.append_basic_block("b")
        end = b.append_basic_block("e")
        b.branch(head)
        b.position_at_end(head)
        i = b.phi(i64)
        i.add_incoming(lo, pre)
        b.cbranch(b.icmp_signed("<", i, hi), bod, end)
        b.position_at_end(bod)
        body(b, i)
        i.add_incoming(b.add(i, ir.Constant(i64, 1)), b.block)
        b.branch(head)
        b.position_at_end(end)

    def row(b, o):
        wrow = b.gep(W, [b.mul(o, I)])

        def per_t(b, t):
            xrow = b.gep(X, [b.mul(t, I)])
            pre = b.block
            head = b.append_basic_block("ih")
            bod = b.append_basic_block("ib")
            end = b.append_basic_block("ie")
            b.branch(head)
            b.position_at_end(head)
            i = b.phi(i64)
            acc = b.phi(f32)
            i.add_incoming(ir.Constant(i64, 0), pre)
            acc.add_incoming(ir.Constant(f32, 0.0), pre)
            b.cbranch(b.icmp_signed("<", i, I), bod, end)
            b.position_at_end(bod)
            w = b.load(b.gep(wrow, [i]))
            wf = b.bitcast(b.shl(b.zext(w, i32), ir.Constant(i32, 16)), f32)
            x = b.load(b.gep(xrow, [i]))
            nxt = b.fadd(acc, b.fmul(wf, x, flags=fast), flags=fast)
            i.add_incoming(b.add(i, ir.Constant(i64, 1)), bod)
            acc.add_incoming(nxt, bod)
            b.branch(head)
            b.position_at_end(end)
            b.store(acc, b.gep(Y, [b.add(b.mul(t, O), o)]))
        loop(b, ir.Constant(i64, 0), T, per_t)
    loop(b, o0, o1, row)
    b.ret_void()
    addr = eng.compile("; apsi hub bf16\n" + str(m), ["apsi_hub_bf16_rows"])["apsi_hub_bf16_rows"]
    proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                             ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    return proto(addr)


def available():
    """True when the native bfloat16 product can run here (compiled on first use)."""
    global _fn, _why
    if _fn is not None:
        return True
    if _why is not None:
        return False
    if os.environ.get("APSI_HUB_NATIVE", "1").strip() in ("0", "false", "no"):
        _why = "turned off by APSI_HUB_NATIVE=0"
        return False
    with _lock:
        if _fn is None and _why is None:
            try:
                _fn = _build()
            except Exception as e:                      # no llvmlite, an odd platform: NumPy instead
                _why = f"{type(e).__name__}: {e}"
    return _fn is not None


def threads():
    from .native.engine import default_threads
    return default_threads()


def matmul(x, w):
    """x [..., I] float32 @ W.T for W [O, I] given as bfloat16 bits (uint16): [..., O] float32, on all cores."""
    global _pool
    if _fn is None and not available():
        raise RuntimeError(f"the native bfloat16 product is not available ({_why})")
    lead = x.shape[:-1]
    O, I = w.shape
    xs = np.ascontiguousarray(x.reshape(-1, I), np.float32)
    T = xs.shape[0]
    if T >= MANY:                                       # many rows (a prompt): decoding once + BLAS is faster
        wf = (np.ascontiguousarray(w).astype(np.uint32) << np.uint32(16)).view(np.float32)
        return (xs @ wf.T).reshape(*lead, O)
    y = np.empty((T, O), np.float32)
    w = np.ascontiguousarray(w)
    n = threads()
    parts = max(1, min(n, O // 64 or 1))
    if parts == 1:
        _fn(w.ctypes.data, xs.ctypes.data, y.ctypes.data, 0, O, I, T, O)
        return y.reshape(*lead, O)
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="apsi-hub-cpu")
    step = (O + parts - 1) // parts
    step = (step + 15) // 16 * 16
    futs = [_pool.submit(_fn, w.ctypes.data, xs.ctypes.data, y.ctypes.data, a, min(O, a + step), I, T, O)
            for a in range(0, O, step)]
    for f in futs:
        f.result()
    return y.reshape(*lead, O)
