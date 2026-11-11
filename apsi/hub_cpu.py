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


QB = 32                                                 # values per int8 block (one float32 scale each)


def _build_q8():
    """y[t, o] = sum_b s[o, b] * sum_{i in block b} int8(W[o, i]) * x[t, i]: half the bytes of bfloat16."""
    import llvmlite.ir as ir
    from .native import engine as _eng
    eng = _eng.engine()
    i8, i64, f32 = ir.IntType(8), ir.IntType(64), ir.FloatType()
    m = ir.Module(name="apsi_hub_cpu_q8")
    fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [i8.as_pointer(), f32.as_pointer(), f32.as_pointer(),
                                                       f32.as_pointer(), i64, i64, i64, i64, i64]),
                     name="apsi_hub_q8_rows")
    W, S, X, Y, o0, o1, I, T, O = fn.args
    for a in (W, S, X, Y):
        a.add_attribute("noalias")
    fast = ("fast",)
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    nb = b.sdiv(I, ir.Constant(i64, QB))

    def loop(b, lo, hi, body, carry=None):
        pre = b.block
        head = b.append_basic_block("h")
        bod = b.append_basic_block("b")
        end = b.append_basic_block("e")
        b.branch(head)
        b.position_at_end(head)
        i = b.phi(i64)
        i.add_incoming(lo, pre)
        acc = None
        if carry is not None:
            acc = b.phi(f32)
            acc.add_incoming(carry, pre)
        b.cbranch(b.icmp_signed("<", i, hi), bod, end)
        b.position_at_end(bod)
        nxt = body(b, i, acc)
        i.add_incoming(b.add(i, ir.Constant(i64, 1)), b.block)
        if carry is not None:
            acc.add_incoming(nxt, b.block)
        b.branch(head)
        b.position_at_end(end)
        return acc

    V = ir.VectorType(f32, QB)
    vi8 = ir.VectorType(i8, QB)
    zero = ir.Constant(V, [0.0] * QB)
    fmav = ir.Function(m, ir.FunctionType(V, [V, V, V]), name=f"llvm.fma.v{QB}f32")
    red = ir.Function(m, ir.FunctionType(f32, [f32, V]), name=f"llvm.vector.reduce.fadd.v{QB}f32")

    def vloop(b, lo, hi, body, carry):
        pre = b.block
        head = b.append_basic_block("vh")
        bod = b.append_basic_block("vb")
        end = b.append_basic_block("ve")
        b.branch(head)
        b.position_at_end(head)
        i = b.phi(i64)
        i.add_incoming(lo, pre)
        acc = b.phi(V)
        acc.add_incoming(carry, pre)
        b.cbranch(b.icmp_signed("<", i, hi), bod, end)
        b.position_at_end(bod)
        nxt = body(b, i, acc)
        i.add_incoming(b.add(i, ir.Constant(i64, 1)), b.block)
        acc.add_incoming(nxt, b.block)
        b.branch(head)
        b.position_at_end(end)
        return acc

    def row(b, o, _):
        wrow = b.gep(W, [b.mul(o, I)])
        srow = b.gep(S, [b.mul(o, nb)])

        def per_t(b, t, _):
            xrow = b.gep(X, [b.mul(t, I)])

            def block(b, k, acc):
                base = b.mul(k, ir.Constant(i64, QB))
                wv = b.load(b.bitcast(b.gep(wrow, [base]), vi8.as_pointer()), align=1)
                xv = b.load(b.bitcast(b.gep(xrow, [base]), V.as_pointer()), align=4)
                sc = b.load(b.gep(srow, [k]))
                sv = b.insert_element(ir.Constant(V, ir.Undefined), sc, ir.Constant(ir.IntType(32), 0))
                sv = b.shuffle_vector(sv, ir.Constant(V, ir.Undefined), ir.Constant(ir.VectorType(ir.IntType(32), QB), [0] * QB))
                p = b.fmul(b.sitofp(wv, V), xv, flags=fast)
                return b.call(fmav, [p, sv, acc])
            tot = vloop(b, ir.Constant(i64, 0), nb, block, zero)
            b.store(b.call(red, [ir.Constant(f32, 0.0), tot], fastmath=fast), b.gep(Y, [b.add(b.mul(t, O), o)]))
        loop(b, ir.Constant(i64, 0), T, per_t)
    loop(b, o0, o1, row)
    b.ret_void()
    addr = eng.compile("; apsi hub q8\n" + str(m), ["apsi_hub_q8_rows"])["apsi_hub_q8_rows"]
    proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    return proto(addr)


class Q8:
    """A matrix as int8 codes [O, I] and one float32 scale per 32 values [O, I/32] (the Q8_0 layout of ggml):
    half the memory of bfloat16, and an error well below bfloat16's own rounding of the activations."""

    def __init__(self, codes, scales):
        self.codes, self.scales = codes, scales
        self.shape = codes.shape
        self.nbytes = codes.nbytes + scales.nbytes

    @staticmethod
    def from_bf16(bits, out=None):
        O, I = bits.shape
        codes = np.empty((O, I), np.int8) if out is None else out[0]
        scales = np.empty((O, I // QB), np.float32) if out is None else out[1]
        step = max(1, (1 << 22) // max(I, 1))
        for a in range(0, O, step):                     # in slices: never a float32 copy of the whole matrix
            w = (np.ascontiguousarray(bits[a:a + step]).astype(np.uint32) << np.uint32(16)).view(np.float32)
            blk = w.reshape(w.shape[0], I // QB, QB)
            amax = np.abs(blk).max(axis=2)
            sc = (amax / 127.0).astype(np.float32)
            inv = np.where(sc > 0, 1.0 / np.where(sc > 0, sc, 1), 0).astype(np.float32)
            codes[a:a + step] = np.clip(np.rint(blk * inv[:, :, None]), -127, 127).astype(np.int8).reshape(-1, I)
            scales[a:a + step] = sc
        return Q8(codes, scales)

    def dense(self):
        O, I = self.shape
        return (self.codes.reshape(O, I // QB, QB).astype(np.float32) * self.scales[:, :, None]).reshape(O, I)


_q8 = None


def q8_available():
    global _q8
    if _q8 is None and available():
        with _lock:
            if _q8 is None:
                try:
                    _q8 = _build_q8()
                except Exception:
                    _q8 = False
    return bool(_q8)


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


# ------------------------------------------------------------------ a native thread team
# Starting work on a Python thread pool costs tens of microseconds per thread, and a token makes about a hundred
# matrix products: on many cores that was a fifth of the time. The team's threads wait in native code (spinning on
# a shared word, then sleeping when idle), so a product starts on every core in about a microsecond.
_S_GEN, _S_DONE, _S_QUIT, _S_PARTS, _S_KIND, _S_W, _S_S, _S_X, _S_Y, _S_I, _S_T, _S_O, _S_SLEEP, _S_STEP = range(14)
_S_LAST = 16


def _build_team(eng, kernels):
    """worker(state, idx, spins) -> 1 to stop, 0 when idle for `spins` checks; wait(state, n): until n are done."""
    import llvmlite.ir as ir
    i8, i16, i64, f32 = ir.IntType(8), ir.IntType(16), ir.IntType(64), ir.FloatType()
    m = ir.Module(name="apsi_hub_team")
    vp = i8.as_pointer()
    ktype = ir.FunctionType(ir.VoidType(), [vp, f32.as_pointer(), f32.as_pointer(), i64, i64, i64, i64, i64])
    qtype = ir.FunctionType(ir.VoidType(), [vp, f32.as_pointer(), f32.as_pointer(), f32.as_pointer(),
                                            i64, i64, i64, i64, i64])
    x86 = eng.triple.startswith(("x86_64", "i686", "i386"))
    pause = ir.Function(m, ir.FunctionType(ir.VoidType(), []), name="llvm.x86.sse2.pause") if x86 else None
    P = i64.as_pointer()

    def slot(b, st, k):
        return b.gep(st, [ir.Constant(i64, k)])

    def aload(b, st, k):
        return b.load_atomic(slot(b, st, k), "acquire", 8)

    # worker
    fn = ir.Function(m, ir.FunctionType(i64, [P, i64, i64]), name="apsi_hub_worker")
    st, idx, spins = fn.args
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    lastp = b.gep(st, [b.add(idx, ir.Constant(i64, _S_LAST))])
    top = b.append_basic_block("top")
    b.branch(top)
    b.position_at_end(top)
    spin_bb = b.append_basic_block("spin")
    work_bb = b.append_basic_block("work")
    idle_bb = b.append_basic_block("idle")
    quit_bb = b.append_basic_block("quit")
    b.branch(spin_bb)
    b.position_at_end(spin_bb)
    n = b.phi(i64)
    n.add_incoming(ir.Constant(i64, 0), top)
    q = aload(b, st, _S_QUIT)
    g = aload(b, st, _S_GEN)
    seen = b.load(lastp)
    newjob = b.icmp_unsigned("!=", g, seen)
    chk_bb = b.append_basic_block("chk")
    b.cbranch(b.icmp_unsigned("!=", q, ir.Constant(i64, 0)), quit_bb, chk_bb)
    b.position_at_end(chk_bb)
    cont_bb = b.append_basic_block("cont")
    b.cbranch(newjob, work_bb, cont_bb)
    b.position_at_end(cont_bb)
    if pause is not None:
        b.call(pause, [])
    n1 = b.add(n, ir.Constant(i64, 1))
    n.add_incoming(n1, cont_bb)
    b.cbranch(b.icmp_signed("<", n1, spins), spin_bb, idle_bb)
    # work: my share of the rows
    b.position_at_end(work_bb)
    b.store(g, lastp)
    parts = b.load(slot(b, st, _S_PARTS))
    run_bb = b.append_basic_block("run")
    fin_bb = b.append_basic_block("fin")
    b.cbranch(b.icmp_signed("<", idx, parts), run_bb, fin_bb)
    b.position_at_end(run_bb)
    step = b.load(slot(b, st, _S_STEP))
    O = b.load(slot(b, st, _S_O))
    a = b.mul(idx, step)
    e0 = b.add(a, step)
    e = b.select(b.icmp_signed("<", e0, O), e0, O)
    I = b.load(slot(b, st, _S_I))
    T = b.load(slot(b, st, _S_T))
    W = b.inttoptr(b.load(slot(b, st, _S_W)), vp)
    X = b.inttoptr(b.load(slot(b, st, _S_X)), f32.as_pointer())
    Y = b.inttoptr(b.load(slot(b, st, _S_Y)), f32.as_pointer())
    kind = b.load(slot(b, st, _S_KIND))
    some = b.append_basic_block("some")
    b.cbranch(b.icmp_signed("<", a, e), some, fin_bb)
    b.position_at_end(some)
    kq_bb = b.append_basic_block("kq")
    kb_bb = b.append_basic_block("kb")
    b.cbranch(b.icmp_unsigned("==", kind, ir.Constant(i64, 1)), kq_bb, kb_bb)
    b.position_at_end(kb_bb)
    kb = b.inttoptr(ir.Constant(i64, kernels[0]), ktype.as_pointer())
    b.call(kb, [W, X, Y, a, e, I, T, O])
    b.branch(fin_bb)
    b.position_at_end(kq_bb)
    kq = b.inttoptr(ir.Constant(i64, kernels[1]), qtype.as_pointer())
    S = b.inttoptr(b.load(slot(b, st, _S_S)), f32.as_pointer())
    b.call(kq, [W, S, X, Y, a, e, I, T, O])
    b.branch(fin_bb)
    b.position_at_end(fin_bb)
    b.atomic_rmw("add", slot(b, st, _S_DONE), ir.Constant(i64, 1), "seq_cst")
    b.branch(top)
    # idle: say so, then look once more (a job may have come in between)
    b.position_at_end(idle_bb)
    b.atomic_rmw("add", slot(b, st, _S_SLEEP), ir.Constant(i64, 1), "seq_cst")
    g2 = b.load_atomic(slot(b, st, _S_GEN), "seq_cst", 8)
    back_bb = b.append_basic_block("back")
    out_bb = b.append_basic_block("out")
    b.cbranch(b.icmp_unsigned("!=", g2, b.load(lastp)), back_bb, out_bb)
    b.position_at_end(back_bb)
    b.atomic_rmw("sub", slot(b, st, _S_SLEEP), ir.Constant(i64, 1), "seq_cst")
    b.branch(top)
    b.position_at_end(out_bb)
    b.ret(ir.Constant(i64, 0))
    b.position_at_end(quit_bb)
    b.ret(ir.Constant(i64, 1))

    # wait until `n` workers are done
    fw = ir.Function(m, ir.FunctionType(ir.VoidType(), [P, i64]), name="apsi_hub_wait")
    st2, want = fw.args
    b = ir.IRBuilder(fw.append_basic_block("entry"))
    loop_bb = b.append_basic_block("l")
    end_bb = b.append_basic_block("e")
    b.branch(loop_bb)
    b.position_at_end(loop_bb)
    d = b.load_atomic(b.gep(st2, [ir.Constant(i64, _S_DONE)]), "acquire", 8)
    more = b.append_basic_block("m")
    b.cbranch(b.icmp_signed("<", d, want), more, end_bb)
    b.position_at_end(more)
    if pause is not None:
        b.call(pause, [])
    b.branch(loop_bb)
    b.position_at_end(end_bb)
    b.ret_void()

    addrs = eng.compile("; apsi hub team\n" + str(m), ["apsi_hub_worker", "apsi_hub_wait"])
    worker = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64)(addrs["apsi_hub_worker"])
    wait = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64)(addrs["apsi_hub_wait"])
    return worker, wait


class _Team:
    SPINS = int(os.environ.get("APSI_HUB_SPIN", "200000"))

    def __init__(self, n):
        from .native import engine as _eng
        eng = _eng.engine()
        kq = ctypes.cast(_q8, ctypes.c_void_p).value if _q8 else ctypes.cast(_fn, ctypes.c_void_p).value
        kb = ctypes.cast(_fn, ctypes.c_void_p).value
        self.worker, self.wait = _build_team(eng, (kb, kq))
        self.state = np.zeros(_S_LAST + n + 8, np.int64)
        self.addr = self.state.ctypes.data
        self.n = n                                       # threads including the caller
        self.pid = os.getpid()                           # a forked child has none of the threads
        self.cond = threading.Condition()
        self.lock = threading.Lock()
        self.threads = []
        for i in range(1, n):
            t = threading.Thread(target=self._run, args=(i,), daemon=True, name=f"apsi-hub-cpu-{i}")
            t.start()
            self.threads.append(t)

    def _run(self, i):
        st = self.state
        while True:
            if self.worker(self.addr, i, self.SPINS):
                return
            with self.cond:
                while st[_S_GEN] == st[_S_LAST + i] and not st[_S_QUIT]:
                    self.cond.wait()
                st[_S_SLEEP] -= 1                        # under the lock: the main thread reads it there too

    def run(self, kind, W, S, X, Y, I, T, O, parts, step):
        st = self.state
        with self.lock:                                  # one product at a time (several models, several threads)
            st[_S_KIND], st[_S_W], st[_S_S], st[_S_X], st[_S_Y] = kind, W, S, X, Y
            st[_S_I], st[_S_T], st[_S_O], st[_S_PARTS], st[_S_STEP] = I, T, O, parts, step
            st[_S_DONE] = 0
            gen = int(st[_S_GEN]) + 1
            ctypes.c_int64.from_address(self.addr + 8 * _S_GEN).value = gen    # publishes the job
            with self.cond:
                if st[_S_SLEEP] > 0:
                    self.cond.notify_all()
            e = min(O, step)
            if e > 0:
                if kind == 1:
                    _q8(W, S, X, Y, 0, e, I, T, O)
                else:
                    _fn(W, X, Y, 0, e, I, T, O)
            self.wait(self.addr, self.n - 1)


_team = None


def _get_team(n):
    global _team
    if _team and _team.pid != os.getpid():              # after fork(): the threads stayed in the parent
        _team = None
    if _team is None and os.environ.get("APSI_HUB_TEAM", "1") not in ("0", "false", "no"):
        q8_available()                                  # takes the lock itself
        with _lock:
            if _team is None:
                try:
                    _team = _Team(n)
                except Exception as e:                  # odd platform: the thread pool instead
                    _team = False
                    global _team_why
                    _team_why = f"{type(e).__name__}: {e}"
    return _team or None


_team_why = None


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
    q = isinstance(w, Q8)
    if T >= MANY:                                       # many rows (a prompt): decoding once + BLAS is faster
        if q:
            out = np.empty((T, O), np.float32)
            step = max(64, (1 << 22) // I)
            for a in range(0, O, step):
                part = Q8(w.codes[a:a + step], w.scales[a:a + step])
                out[:, a:a + step] = xs @ part.dense().T
            return out.reshape(*lead, O)
        wf = (np.ascontiguousarray(w).astype(np.uint32) << np.uint32(16)).view(np.float32)
        return (xs @ wf.T).reshape(*lead, O)
    y = np.empty((T, O), np.float32)
    if q:
        if not q8_available():
            return (xs @ w.dense().T).reshape(*lead, O)
        fn = _q8
        args = (w.codes.ctypes.data, w.scales.ctypes.data)
    else:
        w = np.ascontiguousarray(w)
        fn = _fn
        args = (w.ctypes.data,)
    n = threads()
    parts = max(1, min(n, O // 64 or 1))
    if parts == 1:
        fn(*args, xs.ctypes.data, y.ctypes.data, 0, O, I, T, O)
        return y.reshape(*lead, O)
    team = _get_team(n)
    if team is not None:
        step = (O + parts - 1) // parts
        step = (step + 15) // 16 * 16
        parts = (O + step - 1) // step
        team.run(1 if q else 0, args[0], args[1] if q else 0, xs.ctypes.data, y.ctypes.data, I, T, O, parts, step)
        return y.reshape(*lead, O)
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="apsi-hub-cpu")
    step = (O + parts - 1) // parts
    step = (step + 15) // 16 * 16
    futs = [_pool.submit(fn, *args, xs.ctypes.data, y.ctypes.data, a, min(O, a + step), I, T, O)
            for a in range(0, O, step)]
    for f in futs:
        f.result()
    return y.reshape(*lead, O)
