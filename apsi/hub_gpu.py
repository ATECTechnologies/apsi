"""Hub language models on an NVIDIA GPU, with nothing to install but the display driver.

The whole decoder runs on the card: the weights (copied once, as bfloat16), the key/value cache, the norms, rotary
embeddings, attention and the MLP. Per generated token only the token's embedding goes to the card and the scores of
the next token come back. The kernels are generated with LLVM (already part of Apsi) as PTX, and the driver
(nvcuda.dll / libcuda.so) compiles them for the card that is present, exactly as `@gpu` does: no CUDA Toolkit, no
CuPy, no PyTorch.

bfloat16 is the format most current checkpoints are published in (Qwen, Llama, Gemma, Mistral), so for them the
copy on the GPU is exact and needs half the memory of float32. Sums and activations are float32.

`Emulator` runs the same kernels with NumPy; the tests use it to check everything around the kernels, and the GPU
against it. APSI_NO_GPU=1 (or device="cpu") keeps everything on the processor.
"""
import ctypes
import math
import os
import threading

import numpy as np

_lock = threading.Lock()
_ptx = None

ROWS_PER_BLOCK = 8                      # matrix products: one warp per output row, 8 warps per block
NT = 128                                # threads per block of the other kernels
LOG2E = 1.4426950408889634
KERNELS = ("apsi_kernel", "gemv2", "gemv8n", "gemv2n", "gemv16", "gemv16n", "gemm", "gemm32", "rms", "qkv_prep", "attention", "attn_merge", "qkv_prep_h", "attention_h",
           "act_mul", "lg_prep", "lg_sum", "lg_pick", "embed")                           # apsi_kernel = gemv8 (the module's first entry)
GEMM_ROWS = 8                           # from this many rows on, the tiled product
NTA = 256                               # threads per attention block (8 warps)
MAX_DH = 256                            # head sizes the attention kernel takes


# ======================================================================================== kernels (LLVM -> PTX)
def _cache_path():
    import hashlib
    try:
        import llvmlite
        from .security import config_dir
        with open(__file__, "rb") as fh:
            key = hashlib.sha256(fh.read() + llvmlite.__version__.encode()).hexdigest()[:20]
        return os.path.join(config_dir(), "cache", "gpu", f"hub-kernels-{key}.ptx")
    except Exception:
        return None


def _build_ptx():
    """The kernels as PTX: from memory, else from the disk cache (keyed by this file and LLVM), else built."""
    global _ptx
    if _ptx is not None:
        return _ptx
    path = _cache_path()
    if path and os.environ.get("APSI_NO_CACHE", "").strip() not in ("1", "true", "yes"):
        try:
            with open(path, encoding="ascii") as fh:
                text = fh.read()
            if ".entry apsi_kernel" in text:
                _ptx = text
                return _ptx
        except OSError:
            pass
    text = _generate_ptx()
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="ascii") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except OSError:
            pass
    _ptx = text
    return _ptx


def _generate_ptx():
    import llvmlite.ir as ir
    import llvmlite.binding as llvm
    from .native import engine as _eng

    i32, i64, f32 = ir.IntType(32), ir.IntType(64), ir.FloatType()
    G = lambda t: t.as_pointer(1)                                  # noqa: E731  global memory
    m = ir.Module(name="apsi_hub")
    m.triple = "nvptx64-nvidia-cuda"
    decl = lambda n, r, a: ir.Function(m, ir.FunctionType(r, a), name=n)   # noqa: E731
    tid_f = decl("llvm.nvvm.read.ptx.sreg.tid.x", i32, [])
    cta_f = decl("llvm.nvvm.read.ptx.sreg.ctaid.x", i32, [])
    shfl_f = decl("llvm.nvvm.shfl.sync.bfly.f32", f32, [i32, f32, i32, i32])
    fma_f = decl("llvm.fma.f32", f32, [f32, f32, f32])
    ex2_f = decl("llvm.nvvm.ex2.approx.f", f32, [f32])
    sqrt_f = decl("llvm.sqrt.f32", f32, [f32])
    max_f = decl("llvm.maxnum.f32", f32, [f32, f32])
    bar_f = decl("llvm.nvvm.barrier0", ir.VoidType(), [])
    smem = ir.GlobalVariable(m, ir.ArrayType(f32, 32), "apsi_smem", addrspace=3)
    smem.linkage = "internal"
    smem.initializer = ir.Constant(ir.ArrayType(f32, 32), None)
    C = lambda v: ir.Constant(i32, v)                               # noqa: E731
    F = lambda v: ir.Constant(f32, v)                               # noqa: E731

    class K:
        """A kernel being written: variables live in allocas (promoted to registers by the optimiser)."""

        def __init__(self, name, params):
            fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [t for _, t in params]), name=name)
            fn.calling_convention = "ptx_kernel"
            self.fn = fn
            self.top = fn.append_basic_block("allocas")             # variables; branches to entry at the end
            self.entry = fn.append_basic_block("entry")
            self.a = ir.IRBuilder(self.top)
            self.b = ir.IRBuilder(self.entry)
            self.p = {n: a for (n, _), a in zip(params, fn.args)}
            self.tid = self.b.call(tid_f, [])
            self.cta = self.b.call(cta_f, [])

        def ret(self):
            self.b.ret_void()
            self.a.branch(self.entry)

        def var(self, init):
            v = self.a.alloca(init.type)
            self.b.store(init, v)
            return v

        def ld(self, ptr, idx, align=4):
            return self.b.load(self.b.gep(ptr, [self.b.zext(idx, i64) if idx.type == i32 else idx], inbounds=True),
                               align=align)

        def st(self, val, ptr, idx):
            self.b.store(val, self.b.gep(ptr, [self.b.zext(idx, i64) if idx.type == i32 else idx], inbounds=True),
                         align=4)

        def loop(self, lo, hi, step, body):
            """for i in lo..hi step `step` (i32, unsigned compare)."""
            b = self.b
            iv = self.var(lo)
            head = self.fn.append_basic_block("head")
            bod = self.fn.append_basic_block("body")
            end = self.fn.append_basic_block("end")
            b.branch(head)
            b.position_at_end(head)
            i = b.load(iv)
            b.cbranch(b.icmp_unsigned("<", i, hi), bod, end)
            b.position_at_end(bod)
            body(i)
            b.store(b.add(b.load(iv), step), iv)
            b.branch(head)
            b.position_at_end(end)

        def warp_reduce(self, v, op):
            for k in (16, 8, 4, 2, 1):
                o = self.b.call(shfl_f, [C(-1), v, C(k), C(31)])
                v = op(v, o)
            return v

        def block_reduce(self, v, op, nthreads):
            """Reduce over the whole block; every thread gets the result. All threads must call it."""
            b = self.b
            v = self.warp_reduce(v, op)
            lane = b.and_(self.tid, C(31))
            warp = b.lshr(self.tid, C(5))
            with b.if_then(b.icmp_unsigned("==", lane, C(0))):
                b.store(v, b.gep(smem, [C(0), warp], inbounds=True))
            b.call(bar_f, [])
            r = b.load(b.gep(smem, [C(0), C(0)], inbounds=True))
            for w in range(1, nthreads // 32):
                r = op(r, b.load(b.gep(smem, [C(0), C(w)], inbounds=True)))
            b.call(bar_f, [])
            return r

    fadd = lambda k: (lambda x, y: k.b.fadd(x, y))                  # noqa: E731
    fmax = lambda k: (lambda x, y: k.b.call(max_f, [x, y]))         # noqa: E731

    def exp(k, x):
        return k.b.call(ex2_f, [k.b.fmul(x, F(LOG2E))])

    # ---------------------------------------------------------------- matrix product, any number of rows
    # y[t, o] (+)= sum_i W[o, i] x[t, i] (+ bias[o]); W is bfloat16, 2 (gemv2) or 8 (gemv8) per lane and step
    def gemv(name, per, norm=False):
        """norm=True: the input row is RMS-normalised on the fly (x / rms(x) * (LN + off)): no separate norm pass."""
        params = [("W", G(i32)), ("X", G(f32)), ("Y", G(f32)), ("B", G(f32)), ("O", i32), ("I", i32),
                  ("T", i32), ("has_b", i32), ("acc", i32)]
        if norm:
            params += [("LN", G(f32)), ("eps", f32), ("off", f32)]
        k = K(name, params)
        b, p = k.b, k.p
        lane = b.and_(k.tid, C(31))
        o = b.add(b.mul(k.cta, C(ROWS_PER_BLOCK)), b.lshr(k.tid, C(5)))
        with b.if_then(b.icmp_unsigned("<", o, p["O"])):
            n = b.udiv(p["I"], C(per))                               # steps of `per` weights
            wrow = b.mul(b.zext(o, i64), b.zext(b.udiv(p["I"], C(2)), i64))     # in u32 words

            def per_t(t):
                xrow = b.mul(b.zext(t, i64), b.zext(p["I"], i64))
                acc = k.var(F(0.0))
                if norm:
                    ss = k.var(F(0.0))

                    def sq(i):
                        x = k.ld(p["X"], b.add(xrow, b.zext(i, i64)))
                        b.store(b.call(fma_f, [x, x, b.load(ss)]), ss)
                    k.loop(lane, p["I"], C(32), sq)
                    tot = k.warp_reduce(b.load(ss), fadd(k))
                    fac = b.fdiv(F(1.0), b.call(sqrt_f, [b.fadd(b.fdiv(tot, b.uitofp(p["I"], f32)), p["eps"])]))

                def step(i):
                    i64v = b.zext(i, i64)
                    if per == 2:
                        words = [k.ld(p["W"], b.add(wrow, i64v))]
                    else:                               # per // 8 loads of 16 bytes, all in flight at once
                        words = []
                        for h in range(per // 8):
                            vp = b.bitcast(b.gep(p["W"], [b.add(wrow, b.add(b.mul(i64v, ir.Constant(i64, per // 2)),
                                                                            ir.Constant(i64, 4 * h)))], inbounds=True),
                                           ir.VectorType(i32, 4).as_pointer(1))
                            vec = b.load(vp, align=16)
                            words += [b.extract_element(vec, C(j)) for j in range(4)]
                    xbase = b.add(xrow, b.mul(i64v, ir.Constant(i64, per)))
                    if per == 2:
                        xs = [k.ld(p["X"], xbase), k.ld(p["X"], b.add(xbase, ir.Constant(i64, 1)))]
                    else:
                        xs = []
                        for h in range(per // 4):
                            xp = b.bitcast(b.gep(p["X"], [b.add(xbase, ir.Constant(i64, 4 * h))], inbounds=True),
                                           ir.VectorType(f32, 4).as_pointer(1))
                            xv = b.load(xp, align=16)
                            xs += [b.extract_element(xv, C(j)) for j in range(4)]
                    if norm:
                        lbase = b.mul(i64v, ir.Constant(i64, per))
                        if per == 2:
                            ls = [k.ld(p["LN"], lbase), k.ld(p["LN"], b.add(lbase, ir.Constant(i64, 1)))]
                        else:
                            ls = []
                            for h in range(per // 4):
                                lp = b.bitcast(b.gep(p["LN"], [b.add(lbase, ir.Constant(i64, 4 * h))], inbounds=True),
                                               ir.VectorType(f32, 4).as_pointer(1))
                                lv = b.load(lp, align=16)
                                ls += [b.extract_element(lv, C(j)) for j in range(4)]
                        xs = [b.fmul(b.fmul(x, fac), b.fadd(l_, p["off"])) for x, l_ in zip(xs, ls)]
                    a = b.load(acc)
                    for j, w in enumerate(words):
                        lo = b.bitcast(b.shl(w, C(16)), f32)
                        hi = b.bitcast(b.and_(w, C(-65536)), f32)
                        a = b.call(fma_f, [lo, xs[2 * j], a])
                        a = b.call(fma_f, [hi, xs[2 * j + 1], a])
                    b.store(a, acc)
                k.loop(lane, n, C(32), step)
                s = k.warp_reduce(b.load(acc), fadd(k))
                with b.if_then(b.icmp_unsigned("==", lane, C(0))):
                    yi = b.add(b.mul(b.zext(t, i64), b.zext(p["O"], i64)), b.zext(o, i64))
                    out = k.var(s)
                    with b.if_then(b.icmp_unsigned("!=", p["has_b"], C(0))):
                        b.store(b.fadd(b.load(out), k.ld(p["B"], o)), out)
                    with b.if_then(b.icmp_unsigned("!=", p["acc"], C(0))):
                        b.store(b.fadd(b.load(out), k.ld(p["Y"], yi)), out)
                    k.st(b.load(out), p["Y"], yi)
            k.loop(C(0), p["T"], C(1), per_t)
        k.ret()

    gemv("apsi_kernel", 8)
    gemv("gemv2", 2)
    gemv("gemv8n", 8, norm=True)
    gemv("gemv2n", 2, norm=True)
    gemv("gemv16", 16)
    gemv("gemv16n", 16, norm=True)

    # ---------------------------------------------------------------- many rows (prompts): tiled product
    # Y[T, O] (+)= X[T, I] W[O, I]^T (+ bias): 64 x 64 tiles, 16 deep, through shared memory; 4 x 4 per thread
    BN, BK = 64, 16
    xs_sh = ir.GlobalVariable(m, ir.ArrayType(f32, BK * 64), "apsi_xs", addrspace=3)
    ws_sh = ir.GlobalVariable(m, ir.ArrayType(f32, BK * BN), "apsi_ws", addrspace=3)
    for g_ in (xs_sh, ws_sh):
        g_.linkage = "internal"
        g_.initializer = ir.Constant(g_.type.pointee, None)

    def gemm(name, BM):
        """BM rows of X per tile (64, or 32 for few tiles: more blocks for the SMs); MR = BM / 16 per thread."""
        MR = BM // 16
        k = K(name, [("W", G(i32)), ("X", G(f32)), ("Y", G(f32)), ("B", G(f32)), ("O", i32), ("I", i32),
                     ("T", i32), ("has_b", i32), ("acc", i32)])
        b, p = k.b, k.p
        nbn = b.udiv(b.add(p["O"], C(BN - 1)), C(BN))
        n0 = b.mul(b.urem(k.cta, nbn), C(BN))
        m0 = b.mul(b.udiv(k.cta, nbn), C(BM))
        tx = b.and_(k.tid, C(15))                  # 16 x 16 threads, MR x 4 outputs each
        ty = b.lshr(k.tid, C(4))
        xper = BM * BK // 256                      # X values each thread loads per tile
        xlr = b.udiv(k.tid, C(BK // xper))
        xlk = b.mul(b.urem(k.tid, C(BK // xper)), C(xper))
        lr = b.lshr(k.tid, C(2))                   # W: row 0..63 of the tile, 4 consecutive k
        lk = b.mul(b.and_(k.tid, C(3)), C(4))
        accs = [[k.var(F(0.0)) for _ in range(4)] for _ in range(MR)]
        half_i = b.udiv(p["I"], C(2))

        def tile(kb):
            xm = b.add(m0, xlr)
            xin = b.icmp_unsigned("<", xm, p["T"])
            xrow = b.mul(b.zext(xm, i64), b.zext(p["I"], i64))
            for q in range(xper):
                kk = b.add(b.add(kb, xlk), C(q))
                ok = b.and_(xin, b.icmp_unsigned("<", kk, p["I"]))
                v = k.var(F(0.0))
                with b.if_then(ok):
                    b.store(k.ld(p["X"], b.add(xrow, b.zext(kk, i64))), v)
                b.store(b.load(v), b.gep(xs_sh, [C(0), b.add(b.mul(b.add(xlk, C(q)), C(BM)), xlr)], inbounds=True))
            wn = b.add(n0, lr)
            win = b.icmp_unsigned("<", wn, p["O"])
            wrow = b.mul(b.zext(wn, i64), b.zext(half_i, i64))
            for q in range(2):
                kk = b.add(b.add(kb, lk), C(2 * q))
                ok = b.and_(win, b.icmp_unsigned("<", kk, p["I"]))
                wv = k.var(C(0))
                with b.if_then(ok):
                    b.store(k.ld(p["W"], b.add(wrow, b.zext(b.udiv(kk, C(2)), i64))), wv)
                w = b.load(wv)
                lo = b.bitcast(b.shl(w, C(16)), f32)
                hi = b.bitcast(b.and_(w, C(-65536)), f32)
                for r_, val in ((0, lo), (1, hi)):
                    b.store(val, b.gep(ws_sh, [C(0), b.add(b.mul(b.add(lk, C(2 * q + r_)), C(BN)), lr)],
                                       inbounds=True))
            b.call(bar_f, [])
            for kq in range(BK):
                av = b.load(b.bitcast(b.gep(xs_sh, [C(0), b.add(C(kq * BM), b.mul(ty, C(MR)))], inbounds=True),
                                      ir.VectorType(f32, MR).as_pointer(3)), align=4 * MR)
                bv = b.load(b.bitcast(b.gep(ws_sh, [C(0), b.add(C(kq * BN), b.mul(tx, C(4)))], inbounds=True),
                                      ir.VectorType(f32, 4).as_pointer(3)), align=16)
                a_ = [b.extract_element(av, C(i)) for i in range(MR)]
                b_ = [b.extract_element(bv, C(j)) for j in range(4)]
                for i in range(MR):
                    for j in range(4):
                        b.store(b.call(fma_f, [a_[i], b_[j], b.load(accs[i][j])]), accs[i][j])
            b.call(bar_f, [])
        k.loop(C(0), p["I"], C(BK), tile)
        for i in range(MR):
            mm = b.add(m0, b.add(b.mul(ty, C(MR)), C(i)))
            for j in range(4):
                nn = b.add(n0, b.add(b.mul(tx, C(4)), C(j)))
                with b.if_then(b.and_(b.icmp_unsigned("<", mm, p["T"]), b.icmp_unsigned("<", nn, p["O"]))):
                    yi = b.add(b.mul(b.zext(mm, i64), b.zext(p["O"], i64)), b.zext(nn, i64))
                    out = k.var(b.load(accs[i][j]))
                    with b.if_then(b.icmp_unsigned("!=", p["has_b"], C(0))):
                        b.store(b.fadd(b.load(out), k.ld(p["B"], nn)), out)
                    with b.if_then(b.icmp_unsigned("!=", p["acc"], C(0))):
                        b.store(b.fadd(b.load(out), k.ld(p["Y"], yi)), out)
                    k.st(b.load(out), p["Y"], yi)
        k.ret()

    gemm("gemm", 64)
    gemm("gemm32", 32)

    # ---------------------------------------------------------------- RMSNorm, one block per row
    k = K("rms", [("X", G(f32)), ("Y", G(f32)), ("W", G(f32)), ("n", i32), ("eps", f32), ("off", f32)])
    b, p = k.b, k.p
    base = b.mul(b.zext(k.cta, i64), b.zext(p["n"], i64))
    ss = k.var(F(0.0))

    def sq(d):
        x = k.ld(p["X"], b.add(base, b.zext(d, i64)))
        b.store(b.call(fma_f, [x, x, b.load(ss)]), ss)
    k.loop(k.tid, p["n"], C(NT), sq)
    tot = k.block_reduce(b.load(ss), fadd(k), NT)
    f = b.fdiv(F(1.0), b.call(sqrt_f, [b.fadd(b.fdiv(tot, b.uitofp(p["n"], f32)), p["eps"])]))

    def norm(d):
        i = b.add(base, b.zext(d, i64))
        k.st(b.fmul(b.fmul(k.ld(p["X"], i), f), b.fadd(k.ld(p["W"], d), p["off"])), p["Y"], i)
    k.loop(k.tid, p["n"], C(NT), norm)
    k.ret()

    part = ir.GlobalVariable(m, ir.ArrayType(f32, 8 * MAX_DH), "apsi_att", addrspace=3)
    part.linkage = "internal"
    part.initializer = ir.Constant(part.type.pointee, None)

    def st_kv(k, val, ptr, idx):
        """A key / value into the cache: float32, or float16 (half the memory and reads)."""
        if ptr.type.pointee == ir.HalfType():
            k.b.store(k.b.fptrunc(val, ir.HalfType()), k.b.gep(ptr, [k.b.zext(idx, i64) if idx.type == i32 else idx],
                                                              inbounds=True), align=2)
        else:
            k.st(val, ptr, idx)

    def ld_kv(k, ptr, idx):
        if ptr.type.pointee == ir.HalfType():
            return k.b.fpext(k.ld(ptr, idx, align=2), f32)
        return k.ld(ptr, idx)

    def kv_kernels(suffix, f16):
        kvt = ir.HalfType() if f16 else f32
        # ---------------------------------------------------------------- q / k / v: norm (Qwen3), RoPE, cache
        k = K("qkv_prep" + suffix, [("QKV", G(f32)), ("Q", G(f32)), ("Kc", G(kvt)), ("Vc", G(kvt)), ("COS", G(f32)),
                           ("SIN", G(f32)), ("QN", G(f32)), ("KN", G(f32)), ("has_n", i32), ("nh", i32), ("kvh", i32),
                           ("dh", i32), ("rot", i32), ("start", i32), ("cap", i32), ("eps", f32)])
        b, p = k.b, k.p
        per = b.add(p["nh"], b.mul(p["kvh"], C(2)))
        t = b.udiv(k.cta, per)
        r = b.urem(k.cta, per)
        dh = p["dh"]
        src = b.add(b.mul(b.zext(t, i64), b.zext(b.mul(per, dh), i64)), b.zext(b.mul(r, dh), i64))
        pos = b.add(p["start"], t)
        is_v = b.icmp_unsigned(">=", r, b.add(p["nh"], p["kvh"]))
        with b.if_else(is_v) as (then, other):
            with then:
                kh = b.sub(r, b.add(p["nh"], p["kvh"]))
                dst = b.mul(b.add(b.mul(b.zext(kh, i64), b.zext(p["cap"], i64)), b.zext(pos, i64)), b.zext(dh, i64))
                k.loop(k.tid, dh, C(NT), lambda d: st_kv(k, k.ld(p["QKV"], b.add(src, b.zext(d, i64))), p["Vc"],
                                                           b.add(dst, b.zext(d, i64))))
            with other:
                is_q = b.icmp_unsigned("<", r, p["nh"])
                wn = b.select(is_q, p["QN"], p["KN"])
                fac = k.var(F(1.0))
                with b.if_then(b.icmp_unsigned("!=", p["has_n"], C(0))):
                    ss = k.var(F(0.0))

                    def sq2(d):
                        x = k.ld(p["QKV"], b.add(src, b.zext(d, i64)))
                        b.store(b.call(fma_f, [x, x, b.load(ss)]), ss)
                    k.loop(k.tid, dh, C(NT), sq2)
                    tot = k.block_reduce(b.load(ss), fadd(k), NT)
                    b.store(b.fdiv(F(1.0), b.call(sqrt_f, [b.fadd(b.fdiv(tot, b.uitofp(dh, f32)), p["eps"])])), fac)
                facv = b.load(fac)
                half = b.udiv(p["rot"], C(2))
                has_n = b.icmp_unsigned("!=", p["has_n"], C(0))

                def val(d):
                    x = b.fmul(k.ld(p["QKV"], b.add(src, b.zext(d, i64))), facv)
                    return b.select(has_n, b.fmul(x, k.ld(wn, b.select(has_n, d, C(0)))), x)

                def one(d):
                    out = k.var(val(d))
                    with b.if_then(b.icmp_unsigned("<", d, p["rot"])):
                        first = b.icmp_unsigned("<", d, half)
                        j = b.select(first, d, b.sub(d, half))
                        ci = b.add(b.mul(pos, half), j)          # tables indexed by absolute position
                        c, s = k.ld(p["COS"], ci), k.ld(p["SIN"], ci)
                        partner = val(b.select(first, b.add(d, half), b.sub(d, half)))
                        me = val(d)
                        b.store(b.select(first, b.fsub(b.fmul(me, c), b.fmul(partner, s)),
                                         b.fadd(b.fmul(me, c), b.fmul(partner, s))), out)
                    v = b.load(out)
                    with b.if_else(is_q) as (qt, kt):
                        with qt:
                            qi = b.add(b.mul(b.zext(b.add(b.mul(t, p["nh"]), r), i64), b.zext(dh, i64)), b.zext(d, i64))
                            k.st(v, p["Q"], qi)
                        with kt:
                            kh = b.sub(r, p["nh"])
                            ki = b.add(b.mul(b.add(b.mul(b.zext(kh, i64), b.zext(p["cap"], i64)), b.zext(pos, i64)),
                                             b.zext(dh, i64)), b.zext(d, i64))
                            st_kv(k, v, p["Kc"], ki)
                k.loop(k.tid, dh, C(NT), one)
        k.ret()

        # ---------------------------------------------------------------- attention
        # One block per (position, head, split of the keys). Scores: a thread per key (256 keys at once), the head
        # dimension read 4 at a time. Weighted values: a warp per key, lanes across the dimension, 4 keys per step.
        # With one split the block writes the output; with several it writes its partial result (max, sum, vector)
        # and attn_merge combines them (long contexts: all SMs busy instead of one per head).
        k = K("attention" + suffix, [("Q", G(f32)), ("Kc", G(kvt)), ("Vc", G(kvt)), ("O", G(f32)), ("S", G(f32)),
                            ("PM", G(f32)), ("PL", G(f32)), ("PO", G(f32)), ("nh", i32), ("kvh", i32), ("dh", i32),
                            ("start", i32), ("cap", i32), ("window", i32), ("scale", f32), ("total", i32),
                            ("splits", i32)])
        b, p = k.b, k.p
        v4p = lambda ptr, idx: b.load(b.bitcast(b.gep(ptr, [idx], inbounds=True), ir.VectorType(f32, 4).as_pointer(1)),
                                      align=16)                                     # noqa: E731

        def v4k(ptr, idx):
            if not f16:
                return v4p(ptr, idx)
            v = b.load(b.bitcast(b.gep(ptr, [idx], inbounds=True), ir.VectorType(kvt, 4).as_pointer(1)), align=8)
            out = ir.Constant(ir.VectorType(f32, 4), None)
            for e_ in range(4):
                out = b.insert_element(out, b.fpext(b.extract_element(v, C(e_)), f32), C(e_))
            return out
        lane = b.and_(k.tid, C(31))
        warp = b.lshr(k.tid, C(5))
        nw = NTA // 32
        sp = b.urem(k.cta, p["splits"])
        th = b.udiv(k.cta, p["splits"])                # (position, head)
        t = b.udiv(th, p["nh"])
        h = b.urem(th, p["nh"])
        kh = b.udiv(h, b.udiv(p["nh"], p["kvh"]))
        dh = p["dh"]
        qpos = b.add(p["start"], t)
        n = b.add(qpos, C(1))
        lo = k.var(C(0))
        with b.if_then(b.icmp_unsigned("!=", p["window"], C(0))):
            with b.if_then(b.icmp_unsigned(">=", qpos, p["window"])):
                b.store(b.add(b.sub(qpos, p["window"]), C(1)), lo)
        lov = b.load(lo)
        span = b.sub(n, lov)
        per = b.udiv(b.add(span, b.sub(p["splits"], C(1))), p["splits"])
        j0 = b.add(lov, b.mul(sp, per))
        j1v = k.var(b.add(j0, per))
        with b.if_then(b.icmp_unsigned(">", b.load(j1v), n)):
            b.store(n, j1v)
        j1 = b.load(j1v)
        srow = b.mul(b.zext(th, i64), b.zext(p["total"], i64))
        qrow = b.mul(b.zext(th, i64), b.zext(dh, i64))
        kbase = b.mul(b.zext(kh, i64), b.zext(p["cap"], i64))
        d4 = b.lshr(dh, C(2))
        mx = k.var(F(-3.0e38))

        def score(j):
            krow = b.mul(b.add(kbase, b.zext(j, i64)), b.zext(dh, i64))
            acc = k.var(F(0.0))

            def dot4(c):
                off = b.mul(b.zext(c, i64), ir.Constant(i64, 4))
                qv = v4p(p["Q"], b.add(qrow, off))
                kv = v4k(p["Kc"], b.add(krow, off))
                a_ = b.load(acc)
                for e_ in range(4):
                    a_ = b.call(fma_f, [b.extract_element(qv, C(e_)), b.extract_element(kv, C(e_)), a_])
                b.store(a_, acc)
            k.loop(C(0), d4, C(1), dot4)
            s_ = b.fmul(b.load(acc), p["scale"])
            k.st(s_, p["S"], b.add(srow, b.zext(j, i64)))
            b.store(b.call(max_f, [b.load(mx), s_]), mx)
        k.loop(b.add(j0, k.tid), j1, C(NTA), score)
        M = k.block_reduce(b.load(mx), fmax(k), NTA)
        sm = k.var(F(0.0))

        def expo(j):
            i = b.add(srow, b.zext(j, i64))
            e_ = exp(k, b.fsub(k.ld(p["S"], i), M))
            k.st(e_, p["S"], i)
            b.store(b.fadd(b.load(sm), e_), sm)
        k.loop(b.add(j0, k.tid), j1, C(NTA), expo)
        Z = k.block_reduce(b.load(sm), fadd(k), NTA)
        chunks = MAX_DH // 32
        accs = [k.var(F(0.0)) for _ in range(chunks)]
        # 4 keys per step for the warp (independent loads in flight), then the remainder one by one
        j4 = b.add(j0, b.mul(b.udiv(b.sub(j1, b.select(b.icmp_unsigned(">", j1, j0), j0, j1)), C(4)), C(4)))

        def weigh_n(jb, count):
            ws_ = [k.ld(p["S"], b.add(srow, b.zext(b.add(jb, C(q)), i64))) for q in range(count)]
            rows = [b.mul(b.add(kbase, b.zext(b.add(jb, C(q)), i64)), b.zext(dh, i64)) for q in range(count)]
            for c in range(chunks):
                d = b.add(lane, C(32 * c))
                with b.if_then(b.icmp_unsigned("<", d, dh)):
                    vals = [ld_kv(k, p["Vc"], b.add(r_, b.zext(d, i64))) for r_ in rows]
                    a_ = b.load(accs[c])
                    for w_, v_ in zip(ws_, vals):
                        a_ = b.call(fma_f, [w_, v_, a_])
                    b.store(a_, accs[c])
        k.loop(b.add(j0, b.mul(warp, C(4))), j4, C(4 * nw), lambda jb: weigh_n(jb, 4))
        k.loop(b.add(j4, warp), j1, C(nw), lambda jb: weigh_n(jb, 1))
        for c in range(chunks):
            d = b.add(lane, C(32 * c))
            with b.if_then(b.icmp_unsigned("<", d, dh)):
                b.store(b.load(accs[c]), b.gep(part, [C(0), b.add(b.mul(warp, C(MAX_DH)), d)], inbounds=True))
        b.call(bar_f, [])
        single = b.icmp_unsigned("==", p["splits"], C(1))
        pidx = b.add(b.mul(th, p["splits"]), sp)

        def outd(d):
            tot = F(0.0)
            for w_ in range(nw):
                tot = b.fadd(tot, b.load(b.gep(part, [C(0), b.add(C(w_ * MAX_DH), d)], inbounds=True)))
            with b.if_else(single) as (one, many):
                with one:
                    k.st(b.fdiv(tot, Z), p["O"], b.add(qrow, b.zext(d, i64)))
                with many:
                    k.st(tot, p["PO"], b.add(b.mul(b.zext(pidx, i64), b.zext(dh, i64)), b.zext(d, i64)))
        k.loop(k.tid, dh, C(NTA), outd)
        with b.if_then(b.and_(b.icmp_unsigned("==", k.tid, C(0)), b.not_(single))):
            k.st(M, p["PM"], pidx)
            k.st(Z, p["PL"], pidx)
        k.ret()

    kv_kernels("", False)
    kv_kernels("_h", True)

    # combine the splits: o = sum_s exp(m_s - M) o_s / sum_s exp(m_s - M) l_s
    k = K("attn_merge", [("O", G(f32)), ("PM", G(f32)), ("PL", G(f32)), ("PO", G(f32)), ("dh", i32),
                         ("splits", i32)])
    b, p = k.b, k.p
    base = b.mul(k.cta, p["splits"])
    mm_ = k.var(F(-3.0e38))
    k.loop(C(0), p["splits"], C(1), lambda s_: b.store(b.call(max_f, [b.load(mm_), k.ld(p["PM"], b.add(base, s_))]),
                                                          mm_))
    Mx = b.load(mm_)
    ll = k.var(F(0.0))
    k.loop(C(0), p["splits"], C(1), lambda s_: b.store(b.call(fma_f, [
        k.ld(p["PL"], b.add(base, s_)), exp(k, b.fsub(k.ld(p["PM"], b.add(base, s_)), Mx)), b.load(ll)]), ll))
    L_ = b.load(ll)

    def merge(d):
        acc = k.var(F(0.0))
        k.loop(C(0), p["splits"], C(1), lambda s_: b.store(b.call(fma_f, [
            k.ld(p["PO"], b.add(b.mul(b.zext(b.add(base, s_), i64), b.zext(p["dh"], i64)), b.zext(d, i64))),
            exp(k, b.fsub(k.ld(p["PM"], b.add(base, s_)), Mx)), b.load(acc)]), acc))
        k.st(b.fdiv(b.load(acc), L_), p["O"], b.add(b.mul(b.zext(k.cta, i64), b.zext(p["dh"], i64)), b.zext(d, i64)))
    k.loop(k.tid, p["dh"], C(NT), merge)
    k.ret()

    # ---------------------------------------------------------------- embedding rows from a bfloat16 matrix
    k = K("embed", [("E", G(i32)), ("IDS", G(i32)), ("X", G(f32)), ("H", i32), ("scale", f32)])
    b, p = k.b, k.p
    row = k.ld(p["IDS"], k.cta)
    rbase = b.mul(b.zext(row, i64), b.zext(b.udiv(p["H"], C(2)), i64))
    xbase = b.mul(b.zext(k.cta, i64), b.zext(p["H"], i64))

    def emb(i):
        w = k.ld(p["E"], b.add(rbase, b.zext(b.lshr(i, C(1)), i64)))
        odd = b.icmp_unsigned("!=", b.and_(i, C(1)), C(0))
        v = b.bitcast(b.select(odd, b.and_(w, C(-65536)), b.shl(w, C(16))), f32)
        k.st(b.fmul(v, p["scale"]), p["X"], b.add(xbase, b.zext(i, i64)))
    k.loop(k.tid, p["H"], C(256), emb)
    k.ret()

    # ---------------------------------------------------------------- next-token choice, on the card
    # lg_prep: repetition penalty and temperature, block maxima; lg_sum: the softmax denominator per block;
    # lg_pick: every token within `delta` of the maximum (the only ones top-k / top-p can choose), compacted.
    k = K("lg_prep", [("LOG", G(f32)), ("SEEN", G(f32)), ("OUT", G(f32)), ("PM", G(f32)), ("V", i32), ("rp", f32),
                      ("inv_t", f32), ("chunk", i32)])
    b, p = k.b, k.p
    base = b.mul(k.cta, p["chunk"])
    end = b.select(b.icmp_unsigned("<", b.add(base, p["chunk"]), p["V"]), b.add(base, p["chunk"]), p["V"])
    mx = k.var(F(-3.0e38))

    def prep(i):
        l_ = k.ld(p["LOG"], i)
        pen = b.select(b.fcmp_ordered(">", l_, F(0.0)), b.fdiv(l_, p["rp"]), b.fmul(l_, p["rp"]))
        v = b.fmul(b.select(b.fcmp_unordered("!=", k.ld(p["SEEN"], i), F(0.0)), pen, l_), p["inv_t"])
        k.st(v, p["OUT"], i)
        b.store(b.call(max_f, [b.load(mx), v]), mx)
    k.loop(b.add(base, k.tid), end, C(256), prep)
    M = k.block_reduce(b.load(mx), fmax(k), 256)
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(M, p["PM"], k.cta)
    k.ret()

    def gmax_of(k, p):
        g = k.var(F(-3.0e38))
        k.loop(C(0), p["nb"], C(1), lambda j: k.b.store(k.b.call(max_f, [k.b.load(g), k.ld(p["PM"], j)]), g))
        return k.b.load(g)

    k = K("lg_sum", [("OUT", G(f32)), ("PM", G(f32)), ("PS", G(f32)), ("V", i32), ("nb", i32), ("chunk", i32)])
    b, p = k.b, k.p
    g = gmax_of(k, p)
    base = b.mul(k.cta, p["chunk"])
    end = b.select(b.icmp_unsigned("<", b.add(base, p["chunk"]), p["V"]), b.add(base, p["chunk"]), p["V"])
    sm = k.var(F(0.0))
    k.loop(b.add(base, k.tid), end, C(256), lambda i: b.store(b.fadd(b.load(sm), exp(k, b.fsub(k.ld(p["OUT"], i), g))), sm))
    Z = k.block_reduce(b.load(sm), fadd(k), 256)
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(Z, p["PS"], k.cta)
    k.ret()

    k = K("lg_pick", [("OUT", G(f32)), ("PM", G(f32)), ("CNT", G(i32)), ("CI", G(i32)), ("CV", G(f32)), ("V", i32),
                      ("nb", i32), ("chunk", i32), ("delta", f32), ("cap", i32)])
    b, p = k.b, k.p
    g = gmax_of(k, p)
    thr = b.fsub(g, p["delta"])
    base = b.mul(k.cta, p["chunk"])
    end = b.select(b.icmp_unsigned("<", b.add(base, p["chunk"]), p["V"]), b.add(base, p["chunk"]), p["V"])

    def pick(i):
        v = k.ld(p["OUT"], i)
        with b.if_then(b.fcmp_ordered(">=", v, thr)):
            slot = b.atomic_rmw("add", b.gep(p["CNT"], [ir.Constant(i64, 0)], inbounds=True), C(1), "monotonic")
            with b.if_then(b.icmp_unsigned("<", slot, p["cap"])):
                k.st(i, p["CI"], slot)
                k.st(v, p["CV"], slot)
    k.loop(b.add(base, k.tid), end, C(256), pick)
    k.ret()

    # ---------------------------------------------------------------- act(gate) * up
    k = K("act_mul", [("GU", G(f32)), ("M", G(f32)), ("I", i32), ("n", i32), ("kind", i32)])
    b, p = k.b, k.p
    idx = b.add(b.mul(k.cta, C(256)), k.tid)
    with b.if_then(b.icmp_unsigned("<", idx, p["n"])):
        t = b.udiv(idx, p["I"])
        i = b.urem(idx, p["I"])
        gi = b.add(b.mul(b.zext(t, i64), b.zext(b.mul(p["I"], C(2)), i64)), b.zext(i, i64))
        g = k.ld(p["GU"], gi)
        u = k.ld(p["GU"], b.add(gi, b.zext(p["I"], i64)))
        silu = b.fdiv(g, b.fadd(F(1.0), exp(k, b.fsub(F(0.0), g))))
        z = b.fmul(F(math.sqrt(2.0 / math.pi)), b.fadd(g, b.fmul(F(0.044715), b.fmul(g, b.fmul(g, g)))))
        th = b.fsub(F(1.0), b.fdiv(F(2.0), b.fadd(exp(k, b.fmul(F(2.0), z)), F(1.0))))
        gel = b.fmul(b.fmul(F(0.5), g), b.fadd(F(1.0), th))
        a = b.select(b.icmp_unsigned("==", p["kind"], C(0)), silu, gel)
        k.st(b.fmul(a, u), p["M"], idx)
    k.ret()

    _eng.engine()                                   # initialises LLVM
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    mod = llvm.parse_assembly(str(m))
    mod.verify()
    tm = llvm.Target.from_triple("nvptx64-nvidia-cuda").create_target_machine(cpu="sm_75", opt=3)
    mod.data_layout = str(tm.target_data)
    pb = llvm.create_pass_builder(tm, llvm.create_pipeline_tuning_options(speed_level=3))
    pb.getModulePassManager().run(mod, pb)
    return tm.emit_assembly(mod)


def to_bf16_pairs(w):
    """float32 [O, I] (I even) -> uint32 [O, I/2], each holding two bfloat16 (rounded to nearest even)."""
    w = np.ascontiguousarray(w, np.float32)
    flat = w.reshape(-1).view(np.uint32)
    out = np.empty(flat.size, np.uint16)
    step = 1 << 22                                   # in pieces: small temporaries, stays in cache
    for a in range(0, flat.size, step):
        u = flat[a:a + step]
        t = u >> np.uint32(16)
        t &= np.uint32(1)
        t += u
        t += np.uint32(0x7FFF)
        t >>= np.uint32(16)
        nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
        if nan.any():
            t[nan] = 0x7FC0
        out[a:a + step] = t
    return out.reshape(w.shape).view(np.uint32)


def from_bf16_pairs(p, shape):
    return (np.ascontiguousarray(p).view(np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(shape)


# ======================================================================================== buffers
class Buf:
    """Device memory: `n` elements of float32 (or uint32 words of bfloat16 pairs). `a` holds them in the emulator."""
    __slots__ = ("ptr", "n", "a")

    def __init__(self, ptr, n, a=None):
        self.ptr, self.n, self.a = ptr, n, a


class Matrix:
    __slots__ = ("buf", "rows", "cols")

    def __init__(self, buf, rows, cols):
        self.buf, self.rows, self.cols = buf, rows, cols


class _Base:
    full = True                                     # can run a whole decoder (hub_models.Decoder uses it)

    def __init__(self):
        self.used = 0
        self._own = []
        self._ws = {}

    def matrix(self, w):
        """A [rows, cols] matrix on the device as bfloat16: float32 is rounded; uint16 is taken as bfloat16 bits."""
        rows, cols = w.shape
        if cols % 2:
            return None                             # odd widths stay on the processor
        if w.dtype == np.uint16:
            pairs = np.ascontiguousarray(w).reshape(-1).view(np.uint32)
        else:
            pairs = to_bf16_pairs(np.asarray(w, np.float32)).reshape(-1)
        buf = self.alloc(pairs.size)
        self.put(buf, pairs)
        return Matrix(buf, rows, cols)

    upload = matrix                                 # the older name

    def vec(self, a):
        a = np.ascontiguousarray(np.asarray(a, np.float32).reshape(-1))
        buf = self.alloc(max(1, a.size))
        self.put(buf, a)
        return buf

    def ws(self, name, n):
        """A scratch buffer of at least n float32 that grows as needed."""
        b = self._ws.get(name)
        if b is None or b.n < n:
            if b is not None:
                self.free(b)
            b = self._ws[name] = self.alloc(max(n, 2 * (b.n if b else 0), 64))
        return b

    def linear(self, m, X, Y, T, bias=None, acc=False, ln=None, eps=1e-6, off=0.0, xoff=0):
        """Y (+)= norm(X) @ W.T (+ bias) for T rows. ln: RMSNorm weights applied to each row of X first (fused into
        the product for a few rows; a separate pass before the tiled product for many)."""
        if T < GEMM_ROWS:
            self.gemv(m, X, Y, T, xoff=xoff, bias=bias, acc=acc, ln=ln, eps=eps, off=off)
            return
        if ln is not None:
            Z = self.ws("lnx", T * m.cols)
            self.rms(X, Z, ln, T, m.cols, eps, off, xoff=xoff)
            X, xoff = Z, 0
        self.gemm(m, X, Y, T, xoff=xoff, bias=bias, acc=acc)

    def matmul(self, x, m):
        """x @ W.T with x on the host (the older, per-product path)."""
        lead = x.shape[:-1]
        x2 = np.ascontiguousarray(x.reshape(-1, m.cols), np.float32)
        T = x2.shape[0]
        with _lock:
            xb = self.ws("mm_x", x2.size)
            yb = self.ws("mm_y", T * m.rows)
            self.put(xb, x2.reshape(-1))
            self.begin()
            self.gemv(m, xb, yb, T)
            y = self.get(yb, T * m.rows)
        return y.reshape(*lead, m.rows)

    def trim(self):
        """Free the scratch buffers (they come back, as large as needed, on the next use)."""
        for b in list(self._ws.values()):
            try:
                self.free(b)
            except Exception:
                pass
        self._ws = {}

    def close(self):
        for b in list(self._own):
            try:
                self.free(b)
            except Exception:                       # a lost device: nothing more to free
                pass
        self._own = []
        self._ws = {}
        self.used = 0


# ======================================================================================== the real GPU
class Runner(_Base):
    """One GPU and the kernels; all sizes are in elements."""

    def __init__(self, dev):
        super().__init__()
        self.dev = dev
        self.name = dev.name
        lib = dev.drv.lib
        lib.cuMemcpyDtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        mod, first = dev.load("apsi-hub-2", _build_ptx())
        self._calls = {}
        self._launch = lib.cuLaunchKernel
        self.fn = {"apsi_kernel": first}
        for name in KERNELS[1:]:
            f = ctypes.c_void_p()
            dev.drv.check(lib.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), f"cuModuleGetFunction({name})")
            self.fn[name] = f

    # memory
    def alloc(self, n):
        b = Buf(self.dev.alloc(4 * n), n)
        self._own.append(b)
        self.used += 4 * n
        return b

    def free(self, b):
        if b in self._own:
            self._own.remove(b)
            self.used -= 4 * b.n
            self.dev.free(b.ptr)
            self._calls.clear()                     # cached launches may name this memory

    def put(self, b, a, off=0):
        a = np.ascontiguousarray(a)
        assert off + a.size <= b.n
        self.dev.activate()
        self.dev.to_device(b.ptr + 4 * off, a.ctypes.data, a.nbytes)

    def get(self, b, n, off=0):
        out = np.empty(n, np.float32)
        self.dev.activate()
        self.dev.to_host(out.ctypes.data, b.ptr + 4 * off, out.nbytes)    # waits for the kernels before it
        return out

    def copy(self, dst, doff, src, soff, n):
        if n:
            self.dev.activate()
            self.dev.drv.check(self.dev.drv.lib.cuMemcpyDtoD_v2(ctypes.c_uint64(dst.ptr + 4 * doff),
                                                                 ctypes.c_uint64(src.ptr + 4 * soff),
                                                                 ctypes.c_size_t(4 * n)), "cuMemcpyDtoD")

    _CT = {"p": ctypes.c_uint64, "u": ctypes.c_uint32, "f": ctypes.c_float}

    def _go(self, name, grid, block, *args):
        """Launch a kernel. args: ("p", pointer) / ("u", uint32) / ("f", float32). The argument blocks of
        repeated launches (every layer of every token) are built once and reused."""
        key = (name, grid, block, args)
        hit = self._calls.get(key)
        if hit is None:
            objs = [self._CT[k](v) for k, v in args]
            arr = (ctypes.c_void_p * len(objs))(*[ctypes.addressof(o) for o in objs])
            if len(self._calls) > 8192:
                self._calls.clear()
            hit = self._calls[key] = (objs, arr)
        r = self._launch(self.fn[name], max(1, grid), 1, 1, block, 1, 1, 0, None, hit[1], None)
        if r:
            self.dev.drv.check(r, f"launching {name}")

    def begin(self):
        """Make this GPU current for the calling thread (once per forward pass)."""
        self.dev.activate()

    # kernels
    def gemv(self, m, X, Y, T, xoff=0, yoff=0, bias=None, acc=False, ln=None, eps=1e-6, off=0.0):
        width = 16 if m.cols % 16 == 0 else 8 if m.cols % 8 == 0 else 2
        name = {16: "gemv16", 8: "apsi_kernel", 2: "gemv2"}[width]
        if ln is not None:
            name = {16: "gemv16n", 8: "gemv8n", 2: "gemv2n"}[width]
        args = [("p", m.buf.ptr), ("p", X.ptr + 4 * xoff), ("p", Y.ptr + 4 * yoff), ("p", bias.ptr if bias else 0),
                ("u", m.rows), ("u", m.cols), ("u", T), ("u", 1 if bias else 0), ("u", 1 if acc else 0)]
        if ln is not None:
            args += [("p", ln.ptr), ("f", eps), ("f", off)]
        self._go(name, (m.rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK, 32 * ROWS_PER_BLOCK, *args)

    def gemm(self, m, X, Y, T, xoff=0, bias=None, acc=False):
        tiles = ((m.rows + 63) // 64) * ((T + 63) // 64)
        name = "gemm"
        if tiles < 3 * self.dev.sms:                # few tiles: half-height ones keep more SMs busy
            name, tiles = "gemm32", ((m.rows + 63) // 64) * ((T + 31) // 32)
        self._go(name, tiles, 256, ("p", m.buf.ptr), ("p", X.ptr + 4 * xoff), ("p", Y.ptr), ("p", bias.ptr if bias else 0),
                 ("u", m.rows), ("u", m.cols), ("u", T), ("u", 1 if bias else 0), ("u", 1 if acc else 0))

    def rms(self, X, Y, W, rows, n, eps, off, xoff=0, yoff=0):
        self._go("rms", rows, NT, ("p", X.ptr + 4 * xoff), ("p", Y.ptr + 4 * yoff), ("p", W.ptr), ("u", n),
                 ("f", eps), ("f", off))

    def alloc_kv(self, n, half):
        """A key / value cache buffer of n elements: float16 (half) or float32."""
        return self.alloc((n + 1) // 2 if half else n)

    def copy_kv(self, dst, doff, src, soff, n, half):
        if not half:
            return self.copy(dst, doff, src, soff, n)
        if n:                                           # element offsets of 2 bytes
            self.dev.activate()
            self.dev.drv.check(self.dev.drv.lib.cuMemcpyDtoD_v2(ctypes.c_uint64(dst.ptr + 2 * doff),
                                                                 ctypes.c_uint64(src.ptr + 2 * soff),
                                                                 ctypes.c_size_t(2 * n)), "cuMemcpyDtoD")

    def qkv_prep(self, QKV, Q, Kc, Vc, COS, SIN, QN, KN, T, nh, kvh, dh, rot, start, cap, eps, half=False):
        dummy = self.ws("dummy", 64)                    # read (and unused) when there is no q/k norm
        self._go("qkv_prep_h" if half else "qkv_prep", T * (nh + 2 * kvh), NT,
                 ("p", QKV.ptr), ("p", Q.ptr), ("p", Kc.ptr), ("p", Vc.ptr), ("p", COS.ptr), ("p", SIN.ptr),
                 ("p", (QN or dummy).ptr), ("p", (KN or dummy).ptr), ("u", 1 if QN else 0), ("u", nh), ("u", kvh),
                 ("u", dh), ("u", rot), ("u", start), ("u", cap), ("f", eps))

    def attention(self, Q, Kc, Vc, O, S, T, nh, kvh, dh, start, cap, window, scale, splits=None, half=False):
        if splits is None:                          # long context, few blocks: split the keys over more SMs
            span = min(start + T, window) if window else start + T
            room = max(1, (4 * self.dev.sms) // max(1, T * nh))
            splits = max(1, min(span // 256, room, 64))     # a split is worth its merge from 256 keys on
        blocks = T * nh
        PM, PL = self.ws("att_m", blocks * splits), self.ws("att_l", blocks * splits)
        PO = self.ws("att_o", blocks * splits * dh)
        self._go("attention_h" if half else "attention", blocks * splits, NTA,
                 ("p", Q.ptr), ("p", Kc.ptr), ("p", Vc.ptr), ("p", O.ptr), ("p", S.ptr), ("p", PM.ptr), ("p", PL.ptr),
                 ("p", PO.ptr), ("u", nh), ("u", kvh), ("u", dh), ("u", start), ("u", cap), ("u", window or 0),
                 ("f", scale), ("u", start + T), ("u", splits))
        if splits > 1:
            self._go("attn_merge", blocks, NT, ("p", O.ptr), ("p", PM.ptr), ("p", PL.ptr), ("p", PO.ptr), ("u", dh),
                     ("u", splits))

    def embed(self, m, ids, X, scale=1.0):
        """X[t] = row ids[t] of the bfloat16 matrix m (times scale)."""
        ids = np.ascontiguousarray(ids, np.int32).reshape(-1)
        IDS = self.ws("ids", ids.size)
        self.put(IDS, ids.view(np.float32))
        self._go("embed", ids.size, 256, ("p", m.buf.ptr), ("p", IDS.ptr), ("p", X.ptr), ("u", m.cols), ("f", scale))

    def sample_prep(self, LOG, SEEN, V, rp, inv_t, delta, cap):
        """(max, softmax denominator, candidate ids, their scaled logits, how many there were) for the logits in LOG
        after the repetition penalty (SEEN marks the tokens) and the temperature."""
        chunk = 1024
        nb = (V + chunk - 1) // chunk
        OUT, R = self.ws("smp_out", V), self.ws("smp_res", 2 * nb + 1 + 2 * cap)
        self.put(R, np.zeros(1, np.float32), 2 * nb)                    # the candidate counter
        pm, ps, cnt = R.ptr, R.ptr + 4 * nb, R.ptr + 8 * nb
        ci, cv = cnt + 4, cnt + 4 + 4 * cap
        self._go("lg_prep", nb, 256, ("p", LOG.ptr), ("p", SEEN.ptr), ("p", OUT.ptr), ("p", pm), ("u", V), ("f", rp),
                 ("f", inv_t), ("u", chunk))
        self._go("lg_sum", nb, 256, ("p", OUT.ptr), ("p", pm), ("p", ps), ("u", V), ("u", nb), ("u", chunk))
        self._go("lg_pick", nb, 256, ("p", OUT.ptr), ("p", pm), ("p", cnt), ("p", ci), ("p", cv), ("u", V), ("u", nb),
                 ("u", chunk), ("f", delta), ("u", cap))
        head = self.get(R, 2 * nb + 1)
        count = int(head[2 * nb:].view(np.uint32)[0])
        got = min(count, cap)
        tail = self.get(R, 2 * cap, 2 * nb + 1) if got else np.zeros(2 * cap, np.float32)
        return (float(head[:nb].max()), float(head[nb:2 * nb].astype(np.float64).sum()),
                tail[:got].view(np.int32).astype(np.int64), tail[cap:cap + got], count)

    def act_mul(self, GU, M, T, I, kind):
        n = T * I
        self._go("act_mul", (n + 255) // 256, 256, ("p", GU.ptr), ("p", M.ptr), ("u", I), ("u", n), ("u", kind))

    @property
    def memory(self):
        return self.dev.memory

    def free_memory(self):
        lib = self.dev.drv.lib
        lib.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.dev.activate()
        self.dev.drv.check(lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
        return free.value


# ======================================================================================== NumPy stand-in
class Emulator(_Base):
    """The same operations as Runner, in NumPy (float32, bfloat16 weights): tests the code around the kernels."""

    class dev:
        index, memory = 0, 8 << 30

    name = "emulator"

    def alloc(self, n):
        b = Buf(0, n, np.zeros(n, np.float32))
        self._own.append(b)
        return b

    def free(self, b):
        if b in self._own:
            self._own.remove(b)

    def put(self, b, a, off=0):
        a = np.ascontiguousarray(a).reshape(-1)
        assert off + a.size <= b.n
        b.a[off:off + a.size] = a.view(np.float32) if a.dtype == np.uint32 else a

    def get(self, b, n, off=0):
        return b.a[off:off + n].copy()

    def copy(self, dst, doff, src, soff, n):
        dst.a[doff:doff + n] = src.a[soff:soff + n]

    def begin(self):
        pass

    def gemm(self, m, X, Y, T, xoff=0, bias=None, acc=False):
        self.gemv(m, X, Y, T, xoff=xoff, bias=bias, acc=acc)

    def gemv(self, m, X, Y, T, xoff=0, yoff=0, bias=None, acc=False, ln=None, eps=1e-6, off=0.0):
        if ln is not None:
            Z = Buf(0, T * m.cols, np.zeros(T * m.cols, np.float32))
            self.rms(X, Z, ln, T, m.cols, eps, off, xoff=xoff)
            X, xoff = Z, 0
        w = from_bf16_pairs(m.buf.a.view(np.uint32)[:m.rows * m.cols // 2], (m.rows, m.cols))
        x = X.a[xoff:xoff + T * m.cols].reshape(T, m.cols)
        y = (x @ w.T).astype(np.float32)
        if bias is not None:
            y = y + bias.a[:m.rows]
        sl = slice(yoff, yoff + T * m.rows)
        Y.a[sl] = (Y.a[sl].reshape(T, m.rows) + y).reshape(-1) if acc else y.reshape(-1)

    def rms(self, X, Y, W, rows, n, eps, off, xoff=0, yoff=0):
        x = X.a[xoff:xoff + rows * n].reshape(rows, n)
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (W.a[:n] + off)
        Y.a[yoff:yoff + rows * n] = y.reshape(-1)

    def alloc_kv(self, n, half):
        return self.alloc(n)

    def copy_kv(self, dst, doff, src, soff, n, half):
        self.copy(dst, doff, src, soff, n)

    def qkv_prep(self, QKV, Q, Kc, Vc, COS, SIN, QN, KN, T, nh, kvh, dh, rot, start, cap, eps, half=False):
        kv_half = half
        per = nh + 2 * kvh
        x = QKV.a[:T * per * dh].reshape(T, per, dh)
        half = rot // 2
        cos = COS.a[start * half:(start + T) * half].reshape(T, 1, half)
        sin = SIN.a[start * half:(start + T) * half].reshape(T, 1, half)

        def prep(v, w):
            if w is not None:
                v = v / np.sqrt((v * v).mean(-1, keepdims=True) + eps) * w.a[:dh]
            r1, r2, rest = v[..., :half], v[..., half:rot], v[..., rot:]
            return np.concatenate([r1 * cos - r2 * sin, r2 * cos + r1 * sin, rest], -1)
        q = prep(x[:, :nh], QN)
        k = prep(x[:, nh:nh + kvh], KN)
        v = x[:, nh + kvh:]
        Q.a[:T * nh * dh] = q.reshape(-1)
        kc = Kc.a[:kvh * cap * dh].reshape(kvh, cap, dh)
        vc = Vc.a[:kvh * cap * dh].reshape(kvh, cap, dh)
        if kv_half:                                     # what a float16 cache keeps
            k, v = k.astype(np.float16).astype(np.float32), v.astype(np.float16).astype(np.float32)
        kc[:, start:start + T] = k.transpose(1, 0, 2)
        vc[:, start:start + T] = v.transpose(1, 0, 2)

    def attention(self, Q, Kc, Vc, O, S, T, nh, kvh, dh, start, cap, window, scale, splits=None, half=False):
        q = Q.a[:T * nh * dh].reshape(T, nh, dh)
        kc = Kc.a[:kvh * cap * dh].reshape(kvh, cap, dh)
        vc = Vc.a[:kvh * cap * dh].reshape(kvh, cap, dh)
        out = np.zeros((T, nh, dh), np.float32)
        rep = nh // kvh
        for t in range(T):
            qp = start + t
            lo = max(0, qp - window + 1) if window else 0
            for h in range(nh):
                s = kc[h // rep, lo:qp + 1] @ q[t, h] * scale
                e = np.exp(s - s.max())
                out[t, h] = (e / e.sum()) @ vc[h // rep, lo:qp + 1]
        O.a[:T * nh * dh] = out.reshape(-1)

    def embed(self, m, ids, X, scale=1.0):
        ids = np.asarray(ids, np.int64).reshape(-1)
        w = from_bf16_pairs(m.buf.a.view(np.uint32)[:m.rows * m.cols // 2], (m.rows, m.cols))
        X.a[:ids.size * m.cols] = (w[ids] * np.float32(scale)).reshape(-1)

    def sample_prep(self, LOG, SEEN, V, rp, inv_t, delta, cap):
        lg = LOG.a[:V].copy()
        seen = SEEN.a[:V] != 0
        lg[seen] = np.where(lg[seen] > 0, lg[seen] / rp, lg[seen] * rp)
        lg = lg * np.float32(inv_t)
        g = float(lg.max())
        Z = float(np.exp(lg - g).sum(dtype=np.float64))
        ids = np.flatnonzero(lg >= g - delta)
        return g, Z, ids[:cap], lg[ids[:cap]], len(ids)

    def act_mul(self, GU, M, T, I, kind):
        gu = GU.a[:T * 2 * I].reshape(T, 2 * I)
        g, u = gu[:, :I], gu[:, I:]
        if kind == 0:
            a = g / (1 + np.exp(-g))
        else:
            a = 0.5 * g * (1 + np.tanh(math.sqrt(2 / math.pi) * (g + 0.044715 * g ** 3)))
        M.a[:T * I] = (a * u).reshape(-1)

    @property
    def memory(self):
        return self.dev.memory

    def free_memory(self):
        return self.dev.memory


def runner(device="auto"):
    """(Runner or None, why not). device: "auto", "cpu", "cuda" or "cuda:N"."""
    d = str(device or "auto").lower()
    if d == "cpu":
        return None, "device=cpu"
    from .native import cuda
    drv = cuda.driver()
    if drv is None:
        return None, cuda.unavailable_reason() or "no NVIDIA GPU"
    idx = 0
    if d.startswith("cuda:"):
        try:
            idx = int(d[5:])
        except ValueError:
            return None, f"unknown device '{device}'"
    elif d not in ("auto", "cuda", "gpu"):
        return None, f"unknown device '{device}' (use auto, cpu, cuda or cuda:N)"
    devs = drv.devices
    pick = [x for x in devs if x.index == idx] or (devs[:1] if d == "auto" else [])
    if not pick:
        return None, f"no GPU {idx}"
    try:
        return Runner(pick[0]), ""
    except Exception as e:                              # the driver refused the kernels, out of memory, ...
        return None, f"{type(e).__name__}: {e}"


class Pending:
    """runner(device) started in the background (driver start-up, kernels), joined with result()."""

    def __init__(self, device):
        self.value = (None, "not started")
        self._t = threading.Thread(target=self._run, args=(device,), name="apsi-gpu-prepare", daemon=True)
        self._t.start()

    def _run(self, device):
        try:
            self.value = runner(device)
        except Exception as e:                          # never let the background start-up take the program down
            self.value = (None, f"{type(e).__name__}: {e}")

    def result(self):
        self._t.join()
        return self.value


def prepare(device="auto"):
    return Pending(device)
