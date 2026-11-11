"""The kernels of Apsi's own GPU training (gpu_train.py): LLVM -> PTX through llvmlite, loaded with the CUDA driver.

Two modules: the main one (sm_75: every NVIDIA GPU from the GTX 16xx / RTX 20xx on) with the element-wise,
normalisation, attention-softmax, loss, optimizer and weight-format kernels plus a float32 matrix product, and the
tensor-core one (sm_80: Ampere and newer) with matrix products on mma.sync bfloat16 tensor cores - one pass
(bfloat16 inputs, float32 sums, as mixed precision does) or three passes (each input split into a high and a low
bfloat16 part: hi*hi + lo*hi + hi*lo, about float32 accuracy).

Every matrix product takes any strides, so transposed operands (the backward pass) and the heads of attention
(batched over batch x heads, two batch strides) need no copies:
    C[z][m, n] = alpha * sum_k A[z](m, k) * B[z](k, n) (+ C if beta) (+ bias[n])
    A(m, k) = A[off_a(z) + m * sam + k * sak],  B(k, n) = B[off_b(z) + k * sbk + n * sbn],  C row stride ldc
    off(z) = (z / nb2) * s1 + (z % nb2) * s2,   z = blockIdx.y
"""
import os

LOG2E = 1.4426950408889634
ACTS = {"relu": 0, "gelu": 1, "silu": 2, "swish": 2, "tanh": 3, "sigmoid": 4, "linear": 5, "identity": 5,
        "leaky_relu": 6}
FAKEQ = {"fp8": 0, "fp6": 1, "fp6_e2m3": 2, "fp4": 3, "int8": 4, "int4": 5}
MINIFLOAT = {0: (3, 7, 448.0), 1: (2, 3, 28.0), 2: (3, 1, 7.5), 3: (1, 1, 6.0)}     # mantissa bits, bias, top
TILE = 64

MAIN = ("apsi_kernel", "axpy", "add3", "mul3", "act_f", "act_b", "colsum", "ln_f", "ln_b", "embed_f", "embed_b",
        "pos_f", "pos_b", "softmax_f", "softmax_b", "ce", "mse", "sumsq", "adam", "sgd", "drop_f", "fakeq",
        "gemm_simt", "gather")
TC = ("apsi_kernel", "gemm_tc3")                     # apsi_kernel = the one-pass product


def _setup(name):
    import llvmlite.ir as ir
    m = ir.Module(name=name)
    m.triple = "nvptx64-nvidia-cuda"
    return ir, m


def _emit(m, cpu):
    import llvmlite.binding as llvm
    from .native import engine as _eng
    _eng.engine()
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    mod = llvm.parse_assembly(str(m))
    mod.verify()
    tm = llvm.Target.from_triple("nvptx64-nvidia-cuda").create_target_machine(cpu=cpu, opt=3)
    mod.data_layout = str(tm.target_data)
    pb = llvm.create_pass_builder(tm, llvm.create_pipeline_tuning_options(speed_level=3))
    pb.getModulePassManager().run(mod, pb)
    return tm.emit_assembly(mod)


class _Gen:
    """Helpers shared by both modules."""

    def __init__(self, ir, m):
        self.ir, self.m = ir, m
        self.i32, self.i64, self.f32 = ir.IntType(32), ir.IntType(64), ir.FloatType()
        d = self.decl = lambda n, r, a: ir.Function(m, ir.FunctionType(r, a), name=n)   # noqa: E731
        i32, f32 = self.i32, self.f32
        self.tid_f = d("llvm.nvvm.read.ptx.sreg.tid.x", i32, [])
        self.ctax_f = d("llvm.nvvm.read.ptx.sreg.ctaid.x", i32, [])
        self.ctay_f = d("llvm.nvvm.read.ptx.sreg.ctaid.y", i32, [])
        self.nctax_f = d("llvm.nvvm.read.ptx.sreg.nctaid.x", i32, [])

    def G(self, t):
        return t.as_pointer(1)

    def C(self, v):
        return self.ir.Constant(self.i32, v)

    def F(self, v):
        return self.ir.Constant(self.f32, v)


def _main_module():
    ir, m = _setup("apsi_train")
    g = _Gen(ir, m)
    i32, i64, f32, C, F, G, decl = g.i32, g.i64, g.f32, g.C, g.F, g.G, g.decl
    shfl_f = decl("llvm.nvvm.shfl.sync.bfly.f32", f32, [i32, f32, i32, i32])
    fma_f = decl("llvm.fma.f32", f32, [f32, f32, f32])
    ex2_f = decl("llvm.nvvm.ex2.approx.f", f32, [f32])
    lg2_f = decl("llvm.nvvm.lg2.approx.f", f32, [f32])
    sqrt_f = decl("llvm.sqrt.f32", f32, [f32])
    max_f = decl("llvm.maxnum.f32", f32, [f32, f32])
    min_f = decl("llvm.minnum.f32", f32, [f32, f32])
    fabs_f = decl("llvm.fabs.f32", f32, [f32])
    rint_f = decl("llvm.rint.f32", f32, [f32])
    bar_f = decl("llvm.nvvm.barrier0", ir.VoidType(), [])
    smem = ir.GlobalVariable(m, ir.ArrayType(f32, 32), "apsi_smem", addrspace=3)
    smem.linkage = "internal"
    smem.initializer = ir.Constant(ir.ArrayType(f32, 32), None)

    class K:
        def __init__(self, name, params):
            fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [t for _, t in params]), name=name)
            fn.calling_convention = "ptx_kernel"
            self.fn = fn
            self.top = fn.append_basic_block("allocas")
            self.entry = fn.append_basic_block("entry")
            self.a = ir.IRBuilder(self.top)
            self.b = ir.IRBuilder(self.entry)
            self.p = {n: a for (n, _), a in zip(params, fn.args)}
            self.tid = self.b.call(g.tid_f, [])
            self.cta = self.b.call(g.ctax_f, [])
            self.ctay = self.b.call(g.ctay_f, [])
            self.ncta = self.b.call(g.nctax_f, [])

        def ret(self):
            self.b.ret_void()
            self.a.branch(self.entry)

        def var(self, init):
            v = self.a.alloca(init.type)
            self.b.store(init, v)
            return v

        def gep(self, ptr, idx):
            return self.b.gep(ptr, [self.b.zext(idx, i64) if idx.type == i32 else idx], inbounds=True)

        def ld(self, ptr, idx):
            return self.b.load(self.gep(ptr, idx), align=4)

        def st(self, val, ptr, idx):
            self.b.store(val, self.gep(ptr, idx), align=4)

        def atomic_add(self, ptr, idx, val):
            self.b.atomic_rmw("fadd", self.gep(ptr, idx), val, "monotonic")

        def loop(self, lo, hi, step, body):
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

        def grid_loop(self, n, body):
            """for i in 0..n over the whole grid (256 threads a block)."""
            b = self.b
            self.loop(b.add(b.mul(self.cta, C(256)), self.tid), n, b.mul(self.ncta, C(256)), body)

        def warp_reduce(self, v):
            for k in (16, 8, 4, 2, 1):
                v = self.b.fadd(v, self.b.call(shfl_f, [C(-1), v, C(k), C(31)]))
            return v

        def warp_max(self, v):
            for k in (16, 8, 4, 2, 1):
                v = self.b.call(max_f, [v, self.b.call(shfl_f, [C(-1), v, C(k), C(31)])])
            return v

        def block_sum(self, v, mx=False):
            """Sum (or max) over the block of 256 threads; every thread gets it."""
            b = self.b
            v = self.warp_max(v) if mx else self.warp_reduce(v)
            lane, warp = b.and_(self.tid, C(31)), b.lshr(self.tid, C(5))
            with b.if_then(b.icmp_unsigned("==", lane, C(0))):
                b.store(v, b.gep(smem, [C(0), warp], inbounds=True))
            b.call(bar_f, [])
            r = b.load(b.gep(smem, [C(0), C(0)], inbounds=True))
            for w in range(1, 8):
                o = b.load(b.gep(smem, [C(0), C(w)], inbounds=True))
                r = b.call(max_f, [r, o]) if mx else b.fadd(r, o)
            b.call(bar_f, [])
            return r

    def exp(k, x):
        return k.b.call(ex2_f, [k.b.fmul(x, F(LOG2E))])

    def tanh(k, x):
        b = k.b
        return b.fsub(F(1.0), b.fdiv(F(2.0), b.fadd(exp(k, b.fmul(F(2.0), x)), F(1.0))))

    def sigmoid(k, x):
        b = k.b
        return b.fdiv(F(1.0), b.fadd(F(1.0), exp(k, b.fsub(F(0.0), x))))

    def i64of(b, x):
        return b.zext(x, i64)

    # ------------------------------------------------------------------------- element-wise
    k = K("apsi_kernel", [("X", G(f32)), ("n", i32), ("v", f32)])               # fill
    k.grid_loop(k.p["n"], lambda i: k.st(k.p["v"], k.p["X"], i))
    k.ret()

    k = K("axpy", [("Y", G(f32)), ("X", G(f32)), ("n", i32), ("a", f32)])
    k.grid_loop(k.p["n"], lambda i: k.st(k.b.call(fma_f, [k.p["a"], k.ld(k.p["X"], i), k.ld(k.p["Y"], i)]), k.p["Y"], i))
    k.ret()

    k = K("add3", [("Z", G(f32)), ("X", G(f32)), ("Y", G(f32)), ("n", i32)])
    k.grid_loop(k.p["n"], lambda i: k.st(k.b.fadd(k.ld(k.p["X"], i), k.ld(k.p["Y"], i)), k.p["Z"], i))
    k.ret()

    k = K("mul3", [("Z", G(f32)), ("X", G(f32)), ("Y", G(f32)), ("n", i32)])
    k.grid_loop(k.p["n"], lambda i: k.st(k.b.fmul(k.ld(k.p["X"], i), k.ld(k.p["Y"], i)), k.p["Z"], i))
    k.ret()

    def act_value(k, x, kind):
        """f(x) and f'(x) for each kind, selected at run time."""
        b = k.b
        relu = b.select(b.fcmp_ordered(">", x, F(0.0)), x, F(0.0))
        drelu = b.select(b.fcmp_ordered(">", x, F(0.0)), F(1.0), F(0.0))
        c = 0.7978845608028654
        u = b.fmul(F(c), b.fadd(x, b.fmul(F(0.044715), b.fmul(x, b.fmul(x, x)))))
        t = tanh(k, u)
        gelu = b.fmul(b.fmul(F(0.5), x), b.fadd(F(1.0), t))
        dgelu = b.fadd(b.fmul(F(0.5), b.fadd(F(1.0), t)),
                       b.fmul(b.fmul(b.fmul(F(0.5), x), b.fsub(F(1.0), b.fmul(t, t))),
                              b.fmul(F(c), b.fadd(F(1.0), b.fmul(F(3 * 0.044715), b.fmul(x, x))))))
        s = sigmoid(k, x)
        silu = b.fmul(x, s)
        dsilu = b.fadd(s, b.fmul(x, b.fmul(s, b.fsub(F(1.0), s))))
        th = tanh(k, x)
        dth = b.fsub(F(1.0), b.fmul(th, th))
        dsig = b.fmul(s, b.fsub(F(1.0), s))
        leaky = b.select(b.fcmp_ordered(">", x, F(0.0)), x, b.fmul(F(0.01), x))
        dleaky = b.select(b.fcmp_ordered(">", x, F(0.0)), F(1.0), F(0.01))
        vals = [(relu, drelu), (gelu, dgelu), (silu, dsilu), (th, dth), (s, dsig), (x, F(1.0)), (leaky, dleaky)]
        f, d = vals[-1]
        for j in range(len(vals) - 2, -1, -1):
            sel = b.icmp_unsigned("==", kind, C(j))
            f, d = b.select(sel, vals[j][0], f), b.select(sel, vals[j][1], d)
        return f, d

    k = K("act_f", [("X", G(f32)), ("Y", G(f32)), ("n", i32), ("kind", i32)])
    k.grid_loop(k.p["n"], lambda i: k.st(act_value(k, k.ld(k.p["X"], i), k.p["kind"])[0], k.p["Y"], i))
    k.ret()

    k = K("act_b", [("X", G(f32)), ("DY", G(f32)), ("DX", G(f32)), ("n", i32), ("kind", i32)])
    k.grid_loop(k.p["n"], lambda i: k.st(k.b.fmul(k.ld(k.p["DY"], i), act_value(k, k.ld(k.p["X"], i), k.p["kind"])[1]),
                                         k.p["DX"], i))
    k.ret()

    # ------------------------------------------------------------------------- column sums (bias / norm gradients)
    # grid (ceil(N/256), ceil(M/rows)): DB[n] += sum dy; mode 1 (layer norm) DG[n] += sum dy * (x - mean) * rstd;
    # mode 2 (RMS norm) DG[n] += sum dy * x * rstd and no DB
    k = K("colsum", [("DY", G(f32)), ("X", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)), ("DG", G(f32)),
                     ("DB", G(f32)), ("M", i32), ("N", i32), ("rows", i32), ("mode", i32)])
    b, p = k.b, k.p
    n = b.add(b.mul(k.cta, C(256)), k.tid)
    with b.if_then(b.icmp_unsigned("<", n, p["N"])):
        r0 = b.mul(k.ctay, p["rows"])
        r1 = b.select(b.icmp_unsigned("<", b.add(r0, p["rows"]), p["M"]), b.add(r0, p["rows"]), p["M"])
        sb, sg = k.var(F(0.0)), k.var(F(0.0))

        def row(r):
            idx = b.add(b.mul(i64of(b, r), i64of(b, p["N"])), i64of(b, n))
            dy = k.ld(p["DY"], idx)
            b.store(b.fadd(b.load(sb), dy), sb)
            with b.if_then(b.icmp_unsigned("!=", p["mode"], C(0))):
                x = k.ld(p["X"], idx)
                mu = b.select(b.icmp_unsigned("==", p["mode"], C(1)), k.ld(p["MEAN"], r), F(0.0))
                xh = b.fmul(b.fsub(x, mu), k.ld(p["RSTD"], r))
                b.store(b.call(fma_f, [dy, xh, b.load(sg)]), sg)
        k.loop(r0, r1, C(1), row)
        with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
            k.atomic_add(p["DB"], n, b.load(sb))
        with b.if_then(b.icmp_unsigned("!=", p["mode"], C(0))):
            k.atomic_add(p["DG"], n, b.load(sg))
    k.ret()

    # ------------------------------------------------------------------------- layer norm / RMS norm, a block a row
    k = K("ln_f", [("X", G(f32)), ("G", G(f32)), ("B", G(f32)), ("Y", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)),
                   ("M", i32), ("N", i32), ("eps", f32), ("rms", i32)])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["N"]))
    s1 = k.var(F(0.0))
    k.loop(k.tid, p["N"], C(256), lambda i: b.store(b.fadd(b.load(s1), k.ld(p["X"], b.add(base, i64of(b, i)))), s1))
    isrms = b.icmp_unsigned("!=", p["rms"], C(0))
    mean = b.select(isrms, F(0.0), b.fdiv(k.block_sum(b.load(s1)), b.uitofp(p["N"], f32)))
    s2 = k.var(F(0.0))

    def sq(i):
        d = b.fsub(k.ld(p["X"], b.add(base, i64of(b, i))), mean)
        b.store(b.call(fma_f, [d, d, b.load(s2)]), s2)
    k.loop(k.tid, p["N"], C(256), sq)
    var = b.fdiv(k.block_sum(b.load(s2)), b.uitofp(p["N"], f32))
    rstd = b.fdiv(F(1.0), b.call(sqrt_f, [b.fadd(var, p["eps"])]))

    def out(i):
        idx = b.add(base, i64of(b, i))
        y = b.fmul(b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd), k.ld(p["G"], i))
        y = b.select(isrms, y, b.fadd(y, k.ld(p["B"], i)))
        k.st(y, p["Y"], idx)
    k.loop(k.tid, p["N"], C(256), out)
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(mean, p["MEAN"], k.cta)
        k.st(rstd, p["RSTD"], k.cta)
    k.ret()

    k = K("ln_b", [("X", G(f32)), ("G", G(f32)), ("DY", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)), ("DX", G(f32)),
                   ("M", i32), ("N", i32), ("rms", i32)])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["N"]))
    isrms = b.icmp_unsigned("!=", p["rms"], C(0))
    mean = b.select(isrms, F(0.0), k.ld(p["MEAN"], k.cta))
    rstd = k.ld(p["RSTD"], k.cta)
    a1, a2 = k.var(F(0.0)), k.var(F(0.0))

    def acc(i):
        idx = b.add(base, i64of(b, i))
        gdy = b.fmul(k.ld(p["DY"], idx), k.ld(p["G"], i))
        xh = b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd)
        b.store(b.fadd(b.load(a1), gdy), a1)
        b.store(b.call(fma_f, [gdy, xh, b.load(a2)]), a2)
    k.loop(k.tid, p["N"], C(256), acc)
    nf = b.uitofp(p["N"], f32)
    m1 = b.select(isrms, F(0.0), b.fdiv(k.block_sum(b.load(a1)), nf))
    m2 = b.fdiv(k.block_sum(b.load(a2)), nf)

    def dx(i):
        idx = b.add(base, i64of(b, i))
        gdy = b.fmul(k.ld(p["DY"], idx), k.ld(p["G"], i))
        xh = b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd)
        k.st(b.fmul(rstd, b.fsub(b.fsub(gdy, m1), b.fmul(xh, m2))), p["DX"], idx)
    k.loop(k.tid, p["N"], C(256), dx)
    k.ret()

    # ------------------------------------------------------------------------- embeddings, learned positions
    k = K("embed_f", [("E", G(f32)), ("IDS", G(i32)), ("Y", G(f32)), ("M", i32), ("D", i32)])
    b, p = k.b, k.p

    def emb(i):
        r, d = b.udiv(i, p["D"]), b.urem(i, p["D"])
        t = k.ld(p["IDS"], r)
        k.st(k.ld(p["E"], b.add(b.mul(i64of(b, t), i64of(b, p["D"])), i64of(b, d))), p["Y"], i)
    k.grid_loop(b.mul(p["M"], p["D"]), emb)
    k.ret()

    k = K("embed_b", [("DY", G(f32)), ("IDS", G(i32)), ("DE", G(f32)), ("M", i32), ("D", i32)])
    b, p = k.b, k.p

    def embb(i):
        r, d = b.udiv(i, p["D"]), b.urem(i, p["D"])
        t = k.ld(p["IDS"], r)
        k.atomic_add(p["DE"], b.add(b.mul(i64of(b, t), i64of(b, p["D"])), i64of(b, d)), k.ld(p["DY"], i))
    k.grid_loop(b.mul(p["M"], p["D"]), embb)
    k.ret()

    k = K("pos_f", [("X", G(f32)), ("P", G(f32)), ("Y", G(f32)), ("M", i32), ("D", i32), ("T", i32)])
    b, p = k.b, k.p

    def pos(i):
        r, d = b.udiv(i, p["D"]), b.urem(i, p["D"])
        k.st(b.fadd(k.ld(p["X"], i), k.ld(p["P"], b.add(b.mul(b.urem(r, p["T"]), p["D"]), d))), p["Y"], i)
    k.grid_loop(b.mul(p["M"], p["D"]), pos)
    k.ret()

    k = K("pos_b", [("DY", G(f32)), ("DP", G(f32)), ("M", i32), ("D", i32), ("T", i32)])
    b, p = k.b, k.p
    # thread per (t, d): sums the batch rows in a register, one write (no atomics needed)

    def posb(i):
        t, d = b.udiv(i, p["D"]), b.urem(i, p["D"])
        s = k.var(F(0.0))
        k.loop(t, p["M"], p["T"], lambda r: b.store(b.fadd(b.load(s), k.ld(p["DY"], b.add(b.mul(i64of(b, r),
                                                                                                 i64of(b, p["D"])),
                                                                                         i64of(b, d)))), s))
        k.atomic_add(p["DP"], i, b.load(s))
    k.grid_loop(b.mul(p["T"], p["D"]), posb)
    k.ret()

    # ------------------------------------------------------------------------- attention softmax, a block a row
    # rows are (head, query); query = row % T; causal: keys after the query are left out (probability 0)
    k = K("softmax_f", [("S", G(f32)), ("P", G(f32)), ("rows", i32), ("n", i32), ("T", i32), ("causal", i32)])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["n"]))
    q = b.urem(k.cta, p["T"])
    lim = b.select(b.icmp_unsigned("!=", p["causal"], C(0)), b.add(q, C(1)), p["n"])
    lim = b.select(b.icmp_unsigned("<", lim, p["n"]), lim, p["n"])
    mx = k.var(F(-3.0e38))
    k.loop(k.tid, lim, C(256), lambda i: b.store(b.call(max_f, [b.load(mx), k.ld(p["S"], b.add(base, i64of(b, i)))]), mx))
    M_ = k.block_sum(b.load(mx), mx=True)
    sm = k.var(F(0.0))
    k.loop(k.tid, lim, C(256), lambda i: b.store(b.fadd(b.load(sm), exp(k, b.fsub(k.ld(p["S"], b.add(base, i64of(b, i))),
                                                                                     M_))), sm))
    inv = b.fdiv(F(1.0), k.block_sum(b.load(sm)))

    def wr(i):
        idx = b.add(base, i64of(b, i))
        v = b.fmul(exp(k, b.fsub(k.ld(p["S"], idx), M_)), inv)
        k.st(b.select(b.icmp_unsigned("<", i, lim), v, F(0.0)), p["P"], idx)
    k.loop(k.tid, p["n"], C(256), wr)
    k.ret()

    k = K("softmax_b", [("P", G(f32)), ("DP", G(f32)), ("DS", G(f32)), ("rows", i32), ("n", i32)])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["n"]))
    s = k.var(F(0.0))
    k.loop(k.tid, p["n"], C(256), lambda i: b.store(b.call(fma_f, [k.ld(p["P"], b.add(base, i64of(b, i))),
                                                                   k.ld(p["DP"], b.add(base, i64of(b, i))),
                                                                   b.load(s)]), s))
    tot = k.block_sum(b.load(s))

    def dsw(i):
        idx = b.add(base, i64of(b, i))
        k.st(b.fmul(k.ld(p["P"], idx), b.fsub(k.ld(p["DP"], idx), tot)), p["DS"], idx)
    k.loop(k.tid, p["n"], C(256), dsw)
    k.ret()

    # ------------------------------------------------------------------------- losses (value and gradient at once)
    k = K("ce", [("L", G(f32)), ("LAB", G(i32)), ("DL", G(f32)), ("LOSS", G(f32)), ("M", i32), ("V", i32),
                 ("wg", f32), ("wl", f32)])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["V"]))
    lab = k.ld(p["LAB"], k.cta)
    valid = b.icmp_signed(">=", lab, C(0))
    mx = k.var(F(-3.0e38))
    k.loop(k.tid, p["V"], C(256), lambda i: b.store(b.call(max_f, [b.load(mx), k.ld(p["L"], b.add(base, i64of(b, i)))]), mx))
    M_ = k.block_sum(b.load(mx), mx=True)
    sm = k.var(F(0.0))
    k.loop(k.tid, p["V"], C(256), lambda i: b.store(b.fadd(b.load(sm), exp(k, b.fsub(k.ld(p["L"], b.add(base, i64of(b, i))),
                                                                                     M_))), sm))
    Z = k.block_sum(b.load(sm))
    inv = b.fdiv(F(1.0), Z)

    def grad(i):
        idx = b.add(base, i64of(b, i))
        pr = b.fmul(exp(k, b.fsub(k.ld(p["L"], idx), M_)), inv)
        pr = b.fsub(pr, b.select(b.icmp_signed("==", i, lab), F(1.0), F(0.0)))
        k.st(b.select(valid, b.fmul(pr, p["wg"]), F(0.0)), p["DL"], idx)
    k.loop(k.tid, p["V"], C(256), grad)
    with b.if_then(b.and_(valid, b.icmp_unsigned("==", k.tid, C(0)))):
        lse = b.fadd(M_, b.fdiv(b.call(lg2_f, [Z]), F(LOG2E)))
        k.atomic_add(p["LOSS"], C(0), b.fmul(b.fsub(lse, k.ld(p["L"], b.add(base, i64of(b, lab)))), p["wl"]))
    k.ret()

    k = K("mse", [("Y", G(f32)), ("T", G(f32)), ("DY", G(f32)), ("LOSS", G(f32)), ("n", i32), ("wg", f32),
                  ("wl", f32)])
    b, p = k.b, k.p
    s = k.var(F(0.0))

    def mse(i):
        d = b.fsub(k.ld(p["Y"], i), k.ld(p["T"], i))
        k.st(b.fmul(F(2.0), b.fmul(d, p["wg"])), p["DY"], i)
        b.store(b.call(fma_f, [d, d, b.load(s)]), s)
    k.grid_loop(p["n"], mse)
    tot = k.block_sum(b.load(s))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.atomic_add(p["LOSS"], C(0), b.fmul(tot, p["wl"]))
    k.ret()

    k = K("sumsq", [("X", G(f32)), ("n", i32), ("ACC", G(f32))])
    b, p = k.b, k.p
    s = k.var(F(0.0))
    k.grid_loop(p["n"], lambda i: b.store(b.call(fma_f, [k.ld(p["X"], i), k.ld(p["X"], i), b.load(s)]), s))
    tot = k.block_sum(b.load(s))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.atomic_add(p["ACC"], C(0), tot)
    k.ret()

    # ------------------------------------------------------------------------- optimizers, over all parameters at once
    def clip_scale(k, p):
        b = k.b
        nrm = b.call(sqrt_f, [b.fadd(k.ld(p["SQ"], C(0)), F(1e-30))])
        return b.select(b.fcmp_ordered(">", p["clip"], F(0.0)),
                        b.call(min_f, [F(1.0), b.fdiv(p["clip"], nrm)]), F(1.0))

    k = K("adam", [("P", G(f32)), ("G", G(f32)), ("M", G(f32)), ("V", G(f32)), ("SQ", G(f32)), ("n", i32),
                   ("n_decay", i32), ("lr", f32), ("b1", f32), ("b2", f32), ("eps", f32), ("c1", f32), ("c2", f32),
                   ("wd", f32), ("clip", f32)])
    b, p = k.b, k.p
    sc = clip_scale(k, p)

    def adam(i):
        gr = b.fmul(k.ld(p["G"], i), sc)
        w = k.ld(p["P"], i)
        w = b.select(b.icmp_unsigned("<", i, p["n_decay"]), b.fmul(w, b.fsub(F(1.0), b.fmul(p["lr"], p["wd"]))), w)
        m_ = b.fadd(b.fmul(p["b1"], k.ld(p["M"], i)), b.fmul(b.fsub(F(1.0), p["b1"]), gr))
        v_ = b.fadd(b.fmul(p["b2"], k.ld(p["V"], i)), b.fmul(b.fsub(F(1.0), p["b2"]), b.fmul(gr, gr)))
        k.st(m_, p["M"], i)
        k.st(v_, p["V"], i)
        upd = b.fdiv(b.fmul(p["lr"], b.fdiv(m_, p["c1"])),
                     b.fadd(b.call(sqrt_f, [b.fdiv(v_, p["c2"])]), p["eps"]))
        k.st(b.fsub(w, upd), p["P"], i)
    k.grid_loop(p["n"], adam)
    k.ret()

    k = K("sgd", [("P", G(f32)), ("G", G(f32)), ("VEL", G(f32)), ("SQ", G(f32)), ("n", i32), ("lr", f32),
                  ("mom", f32), ("clip", f32)])
    b, p = k.b, k.p
    sc = clip_scale(k, p)

    def sgd(i):
        v_ = b.call(fma_f, [p["mom"], k.ld(p["VEL"], i), b.fmul(k.ld(p["G"], i), sc)])
        k.st(v_, p["VEL"], i)
        k.st(b.fsub(k.ld(p["P"], i), b.fmul(p["lr"], v_)), p["P"], i)
    k.grid_loop(p["n"], sgd)
    k.ret()

    # ------------------------------------------------------------------------- a batch from the data on the card
    k = K("gather", [("SRC", G(f32)), ("IDX", G(i32)), ("DST", G(f32)), ("rows", i32), ("width", i32), ("start", i32)])
    b, p = k.b, k.p

    def gat(i):
        r, c = b.udiv(i, p["width"]), b.urem(i, p["width"])
        src_row = k.ld(p["IDX"], b.add(p["start"], r))
        k.st(k.ld(p["SRC"], b.add(b.mul(i64of(b, src_row), i64of(b, p["width"])), i64of(b, c))), p["DST"], i)
    k.grid_loop(b.mul(p["rows"], p["width"]), gat)
    k.ret()

    # ------------------------------------------------------------------------- dropout (a hash of index and seed)
    k = K("drop_f", [("X", G(f32)), ("Y", G(f32)), ("MASK", G(f32)), ("n", i32), ("p", f32), ("seed", i32)])
    b, p = k.b, k.p

    def drop(i):
        h = b.xor(b.mul(i, C(-1640531535)), p["seed"])            # 0x9E3779B1
        for sh, mul in ((16, 0x7FEB352D), (15, -2073254261), (16, 0)):   # 0x846CA68B
            h = b.xor(h, b.lshr(h, C(sh)))
            if mul:
                h = b.mul(h, C(mul))
        u = b.fmul(b.uitofp(b.lshr(h, C(8)), f32), F(1.0 / (1 << 24)))
        keep = b.select(b.fcmp_ordered(">=", u, p["p"]), b.fdiv(F(1.0), b.fsub(F(1.0), p["p"])), F(0.0))
        k.st(keep, p["MASK"], i)
        k.st(b.fmul(k.ld(p["X"], i), keep), p["Y"], i)
    k.grid_loop(p["n"], drop)
    k.ret()

    # ------------------------------------------------------------------------- low-bit weights for the forward pass
    # W [K, N] (input side first): each thread one column n and one block of 32 input rows; the same scales and
    # rounding as hub_gpu.qnumpy / lowbit.round_weights
    k = K("fakeq", [("W", G(f32)), ("WQ", G(f32)), ("K", i32), ("N", i32), ("kind", i32)])
    b, p = k.b, k.p
    nblk = b.udiv(b.add(p["K"], C(31)), C(32))

    def fq(i):
        n, blk = b.urem(i, p["N"]), b.udiv(i, p["N"])
        r0 = b.mul(blk, C(32))
        amax, mval = k.var(F(0.0)), k.var(F(0.0))

        def scan(r):
            v = k.ld(p["W"], b.add(b.mul(i64of(b, r), i64of(b, p["N"])), i64of(b, n)))
            av = b.call(fabs_f, [v])
            big = b.fcmp_ordered(">", av, b.load(amax))
            b.store(b.select(big, v, b.load(mval)), mval)
            b.store(b.select(big, av, b.load(amax)), amax)
        r1 = b.select(b.icmp_unsigned("<", b.add(r0, C(32)), p["K"]), b.add(r0, C(32)), p["K"])
        k.loop(r0, r1, C(1), scan)
        kind = p["kind"]
        top = F(0.0)
        for j, (_mb, _bias, t) in MINIFLOAT.items():
            top = b.select(b.icmp_unsigned("==", kind, C(j)), F(t), top)
        isfp = b.icmp_unsigned("<", kind, C(4))
        sc = b.select(isfp, b.fdiv(b.load(amax), top),
                      b.select(b.icmp_unsigned("==", kind, C(4)), b.fdiv(b.load(amax), F(127.0)),
                               b.fdiv(b.load(mval), F(-8.0))))
        inv = b.select(b.fcmp_ordered("!=", sc, F(0.0)), b.fdiv(F(1.0), sc), F(0.0))
        mbits = C(0)
        bias = C(0)
        for j, (mb, bi, _t) in MINIFLOAT.items():
            sel = b.icmp_unsigned("==", kind, C(j))
            mbits, bias = b.select(sel, C(mb), mbits), b.select(sel, C(bi), bias)

        def rnd(r):
            idx = b.add(b.mul(i64of(b, r), i64of(b, p["N"])), i64of(b, n))
            t = b.fmul(k.ld(p["W"], idx), inv)
            # minifloat: the spacing of values near |t| is 2^(max(e, 1 - bias) - mbits); round to it, ties to even
            tc = b.call(max_f, [b.fsub(F(0.0), top), b.call(min_f, [top, t])])
            a = b.call(fabs_f, [tc])
            e = b.sub(b.lshr(b.bitcast(a, i32), C(23)), C(127))
            emin = b.sub(C(1), bias)
            e = b.select(b.icmp_signed("<", e, emin), emin, e)
            step = b.bitcast(b.shl(b.add(b.sub(e, mbits), C(127)), C(23)), f32)
            qf = b.fmul(b.call(rint_f, [b.fdiv(a, step)]), step)
            qf = b.select(b.fcmp_ordered("<", tc, F(0.0)), b.fsub(F(0.0), qf), qf)
            lo = b.select(b.icmp_unsigned("==", kind, C(4)), F(-127.0), F(-8.0))
            hi = b.select(b.icmp_unsigned("==", kind, C(4)), F(127.0), F(7.0))
            qi = b.call(max_f, [lo, b.call(min_f, [hi, b.call(rint_f, [t])])])
            k.st(b.fmul(b.select(isfp, qf, qi), sc), p["WQ"], idx)
        k.loop(r0, r1, C(1), rnd)
    k.grid_loop(b.mul(nblk, p["N"]), fq)
    k.ret()

    # ------------------------------------------------------------------------- float32 matrix product (any GPU)
    # a block of 256 threads computes a 64 x 64 tile, each thread a 4 x 4 piece of it, straight from memory
    gemm_args = [("A", G(f32)), ("B", G(f32)), ("C", G(f32)), ("BIAS", G(f32)), ("M", i32), ("N", i32), ("K", i32),
                 ("sam", i32), ("sak", i32), ("sbk", i32), ("sbn", i32), ("ldc", i32), ("nb2", i32), ("a1", i32),
                 ("a2", i32), ("b1", i32), ("b2", i32), ("c1", i32), ("c2", i32), ("alpha", f32), ("beta", i32),
                 ("has_bias", i32)]
    k = K("gemm_simt", gemm_args)
    b, p = k.b, k.p
    offs = _batch_offsets(b, k.ctay, p, i64)
    nbn = b.udiv(b.add(p["N"], C(TILE - 1)), C(TILE))
    m0 = b.add(b.mul(b.udiv(k.cta, nbn), C(TILE)), b.mul(b.lshr(k.tid, C(4)), C(4)))
    n0 = b.add(b.mul(b.urem(k.cta, nbn), C(TILE)), b.mul(b.and_(k.tid, C(15)), C(4)))
    Mm1, Nm1 = b.sub(p["M"], C(1)), b.sub(p["N"], C(1))

    def umin(x, y):
        return b.select(b.icmp_unsigned("<", x, y), x, y)
    rows = [umin(b.add(m0, C(r)), Mm1) for r in range(4)]
    cols = [umin(b.add(n0, C(c)), Nm1) for c in range(4)]
    arow = [b.add(offs[0], b.mul(i64of(b, r), i64of(b, p["sam"]))) for r in rows]
    bcol = [b.add(offs[1], b.mul(i64of(b, c), i64of(b, p["sbn"]))) for c in cols]
    accs = [[k.var(F(0.0)) for _ in range(4)] for _ in range(4)]

    def kstep(kk):
        ka = b.mul(i64of(b, kk), i64of(b, p["sak"]))
        kb = b.mul(i64of(b, kk), i64of(b, p["sbk"]))
        av = [k.ld(p["A"], b.add(ar, ka)) for ar in arow]
        bv = [k.ld(p["B"], b.add(bc, kb)) for bc in bcol]
        for r in range(4):
            for c in range(4):
                b.store(b.call(fma_f, [av[r], bv[c], b.load(accs[r][c])]), accs[r][c])
    k.loop(C(0), p["K"], C(1), kstep)
    for r in range(4):
        for c in range(4):
            row, col = b.add(m0, C(r)), b.add(n0, C(c))
            with b.if_then(b.and_(b.icmp_unsigned("<", row, p["M"]), b.icmp_unsigned("<", col, p["N"]))):
                _store_c(k, b, p, offs[2], row, col, b.load(accs[r][c]), i64, C, F)
    k.ret()
    return _emit(m, "sm_75")


def _batch_offsets(b, z, p, i64):
    z1, z2 = b.udiv(z, p["nb2"]), b.urem(z, p["nb2"])
    out = []
    for s1, s2 in (("a1", "a2"), ("b1", "b2"), ("c1", "c2")):
        out.append(b.add(b.mul(b.zext(z1, i64), b.zext(p[s1], i64)), b.mul(b.zext(z2, i64), b.zext(p[s2], i64))))
    return out


def _store_c(k, b, p, coff, row, col, v, i64, C, F):
    idx = b.add(coff, b.add(b.mul(b.zext(row, i64), b.zext(p["ldc"], i64)), b.zext(col, i64)))
    ptr = b.gep(p["C"], [idx], inbounds=True)
    v = b.fmul(v, p["alpha"])
    tmp = k.var(v)
    with b.if_then(b.icmp_unsigned("!=", p["has_bias"], C(0))):
        b.store(b.fadd(b.load(tmp), b.load(b.gep(p["BIAS"], [b.zext(col, i64)], inbounds=True), align=4)), tmp)
    with b.if_then(b.icmp_unsigned("!=", p["beta"], C(0))):
        b.store(b.fadd(b.load(tmp), b.load(ptr, align=4)), tmp)
    b.store(b.load(tmp), ptr, align=4)


def _tc_module():
    """Matrix products on tensor cores (sm_80+): mma.sync m16n8k16, bfloat16 inputs, float32 sums. A block of 4
    warps computes 64 x 64 of C, a warp 32 x 32 (2 x 4 tiles of 16 x 8)."""
    ir, m = _setup("apsi_train_tc")
    g = _Gen(ir, m)
    i32, i64, f32, C, F, G = g.i32, g.i64, g.f32, g.C, g.F, g.G
    mma = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([f32] * 4), [i32] * 6 + [f32] * 4),
                       "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, "
                       "{$8,$9}, {$10,$11,$12,$13};", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", side_effect=False)
    gemm_args = [("A", G(f32)), ("B", G(f32)), ("C", G(f32)), ("BIAS", G(f32)), ("M", i32), ("N", i32), ("K", i32),
                 ("sam", i32), ("sak", i32), ("sbk", i32), ("sbn", i32), ("ldc", i32), ("nb2", i32), ("a1", i32),
                 ("a2", i32), ("b1", i32), ("b2", i32), ("c1", i32), ("c2", i32), ("alpha", f32), ("beta", i32),
                 ("has_bias", i32)]

    class KK:
        """Just enough of the kernel helper for _store_c."""

        def __init__(self, b, top):
            self.b, self.topb = b, top

        def var(self, init):
            v = self.topb.alloca(init.type)
            self.b.store(init, v)
            return v

    def kernel(name, passes):
        fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [t for _, t in gemm_args]), name=name)
        fn.calling_convention = "ptx_kernel"
        p = {n: a for (n, _), a in zip(gemm_args, fn.args)}
        allocas = fn.append_basic_block("allocas")
        top = fn.append_basic_block("top")
        ab = ir.IRBuilder(allocas)
        b = ir.IRBuilder(top)
        kk = KK(b, ab)
        tid, cta, ctay = b.call(g.tid_f, []), b.call(g.ctax_f, []), b.call(g.ctay_f, [])
        offs = _batch_offsets(b, ctay, p, i64)
        lane, warp = b.and_(tid, C(31)), b.lshr(tid, C(5))
        gq, tig = b.lshr(lane, C(2)), b.and_(lane, C(3))
        nbn = b.udiv(b.add(p["N"], C(TILE - 1)), C(TILE))
        n0 = b.add(b.mul(b.urem(cta, nbn), C(TILE)), b.mul(b.and_(warp, C(1)), C(32)))
        m0 = b.add(b.mul(b.udiv(cta, nbn), C(TILE)), b.mul(b.lshr(warp, C(1)), C(32)))
        Mm1, Nm1, Km1 = b.sub(p["M"], C(1)), b.sub(p["N"], C(1)), b.sub(p["K"], C(1))

        def umin(x, y):
            return b.select(b.icmp_unsigned("<", x, y), x, y)
        # A rows (mt, h): m0 + mt*16 + h*8 + gq; B columns nt: n0 + nt*8 + gq (clamped; never stored past the end)
        arow = [[b.add(offs[0], b.mul(b.zext(umin(b.add(b.add(m0, C(mt * 16 + h * 8)), gq), Mm1), i64),
                                      b.zext(p["sam"], i64))) for h in range(2)] for mt in range(2)]
        bcol = [b.add(offs[1], b.mul(b.zext(umin(b.add(b.add(n0, C(nt * 8)), gq), Nm1), i64), b.zext(p["sbn"], i64)))
                for nt in range(4)]

        def rne(u):
            return b.lshr(b.add(b.add(u, C(0x7FFF)), b.and_(b.lshr(u, C(16)), C(1))), C(16))

        def pack(x0, x1, low):
            """two float32 -> one i32 of two bfloat16 (x0 in the low half); low=True: the remainders x - bf16(x)."""
            out = []
            for x in (x0, x1):
                h = rne(b.bitcast(x, i32))
                if low:
                    h = rne(b.bitcast(b.fsub(x, b.bitcast(b.shl(h, C(16)), f32)), i32))
                out.append(h)
            return b.or_(out[0], b.shl(out[1], C(16)))

        def kval(base, kidx, stride, ptr):
            """the value at k = kidx (0 past the end; the address stays inside)."""
            ok = b.icmp_unsigned("<", kidx, p["K"])
            v = b.load(b.gep(ptr, [b.add(base, b.mul(b.zext(umin(kidx, Km1), i64), b.zext(stride, i64)))],
                             inbounds=True), align=4)
            return b.select(ok, v, F(0.0))

        loop = fn.append_basic_block("loop")
        body = fn.append_basic_block("body")
        done = fn.append_basic_block("done")
        b.branch(loop)
        b.position_at_end(loop)
        k0 = b.phi(i32)
        k0.add_incoming(C(0), top)
        accs = [[[b.phi(f32) for _ in range(4)] for _ in range(4)] for _ in range(2)]
        for row in accs:
            for frag in row:
                for a in frag:
                    a.add_incoming(ir.Constant(f32, 0.0), top)
        b.cbranch(b.icmp_unsigned("<", k0, p["K"]), body, done)
        b.position_at_end(body)
        kc = b.add(k0, b.mul(tig, C(2)))
        ahi, alo = [], []
        for mt in range(2):
            hi_r, lo_r = [], []
            for kh in range(2):                       # registers: (g, k) (g+8, k) (g, k+8) (g+8, k+8)
                for h in range(2):
                    kx = b.add(kc, C(8 * kh))
                    v0 = kval(arow[mt][h], kx, p["sak"], p["A"])
                    v1 = kval(arow[mt][h], b.add(kx, C(1)), p["sak"], p["A"])
                    hi_r.append(pack(v0, v1, False))
                    lo_r.append(pack(v0, v1, True) if passes == 3 else None)
            ahi.append(hi_r)
            alo.append(lo_r)
        bhi, blo = [], []
        for nt in range(4):
            hi_r, lo_r = [], []
            for kh in range(2):
                kx = b.add(kc, C(8 * kh))
                v0 = kval(bcol[nt], kx, p["sbk"], p["B"])
                v1 = kval(bcol[nt], b.add(kx, C(1)), p["sbk"], p["B"])
                hi_r.append(pack(v0, v1, False))
                lo_r.append(pack(v0, v1, True) if passes == 3 else None)
            bhi.append(hi_r)
            blo.append(lo_r)
        for mt in range(2):
            for nt in range(4):
                acc = list(accs[mt][nt])
                if passes == 3:                       # small terms first
                    r = b.call(mma, alo[mt] + bhi[nt] + acc)
                    acc = [b.extract_value(r, c) for c in range(4)]
                    r = b.call(mma, ahi[mt] + blo[nt] + acc)
                    acc = [b.extract_value(r, c) for c in range(4)]
                r = b.call(mma, ahi[mt] + bhi[nt] + acc)
                for c in range(4):
                    accs[mt][nt][c].add_incoming(b.extract_value(r, c), body)
        k0.add_incoming(b.add(k0, C(16)), body)
        b.branch(loop)
        b.position_at_end(done)
        for mt in range(2):
            for nt in range(4):
                for c in range(4):
                    row = b.add(b.add(m0, C(mt * 16 + (c // 2) * 8)), gq)
                    col = b.add(b.add(n0, C(nt * 8)), b.add(b.mul(tig, C(2)), C(c % 2)))
                    with b.if_then(b.and_(b.icmp_unsigned("<", row, p["M"]), b.icmp_unsigned("<", col, p["N"]))):
                        _store_c(kk, b, p, offs[2], row, col, accs[mt][nt][c], i64, C, F)
        b.ret_void()
        ab.branch(top)

    kernel("apsi_kernel", 1)
    kernel("gemm_tc3", 3)
    return _emit(m, "sm_80")


BM = BN = 128
BK = 32
PAD = 40                        # halves per shared-memory row: 80 bytes, so ldmatrix's 8 rows hit 8 different banks
PADT = 136                      # rows of a transposed tile (k rows of 128 m or n): 272 bytes, conflict-free too
TILED = ("gemm_tkn", "gemm_tkk", "gemm_tmn", "gemm_tmk")


def _tiled_module():
    """Matrix products on tensor cores with shared-memory tiles (sm_80+), the way fast GEMM libraries do them.

    A block of 8 warps computes 128 x 128 of C; each warp 64 x 32 (4 x 4 mma tiles of 16 x 8). The operands are read
    from global memory as float32 vectors of 4, rounded to bfloat16 and stored in shared memory (both A and B with
    k running along each row, rows padded to 40 halves so the reads have no bank conflicts), then read with ldmatrix
    and multiplied with mma.sync m16n8k16 (bfloat16 in, float32 sums). Two shared buffers: while the warps multiply
    one 32-wide slice of k, the next slice is already being read from memory into registers.

    Four kernels by which dimension of A and of B is contiguous in memory (gemm_t{k|m}{n|k}): A(m, k) with k
    contiguous (row-major) or m contiguous (transposed), B(k, n) with n contiguous or k contiguous (transposed).
    blockIdx.z splits k (split-K): each part adds its share to C atomically (for gradients of weights: few output
    tiles, a long k). Requirements (checked by the caller): the contiguous dimension and the other strides are
    multiples of 4 and every operand starts on a 16-byte boundary."""
    ir, m = _setup("apsi_train_tiled")
    g = _Gen(ir, m)
    i16, i32, i64, f32, C, F, G = ir.IntType(16), g.i32, g.i64, g.f32, g.C, g.F, g.G
    d = g.decl
    ctaz_f = d("llvm.nvvm.read.ptx.sreg.ctaid.z", i32, [])
    nctaz_f = d("llvm.nvvm.read.ptx.sreg.nctaid.z", i32, [])
    bar_f = d("llvm.nvvm.barrier0", ir.VoidType(), [])
    HALVES = 2 * 2 * BM * PAD
    smem = ir.GlobalVariable(m, ir.ArrayType(i16, HALVES), "apsi_tiles", addrspace=3)
    smem.linkage = "internal"
    smem.align = 16
    smem.initializer = ir.Constant(ir.ArrayType(i16, HALVES), None)
    mma = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([f32] * 4), [i32] * 6 + [f32] * 4),
                       "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, "
                       "{$8,$9}, {$10,$11,$12,$13};", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", side_effect=False)
    ldm = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([i32] * 4), [i64]),
                       "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l",
                       side_effect=True)
    ldmt = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([i32] * 4), [i64]),
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l",
                        side_effect=True)
    red = ir.InlineAsm(ir.FunctionType(ir.VoidType(), [i64, f32]), "red.global.add.f32 [$0], $1;", "l,f",
                       side_effect=True)
    args = [("A", G(f32)), ("B", G(f32)), ("C", G(f32)), ("BIAS", G(f32)), ("M", i32), ("N", i32), ("K", i32),
            ("sam", i32), ("sak", i32), ("sbk", i32), ("sbn", i32), ("ldc", i32), ("nb2", i32), ("a1", i32),
            ("a2", i32), ("b1", i32), ("b2", i32), ("c1", i32), ("c2", i32), ("alpha", f32), ("beta", i32),
            ("has_bias", i32), ("kchunk", i32)]

    def kernel(name, a_kc, b_kc):
        fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [t for _, t in args]), name=name)
        fn.calling_convention = "ptx_kernel"
        p = {n: a for (n, _), a in zip(args, fn.args)}
        allocas = fn.append_basic_block("allocas")
        top = fn.append_basic_block("top")
        ab = ir.IRBuilder(allocas)
        b = ir.IRBuilder(top)

        def var(init):
            v = ab.alloca(init.type)
            b.store(init, v)
            return v

        def z64(x):
            return b.zext(x, i64)

        def umin(x, y):
            return b.select(b.icmp_unsigned("<", x, y), x, y)
        tid, cta, ctay = b.call(g.tid_f, []), b.call(g.ctax_f, []), b.call(g.ctay_f, [])
        ctaz, nz = b.call(ctaz_f, []), b.call(nctaz_f, [])
        offs = _batch_offsets(b, ctay, p, i64)
        lane, warp = b.and_(tid, C(31)), b.lshr(tid, C(5))
        gq, tig = b.lshr(lane, C(2)), b.and_(lane, C(3))
        wm, wn = b.lshr(warp, C(2)), b.and_(warp, C(3))
        nbn = b.udiv(b.add(p["N"], C(BN - 1)), C(BN))
        bm = b.mul(b.udiv(cta, nbn), C(BM))
        bn = b.mul(b.urem(cta, nbn), C(BN))
        k_beg = b.mul(ctaz, p["kchunk"])
        k_end = umin(b.add(k_beg, p["kchunk"]), p["K"])
        ntiles = b.udiv(b.add(b.sub(k_end, umin(k_beg, k_end)), C(BK - 1)), C(BK))
        Mm1, Nm1 = b.sub(p["M"], C(1)), b.sub(p["N"], C(1))
        Km4 = b.sub(p["K"], C(4))

        def rne(x):
            u = b.bitcast(x, i32)
            return b.lshr(b.add(b.add(u, C(0x7FFF)), b.and_(b.lshr(u, C(16)), C(1))), C(16))

        def load4(base_off, ptr, contig_idx, contig_lim, other_idx, other_lim, contig_stride_is_1, s_other):
            """4 float32 along the contiguous dimension (zeros outside), as 4 bfloat16 bit patterns."""
            ok = b.and_(b.icmp_unsigned("<", contig_idx, contig_lim), b.icmp_unsigned("<", other_idx, other_lim))
            ci = umin(contig_idx, b.sub(contig_lim, C(4)))
            oi = umin(other_idx, b.sub(other_lim, C(1)))
            idx = b.add(base_off, b.add(b.mul(z64(oi), z64(s_other)), z64(ci)))
            vp = b.bitcast(b.gep(ptr, [idx], inbounds=True), ir.VectorType(f32, 4).as_pointer(1))
            vec = b.load(vp, align=16)
            out = []
            for j in range(4):
                v = b.select(ok, b.extract_element(vec, C(j)), F(0.0))
                out.append(rne(v))
            return out

        def sh_ptr(half_index):
            return b.gep(smem, [C(0), half_index], inbounds=True)

        def store_row4(half_index, hs):
            """4 halves at consecutive k (8 bytes)."""
            lo = b.or_(hs[0], b.shl(hs[1], C(16)))
            hi = b.or_(hs[2], b.shl(hs[3], C(16)))
            v = ir.Constant(ir.VectorType(i32, 2), None)
            v = b.insert_element(v, lo, C(0))
            v = b.insert_element(v, hi, C(1))
            b.store(v, b.bitcast(sh_ptr(half_index), ir.VectorType(i32, 2).as_pointer(3)), align=8)

        def store_col4(half_index, hs):
            """4 halves at consecutive rows (stride PAD), the same k."""
            for j in range(4):
                b.store(b.trunc(hs[j], i16), sh_ptr(b.add(half_index, C(j * PAD))), align=2)

        A_BASE = [C(0), C(BM * PAD)]
        B_BASE = [C(2 * BM * PAD), C(2 * BM * PAD + BN * PAD)]

        def fetch(kt):
            """The registers of one k slice: 4 vectors of A and 4 of B per thread."""
            k0 = b.add(k_beg, b.mul(kt, C(BK)))
            regs = []
            for i in range(4):
                idx = b.add(tid, C(256 * i))
                if a_kc:                              # A(m, k) = A[m * sam + k]
                    row, kq = b.lshr(idx, C(3)), b.mul(b.and_(idx, C(7)), C(4))
                    regs.append(("A", row, kq, load4(offs[0], p["A"], b.add(k0, kq), k_end, b.add(bm, row), p["M"],
                                                     True, p["sam"])))
                else:                                 # A(m, k) = A[k * sak + m]
                    kk, mq = b.lshr(idx, C(5)), b.mul(b.and_(idx, C(31)), C(4))
                    regs.append(("At", mq, kk, load4(offs[0], p["A"], b.add(bm, mq), p["M"], b.add(k0, kk), k_end,
                                                     True, p["sak"])))
                if b_kc:                              # B(k, n) = B[n * sbn + k]
                    nrow, kq = b.lshr(idx, C(3)), b.mul(b.and_(idx, C(7)), C(4))
                    regs.append(("B", nrow, kq, load4(offs[1], p["B"], b.add(k0, kq), k_end, b.add(bn, nrow), p["N"],
                                                      True, p["sbn"])))
                else:                                 # B(k, n) = B[k * sbk + n]
                    kk, nq = b.lshr(idx, C(5)), b.mul(b.and_(idx, C(31)), C(4))
                    regs.append(("Bt", nq, kk, load4(offs[1], p["B"], b.add(bn, nq), p["N"], b.add(k0, kk), k_end,
                                                     True, p["sbk"])))
            return regs

        def put(regs, buf):
            for kind, r, c, hs in regs:
                if kind in ("A", "At"):
                    base = b.select(b.icmp_unsigned("==", buf, C(0)), A_BASE[0], A_BASE[1])
                else:
                    base = b.select(b.icmp_unsigned("==", buf, C(0)), B_BASE[0], B_BASE[1])
                if kind in ("A", "B"):                # [m or n][k]: 4 consecutive k
                    store_row4(b.add(base, b.add(b.mul(r, C(PAD)), c)), hs)
                else:                                 # [k][m or n]: 4 consecutive m or n, stored the same way
                    store_row4(b.add(base, b.add(b.mul(c, C(PADT)), r)), hs)

        accs = [[[var(F(0.0)) for _ in range(4)] for _ in range(4)] for _ in range(4)]
        put(fetch(C(0)), C(0))
        b.call(bar_f, [])
        it = var(C(0))
        head = fn.append_basic_block("head")
        body = fn.append_basic_block("body")
        done = fn.append_basic_block("done")
        b.branch(head)
        b.position_at_end(head)
        i = b.load(it)
        b.cbranch(b.icmp_unsigned("<", i, ntiles), body, done)
        b.position_at_end(body)
        nxt = b.add(i, C(1))
        has_next = b.icmp_unsigned("<", nxt, ntiles)
        regs = fetch(umin(nxt, b.sub(ntiles, C(1))))
        buf = b.and_(i, C(1))
        abase = b.select(b.icmp_unsigned("==", buf, C(0)), A_BASE[0], A_BASE[1])
        bbase = b.select(b.icmp_unsigned("==", buf, C(0)), B_BASE[0], B_BASE[1])
        for ks in range(BK // 16):
            l7, l8, l16 = b.and_(lane, C(7)), b.and_(b.lshr(lane, C(3)), C(1)), b.lshr(lane, C(4))
            afr = []
            for mt in range(4):
                if a_kc:                              # [m][k]: rows of A
                    row = b.add(b.add(b.mul(wm, C(64)), C(mt * 16)), b.and_(lane, C(15)))
                    col = b.add(C(ks * 16), b.mul(l16, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(abase, b.add(b.mul(row, C(PAD)), col))), i64)
                    r = b.call(ldm, [addr])
                else:                                 # [k][m]: transposed while loading
                    krow = b.add(C(ks * 16), b.add(l7, b.mul(l16, C(8))))
                    mcol = b.add(b.add(b.mul(wm, C(64)), C(mt * 16)), b.mul(l8, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(abase, b.add(b.mul(krow, C(PADT)), mcol))), i64)
                    r = b.call(ldmt, [addr])
                afr.append([b.extract_value(r, j) for j in range(4)])
            bfr = []
            for pr in range(2):
                if b_kc:                              # [n][k]
                    nrow = b.add(b.add(b.mul(wn, C(32)), C(pr * 16)), b.add(l7, b.mul(l16, C(8))))
                    col = b.add(C(ks * 16), b.mul(l8, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(bbase, b.add(b.mul(nrow, C(PAD)), col))), i64)
                    r = b.call(ldm, [addr])
                else:                                 # [k][n], transposed while loading
                    krow = b.add(C(ks * 16), b.add(l7, b.mul(l8, C(8))))
                    ncol = b.add(b.add(b.mul(wn, C(32)), C(pr * 16)), b.mul(l16, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(bbase, b.add(b.mul(krow, C(PADT)), ncol))), i64)
                    r = b.call(ldmt, [addr])
                q = [b.extract_value(r, j) for j in range(4)]
                bfr += [[q[0], q[1]], [q[2], q[3]]]
            for mt in range(4):
                for nt in range(4):
                    acc = [b.load(a) for a in accs[mt][nt]]
                    r = b.call(mma, afr[mt] + bfr[nt] + acc)
                    for c in range(4):
                        b.store(b.extract_value(r, c), accs[mt][nt][c])
        with b.if_then(has_next):
            put(regs, b.and_(nxt, C(1)))
        b.call(bar_f, [])
        b.store(nxt, it)
        b.branch(head)
        b.position_at_end(done)
        split = b.icmp_unsigned(">", nz, C(1))
        first = b.icmp_unsigned("==", ctaz, C(0))
        for mt in range(4):
            for nt in range(4):
                for c in range(4):
                    row = b.add(b.add(bm, b.add(b.mul(wm, C(64)), C(mt * 16 + (c // 2) * 8))), gq)
                    col = b.add(b.add(bn, b.add(b.mul(wn, C(32)), C(nt * 8))), b.add(b.mul(tig, C(2)), C(c % 2)))
                    with b.if_then(b.and_(b.icmp_unsigned("<", row, p["M"]), b.icmp_unsigned("<", col, p["N"]))):
                        idx = b.add(offs[2], b.add(b.mul(z64(row), z64(p["ldc"])), z64(col)))
                        ptr = b.gep(p["C"], [idx], inbounds=True)
                        v = b.fmul(b.load(accs[mt][nt][c]), p["alpha"])
                        tmp = var(v)
                        with b.if_then(b.and_(b.icmp_unsigned("!=", p["has_bias"], C(0)), first)):
                            bp = b.gep(p["BIAS"], [z64(col)], inbounds=True)
                            b.store(b.fadd(b.load(tmp), b.load(bp, align=4)), tmp)
                        with b.if_else(split) as (then, other):
                            with then:                # each part of k adds its share (C holds the rest)
                                b.call(red, [b.ptrtoint(ptr, i64), b.load(tmp)])
                            with other:
                                with b.if_then(b.icmp_unsigned("!=", p["beta"], C(0))):
                                    b.store(b.fadd(b.load(tmp), b.load(ptr, align=4)), tmp)
                                b.store(b.load(tmp), ptr, align=4)
        b.ret_void()
        ab.branch(top)

    kernel("apsi_kernel", True, False)            # = gemm_tkn: x @ W
    for name, a_kc, b_kc in (("gemm_tkk", True, True), ("gemm_tmn", False, False), ("gemm_tmk", False, True)):
        kernel(name, a_kc, b_kc)
    return _emit(m, "sm_80")


_cache = {}


def ptx(which):
    """The PTX text of "main" or "tc": from memory, the disk cache (keyed by this file and LLVM), or built."""
    if which in _cache:
        return _cache[which]
    import hashlib
    path = None
    try:
        import llvmlite
        from .security import config_dir
        with open(__file__, "rb") as fh:
            key = hashlib.sha256(fh.read() + llvmlite.__version__.encode()).hexdigest()[:20]
        path = os.path.join(config_dir(), "cache", "gpu", f"train-{which}-{key}.ptx")
        if os.environ.get("APSI_NO_CACHE", "").strip() not in ("1", "true", "yes"):
            with open(path, encoding="ascii") as fh:
                text = fh.read()
            if ".entry apsi_kernel" in text:
                _cache[which] = text
                return text
    except (OSError, ImportError):
        pass
    text = {"main": _main_module, "tc": _tc_module, "tiled": _tiled_module}[which]()
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="ascii") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except OSError:
            pass
    _cache[which] = text
    return text
