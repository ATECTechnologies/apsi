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
EW1 = {"exp": 0, "log": 1, "sqrt": 2, "abs": 3, "square": 4, "recip": 5, "neg": 6, "rsqrt": 7}
EW2 = {"div": 0, "max": 1, "min": 2, "pow": 3}
ACTS = {"relu": 0, "gelu": 1, "silu": 2, "swish": 2, "tanh": 3, "sigmoid": 4, "linear": 5, "identity": 5,
        "leaky_relu": 6, "elu": 7, "softplus": 8, "mish": 9, "sin": 10, "cos": 11}
FAKEQ = {"fp8": 0, "fp6": 1, "fp6_e2m3": 2, "fp4": 3, "int8": 4, "int4": 5}
MINIFLOAT = {0: (3, 7, 448.0), 1: (2, 3, 28.0), 2: (3, 1, 7.5), 3: (1, 1, 6.0)}     # mantissa bits, bias, top
TILE = 64

MAIN = ("apsi_kernel", "axpy", "add3", "mul3", "act_f", "act_b", "colsum", "ln_f", "ln_b", "embed_f", "embed_b",
        "pos_f", "pos_b", "softmax_f", "softmax_b", "ce", "mse", "sumsq", "adam", "sgd", "drop_f", "fakeq",
        "gemm_simt", "gather", "rope", "spmm", "edge_dot", "seg_softmax", "seg_softmax_b", "dot_acc", "scale_add",
        "cols", "colmul", "topk_gate", "topk_gate_b", "loss_ew", "im2col", "col2im", "perm_nc", "pool_f", "pool_b",
        "lstm_f", "lstm_b", "gru_a", "gru_b", "gru_b1", "gru_b2", "taylor_f", "taylor_b", "rowbias", "acc2",
        "fin2", "ew1", "ew2", "colsum_fin", "embed_bs", "parts_sum", "ksum", "tanh_s", "axpy_s", "gate_g", "pad2")
TC = ("apsi_kernel", "gemm_tc3")                     # apsi_kernel = the one-pass product
# bytes of dynamic shared memory of the deep builds (3 stages of 128 x 32 tiles of A and B, rows padded to 40 halves)
DYN_BYTES = 3 * 2 * max(128 * 40, 32 * 136) * 2


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
        self.nctay_f = d("llvm.nvvm.read.ptx.sreg.nctaid.y", i32, [])

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
    sin_f = decl("llvm.nvvm.sin.approx.f", f32, [f32])
    cos_f = decl("llvm.nvvm.cos.approx.f", f32, [f32])
    bar_f = decl("llvm.nvvm.barrier0", ir.VoidType(), [])
    membar_f = decl("llvm.nvvm.membar.gl", ir.VoidType(), [])
    ld_cg_f = ir.InlineAsm(ir.FunctionType(f32, [i64]), "ld.global.cg.f32 $0, [$1];", "=f,l", side_effect=True)
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

        def ld_cg(self, ptr, idx):
            """A load that skips the multiprocessor's own cache (values other blocks of this launch wrote)."""
            return self.b.call(ld_cg_f, [self.b.ptrtoint(self.gep(ptr, idx), i64)])

        def when_last(self, CNT, slot, nblocks, body):
            """body() in the last block to finish (of nblocks sharing counter CNT[slot]), after every block's
            writes are visible; the counter is reset for the next launch. Which block runs it changes from run
            to run, what it adds and in what order does not: the same bits every run, one launch instead of two."""
            b = self.b
            b.call(membar_f, [])
            b.call(bar_f, [])
            flag = b.gep(smem, [C(0), C(31)], inbounds=True)
            with b.if_then(b.icmp_unsigned("==", self.tid, C(0))):
                old = b.atomic_rmw("add", self.gep(CNT, slot), C(1), "acq_rel")
                b.store(b.select(b.icmp_unsigned("==", old, b.sub(nblocks, C(1))), F(1.0), F(0.0)), flag)
            b.call(bar_f, [])
            with b.if_then(b.fcmp_ordered("==", b.load(flag), F(1.0))):
                b.call(membar_f, [])
                body()
                with b.if_then(b.icmp_unsigned("==", self.tid, C(0))):
                    b.store(C(0), self.gep(CNT, slot))

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

    def sum_parts(k, PART, n, ACC):
        """ACC[0] += PART[0..n) in a fixed order (parts_sum's order): run by one block."""
        b = k.b
        s_ = k.var(F(0.0))
        k.loop(k.tid, n, C(256), lambda i: b.store(b.fadd(b.load(s_), k.ld_cg(PART, i)), s_))
        tot = k.block_sum(b.load(s_))
        with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
            k.st(b.fadd(k.ld_cg(ACC, C(0)), tot), ACC, C(0))

    def exp_acc(b, x):
        """exp with Cody-Waite reduction: 2^k exp(r), |r| <= ln2/2, k exact; ~2 ulp over the whole range."""
        k = b.call(rint_f, [b.fmul(x, F(LOG2E))])
        r = b.call(fma_f, [b.fsub(F(0.0), k), F(0.693145751953125), x])
        r = b.call(fma_f, [b.fsub(F(0.0), k), F(1.428606765330187e-06), r])
        er = b.call(ex2_f, [b.fmul(r, F(LOG2E))])
        k1 = b.call(rint_f, [b.fmul(k, F(0.5))])          # 2^k as two halves (no overflow for large k)
        return b.fmul(b.fmul(er, b.call(ex2_f, [k1])), b.call(ex2_f, [b.fsub(k, k1)]))

    def log_acc(b, x):
        """log: lg2 then one Newton step on exp(y) = x (accurate near 1 as well)."""
        y = b.fmul(b.call(lg2_f, [x]), F(0.6931471805599453))
        return b.fadd(y, b.fsub(b.fmul(x, exp_acc(b, b.fsub(F(0.0), y))), F(1.0)))

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
        neg = b.fcmp_ordered(">", x, F(0.0))
        ex = exp(k, b.call(min_f, [x, F(0.0)]))
        elu = b.select(neg, x, b.fsub(ex, F(1.0)))
        delu = b.select(neg, F(1.0), ex)
        ax = b.call(fabs_f, [x])                      # softplus = max(x, 0) + log(1 + exp(-|x|))
        sp = b.fadd(b.call(max_f, [x, F(0.0)]),
                    b.fdiv(b.call(lg2_f, [b.fadd(F(1.0), exp(k, b.fsub(F(0.0), ax)))]), F(LOG2E)))
        tsp = tanh(k, sp)
        mish = b.fmul(x, tsp)
        dmish = b.fadd(tsp, b.fmul(b.fmul(x, b.fsub(F(1.0), b.fmul(tsp, tsp))), s))
        red = b.fsub(x, b.fmul(F(6.283185307179586), b.call(rint_f, [b.fmul(x, F(0.15915494309189535))])))
        sn, cs = b.call(sin_f, [red]), b.call(cos_f, [red])
        vals = [(relu, drelu), (gelu, dgelu), (silu, dsilu), (th, dth), (s, dsig), (x, F(1.0)), (leaky, dleaky),
                (elu, delu), (sp, s), (mish, dmish), (sn, cs), (cs, b.fsub(F(0.0), sn))]
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
    # grid (ceil(N/32), chunks): a block is 32 columns x 8 lanes of rows (a warp reads 32 adjacent values of a row);
    # DB[n] += sum dy; mode 1 (layer norm) DG[n] += sum dy * (x - mean) * rstd; mode 2 (RMS norm) DG[n] += sum
    # dy * x * rstd and no DB. The 8 lanes are added in lane order in shared memory; with several chunks each
    # writes its sums to WB / WG [chunk, N] and the last chunk to finish adds them in chunk order: the same result
    # every run, one launch.
    cs_sm = ir.GlobalVariable(m, ir.ArrayType(f32, 512), "apsi_colsum_s", addrspace=3)
    cs_sm.linkage = "internal"
    cs_sm.initializer = ir.Constant(ir.ArrayType(f32, 512), None)
    k = K("colsum", [("DY", G(f32)), ("X", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)), ("DG", G(f32)),
                     ("DB", G(f32)), ("M", i32), ("N", i32), ("rows", i32), ("mode", i32), ("WB", G(f32)),
                     ("WG", G(f32)), ("part", i32), ("CNT", G(i32))])
    b, p = k.b, k.p
    cl, rl = b.and_(k.tid, C(31)), b.lshr(k.tid, C(5))
    n = b.add(b.mul(k.cta, C(32)), cl)
    ncol = b.icmp_unsigned("<", n, p["N"])
    nc = b.select(ncol, n, C(0))
    r0 = b.mul(k.ctay, p["rows"])
    r1 = b.select(b.icmp_unsigned("<", b.add(r0, p["rows"]), p["M"]), b.add(r0, p["rows"]), p["M"])
    sb, sg = k.var(F(0.0)), k.var(F(0.0))
    Mm1 = b.sub(p["M"], C(1))
    is_ln = b.icmp_unsigned("==", p["mode"], C(1))
    with_x = b.icmp_unsigned("!=", p["mode"], C(0))

    def rows4(r):                                     # rows r, r+8, r+16, r+24 of this lane: 4 loads in flight
        dys, xs = [], []
        for u in range(4):
            rr = b.add(r, C(8 * u))
            ok = b.and_(b.icmp_unsigned("<", rr, r1), ncol)
            rc = b.select(ok, rr, Mm1)
            idx = b.add(b.mul(i64of(b, rc), i64of(b, p["N"])), i64of(b, nc))
            dys.append((ok, rc, idx, k.ld(p["DY"], idx)))
        for ok, rc, idx, dy in dys:
            dy = b.select(ok, dy, F(0.0))
            b.store(b.fadd(b.load(sb), dy), sb)
            with b.if_then(with_x):
                mu = b.select(is_ln, k.ld(p["MEAN"], rc), F(0.0))
                xh = b.fmul(b.fsub(k.ld(p["X"], idx), mu), k.ld(p["RSTD"], rc))
                b.store(b.call(fma_f, [dy, xh, b.load(sg)]), sg)
    k.loop(b.add(r0, rl), r1, C(32), rows4)
    b.store(b.load(sb), b.gep(cs_sm, [C(0), k.tid], inbounds=True))
    b.store(b.load(sg), b.gep(cs_sm, [C(0), b.add(k.tid, C(256))], inbounds=True))
    b.call(bar_f, [])
    tb, tg = k.var(F(0.0)), k.var(F(0.0))
    with b.if_then(b.and_(b.icmp_unsigned("==", rl, C(0)), ncol)):
        for lane in range(8):                         # the 8 lanes in order
            b.store(b.fadd(b.load(tb), b.load(b.gep(cs_sm, [C(0), b.add(cl, C(32 * lane))], inbounds=True))), tb)
            b.store(b.fadd(b.load(tg), b.load(b.gep(cs_sm, [C(0), b.add(cl, C(256 + 32 * lane))], inbounds=True))),
                    tg)
        widx = b.add(b.mul(i64of(b, k.ctay), i64of(b, p["N"])), i64of(b, n))
        with b.if_else(b.icmp_unsigned("!=", p["part"], C(0))) as (parted, direct):
            with parted:
                with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
                    k.st(b.load(tb), p["WB"], widx)
                with b.if_then(with_x):
                    k.st(b.load(tg), p["WG"], widx)
            with direct:
                with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
                    k.st(b.fadd(k.ld(p["DB"], n), b.load(tb)), p["DB"], n)
                with b.if_then(with_x):
                    k.st(b.fadd(k.ld(p["DG"], n), b.load(tg)), p["DG"], n)

    def colsum_last():                                # the chunks of these 32 columns, added in chunk order
        with b.if_then(b.and_(b.icmp_unsigned("==", rl, C(0)), ncol)):
            ub, ug = k.var(F(0.0)), k.var(F(0.0))

            def body(c):
                at = b.add(b.mul(i64of(b, c), i64of(b, p["N"])), i64of(b, n))
                with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
                    b.store(b.fadd(b.load(ub), k.ld_cg(p["WB"], at)), ub)
                with b.if_then(with_x):
                    b.store(b.fadd(b.load(ug), k.ld_cg(p["WG"], at)), ug)
            k.loop(C(0), b.call(g.nctay_f, []), C(1), body)
            with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
                k.st(b.fadd(k.ld(p["DB"], n), b.load(ub)), p["DB"], n)
            with b.if_then(with_x):
                k.st(b.fadd(k.ld(p["DG"], n), b.load(ug)), p["DG"], n)
    with b.if_then(b.icmp_unsigned("!=", p["part"], C(0))):
        k.when_last(p["CNT"], k.cta, b.call(g.nctay_f, []), colsum_last)
    k.ret()

    # ksum: C[r, c] += sum over the parts z of W[z, r, c], in order of z (the parts of a split-k product)
    k = K("ksum", [("Cm", G(f32)), ("W", G(f32)), ("M", i32), ("N", i32), ("ldc", i32), ("parts", i32)])
    b, p = k.b, k.p

    def ksum(i):
        r, c = b.udiv(i, p["N"]), b.urem(i, p["N"])
        at = b.add(b.mul(i64of(b, r), i64of(b, p["ldc"])), i64of(b, c))
        plane = b.mul(i64of(b, p["M"]), i64of(b, p["ldc"]))
        s_ = k.var(F(0.0))
        k.loop(C(0), p["parts"], C(1), lambda z: b.store(b.fadd(b.load(s_), k.ld(p["W"], b.add(
            b.mul(i64of(b, z), plane), at))), s_))
        k.st(b.fadd(k.ld(p["Cm"], at), b.load(s_)), p["Cm"], at)
    k.grid_loop(b.mul(p["M"], p["N"]), ksum)
    k.ret()

    # colsum_fin: DB[n] += sum over chunks of WB[c, n] (in chunk order), the same for DG
    k = K("colsum_fin", [("WB", G(f32)), ("WG", G(f32)), ("DG", G(f32)), ("DB", G(f32)), ("N", i32),
                         ("chunks", i32), ("mode", i32)])
    b, p = k.b, k.p

    def cfin(n):
        sb, sg = k.var(F(0.0)), k.var(F(0.0))

        def body(c):
            at = b.add(b.mul(i64of(b, c), i64of(b, p["N"])), i64of(b, n))
            with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
                b.store(b.fadd(b.load(sb), k.ld(p["WB"], at)), sb)
            with b.if_then(b.icmp_unsigned("!=", p["mode"], C(0))):
                b.store(b.fadd(b.load(sg), k.ld(p["WG"], at)), sg)
        k.loop(C(0), p["chunks"], C(1), body)
        with b.if_then(b.icmp_unsigned("!=", p["mode"], C(2))):
            k.st(b.fadd(k.ld(p["DB"], n), b.load(sb)), p["DB"], n)
        with b.if_then(b.icmp_unsigned("!=", p["mode"], C(0))):
            k.st(b.fadd(k.ld(p["DG"], n), b.load(sg)), p["DG"], n)
    k.grid_loop(p["N"], cfin)
    k.ret()

    # ------------------------------------------------------------------------- layer norm / RMS norm, a warp a row
    k = K("ln_f", [("X", G(f32)), ("G", G(f32)), ("B", G(f32)), ("Y", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)),
                   ("M", i32), ("N", i32), ("eps", f32), ("rms", i32), ("Y16", G(f32)), ("has16", i32)])
    b, p = k.b, k.p
    lane = b.and_(k.tid, C(31))
    row = b.add(b.mul(k.cta, C(8)), b.lshr(k.tid, C(5)))
    with b.if_then(b.icmp_unsigned("<", row, p["M"])):
        base = b.mul(i64of(b, row), i64of(b, p["N"]))
        isrms = b.icmp_unsigned("!=", p["rms"], C(0))
        s1 = k.var(F(0.0))
        k.loop(lane, p["N"], C(32), lambda i: b.store(b.fadd(b.load(s1), k.ld(p["X"], b.add(base, i64of(b, i)))), s1))
        nf = b.uitofp(p["N"], f32)
        mean = b.select(isrms, F(0.0), b.fdiv(k.warp_reduce(b.load(s1)), nf))
        s2 = k.var(F(0.0))

        def sq(i):
            dd = b.fsub(k.ld(p["X"], b.add(base, i64of(b, i))), mean)
            b.store(b.call(fma_f, [dd, dd, b.load(s2)]), s2)
        k.loop(lane, p["N"], C(32), sq)
        rstd = b.fdiv(F(1.0), b.call(sqrt_f, [b.fadd(b.fdiv(k.warp_reduce(b.load(s2)), nf), p["eps"])]))

        def out(i):
            idx = b.add(base, i64of(b, i))
            y = b.fmul(b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd), k.ld(p["G"], i))
            yv = k.var(y)
            with b.if_then(b.not_(isrms)):
                b.store(b.fadd(y, k.ld(p["B"], i)), yv)
            yo = b.load(yv)
            k.st(yo, p["Y"], idx)
            # its bfloat16 twin in the same pass (round to nearest even, as to_bf16): the products that read this
            # output take it as it is, no conversion pass per product
            with b.if_then(b.icmp_unsigned("!=", p["has16"], C(0))):
                bits = b.bitcast(yo, i32)
                rnd = b.add(bits, b.add(C(0x7FFF), b.and_(b.lshr(bits, C(16)), C(1))))
                nan = b.fcmp_unordered("uno", yo, yo)
                hb = b.select(nan, b.or_(bits, C(0x00400000)), rnd)
                p16 = b.bitcast(p["Y16"], ir.IntType(16).as_pointer(1))
                b.store(b.trunc(b.lshr(hb, C(16)), ir.IntType(16)), b.gep(p16, [idx], inbounds=True), align=2)
        k.loop(lane, p["N"], C(32), out)
        with b.if_then(b.icmp_unsigned("==", lane, C(0))):
            k.st(mean, p["MEAN"], row)
            k.st(rstd, p["RSTD"], row)
    k.ret()

    k = K("ln_b", [("X", G(f32)), ("G", G(f32)), ("DY", G(f32)), ("MEAN", G(f32)), ("RSTD", G(f32)), ("DX", G(f32)),
                   ("M", i32), ("N", i32), ("rms", i32), ("RES", G(f32)), ("has_res", i32)])
    b, p = k.b, k.p
    lane = b.and_(k.tid, C(31))
    row = b.add(b.mul(k.cta, C(8)), b.lshr(k.tid, C(5)))
    with b.if_then(b.icmp_unsigned("<", row, p["M"])):
        base = b.mul(i64of(b, row), i64of(b, p["N"]))
        isrms = b.icmp_unsigned("!=", p["rms"], C(0))
        mean = b.select(isrms, F(0.0), k.ld(p["MEAN"], row))
        rstd = k.ld(p["RSTD"], row)
        a1, a2 = k.var(F(0.0)), k.var(F(0.0))

        def acc(i):
            idx = b.add(base, i64of(b, i))
            gdy = b.fmul(k.ld(p["DY"], idx), k.ld(p["G"], i))
            xh = b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd)
            b.store(b.fadd(b.load(a1), gdy), a1)
            b.store(b.call(fma_f, [gdy, xh, b.load(a2)]), a2)
        k.loop(lane, p["N"], C(32), acc)
        nf = b.uitofp(p["N"], f32)
        m1 = b.select(isrms, F(0.0), b.fdiv(k.warp_reduce(b.load(a1)), nf))
        m2 = b.fdiv(k.warp_reduce(b.load(a2)), nf)
        hasres = b.icmp_unsigned("!=", p["has_res"], C(0))

        def dx(i):
            idx = b.add(base, i64of(b, i))
            gdy = b.fmul(k.ld(p["DY"], idx), k.ld(p["G"], i))
            xh = b.fmul(b.fsub(k.ld(p["X"], idx), mean), rstd)
            v = b.fmul(rstd, b.fsub(b.fsub(gdy, m1), b.fmul(xh, m2)))
            out = k.var(v)
            with b.if_then(hasres):                   # + the skip connection's gradient
                b.store(b.fadd(v, k.ld(p["RES"], idx)), out)
            k.st(b.load(out), p["DX"], idx)
        k.loop(lane, p["N"], C(32), dx)
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

    # embed_bs: the same sums as embed_b without atomics: POS lists the positions of each target row in order
    # (OFF[u]..OFF[u+1]), ROWS[u] the row; one thread per (u, d) sums its positions in order, one write
    k = K("embed_bs", [("DY", G(f32)), ("POS", G(i32)), ("OFF", G(i32)), ("ROWS", G(i32)), ("DE", G(f32)),
                       ("U", i32), ("D", i32)])
    b, p = k.b, k.p

    def embbs(i):
        u, d = b.udiv(i, p["D"]), b.urem(i, p["D"])
        s = k.var(F(0.0))
        lo, hi = k.ld(p["OFF"], u), k.ld(p["OFF"], b.add(u, C(1)))
        k.loop(lo, hi, C(1), lambda j: b.store(b.fadd(b.load(s), k.ld(p["DY"], b.add(
            b.mul(i64of(b, k.ld(p["POS"], j)), i64of(b, p["D"])), i64of(b, d)))), s))
        at = b.add(b.mul(i64of(b, k.ld(p["ROWS"], u)), i64of(b, p["D"])), i64of(b, d))
        k.st(b.fadd(k.ld(p["DE"], at), b.load(s)), p["DE"], at)
    k.grid_loop(b.mul(p["U"], p["D"]), embbs)
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

    # ------------------------------------------------------------------------- rotary positions, in place
    # X: rows of hd (a head of a token); row r is token r / H, at position (r / H) % T; pairs (j, j + hd / 2):
    # y1 = x1 cos - x2 sin, y2 = x2 cos + x1 sin (sign -1: the inverse rotation, the backward pass)
    k = K("rope", [("X", G(f32)), ("CS", G(f32)), ("SN", G(f32)), ("rows", i32), ("hd", i32), ("H", i32),
                   ("T", i32), ("sign", f32)])
    b, p = k.b, k.p
    half = b.lshr(p["hd"], C(1))

    def rope(i):
        r, j = b.udiv(i, half), b.urem(i, half)
        t = b.urem(b.udiv(r, p["H"]), p["T"])
        base = b.add(b.mul(i64of(b, r), i64of(b, p["hd"])), i64of(b, j))
        tix = b.add(b.mul(i64of(b, t), i64of(b, half)), i64of(b, j))
        c, sn = k.ld(p["CS"], tix), b.fmul(k.ld(p["SN"], tix), p["sign"])
        i2 = b.add(base, i64of(b, half))
        x1, x2 = k.ld(p["X"], base), k.ld(p["X"], i2)
        k.st(b.fsub(b.fmul(x1, c), b.fmul(x2, sn)), p["X"], base)
        k.st(b.fadd(b.fmul(x2, c), b.fmul(x1, sn)), p["X"], i2)
    k.grid_loop(b.mul(p["rows"], half), rope)
    k.ret()

    # ------------------------------------------------------------------------- sparse graphs (rows in CSR order)
    # spmm: Y[n, h, f] (+)= sum over e in OFF[n]..OFF[n+1] of v(e, h) * X[COL[e], h, f]; v = 1, or VAL[e' * H + h]
    # with e' = e or VPERM[e]. mode: 1 values, 2 values through VPERM, 4 add to Y. A thread a (row, head, feature):
    # the sum in a register, one write, no atomics (the same result every run).
    k = K("spmm", [("X", G(f32)), ("COL", G(i32)), ("VAL", G(f32)), ("VPERM", G(i32)), ("OFF", G(i32)),
                   ("Y", G(f32)), ("N", i32), ("H", i32), ("F", i32), ("mode", i32)])
    b, p = k.b, k.p
    HF = b.mul(p["H"], p["F"])

    def spmm(i):
        n, r = b.udiv(i, HF), b.urem(i, HF)
        hh = b.udiv(r, p["F"])
        acc = k.var(F(0.0))
        hasv = b.icmp_unsigned("!=", b.and_(p["mode"], C(1)), C(0))
        perm = b.icmp_unsigned("!=", b.and_(p["mode"], C(2)), C(0))

        def edge(e):
            c = k.ld(p["COL"], e)
            x = k.ld(p["X"], b.add(b.mul(i64of(b, c), i64of(b, HF)), i64of(b, r)))
            ve = b.select(perm, k.ld(p["VPERM"], b.select(perm, e, C(0))), e)
            v = b.select(hasv, k.ld(p["VAL"], b.select(hasv, b.add(b.mul(i64of(b, ve), i64of(b, p["H"])),
                                                                     i64of(b, hh)), b.zext(C(0), i64))), F(1.0))
            b.store(b.call(fma_f, [v, x, b.load(acc)]), acc)
        k.loop(k.ld(p["OFF"], n), k.ld(p["OFF"], b.add(n, C(1))), C(1), edge)
        add = b.icmp_unsigned("!=", b.and_(p["mode"], C(4)), C(0))
        tot = b.load(acc)
        tot = b.select(add, b.fadd(tot, k.ld(p["Y"], i)), tot)
        k.st(tot, p["Y"], i)
    k.grid_loop(b.mul(p["N"], HF), spmm)
    k.ret()

    # edge_dot: DA[e, h] = sum_f DY[ROW[e], h, f] * X[COL[e], h, f]  (the gradient of an edge's weight)
    k = K("edge_dot", [("DY", G(f32)), ("X", G(f32)), ("ROW", G(i32)), ("COL", G(i32)), ("DA", G(f32)),
                       ("E", i32), ("H", i32), ("F", i32)])
    b, p = k.b, k.p
    HF = b.mul(p["H"], p["F"])

    def edot(i):
        e, hh = b.udiv(i, p["H"]), b.urem(i, p["H"])
        r0 = b.add(b.mul(i64of(b, k.ld(p["ROW"], e)), i64of(b, HF)), i64of(b, b.mul(hh, p["F"])))
        c0 = b.add(b.mul(i64of(b, k.ld(p["COL"], e)), i64of(b, HF)), i64of(b, b.mul(hh, p["F"])))
        acc = k.var(F(0.0))
        k.loop(C(0), p["F"], C(1), lambda f: b.store(b.call(fma_f, [k.ld(p["DY"], b.add(r0, i64of(b, f))),
                                                                     k.ld(p["X"], b.add(c0, i64of(b, f))),
                                                                     b.load(acc)]), acc))
        k.st(b.load(acc), p["DA"], i)
    k.grid_loop(b.mul(p["E"], p["H"]), edot)
    k.ret()

    # seg_softmax: per (row n, head h), over its edges: z = leaky(S, slope), A = exp(z - max) / (sum + 1e-16)
    k = K("seg_softmax", [("S", G(f32)), ("OFF", G(i32)), ("A", G(f32)), ("N", i32), ("H", i32), ("slope", f32)])
    b, p = k.b, k.p

    def leaky(x):
        return b.select(b.fcmp_ordered(">", x, F(0.0)), x, b.fmul(x, p["slope"]))

    def ssm(i):
        n, hh = b.udiv(i, p["H"]), b.urem(i, p["H"])
        lo, hi = k.ld(p["OFF"], n), k.ld(p["OFF"], b.add(n, C(1)))

        def at(e):
            return b.add(b.mul(i64of(b, e), i64of(b, p["H"])), i64of(b, hh))
        mx = k.var(F(-3.0e38))
        k.loop(lo, hi, C(1), lambda e: b.store(b.call(max_f, [b.load(mx), leaky(k.ld(p["S"], at(e)))]), mx))
        sm = k.var(F(0.0))

        def ex(e):
            v = exp(k, b.fsub(leaky(k.ld(p["S"], at(e))), b.load(mx)))
            k.st(v, p["A"], at(e))
            b.store(b.fadd(b.load(sm), v), sm)
        k.loop(lo, hi, C(1), ex)
        inv = b.fdiv(F(1.0), b.fadd(b.load(sm), F(1e-16)))
        k.loop(lo, hi, C(1), lambda e: k.st(b.fmul(k.ld(p["A"], at(e)), inv), p["A"], at(e)))
    k.grid_loop(b.mul(p["N"], p["H"]), ssm)
    k.ret()

    # seg_softmax_b: DS = A (DA - sum A DA) leaky'(S)
    k = K("seg_softmax_b", [("S", G(f32)), ("A", G(f32)), ("DA", G(f32)), ("OFF", G(i32)), ("DS", G(f32)),
                            ("N", i32), ("H", i32), ("slope", f32)])
    b, p = k.b, k.p

    def ssmb(i):
        n, hh = b.udiv(i, p["H"]), b.urem(i, p["H"])
        lo, hi = k.ld(p["OFF"], n), k.ld(p["OFF"], b.add(n, C(1)))

        def at(e):
            return b.add(b.mul(i64of(b, e), i64of(b, p["H"])), i64of(b, hh))
        t = k.var(F(0.0))
        k.loop(lo, hi, C(1), lambda e: b.store(b.call(fma_f, [k.ld(p["A"], at(e)), k.ld(p["DA"], at(e)),
                                                               b.load(t)]), t))

        def back(e):
            a = k.ld(p["A"], at(e))
            g = b.fmul(a, b.fsub(k.ld(p["DA"], at(e)), b.load(t)))
            g = b.select(b.fcmp_ordered(">", k.ld(p["S"], at(e)), F(0.0)), g, b.fmul(g, p["slope"]))
            k.st(g, p["DS"], at(e))
        k.loop(lo, hi, C(1), back)
    k.grid_loop(b.mul(p["N"], p["H"]), ssmb)
    k.ret()

    # dot_acc: ACC[0] += sum X * Y
    k = K("dot_acc", [("X", G(f32)), ("Y", G(f32)), ("n", i32), ("ACC", G(f32)), ("PART", G(f32)),
                      ("CNT", G(i32))])
    b, p = k.b, k.p
    s_ = k.var(F(0.0))
    k.grid_loop(p["n"], lambda i: b.store(b.call(fma_f, [k.ld(p["X"], i), k.ld(p["Y"], i), b.load(s_)]), s_))
    tot = k.block_sum(b.load(s_))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(tot, p["PART"], k.cta)                   # one value a block; the last block adds them in order
    k.when_last(p["CNT"], C(0), k.ncta, lambda: sum_parts(k, p["PART"], k.ncta, p["ACC"]))
    k.ret()

    # parts_sum: ACC[0] += sum of PART[0..n) in a fixed order (one block): the same result every run
    k = K("parts_sum", [("PART", G(f32)), ("n", i32), ("ACC", G(f32))])
    b, p = k.b, k.p
    s_ = k.var(F(0.0))
    k.loop(k.tid, p["n"], C(256), lambda i: b.store(b.fadd(b.load(s_), k.ld(p["PART"], i)), s_))
    tot = k.block_sum(b.load(s_))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(b.fadd(k.ld(p["ACC"], C(0)), tot), p["ACC"], C(0))
    k.ret()

    # scale_add: Z += X * (S[0] + c)   (a scale that lives on the card, e.g. 1 + eps of a GIN layer)
    k = K("scale_add", [("Z", G(f32)), ("X", G(f32)), ("S", G(f32)), ("n", i32), ("c", f32)])
    b, p = k.b, k.p
    k.grid_loop(p["n"], lambda i: k.st(b.call(fma_f, [k.ld(p["X"], i), b.fadd(k.ld(p["S"], C(0)), p["c"]),
                                                      k.ld(p["Z"], i)]), p["Z"], i))
    k.ret()

    # acc2: exact sums of gradients (double-float, TwoSum): HI + LO += G, error-free; clear=1 also sets G to 0. The
    # sum of many micro-batches (or of the copies of a stage on several cards) then does not depend on their order.
    k = K("acc2", [("HI", G(f32)), ("LO", G(f32)), ("G", G(f32)), ("n", i32), ("clear", i32)])
    b, p = k.b, k.p

    def acc2(i):
        g, h = k.ld(p["G"], i), k.ld(p["HI"], i)
        s_ = b.fadd(h, g)
        bp = b.fsub(s_, h)
        e = b.fadd(b.fsub(h, b.fsub(s_, bp)), b.fsub(g, bp))
        k.st(s_, p["HI"], i)
        k.st(b.fadd(k.ld(p["LO"], i), e), p["LO"], i)
        with b.if_then(b.icmp_unsigned("!=", p["clear"], C(0))):
            k.st(F(0.0), p["G"], i)
    k.grid_loop(p["n"], acc2)
    k.ret()

    # ew1: Y = f(X), f by kind (EW1): exp, log, sqrt (IEEE), abs, square, 1/x (IEEE), -x, 1/sqrt(x) (IEEE)
    k = K("ew1", [("X", G(f32)), ("Y", G(f32)), ("n", i32), ("kind", i32)])
    b, p = k.b, k.p

    def ew1(i):
        x = k.ld(p["X"], i)
        sq = b.call(sqrt_f, [x])
        vals = [exp_acc(b, x), log_acc(b, x), sq, b.call(fabs_f, [x]), b.fmul(x, x), b.fdiv(F(1.0), x),
                b.fsub(F(0.0), x), b.fdiv(F(1.0), sq)]
        out = vals[-1]
        for j in range(len(vals) - 2, -1, -1):
            out = b.select(b.icmp_unsigned("==", p["kind"], C(j)), vals[j], out)
        k.st(out, p["Y"], i)
    k.grid_loop(p["n"], ew1)
    k.ret()

    # ew2: Z = X op Y (EW2): x / y (IEEE), max, min, x ** y (exp(y log x), x > 0)
    k = K("ew2", [("Z", G(f32)), ("X", G(f32)), ("Y", G(f32)), ("n", i32), ("kind", i32)])
    b, p = k.b, k.p

    def ew2(i):
        x, y = k.ld(p["X"], i), k.ld(p["Y"], i)
        vals = [b.fdiv(x, y), b.call(max_f, [x, y]), b.call(min_f, [x, y]), exp_acc(b, b.fmul(y, log_acc(b, x)))]
        out = vals[-1]
        for j in range(len(vals) - 2, -1, -1):
            out = b.select(b.icmp_unsigned("==", p["kind"], C(j)), vals[j], out)
        k.st(out, p["Z"], i)
    k.grid_loop(p["n"], ew2)
    k.ret()

    # Scalars that stay on the card (a gate read on the host would make the processor wait for the card):
    # tanh_s: OUT[0] = tanh(W[0]) in float64, rounded once (math.tanh's value); axpy_s: Y += S[0] X;
    # gate_g: G[0] += float32(DG[0] (1 - TH[0]^2)) (the gradient of a tanh gate, the product in float64)
    f64 = ir.DoubleType()

    def tanh64(b, x):
        a = b.call(decl("llvm.fabs.f64", f64, [f64]), [x])
        a = b.select(b.fcmp_ordered(">", a, ir.Constant(f64, 20.0)), ir.Constant(f64, 20.0), a)
        y = b.fmul(a, ir.Constant(f64, 2.0))                      # tanh(a) = expm1(2a) / (expm1(2a) + 2)
        poly = ir.Constant(f64, 0.0)                              # expm1(y), |y| < 1: Taylor to y^21
        import math as _m
        for n_ in range(21, 0, -1):
            poly = b.fmul(b.fadd(poly, ir.Constant(f64, 1.0 / _m.factorial(n_))), y)
        kf = b.call(decl("llvm.floor.f64", f64, [f64]), [b.fadd(b.fmul(y, ir.Constant(f64, 1.4426950408889634)),
                                                                 ir.Constant(f64, 0.5))])
        r = b.fsub(b.fsub(y, b.fmul(kf, ir.Constant(f64, 0.6931471803691238))),
                   b.fmul(kf, ir.Constant(f64, 1.9082149292705877e-10)))         # y - k ln 2 (Cody-Waite)
        er = ir.Constant(f64, 0.0)                                # exp(r), |r| <= ln2 / 2: Taylor to r^16
        for n_ in range(16, 0, -1):
            er = b.fmul(b.fadd(er, ir.Constant(f64, 1.0 / _m.factorial(n_))), r)
        er = b.fadd(er, ir.Constant(f64, 1.0))
        two_k = b.bitcast(b.shl(b.add(b.fptosi(kf, i64), ir.Constant(i64, 1023)), ir.Constant(i64, 52)), f64)
        big = b.fsub(b.fmul(er, two_k), ir.Constant(f64, 1.0))
        t = b.select(b.fcmp_ordered("<", y, ir.Constant(f64, 1.0)), poly, big)
        th = b.fdiv(t, b.fadd(t, ir.Constant(f64, 2.0)))
        return b.select(b.fcmp_ordered("<", x, ir.Constant(f64, 0.0)), b.fsub(ir.Constant(f64, 0.0), th), th)

    k = K("tanh_s", [("OUT", G(f32)), ("W", G(f32))])
    b, p = k.b, k.p
    with b.if_then(b.icmp_unsigned("==", b.add(k.tid, k.cta), C(0))):
        k.st(b.fptrunc(tanh64(b, b.fpext(k.ld(p["W"], C(0)), f64)), f32), p["OUT"], C(0))
    k.ret()

    k = K("axpy_s", [("Y", G(f32)), ("X", G(f32)), ("n", i32), ("S", G(f32))])
    k.grid_loop(k.p["n"], lambda i: k.st(k.b.call(fma_f, [k.ld(k.p["S"], C(0)), k.ld(k.p["X"], i),
                                                          k.ld(k.p["Y"], i)]), k.p["Y"], i))
    k.ret()

    k = K("gate_g", [("G", G(f32)), ("DG", G(f32)), ("TH", G(f32))])
    b, p = k.b, k.p
    with b.if_then(b.icmp_unsigned("==", b.add(k.tid, k.cta), C(0))):
        th = b.fpext(k.ld(p["TH"], C(0)), f64)
        v = b.fmul(b.fpext(k.ld(p["DG"], C(0)), f64), b.fsub(ir.Constant(f64, 1.0), b.fmul(th, th)))
        k.st(b.fadd(k.ld(p["G"], C(0)), b.fptrunc(v, f32)), p["G"], C(0))
    k.ret()

    # fin2: G <- HI + LO (rounded once), HI = LO = 0
    k = K("fin2", [("HI", G(f32)), ("LO", G(f32)), ("G", G(f32)), ("n", i32)])
    b, p = k.b, k.p

    def fin2(i):
        k.st(b.fadd(k.ld(p["HI"], i), k.ld(p["LO"], i)), p["G"], i)
        k.st(F(0.0), p["HI"], i)
        k.st(F(0.0), p["LO"], i)
    k.grid_loop(p["n"], fin2)
    k.ret()

    # cols: columns off..off+w of Y (row stride ld) <- X (rows of w); back=1: X <- those columns (concatenation)
    k = K("cols", [("X", G(f32)), ("Y", G(f32)), ("rows", i32), ("w", i32), ("ld", i32), ("off", i32),
                   ("back", i32)])
    b, p = k.b, k.p

    def cols(i):
        r, c = b.udiv(i, p["w"]), b.urem(i, p["w"])
        j = b.add(b.mul(i64of(b, r), i64of(b, p["ld"])), i64of(b, b.add(p["off"], c)))
        with b.if_else(b.icmp_unsigned("!=", p["back"], C(0))) as (bk, fw):
            with bk:
                k.st(k.ld(p["Y"], j), p["X"], i)
            with fw:
                k.st(k.ld(p["X"], i), p["Y"], j)
    k.grid_loop(b.mul(p["rows"], p["w"]), cols)
    k.ret()

    # pad2: Y (rows_p x cols_p) <- X (rows x cols, dense), zeros outside: one pass, no separate fill (operands of
    # products whose sizes are not multiples of 4, made tile-friendly)
    k = K("pad2", [("X", G(f32)), ("Y", G(f32)), ("rows", i32), ("cols", i32), ("rows_p", i32), ("cols_p", i32)])
    b, p = k.b, k.p

    def pad2(i):
        r, c = b.udiv(i, p["cols_p"]), b.urem(i, p["cols_p"])
        inside = b.and_(b.icmp_unsigned("<", r, p["rows"]), b.icmp_unsigned("<", c, p["cols"]))
        with b.if_else(inside) as (yes, no):
            with yes:
                j = b.add(b.mul(i64of(b, r), i64of(b, p["cols"])), i64of(b, c))
                k.st(k.ld(p["X"], j), p["Y"], i)
            with no:
                k.st(F(0.0), p["Y"], i)
    k.grid_loop(b.mul(p["rows_p"], p["cols_p"]), pad2)
    k.ret()

    # colmul: Y[i] = X[i] * S[i % D]   (a scale per feature)
    k = K("colmul", [("X", G(f32)), ("S", G(f32)), ("Y", G(f32)), ("n", i32), ("D", i32)])
    b, p = k.b, k.p
    k.grid_loop(p["n"], lambda i: k.st(b.fmul(k.ld(p["X"], i), k.ld(p["S"], b.urem(i, p["D"]))), p["Y"], i))
    k.ret()

    # topk_gate: per row of router scores L [M, E]: the top k (larger first, lower index on a tie) -> IDX [M, k],
    # and the softmax of their scores -> GATE [M, k]. A thread a row (k and E are small).
    k = K("topk_gate", [("L", G(f32)), ("IDX", G(i32)), ("GATE", G(f32)), ("M", i32), ("E", i32), ("k", i32)])
    b, p = k.b, k.p

    def topk(r):
        base = b.mul(i64of(b, r), i64of(b, p["E"]))
        obase = b.mul(i64of(b, r), i64of(b, p["k"]))

        def pick(j):
            best, bi = k.var(F(-3.4e38)), k.var(C(0))
            found = k.var(C(0))

            def scan(e):
                v = k.ld(p["L"], b.add(base, i64of(b, e)))
                taken = k.var(C(0))
                k.loop(C(0), j, C(1), lambda jj: b.store(
                    b.or_(b.load(taken), b.zext(b.icmp_unsigned("==", k.ld(p["IDX"], b.add(obase, i64of(b, jj))), e),
                                                i32)), taken))
                better = b.and_(b.icmp_unsigned("==", b.load(taken), C(0)),
                                b.or_(b.fcmp_ordered(">", v, b.load(best)),
                                      b.icmp_unsigned("==", b.load(found), C(0))))
                with b.if_then(better):
                    b.store(v, best)
                    b.store(e, bi)
                    b.store(C(1), found)
            k.loop(C(0), p["E"], C(1), scan)
            k.st(b.load(bi), p["IDX"], b.add(obase, i64of(b, j)))
            k.st(b.load(best), p["GATE"], b.add(obase, i64of(b, j)))
        k.loop(C(0), p["k"], C(1), pick)
        mx = k.ld(p["GATE"], obase)                      # the first is the largest
        sm = k.var(F(0.0))

        def ex(j):
            v = exp(k, b.fsub(k.ld(p["GATE"], b.add(obase, i64of(b, j))), mx))
            k.st(v, p["GATE"], b.add(obase, i64of(b, j)))
            b.store(b.fadd(b.load(sm), v), sm)
        k.loop(C(0), p["k"], C(1), ex)
        inv = b.fdiv(F(1.0), b.load(sm))
        k.loop(C(0), p["k"], C(1), lambda j: k.st(b.fmul(k.ld(p["GATE"], b.add(obase, i64of(b, j))), inv), p["GATE"],
                                                  b.add(obase, i64of(b, j))))
    k.grid_loop(p["M"], topk)
    k.ret()

    # topk_gate_b: DL [M, E] = 0 except at the chosen experts: g_j (dg_j - sum g dg)
    k = K("topk_gate_b", [("GATE", G(f32)), ("DG", G(f32)), ("IDX", G(i32)), ("DL", G(f32)), ("M", i32), ("E", i32),
                          ("k", i32)])
    b, p = k.b, k.p

    def topkb(r):
        base = b.mul(i64of(b, r), i64of(b, p["E"]))
        obase = b.mul(i64of(b, r), i64of(b, p["k"]))
        k.loop(C(0), p["E"], C(1), lambda e: k.st(F(0.0), p["DL"], b.add(base, i64of(b, e))))
        t = k.var(F(0.0))
        k.loop(C(0), p["k"], C(1), lambda j: b.store(b.call(fma_f, [k.ld(p["GATE"], b.add(obase, i64of(b, j))),
                                                                     k.ld(p["DG"], b.add(obase, i64of(b, j))),
                                                                     b.load(t)]), t))

        def put(j):
            g_ = k.ld(p["GATE"], b.add(obase, i64of(b, j)))
            v = b.fmul(g_, b.fsub(k.ld(p["DG"], b.add(obase, i64of(b, j))), b.load(t)))
            e = k.ld(p["IDX"], b.add(obase, i64of(b, j)))
            k.st(v, p["DL"], b.add(base, i64of(b, e)))
        k.loop(C(0), p["k"], C(1), put)
    k.grid_loop(p["M"], topkb)
    k.ret()

    # ------------------------------------------------------------------------- images (N, C, H, W)
    # im2col: COL[(n, oh, ow), (c, i, j)] = X[n, c, oh s + i - p, ow s + j - p] (0 outside)
    shp = [("N", i32), ("C", i32), ("H", i32), ("W", i32), ("kh", i32), ("kw", i32), ("s", i32), ("p", i32),
           ("OH", i32), ("OW", i32)]
    k = K("im2col", [("X", G(f32)), ("COL", G(f32))] + shp)
    b, p = k.b, k.p
    CK = b.mul(p["C"], b.mul(p["kh"], p["kw"]))

    def i2c(t):
        r, c = b.udiv(t, CK), b.urem(t, CK)
        ow = b.urem(r, p["OW"])
        oh = b.urem(b.udiv(r, p["OW"]), p["OH"])
        n = b.udiv(r, b.mul(p["OW"], p["OH"]))
        j = b.urem(c, p["kw"])
        i = b.urem(b.udiv(c, p["kw"]), p["kh"])
        ci = b.udiv(c, b.mul(p["kw"], p["kh"]))
        ih = b.sub(b.add(b.mul(oh, p["s"]), i), p["p"])          # unsigned: below zero wraps above H
        iw = b.sub(b.add(b.mul(ow, p["s"]), j), p["p"])
        inside = b.and_(b.icmp_unsigned("<", ih, p["H"]), b.icmp_unsigned("<", iw, p["W"]))
        src = b.add(b.mul(i64of(b, b.add(b.mul(n, p["C"]), ci)), i64of(b, b.mul(p["H"], p["W"]))),
                    i64of(b, b.add(b.mul(b.select(inside, ih, C(0)), p["W"]), b.select(inside, iw, C(0)))))
        k.st(b.select(inside, k.ld(p["X"], src), F(0.0)), p["COL"], t)
    k.grid_loop(b.mul(b.mul(p["N"], b.mul(p["OH"], p["OW"])), CK), i2c)
    k.ret()

    # col2im: DX[n, c, h, w] = sum over the (oh, ow, i, j) that read it of DCOL (a thread an input value: no atomics)
    k = K("col2im", [("DCOL", G(f32)), ("DX", G(f32))] + shp)
    b, p = k.b, k.p
    CK = b.mul(p["C"], b.mul(p["kh"], p["kw"]))

    def c2i(t):
        w = b.urem(t, p["W"])
        h = b.urem(b.udiv(t, p["W"]), p["H"])
        nc = b.udiv(t, b.mul(p["W"], p["H"]))
        ci, n = b.urem(nc, p["C"]), b.udiv(nc, p["C"])
        acc = k.var(F(0.0))

        def over_i(i):
            def over_j(j):
                a, c_ = b.sub(b.add(h, p["p"]), i), b.sub(b.add(w, p["p"]), j)
                ok = b.and_(b.icmp_signed(">=", a, C(0)), b.icmp_signed(">=", c_, C(0)))
                a0, c0 = b.select(ok, a, C(0)), b.select(ok, c_, C(0))
                ok = b.and_(ok, b.and_(b.icmp_unsigned("==", b.urem(a0, p["s"]), C(0)),
                                       b.icmp_unsigned("==", b.urem(c0, p["s"]), C(0))))
                oh, ow = b.udiv(a0, p["s"]), b.udiv(c0, p["s"])
                ok = b.and_(ok, b.and_(b.icmp_unsigned("<", oh, p["OH"]), b.icmp_unsigned("<", ow, p["OW"])))
                r = b.add(b.mul(b.add(b.mul(n, p["OH"]), b.select(ok, oh, C(0))), p["OW"]), b.select(ok, ow, C(0)))
                c = b.add(b.mul(b.add(b.mul(ci, p["kh"]), i), p["kw"]), j)
                v = k.ld(p["DCOL"], b.add(b.mul(i64of(b, r), i64of(b, CK)), i64of(b, c)))
                b.store(b.fadd(b.load(acc), b.select(ok, v, F(0.0))), acc)
            k.loop(C(0), p["kw"], C(1), over_j)
        k.loop(C(0), p["kh"], C(1), over_i)
        k.st(b.load(acc), p["DX"], t)
    k.grid_loop(b.mul(b.mul(p["N"], p["C"]), b.mul(p["H"], p["W"])), c2i)
    k.ret()

    # perm_nc: back=0: Y[n, c, s] = Z[(n, s), c] (+ bias[c]); back=1: Z[(n, s), c] = Y[n, c, s]
    k = K("perm_nc", [("Z", G(f32)), ("Y", G(f32)), ("BIAS", G(f32)), ("N", i32), ("Cc", i32), ("S", i32),
                      ("back", i32), ("hasb", i32)])
    b, p = k.b, k.p

    def perm(t):
        sp = b.urem(t, p["S"])
        c = b.urem(b.udiv(t, p["S"]), p["Cc"])
        n = b.udiv(t, b.mul(p["S"], p["Cc"]))
        z = b.add(b.mul(i64of(b, b.add(b.mul(n, p["S"]), sp)), i64of(b, p["Cc"])), i64of(b, c))
        with b.if_else(b.icmp_unsigned("!=", p["back"], C(0))) as (bk, fw):
            with bk:
                k.st(k.ld(p["Y"], t), p["Z"], z)
            with fw:
                hb = b.icmp_unsigned("!=", p["hasb"], C(0))
                bias = b.select(hb, k.ld(p["BIAS"], b.select(hb, c, C(0))), F(0.0))
                k.st(b.fadd(k.ld(p["Z"], z), bias), p["Y"], t)
    k.grid_loop(b.mul(b.mul(p["N"], p["Cc"]), p["S"]), perm)
    k.ret()

    # pool_f: Y[n, c, a, b] = max (mode 0) or mean (mode 1) of the k x k window; pool_b: the gradient back, the max
    # shared equally between ties (as the NumPy backend does); values past the last whole window get nothing
    pshp = [("NC", i32), ("H", i32), ("W", i32), ("kk", i32), ("mode", i32)]
    k = K("pool_f", [("X", G(f32)), ("Y", G(f32))] + pshp)
    b, p = k.b, k.p
    H2, W2 = b.udiv(p["H"], p["kk"]), b.udiv(p["W"], p["kk"])

    def poolf(t):
        bb, a = b.urem(t, W2), b.urem(b.udiv(t, W2), H2)
        nc = b.udiv(t, b.mul(W2, H2))
        base = b.mul(i64of(b, nc), i64of(b, b.mul(p["H"], p["W"])))
        mx, sm = k.var(F(-3.4e38)), k.var(F(0.0))

        def over(i):
            def over_j(j):
                v = k.ld(p["X"], b.add(base, i64of(b, b.add(b.mul(b.add(b.mul(a, p["kk"]), i), p["W"]),
                                                             b.add(b.mul(bb, p["kk"]), j)))))
                b.store(b.call(max_f, [b.load(mx), v]), mx)
                b.store(b.fadd(b.load(sm), v), sm)
            k.loop(C(0), p["kk"], C(1), over_j)
        k.loop(C(0), p["kk"], C(1), over)
        avg = b.fdiv(b.load(sm), b.uitofp(b.mul(p["kk"], p["kk"]), f32))
        k.st(b.select(b.icmp_unsigned("==", p["mode"], C(0)), b.load(mx), avg), p["Y"], t)
    k.grid_loop(b.mul(p["NC"], b.mul(H2, W2)), poolf)
    k.ret()

    k = K("pool_b", [("X", G(f32)), ("Y", G(f32)), ("DY", G(f32)), ("DX", G(f32))] + pshp)
    b, p = k.b, k.p
    H2, W2 = b.udiv(p["H"], p["kk"]), b.udiv(p["W"], p["kk"])

    def poolb(t):
        w, h = b.urem(t, p["W"]), b.urem(b.udiv(t, p["W"]), p["H"])
        nc = b.udiv(t, b.mul(p["W"], p["H"]))
        a, bb = b.udiv(h, p["kk"]), b.udiv(w, p["kk"])
        inside = b.and_(b.icmp_unsigned("<", a, H2), b.icmp_unsigned("<", bb, W2))
        yi = b.add(b.mul(i64of(b, nc), i64of(b, b.mul(H2, W2))),
                   i64of(b, b.add(b.mul(b.select(inside, a, C(0)), W2), b.select(inside, bb, C(0)))))
        dy, ym = k.ld(p["DY"], yi), k.ld(p["Y"], yi)
        base = b.mul(i64of(b, nc), i64of(b, b.mul(p["H"], p["W"])))
        ties = k.var(F(0.0))

        def over(i):
            def over_j(j):
                v = k.ld(p["X"], b.add(base, i64of(b, b.add(b.mul(b.add(b.mul(a, p["kk"]), i), p["W"]),
                                                             b.add(b.mul(bb, p["kk"]), j)))))
                b.store(b.fadd(b.load(ties), b.select(b.fcmp_ordered("==", v, ym), F(1.0), F(0.0))), ties)
            k.loop(C(0), p["kk"], C(1), over_j)
        k.loop(C(0), p["kk"], C(1), over)
        mine = b.select(b.fcmp_ordered("==", k.ld(p["X"], t), ym), b.fdiv(dy, b.load(ties)), F(0.0))
        avg = b.fdiv(dy, b.uitofp(b.mul(p["kk"], p["kk"]), f32))
        g_ = b.select(b.icmp_unsigned("==", p["mode"], C(0)), mine, avg)
        k.st(b.select(inside, g_, F(0.0)), p["DX"], t)
    k.grid_loop(b.mul(p["NC"], b.mul(p["H"], p["W"])), poolb)
    k.ret()

    # ------------------------------------------------------------------------- recurrent cells, one time step
    # [B, T, *] buffers; a thread a (b, j); t the step. LSTM gates i f g o from Z (pre-activations, x W_x + h W_h + b)
    rshp = [("B", i32), ("H", i32), ("T", i32), ("t", i32)]

    def rix(k, b_, w, j, p, step):
        """index of [b, step, j] in a [B, T, w] buffer"""
        bb = k.b
        return bb.add(bb.mul(i64of(bb, bb.add(bb.mul(b_, p["T"]), step)), i64of(bb, w)), i64of(bb, j))

    def prev_or_zero(k, buf, b_, j, p):
        bb = k.b
        has = bb.icmp_unsigned("!=", p["t"], C(0))
        tp = bb.select(has, bb.sub(p["t"], C(1)), C(0))
        return bb.select(has, k.ld(buf, rix(k, b_, p["H"], j, p, tp)), F(0.0))

    k = K("lstm_f", [("Z", G(f32)), ("CS", G(f32)), ("HS", G(f32))] + rshp)
    b, p = k.b, k.p

    def lstmf(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        H4 = b.mul(p["H"], C(4))
        zi = k.ld(p["Z"], rix(k, bb, H4, j, p, p["t"]))
        zf = k.ld(p["Z"], rix(k, bb, H4, b.add(j, p["H"]), p, p["t"]))
        zg = k.ld(p["Z"], rix(k, bb, H4, b.add(j, b.mul(p["H"], C(2))), p, p["t"]))
        zo = k.ld(p["Z"], rix(k, bb, H4, b.add(j, b.mul(p["H"], C(3))), p, p["t"]))
        c = b.fadd(b.fmul(sigmoid(k, zf), prev_or_zero(k, p["CS"], bb, j, p)), b.fmul(sigmoid(k, zi), tanh(k, zg)))
        k.st(c, p["CS"], rix(k, bb, p["H"], j, p, p["t"]))
        k.st(b.fmul(sigmoid(k, zo), tanh(k, c)), p["HS"], rix(k, bb, p["H"], j, p, p["t"]))
    k.grid_loop(b.mul(p["B"], p["H"]), lstmf)
    k.ret()

    # lstm_b: DZ[b, t] from dh = DY[b, t] + DHN[b] and dc = ... + DCN[b]; DCN[b] <- the cell gradient for t - 1
    k = K("lstm_b", [("Z", G(f32)), ("CS", G(f32)), ("DY", G(f32)), ("DHN", G(f32)), ("DCN", G(f32)),
                     ("DZ", G(f32))] + rshp)
    b, p = k.b, k.p

    def lstmb(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        H4 = b.mul(p["H"], C(4))
        idx = [rix(k, bb, H4, b.add(j, b.mul(p["H"], C(q))), p, p["t"]) for q in range(4)]
        ig, fg, og = sigmoid(k, k.ld(p["Z"], idx[0])), sigmoid(k, k.ld(p["Z"], idx[1])), \
            sigmoid(k, k.ld(p["Z"], idx[3]))
        gg = tanh(k, k.ld(p["Z"], idx[2]))
        c = k.ld(p["CS"], rix(k, bb, p["H"], j, p, p["t"]))
        cp = prev_or_zero(k, p["CS"], bb, j, p)
        dh = b.fadd(k.ld(p["DY"], rix(k, bb, p["H"], j, p, p["t"])), k.ld(p["DHN"], i))
        tc = tanh(k, c)
        dc = b.fadd(b.fmul(b.fmul(dh, og), b.fsub(F(1.0), b.fmul(tc, tc))), k.ld(p["DCN"], i))
        one = F(1.0)
        k.st(b.fmul(b.fmul(dc, gg), b.fmul(ig, b.fsub(one, ig))), p["DZ"], idx[0])
        k.st(b.fmul(b.fmul(dc, cp), b.fmul(fg, b.fsub(one, fg))), p["DZ"], idx[1])
        k.st(b.fmul(b.fmul(dc, ig), b.fsub(one, b.fmul(gg, gg))), p["DZ"], idx[2])
        k.st(b.fmul(b.fmul(dh, tc), b.fmul(og, b.fsub(one, og))), p["DZ"], idx[3])
        k.st(b.fmul(dc, fg), p["DCN"], i)
    k.grid_loop(b.mul(p["B"], p["H"]), lstmb)
    k.ret()

    # GRU: ZR [B, T, 2H] (pre-activations in, z and r out), RH = r h_prev; CX [B, T, H] (pre-activation in, the
    # candidate out), HS = (1 - z) h_prev + z cand
    k = K("gru_a", [("ZR", G(f32)), ("HS", G(f32)), ("RH", G(f32))] + rshp)
    b, p = k.b, k.p

    def grua(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        H2 = b.mul(p["H"], C(2))
        iz, ir = rix(k, bb, H2, j, p, p["t"]), rix(k, bb, H2, b.add(j, p["H"]), p, p["t"])
        z, r = sigmoid(k, k.ld(p["ZR"], iz)), sigmoid(k, k.ld(p["ZR"], ir))
        k.st(z, p["ZR"], iz)
        k.st(r, p["ZR"], ir)
        k.st(b.fmul(r, prev_or_zero(k, p["HS"], bb, j, p)), p["RH"], rix(k, bb, p["H"], j, p, p["t"]))
    k.grid_loop(b.mul(p["B"], p["H"]), grua)
    k.ret()

    k = K("gru_b", [("CX", G(f32)), ("ZR", G(f32)), ("HS", G(f32))] + rshp)
    b, p = k.b, k.p

    def grub(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        ic = rix(k, bb, p["H"], j, p, p["t"])
        cand = tanh(k, k.ld(p["CX"], ic))
        k.st(cand, p["CX"], ic)
        z = k.ld(p["ZR"], rix(k, bb, b.mul(p["H"], C(2)), j, p, p["t"]))
        hp = prev_or_zero(k, p["HS"], bb, j, p)
        k.st(b.fadd(b.fmul(b.fsub(F(1.0), z), hp), b.fmul(z, cand)), p["HS"], ic)
    k.grid_loop(b.mul(p["B"], p["H"]), grub)
    k.ret()

    # gru_b1: dh = DY + DHN; DCX = dh z (1 - cand^2); DZR[z] = dh (cand - h_prev) z (1 - z); DHP = dh (1 - z)
    k = K("gru_b1", [("CX", G(f32)), ("ZR", G(f32)), ("HS", G(f32)), ("DY", G(f32)), ("DHN", G(f32)),
                     ("DCX", G(f32)), ("DZR", G(f32)), ("DHP", G(f32))] + rshp)
    b, p = k.b, k.p

    def grub1(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        ic = rix(k, bb, p["H"], j, p, p["t"])
        iz = rix(k, bb, b.mul(p["H"], C(2)), j, p, p["t"])
        cand, z = k.ld(p["CX"], ic), k.ld(p["ZR"], iz)
        hp = prev_or_zero(k, p["HS"], bb, j, p)
        dh = b.fadd(k.ld(p["DY"], ic), k.ld(p["DHN"], i))
        k.st(b.fmul(b.fmul(dh, z), b.fsub(F(1.0), b.fmul(cand, cand))), p["DCX"], ic)
        k.st(b.fmul(b.fmul(dh, b.fsub(cand, hp)), b.fmul(z, b.fsub(F(1.0), z))), p["DZR"], iz)
        k.st(b.fmul(dh, b.fsub(F(1.0), z)), p["DHP"], i)
    k.grid_loop(b.mul(p["B"], p["H"]), grub1)
    k.ret()

    # gru_b2: DRH = dcx W_h,h^T given; DZR[r] = DRH h_prev r (1 - r); DHP += DRH r
    k = K("gru_b2", [("DRH", G(f32)), ("ZR", G(f32)), ("HS", G(f32)), ("DZR", G(f32)), ("DHP", G(f32))] + rshp)
    b, p = k.b, k.p

    def grub2(i):
        bb, j = b.udiv(i, p["H"]), b.urem(i, p["H"])
        ir = rix(k, bb, b.mul(p["H"], C(2)), b.add(j, p["H"]), p, p["t"])
        r, d = k.ld(p["ZR"], ir), k.ld(p["DRH"], i)
        hp = prev_or_zero(k, p["HS"], bb, j, p)
        k.st(b.fmul(b.fmul(d, hp), b.fmul(r, b.fsub(F(1.0), r))), p["DZR"], ir)
        k.st(b.fadd(k.ld(p["DHP"], i), b.fmul(d, r)), p["DHP"], i)
    k.grid_loop(b.mul(p["B"], p["H"]), grub2)
    k.ret()

    # ------------------------------------------------------------------------- physics-informed networks
    # Taylor mode: a stack [(1 + 2 d) N, F] holds the values (block 0), their derivatives along each input (blocks
    # 1..d) and their second derivatives along each input (blocks d+1..2d). An activation f maps value z, first j,
    # second h to f(z), f'(z) j, f''(z) j^2 + f'(z) h. kind: 0 tanh, 1 sin, 2 sigmoid, 3 silu, 4 softplus,
    # 5 identity, 6 cos
    def derivs(k, z, kind):
        """the function and its first three derivatives at z"""
        b = k.b
        red = b.fsub(z, b.fmul(F(6.283185307179586), b.call(rint_f, [b.fmul(z, F(0.15915494309189535))])))
        sn, cs = b.call(sin_f, [red]), b.call(cos_f, [red])
        t = tanh(k, z)
        t1 = b.fsub(F(1.0), b.fmul(t, t))
        s = sigmoid(k, z)
        s1 = b.fmul(s, b.fsub(F(1.0), s))
        s2 = b.fmul(s1, b.fsub(F(1.0), b.fmul(F(2.0), s)))
        s3 = b.fsub(b.fmul(s2, b.fsub(F(1.0), b.fmul(F(2.0), s))), b.fmul(F(2.0), b.fmul(s1, s1)))
        az = b.call(fabs_f, [z])
        sp = b.fadd(b.call(max_f, [z, F(0.0)]),
                    b.fdiv(b.call(lg2_f, [b.fadd(F(1.0), exp(k, b.fsub(F(0.0), az)))]), F(LOG2E)))

        def neg(v):
            return b.fsub(F(0.0), v)
        table = [
            (t, t1, b.fmul(F(-2.0), b.fmul(t, t1)), b.fmul(t1, b.fsub(b.fmul(F(6.0), b.fmul(t, t)), F(2.0)))),
            (sn, cs, neg(sn), neg(cs)),
            (s, s1, s2, s3),
            (b.fmul(z, s), b.fadd(s, b.fmul(z, s1)), b.fadd(b.fmul(F(2.0), s1), b.fmul(z, s2)),
             b.fadd(b.fmul(F(3.0), s2), b.fmul(z, s3))),
            (sp, s, s1, s2),
            (z, F(1.0), F(0.0), F(0.0)),
            (cs, neg(sn), neg(cs), sn),
        ]
        out = list(table[-1])
        for idx in range(len(table) - 2, -1, -1):
            is_k = b.icmp_unsigned("==", kind, C(idx))
            out = [b.select(is_k, table[idx][q], out[q]) for q in range(4)]
        return out

    tshape = [("N", i32), ("F", i32), ("d", i32), ("kind", i32)]
    k = K("taylor_f", [("Z", G(f32)), ("Y", G(f32))] + tshape)
    b, p = k.b, k.p
    NF = b.mul(p["N"], p["F"])

    def tf(i):
        z = k.ld(p["Z"], i)
        f0, f1, f2, _ = derivs(k, z, p["kind"])
        k.st(f0, p["Y"], i)

        def each(q):
            ij = b.add(i64of(b, i), b.mul(i64of(b, b.add(q, C(1))), i64of(b, NF)))
            ih = b.add(i64of(b, i), b.mul(i64of(b, b.add(b.add(q, C(1)), p["d"])), i64of(b, NF)))
            j, h = k.ld(p["Z"], ij), k.ld(p["Z"], ih)
            k.st(b.fmul(f1, j), p["Y"], ij)
            k.st(b.fadd(b.fmul(f2, b.fmul(j, j)), b.fmul(f1, h)), p["Y"], ih)
        k.loop(C(0), p["d"], C(1), each)
    k.grid_loop(NF, tf)
    k.ret()

    k = K("taylor_b", [("Z", G(f32)), ("DY", G(f32)), ("DZ", G(f32))] + tshape)
    b, p = k.b, k.p
    NF = b.mul(p["N"], p["F"])

    def tb(i):
        z = k.ld(p["Z"], i)
        _, f1, f2, f3 = derivs(k, z, p["kind"])
        dz = k.var(b.fmul(k.ld(p["DY"], i), f1))

        def each(q):
            ij = b.add(i64of(b, i), b.mul(i64of(b, b.add(q, C(1))), i64of(b, NF)))
            ih = b.add(i64of(b, i), b.mul(i64of(b, b.add(b.add(q, C(1)), p["d"])), i64of(b, NF)))
            j, h = k.ld(p["Z"], ij), k.ld(p["Z"], ih)
            dj_, dh_ = k.ld(p["DY"], ij), k.ld(p["DY"], ih)
            acc = b.fadd(b.fmul(dj_, b.fmul(f2, j)), b.fmul(dh_, b.fadd(b.fmul(f3, b.fmul(j, j)), b.fmul(f2, h))))
            b.store(b.fadd(b.load(dz), acc), dz)
            k.st(b.fadd(b.fmul(dj_, f1), b.fmul(dh_, b.fmul(F(2.0), b.fmul(f2, j)))), p["DZ"], ij)
            k.st(b.fmul(dh_, f1), p["DZ"], ih)
        k.loop(C(0), p["d"], C(1), each)
        k.st(b.load(dz), p["DZ"], i)
    k.grid_loop(NF, tb)
    k.ret()

    # rowbias: Y[r, c] += B[c] for the first `rows` rows (the values block of a Taylor stack)
    k = K("rowbias", [("Y", G(f32)), ("B", G(f32)), ("rows", i32), ("F", i32)])
    b, p = k.b, k.p
    k.grid_loop(b.mul(p["rows"], p["F"]), lambda i: k.st(b.fadd(k.ld(p["Y"], i), k.ld(p["B"], b.urem(i, p["F"]))),
                                                         p["Y"], i))
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
    # the loss values go to PART (one a row / a block); parts_sum adds them in order: the same loss every run
    k = K("ce", [("L", G(f32)), ("LAB", G(i32)), ("DL", G(f32)), ("LOSS", G(f32)), ("M", i32), ("V", i32),
                 ("wg", f32), ("wl", f32), ("PART", G(f32)), ("CNT", G(i32))])
    b, p = k.b, k.p
    base = b.mul(i64of(b, k.cta), i64of(b, p["V"]))
    lab = k.ld(p["LAB"], k.cta)
    valid = b.icmp_signed(">=", lab, C(0))
    mx = k.var(F(-3.0e38))
    k.loop(k.tid, p["V"], C(256), lambda i: b.store(b.call(max_f, [b.load(mx), k.ld(p["L"], b.add(base, i64of(b, i)))]), mx))
    M_ = k.block_sum(b.load(mx), mx=True)
    sm = k.var(F(0.0))
    k.loop(k.tid, p["V"], C(256), lambda i: b.store(b.fadd(b.load(sm), exp_acc(b, b.fsub(
        k.ld(p["L"], b.add(base, i64of(b, i))), M_))), sm))
    Z = k.block_sum(b.load(sm))
    inv = b.fdiv(F(1.0), Z)

    def grad(i):
        idx = b.add(base, i64of(b, i))
        pr = b.fmul(exp_acc(b, b.fsub(k.ld(p["L"], idx), M_)), inv)
        pr = b.fsub(pr, b.select(b.icmp_signed("==", i, lab), F(1.0), F(0.0)))
        k.st(b.select(valid, b.fmul(pr, p["wg"]), F(0.0)), p["DL"], idx)
    k.loop(k.tid, p["V"], C(256), grad)
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        lse = b.fadd(M_, log_acc(b, Z))
        safe = b.select(valid, lab, C(0))
        lv = b.fmul(b.fsub(lse, k.ld(p["L"], b.add(base, i64of(b, safe)))), p["wl"])
        k.st(b.select(valid, lv, F(0.0)), p["PART"], k.cta)
    k.when_last(p["CNT"], C(0), k.ncta, lambda: sum_parts(k, p["PART"], k.ncta, p["LOSS"]))
    k.ret()

    # loss_ew: element-wise losses with an optional weight per element (a mask). kind 0: mae |y - t|;
    # kind 1: bce -(t log(s + 1e-12) + (1 - t) log(1 - s + 1e-12)), s = sigmoid(y) (the formula of the NumPy backend)
    k = K("loss_ew", [("Y", G(f32)), ("T", G(f32)), ("W", G(f32)), ("DY", G(f32)), ("LOSS", G(f32)), ("n", i32),
                      ("wg", f32), ("wl", f32), ("kind", i32), ("hasw", i32), ("PART", G(f32)),
                      ("CNT", G(i32))])
    b, p = k.b, k.p
    s = k.var(F(0.0))

    def lew(i):
        y, t = k.ld(p["Y"], i), k.ld(p["T"], i)
        w = b.select(b.icmp_unsigned("!=", p["hasw"], C(0)), k.ld(p["W"], b.select(
            b.icmp_unsigned("!=", p["hasw"], C(0)), i, C(0))), F(1.0))
        d = b.fsub(y, t)
        l_mae = b.call(fabs_f, [d])
        g_mae = b.select(b.fcmp_ordered(">", d, F(0.0)), F(1.0), b.select(b.fcmp_ordered("<", d, F(0.0)), F(-1.0),
                                                                            F(0.0)))
        sg = sigmoid(k, y)
        a, c = b.fadd(sg, F(1e-12)), b.fadd(b.fsub(F(1.0), sg), F(1e-12))
        ln = lambda v: b.fdiv(b.call(lg2_f, [v]), F(LOG2E))          # noqa: E731
        l_bce = b.fsub(F(0.0), b.fadd(b.fmul(t, ln(a)), b.fmul(b.fsub(F(1.0), t), ln(c))))
        ds = b.fmul(sg, b.fsub(F(1.0), sg))
        g_bce = b.fsub(F(0.0), b.fsub(b.fdiv(b.fmul(t, ds), a), b.fdiv(b.fmul(b.fsub(F(1.0), t), ds), c)))
        bce = b.icmp_unsigned("==", p["kind"], C(1))
        lv, gv = b.select(bce, l_bce, l_mae), b.select(bce, g_bce, g_mae)
        k.st(b.fmul(b.fmul(gv, w), p["wg"]), p["DY"], i)
        b.store(b.call(fma_f, [lv, w, b.load(s)]), s)
    k.grid_loop(p["n"], lew)
    tot = k.block_sum(b.load(s))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(b.fmul(tot, p["wl"]), p["PART"], k.cta)
    k.when_last(p["CNT"], C(0), k.ncta, lambda: sum_parts(k, p["PART"], k.ncta, p["LOSS"]))
    k.ret()

    k = K("mse", [("Y", G(f32)), ("T", G(f32)), ("DY", G(f32)), ("LOSS", G(f32)), ("n", i32), ("wg", f32),
                  ("wl", f32), ("PART", G(f32)), ("CNT", G(i32))])
    b, p = k.b, k.p
    s = k.var(F(0.0))

    def mse(i):
        d = b.fsub(k.ld(p["Y"], i), k.ld(p["T"], i))
        k.st(b.fmul(F(2.0), b.fmul(d, p["wg"])), p["DY"], i)
        b.store(b.call(fma_f, [d, d, b.load(s)]), s)
    k.grid_loop(p["n"], mse)
    tot = k.block_sum(b.load(s))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(b.fmul(tot, p["wl"]), p["PART"], k.cta)
    k.when_last(p["CNT"], C(0), k.ncta, lambda: sum_parts(k, p["PART"], k.ncta, p["LOSS"]))
    k.ret()

    k = K("sumsq", [("X", G(f32)), ("n", i32), ("ACC", G(f32)), ("PART", G(f32)), ("CNT", G(i32))])
    b, p = k.b, k.p
    s = k.var(F(0.0))
    k.grid_loop(p["n"], lambda i: b.store(b.call(fma_f, [k.ld(p["X"], i), k.ld(p["X"], i), b.load(s)]), s))
    tot = k.block_sum(b.load(s))
    with b.if_then(b.icmp_unsigned("==", k.tid, C(0))):
        k.st(tot, p["PART"], k.cta)
    k.when_last(p["CNT"], C(0), k.ncta, lambda: sum_parts(k, p["PART"], k.ncta, p["ACC"]))
    k.ret()

    # ------------------------------------------------------------------------- optimizers, over all parameters at once
    def clip_scale(k, p):
        b = k.b
        nrm = b.call(sqrt_f, [b.fadd(k.ld(p["SQ"], C(0)), F(1e-30))])
        return b.select(b.fcmp_ordered(">", p["clip"], F(0.0)),
                        b.call(min_f, [F(1.0), b.fdiv(p["clip"], nrm)]), F(1.0))

    def finite_sq(k, p):
        """False when the step's sum of squares of the gradient is inf or NaN: the step is then skipped."""
        b = k.b
        return b.fcmp_ordered("<", b.call(fabs_f, [k.ld(p["SQ"], C(0))]), F(float("inf")))

    def kahan(k, p, w_old, delta, w_new, i, ok=None):
        """With COMP: w_old + delta, the rounding error of every earlier update carried in COMP (compensated
        summation), so the weights keep changes far below float32's step at their size (small learning rates, long
        training); without: w_new as computed."""
        b = k.b
        hc = b.icmp_unsigned("!=", p["hasc"], C(0))
        ci = b.select(hc, i, C(0))
        c0 = b.select(hc, k.ld(p["COMP"], ci), F(0.0))
        y = b.fsub(delta, c0)
        t = b.fadd(w_old, y)
        with b.if_then(hc if ok is None else b.and_(hc, ok)):
            k.st(b.fsub(b.fsub(t, w_old), y), p["COMP"], i)
        out = b.select(hc, t, w_new)
        return out if ok is None else b.select(ok, out, w_old)

    k = K("adam", [("P", G(f32)), ("G", G(f32)), ("M", G(f32)), ("V", G(f32)), ("SQ", G(f32)), ("n", i32),
                   ("n_decay", i32), ("lr", f32), ("b1", f32), ("b2", f32), ("eps", f32), ("c1", f32), ("c2", f32),
                   ("wd", f32), ("clip", f32), ("COMP", G(f32)), ("hasc", i32)])
    b, p = k.b, k.p
    sc = clip_scale(k, p)
    ok = finite_sq(k, p)

    def adam(i):
        # every load first, every store last: a store between them (to arrays the compiler must assume may overlap)
        # would hold the later loads back, and this kernel is all memory traffic
        g0, w, m0, v0 = k.ld(p["G"], i), k.ld(p["P"], i), k.ld(p["M"], i), k.ld(p["V"], i)
        w_old = w
        gr = b.fmul(g0, sc)
        w = b.select(b.icmp_unsigned("<", i, p["n_decay"]), b.fmul(w, b.fsub(F(1.0), b.fmul(p["lr"], p["wd"]))), w)
        m_ = b.fadd(b.fmul(p["b1"], m0), b.fmul(b.fsub(F(1.0), p["b1"]), gr))
        v_ = b.fadd(b.fmul(p["b2"], v0), b.fmul(b.fsub(F(1.0), p["b2"]), b.fmul(gr, gr)))
        upd = b.fdiv(b.fmul(p["lr"], b.fdiv(m_, p["c1"])),
                     b.fadd(b.call(sqrt_f, [b.fdiv(v_, p["c2"])]), p["eps"]))
        k.st(b.select(ok, m_, m0), p["M"], i)
        k.st(b.select(ok, v_, v0), p["V"], i)
        dec = b.select(b.icmp_unsigned("<", i, p["n_decay"]), b.fmul(w_old, b.fmul(p["lr"], p["wd"])), F(0.0))
        k.st(kahan(k, p, w_old, b.fsub(F(0.0), b.fadd(dec, upd)), b.fsub(w, upd), i, ok), p["P"], i)
        k.st(F(0.0), p["G"], i)                       # the gradient is used up: cleared for the next step here
    k.grid_loop(p["n"], adam)
    k.ret()

    k = K("sgd", [("P", G(f32)), ("G", G(f32)), ("VEL", G(f32)), ("SQ", G(f32)), ("n", i32), ("lr", f32),
                  ("mom", f32), ("clip", f32), ("COMP", G(f32)), ("hasc", i32)])
    b, p = k.b, k.p
    sc = clip_scale(k, p)
    ok = finite_sq(k, p)

    def sgd(i):
        g0, vel, w = k.ld(p["G"], i), k.ld(p["VEL"], i), k.ld(p["P"], i)
        v_ = b.call(fma_f, [p["mom"], vel, b.fmul(g0, sc)])
        k.st(b.select(ok, v_, vel), p["VEL"], i)
        step = b.fmul(p["lr"], v_)
        k.st(kahan(k, p, w, b.fsub(F(0.0), step), b.fsub(w, step), i, ok), p["P"], i)
        k.st(F(0.0), p["G"], i)
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


def _act_ir(b, x, kind, F, C, ex2_f):
    """f(x) and f'(x) of the activation numbered `kind` (ACTS), selected at run time."""
    def exp(v):
        return b.call(ex2_f, [b.fmul(v, F(LOG2E))])

    def tanh(v):
        return b.fsub(F(1.0), b.fdiv(F(2.0), b.fadd(exp(b.fmul(F(2.0), v)), F(1.0))))
    pos = b.fcmp_ordered(">", x, F(0.0))
    relu, drelu = b.select(pos, x, F(0.0)), b.select(pos, F(1.0), F(0.0))
    c = 0.7978845608028654
    t = tanh(b.fmul(F(c), b.fadd(x, b.fmul(F(0.044715), b.fmul(x, b.fmul(x, x))))))
    gelu = b.fmul(b.fmul(F(0.5), x), b.fadd(F(1.0), t))
    dgelu = b.fadd(b.fmul(F(0.5), b.fadd(F(1.0), t)),
                   b.fmul(b.fmul(b.fmul(F(0.5), x), b.fsub(F(1.0), b.fmul(t, t))),
                          b.fmul(F(c), b.fadd(F(1.0), b.fmul(F(3 * 0.044715), b.fmul(x, x))))))
    sg = b.fdiv(F(1.0), b.fadd(F(1.0), exp(b.fsub(F(0.0), x))))
    th = tanh(x)
    vals = [(relu, drelu), (gelu, dgelu), (b.fmul(x, sg), b.fadd(sg, b.fmul(x, b.fmul(sg, b.fsub(F(1.0), sg))))),
            (th, b.fsub(F(1.0), b.fmul(th, th))), (sg, b.fmul(sg, b.fsub(F(1.0), sg))), (x, F(1.0)),
            (b.select(pos, x, b.fmul(F(0.01), x)), b.select(pos, F(1.0), F(0.01)))]
    f, d = vals[-1]
    for j in range(len(vals) - 2, -1, -1):
        sel = b.icmp_unsigned("==", kind, C(j))
        f, d = b.select(sel, vals[j][0], f), b.select(sel, vals[j][1], d)
    return f, d


def _act_one(b, x, j, F, ex2_f, deriv):
    """f(x) (or f'(x)) of the one activation numbered j (approximate reciprocals: the products round to bfloat16
    anyway)."""
    import llvmlite.ir as ir
    m = b.module
    try:
        rcp_f = m.get_global("llvm.nvvm.rcp.approx.ftz.f")
    except KeyError:
        rcp_f = ir.Function(m, ir.FunctionType(ir.FloatType(), [ir.FloatType()]), name="llvm.nvvm.rcp.approx.ftz.f")

    def exp(v):
        return b.call(ex2_f, [b.fmul(v, F(LOG2E))])

    def tanh(v):
        return b.fsub(F(1.0), b.fmul(F(2.0), b.call(rcp_f, [b.fadd(exp(b.fmul(F(2.0), v)), F(1.0))])))
    pos = b.fcmp_ordered(">", x, F(0.0))
    if j == 0:
        return b.select(pos, F(1.0), F(0.0)) if deriv else b.select(pos, x, F(0.0))
    if j == 1:
        c = 0.7978845608028654
        t = tanh(b.fmul(F(c), b.fadd(x, b.fmul(F(0.044715), b.fmul(x, b.fmul(x, x))))))
        if not deriv:
            return b.fmul(b.fmul(F(0.5), x), b.fadd(F(1.0), t))
        return b.fadd(b.fmul(F(0.5), b.fadd(F(1.0), t)),
                      b.fmul(b.fmul(b.fmul(F(0.5), x), b.fsub(F(1.0), b.fmul(t, t))),
                             b.fmul(F(c), b.fadd(F(1.0), b.fmul(F(3 * 0.044715), b.fmul(x, x))))))
    if j in (2, 4):
        sg = b.call(rcp_f, [b.fadd(F(1.0), exp(b.fsub(F(0.0), x)))])
        if j == 2:
            return b.fadd(sg, b.fmul(x, b.fmul(sg, b.fsub(F(1.0), sg)))) if deriv else b.fmul(x, sg)
        return b.fmul(sg, b.fsub(F(1.0), sg)) if deriv else sg
    if j == 3:
        th = tanh(x)
        return b.fsub(F(1.0), b.fmul(th, th)) if deriv else th
    if j == 5:
        return F(1.0) if deriv else x
    return b.select(pos, F(1.0), F(0.01)) if deriv else b.select(pos, x, b.fmul(F(0.01), x))


def _act_switch(b, x, kind, F, C, ex2_f, deriv, var):
    """Only the activation in use is computed (a branch per kind; every thread takes the same one)."""
    out = var(x)
    for j in range(7):
        with b.if_then(b.icmp_unsigned("==", kind, C(j))):
            b.store(_act_one(b, x, j, F, ex2_f, deriv), out)
    return b.load(out)


BM = BN = 128
BK = 32
PAD = 40                        # halves per shared-memory row: 80 bytes, so ldmatrix's 8 rows hit 8 different banks
PADT = 136                      # rows of a transposed tile (k rows of 128 m or n): 272 bytes, conflict-free too
TILED = ("gemm_tkn", "gemm_tkk", "gemm_tmn", "gemm_tmk")
# products with a last step compiled in: x W + b -> act (forward), dy W^T * act'(h) (backward), x W + b + residual
EPI_ACTS = (0, 1, 2)            # relu, gelu, silu: compiled into their own product kernels (no choice per element)
BF16_KERNELS = tuple([f"gemm_tkn_e1a{j}_bf" for j in EPI_ACTS] + [f"gemm_tkk_e2a{j}_bf" for j in EPI_ACTS] +
                     [f"gemm_tmn_aa{j}_bf" for j in EPI_ACTS] + ["gemm_tkn_bf", "gemm_tkn_e3_bf"])
EPI_KERNELS = tuple([(f"gemm_tkn_e1a{j}", True, False, 1, j) for j in EPI_ACTS] +
                    [(f"gemm_tkk_e2a{j}", True, True, 2, j) for j in EPI_ACTS] + [("gemm_tkn_e3", True, False, 3, None)])


def _tiled_module(early=False, wide=False, h=False, deep=False):
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
    # wide: 4 warps of 64 x 64 (4 x 8 mma tiles) over slices of 16, the shape fast GEMM libraries use on Ampere:
    # every value read from shared memory feeds twice as many products as a 64 x 32 warp tile
    if wide:
        early = True
    # h: the operands already in bfloat16 in memory (copies made just before), copied into shared memory with
    # cp.async (no registers, no conversion) four slices ahead: the tensor cores never wait for memory
    # deep (with h): slices of 32 k, three in flight (the shape CUTLASS uses on Ampere for bfloat16: 128 x 128 x 32,
    # 3 stages): half the barriers of slices of 16; the 60 KB of tiles are dynamic shared memory (above the 48 KB a
    # static array may take), so one block of 8 warps (or 4 with wide) per multiprocessor
    STAGES = 3 if deep else 4 if h else 2
    BKv, NTH, WNW = ((32, 128, 2) if wide else (32, 256, 4)) if deep else \
        (16, 128, 2) if wide else (16, 256, 4) if h else (BK, 256, 4)
    NTL = BN // WNW // 8                          # n8 tiles per warp
    PADv = BKv + 8
    VPR = BKv // 4                                # float32 vectors of 4 per row of a k-contiguous slice
    ir, m = _setup(("apsi_train_tiled_dw" if wide else "apsi_train_tiled_d") if deep else
                   "apsi_train_tiled_v" if (h and wide) else "apsi_train_tiled_h" if h else "apsi_train_tiled_w"
                   if wide else "apsi_train_tiled_x" if early else "apsi_train_tiled")
    annotations = m.add_named_metadata("nvvm.annotations")
    g = _Gen(ir, m)
    i16, i32, i64, f32, C, F, G = ir.IntType(16), g.i32, g.i64, g.f32, g.C, g.F, g.G
    d = g.decl
    ctaz_f = d("llvm.nvvm.read.ptx.sreg.ctaid.z", i32, [])
    nctaz_f = d("llvm.nvvm.read.ptx.sreg.nctaid.z", i32, [])
    bar_f = d("llvm.nvvm.barrier0", ir.VoidType(), [])
    HALVES = STAGES * 2 * max(BM * PADv, BKv * PADT)
    if deep:                                      # extern __shared__: sized at launch (DYN_BYTES)
        smem = ir.GlobalVariable(m, ir.ArrayType(i16, 0), "apsi_tiles", addrspace=3)
        smem.linkage = "external"
        smem.align = 16
    else:
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
    ldg4 = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([f32] * 4), [i64]),
                        "ld.global.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,l", side_effect=True)
    cpa = ir.InlineAsm(ir.FunctionType(ir.VoidType(), [i64, i64, i32]), "cp.async.cg.shared.global [$0], [$1], 16, $2;",
                       "l,l,r", side_effect=True)
    cpa_commit = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), "cp.async.commit_group;", "", side_effect=True)
    cpa_wait = ir.InlineAsm(ir.FunctionType(ir.VoidType(), []), f"cp.async.wait_group {max(0, STAGES - 2)};", "",
                            side_effect=True)
    cvt2 = ir.InlineAsm(ir.FunctionType(i32, [f32, f32]), "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f",
                        side_effect=False)
    ex2_f = d("llvm.nvvm.ex2.approx.f", f32, [f32])
    args = [("A", G(f32)), ("B", G(f32)), ("C", G(f32)), ("BIAS", G(f32)), ("M", i32), ("N", i32), ("K", i32),
            ("sam", i32), ("sak", i32), ("sbk", i32), ("sbn", i32), ("ldc", i32), ("nb2", i32), ("a1", i32),
            ("a2", i32), ("b1", i32), ("b2", i32), ("c1", i32), ("c2", i32), ("alpha", f32), ("beta", i32),
            ("has_bias", i32), ("kchunk", i32), ("X2", G(f32)), ("epi", i32), ("act", i32), ("DB", G(f32)),
            ("want_db", i32), ("CNT", G(i32))]
    membar_t = d("llvm.nvvm.membar.gl", ir.VoidType(), [])
    ld_cg_t = ir.InlineAsm(ir.FunctionType(f32, [i64]), "ld.global.cg.f32 $0, [$1];", "=f,l", side_effect=True)

    # DB (kernels with B n-contiguous and A transposed: weight gradients x^T dy): also DB[n] += sum over k of B,
    # the bias gradient, from the values already in registers (no second pass over dy)
    # epi (after alpha, bias and beta): 1 = X2 <- v, C <- act(v) (a linear layer and its activation in one pass);
    # 2 = C <- v * act'(X2) (the backward pass through that activation); 3 = C <- v + X2 (a residual connection)

    def kernel(name, a_kc, b_kc, epi_mode=0, act_j=None, a_act=None, a16=False, x16=False, c16=False):
        """a16: A is stored in bfloat16 (2 bytes a value); x16: X2 (the kept pre-activation of epi 1 / 2) is
        bfloat16; c16: C is written in bfloat16. The hidden layer of an MLP kept this way moves half the bytes."""
        fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [t for _, t in args]), name=name)
        fn.calling_convention = "ptx_kernel"
        if early and not deep:                    # two blocks per multiprocessor (at most 128 / 255 registers)
            for key, val in (("maxntidx", NTH), ("minctasm", 2)):
                annotations.add(m.add_metadata([fn, ir.MetaDataString(m, key), C(val)]))
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
        wm, wn = b.udiv(warp, C(WNW)), b.urem(warp, C(WNW))
        # tiles in groups of 8 rows: the blocks running at the same time share 8 rows of A and a few columns of B, so
        # a large B (weights of a big layer) is read from memory about 8 times less often (the rest hits the L2 cache)
        nbn = b.udiv(b.add(p["N"], C(BN - 1)), C(BN))
        nbm = b.udiv(b.add(p["M"], C(BM - 1)), C(BM))
        per_group = b.mul(nbn, C(8))
        first_m = b.mul(b.udiv(cta, per_group), C(8))
        gsize = umin(b.sub(nbm, umin(first_m, nbm)), C(8))
        gsize = b.select(b.icmp_unsigned("==", gsize, C(0)), C(1), gsize)
        within = b.urem(cta, per_group)
        bm = b.mul(b.add(first_m, b.urem(within, gsize)), C(BM))
        bn = b.mul(b.udiv(within, gsize), C(BN))
        k_beg = b.mul(ctaz, p["kchunk"])
        k_end = umin(b.add(k_beg, p["kchunk"]), p["K"])
        ntiles = b.udiv(b.add(b.sub(k_end, umin(k_beg, k_end)), C(BKv - 1)), C(BKv))
        Mm1, Nm1 = b.sub(p["M"], C(1)), b.sub(p["N"], C(1))
        Km4 = b.sub(p["K"], C(4))

        def rne(x):
            u = b.bitcast(x, i32)
            return b.lshr(b.add(b.add(u, C(0x7FFF)), b.and_(b.lshr(u, C(16)), C(1))), C(16))

        def load4(base_off, ptr, contig_idx, contig_lim, other_idx, other_lim, contig_stride_is_1, s_other, act=None,
                  raw=None, half=False):
            """Issues the load of 4 float32 along the contiguous dimension; the values are used (zeros outside,
            the activation, the rounding to bfloat16) only after the products of the current slice, so the wait
            for memory overlaps the tensor-core work instead of preceding it. act: the activation applied on the
            way (A is a pre-activation kept instead of its activation); raw: collects the float32 values."""
            ok = b.and_(b.icmp_unsigned("<", contig_idx, contig_lim), b.icmp_unsigned("<", other_idx, other_lim))
            ci = umin(contig_idx, b.sub(contig_lim, C(4)))
            oi = umin(other_idx, b.sub(other_lim, C(1)))
            idx = b.add(base_off, b.add(b.mul(z64(oi), z64(s_other)), z64(ci)))
            if half:                              # 4 bfloat16 (8 bytes): already rounded, widened to float32
                p16 = b.bitcast(ptr, i16.as_pointer(1))
                hv = b.load(b.bitcast(b.gep(p16, [idx], inbounds=True), ir.VectorType(i16, 4).as_pointer(1)), align=8)
                vec = ir.Constant(ir.VectorType(f32, 4), None)
                for j in range(4):
                    w = b.shl(b.zext(b.extract_element(hv, C(j)), i32), C(16))
                    vec = b.insert_element(vec, b.bitcast(w, f32), C(j))
            elif early:
                # an inline load with side effects: LLVM would otherwise sink it into the block that stores the next
                # slice (after the products); with two blocks per multiprocessor the other block hides that wait,
                # so this variant is chosen per product only where the card measures it faster
                addr = b.ptrtoint(b.gep(ptr, [idx], inbounds=True), i64)
                return ("pending", b.call(ldg4, [addr]), ok, act, raw)
            else:
                vp = b.bitcast(b.gep(ptr, [idx], inbounds=True), ir.VectorType(f32, 4).as_pointer(1))
                vec = b.load(vp, align=16)
            hs = []
            for j in range(4):                    # rounded to bfloat16 at once (round to nearest even)
                x = b.extract_element(vec, C(j))
                if act is not None:
                    x = _act_one(b, x, act, F, ex2_f, False)
                v = b.select(ok, x, F(0.0))
                if raw is not None:
                    raw.append(v)
                u = b.bitcast(v, i32)
                hs.append(b.lshr(b.add(b.add(u, C(0x7FFF)), b.and_(b.lshr(u, C(16)), C(1))), C(16)))
            return ("now", [b.or_(hs[0], b.shl(hs[1], C(16))), b.or_(hs[2], b.shl(hs[3], C(16)))])

        def convert(pend):
            """The 4 values of a pending load as 2 words of 2 bfloat16 each (one cvt.rn.bf16x2 per pair: round to
            nearest even, the same as the emulator)."""
            if pend[0] == "now":
                return pend[1]
            _, vec, ok, act, raw = pend
            xs = []
            for j in range(4):
                x = b.extract_value(vec, j)
                if act is not None:
                    x = _act_one(b, x, act, F, ex2_f, False)
                v = b.select(ok, x, F(0.0))
                if raw is not None:
                    raw.append(v)
                xs.append(v)
            return [b.call(cvt2, [xs[1], xs[0]]), b.call(cvt2, [xs[3], xs[2]])]

        def sh_ptr(half_index):
            return b.gep(smem, [C(0), half_index], inbounds=True)

        def store_row4(half_index, ws):
            """4 halves at consecutive k (8 bytes), given as 2 packed words."""
            v = ir.Constant(ir.VectorType(i32, 2), None)
            v = b.insert_element(v, ws[0], C(0))
            v = b.insert_element(v, ws[1], C(1))
            b.store(v, b.bitcast(sh_ptr(half_index), ir.VectorType(i32, 2).as_pointer(3)), align=8)

        REG = max(BM * PADv, BKv * PADT)
        A_BASE = [C(0), C(REG)]
        B_BASE = [C(2 * REG), C(3 * REG)]

        def fetch(kt):
            """The registers of one k slice: 4 vectors of A and 4 of B per thread."""
            k0 = b.add(k_beg, b.mul(kt, C(BKv)))
            regs = []
            for i in range(BM * BKv // 4 // NTH):
                idx = b.add(tid, C(NTH * i))
                if a_kc:                              # A(m, k) = A[m * sam + k]
                    row, kq = b.udiv(idx, C(VPR)), b.mul(b.urem(idx, C(VPR)), C(4))
                    regs.append(("A", row, kq, load4(offs[0], p["A"], b.add(k0, kq), k_end, b.add(bm, row), p["M"],
                                                     True, p["sam"], half=a16)))
                else:                                 # A(m, k) = A[k * sak + m]
                    kk, mq = b.lshr(idx, C(5)), b.mul(b.and_(idx, C(31)), C(4))
                    regs.append(("At", mq, kk, load4(offs[0], p["A"], b.add(bm, mq), p["M"], b.add(k0, kk), k_end,
                                                     True, p["sak"], a_act, half=a16)))
                if b_kc:                              # B(k, n) = B[n * sbn + k]
                    nrow, kq = b.udiv(idx, C(VPR)), b.mul(b.urem(idx, C(VPR)), C(4))
                    regs.append(("B", nrow, kq, load4(offs[1], p["B"], b.add(k0, kq), k_end, b.add(bn, nrow), p["N"],
                                                      True, p["sbn"])))
                else:                                 # B(k, n) = B[k * sbk + n]
                    kk, nq = b.lshr(idx, C(5)), b.mul(b.and_(idx, C(31)), C(4))
                    raw = [] if (not a_kc and epi_mode == 0) else None
                    regs.append(("Bt", nq, kk, load4(offs[1], p["B"], b.add(bn, nq), p["N"], b.add(k0, kk), k_end,
                                                     True, p["sbk"], None, raw), raw))
            return regs

        def put(regs, buf):
            for kind, r, c, pend, *_ in regs:
                hs = convert(pend)
                if kind in ("A", "At"):
                    base = b.select(b.icmp_unsigned("==", buf, C(0)), A_BASE[0], A_BASE[1])
                else:
                    base = b.select(b.icmp_unsigned("==", buf, C(0)), B_BASE[0], B_BASE[1])
                if kind in ("A", "B"):                # [m or n][k]: 4 consecutive k
                    store_row4(b.add(base, b.add(b.mul(r, C(PADv)), c)), hs)
                else:                                 # [k][m or n]: 4 consecutive m or n, stored the same way
                    store_row4(b.add(base, b.add(b.mul(c, C(PADT)), r)), hs)

        accs = [[[var(F(0.0)) for _ in range(4)] for _ in range(NTL)] for _ in range(4)]
        dbs = [var(F(0.0)) for _ in range(4)] if (not a_kc and not b_kc and epi_mode == 0 and not h) else None

        def add_db(regs):
            if dbs is None:
                return
            for r_ in regs:
                if r_[0] == "Bt" and r_[4]:
                    for j in range(4):
                        b.store(b.fadd(b.load(dbs[j]), r_[4][j]), dbs[j])
        def issue(kt, stage):
            """cp.async of slice kt (8 bfloat16 = 16 bytes per thread and operand) into a stage of shared memory;
            outside the matrix the bytes are zeros."""
            k0 = b.add(k_beg, b.mul(kt, C(BKv)))
            sbase_a = b.mul(stage, C(2 * REG))
            sbase_b = b.add(sbase_a, C(REG))
            A16 = b.bitcast(p["A"], i16.as_pointer(1))
            B16 = b.bitcast(p["B"], i16.as_pointer(1))
            for which, ptr, kc, base_off, lim_o, s_o, o0 in (("A", A16, a_kc, offs[0], p["M"], p["sam"] if a_kc else p["sak"], bm),
                                                           ("B", B16, b_kc, offs[1], p["N"], p["sbn"] if b_kc else p["sbk"], bn)):
                sb_ = sbase_a if which == "A" else sbase_b
                # 128 rows x BKv k = 128 * BKv / 8 chunks of 8 halves (16 bytes) per operand, NTH threads: each
                # thread copies its share (one chunk with 256 threads, two with the 4-warp build)
                for part in range(128 * BKv // 8 // NTH):
                    cid = b.add(tid, C(NTH * part))
                    if kc:                            # [row][k]: BKv / 8 chunks per row
                        row, kq = b.udiv(cid, C(BKv // 8)), b.mul(b.urem(cid, C(BKv // 8)), C(8))
                        ci, cl, oi, ol = b.add(k0, kq), k_end, b.add(o0, row), lim_o
                        dst = b.add(sb_, b.add(b.mul(row, C(PADv)), kq))
                    else:                             # [k][row]: 16 chunks of 8 per k
                        kk, q8 = b.lshr(cid, C(4)), b.mul(b.and_(cid, C(15)), C(8))
                        ci, cl, oi, ol = b.add(o0, q8), lim_o, b.add(k0, kk), k_end
                        dst = b.add(sb_, b.add(b.mul(kk, C(PADT)), q8))
                    ok = b.and_(b.icmp_unsigned("<", ci, cl), b.icmp_unsigned("<", oi, ol))
                    ci2 = umin(ci, b.sub(cl, C(8)))
                    oi2 = umin(oi, b.sub(ol, C(1)))
                    idx = b.add(base_off, b.add(b.mul(z64(oi2), z64(s_o)), z64(ci2)))
                    src = b.ptrtoint(b.gep(ptr, [idx], inbounds=True), i64)
                    b.call(cpa, [b.ptrtoint(sh_ptr(dst), i64), src, b.select(ok, C(16), C(0))])

        if h:
            for st_ in range(STAGES - 1):
                with b.if_then(b.icmp_unsigned("<", C(st_), ntiles)):
                    issue(C(st_), C(st_))
                b.call(cpa_commit, [])
        else:
            first_regs = fetch(C(0))
            put(first_regs, C(0))
            add_db(first_regs)
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
        if h:
            b.call(cpa_wait, [])                      # slice i has arrived (the later ones may still be coming)
            b.call(bar_f, [])                         # ... for every thread, and everyone is done with slice i - 1
            ahead = b.add(i, C(STAGES - 1))
            with b.if_then(b.icmp_unsigned("<", ahead, ntiles)):
                issue(ahead, b.urem(ahead, C(STAGES)))
            b.call(cpa_commit, [])
            abase = b.mul(b.urem(i, C(STAGES)), C(2 * REG))
            bbase = b.add(abase, C(REG))
        else:
            regs = fetch(umin(nxt, b.sub(ntiles, C(1))))
            buf = b.and_(i, C(1))
            abase = b.select(b.icmp_unsigned("==", buf, C(0)), A_BASE[0], A_BASE[1])
            bbase = b.select(b.icmp_unsigned("==", buf, C(0)), B_BASE[0], B_BASE[1])
        for ks in range(BKv // 16):
            l7, l8, l16 = b.and_(lane, C(7)), b.and_(b.lshr(lane, C(3)), C(1)), b.lshr(lane, C(4))
            afr = []
            for mt in range(4):
                if a_kc:                              # [m][k]: rows of A
                    row = b.add(b.add(b.mul(wm, C(64)), C(mt * 16)), b.and_(lane, C(15)))
                    col = b.add(C(ks * 16), b.mul(l16, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(abase, b.add(b.mul(row, C(PADv)), col))), i64)
                    r = b.call(ldm, [addr])
                else:                                 # [k][m]: transposed while loading
                    krow = b.add(C(ks * 16), b.add(l7, b.mul(l16, C(8))))
                    mcol = b.add(b.add(b.mul(wm, C(64)), C(mt * 16)), b.mul(l8, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(abase, b.add(b.mul(krow, C(PADT)), mcol))), i64)
                    r = b.call(ldmt, [addr])
                afr.append([b.extract_value(r, j) for j in range(4)])
            bfr = []
            for pr in range(NTL // 2):
                if b_kc:                              # [n][k]
                    nrow = b.add(b.add(b.mul(wn, C(NTL * 8)), C(pr * 16)), b.add(l7, b.mul(l16, C(8))))
                    col = b.add(C(ks * 16), b.mul(l8, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(bbase, b.add(b.mul(nrow, C(PADv)), col))), i64)
                    r = b.call(ldm, [addr])
                else:                                 # [k][n], transposed while loading
                    krow = b.add(C(ks * 16), b.add(l7, b.mul(l8, C(8))))
                    ncol = b.add(b.add(b.mul(wn, C(NTL * 8)), C(pr * 16)), b.mul(l16, C(8)))
                    addr = b.ptrtoint(sh_ptr(b.add(bbase, b.add(b.mul(krow, C(PADT)), ncol))), i64)
                    r = b.call(ldmt, [addr])
                q = [b.extract_value(r, j) for j in range(4)]
                bfr += [[q[0], q[1]], [q[2], q[3]]]
            for mt in range(4):
                for nt in range(NTL):
                    acc = [b.load(a) for a in accs[mt][nt]]
                    r = b.call(mma, afr[mt] + bfr[nt] + acc)
                    for c in range(4):
                        b.store(b.extract_value(r, c), accs[mt][nt][c])
        if not h:
            with b.if_then(has_next):
                put(regs, b.and_(nxt, C(1)))
                add_db(regs)
            b.call(bar_f, [])
        b.store(nxt, it)
        b.branch(head)
        b.position_at_end(done)
        if dbs is not None:                           # the bias gradient: once per column, from the first row of tiles
            nq = b.mul(b.and_(tid, C(31)), C(4))
            with b.if_then(b.and_(b.icmp_unsigned("!=", p["want_db"], C(0)), b.icmp_unsigned("==", bm, C(0)))):
                for j in range(4):
                    col = b.add(bn, b.add(nq, C(j)))
                    with b.if_then(b.icmp_unsigned("<", col, p["N"])):
                        b.call(red, [b.ptrtoint(b.gep(p["DB"], [z64(col)], inbounds=True), i64), b.load(dbs[j])])
        split = b.icmp_unsigned(">", nz, C(1))
        first = b.icmp_unsigned("==", ctaz, C(0))
        for mt in range(4):
            for nt in range(NTL):
                for c in range(4):
                    row = b.add(b.add(bm, b.add(b.mul(wm, C(64)), C(mt * 16 + (c // 2) * 8))), gq)
                    col = b.add(b.add(bn, b.add(b.mul(wn, C(NTL * 8)), C(nt * 8))), b.add(b.mul(tig, C(2)), C(c % 2)))
                    with b.if_then(b.and_(b.icmp_unsigned("<", row, p["M"]), b.icmp_unsigned("<", col, p["N"]))):
                        idx = b.add(offs[2], b.add(b.mul(z64(row), z64(p["ldc"])), z64(col)))
                        ptr = b.gep(p["C"], [idx], inbounds=True)
                        v = b.fmul(b.load(accs[mt][nt][c]), p["alpha"])
                        tmp = var(v)
                        with b.if_then(b.and_(b.icmp_unsigned("!=", p["has_bias"], C(0)), first)):
                            bp = b.gep(p["BIAS"], [z64(col)], inbounds=True)
                            b.store(b.fadd(b.load(tmp), b.load(bp, align=4)), tmp)
                        with b.if_else(split) as (then, other):
                            with then:                # each part of k writes its share to its own slice of the
                                # workspace (X2: [parts, M, ldc]); ksum adds the parts in order (no atomics)
                                widx = b.add(b.mul(z64(ctaz), b.mul(z64(p["M"]), z64(p["ldc"]))),
                                             b.add(b.mul(z64(row), z64(p["ldc"])), z64(col)))
                                b.store(b.load(tmp), b.gep(p["X2"], [widx], inbounds=True), align=4)
                            with other:
                                with b.if_then(b.icmp_unsigned("!=", p["beta"], C(0))):
                                    b.store(b.fadd(b.load(tmp), b.load(ptr, align=4)), tmp)
                                if epi_mode:              # this kernel's own last step, compiled in
                                    xp = b.gep(p["X2"], [idx], inbounds=True)
                                    xp16 = b.gep(b.bitcast(p["X2"], i16.as_pointer(1)), [idx], inbounds=True)
                                    vv = b.load(tmp)

                                    def ld_x():
                                        if x16:
                                            w = b.shl(b.zext(b.load(xp16, align=2), i32), C(16))
                                            return b.bitcast(w, f32)
                                        return b.load(xp, align=4)
                                    if epi_mode == 1:
                                        if x16:           # the pre-activation kept in bfloat16; act() of that value
                                            hb = b.trunc(rne(vv), i16)
                                            b.store(hb, xp16, align=2)
                                            vv = b.bitcast(b.shl(b.zext(hb, i32), C(16)), f32)
                                        else:
                                            b.store(vv, xp, align=4)
                                        out = _act_one(b, vv, act_j, F, ex2_f, False)
                                    elif epi_mode == 2:
                                        out = b.fmul(vv, _act_one(b, ld_x(), act_j, F, ex2_f, True))
                                    else:
                                        out = b.fadd(vv, ld_x())
                                    b.store(out, tmp)
                                if c16:
                                    cp16 = b.gep(b.bitcast(p["C"], i16.as_pointer(1)), [idx], inbounds=True)
                                    b.store(b.trunc(rne(b.load(tmp)), i16), cp16, align=2)
                                else:
                                    b.store(b.load(tmp), ptr, align=4)
        # split k: the last part of this tile to finish adds the parts in order (what ksum did, in the same order,
        # without a second launch); the counter of the tile is reset for the next product. The flag lives in the
        # tiles' shared memory (free by now): no shared memory of its own.
        with b.if_then(split):
            b.call(membar_t, [])
            b.call(bar_f, [])
            flag = b.bitcast(b.gep(smem, [C(0), C(0)], inbounds=True), i32.as_pointer(3))
            with b.if_then(b.icmp_unsigned("==", tid, C(0))):
                old = b.atomic_rmw("add", b.gep(p["CNT"], [z64(cta)], inbounds=True), C(1), "acq_rel")
                b.store(b.zext(b.icmp_unsigned("==", old, b.sub(nz, C(1))), i32), flag)
            b.call(bar_f, [])
            with b.if_then(b.icmp_unsigned("!=", b.load(flag), C(0))):
                b.call(membar_t, [])
                plane = b.mul(z64(p["M"]), z64(p["ldc"]))
                els = []
                for mt in range(4):
                    for nt in range(NTL):
                        for c in range(4):
                            row = b.add(b.add(bm, b.add(b.mul(wm, C(64)), C(mt * 16 + (c // 2) * 8))), gq)
                            col = b.add(b.add(bn, b.add(b.mul(wn, C(NTL * 8)), C(nt * 8))),
                                        b.add(b.mul(tig, C(2)), C(c % 2)))
                            ok = b.and_(b.icmp_unsigned("<", row, p["M"]), b.icmp_unsigned("<", col, p["N"]))
                            at = b.add(b.mul(z64(b.select(ok, row, C(0))), z64(p["ldc"])),
                                       z64(b.select(ok, col, C(0))))
                            els.append((ok, at, var(F(0.0))))
                zv = var(C(0))
                head = fn.append_basic_block("ks_head")
                body = fn.append_basic_block("ks_body")
                end = fn.append_basic_block("ks_end")
                b.branch(head)
                b.position_at_end(head)
                zi = b.load(zv)
                b.cbranch(b.icmp_unsigned("<", zi, nz), body, end)
                b.position_at_end(body)
                zo = b.mul(z64(zi), plane)                # every element's part z: independent loads in flight
                vals = [b.call(ld_cg_t, [b.ptrtoint(b.gep(p["X2"], [b.add(zo, at)], inbounds=True), i64)])
                        for _, at, _ in els]
                for (_, _, sv), v in zip(els, vals):
                    b.store(b.fadd(b.load(sv), v), sv)
                b.store(b.add(zi, C(1)), zv)
                b.branch(head)
                b.position_at_end(end)
                for ok, at, sv in els:
                    with b.if_then(ok):
                        cp = b.gep(p["C"], [at], inbounds=True)
                        b.store(b.fadd(b.load(cp, align=4), b.load(sv)), cp, align=4)
                with b.if_then(b.icmp_unsigned("==", tid, C(0))):
                    b.store(C(0), b.gep(p["CNT"], [z64(cta)], inbounds=True))
        b.ret_void()
        ab.branch(top)

    kernel("apsi_kernel", True, False)            # = gemm_tkn: x @ W
    for name, a_kc, b_kc in (("gemm_tkk", True, True), ("gemm_tmn", False, False), ("gemm_tmk", False, True)):
        kernel(name, a_kc, b_kc)
    for name, a_kc, b_kc, e, j in EPI_KERNELS:
        kernel(name, a_kc, b_kc, e, j)
    if not h:
        for j in EPI_ACTS:                        # dW = act(h)^T dy without keeping act(h)
            kernel(f"gemm_tmn_aa{j}", False, False, 0, None, j)
        # the hidden layer of an MLP in bfloat16 (BF16_KERNELS): h = x W1 + b kept and act(h) written in bfloat16,
        # act(h) W2 (+ residual) reading it, and the backward pass reading h in bfloat16
        for j in EPI_ACTS:
            kernel(f"gemm_tkn_e1a{j}_bf", True, False, 1, j, None, False, True, True)
            kernel(f"gemm_tkk_e2a{j}_bf", True, True, 2, j, None, False, True, False)
            kernel(f"gemm_tmn_aa{j}_bf", False, False, 0, None, j, True)
        kernel("gemm_tkn_bf", True, False, 0, None, None, True)
        kernel("gemm_tkn_e3_bf", True, False, 3, None, None, True)

    if early or h:
        return _emit(m, "sm_80")
    # to_bf16: Y16 = bfloat16(X), two values per thread (the operands of the h build)
    fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [G(f32), G(i32), i32]), name="to_bf16")
    fn.calling_convention = "ptx_kernel"
    X_, Y_, n_ = fn.args
    b = ir.IRBuilder(fn.append_basic_block("top"))
    i = b.add(b.mul(b.call(g.ctax_f, []), C(256)), b.call(g.tid_f, []))
    with b.if_then(b.icmp_unsigned("<", b.mul(i, C(2)), n_)):
        ix = b.zext(b.mul(i, C(2)), i64)
        v2 = b.load(b.bitcast(b.gep(X_, [ix], inbounds=True), ir.VectorType(f32, 2).as_pointer(1)), align=8)
        w = b.call(cvt2, [b.extract_element(v2, C(1)), b.extract_element(v2, C(0))])
        b.store(w, b.gep(Y_, [b.zext(i, i64)], inbounds=True), align=4)
    b.ret_void()
    # bf_lo: Y = X - bfloat16(X), the part bfloat16 drops. float32 products as three bfloat16 products on the
    # tensor cores: A B = hi(A) hi(B) + lo(A) hi(B) + hi(A) lo(B), with an error near float32's own
    fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [G(f32), G(f32), i32]), name="bf_lo")
    fn.calling_convention = "ptx_kernel"
    X_, Y_, n_ = fn.args
    b = ir.IRBuilder(fn.append_basic_block("top"))
    i = b.add(b.mul(b.call(g.ctax_f, []), C(256)), b.call(g.tid_f, []))
    with b.if_then(b.icmp_unsigned("<", i, n_)):
        ix = b.zext(i, i64)
        x = b.load(b.gep(X_, [ix], inbounds=True), align=4)
        w = b.call(cvt2, [F(0.0), x])                       # low half: bfloat16(x)
        hi = b.bitcast(b.shl(w, C(16)), f32)
        b.store(b.fsub(x, hi), b.gep(Y_, [ix], inbounds=True), align=4)
    b.ret_void()
    return _emit(m, "sm_80")


FLASH_HD = (32, 64)               # head sizes the fused attention takes (others: the separate products)
BQ = 64                          # queries (forward) or keys (backward) per block of 4 warps


def _flash_module():
    """Attention fused into one kernel each way (the FlashAttention-2 idea, written for Apsi): scores, the softmax
    and the weighted sum stay in registers and shared memory, so the T x T matrices are never written to memory.

    Forward, per block of 64 queries of one (batch, head): Q in shared memory; for each tile of 64 keys, S = Q K^T
    on the tensor cores, a running (online) softmax per row, O += P V with P taken straight from the registers that
    hold S. It writes O and, per row, L = log2 of the softmax denominator (with the row's maximum folded in).
    Backward, per block of 64 keys: for each tile of 64 queries it recomputes P from Q, K and L, then dV += P^T dO,
    dP = dO V^T, dS = P (dP - D), dK += dS^T Q, and dQ += dS K added atomically (several key tiles add to one dQ).
    D (one number per row: the dot product of dO and O) comes from attn_dot. Inputs and outputs are float32 in the
    trainer's [batch, time, heads x head_size] layout; the products run in bfloat16 with float32 sums."""
    ir, m = _setup("apsi_train_flash")
    g = _Gen(ir, m)
    i16, i32, i64, f32, C, F, G = ir.IntType(16), g.i32, g.i64, g.f32, g.C, g.F, g.G
    d = g.decl
    bar_f = d("llvm.nvvm.barrier0", ir.VoidType(), [])
    shfl_f = d("llvm.nvvm.shfl.sync.bfly.f32", f32, [i32, f32, i32, i32])
    ex2_f = d("llvm.nvvm.ex2.approx.f", f32, [f32])
    lg2_f = d("llvm.nvvm.lg2.approx.f", f32, [f32])
    max_f = d("llvm.maxnum.f32", f32, [f32, f32])
    mma = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([f32] * 4), [i32] * 6 + [f32] * 4),
                       "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, "
                       "{$8,$9}, {$10,$11,$12,$13};", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", side_effect=False)
    ldm = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([i32] * 4), [i64]),
                       "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l", side_effect=True)
    ldmt = ir.InlineAsm(ir.FunctionType(ir.LiteralStructType([i32] * 4), [i64]),
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l",
                        side_effect=True)
    red = ir.InlineAsm(ir.FunctionType(ir.VoidType(), [i64, f32]), "red.global.add.f32 [$0], $1;", "l,f",
                       side_effect=True)
    NEG = -1.0e30

    def build(name, hd, kind):
        """kind "f": forward; "b": backward with dQ added atomically; "k": backward without dQ (dK, dV only);
        "q": dQ alone, a block per 64 queries summing over the key tiles in registers (no atomics, so the same bits
        every run: "k" and "q" together are the deterministic backward)."""
        backward = kind in "bk"
        SP = hd + 8
        TILE = BQ * SP
        SPS = BQ + 8                  # rows of dS^T: 64 queries (+8), whatever the head size
        halves = {"b": 4 * TILE + BQ * SPS, "k": 4 * TILE}.get(kind, 3 * TILE)
        sm = ir.GlobalVariable(m, ir.ArrayType(i16, halves), f"{name}_s", addrspace=3)
        sm.linkage = "internal"
        sm.align = 16
        sm.initializer = ir.Constant(ir.ArrayType(i16, halves), None)
        smf = ir.GlobalVariable(m, ir.ArrayType(f32, 2 * BQ), f"{name}_f", addrspace=3)
        smf.linkage = "internal"
        smf.align = 16
        smf.initializer = ir.Constant(ir.ArrayType(f32, 2 * BQ), None)
        if backward:
            args = [("Q", G(f32)), ("K", G(f32)), ("V", G(f32)), ("DO", G(f32)), ("L", G(f32)), ("DD", G(f32)),
                    ("DQ", G(f32)), ("DK", G(f32)), ("DV", G(f32)), ("T", i32), ("D", i32), ("H", i32),
                    ("scale", f32), ("scale2", f32), ("causal", i32)]
        elif kind == "q":
            args = [("Q", G(f32)), ("K", G(f32)), ("V", G(f32)), ("DO", G(f32)), ("L", G(f32)), ("DD", G(f32)),
                    ("DQ", G(f32)), ("T", i32), ("D", i32), ("H", i32), ("scale", f32), ("scale2", f32),
                    ("causal", i32)]
        else:
            args = [("Q", G(f32)), ("K", G(f32)), ("V", G(f32)), ("O", G(f32)), ("L", G(f32)), ("T", i32), ("D", i32),
                    ("H", i32), ("scale2", f32), ("causal", i32)]
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

        def rne(x):
            u = b.bitcast(x, i32)
            return b.lshr(b.add(b.add(u, C(0x7FFF)), b.and_(b.lshr(u, C(16)), C(1))), C(16))

        def pack(x0, x1):
            return b.or_(rne(x0), b.shl(rne(x1), C(16)))

        def shp(tile, row, col):
            stride = SPS if (backward and tile == 4) else SP
            return b.gep(sm, [C(0), b.add(C(tile * TILE), b.add(b.mul(row, C(stride)), col))], inbounds=True)

        def fp(i):
            return b.gep(smf, [C(0), i], inbounds=True)

        def quad(v, op):
            for k in (1, 2):
                o = b.call(shfl_f, [C(-1), v, C(k), C(31)])
                v = b.call(max_f, [v, o]) if op == "max" else b.fadd(v, o)
            return v

        tid, cta, z = b.call(g.tid_f, []), b.call(g.ctax_f, []), b.call(g.ctay_f, [])
        lane, w = b.and_(tid, C(31)), b.lshr(tid, C(5))
        gq, tig = b.lshr(lane, C(2)), b.and_(lane, C(3))
        l7, l8, l16 = b.and_(lane, C(7)), b.and_(b.lshr(lane, C(3)), C(1)), b.lshr(lane, C(4))
        T, D = p["T"], p["D"]
        base = b.add(b.mul(z64(b.udiv(z, p["H"])), b.mul(z64(T), z64(D))), z64(b.mul(b.urem(z, p["H"]), C(hd))))
        causal = b.icmp_unsigned("!=", p["causal"], C(0))

        def load_tile(tile, src, row0):
            """64 rows x hd float32 of one head -> bfloat16 in shared memory (zeros past T)."""
            per = hd // 4
            for i in range(BQ * per // 128):
                idx = b.add(tid, C(128 * i))
                row, c4 = b.udiv(idx, C(per)), b.mul(b.urem(idx, C(per)), C(4))
                r = b.add(row0, row)
                ok = b.icmp_unsigned("<", r, T)
                gi = b.add(base, b.add(b.mul(z64(umin(r, b.sub(T, C(1)))), z64(D)), z64(c4)))
                vec = b.load(b.bitcast(b.gep(src, [gi], inbounds=True), ir.VectorType(f32, 4).as_pointer(1)), align=16)
                hs = [rne(b.select(ok, b.extract_element(vec, C(j)), F(0.0))) for j in range(4)]
                v2 = ir.Constant(ir.VectorType(i32, 2), None)
                v2 = b.insert_element(v2, b.or_(hs[0], b.shl(hs[1], C(16))), C(0))
                v2 = b.insert_element(v2, b.or_(hs[2], b.shl(hs[3], C(16))), C(1))
                b.store(v2, b.bitcast(shp(tile, row, c4), ir.VectorType(i32, 2).as_pointer(3)), align=8)

        def frag_rows(tile, row0, ks):
            """A fragment (16 rows x 16 of k) from a [row][k] tile."""
            a = b.ptrtoint(shp(tile, b.add(row0, b.and_(lane, C(15))), b.add(C(ks * 16), b.mul(l16, C(8)))), i64)
            r = b.call(ldm, [a])
            return [b.extract_value(r, j) for j in range(4)]

        def frag_rows_t(tile, col0, kk):
            """A fragment (16 rows x 16 of k) from a [k][row] tile (transposed while loading)."""
            a = b.ptrtoint(shp(tile, b.add(C(kk * 16), b.add(l7, b.mul(l16, C(8)))), b.add(col0, b.mul(l8, C(8)))), i64)
            r = b.call(ldmt, [a])
            return [b.extract_value(r, j) for j in range(4)]

        def bfrag_nk(tile, n0, ks):
            """B fragments of two 8-wide n tiles from an [n][k] tile."""
            a = b.ptrtoint(shp(tile, b.add(n0, b.add(l7, b.mul(l16, C(8)))), b.add(C(ks * 16), b.mul(l8, C(8)))), i64)
            r = b.call(ldm, [a])
            q = [b.extract_value(r, j) for j in range(4)]
            return [q[0], q[1]], [q[2], q[3]]

        def bfrag_kn(tile, n0, kk):
            """B fragments of two 8-wide n tiles from a [k][n] tile (transposed while loading)."""
            a = b.ptrtoint(shp(tile, b.add(C(kk * 16), b.add(l7, b.mul(l8, C(8)))), b.add(n0, b.mul(l16, C(8)))), i64)
            r = b.call(ldmt, [a])
            q = [b.extract_value(r, j) for j in range(4)]
            return [q[0], q[1]], [q[2], q[3]]

        def mma_acc(acc_vars, a, bb):
            r = b.call(mma, a + bb + [b.load(v) for v in acc_vars])
            for c in range(4):
                b.store(b.extract_value(r, c), acc_vars[c])

        def scores(a_frags, tile_nk):
            """16 x 64 scores: A (16 x hd, in registers) times the [n][k] tile's rows."""
            s = [[var(F(0.0)) for _ in range(4)] for _ in range(8)]
            for ks in range(hd // 16):
                for np_ in range(4):
                    b0, b1 = bfrag_nk(tile_nk, C(np_ * 16), ks)
                    mma_acc(s[2 * np_], a_frags[ks], b0)
                    mma_acc(s[2 * np_ + 1], a_frags[ks], b1)
            return [[b.load(v) for v in row] for row in s]

        def a_from_c(vals, kk):
            """The registers of a 16 x 64 result, as the A fragment of k step kk (16 of the 64)."""
            x, y = vals[2 * kk], vals[2 * kk + 1]
            return [pack(x[0], x[1]), pack(x[2], x[3]), pack(y[0], y[1]), pack(y[2], y[3])]

        def loop(lo, hi, body):
            iv = var(lo)
            head = fn.append_basic_block("head")
            bod = fn.append_basic_block("body")
            end = fn.append_basic_block("end")
            b.branch(head)
            b.position_at_end(head)
            i = b.load(iv)
            b.cbranch(b.icmp_unsigned("<", i, hi), bod, end)
            b.position_at_end(bod)
            body(i)
            b.store(b.add(b.load(iv), C(1)), iv)
            b.branch(head)
            b.position_at_end(end)

        ntiles = b.udiv(b.add(T, C(BQ - 1)), C(BQ))
        if kind == "q":
            sQ, sK, sV = 0, 1, 2
            q0 = b.mul(cta, C(BQ))
            load_tile(sQ, p["Q"], q0)
            b.call(bar_f, [])
            qa = [frag_rows(sQ, b.mul(w, C(16)), ks) for ks in range(hd // 16)]
            b.call(bar_f, [])
            load_tile(sQ, p["DO"], q0)
            b.call(bar_f, [])
            doa = [frag_rows(sQ, b.mul(w, C(16)), ks) for ks in range(hd // 16)]
            lr, dr = [], []
            for r in range(2):
                row = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C(r * 8)))
                ok = b.icmp_unsigned("<", row, T)
                li = b.add(b.mul(z64(z), z64(T)), z64(umin(row, b.sub(T, C(1)))))
                lr.append(b.select(ok, b.load(b.gep(p["L"], [li], inbounds=True), align=4), F(0.0)))
                dr.append(b.select(ok, b.load(b.gep(p["DD"], [li], inbounds=True), align=4), F(0.0)))
            dq = [[var(F(0.0)) for _ in range(4)] for _ in range(hd // 8)]
            last = b.select(causal, umin(b.add(cta, C(1)), ntiles), ntiles)

            def kv_tile_q(j):
                b.call(bar_f, [])
                k0 = b.mul(j, C(BQ))
                load_tile(sK, p["K"], k0)
                load_tile(sV, p["V"], k0)
                b.call(bar_f, [])
                s = scores(qa, sK)                     # S = Q K^T: 16 queries x 64 keys
                dp = scores(doa, sV)                   # dP = dO V^T
                ds = [[None] * 4 for _ in range(8)]
                for nt in range(8):
                    for c in range(4):
                        key = b.add(b.add(k0, C(nt * 8 + (c & 1))), b.mul(tig, C(2)))
                        qrow = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C((c >> 1) * 8)))
                        bad = b.or_(b.icmp_unsigned(">=", key, T), b.and_(causal, b.icmp_unsigned(">", key, qrow)))
                        pv = b.call(ex2_f, [b.fsub(b.fmul(s[nt][c], p["scale2"]), lr[c >> 1])])
                        pv = b.select(bad, F(0.0), pv)
                        ds[nt][c] = b.fmul(pv, b.fsub(dp[nt][c], dr[c >> 1]))
                for kk in range(4):                    # dQ += dS K over this tile's 64 keys
                    a = a_from_c(ds, kk)
                    for pr in range(hd // 16):
                        b0, b1 = bfrag_kn(sK, C(pr * 16), kk)
                        mma_acc(dq[2 * pr], a, b0)
                        mma_acc(dq[2 * pr + 1], a, b1)
            loop(C(0), last, kv_tile_q)
            for r in range(2):
                row = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C(r * 8)))
                with b.if_then(b.icmp_unsigned("<", row, T)):
                    for nt in range(hd // 8):
                        for c in (2 * r, 2 * r + 1):
                            col = b.add(C(nt * 8 + (c & 1)), b.mul(tig, C(2)))
                            gi = b.add(base, b.add(b.mul(z64(row), z64(D)), z64(col)))
                            b.store(b.fmul(b.load(dq[nt][c]), p["scale"]), b.gep(p["DQ"], [gi], inbounds=True),
                                    align=4)
        elif not backward:
            sQ, sK, sV = 0, 1, 2
            q0 = b.mul(cta, C(BQ))
            load_tile(sQ, p["Q"], q0)
            b.call(bar_f, [])
            qa = [frag_rows(sQ, b.mul(w, C(16)), ks) for ks in range(hd // 16)]
            mrow = [var(F(NEG)), var(F(NEG))]
            lrow = [var(F(0.0)), var(F(0.0))]
            o = [[var(F(0.0)) for _ in range(4)] for _ in range(hd // 8)]
            last = b.select(causal, umin(b.add(cta, C(1)), ntiles), ntiles)

            def kv_tile(j):
                b.call(bar_f, [])
                k0 = b.mul(j, C(BQ))
                load_tile(sK, p["K"], k0)
                load_tile(sV, p["V"], k0)
                b.call(bar_f, [])
                s = scores(qa, sK)
                for nt in range(8):
                    for c in range(4):
                        key = b.add(k0, C(nt * 8 + (c & 1)))
                        key = b.add(key, b.mul(tig, C(2)))
                        qrow = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C((c >> 1) * 8)))
                        bad = b.or_(b.icmp_unsigned(">=", key, T), b.and_(causal, b.icmp_unsigned(">", key, qrow)))
                        s[nt][c] = b.select(bad, F(NEG), b.fmul(s[nt][c], p["scale2"]))
                for r in range(2):
                    mx = F(NEG)
                    for nt in range(8):
                        for c in (2 * r, 2 * r + 1):
                            mx = b.call(max_f, [mx, s[nt][c]])
                    mx = quad(mx, "max")
                    mold = b.load(mrow[r])
                    mnew = b.call(max_f, [mold, mx])
                    corr = b.call(ex2_f, [b.fsub(mold, mnew)])
                    tot = F(0.0)
                    for nt in range(8):
                        for c in (2 * r, 2 * r + 1):
                            s[nt][c] = b.call(ex2_f, [b.fsub(s[nt][c], mnew)])
                            tot = b.fadd(tot, s[nt][c])
                    b.store(b.fadd(b.fmul(b.load(lrow[r]), corr), tot), lrow[r])
                    b.store(mnew, mrow[r])
                    for nt in range(hd // 8):
                        for c in (2 * r, 2 * r + 1):
                            b.store(b.fmul(b.load(o[nt][c]), corr), o[nt][c])
                for kk in range(4):
                    a = a_from_c(s, kk)
                    for pr in range(hd // 16):
                        b0, b1 = bfrag_kn(sV, C(pr * 16), kk)
                        mma_acc(o[2 * pr], a, b0)
                        mma_acc(o[2 * pr + 1], a, b1)
            loop(C(0), last, kv_tile)
            for r in range(2):
                lt = quad(b.load(lrow[r]), "sum")
                inv = b.fdiv(F(1.0), lt)
                row = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C(r * 8)))
                with b.if_then(b.icmp_unsigned("<", row, T)):
                    for nt in range(hd // 8):
                        for c in (2 * r, 2 * r + 1):
                            col = b.add(C(nt * 8 + (c & 1)), b.mul(tig, C(2)))
                            gi = b.add(base, b.add(b.mul(z64(row), z64(D)), z64(col)))
                            b.store(b.fmul(b.load(o[nt][c]), inv), b.gep(p["O"], [gi], inbounds=True), align=4)
                    with b.if_then(b.icmp_unsigned("==", tig, C(0))):
                        li = b.add(b.mul(z64(z), z64(T)), z64(row))
                        b.store(b.fadd(b.load(mrow[r]), b.call(lg2_f, [lt])), b.gep(p["L"], [li], inbounds=True),
                                align=4)
        else:
            sK, sV, sQ, sO, sS = 0, 1, 2, 3, 4
            k0 = b.mul(cta, C(BQ))
            load_tile(sK, p["K"], k0)
            load_tile(sV, p["V"], k0)
            b.call(bar_f, [])
            ka = [frag_rows(sK, b.mul(w, C(16)), ks) for ks in range(hd // 16)]
            va = [frag_rows(sV, b.mul(w, C(16)), ks) for ks in range(hd // 16)]
            dv = [[var(F(0.0)) for _ in range(4)] for _ in range(hd // 8)]
            dk = [[var(F(0.0)) for _ in range(4)] for _ in range(hd // 8)]
            first = b.select(causal, cta, C(0))

            def q_tile(i):
                b.call(bar_f, [])
                q0 = b.mul(i, C(BQ))
                load_tile(sQ, p["Q"], q0)
                load_tile(sO, p["DO"], q0)
                with b.if_then(b.icmp_unsigned("<", tid, C(BQ))):
                    r = b.add(q0, tid)
                    ok = b.icmp_unsigned("<", r, T)
                    li = b.add(b.mul(z64(z), z64(T)), z64(umin(r, b.sub(T, C(1)))))
                    b.store(b.select(ok, b.load(b.gep(p["L"], [li], inbounds=True), align=4), F(0.0)), fp(tid))
                    b.store(b.select(ok, b.load(b.gep(p["DD"], [li], inbounds=True), align=4), F(0.0)),
                            fp(b.add(tid, C(BQ))))
                b.call(bar_f, [])
                s = scores(ka, sQ)                     # S^T: 16 keys x 64 queries
                dp = scores(va, sO)                    # dP^T = V dO^T
                pt = [[None] * 4 for _ in range(8)]
                ds = [[None] * 4 for _ in range(8)]
                for nt in range(8):
                    for c in range(4):
                        ql = b.add(C(nt * 8 + (c & 1)), b.mul(tig, C(2)))
                        qg = b.add(q0, ql)
                        key = b.add(b.add(k0, b.mul(w, C(16))), b.add(gq, C((c >> 1) * 8)))
                        bad = b.or_(b.icmp_unsigned(">=", qg, T), b.and_(causal, b.icmp_unsigned(">", key, qg)))
                        pv = b.call(ex2_f, [b.fsub(b.fmul(s[nt][c], p["scale2"]), b.load(fp(ql)))])
                        pv = b.select(bad, F(0.0), pv)
                        pt[nt][c] = pv
                        ds[nt][c] = b.fmul(pv, b.fsub(dp[nt][c], b.load(fp(b.add(ql, C(BQ))))))
                for kk in range(4):
                    a_p, a_s = a_from_c(pt, kk), a_from_c(ds, kk)
                    for pr in range(hd // 16):
                        b0, b1 = bfrag_kn(sO, C(pr * 16), kk)
                        mma_acc(dv[2 * pr], a_p, b0)
                        mma_acc(dv[2 * pr + 1], a_p, b1)
                        b0, b1 = bfrag_kn(sQ, C(pr * 16), kk)
                        mma_acc(dk[2 * pr], a_s, b0)
                        mma_acc(dk[2 * pr + 1], a_s, b1)
                if kind == "k":
                    return
                for nt in range(8):                    # dS^T to shared memory: [key][query] in bfloat16
                    for r in range(2):
                        kl = b.add(b.mul(w, C(16)), b.add(gq, C(r * 8)))
                        ql = b.add(C(nt * 8), b.mul(tig, C(2)))
                        b.store(pack(ds[nt][2 * r], ds[nt][2 * r + 1]),
                                b.bitcast(shp(sS, kl, ql), i32.as_pointer(3)), align=4)
                b.call(bar_f, [])
                dq = [[var(F(0.0)) for _ in range(4)] for _ in range(hd // 8)]
                for kk in range(4):                    # dQ (this warp's 16 queries) = dS K over the 64 keys
                    a = frag_rows_t(sS, b.mul(w, C(16)), kk)
                    for pr in range(hd // 16):
                        b0, b1 = bfrag_kn(sK, C(pr * 16), kk)
                        mma_acc(dq[2 * pr], a, b0)
                        mma_acc(dq[2 * pr + 1], a, b1)
                for nt in range(hd // 8):
                    for c in range(4):
                        row = b.add(b.add(q0, b.mul(w, C(16))), b.add(gq, C((c >> 1) * 8)))
                        col = b.add(C(nt * 8 + (c & 1)), b.mul(tig, C(2)))
                        with b.if_then(b.icmp_unsigned("<", row, T)):
                            gi = b.add(base, b.add(b.mul(z64(row), z64(D)), z64(col)))
                            b.call(red, [b.ptrtoint(b.gep(p["DQ"], [gi], inbounds=True), i64),
                                         b.fmul(b.load(dq[nt][c]), p["scale"])])
            loop(first, ntiles, q_tile)
            for nt in range(hd // 8):
                for c in range(4):
                    row = b.add(b.add(k0, b.mul(w, C(16))), b.add(gq, C((c >> 1) * 8)))
                    col = b.add(C(nt * 8 + (c & 1)), b.mul(tig, C(2)))
                    with b.if_then(b.icmp_unsigned("<", row, T)):
                        gi = b.add(base, b.add(b.mul(z64(row), z64(D)), z64(col)))
                        b.store(b.fmul(b.load(dk[nt][c]), p["scale"]), b.gep(p["DK"], [gi], inbounds=True), align=4)
                        b.store(b.load(dv[nt][c]), b.gep(p["DV"], [gi], inbounds=True), align=4)
        b.ret_void()
        ab.branch(top)

    # D[z, t] = sum over the head of dO * O: a warp a row (the 32 lanes read the row together), 4 rows a block
    fn = ir.Function(m, ir.FunctionType(ir.VoidType(), [G(f32), G(f32), G(f32), i32, i32, i32, i32, i32]),
                     name="apsi_kernel")
    fn.calling_convention = "ptx_kernel"
    DO_, O_, DD_, T_, D_, H_, hd_, rows_ = fn.args
    b = ir.IRBuilder(fn.append_basic_block("e"))
    tid = b.call(g.tid_f, [])
    lane = b.and_(tid, C(31))
    i = b.add(b.mul(b.call(g.ctax_f, []), C(4)), b.lshr(tid, C(5)))
    with b.if_then(b.icmp_unsigned("<", i, rows_)):
        zz, t = b.udiv(i, T_), b.urem(i, T_)
        bs = b.add(b.mul(b.zext(b.udiv(zz, H_), i64), b.mul(b.zext(T_, i64), b.zext(D_, i64))),
                   b.zext(b.add(b.mul(b.urem(zz, H_), hd_), b.mul(t, D_)), i64))
        acc = F(0.0)
        for j in range(4):                            # head sizes up to 128
            col = b.add(lane, C(32 * j))
            ok = b.icmp_unsigned("<", col, hd_)
            gi = b.add(bs, b.zext(b.select(ok, col, C(0)), i64))
            prod = b.fmul(b.load(b.gep(DO_, [gi], inbounds=True), align=4), b.load(b.gep(O_, [gi], inbounds=True), align=4))
            acc = b.fadd(acc, b.select(ok, prod, F(0.0)))
        for k in (16, 8, 4, 2, 1):
            acc = b.fadd(acc, b.call(shfl_f, [C(-1), acc, C(k), C(31)]))
        with b.if_then(b.icmp_unsigned("==", lane, C(0))):
            b.store(acc, b.gep(DD_, [b.zext(i, i64)], inbounds=True), align=4)
    b.ret_void()
    for hd in FLASH_HD:
        for kind in "fbkq":
            build(f"flash_{kind}{hd}", hd, kind)
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
    text = {"main": _main_module, "tc": _tc_module, "tiled": _tiled_module, "tiled_x": lambda: _tiled_module(True),
            "tiled_w": lambda: _tiled_module(True, True),
            "tiled_h": lambda: _tiled_module(h=True),
            "tiled_v": lambda: _tiled_module(True, True, True),
            "tiled_d": lambda: _tiled_module(False, False, True, True),
            "tiled_dw": lambda: _tiled_module(True, True, True, True),
            "flash": _flash_module}[which]()
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
