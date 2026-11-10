"""@gpu: run a function's `parallel for` on every NVIDIA GPU in the machine, one GPU thread per iteration.

    @gpu
    fn saxpy(a: Float, x, y):
        parallel for i in 0..len(x):
            y[i] = a * x[i] + y[i]

    @gpu
    fn total(x) -> Float:
        var s = 0.0
        parallel for i in 0..len(x):
            s += x[i] * x[i]
        return s

What a GPU function looks like: optional `let`/`var` lines, then one `parallel for`, then optionally `return` of a
`+=` reduction variable. The loop body may use arithmetic, comparisons, if/while, tensor element reads and writes,
sqrt/abs/floor/ceil/min/max/where/clip and other functions that follow the same rules.

* **No CUDA toolkit**: Apsi generates PTX for sm_75 itself (LLVM NVPTX); the NVIDIA driver compiles it for the GPU
  that is present (GTX 1660 / RTX 20xx and newer) and caches the result.
* **Several GPUs**: the iterations are split evenly across all GPUs (or those in `APSI_GPUS=0,2`). Each GPU copies
  back only the rows it wrote, so results are combined without conflicts; reductions are summed.
* **Always runs**: with no NVIDIA GPU, or for a function that does not fit the rules (the note says why), the same
  function runs as native code on all CPU cores. Results are the same.
* Arithmetic is IEEE double/single precision like the CPU (sqrt is correctly rounded). Consumer GPUs run f32 much
  faster than f64; use f32 tensors for speed.
"""
import ctypes
import os
import sys
import threading
import time

import numpy as np

from .. import ast as A
from ..errors import RuntimeErr
from ..units import Quantity
from . import NativeFunction, _key_of, _REPORTS, _Fallback, Specialization
from . import engine as _eng

BLOCK = 256
ALLOWED_DECL = ("llvm.sqrt.", "llvm.fabs.", "llvm.floor.", "llvm.ceil.", "llvm.copysign.", "llvm.minnum.",
                "llvm.maxnum.", "llvm.minimum.", "llvm.maximum.", "llvm.sadd.with.overflow", "llvm.ssub.with.overflow",
                "llvm.smul.with.overflow", "llvm.fptosi.sat.", "llvm.roundeven.", "llvm.nvvm.")
_lock = threading.RLock()
STATS = {"gpu_calls": 0, "cpu_calls": 0, "blas_calls": 0}
_NO_GPU_NOTED = False


def gpu(*args, **kwargs):
    """@gpu, @gpu("fast"): run on the GPU(s) when possible, otherwise on all CPU cores."""
    if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], str):
        return _gwrap(args[0], {})
    opts = {a: True for a in args}
    opts.update({k: bool(v) for k, v in kwargs.items()})
    bad = set(opts) - {"fast", "quiet", "strict"}
    if bad:
        raise RuntimeErr(f"@gpu options are fast, quiet, strict; got {', '.join(sorted(bad))}")
    return lambda f: _gwrap(f, opts)


def _gwrap(f, opts):
    from . import _wrap
    from ..values import Function
    if getattr(f, "_apsi_def", None) is None and isinstance(f, Function):
        return _wrap(f, opts)                    # the interpreter engine: runs as an ordinary function
    return GpuFunction(f, opts)


class GpuArray:
    """A tensor that stays in GPU memory between @gpu calls: to_gpu(x). Kernels read and write it in place, with no
    copy over PCIe per call; to_host(g) (or g.numpy()) copies it back when you need the values. If a call runs on
    the CPU instead (no GPU, small launch), the data moves once and the copy that is out of date is refreshed lazily.
    """

    def __init__(self, host, device=None):
        from . import cuda
        self.ptr = 0
        drv = cuda.driver()
        if drv is None:
            raise RuntimeErr(f"to_gpu(): {cuda.unavailable_reason()}")
        arr = np.ascontiguousarray(_raw(host))
        if _key_of(arr) is None:
            raise RuntimeErr(f"to_gpu(): a {arr.dtype} tensor cannot be used in GPU kernels")
        self.host = arr.copy() if arr is _raw(host) else arr
        self.dev = drv.devices[int(device or 0)]
        self.dev.activate()
        self.ptr = self.dev.alloc(self.host.nbytes)
        self.dev.to_device(self.ptr, self.host.ctypes.data, self.host.nbytes)
        self.host_ok, self.dev_ok = True, True

    shape = property(lambda self: self.host.shape)
    dtype = property(lambda self: self.host.dtype)
    nbytes = property(lambda self: self.host.nbytes)

    def device_ptr(self):
        if not self.dev_ok:
            self.dev.activate()
            self.dev.to_device(self.ptr, self.host.ctypes.data, self.host.nbytes)
            self.dev_ok = True
        return self.ptr

    def numpy(self):
        """The values on the host (copied from the GPU when the GPU has newer ones)."""
        if not self.host_ok:
            self.dev.activate()
            self.dev.sync()
            self.dev.to_host(self.host.ctypes.data, self.ptr, self.host.nbytes)
            self.host_ok = True
        return self.host

    def written_on_gpu(self):
        self.host_ok = False

    def written_on_host(self):
        self.dev_ok = False

    def free(self):
        if getattr(self, "ptr", 0):
            try:
                self.dev.activate()
                self.dev.free(self.ptr)
            except Exception:                              # noqa: BLE001 - driver already gone at exit
                pass
            self.ptr = 0

    __del__ = free

    def __repr__(self):
        return f"<gpu tensor {self.shape} {self.dtype} on {self.dev.name}>"


def to_gpu(x, device=0):
    """Keep a tensor in GPU memory between @gpu calls (see GpuArray). Without a GPU the tensor itself is returned,
    so the same program runs everywhere."""
    if isinstance(x, GpuArray):
        return x
    from . import cuda
    try:
        if cuda.driver() is None:
            return x
    except Exception:                                      # noqa: BLE001
        return x
    return GpuArray(x, device)


def to_host(x):
    """The values of a GPU tensor as an ordinary tensor (a copy); other values unchanged."""
    return x.numpy().copy() if isinstance(x, GpuArray) else x


def _host_args(args):
    return tuple(a.numpy() if isinstance(a, GpuArray) else a for a in args)


# Automatic CPU/GPU choice for small launches (off with APSI_GPU_AUTO=0; @gpu("strict") always uses the GPU).
AUTO = os.environ.get("APSI_GPU_AUTO", "1") != "0"
RECHECK = 64                               # automatic CPU/GPU choice: time the losing side again every RECHECK calls


class _Choice:
    """Measured CPU-or-GPU choice for one size class: warm-up on the GPU, then one timed call on each side, then
    the faster side, with the losing side re-timed every RECHECK calls (a moving best of the last two samples)."""
    __slots__ = ("gpu", "cpu", "calls", "warm")

    def __init__(self):
        self.gpu, self.cpu, self.calls, self.warm = [], [], 0, False

    def next(self):
        self.calls += 1
        if not self.warm or not self.gpu:
            return "gpu"
        if not self.cpu:
            return "cpu"
        g, c = min(self.gpu), min(self.cpu)
        if self.calls % RECHECK == 0:
            return "cpu" if g <= c else "gpu"
        return "gpu" if g <= c else "cpu"

    def record(self, side, seconds):
        if side == "gpu" and not self.warm:
            self.warm = True                     # the first launch loads the kernel: not representative
            return
        xs = self.gpu if side == "gpu" else self.cpu
        xs.append(seconds)
        del xs[:-2]
MIN_TRIPS = int(os.environ.get("APSI_GPU_MIN", "4096"))     # fewer iterations than this: the CPU, without trying


class GpuFunction:
    def __init__(self, fn, options):
        if isinstance(fn, (NativeFunction, GpuFunction)):
            fn = fn.fn
        self.cpu = NativeFunction(fn, {k: v for k, v in options.items() if k != "strict"})
        self.fn = fn
        self.options = options
        self.fndef = self.cpu.fndef
        self.name = self.cpu.name
        self._apsi_def = self.fndef
        self._apsi_free = self.cpu._apsi_free
        self._apsi_e = self.cpu._apsi_e
        self._apsi_native_e = self.cpu._apsi_native_e
        self.plans = {}
        self.warned = False
        try:
            self.gpu_fndef, self.collapsed = collapse(self.fndef)
        except Exception:                                  # noqa: BLE001 - analysis problems: keep the loop as is
            self.gpu_fndef, self.collapsed = self.fndef, None

    def __repr__(self):
        return f"<gpu fn {self.name}>"

    def __call__(self, *args, **kwargs):
        if kwargs or len(args) != len(self.fndef.params):
            bound = self.cpu._bind(args, kwargs)
            if bound is None:
                return self.cpu(*args, **kwargs)
            args = bound
        on_gpu = [a for a in args if isinstance(a, GpuArray)]
        key = []
        for a in args:
            tv = _key_of(a.host if isinstance(a, GpuArray) else a)
            if tv is None:
                return self._on_cpu(args, on_gpu)
            key.append(tv)
        key = tuple(key)
        plan = self.plans.get(key)
        if plan is None:
            plan = self._plan(key)
        if isinstance(plan, _Fallback):
            STATS["cpu_calls"] += 1
            return self._on_cpu(args, on_gpu)
        r = plan.ranger(*plan.range_args(args))
        if not self.options.get("strict") and AUTO:
            # Small launches: the GPU's fixed costs (launch, copies) can exceed the work. Per size class (powers of
            # two of the iteration count) the choice is measured: the first GPU call is a warm-up (it loads the
            # kernel), then one timed GPU call and one timed CPU call decide; every RECHECK calls the side that lost
            # is timed once more, so a choice made under a passing load does not stick. Each call runs once, so
            # side effects never repeat.
            trips = plan.trips(r)
            if trips < MIN_TRIPS and not on_gpu:
                STATS["cpu_calls"] += 1
                return self.cpu(*args)
            if on_gpu:                       # data already on the GPU: moving it to the CPU is never cheaper
                STATS["gpu_calls"] += 1
                return plan.run(args, r)
            c = plan.timing.get(trips.bit_length())
            if c is None:
                c = plan.timing[trips.bit_length()] = _Choice()
            side = c.next()
            t = time.perf_counter()
            if side == "cpu":
                out = self.cpu(*args)
                STATS["cpu_calls"] += 1
            else:
                out = plan.run(args, r)
                STATS["gpu_calls"] += 1
            c.record(side, time.perf_counter() - t)
            return out
        STATS["gpu_calls"] += 1
        return plan.run(args, r)

    def _on_cpu(self, args, on_gpu):
        out = self.cpu(*_host_args(args))
        for a in on_gpu:                       # the CPU may have changed them: refresh the GPU copy when next needed
            a.written_on_host()
        return out

    def _plan(self, key):
        from . import cuda
        with _lock:
            if key in self.plans:
                return self.plans[key]
            try:
                drv = cuda.driver()
                if drv is None:
                    global _NO_GPU_NOTED
                    if not _NO_GPU_NOTED and not self.options.get("quiet") and \
                            os.environ.get("APSI_NATIVE_QUIET", "") not in ("1", "true"):
                        sys.stderr.write(f"apsi: note: {cuda.unavailable_reason()}; @gpu functions run on all CPU "
                                         f"cores\n")
                    _NO_GPU_NOTED = True
                    self.plans[key] = _Fallback("no GPU")
                    return self.plans[key]
                plan = GpuPlan(self, key, drv)
            except Exception as e:
                msg = str(e)
                if self.options.get("strict"):
                    raise RuntimeErr(f"@gpu(strict): {self.name}() cannot run on the GPU: {msg}") from None
                _REPORTS.append((self.name, "gpu: " + msg))
                if not self.warned and not self.options.get("quiet") and \
                        os.environ.get("APSI_NATIVE_QUIET", "") not in ("1", "true"):
                    sys.stderr.write(f"apsi: note: {self.name}() runs on the CPU: {msg}\n")
                self.warned = True
                plan = _Fallback(msg)
            self.plans[key] = plan
            return plan

    def ptx(self, *args):
        """The PTX assembly Apsi generates for these argument types (works without a GPU)."""
        key = tuple(_key_of(a) for a in args)
        return build_kernel(self, key)[0]


# ----------------------------------------------------------------------------- analysis
def kernel_shape(fd):
    """(prelude statements, the parallel for, reduction variable returned or None) or raise Unsupported."""
    from .compiler import Unsupported
    body = fd.body if not fd.is_expr else None
    if body is None:
        raise Unsupported("a GPU function needs a parallel for loop", fd)
    loops = [k for k, s in enumerate(body) if isinstance(s, A.For) and s.parallel]
    if len(loops) != 1:
        raise Unsupported("a GPU function needs exactly one top-level parallel for", fd)
    k = loops[0]
    prelude, loop, rest = body[:k], body[k], body[k + 1:]
    for s in prelude:
        if not isinstance(s, (A.Let, A.Pass, A.Assert)):
            raise Unsupported("only let/var lines may come before the parallel for in a GPU function", s)
    ret = None
    if rest:
        if len(rest) != 1 or not isinstance(rest[0], A.Return) or not isinstance(rest[0].value, A.Name):
            raise Unsupported("after the parallel for, a GPU function may only `return` a reduction variable",
                              rest[0])
        ret = rest[0].value.id
    if len(loop.targets) != 1:
        raise Unsupported("the parallel for needs a single loop variable", loop)
    return prelude, loop, ret


def collapse(fd):
    """GPU kernels get one thread per iteration of the parallel for. When its body is exactly one inner counting loop
    whose iterations are independent (`c[i, j] = …`), both loops become one loop over i×j iterations, so a GPU gets
    millions of threads instead of one per row (OpenMP calls this collapse(2)). Returns (fndef, info) where info is
    None when nothing was changed. The CPU version of the function keeps the original loops."""
    try:
        prelude, loop, ret = kernel_shape(fd)
    except Exception:                                      # noqa: BLE001
        return fd, None
    if len(loop.body) != 1 or not isinstance(loop.body[0], A.For):
        return fd, None
    inner = loop.body[0]
    if inner.parallel or inner.else_body or inner.is_async or len(inner.targets) != 1 \
            or not isinstance(inner.targets[0], str):
        return fd, None
    i, j = loop.targets[0], inner.targets[0]
    if not isinstance(i, str) or i == j:
        return fd, None
    lo_i, hi_i, st_i = _range_parts(loop)
    lo_j, hi_j, st_j = _range_parts(inner)
    if not (_is_one(st_i) and _is_one(st_j)):
        return fd, None
    body = inner.body
    declared = {t for n in _walk(body) if isinstance(n, A.Let) for t in n.targets if isinstance(t, str)}
    declared |= {t for n in _walk(body) if isinstance(n, A.For) for t in n.targets if isinstance(t, str)}
    outer_vars = {t for n in prelude if isinstance(n, A.Let) for t in n.targets if isinstance(t, str)}
    # the inner bounds may not depend on the outer variable or on anything the loop body changes
    used = set()
    for e in (lo_j, hi_j):
        for n in _walk([e]):
            if isinstance(n, A.Name):
                used.add(n.id)
            elif isinstance(n, A.Call) and not (isinstance(n.func, A.Name) and n.func.id in _PURE):
                return fd, None
    if i in used or used & declared:
        return fd, None
    writes = {}                                            # array name -> the index text every access must use
    for n in _walk(body):
        if isinstance(n, (A.Break, A.Continue, A.Return, A.Yield, A.YieldExpr)):
            return fd, None
        if isinstance(n, A.Call) and not (isinstance(n.func, A.Name) and n.func.id in _PURE):
            return fd, None
        if isinstance(n, (A.Assign, A.AugAssign)):
            t = n.target
            if isinstance(n, A.Assign) and n.more:
                return fd, None
            if isinstance(t, A.Name):
                if t.id in declared:
                    continue
                if isinstance(n, A.AugAssign) and n.op == "+" and t.id in outer_vars:
                    continue                               # a reduction: added up across all threads anyway
                return fd, None
            if not (isinstance(t, A.Index) and isinstance(t.obj, A.Name) and isinstance(t.index, A.TupleLit)):
                return fd, None
            names = [x.id for x in t.index.items if isinstance(x, A.Name)]
            if i not in names or j not in names:
                return fd, None
            key = repr(t.index)
            if writes.setdefault(t.obj.id, key) != key:
                return fd, None
    if writes:                                             # written tensors are read only at the same element
        for n in _walk(body):
            if isinstance(n, A.Index) and isinstance(n.obj, A.Name) and n.obj.id in writes \
                    and repr(n.index) != writes[n.obj.id]:
                return fd, None
        for n in _walk(body):                              # a written tensor used whole (e.g. passed on) is unsafe
            if isinstance(n, A.Call):
                for a in n.args:
                    if isinstance(a, A.Name) and a.id in writes:
                        return fd, None
    L, C = loop.line, loop.col

    def nm(x):
        return A.Name(x, line=L, col=C)

    def bi(op, a, b):
        return A.Binary(op, a, b, line=L, col=C)

    def nonneg(e):
        return A.Call(nm("max"), [A.Num(0, "0", None, line=L, col=C), e], [], line=L, col=C)
    t = "__apsi_flat"
    n_inner = nonneg(bi("-", hi_j, lo_j))
    total = bi("*", nonneg(bi("-", hi_i, lo_i)), n_inner)
    lets = [A.Let([i], None, bi("+", lo_i, bi("//", nm(t), n_inner)), False, line=L, col=C),
            A.Let([j], None, bi("+", lo_j, bi("%", nm(t), n_inner)), False, line=L, col=C)]
    flat = A.For([t], A.Range(A.Num(0, "0", None, line=L, col=C), total, False, line=L, col=C),
                 lets + list(body), True, None, None, line=L, col=C)
    new_body = [flat if s is loop else s for s in fd.body]
    nfd = A.FnDef(fd.name, fd.shape_params, fd.params, fd.ret, new_body, fd.is_expr, fd.decorators, fd.is_async,
                  line=fd.line, col=fd.col)
    return nfd, {"outer": i, "inner": j, "n_inner": n_inner, "outer_start": lo_i}


def _is_one(e):
    return isinstance(e, A.Num) and e.value == 1 and e.unit is None


def _range_parts(loop):
    """AST for (start, stop, step) of the parallel for."""
    from .compiler import Unsupported
    it = loop.iter
    one = A.Num(1, "1", None, line=loop.line, col=loop.col)
    if isinstance(it, A.Range):
        stop = A.Binary("+", it.end, one, line=loop.line, col=loop.col) if it.inclusive else it.end
        return it.start, stop, one
    if isinstance(it, A.Call) and isinstance(it.func, A.Name) and it.func.id == "range" and not it.kwargs \
            and 1 <= len(it.args) <= 3:
        a = it.args
        zero = A.Num(0, "0", None, line=loop.line, col=loop.col)
        if len(a) == 1:
            return zero, a[0], one
        if len(a) == 2:
            return a[0], a[1], one
        return a[0], a[1], a[2]
    raise Unsupported("the parallel for of a GPU function must count over a range", loop)


def _written_arrays(body, loopvar):
    """{array name: True when every write is `name[loopvar, ...]` (so rows can be split across GPUs)}."""
    out = {}
    for n in _walk(body):
        tgt = None
        if isinstance(n, (A.Assign, A.AugAssign)):
            tgt = n.target
        if isinstance(tgt, A.Index):
            base = tgt
            while isinstance(base.obj, A.Index):
                base = base.obj
            if isinstance(base.obj, A.Name):
                first = base.index.items[0] if isinstance(base.index, A.TupleLit) else base.index
                rowwise = isinstance(first, A.Name) and first.id == loopvar
                out[base.obj.id] = out.get(base.obj.id, True) and rowwise
        if isinstance(n, A.Call) and not (isinstance(n.func, A.Name) and n.func.id in _PURE):
            for a in n.args:                                   # a called function might write a tensor argument
                if isinstance(a, A.Name):
                    out[a.id] = False
    return out


_PURE = {"len", "sqrt", "abs", "floor", "ceil", "min", "max", "where", "clip", "float", "int", "bool", "round",
         "shape", "ndim"}


def _walk(nodes):
    from .compiler import _walk as w
    return w(nodes)


# ----------------------------------------------------------------------------- the kernel
def build_kernel(gf, key):
    """(ptx text, root spec, module ctx) for these argument types."""
    from . import compiler as C
    import llvmlite.binding as llvm
    fd = gf.gpu_fndef
    _, loop, _ = kernel_shape(fd)
    opts = dict(gf.cpu.options)
    opts["gpu"] = True
    opts.pop("auto", None)
    mc = C.ModuleCtx(opts)
    mc.module.triple = "nvptx64-nvidia-cuda"
    mc.gpu_loop = loop
    pyfn = getattr(gf.fn, "_apsi_e", None) or gf.fn
    resolver = C.Resolver(pyfn, gf._apsi_free, opts)
    targs = [C.TV(C.ARR(t[1], t[2], True) if t[0] == "a" else t, d) for t, d in key]   # device copies are dense
    root = mc.spec(fd, targs, resolver)
    if any(s.globals for s in mc.order):
        raise Unsupported_("global tensors inside a GPU kernel (pass them as arguments)")
    mc.emit_all()
    ir = C.ir
    fn = ir.Function(mc.module, ir.FunctionType(ir.VoidType(), [C.i64t().as_pointer()]), name="apsi_kernel")
    fn.calling_convention = "ptx_kernel"
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    pos = [0]

    def slot():
        p = b.gep(fn.args[0], [ir.Constant(C.i64t(), pos[0])])
        pos[0] += 1
        return b.load(p, align=8)
    from . import _unpack
    vals = [_unpack(b, t, slot, C) for t, _ in [(a.t, a.d) for a in targs]]
    b.call(root.fn, vals)
    b.ret_void()
    mc.seal()
    text = str(mc.module)
    for line in text.splitlines():
        if line.startswith("declare"):
            name = line.split("@", 1)[1].split("(", 1)[0].strip('"')
            if not name.startswith(ALLOWED_DECL):
                what = name.replace("llvm.", "").split(".")[0]
                raise Unsupported_(f"{what}() is not available inside GPU kernels yet")
    _eng.engine()                           # initialises LLVM
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    mod = llvm.parse_assembly(text)
    mod.verify()
    tm = llvm.Target.from_triple("nvptx64-nvidia-cuda").create_target_machine(cpu="sm_75", opt=3)
    mod.data_layout = str(tm.target_data)
    pto = llvm.create_pipeline_tuning_options(speed_level=3)
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod, pb)
    return tm.emit_assembly(mod), root, mc, targs


def Unsupported_(msg):
    from .compiler import Unsupported
    return Unsupported(msg)


class GpuPlan:
    def __init__(self, gf, key, drv):
        from . import compiler as C
        self.gf, self.key, self.drv = gf, key, drv
        prelude, loop, ret = kernel_shape(gf.gpu_fndef)
        self.ptx, self.root, self.mc, self.targs = build_kernel(gf, key)
        self.ret_name = ret
        self.reds = [k for k, _ in self.mc.gpu_reds]
        self.red_types = [t for _, t in self.mc.gpu_reds]
        if ret is not None:
            names = {}
            for k in self.reds:
                names[k] = None
            if not any(self._slot_name(k) == ret for k in self.reds):
                raise Unsupported_(f"'{ret}' is not reduced with += in the parallel for")
        # host-side helper: start, stop, step and the reductions' initial values, computed natively on the CPU
        start, stop, step = _range_parts(loop)
        cl = gf.collapsed
        extra = [cl["n_inner"], cl["outer_start"]] if cl else []
        items = [start, stop, step] + extra + ([A.Name(ret, line=loop.line, col=loop.col)] if ret else [])
        helper = A.FnDef(gf.fndef.name + "__gpu_range", [], gf.fndef.params, None,
                         list(prelude) + [A.Return(A.TupleLit(items, line=loop.line, col=loop.col),
                                                   line=loop.line, col=loop.col)],
                         False, [], line=gf.fndef.line, col=gf.fndef.col)
        self.ranger = NativeFunction(_Shim(gf.fn, helper), {"strict": True, "quiet": True})
        self.written = _written_arrays(loop.body, cl["outer"] if cl else loop.targets[0])
        self.n_extra = len(extra)
        self.params = [p.name for p in gf.fndef.params]
        self.devices = list(drv.devices)
        self.ptx_key = hash(self.ptx)
        self.timing = {}
        self.blas = self._blas_plan(gf, key)

    def _blas_plan(self, gf, key):
        """A matrix-product loop nest over parameters of one float type: (Gemm, a/b/c positions, elem, dims fn)."""
        from ..idioms import match_gemm
        try:
            prelude, loop, ret = kernel_shape(gf.fndef)
            g = match_gemm(loop) if ret is None else None
            if g is None:
                return None
            names = [p.name for p in gf.fndef.params]
            if not all(n in names for n in (g.a, g.b, g.c)):
                return None
            pos = [names.index(n) for n in (g.a, g.b, g.c)]
            ts = [key[k][0] for k in pos]
            if not all(t[0] == "a" and t[2] == 2 for t in ts) or len({t[1] for t in ts}) != 1:
                return None
            from . import cublas
            if ts[0][1] not in cublas.TYPES:
                return None
            L, C = loop.line, loop.col
            helper = A.FnDef(gf.fndef.name + "__gpu_dims", [], gf.fndef.params, None,
                             list(prelude) + [A.Return(A.TupleLit([g.M, g.N, g.K], line=L, col=C), line=L, col=C)],
                             False, [], line=gf.fndef.line, col=gf.fndef.col)
            dims = NativeFunction(_Shim(gf.fn, helper), {"strict": True, "quiet": True})
            return g, pos, ts[0][1], dims
        except Exception:                                  # noqa: BLE001 - no cuBLAS path; the kernel runs
            return None

    def _run_blas_resident(self, a, b, c, g, elem, M, N, K):
        """cuBLAS with some operands already on the GPU: only host operands are copied."""
        from . import cublas
        d = self.devices[0]
        d.activate()
        tmp, ptrs = [], []
        try:
            for x in (a, b, c):
                if isinstance(x, GpuArray):
                    ptrs.append((x.device_ptr(), x.host.shape[1]))
                else:
                    dense = np.ascontiguousarray(x)
                    p = d.alloc(dense.nbytes)
                    tmp.append((p, dense, x))
                    d.to_device(p, dense.ctypes.data, dense.nbytes)
                    ptrs.append((p, dense.shape[1]))
            if isinstance(c, GpuArray) and (c.ptr in (ptrs[0][0], ptrs[1][0])):
                return False
            cublas.gemm(d, elem, g.mode == 0, M, N, K, ptrs[0][0], ptrs[0][1], ptrs[1][0], ptrs[1][1],
                        ptrs[2][0], ptrs[2][1])
            if isinstance(c, GpuArray):
                c.written_on_gpu()
            else:
                d.sync()
                p, dense, orig = tmp[-1]
                d.to_host(dense.ctypes.data, p, dense.nbytes)
                if dense is not orig:
                    orig[...] = dense
        finally:
            for p, _, _ in tmp:
                d.free(p)
        STATS["blas_calls"] = STATS.get("blas_calls", 0) + 1
        return True

    def _run_blas(self, args):
        """c (+)= a @ b with cuBLAS; False when this call cannot use it (nothing changed)."""
        from . import cublas
        g, pos, elem, dims = self.blas
        if cublas.library() is None:
            return False
        a, b, c = (_raw(args[k]) for k in pos)
        if not all(isinstance(x, (np.ndarray, GpuArray)) for x in (a, b, c)):
            return False
        res = [x for x in (a, b, c) if isinstance(x, GpuArray)]
        if res and any(x.dev is not self.devices[0] for x in res):
            return False
        M, N, K = (int(v) for v in dims(*self.range_args(args)))
        shp = [x.shape for x in (a, b, c)]
        if min(M, N, K) <= 0 or M > shp[0][0] or K > shp[0][1] or K > shp[1][0] or N > shp[1][1] \
                or M > shp[2][0] or N > shp[2][1]:
            return False
        if res:
            return self._run_blas_resident(a, b, c, g, elem, M, N, K)
        if min(M, N, K) <= 0 or M > a.shape[0] or K > a.shape[1] or K > b.shape[0] or N > b.shape[1] \
                or M > c.shape[0] or N > c.shape[1] or np.may_share_memory(c, a) or np.may_share_memory(c, b):
            return False
        d = self.devices[0]
        d.activate()
        dense = [np.ascontiguousarray(x) for x in (a, b, c)]
        bufs = []
        try:
            for x in dense:
                p = d.alloc(x.nbytes)
                bufs.append(p)
                d.to_device(p, x.ctypes.data, x.nbytes)
            cublas.gemm(d, elem, g.mode == 0, M, N, K, bufs[0], dense[0].shape[1], bufs[1], dense[1].shape[1],
                        bufs[2], dense[2].shape[1])
            d.sync()
            d.to_host(dense[2].ctypes.data, bufs[2], dense[2].nbytes)
        finally:
            for p in bufs:
                d.free(p)
        if dense[2] is not c:
            c[...] = dense[2]
        STATS["blas_calls"] = STATS.get("blas_calls", 0) + 1
        return True

    def _slot_name(self, key):
        # slot keys are ("let", id(node), k) or ("param", i); recover the declared name
        kind = key[0]
        if kind == "param":
            return self.gf.fndef.params[key[1]].name
        for n in _walk(self.gf.gpu_fndef.body):
            if id(n) == key[1] and isinstance(n, A.Let):
                return n.targets[key[2]]
        return None

    def range_args(self, args):
        """Arguments for the host-side helpers (loop bounds, matrix sizes): GPU tensors by their host copy, which
        has the right shape; refreshed from the GPU only if the bounds read tensor values."""
        if not any(isinstance(a, GpuArray) for a in args):
            return args
        reads = getattr(self, "_reads", None)
        if reads is None:
            prelude, loop, _ = kernel_shape(self.gf.gpu_fndef)
            reads = set()
            for n in _walk(list(prelude) + [loop.iter]):
                if isinstance(n, A.Index) and isinstance(n.obj, A.Name):
                    reads.add(n.obj.id)
            self._reads = reads
        return tuple((a.numpy() if p in reads else a.host) if isinstance(a, GpuArray) else a
                     for a, p in zip(args, self.params))

    def trips(self, r):
        start, stop, step = (int(x) for x in r[:3])
        if step == 0:
            return 0
        return max(0, (stop - start + (step - 1 if step > 0 else step + 1)) // step)

    def run(self, args, r=None):
        from . import compiler as C            # noqa: F401  (types for the reductions below)
        if self.blas is not None:
            try:
                if self._run_blas(args):
                    return None
            except Exception as e:                         # noqa: BLE001 - the generated kernel instead
                _REPORTS.append((self.gf.name, "cuBLAS: " + str(e)))
                self.blas = None
        if r is None:
            r = self.ranger(*self.range_args(args))
        start, stop, step = (int(x) for x in r[:3])
        init = r[3 + self.n_extra] if self.ret_name else None
        unit = 1                                      # iterations per row of the written tensors
        if self.n_extra:
            unit, outer_start = max(1, int(r[3])), int(r[4])
            if outer_start != 0:
                unit = -1                             # rows do not start at 0: no splitting across GPUs
        if step == 0:
            raise RuntimeErr("range() step must not be zero")
        span = stop - start
        trips = max(0, (span + (step - 1 if step > 0 else step + 1)) // step)
        devs = self.devices
        split = len(devs) > 1 and start == 0 and step == 1 and unit > 0 and all(
            self.written.get(n, True) for n, a in zip(self.params, args) if isinstance(_raw(a), np.ndarray)
            and n in self.written)
        resident = [_raw(a) for a in args if isinstance(_raw(a), GpuArray)]
        if resident:                                   # tensors kept on a GPU: run there
            devs = [resident[0].dev]
        elif not split or trips < 2 * BLOCK * len(devs):
            devs = devs[:1]
        parts = []
        rows = trips // unit if unit > 0 else trips
        for j, d in enumerate(devs):
            if len(devs) > 1:                         # whole rows per GPU: lo/hi below count iterations
                lo = rows * j // len(devs) * unit
                hi = (rows * (j + 1) // len(devs) * unit) if j < len(devs) - 1 else trips
            else:
                lo, hi = 0, trips
            parts.append((d, lo, hi))
        hosts = []
        for a in args:
            raw = _raw(a)
            if isinstance(raw, GpuArray):
                if raw.dev is not devs[0]:
                    raw = raw.numpy()                  # on another GPU: through the host this once
                    hosts.append((raw, np.ascontiguousarray(raw)))
                else:
                    hosts.append((raw, "gpu"))
            elif isinstance(raw, np.ndarray):
                dense = np.ascontiguousarray(raw)
                hosts.append((raw, dense))
            else:
                hosts.append((raw, None))
        totals = [0] * len(self.reds)
        running = []
        try:
            for d, lo, hi in parts:
                d.activate()
                mod, fn = d.load(self.ptx_key, self.ptx)
                bufs = []
                slots = []
                for (t, _), (raw, dense) in zip(self.key, hosts):
                    if dense is None:
                        if t[0] in ("f", "s"):
                            slots.append(ctypes.c_int64.from_buffer(ctypes.c_double(float(raw))).value)
                        else:
                            slots.append(int(raw))
                        continue
                    if isinstance(dense, str):                     # already in this GPU's memory
                        h = raw.host
                        slots += [raw.device_ptr(), 0] + list(h.shape) + [s // h.itemsize for s in h.strides]
                        continue
                    dp = d.alloc(dense.nbytes)
                    bufs.append(dp)
                    d.to_device(dp, dense.ctypes.data, dense.nbytes)
                    slots += [dp, 0] + list(dense.shape) + [s // dense.itemsize for s in dense.strides]
                arg = np.array(slots or [0], dtype=np.int64)
                dargs = d.alloc(arg.nbytes)
                bufs.append(dargs)
                d.to_device(dargs, arg.ctypes.data, arg.nbytes)
                kr, _ = d.global_ptr(mod, "apsi_krange")
                krange = np.array([lo, hi], dtype=np.int64)
                d.to_device(kr, krange.ctypes.data, 16)
                zeros = np.zeros(16, dtype=np.int64)
                for g in ("apsi_err", "apsi_red"):
                    try:
                        gp, size = d.global_ptr(mod, g)
                        d.to_device(gp, zeros.ctypes.data, min(size, 128))
                    except Exception:
                        pass
                n = hi - lo
                if n > 0:
                    d.launch(fn, (n + BLOCK - 1) // BLOCK, BLOCK, [ctypes.c_uint64(dargs)])
                running.append((d, mod, lo, hi, bufs))
            for d, mod, lo, hi, bufs in running:
                d.sync()
                err = np.zeros(8, dtype=np.int64)
                gp, _ = d.global_ptr(mod, "apsi_err")
                d.to_host(err.ctypes.data, gp, 64)
                if err[0]:
                    slots = _eng.error_slots()
                    for i in range(8):
                        slots[i] = int(err[i])
                    raise Specialization._error(_ErrCtx(self), slots)
                if self.reds:
                    red = np.zeros(16, dtype=np.int64)
                    gp, _ = d.global_ptr(mod, "apsi_red")
                    d.to_host(red.ctypes.data, gp, 128)
                    for j, t in enumerate(self.red_types):
                        totals[j] += int(red[j]) if t == C.I64 else float(red[j:j + 1].view(np.float64)[0])
                bi = 0
                for (t, _), (raw, dense), name in zip(self.key, hosts, self.params):
                    if dense is None:
                        continue
                    if isinstance(dense, str):
                        if name in self.written:
                            raw.written_on_gpu()               # the values stay on the GPU
                        continue
                    dp = bufs[bi]
                    bi += 1
                    if name not in self.written:
                        continue
                    if len(running) == 1:
                        d.to_host(dense.ctypes.data, dp, dense.nbytes)
                    else:                                         # only this GPU's rows
                        row = dense.strides[0]
                        rlo, rhi = lo // max(unit, 1), hi // max(unit, 1)
                        d.to_host(dense.ctypes.data + rlo * row, dp + rlo * row, (rhi - rlo) * row)
        finally:
            for d, mod, lo, hi, bufs in running:
                d.activate()
                for p in bufs:
                    d.free(p)
        for (raw, dense), name in zip(hosts, self.params):
            if dense is not None and not isinstance(dense, str) and dense is not raw and name in self.written:
                raw[...] = dense
        if self.ret_name is None:
            return None
        j = [self._slot_name(k) for k in self.reds].index(self.ret_name)
        val = init + totals[j]
        d = self.root.ret.d
        return Quantity(val, d) if not d.dimensionless else val


class _ErrCtx:
    """Lets Specialization._error format a GPU error with this plan's tables."""

    def __init__(self, plan):
        self.fn_names = plan.mc.fn_names
        self.assert_msgs = plan.mc.assert_msgs
        self.nf = plan.gf.cpu


class _Shim:
    """A function-like object carrying a synthesised definition (for the host-side range helper)."""

    def __init__(self, fn, fndef):
        self._apsi_def = fndef
        self._apsi_e = getattr(fn, "_apsi_e", None) or fn
        self._apsi_free = getattr(fn, "_apsi_free", None)
        self.name = fndef.name

    def __call__(self, *a, **k):
        raise RuntimeErr(f"{self.name}: internal helper cannot run on the bytecode engine")


def _raw(a):
    return a.v if isinstance(a, Quantity) else a


def info():
    from . import cuda
    drv = cuda.driver()
    if drv is None:
        return {"available": False, "reason": cuda.unavailable_reason(), "devices": []}
    return {"available": True, "devices": [d.describe() for d in drv.devices],
            "skipped": [d.describe() for d in drv.skipped]}
