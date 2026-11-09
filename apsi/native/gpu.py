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
STATS = {"gpu_calls": 0, "cpu_calls": 0}
_NO_GPU_NOTED = False


def gpu(*args, **kwargs):
    """@gpu, @gpu("fast"): run on the GPU(s) when possible, otherwise on all CPU cores."""
    if len(args) == 1 and not kwargs and callable(args[0]) and not isinstance(args[0], str):
        return GpuFunction(args[0], {})
    opts = {a: True for a in args}
    opts.update({k: bool(v) for k, v in kwargs.items()})
    bad = set(opts) - {"fast", "quiet", "strict"}
    if bad:
        raise RuntimeErr(f"@gpu options are fast, quiet, strict; got {', '.join(sorted(bad))}")
    return lambda f: GpuFunction(f, opts)


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

    def __repr__(self):
        return f"<gpu fn {self.name}>"

    def __call__(self, *args, **kwargs):
        if kwargs or len(args) != len(self.fndef.params):
            bound = self.cpu._bind(args, kwargs)
            if bound is None:
                return self.cpu(*args, **kwargs)
            args = bound
        key = []
        for a in args:
            tv = _key_of(a)
            if tv is None:
                return self.cpu(*args)
            key.append(tv)
        key = tuple(key)
        plan = self.plans.get(key)
        if plan is None:
            plan = self._plan(key)
        if isinstance(plan, _Fallback):
            STATS["cpu_calls"] += 1
            return self.cpu(*args)
        STATS["gpu_calls"] += 1
        return plan.run(args)

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
    fd = gf.fndef
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
        prelude, loop, ret = kernel_shape(gf.fndef)
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
        items = [start, stop, step] + ([A.Name(ret, line=loop.line, col=loop.col)] if ret else [])
        helper = A.FnDef(gf.fndef.name + "__gpu_range", [], gf.fndef.params, None,
                         list(prelude) + [A.Return(A.TupleLit(items, line=loop.line, col=loop.col),
                                                   line=loop.line, col=loop.col)],
                         False, [], line=gf.fndef.line, col=gf.fndef.col)
        self.ranger = NativeFunction(_Shim(gf.fn, helper), {"strict": True, "quiet": True})
        self.written = _written_arrays(loop.body, loop.targets[0])
        self.params = [p.name for p in gf.fndef.params]
        self.devices = list(drv.devices)
        self.ptx_key = hash(self.ptx)

    def _slot_name(self, key):
        # slot keys are ("let", id(node), k) or ("param", i); recover the declared name
        kind = key[0]
        if kind == "param":
            return self.gf.fndef.params[key[1]].name
        for n in _walk(self.gf.fndef.body):
            if id(n) == key[1] and isinstance(n, A.Let):
                return n.targets[key[2]]
        return None

    def run(self, args):
        from . import compiler as C            # noqa: F401  (types for the reductions below)
        r = self.ranger(*args)
        start, stop, step = (int(x) for x in r[:3])
        init = r[3] if self.ret_name else None
        if step == 0:
            raise RuntimeErr("range() step must not be zero")
        span = stop - start
        trips = max(0, (span + (step - 1 if step > 0 else step + 1)) // step)
        devs = self.devices
        split = len(devs) > 1 and start == 0 and step == 1 and all(
            self.written.get(n, True) for n, a in zip(self.params, args) if isinstance(_raw(a), np.ndarray)
            and n in self.written)
        if not split or trips < 2 * BLOCK * len(devs):
            devs = devs[:1]
        parts = []
        for j, d in enumerate(devs):
            lo = trips * j // len(devs)
            hi = trips * (j + 1) // len(devs)
            parts.append((d, lo, hi))
        hosts = []
        for a in args:
            raw = _raw(a)
            if isinstance(raw, np.ndarray):
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
                    dp = bufs[bi]
                    bi += 1
                    if name not in self.written:
                        continue
                    if len(running) == 1:
                        d.to_host(dense.ctypes.data, dp, dense.nbytes)
                    else:                                         # only this GPU's rows
                        row = dense.strides[0]
                        d.to_host(dense.ctypes.data + lo * row, dp + lo * row, (hi - lo) * row)
        finally:
            for d, mod, lo, hi, bufs in running:
                d.activate()
                for p in bufs:
                    d.free(p)
        for (raw, dense), name in zip(hosts, self.params):
            if dense is not None and dense is not raw and name in self.written:
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
