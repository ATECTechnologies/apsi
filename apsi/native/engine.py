"""Machine-code engine for Apsi native functions.

One LLVM MCJIT per process turns the IR produced by `apsi.native.compiler` into machine code for the exact CPU
it runs on (AVX2 / AVX-512 / NEON are used when the CPU has them), at -O3 with loop and SLP vectorization.

* **Machine-code cache** - every compiled module is stored in ``~/.apsi/cache/native`` as an object file signed
  with HMAC-SHA256 (the same per-user key as the compile cache). A signature that does not verify is never
  loaded. The key covers the IR, the CPU name and features, LLVM and Apsi versions, so a cached object is only
  reused on a machine where it is valid. ``APSI_NO_CACHE=1`` turns it off.
* **Parallel runtime** - ``apsi_parallel`` splits an iteration range into a fixed number of chunks and runs them on
  a pool of OS threads. Each thread calls a small native runner once and claims chunks with an atomic counter, so a
  parallel region costs one thread wake-up per core, not one Python call per chunk. Chunk boundaries do not depend
  on the number of threads, so parallel reductions give bit-identical results on every machine.
* **Errors** - native code never raises. It records an error code, the source line/column and two operands in
  ``apsi_err`` and returns; the Python side turns that into the usual Apsi error.
"""
import ctypes
import ctypes.util
import hashlib
import hmac
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

NCHUNKS = 256                     # chunks per parallel region (fixed -> reproducible reductions)
# parallel runtime state: one 64-byte cache line per field (no false sharing between threads)
P_GEN, P_INF, P_SPIN, P_BUSY, P_NTH, P_CNT = 0, 8, 16, 24, 32, 40
P_FN, P_CTX, P_LO, P_HI, P_N = 48, 49, 50, 51, 52
P_STOP = 56
P_SLOTS = {"P_STOP": P_STOP, "P_GEN": P_GEN, "P_INF": P_INF, "P_SPIN": P_SPIN, "P_BUSY": P_BUSY, "P_NTH": P_NTH, "P_CNT": P_CNT,
           "P_FN": P_FN, "P_CTX": P_CTX, "P_LO": P_LO, "P_HI": P_HI, "P_N": P_N}
SPIN_LIMIT = int(os.environ.get("APSI_SPIN", "200000"))   # idle spins (~2-10 ms) before a worker sleeps
_PSTATE = None
OPT_VERSION = "apsi-native-3"     # bump when the optimisation pipeline changes

_lock = threading.RLock()
_engine = None
_error = None


class NativeUnavailable(Exception):
    pass


# ----------------------------------------------------------------------------- threads
def default_threads():
    env = os.environ.get("APSI_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


class _Pool:
    """Worker threads for parallel regions. Each runs the native worker loop with the GIL released; when that loop
    has been idle for a while it returns and the thread sleeps on a semaphore until the dispatcher wakes it."""

    def __init__(self):
        self.threads = default_threads()
        self.pool = None
        self.local = threading.local()
        self.sem = threading.Semaphore(0)
        self.workers = []
        self.pending = 0

    def wake(self, k):
        with _lock:
            want = max(0, self.threads - 1)
            while len(self.workers) < want:                       # start workers on first use
                t = threading.Thread(target=self._run, name=f"apsi-native-{len(self.workers)}", daemon=True)
                self.workers.append(t)
                t.start()
            k = min(int(k), want) - self.pending
            for _ in range(max(0, k)):
                self.pending += 1
                self.sem.release()

    def _run(self):
        while True:
            self.sem.acquire()
            with _lock:
                self.pending -= 1
            eng = _engine
            if eng is not None and not _STOPPING:
                eng.worker(SPIN_LIMIT)

    @staticmethod
    def shutdown():
        """At exit: make spinning workers leave machine code before the interpreter (and the JIT) goes away."""
        global _STOPPING
        _STOPPING = True
        if _PSTATE is None:
            return
        import time
        _PSTATE[P_STOP] = 1
        deadline = time.monotonic() + 2.0
        while _PSTATE[P_SPIN] > 0 and time.monotonic() < deadline:
            time.sleep(0.0005)

    def set_threads(self, n):
        n = max(1, int(n))
        with _lock:
            self.threads = n
            if _PSTATE is not None:
                _PSTATE[P_NTH] = n

    def executor(self):
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=max(1, self.threads - 1),
                                           thread_name_prefix="apsi-native")
        return self.pool


_STOPPING = False
POOL = _Pool()
import atexit                                                   # noqa: E402
atexit.register(_Pool.shutdown)


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self):
        try:
            import llvmlite
            import llvmlite.binding as llvm
        except ImportError as e:              # pragma: no cover - depends on the install
            raise NativeUnavailable("the native compiler needs llvmlite (pip install llvmlite)") from e
        self.llvm = llvm
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        try:
            llvm.initialize_native_asmparser()      # inline assembly (the ARM64 spin-wait hint) needs the parser
        except Exception:                           # noqa: BLE001 - older llvmlite: x86 needs no parser
            pass
        self.llvm_version = ".".join(map(str, llvm.llvm_version_info))
        self.llvmlite_version = llvmlite.__version__
        self.triple = llvm.get_process_triple()
        self.cpu = llvm.get_host_cpu_name()
        try:
            self.features = llvm.get_host_cpu_features().flatten()
        except RuntimeError:                 # some VMs hide the feature list
            self.features = ""
        target = llvm.Target.from_triple(self.triple)
        self.tm = target.create_target_machine(cpu=self.cpu, features=self.features, opt=3,
                                               reloc="default", codemodel="jitdefault", jit=True)
        self.data_layout = str(self.tm.target_data)
        backing = llvm.parse_assembly("")
        backing.triple = self.triple
        self.ee = llvm.create_mcjit_compiler(backing, self.tm)
        self._cache_dir = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.compile_seconds = 0.0
        self.ee.set_object_cache(self._notify, self._getbuffer)
        self._callbacks = []
        self._register_symbols()
        self._build_runtime()

    # ---------------------------------------------------------------- symbols
    def _register_symbols(self):
        global _error
        llvm = self.llvm
        _error = (ctypes.c_int64 * 8)()
        llvm.add_symbol("apsi_err", ctypes.addressof(_error))
        libs = []
        if sys.platform == "win32":
            for name in ("ucrtbase", "msvcrt"):
                try:
                    libs.append(ctypes.CDLL(name))
                except OSError:
                    pass
        else:
            for name in (ctypes.util.find_library("m"), ctypes.util.find_library("c"), None):
                try:
                    libs.append(ctypes.CDLL(name))
                except OSError:
                    pass
        names = ["malloc", "calloc", "free", "memset", "memcpy", "memmove",
                 "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "expm1",
                 "log", "log10", "log2", "log1p", "pow", "fmod", "cbrt", "hypot", "erf", "erfc",
                 "sinf", "cosf", "tanf", "asinf", "acosf", "atanf", "atan2f", "sinhf", "coshf", "tanhf", "expf",
                 "logf", "log10f", "log2f", "powf", "fmodf", "cbrtf", "exp2", "exp2f", "floor", "ceil", "trunc",
                 "round", "nearbyint", "rint", "fmin", "fmax", "floorf", "ceilf", "truncf", "roundf"]
        for n in names:
            for lib in libs:
                try:
                    fn = getattr(lib, n)
                except AttributeError:
                    continue
                llvm.add_symbol(n, ctypes.cast(fn, ctypes.c_void_p).value)
                break
        if sys.platform == "win32":
            # stack frames over 4 KB (e.g. several parallel reductions) call the stack probe on Windows x64
            try:
                nt = ctypes.WinDLL("ntdll")
                for n in ("__chkstk", "_chkstk"):
                    try:
                        llvm.add_symbol(n, ctypes.cast(getattr(nt, n), ctypes.c_void_p).value)
                    except AttributeError:
                        pass
            except OSError:
                pass
        # parallel runtime state (one cache line per field) and the wake-up callback for sleeping workers
        global _PSTATE
        _PSTATE = (ctypes.c_int64 * 64)()
        _PSTATE[P_NTH * 1] = POOL.threads
        llvm.add_symbol("apsi_pstate", ctypes.addressof(_PSTATE))
        wcb = ctypes.CFUNCTYPE(None, ctypes.c_int64)(_wake)
        self._callbacks.append(wcb)
        llvm.add_symbol("apsi_wake", ctypes.cast(wcb, ctypes.c_void_p).value)
        mcb = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(_matmul_cb)
        self._callbacks.append(mcb)
        llvm.add_symbol("apsi_matmul", ctypes.cast(mcb, ctypes.c_void_p).value)
        gcb = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)(_gemm_cb)
        self._callbacks.append(gcb)
        llvm.add_symbol("apsi_gemm", ctypes.cast(gcb, ctypes.c_void_p).value)
        pcb = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_void_p)(_print_cb)
        self._callbacks.append(pcb)
        llvm.add_symbol("apsi_print", ctypes.cast(pcb, ctypes.c_void_p).value)

    def _build_runtime(self):
        """The chunk runner every worker thread executes (claims chunks with an atomic counter)."""
        if self.triple.startswith(("x86_64", "i686", "i386")):
            pause_decl, pause = "declare void @llvm.x86.sse2.pause()", "call void @llvm.x86.sse2.pause()"
        elif self.triple.startswith(("aarch64", "arm64")):
            pause_decl, pause = "declare void @llvm.aarch64.hint(i32)", "call void @llvm.aarch64.hint(i32 1)"
        else:
            pause_decl, pause = "", ""
        ir_text = r"""
@apsi_err = external global [8 x i64]
@apsi_pstate = external global [64 x i64]
declare void @apsi_wake(i64)
PAUSE_DECL

define void @apsi_runner(i8* %chunk, i8* %ctx, i64 %lo, i64 %hi, i64 %n, i64* %counter) {
entry:
  %fn = bitcast i8* %chunk to void (i8*, i64, i64, i64)*
  %len = sub i64 %hi, %lo
  %base = sdiv i64 %len, %n
  %rem = srem i64 %len, %n
  ; claim several consecutive chunks per atomic operation (fewer contended cache-line transfers); chunk
  ; boundaries never change, so reductions stay bit-identical
  %nthp = getelementptr [64 x i64], [64 x i64]* @apsi_pstate, i64 0, i64 P_NTH
  %nth = load atomic i64, i64* %nthp monotonic, align 8
  %pos = icmp sgt i64 %nth, 0
  %nthv = select i1 %pos, i64 %nth, i64 1
  %den = mul i64 %nthv, 8
  %q = sdiv i64 %n, %den
  %big = icmp sgt i64 %q, 1
  %step = select i1 %big, i64 %q, i64 1
  br label %claim
claim:
  %first = atomicrmw add i64* %counter, i64 %step seq_cst
  %done = icmp sge i64 %first, %n
  br i1 %done, label %out, label %batch
batch:
  %end0 = add i64 %first, %step
  %over = icmp sgt i64 %end0, %n
  %end = select i1 %over, i64 %n, i64 %end0
  br label %loop
loop:
  %id = phi i64 [ %first, %batch ], [ %next, %work ]
  %more = icmp slt i64 %id, %end
  br i1 %more, label %check, label %claim
check:
  %ep = getelementptr [8 x i64], [8 x i64]* @apsi_err, i64 0, i64 0
  %e = load atomic i64, i64* %ep monotonic, align 8
  %bad = icmp ne i64 %e, 0
  br i1 %bad, label %out, label %work
work:
  %lt = icmp slt i64 %id, %rem
  %extra = select i1 %lt, i64 %id, i64 %rem
  %off = mul i64 %id, %base
  %a0 = add i64 %lo, %off
  %a = add i64 %a0, %extra
  %one = zext i1 %lt to i64
  %sz = add i64 %base, %one
  %b = add i64 %a, %sz
  call void %fn(i8* %ctx, i64 %a, i64 %b, i64 %id)
  %next = add i64 %id, 1
  br label %loop
out:
  ret void
}

define internal i64* @slot(i64 %k) {
  %p = getelementptr [64 x i64], [64 x i64]* @apsi_pstate, i64 0, i64 %k
  ret i64* %p
}

; Spin until no worker is inside a job.
define internal void @wait_idle() {
entry:
  %inf = call i64* @slot(i64 P_INF)
  br label %loop
loop:
  %v = load atomic i64, i64* %inf seq_cst, align 8
  %z = icmp eq i64 %v, 0
  br i1 %z, label %out, label %spin
spin:
  PAUSE
  br label %loop
out:
  ret void
}

; apsi_parallel(chunk, ctx, lo, hi, nchunks): run on this thread and every worker.
define void @apsi_parallel(i8* %chunk, i8* %ctx, i64 %lo, i64 %hi, i64 %n) {
entry:
  %busy = call i64* @slot(i64 P_BUSY)
  %nthp = call i64* @slot(i64 P_NTH)
  %nth = load atomic i64, i64* %nthp monotonic, align 8
  %few = icmp sle i64 %nth, 1
  %small = icmp sle i64 %n, 1
  %serial = or i1 %few, %small
  br i1 %serial, label %inline, label %claim
claim:
  %cx = cmpxchg i64* %busy, i64 0, i64 1 seq_cst seq_cst
  %got = extractvalue { i64, i1 } %cx, 1
  br i1 %got, label %publish, label %inline
inline:                                    ; nested region, one thread, or another region is running
  %lc = alloca i64
  store i64 0, i64* %lc
  call void @apsi_runner(i8* %chunk, i8* %ctx, i64 %lo, i64 %hi, i64 %n, i64* %lc)
  ret void
publish:
  %genp = call i64* @slot(i64 P_GEN)
  %g = load atomic i64, i64* %genp seq_cst, align 8
  %godd = add i64 %g, 1
  store atomic i64 %godd, i64* %genp seq_cst, align 8          ; odd: the job is being written
  call void @wait_idle()
  %p1 = call i64* @slot(i64 P_FN)
  %fni = ptrtoint i8* %chunk to i64
  store i64 %fni, i64* %p1
  %p2 = call i64* @slot(i64 P_CTX)
  %cti = ptrtoint i8* %ctx to i64
  store i64 %cti, i64* %p2
  %p3 = call i64* @slot(i64 P_LO)
  store i64 %lo, i64* %p3
  %p4 = call i64* @slot(i64 P_HI)
  store i64 %hi, i64* %p4
  %p5 = call i64* @slot(i64 P_N)
  store i64 %n, i64* %p5
  %cnt = call i64* @slot(i64 P_CNT)
  store atomic i64 0, i64* %cnt seq_cst, align 8
  %gnew = add i64 %g, 2
  store atomic i64 %gnew, i64* %genp seq_cst, align 8          ; even: published
  %spp = call i64* @slot(i64 P_SPIN)
  %sp = load atomic i64, i64* %spp seq_cst, align 8
  %workers = sub i64 %nth, 1
  %asleep = sub i64 %workers, %sp
  %wake = icmp sgt i64 %asleep, 0
  br i1 %wake, label %dowake, label %run
dowake:
  call void @apsi_wake(i64 %asleep)
  br label %run
run:
  call void @apsi_runner(i8* %chunk, i8* %ctx, i64 %lo, i64 %hi, i64 %n, i64* %cnt)
  ; close the job before its context (on this thread's stack) goes away: after an error some chunks are still
  ; unclaimed, and a worker joining late must find none left to claim
  store atomic i64 4611686018427387904, i64* %cnt seq_cst, align 8
  call void @wait_idle()
  store atomic i64 0, i64* %busy seq_cst, align 8
  ret void
}

; apsi_worker(limit): a worker thread spins on the generation counter, runs every job it sees, and returns to
; Python (which then sleeps on a semaphore) after `limit` idle spins.
define void @apsi_worker(i64 %limit) {
entry:
  %spp = call i64* @slot(i64 P_SPIN)
  %genp = call i64* @slot(i64 P_GEN)
  %inf = call i64* @slot(i64 P_INF)
  %cnt = call i64* @slot(i64 P_CNT)
  atomicrmw add i64* %spp, i64 1 seq_cst
  ; join the job that is running right now, if any (joining a finished one is harmless: no chunks are left)
  %start_even = add i64 0, -2
  br label %loop
loop:
  %last = phi i64 [ %start_even, %entry ], [ %last, %idle ], [ %g, %finished ], [ %last, %retry ]
  %i = phi i64 [ 0, %entry ], [ %inext, %idle ], [ 0, %finished ], [ 0, %retry ]
  %g = load atomic i64, i64* %genp seq_cst, align 8
  %odd = and i64 %g, 1
  %ready = icmp eq i64 %odd, 0
  %new = icmp ne i64 %g, %last
  %go = and i1 %ready, %new
  br i1 %go, label %enter, label %idle
idle:
  %inext = add i64 %i, 1
  PAUSE
  %tired = icmp sge i64 %inext, %limit
  %stopp = call i64* @slot(i64 P_STOP)
  %stop = load atomic i64, i64* %stopp monotonic, align 8
  %halt = icmp ne i64 %stop, 0
  %quit = or i1 %tired, %halt
  br i1 %quit, label %leave, label %loop
enter:
  atomicrmw add i64* %inf, i64 1 seq_cst
  %g2 = load atomic i64, i64* %genp seq_cst, align 8
  %same = icmp eq i64 %g2, %g
  br i1 %same, label %work, label %retry
retry:
  atomicrmw sub i64* %inf, i64 1 seq_cst
  br label %loop
work:
  %f1 = call i64* @slot(i64 P_FN)
  %fv = load i64, i64* %f1
  %fp = inttoptr i64 %fv to i8*
  %c1 = call i64* @slot(i64 P_CTX)
  %cv = load i64, i64* %c1
  %cp = inttoptr i64 %cv to i8*
  %l1 = call i64* @slot(i64 P_LO)
  %lv = load i64, i64* %l1
  %h1 = call i64* @slot(i64 P_HI)
  %hv = load i64, i64* %h1
  %n1 = call i64* @slot(i64 P_N)
  %nv = load i64, i64* %n1
  call void @apsi_runner(i8* %fp, i8* %cp, i64 %lv, i64 %hv, i64 %nv, i64* %cnt)
  atomicrmw sub i64* %inf, i64 1 seq_cst
  br label %finished
finished:
  br label %loop
leave:
  atomicrmw sub i64* %spp, i64 1 seq_cst
  ret void
}
"""
        for k, v in P_SLOTS.items():
            ir_text = ir_text.replace(k, str(v))
        ir_text = ir_text.replace("PAUSE_DECL", pause_decl).replace("PAUSE", pause)
        addrs = self.compile(ir_text, ["apsi_runner", "apsi_parallel", "apsi_worker"], tag="runtime")
        self.runner = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_void_p)(addrs["apsi_runner"])
        self.worker = ctypes.CFUNCTYPE(None, ctypes.c_int64)(addrs["apsi_worker"])

    # ---------------------------------------------------------------- cache
    def cache_dir(self):
        if self._cache_dir is None:
            from ..security import config_dir
            self._cache_dir = os.path.join(config_dir(), "cache", "native")
        return self._cache_dir

    @staticmethod
    def _cache_on():
        return os.environ.get("APSI_NO_CACHE", "").strip() not in ("1", "true", "yes")

    @staticmethod
    def _secret():
        from ..data import _cache_key
        return _cache_key()

    def _path(self, key):
        return os.path.join(self.cache_dir(), key[:2], key + ".o")

    def _getbuffer(self, module):
        key = module.name
        if not key.startswith("apsi_") or not self._cache_on():
            return None
        path = self._path(key[5:])
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            return None
        sig, body = blob[:32], blob[32:]
        if not hmac.compare_digest(sig, hmac.new(self._secret(), body, hashlib.sha256).digest()):
            try:
                os.remove(path)                       # tampered or from another user: never loaded
            except OSError:
                pass
            return None
        self.cache_hits += 1
        return body

    def _notify(self, module, buf):
        key = module.name
        if not key.startswith("apsi_") or not self._cache_on():
            return
        path = self._path(key[5:])
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".{os.getpid()}.tmp"
            body = bytes(buf)
            with open(tmp, "wb") as fh:
                fh.write(hmac.new(self._secret(), body, hashlib.sha256).digest() + body)
            os.replace(tmp, path)
        except OSError:
            pass

    def key(self, ir_text):
        from .. import __version__
        h = hashlib.sha256()
        for part in (OPT_VERSION, __version__, self.llvm_version, self.triple, self.cpu, self.features, ir_text):
            h.update(part.encode())
            h.update(b"\0")
        return h.hexdigest()

    # ---------------------------------------------------------------- compile
    def optimize(self, mod):
        llvm = self.llvm
        pto = llvm.create_pipeline_tuning_options(speed_level=3)
        pto.loop_vectorization = True
        pto.slp_vectorization = True
        pto.loop_unrolling = True
        pto.loop_interleaving = True
        pb = llvm.create_pass_builder(self.tm, pto)
        mpm = pb.getModulePassManager()
        mpm.run(mod, pb)

    def compile(self, ir_text, entries, tag="fn"):
        """Compile LLVM IR text; return {entry name: address}."""
        import time
        llvm = self.llvm
        t0 = time.perf_counter()
        key = self.key(ir_text)
        with _lock:
            mod = llvm.parse_assembly(ir_text)
            mod.triple = self.triple
            mod.data_layout = self.data_layout
            mod.name = "apsi_" + key
            mod.verify()
            cached = self._cache_on() and os.path.exists(self._path(key))
            if not cached:
                self.cache_misses += 1
                self.optimize(mod)
            self.ee.add_module(mod)
            self.ee.finalize_object()
            addrs = {e: self.ee.get_function_address(e) for e in entries}
        self.compile_seconds += time.perf_counter() - t0
        for e, a in addrs.items():
            if not a:
                raise RuntimeError(f"native: entry point {e} was not produced")
        return addrs

    def assembly(self, ir_text):
        mod = self.llvm.parse_assembly(ir_text)
        mod.triple = self.triple
        mod.data_layout = self.data_layout
        self.optimize(mod)
        return str(mod), self.tm.emit_assembly(mod)


def engine():
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = Engine()
    return _engine


def available():
    try:
        engine()
        return True
    except NativeUnavailable:
        return False


def error_slots():
    engine()
    return _error


# ----------------------------------------------------------------------------- callbacks from native code
def _wake(k):
    try:
        POOL.wake(k)
    except BaseException as e:                                   # never let an exception cross into native code
        sys.stderr.write(f"apsi native: could not start worker threads: {e}\n")


def _dispatch(chunk, ctx, lo, hi, n):
    """apsi_parallel(chunk, ctx, lo, hi, nchunks): run chunks [0, n) of [lo, hi) on all cores."""
    try:
        eng = _engine
        counter = ctypes.c_int64(0)
        cptr = ctypes.addressof(counter)
        threads = POOL.threads
        if threads <= 1 or n <= 1 or getattr(POOL.local, "busy", False):
            eng.runner(chunk, ctx, lo, hi, n, cptr)          # nested region or one core: run here
            return

        def work():
            POOL.local.busy = True
            try:
                eng.runner(chunk, ctx, lo, hi, n, cptr)
            finally:
                POOL.local.busy = False
        ex = POOL.executor()
        futs = [ex.submit(work) for _ in range(min(threads - 1, n - 1))]
        work()
        for f in futs:
            if not f.cancel():                               # not started yet -> nothing left for it to do
                f.result()
    except BaseException as e:                               # never let an exception cross into native code
        if _error is not None and _error[0] == 0:
            _error[0] = 99
        sys.stderr.write(f"apsi native: parallel runtime error: {e}\n")


PRINT_SITES = []


def _as_numpy(desc, nd, dtype):
    """A NumPy view of a native tensor descriptor {data*, base*, shape[nd], strides[nd]} (strides in elements)."""
    import numpy as np
    w = ctypes.cast(desc, ctypes.POINTER(ctypes.c_int64))
    data = w[0]
    shape = tuple(int(w[2 + k]) for k in range(nd))
    strides = tuple(int(w[2 + nd + k]) for k in range(nd))
    span = 1 + sum((n - 1) * st for n, st in zip(shape, strides) if n)
    buf = (ctypes.c_char * max(1, span * dtype.itemsize)).from_address(data) if data else bytearray(dtype.itemsize)
    return np.ndarray(shape, dtype=dtype, buffer=buf, strides=tuple(st * dtype.itemsize for st in strides))


def _matmul_cb(code, da, db, dout):
    """apsi_matmul(code, a, b, out): out[...] = a @ b with NumPy's BLAS (multi-threaded, releases the GIL)."""
    try:
        import numpy as np
        dtype = np.dtype(np.float64 if code // 100 == 8 else np.float32)
        na, nb = code // 10 % 10, code % 10
        a = _as_numpy(da, na, dtype)
        b = _as_numpy(db, nb, dtype)
        out = _as_numpy(dout, 2 if na == 2 and nb == 2 else 1, dtype)
        np.matmul(a, b, out=out)
    except BaseException as e:
        if _error is not None and _error[0] == 0:
            _error[0] = 99
        sys.stderr.write(f"apsi native: matmul failed: {e}\n")


def _gemm_cb(code, da, db, dc, M, N, K):
    """apsi_gemm: a loop nest recognised as a matrix product (see apsi/idioms.py). 0 = done with BLAS, 1 = the
    compiled loops must run (nothing was written)."""
    try:
        import numpy as np
        from ..idioms import gemm
        dtype = np.dtype(np.float64 if code // 100 == 8 else np.float32)
        a, b, c = (_as_numpy(d, 2, dtype) for d in (da, db, dc))
        return 0 if gemm(code % 100, a, b, c, M, N, K) else 1
    except BaseException:                                     # noqa: BLE001
        return 1


def _print_cb(site, slots):
    try:
        PRINT_SITES[site](slots)
    except BaseException as e:
        sys.stderr.write(f"apsi native: print failed: {e}\n")
