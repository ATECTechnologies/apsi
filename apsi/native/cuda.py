"""Minimal CUDA driver API binding (ctypes): no CUDA toolkit needed, only the NVIDIA driver.

The driver itself compiles the PTX that Apsi generates for sm_75 (GTX 1660 / RTX 20xx) to the machine code of
whatever GPU is present (Ampere, Ada, Hopper, Blackwell ...), and caches the result.
Windows: nvcuda.dll (installed with the display driver). Linux: libcuda.so.1.
"""
import ctypes
import os
import sys
import threading

_lock = threading.RLock()
_driver = None
_failed = None

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
MIN_CC = (7, 5)


class CudaError(Exception):
    pass


class Device:
    def __init__(self, drv, index):
        self.drv, self.index = drv, index
        h = ctypes.c_int()
        drv.check(drv.lib.cuDeviceGet(ctypes.byref(h), index), "cuDeviceGet")
        self.handle = h.value
        name = ctypes.create_string_buffer(256)
        drv.check(drv.lib.cuDeviceGetName(name, 256, self.handle), "cuDeviceGetName")
        self.name = name.value.decode(errors="replace")
        self.cc = (self.attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                   self.attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
        self.sms = self.attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        mem = ctypes.c_size_t()
        drv.check(drv.lib.cuDeviceTotalMem_v2(ctypes.byref(mem), self.handle), "cuDeviceTotalMem")
        self.memory = mem.value
        self.ctx = None
        self.modules = {}

    def attr(self, a):
        v = ctypes.c_int()
        self.drv.check(self.drv.lib.cuDeviceGetAttribute(ctypes.byref(v), a, self.handle), "cuDeviceGetAttribute")
        return v.value

    def activate(self):
        lib = self.drv.lib
        if self.ctx is None:
            c = ctypes.c_void_p()
            self.drv.check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(c), self.handle), "cuDevicePrimaryCtxRetain")
            self.ctx = c
        self.drv.check(lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    # ---------------------------------------------------------------- modules
    def load(self, key, ptx):
        m = self.modules.get(key)
        if m is None:
            self.activate()
            mod = ctypes.c_void_p()
            data = ptx.encode() + b"\0"
            log = ctypes.create_string_buffer(8192)
            # CU_JIT_ERROR_LOG_BUFFER = 5, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
            opts = (ctypes.c_int * 2)(5, 6)
            vals = (ctypes.c_void_p * 2)(ctypes.cast(log, ctypes.c_void_p), ctypes.c_void_p(8192))
            r = self.drv.lib.cuModuleLoadDataEx(ctypes.byref(mod), data, 2, opts, vals)
            if r:
                raise CudaError(f"the GPU driver rejected the kernel ({self.drv.err(r)}): "
                                f"{log.value.decode(errors='replace')}")
            fn = ctypes.c_void_p()
            self.drv.check(self.drv.lib.cuModuleGetFunction(ctypes.byref(fn), mod, b"apsi_kernel"),
                           "cuModuleGetFunction")
            m = self.modules[key] = (mod, fn)
        return m

    def global_ptr(self, mod, name):
        p = ctypes.c_uint64()
        size = ctypes.c_size_t()
        self.drv.check(self.drv.lib.cuModuleGetGlobal_v2(ctypes.byref(p), ctypes.byref(size), mod, name.encode()),
                       f"cuModuleGetGlobal({name})")
        return p.value, size.value

    # ---------------------------------------------------------------- memory
    def alloc(self, n):
        p = ctypes.c_uint64()
        self.drv.check(self.drv.lib.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(max(1, n))), "cuMemAlloc")
        return p.value

    def free(self, p):
        if p:
            self.drv.lib.cuMemFree_v2(ctypes.c_uint64(p))

    def to_device(self, dptr, host_addr, n):
        if n:
            self.drv.check(self.drv.lib.cuMemcpyHtoD_v2(ctypes.c_uint64(dptr), ctypes.c_void_p(host_addr),
                                                         ctypes.c_size_t(n)), "cuMemcpyHtoD")

    def to_host(self, host_addr, dptr, n):
        if n:
            self.drv.check(self.drv.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(host_addr), ctypes.c_uint64(dptr),
                                                         ctypes.c_size_t(n)), "cuMemcpyDtoH")

    def launch(self, fn, grid, block, params):
        """params: list of ctypes objects; the kernel receives pointers to them."""
        arr = (ctypes.c_void_p * len(params))(*[ctypes.cast(ctypes.pointer(p), ctypes.c_void_p) for p in params])
        self.drv.check(self.drv.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, None, arr, None),
                       "cuLaunchKernel")

    def sync(self):
        self.activate()
        self.drv.check(self.drv.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def describe(self):
        return {"index": self.index, "name": self.name, "compute_capability": f"{self.cc[0]}.{self.cc[1]}",
                "multiprocessors": self.sms, "memory_gb": round(self.memory / 2 ** 30, 2)}


class Driver:
    def __init__(self):
        names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
        lib = None
        errs = []
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError as e:
                errs.append(str(e))
        if lib is None:
            raise CudaError("no NVIDIA driver (CUDA) was found")
        self.lib = lib
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_uint,
                                           ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.c_void_p, ctypes.c_char_p]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [ctypes.c_uint, ctypes.c_void_p,
                                                                                 ctypes.c_void_p, ctypes.c_void_p]
        lib.cuCtxSynchronize.argtypes = []
        self.check(lib.cuInit(0), "cuInit")
        n = ctypes.c_int()
        self.check(lib.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
        devs = [Device(self, i) for i in range(n.value)]
        self.devices = [d for d in devs if d.cc >= MIN_CC]
        self.skipped = [d for d in devs if d.cc < MIN_CC]
        limit = os.environ.get("APSI_GPUS", "").strip()
        if limit:
            keep = {int(x) for x in limit.split(",") if x.strip().isdigit()}
            self.devices = [d for d in self.devices if d.index in keep]
        if not self.devices:
            raise CudaError("no CUDA GPU with compute capability 7.5 or newer (GTX 1660 / RTX 20xx and newer)")

    def err(self, r):
        s = ctypes.c_char_p()
        try:
            self.lib.cuGetErrorString(r, ctypes.byref(s))
        except Exception:
            pass
        return f"CUDA error {r}" + (f": {s.value.decode()}" if s.value else "")

    def check(self, r, what):
        if r:
            raise CudaError(f"{what} failed: {self.err(r)}")


def driver():
    """The CUDA driver, or None (the reason is in `unavailable_reason()`)."""
    global _driver, _failed
    if _driver is not None or _failed is not None:
        return _driver
    with _lock:
        if _driver is None and _failed is None:
            if os.environ.get("APSI_NO_GPU", "").strip() in ("1", "true", "yes"):
                _failed = "disabled by APSI_NO_GPU"
                return None
            try:
                _driver = Driver()
            except (CudaError, OSError, AttributeError) as e:
                _failed = str(e)
    return _driver


def unavailable_reason():
    driver()
    return _failed


def devices():
    d = driver()
    return [x.describe() for x in d.devices] if d else []
