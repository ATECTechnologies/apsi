"""cuBLAS for matrix products in @gpu functions: Tensor Cores for float16 / bfloat16, tuned kernels for f32 / f64.

A @gpu function whose loop nest is a matrix product (see apsi/idioms.py) runs as one cuBLAS GEMM instead of the
generated kernel when cuBLAS is installed (the CUDA Toolkit, or `pip install nvidia-cublas-cu12`):

* float16 and bfloat16 tensors use Tensor Cores with 32-bit accumulation (CUBLAS_COMPUTE_32F);
* float32 uses ordinary 32-bit arithmetic (no TF32 unless APSI_TF32=1, which trades 13 mantissa bits for speed);
* float64 uses DGEMM.

As on the CPU, only the order of the additions differs from the loops. Without cuBLAS, or when anything does not
fit (aliasing, bounds, other types), the generated kernel runs instead. APSI_CUBLAS=0 turns this off.
"""
import ctypes
import glob
import os
import sys
import threading

ENABLED = os.environ.get("APSI_CUBLAS", "1") != "0"
TF32 = os.environ.get("APSI_TF32", "0") == "1"

CUDA_R_32F, CUDA_R_64F, CUDA_R_16F, CUDA_R_16BF = 0, 1, 2, 14
COMPUTE_32F, COMPUTE_32F_FAST_TF32, COMPUTE_64F = 68, 77, 70
OP_N = 0
GEMM_DEFAULT = -1
TYPES = {"f8": CUDA_R_64F, "f4": CUDA_R_32F, "f2": CUDA_R_16F, "h2": CUDA_R_16BF}

_lock = threading.Lock()
_lib = None
_tried = False
_reason = ""
_handles = {}


def _candidates():
    names = (["cublas64_12.dll", "cublas64_11.dll"] if sys.platform == "win32"
             else ["libcublas.so.12", "libcublas.so.11", "libcublas.so"])
    dirs = []
    for env in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env)
        if root:
            dirs += [os.path.join(root, "bin"), os.path.join(root, "lib64"), os.path.join(root, "lib", "x64")]
    for p in sys.path:                                  # pip install nvidia-cublas-cu12
        for sub in ("lib", "bin"):
            d = os.path.join(p, "nvidia", "cublas", sub)
            if os.path.isdir(d):
                dirs.append(d)
    if sys.platform == "win32":
        dirs += glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin")
    out = []
    for d in dirs:
        for n in names:
            f = os.path.join(d, n)
            if os.path.exists(f):
                out.append(f)
    return out + names                                  # finally the system search path


def library():
    """The loaded cuBLAS library, or None (see reason())."""
    global _lib, _tried, _reason
    if _tried:
        return _lib
    with _lock:
        if _tried:
            return _lib
        _tried = True
        if not ENABLED:
            _reason = "APSI_CUBLAS=0"
            return None
        for c in _candidates():
            try:
                if sys.platform == "win32" and os.path.isabs(c):
                    os.add_dll_directory(os.path.dirname(c))
                lib = ctypes.CDLL(c)
                lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                lib.cublasGemmEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_int]
                _lib = lib
                return lib
            except (OSError, AttributeError) as e:
                _reason = str(e)
        _reason = _reason or "cuBLAS not found (CUDA Toolkit, or pip install nvidia-cublas-cu12)"
        return None


def reason():
    library()
    return _reason


def _handle(dev):
    h = _handles.get(dev.index)
    if h is None:
        dev.activate()
        h = ctypes.c_void_p()
        st = _lib.cublasCreate_v2(ctypes.byref(h))
        if st != 0:
            raise RuntimeError(f"cublasCreate failed ({st})")
        _handles[dev.index] = h
    return h


def gemm(dev, elem, accumulate, M, N, K, a, lda, b, ldb, c, ldc):
    """Row-major c[:M, :N] (+)= a[:M, :K] @ b[:K, :N] on device pointers (cuBLAS is column-major, so this computes
    cᵀ = bᵀ aᵀ). Raises on failure."""
    lib = library()
    if lib is None:
        raise RuntimeError(reason())
    h = _handle(dev)
    dt = TYPES[elem]
    if elem == "f8":
        alpha, beta, compute = ctypes.c_double(1.0), ctypes.c_double(1.0 if accumulate else 0.0), COMPUTE_64F
    else:
        compute = COMPUTE_32F_FAST_TF32 if (elem == "f4" and TF32) else COMPUTE_32F
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(1.0 if accumulate else 0.0)
    st = lib.cublasGemmEx(h, OP_N, OP_N, N, M, K, ctypes.byref(alpha), ctypes.c_void_p(b), dt, ldb,
                          ctypes.c_void_p(a), dt, lda, ctypes.byref(beta), ctypes.c_void_p(c), dt, ldc, compute,
                          GEMM_DEFAULT)
    if st != 0:
        raise RuntimeError(f"cublasGemmEx failed ({st})")
