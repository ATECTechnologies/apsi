/* Apsi native runtime: the GPU. The CUDA driver is loaded when the program first asks for the card (nvcuda.dll /
 * libcuda.so: it comes with the NVIDIA driver, nothing else is needed); the kernels are PTX built into the program
 * (kernels.c, compiled by clang), so the driver compiles them for whatever card it finds. With APSI_GPU_EMU=1 the
 * same kernels run on host threads instead: the GPU path can be tested anywhere. */
#include "apsi_rt.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif
#include "kernels_ptx.h"

typedef int CUresult; typedef int CUdevice; typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef unsigned long long CUdeviceptr;
static struct {
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetCount)(int *);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DeviceTotalMem)(size_t *, CUdevice);
    CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxSynchronize)(void);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*MemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
} cu;
static int g_state = 0;            /* 0 not tried, 1 ready (card), 2 ready (emulated), -1 no card */
static char g_why[256] = "";
static char g_name[256] = "";
static CUmodule g_mod;
size_t a_dev_since = 0, a_dev_limit = (size_t)1 << 30;   /* card bytes allocated since the last collection, and when to collect */

#ifdef APSI_GPU_EMU_BUILD
/* ---- the emulated card: kernels.c compiled for the host */
void k_binary(KArgs); void k_scalar(KArgs); void k_unary(KArgs); void k_fill(KArgs); void k_where(KArgs);
void k_permute(KArgs); void k_reduce(KArgs); void k_softmax(KArgs); void k_gemm(KArgs); void k_gather(KArgs);
void k_scatter_add(KArgs); void k_adam(KArgs);
extern __thread unsigned emu_tid, emu_bid, emu_ntid, emu_nbid;
static __thread pthread_barrier_t *emu_bar;
void emu_sync(void) { pthread_barrier_wait(emu_bar); }
typedef struct { void (*fn)(KArgs); KArgs k; unsigned tid, bid, ntid, nbid; pthread_barrier_t *bar; } EmuT;
static void *emu_thread(void *p) {
    EmuT *e = (EmuT *)p; emu_tid = e->tid; emu_bid = e->bid; emu_ntid = e->ntid; emu_nbid = e->nbid; emu_bar = e->bar;
    e->fn(e->k); return NULL;
}
static struct { const char *name; void (*fn)(KArgs); int sync; } emu_table[] = {
    {"k_binary", k_binary, 0}, {"k_scalar", k_scalar, 0}, {"k_unary", k_unary, 0}, {"k_fill", k_fill, 0},
    {"k_where", k_where, 0}, {"k_permute", k_permute, 0}, {"k_reduce", k_reduce, 1}, {"k_softmax", k_softmax, 1},
    {"k_gemm", k_gemm, 1}, {"k_gather", k_gather, 0}, {"k_scatter_add", k_scatter_add, 0}, {"k_adam", k_adam, 0}, {NULL, NULL, 0}};
static void emu_launch(const char *name, KArgs *k, unsigned grid, unsigned block) {
    for (int i = 0; emu_table[i].name; i++) if (!strcmp(emu_table[i].name, name)) {
        for (unsigned b = 0; b < grid; b++) {
            if (!emu_table[i].sync) {
                for (unsigned t = 0; t < block; t++) { emu_tid = t; emu_bid = b; emu_ntid = block; emu_nbid = grid; emu_table[i].fn(*k); }
                continue;
            }
            pthread_barrier_t bar; pthread_barrier_init(&bar, NULL, block);
            pthread_t *th = malloc(sizeof(pthread_t) * block); EmuT *es = malloc(sizeof(EmuT) * block);
            for (unsigned t = 0; t < block; t++) { es[t] = (EmuT){emu_table[i].fn, *k, t, b, block, grid, &bar}; pthread_create(&th[t], NULL, emu_thread, &es[t]); }
            for (unsigned t = 0; t < block; t++) pthread_join(th[t], NULL);
            pthread_barrier_destroy(&bar); free(th); free(es);
        }
        return;
    }
    a_error("RuntimeError", "no kernel %s", name);
}
#endif

static size_t cache_cap;
static void *sym(void *lib, const char *n) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)lib, n);
#else
    return dlsym(lib, n);
#endif
}
static int g_open(void) {
    if (g_state) return g_state > 0;
    const char *emu = getenv("APSI_GPU_EMU");
#ifdef APSI_GPU_EMU_BUILD
    if (emu && *emu == '1') { g_state = 2; snprintf(g_name, sizeof g_name, "emulated card (host threads)"); return 1; }
#else
    (void)emu;
#endif
#ifdef _WIN32
    void *lib = (void *)LoadLibraryA("nvcuda.dll");
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW); if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
#endif
    if (!lib) { g_state = -1; snprintf(g_why, sizeof g_why, "no NVIDIA driver (CUDA) was found"); return 0; }
    #define L(f, n) *(void **)&cu.f = sym(lib, n); if (!cu.f) { g_state = -1; snprintf(g_why, sizeof g_why, "the NVIDIA driver lacks %s", n); return 0; }
    L(Init, "cuInit") L(DeviceGet, "cuDeviceGet") L(DeviceGetCount, "cuDeviceGetCount") L(DeviceGetName, "cuDeviceGetName")
    L(DeviceTotalMem, "cuDeviceTotalMem_v2") L(DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") L(CtxSetCurrent, "cuCtxSetCurrent")
    L(CtxSynchronize, "cuCtxSynchronize") L(MemAlloc, "cuMemAlloc_v2") L(MemFree, "cuMemFree_v2") L(MemcpyHtoD, "cuMemcpyHtoD_v2")
    L(MemcpyDtoH, "cuMemcpyDtoH_v2") L(MemcpyDtoD, "cuMemcpyDtoD_v2") L(MemGetInfo, "cuMemGetInfo_v2")
    L(ModuleLoadDataEx, "cuModuleLoadDataEx") L(ModuleGetFunction, "cuModuleGetFunction") L(LaunchKernel, "cuLaunchKernel")
    L(GetErrorString, "cuGetErrorString")
    #undef L
    int count = 0; CUdevice dev; CUcontext ctx;
    if (cu.Init(0) || cu.DeviceGetCount(&count) || count < 1) { g_state = -1; snprintf(g_why, sizeof g_why, "no CUDA card is available"); return 0; }
    int which = getenv("APSI_GPU") ? atoi(getenv("APSI_GPU")) : 0;
    if (cu.DeviceGet(&dev, which) || cu.DevicePrimaryCtxRetain(&ctx, dev) || cu.CtxSetCurrent(ctx)) { g_state = -1; snprintf(g_why, sizeof g_why, "the card could not be opened"); return 0; }
    cu.DeviceGetName(g_name, sizeof g_name, dev);
    size_t total = 0; if (!cu.DeviceTotalMem(&total, dev) && total) { a_dev_limit = total / 6; cache_cap = total / 4; }
    if (getenv("APSI_GPU_CACHE_MB")) cache_cap = (size_t)atoll(getenv("APSI_GPU_CACHE_MB")) << 20;
    char log[4096] = ""; int opts[2] = {4, 5}; void *vals[2] = {log, (void *)(size_t)sizeof log};   /* CU_JIT_ERROR_LOG_BUFFER(_SIZE) */
    CUresult r = cu.ModuleLoadDataEx(&g_mod, apsi_kernels_ptx, 2, opts, vals);
    if (r) { g_state = -1; snprintf(g_why, sizeof g_why, "the driver did not take the kernels (error %d): %.180s", r, log); return 0; }
    g_state = 1;
    return 1;
}
static void check(CUresult r, const char *what) {
    if (!r) return;
    const char *s = "?"; if (cu.GetErrorString) cu.GetErrorString(r, &s);
    a_error("RuntimeError", "GPU: %s failed: %s (error %d)", what, s, r);
}
int a_gpu_available(void) { return g_open(); }
const char *a_gpu_name(void) { g_open(); return g_state > 0 ? g_name : g_why; }
/* the card's memory: freed blocks are kept by size and handed out again (cuMemAlloc / cuMemFree synchronise the
 * card and are slow); the cache is emptied when an allocation fails, before the collector is asked */
#define GC_SLOTS 4096
static struct { CUdeviceptr p; size_t bytes; } blk[GC_SLOTS * 2];          /* live blocks: pointer -> size (open addressing) */
static struct { CUdeviceptr p; size_t bytes; } freel[GC_SLOTS]; static int nfree = 0;
static size_t cached = 0;
static size_t round_up(size_t b) { if (b < 512) return 512; if (b < ((size_t)1 << 20)) return (b + 511) & ~(size_t)511; return (b + ((size_t)1 << 20) - 1) & ~(((size_t)1 << 20) - 1); }
static void blk_put(CUdeviceptr p, size_t bytes) {
    size_t h = (size_t)(p >> 8) % (GC_SLOTS * 2);
    for (int i = 0; i < GC_SLOTS * 2; i++, h = (h + 1) % (GC_SLOTS * 2)) if (!blk[h].p || blk[h].p == (CUdeviceptr)-1) { blk[h].p = p; blk[h].bytes = bytes; return; }
}
static size_t blk_take(CUdeviceptr p) {
    size_t h = (size_t)(p >> 8) % (GC_SLOTS * 2);
    for (int i = 0; i < GC_SLOTS * 2 && blk[h].p; i++, h = (h + 1) % (GC_SLOTS * 2)) if (blk[h].p == p) { size_t b = blk[h].bytes; blk[h].p = (CUdeviceptr)-1; return b; }
    return 0;
}
static void cache_release(void) { for (int i = 0; i < nfree; i++) cu.MemFree(freel[i].p); nfree = 0; cached = 0; }
void a_gpu_trim(void) { if (g_state == 1) cache_release(); }
void *a_gpu_alloc(size_t bytes) {
    if (!g_open()) a_error("RuntimeError", "no GPU: %s", g_why);
    if (!bytes) bytes = 4;
    if (g_state == 2) { void *p = calloc(1, bytes); if (!p) a_error("MemoryError", "GPU memory"); return p; }
    size_t want = round_up(bytes);
    for (int i = nfree - 1; i >= 0; i--) if (freel[i].bytes == want) {          /* a block of this size, from the cache */
        CUdeviceptr p = freel[i].p; freel[i] = freel[--nfree]; cached -= want; blk_put(p, want); a_dev_since += want; return (void *)(size_t)p;
    }
    CUdeviceptr p = 0;
    CUresult r = cu.MemAlloc(&p, want);
    if (r == 2) { cache_release(); r = cu.MemAlloc(&p, want); }
    if (r == 2) { a_gc_full(); cache_release(); r = cu.MemAlloc(&p, want); }
    if (r == 2) {
        size_t fb = 0, tb = 0; if (cu.MemGetInfo) cu.MemGetInfo(&fb, &tb);
        a_error("MemoryError", "the card is out of memory: %.1f MB asked, %.1f MB free of %.1f MB. Use smaller batches, "
                "f16 / bf16 tensors, or free tensors you no longer need (set them to none)", want / 1e6, fb / 1e6, tb / 1e6);
    }
    check(r, "memory allocation");
    blk_put(p, want);
    a_dev_since += want;
    return (void *)(size_t)p;
}
void a_gpu_free(void *p) {
    if (!p) return;
    if (g_state == 2) { free(p); return; }
    if (g_state != 1) return;
    CUdeviceptr d = (CUdeviceptr)(size_t)p; size_t b = blk_take(d);
    if (b && nfree < GC_SLOTS && cached + b <= cache_cap) { freel[nfree].p = d; freel[nfree++].bytes = b; cached += b; return; }
    cu.MemFree(d);
}
void a_gpu_upload(void *dst, const void *src, size_t n) { if (g_state == 2) { memcpy(dst, src, n); return; } check(cu.MemcpyHtoD((CUdeviceptr)(size_t)dst, src, n), "copy to the card"); }
void a_gpu_download(void *dst, const void *src, size_t n) { if (g_state == 2) { memcpy(dst, src, n); return; } check(cu.MemcpyDtoH(dst, (CUdeviceptr)(size_t)src, n), "copy from the card"); }
void a_gpu_copy(void *dst, const void *src, size_t n) { if (g_state == 2) { memcpy(dst, src, n); return; } check(cu.MemcpyDtoD((CUdeviceptr)(size_t)dst, (CUdeviceptr)(size_t)src, n), "copy on the card"); }
void a_gpu_sync(void) { if (g_state == 1) check(cu.CtxSynchronize(), "synchronize"); }
void a_gpu_launch(const char *name, KArgs *k, unsigned grid, unsigned block) {
    if (grid == 0) return;
#ifdef APSI_GPU_EMU_BUILD
    if (g_state == 2) { emu_launch(name, k, grid, block); return; }
#endif
    static struct { char name[32]; CUfunction f; } cache[32]; static int ncache = 0;
    CUfunction f = NULL;
    for (int i = 0; i < ncache; i++) if (!strcmp(cache[i].name, name)) f = cache[i].f;
    if (!f) {
        check(cu.ModuleGetFunction(&f, g_mod, name), name);
        if (ncache < 32) { snprintf(cache[ncache].name, 32, "%s", name); cache[ncache++].f = f; }
    }
    void *params[1] = {k};
    check(cu.LaunchKernel(f, grid, 1, 1, block, 1, 1, 0, NULL, params, NULL), name);
}
void a_gpu_info(size_t *free_b, size_t *total_b) {
    *free_b = *total_b = 0;
    if (g_state == 1 && cu.MemGetInfo) cu.MemGetInfo(free_b, total_b);
}

/* ------------------------------------------------------------------ cuBLAS, when it is installed: the vendor's matrix
 * products (Tensor Cores with APSI_GPU_MATH=tf32 / bf16 / f16). Found in CUDA_PATH / CUDA_HOME, in the NVIDIA wheels of
 * an active or nearby .venv (pip install nvidia-cublas-cu12 / -cu13), in the CUDA Toolkit folders, then the system
 * search path. Any CUDA 11, 12 or 13 cuBLAS works; without one Apsi's own kernels run. APSI_CUBLAS=0 turns it off. */
typedef int (*blas_create_t)(void **);
typedef int (*blas_gemm_ex_t)(void *, int, int, int, int, int, const void *, const void *, int, int, const void *, int, int,
                              const void *, void *, int, int, int, int);
typedef int (*blas_gemm_sb_t)(void *, int, int, int, int, int, const void *, const void *, int, int, long long,
                              const void *, int, int, long long, const void *, void *, int, int, long long, int, int, int);
static struct { int state; void *h; blas_gemm_ex_t gemm; blas_gemm_sb_t gemm_sb; char path[512]; } blas;
static void *blas_try(const char *path) {
#ifdef _WIN32
    return (void *)LoadLibraryExA(path, NULL, strchr(path, '\\') || strchr(path, '/') ? 0x00000100 | 0x00001000 : 0);   /* the DLL's own folder for its dependencies */
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}
static void *blas_in(const char *dir, char *found) {
#ifdef _WIN32
    static const char *names[] = {"cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll", NULL}; const char sep = '\\';
#else
    static const char *names[] = {"libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so", NULL}; const char sep = '/';
#endif
    for (int i = 0; names[i]; i++) {
        char p[512]; if (dir) snprintf(p, sizeof p, "%s%c%s", dir, sep, names[i]); else snprintf(p, sizeof p, "%s", names[i]);
        void *h = blas_try(p); if (h) { snprintf(found, 512, "%s", p); return h; }
    }
    return NULL;
}
static void *blas_venv(const char *venv, char *found) {          /* the NVIDIA wheels of a virtual environment (lib/python3.N/site-packages/nvidia/cublas/lib, or Lib\\site-packages\\nvidia\\cublas\\bin) */
    char d[512]; void *h;
#ifdef _WIN32
    snprintf(d, sizeof d, "%s\\Lib\\site-packages\\nvidia\\cublas\\bin", venv); if ((h = blas_in(d, found))) return h;
    snprintf(d, sizeof d, "%s\\Lib\\site-packages\\nvidia\\cu13\\bin", venv); if ((h = blas_in(d, found))) return h;
#else
    for (int minor = 20; minor >= 8; minor--) {
        snprintf(d, sizeof d, "%s/lib/python3.%d/site-packages/nvidia/cublas/lib", venv, minor); if ((h = blas_in(d, found))) return h;
        snprintf(d, sizeof d, "%s/lib/python3.%d/site-packages/nvidia/cu13/lib", venv, minor); if ((h = blas_in(d, found))) return h;
    }
#endif
    return NULL;
}
#ifdef _WIN32
static const char *sep_cuda = "\\.apsi\\cuda";
#else
static const char *sep_cuda = "/.apsi/cuda";
#endif
static int blas_open(void) {
    if (blas.state) return blas.state > 0;
    blas.state = -1;
    const char *off = getenv("APSI_CUBLAS"); if (off && *off == '0') return 0;
    void *h = NULL; char found[512] = ""; char d[512];
    const char *ex = getenv("APSI_CUBLAS_PATH"); if (ex && *ex) { h = blas_try(ex); if (h) snprintf(found, sizeof found, "%s", ex); }
    const char *envs[] = {"CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", NULL};
    for (int i = 0; !h && envs[i]; i++) {
        const char *r = getenv(envs[i]); if (!r || !*r) continue;
#ifdef _WIN32
        snprintf(d, sizeof d, "%s\\bin", r); h = blas_in(d, found);
        if (!h) { snprintf(d, sizeof d, "%s\\bin\\x64", r); h = blas_in(d, found); }
#else
        snprintf(d, sizeof d, "%s/lib64", r); h = blas_in(d, found);
        if (!h) { snprintf(d, sizeof d, "%s/lib", r); h = blas_in(d, found); }
#endif
    }
    if (!h && getenv("VIRTUAL_ENV")) h = blas_venv(getenv("VIRTUAL_ENV"), found);
    if (!h) h = blas_venv(".venv", found);
    if (!h) {                                                            /* `apsi cuda install --user` */
        const char *home = getenv("USERPROFILE"); if (!home) home = getenv("HOME");
        if (home) { char u[512]; snprintf(u, sizeof u, "%s%s", home, sep_cuda); h = blas_venv(u, found); }
    }
#ifdef _WIN32
    for (int v = 14; !h && v >= 11; v--) for (int m = 9; !h && m >= 0; m--) {
        snprintf(d, sizeof d, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%d.%d\\bin", v, m); h = blas_in(d, found);
        if (!h) { snprintf(d, sizeof d, "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v%d.%d\\bin\\x64", v, m); h = blas_in(d, found); }
    }
#else
    if (!h) h = blas_in("/usr/local/cuda/lib64", found);
#endif
    if (!h) h = blas_in(NULL, found);
    if (!h) return 0;
    blas_create_t create = (blas_create_t)sym(h, "cublasCreate_v2");
    blas.gemm = (blas_gemm_ex_t)sym(h, "cublasGemmEx");
    blas.gemm_sb = (blas_gemm_sb_t)sym(h, "cublasGemmStridedBatchedEx");
    if (!create || !blas.gemm || !blas.gemm_sb || create(&blas.h)) return 0;
    snprintf(blas.path, sizeof blas.path, "%s", found);
    blas.state = 1;
    return 1;
}
const char *a_gpu_blas(void) { return g_state == 1 && blas_open() ? blas.path : NULL; }
/* C = A @ B, row-major, float32 on the card (batched with strides sa, sb; C packed): cuBLAS sees the transposes */
int a_gpu_gemm_blas(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K, int64_t batch, int64_t sa, int64_t sb) {
    if (g_state != 1 || M > 0x7fffffff || N > 0x7fffffff || K > 0x7fffffff || !blas_open()) return 0;
    const char *m = getenv("APSI_GPU_MATH"); int compute = 68;                              /* CUBLAS_COMPUTE_32F: exact float32 */
    if (m && !strcmp(m, "tf32")) compute = 77; else if (m && !strcmp(m, "bf16")) compute = 75; else if (m && !strcmp(m, "f16")) compute = 74;
    float one = 1.0f, zero = 0.0f;
    int r;
    if (batch == 1) r = blas.gemm(blas.h, 0, 0, (int)N, (int)M, (int)K, &one, B, 0, (int)N, A, 0, (int)K, &zero, C, 0, (int)N, compute, -1);
    else r = blas.gemm_sb(blas.h, 0, 0, (int)N, (int)M, (int)K, &one, B, 0, (int)N, (long long)sb, A, 0, (int)K, (long long)sa, &zero, C, 0, (int)N, (long long)(M * N), (int)batch, compute, -1);
    return r == 0;
}
