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
    size_t total = 0; if (!cu.DeviceTotalMem(&total, dev) && total) a_dev_limit = total / 6;
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
void *a_gpu_alloc(size_t bytes) {
    if (!g_open()) a_error("RuntimeError", "no GPU: %s", g_why);
    if (!bytes) bytes = 4;
    if (g_state == 2) { void *p = calloc(1, bytes); if (!p) a_error("MemoryError", "GPU memory"); return p; }
    CUdeviceptr p = 0;
    CUresult r = cu.MemAlloc(&p, bytes);
    if (r == 2) a_error("MemoryError", "the card is out of memory (%zu bytes asked); tensors no longer used are freed at the next loop turn or call", bytes);
    check(r, "memory allocation");
    a_dev_since += bytes;
    return (void *)(size_t)p;
}
void a_gpu_free(void *p) {
    if (!p) return;
    if (g_state == 2) { free(p); return; }
    if (g_state == 1) cu.MemFree((CUdeviceptr)(size_t)p);
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
