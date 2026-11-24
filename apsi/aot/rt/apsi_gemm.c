/* Apsi native runtime: matrix products on the CPU, and the worker threads every parallel loop shares.
 * The product is the usual packed form (as in BLIS / OpenBLAS): B is packed in KC x NR column panels, A in
 * MR x KC row panels, and a register-blocked micro-kernel written with vector types does the work; clang turns it
 * into AVX2 / AVX-512 FMA code when the program is built for this machine (-march=native). */
#include <stdint.h>
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
#include <pthread.h>
#include <unistd.h>
#endif

/* ================================================================== the worker threads */
typedef void (*PoolFn)(void *ctx, int64_t task);
static int pool_n = 0;                    /* threads, the calling one included */
static volatile int64_t pool_next, pool_ntasks, pool_done;
static PoolFn pool_fn; static void *pool_ctx;
static volatile int64_t pool_gen = 0;
#ifdef _WIN32
static SRWLOCK pool_lock = SRWLOCK_INIT; static CONDITION_VARIABLE pool_cv = CONDITION_VARIABLE_INIT, pool_cv_done = CONDITION_VARIABLE_INIT;
#define LOCK() AcquireSRWLockExclusive(&pool_lock)
#define UNLOCK() ReleaseSRWLockExclusive(&pool_lock)
#define WAIT(cv) SleepConditionVariableSRW(&(cv), &pool_lock, INFINITE, 0)
#define WAKE_ALL(cv) WakeAllConditionVariable(&(cv))
#else
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t pool_cv = PTHREAD_COND_INITIALIZER, pool_cv_done = PTHREAD_COND_INITIALIZER;
#define LOCK() pthread_mutex_lock(&pool_lock)
#define UNLOCK() pthread_mutex_unlock(&pool_lock)
#define WAIT(cv) pthread_cond_wait(&(cv), &pool_lock)
#define WAKE_ALL(cv) pthread_cond_broadcast(&(cv))
#endif
static void pool_work(void) {
    for (;;) {
        int64_t t = __atomic_fetch_add(&pool_next, 1, __ATOMIC_ACQ_REL);
        if (t >= pool_ntasks) break;
        pool_fn(pool_ctx, t);
        if (__atomic_add_fetch(&pool_done, 1, __ATOMIC_ACQ_REL) == pool_ntasks) { LOCK(); WAKE_ALL(pool_cv_done); UNLOCK(); }
    }
}
#ifdef _WIN32
static DWORD WINAPI pool_thread(LPVOID p)
#else
static void *pool_thread(void *p)
#endif
{
    (void)p;
    int64_t seen = 0;
    for (;;) {
        LOCK();
        while (pool_gen == seen) WAIT(pool_cv);
        seen = pool_gen;
        UNLOCK();
        pool_work();
    }
#ifndef _WIN32
    return NULL;
#endif
}
int a_threads(void) {
    if (!pool_n) {
        const char *e = getenv("APSI_THREADS");
        int n;
#ifdef _WIN32
        SYSTEM_INFO si; GetSystemInfo(&si); n = (int)si.dwNumberOfProcessors;
#else
        long c = sysconf(_SC_NPROCESSORS_ONLN); n = c > 0 ? (int)c : 1;
#endif
        if (e && atoi(e) > 0) n = atoi(e);
        if (n > 64) n = 64;
        pool_n = n;
        for (int i = 1; i < n; i++) {
#ifdef _WIN32
            CloseHandle(CreateThread(NULL, 0, pool_thread, NULL, 0, NULL));
#else
            pthread_t th; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
            pthread_create(&th, &at, pool_thread, NULL); pthread_attr_destroy(&at);
#endif
        }
    }
    return pool_n;
}
static volatile int pool_busy = 0;
/* run fn(ctx, 0 .. ntasks-1) on every core; nested calls run in the calling thread */
void a_parallel(PoolFn fn, void *ctx, int64_t ntasks) {
    if (ntasks <= 0) return;
    if (ntasks == 1 || a_threads() == 1 || __atomic_exchange_n(&pool_busy, 1, __ATOMIC_ACQ_REL)) {
        for (int64_t t = 0; t < ntasks; t++) fn(ctx, t);
        return;
    }
    LOCK();
    pool_fn = fn; pool_ctx = ctx; pool_ntasks = ntasks; pool_done = 0;
    __atomic_store_n(&pool_next, 0, __ATOMIC_RELEASE);
    pool_gen++;
    WAKE_ALL(pool_cv);
    UNLOCK();
    pool_work();
    LOCK();
    while (__atomic_load_n(&pool_done, __ATOMIC_ACQUIRE) < ntasks) WAIT(pool_cv_done);
    UNLOCK();
    __atomic_store_n(&pool_busy, 0, __ATOMIC_RELEASE);
}

#ifdef _WIN32
#include <malloc.h>
#define ALLOC64(n) _aligned_malloc((n), 64)
#define FREE64(p) _aligned_free(p)
#else
#define ALLOC64(n) aligned_alloc(64, (((n) + 63) / 64) * 64)
#define FREE64(p) free(p)
#endif
/* ================================================================== the packed product, float64 */
#define MR 6
#define KC 256
#define MC 96
#define NC 2048
#ifdef __AVX512F__                  /* 512-bit registers: twice the columns per micro-tile */
#define VB 64
#else
#define VB 32
#endif
#define VD (VB / 8)                 /* doubles per vector */
#define VF (VB / 4)                 /* floats per vector */
#define NR (2 * VD)
typedef double v4d __attribute__((vector_size(VB), aligned(8)));
typedef float v8f __attribute__((vector_size(VB), aligned(4)));

static inline void kernel_d(int64_t kc, const double *a, const double *b, double *c, int64_t ldc, int mr, int nr, int add) {
    v4d c0[MR], c1[MR];
    for (int i = 0; i < MR; i++) { c0[i] = (v4d){0}; c1[i] = (v4d){0}; }
    for (int64_t p = 0; p < kc; p++) {
        v4d b0, b1; memcpy(&b0, b + p * NR, VB); memcpy(&b1, b + p * NR + VD, VB);
        const double *ap = a + p * MR;
        for (int i = 0; i < MR; i++) { v4d x = (v4d){0} + ap[i]; c0[i] += x * b0; c1[i] += x * b1; }
    }
    if (mr == MR && nr == NR) {
        for (int i = 0; i < MR; i++) {
            double *ci = c + i * ldc; v4d o0 = c0[i], o1 = c1[i];
            if (add) { v4d p0, p1; memcpy(&p0, ci, VB); memcpy(&p1, ci + VD, VB); o0 += p0; o1 += p1; }
            memcpy(ci, &o0, VB); memcpy(ci + VD, &o1, VB);
        }
    } else {
        for (int i = 0; i < mr; i++) for (int j = 0; j < nr; j++) { double v = j < VD ? c0[i][j] : c1[i][j - VD]; c[i * ldc + j] = add ? c[i * ldc + j] + v : v; }
    }
}
static void pack_b_d(const double *B, int64_t ldb, int64_t kc, int64_t nc, double *out) {
    for (int64_t j = 0; j < nc; j += NR) {
        int64_t w = nc - j < NR ? nc - j : NR;
        for (int64_t p = 0; p < kc; p++) {
            const double *src = B + p * ldb + j;
            for (int64_t q = 0; q < NR; q++) out[q] = q < w ? src[q] : 0.0;
            out += NR;
        }
    }
}
static void pack_a_d(const double *A, int64_t lda, int64_t mc, int64_t kc, double *out) {
    for (int64_t i = 0; i < mc; i += MR) {
        int64_t h = mc - i < MR ? mc - i : MR;
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t q = 0; q < MR; q++) out[q] = q < h ? A[(i + q) * lda + p] : 0.0;
            out += MR;
        }
    }
}
typedef struct { const double *A, *B; double *C; int64_t M, N, K, lda, ldb, ldc; const double *Bp; int64_t kc, pc, jc, nc; int add; } GD;
static void gd_task(void *p, int64_t t) {          /* one MC block of rows against the packed B panel */
    GD *g = (GD *)p;
    int64_t ic = t * MC, mc = g->M - ic < MC ? g->M - ic : MC;
    static _Thread_local double Ap[MC * KC] __attribute__((aligned(64)));
    pack_a_d(g->A + ic * g->lda + g->pc, g->lda, mc, g->kc, Ap);
    for (int64_t jr = 0; jr < g->nc; jr += NR)
        for (int64_t ir = 0; ir < mc; ir += MR)
            kernel_d(g->kc, Ap + ir * g->kc, g->Bp + jr * g->kc, g->C + (ic + ir) * g->ldc + g->jc + jr, g->ldc,
                     (int)(mc - ir < MR ? mc - ir : MR), (int)(g->nc - jr < NR ? g->nc - jr : NR), g->add);
}
static void small_d(const double *A, const double *B, double *C, int64_t M, int64_t N, int64_t K, int add) {
    for (int64_t i = 0; i < M; i++) {
        double *c = C + i * N; const double *a = A + i * K;
        if (!add) memset(c, 0, sizeof(double) * (size_t)N);
        for (int64_t k = 0; k < K; k++) { double x = a[k]; const double *b = B + k * N; for (int64_t j = 0; j < N; j++) c[j] += x * b[j]; }
    }
}
/* C = A (M x K) @ B (K x N), or C += with add; all row-major and contiguous */
void a_gemm_d(const double *A, const double *B, double *C, int64_t M, int64_t N, int64_t K, int parallel, int add) {
    if (K == 0) { if (!add) memset(C, 0, sizeof(double) * (size_t)(M * N)); return; }
    if (M * N * K <= 32 * 32 * 32 || M < 4) { small_d(A, B, C, M, N, K, add); return; }
    double *Bp = (double *)ALLOC64(sizeof(double) * (size_t)KC * (size_t)((NC + NR - 1) / NR * NR) + 64);
    for (int64_t jc = 0; jc < N; jc += NC) {
        int64_t nc = N - jc < NC ? N - jc : NC;
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = K - pc < KC ? K - pc : KC;
            pack_b_d(B + pc * N + jc, N, kc, nc, Bp);
            GD g = {A, B, C, M, N, K, K, N, N, Bp, kc, pc, jc, nc, add || pc > 0};
            int64_t tasks = (M + MC - 1) / MC;
            if (parallel) a_parallel(gd_task, &g, tasks); else for (int64_t t = 0; t < tasks; t++) gd_task(&g, t);
        }
    }
    FREE64(Bp);
}

/* ================================================================== float32: 6 x 16 */
#define NRF (2 * VF)
static inline void kernel_f(int64_t kc, const float *a, const float *b, float *c, int64_t ldc, int mr, int nr, int add) {
    v8f c0[MR], c1[MR];
    for (int i = 0; i < MR; i++) { c0[i] = (v8f){0}; c1[i] = (v8f){0}; }
    for (int64_t p = 0; p < kc; p++) {
        v8f b0, b1; memcpy(&b0, b + p * NRF, VB); memcpy(&b1, b + p * NRF + VF, VB);
        const float *ap = a + p * MR;
        for (int i = 0; i < MR; i++) { v8f x = (v8f){0} + ap[i]; c0[i] += x * b0; c1[i] += x * b1; }
    }
    if (mr == MR && nr == NRF) {
        for (int i = 0; i < MR; i++) {
            float *ci = c + i * ldc; v8f o0 = c0[i], o1 = c1[i];
            if (add) { v8f p0, p1; memcpy(&p0, ci, VB); memcpy(&p1, ci + VF, VB); o0 += p0; o1 += p1; }
            memcpy(ci, &o0, VB); memcpy(ci + VF, &o1, VB);
        }
    } else {
        for (int i = 0; i < mr; i++) for (int j = 0; j < nr; j++) { float v = j < VF ? c0[i][j] : c1[i][j - VF]; c[i * ldc + j] = add ? c[i * ldc + j] + v : v; }
    }
}
static void pack_b_f(const float *B, int64_t ldb, int64_t kc, int64_t nc, float *out) {
    for (int64_t j = 0; j < nc; j += NRF) {
        int64_t w = nc - j < NRF ? nc - j : NRF;
        for (int64_t p = 0; p < kc; p++) {
            const float *src = B + p * ldb + j;
            for (int64_t q = 0; q < NRF; q++) out[q] = q < w ? src[q] : 0.0f;
            out += NRF;
        }
    }
}
static void pack_a_f(const float *A, int64_t lda, int64_t mc, int64_t kc, float *out) {
    for (int64_t i = 0; i < mc; i += MR) {
        int64_t h = mc - i < MR ? mc - i : MR;
        for (int64_t p = 0; p < kc; p++) {
            for (int64_t q = 0; q < MR; q++) out[q] = q < h ? A[(i + q) * lda + p] : 0.0f;
            out += MR;
        }
    }
}
typedef struct { const float *A, *B; float *C; int64_t M, N, K, lda, ldb, ldc; const float *Bp; int64_t kc, pc, jc, nc; int add; } GF;
static void gf_task(void *p, int64_t t) {
    GF *g = (GF *)p;
    int64_t ic = t * MC, mc = g->M - ic < MC ? g->M - ic : MC;
    static _Thread_local float Ap[MC * KC] __attribute__((aligned(64)));
    pack_a_f(g->A + ic * g->lda + g->pc, g->lda, mc, g->kc, Ap);
    for (int64_t jr = 0; jr < g->nc; jr += NRF)
        for (int64_t ir = 0; ir < mc; ir += MR)
            kernel_f(g->kc, Ap + ir * g->kc, g->Bp + jr * g->kc, g->C + (ic + ir) * g->ldc + g->jc + jr, g->ldc,
                     (int)(mc - ir < MR ? mc - ir : MR), (int)(g->nc - jr < NRF ? g->nc - jr : NRF), g->add);
}
static void small_f(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K, int add) {
    for (int64_t i = 0; i < M; i++) {
        float *c = C + i * N; const float *a = A + i * K;
        if (!add) memset(c, 0, sizeof(float) * (size_t)N);
        for (int64_t k = 0; k < K; k++) { float x = a[k]; const float *b = B + k * N; for (int64_t j = 0; j < N; j++) c[j] += x * b[j]; }
    }
}
void a_gemm_f(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K, int parallel, int add) {
    if (K == 0) { if (!add) memset(C, 0, sizeof(float) * (size_t)(M * N)); return; }
    if (M * N * K <= 32 * 32 * 32 || M < 4) { small_f(A, B, C, M, N, K, add); return; }
    float *Bp = (float *)ALLOC64(sizeof(float) * (size_t)KC * (size_t)((NC + NRF - 1) / NRF * NRF) + 64);
    for (int64_t jc = 0; jc < N; jc += NC) {
        int64_t nc = N - jc < NC ? N - jc : NC;
        for (int64_t pc = 0; pc < K; pc += KC) {
            int64_t kc = K - pc < KC ? K - pc : KC;
            pack_b_f(B + pc * N + jc, N, kc, nc, Bp);
            GF g = {A, B, C, M, N, K, K, N, N, Bp, kc, pc, jc, nc, add || pc > 0};
            int64_t tasks = (M + MC - 1) / MC;
            if (parallel) a_parallel(gf_task, &g, tasks); else for (int64_t t = 0; t < tasks; t++) gf_task(&g, t);
        }
    }
    FREE64(Bp);
}
