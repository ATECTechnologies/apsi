/* Apsi's GPU kernels, written in C. Built for the card by clang (-target nvptx64-nvidia-cuda) into PTX that the
 * runtime loads through the CUDA driver; built for the CPU with -DAPSI_EMU they run on host threads, which is how
 * the GPU path is tested on machines without a card. Every kernel takes one KArgs by value. */
#include "kernels.h"

#ifdef APSI_EMU
#include <math.h>
__thread unsigned emu_tid, emu_bid, emu_ntid, emu_nbid;
#define TID emu_tid
#define BID emu_bid
#define NTID emu_ntid
#define NBID emu_nbid
#define SYNC() emu_sync()
#define SHARED static
void emu_sync(void);
#define KERNEL
#define EXPF expf
#define LOGF logf
#define SQRTF sqrtf
#define TANHF tanhf
#define ERFF erff
#define POWF powf
#define FMODF fmodf
#define FLOORF floorf
#define ATOMIC_ADD(p, v) __atomic_fetch_add_float((p), (v))
static inline void __atomic_fetch_add_float(float *p, float v) {
    float old = *p, nw;
    do { nw = old + v; } while (!__atomic_compare_exchange(p, &old, &nw, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
}
#else
#define TID __nvvm_read_ptx_sreg_tid_x()
#define BID __nvvm_read_ptx_sreg_ctaid_x()
#define NTID __nvvm_read_ptx_sreg_ntid_x()
#define NBID __nvvm_read_ptx_sreg_nctaid_x()
#define SYNC() __nvvm_bar_sync(0)
#define SHARED static __attribute__((address_space(3)))
#define KERNEL __attribute__((nvptx_kernel))
/* the card's maths without libdevice: hardware ex2 / lg2 / sqrt, the rest built on them */
static inline float EXPF(float x) { return __nvvm_ex2_approx_f(x * 1.4426950408889634f); }
static inline float LOGF(float x) { return __nvvm_lg2_approx_f(x) * 0.6931471805599453f; }
static inline float SQRTF(float x) { return __nvvm_sqrt_rn_f(x); }
static inline float FLOORF(float x) { return __nvvm_floor_f(x); }
static inline float TANHF(float x) { if (x > 15) return 1; if (x < -15) return -1; float e = EXPF(2 * x); return (e - 1) / (e + 1); }
static inline float ERFF(float x) {        /* Abramowitz-Stegun 7.1.26 with a sign: |error| < 1.5e-7 */
    float s = x < 0 ? -1.0f : 1.0f; x = x < 0 ? -x : x;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * EXPF(-x * x);
    return s * y;
}
static inline float POWF(float x, float y) {
    if (y == 2.0f) return x * x;
    if (y == 1.0f) return x;
    if (y == 0.5f) return SQRTF(x);
    if (x > 0) return __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(x));
    if (x == 0) return y > 0 ? 0.0f : (y == 0 ? 1.0f : 1.0f / 0.0f);
    float yi = FLOORF(y);
    if (yi != y) return 0.0f / 0.0f;
    float r = __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(-x));
    long q = (long)yi; return (q & 1) ? -r : r;
}
static inline float FMODF(float x, float y) { float q = x / y; q = q < 0 ? -FLOORF(-q) : FLOORF(q); return x - q * y; }
#define ATOMIC_ADD(p, v) __nvvm_atom_add_gen_f((p), (v))
#endif

static inline float ebin(int op, float x, float y) {
    switch (op) {
    case K_ADD: return x + y; case K_SUB: return x - y; case K_MUL: return x * y; case K_DIV: return x / y;
    case K_POW: return POWF(x, y);
    case K_MOD: { float r = FMODF(x, y); if (r != 0 && ((r < 0) != (y < 0))) r += y; return r; }
    case K_FLOORDIV: return FLOORF(x / y);
    case K_MAX: return x > y ? x : y; case K_MIN: return x < y ? x : y;
    case K_LT: return x < y; case K_LE: return x <= y; case K_GT: return x > y; case K_GE: return x >= y;
    case K_EQ: return x == y; case K_NE: return x != y;
    case K_AND: return (x != 0) && (y != 0); case K_OR: return (x != 0) || (y != 0);
    default: return 0;
    }
}
static inline float eun(int op, float x) {
    switch (op) {
    case KU_NEG: return -x; case KU_ABS: return x < 0 ? -x : x; case KU_EXP: return EXPF(x); case KU_LOG: return LOGF(x);
    case KU_SQRT: return SQRTF(x); case KU_TANH: return TANHF(x); case KU_RELU: return x > 0 ? x : 0;
    case KU_SIGMOID: return 1.0f / (1.0f + EXPF(-x));
    case KU_GELU: return 0.5f * x * (1.0f + TANHF(0.7978845608f * (x + 0.044715f * x * x * x)));
    case KU_GELU_ERF: return 0.5f * x * (1.0f + ERFF(x * 0.70710678f));
    case KU_SOFTPLUS: return x > 20 ? x : LOGF(1.0f + EXPF(x));
    case KU_SILU: return x / (1.0f + EXPF(-x));
    case KU_SQUARE: return x * x; case KU_RSQRT: return 1.0f / SQRTF(x);
    case KU_SIGN: return (float)((x > 0) - (x < 0));
    case KU_NOT: return x == 0;
    case KU_RELU_GRAD: return x > 0 ? 1.0f : 0.0f;
    case KU_GELU_GRAD: { float k = 0.7978845608f, u = k * (x + 0.044715f * x * x * x), t = TANHF(u); return 0.5f * (1 + t) + 0.5f * x * (1 - t * t) * k * (1 + 3 * 0.044715f * x * x); }
    case KU_GELU_ERF_GRAD: return 0.5f * (1 + ERFF(x * 0.70710678f)) + x * EXPF(-0.5f * x * x) * 0.39894228f;
    case KU_RECIP: return 1.0f / x;
    default: return 0;
    }
}

/* c = a op b, both broadcast to the output shape (strides 0 on broadcast axes) */
KERNEL void k_binary(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) {
        long ia = 0, ib = 0, r = i;
        for (int d = k.nd - 1; d >= 0; d--) { long q = r % k.sh[d]; r /= k.sh[d]; ia += q * k.sa[d]; ib += q * k.sb[d]; }
        k.c[i] = ebin(k.op, k.a[ia], k.b[ib]);
    }
}
/* c = a op s  (or s op a when k.flag) */
KERNEL void k_scalar(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) k.c[i] = k.flag ? ebin(k.op, k.s, k.a[i]) : ebin(k.op, k.a[i], k.s);
}
KERNEL void k_unary(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) k.c[i] = eun(k.op, k.a[i]);
}
KERNEL void k_fill(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) k.c[i] = k.s;
}
/* c = cond ? a : b, broadcast (strides in sa / sb / sc) */
KERNEL void k_where(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) {
        long ia = 0, ib = 0, ic = 0, r = i;
        for (int d = k.nd - 1; d >= 0; d--) { long q = r % k.sh[d]; r /= k.sh[d]; ia += q * k.sa[d]; ib += q * k.sb[d]; ic += q * k.sc[d]; }
        k.c[i] = k.d[ic] != 0 ? k.a[ia] : k.b[ib];
    }
}
/* out[i] = a[the element of a at output index i under a permutation]: sa holds a's strides per output axis */
KERNEL void k_permute(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) {
        long src = 0, r = i;
        for (int d = k.nd - 1; d >= 0; d--) { long q = r % k.sh[d]; r /= k.sh[d]; src += q * k.sa[d]; }
        k.c[i] = k.a[src];
    }
}
/* one block per output element of a reduction over an axis: a is viewed as [outer, len, inner] */
KERNEL void k_reduce(KArgs k) {
    SHARED float part[256];
    long o = BID;
    long outer_i = o / k.inner, inner_i = o % k.inner;
    const float *base = k.a + outer_i * k.len * k.inner + inner_i;
    float acc = k.op == K_MAX ? -3.4e38f : k.op == K_MIN ? 3.4e38f : 0.0f;
    for (long j = TID; j < k.len; j += NTID) {
        float x = base[j * k.inner];
        if (k.op == K_MAX) acc = x > acc ? x : acc; else if (k.op == K_MIN) acc = x < acc ? x : acc; else acc += x;
    }
    part[TID] = acc;
    SYNC();
    for (unsigned s = NTID / 2; s > 0; s >>= 1) {
        if (TID < s) {
            float x = part[TID + s], y = part[TID];
            part[TID] = k.op == K_MAX ? (x > y ? x : y) : k.op == K_MIN ? (x < y ? x : y) : x + y;
        }
        SYNC();
    }
    if (TID == 0) k.c[o] = k.op == K_MEAN ? part[0] / (float)k.len : part[0];
}
/* softmax / log-softmax along the last axis: one block per row */
KERNEL void k_softmax(KArgs k) {
    SHARED float part[256];
    const float *x = k.a + (long)BID * k.len;
    float *y = k.c + (long)BID * k.len;
    float m = -3.4e38f;
    for (long j = TID; j < k.len; j += NTID) m = x[j] > m ? x[j] : m;
    part[TID] = m; SYNC();
    for (unsigned s = NTID / 2; s > 0; s >>= 1) { if (TID < s && part[TID + s] > part[TID]) part[TID] = part[TID + s]; SYNC(); }
    m = part[0]; SYNC();
    float sum = 0;
    for (long j = TID; j < k.len; j += NTID) sum += EXPF(x[j] - m);
    part[TID] = sum; SYNC();
    for (unsigned s = NTID / 2; s > 0; s >>= 1) { if (TID < s) part[TID] += part[TID + s]; SYNC(); }
    float ls = LOGF(part[0]);
    for (long j = TID; j < k.len; j += NTID) y[j] = k.flag ? x[j] - m - ls : EXPF(x[j] - m - ls);
}
/* C[b] = A[b] @ B[b]: 64x64 tiles of C, 16 deep, each of 256 threads makes a 4x4 block */
KERNEL void k_gemm(KArgs k) {
    SHARED float As[16][64 + 1];
    SHARED float Bs[16][64 + 1];
    long tiles_n = (k.N + 63) / 64, tiles_m = (k.M + 63) / 64;
    long t = BID % (tiles_m * tiles_n), batch = BID / (tiles_m * tiles_n);
    long tm = t / tiles_n, tn = t % tiles_n;
    const float *A = k.a + batch * k.sa[0], *B = k.b + batch * k.sb[0];
    float *C = k.c + batch * k.M * k.N;
    int tx = TID % 16, ty = TID / 16;
    float acc[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (long k0 = 0; k0 < k.K; k0 += 16) {
        for (int q = TID; q < 16 * 64; q += 256) {
            int kk = q / 64, mm = q % 64;
            long gm = tm * 64 + mm, gk = k0 + kk;
            As[kk][mm] = (gm < k.M && gk < k.K) ? A[gm * k.K + gk] : 0.0f;
            long gn = tn * 64 + mm;
            Bs[kk][mm] = (gk < k.K && gn < k.N) ? B[gk * k.N + gn] : 0.0f;
        }
        SYNC();
        for (int kk = 0; kk < 16; kk++) {
            float av[4], bv[4];
            for (int i = 0; i < 4; i++) av[i] = As[kk][ty * 4 + i];
            for (int j = 0; j < 4; j++) bv[j] = Bs[kk][tx * 4 + j];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
        }
        SYNC();
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
        long gm = tm * 64 + ty * 4 + i, gn = tn * 64 + tx * 4 + j;
        if (gm < k.M && gn < k.N) C[gm * k.N + gn] = acc[i][j];
    }
}
/* gather rows: c[i, :] = a[idx[i], :] (row length k.len); idx are floats holding whole numbers */
KERNEL void k_gather(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) { long r = i / k.len, c = i % k.len; k.c[i] = k.a[(long)k.d[r] * k.len + c]; }
}
/* scatter-add rows: c[idx[i], :] += a[i, :] */
KERNEL void k_scatter_add(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) { long r = i / k.len, c = i % k.len; ATOMIC_ADD(&k.c[(long)k.d[r] * k.len + c], k.a[i]); }
}
/* the whole Adam update in one pass: p, m, v updated in place from g */
KERNEL void k_adam(KArgs k) {
    long i = (long)BID * NTID + TID;
    for (; i < k.n; i += (long)NBID * NTID) {
        float g = k.b[i];
        float m = k.s2 * k.c[i] + (1 - k.s2) * g, v = k.s3 * k.d[i] + (1 - k.s3) * g * g;
        k.c[i] = m; k.d[i] = v;
        k.a[i] = k.a[i] * (1 - k.s5) - k.s * (m / k.s4) / (SQRTF(v / k.s6) + k.s7);
    }
}
