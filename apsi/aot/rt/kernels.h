/* What the GPU kernels share with the runtime: one argument block for every kernel. */
#ifndef APSI_KERNELS_H
#define APSI_KERNELS_H
typedef struct KArgs {
    long n, len, inner, M, N, K;
    int op, nd, flag, pad;
    float s, s2, s3, s4, s5, s6, s7, s8;
    long sh[8], sa[8], sb[8], sc[8];
    float *a, *b, *c, *d;
} KArgs;
enum { K_ADD, K_SUB, K_MUL, K_DIV, K_POW, K_MOD, K_FLOORDIV, K_MAX, K_MIN, K_LT, K_LE, K_GT, K_GE, K_EQ, K_NE, K_AND, K_OR,
       K_SUM, K_MEAN };
enum { KU_NEG, KU_ABS, KU_EXP, KU_LOG, KU_SQRT, KU_TANH, KU_RELU, KU_SIGMOID, KU_GELU, KU_GELU_ERF, KU_SOFTPLUS, KU_SILU,
       KU_SQUARE, KU_RSQRT, KU_SIGN, KU_NOT, KU_RELU_GRAD, KU_GELU_GRAD, KU_GELU_ERF_GRAD, KU_RECIP };
#endif
