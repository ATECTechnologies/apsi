/* Apsi native runtime: tensors (n-dimensional arrays), their maths, and reverse-mode automatic differentiation.
 * Tensors are contiguous, row-major; float64 by default as in Apsi, with float32, int64 and bool. Every operation
 * broadcasts like NumPy. A value being differentiated is a Node: the operations on it are recorded, and grad()
 * walks them backwards with the same generic operations, so gradients work on numbers, quantities and tensors. */
#include "apsi_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <pthread.h>
#include <unistd.h>
#endif

enum { DT_F64 = 0, DT_F32, DT_I64, DT_BOOL };
static const char *dt_name[] = {"float64", "float32", "int64", "bool"};
static const int dt_size[] = {8, 4, 8, 1};
#define MAXD 8
typedef struct ATensor { AObj h; uint8_t dt, nd, dev, pad; int64_t sh[MAXD]; int64_t n; void *p; } ATensor;
typedef struct ANode { AObj h; AV val; int op; AV a, b, x; AV grad; int64_t id; int seen; } ANode;
#define TT(v) ((ATensor *)(v).o)
#define ND(v) ((ANode *)(v).o)

static int is_t(AV v) { return v.t == T_OBJ && v.o->kind == O_TENSOR; }
static int is_n(AV v) { return v.t == T_OBJ && v.o->kind == O_NODE; }
static int is_num(AV v) { return v.t == T_INT || v.t == T_FLOAT || v.t == T_BOOL; }

/* ------------------------------------------------------------------ making tensors */
static void tfree(AObj *o) { free(((ATensor *)o)->p); }
static void tmark(AObj *o) { (void)o; }
static ATensor *t_new(int dt, int nd, const int64_t *sh) {
    ATensor *t = (ATensor *)a_alloc(sizeof(ATensor), O_TENSOR);
    t->dt = (uint8_t)dt; t->nd = (uint8_t)nd; t->n = 1;
    for (int i = 0; i < nd; i++) { if (sh[i] < 0) a_error("ValueError", "negative dimension"); t->sh[i] = sh[i]; t->n *= sh[i]; }
    size_t bytes = (size_t)t->n * (size_t)dt_size[dt];
    t->p = calloc(bytes ? bytes : 1, 1);
    if (!t->p) a_error("MemoryError", "cannot allocate a tensor of %lld elements", (long long)t->n);
    a_note_alloc(bytes);
    return t;
}
static ATensor *t_like(ATensor *a, int dt) { return t_new(dt, a->nd, a->sh); }
static inline double t_get(ATensor *t, int64_t i) {
    switch (t->dt) {
    case DT_F64: return ((double *)t->p)[i];
    case DT_F32: return ((float *)t->p)[i];
    case DT_I64: return (double)((int64_t *)t->p)[i];
    default: return ((uint8_t *)t->p)[i];
    }
}
static inline void t_set(ATensor *t, int64_t i, double v) {
    switch (t->dt) {
    case DT_F64: ((double *)t->p)[i] = v; break;
    case DT_F32: ((float *)t->p)[i] = (float)v; break;
    case DT_I64: ((int64_t *)t->p)[i] = (int64_t)v; break;
    default: ((uint8_t *)t->p)[i] = v != 0; break;
    }
}
static AV t_scalar(ATensor *t, int64_t i) {
    switch (t->dt) {
    case DT_I64: return a_int(((int64_t *)t->p)[i]);
    case DT_BOOL: return a_bool(((uint8_t *)t->p)[i]);
    default: return a_float(t_get(t, i));
    }
}
static int64_t prod(const int64_t *s, int nd) { int64_t n = 1; for (int i = 0; i < nd; i++) n *= s[i]; return n; }

/* nested lists -> a tensor */
static int infer(AV v, int depth, int64_t *sh, int *nd, int *dt) {
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) {
        AList *l = (AList *)v.o;
        if (depth >= MAXD) a_error("ValueError", "too many dimensions");
        if (depth >= *nd) { *nd = depth + 1; sh[depth] = l->len; }
        else if (sh[depth] != l->len) a_error("ValueError", "a tensor needs lists of equal length (ragged nesting)");
        for (int64_t i = 0; i < l->len; i++) infer(l->v[i], depth + 1, sh, nd, dt);
        return 1;
    }
    if (is_t(v)) {
        ATensor *t = TT(v);
        for (int i = 0; i < t->nd; i++) {
            if (depth + i >= *nd) { *nd = depth + i + 1; sh[depth + i] = t->sh[i]; }
            else if (sh[depth + i] != t->sh[i]) a_error("ValueError", "a tensor needs parts of equal shape");
        }
        if (t->dt == DT_F64 || t->dt == DT_F32) *dt = DT_F64; else if (t->dt == DT_I64 && *dt == DT_BOOL) *dt = DT_I64;
        return 1;
    }
    if (depth != *nd && *nd > 0 && depth < *nd) a_error("ValueError", "a tensor needs lists of equal length (ragged nesting)");
    if (v.t == T_FLOAT) *dt = DT_F64;
    else if (v.t == T_INT) { if (*dt == DT_BOOL) *dt = DT_I64; }
    else if (v.t == T_BOOL) {}
    else if (is_n(v)) *dt = DT_F64;
    else a_error("TypeError", "a tensor holds numbers, not %s", a_is(v, O_STR) ? "text" : "this value");
    return 0;
}
static void fill_from(ATensor *t, AV v, int64_t *pos) {
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) { AList *l = (AList *)v.o; for (int64_t i = 0; i < l->len; i++) fill_from(t, l->v[i], pos); return; }
    if (is_t(v)) { ATensor *s = TT(v); for (int64_t i = 0; i < s->n; i++) t_set(t, (*pos)++, t_get(s, i)); return; }
    if (is_n(v)) v = ND(v)->val;
    t_set(t, (*pos)++, v.t == T_FLOAT ? v.f : (double)v.i);
}
static int dt_from_name(AV d) {
    if (d.t == T_NONE) return -1;
    const char *s = a_is(d, O_STR) ? ((AStr *)d.o)->s : "";
    if (!strcmp(s, "float64") || !strcmp(s, "f64") || !strcmp(s, "fp64")) return DT_F64;
    if (!strcmp(s, "float32") || !strcmp(s, "f32") || !strcmp(s, "fp32")) return DT_F32;
    if (!strcmp(s, "int64") || !strcmp(s, "i64") || !strcmp(s, "int")) return DT_I64;
    if (!strcmp(s, "bool")) return DT_BOOL;
    a_error("ValueError", "unknown dtype '%s'", s);
}
static AV kwget(ADict *kw, const char *name) { AV v; if (kw && a_dict_get(kw, a_str(name), &v)) return v; return A_NONE; }
AV t_from(AV v, int dt) {
    int64_t sh[MAXD]; int nd = 0, d = DT_BOOL;
    infer(v, 0, sh, &nd, &d);
    if (dt >= 0) d = dt;
    ATensor *t = t_new(d, nd, sh);
    int64_t pos = 0; fill_from(t, v, &pos);
    return a_obj(t);
}
static AV tcopy(ATensor *a, int dt) {
    ATensor *t = t_like(a, dt < 0 ? a->dt : dt);
    if (t->dt == a->dt) memcpy(t->p, a->p, (size_t)a->n * (size_t)dt_size[a->dt]);
    else for (int64_t i = 0; i < a->n; i++) t_set(t, i, t_get(a, i));
    return a_obj(t);
}
static AV as_tensor(AV v) {
    if (is_t(v)) return v;
    if (is_num(v)) { ATensor *t = t_new(v.t == T_FLOAT ? DT_F64 : v.t == T_INT ? DT_I64 : DT_BOOL, 0, NULL); t_set(t, 0, v.t == T_FLOAT ? v.f : (double)v.i); return a_obj(t); }
    return t_from(v, -1);
}
static int shape_args(int n, AV *a, int64_t *sh) {
    if (n == 1 && (a_is(a[0], O_LIST) || a_is(a[0], O_TUPLE))) {
        AList *l = (AList *)a[0].o; if (l->len > MAXD) a_error("ValueError", "too many dimensions");
        for (int64_t i = 0; i < l->len; i++) sh[i] = l->v[i].i;
        return (int)l->len;
    }
    if (n > MAXD) a_error("ValueError", "too many dimensions");
    for (int i = 0; i < n; i++) { if (a[i].t != T_INT) a_error("TypeError", "a shape is whole numbers"); sh[i] = a[i].i; }
    return n;
}

/* ------------------------------------------------------------------ printing (as Apsi prints NumPy arrays) */
typedef struct { char *p; int64_t n, cap; } TB;
static void tb_put(TB *b, const char *s) {
    int64_t k = (int64_t)strlen(s);
    if (b->n + k + 1 > b->cap) { b->cap = (b->n + k + 1) * 2 + 64; b->p = (char *)realloc(b->p, (size_t)b->cap); }
    memcpy(b->p + b->n, s, (size_t)k); b->n += k; b->p[b->n] = 0;
}
static void elem_text(ATensor *t, int64_t i, char *out, int width) {
    char s[64];
    if (t->dt == DT_BOOL) snprintf(s, sizeof s, "%s", t_get(t, i) != 0 ? "true" : "false");
    else if (t->dt == DT_I64) snprintf(s, sizeof s, "%lld", (long long)((int64_t *)t->p)[i]);
    else {
        double v = t_get(t, i);
        if (isnan(v)) snprintf(s, sizeof s, "nan");
        else if (isinf(v)) snprintf(s, sizeof s, v > 0 ? "inf" : "-inf");
        else {
            snprintf(s, sizeof s, "%.6g", v);
            if (!strchr(s, '.') && !strchr(s, 'e') && !strchr(s, 'n')) strcat(s, ".");
        }
    }
    if (width > 0) snprintf(out, 64, "%*s", width, s); else strcpy(out, s);
}
static int summarize(ATensor *t) { return t->n > 40; }
static void print_rec(TB *b, ATensor *t, int ax, int64_t off, int64_t stride_total, int width, int indent, int sum) {
    int64_t len = t->sh[ax];
    int64_t inner = stride_total / (len ? len : 1);
    tb_put(b, "[");
    int last = ax == t->nd - 1;
    int line = indent + 1;
    for (int64_t i = 0; i < len; i++) {
        if (sum && len > 6 && i == 3) {
            if (last) { tb_put(b, "..., "); line += 5; }
            else {
                tb_put(b, "...,");
                for (int k = 0; k < t->nd - ax - 1; k++) tb_put(b, "\n");
                for (int k = 0; k <= indent; k++) tb_put(b, " ");
            }
            i = len - 3;
        }
        if (last) {
            char s[64]; elem_text(t, off + i, s, width);
            int k = (int)strlen(s);
            if (i > 0 && line + k + 2 > 100) { tb_put(b, "\n"); for (int q = 0; q <= indent; q++) tb_put(b, " "); line = indent + 1; }
            tb_put(b, s); line += k;
            if (i < len - 1) { tb_put(b, ", "); line += 2; }
        } else {
            print_rec(b, t, ax + 1, off + i * inner, inner, width, indent + 1, sum);
            if (i < len - 1) {
                tb_put(b, ",");
                for (int k = 0; k < t->nd - ax - 1; k++) tb_put(b, "\n");
                for (int k = 0; k <= indent; k++) tb_put(b, " ");
            }
        }
    }
    tb_put(b, "]");
}
char *t_text(AV v, int top) {
    ATensor *t = TT(v);
    TB b = {0};
    if (!top && t->n > 8) {
        char s[128]; int k = snprintf(s, sizeof s, "Tensor[");
        for (int i = 0; i < t->nd; i++) k += snprintf(s + k, sizeof s - (size_t)k, i ? "x%lld" : "%lld", (long long)t->sh[i]);
        snprintf(s + k, sizeof s - (size_t)k, "]");
        tb_put(&b, s); return b.p;
    }
    if (t->nd == 0) { char s[64]; elem_text(t, 0, s, 0); tb_put(&b, s); return b.p; }
    int width = 0, sum = summarize(t);
    if (t->dt == DT_I64) for (int64_t i = 0; i < t->n; i++) { char s[64]; elem_text(t, i, s, 0); int k = (int)strlen(s); if (k > width) width = k; }
    if (t->n == 0) { tb_put(&b, "[]"); return b.p; }
    print_rec(&b, t, 0, 0, t->n, width, 0, sum);
    return b.p;
}

/* ------------------------------------------------------------------ broadcasting element-wise operations */
enum { E_ADD, E_SUB, E_MUL, E_DIV, E_POW, E_MOD, E_FLOORDIV, E_MAX, E_MIN, E_LT, E_LE, E_GT, E_GE, E_EQ, E_NE, E_AND, E_OR, E_XOR, E_ATAN2 };
static int bshape(ATensor *a, ATensor *b, int64_t *out) {
    int nd = a->nd > b->nd ? a->nd : b->nd;
    for (int i = 0; i < nd; i++) {
        int64_t x = i < nd - a->nd ? 1 : a->sh[i - (nd - a->nd)];
        int64_t y = i < nd - b->nd ? 1 : b->sh[i - (nd - b->nd)];
        if (x != y && x != 1 && y != 1) {
            char s1[64] = "", s2[64] = "";
            for (int k = 0; k < a->nd; k++) snprintf(s1 + strlen(s1), 64 - strlen(s1), k ? "x%lld" : "%lld", (long long)a->sh[k]);
            for (int k = 0; k < b->nd; k++) snprintf(s2 + strlen(s2), 64 - strlen(s2), k ? "x%lld" : "%lld", (long long)b->sh[k]);
            a_error("ValueError", "shapes [%s] and [%s] do not broadcast", s1, s2);
        }
        out[i] = x == 1 ? y : x;
    }
    return nd;
}
static void bstrides(ATensor *a, int nd, const int64_t *osh, int64_t *st) {
    int64_t s = 1;
    for (int i = nd - 1; i >= 0; i--) {
        int ai = i - (nd - a->nd);
        if (ai < 0 || a->sh[ai] == 1) st[i] = 0; else { st[i] = s; }
        if (ai >= 0) s *= a->sh[ai];
    }
    (void)osh;
}
static inline double eop(int op, double x, double y) {
    switch (op) {
    case E_ADD: return x + y; case E_SUB: return x - y; case E_MUL: return x * y; case E_DIV: return x / y;
    case E_POW: return pow(x, y);
    case E_MOD: { double r = fmod(x, y); if (r != 0 && ((r < 0) != (y < 0))) r += y; return r; }
    case E_FLOORDIV: return floor(x / y);
    case E_MAX: return x > y || isnan(x) ? x : y; case E_MIN: return x < y || isnan(x) ? x : y;
    case E_LT: return x < y; case E_LE: return x <= y; case E_GT: return x > y; case E_GE: return x >= y;
    case E_EQ: return x == y; case E_NE: return x != y;
    case E_AND: return (x != 0) && (y != 0); case E_OR: return (x != 0) || (y != 0); case E_XOR: return (x != 0) != (y != 0);
    case E_ATAN2: return atan2(x, y);
    default: return NAN;
    }
}
static int res_dt(int op, ATensor *a, ATensor *b) {
    if (op >= E_LT && op <= E_XOR) return DT_BOOL;
    if (op == E_DIV || op == E_ATAN2) return (a->dt == DT_F32 && (b->dt == DT_F32 || b->nd == 0)) ? DT_F32 : DT_F64;
    int fa = a->dt <= DT_F32, fb = b->dt <= DT_F32;
    if (fa || fb) {
        if (a->dt == DT_F64 || b->dt == DT_F64) {
            if (a->dt == DT_F32 && b->nd == 0 && b->dt != DT_F64) return DT_F32;
            if (b->dt == DT_F32 && a->nd == 0 && a->dt != DT_F64) return DT_F32;
            if (a->dt == DT_F32 && b->nd == 0) return DT_F32;      /* a scalar does not widen a float32 tensor */
            if (b->dt == DT_F32 && a->nd == 0) return DT_F32;
            return DT_F64;
        }
        return DT_F32;
    }
    if (op == E_POW && b->dt == DT_I64) return DT_I64;
    return DT_I64;
}
static AV ewise(int op, AV av, AV bv) {
    ATensor *a = TT(av), *b = TT(bv);
    int64_t osh[MAXD]; int nd = bshape(a, b, osh);
    int dt = res_dt(op, a, b);
    ATensor *o = t_new(dt, nd, osh);
    if (dt == DT_I64 && (op == E_FLOORDIV || op == E_MOD)) {
        for (int64_t i = 0; i < b->n; i++) if (t_get(b, i) == 0) a_error("ZeroDivisionError", "integer division by zero in a tensor");
    }
    /* the fast paths: same shapes, or a scalar on one side, in float64 / float32 */
    if (dt == DT_F64 && a->dt == DT_F64 && b->dt == DT_F64 && (a->n == b->n || a->n == 1 || b->n == 1) && (a->nd == b->nd || a->n == 1 || b->n == 1)) {
        double *x = (double *)a->p, *y = (double *)b->p, *r = (double *)o->p;
        int64_t n = o->n, sa = a->n == 1 ? 0 : 1, sb = b->n == 1 ? 0 : 1;
        switch (op) {
        case E_ADD: for (int64_t i = 0; i < n; i++) r[i] = x[i * sa] + y[i * sb]; return a_obj(o);
        case E_SUB: for (int64_t i = 0; i < n; i++) r[i] = x[i * sa] - y[i * sb]; return a_obj(o);
        case E_MUL: for (int64_t i = 0; i < n; i++) r[i] = x[i * sa] * y[i * sb]; return a_obj(o);
        case E_DIV: for (int64_t i = 0; i < n; i++) r[i] = x[i * sa] / y[i * sb]; return a_obj(o);
        default: for (int64_t i = 0; i < n; i++) r[i] = eop(op, x[i * sa], y[i * sb]); return a_obj(o);
        }
    }
    if (dt == DT_F32 && (a->dt == DT_F32 || a->n == 1) && (b->dt == DT_F32 || b->n == 1) && (a->n == b->n || a->n == 1 || b->n == 1) && (a->nd == b->nd || a->n == 1 || b->n == 1)) {
        float *r = (float *)o->p; int64_t n = o->n;
        if (a->dt == DT_F32 && b->dt == DT_F32) {
            float *x = (float *)a->p, *y = (float *)b->p; int64_t sa = a->n == 1 ? 0 : 1, sb = b->n == 1 ? 0 : 1;
            for (int64_t i = 0; i < n; i++) r[i] = (float)eop(op, x[i * sa], y[i * sb]);
        } else if (a->dt == DT_F32) { float *x = (float *)a->p; double y = t_get(b, 0); for (int64_t i = 0; i < n; i++) r[i] = (float)eop(op, x[i], y); }
        else { double x = t_get(a, 0); float *y = (float *)b->p; for (int64_t i = 0; i < n; i++) r[i] = (float)eop(op, x, y[i]); }
        return a_obj(o);
    }
    int64_t sa[MAXD], sb[MAXD], idx[MAXD] = {0};
    bstrides(a, nd, osh, sa); bstrides(b, nd, osh, sb);
    int64_t ia = 0, ib = 0;
    for (int64_t i = 0; i < o->n; i++) {
        if (dt == DT_I64 && a->dt != DT_F64 && b->dt != DT_F64 && a->dt != DT_F32 && b->dt != DT_F32) {
            int64_t x = (int64_t)t_get(a, ia), y = (int64_t)t_get(b, ib), r;
            switch (op) {
            case E_ADD: r = x + y; break; case E_SUB: r = x - y; break; case E_MUL: r = x * y; break;
            case E_FLOORDIV: r = x / y; if ((x % y != 0) && ((x < 0) != (y < 0))) r--; break;
            case E_MOD: r = x % y; if (r != 0 && ((r < 0) != (y < 0))) r += y; break;
            case E_POW: { r = 1; for (int64_t k = 0; k < y; k++) r *= x; break; }
            case E_MAX: r = x > y ? x : y; break; case E_MIN: r = x < y ? x : y; break;
            default: r = (int64_t)eop(op, (double)x, (double)y);
            }
            ((int64_t *)o->p)[i] = r;
        } else t_set(o, i, eop(op, t_get(a, ia), t_get(b, ib)));
        for (int d = nd - 1; d >= 0; d--) {
            idx[d]++; ia += sa[d]; ib += sb[d];
            if (idx[d] < osh[d]) break;
            ia -= sa[d] * osh[d]; ib -= sb[d] * osh[d]; idx[d] = 0;
        }
    }
    return a_obj(o);
}
enum { U_NEG, U_ABS, U_EXP, U_LOG, U_SQRT, U_SIN, U_COS, U_TAN, U_TANH, U_RELU, U_SIGMOID, U_GELU, U_SOFTPLUS,
       U_FLOOR, U_CEIL, U_ROUND, U_NOT, U_LOG1P, U_EXPM1, U_ASIN, U_ACOS, U_ATAN, U_SINH, U_COSH, U_SIGN, U_ISNAN,
       U_ISFINITE, U_LOG2, U_LOG10, U_ERF, U_SQUARE, U_RSQRT, U_SILU, U_GELU_ERF };
static inline double uop(int op, double x) {
    switch (op) {
    case U_NEG: return -x; case U_ABS: return fabs(x); case U_EXP: return exp(x); case U_LOG: return log(x);
    case U_SQRT: return sqrt(x); case U_SIN: return sin(x); case U_COS: return cos(x); case U_TAN: return tan(x);
    case U_TANH: return tanh(x); case U_RELU: return x > 0 ? x : 0; case U_SIGMOID: return 1 / (1 + exp(-x));
    case U_GELU: return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    case U_GELU_ERF: return 0.5 * x * (1 + erf(x * 0.7071067811865476));
    case U_SOFTPLUS: return x > 30 ? x : log1p(exp(x));
    case U_FLOOR: return floor(x); case U_CEIL: return ceil(x); case U_ROUND: return nearbyint(x); case U_NOT: return x == 0;
    case U_LOG1P: return log1p(x); case U_EXPM1: return expm1(x); case U_ASIN: return asin(x); case U_ACOS: return acos(x);
    case U_ATAN: return atan(x); case U_SINH: return sinh(x); case U_COSH: return cosh(x);
    case U_SIGN: return (x > 0) - (x < 0); case U_ISNAN: return isnan(x); case U_ISFINITE: return isfinite(x);
    case U_LOG2: return log2(x); case U_LOG10: return log10(x); case U_ERF: return erf(x); case U_SQUARE: return x * x;
    case U_RSQRT: return 1 / sqrt(x); case U_SILU: return x / (1 + exp(-x));
    default: return NAN;
    }
}
static AV unary(int op, AV v) {
    ATensor *a = TT(v);
    int dt = a->dt;
    if (op == U_NOT || op == U_ISNAN || op == U_ISFINITE) dt = DT_BOOL;
    else if (dt == DT_BOOL || (dt == DT_I64 && !(op == U_NEG || op == U_ABS || op == U_RELU || op == U_SIGN || op == U_SQUARE || op == U_FLOOR || op == U_CEIL || op == U_ROUND))) dt = DT_F64;
    ATensor *o = t_like(a, dt);
    if (a->dt == DT_F64 && dt == DT_F64) { double *x = (double *)a->p, *r = (double *)o->p; for (int64_t i = 0; i < a->n; i++) r[i] = uop(op, x[i]); }
    else if (a->dt == DT_F32 && dt == DT_F32) { float *x = (float *)a->p, *r = (float *)o->p; for (int64_t i = 0; i < a->n; i++) r[i] = (float)uop(op, x[i]); }
    else for (int64_t i = 0; i < a->n; i++) t_set(o, i, uop(op, t_get(a, i)));
    return a_obj(o);
}

/* ------------------------------------------------------------------ matrix products (blocked, on every core) */
typedef struct { const double *A, *B; double *C; int64_t M, N, K, r0, r1; } GJob;
typedef struct { const float *A, *B; float *C; int64_t M, N, K, r0, r1; } GJobF;
static void gemm_rows(GJob *j) {
    const int64_t N = j->N, K = j->K; const double *A = j->A, *B = j->B; double *C = j->C;
    for (int64_t kk = 0; kk < K; kk += 256) {
        int64_t ke = kk + 256 < K ? kk + 256 : K;
        for (int64_t i = j->r0; i < j->r1; i++) {
            double *c = C + i * N; const double *a = A + i * K;
            for (int64_t k = kk; k < ke; k++) { double x = a[k]; if (x == 0) continue; const double *b = B + k * N; for (int64_t q = 0; q < N; q++) c[q] += x * b[q]; }
        }
    }
}
static void gemm_rows_f(GJobF *j) {
    const int64_t N = j->N, K = j->K; const float *A = j->A, *B = j->B; float *C = j->C;
    for (int64_t kk = 0; kk < K; kk += 512) {
        int64_t ke = kk + 512 < K ? kk + 512 : K;
        for (int64_t i = j->r0; i < j->r1; i++) {
            float *c = C + i * N; const float *a = A + i * K;
            for (int64_t k = kk; k < ke; k++) { float x = a[k]; if (x == 0) continue; const float *b = B + k * N; for (int64_t q = 0; q < N; q++) c[q] += x * b[q]; }
        }
    }
}
static int n_threads(void) {
    static int n = 0;
    if (!n) {
        const char *e = getenv("APSI_THREADS");
#ifdef _WIN32
        n = e ? atoi(e) : 4;
#else
        long c = sysconf(_SC_NPROCESSORS_ONLN); n = e ? atoi(e) : (int)(c > 0 ? c : 1);
#endif
        if (n < 1) n = 1; if (n > 64) n = 64;
    }
    return n;
}
#ifndef _WIN32
static void *gemm_thread(void *p) { gemm_rows((GJob *)p); return NULL; }
static void *gemm_thread_f(void *p) { gemm_rows_f((GJobF *)p); return NULL; }
#endif
static void gemm(const double *A, const double *B, double *C, int64_t M, int64_t N, int64_t K) {
    int T = n_threads(); if ((double)M * N * K < 2e5 || M < 2) T = 1; if (T > M) T = (int)M;
    GJob jobs[64];
    for (int t = 0; t < T; t++) { jobs[t] = (GJob){A, B, C, M, N, K, M * t / T, M * (t + 1) / T}; }
#ifndef _WIN32
    pthread_t th[64];
    for (int t = 1; t < T; t++) pthread_create(&th[t], NULL, gemm_thread, &jobs[t]);
    gemm_rows(&jobs[0]);
    for (int t = 1; t < T; t++) pthread_join(th[t], NULL);
#else
    for (int t = 0; t < T; t++) gemm_rows(&jobs[t]);
#endif
}
static void gemm_f(const float *A, const float *B, float *C, int64_t M, int64_t N, int64_t K) {
    int T = n_threads(); if ((double)M * N * K < 2e5 || M < 2) T = 1; if (T > M) T = (int)M;
    GJobF jobs[64];
    for (int t = 0; t < T; t++) { jobs[t] = (GJobF){A, B, C, M, N, K, M * t / T, M * (t + 1) / T}; }
#ifndef _WIN32
    pthread_t th[64];
    for (int t = 1; t < T; t++) pthread_create(&th[t], NULL, gemm_thread_f, &jobs[t]);
    gemm_rows_f(&jobs[0]);
    for (int t = 1; t < T; t++) pthread_join(th[t], NULL);
#else
    for (int t = 0; t < T; t++) gemm_rows_f(&jobs[t]);
#endif
}
static AV matmul(AV av, AV bv) {
    ATensor *a = TT(av), *b = TT(bv);
    if (a->nd == 0 || b->nd == 0) a_error("ValueError", "matmul needs tensors with at least 1 dimension; use * for scalars");
    int dt = (a->dt == DT_F32 && b->dt == DT_F32) ? DT_F32 : DT_F64;
    int a1 = a->nd == 1, b1 = b->nd == 1;
    int64_t M = a1 ? 1 : a->sh[a->nd - 2], K = a->sh[a->nd - 1];
    int64_t Kb = b1 ? b->sh[0] : b->sh[b->nd - 2], N = b1 ? 1 : b->sh[b->nd - 1];
    if (K != Kb) a_error("ValueError", "matmul: inner sizes differ (%lld and %lld)", (long long)K, (long long)Kb);
    int ba = a->nd > 2 ? a->nd - 2 : 0, bb = b->nd > 2 ? b->nd - 2 : 0;
    int nb = ba > bb ? ba : bb; int64_t bsh[MAXD];
    for (int i = 0; i < nb; i++) {
        int64_t x = i < nb - ba ? 1 : a->sh[i - (nb - ba)], y = i < nb - bb ? 1 : b->sh[i - (nb - bb)];
        if (x != y && x != 1 && y != 1) a_error("ValueError", "matmul: batch shapes do not broadcast");
        bsh[i] = x == 1 ? y : x;
    }
    int64_t osh[MAXD]; int ond = 0;
    for (int i = 0; i < nb; i++) osh[ond++] = bsh[i];
    if (!a1) osh[ond++] = M;
    if (!b1) osh[ond++] = N;
    ATensor *o = t_new(dt, ond, osh);
    AV ac = a->dt == dt ? av : tcopy(a, dt); int sp = a_ntmp; A_PUSH(ac);
    AV bc = b->dt == dt ? bv : tcopy(b, dt); A_PUSH(bc);
    A_PUSH(a_obj(o));
    int64_t nbatch = prod(bsh, nb);
    int64_t abatch = prod(a->sh, ba), bbatch = prod(b->sh, bb);
    for (int64_t q = 0; q < nbatch; q++) {
        int64_t ia = 0, ib = 0, rem = q, mul_a = 1, mul_b = 1;
        for (int i = nb - 1; i >= 0; i--) {
            int64_t d = rem % bsh[i]; rem /= bsh[i];
            int ai = i - (nb - ba), bi = i - (nb - bb);
            if (ai >= 0) { if (a->sh[ai] != 1) ia += d * mul_a; mul_a *= a->sh[ai]; }
            if (bi >= 0) { if (b->sh[bi] != 1) ib += d * mul_b; mul_b *= b->sh[bi]; }
        }
        (void)abatch; (void)bbatch;
        if (dt == DT_F64) gemm((double *)TT(ac)->p + ia * M * K, (double *)TT(bc)->p + ib * K * N, (double *)o->p + q * M * N, M, N, K);
        else gemm_f((float *)TT(ac)->p + ia * M * K, (float *)TT(bc)->p + ib * K * N, (float *)o->p + q * M * N, M, N, K);
    }
    a_ntmp = sp;
    if (ond == 0) return t_scalar(o, 0);
    return a_obj(o);
}

/* ------------------------------------------------------------------ reductions */
enum { R_SUM, R_MEAN, R_MAX, R_MIN, R_PROD, R_ARGMAX, R_ARGMIN, R_ANY, R_ALL, R_VAR, R_STD, R_LOGSUMEXP };
static int norm_axis(int ax, int nd) { if (ax < 0) ax += nd; if (ax < 0 || ax >= nd) a_error("ValueError", "axis %d is out of range for %d dimensions", ax, nd); return ax; }
static AV reduce(int op, AV v, AV axis, int keep, int ddof) {
    ATensor *a = TT(v);
    if (axis.t == T_NONE) {
        double acc; int64_t best = 0;
        if (a->n == 0 && (op == R_MAX || op == R_MIN || op == R_ARGMAX || op == R_ARGMIN)) a_error("ValueError", "max/min of an empty tensor");
        switch (op) {
        case R_SUM: case R_MEAN: { double s = 0, c = 0; for (int64_t i = 0; i < a->n; i++) { double y = t_get(a, i) - c, t = s + y; c = (t - s) - y; s = t; } acc = op == R_MEAN ? s / (double)a->n : s;
            if (op == R_SUM && a->dt == DT_I64) return a_int((int64_t)s); if (op == R_SUM && a->dt == DT_BOOL) return a_int((int64_t)s); break; }
        case R_MAX: acc = t_get(a, 0); for (int64_t i = 1; i < a->n; i++) { double x = t_get(a, i); if (x > acc || isnan(x)) acc = x; } if (a->dt == DT_I64) return a_int((int64_t)acc); break;
        case R_MIN: acc = t_get(a, 0); for (int64_t i = 1; i < a->n; i++) { double x = t_get(a, i); if (x < acc || isnan(x)) acc = x; } if (a->dt == DT_I64) return a_int((int64_t)acc); break;
        case R_PROD: acc = 1; for (int64_t i = 0; i < a->n; i++) acc *= t_get(a, i); if (a->dt == DT_I64) return a_int((int64_t)acc); break;
        case R_ARGMAX: for (int64_t i = 1; i < a->n; i++) if (t_get(a, i) > t_get(a, best)) best = i; return a_int(best);
        case R_ARGMIN: for (int64_t i = 1; i < a->n; i++) if (t_get(a, i) < t_get(a, best)) best = i; return a_int(best);
        case R_ANY: for (int64_t i = 0; i < a->n; i++) if (t_get(a, i) != 0) return a_bool(1); return a_bool(0);
        case R_ALL: for (int64_t i = 0; i < a->n; i++) if (t_get(a, i) == 0) return a_bool(0); return a_bool(1);
        case R_VAR: case R_STD: { double m = 0; for (int64_t i = 0; i < a->n; i++) m += t_get(a, i); m /= (double)a->n; double s = 0; for (int64_t i = 0; i < a->n; i++) { double d = t_get(a, i) - m; s += d * d; } acc = s / (double)(a->n - ddof); if (op == R_STD) acc = sqrt(acc); break; }
        case R_LOGSUMEXP: { double m = -INFINITY; for (int64_t i = 0; i < a->n; i++) if (t_get(a, i) > m) m = t_get(a, i); if (!isfinite(m)) m = 0; double s = 0; for (int64_t i = 0; i < a->n; i++) s += exp(t_get(a, i) - m); acc = log(s) + m; break; }
        default: acc = NAN;
        }
        if (keep) { int64_t sh[MAXD]; for (int i = 0; i < a->nd; i++) sh[i] = 1; ATensor *o = t_new(a->dt == DT_F32 ? DT_F32 : DT_F64, a->nd, sh); t_set(o, 0, acc); return a_obj(o); }
        return a_float(acc);
    }
    int ax = norm_axis((int)axis.i, a->nd);
    int64_t outer = prod(a->sh, ax), len = a->sh[ax], inner = prod(a->sh + ax + 1, a->nd - ax - 1);
    int64_t osh[MAXD]; int ond = 0;
    for (int i = 0; i < a->nd; i++) { if (i == ax) { if (keep) osh[ond++] = 1; } else osh[ond++] = a->sh[i]; }
    int dt = (op == R_ARGMAX || op == R_ARGMIN) ? DT_I64 : (op == R_ANY || op == R_ALL) ? DT_BOOL :
             ((op == R_SUM || op == R_MAX || op == R_MIN || op == R_PROD) && (a->dt == DT_I64 || a->dt == DT_BOOL)) ? DT_I64 :
             a->dt == DT_F32 ? DT_F32 : DT_F64;
    ATensor *o = t_new(dt, ond, osh);
    for (int64_t p = 0; p < outer; p++) for (int64_t q = 0; q < inner; q++) {
        int64_t base = p * len * inner + q;
        double acc = 0; int64_t best = 0;
        switch (op) {
        case R_SUM: case R_MEAN: for (int64_t k = 0; k < len; k++) acc += t_get(a, base + k * inner); if (op == R_MEAN) acc /= (double)len; break;
        case R_MAX: acc = t_get(a, base); for (int64_t k = 1; k < len; k++) { double x = t_get(a, base + k * inner); if (x > acc) acc = x; } break;
        case R_MIN: acc = t_get(a, base); for (int64_t k = 1; k < len; k++) { double x = t_get(a, base + k * inner); if (x < acc) acc = x; } break;
        case R_PROD: acc = 1; for (int64_t k = 0; k < len; k++) acc *= t_get(a, base + k * inner); break;
        case R_ARGMAX: for (int64_t k = 1; k < len; k++) if (t_get(a, base + k * inner) > t_get(a, base + best * inner)) best = k; acc = (double)best; break;
        case R_ARGMIN: for (int64_t k = 1; k < len; k++) if (t_get(a, base + k * inner) < t_get(a, base + best * inner)) best = k; acc = (double)best; break;
        case R_ANY: for (int64_t k = 0; k < len; k++) if (t_get(a, base + k * inner) != 0) { acc = 1; break; } break;
        case R_ALL: acc = 1; for (int64_t k = 0; k < len; k++) if (t_get(a, base + k * inner) == 0) { acc = 0; break; } break;
        case R_VAR: case R_STD: { double m = 0; for (int64_t k = 0; k < len; k++) m += t_get(a, base + k * inner); m /= (double)len;
            for (int64_t k = 0; k < len; k++) { double d = t_get(a, base + k * inner) - m; acc += d * d; } acc /= (double)(len - ddof); if (op == R_STD) acc = sqrt(acc); break; }
        case R_LOGSUMEXP: { double m = -INFINITY; for (int64_t k = 0; k < len; k++) { double x = t_get(a, base + k * inner); if (x > m) m = x; } if (!isfinite(m)) m = 0;
            for (int64_t k = 0; k < len; k++) acc += exp(t_get(a, base + k * inner) - m); acc = log(acc) + m; break; }
        }
        t_set(o, p * inner + q, acc);
    }
    if (ond == 0) return t_scalar(o, 0);
    return a_obj(o);
}

/* ------------------------------------------------------------------ shapes */
static AV reshape(AV v, int nd, int64_t *sh) {
    ATensor *a = TT(v); int64_t known = 1; int neg = -1;
    for (int i = 0; i < nd; i++) { if (sh[i] == -1) { if (neg >= 0) a_error("ValueError", "only one -1 in a shape"); neg = i; } else known *= sh[i]; }
    if (neg >= 0) { if (known == 0 || a->n % known) a_error("ValueError", "cannot reshape %lld elements", (long long)a->n); sh[neg] = a->n / known; known *= sh[neg]; }
    if (known != a->n) a_error("ValueError", "cannot reshape %lld elements into a shape of %lld", (long long)a->n, (long long)known);
    ATensor *o = t_new(a->dt, nd, sh);
    memcpy(o->p, a->p, (size_t)a->n * (size_t)dt_size[a->dt]);
    return a_obj(o);
}
static AV permute(AV v, const int *perm) {
    ATensor *a = TT(v); int nd = a->nd; int64_t osh[MAXD], ist[MAXD], ost[MAXD], idx[MAXD] = {0};
    int64_t s = 1; for (int i = nd - 1; i >= 0; i--) { ist[i] = s; s *= a->sh[i]; }
    for (int i = 0; i < nd; i++) { osh[i] = a->sh[perm[i]]; ost[i] = ist[perm[i]]; }
    ATensor *o = t_new(a->dt, nd, osh);
    int es = dt_size[a->dt]; int64_t src = 0;
    for (int64_t i = 0; i < o->n; i++) {
        memcpy((char *)o->p + i * es, (char *)a->p + src * es, (size_t)es);
        for (int d = nd - 1; d >= 0; d--) { idx[d]++; src += ost[d]; if (idx[d] < osh[d]) break; src -= ost[d] * osh[d]; idx[d] = 0; }
    }
    return a_obj(o);
}
static AV transpose2(AV v) {
    ATensor *a = TT(v);
    if (a->nd < 2) return v;
    int perm[MAXD]; for (int i = 0; i < a->nd; i++) perm[i] = i;
    perm[a->nd - 1] = a->nd - 2; perm[a->nd - 2] = a->nd - 1;
    if (a->nd > 2) return permute(v, perm);
    int full[MAXD] = {1, 0}; return permute(v, full);
}
static AV t_concat(AV items, int ax, int stack) {
    AList *l = (AList *)items.o;
    if (!l->len) a_error("ValueError", "nothing to concatenate");
    int sp = a_ntmp;
    AList *ts = (AList *)a_list(l->len, l->v).o; A_PUSH(a_obj(ts));
    int dt = DT_BOOL;
    for (int64_t i = 0; i < l->len; i++) {
        ts->v[i] = as_tensor(l->v[i]);
        int d = TT(ts->v[i])->dt;
        if (d == DT_F64 || (d == DT_F32 && dt != DT_F64)) dt = d; else if (d == DT_I64 && dt == DT_BOOL) dt = d;
    }
    ATensor *f = TT(ts->v[0]);
    int nd = f->nd + (stack ? 1 : 0);
    ax = norm_axis(ax, nd);
    int64_t osh[MAXD];
    for (int i = 0, j = 0; i < nd; i++) { if (stack && i == ax) osh[i] = (int64_t)l->len; else osh[i] = f->sh[j++]; }
    if (!stack) {
        osh[ax] = 0;
        for (int64_t i = 0; i < l->len; i++) {
            ATensor *t = TT(ts->v[i]);
            if (t->nd != f->nd) a_error("ValueError", "concat needs tensors with the same number of dimensions");
            for (int d = 0; d < nd; d++) if (d != ax && t->sh[d] != f->sh[d]) a_error("ValueError", "concat: shapes differ off the joining axis");
            osh[ax] += t->sh[ax];
        }
    } else for (int64_t i = 0; i < l->len; i++) {
        ATensor *t = TT(ts->v[i]);
        if (t->nd != f->nd) a_error("ValueError", "stack needs tensors of the same shape");
        for (int d = 0; d < f->nd; d++) if (t->sh[d] != f->sh[d]) a_error("ValueError", "stack needs tensors of the same shape");
    }
    ATensor *o = t_new(dt, nd, osh);
    int64_t outer = prod(osh, ax), off = 0;
    int64_t inner_o = prod(osh + ax + 1, nd - ax - 1);
    for (int64_t i = 0; i < l->len; i++) {
        ATensor *t = TT(ts->v[i]);
        int64_t len = stack ? 1 : t->sh[ax];
        for (int64_t p = 0; p < outer; p++) for (int64_t k = 0; k < len * inner_o; k++)
            t_set(o, p * osh[ax] * inner_o + off * inner_o + k, t_get(t, p * len * inner_o + k));
        off += len;
    }
    a_ntmp = sp;
    return a_obj(o);
}

/* ------------------------------------------------------------------ indexing */
static int64_t idx_norm(int64_t i, int64_t n) {
    int64_t k = i < 0 ? i + n : i;
    if (k < 0 || k >= n) a_error("IndexError", "index %lld is out of range for an axis of size %lld", (long long)i, (long long)n);
    return k;
}
typedef struct { int kind; int64_t start, step, len; AV list; } Sel;   /* 0 an index, 1 a range, 2 a list of positions */
static void slice_sel(AV s, int64_t n, Sel *o) {
    ASlice *sl = (ASlice *)s.o;
    int64_t step = sl->step.t == T_NONE ? 1 : sl->step.i, a, b;
    if (step == 0) a_error("ValueError", "slice step cannot be zero");
    if (step > 0) {
        a = sl->lo.t == T_NONE ? 0 : sl->lo.i; b = sl->hi.t == T_NONE ? n : sl->hi.i;
        if (a < 0) a += n; if (a < 0) a = 0; if (a > n) a = n; if (b < 0) b += n; if (b < 0) b = 0; if (b > n) b = n;
        o->len = b > a ? (b - a + step - 1) / step : 0;
    } else {
        a = sl->lo.t == T_NONE ? n - 1 : sl->lo.i; b = sl->hi.t == T_NONE ? -1 - n : sl->hi.i;
        if (a < 0) a += n; if (a < -1) a = -1; if (a >= n) a = n - 1; if (b < 0) b += n; if (b < -1) b = -1; if (b >= n) b = n - 1;
        o->len = a > b ? (a - b - step - 1) / (-step) : 0;
    }
    o->kind = 1; o->start = a; o->step = step;
}
static int build_sels(ATensor *a, AV index, Sel *sels, int *newaxes) {
    AV items[MAXD + 1]; int ni = 0;
    if (a_is(index, O_TUPLE)) { AList *l = (AList *)index.o; if (l->len > MAXD) a_error("IndexError", "too many indices"); for (int64_t i = 0; i < l->len; i++) items[ni++] = l->v[i]; }
    else items[ni++] = index;
    if (ni > a->nd) a_error("IndexError", "too many indices for a tensor of %d dimensions", a->nd);
    for (int d = 0; d < a->nd; d++) {
        if (d >= ni) { sels[d] = (Sel){1, 0, 1, a->sh[d], A_NONE}; continue; }
        AV it = items[d];
        if (it.t == T_INT || it.t == T_BOOL) sels[d] = (Sel){0, idx_norm(it.i, a->sh[d]), 0, 1, A_NONE};
        else if (a_is(it, O_SLICE)) slice_sel(it, a->sh[d], &sels[d]);
        else if (a_is(it, O_LIST) || is_t(it)) {
            AV t = is_t(it) ? it : t_from(it, -1);
            if (TT(t)->dt == DT_BOOL) {             /* a mask: the positions where it is true */
                if (TT(t)->nd != 1 && !(ni == 1 && TT(t)->nd == a->nd)) a_error("IndexError", "a boolean mask must match the tensor");
                if (TT(t)->nd > 1) return -1;
                AList *pos = (AList *)a_list(0, NULL).o;
                for (int64_t i = 0; i < TT(t)->n; i++) if (t_get(TT(t), i) != 0) a_list_append(pos, a_int(i));
                t = t_from(a_obj(pos), DT_I64);
            }
            sels[d] = (Sel){2, 0, 0, TT(t)->n, t};
            for (int64_t i = 0; i < TT(t)->n; i++) idx_norm((int64_t)t_get(TT(t), i), a->sh[d]);
        } else a_error("TypeError", "a tensor index is a number, a slice, or a list of positions");
    }
    *newaxes = 0;
    return 0;
}
static int64_t sel_pos(Sel *s, int64_t k, int64_t n) {
    if (s->kind == 0) return s->start;
    if (s->kind == 1) return s->start + k * s->step;
    int64_t i = (int64_t)t_get(TT(s->list), k); return i < 0 ? i + n : i;
}

/* NumPy's rule for two or more lists of positions: they go together (zipped), not as an outer product */
static int n_adv(Sel *sels, int nd) { int k = 0; for (int d = 0; d < nd; d++) if (sels[d].kind == 2) k++; return k; }
static void zip_walk(ATensor *a, Sel *sels, ATensor *o, ATensor *src, int mode) {
    /* mode 0: o[...] = a[sel]; 1: a[sel] = src (broadcast); 2: a[sel] += src */
    int nd = a->nd; int64_t k = -1;
    for (int d = 0; d < nd; d++) if (sels[d].kind == 2) { if (k < 0 || k == 1) k = sels[d].len; else if (sels[d].len != k && sels[d].len != 1) a_error("IndexError", "lists of positions of different lengths"); }
    int64_t st[MAXD], s = 1; for (int d = nd - 1; d >= 0; d--) { st[d] = s; s *= a->sh[d]; }
    int64_t rest_sh[MAXD]; int nr = 0; int rest_ax[MAXD];
    for (int d = 0; d < nd; d++) if (sels[d].kind == 1) { rest_ax[nr] = d; rest_sh[nr++] = sels[d].len; }
    int64_t inner = prod(rest_sh, nr), out_i = 0;
    int64_t osh[MAXD]; osh[0] = k; memcpy(osh + 1, rest_sh, sizeof(int64_t) * (size_t)nr);
    int64_t sst[MAXD];
    if (src) { ATensor shp; shp.nd = (uint8_t)(nr + 1); memcpy(shp.sh, osh, sizeof osh); int64_t tmp[MAXD]; bshape(&shp, src, tmp); bstrides(src, nr + 1, osh, sst); }
    for (int64_t p = 0; p < k; p++) {
        int64_t base = 0;
        for (int d = 0; d < nd; d++) {
            if (sels[d].kind == 2) base += sel_pos(&sels[d], sels[d].len == 1 ? 0 : p, a->sh[d]) * st[d];
            else if (sels[d].kind == 0) base += sels[d].start * st[d];
        }
        int64_t cnt[MAXD] = {0};
        for (int64_t q = 0; q < inner; q++) {
            int64_t off = base;
            for (int r = 0; r < nr; r++) off += sel_pos(&sels[rest_ax[r]], cnt[r], a->sh[rest_ax[r]]) * st[rest_ax[r]];
            if (mode == 0) t_set(o, out_i, t_get(a, off));
            else {
                int64_t si = p * (sst[0]);
                for (int r = 0; r < nr; r++) si += cnt[r] * sst[r + 1];
                double x = t_get(src, si);
                t_set(a, off, mode == 2 ? t_get(a, off) + x : x);
            }
            out_i++;
            for (int r = nr - 1; r >= 0; r--) { if (++cnt[r] < rest_sh[r]) break; cnt[r] = 0; }
        }
    }
}

static AV t_getitem(AV v, AV index) {
    ATensor *a = TT(v);
    if ((index.t == T_INT || index.t == T_BOOL) && a->nd == 1) return t_scalar(a, idx_norm(index.i, a->sh[0]));
    if (is_t(index) && TT(index)->dt == DT_BOOL && TT(index)->nd == a->nd && a->nd > 1) {
        ATensor *m = TT(index); int64_t cnt = 0;
        for (int64_t i = 0; i < m->n; i++) if (t_get(m, i) != 0) cnt++;
        ATensor *o = t_new(a->dt, 1, &cnt); int64_t k = 0;
        for (int64_t i = 0; i < a->n; i++) if (t_get(m, i) != 0) t_set(o, k++, t_get(a, i));
        return a_obj(o);
    }
    Sel sels[MAXD]; int na;
    int sp = a_ntmp;
    build_sels(a, index, sels, &na);
    for (int d = 0; d < a->nd; d++) if (sels[d].kind == 2) A_PUSH(sels[d].list);
    if (n_adv(sels, a->nd) >= 2) {
        int64_t k = 1, osh[MAXD]; int ond = 1;
        for (int d = 0; d < a->nd; d++) if (sels[d].kind == 2 && sels[d].len != 1) k = sels[d].len;
        osh[0] = k; for (int d = 0; d < a->nd; d++) if (sels[d].kind == 1) osh[ond++] = sels[d].len;
        ATensor *o = t_new(a->dt, ond, osh);
        zip_walk(a, sels, o, NULL, 0);
        a_ntmp = sp; return a_obj(o);
    }
    int64_t osh[MAXD]; int ond = 0;
    for (int d = 0; d < a->nd; d++) if (sels[d].kind != 0) osh[ond++] = sels[d].len;
    ATensor *o = t_new(a->dt, ond, osh);
    int64_t st[MAXD], s = 1; for (int d = a->nd - 1; d >= 0; d--) { st[d] = s; s *= a->sh[d]; }
    int64_t cnt[MAXD] = {0};
    int es = dt_size[a->dt];
    for (int64_t i = 0; i < o->n; i++) {
        int64_t src = 0;
        for (int d = 0; d < a->nd; d++) src += sel_pos(&sels[d], cnt[d], a->sh[d]) * st[d];
        memcpy((char *)o->p + i * es, (char *)a->p + src * es, (size_t)es);
        for (int d = a->nd - 1; d >= 0; d--) { if (sels[d].kind == 0) continue; if (++cnt[d] < sels[d].len) break; cnt[d] = 0; }
    }
    a_ntmp = sp;
    if (ond == 0) return t_scalar(o, 0);
    return a_obj(o);
}
static void t_setitem(AV v, AV index, AV val, int add) {
    ATensor *a = TT(v);
    if (is_t(index) && TT(index)->dt == DT_BOOL && TT(index)->nd == a->nd && a->nd > 1) {
        ATensor *m = TT(index); AV src = as_tensor(val); int64_t k = 0;
        for (int64_t i = 0; i < a->n; i++) if (t_get(m, i) != 0) { double x = t_get(TT(src), TT(src)->n == 1 ? 0 : k++); t_set(a, i, add ? t_get(a, i) + x : x); }
        return;
    }
    Sel sels[MAXD]; int na;
    int sp = a_ntmp;
    build_sels(a, index, sels, &na);
    for (int d = 0; d < a->nd; d++) if (sels[d].kind == 2) A_PUSH(sels[d].list);
    if (n_adv(sels, a->nd) >= 2) {
        AV src = as_tensor(val); A_PUSH(src);
        zip_walk(a, sels, NULL, TT(src), add ? 2 : 1);
        a_ntmp = sp; return;
    }
    int64_t osh[MAXD]; int ond = 0;
    for (int d = 0; d < a->nd; d++) if (sels[d].kind != 0) osh[ond++] = sels[d].len;
    AV src = as_tensor(val); A_PUSH(src);
    ATensor *s = TT(src);
    int64_t total = prod(osh, ond);
    /* the value broadcasts against the selection */
    int64_t sst[MAXD]; ATensor shp; shp.nd = (uint8_t)ond; memcpy(shp.sh, osh, sizeof osh);
    int64_t tmp[MAXD]; bshape(&shp, s, tmp);
    bstrides(s, ond, osh, sst);
    int64_t st[MAXD], q = 1; for (int d = a->nd - 1; d >= 0; d--) { st[d] = q; q *= a->sh[d]; }
    int64_t cnt[MAXD] = {0}, ocnt[MAXD] = {0};
    for (int64_t i = 0; i < total; i++) {
        int64_t dst = 0, si = 0;
        for (int d = 0; d < a->nd; d++) dst += sel_pos(&sels[d], cnt[d], a->sh[d]) * st[d];
        for (int d = 0; d < ond; d++) si += ocnt[d] * sst[d];
        double x = t_get(s, si);
        t_set(a, dst, add ? t_get(a, dst) + x : x);
        for (int d = a->nd - 1, od = ond - 1; d >= 0; d--) { if (sels[d].kind == 0) continue; ocnt[od]++; if (++cnt[d] < sels[d].len) break; cnt[d] = 0; ocnt[od] = 0; od--; }
    }
    a_ntmp = sp;
}

/* ================================================================== automatic differentiation */
enum { N_LEAF, N_ADD, N_SUB, N_MUL, N_DIV, N_POW, N_NEG, N_MATMUL, N_UNARY, N_SUM, N_MEAN, N_RESHAPE, N_TRANSPOSE,
       N_INDEX, N_SOFTMAX, N_LOGSOFTMAX, N_MAXRED, N_CONCAT, N_STACK, N_WHERE, N_MAXIMUM, N_PERMUTE, N_ABS, N_SQRT_Q };
static int64_t node_ids = 0;
static AV node_new(AV val, int op, AV a, AV b, AV x) {
    ANode *n = (ANode *)a_alloc(sizeof(ANode), O_NODE);
    n->val = val; n->op = op; n->a = a; n->b = b; n->x = x; n->grad = A_NONE; n->id = ++node_ids;
    return a_obj(n);
}
static AV val_of(AV v) { return is_n(v) ? ND(v)->val : v; }

/* every operation on values, whatever their kind */
AV a_tensor_binop(int op, AV a, AV b);
static AV gv_add(AV a, AV b) { return a_binop(OP_ADD, a, b); }
static AV gv_mul(AV a, AV b) { return a_binop(OP_MUL, a, b); }
static AV gv_div(AV a, AV b) { return a_binop(OP_DIV, a, b); }
static AV gv_neg(AV a) { return a_unop(OP_NEG, a); }

/* a gradient for a value of a smaller shape: summed over the broadcast axes */
static AV unbroadcast(AV g, AV like) {
    if (!is_t(g)) return g;
    if (!is_t(like)) return reduce(R_SUM, g, A_NONE, 0, 0);
    ATensor *G = TT(g), *L = TT(like);
    if (G->nd == L->nd) { int same = 1; for (int i = 0; i < G->nd; i++) if (G->sh[i] != L->sh[i]) same = 0; if (same) return g; }
    int sp = a_ntmp; A_PUSH(g);
    while (TT(g)->nd > L->nd) { g = reduce(R_SUM, g, a_int(0), 0, 0); a_tmp[sp] = g; }
    for (int i = 0; i < L->nd; i++) if (L->sh[i] == 1 && TT(g)->sh[i] != 1) { g = reduce(R_SUM, g, a_int(i), 1, 0); a_tmp[sp] = g; }
    a_ntmp = sp;
    return g;
}
static void acc_grad(AV node, AV g) {
    if (!is_n(node)) return;
    ANode *n = ND(node);
    if (n->grad.t == T_NONE) n->grad = g; else n->grad = gv_add(n->grad, g);
}
static void collect_nodes(AV v, AList *out) {
    if (!is_n(v)) return;
    ANode *n = ND(v);
    if (n->seen) return;
    n->seen = 1;
    collect_nodes(n->a, out); collect_nodes(n->b, out);
    if (n->op == N_CONCAT || n->op == N_STACK) { AList *l = (AList *)n->a.o; (void)l; }
    a_list_append(out, v);
}
static AV unary_grad(int u, AV x, AV y, AV g) {     /* d op(x) / dx times g, from x and y = op(x) */
    AV d;
    int sp = a_ntmp;
    switch (u) {
    case U_EXP: d = y; break;
    case U_LOG: d = a_binop(OP_DIV, a_float(1.0), x); break;
    case U_SQRT: d = a_binop(OP_DIV, a_float(0.5), y); break;
    case U_SIN: d = a_tensor_unary(U_COS, x); break;
    case U_COS: d = a_unop(OP_NEG, a_tensor_unary(U_SIN, x)); break;
    case U_TAN: { AV c = a_tensor_unary(U_COS, x); A_PUSH(c); d = a_binop(OP_DIV, a_float(1.0), a_binop(OP_MUL, c, c)); break; }
    case U_TANH: { AV yy = a_binop(OP_MUL, y, y); A_PUSH(yy); d = a_binop(OP_SUB, a_float(1.0), yy); break; }
    case U_RELU: d = a_binop(OP_MUL, a_tensor_cmp(OP_GT, x, a_float(0.0)), a_float(1.0)); break;
    case U_SIGMOID: { AV one = a_binop(OP_SUB, a_float(1.0), y); A_PUSH(one); d = a_binop(OP_MUL, y, one); break; }
    case U_SOFTPLUS: d = a_tensor_unary(U_SIGMOID, x); break;
    case U_LOG1P: d = a_binop(OP_DIV, a_float(1.0), a_binop(OP_ADD, x, a_float(1.0))); break;
    case U_EXPM1: d = a_binop(OP_ADD, y, a_float(1.0)); break;
    case U_ATAN: d = a_binop(OP_DIV, a_float(1.0), a_binop(OP_ADD, a_binop(OP_MUL, x, x), a_float(1.0))); break;
    case U_SINH: d = a_tensor_unary(U_COSH, x); break;
    case U_COSH: d = a_tensor_unary(U_SINH, x); break;
    case U_SQUARE: d = a_binop(OP_MUL, x, a_float(2.0)); break;
    case U_ABS: d = a_tensor_unary(U_SIGN, x); break;
    case U_SILU: { AV s = a_tensor_unary(U_SIGMOID, x); A_PUSH(s); AV t = a_binop(OP_MUL, x, a_binop(OP_SUB, a_float(1.0), s)); A_PUSH(t); d = a_binop(OP_MUL, s, a_binop(OP_ADD, t, a_float(1.0))); break; }
    case U_GELU: {
        /* d/dx 0.5 x (1 + tanh(k (x + c x^3))) */
        if (!is_t(x)) { double v = x.t == T_FLOAT ? x.f : (double)x.i, k = 0.7978845608028654, u = k * (v + 0.044715 * v * v * v), t = tanh(u);
            d = a_float(0.5 * (1 + t) + 0.5 * v * (1 - t * t) * k * (1 + 3 * 0.044715 * v * v)); break; }
        ATensor *X = TT(x), *o = t_like(X, X->dt == DT_F32 ? DT_F32 : DT_F64);
        for (int64_t i = 0; i < X->n; i++) { double v = t_get(X, i), k = 0.7978845608028654, u = k * (v + 0.044715 * v * v * v), t = tanh(u);
            t_set(o, i, 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * k * (1 + 3 * 0.044715 * v * v)); }
        d = a_obj(o); break;
    }
    case U_GELU_ERF: {
        if (!is_t(x)) { double v = x.t == T_FLOAT ? x.f : (double)x.i; d = a_float(0.5 * (1 + erf(v * 0.7071067811865476)) + v * exp(-0.5 * v * v) * 0.3989422804014327); break; }
        ATensor *X = TT(x), *o = t_like(X, X->dt == DT_F32 ? DT_F32 : DT_F64);
        for (int64_t i = 0; i < X->n; i++) { double v = t_get(X, i); t_set(o, i, 0.5 * (1 + erf(v * 0.7071067811865476)) + v * exp(-0.5 * v * v) * 0.3989422804014327); }
        d = a_obj(o); break;
    }
    default: a_error("NotImplementedError", "no derivative for this function");
    }
    A_PUSH(d);
    AV r = a_binop(OP_MUL, g, d);
    a_ntmp = sp;
    return r;
}
static AV expand_axis(AV g, AV like, AV axis, int keep) {   /* the gradient of a reduction, back to the input's shape */
    if (!is_t(like)) return g;
    ATensor *L = TT(like);
    if (axis.t != T_NONE && !keep && is_t(g)) {
        int ax = norm_axis((int)axis.i, L->nd); int64_t sh[MAXD]; int nd = 0;
        for (int i = 0; i < L->nd; i++) sh[nd++] = i == ax ? 1 : L->sh[i];
        g = reshape(g, nd, sh);
    } else if (axis.t != T_NONE && !keep) {
        int ax = norm_axis((int)axis.i, L->nd); int64_t sh[MAXD]; int nd = 0;
        for (int i = 0; i < L->nd; i++) sh[nd++] = i == ax ? 1 : L->sh[i];
        g = reshape(as_tensor(g), nd, sh);
    }
    int sp = a_ntmp; A_PUSH(g);
    AV z = a_obj(t_like(L, L->dt == DT_F32 ? DT_F32 : DT_F64)); A_PUSH(z);
    AV r = a_binop(OP_ADD, z, g);
    a_ntmp = sp;
    return r;
}
static AV t_softmax(AV v, int ax, int logsm);
static void backward_one(ANode *n) {
    AV g = n->grad, a = n->a, b = n->b, va = val_of(a), vb = val_of(b);
    int sp = a_ntmp; A_PUSH(g);
    switch (n->op) {
    case N_ADD: acc_grad(a, unbroadcast(g, va)); acc_grad(b, unbroadcast(g, vb)); break;
    case N_SUB: acc_grad(a, unbroadcast(g, va)); if (is_n(b)) acc_grad(b, unbroadcast(gv_neg(g), vb)); break;
    case N_MUL: if (is_n(a)) acc_grad(a, unbroadcast(gv_mul(g, vb), va)); if (is_n(b)) acc_grad(b, unbroadcast(gv_mul(g, va), vb)); break;
    case N_DIV:
        if (is_n(a)) acc_grad(a, unbroadcast(gv_div(g, vb), va));
        if (is_n(b)) { AV t = gv_div(gv_mul(g, va), gv_mul(vb, vb)); A_PUSH(t); acc_grad(b, unbroadcast(gv_neg(t), vb)); }
        break;
    case N_POW:
        if (is_n(a)) { AV pm1 = a_binop(OP_SUB, vb, a_float(1.0)); A_PUSH(pm1); AV t = gv_mul(vb, a_binop(OP_POW, va, pm1)); A_PUSH(t); acc_grad(a, unbroadcast(gv_mul(g, t), va)); }
        if (is_n(b)) { AV t = gv_mul(n->val, a_tensor_unary(U_LOG, va)); A_PUSH(t); acc_grad(b, unbroadcast(gv_mul(g, t), vb)); }
        break;
    case N_NEG: acc_grad(a, gv_neg(g)); break;
    case N_MATMUL: {
        ATensor *A = TT(va), *B = TT(vb);
        if (is_n(a)) {
            AV ga;
            if (A->nd == 1 && B->nd == 1) ga = a_binop(OP_MUL, g, vb);
            else if (A->nd == 1) ga = matmul(transpose2(vb), as_tensor(g)), ga = is_t(ga) ? ga : ga;
            else if (B->nd == 1) { AV gg = as_tensor(g); A_PUSH(gg); int64_t sh[MAXD]; int nd = TT(gg)->nd; memcpy(sh, TT(gg)->sh, sizeof(int64_t) * (size_t)nd); sh[nd] = 1; AV g2 = reshape(gg, nd + 1, sh); A_PUSH(g2);
                int64_t bs[2] = {1, B->sh[0]}; AV b2 = reshape(vb, 2, bs); A_PUSH(b2); ga = matmul(g2, b2); }
            else { AV bt = transpose2(vb); A_PUSH(bt); ga = matmul(g, bt); }
            A_PUSH(ga); acc_grad(a, unbroadcast(ga, va));
        }
        if (is_n(b)) {
            AV gb;
            if (A->nd == 1 && B->nd == 1) gb = a_binop(OP_MUL, g, va);
            else if (A->nd == 1) { int64_t as[2] = {A->sh[0], 1}; AV a2 = reshape(va, 2, as); A_PUSH(a2); AV gg = as_tensor(g); A_PUSH(gg); int64_t gs[2] = {1, TT(gg)->n}; AV g2 = reshape(gg, 2, gs); A_PUSH(g2); gb = matmul(a2, g2); }
            else if (B->nd == 1) { AV at = transpose2(va); A_PUSH(at); gb = matmul(at, g); }
            else { AV at = transpose2(va); A_PUSH(at); gb = matmul(at, g); }
            A_PUSH(gb); acc_grad(b, unbroadcast(gb, vb));
        }
        break;
    }
    case N_UNARY: acc_grad(a, unary_grad((int)n->x.i, va, n->val, g)); break;
    case N_SUM: acc_grad(a, expand_axis(g, va, ((AList *)n->x.o)->v[0], (int)((AList *)n->x.o)->v[1].i)); break;
    case N_MEAN: {
        AV axis = ((AList *)n->x.o)->v[0]; int keep = (int)((AList *)n->x.o)->v[1].i;
        double cnt = is_t(va) ? (axis.t == T_NONE ? (double)TT(va)->n : (double)TT(va)->sh[norm_axis((int)axis.i, TT(va)->nd)]) : 1.0;
        AV e = expand_axis(g, va, axis, keep); A_PUSH(e);
        acc_grad(a, a_binop(OP_DIV, e, a_float(cnt))); break;
    }
    case N_MAXRED: {         /* the gradient goes to the places that held the maximum */
        AV axis = ((AList *)n->x.o)->v[0]; int keep = (int)((AList *)n->x.o)->v[1].i;
        AV m = expand_axis(n->val, va, axis, keep); A_PUSH(m);
        AV mask = a_binop(OP_MUL, a_tensor_cmp(OP_EQ, va, m), a_float(1.0)); A_PUSH(mask);
        AV e = expand_axis(g, va, axis, keep); A_PUSH(e);
        acc_grad(a, a_binop(OP_MUL, e, mask)); break;
    }
    case N_RESHAPE: { AV gg = as_tensor(g); A_PUSH(gg); acc_grad(a, is_t(va) ? reshape(gg, TT(va)->nd, TT(va)->sh) : reduce(R_SUM, gg, A_NONE, 0, 0)); break; }
    case N_TRANSPOSE: acc_grad(a, transpose2(g)); break;
    case N_PERMUTE: {
        AList *pl = (AList *)n->x.o; int perm[MAXD], inv[MAXD];
        for (int64_t i = 0; i < pl->len; i++) perm[i] = (int)pl->v[i].i;
        for (int64_t i = 0; i < pl->len; i++) inv[perm[i]] = (int)i;
        acc_grad(a, permute(g, inv)); break;
    }
    case N_INDEX: {
        AV z = a_obj(t_like(TT(va), TT(va)->dt == DT_F32 ? DT_F32 : DT_F64)); A_PUSH(z);
        t_setitem(z, n->x, g, 1);
        acc_grad(a, z); break;
    }
    case N_SOFTMAX: {        /* g_x = y * (g - sum(g * y, axis)) */
        int ax = (int)n->x.i; AV y = n->val;
        AV gy = gv_mul(g, y); A_PUSH(gy);
        AV s = reduce(R_SUM, gy, a_int(ax), 1, 0); A_PUSH(s);
        AV d = a_binop(OP_SUB, g, s); A_PUSH(d);
        acc_grad(a, gv_mul(y, d)); break;
    }
    case N_LOGSOFTMAX: {     /* g_x = g - softmax(x) * sum(g, axis) */
        int ax = (int)n->x.i;
        AV sm = a_tensor_unary(U_EXP, n->val); A_PUSH(sm);
        AV s = reduce(R_SUM, g, a_int(ax), 1, 0); A_PUSH(s);
        AV t = gv_mul(sm, s); A_PUSH(t);
        acc_grad(a, a_binop(OP_SUB, g, t)); break;
    }
    case N_CONCAT: case N_STACK: {
        AList *parts = (AList *)a.o; int ax = (int)n->x.i; int64_t off = 0;
        ATensor *G = TT(g);
        int gax = norm_axis(ax, G->nd);
        for (int64_t i = 0; i < parts->len; i++) {
            AV p = parts->v[i]; AV pv = as_tensor(val_of(p)); A_PUSH(pv);
            int64_t len = n->op == N_STACK ? 1 : TT(pv)->sh[gax];
            if (is_n(p)) {
                AV sl = a_slice(a_int(off), a_int(off + len), A_NONE); A_PUSH(sl);
                AV items[MAXD]; for (int d = 0; d < G->nd; d++) items[d] = d == gax ? (n->op == N_STACK ? a_int(off) : sl) : a_slice(A_NONE, A_NONE, A_NONE);
                AV idx = a_tuple(G->nd, items); A_PUSH(idx);
                acc_grad(p, t_getitem(g, idx));
            }
            off += len;
        }
        break;
    }
    case N_WHERE: {
        AV c = n->x; AV z = a_float(0.0);
        if (is_n(a)) { AV r = a_tensor_where(c, g, z); A_PUSH(r); acc_grad(a, unbroadcast(r, va)); }
        if (is_n(b)) { AV r = a_tensor_where(c, z, g); A_PUSH(r); acc_grad(b, unbroadcast(r, vb)); }
        break;
    }
    case N_MAXIMUM: {
        AV m = a_binop(OP_MUL, a_tensor_cmp(OP_GE, va, vb), a_float(1.0)); A_PUSH(m);
        if (is_n(a)) acc_grad(a, unbroadcast(gv_mul(g, m), va));
        if (is_n(b)) { AV inv = a_binop(OP_SUB, a_float(1.0), m); A_PUSH(inv); acc_grad(b, unbroadcast(gv_mul(g, inv), vb)); }
        break;
    }
    default: break;
    }
    a_ntmp = sp;
}
/* d out / d leaves: seeds out with 1 and walks the recorded operations backwards */
static void backward(AV out) {
    int sp = a_ntmp;
    AList *order = (AList *)a_list(0, NULL).o; A_PUSH(a_obj(order));
    collect_nodes(out, order);
    for (int64_t i = 0; i < order->len; i++) ND(order->v[i])->seen = 0;
    AV ov = ND(out)->val;
    ND(out)->grad = is_t(ov) ? a_binop(OP_ADD, a_obj(t_like(TT(ov), TT(ov)->dt == DT_F32 ? DT_F32 : DT_F64)), a_float(1.0)) : a_float(1.0);
    for (int64_t i = order->len - 1; i >= 0; i--) {
        ANode *n = ND(order->v[i]);
        if (n->grad.t != T_NONE && n->op != N_LEAF) backward_one(n);
    }
    a_ntmp = sp;
}

/* ================================================================== the hooks the core runtime calls */
static AV wrap_unary(int u, AV v) {
    if (is_n(v)) { AV r = a_tensor_unary(u, ND(v)->val); int sp = a_ntmp; A_PUSH(r); AV n = node_new(r, N_UNARY, v, A_NONE, a_int(u)); a_ntmp = sp; return n; }
    if (is_t(v)) return unary(u, v);
    if (a_is(v, O_QTY) && u == U_ABS) return a_call(a_global_builtin("abs"), 1, &v, NULL);
    if (!is_num(v)) a_error("TypeError", "this function needs a number or a tensor");
    double x = v.t == T_FLOAT ? v.f : (double)v.i;
    if (u == U_RELU && v.t == T_INT) return a_int(v.i > 0 ? v.i : 0);
    if (u == U_ABS && v.t == T_INT) return a_int(v.i < 0 ? -v.i : v.i);
    if (u == U_NEG && v.t == T_INT) return a_int(-v.i);
    return a_float(uop(u, x));
}
AV a_tensor_unary(int u, AV v) { return wrap_unary(u, v); }
AV a_tensor_cmp(int op, AV a, AV b) {
    a = val_of(a); b = val_of(b);
    int e = op == OP_LT ? E_LT : op == OP_LE ? E_LE : op == OP_GT ? E_GT : op == OP_GE ? E_GE : op == OP_EQ ? E_EQ : E_NE;
    if (!is_t(a) && !is_t(b)) return a_bool(a_compare(op, a, b));
    int sp = a_ntmp; AV ta = as_tensor(a); A_PUSH(ta); AV tb = as_tensor(b); A_PUSH(tb);
    AV r = ewise(e, ta, tb); a_ntmp = sp; return r;
}
AV a_tensor_where(AV c, AV a, AV b) {
    int sp = a_ntmp;
    if (is_n(a) || is_n(b)) {
        AV v = a_tensor_where(c, val_of(a), val_of(b)); A_PUSH(v);
        AV n = node_new(v, N_WHERE, a, b, c); a_ntmp = sp; return n;
    }
    AV tc = as_tensor(val_of(c)); A_PUSH(tc); AV ta = as_tensor(a); A_PUSH(ta); AV tb = as_tensor(b); A_PUSH(tb);
    ATensor *C = TT(tc), *A = TT(ta), *B = TT(tb);
    int64_t s1[MAXD], s2[MAXD]; int nd1 = bshape(C, A, s1); ATensor tmp; tmp.nd = (uint8_t)nd1; memcpy(tmp.sh, s1, sizeof s1);
    int nd = bshape(&tmp, B, s2);
    int dt = (A->dt == DT_F64 || B->dt == DT_F64) ? DT_F64 : (A->dt == DT_F32 || B->dt == DT_F32) ? DT_F32 : A->dt;
    ATensor *o = t_new(dt, nd, s2);
    int64_t sc[MAXD], sa[MAXD], sb[MAXD], idx[MAXD] = {0}, ic = 0, ia = 0, ib = 0;
    bstrides(C, nd, s2, sc); bstrides(A, nd, s2, sa); bstrides(B, nd, s2, sb);
    for (int64_t i = 0; i < o->n; i++) {
        t_set(o, i, t_get(C, ic) != 0 ? t_get(A, ia) : t_get(B, ib));
        for (int d = nd - 1; d >= 0; d--) { idx[d]++; ic += sc[d]; ia += sa[d]; ib += sb[d]; if (idx[d] < s2[d]) break; ic -= sc[d] * s2[d]; ia -= sa[d] * s2[d]; ib -= sb[d] * s2[d]; idx[d] = 0; }
    }
    a_ntmp = sp;
    return a_obj(o);
}
/* a_binop sends here every operation with a tensor or a node on either side */
AV a_tensor_binop(int op, AV a, AV b) {
    if (is_n(a) || is_n(b)) {
        if (op == OP_MATMUL) { AV v = a_tensor_binop(OP_MATMUL, val_of(a), val_of(b)); int sp = a_ntmp; A_PUSH(v); AV r = node_new(v, N_MATMUL, a, b, A_NONE); a_ntmp = sp; return r; }
        AV v = a_binop(op, val_of(a), val_of(b));
        int nop = op == OP_ADD ? N_ADD : op == OP_SUB ? N_SUB : op == OP_MUL ? N_MUL : op == OP_DIV ? N_DIV : op == OP_POW ? N_POW : -1;
        if (nop < 0) return v;                           /* //, % ...: no derivative, the plain value */
        int sp = a_ntmp; A_PUSH(v); AV r = node_new(v, nop, a, b, A_NONE); a_ntmp = sp; return r;
    }
    if (a_is(a, O_QTY) || a_is(b, O_QTY)) a_error("TypeError", "a tensor with units is not supported natively yet");
    int sp = a_ntmp;
    AV ta = as_tensor(a); A_PUSH(ta); AV tb = as_tensor(b); A_PUSH(tb);
    AV r;
    switch (op) {
    case OP_ADD: r = ewise(E_ADD, ta, tb); break; case OP_SUB: r = ewise(E_SUB, ta, tb); break;
    case OP_MUL: r = ewise(E_MUL, ta, tb); break; case OP_DIV: r = ewise(E_DIV, ta, tb); break;
    case OP_POW: r = ewise(E_POW, ta, tb); break; case OP_MOD: r = ewise(E_MOD, ta, tb); break;
    case OP_FLOORDIV: r = ewise(E_FLOORDIV, ta, tb); break;
    case OP_AND: r = ewise(E_AND, ta, tb); break; case OP_OR: r = ewise(E_OR, ta, tb); break; case OP_XOR: r = ewise(E_XOR, ta, tb); break;
    case OP_MATMUL: r = matmul(ta, tb); break;
    default: a_error("TypeError", "this operator does not apply to tensors");
    }
    a_ntmp = sp;
    return r;
}
AV a_tensor_neg(AV a) {
    if (is_n(a)) { AV v = a_unop(OP_NEG, ND(a)->val); int sp = a_ntmp; A_PUSH(v); AV r = node_new(v, N_NEG, a, A_NONE, A_NONE); a_ntmp = sp; return r; }
    return unary(U_NEG, a);
}
int a_tensor_truthy(AV v) {
    v = val_of(v);
    if (!is_t(v)) return a_truthy(v);
    if (TT(v)->n != 1) a_error("ValueError", "the truth of a tensor with more than one element is ambiguous; use any() or all()");
    return t_get(TT(v), 0) != 0;
}
char *a_tensor_text(AV v, int top) {
    if (is_n(v)) { AV x = ND(v)->val; if (is_t(x)) return t_text(x, top); AV s = a_to_str(x); return strdup(((AStr *)s.o)->s); }
    return t_text(v, top);
}
int64_t a_tensor_len(AV v) { v = val_of(v); if (!is_t(v)) a_error("TypeError", "no len() of a number"); if (TT(v)->nd == 0) a_error("TypeError", "len() of a 0-d tensor"); return TT(v)->sh[0]; }
AV a_tensor_getitem(AV v, AV i) {
    if (is_n(v)) { AV r = t_getitem(ND(v)->val, i); int sp = a_ntmp; A_PUSH(r); AV n = node_new(r, N_INDEX, v, A_NONE, i); a_ntmp = sp; return n; }
    return t_getitem(v, i);
}
void a_tensor_setitem(AV v, AV i, AV x) { if (is_n(v)) a_error("TypeError", "cannot assign into a value being differentiated"); t_setitem(v, i, val_of(x), 0); }
int a_tensor_eq(AV a, AV b) {
    a = val_of(a); b = val_of(b);
    if (!is_t(a) || !is_t(b)) return 0;
    if (TT(a)->n != TT(b)->n || TT(a)->nd != TT(b)->nd) return 0;
    for (int64_t i = 0; i < TT(a)->n; i++) if (t_get(TT(a), i) != t_get(TT(b), i)) return 0;
    return 1;
}
double a_tensor_float(AV v) { v = val_of(v); if (is_t(v)) { if (TT(v)->n != 1) a_error("TypeError", "only a one-element tensor converts to a number"); return t_get(TT(v), 0); } return v.t == T_FLOAT ? v.f : (double)v.i; }

/* ---- reductions and shapes, recorded when differentiated */
static AV axis_kw(int n, AV *a, ADict *kw, int pos) { AV ax = kwget(kw, "axis"); if (ax.t == T_NONE && n > pos) ax = a[pos]; if (ax.t == T_NONE) ax = kwget(kw, "dim"); return ax; }
static AV red(int op, int n, AV *a, ADict *kw) {
    AV x = a[0], axis = axis_kw(n, a, kw, 1);
    int keep = a_truthy(kwget(kw, "keepdims")) || a_truthy(kwget(kw, "keepdim"));
    if (a_is(x, O_LIST) || a_is(x, O_TUPLE)) {
        int has_t = 0; AList *l = (AList *)x.o;
        for (int64_t i = 0; i < l->len; i++) if (is_t(l->v[i]) || is_n(l->v[i])) has_t = 1;
        if (!has_t && op != R_MEAN && op != R_STD && op != R_VAR) return A_NONE;     /* the list built-in decides */
        if (has_t || op == R_MEAN || op == R_STD || op == R_VAR) {
            if (!has_t) x = t_from(x, -1);
            else { AV acc = l->len ? l->v[0] : a_float(0.0); int sp = a_ntmp; A_PUSH(acc);
                if (op == R_SUM || op == R_MEAN) { for (int64_t i = 1; i < l->len; i++) { acc = a_binop(OP_ADD, acc, l->v[i]); a_tmp[sp] = acc; }
                    if (op == R_MEAN) acc = a_binop(OP_DIV, acc, a_float((double)l->len)); a_ntmp = sp; return acc; }
                a_ntmp = sp; x = t_from(x, -1); }
        }
    }
    if (is_n(x)) {
        int nop = op == R_SUM ? N_SUM : op == R_MEAN ? N_MEAN : (op == R_MAX || op == R_MIN) ? N_MAXRED : -1;
        AV v = red(op, n, (AV[]){ND(x)->val, n > 1 ? a[1] : A_NONE}, kw);
        if (nop < 0) return v;
        int sp = a_ntmp; A_PUSH(v);
        AV info = a_list(2, (AV[]){axis, a_int(keep)}); A_PUSH(info);
        AV r = node_new(v, nop, x, A_NONE, info); a_ntmp = sp; return r;
    }
    if (!is_t(x)) {
        if (is_num(x)) return op == R_ARGMAX || op == R_ARGMIN ? a_int(0) : x;
        x = t_from(x, -1);
    }
    int ddof = (int)(kwget(kw, "ddof").t == T_INT ? kwget(kw, "ddof").i : 0);
    return reduce(op, x, axis, keep, ddof);
}
static AV t_reshape_any(AV x, int nd, int64_t *sh) {
    if (is_n(x)) { AV v = reshape(as_tensor(ND(x)->val), nd, sh); int sp = a_ntmp; A_PUSH(v); AV r = node_new(v, N_RESHAPE, x, A_NONE, A_NONE); a_ntmp = sp; return r; }
    return reshape(as_tensor(x), nd, sh);
}
static AV t_permute_any(AV x, int nd, const int *perm) {
    if (is_n(x)) {
        AV v = permute(ND(x)->val, perm); int sp = a_ntmp; A_PUSH(v);
        AV pl = a_list(0, NULL); A_PUSH(pl); for (int i = 0; i < nd; i++) a_list_append((AList *)pl.o, a_int(perm[i]));
        AV r = node_new(v, N_PERMUTE, x, A_NONE, pl); a_ntmp = sp; return r;
    }
    return permute(x, perm);
}
static AV t_transpose_any(AV x) {
    AV v = as_tensor(val_of(x)); int nd = TT(v)->nd;
    if (nd < 2) return x;
    int perm[MAXD]; for (int i = 0; i < nd; i++) perm[i] = i; perm[nd - 1] = nd - 2; perm[nd - 2] = nd - 1;
    return t_permute_any(is_n(x) ? x : v, nd, perm);
}
static AV t_softmax(AV v, int ax, int logsm) {
    ATensor *a = TT(v); ax = norm_axis(ax, a->nd);
    int64_t outer = prod(a->sh, ax), len = a->sh[ax], inner = prod(a->sh + ax + 1, a->nd - ax - 1);
    ATensor *o = t_like(a, a->dt == DT_F32 ? DT_F32 : DT_F64);
    for (int64_t p = 0; p < outer; p++) for (int64_t q = 0; q < inner; q++) {
        int64_t base = p * len * inner + q; double m = -INFINITY, s = 0;
        for (int64_t k = 0; k < len; k++) { double x = t_get(a, base + k * inner); if (x > m) m = x; }
        for (int64_t k = 0; k < len; k++) s += exp(t_get(a, base + k * inner) - m);
        double ls = log(s);
        for (int64_t k = 0; k < len; k++) { double x = t_get(a, base + k * inner) - m; t_set(o, base + k * inner, logsm ? x - ls : exp(x - ls)); }
    }
    return a_obj(o);
}
static AV softmax_any(AV x, int ax, int logsm) {
    if (is_n(x)) { AV v = t_softmax(as_tensor(ND(x)->val), ax, logsm); int sp = a_ntmp; A_PUSH(v); AV r = node_new(v, logsm ? N_LOGSOFTMAX : N_SOFTMAX, x, A_NONE, a_int(ax)); a_ntmp = sp; return r; }
    return t_softmax(as_tensor(x), ax, logsm);
}

/* ================================================================== the built-in functions */
#define BI(name) static AV name(int n, AV *a, ADict *kw, AV self)
#define UNUSED (void)n; (void)a; (void)kw; (void)self
BI(tb_tensor) { (void)self; if (!n) a_error("TypeError", "tensor() needs a value"); AV x = val_of(a[0]); if (is_t(x)) return tcopy(TT(x), dt_from_name(n > 1 ? a[1] : kwget(kw, "dtype"))); return t_from(x, dt_from_name(n > 1 ? a[1] : kwget(kw, "dtype"))); }
static AV filled(int n, AV *a, ADict *kw, double v) { int64_t sh[MAXD]; int nd = shape_args(n, a, sh); int dt = dt_from_name(kwget(kw, "dtype")); ATensor *t = t_new(dt < 0 ? DT_F64 : dt, nd, sh); if (v != 0) for (int64_t i = 0; i < t->n; i++) t_set(t, i, v); return a_obj(t); }
BI(tb_zeros) { (void)self; return filled(n, a, kw, 0.0); }
BI(tb_ones) { (void)self; return filled(n, a, kw, 1.0); }
BI(tb_full) { UNUSED; int64_t sh[MAXD]; int nd = shape_args(1, a, sh); if (!(a_is(a[0], O_LIST) || a_is(a[0], O_TUPLE))) { nd = 1; sh[0] = a[0].i; } ATensor *t = t_new(a[1].t == T_INT ? DT_I64 : a[1].t == T_BOOL ? DT_BOOL : DT_F64, nd, sh); double v = a[1].t == T_FLOAT ? a[1].f : (double)a[1].i; for (int64_t i = 0; i < t->n; i++) t_set(t, i, v); return a_obj(t); }
BI(tb_zeros_like) { UNUSED; AV x = as_tensor(val_of(a[0])); return a_obj(t_like(TT(x), TT(x)->dt)); }
BI(tb_ones_like) { UNUSED; AV x = as_tensor(val_of(a[0])); ATensor *t = t_like(TT(x), TT(x)->dt); for (int64_t i = 0; i < t->n; i++) t_set(t, i, 1); return a_obj(t); }
BI(tb_eye) { UNUSED; int64_t sh[2] = {a[0].i, n > 1 ? a[1].i : a[0].i}; ATensor *t = t_new(DT_F64, 2, sh); for (int64_t i = 0; i < sh[0] && i < sh[1]; i++) t_set(t, i * sh[1] + i, 1); return a_obj(t); }
BI(tb_arange) {
    UNUSED;
    AV start = n > 1 ? a[0] : a_int(0), stop = n > 1 ? a[1] : a[0], step = n > 2 ? a[2] : a_int(1);
    int isint = start.t == T_INT && stop.t == T_INT && step.t == T_INT;
    double s0 = start.t == T_FLOAT ? start.f : (double)start.i, s1 = stop.t == T_FLOAT ? stop.f : (double)stop.i, st = step.t == T_FLOAT ? step.f : (double)step.i;
    if (st == 0) a_error("ValueError", "arange step must not be zero");
    int64_t cnt = (int64_t)ceil((s1 - s0) / st); if (cnt < 0) cnt = 0;
    ATensor *t = t_new(isint ? DT_I64 : DT_F64, 1, &cnt);
    for (int64_t i = 0; i < cnt; i++) t_set(t, i, s0 + (double)i * st);
    return a_obj(t);
}
BI(tb_linspace) {
    UNUSED;
    double s0 = a_tensor_float(a[0]), s1 = a_tensor_float(a[1]); int64_t cnt = n > 2 ? a[2].i : kwget(kw, "n").t == T_INT ? kwget(kw, "n").i : 50;
    ATensor *t = t_new(DT_F64, 1, &cnt);
    for (int64_t i = 0; i < cnt; i++) t_set(t, i, cnt == 1 ? s0 : s0 + (s1 - s0) * (double)i / (double)(cnt - 1));
    if (cnt > 1) t_set(t, cnt - 1, s1);
    return a_obj(t);
}
static AV t_random(int kind, int n, AV *a, AV lo, AV hi) {
    int64_t sh[MAXD]; int nd = shape_args(n, a, sh);
    ATensor *t = t_new(kind == 2 ? DT_I64 : DT_F64, nd, sh);
    for (int64_t i = 0; i < t->n; i++) {
        if (kind == 0) t_set(t, i, a_rand_f());
        else if (kind == 1) t_set(t, i, a_rand_n());
        else ((int64_t *)t->p)[i] = lo.i + (int64_t)(a_rand_u64() % (uint64_t)(hi.i - lo.i));
    }
    return a_obj(t);
}
BI(tb_shape) {
    UNUSED; AV x = val_of(a[0]);
    if (a_is(x, O_LIST) || a_is(x, O_TUPLE)) x = t_from(x, -1);
    if (!is_t(x)) return a_tuple(0, NULL);
    AV items[MAXD]; for (int i = 0; i < TT(x)->nd; i++) items[i] = a_int(TT(x)->sh[i]);
    return a_tuple(TT(x)->nd, items);
}
BI(tb_ndim) { UNUSED; AV x = val_of(a[0]); return a_int(is_t(x) ? TT(x)->nd : 0); }
BI(tb_reshape) {
    (void)self; int64_t sh[MAXD]; int nd = shape_args(n - 1, a + 1, sh); (void)kw;
    return t_reshape_any(a[0], nd, sh);
}
BI(tb_flatten) { UNUSED; AV x = as_tensor(val_of(a[0])); int64_t sh[1] = {TT(x)->n}; return t_reshape_any(is_n(a[0]) ? a[0] : x, 1, sh); }
BI(tb_transpose) {
    (void)self; (void)kw;
    if (n > 1 && a[1].t != T_NONE) { AV x = as_tensor(val_of(a[0])); int perm[MAXD]; AList *l = (AList *)a[1].o; for (int64_t i = 0; i < l->len; i++) perm[i] = (int)norm_axis((int)l->v[i].i, TT(x)->nd); return t_permute_any(is_n(a[0]) ? a[0] : x, TT(x)->nd, perm); }
    AV x = is_n(a[0]) ? a[0] : as_tensor(a[0]);
    AV v = val_of(x); int nd = TT(v)->nd; if (nd < 2) return x;
    int perm[MAXD]; for (int i = 0; i < nd; i++) perm[i] = nd - 1 - i;   /* NumPy's transpose: every axis reversed */
    return t_permute_any(x, nd, perm);
}
BI(tb_permute) {
    (void)self; (void)kw; AV x = is_n(a[0]) ? a[0] : as_tensor(a[0]); AV v = val_of(x); int nd = TT(v)->nd; int perm[MAXD];
    if (n == 2 && (a_is(a[1], O_LIST) || a_is(a[1], O_TUPLE))) { AList *l = (AList *)a[1].o; for (int64_t i = 0; i < l->len; i++) perm[i] = norm_axis((int)l->v[i].i, nd); }
    else for (int i = 1; i < n; i++) perm[i - 1] = norm_axis((int)a[i].i, nd);
    return t_permute_any(x, nd, perm);
}
BI(tb_swapaxes) {
    UNUSED; AV x = is_n(a[0]) ? a[0] : as_tensor(a[0]); AV v = val_of(x); int nd = TT(v)->nd; int perm[MAXD];
    for (int i = 0; i < nd; i++) perm[i] = i;
    int p = norm_axis((int)a[1].i, nd), q = norm_axis((int)a[2].i, nd); perm[p] = q; perm[q] = p;
    return t_permute_any(x, nd, perm);
}
BI(tb_unsqueeze) {
    UNUSED; AV v = as_tensor(val_of(a[0])); int nd = TT(v)->nd; int ax = (int)a[1].i; if (ax < 0) ax += nd + 1;
    int64_t sh[MAXD]; for (int i = 0, j = 0; i <= nd; i++) sh[i] = i == ax ? 1 : TT(v)->sh[j++];
    return t_reshape_any(is_n(a[0]) ? a[0] : v, nd + 1, sh);
}
BI(tb_squeeze) {
    (void)self; AV v = as_tensor(val_of(a[0])); AV ax = axis_kw(n, a, kw, 1); int nd = TT(v)->nd; int64_t sh[MAXD]; int k = 0;
    int axi = ax.t == T_NONE ? -1 : norm_axis((int)ax.i, nd);
    for (int i = 0; i < nd; i++) if (!(TT(v)->sh[i] == 1 && (axi < 0 || axi == i))) sh[k++] = TT(v)->sh[i];
    return t_reshape_any(is_n(a[0]) ? a[0] : v, k, sh);
}
static AV join(int n, AV *a, ADict *kw, int stack) {
    AV items = a[0]; AV ax = axis_kw(n, a, kw, 1); int axi = ax.t == T_NONE ? 0 : (int)ax.i;
    AList *l = (AList *)items.o; int any_node = 0;
    for (int64_t i = 0; i < l->len; i++) if (is_n(l->v[i])) any_node = 1;
    if (!any_node) return t_concat(items, axi, stack);
    int sp = a_ntmp;
    AV vals = a_list(l->len, l->v); A_PUSH(vals);
    for (int64_t i = 0; i < l->len; i++) ((AList *)vals.o)->v[i] = val_of(l->v[i]);
    AV v = t_concat(vals, axi, stack); A_PUSH(v);
    AV keep = a_list(l->len, l->v); A_PUSH(keep);
    AV r = node_new(v, stack ? N_STACK : N_CONCAT, keep, A_NONE, a_int(axi)); a_ntmp = sp; return r;
}
BI(tb_concat) { (void)self; return join(n, a, kw, 0); }
BI(tb_stack) { (void)self; return join(n, a, kw, 1); }
BI(tb_matmul) { UNUSED; return a_tensor_binop(OP_MATMUL, a[0], a[1]); }
BI(tb_dot) {
    UNUSED;
    AV x = val_of(a[0]), y = val_of(a[1]);
    if (is_t(x) && is_t(y) && TT(x)->nd == 1 && TT(y)->nd == 1 && !is_n(a[0]) && !is_n(a[1])) {
        if (TT(x)->n != TT(y)->n) a_error("ValueError", "dot: lengths differ");
        double s = 0; for (int64_t i = 0; i < TT(x)->n; i++) s += t_get(TT(x), i) * t_get(TT(y), i); return a_float(s);
    }
    if (a_is(x, O_LIST)) { AV tx = t_from(x, -1); int sp = a_ntmp; A_PUSH(tx); AV ty = as_tensor(y); A_PUSH(ty); AV r = matmul(tx, ty); a_ntmp = sp; return r; }
    return a_tensor_binop(OP_MATMUL, a[0], a[1]);
}
BI(tb_outer) {
    UNUSED; AV x = as_tensor(val_of(a[0])), y; int sp = a_ntmp; A_PUSH(x); y = as_tensor(val_of(a[1])); A_PUSH(y);
    int64_t s1[2] = {TT(x)->n, 1}, s2[2] = {1, TT(y)->n};
    AV x2 = t_reshape_any(is_n(a[0]) ? a[0] : x, 2, s1); A_PUSH(x2);
    AV y2 = t_reshape_any(is_n(a[1]) ? a[1] : y, 2, s2); A_PUSH(y2);
    AV r = a_binop(OP_MUL, x2, y2); a_ntmp = sp; return r;
}
BI(tb_softmax) { (void)self; AV ax = axis_kw(n, a, kw, 1); return softmax_any(a[0], ax.t == T_NONE ? -1 : (int)ax.i, 0); }
BI(tb_log_softmax) { (void)self; AV ax = axis_kw(n, a, kw, 1); return softmax_any(a[0], ax.t == T_NONE ? -1 : (int)ax.i, 1); }
BI(tb_mean) { (void)self; return red(R_MEAN, n, a, kw); }
BI(tb_prod) { (void)self; AV r = red(R_PROD, n, a, kw); if (r.t != T_NONE) return r; AV acc = a_int(1), it = a_iter(a[0]), x; int sp = a_ntmp; A_PUSH(it); A_PUSH(acc); while (a_next(it, &x)) { acc = a_binop(OP_MUL, acc, x); a_tmp[sp + 1] = acc; } a_ntmp = sp; return acc; }
BI(tb_std) { (void)self; return red(R_STD, n, a, kw); }
BI(tb_var) { (void)self; return red(R_VAR, n, a, kw); }
BI(tb_argmax) { (void)self; return red(R_ARGMAX, n, a, kw); }
BI(tb_argmin) { (void)self; return red(R_ARGMIN, n, a, kw); }
BI(tb_norm) {
    (void)self; AV x = a[0];
    int sp = a_ntmp; AV sq = a_binop(OP_MUL, x, x); A_PUSH(sq);
    AV s = red(R_SUM, n, (AV[]){sq, n > 1 ? a[1] : A_NONE}, kw); A_PUSH(s);
    AV r = a_tensor_unary(U_SQRT, s); a_ntmp = sp; return r;
}
BI(tb_logsumexp) {
    (void)self; AV x = a[0];
    if (is_n(x)) {   /* log(sum(exp(x - m))) + m, with m held constant */
        AV ax = axis_kw(n, a, kw, 1); int sp = a_ntmp;
        AV m = reduce(R_MAX, as_tensor(ND(x)->val), ax, 1, 0); A_PUSH(m);
        AV d = a_binop(OP_SUB, x, m); A_PUSH(d);
        AV e = a_tensor_unary(U_EXP, d); A_PUSH(e);
        AV s = red(R_SUM, 2, (AV[]){e, ax}, NULL); A_PUSH(s);
        AV l = a_tensor_unary(U_LOG, s); A_PUSH(l);
        AV m2 = ax.t == T_NONE ? a_float(t_get(TT(m), 0)) : reshape(m, TT(val_of(l))->nd, TT(val_of(l))->sh); A_PUSH(m2);
        AV r = a_binop(OP_ADD, l, m2); a_ntmp = sp; return r;
    }
    if (a_is(x, O_LIST)) x = t_from(x, -1);
    return reduce(R_LOGSUMEXP, x, axis_kw(n, a, kw, 1), 0, 0);
}
BI(tb_cumsum) {
    UNUSED; AV x = as_tensor(val_of(a[0])); ATensor *t = TT(x);
    AV axv = n > 1 ? a[1] : kwget(kw, "axis");
    if (axv.t == T_NONE) { int64_t cnt = t->n; ATensor *o = t_new(t->dt == DT_BOOL ? DT_I64 : t->dt, 1, &cnt); double s = 0; for (int64_t i = 0; i < t->n; i++) { s += t_get(t, i); t_set(o, i, s); } return a_obj(o); }
    int ax = norm_axis((int)axv.i, t->nd); int64_t outer = prod(t->sh, ax), len = t->sh[ax], inner = prod(t->sh + ax + 1, t->nd - ax - 1);
    ATensor *o = t_like(t, t->dt == DT_BOOL ? DT_I64 : t->dt);
    for (int64_t p = 0; p < outer; p++) for (int64_t q = 0; q < inner; q++) { double s = 0; for (int64_t k = 0; k < len; k++) { int64_t i = p * len * inner + k * inner + q; s += t_get(t, i); t_set(o, i, s); } }
    return a_obj(o);
}
BI(tb_where) { UNUSED; return a_tensor_where(a[0], a[1], a[2]); }
BI(tb_clip) {
    UNUSED; AV x = a[0];
    int sp = a_ntmp;
    AV lo = a[1], hi = a[2];
    if (is_n(x)) {
        AV r = x;
        if (lo.t != T_NONE) { AV c = a_tensor_cmp(OP_LT, r, lo); A_PUSH(c); r = a_tensor_where(c, lo, r); A_PUSH(r); }
        if (hi.t != T_NONE) { AV c = a_tensor_cmp(OP_GT, r, hi); A_PUSH(c); r = a_tensor_where(c, hi, r); }
        a_ntmp = sp; return r;
    }
    if (!is_t(x)) { if (is_num(x)) { double v = x.t == T_FLOAT ? x.f : (double)x.i; if (lo.t != T_NONE && v < a_tensor_float(lo)) return lo; if (hi.t != T_NONE && v > a_tensor_float(hi)) return hi; return x; } x = t_from(x, -1); }
    ATensor *t = TT(x), *o = t_like(t, t->dt);
    for (int64_t i = 0; i < t->n; i++) { double v = t_get(t, i); if (lo.t != T_NONE && v < a_tensor_float(lo)) v = a_tensor_float(lo); if (hi.t != T_NONE && v > a_tensor_float(hi)) v = a_tensor_float(hi); t_set(o, i, v); }
    return a_obj(o);
}
static int cmp_dbl(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }
BI(tb_sort) {
    UNUSED; AV x = val_of(a[0]);
    if (!is_t(x)) return a_call(a_global_builtin("sorted"), n, a, kw);
    ATensor *t = TT(x); AV r = tcopy(t, t->dt == DT_F32 ? DT_F32 : t->dt);
    if (t->nd != 1) a_error("NotImplementedError", "sort of a tensor with more than one dimension");
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(t->n ? t->n : 1));
    for (int64_t i = 0; i < t->n; i++) tmp[i] = t_get(t, i);
    qsort(tmp, (size_t)t->n, sizeof(double), cmp_dbl);
    for (int64_t i = 0; i < t->n; i++) t_set(TT(r), i, tmp[i]);
    free(tmp); return r;
}
#define UFN(nm, code) BI(tb_##nm) { (void)self; (void)kw; (void)n; return wrap_unary(code, a[0]); }
UFN(relu, U_RELU) UFN(sigmoid, U_SIGMOID) UFN(softplus, U_SOFTPLUS) UFN(silu, U_SILU) UFN(t_exp, U_EXP)
UFN(t_log, U_LOG) UFN(t_sqrt, U_SQRT) UFN(t_sin, U_SIN) UFN(t_cos, U_COS) UFN(t_tan, U_TAN) UFN(t_tanh, U_TANH)
UFN(t_abs, U_ABS) UFN(t_floor, U_FLOOR) UFN(t_ceil, U_CEIL) UFN(t_log1p, U_LOG1P) UFN(t_expm1, U_EXPM1)
UFN(t_asin, U_ASIN) UFN(t_acos, U_ACOS) UFN(t_atan, U_ATAN) UFN(t_sinh, U_SINH) UFN(t_cosh, U_COSH)
UFN(t_log2, U_LOG2) UFN(t_log10, U_LOG10) UFN(t_erf, U_ERF) UFN(sign, U_SIGN)
BI(tb_gelu) { (void)self; AV ap = n > 1 ? a[1] : kwget(kw, "approximate"); int erfv = a_is(ap, O_STR) && !strcmp(((AStr *)ap.o)->s, "none"); return wrap_unary(erfv ? U_GELU_ERF : U_GELU, a[0]); }
BI(tb_t_isnan) { UNUSED; AV x = val_of(a[0]); if (is_t(x)) return unary(U_ISNAN, x); return a_bool(x.t == T_FLOAT && isnan(x.f)); }
BI(tb_t_isfinite) { UNUSED; AV x = val_of(a[0]); if (is_t(x)) return unary(U_ISFINITE, x); return a_bool(x.t != T_FLOAT || isfinite(x.f)); }

/* ---- differentiation */
static AV wrap_tree(AV v, AList *leaves) {           /* every number and tensor in v becomes a recorded leaf */
    if (is_t(v) || v.t == T_FLOAT || v.t == T_INT || a_is(v, O_QTY)) {
        AV x = v.t == T_INT ? a_float((double)v.i) : v;
        AV nd = node_new(x, N_LEAF, A_NONE, A_NONE, A_NONE); a_list_append(leaves, nd); return nd;
    }
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) {
        AList *l = (AList *)v.o; AV out = a_obj(v.o->kind == O_LIST ? (AList *)a_list(l->len, l->v).o : (AList *)a_tuple(l->len, l->v).o);
        int sp = a_ntmp; A_PUSH(out);
        for (int64_t i = 0; i < l->len; i++) ((AList *)out.o)->v[i] = wrap_tree(l->v[i], leaves);
        a_ntmp = sp; return out;
    }
    if (a_is(v, O_DICT)) {
        ADict *d = (ADict *)v.o; AV out = a_dict(); int sp = a_ntmp; A_PUSH(out);
        for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set((ADict *)out.o, d->e[i].k, wrap_tree(d->e[i].v, leaves));
        a_ntmp = sp; return out;
    }
    if (a_is(v, O_INST)) {               /* a record or a struct: its fields */
        AInst *s = (AInst *)v.o; AV out = a_call(a_obj(s->cls), 0, NULL, NULL);
        int sp = a_ntmp; A_PUSH(out);
        ADict *d = s->attrs;
        for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set(((AInst *)out.o)->attrs, d->e[i].k, wrap_tree(d->e[i].v, leaves));
        a_ntmp = sp; return out;
    }
    return v;
}
static AV grads_tree(AV v) {                         /* the same structure, with each leaf's gradient */
    if (is_n(v)) {
        ANode *n = ND(v);
        if (n->grad.t != T_NONE) return n->grad;
        if (is_t(n->val)) return a_obj(t_like(TT(n->val), TT(n->val)->dt == DT_F32 ? DT_F32 : DT_F64));
        return a_float(0.0);
    }
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) {
        AList *l = (AList *)v.o; AV out = v.o->kind == O_LIST ? a_list(l->len, l->v) : a_tuple(l->len, l->v);
        int sp = a_ntmp; A_PUSH(out);
        for (int64_t i = 0; i < l->len; i++) ((AList *)out.o)->v[i] = grads_tree(l->v[i]);
        a_ntmp = sp; return out;
    }
    if (a_is(v, O_DICT)) {
        ADict *d = (ADict *)v.o; AV out = a_dict(); int sp = a_ntmp; A_PUSH(out);
        for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set((ADict *)out.o, d->e[i].k, grads_tree(d->e[i].v));
        a_ntmp = sp; return out;
    }
    if (a_is(v, O_INST)) {
        AInst *s = (AInst *)v.o; AV out = a_call(a_obj(s->cls), 0, NULL, NULL);
        int sp = a_ntmp; A_PUSH(out);
        ADict *d = s->attrs;
        for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set(((AInst *)out.o)->attrs, d->e[i].k, grads_tree(d->e[i].v));
        a_ntmp = sp; return out;
    }
    return v;
}
static AV run_grad(AV f, int n, AV *a, ADict *kw, int with_value) {
    if (!n) a_error("TypeError", "grad(f)(x) needs an argument");
    int sp = a_ntmp;
    AV leaves = a_list(0, NULL); A_PUSH(leaves);
    AV args = a_list(n, a); A_PUSH(args);
    AV x = wrap_tree(a[0], (AList *)leaves.o);
    ((AList *)args.o)->v[0] = x;
    AV out = a_call(f, n, ((AList *)args.o)->v, kw); A_PUSH(out);
    AV value = val_of(out);
    if (is_n(out)) {
        if (is_t(value) && TT(value)->n != 1) a_error("ValueError", "grad needs a function with one number as its result");
        backward(out);
    }
    AV g = grads_tree(x); A_PUSH(g);
    a_ntmp = sp;
    if (!with_value) return g;
    AV pair[2] = {is_t(value) && TT(value)->n == 1 ? t_scalar(TT(value), 0) : value, g};
    return a_tuple(2, pair);
}
static AV grad_apply(int n, AV *a, ADict *kw, AV self) { return run_grad(self, n, a, kw, 0); }
static AV vgrad_apply(int n, AV *a, ADict *kw, AV self) { return run_grad(self, n, a, kw, 1); }
BI(tb_grad) { UNUSED; return a_builtin("grad", grad_apply, a[0]); }
BI(tb_value_and_grad) { UNUSED; return a_builtin("value_and_grad", vgrad_apply, a[0]); }
BI(tb_stop_gradient) { UNUSED; return val_of(a[0]); }


/* ---- trees of values (lists, tuples, dicts, records) */
static int leaf(AV v) { return is_t(v) || is_n(v) || is_num(v) || a_is(v, O_QTY); }
static AV tree_map_rec(AV f, int k, AV *trees) {
    AV t0 = trees[0];
    if (leaf(t0) || t0.t == T_NONE) return t0.t == T_NONE ? A_NONE : a_call(f, k, trees, NULL);
    int sp = a_ntmp;
    if (a_is(t0, O_LIST) || a_is(t0, O_TUPLE)) {
        AList *l = (AList *)t0.o; AV out = t0.o->kind == O_LIST ? a_list(l->len, l->v) : a_tuple(l->len, l->v); A_PUSH(out);
        for (int64_t i = 0; i < l->len; i++) {
            AV sub[8]; for (int j = 0; j < k; j++) sub[j] = ((AList *)trees[j].o)->v[i];
            ((AList *)out.o)->v[i] = tree_map_rec(f, k, sub);
        }
        a_ntmp = sp; return out;
    }
    if (a_is(t0, O_DICT) || a_is(t0, O_INST)) {
        int inst = a_is(t0, O_INST);
        ADict *d = inst ? ((AInst *)t0.o)->attrs : (ADict *)t0.o;
        AV out = inst ? a_call(a_obj(((AInst *)t0.o)->cls), 0, NULL, NULL) : a_dict(); A_PUSH(out);
        ADict *od = inst ? ((AInst *)out.o)->attrs : (ADict *)out.o;
        for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) {
            AV sub[8];
            for (int j = 0; j < k; j++) { ADict *dj = a_is(trees[j], O_INST) ? ((AInst *)trees[j].o)->attrs : (ADict *)trees[j].o; if (!a_dict_get(dj, d->e[i].k, &sub[j])) a_error("ValueError", "tree_map: the trees differ in structure"); }
            a_dict_set(od, d->e[i].k, tree_map_rec(f, k, sub));
        }
        a_ntmp = sp; return out;
    }
    return a_call(f, k, trees, NULL);
}
BI(tb_tree_map) { (void)kw; (void)self; if (n < 2 || n > 8) a_error("TypeError", "tree_map(f, tree, ...)"); return tree_map_rec(a[0], n - 1, a + 1); }
static void leaves_rec(AV v, AList *out) {
    if (is_t(v) || is_n(v) || a_is(v, O_QTY) || v.t == T_FLOAT) { a_list_append(out, v); return; }
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) { AList *l = (AList *)v.o; for (int64_t i = 0; i < l->len; i++) leaves_rec(l->v[i], out); return; }
    if (a_is(v, O_DICT) || a_is(v, O_INST)) { ADict *d = a_is(v, O_INST) ? ((AInst *)v.o)->attrs : (ADict *)v.o; for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) leaves_rec(d->e[i].v, out); }
}
BI(tb_tree_leaves) { UNUSED; AV out = a_list(0, NULL); int sp = a_ntmp; A_PUSH(out); leaves_rec(a[0], (AList *)out.o); a_ntmp = sp; return out; }

/* ---- records */
BI(tb_record) {
    UNUSED;
    static AV cls; static int made = 0;
    if (!made) { cls = a_class("record", A_NONE, 2, 0, NULL); a_root(&cls, 1); made = 1; }
    AV r = a_call(cls, 0, NULL, NULL);
    if (kw) for (int64_t i = 0; i < kw->nent; i++) if (kw->e[i].used) a_dict_set(((AInst *)r.o)->attrs, kw->e[i].k, kw->e[i].v);
    return r;
}
BI(tb_item) { UNUSED; return a_float(a_tensor_float(a[0])); }
BI(tb_to_float) { UNUSED; AV x = val_of(a[0]); if (is_t(x)) return tcopy(TT(x), DT_F64); return a_float(a_tensor_float(x)); }
BI(tb_tolist) {
    UNUSED; AV x = val_of(a[0]);
    if (!is_t(x)) return x;
    ATensor *t = TT(x);
    if (t->nd == 0) return t_scalar(t, 0);
    if (t->nd == 1) { AList *l = (AList *)a_list(0, NULL).o; for (int64_t i = 0; i < t->n; i++) a_list_append(l, t_scalar(t, i)); return a_obj(l); }
    int sp = a_ntmp; AList *l = (AList *)a_list(0, NULL).o; A_PUSH(a_obj(l));
    for (int64_t i = 0; i < t->sh[0]; i++) { AV row = t_getitem(x, a_int(i)); A_PUSH(row); AV r = tb_tolist(1, &row, NULL, A_NONE); a_ntmp--; a_list_append(l, r); }
    a_ntmp = sp; return a_obj(l);
}
BI(tb_astype) { UNUSED; AV x = as_tensor(val_of(a[0])); return tcopy(TT(x), dt_from_name(a[1])); }
BI(tb_param_count) {
    UNUSED; AV v = a[0];
    if (is_t(val_of(v))) return a_int(TT(val_of(v))->n);
    int64_t total = 0;
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) { AList *l = (AList *)v.o; for (int64_t i = 0; i < l->len; i++) total += tb_param_count(1, &l->v[i], NULL, A_NONE).i; }
    else if (a_is(v, O_DICT) || a_is(v, O_INST)) { ADict *d = a_is(v, O_DICT) ? (ADict *)v.o : ((AInst *)v.o)->attrs; for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) total += tb_param_count(1, &d->e[i].v, NULL, A_NONE).i; }
    else if (is_num(v)) total = 1;
    return a_int(total);
}

/* methods of a tensor: x.shape, x.T, x.sum(), x.reshape(...) */
static AV method_bound(const char *name, ABuiltinFn fn, AV self) { return a_builtin(name, fn, self); }
static AV tm_call(ABuiltinFn fn, int n, AV *a, ADict *kw, AV self) { AV args[9]; args[0] = self; for (int i = 0; i < n && i < 8; i++) args[i + 1] = a[i]; return fn(n + 1, args, kw, A_NONE); }
#define TMD(nm, fn) static AV tmm_##nm(int n, AV *a, ADict *kw, AV self) { return tm_call(fn, n, a, kw, self); }
static AV tb_sum_bi(int n, AV *a, ADict *kw, AV self) { (void)self; AV r = red(R_SUM, n, a, kw); if (r.t == T_NONE && !is_t(a[0])) return a_call(a_global_builtin("sum"), n, a, kw); return r; }
static AV tb_max_bi(int n, AV *a, ADict *kw, AV self) { (void)self; return red(R_MAX, n, a, kw); }
static AV tb_min_bi(int n, AV *a, ADict *kw, AV self) { (void)self; return red(R_MIN, n, a, kw); }
static AV tb_any_bi(int n, AV *a, ADict *kw, AV self) { (void)self; return red(R_ANY, n, a, kw); }
static AV tb_all_bi(int n, AV *a, ADict *kw, AV self) { (void)self; return red(R_ALL, n, a, kw); }
TMD(sum, tb_sum_bi) TMD(mean, tb_mean) TMD(max, tb_max_bi) TMD(min, tb_min_bi) TMD(std, tb_std) TMD(var, tb_var)
TMD(argmax, tb_argmax) TMD(argmin, tb_argmin) TMD(prod, tb_prod) TMD(reshape, tb_reshape) TMD(flatten, tb_flatten)
TMD(transpose, tb_transpose) TMD(permute, tb_permute) TMD(tolist, tb_tolist) TMD(item, tb_item) TMD(astype, tb_astype)
TMD(squeeze, tb_squeeze) TMD(unsqueeze, tb_unsqueeze) TMD(any, tb_any_bi) TMD(all, tb_all_bi) TMD(cumsum, tb_cumsum)
TMD(clip, tb_clip) TMD(norm, tb_norm) TMD(softmax, tb_softmax) TMD(swapaxes, tb_swapaxes) TMD(exp, tb_t_exp)
TMD(log, tb_t_log) TMD(sqrt, tb_t_sqrt) TMD(abs, tb_t_abs) TMD(tanh, tb_t_tanh) TMD(relu, tb_relu) TMD(sigmoid, tb_sigmoid)
static AV tmm_copy(int n, AV *a, ADict *kw, AV self) { (void)n; (void)a; (void)kw; AV x = val_of(self); return tcopy(TT(x), -1); }
AV a_tensor_attr(AV v, const char *name) {
    AV x = val_of(v);
    if (!is_t(x)) { AV none; none.t = 255; none.i = 0; return none; }
    ATensor *t = TT(x);
    if (!strcmp(name, "shape")) return tb_shape(1, &x, NULL, A_NONE);
    if (!strcmp(name, "ndim")) return a_int(t->nd);
    if (!strcmp(name, "size")) return a_int(t->n);
    if (!strcmp(name, "dtype")) return a_str(dt_name[t->dt]);
    if (!strcmp(name, "T")) return t_transpose_any(is_n(v) ? v : x);
    if (!strcmp(name, "value")) return x;
    #define M(nm) if (!strcmp(name, #nm)) return method_bound(#nm, tmm_##nm, v)
    M(sum); M(mean); M(max); M(min); M(std); M(var); M(argmax); M(argmin); M(prod); M(reshape); M(flatten);
    M(transpose); M(permute); M(tolist); M(item); M(astype); M(squeeze); M(unsqueeze); M(any); M(all); M(cumsum);
    M(clip); M(norm); M(softmax); M(swapaxes); M(exp); M(log); M(sqrt); M(abs); M(tanh); M(relu); M(sigmoid); M(copy);
    #undef M
    if (!strcmp(name, "view")) return method_bound("view", tmm_reshape, v);
    AV none; none.t = 255; none.i = 0; return none;
}

typedef struct { const char *name; ABuiltinFn fn; } TEnt;
static const TEnt tensor_builtins[] = {
    {"tensor", tb_tensor}, {"zeros", tb_zeros}, {"ones", tb_ones}, {"full", tb_full}, {"zeros_like", tb_zeros_like},
    {"ones_like", tb_ones_like}, {"eye", tb_eye}, {"arange", tb_arange}, {"linspace", tb_linspace}, {"shape", tb_shape},
    {"ndim", tb_ndim}, {"reshape", tb_reshape}, {"flatten", tb_flatten}, {"transpose", tb_transpose},
    {"permute", tb_permute}, {"swapaxes", tb_swapaxes}, {"unsqueeze", tb_unsqueeze}, {"squeeze", tb_squeeze},
    {"concat", tb_concat}, {"stack", tb_stack}, {"matmul", tb_matmul}, {"dot", tb_dot}, {"outer", tb_outer},
    {"softmax", tb_softmax}, {"log_softmax", tb_log_softmax}, {"mean", tb_mean}, {"prod", tb_prod}, {"std", tb_std},
    {"var", tb_var}, {"argmax", tb_argmax}, {"argmin", tb_argmin}, {"norm", tb_norm}, {"logsumexp", tb_logsumexp},
    {"cumsum", tb_cumsum}, {"where", tb_where}, {"clip", tb_clip}, {"sort", tb_sort}, {"relu", tb_relu},
    {"sigmoid", tb_sigmoid}, {"softplus", tb_softplus}, {"silu", tb_silu}, {"gelu", tb_gelu}, {"sign", tb_sign},
    {"grad", tb_grad}, {"value_and_grad", tb_value_and_grad}, {"stop_gradient", tb_stop_gradient},
    {"record", tb_record}, {"item", tb_item}, {"to_float", tb_to_float}, {"tolist", tb_tolist},
    {"param_count", tb_param_count}, {"tree_map", tb_tree_map}, {"tree_leaves", tb_tree_leaves},
    {NULL, NULL}};
/* the maths built-ins of the core take tensors and nodes through these */
static const TEnt tensor_overrides[] = {
    {"exp", tb_t_exp}, {"log", tb_t_log}, {"sqrt", tb_t_sqrt}, {"sin", tb_t_sin}, {"cos", tb_t_cos},
    {"tan", tb_t_tan}, {"tanh", tb_t_tanh}, {"abs", tb_t_abs}, {"floor", tb_t_floor}, {"ceil", tb_t_ceil},
    {"log1p", tb_t_log1p}, {"expm1", tb_t_expm1}, {"asin", tb_t_asin}, {"acos", tb_t_acos}, {"atan", tb_t_atan},
    {"sinh", tb_t_sinh}, {"cosh", tb_t_cosh}, {"log2", tb_t_log2}, {"log10", tb_t_log10}, {"erf", tb_t_erf},
    {"isnan", tb_t_isnan}, {"isfinite", tb_t_isfinite}, {"sum", tb_sum_bi}, {"max", tb_max_bi}, {"min", tb_min_bi},
    {NULL, NULL}};
ABuiltinFn a_tensor_builtin(const char *name, int override_only) {
    for (const TEnt *e = tensor_overrides; e->name; e++) if (!strcmp(e->name, name)) return e->fn;
    if (override_only) return NULL;
    for (const TEnt *e = tensor_builtins; e->name; e++) if (!strcmp(e->name, name)) return e->fn;
    return NULL;
}
void a_tensor_mark(AObj *o) {
    if (o->kind == O_NODE) { ANode *n = (ANode *)o; a_mark_value(n->val); a_mark_value(n->a); a_mark_value(n->b); a_mark_value(n->x); a_mark_value(n->grad); }
}
void a_tensor_free(AObj *o) { if (o->kind == O_TENSOR) tfree(o); }
void a_tensor_init(void) { a_tensor_random = t_random; (void)tmark; }
