/* Complex numbers and complex128 tensors for native programs, as apsi run has them (Python complex, NumPy
   complex128): arithmetic with broadcasting, matrix products, sums, the elementary functions, indexing, and the
   same printed text (scalars as 1.0+2.0i, arrays as NumPy writes them). Included at the end of apsi_tensor.c.
   A complex tensor keeps its numbers interleaved (re, im) in one block after its header. */

typedef struct ACT { AObj h; int nd; int64_t sh[MAXD]; int64_t n; double p[]; } ACT;
typedef struct ACplx { AObj h; double re, im; } ACplx;
#define CT(v) ((ACT *)(v).o)
#define CX(v) ((ACplx *)(v).o)
static int is_ct(AV v) { return v.t == T_OBJ && v.o->kind == O_CTENSOR; }
static int is_cx(AV v) { return v.t == T_OBJ && v.o->kind == O_COMPLEX; }

AV a_cplx(double re, double im) { ACplx *c = (ACplx *)a_alloc(sizeof(ACplx), O_COMPLEX); c->re = re; c->im = im; return a_obj(c); }
static AV ct_new(int nd, const int64_t *sh) {
    int64_t n = 1; for (int i = 0; i < nd; i++) n *= sh[i];
    ACT *t = (ACT *)a_alloc(sizeof(ACT) + sizeof(double) * 2 * (size_t)(n ? n : 1), O_CTENSOR);
    t->nd = nd; for (int i = 0; i < nd; i++) t->sh[i] = sh[i]; t->n = n;
    memset(t->p, 0, sizeof(double) * 2 * (size_t)n);
    return a_obj(t);
}

/* ---- any value as a complex tensor */
static int cx_scalar(AV v, double *re, double *im) {
    if (v.t == T_INT || v.t == T_BOOL) { *re = (double)v.i; *im = 0; return 1; }
    if (v.t == T_FLOAT) { *re = v.f; *im = 0; return 1; }
    if (is_cx(v)) { *re = CX(v)->re; *im = CX(v)->im; return 1; }
    return 0;
}
static void ct_shape_of(AV v, int depth, int *nd, int64_t *sh) {
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) {
        AList *l = (AList *)v.o;
        if (depth >= MAXD) a_error("ValueError", "too many dimensions");
        if (depth >= *nd) { *nd = depth + 1; sh[depth] = l->len; }
        else if (sh[depth] != l->len) a_error("ValueError", "the rows have different lengths");
        if (l->len) ct_shape_of(l->v[0], depth + 1, nd, sh);
    } else if (is_ct(v)) {
        for (int i = 0; i < CT(v)->nd; i++) { if (depth + i >= *nd) { *nd = depth + i + 1; sh[depth + i] = CT(v)->sh[i]; } }
    } else if (is_t(v)) {
        for (int i = 0; i < TT(v)->nd; i++) { if (depth + i >= *nd) { *nd = depth + i + 1; sh[depth + i] = TT(v)->sh[i]; } }
    }
}
static void ct_fill(AV v, double **w) {
    double re, im;
    if (cx_scalar(v, &re, &im)) { (*w)[0] = re; (*w)[1] = im; *w += 2; return; }
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) { AList *l = (AList *)v.o; for (int64_t i = 0; i < l->len; i++) ct_fill(l->v[i], w); return; }
    if (is_ct(v)) { memcpy(*w, CT(v)->p, sizeof(double) * 2 * (size_t)CT(v)->n); *w += 2 * CT(v)->n; return; }
    if (is_t(v) || is_n(v)) {
        AV t = to_host(val_of(v)); ATensor *T = TT(t);
        for (int64_t i = 0; i < T->n; i++) { (*w)[0] = t_get(T, i); (*w)[1] = 0; *w += 2; }
        return;
    }
    if (a_is(v, O_QTY)) a_error("UnitError", "complex numbers take plain values; strip the unit first");
    a_error("TypeError", "cannot make a complex number of a %s", a_tname(v));
}
static AV as_ct(AV v) {
    if (is_ct(v)) return v;
    int nd = 0; int64_t sh[MAXD];
    ct_shape_of(v, 0, &nd, sh);
    AV r = ct_new(nd, sh); double *w = CT(r)->p; int sp = a_ntmp; A_PUSH(r);
    ct_fill(v, &w); a_ntmp = sp;
    return r;
}
static AV ct_result(AV t) {                                     /* a 0-d complex tensor is a complex number */
    if (is_ct(t) && CT(t)->nd == 0) return a_cplx(CT(t)->p[0], CT(t)->p[1]);
    return t;
}
static AV ct_real_tensor(int nd, const int64_t *sh) { ATensor *t = t_new(DT_F64, nd, sh); return a_obj(t); }

/* ---- complex arithmetic of two numbers (Python's rules: Smith's division, integer powers multiplied out) */
static void cmul(double a, double b, double c, double d, double *r, double *i) { *r = a * c - b * d; *i = a * d + b * c; }
static void cdiv(double a, double b, double c, double d, double *r, double *i) {
    if (c == 0 && d == 0) a_error("ZeroDivisionError", "complex division by zero");
    if (fabs(c) >= fabs(d)) { double q = d / c, den = c + d * q; *r = (a + b * q) / den; *i = (b - a * q) / den; }
    else { double q = c / d, den = c * q + d; *r = (a * q + b) / den; *i = (b * q - a) / den; }
}
static void cpow_(double a, double b, double c, double d, double *r, double *i) {
    if (d == 0 && c == floor(c) && fabs(c) <= 100) {
        long n = (long)c; double xr = 1, xi = 0, pr = a, pi_ = b; long k = n < 0 ? -n : n;
        while (k) { if (k & 1) { double t = xr * pr - xi * pi_; xi = xr * pi_ + xi * pr; xr = t; } k >>= 1; if (k) { double t = pr * pr - pi_ * pi_; pi_ = 2 * pr * pi_; pr = t; } }
        if (n < 0) cdiv(1, 0, xr, xi, &xr, &xi);
        *r = xr; *i = xi; return;
    }
    if (a == 0 && b == 0) { if (c == 0 && d == 0) { *r = 1; *i = 0; return; } if (d != 0 || c < 0) a_error("ZeroDivisionError", "0.0 to a negative or complex power"); *r = 0; *i = 0; return; }
    double lr = log(hypot(a, b)), th = atan2(b, a);
    double er = c * lr - d * th, ei = d * lr + c * th, m = exp(er);
    *r = m * cos(ei); *i = m * sin(ei);
}
static void cop(int op, double a, double b, double c, double d, double *r, double *i) {
    switch (op) {
    case OP_ADD: *r = a + c; *i = b + d; return;
    case OP_SUB: *r = a - c; *i = b - d; return;
    case OP_MUL: cmul(a, b, c, d, r, i); return;
    case OP_DIV: cdiv(a, b, c, d, r, i); return;
    case OP_POW: cpow_(a, b, c, d, r, i); return;
    default: a_error("TypeError", "this operator does not apply to complex numbers");
    }
}

/* ---- broadcasting */
static int ct_bshape(int na, const int64_t *sa, int nb, const int64_t *sb, int64_t *out) {
    int nd = na > nb ? na : nb;
    for (int i = 0; i < nd; i++) {
        int64_t x = i < nd - na ? 1 : sa[i - (nd - na)], y = i < nd - nb ? 1 : sb[i - (nd - nb)];
        if (x != y && x != 1 && y != 1) a_error("ValueError", "operands could not be broadcast together");
        out[i] = x == 1 ? y : x;
    }
    return nd;
}
static void ct_bstr(int n_, const int64_t *s_, int nd, int64_t *st) {
    int64_t s = 1;
    for (int i = nd - 1; i >= 0; i--) { int ai = i - (nd - n_); if (ai < 0 || s_[ai] == 1) st[i] = 0; else st[i] = s; if (ai >= 0) s *= s_[ai]; }
}

AV a_cplx_binop(int op, AV a, AV b) {
    double ar, ai, br, bi;
    if (cx_scalar(a, &ar, &ai) && cx_scalar(b, &br, &bi)) {
        if (op == OP_EQ || op == OP_NE) return a_bool((ar == br && ai == bi) == (op == OP_EQ));
        double r, i; cop(op, ar, ai, br, bi, &r, &i); return a_cplx(r, i);
    }
    if (op == OP_MATMUL) {
        int sp = a_ntmp; AV A = as_ct(a); A_PUSH(A); AV B = as_ct(b); A_PUSH(B);
        ACT *x = CT(A), *y = CT(B);
        int64_t m = x->nd == 2 ? x->sh[0] : 1, k = x->nd >= 1 ? x->sh[x->nd - 1] : 1, k2 = y->nd == 2 ? y->sh[0] : y->sh[0], nn = y->nd == 2 ? y->sh[1] : 1;
        if (x->nd < 1 || y->nd < 1 || x->nd > 2 || y->nd > 2 || k != k2) a_error("ValueError", "matmul: shapes do not match");
        int64_t sh[2]; int nd = 0; if (x->nd == 2) sh[nd++] = m; if (y->nd == 2) sh[nd++] = nn;
        AV R = ct_new(nd, sh); double *rp = CT(R)->p;
        for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < nn; j++) {
            double sr = 0, si = 0;
            for (int64_t q = 0; q < k; q++) {
                double xr = x->p[2 * (i * k + q)], xi = x->p[2 * (i * k + q) + 1], yr = y->p[2 * (q * nn + j)], yi = y->p[2 * (q * nn + j) + 1];
                sr += xr * yr - xi * yi; si += xr * yi + xi * yr;
            }
            rp[2 * (i * nn + j)] = sr; rp[2 * (i * nn + j) + 1] = si;
        }
        a_ntmp = sp; return ct_result(R);
    }
    if (op != OP_ADD && op != OP_SUB && op != OP_MUL && op != OP_DIV && op != OP_POW && op != OP_EQ && op != OP_NE)
        a_error("TypeError", "this operator does not apply to complex numbers");
    int sp = a_ntmp; AV A = as_ct(a); A_PUSH(A); AV B = as_ct(b); A_PUSH(B);
    ACT *x = CT(A), *y = CT(B); int64_t sh[MAXD], sa[MAXD], sb[MAXD];
    int nd = ct_bshape(x->nd, x->sh, y->nd, y->sh, sh);
    ct_bstr(x->nd, x->sh, nd, sa); ct_bstr(y->nd, y->sh, nd, sb);
    int cmp = op == OP_EQ || op == OP_NE;
    AV R = cmp ? a_obj(t_new(DT_BOOL, nd, sh)) : ct_new(nd, sh); A_PUSH(R);
    int64_t n = 1; for (int i = 0; i < nd; i++) n *= sh[i];
    int64_t idx[MAXD] = {0}, oa = 0, ob = 0;
    for (int64_t e = 0; e < n; e++) {
        double r, i_, xr = x->p[2 * oa], xi = x->p[2 * oa + 1], yr = y->p[2 * ob], yi = y->p[2 * ob + 1];
        if (cmp) ((uint8_t *)TT(R)->p)[e] = (xr == yr && xi == yi) == (op == OP_EQ);
        else { cop(op, xr, xi, yr, yi, &r, &i_); CT(R)->p[2 * e] = r; CT(R)->p[2 * e + 1] = i_; }
        for (int d = nd - 1; d >= 0; d--) { idx[d]++; oa += sa[d]; ob += sb[d]; if (idx[d] < sh[d]) break; oa -= sa[d] * sh[d]; ob -= sb[d] * sh[d]; idx[d] = 0; }
    }
    a_ntmp = sp; return cmp ? R : ct_result(R);
}
AV a_cplx_neg(AV a) {
    if (is_cx(a)) return a_cplx(-CX(a)->re, -CX(a)->im);
    AV r = ct_new(CT(a)->nd, CT(a)->sh); for (int64_t i = 0; i < 2 * CT(a)->n; i++) CT(r)->p[i] = -CT(a)->p[i]; return r;
}
int a_cplx_eq(AV a, AV b) {
    double ar, ai, br, bi;
    if (cx_scalar(a, &ar, &ai) && cx_scalar(b, &br, &bi)) return ar == br && ai == bi;
    if (is_ct(a) && is_ct(b)) { if (CT(a)->n != CT(b)->n || CT(a)->nd != CT(b)->nd) return 0; for (int64_t i = 0; i < 2 * CT(a)->n; i++) if (CT(a)->p[i] != CT(b)->p[i]) return 0; return 1; }
    return 0;
}

/* ---- printing: a number as apsi run writes it (1.0+2.0i), an array as NumPy writes complex128 */
static void cx_num(char *out, size_t cap, double x) {
    if (isnan(x)) snprintf(out, cap, "nan"); else if (isinf(x)) snprintf(out, cap, x > 0 ? "inf" : "-inf"); else a_float_repr(x, out, cap);
}
char *a_cplx_scalar_text(AV v) {
    char r[64], i[64], *s = (char *)malloc(160); double im = CX(v)->im;
    cx_num(r, sizeof r, CX(v)->re); cx_num(i, sizeof i, fabs(im));
    snprintf(s, 160, "%s%s%si", r, im >= 0 || isnan(im) ? "+" : "-", i); return s;
}
/* NumPy's 'maxprec' positional text of one float: the fewest fraction digits (at most 8) that read back the same */
static void np_pos(double x, char *out, size_t cap) {
    if (isnan(x)) { snprintf(out, cap, "nan"); return; }
    if (isinf(x)) { snprintf(out, cap, x > 0 ? "inf" : "-inf"); return; }
    int d; char t[128];
    for (d = 0; d <= 8; d++) { snprintf(t, sizeof t, "%.*f", d, x); if (strtod(t, NULL) == x) break; }
    if (d > 8) { snprintf(t, sizeof t, "%.8f", x); size_t L = strlen(t); while (L > 0 && t[L - 1] == '0') t[--L] = 0; }
    if (!strchr(t, '.')) strcat(t, ".");
    if (x == 0 && signbit(x) && t[0] != '-') { memmove(t + 1, t, strlen(t) + 1); t[0] = '-'; }
    snprintf(out, cap, "%s", t);
}
typedef struct { int sci, pad_l, pad_r, forced_sign, prec, exp_w; } NpFmt;
/* the shortest mantissa digits (at most 8 after the point) of x in scientific form */
static int np_sci_prec(double x) {
    char t[64]; int d;
    for (d = 0; d <= 8; d++) { snprintf(t, sizeof t, "%.*e", d, x); if (strtod(t, NULL) == x) break; }
    if (d > 8) { snprintf(t, sizeof t, "%.8e", x); char *e = strchr(t, 'e'); char *q = e - 1; d = 8; while (*q == '0' && d > 0) { q--; d--; } }
    return d;
}
static NpFmt np_fill(const double *v, int64_t n, int stride, int force_sign) {
    NpFmt f = {0, 0, 0, force_sign, 0, 2}; double mx = 0, mn = INFINITY; int any = 0;
    for (int64_t i = 0; i < n; i++) { double x = fabs(v[i * stride]); if (!isfinite(x) || x == 0) continue; any = 1; if (x > mx) mx = x; if (x < mn) mn = x; }
    if (any && (mx >= 1e8 || mn < 1e-4 || mx / mn > 1e3)) f.sci = 1;
    for (int64_t i = 0; i < n; i++) {
        double x = v[i * stride]; char t[128];
        if (f.sci) {
            if (!isfinite(x)) { np_pos(x, t, sizeof t); int il = (int)strlen(t) + (force_sign && t[0] != '-' && !isnan(x)); if (il > f.pad_l) f.pad_l = il; continue; }
            int p = np_sci_prec(x); if (p > f.prec) f.prec = p;
            snprintf(t, sizeof t, "%.*e", p, x); char *e = strchr(t, 'e'); int ev = abs(atoi(e + 1)); int ew = ev >= 100 ? 3 : 2; if (ew > f.exp_w) f.exp_w = ew;
            int il = (int)(strchr(t, '.') ? strchr(t, '.') - t : e - t) + (force_sign && t[0] != '-');
            if (il > f.pad_l) f.pad_l = il;
        } else {
            np_pos(x, t, sizeof t);
            const char *dot = strchr(t, '.');
            int il = dot ? (int)(dot - t) : (int)strlen(t);
            if (force_sign && t[0] != '-' && !isnan(x)) il++;
            int fl = dot ? (int)strlen(dot + 1) : 0;
            if (il > f.pad_l) f.pad_l = il; if (fl > f.pad_r) f.pad_r = fl;
        }
    }
    if (f.sci) f.pad_r = f.prec;
    return f;
}
static void np_one(double x, const NpFmt *f, char *out, size_t cap) {
    char t[128], u[160];
    if (f->sci && isfinite(x)) {
        snprintf(t, sizeof t, "%.*e", f->prec, x);
        char *e = strchr(t, 'e'); int ev = atoi(e + 1); char mant[96]; snprintf(mant, sizeof mant, "%.*s", (int)(e - t), t);
        if (!strchr(mant, '.')) strcat(mant, ".");
        snprintf(u, sizeof u, "%s%se%c%0*d", f->forced_sign && t[0] != '-' ? "+" : "", mant, ev < 0 ? '-' : '+', f->exp_w, ev < 0 ? -ev : ev);
        int il = (int)(strchr(u, '.') - u);
        snprintf(out, cap, "%*s%s", f->pad_l - il > 0 ? f->pad_l - il : 0, "", u); return;
    }
    np_pos(x, t, sizeof t);
    if (f->forced_sign && t[0] != '-' && !isnan(x)) snprintf(u, sizeof u, "+%s", t); else snprintf(u, sizeof u, "%s", t);
    if (isnan(x) || isinf(x)) { int w = f->sci ? f->pad_l + 1 + f->prec + 2 + f->exp_w : f->pad_l + 1 + f->pad_r; snprintf(out, cap, "%*s", w, u); return; }
    const char *dot = strchr(u, '.'); int il = dot ? (int)(dot - u) : (int)strlen(u); int fl = dot ? (int)strlen(dot + 1) : 0;
    snprintf(out, cap, "%*s%s%*s", f->pad_l - il, "", u, f->pad_r - fl, "");
}
static void ct_elem(ACT *t, int64_t i, const NpFmt *fr, const NpFmt *fi, char *out, size_t cap) {
    char r[160], im[160]; np_one(t->p[2 * i], fr, r, sizeof r); np_one(t->p[2 * i + 1], fi, im, sizeof im);
    size_t L = strlen(im); size_t k = L; while (k > 0 && im[k - 1] == ' ') k--;
    snprintf(out, cap, "%s%.*sj%s", r, (int)k, im, im + k);
}
static void ct_rec(TB *b, ACT *t, int ax, int64_t off, int64_t stride_total, const NpFmt *fr, const NpFmt *fi, int indent, int sum) {
    int64_t len = t->sh[ax], inner = stride_total / (len ? len : 1); int last = ax == t->nd - 1, line = indent + 1;
    tb_put(b, "[");
    for (int64_t i = 0; i < len; i++) {
        if (sum && len > 6 && i == 3) {
            if (last) { tb_put(b, "..., "); line += 5; }
            else { tb_put(b, "...,"); for (int k = 0; k < t->nd - ax - 1; k++) tb_put(b, "\n"); for (int k = 0; k <= indent; k++) tb_put(b, " "); }
            i = len - 3;
        }
        if (last) {
            char s[400]; ct_elem(t, off + i, fr, fi, s, sizeof s); int k = (int)strlen(s);
            if (i > 0 && line + k + 2 > 100) { if (b->n && b->p[b->n - 1] == ' ') b->p[--b->n] = 0; tb_put(b, "\n"); for (int q = 0; q <= indent; q++) tb_put(b, " "); line = indent + 1; }
            tb_put(b, s); line += k;
            if (i < len - 1) { tb_put(b, ", "); line += 2; }
        } else {
            ct_rec(b, t, ax + 1, off + i * inner, inner, fr, fi, indent + 1, sum);
            if (i < len - 1) { tb_put(b, ","); for (int k = 0; k < t->nd - ax - 1; k++) tb_put(b, "\n"); for (int k = 0; k <= indent; k++) tb_put(b, " "); }
        }
    }
    tb_put(b, "]");
}
char *a_cplx_text(AV v, int top) {
    if (is_cx(v)) return a_cplx_scalar_text(v);
    ACT *t = CT(v); TB b = {0};
    if (!top && t->n > 8) {
        char s[128]; int k = snprintf(s, sizeof s, "Tensor[");
        for (int i = 0; i < t->nd; i++) k += snprintf(s + k, sizeof s - (size_t)k, i ? "x%lld" : "%lld", (long long)t->sh[i]);
        snprintf(s + k, sizeof s - (size_t)k, "]"); tb_put(&b, s); return b.p;
    }
    if (t->n == 0) { tb_put(&b, "[]"); return b.p; }
    NpFmt fr = np_fill(t->p, t->n, 2, 0), fi = np_fill(t->p + 1, t->n, 2, 1);
    if (t->nd == 0) { char s[400]; NpFmt z = {0, 0, 0, 0, 0, 2}, zi = {0, 0, 0, 1, 0, 2}; ct_elem(t, 0, &z, &zi, s, sizeof s); tb_put(&b, s); return b.p; }
    ct_rec(&b, t, 0, 0, t->n, &fr, &fi, 0, t->n > 40);
    return b.p;
}

/* ---- indexing: int, slice, or a tuple of them */
static AV ct_index(AV v, AV idx, AV set) {
    ACT *t = CT(v); AV items[MAXD]; int ni = 1;
    if (a_is(idx, O_TUPLE)) { ni = (int)((AList *)idx.o)->len; if (ni > t->nd) a_error("IndexError", "too many indices"); for (int i = 0; i < ni; i++) items[i] = ((AList *)idx.o)->v[i]; }
    else items[0] = idx;
    if (ni > t->nd) a_error("IndexError", "too many indices for a %d-d tensor", t->nd);
    int64_t lo[MAXD], st[MAXD], cnt[MAXD], strides[MAXD]; int keep[MAXD], ond = 0; int64_t osh[MAXD];
    int64_t s = 1; for (int d = t->nd - 1; d >= 0; d--) { strides[d] = s; s *= t->sh[d]; }
    for (int d = 0; d < t->nd; d++) {
        AV it = d < ni ? items[d] : A_NONE;
        if (d < ni && (it.t == T_INT || it.t == T_BOOL)) {
            int64_t k = it.i; if (k < 0) k += t->sh[d]; if (k < 0 || k >= t->sh[d]) a_error("IndexError", "index %lld is out of range for an axis of size %lld", (long long)it.i, (long long)t->sh[d]);
            lo[d] = k; st[d] = 1; cnt[d] = 1; keep[d] = 0;
        } else if (d >= ni || a_is(it, O_SLICE)) {
            int64_t a0 = 0, b0 = t->sh[d], s0 = 1;
            if (d < ni) a_slice_bounds(it, t->sh[d], &a0, &b0, &s0);
            int64_t c = s0 > 0 ? (b0 > a0 ? (b0 - a0 + s0 - 1) / s0 : 0) : (a0 > b0 ? (a0 - b0 - s0 - 1) / (-s0) : 0);
            lo[d] = a0; st[d] = s0; cnt[d] = c; keep[d] = 1; osh[ond++] = c;
        } else a_error("IndexError", "a complex tensor takes integers and slices as indices");
    }
    int64_t n = 1; for (int d = 0; d < t->nd; d++) n *= cnt[d];
    int sp = a_ntmp; AV R = A_NONE; ACT *src = NULL;
    if (set.t == 255) { R = ct_new(ond, osh); A_PUSH(R); }
    else { AV sv = as_ct(set); A_PUSH(sv); src = CT(sv); if (src->n != 1 && src->n != n) a_error("ValueError", "cannot assign %lld values to %lld places", (long long)src->n, (long long)n); }
    int64_t pos[MAXD] = {0};
    for (int64_t e = 0; e < n; e++) {
        int64_t off = 0; for (int d = 0; d < t->nd; d++) off += (lo[d] + pos[d] * st[d]) * strides[d];
        if (set.t == 255) { CT(R)->p[2 * e] = t->p[2 * off]; CT(R)->p[2 * e + 1] = t->p[2 * off + 1]; }
        else { int64_t q = src->n == 1 ? 0 : e; t->p[2 * off] = src->p[2 * q]; t->p[2 * off + 1] = src->p[2 * q + 1]; }
        for (int d = t->nd - 1; d >= 0; d--) { if (++pos[d] < cnt[d]) break; pos[d] = 0; }
    }
    a_ntmp = sp;
    return set.t == 255 ? ct_result(R) : A_NONE;
}
AV a_cplx_getitem(AV v, AV i) { return ct_index(v, i, (AV){.t = 255}); }
void a_cplx_setitem(AV v, AV i, AV x) { ct_index(v, i, x); }
int64_t a_cplx_len(AV v) { if (CT(v)->nd == 0) a_error("TypeError", "len() of a 0-d tensor"); return CT(v)->sh[0]; }

static AV ct_tolist_rec(ACT *t, int ax, int64_t off) {
    if (ax == t->nd) return a_cplx(t->p[2 * off], t->p[2 * off + 1]);
    int64_t len = t->sh[ax], inner = 1; for (int d = ax + 1; d < t->nd; d++) inner *= t->sh[d];
    AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l);
    for (int64_t i = 0; i < len; i++) a_list_append((AList *)l.o, ct_tolist_rec(t, ax + 1, off + i * inner));
    a_ntmp = sp; return l;
}

/* ---- element-wise functions and reductions */
static AV ct_map(AV v, int what) {      /* 0 conj, 1 exp, 2 log, 3 sqrt, 4 sin, 5 cos, 6 tanh */
    int sp = a_ntmp; AV a = as_ct(v); A_PUSH(a); AV r = ct_new(CT(a)->nd, CT(a)->sh); A_PUSH(r);
    for (int64_t i = 0; i < CT(a)->n; i++) {
        double x = CT(a)->p[2 * i], y = CT(a)->p[2 * i + 1], u, w;
        switch (what) {
        case 0: u = x; w = -y; break;
        case 1: { double m = exp(x); u = m * cos(y); w = m * sin(y); break; }
        case 2: u = log(hypot(x, y)); w = atan2(y, x); break;
        case 3: { double m = hypot(x, y); if (m == 0) { u = 0; w = y; break; } double t = sqrt((m + fabs(x)) / 2); if (x >= 0) { u = t; w = y / (2 * t); } else { u = fabs(y) / (2 * t); w = copysign(t, y); } break; }
        case 4: u = sin(x) * cosh(y); w = cos(x) * sinh(y); break;
        case 5: u = cos(x) * cosh(y); w = -sin(x) * sinh(y); break;
        default: { double den = cosh(2 * x) + cos(2 * y); u = sinh(2 * x) / den; w = sin(2 * y) / den; break; }
        }
        CT(r)->p[2 * i] = u; CT(r)->p[2 * i + 1] = w;
    }
    a_ntmp = sp; return ct_result(r);
}
static AV ct_part(AV v, int what) {     /* 0 real, 1 imag, 2 abs, 3 angle */
    int sp = a_ntmp; AV a = as_ct(v); A_PUSH(a); ACT *t = CT(a);
    AV r = ct_real_tensor(t->nd, t->sh); A_PUSH(r); double *o = (double *)TT(r)->p;
    for (int64_t i = 0; i < t->n; i++) { double x = t->p[2 * i], y = t->p[2 * i + 1]; o[i] = what == 0 ? x : what == 1 ? y : what == 2 ? hypot(x, y) : atan2(y, x); }
    a_ntmp = sp;
    if (t->nd == 0) return a_float(o[0]);
    return r;
}
static AV ct_sum(AV v, AV axis, int mean) {
    int sp = a_ntmp; AV a = as_ct(v); A_PUSH(a); ACT *t = CT(a);
    if (axis.t == T_NONE) {
        double sr = 0, si = 0; for (int64_t i = 0; i < t->n; i++) { sr += t->p[2 * i]; si += t->p[2 * i + 1]; }
        if (mean) { sr /= (double)t->n; si /= (double)t->n; }
        a_ntmp = sp; return a_cplx(sr, si);
    }
    int ax = (int)axis.i; if (ax < 0) ax += t->nd; if (ax < 0 || ax >= t->nd) a_error("ValueError", "axis out of range");
    int64_t osh[MAXD]; int ond = 0; for (int d = 0; d < t->nd; d++) if (d != ax) osh[ond++] = t->sh[d];
    AV r = ct_new(ond, osh); A_PUSH(r);
    int64_t outer = 1, inner = 1; for (int d = 0; d < ax; d++) outer *= t->sh[d]; for (int d = ax + 1; d < t->nd; d++) inner *= t->sh[d];
    int64_t len = t->sh[ax];
    for (int64_t o = 0; o < outer; o++) for (int64_t i = 0; i < inner; i++) {
        double sr = 0, si = 0; for (int64_t k = 0; k < len; k++) { int64_t q = (o * len + k) * inner + i; sr += t->p[2 * q]; si += t->p[2 * q + 1]; }
        if (mean) { sr /= (double)len; si /= (double)len; }
        CT(r)->p[2 * (o * inner + i)] = sr; CT(r)->p[2 * (o * inner + i) + 1] = si;
    }
    a_ntmp = sp; return ct_result(r);
}
static AV ct_reshape(AV v, int nd, const int64_t *sh) {
    int sp = a_ntmp; AV a = as_ct(v); A_PUSH(a); int64_t s2[MAXD], n = 1; int neg = -1;
    for (int i = 0; i < nd; i++) { s2[i] = sh[i]; if (sh[i] < 0) neg = i; else n *= sh[i]; }
    if (neg >= 0) s2[neg] = n ? CT(a)->n / n : 0;
    int64_t m = 1; for (int i = 0; i < nd; i++) m *= s2[i];
    if (m != CT(a)->n) a_error("ValueError", "cannot reshape %lld values", (long long)CT(a)->n);
    AV r = ct_new(nd, s2); memcpy(CT(r)->p, CT(a)->p, sizeof(double) * 2 * (size_t)m); a_ntmp = sp; return r;
}
static AV ct_permute(AV v, int nd, const int *perm) {
    int sp = a_ntmp; AV a = as_ct(v); A_PUSH(a); ACT *t = CT(a);
    if (nd != t->nd) a_error("ValueError", "axes don't match the tensor");
    int64_t osh[MAXD], st[MAXD], s = 1; for (int d = t->nd - 1; d >= 0; d--) { st[d] = s; s *= t->sh[d]; }
    for (int d = 0; d < nd; d++) osh[d] = t->sh[perm[d]];
    AV r = ct_new(nd, osh); A_PUSH(r); int64_t pos[MAXD] = {0};
    for (int64_t e = 0; e < t->n; e++) {
        int64_t off = 0; for (int d = 0; d < nd; d++) off += pos[d] * st[perm[d]];
        CT(r)->p[2 * e] = t->p[2 * off]; CT(r)->p[2 * e + 1] = t->p[2 * off + 1];
        for (int d = nd - 1; d >= 0; d--) { if (++pos[d] < osh[d]) break; pos[d] = 0; }
    }
    a_ntmp = sp; return r;
}
static AV ct_transpose(AV v) { int nd = is_ct(v) ? CT(v)->nd : 0; int perm[MAXD]; for (int i = 0; i < nd; i++) perm[i] = nd - 1 - i; return nd < 2 ? v : ct_permute(v, nd, perm); }
static AV ct_join(AV list, int axis, int stack) {
    AList *l = (AList *)list.o; if (!l->len) a_error("ValueError", "need at least one array to join");
    int sp = a_ntmp; AV items = a_list(0, NULL); A_PUSH(items);
    for (int64_t i = 0; i < l->len; i++) { AV c = as_ct(l->v[i]); if (stack) { int64_t sh[MAXD + 1]; int nd = CT(c)->nd + 1, ax = axis < 0 ? axis + nd : axis; for (int d = 0, s = 0; d < nd; d++) sh[d] = d == ax ? 1 : CT(c)->sh[s++]; c = ct_reshape(c, nd, sh); } a_list_append((AList *)items.o, c); }
    AList *L = (AList *)items.o; ACT *f = CT(L->v[0]); int nd = f->nd, ax = axis < 0 ? axis + nd : axis;
    int64_t sh[MAXD]; for (int d = 0; d < nd; d++) sh[d] = f->sh[d]; sh[ax] = 0;
    for (int64_t i = 0; i < L->len; i++) { ACT *c = CT(L->v[i]); if (c->nd != nd) a_error("ValueError", "all arrays need the same number of dimensions"); sh[ax] += c->sh[ax]; }
    AV r = ct_new(nd, sh); A_PUSH(r);
    int64_t outer = 1, inner = 1; for (int d = 0; d < ax; d++) outer *= sh[d]; for (int d = ax + 1; d < nd; d++) inner *= sh[d];
    int64_t at = 0;
    for (int64_t i = 0; i < L->len; i++) {
        ACT *c = CT(L->v[i]); int64_t len = c->sh[ax];
        for (int64_t o = 0; o < outer; o++) memcpy(CT(r)->p + 2 * ((o * sh[ax] + at) * inner), c->p + 2 * (o * len * inner), sizeof(double) * 2 * (size_t)(len * inner));
        at += len;
    }
    a_ntmp = sp; return r;
}
static AV ct_kron(AV x, AV y) {
    int sp = a_ntmp; AV A = as_ct(x); A_PUSH(A); AV B = as_ct(y); A_PUSH(B);
    ACT *a = CT(A), *b = CT(B);
    if (a->nd == 1 && b->nd == 1) {
        int64_t sh[1] = {a->n * b->n}; AV r = ct_new(1, sh); A_PUSH(r);
        for (int64_t i = 0; i < a->n; i++) for (int64_t j = 0; j < b->n; j++) { double u, w; cmul(a->p[2 * i], a->p[2 * i + 1], b->p[2 * j], b->p[2 * j + 1], &u, &w); CT(r)->p[2 * (i * b->n + j)] = u; CT(r)->p[2 * (i * b->n + j) + 1] = w; }
        a_ntmp = sp; return r;
    }
    if (a->nd != 2 || b->nd != 2) a_error("ValueError", "kron of vectors or matrices");
    int64_t m = a->sh[0], n = a->sh[1], p = b->sh[0], q = b->sh[1], sh[2] = {m * p, n * q};
    AV r = ct_new(2, sh); A_PUSH(r);
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) for (int64_t k = 0; k < p; k++) for (int64_t l = 0; l < q; l++) {
        double u, w; cmul(a->p[2 * (i * n + j)], a->p[2 * (i * n + j) + 1], b->p[2 * (k * q + l)], b->p[2 * (k * q + l) + 1], &u, &w);
        int64_t o = (i * p + k) * (n * q) + j * q + l; CT(r)->p[2 * o] = u; CT(r)->p[2 * o + 1] = w;
    }
    a_ntmp = sp; return r;
}

/* the built-ins that meet a complex argument (sum, abs, exp, reshape, ...); 255 when this one is not for complex */
static int ct_shape_args(int n, AV *a, int64_t *sh) {
    if (n == 1 && (a_is(a[0], O_LIST) || a_is(a[0], O_TUPLE))) { AList *l = (AList *)a[0].o; for (int64_t i = 0; i < l->len; i++) sh[i] = l->v[i].i; return (int)l->len; }
    for (int i = 0; i < n; i++) sh[i] = a[i].i;
    return n;
}
AV a_cplx_builtin(const char *name, int n, AV *a, ADict *kw) {
    AV ax = kwget(kw, "axis");
    if (!strcmp(name, "complex")) { double r = 0, i = 0, r2, i2; if (n > 0 && !cx_scalar(a[0], &r, &i)) a_error("TypeError", "complex() takes numbers"); if (n > 1) { if (!cx_scalar(a[1], &r2, &i2)) a_error("TypeError", "complex() takes numbers"); r -= i2; i += r2; } return a_cplx(r, i); }
    if (!strcmp(name, "sum") || !strcmp(name, "mean")) return ct_sum(a[0], n > 1 ? a[1] : ax, name[0] == 'm');
    if (!strcmp(name, "abs")) return ct_part(a[0], 2);
    if (!strcmp(name, "real")) return ct_part(a[0], 0);
    if (!strcmp(name, "imag")) return ct_part(a[0], 1);
    if (!strcmp(name, "angle")) return ct_part(a[0], 3);
    if (!strcmp(name, "conj") || !strcmp(name, "conjugate")) return ct_map(a[0], 0);
    if (!strcmp(name, "exp")) return ct_map(a[0], 1);
    if (!strcmp(name, "log")) return ct_map(a[0], 2);
    if (!strcmp(name, "sqrt")) return ct_map(a[0], 3);
    if (!strcmp(name, "sin")) return ct_map(a[0], 4);
    if (!strcmp(name, "cos")) return ct_map(a[0], 5);
    if (!strcmp(name, "tanh")) return ct_map(a[0], 6);
    if (!strcmp(name, "reshape")) { int64_t sh[MAXD]; int nd = ct_shape_args(n - 1, a + 1, sh); return ct_reshape(a[0], nd, sh); }
    if (!strcmp(name, "flatten")) { int64_t sh[1] = {-1}; return ct_reshape(a[0], 1, sh); }
    if (!strcmp(name, "transpose")) {
        if (n > 1) { int perm[MAXD]; int nd = 0; if (a_is(a[1], O_LIST) || a_is(a[1], O_TUPLE)) { AList *l = (AList *)a[1].o; for (int64_t i = 0; i < l->len; i++) perm[nd++] = (int)l->v[i].i; } else for (int i = 1; i < n; i++) perm[nd++] = (int)a[i].i; return ct_permute(a[0], nd, perm); }
        return ct_transpose(as_ct(a[0]));
    }
    if (!strcmp(name, "shape")) { AV c = as_ct(a[0]); AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l); for (int i = 0; i < CT(c)->nd; i++) a_list_append((AList *)l.o, a_int(CT(c)->sh[i])); AV t = a_tuple(((AList *)l.o)->len, ((AList *)l.o)->v); a_ntmp = sp; return t; }
    if (!strcmp(name, "ndim")) return a_int(is_ct(a[0]) ? CT(a[0])->nd : 0);
    if (!strcmp(name, "tolist")) { AV c = as_ct(a[0]); return CT(c)->nd == 0 ? ct_result(c) : ct_tolist_rec(CT(c), 0, 0); }
    if (!strcmp(name, "tensor") || !strcmp(name, "array")) {
        AV dt = n > 1 ? a[1] : kwget(kw, "dtype");
        if (dt.t != T_NONE && a_is(dt, O_STR) && strncmp(((AStr *)dt.o)->s, "complex", 7)) a_error("TypeError", "cannot convert complex numbers to %s", ((AStr *)dt.o)->s);
        AV c = as_ct(a[0]); if (c.o == a[0].o) { AV r = ct_new(CT(c)->nd, CT(c)->sh); memcpy(CT(r)->p, CT(c)->p, sizeof(double) * 2 * (size_t)CT(c)->n); return r; } return c;
    }
    if (!strcmp(name, "matmul") || !strcmp(name, "dot")) return a_cplx_binop(OP_MATMUL, a[0], a[1]);
    if (!strcmp(name, "kron")) return ct_kron(a[0], a[1]);
    if (!strcmp(name, "outer")) { int sp = a_ntmp; AV x = as_ct(a[0]); A_PUSH(x); AV y = as_ct(a[1]); A_PUSH(y); int64_t s1[2] = {CT(x)->n, 1}, s2[2] = {1, CT(y)->n}; AV xr = ct_reshape(x, 2, s1); A_PUSH(xr); AV yr = ct_reshape(y, 2, s2); A_PUSH(yr); AV r = a_cplx_binop(OP_MUL, xr, yr); a_ntmp = sp; return r; }
    if (!strcmp(name, "concat") || !strcmp(name, "concatenate") || !strcmp(name, "stack")) return ct_join(a[0], (int)(n > 1 ? a[1].i : ax.t == T_INT ? ax.i : 0), name[0] == 's');
    if (!strcmp(name, "zeros_like") || !strcmp(name, "ones_like")) { AV c = as_ct(a[0]); AV r = ct_new(CT(c)->nd, CT(c)->sh); if (name[0] == 'o') for (int64_t i = 0; i < CT(r)->n; i++) CT(r)->p[2 * i] = 1; return r; }
    if (!strcmp(name, "zeros") || !strcmp(name, "ones")) { int64_t sh[MAXD]; int nd = ct_shape_args(n, a, sh); AV r = ct_new(nd, sh); if (name[0] == 'o') for (int64_t i = 0; i < CT(r)->n; i++) CT(r)->p[2 * i] = 1; return r; }
    if (!strcmp(name, "isnan")) { int sp = a_ntmp; AV c = as_ct(a[0]); A_PUSH(c); AV r = a_obj(t_new(DT_BOOL, CT(c)->nd, CT(c)->sh)); for (int64_t i = 0; i < CT(c)->n; i++) ((uint8_t *)TT(r)->p)[i] = isnan(CT(c)->p[2 * i]) || isnan(CT(c)->p[2 * i + 1]); a_ntmp = sp; return r; }
    if (!strcmp(name, "len")) return a_int(a_cplx_len(as_ct(a[0])));
    if (!strcmp(name, "float") || !strcmp(name, "int")) a_error("TypeError", "can't convert complex to %s", name);
    return (AV){.t = 255};
}
int a_cplx_wants(const char *name, int n, AV *a, ADict *kw) {
    for (int i = 0; i < n && i < 2; i++) if (is_ct(a[i]) || is_cx(a[i])) return 1;
    AV dt = kwget(kw, "dtype"); if (a_is(dt, O_STR) && !strncmp(((AStr *)dt.o)->s, "complex", 7)) return 1;
    if (n > 1 && a_is(a[1], O_STR) && !strncmp(((AStr *)a[1].o)->s, "complex", 7) && (!strcmp(name, "tensor") || !strcmp(name, "array"))) return 1;
    if (n >= 1 && (a_is(a[0], O_LIST) || a_is(a[0], O_TUPLE)) && (!strcmp(name, "tensor") || !strcmp(name, "array") || !strcmp(name, "concat") || !strcmp(name, "stack") || !strcmp(name, "concatenate") || !strcmp(name, "sum"))) {
        AList *l = (AList *)a[0].o; int64_t k = l->len < 64 ? l->len : 64;
        for (int64_t i = 0; i < k; i++) { AV x = l->v[i]; if (is_ct(x) || is_cx(x)) return 1; if ((a_is(x, O_LIST) || a_is(x, O_TUPLE)) && ((AList *)x.o)->len && (is_cx(((AList *)x.o)->v[0]) || is_ct(((AList *)x.o)->v[0]))) return 1; }
    }
    return 0;
}
/* attributes: .real .imag .T .shape .dtype .ndim, and conj() / conjugate() / tolist() / reshape() */
static AV cm_conj(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; return ct_map(s, 0); }
static AV cm_tolist(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; return is_cx(s) ? s : ct_tolist_rec(CT(s), 0, 0); }
static AV cm_reshape(int n, AV *a, ADict *kw, AV s) { (void)kw; int64_t sh[MAXD]; int nd = ct_shape_args(n, a, sh); return ct_reshape(s, nd, sh); }
static AV cm_copy(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (is_cx(s)) return s; AV r = ct_new(CT(s)->nd, CT(s)->sh); memcpy(CT(r)->p, CT(s)->p, sizeof(double) * 2 * (size_t)CT(s)->n); return r; }
AV a_cplx_attr(AV v, const char *name) {
    if (!strcmp(name, "real") || !strcmp(name, "re")) return is_cx(v) ? a_float(CX(v)->re) : ct_part(v, 0);
    if (!strcmp(name, "imag") || !strcmp(name, "im")) return is_cx(v) ? a_float(CX(v)->im) : ct_part(v, 1);
    if (!strcmp(name, "conjugate") || !strcmp(name, "conj")) return a_builtin("conj", cm_conj, v);
    if (!strcmp(name, "tolist")) return a_builtin("tolist", cm_tolist, v);
    if (!strcmp(name, "copy")) return a_builtin("copy", cm_copy, v);
    if (is_ct(v)) {
        if (!strcmp(name, "T")) return ct_transpose(v);
        if (!strcmp(name, "shape")) return a_cplx_builtin("shape", 1, &v, NULL);
        if (!strcmp(name, "ndim")) return a_int(CT(v)->nd);
        if (!strcmp(name, "size")) return a_int(CT(v)->n);
        if (!strcmp(name, "dtype")) return a_str("complex128");
        if (!strcmp(name, "reshape")) return a_builtin("reshape", cm_reshape, v);
    }
    return (AV){.t = 255};
}
AV a_cplx_iter_list(AV v) {               /* iteration: the rows (or the numbers of a vector) */
    ACT *t = CT(v); if (t->nd == 0) a_error("TypeError", "iteration over a 0-d tensor");
    AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l);
    for (int64_t i = 0; i < t->sh[0]; i++) a_list_append((AList *)l.o, a_cplx_getitem(v, a_int(i)));
    a_ntmp = sp; return l;
}
const char *a_cplx_type(AV v) { static char b[96]; if (is_cx(v)) return "complex"; int k = snprintf(b, sizeof b, "Tensor["); for (int i = 0; i < CT(v)->nd; i++) k += snprintf(b + k, sizeof b - (size_t)k, i ? ", %lld" : "%lld", (long long)CT(v)->sh[i]); snprintf(b + k, sizeof b - (size_t)k, "%scomplex128]", CT(v)->nd ? ", " : ""); return b; }

/* __ct(re, im): a complex tensor from its parts (the library modules build FFT results with it) */
BI(lb_ct) {
    (void)self; (void)kw;
    if (n == 1) return as_ct(a[0]);
    int sp = a_ntmp; AV re = a[0], im = a[1];
    if ((re.t == T_FLOAT || re.t == T_INT) && (im.t == T_FLOAT || im.t == T_INT)) return a_cplx(re.t == T_FLOAT ? re.f : (double)re.i, im.t == T_FLOAT ? im.f : (double)im.i);
    AV R = as_tensor(val_of(re)); A_PUSH(R); AV I = as_tensor(val_of(im)); A_PUSH(I);
    R = to_host(R); A_PUSH(R); I = to_host(I); A_PUSH(I);
    int64_t sh[MAXD]; int nd = bshape(TT(R), TT(I), sh);
    AV out = ct_new(nd, sh); A_PUSH(out);
    int64_t sr[MAXD], si[MAXD]; bstrides(TT(R), nd, sh, sr); bstrides(TT(I), nd, sh, si);
    int64_t pos[MAXD] = {0}, orr = 0, oi = 0;
    for (int64_t e = 0; e < CT(out)->n; e++) {
        CT(out)->p[2 * e] = t_get(TT(R), orr); CT(out)->p[2 * e + 1] = t_get(TT(I), oi);
        for (int d = nd - 1; d >= 0; d--) { pos[d]++; orr += sr[d]; oi += si[d]; if (pos[d] < sh[d]) break; orr -= sr[d] * sh[d]; oi -= si[d] * sh[d]; pos[d] = 0; }
    }
    a_ntmp = sp; return ct_result(out);
}
BI(lb_is_cplx) { (void)kw; (void)self; return a_bool(n && (is_ct(a[0]) || is_cx(a[0]))); }
