/* apsi_dec.c: decimal numbers of any precision for native programs (`use hp`), with the arithmetic, rounding
   (half even), exponents and text of Python's decimal module: a value is (-1)^sign · coefficient · 10^exp, the
   coefficient a natural number in base 10^9 limbs. + - * / and integer powers use the precision of the current
   context (__dec("prec")); the hp library sets it the way apsi run's decimal context is set. */

#define DB 1000000000u
typedef struct { uint32_t *d; int64_t n, cap; } BN;       /* little-endian limbs, n == 0 for zero */
typedef struct ADec { AObj h; int sign; int64_t exp; int64_t n; uint32_t d[]; } ADec;
#define DEC(v) ((ADec *)(v).o)

static int64_t dec_prec = 50;
typedef struct { char *p; int64_t n; size_t cap; } DSB;     /* a growing text, always 0-terminated */
static void ds_put(DSB *b, const char *s, int64_t n) { if ((size_t)(b->n + n + 1) > b->cap) { b->cap = (size_t)(b->n + n + 1) * 2; b->p = (char *)realloc(b->p, b->cap); } if (n) memcpy(b->p + b->n, s, (size_t)n); b->n += n; b->p[b->n] = 0; }
static void ds_puts(DSB *b, const char *s) { ds_put(b, s, (int64_t)strlen(s)); }                              /* the context: apsi run's hp sets 50 at start-up */

/* ------------------------------------------------------------------------------------------- natural numbers */
static void bn_init(BN *a) { a->d = NULL; a->n = a->cap = 0; }
static void bn_free(BN *a) { free(a->d); a->d = NULL; a->n = a->cap = 0; }
static void bn_reserve(BN *a, int64_t n) { if (n > a->cap) { int64_t c = n < 4 ? 4 : n; a->d = (uint32_t *)realloc(a->d, sizeof(uint32_t) * (size_t)c); if (!a->d) a_error("MemoryError", "out of memory"); a->cap = c; } }
static void bn_norm(BN *a) { while (a->n > 0 && a->d[a->n - 1] == 0) a->n--; }
static void bn_set_u64(BN *a, uint64_t v) { a->n = 0; bn_reserve(a, 3); while (v) { a->d[a->n++] = (uint32_t)(v % DB); v /= DB; } }
static void bn_copy(BN *a, const BN *b) { bn_reserve(a, b->n); if (b->n) memcpy(a->d, b->d, sizeof(uint32_t) * (size_t)b->n); a->n = b->n; }
static int bn_cmp(const BN *a, const BN *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (int64_t i = a->n - 1; i >= 0; i--) if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    return 0;
}
static void bn_add(BN *r, const BN *a, const BN *b) {                 /* r may be a or b */
    const BN *x = a->n >= b->n ? a : b, *y = a->n >= b->n ? b : a; int64_t n = x->n, m = y->n;
    BN t; bn_init(&t); bn_reserve(&t, n + 1); uint32_t c = 0;
    for (int64_t i = 0; i < n; i++) { uint32_t s = x->d[i] + (i < m ? y->d[i] : 0) + c; c = s >= DB; t.d[i] = c ? s - DB : s; }
    t.n = n; if (c) t.d[t.n++] = 1;
    bn_free(r); *r = t;
}
static void bn_sub(BN *r, const BN *a, const BN *b) {                 /* a >= b; r may be a or b */
    BN t; bn_init(&t); bn_reserve(&t, a->n + 1); int64_t br = 0;
    for (int64_t i = 0; i < a->n; i++) { int64_t s = (int64_t)a->d[i] - (i < b->n ? b->d[i] : 0) - br; br = s < 0; t.d[i] = (uint32_t)(br ? s + DB : s); }
    t.n = a->n; bn_norm(&t); bn_free(r); *r = t;
}
/* column by column (Comba): the products of a column summed in 128 bits, one carry division per column */
static void bn_mul_comba(uint32_t *out, const uint32_t *a, int64_t na, const uint32_t *b, int64_t nb) {
    unsigned __int128 acc = 0;
    for (int64_t k = 0; k < na + nb - 1; k++) {
        int64_t lo = k - nb + 1 > 0 ? k - nb + 1 : 0, hi = k < na - 1 ? k : na - 1;
        uint64_t part = 0; int cnt = 0;
        for (int64_t i = lo; i <= hi; i++) {
            part += (uint64_t)a[i] * b[k - i];
            if (++cnt == 17) { acc += part; part = 0; cnt = 0; }   /* 17 products of < 1e18 stay below 2^64 */
        }
        acc += part;
        uint64_t low = (uint64_t)(acc % DB); out[k] = (uint32_t)low; acc /= DB;
    }
    int64_t k = na + nb - 1; out[k] = (uint32_t)(acc % DB);
}
/* Karatsuba above a few dozen limbs: (a1 B + a0)(b1 B + b0) with three products of half the size */
static void bn_mul_rec(uint32_t *out, const uint32_t *a, int64_t na, const uint32_t *b, int64_t nb) {
    if (na < 48 || nb < 48) {
        if (na == 0 || nb == 0) { memset(out, 0, sizeof(uint32_t) * (size_t)(na + nb)); return; }
        bn_mul_comba(out, a, na, b, nb); return;
    }
    int64_t h = (na > nb ? na : nb) / 2;
    if (h >= na || h >= nb) {                                 /* very unequal sizes: split the longer one */
        const uint32_t *L = na >= nb ? a : b, *S = na >= nb ? b : a; int64_t nl = na >= nb ? na : nb, nsh = na >= nb ? nb : na;
        memset(out, 0, sizeof(uint32_t) * (size_t)(na + nb));
        uint32_t *tmp = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(nsh * 2 + 2));
        for (int64_t off = 0; off < nl; off += nsh) {
            int64_t len = nl - off < nsh ? nl - off : nsh;
            bn_mul_rec(tmp, L + off, len, S, nsh);
            uint32_t c = 0; int64_t i;
            for (i = 0; i < len + nsh; i++) { uint32_t v = out[off + i] + tmp[i] + c; c = v >= DB; out[off + i] = c ? v - DB : v; }
            for (i = off + len + nsh; c && i < na + nb; i++) { uint32_t v = out[i] + c; c = v >= DB; out[i] = c ? v - DB : v; }
        }
        free(tmp); return;
    }
    BN a0 = {(uint32_t *)a, h, h}, a1 = {(uint32_t *)a + h, na - h, na - h}, b0 = {(uint32_t *)b, h, h}, b1 = {(uint32_t *)b + h, nb - h, nb - h};
    BN x0, x1, sa, sb, m; bn_init(&x0); bn_init(&x1); bn_init(&sa); bn_init(&sb); bn_init(&m);
    BN a0n, b0n; bn_init(&a0n); bn_init(&b0n); bn_copy(&a0n, &a0); bn_norm(&a0n); bn_copy(&b0n, &b0); bn_norm(&b0n);
    BN a1n, b1n; bn_init(&a1n); bn_init(&b1n); bn_copy(&a1n, &a1); bn_norm(&a1n); bn_copy(&b1n, &b1); bn_norm(&b1n);
    bn_reserve(&x0, 2 * h + 1); bn_mul_rec(x0.d, a0n.d, a0n.n, b0n.d, b0n.n); x0.n = a0n.n + b0n.n; bn_norm(&x0);
    bn_reserve(&x1, a1n.n + b1n.n + 1); bn_mul_rec(x1.d, a1n.d, a1n.n, b1n.d, b1n.n); x1.n = a1n.n + b1n.n; bn_norm(&x1);
    bn_add(&sa, &a0n, &a1n); bn_add(&sb, &b0n, &b1n);
    bn_reserve(&m, sa.n + sb.n + 1); bn_mul_rec(m.d, sa.d, sa.n, sb.d, sb.n); m.n = sa.n + sb.n; bn_norm(&m);
    bn_sub(&m, &m, &x0); bn_sub(&m, &m, &x1);
    memset(out, 0, sizeof(uint32_t) * (size_t)(na + nb));
    const BN *parts[3] = {&x0, &m, &x1}; int64_t offs[3] = {0, h, 2 * h};
    for (int q = 0; q < 3; q++) {
        uint32_t c = 0; int64_t i, off = offs[q];
        for (i = 0; i < parts[q]->n; i++) { uint32_t v = out[off + i] + parts[q]->d[i] + c; c = v >= DB; out[off + i] = c ? v - DB : v; }
        for (i = off + parts[q]->n; c && i < na + nb; i++) { uint32_t v = out[i] + c; c = v >= DB; out[i] = c ? v - DB : v; }
    }
    bn_free(&x0); bn_free(&x1); bn_free(&sa); bn_free(&sb); bn_free(&m); bn_free(&a0n); bn_free(&b0n); bn_free(&a1n); bn_free(&b1n);
}
static void bn_mul(BN *r, const BN *a, const BN *b) {
    if (!a->n || !b->n) { r->n = 0; return; }
    BN t; bn_init(&t); bn_reserve(&t, a->n + b->n + 1);
    bn_mul_rec(t.d, a->d, a->n, b->d, b->n);
    t.n = a->n + b->n; bn_norm(&t); bn_free(r); *r = t;
}
static void bn_mul_small(BN *a, uint32_t m) {
    uint64_t c = 0; for (int64_t i = 0; i < a->n; i++) { uint64_t p = (uint64_t)a->d[i] * m + c; a->d[i] = (uint32_t)(p % DB); c = p / DB; }
    if (c) { bn_reserve(a, a->n + 1); a->d[a->n++] = (uint32_t)c; }
    if (!m) a->n = 0;
}
static void bn_add_small(BN *a, uint32_t m) {
    int64_t i = 0; uint64_t c = m;
    while (c) { if (i == a->n) { bn_reserve(a, a->n + 1); a->d[a->n++] = 0; } uint64_t s = (uint64_t)a->d[i] + c; a->d[i] = (uint32_t)(s % DB); c = s / DB; i++; }
}
static uint32_t bn_div_small(BN *a, uint32_t m) {
    uint64_t r = 0; for (int64_t i = a->n - 1; i >= 0; i--) { uint64_t cur = r * DB + a->d[i]; a->d[i] = (uint32_t)(cur / m); r = cur % m; }
    bn_norm(a); return (uint32_t)r;
}
static const uint32_t P10[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
static void bn_mul_pow10(BN *a, int64_t k) {                          /* a *= 10^k, k >= 0 */
    if (!a->n || k <= 0) return;
    int64_t q = k / 9; int r = (int)(k % 9);
    if (r) bn_mul_small(a, P10[r]);
    if (q) { bn_reserve(a, a->n + q); memmove(a->d + q, a->d, sizeof(uint32_t) * (size_t)a->n); memset(a->d, 0, sizeof(uint32_t) * (size_t)q); a->n += q; }
}
static void bn_pow10(BN *a, int64_t k) { bn_set_u64(a, 1); bn_mul_pow10(a, k); }
static int64_t bn_ndigits(const BN *a) {
    if (!a->n) return 1;
    uint32_t t = a->d[a->n - 1]; int k = 1; while (k < 10 && t >= P10[k]) k++;
    return (a->n - 1) * 9 + k;
}
static void bn_divmod(const BN *a, const BN *b, BN *q, BN *r) {     /* q, r: fresh or reusable, not a or b */
    if (!b->n) a_error("DivisionByZero", "[<class 'decimal.DivisionByZero'>]");
    if (bn_cmp(a, b) < 0) { if (r) bn_copy(r, a); if (q) q->n = 0; return; }
    if (b->n == 1) { BN t; bn_init(&t); bn_copy(&t, a); uint32_t rm = bn_div_small(&t, b->d[0]); if (r) bn_set_u64(r, rm); if (q) { bn_free(q); *q = t; } else bn_free(&t); return; }
    int64_t n = b->n, m = a->n - b->n; uint32_t f = (uint32_t)(DB / ((uint64_t)b->d[n - 1] + 1));
    BN u, v, qq; bn_init(&u); bn_init(&v); bn_init(&qq);
    bn_copy(&u, a); bn_mul_small(&u, f); bn_reserve(&u, a->n + 1); while (u.n < a->n + 1) u.d[u.n++] = 0;
    bn_copy(&v, b); bn_mul_small(&v, f);
    bn_reserve(&qq, m + 1); qq.n = m + 1;
    for (int64_t j = m; j >= 0; j--) {
        uint64_t num = (uint64_t)u.d[j + n] * DB + u.d[j + n - 1];
        uint64_t qh = num / v.d[n - 1], rh = num % v.d[n - 1];
        while (qh >= DB || qh * v.d[n - 2] > rh * DB + u.d[j + n - 2]) { qh--; rh += v.d[n - 1]; if (rh >= DB) break; }
        int64_t br = 0; uint64_t c = 0;
        for (int64_t i = 0; i < n; i++) {
            uint64_t p = qh * v.d[i] + c; c = p / DB; int64_t t = (int64_t)u.d[i + j] - (int64_t)(p % DB) - br;
            if (t < 0) { t += DB; br = 1; } else br = 0; u.d[i + j] = (uint32_t)t;
        }
        int64_t t = (int64_t)u.d[j + n] - (int64_t)c - br;
        if (t < 0) {
            u.d[j + n] = (uint32_t)(t + DB); qh--; uint64_t cc = 0;
            for (int64_t i = 0; i < n; i++) { uint64_t s = (uint64_t)u.d[i + j] + v.d[i] + cc; u.d[i + j] = (uint32_t)(s % DB); cc = s / DB; }
            u.d[j + n] = (uint32_t)(((uint64_t)u.d[j + n] + cc) % DB);
        } else u.d[j + n] = (uint32_t)t;
        qq.d[j] = (uint32_t)qh;
    }
    bn_norm(&qq);
    if (r) { u.n = n; bn_norm(&u); bn_div_small(&u, f); bn_free(r); *r = u; } else bn_free(&u);
    if (q) { bn_free(q); *q = qq; } else bn_free(&qq);
    bn_free(&v);
}
/* q = a / 10^k; returns how the dropped part compares with half a unit: 0 nothing, 1 below, 2 half, 3 above */
static int bn_digit(const BN *a, int64_t pos) { int64_t l = pos / 9; if (l >= a->n) return 0; return (int)((a->d[l] / P10[pos % 9]) % 10); }
static int bn_any_below(const BN *a, int64_t pos) {          /* a digit below position pos is not zero */
    int64_t l = pos / 9; for (int64_t i = 0; i < l && i < a->n; i++) if (a->d[i]) return 1;
    return l < a->n && (a->d[l] % P10[pos % 9]) != 0;
}
static int bn_split10(const BN *a, int64_t k, BN *q) {
    if (k <= 0) { bn_copy(q, a); return 0; }
    if (k > bn_ndigits(a) + 1) { int nz = a->n != 0; q->n = 0; return nz ? 1 : 0; }
    int d = bn_digit(a, k - 1), rest = bn_any_below(a, k - 1);
    int cls = d > 5 || (d == 5 && rest) ? 3 : d == 5 ? 2 : (d || rest) ? 1 : 0;
    int64_t l = k / 9; BN t; bn_init(&t);
    if (l < a->n) { bn_reserve(&t, a->n - l); memcpy(t.d, a->d + l, sizeof(uint32_t) * (size_t)(a->n - l)); t.n = a->n - l; }
    if (k % 9) bn_div_small(&t, P10[k % 9]);
    bn_norm(&t); bn_free(q); *q = t;
    return cls;
}
static void bn_from_digits(BN *a, const char *s, int64_t len) {
    a->n = 0; bn_reserve(a, len / 9 + 2);
    for (int64_t end = len; end > 0; end -= 9) {
        int64_t st = end - 9 < 0 ? 0 : end - 9; uint32_t v = 0;
        for (int64_t i = st; i < end; i++) v = v * 10 + (uint32_t)(s[i] - '0');
        a->d[a->n++] = v;
    }
    bn_norm(a);
}
static char *bn_text(const BN *a, int64_t *len) {               /* decimal digits, "0" for zero; malloc'd */
    if (!a->n) { char *s = (char *)malloc(2); s[0] = '0'; s[1] = 0; *len = 1; return s; }
    char *s = (char *)malloc((size_t)a->n * 9 + 2); int k = snprintf(s, 12, "%u", a->d[a->n - 1]);
    for (int64_t i = a->n - 2; i >= 0; i--) k += snprintf(s + k, 10, "%09u", a->d[i]);
    *len = k; return s;
}
static int bn_low_digit(const BN *a) { return a->n ? (int)(a->d[0] % 10) : 0; }

/* ------------------------------------------------------------------------------------------- decimal values */
typedef struct { int sign; int64_t exp; BN c; } DV;
static void dv_init(DV *x) { x->sign = 0; x->exp = 0; bn_init(&x->c); }
static void dv_free(DV *x) { bn_free(&x->c); }
static void dv_copy(DV *r, const DV *x) { r->sign = x->sign; r->exp = x->exp; bn_copy(&r->c, &x->c); }
static int dv_zero(const DV *x) { return x->c.n == 0; }
static int64_t dv_adj(const DV *x) { return x->exp + bn_ndigits(&x->c) - 1; }
static void dv_set_int(DV *x, int64_t v) { x->sign = v < 0; x->exp = 0; bn_set_u64(&x->c, v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v); }

static AV dv_av(const DV *x) {
    ADec *o = (ADec *)a_alloc(sizeof(ADec) + sizeof(uint32_t) * (size_t)(x->c.n ? x->c.n : 1), O_DEC);
    o->sign = x->sign; o->exp = x->exp; o->n = x->c.n; if (x->c.n) memcpy(o->d, x->c.d, sizeof(uint32_t) * (size_t)x->c.n);
    return a_obj(o);
}
static void dv_load(DV *x, AV v) {                          /* an hp number or an integer, exactly */
    if (a_is(v, O_DEC)) { ADec *o = DEC(v); x->sign = o->sign; x->exp = o->exp; bn_reserve(&x->c, o->n); if (o->n) memcpy(x->c.d, o->d, sizeof(uint32_t) * (size_t)o->n); x->c.n = o->n; return; }
    if (v.t == T_INT || v.t == T_BOOL) { dv_set_int(x, v.i); return; }
    a_error("TypeError", "unsupported operand type for Decimal: '%s'", a_tname(v));
}
static void dv_from_double(DV *x, double d) {             /* the exact binary value */
    if (!isfinite(d)) a_error("ValueError", "hp numbers are finite (got %g)", d);
    x->sign = signbit(d) ? 1 : 0; d = fabs(d); x->exp = 0;
    if (d == 0) { x->c.n = 0; return; }
    int e; double m = frexp(d, &e); uint64_t mi = (uint64_t)ldexp(m, 53); e -= 53;
    while (!(mi & 1)) { mi >>= 1; e++; }
    bn_set_u64(&x->c, mi);
    if (e >= 0) { for (int i = 0; i < e; i++) bn_mul_small(&x->c, 2); }
    else { for (int i = 0; i < -e; i++) bn_mul_small(&x->c, 5); x->exp = e; }
}
static void dv_parse(DV *x, const char *s) {
    const char *p = s; while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    x->sign = 0; if (*p == '+' || *p == '-') { x->sign = *p == '-'; p++; }
    size_t L = strlen(p); char *dig = (char *)malloc(L + 1); int64_t nd = 0, frac = 0; int seen = 0, dot = 0;
    for (; *p; p++) {
        if (*p >= '0' && *p <= '9') { dig[nd++] = *p; seen = 1; if (dot) frac++; }
        else if (*p == '.' && !dot) dot = 1;
        else break;
    }
    int64_t e = 0;
    if (seen && (*p == 'e' || *p == 'E')) {
        p++; int neg = 0; if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
        if (!(*p >= '0' && *p <= '9')) seen = 0;
        while (*p >= '0' && *p <= '9') { e = e * 10 + (*p - '0'); if (e > 4000000000000000LL) break; p++; }
        if (neg) e = -e;
    }
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (!seen || *p) { free(dig); a_error("InvalidOperation", "[<class 'decimal.ConversionSyntax'>]"); }
    int64_t st = 0; while (st < nd - 1 && dig[st] == '0') st++;
    bn_from_digits(&x->c, dig + st, nd - st); free(dig);
    x->exp = e - frac;
}
/* round to prec significant digits, half even (prec <= 0: exact) */
static void dv_round(DV *x, int64_t prec) {
    if (prec <= 0 || dv_zero(x)) return;
    int64_t nd = bn_ndigits(&x->c); if (nd <= prec) return;
    int64_t k = nd - prec; BN q; bn_init(&q); int cls = bn_split10(&x->c, k, &q);
    if (cls == 3 || (cls == 2 && (bn_low_digit(&q) & 1))) bn_add_small(&q, 1);
    if (bn_ndigits(&q) > prec) { bn_div_small(&q, 10); k++; }
    bn_free(&x->c); x->c = q; x->exp += k;
}
/* the value with exponent e (more digits, or rounded half even to fewer) */
static void dv_rescale(DV *x, int64_t e) {
    if (x->exp >= e) { if (x->exp - e > 100000000) a_error("OverflowError", "hp number too long"); bn_mul_pow10(&x->c, x->exp - e); x->exp = e; return; }
    BN q; bn_init(&q); int cls = bn_split10(&x->c, e - x->exp, &q);
    if (cls == 3 || (cls == 2 && (bn_low_digit(&q) & 1))) bn_add_small(&q, 1);
    bn_free(&x->c); x->c = q; x->exp = e;
}
static int dv_cmp_abs(const DV *a, const DV *b) {
    if (dv_zero(a) || dv_zero(b)) return dv_zero(a) ? (dv_zero(b) ? 0 : -1) : 1;
    int64_t aa = dv_adj(a), ab = dv_adj(b); if (aa != ab) return aa < ab ? -1 : 1;
    DV x, y; dv_init(&x); dv_init(&y); dv_copy(&x, a); dv_copy(&y, b);
    int64_t e = x.exp < y.exp ? x.exp : y.exp; bn_mul_pow10(&x.c, x.exp - e); bn_mul_pow10(&y.c, y.exp - e);
    int c = bn_cmp(&x.c, &y.c); dv_free(&x); dv_free(&y); return c;
}
static int dv_cmp(const DV *a, const DV *b) {
    int za = dv_zero(a), zb = dv_zero(b);
    if (za && zb) return 0;
    int sa = za ? 0 : (a->sign ? -1 : 1), sb = zb ? 0 : (b->sign ? -1 : 1);
    if (sa != sb) return sa < sb ? -1 : 1;
    int c = dv_cmp_abs(a, b); return sa < 0 ? -c : c;
}
static void dv_add(DV *r, const DV *a0, const DV *b0, int negb, int64_t prec) {
    DV a, b; dv_init(&a); dv_init(&b); dv_copy(&a, a0); dv_copy(&b, b0); if (negb) b.sign ^= 1;
    if (dv_zero(&a) || dv_zero(&b)) {                       /* Python: the zero only contributes its exponent */
        int64_t e = a.exp < b.exp ? a.exp : b.exp;
        DV *nz = dv_zero(&a) ? &b : &a;
        if (dv_zero(&a) && dv_zero(&b)) { r->sign = a.sign && b.sign; r->exp = e; r->c.n = 0; dv_free(&a); dv_free(&b); return; }
        int64_t ne = nz->exp, nd = bn_ndigits(&nz->c);
        if (prec > 0 && ne > e) { int64_t extra = prec - nd; if (extra < 0) extra = 0; int64_t te = ne - extra; e = te > e ? te : e; }
        dv_copy(r, nz); if (r->exp > e) dv_rescale(r, e); dv_round(r, prec); dv_free(&a); dv_free(&b); return;
    }
    if (prec > 0) {                                          /* Python's _normalize: a far smaller operand is only a sticky digit */
        DV *tmp = a.exp >= b.exp ? &a : &b, *oth = a.exp >= b.exp ? &b : &a;
        int64_t tl = bn_ndigits(&tmp->c), ol = bn_ndigits(&oth->c);
        int64_t lim = tl - prec - 2; int64_t ex = tmp->exp + (lim < -1 ? lim : -1);
        if (ol + oth->exp - 1 < ex) { bn_set_u64(&oth->c, 1); oth->exp = ex; }
    }
    int64_t e = a.exp < b.exp ? a.exp : b.exp;
    bn_mul_pow10(&a.c, a.exp - e); bn_mul_pow10(&b.c, b.exp - e); a.exp = b.exp = e;
    r->exp = e;
    if (a.sign == b.sign) { bn_add(&r->c, &a.c, &b.c); r->sign = a.sign; }
    else {
        int c = bn_cmp(&a.c, &b.c);
        if (c == 0) { r->c.n = 0; r->sign = 0; }
        else if (c > 0) { bn_sub(&r->c, &a.c, &b.c); r->sign = a.sign; }
        else { bn_sub(&r->c, &b.c, &a.c); r->sign = b.sign; }
    }
    dv_free(&a); dv_free(&b); dv_round(r, prec);
}
static void dv_mul(DV *r, const DV *a, const DV *b, int64_t prec) {
    BN t; bn_init(&t); bn_mul(&t, &a->c, &b->c); bn_free(&r->c); r->c = t;
    r->sign = a->sign ^ b->sign; r->exp = a->exp + b->exp; dv_round(r, prec);
}
static void dv_div(DV *r, const DV *a, const DV *b, int64_t prec) {    /* Python's _divide */
    if (dv_zero(b)) { if (dv_zero(a)) a_error("InvalidOperation", "[<class 'decimal.DivisionUndefined'>]"); a_error("DivisionByZero", "[<class 'decimal.DivisionByZero'>]"); }
    int sign = a->sign ^ b->sign;
    if (dv_zero(a)) { r->sign = sign; r->exp = a->exp - b->exp; r->c.n = 0; return; }
    if (prec <= 0) prec = 28;
    if (b->c.n == 1 && b->c.d[0] == 1) {                        /* by ±10^e: exact, then rounded */
        BN c; bn_init(&c); bn_copy(&c, &a->c); bn_free(&r->c); r->c = c; r->sign = sign; r->exp = a->exp - b->exp; dv_round(r, prec); return;
    }
    int64_t shift = bn_ndigits(&b->c) - bn_ndigits(&a->c) + prec + 1, e = a->exp - b->exp - shift;
    BN op1, op2, q, rm; bn_init(&op1); bn_init(&op2); bn_init(&q); bn_init(&rm);
    bn_copy(&op1, &a->c); bn_copy(&op2, &b->c);
    if (shift >= 0) bn_mul_pow10(&op1, shift); else bn_mul_pow10(&op2, -shift);
    bn_divmod(&op1, &op2, &q, &rm);
    if (rm.n) { if (bn_low_digit(&q) % 5 == 0) bn_add_small(&q, 1); }
    else { int64_t ideal = a->exp - b->exp; while (e < ideal && q.n && bn_low_digit(&q) == 0) { bn_div_small(&q, 10); e++; } }
    bn_free(&r->c); r->c = q; r->sign = sign; r->exp = e;
    bn_free(&op1); bn_free(&op2); bn_free(&rm);
    dv_round(r, prec);
}
static void dv_sqrt(DV *r, const DV *x, int64_t prec) {                 /* Python's sqrt */
    if (dv_zero(x)) { r->sign = x->sign; r->exp = x->exp >= 0 ? x->exp / 2 : -((-x->exp + 1) / 2); r->c.n = 0; return; }
    if (x->sign) a_error("InvalidOperation", "[<class 'decimal.InvalidOperation'>]");
    if (prec <= 0) prec = 28;
    int64_t p = prec + 1, e = x->exp >= 0 ? x->exp / 2 : -((-x->exp + 1) / 2), l;   /* e = op.exp >> 1 */
    BN c; bn_init(&c); bn_copy(&c, &x->c); int64_t len = bn_ndigits(&x->c);
    if (x->exp & 1) { bn_mul_small(&c, 10); l = (len >> 1) + 1; } else l = (len + 1) >> 1;
    int64_t shift = p - l; int exact = 1;
    if (shift >= 0) bn_mul_pow10(&c, 2 * shift);
    else { BN q, rm, d; bn_init(&q); bn_init(&rm); bn_init(&d); bn_pow10(&d, -2 * shift); bn_divmod(&c, &d, &q, &rm); exact = rm.n == 0; bn_free(&c); c = q; bn_free(&rm); bn_free(&d); }
    e -= shift;
    BN n, q; bn_init(&n); bn_init(&q); bn_pow10(&n, p);
    for (;;) {
        bn_divmod(&c, &n, &q, NULL);
        if (bn_cmp(&n, &q) <= 0) break;
        bn_add(&n, &n, &q); bn_div_small(&n, 2);
    }
    if (exact) { BN sq; bn_init(&sq); bn_mul(&sq, &n, &n); exact = bn_cmp(&sq, &c) == 0; bn_free(&sq); }
    if (exact) {
        if (shift >= 0) { BN d, qq; bn_init(&d); bn_init(&qq); bn_pow10(&d, shift); bn_divmod(&n, &d, &qq, NULL); bn_free(&n); n = qq; bn_free(&d); }
        else bn_mul_pow10(&n, -shift);
        e += shift;
    } else if (bn_low_digit(&n) % 5 == 0) bn_add_small(&n, 1);
    bn_free(&r->c); r->c = n; r->sign = 0; r->exp = e;
    bn_free(&c); bn_free(&q);
    dv_round(r, prec);
}
/* x ** k for an integer k: exact when it fits in prec digits, else correctly rounded with guard digits */
static void dv_powi_(DV *r, const DV *x, int64_t k, int64_t prec, int ideal_exp);
static void dv_powi(DV *r, const DV *x, int64_t k, int64_t prec) { dv_powi_(r, x, k, prec, 1); }
static void dv_powi_(DV *r, const DV *x, int64_t k, int64_t prec, int ideal_exp) {
    if (k == 0 && dv_zero(x)) a_error("InvalidOperation", "[<class 'decimal.InvalidOperation'>]");
    if (k == 0) { r->sign = 0; r->exp = 0; bn_set_u64(&r->c, 1); return; }
    if (k < 0) { DV p, one; dv_init(&p); dv_init(&one); int64_t wp = prec > 0 ? prec + 10 : 60; dv_powi_(&p, x, -k, wp + 10, 0); dv_set_int(&one, 1); dv_div(r, &one, &p, prec > 0 ? prec : 28); dv_free(&p); dv_free(&one); return; }
    int64_t nd = bn_ndigits(&x->c);
    int exactly = prec <= 0 || (double)nd * (double)k <= 4.0 * (double)prec + 60;
    int64_t wp = exactly ? 0 : prec + 20 + (int64_t)(log10((double)k) + 1) * 2;
    DV base, acc; dv_init(&base); dv_init(&acc); dv_copy(&base, x); dv_set_int(&acc, 1);
    int64_t z = 0; while (base.c.n && bn_low_digit(&base.c) == 0) { bn_div_small(&base.c, 10); base.exp++; z++; }   /* 10^z apart */
    uint64_t m = (uint64_t)k;
    while (m) { if (m & 1) dv_mul(&acc, &acc, &base, wp); m >>= 1; if (m) dv_mul(&base, &base, &base, wp); }
    if (ideal_exp && exactly && z && acc.c.n) {                            /* back to the exponent Python keeps: x.exp · k */
        int64_t ideal = x->exp * k, room = (prec > 0 ? prec : INT64_MAX / 4) - bn_ndigits(&acc.c);
        int64_t to = acc.exp - room > ideal ? acc.exp - room : ideal; if (to < acc.exp) dv_rescale(&acc, to);
    }
    dv_free(&base); dv_free(r); *r = acc; dv_round(r, prec);
}
static void dv_one(DV *x) { x->sign = 0; x->exp = 0; bn_set_u64(&x->c, 1); }
static double dv_to_double_approx(const DV *x, int64_t *e10) {          /* x ≈ m · 10^e10 with m a double */
    int64_t len; char *s = bn_text(&x->c, &len); int k = len > 17 ? 17 : (int)len; char t[40]; memcpy(t, s, (size_t)k); t[k] = 0; free(s);
    *e10 = x->exp + (len - k); double m = strtod(t, NULL); return x->sign ? -m : m;
}
/* e^x to prec digits: x / 2^m by Taylor's series, then m squarings, with guard digits */
static void dv_exp(DV *r, const DV *x, int64_t prec) {
    if (dv_zero(x)) { dv_one(r); return; }
    int64_t e10; double m0 = dv_to_double_approx(x, &e10); double ax = fabs(m0) * pow(10.0, (double)(e10 > 400 ? 400 : e10 < -400 ? -400 : e10));
    if (ax > 1e10) a_error("OverflowError", "hp.exp: the argument is too large");
    int m = ax > 1e-3 ? (int)ceil(log2(ax * 1000.0)) : 0; m += (int)(sqrt((double)prec) / 2);
    int64_t wp = prec + 15 + (int64_t)(m * 0.302) + 5;
    DV y, two, term, sum, kk; dv_init(&y); dv_init(&two); dv_init(&term); dv_init(&sum); dv_init(&kk);
    dv_copy(&y, x);
    for (int i = 0; i < m; i++) { dv_set_int(&two, 2); dv_div(&y, &y, &two, wp); }
    dv_one(&sum); dv_one(&term);
    for (int64_t k = 1; k < 1000000; k++) {
        dv_mul(&term, &term, &y, wp); dv_set_int(&kk, k); dv_div(&term, &term, &kk, wp);
        if (dv_zero(&term)) break;
        dv_add(&sum, &sum, &term, 0, wp);
        if (dv_adj(&term) < dv_adj(&sum) - wp - 2) break;
    }
    for (int i = 0; i < m; i++) dv_mul(&sum, &sum, &sum, wp);
    dv_free(&y); dv_free(&two); dv_free(&term); dv_free(&kk);
    dv_free(r); *r = sum; dv_round(r, prec);
}
/* ln x by Halley's iteration y += 2(x - e^y)/(x + e^y), the working precision tripling each step */
static void dv_ln(DV *r, const DV *x, int64_t prec) {
    if (dv_zero(x) || x->sign) { if (dv_zero(x)) a_error("InvalidOperation", "[<class 'decimal.InvalidOperation'>]"); a_error("InvalidOperation", "[<class 'decimal.InvalidOperation'>]"); }
    DV one; dv_init(&one); dv_one(&one);
    if (dv_cmp(x, &one) == 0) { dv_free(&one); r->sign = 0; r->exp = 0; r->c.n = 0; return; }
    DV t; dv_init(&t); dv_add(&t, x, &one, 1, 0);                          /* x - 1, exactly */
    int64_t nearloss = dv_adj(&t) < 0 ? -dv_adj(&t) : 0;           /* digits lost to x - e^y when x is near 1 */
    int64_t e10; double m0 = dv_to_double_approx(x, &e10); double y0 = log(m0) + (double)e10 * log(10.0);
    if (dv_adj(&t) < -8) { int64_t te; double tm = dv_to_double_approx(&t, &te); y0 = tm * pow(10.0, (double)te); }
    int64_t wp = prec + 12, steps[64]; int ns = 0; int64_t p = wp; while (p > 14 && ns < 60) { steps[ns++] = p; p = p / 3 + 2; }
    DV y, ey, num, den, two; dv_init(&y); dv_init(&ey); dv_init(&num); dv_init(&den); dv_init(&two);
    dv_from_double(&y, y0); dv_round(&y, 17);
    for (int i = ns - 1; i >= -1; i--) {
        int64_t pp = (i >= 0 ? steps[i] : wp) + nearloss + 5;
        dv_exp(&ey, &y, pp);
        dv_add(&num, x, &ey, 1, pp); dv_add(&den, x, &ey, 0, pp);
        if (dv_zero(&num)) continue;
        dv_div(&num, &num, &den, pp); dv_set_int(&two, 2); dv_mul(&num, &num, &two, pp);
        dv_add(&y, &y, &num, 0, pp);
    }
    dv_free(&ey); dv_free(&num); dv_free(&den); dv_free(&two); dv_free(&one); dv_free(&t);
    dv_free(r); *r = y; dv_round(r, prec);
}
static void dv_log10(DV *r, const DV *x, int64_t prec) {
    if (dv_zero(x) || x->sign) a_error("InvalidOperation", "[<class 'decimal.InvalidOperation'>]");
    BN q; bn_init(&q); bn_copy(&q, &x->c); int64_t z = 0; while (q.n && bn_low_digit(&q) == 0) { bn_div_small(&q, 10); z++; }
    if (q.n == 1 && q.d[0] == 1) { bn_free(&q); dv_set_int(r, x->exp + z); return; }   /* a power of ten: exact */
    bn_free(&q);
    DV a, ten, l10; dv_init(&a); dv_init(&ten); dv_init(&l10);
    int64_t wp = prec + 10; dv_ln(&a, x, wp); dv_set_int(&ten, 10); dv_ln(&l10, &ten, wp); dv_div(r, &a, &l10, prec);
    dv_free(&a); dv_free(&ten); dv_free(&l10);
}
/* the text Python's str(Decimal) gives (scientific when the exponent is positive or the number is tiny) */
static char *dv_text(const DV *x) {
    int64_t len; char *co = bn_text(&x->c, &len);
    int64_t left = x->exp + len, dot = (x->exp <= 0 && left > -6) ? left : 1;
    DSB b = {0}; if (x->sign) ds_puts(&b, "-");
    if (dot <= 0) { ds_puts(&b, "0."); for (int64_t i = 0; i < -dot; i++) ds_puts(&b, "0"); ds_put(&b, co, len); }
    else if (dot >= len) { ds_put(&b, co, len); for (int64_t i = 0; i < dot - len; i++) ds_puts(&b, "0"); }
    else { ds_put(&b, co, dot); ds_puts(&b, "."); ds_put(&b, co + dot, len - dot); }
    if (left != dot) { char t[32]; snprintf(t, sizeof t, "E%+lld", (long long)(left - dot)); ds_puts(&b, t); }
    free(co); ds_put(&b, "", 0); char *s = b.p ? b.p : strdup(""); return s;
}
static double dv_double(const DV *x) { char *s = dv_text(x); double d = strtod(s, NULL); free(s); return d; }

/* ------------------------------------------------------------------------------------------- hooks */
static void dv_operand(DV *x, AV v, int op, AV a, AV b) {
    if (a_is(v, O_DEC) || v.t == T_INT || v.t == T_BOOL) { dv_load(x, v); return; }
    static const char *sym[] = {"", "+", "-", "*", "/", "//", "%", "**"};
    a_error("TypeError", "unsupported operand type(s) for %s: '%s' and '%s'", op >= 1 && op <= 7 ? sym[op] : "?",
            a_is(a, O_DEC) ? "decimal.Decimal" : a_tname(a), a_is(b, O_DEC) ? "decimal.Decimal" : a_tname(b));
}
int a_dec_cmp(AV a, AV b, int *ok) {
    DV x, y; dv_init(&x); dv_init(&y); *ok = 1;
    if (a.t == T_FLOAT) { if (isnan(a.f)) { *ok = 0; return 0; } if (isinf(a.f)) return a.f < 0 ? -1 : 1; dv_from_double(&x, a.f); }
    else if (a_is(a, O_DEC) || a.t == T_INT || a.t == T_BOOL) dv_load(&x, a); else { *ok = 0; return 0; }
    if (b.t == T_FLOAT) { if (isnan(b.f)) { *ok = 0; dv_free(&x); return 0; } if (isinf(b.f)) { dv_free(&x); return b.f < 0 ? 1 : -1; } dv_from_double(&y, b.f); }
    else if (a_is(b, O_DEC) || b.t == T_INT || b.t == T_BOOL) dv_load(&y, b); else { *ok = 0; dv_free(&x); return 0; }
    int c = dv_cmp(&x, &y); dv_free(&x); dv_free(&y); return c;
}
AV a_dec_binop(int op, AV a, AV b) {
    DV x, y, r; dv_init(&x); dv_init(&y); dv_init(&r);
    if (op >= OP_ADD && op <= OP_POW) { dv_operand(&x, a, op, a, b); dv_operand(&y, b, op, a, b); }
    switch (op) {
    case OP_ADD: dv_add(&r, &x, &y, 0, dec_prec); break;
    case OP_SUB: dv_add(&r, &x, &y, 1, dec_prec); break;
    case OP_MUL: dv_mul(&r, &x, &y, dec_prec); break;
    case OP_DIV: dv_div(&r, &x, &y, dec_prec); break;
    case OP_FLOORDIV: case OP_MOD: {
        if (dv_zero(&y)) { if (dv_zero(&x)) a_error("InvalidOperation", "[<class 'decimal.DivisionUndefined'>]"); a_error("DivisionByZero", "[<class 'decimal.DivisionByZero'>]"); }
        int64_t e = x.exp < y.exp ? x.exp : y.exp; DV ax, ay; dv_init(&ax); dv_init(&ay); dv_copy(&ax, &x); dv_copy(&ay, &y);
        bn_mul_pow10(&ax.c, ax.exp - e); bn_mul_pow10(&ay.c, ay.exp - e);
        BN q, rm; bn_init(&q); bn_init(&rm); bn_divmod(&ax.c, &ay.c, &q, &rm);
        if (op == OP_FLOORDIV) { bn_free(&r.c); r.c = q; r.exp = 0; r.sign = x.sign ^ y.sign; bn_free(&rm); }
        else { bn_free(&r.c); r.c = rm; r.exp = e; r.sign = x.sign; bn_free(&q); }
        dv_free(&ax); dv_free(&ay); dv_round(&r, dec_prec); break;
    }
    case OP_POW: {
        DV yi; dv_init(&yi); dv_copy(&yi, &y); if (yi.exp < 0) dv_rescale(&yi, 0);
        DV back; dv_init(&back); dv_copy(&back, &yi);
        if (dv_cmp(&back, &y) == 0 && bn_ndigits(&yi.c) + yi.exp <= 15) {
            bn_mul_pow10(&yi.c, yi.exp); int64_t k = 0; for (int64_t i = yi.c.n - 1; i >= 0; i--) k = k * DB + yi.c.d[i]; if (yi.sign) k = -k;
            if (k < 0 && dv_zero(&x)) a_error("ZeroDivisionError", "0 cannot be raised to a negative power");
            dv_powi(&r, &x, k, dec_prec);
        } else {
            if (x.sign && !dv_zero(&x)) a_error("ValueError", "a negative number to a non-integer power");
            if (dv_zero(&x)) { r.c.n = 0; r.exp = 0; r.sign = 0; }
            else { DV l; dv_init(&l); int64_t wp = dec_prec + 20; dv_ln(&l, &x, wp); dv_mul(&l, &l, &y, wp); dv_exp(&r, &l, dec_prec); dv_free(&l); }
        }
        dv_free(&yi); dv_free(&back); break;
    }
    case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
        int ok; int c = a_dec_cmp(a, b, &ok);
        if (!ok) { if (op == OP_EQ) return a_bool(0); if (op == OP_NE) return a_bool(1); a_error("TypeError", "'<' not supported between '%s' and '%s'", a_tname(a), a_tname(b)); }
        int v = op == OP_EQ ? c == 0 : op == OP_NE ? c != 0 : op == OP_LT ? c < 0 : op == OP_LE ? c <= 0 : op == OP_GT ? c > 0 : c >= 0;
        dv_free(&x); dv_free(&y); dv_free(&r); return a_bool(v);
    }
    default: a_error("TypeError", "unsupported operand type(s) for an hp number");
    }
    AV out = dv_av(&r); dv_free(&x); dv_free(&y); dv_free(&r); return out;
}
AV a_dec_unop(int op, AV a) {
    DV x; dv_init(&x); dv_load(&x, a);
    if (op == OP_NEG) { if (!dv_zero(&x) || 1) x.sign ^= 1; dv_round(&x, dec_prec); }
    else if (op == OP_POS) { if (dv_zero(&x)) x.sign = 0; dv_round(&x, dec_prec); }
    AV r = dv_av(&x); dv_free(&x); return r;
}
int a_dec_truthy(AV v) { return DEC(v)->n != 0; }
char *a_dec_text(AV v) { DV x; dv_init(&x); dv_load(&x, v); char *s = dv_text(&x); dv_free(&x); return s; }
double a_dec_float(AV v) { DV x; dv_init(&x); dv_load(&x, v); double d = dv_double(&x); dv_free(&x); return d; }
uint64_t a_dec_hash(AV v) {
    DV x; dv_init(&x); dv_load(&x, v); DV t; dv_init(&t); dv_copy(&t, &x); uint64_t h;
    if (t.exp < 0) dv_rescale(&t, 0);
    if (dv_cmp(&t, &x) == 0 && bn_ndigits(&t.c) + t.exp <= 18) { bn_mul_pow10(&t.c, t.exp); int64_t k = 0; for (int64_t i = t.c.n - 1; i >= 0; i--) k = k * DB + t.c.d[i]; h = a_hash(a_int(t.sign ? -k : k)); }
    else h = a_hash(a_float(dv_double(&x)));
    dv_free(&x); dv_free(&t); return h;
}
static AV dv_trunc_int(DV *x) {                             /* toward zero, as int(Decimal) */
    if (x->exp < 0) { BN q; bn_init(&q); bn_split10(&x->c, -x->exp, &q); bn_free(&x->c); x->c = q; x->exp = 0; }
    if (bn_ndigits(&x->c) + x->exp > 18) { char *s = dv_text(x); a_error("OverflowError", "int too large for a native Int: %s", s); }
    bn_mul_pow10(&x->c, x->exp); int64_t k = 0; for (int64_t i = x->c.n - 1; i >= 0; i--) k = k * DB + x->c.d[i];
    return a_int(x->sign ? -k : k);
}
/* format(x, spec) as Decimal.__format__: [[fill]align][sign][0][width][.precision][type] with e E f F g G % */
char *a_dec_format(AV v, const char *spec) {
    DV x; dv_init(&x); dv_load(&x, v);
    const char *p = spec; char fill = ' ', align = 0, sgn = '-', type = 0; int zero = 0, alt = 0; int64_t width = 0, prec = -1;
    if (p[0] && (p[1] == '<' || p[1] == '>' || p[1] == '^' || p[1] == '=')) { fill = p[0]; align = p[1]; p += 2; }
    else if (*p == '<' || *p == '>' || *p == '^' || *p == '=') align = *p++;
    if (*p == '+' || *p == '-' || *p == ' ') sgn = *p++;
    if (*p == '#') { alt = 1; p++; }
    if (*p == '0') { zero = 1; p++; }
    while (*p >= '0' && *p <= '9') width = width * 10 + (*p++ - '0');
    if (*p == ',' || *p == '_') p++;
    if (*p == '.') { p++; prec = 0; while (*p >= '0' && *p <= '9') prec = prec * 10 + (*p++ - '0'); }
    if (*p) type = *p;
    if (!type) type = 'g';
    if (type == '%') x.exp += 2;
    if (prec >= 0) {
        if (type == 'e' || type == 'E') { if (!dv_zero(&x)) { int64_t ad = dv_adj(&x); dv_rescale(&x, ad + 1 - (prec + 1)); if (dv_adj(&x) != ad) dv_rescale(&x, dv_adj(&x) + 1 - (prec + 1)); } }
        else if (type == 'f' || type == 'F' || type == '%') dv_rescale(&x, -prec);
        else if ((type == 'g' || type == 'G') && bn_ndigits(&x.c) > prec && !dv_zero(&x)) { int64_t pr = prec ? prec : 1, ad = dv_adj(&x); dv_rescale(&x, ad + 1 - pr); if (dv_adj(&x) != ad) dv_rescale(&x, dv_adj(&x) + 1 - pr); }
    }
    if (dv_zero(&x) && x.exp > 0 && (type == 'f' || type == 'F' || type == '%')) dv_rescale(&x, 0);
    int64_t len; char *co = bn_text(&x.c, &len); int64_t left = x.exp + len, dot;
    if (type == 'e' || type == 'E') dot = (dv_zero(&x) && prec >= 0) ? 1 - prec : 1;
    else if (type == 'f' || type == 'F' || type == '%') dot = left;
    else dot = (x.exp <= 0 && left > -6) ? left : 1;
    DSB ip = {0}, fp = {0};
    if (dot < 0) { ds_puts(&ip, "0"); for (int64_t i = 0; i < -dot; i++) ds_puts(&fp, "0"); ds_put(&fp, co, len); }
    else if (dot > len) { ds_put(&ip, co, len); for (int64_t i = 0; i < dot - len; i++) ds_puts(&ip, "0"); }
    else { if (dot) ds_put(&ip, co, dot); else ds_puts(&ip, "0"); ds_put(&fp, co + dot, len - dot); }
    int64_t ex = left - dot;
    DSB body = {0}; ds_put(&body, ip.p ? ip.p : "", ip.n);
    if (fp.n || alt) { ds_puts(&body, "."); ds_put(&body, fp.p ? fp.p : "", fp.n); }
    if (ex != 0 || type == 'e' || type == 'E') { char t[32]; char ec = (type == 'E' || type == 'G') ? 'E' : 'e'; snprintf(t, sizeof t, "%c%+lld", ec, (long long)ex); ds_puts(&body, t); }
    if (type == '%') ds_puts(&body, "%");
    const char *sign = x.sign ? "-" : sgn == '+' ? "+" : sgn == ' ' ? " " : "";
    ds_put(&body, "", 0);
    int64_t blen = body.n + (int64_t)strlen(sign), padn = width > blen ? width - blen : 0;
    DSB out = {0};
    if (zero && !align) { align = '='; fill = '0'; }
    if (!align) align = '>';
    if (align == '=') { ds_puts(&out, sign); for (int64_t i = 0; i < padn; i++) ds_put(&out, &fill, 1); ds_put(&out, body.p, body.n); }
    else {
        int64_t lp = align == '<' ? 0 : align == '^' ? padn / 2 : padn, rp = padn - lp;
        for (int64_t i = 0; i < lp; i++) ds_put(&out, &fill, 1);
        ds_puts(&out, sign); ds_put(&out, body.p, body.n);
        for (int64_t i = 0; i < rp; i++) ds_put(&out, &fill, 1);
    }
    ds_put(&out, "", 0);
    free(co); free(ip.p); free(fp.p); free(body.p); dv_free(&x);
    return out.p ? out.p : strdup("");
}
/* builtins on an hp number: abs, float, int, round, bool */
int a_dec_wants(const char *name, int n, AV *a) {
    return n >= 1 && a_is(a[0], O_DEC) && (!strcmp(name, "abs") || !strcmp(name, "float") || !strcmp(name, "int") || !strcmp(name, "round"));
}
AV a_dec_builtin(const char *name, int n, AV *a) {
    DV x; dv_init(&x); dv_load(&x, a[0]); AV r;
    if (!strcmp(name, "abs")) { x.sign = 0; dv_round(&x, dec_prec); r = dv_av(&x); }
    else if (!strcmp(name, "float")) r = a_float(dv_double(&x));
    else if (!strcmp(name, "int")) r = dv_trunc_int(&x);
    else {                                                  /* round(x) -> Int, round(x, n) -> hp number */
        if (n > 1 && a[1].t != T_NONE) { dv_rescale(&x, -a[1].i); dv_round(&x, dec_prec); r = dv_av(&x); }
        else { dv_rescale(&x, 0); r = dv_trunc_int(&x); }
    }
    dv_free(&x); return r;
}
/* x.sqrt() x.exp() x.ln() x.log10() x.adjusted() x.to_integral_value() x.quantize(y) x.is_zero() ... */
static AV m_dec(int n, AV *a, ADict *kw, AV self, const char *name) {
    (void)kw; DV x, r; dv_init(&x); dv_init(&r); dv_load(&x, self); AV out;
    if (!strcmp(name, "sqrt")) { dv_sqrt(&r, &x, dec_prec); out = dv_av(&r); }
    else if (!strcmp(name, "exp")) { dv_exp(&r, &x, dec_prec); out = dv_av(&r); }
    else if (!strcmp(name, "ln")) { dv_ln(&r, &x, dec_prec); out = dv_av(&r); }
    else if (!strcmp(name, "log10")) { dv_log10(&r, &x, dec_prec); out = dv_av(&r); }
    else if (!strcmp(name, "adjusted")) out = a_int(dv_zero(&x) ? x.exp : dv_adj(&x));
    else if (!strcmp(name, "to_integral_value") || !strcmp(name, "to_integral")) { if (x.exp < 0) dv_rescale(&x, 0); out = dv_av(&x); }
    else if (!strcmp(name, "quantize")) { DV q; dv_init(&q); dv_load(&q, a[0]); dv_rescale(&x, q.exp); dv_free(&q); out = dv_av(&x); }
    else if (!strcmp(name, "is_zero")) out = a_bool(dv_zero(&x));
    else if (!strcmp(name, "is_signed")) out = a_bool(x.sign);
    else if (!strcmp(name, "copy_abs")) { x.sign = 0; out = dv_av(&x); }
    else if (!strcmp(name, "normalize")) { if (dv_zero(&x)) x.exp = 0; else { dv_round(&x, dec_prec); while (bn_low_digit(&x.c) == 0) { bn_div_small(&x.c, 10); x.exp++; } } out = dv_av(&x); }
    else { (void)n; out = A_NONE; }
    dv_free(&x); dv_free(&r); return out;
}
#define DM(nm) static AV m_dec_##nm(int n, AV *a, ADict *kw, AV s) { return m_dec(n, a, kw, s, #nm); }
DM(sqrt) DM(exp) DM(ln) DM(log10) DM(adjusted) DM(to_integral_value) DM(to_integral) DM(quantize) DM(is_zero) DM(is_signed) DM(copy_abs) DM(normalize)
AV a_dec_attr(AV v, const char *name) {
    #define DA(nm) if (!strcmp(name, #nm)) return a_builtin(#nm, m_dec_##nm, v)
    DA(sqrt); DA(exp); DA(ln); DA(log10); DA(adjusted); DA(to_integral_value); DA(to_integral); DA(quantize); DA(is_zero); DA(is_signed); DA(copy_abs); DA(normalize);
    #undef DA
    AV none; none.t = 255; none.i = 0; return none;
}

/* __dec(op, ...): the hp library's access to decimals
     "prec" [n]          the context's precision (set it, get the old one)
     "new" x             an hp number from text, an Int, a Float (its exact binary value) or an hp number
     "round" x p         x rounded to p significant digits (p <= 0: unchanged)
     "add"/"sub"/"mul"/"div" a b p      with p digits (p <= 0 for add/sub/mul: exact)
     "is" x              x is an hp number
     "exact_text" x      x as text (str)  */
BI(lb_dec) {
    (void)self; (void)kw;
    const char *op = lib_str(a[0], "__dec")->s;
    if (!strcmp(op, "prec")) { int64_t old = dec_prec; if (n > 1 && a[1].t != T_NONE) { if (a[1].i < 1) a_error("ValueError", "precision of 1 or more"); dec_prec = a[1].i; } return a_int(old); }
    if (!strcmp(op, "is")) return a_bool(a_is(a[1], O_DEC));
    DV x; dv_init(&x);
    if (!strcmp(op, "new")) {
        AV v = a[1];
        if (a_is(v, O_DEC)) return v;
        if (a_is(v, O_STR)) dv_parse(&x, ((AStr *)v.o)->s);
        else if (v.t == T_FLOAT) dv_from_double(&x, v.f);
        else if (v.t == T_INT || v.t == T_BOOL) dv_set_int(&x, v.t == T_BOOL ? (v.i != 0) : v.i);
        else a_error("TypeError", "hp works with plain numbers (got '%s')", a_tname(v));
        AV r = dv_av(&x); dv_free(&x); return r;
    }
    if (!strcmp(op, "round")) { dv_load(&x, a[1]); dv_round(&x, a[2].i); AV r = dv_av(&x); dv_free(&x); return r; }
    if (!strcmp(op, "add") || !strcmp(op, "sub") || !strcmp(op, "mul") || !strcmp(op, "div")) {
        DV y, r; dv_init(&y); dv_init(&r); dv_load(&x, a[1]); dv_load(&y, a[2]); int64_t p = a[3].i;
        if (op[0] == 'a') dv_add(&r, &x, &y, 0, p); else if (op[0] == 's') dv_add(&r, &x, &y, 1, p);
        else if (op[0] == 'm') dv_mul(&r, &x, &y, p); else dv_div(&r, &x, &y, p);
        AV out = dv_av(&r); dv_free(&x); dv_free(&y); dv_free(&r); return out;
    }
    a_error("ValueError", "__dec: unknown operation '%s'", op); return A_NONE;
}
