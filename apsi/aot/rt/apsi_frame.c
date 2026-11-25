/* The kernels behind `use dataframe` in native programs: CSV parsing into typed columns, group ids, grouped
   reductions, hash joins, stable multi-key sorts and membership tests. Included at the end of apsi_tensor.c.
   One entry point, __frame(op, ...), keeps the library's surface small. Keys are compared as apsi run's numpy
   code compares them: integers exactly, floats with every NaN equal and -0.0 == 0.0. */

#define FR_STR(v) ((AStr *)(v).o)
#define FR_LIST(v) ((AList *)(v).o)
typedef struct { char *p; size_t len, cap; } FrBuf;
static void fr_put(FrBuf *b, const char *s, size_t n) { if (b->len + n + 1 > b->cap) { b->cap = (b->len + n + 1) * 2; b->p = (char *)realloc(b->p, b->cap); } memcpy(b->p + b->len, s, n); b->len += n; }
static ATensor *fr_f64(AV v) { AV t = as_tensor(val_of(v)); if (TT(t)->dev) t = to_host(t); return TT(tcopy(TT(t), DT_F64)); }
static ATensor *fr_i64(AV v) { AV t = as_tensor(val_of(v)); if (TT(t)->dev) t = to_host(t); return TT(tcopy(TT(t), DT_I64)); }
static ATensor *fr_vec(int dt, int64_t n) { int64_t sh[1] = {n}; return t_new(dt, 1, sh); }

static uint64_t fr_dbits(double x) {                     /* canonical bits: one NaN, one zero */
    if (x != x) return 0x7FF8000000000000ULL;
    if (x == 0) x = 0.0;
    uint64_t u; memcpy(&u, &x, 8); return u;
}
static uint64_t fr_mix(uint64_t h) { h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33; return h; }

/* key matrix: m columns of n rows as canonical 64-bit words (an int column stays exact, a float column canonical) */
typedef struct { int64_t n; int m; uint64_t **col; } FrKeys;
static FrKeys fr_keys(AV list, int as_float) {
    AList *l = FR_LIST(list); FrKeys k = {0, (int)l->len, (uint64_t **)calloc((size_t)(l->len ? l->len : 1), sizeof(uint64_t *))};
    for (int j = 0; j < k.m; j++) {
        AV t = as_tensor(val_of(l->v[j])); if (TT(t)->dev) t = to_host(t);
        ATensor *T = TT(t); int64_t n = T->n; if (j == 0) k.n = n; else if (n != k.n) a_error("ShapeError", "key columns differ in length");
        k.col[j] = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(n ? n : 1));
        if (T->dt == DT_F64 || T->dt == DT_F32 || as_float) {
            ATensor *F = TT(tcopy(T, DT_F64)); const double *d = (const double *)F->p; for (int64_t i = 0; i < n; i++) k.col[j][i] = fr_dbits(d[i]);
        } else {
            ATensor *I = TT(tcopy(T, DT_I64)); const int64_t *d = (const int64_t *)I->p; for (int64_t i = 0; i < n; i++) k.col[j][i] = (uint64_t)d[i];
        }
    }
    return k;
}
static void fr_keys_free(FrKeys *k) { for (int j = 0; j < k->m; j++) free(k->col[j]); free(k->col); }
static uint64_t fr_hash(const FrKeys *k, int64_t i) { uint64_t h = 0x9E3779B97F4A7C15ULL; for (int j = 0; j < k->m; j++) h = fr_mix(h ^ (k->col[j][i] + 0x9E3779B97F4A7C15ULL + (h << 6))); return h; }
static int fr_eq(const FrKeys *a, int64_t i, const FrKeys *b, int64_t r) { for (int j = 0; j < a->m; j++) if (a->col[j][i] != b->col[j][r]) return 0; return 1; }

/* open-addressing table of representative rows */
typedef struct { int64_t cap; int64_t *slot; } FrTab;
static FrTab fr_tab(int64_t n) { FrTab t; t.cap = 16; while (t.cap < 2 * n + 16) t.cap <<= 1; t.slot = (int64_t *)malloc(sizeof(int64_t) * (size_t)t.cap); for (int64_t i = 0; i < t.cap; i++) t.slot[i] = -1; return t; }

/* group ids in order of first appearance, and the first row of every group */
static AV fr_group(AV keys) {
    FrKeys k = fr_keys(keys, 0); int64_t n = k.n;
    FrTab t = fr_tab(n);
    ATensor *G = fr_vec(DT_I64, n); int sp = a_ntmp; A_PUSH(a_obj(G));
    int64_t *gid = (int64_t *)G->p, *first = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n ? n : 1)), *rep_gid = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n ? n : 1)), ng = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t h = fr_hash(&k, i); int64_t s = (int64_t)(h & (uint64_t)(t.cap - 1));
        while (t.slot[s] >= 0 && !fr_eq(&k, i, &k, t.slot[s])) s = (s + 1) & (t.cap - 1);
        if (t.slot[s] < 0) { t.slot[s] = i; rep_gid[i] = ng; first[ng] = i; ng++; }
        gid[i] = rep_gid[t.slot[s]];
    }
    ATensor *F = fr_vec(DT_I64, ng); memcpy(F->p, first, sizeof(int64_t) * (size_t)ng); A_PUSH(a_obj(F));
    free(first); free(rep_gid); free(t.slot); fr_keys_free(&k);
    AV pair[2] = {a_obj(G), a_obj(F)}; AV r = a_tuple(2, pair); a_ntmp = sp; return r;
}

static int fr_dcmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* one value per group: sum, count, mean, var, std, min, max, median, first, last, n_unique (NaN skipped as numpy's nan*) */
static AV fr_reduce(AV gidv, int64_t k, AV valv, const char *kind) {
    ATensor *G = fr_i64(gidv); int sp = a_ntmp; A_PUSH(a_obj(G));
    ATensor *V = fr_f64(valv); A_PUSH(a_obj(V));
    const int64_t *g = (const int64_t *)G->p; const double *x = (const double *)V->p; int64_t n = G->n;
    ATensor *R = fr_vec(DT_F64, k); A_PUSH(a_obj(R)); double *r = (double *)R->p;
    if (!strcmp(kind, "sum") || !strcmp(kind, "count") || !strcmp(kind, "mean") || !strcmp(kind, "var") || !strcmp(kind, "std")) {
        double *s = (double *)calloc((size_t)(k ? k : 1), sizeof(double)), *c = (double *)calloc((size_t)(k ? k : 1), sizeof(double));
        for (int64_t i = 0; i < n; i++) if (x[i] == x[i]) { s[g[i]] += x[i]; c[g[i]] += 1; }
        if (!strcmp(kind, "sum")) memcpy(r, s, sizeof(double) * (size_t)k);
        else if (!strcmp(kind, "count")) memcpy(r, c, sizeof(double) * (size_t)k);
        else if (!strcmp(kind, "mean")) for (int64_t j = 0; j < k; j++) r[j] = s[j] / c[j];
        else {
            double *ss = (double *)calloc((size_t)(k ? k : 1), sizeof(double));
            for (int64_t i = 0; i < n; i++) if (x[i] == x[i]) { double d = x[i] - s[g[i]] / c[g[i]]; ss[g[i]] += d * d; }
            for (int64_t j = 0; j < k; j++) { double v = ss[j] / (c[j] - 1); r[j] = !strcmp(kind, "std") ? sqrt(v) : v; }
            free(ss);
        }
        free(s); free(c);
    } else if (!strcmp(kind, "min") || !strcmp(kind, "max")) {
        int mx = !strcmp(kind, "max");
        for (int64_t j = 0; j < k; j++) r[j] = mx ? -INFINITY : INFINITY;
        for (int64_t i = 0; i < n; i++) if (x[i] == x[i]) { double *p = &r[g[i]]; if (mx ? x[i] > *p : x[i] < *p) *p = x[i]; }
    } else if (!strcmp(kind, "first") || !strcmp(kind, "last")) {
        int last = !strcmp(kind, "last"); uint8_t *seen = (uint8_t *)calloc((size_t)(k ? k : 1), 1);
        for (int64_t i = 0; i < n; i++) if (last || !seen[g[i]]) { r[g[i]] = x[i]; seen[g[i]] = 1; }
        free(seen);
    } else if (!strcmp(kind, "median") || !strcmp(kind, "n_unique")) {
        int64_t *cnt = (int64_t *)calloc((size_t)(k + 1), sizeof(int64_t)), *pos = (int64_t *)malloc(sizeof(int64_t) * (size_t)(k + 1));
        for (int64_t i = 0; i < n; i++) cnt[g[i] + 1]++;
        for (int64_t j = 0; j < k; j++) cnt[j + 1] += cnt[j];
        memcpy(pos, cnt, sizeof(int64_t) * (size_t)(k + 1));
        double *buf = (double *)malloc(sizeof(double) * (size_t)(n ? n : 1));
        for (int64_t i = 0; i < n; i++) buf[pos[g[i]]++] = x[i];
        for (int64_t j = 0; j < k; j++) {
            double *b = buf + cnt[j]; int64_t m = cnt[j + 1] - cnt[j], q = 0;
            if (!strcmp(kind, "median")) {
                for (int64_t i = 0; i < m; i++) if (b[i] == b[i]) b[q++] = b[i];
                qsort(b, (size_t)q, sizeof(double), fr_dcmp);
                r[j] = q == 0 ? NAN : (q & 1) ? b[q / 2] : (b[q / 2 - 1] + b[q / 2]) / 2;
            } else {
                for (int64_t i = 0; i < m; i++) b[i] = b[i] == 0 ? 0.0 : b[i];
                qsort(b, (size_t)m, sizeof(double), fr_dcmp);       /* NaNs: numpy's unique keeps one */
                int64_t u = 0, nan = 0;
                for (int64_t i = 0; i < m; i++) { if (b[i] != b[i]) { nan = 1; continue; } if (i == 0 || b[i] != b[i - 1] || b[i - 1] != b[i - 1]) u++; }
                r[j] = (double)(u + nan);
            }
        }
        free(cnt); free(pos); free(buf);
    } else a_error("ValueError", "unknown aggregation '%s'", kind);
    a_ntmp = sp; return a_obj(R);
}

/* rows of left and right with equal keys: (left rows, right rows), -1 where a side has no row (how: inner, left,
   right, outer); left rows in order, each with its matches in right order, unmatched right rows last */
static AV fr_join(AV lkeys, AV rkeys, const char *how, int as_float) {
    FrKeys L = fr_keys(lkeys, as_float), R = fr_keys(rkeys, as_float);
    int64_t nl = L.n, nr = R.n;
    FrTab t = fr_tab(nr);
    int64_t *next = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nr ? nr : 1)), *last = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nr ? nr : 1));
    for (int64_t i = 0; i < nr; i++) {                      /* chains of right rows with the same key, in order */
        next[i] = -1;
        uint64_t h = fr_hash(&R, i); int64_t s = (int64_t)(h & (uint64_t)(t.cap - 1));
        while (t.slot[s] >= 0 && !fr_eq(&R, i, &R, t.slot[s])) s = (s + 1) & (t.cap - 1);
        if (t.slot[s] < 0) { t.slot[s] = i; last[i] = i; } else { int64_t hd = t.slot[s]; next[last[hd]] = i; last[hd] = i; }
    }
    int left = !strcmp(how, "left") || !strcmp(how, "outer") || !strcmp(how, "full");
    int right = !strcmp(how, "right") || !strcmp(how, "outer") || !strcmp(how, "full");
    int64_t cap = nl + nr + 16, m = 0;
    int64_t *li = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap), *ri = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap);
    uint8_t *used = (uint8_t *)calloc((size_t)(nr ? nr : 1), 1);
    for (int64_t i = 0; i < nl; i++) {
        uint64_t h = fr_hash(&L, i); int64_t s = (int64_t)(h & (uint64_t)(t.cap - 1));
        while (t.slot[s] >= 0 && !fr_eq(&L, i, &R, t.slot[s])) s = (s + 1) & (t.cap - 1);
        int64_t j = t.slot[s];
        if (j < 0) { if (left) { if (m == cap) { cap *= 2; li = (int64_t *)realloc(li, sizeof(int64_t) * (size_t)cap); ri = (int64_t *)realloc(ri, sizeof(int64_t) * (size_t)cap); } li[m] = i; ri[m++] = -1; } continue; }
        for (; j >= 0; j = next[j]) {
            if (m == cap) { cap *= 2; li = (int64_t *)realloc(li, sizeof(int64_t) * (size_t)cap); ri = (int64_t *)realloc(ri, sizeof(int64_t) * (size_t)cap); }
            li[m] = i; ri[m++] = j; used[j] = 1;
        }
    }
    if (right) for (int64_t j = 0; j < nr; j++) if (!used[j]) {
        if (m == cap) { cap *= 2; li = (int64_t *)realloc(li, sizeof(int64_t) * (size_t)cap); ri = (int64_t *)realloc(ri, sizeof(int64_t) * (size_t)cap); }
        li[m] = -1; ri[m++] = j;
    }
    ATensor *A = fr_vec(DT_I64, m); int sp = a_ntmp; A_PUSH(a_obj(A)); ATensor *B = fr_vec(DT_I64, m); A_PUSH(a_obj(B));
    memcpy(A->p, li, sizeof(int64_t) * (size_t)m); memcpy(B->p, ri, sizeof(int64_t) * (size_t)m);
    free(li); free(ri); free(used); free(next); free(last); free(t.slot); fr_keys_free(&L); fr_keys_free(&R);
    AV pair[2] = {a_obj(A), a_obj(B)}; AV r = a_tuple(2, pair); a_ntmp = sp; return r;
}

/* stable sort by several keys (the first is the primary key); keys are float or int tensors */
static const double **fr_sk; static int fr_nsk;
static int fr_sort_cmp(const void *a, const void *b) {
    int64_t i = *(const int64_t *)a, j = *(const int64_t *)b;
    for (int q = 0; q < fr_nsk; q++) {
        double x = fr_sk[q][i], y = fr_sk[q][j]; int xn = x != x, yn = y != y;   /* NaN last, as numpy */
        if (xn != yn) return xn - yn;
        if (!xn && x != y) return x < y ? -1 : 1;
    }
    return i < j ? -1 : i > j;
}
static AV fr_argsort(AV keys) {
    AList *l = FR_LIST(keys); int m = (int)l->len; int sp = a_ntmp;
    const double **ks = (const double **)malloc(sizeof(double *) * (size_t)(m ? m : 1)); int64_t n = 0;
    for (int q = 0; q < m; q++) { ATensor *T = fr_f64(l->v[q]); A_PUSH(a_obj(T)); ks[q] = (const double *)T->p; n = T->n; }
    ATensor *O = fr_vec(DT_I64, n); A_PUSH(a_obj(O)); int64_t *o = (int64_t *)O->p;
    for (int64_t i = 0; i < n; i++) o[i] = i;
    const double **sk = fr_sk; int nsk = fr_nsk; fr_sk = ks; fr_nsk = m;
    qsort(o, (size_t)n, sizeof(int64_t), fr_sort_cmp);
    fr_sk = sk; fr_nsk = nsk; free(ks);
    a_ntmp = sp; return a_obj(O);
}

/* ---- CSV: Python's csv module rules (quotes, doubled quotes, newlines in quotes, \r\n) */
static AV fr_csv(AV textv, const char *sepstr) {
    AStr *S = FR_STR(textv); const char *p = S->s, *end = S->s + S->len; char sep = sepstr[0] ? sepstr[0] : ',';
    if (end - p >= 3 && (unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3;
    AV rows = a_list(0, NULL); int sp = a_ntmp; A_PUSH(rows);
    FrBuf cell = {0};
    while (p < end) {
        AV row = a_list(0, NULL); A_PUSH(row);
        int done = 0;
        while (!done) {
            cell.len = 0; int quoted = 0;
            if (p < end && *p == '"') {
                quoted = 1; p++;
                while (p < end) {
                    if (*p == '"') { if (p + 1 < end && p[1] == '"') { fr_put(&cell, "\"", 1); p += 2; continue; } p++; break; }
                    fr_put(&cell, p, 1); p++;
                }
            }
            while (p < end && *p != sep && *p != '\n' && *p != '\r') { fr_put(&cell, p, 1); p++; }
            (void)quoted;
            a_list_append((AList *)row.o, a_strn(cell.p ? cell.p : "", (int64_t)cell.len));
            if (p >= end) done = 1;
            else if (*p == sep) p++;
            else { if (*p == '\r') p++; if (p < end && *p == '\n') p++; done = 1; }
        }
        if (!(FR_LIST(row)->len == 1 && FR_STR(FR_LIST(row)->v[0])->len == 0)) a_list_append((AList *)rows.o, row);   /* blank lines */
        a_ntmp--;
    }
    free(cell.p);
    a_ntmp = sp; return rows;
}

/* a typed column from CSV text, as apsi run converts it: int64 when every value is an integer, float64 when every
   value parses as a number (empty = NaN), bool for true / false, else ("str", codes, values) with "" as null */
static int fr_is_int(const char *s, int64_t n) { int64_t i = 0; while (i < n && s[i] == '-') i++; if (i == n) return 0; for (; i < n; i++) if (s[i] < '0' || s[i] > '9') return 0; return 1; }
static int fr_parse_f(const char *s, int64_t n, double *out) {
    if (n == 0) { *out = NAN; return 1; }
    char buf[128]; char *tmp = n < 127 ? buf : (char *)malloc((size_t)n + 1); int64_t q = 0;
    for (int64_t i = 0; i < n; i++) if (s[i] != '_') tmp[q++] = s[i];
    tmp[q] = 0; char *e; double v = strtod(tmp, &e);
    while (*e == ' ' || *e == '\t' || *e == '\n' || *e == '\r') e++;
    int ok = e != tmp && *e == 0; const char *t = tmp; while (*t == ' ' || *t == '\t') t++;
    if (t != tmp && *t == 0) ok = 0;
    if (tmp != buf) free(tmp);
    *out = v; return ok;
}
static int fr_ieq(const char *s, const char *w, int64_t n) { for (int64_t i = 0; i < n; i++) { char c = s[i]; if (c >= 'A' && c <= 'Z') c = (char)(c + 32); if (c != w[i]) return 0; } return 1; }
static AV fr_convert(AV list) {
    AList *l = FR_LIST(list); int64_t n = l->len, look = n < 1000 ? n : 1000; int sp = a_ntmp;
    int allint = n > 0;
    for (int64_t i = 0; i < look && allint; i++) { AStr *s = FR_STR(l->v[i]); if (!s->len || !fr_is_int(s->s, s->len)) allint = 0; }
    if (allint) {
        ATensor *T = fr_vec(DT_I64, n); A_PUSH(a_obj(T)); int64_t *d = (int64_t *)T->p; int ok = 1;
        for (int64_t i = 0; i < n && ok; i++) { AStr *s = FR_STR(l->v[i]); char *e; errno = 0; long long v = strtoll(s->s, &e, 10); if (!s->len || e != s->s + s->len || errno) ok = 0; else d[i] = v; }
        if (ok) { a_ntmp = sp; return a_obj(T); }
    }
    {
        ATensor *T = fr_vec(DT_F64, n); A_PUSH(a_obj(T)); double *d = (double *)T->p; int ok = 1;
        for (int64_t i = 0; i < n && ok; i++) { AStr *s = FR_STR(l->v[i]); if (!fr_parse_f(s->s, s->len, &d[i])) ok = 0; }
        if (ok) { a_ntmp = sp; return a_obj(T); }
    }
    int anyv = 0, bools = 1;
    for (int64_t i = 0; i < look; i++) {
        AStr *s = FR_STR(l->v[i]); if (!s->len) continue; anyv = 1;
        if (!(s->len == 4 && fr_ieq(s->s, "true", 4)) && !(s->len == 5 && fr_ieq(s->s, "false", 5))) { bools = 0; break; }
    }
    if (anyv && bools) {
        ATensor *T = fr_vec(DT_BOOL, n); A_PUSH(a_obj(T)); uint8_t *d = (uint8_t *)T->p;
        for (int64_t i = 0; i < n; i++) { AStr *s = FR_STR(l->v[i]); d[i] = s->len == 4 && fr_ieq(s->s, "true", 4); }
        a_ntmp = sp; return a_obj(T);
    }
    /* strings: a dictionary in order of first appearance; "" is null (code -1) */
    ATensor *C = fr_vec(DT_I64, n); A_PUSH(a_obj(C)); int64_t *c = (int64_t *)C->p;
    AV vals = a_list(0, NULL); A_PUSH(vals); AV index = a_dict(); A_PUSH(index);
    for (int64_t i = 0; i < n; i++) {
        AStr *s = FR_STR(l->v[i]); if (!s->len) { c[i] = -1; continue; }
        AV got; if (a_dict_get((ADict *)index.o, l->v[i], &got)) { c[i] = got.i; continue; }
        c[i] = FR_LIST(vals)->len; a_dict_set((ADict *)index.o, l->v[i], a_int(c[i])); a_list_append((AList *)vals.o, l->v[i]);
    }
    AV parts[3] = {a_str("str"), a_obj(C), vals}; AV r = a_tuple(3, parts); a_ntmp = sp; return r;
}


/* ---- CSV straight to typed columns: cells are indexed once, numbers parsed from the bytes, text interned through
   a hash table (one string object per distinct value) */
typedef struct { int64_t a, b; uint8_t q; } FrCell;      /* bytes [a, b); q: 1 quoted, 2 quoted with "" inside */
static const char *fr_cell_text(const char *base, const FrCell *c, FrBuf *tmp, int64_t *len) {
    if (c->q != 2) { *len = c->b - c->a; return base + c->a; }
    tmp->len = 0;
    for (int64_t i = c->a; i < c->b; i++) { if (base[i] == '"' && i + 1 < c->b && base[i + 1] == '"') i++; fr_put(tmp, base + i, 1); }
    *len = (int64_t)tmp->len; return tmp->p ? tmp->p : "";
}
static AV fr_read(AV textv, const char *sepstr, AV want) {
    AStr *S = FR_STR(textv); const char *base = S->s, *end = S->s + S->len; char sep = sepstr[0] ? sepstr[0] : ',';
    const char *p = base;
    if (end - p >= 3 && (unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3;
    int64_t ccap = 1024, nc = 0, rcap = 256, nr = 0;
    FrCell *cells = (FrCell *)malloc(sizeof(FrCell) * (size_t)ccap); int64_t *rows = (int64_t *)malloc(sizeof(int64_t) * (size_t)(rcap + 1));
    while (p < end) {
        int64_t first = nc; int done = 0;
        while (!done) {
            if (nc == ccap) { ccap *= 2; cells = (FrCell *)realloc(cells, sizeof(FrCell) * (size_t)ccap); }
            FrCell c = {p - base, p - base, 0};
            if (p < end && *p == '"') {
                c.q = 1; p++; c.a = p - base;
                while (p < end) { if (*p == '"') { if (p + 1 < end && p[1] == '"') { c.q = 2; p += 2; continue; } break; } p++; }
                c.b = p - base; if (p < end) p++;
                while (p < end && *p != sep && *p != '\n' && *p != '\r') p++;       /* text after a closing quote: dropped */
            } else {
                while (p < end && *p != sep && *p != '\n' && *p != '\r') p++;
                c.b = p - base;
            }
            cells[nc++] = c;
            if (p >= end) done = 1;
            else if (*p == sep) p++;
            else { if (*p == '\r') p++; if (p < end && *p == '\n') p++; done = 1; }
        }
        if (nc - first == 1 && cells[first].q == 0 && cells[first].a == cells[first].b) { nc = first; continue; }   /* blank line */
        if (nr == rcap) { rcap *= 2; rows = (int64_t *)realloc(rows, sizeof(int64_t) * (size_t)(rcap + 1)); }
        rows[nr++] = first;
    }
    rows[nr] = nc;
    int sp = a_ntmp; AV header = a_list(0, NULL); A_PUSH(header); AV cols = a_dict(); A_PUSH(cols);
    FrBuf tmp = {0};
    if (nr == 0) { free(cells); free(rows); AV pair[2] = {header, cols}; AV r = a_tuple(2, pair); a_ntmp = sp; return r; }
    int64_t width = rows[1] - rows[0];
    for (int64_t j = 0; j < width; j++) { int64_t L; const char *t = fr_cell_text(base, &cells[rows[0] + j], &tmp, &L); a_list_append((AList *)header.o, a_strn(t, L)); }
    for (int64_t r = 1; r < nr; r++) if (rows[r + 1] - rows[r] < width) width = rows[r + 1] - rows[r];   /* ragged rows: the shortest, as zip */
    int64_t n = nr - 1, look = n < 1000 ? n : 1000;
    for (int64_t j = 0; j < width; j++) {
        AV name = FR_LIST(header)->v[j];
        if (want.t != T_NONE && !a_contains(want, name)) continue;
        #define CELL(r) (&cells[rows[(r) + 1] + j])
        int allint = n > 0;
        for (int64_t r = 0; r < look && allint; r++) { int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L); if (!L || !fr_is_int(t, L)) allint = 0; }
        AV col = A_NONE;
        if (allint) {
            ATensor *T = fr_vec(DT_I64, n); A_PUSH(a_obj(T)); int64_t *d = (int64_t *)T->p; int ok = 1;
            for (int64_t r = 0; r < n && ok; r++) {
                int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L); char nb[32];
                if (!L || L > 30 || !fr_is_int(t, L)) { ok = 0; break; }
                memcpy(nb, t, (size_t)L); nb[L] = 0; errno = 0; char *e; long long v = strtoll(nb, &e, 10); if (errno || *e) ok = 0; else d[r] = v;
            }
            if (ok) col = a_obj(T); else a_ntmp--;
        }
        if (col.t == T_NONE) {
            ATensor *T = fr_vec(DT_F64, n); A_PUSH(a_obj(T)); double *d = (double *)T->p; int ok = 1;
            for (int64_t r = 0; r < n && ok; r++) { int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L); if (!fr_parse_f(t, L, &d[r])) ok = 0; }
            if (ok) col = a_obj(T); else a_ntmp--;
        }
        if (col.t == T_NONE) {
            int anyv = 0, bools = 1;
            for (int64_t r = 0; r < look; r++) {
                int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L); if (!L) continue; anyv = 1;
                if (!(L == 4 && fr_ieq(t, "true", 4)) && !(L == 5 && fr_ieq(t, "false", 5))) { bools = 0; break; }
            }
            if (anyv && bools) {
                ATensor *T = fr_vec(DT_BOOL, n); A_PUSH(a_obj(T)); uint8_t *d = (uint8_t *)T->p;
                for (int64_t r = 0; r < n; r++) { int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L); d[r] = L == 4 && fr_ieq(t, "true", 4); }
                col = a_obj(T);
            }
        }
        if (col.t == T_NONE) {                                /* text: codes into the distinct values, "" = null */
            ATensor *C = fr_vec(DT_I64, n); A_PUSH(a_obj(C)); int64_t *c = (int64_t *)C->p;
            AV vals = a_list(0, NULL); A_PUSH(vals);
            int64_t cap = 64; while (cap < 2 * n + 16) cap <<= 1;
            int64_t *slot = (int64_t *)malloc(sizeof(int64_t) * (size_t)cap); uint64_t *hs = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)cap);
            for (int64_t i = 0; i < cap; i++) slot[i] = -1;
            for (int64_t r = 0; r < n; r++) {
                int64_t L; const char *t = fr_cell_text(base, CELL(r), &tmp, &L);
                if (!L) { c[r] = -1; continue; }
                uint64_t h = 1469598103934665603ULL; for (int64_t i = 0; i < L; i++) { h ^= (unsigned char)t[i]; h *= 1099511628211ULL; }
                int64_t s = (int64_t)(fr_mix(h) & (uint64_t)(cap - 1));
                while (slot[s] >= 0) {
                    AStr *v = FR_STR(FR_LIST(vals)->v[slot[s]]);
                    if (hs[s] == h && v->len == L && !memcmp(v->s, t, (size_t)L)) break;
                    s = (s + 1) & (cap - 1);
                }
                if (slot[s] < 0) { slot[s] = FR_LIST(vals)->len; hs[s] = h; a_list_append((AList *)vals.o, a_strn(t, L)); }
                c[r] = slot[s];
            }
            free(slot); free(hs);
            AV parts[3] = {a_str("str"), a_obj(C), vals}; col = a_tuple(3, parts);
        }
        #undef CELL
        a_dict_set((ADict *)cols.o, name, col);
        a_ntmp = sp + 2;
    }
    free(cells); free(rows); free(tmp.p);
    AV pair[2] = {header, cols}; AV r = a_tuple(2, pair); a_ntmp = sp; return r;
}

/* membership of every value of a in the values b (numbers; NaN never matches, as numpy.isin) */
static AV fr_isin(AV av, AV bv) {
    ATensor *A = fr_f64(av); int sp = a_ntmp; A_PUSH(a_obj(A)); ATensor *B = fr_f64(bv); A_PUSH(a_obj(B));
    const double *a = (const double *)A->p, *b = (const double *)B->p; int64_t na = A->n, nb = B->n;
    double *s = (double *)malloc(sizeof(double) * (size_t)(nb ? nb : 1)); int64_t q = 0;
    for (int64_t i = 0; i < nb; i++) if (b[i] == b[i]) s[q++] = b[i];
    qsort(s, (size_t)q, sizeof(double), fr_dcmp);
    ATensor *R = fr_vec(DT_BOOL, na); A_PUSH(a_obj(R)); uint8_t *r = (uint8_t *)R->p;
    for (int64_t i = 0; i < na; i++) { r[i] = 0; if (a[i] != a[i]) continue; int64_t lo = 0, hi = q; while (lo < hi) { int64_t mid = (lo + hi) / 2; if (s[mid] < a[i]) lo = mid + 1; else hi = mid; } r[i] = lo < q && s[lo] == a[i]; }
    free(s); a_ntmp = sp; return a_obj(R);
}

/* rows where a mask is true */
static AV fr_nonzero(AV mv) {
    AV t = as_tensor(val_of(mv)); if (TT(t)->dev) t = to_host(t);
    ATensor *M = TT(tcopy(TT(t), DT_F64)); int sp = a_ntmp; A_PUSH(a_obj(M)); const double *m = (const double *)M->p;
    int64_t k = 0; for (int64_t i = 0; i < M->n; i++) if (m[i] != 0) k++;
    ATensor *R = fr_vec(DT_I64, k); A_PUSH(a_obj(R)); int64_t *r = (int64_t *)R->p; k = 0;
    for (int64_t i = 0; i < M->n; i++) if (m[i] != 0) r[k++] = i;
    a_ntmp = sp; return a_obj(R);
}

/* rows by index, any dtype; index -1 gives the fill value (a float column, or the given int / bool) */
static AV fr_take(AV cv, AV iv, int fill_nan) {
    AV t = as_tensor(val_of(cv)); if (TT(t)->dev) t = to_host(t);
    ATensor *C = TT(t); ATensor *I = fr_i64(iv); int sp = a_ntmp; A_PUSH(a_obj(I)); const int64_t *ix = (const int64_t *)I->p;
    int miss = 0; for (int64_t i = 0; i < I->n; i++) { if (ix[i] < 0) miss = 1; else if (ix[i] >= C->n) a_error("IndexError", "row %lld is out of range for %lld rows", (long long)ix[i], (long long)C->n); }
    int dt = miss && fill_nan ? DT_F64 : C->dt;
    ATensor *R = fr_vec(dt, I->n); A_PUSH(a_obj(R));
    if (dt == C->dt && C->dt == DT_I64) { const int64_t *c = (const int64_t *)C->p; int64_t *r = (int64_t *)R->p; for (int64_t i = 0; i < I->n; i++) r[i] = ix[i] < 0 ? -1 : c[ix[i]]; }
    else if (dt == C->dt && C->dt == DT_BOOL) { const uint8_t *c = (const uint8_t *)C->p; uint8_t *r = (uint8_t *)R->p; for (int64_t i = 0; i < I->n; i++) r[i] = ix[i] < 0 ? 0 : c[ix[i]]; }
    else if (dt == C->dt && C->dt == DT_F32) { const float *c = (const float *)C->p; float *r = (float *)R->p; for (int64_t i = 0; i < I->n; i++) r[i] = ix[i] < 0 ? NAN : c[ix[i]]; }
    else { ATensor *F = TT(tcopy(C, DT_F64)); A_PUSH(a_obj(F)); const double *c = (const double *)F->p; double *r = (double *)R->p; for (int64_t i = 0; i < I->n; i++) r[i] = ix[i] < 0 ? NAN : c[ix[i]]; }
    a_ntmp = sp; return a_obj(R);
}

BI(lb_frame) {
    (void)self; (void)kw; if (n < 1) a_error("TypeError", "__frame(op, ...)");
    const char *op = lib_str(a[0], "dataframe")->s;
    if (!strcmp(op, "group")) return fr_group(a[1]);
    if (!strcmp(op, "reduce")) return fr_reduce(a[1], a[2].i, a[3], lib_str(a[4], "dataframe")->s);
    if (!strcmp(op, "join")) return fr_join(a[1], a[2], lib_str(a[3], "dataframe")->s, n > 4 && a_truthy(a[4]));
    if (!strcmp(op, "argsort")) return fr_argsort(a[1]);
    if (!strcmp(op, "csv")) return fr_csv(a[1], lib_str(a[2], "dataframe")->s);
    if (!strcmp(op, "convert")) return fr_convert(a[1]);
    if (!strcmp(op, "read")) return fr_read(a[1], lib_str(a[2], "dataframe")->s, n > 3 ? a[3] : A_NONE);
    if (!strcmp(op, "isin")) return fr_isin(a[1], a[2]);
    if (!strcmp(op, "nonzero")) return fr_nonzero(a[1]);
    if (!strcmp(op, "take")) return fr_take(a[1], a[2], n > 3 && a_truthy(a[3]));
    a_error("ValueError", "unknown dataframe kernel '%s'", op);
}
