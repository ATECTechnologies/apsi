/* Apsi native runtime: files and folders, model and array files (safetensors, .npy), time, and an HTTP/1.1 server
 * (one event loop over non-blocking sockets: many idle keep-alive clients cost little; the handlers, which are Apsi
 * code, run one at a time on that loop). Included by apsi_tensor.c; the library modules written in Apsi wrap these. */
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <dirent.h>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <direct.h>
#include <io.h>
typedef SOCKET sock_t;
#define SOCK_BAD INVALID_SOCKET
#define sock_close closesocket
#define sock_poll WSAPoll
#define SOCK_WOULDBLOCK(e) ((e) == WSAEWOULDBLOCK)
static int sock_err(void) { return WSAGetLastError(); }
#else
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
typedef int sock_t;
#define SOCK_BAD (-1)
#define sock_close close
#define sock_poll poll
#define SOCK_WOULDBLOCK(e) ((e) == EAGAIN || (e) == EWOULDBLOCK)
static int sock_err(void) { return errno; }
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

static double now_s(void) {                 /* wall-clock seconds since 1970 */
#ifdef _WIN32
    FILETIME ft; GetSystemTimePreciseAsFileTime(&ft);
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;     /* 100 ns since 1601 */
    return (double)(t - 116444736000000000ULL) * 1e-7;
#else
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}
BI(lb_time) { (void)self; (void)kw; (void)n; (void)a; return a_float(now_s()); }
BI(lb_sleep) {
    (void)self; (void)kw; (void)n; double s = a_c_f64(a[0]); if (s <= 0) return A_NONE;
#ifdef _WIN32
    Sleep((DWORD)(s * 1000));
#else
    struct timespec ts = {(time_t)s, (long)((s - (double)(time_t)s) * 1e9)}; while (nanosleep(&ts, &ts) && errno == EINTR) {}
#endif
    return A_NONE;
}

/* ------------------------------------------------------------------ files and folders: __fs(op, path, arg) */
static int cmp_names(const void *x, const void *y) { return strcmp(STR(*(const AV *)x)->s, STR(*(const AV *)y)->s); }
static int mkdirs(const char *path) {
    char p[4096]; snprintf(p, sizeof p, "%s", path); size_t n = strlen(p);
    for (size_t i = 1; i <= n; i++) if (p[i] == '/' || p[i] == '\\' || p[i] == 0) {
        char c = p[i]; p[i] = 0;
        if (p[0] && !(i == 2 && p[1] == ':')) {
#ifdef _WIN32
            if (_mkdir(p) && errno != EEXIST) return -1;
#else
            if (mkdir(p, 0777) && errno != EEXIST) return -1;
#endif
        }
        p[i] = c;
    }
    return 0;
}
BI(lb_fs) {
    (void)self; (void)kw; const char *op = lib_str(a[0], "fs")->s, *path = lib_str(a[1], "fs")->s; struct stat st;
    if (!strcmp(op, "list")) {
        DIR *d = opendir(path); if (!d) a_error("FileNotFoundError", "cannot list the folder '%s': %s", path, strerror(errno));
        AV lst = a_list(0, NULL); int sp = a_ntmp; A_PUSH(lst); struct dirent *e;
        while ((e = readdir(d))) { if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue; a_list_append((AList *)lst.o, a_str(e->d_name)); }
        closedir(d);
        AList *l = (AList *)lst.o; qsort(l->v, (size_t)l->len, sizeof(AV), cmp_names);
        a_ntmp = sp; return lst;
    }
    if (!strcmp(op, "isdir")) return a_bool(!stat(path, &st) && S_ISDIR(st.st_mode));
    if (!strcmp(op, "isfile")) return a_bool(!stat(path, &st) && S_ISREG(st.st_mode));
    if (!strcmp(op, "size")) { if (stat(path, &st)) a_error("FileNotFoundError", "no file '%s'", path); return a_int((int64_t)st.st_size); }
    if (!strcmp(op, "mtime")) { if (stat(path, &st)) a_error("FileNotFoundError", "no file '%s'", path); return a_float((double)st.st_mtime); }
    if (!strcmp(op, "mkdir")) { if (mkdirs(path)) a_error("OSError", "cannot make the folder '%s': %s", path, strerror(errno)); return A_NONE; }
    if (!strcmp(op, "remove")) { if (remove(path)) a_error("OSError", "cannot remove '%s': %s", path, strerror(errno)); return A_NONE; }
    if (!strcmp(op, "rmdir")) {
#ifdef _WIN32
        if (_rmdir(path))
#else
        if (rmdir(path))
#endif
            a_error("OSError", "cannot remove the folder '%s': %s", path, strerror(errno));
        return A_NONE;
    }
    if (!strcmp(op, "rename")) { if (rename(path, lib_str(a[2], "fs")->s)) a_error("OSError", "cannot rename '%s': %s", path, strerror(errno)); return A_NONE; }
    if (!strcmp(op, "append")) {
        FILE *f = fopen(path, "ab"); if (!f) a_error("IOError", "cannot open '%s': %s", path, strerror(errno));
        AStr *s = lib_str(a[2], "append_text"); fwrite(s->s, 1, (size_t)s->len, f); fclose(f); return A_NONE;
    }
    if (!strcmp(op, "cwd")) { char b[4096]; if (!getcwd(b, sizeof b)) return a_str("."); return a_str(b); }
    if (!strcmp(op, "readb")) {                                      /* the whole file as bytes */
        FILE *f = fopen(path, "rb"); if (!f) a_error("FileNotFoundError", "no such file: '%s'", path);
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); if (sz < 0) sz = 0;
        char *buf = (char *)malloc((size_t)sz + 1); size_t got = fread(buf, 1, (size_t)sz, f); fclose(f);
        AV r = a_bytes(buf, (int64_t)got); free(buf); return r;
    }
    if (!strcmp(op, "writeb")) {
        AStr *d = lib_str(a[2], "fs"); FILE *f = fopen(path, "wb"); if (!f) a_error("OSError", "cannot write '%s': %s", path, strerror(errno));
        fwrite(d->s, 1, (size_t)d->len, f); fclose(f); return A_NONE;
    }
    a_error("ValueError", "fs: unknown operation '%s'", op);
}

/* ------------------------------------------------------------------ safetensors and .npy */
static unsigned char *read_all(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) a_error("FileNotFoundError", "cannot open '%s': %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END)) { fclose(f); a_error("IOError", "cannot read '%s'", path); }
    long long n = (long long)ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc((size_t)n + 1); if (!b) { fclose(f); a_error("MemoryError", "the file '%s' (%lld bytes) does not fit in memory", path, n); }
    if (n && fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); a_error("IOError", "cannot read '%s'", path); }
    fclose(f); *len = (size_t)n; return b;
}
static float half_to_f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) { if (!m) f = s; else { int k = -1; do { k++; m <<= 1; } while (!(m & 1024)); m &= 1023; f = s | (uint32_t)(112 - k) << 23 | m << 13; } }
    else if (e == 31) f = s | 0x7f800000 | m << 13;
    else f = s | (e + 112) << 23 | m << 13;
    float r; memcpy(&r, &f, 4); return r;
}
static void put_tensor_bytes(ATensor *t, const unsigned char *src, const char *code, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        double v;
        if (!strcmp(code, "F64")) { double x; memcpy(&x, src + 8 * i, 8); v = x; }
        else if (!strcmp(code, "F32")) { float x; memcpy(&x, src + 4 * i, 4); v = x; }
        else if (!strcmp(code, "F16")) { uint16_t x; memcpy(&x, src + 2 * i, 2); v = half_to_f(x); }
        else if (!strcmp(code, "BF16")) { uint16_t x; memcpy(&x, src + 2 * i, 2); uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, 4); v = f; }
        else if (!strcmp(code, "I64")) { int64_t x; memcpy(&x, src + 8 * i, 8); ((int64_t *)t->p)[i] = x; continue; }
        else if (!strcmp(code, "I32")) { int32_t x; memcpy(&x, src + 4 * i, 4); v = x; }
        else if (!strcmp(code, "I16")) { int16_t x; memcpy(&x, src + 2 * i, 2); v = x; }
        else if (!strcmp(code, "I8")) v = (int8_t)src[i];
        else if (!strcmp(code, "U8") || !strcmp(code, "BOOL")) v = src[i];
        else a_error("ValueError", "safetensors: the type %s is not supported natively", code);
        t_set(t, i, v);
    }
}
static int st_dtype(const char *code, int *size) {
    static const struct { const char *c; int dt, sz; } m[] = {{"F64", DT_F64, 8}, {"F32", DT_F32, 4}, {"F16", DT_F32, 2}, {"BF16", DT_F32, 2},
        {"I64", DT_I64, 8}, {"I32", DT_I64, 4}, {"I16", DT_I64, 2}, {"I8", DT_I64, 1}, {"U8", DT_I64, 1}, {"BOOL", DT_BOOL, 1}};
    for (size_t i = 0; i < sizeof m / sizeof m[0]; i++) if (!strcmp(m[i].c, code)) { *size = m[i].sz; return m[i].dt; }
    return -1;
}
static AV json_parse_text(const char *s, size_t n) { AV f = a_global_builtin("json_parse"); AV t = a_strn(s, (int64_t)n); return a_call(f, 1, &t, NULL); }
/* the tensor table as Python's json.dumps(sort_keys=True, separators=(",", ":")) writes it (ASCII, \\uXXXX escapes):
 * Apsi checkpoints hash it together with the data */
typedef struct { const char *name; const char *code; int nd; const int64_t *sh; uint64_t a, b; } LayRow;
static int lay_cmp(const void *x, const void *y) { return strcmp(((const LayRow *)x)->name, ((const LayRow *)y)->name); }
static void py_json_str(HCtx *h, const char *s) {
    h_update(h, "\"", 1);
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        uint32_t cp = *p; int k = 1;
        if (cp >= 0xF0) { cp = ((cp & 7) << 18) | ((p[1] & 63) << 12) | ((p[2] & 63) << 6) | (p[3] & 63); k = 4; }
        else if (cp >= 0xE0) { cp = ((cp & 15) << 12) | ((p[1] & 63) << 6) | (p[2] & 63); k = 3; }
        else if (cp >= 0xC0) { cp = ((cp & 31) << 6) | (p[1] & 63); k = 2; }
        char b[16]; int bl;
        if (cp == '"') bl = snprintf(b, sizeof b, "\\\""); else if (cp == '\\') bl = snprintf(b, sizeof b, "\\\\");
        else if (cp == '\n') bl = snprintf(b, sizeof b, "\\n"); else if (cp == '\r') bl = snprintf(b, sizeof b, "\\r");
        else if (cp == '\t') bl = snprintf(b, sizeof b, "\\t"); else if (cp == 8) bl = snprintf(b, sizeof b, "\\b"); else if (cp == 12) bl = snprintf(b, sizeof b, "\\f");
        else if (cp < 32 || (cp > 126 && cp < 0x10000)) bl = snprintf(b, sizeof b, "\\u%04x", cp);
        else if (cp >= 0x10000) { uint32_t v = cp - 0x10000; bl = snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 1023)); }
        else { b[0] = (char)cp; bl = 1; }
        h_update(h, b, (size_t)bl); p += k;
    }
    h_update(h, "\"", 1);
}
static void layout_hash(HCtx *h, LayRow *rows, int64_t n) {
    qsort(rows, (size_t)n, sizeof(LayRow), lay_cmp);
    h_update(h, "{", 1);
    for (int64_t i = 0; i < n; i++) {
        char b[64]; int bl;
        if (i) h_update(h, ",", 1);
        py_json_str(h, rows[i].name); h_update(h, ":[", 2); py_json_str(h, rows[i].code); h_update(h, ",[", 2);
        for (int d = 0; d < rows[i].nd; d++) { bl = snprintf(b, sizeof b, d ? ",%lld" : "%lld", (long long)rows[i].sh[d]); h_update(h, b, (size_t)bl); }
        bl = snprintf(b, sizeof b, "],[%llu,%llu]]", (unsigned long long)rows[i].a, (unsigned long long)rows[i].b); h_update(h, b, (size_t)bl);
    }
    h_update(h, "}", 1);
}
/* __st_load(path, names_or_none) -> (tensors dict, metadata dict) */
BI(lb_st_load) {
    (void)self; (void)kw; const char *path = lib_str(a[0], "load_safetensors")->s; size_t len; unsigned char *b = read_all(path, &len);
    if (len < 8) { free(b); a_error("ValueError", "'%s' is not a safetensors file", path); }
    uint64_t hn = le64(b); if (hn > len - 8 || hn > ((uint64_t)100 << 20)) { free(b); a_error("ValueError", "'%s' is not a safetensors file (bad header size)", path); }
    int sp = a_ntmp; AV head = json_parse_text((const char *)b + 8, (size_t)hn); A_PUSH(head);
    if (!a_is(head, O_DICT)) { free(b); a_error("ValueError", "'%s': the header is not a JSON object", path); }
    AV out = a_dict(); A_PUSH(out); AV meta = a_dict(); A_PUSH(meta);
    ADict *H = (ADict *)head.o; const unsigned char *data = b + 8 + hn; uint64_t dlen = len - 8 - hn;
    AV want = n > 1 ? a[1] : A_NONE;
    if (n > 2 && a_truthy(a[2])) {                       /* verify an Apsi checkpoint's SHA-256 */
        AV mv; const char *want_hex = NULL;
        if (a_dict_get(H, a_str("__metadata__"), &mv) && a_is(mv, O_DICT)) { AV sv; if (a_dict_get((ADict *)mv.o, a_str("sha256"), &sv) && a_is(sv, O_STR)) want_hex = STR(sv)->s; }
        if (want_hex) {
            LayRow *rows = (LayRow *)malloc(sizeof(LayRow) * (size_t)(H->nent + 1)); int64_t *shs = (int64_t *)malloc(sizeof(int64_t) * MAXD * (size_t)(H->nent + 1)); int64_t k = 0; uint64_t end = 0;
            for (int64_t i = 0; i < H->nent; i++) {
                if (!H->e[i].used || !a_is(H->e[i].k, O_STR) || !strcmp(STR(H->e[i].k)->s, "__metadata__")) continue;
                AV v = H->e[i].v; AList *sl = (AList *)a_getitem(v, a_str("shape")).o, *ol = (AList *)a_getitem(v, a_str("data_offsets")).o;
                int nd = (int)(sl->len < MAXD ? sl->len : MAXD); for (int d = 0; d < nd; d++) shs[k * MAXD + d] = sl->v[d].i;
                rows[k] = (LayRow){STR(H->e[i].k)->s, STR(a_getitem(v, a_str("dtype")))->s, nd, shs + k * MAXD, (uint64_t)ol->v[0].i, (uint64_t)ol->v[1].i};
                if (rows[k].b > end) end = rows[k].b; k++;
            }
            HCtx hc; h_init(&hc, H_SHA256); layout_hash(&hc, rows, k); free(rows); free(shs);
            if (end > dlen) end = dlen;
            h_update(&hc, data, (size_t)end); uint8_t d[32]; h_final(&hc, d); char got[65]; static const char *hx = "0123456789abcdef";
            for (int i = 0; i < 32; i++) { got[2 * i] = hx[d[i] >> 4]; got[2 * i + 1] = hx[d[i] & 15]; } got[64] = 0;
            if (strcmp(got, want_hex)) { free(b); a_error("ValueError", "%s is damaged or was modified: its data does not match the SHA-256 stored in the file", path); }
        }
    }
    for (int64_t i = 0; i < H->nent; i++) {
        if (!H->e[i].used) continue;
        AV k = H->e[i].k, v = H->e[i].v; if (!a_is(k, O_STR)) continue;
        if (!strcmp(STR(k)->s, "__metadata__")) { if (a_is(v, O_DICT)) a_tmp[sp + 2] = meta = v; continue; }
        if (want.t != T_NONE && !a_contains(want, k)) continue;
        AV dt = a_getitem(v, a_str("dtype")), shp = a_getitem(v, a_str("shape")), off = a_getitem(v, a_str("data_offsets"));
        int esz; int dtn = st_dtype(STR(dt)->s, &esz);
        if (dtn < 0) { free(b); a_error("ValueError", "'%s': tensor '%s' has the type %s, which native programs do not read", path, STR(k)->s, STR(dt)->s); }
        AList *sl = (AList *)shp.o; int64_t sh[MAXD]; if (sl->len > MAXD) { free(b); a_error("ValueError", "too many dimensions"); }
        int64_t count = 1; for (int64_t j = 0; j < sl->len; j++) { sh[j] = sl->v[j].i; count *= sh[j]; }
        int64_t o0 = ((AList *)off.o)->v[0].i, o1 = ((AList *)off.o)->v[1].i;
        if (o0 < 0 || o1 < o0 || (uint64_t)o1 > dlen || o1 - o0 != count * esz) { free(b); a_error("ValueError", "'%s': tensor '%s' has bad offsets", path, STR(k)->s); }
        ATensor *t = t_new(dtn, (int)sl->len, sh); A_PUSH(a_obj(t));
        if ((dtn == DT_F64 && esz == 8) || (dtn == DT_F32 && esz == 4 && !strcmp(STR(dt)->s, "F32")) || (dtn == DT_I64 && esz == 8) || dtn == DT_BOOL) memcpy(t->p, data + o0, (size_t)(count * esz));
        else put_tensor_bytes(t, data + o0, STR(dt)->s, count);
        a_dict_set((ADict *)out.o, k, a_obj(t)); a_ntmp--;
    }
    free(b);
    AV r = ltuple2(out, meta); a_ntmp = sp; return r;
}
static void json_esc(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) { unsigned char c = (unsigned char)*s; if (c == '"' || c == '\\') fprintf(f, "\\%c", c); else if (c < 32) fprintf(f, "\\u%04x", c); else fputc(c, f); }
    fputc('"', f);
}
/* __st_save(path, {name: tensor}, {key: text}) */
BI(lb_st_save) {
    (void)self; (void)kw; const char *path = lib_str(a[0], "save_safetensors")->s;
    if (!a_is(a[1], O_DICT)) a_error("TypeError", "save_safetensors(path, tensors): tensors is a dict of name -> tensor");
    ADict *D = (ADict *)a[1].o; int sp = a_ntmp;
    int64_t cnt = 0; for (int64_t i = 0; i < D->nent; i++) if (D->e[i].used) cnt++;
    ATensor **ts = (ATensor **)malloc(sizeof(ATensor *) * (size_t)(cnt + 1)); const char **nm = (const char **)malloc(sizeof(char *) * (size_t)(cnt + 1)); int64_t k = 0;
    for (int64_t i = 0; i < D->nent; i++) if (D->e[i].used) {
        if (!a_is(D->e[i].k, O_STR)) { free(ts); free(nm); a_error("TypeError", "save_safetensors: the names are text"); }
        AV t = deepval(D->e[i].v); if (!is_t(t)) t = as_tensor(t); if (TT(t)->dev) t = to_host(t); A_PUSH(t);
        ts[k] = TT(t); nm[k++] = STR(D->e[i].k)->s;
    }
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (!f) { free(ts); free(nm); a_error("IOError", "cannot write '%s': %s", path, strerror(errno)); }
    /* the header first in a memory buffer, so its length is known */
    char *hb = NULL; size_t hl = 0;
#ifdef _WIN32
    FILE *h = tmpfile();
#else
    FILE *h = open_memstream(&hb, &hl);
#endif
    fputc('{', h); int first = 1;
    static const char *codes[] = {"F64", "F32", "I64", "BOOL"};
    char digest_hex[65] = "";
    if (n > 3 && a_truthy(a[3])) {                       /* an Apsi checkpoint: the SHA-256 of the table and the data */
        LayRow *rows = (LayRow *)malloc(sizeof(LayRow) * (size_t)(k + 1)); uint64_t o2 = 0;
        for (int64_t i = 0; i < k; i++) { uint64_t by = (uint64_t)ts[i]->n * (uint64_t)dt_size[ts[i]->dt];
            rows[i] = (LayRow){nm[i], codes[ts[i]->dt], ts[i]->nd, ts[i]->sh, o2, o2 + by}; o2 += by; }
        HCtx hc; h_init(&hc, H_SHA256); layout_hash(&hc, rows, k); free(rows);
        for (int64_t i = 0; i < k; i++) h_update(&hc, ts[i]->p, (size_t)ts[i]->n * (size_t)dt_size[ts[i]->dt]);
        uint8_t d[32]; h_final(&hc, d); static const char *hx = "0123456789abcdef";
        for (int i = 0; i < 32; i++) { digest_hex[2 * i] = hx[d[i] >> 4]; digest_hex[2 * i + 1] = hx[d[i] & 15]; } digest_hex[64] = 0;
    }
    if ((n > 2 && a_is(a[2], O_DICT)) || digest_hex[0]) {
        fputs("\"__metadata__\":{", h); int f1 = 1;
        if (n > 2 && a_is(a[2], O_DICT)) { ADict *M = (ADict *)a[2].o;
            for (int64_t i = 0; i < M->nent; i++) if (M->e[i].used) { if (!f1) fputc(',', h); f1 = 0; json_esc(h, STR(a_to_str(M->e[i].k))->s); fputc(':', h); json_esc(h, STR(a_to_str(M->e[i].v))->s); } }
        if (digest_hex[0]) { if (!f1) fputc(',', h); fprintf(h, "\"sha256\":\"%s\"", digest_hex); }
        fputc('}', h); first = 0;
    }
    uint64_t off = 0;
    for (int64_t i = 0; i < k; i++) {
        if (!first) fputc(',', h); first = 0;
        uint64_t bytes = (uint64_t)ts[i]->n * (uint64_t)dt_size[ts[i]->dt];
        json_esc(h, nm[i]); fprintf(h, ":{\"dtype\":\"%s\",\"shape\":[", codes[ts[i]->dt]);
        for (int d = 0; d < ts[i]->nd; d++) fprintf(h, d ? ",%lld" : "%lld", (long long)ts[i]->sh[d]);
        fprintf(h, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off, (unsigned long long)(off + bytes)); off += bytes;
    }
    fputc('}', h);
#ifdef _WIN32
    hl = (size_t)ftell(h); hb = (char *)malloc(hl + 8); fseek(h, 0, SEEK_SET); if (fread(hb, 1, hl, h) != hl) hl = 0; fclose(h);
#else
    fclose(h);
#endif
    size_t padded = (hl + 7) & ~(size_t)7; unsigned char lenb[8]; put_le64(lenb, (uint64_t)padded);
    int ok = fwrite(lenb, 1, 8, f) == 8 && fwrite(hb, 1, hl, f) == hl;
    for (size_t i = hl; i < padded; i++) ok = ok && fputc(' ', f) != EOF;
    for (int64_t i = 0; i < k && ok; i++) ok = fwrite(ts[i]->p, (size_t)dt_size[ts[i]->dt], (size_t)ts[i]->n, f) == (size_t)ts[i]->n;
    ok = fclose(f) == 0 && ok; free(hb); free(ts); free(nm); a_ntmp = sp;
    if (!ok) { remove(tmp); a_error("IOError", "writing '%s' failed (disk full?)", path); }
    remove(path);
    if (rename(tmp, path)) a_error("IOError", "cannot replace '%s': %s", path, strerror(errno));   /* a crash never leaves half a file */
    return A_NONE;
}
/* .npy (NumPy's format, version 1 / 2 / 3) */
BI(lb_npy_load) {
    (void)self; (void)kw; (void)n; const char *path = lib_str(a[0], "load_tensor")->s; size_t len; unsigned char *b = read_all(path, &len);
    if (len < 12 || memcmp(b, "\x93NUMPY", 6)) { free(b); a_error("ValueError", "'%s' is not a .npy file", path); }
    size_t hl, ho; if (b[6] == 1) { hl = (size_t)b[8] | (size_t)b[9] << 8; ho = 10; } else { hl = le32(b + 8); ho = 12; }
    if (ho + hl > len) { free(b); a_error("ValueError", "'%s': a broken .npy header", path); }
    char head[4096]; size_t cp = hl < sizeof head - 1 ? hl : sizeof head - 1; memcpy(head, b + ho, cp); head[cp] = 0;
    char *d = strstr(head, "descr"), *fo = strstr(head, "fortran_order"), *sh = strstr(head, "shape");
    if (!d || !fo || !sh) { free(b); a_error("ValueError", "'%s': a broken .npy header", path); }
    char descr[16] = ""; { char *q = strchr(d + 6, '\''); if (q) { char *e = strchr(q + 1, '\''); if (e && e - q - 1 < 15) { memcpy(descr, q + 1, (size_t)(e - q - 1)); descr[e - q - 1] = 0; } } }
    if (strstr(fo, "True")) { free(b); a_error("ValueError", "'%s': Fortran-ordered arrays are not read natively; save with np.ascontiguousarray", path); }
    int64_t shp[MAXD]; int nd = 0; char *q = strchr(sh, '(');
    for (q = q ? q + 1 : sh; *q && *q != ')'; ) { while (*q == ' ' || *q == ',') q++; if (*q == ')' || !*q) break; if (nd == MAXD) { free(b); a_error("ValueError", "too many dimensions"); } shp[nd++] = strtoll(q, &q, 10); }
    static const struct { const char *d, *code; } m[] = {{"<f8", "F64"}, {"<f4", "F32"}, {"<f2", "F16"}, {"<i8", "I64"}, {"<i4", "I32"}, {"<i2", "I16"}, {"|i1", "I8"}, {"|u1", "U8"}, {"|b1", "BOOL"}};
    const char *code = NULL; for (size_t i = 0; i < sizeof m / sizeof m[0]; i++) if (!strcmp(m[i].d, descr)) code = m[i].code;
    if (!code) { free(b); a_error("ValueError", "'%s': the type %s is not read natively (float64/32/16, int64/32/16/8, uint8, bool are)", path, descr); }
    int esz; int dt = st_dtype(code, &esz); ATensor *t = t_new(dt, nd, shp);
    if ((uint64_t)t->n * (uint64_t)esz > len - ho - hl) { free(b); a_error("ValueError", "'%s': the file is shorter than its shape says", path); }
    put_tensor_bytes(t, b + ho + hl, code, t->n); free(b);
    return a_obj(t);
}
BI(lb_npy_save) {
    (void)self; (void)kw; (void)n; const char *path = lib_str(a[0], "save_tensor")->s; AV v = deepval(a[1]); if (!is_t(v)) v = as_tensor(v); if (TT(v)->dev) v = to_host(v);
    ATensor *t = TT(v); static const char *descr[] = {"<f8", "<f4", "<i8", "|b1"};
    char head[512]; int k = snprintf(head, sizeof head, "{'descr': '%s', 'fortran_order': False, 'shape': (", descr[t->dt]);
    for (int i = 0; i < t->nd; i++) k += snprintf(head + k, sizeof head - (size_t)k, "%lld,%s", (long long)t->sh[i], t->nd > 1 && i < t->nd - 1 ? " " : "");
    if (t->nd > 1) k--;
    k += snprintf(head + k, sizeof head - (size_t)k, "), }");
    int total = 10 + k + 1; int pad = (64 - total % 64) % 64;
    FILE *f = fopen(path, "wb"); if (!f) a_error("IOError", "cannot write '%s': %s", path, strerror(errno));
    unsigned char pre[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0}; int hl = k + pad + 1; pre[8] = (unsigned char)hl; pre[9] = (unsigned char)(hl >> 8);
    int ok = fwrite(pre, 1, 10, f) == 10 && fwrite(head, 1, (size_t)k, f) == (size_t)k;
    for (int i = 0; i < pad; i++) ok = ok && fputc(' ', f) != EOF;
    ok = ok && fputc('\n', f) != EOF && fwrite(t->p, (size_t)dt_size[t->dt], (size_t)t->n, f) == (size_t)t->n;
    ok = fclose(f) == 0 && ok; if (!ok) a_error("IOError", "writing '%s' failed", path);
    return A_NONE;
}

/* ------------------------------------------------------------------ the HTTP server */
typedef struct { sock_t fd; char *in; size_t inl, incap; char *out; size_t outl, outpos; double deadline; int state, close_after; char peer[64]; } Conn;
enum { C_HEAD = 0, C_BODY, C_WRITE, C_IDLE };
static volatile sig_atomic_t http_stop = 0;
static void http_on_signal(int s) { (void)s; http_stop = 1; }
static const char *reason(int st) {
    switch (st) {
    case 200: return "OK"; case 201: return "Created"; case 202: return "Accepted"; case 204: return "No Content"; case 206: return "Partial Content";
    case 301: return "Moved Permanently"; case 302: return "Found"; case 303: return "See Other"; case 304: return "Not Modified"; case 307: return "Temporary Redirect"; case 308: return "Permanent Redirect";
    case 400: return "Bad Request"; case 401: return "Unauthorized"; case 403: return "Forbidden"; case 404: return "Not Found"; case 405: return "Method Not Allowed";
    case 408: return "Request Timeout"; case 409: return "Conflict"; case 411: return "Length Required"; case 413: return "Content Too Large"; case 415: return "Unsupported Media Type";
    case 421: return "Misdirected Request"; case 422: return "Unprocessable Content"; case 429: return "Too Many Requests"; case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error"; case 501: return "Not Implemented"; case 502: return "Bad Gateway"; case 503: return "Service Unavailable"; case 504: return "Gateway Timeout";
    default: return "Status";
    }
}
static void out_add(Conn *c, const char *s, size_t n) {
    if (c->outl + n + 1 > (size_t)-1 / 2) return;
    char *p = (char *)realloc(c->out, c->outl + n + 1); if (!p) return; c->out = p; memcpy(c->out + c->outl, s, n); c->outl += n;
}
static void respond_raw(Conn *c, int status, const char *ctype, const char *body, size_t bl, int close_after, const char *extra) {
    char h[512]; int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n%sConnection: %s\r\n\r\n",
        status, reason(status), ctype, bl, extra ? extra : "", close_after ? "close" : "keep-alive");
    out_add(c, h, (size_t)k); out_add(c, body, bl);
    c->close_after |= close_after; c->state = C_WRITE;
}
static void respond_error(Conn *c, int status, const char *msg) {
    char b[256]; int k = snprintf(b, sizeof b, "{\"error\":\"%s\",\"status\":%d}", msg, status);
    respond_raw(c, status, "application/json", b, (size_t)k, 1, "X-Content-Type-Options: nosniff\r\n");
}
static int tok_char(int ch) { return ch > 32 && ch < 127 && !strchr("()<>@,;:\\\"/[]?={}", ch); }
typedef struct { AV handler; int64_t max_body; double t_head, t_body, t_idle; int log; int64_t served; } Srv;
/* a whole request is in c->in: call the program and queue its answer; returns the bytes the request used, 0: wait for more */
static size_t http_request(Srv *S, Conn *c) {
    char *end = NULL; size_t hlen = 0;
    for (size_t i = 3; i < c->inl; i++) if (c->in[i] == '\n' && c->in[i - 1] == '\r' && c->in[i - 2] == '\n' && c->in[i - 3] == '\r') { end = c->in + i + 1; break; }
    if (!end) { if (c->inl > 16384) respond_error(c, 431, "headers too large"); return 0; }
    hlen = (size_t)(end - c->in);
    if (hlen > 16384) { respond_error(c, 431, "headers too large"); return 0; }
    /* the request line */
    char *p = c->in, *le = strstr(p, "\r\n"); char method[16], target[8192], ver[16];
    { char *sp1 = memchr(p, ' ', (size_t)(le - p)), *sp2 = sp1 ? memchr(sp1 + 1, ' ', (size_t)(le - sp1 - 1)) : NULL;
      if (!sp1 || !sp2 || sp1 - p >= 16 || sp2 - sp1 - 1 >= 8192 || le - sp2 - 1 >= 16 || sp2 - sp1 - 1 < 1) { respond_error(c, 400, "bad request line"); return 0; }
      memcpy(method, p, (size_t)(sp1 - p)); method[sp1 - p] = 0; memcpy(target, sp1 + 1, (size_t)(sp2 - sp1 - 1)); target[sp2 - sp1 - 1] = 0;
      memcpy(ver, sp2 + 1, (size_t)(le - sp2 - 1)); ver[le - sp2 - 1] = 0; }
    for (char *m = method; *m; m++) if (!tok_char((unsigned char)*m)) { respond_error(c, 400, "bad method"); return 0; }
    if (strcmp(ver, "HTTP/1.1") && strcmp(ver, "HTTP/1.0")) { respond_error(c, 505 == 505 ? 400 : 400, "unsupported HTTP version"); return 0; }
    if (target[0] != '/' && strcmp(method, "OPTIONS")) { respond_error(c, 400, "bad target"); return 0; }
    for (char *t = target; *t; t++) if ((unsigned char)*t <= 32 || *t == 127) { respond_error(c, 400, "bad target"); return 0; }
    /* the headers */
    int sp = a_ntmp; AV hd = a_dict(); A_PUSH(hd); int nh = 0; int64_t clen = -1; int keep = !strcmp(ver, "HTTP/1.1");
    for (char *q = le + 2; q < end - 2; ) {
        char *e = strstr(q, "\r\n"); if (!e || e > end - 2) break;
        if (*q == ' ' || *q == '\t') { a_ntmp = sp; respond_error(c, 400, "folded headers are not accepted"); return 0; }
        char *colon = memchr(q, ':', (size_t)(e - q)); if (!colon || colon == q) { a_ntmp = sp; respond_error(c, 400, "bad header"); return 0; }
        if (++nh > 100) { a_ntmp = sp; respond_error(c, 431, "too many headers"); return 0; }
        for (char *z = q; z < colon; z++) { if (!tok_char((unsigned char)*z)) { a_ntmp = sp; respond_error(c, 400, "bad header name"); return 0; } *z = (char)((*z >= 'A' && *z <= 'Z') ? *z + 32 : *z); }
        char *v = colon + 1; while (v < e && (*v == ' ' || *v == '\t')) v++; char *ve = e; while (ve > v && (ve[-1] == ' ' || ve[-1] == '\t')) ve--;
        AV key = a_strn(q, colon - q); A_PUSH(key); AV val = a_strn(v, ve - v); A_PUSH(val);
        if (!strcmp(STR(key)->s, "content-length")) {
            if (clen >= 0) { a_ntmp = sp; respond_error(c, 400, "duplicate Content-Length"); return 0; }
            char *z; long long L = strtoll(STR(val)->s, &z, 10); if (*z || L < 0 || STR(val)->len == 0) { a_ntmp = sp; respond_error(c, 400, "bad Content-Length"); return 0; }
            clen = L;
        }
        if (!strcmp(STR(key)->s, "transfer-encoding")) { a_ntmp = sp; respond_error(c, 501, "chunked request bodies are not accepted; send Content-Length"); return 0; }
        if (!strcmp(STR(key)->s, "connection")) { if (strstr(STR(val)->s, "close") || strstr(STR(val)->s, "Close")) keep = 0; else if (strstr(STR(val)->s, "eep-alive")) keep = 1; }
        AV old; if (a_dict_get((ADict *)hd.o, key, &old)) { AV parts[3] = {old, a_str(", "), val}; val = a_concat_strs(3, parts); A_PUSH(val); }
        a_dict_set((ADict *)hd.o, key, val); a_ntmp = sp + 1;
        q = e + 2;
    }
    if (clen < 0) clen = 0;
    if (clen > S->max_body) { a_ntmp = sp; respond_error(c, 413, "body too large"); return 0; }
    if (c->inl < hlen + (size_t)clen) { a_ntmp = sp; if (c->state == C_HEAD) { c->state = C_BODY; c->deadline = now_s() + S->t_body; } return 0; }
    /* the request, to the program */
    AV req = a_dict(); A_PUSH(req); ADict *R = (ADict *)req.o;
    char *qm = strchr(target, '?'); AV path = a_strn(target, qm ? qm - target : (int64_t)strlen(target)); A_PUSH(path);
    a_dict_set(R, a_str("method"), a_str(method)); a_dict_set(R, a_str("path"), path); a_dict_set(R, a_str("query"), a_str(qm ? qm + 1 : ""));
    a_dict_set(R, a_str("target"), a_str(target)); a_dict_set(R, a_str("headers"), hd); a_dict_set(R, a_str("peer"), a_str(c->peer));
    a_dict_set(R, a_str("version"), a_str(ver)); a_dict_set(R, a_str("body"), a_strn(c->in + hlen, clen));
    AV res = A_NONE; AHandler h; double t0 = now_s(); int failed = 0;
    if (!A_TRY(h)) { res = a_call(S->handler, 1, &req, NULL); a_handler = h.prev; }
    else { failed = 1; AV m = a_to_str(a_exc); fprintf(stderr, "[apsi web] %s %s: %s\n", method, target, STR(m)->s); }
    A_PUSH(res);
    if (failed || !a_is(res, O_TUPLE) || ((AList *)res.o)->len != 3) respond_error(c, 500, "internal error");
    else {
        AList *t = (AList *)res.o; int status = t->v[0].t == T_INT ? (int)t->v[0].i : 500;
        AV body = t->v[2]; if (!a_is(body, O_STR)) body = a_to_str(body); A_PUSH(body);
        char line[160]; int k = snprintf(line, sizeof line, "HTTP/1.1 %d %s\r\n", status, reason(status)); out_add(c, line, (size_t)k);
        AV hs = t->v[1];
        if (a_is(hs, O_LIST) || a_is(hs, O_TUPLE)) for (int64_t i = 0; i < ((AList *)hs.o)->len; i++) {
            AV pr = ((AList *)hs.o)->v[i]; if (!(a_is(pr, O_LIST) || a_is(pr, O_TUPLE)) || ((AList *)pr.o)->len != 2) continue;
            AV hk = a_to_str(((AList *)pr.o)->v[0]); A_PUSH(hk); AV hv = a_to_str(((AList *)pr.o)->v[1]); A_PUSH(hv);
            if (strpbrk(STR(hk)->s, "\r\n:") || strpbrk(STR(hv)->s, "\r\n")) continue;      /* no header injection */
            out_add(c, STR(hk)->s, (size_t)STR(hk)->len); out_add(c, ": ", 2); out_add(c, STR(hv)->s, (size_t)STR(hv)->len); out_add(c, "\r\n", 2);
        }
        int head_only = !strcmp(method, "HEAD");
        k = snprintf(line, sizeof line, "Content-Length: %lld\r\nConnection: %s\r\n\r\n", (long long)STR(body)->len, keep ? "keep-alive" : "close");
        out_add(c, line, (size_t)k); if (!head_only) out_add(c, STR(body)->s, (size_t)STR(body)->len);
        c->state = C_WRITE; c->close_after |= !keep;
        if (S->log) fprintf(stderr, "[apsi web] %s %s %d %.1f ms\n", method, target, status, (now_s() - t0) * 1e3);
    }
    a_ntmp = sp; S->served++;
    return hlen + (size_t)clen;
}
static void set_nonblock(sock_t fd) {
#ifdef _WIN32
    u_long one = 1; ioctlsocket(fd, FIONBIO, &one);
#else
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
#endif
}
/* __http_serve(host, port, handler, options): runs until Ctrl+C (or SIGTERM); returns the number of requests served */
BI(lb_http_serve) {
    (void)self; (void)kw; const char *host = lib_str(a[0], "serve")->s; int64_t port = a[1].i; AV opt = n > 3 ? a[3] : A_NONE;
    Srv S = {a[2], 1 << 20, 10, 30, 15, 1, 0}; int64_t maxc = 10000;
    if (a_is(opt, O_DICT)) {
        AV v; ADict *o = (ADict *)opt.o;
        if (a_dict_get(o, a_str("max_body"), &v)) S.max_body = v.i;
        if (a_dict_get(o, a_str("header_timeout"), &v)) S.t_head = a_c_f64(v);
        if (a_dict_get(o, a_str("body_timeout"), &v)) S.t_body = a_c_f64(v);
        if (a_dict_get(o, a_str("keepalive_timeout"), &v)) S.t_idle = a_c_f64(v);
        if (a_dict_get(o, a_str("max_connections"), &v)) maxc = v.i;
        if (a_dict_get(o, a_str("log"), &v)) S.log = a_truthy(v);
    }
    int sp = a_ntmp; A_PUSH(S.handler);
#ifdef _WIN32
    WSADATA wd; WSAStartup(MAKEWORD(2, 2), &wd);
#else
    signal(SIGPIPE, SIG_IGN);
#endif
    struct addrinfo hints, *ai = NULL; memset(&hints, 0, sizeof hints); hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM; hints.ai_flags = AI_PASSIVE;
    char ps[16]; snprintf(ps, sizeof ps, "%lld", (long long)port);
    if (getaddrinfo(*host ? host : NULL, ps, &hints, &ai) || !ai) a_error("OSError", "serve: cannot use the address %s:%lld", host, (long long)port);
    sock_t L = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (L == SOCK_BAD) { freeaddrinfo(ai); a_error("OSError", "serve: no socket"); }
    int one = 1; setsockopt(L, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof one);
    if (bind(L, ai->ai_addr, (int)ai->ai_addrlen) || listen(L, 511)) { int e = sock_err(); freeaddrinfo(ai); sock_close(L);
        a_error("OSError", "serve: cannot listen on %s:%lld (error %d)%s", host, (long long)port, e, e == 98 || e == 10048 ? ": the port is in use; pick another port= or stop the other program" : ""); }
    freeaddrinfo(ai); set_nonblock(L);
    http_stop = 0; void (*old_int)(int) = signal(SIGINT, http_on_signal);
#ifdef SIGTERM
    void (*old_term)(int) = signal(SIGTERM, http_on_signal);
#endif
    fprintf(stderr, "[apsi web] listening on http://%s:%lld  (Ctrl+C stops)\n", *host ? host : "0.0.0.0", (long long)port);
    int cap = 64; Conn *cs = (Conn *)calloc((size_t)cap, sizeof(Conn)); int nc = 0;
    struct pollfd *pf = (struct pollfd *)calloc((size_t)cap + 1, sizeof(struct pollfd));
    while (!http_stop || nc) {
        int np = 0;
        if (!http_stop && nc < maxc) { pf[np].fd = L; pf[np].events = POLLIN; pf[np].revents = 0; np++; }
        int base = np;
        for (int i = 0; i < nc; i++) { pf[np].fd = cs[i].fd; pf[np].events = cs[i].state == C_WRITE ? POLLOUT : POLLIN; pf[np].revents = 0; np++; }
        int r = sock_poll(pf, (unsigned)np, 250);
        double t = now_s();
        if (http_stop) for (int i = 0; i < nc; i++) if (cs[i].state != C_WRITE) cs[i].deadline = 0;     /* shutting down: finish answers only */
        if (r > 0 && base && (pf[0].revents & POLLIN)) for (int acc = 0; acc < 64 && nc < maxc; acc++) {
            struct sockaddr_storage sa; socklen_t sl = sizeof sa; sock_t fd = accept(L, (struct sockaddr *)&sa, &sl);
            if (fd == SOCK_BAD) break;
            set_nonblock(fd); setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof one);
            if (nc == cap) { cap *= 2; cs = (Conn *)realloc(cs, sizeof(Conn) * (size_t)cap); pf = (struct pollfd *)realloc(pf, sizeof(struct pollfd) * ((size_t)cap + 1)); }
            Conn *c = &cs[nc++]; memset(c, 0, sizeof *c); c->fd = fd; c->deadline = t + S.t_head; c->state = C_IDLE;
            if (sa.ss_family == AF_INET) inet_ntop(AF_INET, &((struct sockaddr_in *)&sa)->sin_addr, c->peer, sizeof c->peer);
            else inet_ntop(AF_INET6, &((struct sockaddr_in6 *)&sa)->sin6_addr, c->peer, sizeof c->peer);
        }
        for (int i = 0; i < nc; i++) {
            Conn *c = &cs[i]; short ev = 0;
            for (int j = base; j < np; j++) if (pf[j].fd == c->fd) { ev = pf[j].revents; break; }
            int drop = 0;
            if (ev & (POLLERR | POLLNVAL)) drop = 1;
            else if (c->state != C_WRITE && (ev & (POLLIN | POLLHUP))) {
                if (c->incap - c->inl < 16384) { size_t nc2 = c->incap ? c->incap * 2 : 32768; char *q = (char *)realloc(c->in, nc2 + 1); if (!q) { drop = 1; goto next; } c->in = q; c->incap = nc2; }
                long got = (long)recv(c->fd, c->in + c->inl, (int)(c->incap - c->inl), 0);
                if (got <= 0) { if (got < 0 && SOCK_WOULDBLOCK(sock_err())) goto next; drop = 1; goto next; }
                if (c->state == C_IDLE) { c->state = C_HEAD; c->deadline = t + S.t_head; }
                c->inl += (size_t)got;
                if (c->inl > 16384 + (size_t)S.max_body + 64) { respond_error(c, 413, "request too large"); goto next; }
                size_t used = http_request(&S, c);
                if (used) { memmove(c->in, c->in + used, c->inl - used); c->inl -= used; }
            } else if (c->state == C_WRITE && (ev & POLLOUT)) {
                long put = (long)send(c->fd, c->out + c->outpos, (int)(c->outl - c->outpos), MSG_NOSIGNAL);
                if (put < 0) { if (!SOCK_WOULDBLOCK(sock_err())) drop = 1; goto next; }
                c->outpos += (size_t)put;
                if (c->outpos == c->outl) {
                    free(c->out); c->out = NULL; c->outl = c->outpos = 0;
                    if (c->close_after || http_stop) { drop = 1; goto next; }
                    c->state = C_IDLE; c->deadline = t + S.t_idle;
                    if (c->inl) { c->state = C_HEAD; size_t used = http_request(&S, c); if (used) { memmove(c->in, c->in + used, c->inl - used); c->inl -= used; } }   /* a pipelined request */
                }
            }
            if (!drop && c->state != C_WRITE && t > c->deadline) {
                if (c->state == C_HEAD || c->state == C_BODY) respond_error(c, 408, "request timeout"); else drop = 1;
            }
        next:
            if (drop) { sock_close(c->fd); free(c->in); free(c->out); cs[i] = cs[--nc]; i--; }
        }
        a_gc_safe();
    }
    sock_close(L); free(cs); free(pf);
    signal(SIGINT, old_int);
#ifdef SIGTERM
    signal(SIGTERM, old_term);
#endif
    fprintf(stderr, "[apsi web] stopped after %lld requests\n", (long long)S.served);
    a_ntmp = sp;
    return a_int(S.served);
}

/* ------------------------------------------------------------------ small global built-ins */
static int64_t ival(AV v, const char *what) { if (v.t == T_INT || v.t == T_BOOL) return v.i; a_error("TypeError", "%s takes whole numbers", what); }
BI(lg_band) { (void)self; (void)kw; (void)n; return a_int(ival(a[0], "band") & ival(a[1], "band")); }
BI(lg_bor) { (void)self; (void)kw; (void)n; return a_int(ival(a[0], "bor") | ival(a[1], "bor")); }
BI(lg_bxor) { (void)self; (void)kw; (void)n; return a_int(ival(a[0], "bxor") ^ ival(a[1], "bxor")); }
BI(lg_bnot) { (void)self; (void)kw; (void)n; return a_int(~ival(a[0], "bnot")); }
BI(lg_shl) { (void)self; (void)kw; (void)n; int64_t k = ival(a[1], "shl"); if (k < 0 || k > 63) a_error("ValueError", "shl: the shift is 0 to 63"); return a_int((int64_t)((uint64_t)ival(a[0], "shl") << k)); }
BI(lg_shr) { (void)self; (void)kw; (void)n; int64_t k = ival(a[1], "shr"); if (k < 0 || k > 63) a_error("ValueError", "shr: the shift is 0 to 63"); return a_int(ival(a[0], "shr") >> k); }
BI(lg_platform) {
    (void)self; (void)kw; (void)n; (void)a;
#ifdef _WIN32
    return a_str("windows");
#elif defined(__APPLE__)
    return a_str("macos");
#else
    return a_str("linux");
#endif
}
BI(lg_id) { (void)self; (void)kw; (void)n; return a[0].t == T_OBJ ? a_int((int64_t)(intptr_t)a[0].o) : a_int((int64_t)a[0].i ^ ((int64_t)a[0].t << 56)); }
BI(lg_nbytes) {
    (void)self; (void)kw; (void)n; AV v = deepval(a[0]);
    if (is_t(v)) return a_int(TT(v)->n * (int64_t)dt_size[TT(v)->dt]);
    if (a_is(v, O_STR)) return a_int(STR(v)->len);
    return a_int(8);
}
BI(lg_format) { (void)self; (void)kw; return a_format(a[0], n > 1 ? lib_str(a[1], "format")->s : ""); }
BI(lg_native_threads) { (void)self; (void)kw; (void)n; (void)a; return a_int(a_threads()); }
BI(lg_native_info) {
    (void)self; (void)kw; (void)n; (void)a; AV d = a_dict(); int sp = a_ntmp; A_PUSH(d); ADict *D = (ADict *)d.o;
#if defined(__x86_64__) || defined(_M_X64)
    a_dict_set(D, a_str("arch"), a_str("x86_64"));
    __builtin_cpu_init();
    AV simd = a_list(0, NULL); A_PUSH(simd);
    if (__builtin_cpu_supports("sse4.2")) a_list_append((AList *)simd.o, a_str("sse4.2"));
    if (__builtin_cpu_supports("avx2")) a_list_append((AList *)simd.o, a_str("avx2"));
    if (__builtin_cpu_supports("fma")) a_list_append((AList *)simd.o, a_str("fma"));
    if (__builtin_cpu_supports("avx512f")) a_list_append((AList *)simd.o, a_str("avx512f"));
    a_dict_set(D, a_str("simd"), simd);
#elif defined(__aarch64__)
    a_dict_set(D, a_str("arch"), a_str("arm64")); AV simd = a_list(0, NULL); A_PUSH(simd); a_list_append((AList *)simd.o, a_str("neon")); a_dict_set(D, a_str("simd"), simd);
#else
    a_dict_set(D, a_str("arch"), a_str("other"));
#endif
    a_dict_set(D, a_str("threads"), a_int(a_threads()));
    AV nm[1] = {A_NONE}; (void)nm;
    a_dict_set(D, a_str("os"), lg_platform(0, NULL, NULL, A_NONE));
    a_dict_set(D, a_str("engine"), a_str("native"));
    a_ntmp = sp; return d;
}
static const TEnt lib_globals[] = {
    {"band", lg_band}, {"bor", lg_bor}, {"bxor", lg_bxor}, {"bnot", lg_bnot}, {"shl", lg_shl}, {"shr", lg_shr},
    {"platform", lg_platform}, {"id", lg_id}, {"nbytes", lg_nbytes}, {"format", lg_format},
    {"native_threads", lg_native_threads}, {"_native_info", lg_native_info},
    {NULL, NULL}};

BI(lb_gen_step) { (void)self; (void)kw; return a_gen_step(a[0], n > 1 ? a[1] : A_NONE); }
BI(lb_is_coro) { (void)self; (void)kw; (void)n; return a_bool(a_is_coro(a[0])); }
BI(lb_builtin) { (void)self; (void)kw; (void)n; return a_global_builtin(lib_str(a[0], "builtin")->s); }
BI(lb_setenv) {
    (void)self; (void)kw; (void)n; const char *k = lib_str(a[0], "setenv")->s;
#ifdef _WIN32
    _putenv_s(k, a[1].t == T_NONE ? "" : lib_str(a[1], "setenv")->s);
#else
    if (a[1].t == T_NONE) unsetenv(k); else setenv(k, lib_str(a[1], "setenv")->s, 1);
#endif
    return A_NONE;
}
extern int a_argc; extern char **a_argv;
#ifndef _WIN32
extern char **environ;
#endif
BI(lb_argv) {                                /* the program's arguments after its own name */
    (void)self; (void)kw; (void)n; (void)a; AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l);
    for (int i = 1; i < a_argc; i++) a_list_append((AList *)l.o, a_str(a_argv[i]));
    a_ntmp = sp; return l;
}
BI(lb_environ_names) {
    (void)self; (void)kw; (void)n; (void)a; AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l);
#ifdef _WIN32
    char **env = _environ;
#else
    char **env = environ;
#endif
    for (char **e = env; e && *e; e++) { const char *q = strchr(*e, '='); if (q && q != *e) a_list_append((AList *)l.o, a_strn(*e, q - *e)); }
    a_ntmp = sp; return l;
}
BI(lb_rec_dict) {                           /* a record's (or an instance's) fields as a new dict */
    (void)self; (void)kw; (void)n; if (!a_is(a[0], O_INST)) return a[0];
    ADict *d = ((AInst *)a[0].o)->attrs; AV out = a_dict(); int sp = a_ntmp; A_PUSH(out);
    for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set((ADict *)out.o, d->e[i].k, d->e[i].v);
    a_ntmp = sp; return out;
}
BI(lb_is_rec) { (void)self; (void)kw; (void)n; return a_bool(a_is(a[0], O_INST) && ((AInst *)a[0].o)->cls->name && !strcmp(((AInst *)a[0].o)->cls->name->s, "record")); }
/* __localtime(form): "stamp" -> 2026-11-25 07:30:12.345, "iso" -> 2026-11-25T07:30:12.345+07:00, "date" */
BI(lb_localtime) {
    (void)self; (void)kw; const char *form = n ? lib_str(a[0], "localtime")->s : "stamp";
    double t = now_s(); time_t sec = (time_t)t; int ms = (int)((t - (double)sec) * 1000); struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &sec);
#else
    localtime_r(&sec, &tmv);
#endif
    char b[64];
    if (!strcmp(form, "utc")) {                                  /* 2026-01-02T03:04:05Z */
#ifdef _WIN32
        gmtime_s(&tmv, &sec);
#else
        gmtime_r(&sec, &tmv);
#endif
        strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tmv); return a_str(b);
    }
    if (!strcmp(form, "date")) strftime(b, sizeof b, "%Y-%m-%d", &tmv);
    else if (!strcmp(form, "iso")) {
        char z[16]; strftime(z, sizeof z, "%z", &tmv); size_t k = strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", &tmv);
        snprintf(b + k, sizeof b - k, ".%03d%.3s:%.2s", ms, z, z[0] ? z + 3 : "00");
    } else { size_t k = strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S", &tmv); snprintf(b + k, sizeof b - k, ".%03d", ms); }
    return a_str(b);
}
BI(lb_isatty) {
    (void)self; (void)kw; (void)n; (void)a;
#ifdef _WIN32
    return a_bool(_isatty(2));
#else
    return a_bool(isatty(2));
#endif
}
/* ------------------------------------------------------------------ numeric formats (use precision)
   __qfmt(x, fmt, "enc" | "dec"): round-to-nearest-even codes of BF16, the OCP minifloats (FP8 E4M3 / E5M2,
   FP6 E3M2 / E2M3, FP4 E2M1) and symmetric integers, and their decoded FP32 values: bit-exact with apsi run. */
typedef struct { int kind, e, m, bias, finite_only, nan_top, qmax; } QFmt;   /* kind: 1 bf16, 2 minifloat, 3 int */
static int qfmt_get(const char *nm, QFmt *f) {
    memset(f, 0, sizeof *f);
    if (!strcmp(nm, "bf16")) { f->kind = 1; return 1; }
    if (!strcmp(nm, "f16")) { f->kind = 4; return 1; }
    if (!strcmp(nm, "f32")) { f->kind = 5; return 1; }
    static const struct { const char *n; int e, m, b, fo, nt; } mf[] = {
        {"fp8_e4m3", 4, 3, 7, 1, 1}, {"fp8_e5m2", 5, 2, 15, 0, 0}, {"fp6_e3m2", 3, 2, 3, 1, 0},
        {"fp6_e2m3", 2, 3, 1, 1, 0}, {"fp4_e2m1", 2, 1, 1, 1, 0}};
    for (int i = 0; i < 5; i++) if (!strcmp(nm, mf[i].n)) { f->kind = 2; f->e = mf[i].e; f->m = mf[i].m; f->bias = mf[i].b; f->finite_only = mf[i].fo; f->nan_top = mf[i].nt; return 1; }
    if (!strcmp(nm, "int8")) { f->kind = 3; f->qmax = 127; return 1; }
    if (!strcmp(nm, "int4")) { f->kind = 3; f->qmax = 7; return 1; }
    return 0;
}
static double qmf_value(const QFmt *f, int code) {
    int e = code >> f->m, m = code & ((1 << f->m) - 1), top = (1 << f->e) - 1;
    double v = e == 0 ? ((double)m / (1 << f->m)) * pow(2.0, 1 - f->bias) : (1 + (double)m / (1 << f->m)) * pow(2.0, e - f->bias);
    if (!f->finite_only && e == top) v = m == 0 ? INFINITY : NAN;
    if (f->nan_top && e == top && m == (1 << f->m) - 1) v = NAN;
    return v;
}
static uint16_t d2h(double x) {                          /* float64 -> IEEE half, round to nearest even (as numpy) */
    uint16_t sg = signbit(x) ? 0x8000 : 0; double a = fabs(x);
    if (a != a) return 0x7E00;
    if (a >= 65520.0) return sg | 0x7C00;
    if (a < ldexp(1.0, -14)) return sg | (uint16_t)rint(a / ldexp(1.0, -24));
    int ex; frexp(a, &ex); int e = ex - 1; double q = rint((ldexp(a, -e) - 1) * 1024);
    if (q >= 1024) { q = 0; e++; }
    if (e > 15) return sg | 0x7C00;
    return sg | (uint16_t)((e + 15) << 10) | (uint16_t)q;
}
static double h2d(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023; double v;
    if (e == 0) v = ldexp((double)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexp(1 + m / 1024.0, e - 15);
    return (h & 0x8000) ? -v : v;
}
BI(lb_qfmt) {
    (void)self; (void)kw; if (n < 3) a_error("TypeError", "__qfmt(x, fmt, op)");
    QFmt f; if (!qfmt_get(lib_str(a[1], "precision")->s, &f)) a_error("ValueError", "unknown numeric format '%s'", lib_str(a[1], "precision")->s);
    int dec = !strcmp(lib_str(a[2], "precision")->s, "dec");
    ATensor *X = lf64(a[0], "precision"); int sp = a_ntmp; A_PUSH(a_obj(X));
    ATensor *Y = t_new(DT_F64, X->nd, X->sh); A_PUSH(a_obj(Y));
    const double *x = (const double *)X->p; double *y = (double *)Y->p; int64_t cnt = X->n;
    if (f.kind == 4) for (int64_t i = 0; i < cnt; i++) y[i] = dec ? h2d((uint16_t)(int64_t)x[i]) : (double)d2h(x[i]);
    else if (f.kind == 5) for (int64_t i = 0; i < cnt; i++) y[i] = (float)x[i];
    else if (f.kind == 1) {
        for (int64_t i = 0; i < cnt; i++) {
            if (!dec) { float v = (float)x[i]; uint32_t u; memcpy(&u, &v, 4);
                        y[i] = v != v ? (double)0x7FC0 : (double)(uint16_t)(((uint64_t)u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
            else { uint32_t u = (uint32_t)(int64_t)x[i] << 16; float v; memcpy(&v, &u, 4); y[i] = v; }
        }
    } else if (f.kind == 3) {
        for (int64_t i = 0; i < cnt; i++) {
            if (dec) { y[i] = (float)x[i]; continue; }
            double r = rint(x[i]); if (r > f.qmax) r = f.qmax; if (r < -f.qmax) r = -f.qmax; if (r != r) r = 0; y[i] = r + 0.0;   /* an int8 code has no -0 */
        }
    } else {
        int nm = f.e + f.m, ncode = 1 << nm; double tab[128]; int ord[128], k = 0, nan_code = ncode - 1, have_nan = 0;
        for (int c = 0; c < ncode; c++) { tab[c] = qmf_value(&f, c); if (tab[c] != tab[c] && !have_nan) { nan_code = c; have_nan = 1; } }
        for (int c = 0; c < ncode; c++) if (isfinite(tab[c])) ord[k++] = c;
        for (int i = 1; i < k; i++) { int c = ord[i], j = i; while (j > 0 && tab[ord[j - 1]] > tab[c]) { ord[j] = ord[j - 1]; j--; } ord[j] = c; }
        double mx = tab[ord[k - 1]];
        for (int64_t i = 0; i < cnt; i++) {
            if (dec) {
                int64_t c = (int64_t)x[i]; double mag = tab[c & (ncode - 1)]; y[i] = (float)(((c >> nm) & 1) ? -mag : mag); continue;
            }
            double v = x[i];
            if (v != v) { y[i] = nan_code; continue; }
            int sg = signbit(v) ? 1 : 0; double ac = fabs(v); if (ac > mx) ac = mx;
            int lo = 0, hi = k;                                   /* searchsorted (left) */
            while (lo < hi) { int mid = (lo + hi) / 2; if (tab[ord[mid]] < ac) lo = mid + 1; else hi = mid; }
            int idx = lo < 1 ? 1 : lo > k - 1 ? k - 1 : lo;
            double l = tab[ord[idx - 1]], h = tab[ord[idx]];
            int pick = (ac - l) > (h - ac) || ((ac - l) == (h - ac) && (ord[idx] & 1) == 0);
            y[i] = (double)((pick ? ord[idx] : ord[idx - 1]) | (sg << nm));
        }
    }
    a_ntmp = sp;
    return X->nd == 0 ? a_float(y[0]) : a_obj(Y);
}
BI(lb_is_traced) { (void)kw; (void)self; return a_bool(n && is_n(a[0])); }   /* recorded by grad() */
/* __frombuffer(bytes, offset, count, size, signed, little): integers packed in bytes as a float64 tensor */
BI(lb_frombuffer) {
    (void)self; (void)kw; (void)n; AStr *b = lib_str(a[0], "frombuffer"); int64_t off = a[1].i, cnt = a[2].i, sz = a[3].i; int sg = a_truthy(a[4]), le = a_truthy(a[5]);
    if (sz != 1 && sz != 2 && sz != 4 && sz != 8) a_error("ValueError", "frombuffer: size 1, 2, 4 or 8");
    if (off < 0 || cnt < 0 || off + cnt * sz > b->len) a_error("ValueError", "frombuffer: the data is shorter than %lld values", (long long)cnt);
    int64_t sh[1] = {cnt}; ATensor *T = t_new(DT_F64, 1, sh); double *o = (double *)T->p; const unsigned char *p = (const unsigned char *)b->s + off;
    for (int64_t i = 0; i < cnt; i++, p += sz) {
        uint64_t u = 0; for (int k = 0; k < sz; k++) u |= (uint64_t)p[le ? k : sz - 1 - k] << (8 * k);
        if (sg && sz < 8 && (u >> (8 * sz - 1)) & 1) u |= ~0ULL << (8 * sz);
        o[i] = sg ? (double)(int64_t)u : (double)u;
    }
    return a_obj(T);
}
/* __pk(code, little, value) -> bytes, __upk(bytes, offset, code, little) -> value: one value as struct.pack writes it
   (codes B H I Q b h i q f d ? c) */
static int pk_size(const char *c) { switch (*c) { case 'B': case 'b': case '?': case 'c': return 1; case 'H': case 'h': return 2; case 'I': case 'i': case 'f': return 4; case 'Q': case 'q': case 'd': return 8; } a_error("ValueError", "pack: unknown code '%s'", c); return 0; }
BI(lb_pk) {
    (void)self; (void)kw; (void)n; const char *c = lib_str(a[0], "pack")->s; int le = a_truthy(a[1]), sz = pk_size(c); AV v = a[2];
    unsigned char raw[8]; uint64_t u = 0;
    if (*c == 'f') { float f = (float)(v.t == T_FLOAT ? v.f : (double)v.i); uint32_t w; memcpy(&w, &f, 4); u = w; }
    else if (*c == 'd') { double d = v.t == T_FLOAT ? v.f : (double)v.i; memcpy(&u, &d, 8); }
    else if (*c == 'c') { AStr *t = lib_str(v, "pack"); u = (unsigned char)t->s[0]; }
    else if (*c == '?') u = a_truthy(v) ? 1 : 0;
    else u = (uint64_t)(v.t == T_FLOAT ? (int64_t)v.f : v.i);
    for (int k = 0; k < sz; k++) raw[le ? k : sz - 1 - k] = (unsigned char)(u >> (8 * k));
    return a_bytes((const char *)raw, sz);
}
BI(lb_upk) {
    (void)self; (void)kw; (void)n; AStr *b = lib_str(a[0], "unpack"); int64_t off = a[1].i; const char *c = lib_str(a[2], "unpack")->s; int le = a_truthy(a[3]), sz = pk_size(c);
    if (off < 0 || off + sz > b->len) a_error("ValueError", "unpack: %d bytes needed at %lld", sz, (long long)off);
    const unsigned char *p = (const unsigned char *)b->s + off; uint64_t u = 0;
    for (int k = 0; k < sz; k++) u |= (uint64_t)p[le ? k : sz - 1 - k] << (8 * k);
    switch (*c) {
    case 'f': { uint32_t w = (uint32_t)u; float f; memcpy(&f, &w, 4); return a_float((double)f); }
    case 'd': { double d; memcpy(&d, &u, 8); return a_float(d); }
    case '?': return a_bool(u != 0);
    case 'c': { char ch = (char)u; unsigned char uc = (unsigned char)u; if (uc < 128) return a_strn(&ch, 1); char t2[2] = {(char)(0xC0 | (uc >> 6)), (char)(0x80 | (uc & 63))}; return a_strn(t2, 2); }
    case 'b': return a_int((int8_t)u); case 'h': return a_int((int16_t)u); case 'i': return a_int((int32_t)u);
    default: return a_int((int64_t)u);
    }
}
BI(lb_dec);
BI(lb_task) { (void)self; return a_task_new(n, a, kw); }               /* __task(f, args...) */
void a_set_threads(int n);
BI(lb_set_threads) { (void)self; (void)kw; (void)n; a_set_threads((int)a[0].i); return a_int(a_threads()); }
BI(lb_in_task) { (void)self; (void)kw; (void)n; (void)a; return a_bool(a_in_task()); }
BI(lb_suspend) { (void)self; (void)kw; return a_yield(n ? a[0] : A_NONE); }
BI(lb_cpu_count) {
    (void)self; (void)kw; (void)n; (void)a;
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); return a_int((int64_t)si.dwNumberOfProcessors);
#else
    long c = sysconf(_SC_NPROCESSORS_ONLN); return a_int(c > 0 ? c : 1);
#endif
}
BI(lb_setraw) { (void)self; (void)kw; (void)n; if (!a_is(a[0], O_INST)) a_error("TypeError", "__setraw: an object"); a_dict_set(((AInst *)a[0].o)->attrs, a[1], a[2]); return A_NONE; }
#define APSI_VERSION "2.6.6"                                  /* kept equal to apsi.__version__ by the tests */
BI(lb_apsi_version) { (void)n; (void)a; (void)kw; (void)self; return a_str(APSI_VERSION); }
BI(lb_frame);
BI(lb_sqlite);
BI(lb_net);
BI(lb_ct);
BI(lb_is_cplx);
BI(lb_ckron);
BI(lb_arity) {                                                /* required positional parameters of an Apsi function */
    (void)kw; (void)self; AV f = n ? a[0] : A_NONE; int d = n > 1 ? (int)a[1].i : -1;
    if (a_is(f, O_BOUND)) { AV g = ((ABound *)f.o)->fn; if (a_is(g, O_FUNC) && ((AFunc *)g.o)->nreq > 0) return a_int(((AFunc *)g.o)->nreq - 1); return a_int(d); }
    if (a_is(f, O_FUNC) && ((AFunc *)f.o)->nreq >= 0) return a_int(((AFunc *)f.o)->nreq);
    return a_int(d);
}
BI(lb_is_dev) { (void)self; (void)kw; (void)n; AV v = deepval(a[0]); return a_bool(is_t(v) && TT(v)->dev); }
static const TEnt lib_builtins[] = {
    {"__hash", lb_hash}, {"__pbkdf2", lb_pbkdf2}, {"__hmac", lb_hmac}, {"__b64enc", lb_b64enc}, {"__b64dec", lb_b64dec}, {"__crc32", lb_crc32},
    {"__secure_eq", lb_secure_eq}, {"__random_bytes", lb_random_bytes}, {"__secure_randint", lb_secure_randint},
    {"__password_hash", lb_password_hash}, {"__password_verify", lb_password_verify},
    {"__la", lb_la}, {"__fft", lb_fft}, {"__special", lb_special}, {"__conv2d", lb_conv2d}, {"__maxpool2d", lb_maxpool2d},
    {"__time", lb_time}, {"__sleep", lb_sleep}, {"__fs", lb_fs}, {"__st_load", lb_st_load}, {"__st_save", lb_st_save},
    {"__npy_load", lb_npy_load}, {"__npy_save", lb_npy_save}, {"__http_serve", lb_http_serve}, {"__is_dev", lb_is_dev}, {"__frombuffer", lb_frombuffer}, {"__pk", lb_pk}, {"__dec", lb_dec}, {"__task", lb_task}, {"__in_task", lb_in_task}, {"__suspend", lb_suspend}, {"__cpu_count", lb_cpu_count}, {"__set_threads", lb_set_threads}, {"__setraw", lb_setraw}, {"__upk", lb_upk}, {"__frame", lb_frame}, {"__sqlite", lb_sqlite}, {"__net", lb_net}, {"__ct", lb_ct}, {"__is_cplx", lb_is_cplx}, {"__ckron", lb_ckron}, {"__is_traced", lb_is_traced}, {"__qfmt", lb_qfmt}, {"__apsi_version", lb_apsi_version}, {"__arity", lb_arity}, {"__localtime", lb_localtime}, {"__isatty", lb_isatty}, {"__rec_dict", lb_rec_dict}, {"__is_rec", lb_is_rec}, {"__argv", lb_argv}, {"__environ_names", lb_environ_names}, {"__builtin", lb_builtin}, {"__setenv", lb_setenv}, {"__gen_step", lb_gen_step}, {"__is_coro", lb_is_coro},
    {NULL, NULL}};
