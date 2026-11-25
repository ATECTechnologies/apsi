/* SQLite for native programs (`use db` → db.sqlite): the SQLite library the system already has, loaded when first
   used (Windows 10+: winsqlite3.dll; Linux: libsqlite3.so.0; macOS: libsqlite3.dylib; APSI_SQLITE names another),
   so nothing is linked in and nothing has to be installed. __sqlite(op, ...) returns ("error", message) on failure
   so the library raises its own DatabaseError. Included at the end of apsi_tensor.c. */
#ifndef _WIN32
#include <dlfcn.h>
#endif

typedef struct sqlite3 sqlite3; typedef struct sqlite3_stmt sqlite3_stmt;
static struct {
    int state; void *lib; char path[512];
    int (*open_v2)(const char *, sqlite3 **, int, const char *);
    int (*close_v2)(sqlite3 *);
    int (*prepare_v2)(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
    int (*step)(sqlite3_stmt *); int (*finalize)(sqlite3_stmt *);
    int (*column_count)(sqlite3_stmt *); const char *(*column_name)(sqlite3_stmt *, int);
    int (*column_type)(sqlite3_stmt *, int); long long (*column_int64)(sqlite3_stmt *, int);
    double (*column_double)(sqlite3_stmt *, int); const unsigned char *(*column_text)(sqlite3_stmt *, int);
    int (*column_bytes)(sqlite3_stmt *, int); const void *(*column_blob)(sqlite3_stmt *, int);
    int (*bind_int64)(sqlite3_stmt *, int, long long); int (*bind_double)(sqlite3_stmt *, int, double);
    int (*bind_text)(sqlite3_stmt *, int, const char *, int, void (*)(void *)); int (*bind_null)(sqlite3_stmt *, int);
    int (*bind_blob)(sqlite3_stmt *, int, const void *, int, void (*)(void *));
    int (*bind_parameter_count)(sqlite3_stmt *);
    const char *(*errmsg)(sqlite3 *); int (*changes)(sqlite3 *);
    int (*exec)(sqlite3 *, const char *, void *, void *, char **); void (*free_)(void *);
    int (*busy_timeout)(sqlite3 *, int);
} sq;

static void *sq_try(const char *p) {
#ifdef _WIN32
    return (void *)LoadLibraryA(p);
#else
    return dlopen(p, RTLD_NOW | RTLD_LOCAL);
#endif
}
static void *sq_sym(const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress((HMODULE)sq.lib, name);
#else
    return dlsym(sq.lib, name);
#endif
}
static int sq_load(void) {
    if (sq.state) return sq.state > 0;
    sq.state = -1;
    const char *env = getenv("APSI_SQLITE");
#ifdef _WIN32
    const char *cands[] = {env, "winsqlite3.dll", "sqlite3.dll", NULL};
#elif defined(__APPLE__)
    const char *cands[] = {env, "libsqlite3.dylib", "/usr/lib/libsqlite3.dylib", "libsqlite3.0.dylib", NULL};
#else
    const char *cands[] = {env, "libsqlite3.so.0", "libsqlite3.so", NULL};
#endif
    for (int i = 0; i < (int)(sizeof cands / sizeof *cands); i++) {
        if (i > 0 && !cands[i]) break;
        if (!cands[i] || !*cands[i]) continue;
        if ((sq.lib = sq_try(cands[i]))) { snprintf(sq.path, sizeof sq.path, "%s", cands[i]); break; }
    }
    if (!sq.lib) return 0;
    #define SQ(f, n) *(void **)&sq.f = sq_sym("sqlite3_" n)
    SQ(open_v2, "open_v2"); SQ(close_v2, "close_v2"); SQ(prepare_v2, "prepare_v2"); SQ(step, "step"); SQ(finalize, "finalize");
    SQ(column_count, "column_count"); SQ(column_name, "column_name"); SQ(column_type, "column_type");
    SQ(column_int64, "column_int64"); SQ(column_double, "column_double"); SQ(column_text, "column_text");
    SQ(column_bytes, "column_bytes"); SQ(column_blob, "column_blob"); SQ(bind_int64, "bind_int64");
    SQ(bind_double, "bind_double"); SQ(bind_text, "bind_text"); SQ(bind_null, "bind_null"); SQ(bind_blob, "bind_blob");
    SQ(bind_parameter_count, "bind_parameter_count"); SQ(errmsg, "errmsg"); SQ(changes, "changes"); SQ(exec, "exec");
    SQ(free_, "free"); SQ(busy_timeout, "busy_timeout");
    #undef SQ
    if (!sq.open_v2 || !sq.prepare_v2 || !sq.step || !sq.finalize || !sq.column_text || !sq.bind_text || !sq.exec || !sq.errmsg) return 0;
    sq.state = 1; return 1;
}
static AV sq_err(const char *msg) { AV parts[2] = {a_str("error"), a_str(msg ? msg : "SQLite error")}; return a_tuple(2, parts); }
static sqlite3 *sq_handle(AV h) { if (h.t != T_INT || !h.i) a_error("ValueError", "this SQLite connection is closed"); return (sqlite3 *)(intptr_t)h.i; }

/* the statements Python's sqlite3 counts rows for */
static int sq_is_dml(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || *s == '(') s++;
    const char *kw[] = {"insert", "update", "delete", "replace"};
    for (int i = 0; i < 4; i++) { size_t n = strlen(kw[i]); int ok = 1; for (size_t j = 0; j < n; j++) { char c = s[j]; if (c >= 'A' && c <= 'Z') c = (char)(c + 32); if (c != kw[i][j]) { ok = 0; break; } } if (ok) return 1; }
    return 0;
}

static AV sq_query(AV hv, AV sqlv, AV params) {
    sqlite3 *db = sq_handle(hv); const char *sql = ((AStr *)sqlv.o)->s;
    sqlite3_stmt *st = NULL; const char *tail = NULL;
    if (sq.prepare_v2(db, sql, -1, &st, &tail) != 0) return sq_err(sq.errmsg(db));
    if (tail) { const char *t = tail; while (*t == ' ' || *t == '\t' || *t == '\n' || *t == '\r' || *t == ';') t++; if (*t) { sq.finalize(st); return sq_err("You can only execute one statement at a time."); } }
    if (!st) { AV parts[4] = {a_list(0, NULL), a_list(0, NULL), a_int(-1), a_bool(0)}; return a_tuple(4, parts); }
    AList *pl = a_is(params, O_LIST) || a_is(params, O_TUPLE) ? (AList *)params.o : NULL;
    int np = sq.bind_parameter_count ? sq.bind_parameter_count(st) : 0;
    if (pl && pl->len != np) {
        sq.finalize(st); char m[160];
        snprintf(m, sizeof m, "Incorrect number of bindings supplied. The current statement uses %d, and there are %lld supplied.", np, (long long)pl->len);
        return sq_err(m);
    }
    for (int i = 0; pl && i < np; i++) {
        AV v = pl->v[i]; int rc = 0;
        if (v.t == T_NONE) rc = sq.bind_null(st, i + 1);
        else if (v.t == T_INT || v.t == T_BOOL) rc = sq.bind_int64(st, i + 1, (long long)v.i);
        else if (v.t == T_FLOAT) rc = sq.bind_double(st, i + 1, v.f);
        else if (a_is(v, O_BYTEARRAY) && sq.bind_blob) rc = sq.bind_blob(st, i + 1, ((AStr *)((ABArr *)v.o)->data.o)->s, (int)((AStr *)((ABArr *)v.o)->data.o)->len, (void (*)(void *))(intptr_t)-1);
        else if (a_is(v, O_BYTES) && sq.bind_blob) rc = sq.bind_blob(st, i + 1, ((AStr *)v.o)->s, (int)((AStr *)v.o)->len, (void (*)(void *))(intptr_t)-1);
        else if (a_is(v, O_STR)) rc = sq.bind_text(st, i + 1, ((AStr *)v.o)->s, (int)((AStr *)v.o)->len, (void (*)(void *))(intptr_t)-1);
        else { AV s = a_to_str(v); rc = sq.bind_text(st, i + 1, ((AStr *)s.o)->s, (int)((AStr *)s.o)->len, (void (*)(void *))(intptr_t)-1); }
        if (rc) { sq.finalize(st); return sq_err(sq.errmsg(db)); }
    }
    int nc = sq.column_count(st), sp = a_ntmp;
    AV cols = a_list(0, NULL); A_PUSH(cols); AV rows = a_list(0, NULL); A_PUSH(rows);
    for (int j = 0; j < nc; j++) a_list_append((AList *)cols.o, a_str(sq.column_name(st, j)));
    for (;;) {
        int rc = sq.step(st);
        if (rc == 101) break;                                   /* SQLITE_DONE */
        if (rc != 100) { AV e = sq_err(sq.errmsg(db)); sq.finalize(st); a_ntmp = sp; return e; }   /* not SQLITE_ROW */
        if (!nc) continue;
        AV row = a_list(0, NULL); A_PUSH(row);
        for (int j = 0; j < nc; j++) {
            int t = sq.column_type(st, j); AV v;
            if (t == 1) v = a_int(sq.column_int64(st, j));
            else if (t == 2) v = a_float(sq.column_double(st, j));
            else if (t == 5) v = A_NONE;
            else if (t == 4) { const void *s = sq.column_blob(st, j); int n = sq.column_bytes(st, j); v = a_bytes(s ? (const char *)s : "", n); }
            else { const unsigned char *s = sq.column_text(st, j); int n = sq.column_bytes(st, j); v = a_strn(s ? (const char *)s : "", n); }
            a_list_append((AList *)row.o, v);
        }
        a_list_append((AList *)rows.o, row); a_ntmp--;
    }
    sq.finalize(st);
    int dml = sq_is_dml(sql);
    AV parts[4] = {cols, rows, a_int(dml && sq.changes ? sq.changes(db) : -1), a_bool(dml)};
    AV r = a_tuple(4, parts); a_ntmp = sp; return r;
}

BI(lb_sqlite) {
    (void)self; (void)kw; const char *op = lib_str(a[0], "sqlite")->s;
    if (!strcmp(op, "library")) return sq_load() ? a_str(sq.path) : A_NONE;
    if (!sq_load()) return sq_err("no SQLite library was found on this computer (Windows 10 and later have winsqlite3.dll; "
                                  "on Linux install libsqlite3; APSI_SQLITE can name the library file)");
    if (!strcmp(op, "open")) {
        sqlite3 *db = NULL; const char *path = lib_str(a[1], "sqlite")->s;
        int rc = sq.open_v2(path, &db, 0x00000002 | 0x00000004 | 0x00010000, NULL);   /* READWRITE | CREATE | FULLMUTEX */
        if (rc) { AV e = sq_err(db ? sq.errmsg(db) : "cannot open the database"); if (db) sq.close_v2(db); return e; }
        if (sq.busy_timeout) sq.busy_timeout(db, 5000);
        return a_int((int64_t)(intptr_t)db);
    }
    if (!strcmp(op, "query")) return sq_query(a[1], a[2], n > 3 ? a[3] : A_NONE);
    if (!strcmp(op, "exec")) {
        char *err = NULL; int rc = sq.exec(sq_handle(a[1]), lib_str(a[2], "sqlite")->s, NULL, NULL, &err);
        if (rc) { AV e = sq_err(err ? err : sq.errmsg(sq_handle(a[1]))); if (err && sq.free_) sq.free_(err); return e; }
        return A_NONE;
    }
    if (!strcmp(op, "close")) { if (a[1].t == T_INT && a[1].i) sq.close_v2((sqlite3 *)(intptr_t)a[1].i); return A_NONE; }
    a_error("ValueError", "unknown sqlite operation '%s'", op);
}
