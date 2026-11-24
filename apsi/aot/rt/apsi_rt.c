/* Apsi native runtime (see apsi_rt.h). */
#include "apsi_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdbool.h>

/* =================================================================== memory: mark and sweep at safe points */
AFrame *a_frame = NULL;
static AObj *a_objects = NULL;
static size_t a_allocated = 0, a_threshold = 64u << 20;
typedef struct ARoots { AV *slots; int n; struct ARoots *next; } ARoots;
static ARoots *a_roots = NULL;
AHandler *a_handler = NULL;
AV a_tmp[65536];
int a_ntmp = 0;
void a_overflow(void) { fprintf(stderr, "RecursionError: runtime stack overflow\n"); exit(1); }
AV a_exc;
const char *a_file = "", *a_source = "";

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "MemoryError: out of memory\n"); exit(3); }
    return p;
}
static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) { fprintf(stderr, "MemoryError: out of memory\n"); exit(3); }
    return q;
}

void *a_alloc(size_t size, AKind kind) {
    AObj *o = (AObj *)xmalloc(size);
    memset(o, 0, size);
    o->kind = (uint8_t)kind;
    o->gcnext = a_objects;
    a_objects = o;
    a_allocated += size;
    return o;
}

void a_note_alloc(size_t bytes) { a_allocated += bytes; }
void a_root(AV *slots, int n) {
    ARoots *r = (ARoots *)xmalloc(sizeof(ARoots));
    r->slots = slots; r->n = n; r->next = a_roots; a_roots = r;
}

/* the class objects of the built-in exceptions (kept alive as roots) */
#define A_NEXC 34
static AV a_exc_classes[A_NEXC];
static const char *a_exc_names[A_NEXC] = {
    "BaseException", "Exception", "ArithmeticError", "ZeroDivisionError", "OverflowError", "ValueError", "TypeError",
    "LookupError", "IndexError", "KeyError", "AttributeError", "NameError", "RuntimeError", "NotImplementedError",
    "StopIteration", "AssertionError", "UnitError", "OSError", "RecursionError", "KeyboardInterrupt",
    "FloatingPointError", "FileNotFoundError", "FileExistsError", "PermissionError", "TimeoutError", "ConnectionError",
    "UnicodeError", "MemoryError", "EOFError", "ImportError", "ShapeError", "UserError", "IsADirectoryError",
    "ModuleNotFoundError"};
static const int a_exc_base[A_NEXC] = {-1, 0, 1, 2, 2, 1, 1, 1, 7, 7, 1, 1, 1, 12, 1, 1, 5, 1, 12, 0,
    2, 17, 17, 17, 17, 17, 5, 1, 1, 1, 5, 12, 17, 29};

static void mark(AV v);
struct AGen; static void gen_mark(struct AGen *g); static void gen_free(struct AGen *g); static const char *gen_name(AObj *o);
static void mark_obj(AObj *o) {
    while (o && !o->mark) {
        o->mark = 1;
        switch (o->kind) {
        case O_LIST: case O_TUPLE: {
            AList *l = (AList *)o;
            for (int64_t i = 0; i < l->len; i++) mark(l->v[i]);
            return;
        }
        case O_DICT: case O_SET: {
            ADict *d = (ADict *)o;
            for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) { mark(d->e[i].k); mark(d->e[i].v); }
            return;
        }
        case O_FUNC: {
            AFunc *f = (AFunc *)o;
            for (int i = 0; i < f->ncells; i++) if (f->cells[i]) mark_obj((AObj *)f->cells[i]);
            for (int i = 0; i < f->ndef; i++) mark(f->defaults[i]);
            return;
        }
        case O_CELL: mark(((ACell *)o)->v); return;
        case O_CLASS: {
            AClass *c = (AClass *)o;
            mark_obj((AObj *)c->name);
            for (int i = 0; i < c->nfields; i++) mark_obj((AObj *)c->fields[i]);
            mark_obj((AObj *)c->attrs);
            o = (AObj *)c->base;
            continue;
        }
        case O_INST: { AInst *s = (AInst *)o; mark_obj((AObj *)s->cls); o = (AObj *)s->attrs; continue; }
        case O_BOUND: { ABound *b = (ABound *)o; mark(b->self); mark(b->fn); return; }
        case O_BUILTIN: mark(((ABuiltin *)o)->self); return;
        case O_ITER: { AIter *it = (AIter *)o; mark(it->src); mark(it->aux); return; }
        case O_SLICE: { ASlice *s = (ASlice *)o; mark(s->lo); mark(s->hi); mark(s->step); return; }
        case O_QTY: o = (AObj *)((AQty *)o)->dtext; continue;
        case O_GEN: gen_mark((struct AGen *)o); return;
        case O_NODE: a_tensor_mark(o); return;
        default: return;
        }
    }
}
static void mark(AV v) { if (v.t == T_OBJ && v.o) mark_obj(v.o); }
void a_mark_value(AV v) { mark(v); }

static void free_obj(AObj *o) {
    switch (o->kind) {
    case O_LIST: case O_TUPLE: free(((AList *)o)->v); break;
    case O_DICT: case O_SET: free(((ADict *)o)->e); free(((ADict *)o)->idx); break;
    case O_FUNC: free(((AFunc *)o)->cells); free(((AFunc *)o)->defaults); break;
    case O_CLASS: free(((AClass *)o)->fields); break;
    case O_GEN: gen_free((struct AGen *)o); break;
    case O_TENSOR: a_tensor_free(o); break;
    default: break;
    }
    free(o);
}

static void a_collect(void) {
    for (AFrame *f = a_frame; f; f = f->prev) for (int i = 0; i < f->n; i++) mark(f->slots[i]);
    for (ARoots *r = a_roots; r; r = r->next) for (int i = 0; i < r->n; i++) mark(r->slots[i]);
    for (int i = 0; i < A_NEXC; i++) mark(a_exc_classes[i]);
    mark(a_exc);
    for (int i = 0; i < a_ntmp; i++) mark(a_tmp[i]);
    AObj **pp = &a_objects;
    size_t live = 0;
    while (*pp) {
        AObj *o = *pp;
        if (o->mark) { o->mark = 0; live += 64; pp = &o->gcnext; }
        else { *pp = o->gcnext; free_obj(o); }
    }
    a_allocated = 0;
    if (a_threshold < live * 2) a_threshold = live * 2;
}
void a_gc_full(void) { a_collect(); }
extern size_t a_dev_since, a_dev_limit;
void a_gc_safe(void) { if (a_allocated > a_threshold || a_dev_since > a_dev_limit) { a_collect(); a_dev_since = 0; } }

static _Noreturn void a_key_error(AV k);
#define QTY(v) ((AQty *)(v).o)
static int same_dim(const int16_t *a, const int16_t *b);
static int is_zero_num(AV v);
struct SB;
static void qty_text(struct SB *b, AQty *q, const char *spec);
static AV qty_binop(int op, AV a, AV b);
static double qty_same(AV a, AV b, const char *verb, int *ok);
static int num(AV v); static double fval(AV v); static const char *tname(AV v); static const char *opname(int op);
/* =================================================================== strings */
static uint64_t fnv(const char *s, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (int64_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h ? h : 1;
}
static int64_t utf8_count(const char *s, int64_t n, int *ascii) {
    int64_t c = 0; *ascii = 1;
    for (int64_t i = 0; i < n; i++) { unsigned char b = (unsigned char)s[i]; if (b >= 0x80) *ascii = 0; if ((b & 0xC0) != 0x80) c++; }
    return c;
}
AV a_strn(const char *s, int64_t n) {
    AStr *o = (AStr *)a_alloc(sizeof(AStr) + (size_t)n + 1, O_STR);
    memcpy(o->s, s, (size_t)n); o->s[n] = 0; o->len = n;
    o->nchars = utf8_count(o->s, n, &o->ascii);
    return a_obj(o);
}
AV a_str(const char *s) { return a_strn(s, (int64_t)strlen(s)); }
#define STR(v) ((AStr *)(v).o)
#define LIST(v) ((AList *)(v).o)
#define DICT(v) ((ADict *)(v).o)

/* byte offset of code point i */
static int64_t str_off(AStr *s, int64_t i) {
    if (s->ascii) return i;
    int64_t c = 0;
    for (int64_t b = 0; b < s->len; b++) {
        if ((((unsigned char)s->s[b]) & 0xC0) != 0x80) { if (c == i) return b; c++; }
    }
    return s->len;
}

typedef struct SB { char *p; int64_t n, cap; } SB;
static void sb_put(SB *b, const char *s, int64_t n) {
    if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2 + 16; b->p = (char *)xrealloc(b->p, (size_t)b->cap); }
    memcpy(b->p + b->n, s, (size_t)n); b->n += n; b->p[b->n] = 0;
}
static void sb_puts(SB *b, const char *s) { sb_put(b, s, (int64_t)strlen(s)); }
static AV sb_done(SB *b) { AV r = a_strn(b->p ? b->p : "", b->n); free(b->p); return r; }

/* =================================================================== containers */
static AList *new_list(int64_t n, AKind kind) {
    AList *l = (AList *)a_alloc(sizeof(AList), kind);
    l->cap = n > 4 ? n : 4; l->len = n;
    l->v = (AV *)xmalloc(sizeof(AV) * (size_t)l->cap);
    a_allocated += sizeof(AV) * (size_t)l->cap;
    return l;
}
AV a_list(int64_t n, AV *items) {
    AList *l = new_list(n, O_LIST);
    if (n) memcpy(l->v, items, sizeof(AV) * (size_t)n);
    return a_obj(l);
}
AV a_tuple(int64_t n, AV *items) {
    AList *l = new_list(n, O_TUPLE);
    if (n) memcpy(l->v, items, sizeof(AV) * (size_t)n);
    return a_obj(l);
}
void a_list_append(AList *l, AV v) {
    if (l->len == l->cap) { l->cap *= 2; l->v = (AV *)xrealloc(l->v, sizeof(AV) * (size_t)l->cap); a_allocated += sizeof(AV) * (size_t)l->len; }
    l->v[l->len++] = v;
}

static ADict *new_dict(AKind kind) {
    ADict *d = (ADict *)a_alloc(sizeof(ADict), kind);
    d->cap = 8; d->e = (ADEnt *)xmalloc(sizeof(ADEnt) * 8); memset(d->e, 0, sizeof(ADEnt) * 8);
    d->icap = 16; d->idx = (int64_t *)xmalloc(sizeof(int64_t) * 16);
    for (int i = 0; i < 16; i++) d->idx[i] = -1;
    return d;
}
AV a_dict(void) { return a_obj(new_dict(O_DICT)); }
AV a_set(void) { return a_obj(new_dict(O_SET)); }

static void dict_rehash(ADict *d) {
    /* compact the entries, then rebuild the index at twice the room */
    int64_t j = 0;
    for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) d->e[j++] = d->e[i];
    d->nent = j;
    while (d->icap < (d->len + 1) * 3) d->icap *= 2;
    d->idx = (int64_t *)xrealloc(d->idx, sizeof(int64_t) * (size_t)d->icap);
    for (int64_t i = 0; i < d->icap; i++) d->idx[i] = -1;
    for (int64_t i = 0; i < d->nent; i++) {
        uint64_t p = d->e[i].hash & (uint64_t)(d->icap - 1);
        while (d->idx[p] >= 0) p = (p + 1) & (uint64_t)(d->icap - 1);
        d->idx[p] = i;
    }
}
static int64_t dict_find(ADict *d, AV k, uint64_t h, uint64_t *slot) {
    uint64_t p = h & (uint64_t)(d->icap - 1);
    for (;;) {
        int64_t i = d->idx[p];
        if (i == -1) { if (slot) *slot = p; return -1; }
        if (i >= 0 && d->e[i].used && d->e[i].hash == h && a_eq(d->e[i].k, k)) { if (slot) *slot = p; return i; }
        p = (p + 1) & (uint64_t)(d->icap - 1);
    }
}
void a_dict_set(ADict *d, AV k, AV v) {
    uint64_t h = a_hash(k), slot;
    int64_t i = dict_find(d, k, h, &slot);
    if (i >= 0) { d->e[i].v = v; return; }
    if ((d->nent + 1) * 3 > d->icap * 2 || d->nent + 1 > d->cap) {
        if (d->nent + 1 > d->cap) { d->cap *= 2; d->e = (ADEnt *)xrealloc(d->e, sizeof(ADEnt) * (size_t)d->cap); }
        dict_rehash(d);
        dict_find(d, k, h, &slot);
    }
    i = d->nent++;
    d->e[i].k = k; d->e[i].v = v; d->e[i].hash = h; d->e[i].used = 1;
    d->idx[slot] = i; d->len++;
}
int a_dict_get(ADict *d, AV k, AV *out) {
    int64_t i = dict_find(d, k, a_hash(k), NULL);
    if (i < 0) return 0;
    if (out) *out = d->e[i].v;
    return 1;
}
int a_dict_del(ADict *d, AV k) {
    uint64_t slot;
    int64_t i = dict_find(d, k, a_hash(k), &slot);
    if (i < 0) return 0;
    d->e[i].used = 0; d->idx[slot] = -2; d->len--;
    return 1;
}
static AV dict_get_str(ADict *d, const char *name) {
    AV k = a_str(name), v;
    if (a_dict_get(d, k, &v)) return v;
    AV none; none.t = 255; none.i = 0; return none;
}

AV a_range(int64_t a, int64_t b, int64_t step) {
    ARange *r = (ARange *)a_alloc(sizeof(ARange), O_RANGE);
    if (step == 0) a_error("ValueError", "range() step must not be zero");
    r->start = a; r->stop = b; r->step = step;
    return a_obj(r);
}
AV a_slice(AV lo, AV hi, AV step) {
    ASlice *s = (ASlice *)a_alloc(sizeof(ASlice), O_SLICE);
    s->lo = lo; s->hi = hi; s->step = step;
    return a_obj(s);
}
AV a_func(const char *name, AFnCode code, int ncells, int ndef) {
    AFunc *f = (AFunc *)a_alloc(sizeof(AFunc), O_FUNC);
    f->name = name; f->code = code; f->ncells = ncells; f->ndef = ndef;
    f->cells = ncells ? (ACell **)xmalloc(sizeof(ACell *) * (size_t)ncells) : NULL;
    for (int i = 0; i < ncells; i++) f->cells[i] = NULL;
    f->defaults = ndef ? (AV *)xmalloc(sizeof(AV) * (size_t)ndef) : NULL;
    for (int i = 0; i < ndef; i++) f->defaults[i] = A_NONE;
    return a_obj(f);
}
ACell *a_cell(AV v) { ACell *c = (ACell *)a_alloc(sizeof(ACell), O_CELL); c->v = v; return c; }
AV a_builtin(const char *name, ABuiltinFn fn, AV self) {
    ABuiltin *b = (ABuiltin *)a_alloc(sizeof(ABuiltin), O_BUILTIN);
    b->name = name; b->fn = fn; b->self = self;
    return a_obj(b);
}
AV a_class(const char *name, AV base, int is_struct, int nfields, const char **fields) {
    AClass *c = (AClass *)a_alloc(sizeof(AClass), O_CLASS);
    c->name = STR(a_str(name)); c->base = a_is(base, O_CLASS) ? (AClass *)base.o : NULL;
    c->attrs = new_dict(O_DICT); c->is_struct = is_struct; c->nfields = nfields;
    c->fields = nfields ? (AStr **)xmalloc(sizeof(AStr *) * (size_t)nfields) : NULL;
    for (int i = 0; i < nfields; i++) c->fields[i] = STR(a_str(fields[i]));
    return a_obj(c);
}
static AV new_inst(AClass *c) {
    AInst *s = (AInst *)a_alloc(sizeof(AInst), O_INST);
    s->cls = c; s->attrs = new_dict(O_DICT);
    return a_obj(s);
}
static AV a_bound(AV self, AV fn) {
    ABound *b = (ABound *)a_alloc(sizeof(ABound), O_BOUND);
    b->self = self; b->fn = fn;
    return a_obj(b);
}

/* =================================================================== exceptions */
static int exc_index(const char *type) {
    for (int i = 0; i < A_NEXC; i++) if (!strcmp(a_exc_names[i], type)) return i;
    return 12;   /* RuntimeError */
}
static AStr *exc_kind(AClass *c) {            /* the nearest built-in kind; Exception, Error and a raised text: RuntimeError */
    for (; c; c = c->base)
        for (int i = 0; i < A_NEXC; i++) if ((AObj *)c == a_exc_classes[i].o) {
            if (i == 0 || i == 1 || i == 31) return ((AClass *)a_exc_classes[12].o)->name;
            return c->name;
        }
    return ((AClass *)a_exc_classes[12].o)->name;
}
static AV make_exc(AV cls, AV msg) {
    AV e = new_inst((AClass *)cls.o);
    AInst *s = (AInst *)e.o;
    AV args = a_tuple(1, &msg);
    a_dict_set(s->attrs, a_str("args"), args);
    a_dict_set(s->attrs, a_str("message"), msg);
    a_dict_set(s->attrs, a_str("kind"), a_obj(exc_kind(s->cls)));
    return e;
}
static void print_source_line(int line, int col) {
    const char *p = a_source; int l = 1;
    while (*p && l < line) { if (*p == '\n') l++; p++; }
    if (!*p) return;
    const char *e = strchr(p, '\n'); int n = e ? (int)(e - p) : (int)strlen(p);
    fprintf(stderr, "    %.*s\n    %*s^\n", n, p, col > 1 ? col - 1 : 0, "");
}
_Noreturn void a_raise(AV exc) {
    if (a_is(exc, O_CLASS)) exc = make_exc(exc, a_str(""));
    if (a_is(exc, O_STR)) exc = make_exc(a_exc_classes[31], exc);     /* raise "message" */
    a_exc = exc;
    if (!a_handler) {
        AV msg = a_to_str(exc);
        AInst *s = a_is(exc, O_INST) ? (AInst *)exc.o : NULL;
        int pos = a_frame ? a_frame->pos : 0, line = pos >> 12, col = pos & 4095;
        fflush(stdout);
        fprintf(stderr, "%s:%d:%d: RuntimeError: %s: %s\n", a_file, line, col, s ? s->cls->name->s : "Exception", STR(msg)->s);
        print_source_line(line, col);
        for (AFrame *f = a_frame; f; f = f->prev) if (f->name && strcmp(f->name, "<module>")) fprintf(stderr, "  in %s()\n", f->name);
        exit(1);
    }
    AHandler *h = a_handler;
    a_handler = h->prev;
    a_frame = h->frame;
    a_ntmp = h->ntmp;
    longjmp(h->jb, 1);
}
_Noreturn void a_error(const char *type, const char *fmt, ...) {
    char buf[1024];
    va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    a_raise(make_exc(a_exc_classes[exc_index(type)], a_str(buf)));
}
int a_isinstance(AV v, AV cls) {
    if (a_is(cls, O_TUPLE)) {
        for (int64_t i = 0; i < LIST(cls)->len; i++) if (a_isinstance(v, LIST(cls)->v[i])) return 1;
        return 0;
    }
    if (a_is(cls, O_BUILTIN)) {               /* int, float, str, list, dict, ... */
        const char *n = ((ABuiltin *)cls.o)->name;
        if (!strcmp(n, "int")) return v.t == T_INT || v.t == T_BOOL;
        if (!strcmp(n, "float")) return v.t == T_FLOAT;
        if (!strcmp(n, "bool")) return v.t == T_BOOL;
        if (!strcmp(n, "str")) return a_is(v, O_STR);
        if (!strcmp(n, "list")) return a_is(v, O_LIST);
        if (!strcmp(n, "tuple")) return a_is(v, O_TUPLE);
        if (!strcmp(n, "dict")) return a_is(v, O_DICT);
        if (!strcmp(n, "set")) return a_is(v, O_SET);
        return 0;
    }
    if (!a_is(cls, O_CLASS) || !a_is(v, O_INST)) return 0;
    for (AClass *c = ((AInst *)v.o)->cls; c; c = c->base) if ((AObj *)c == cls.o) return 1;
    return 0;
}
int a_exc_matches(AV exc, AV cls) { return a_isinstance(exc, cls); }

/* =================================================================== hashing, equality, truth */
uint64_t a_hash(AV v) {
    switch (v.t) {
    case T_NONE: return 0x9e3779b97f4a7c15ull;
    case T_BOOL: case T_INT: return (uint64_t)v.i * 0x9e3779b97f4a7c15ull + 1;
    case T_FLOAT: {
        if (v.f == floor(v.f) && fabs(v.f) < 9.2e18) return (uint64_t)(int64_t)v.f * 0x9e3779b97f4a7c15ull + 1;
        uint64_t b; memcpy(&b, &v.f, 8); return b * 0xff51afd7ed558ccdull + 7;
    }
    default: break;
    }
    switch (v.o->kind) {
    case O_STR: { AStr *s = STR(v); if (!s->hash) s->hash = fnv(s->s, s->len); return s->hash; }
    case O_TUPLE: {
        uint64_t h = 0x345678;
        for (int64_t i = 0; i < LIST(v)->len; i++) h = (h ^ a_hash(LIST(v)->v[i])) * 1000003ull;
        return h;
    }
    case O_LIST: case O_DICT: case O_SET: a_error("TypeError", "unhashable type");
    default: return (uint64_t)(uintptr_t)v.o * 0x9e3779b97f4a7c15ull;
    }
}
static int num(AV v) { return v.t == T_INT || v.t == T_BOOL || v.t == T_FLOAT; }
static double fval(AV v) { return v.t == T_FLOAT ? v.f : (double)v.i; }
int a_eq(AV a, AV b) {
    if (a_is_tn(a) || a_is_tn(b)) return a_tensor_eq(a, b);
    if (a_is(a, O_QTY) != a_is(b, O_QTY)) {
        AV q = a_is(a, O_QTY) ? a : b, o = a_is(a, O_QTY) ? b : a;
        return is_zero_num(o) && ((AQty *)q.o)->v == 0.0;
    }
    if (num(a) && num(b)) {
        if (a.t == T_FLOAT || b.t == T_FLOAT) return fval(a) == fval(b);
        return a.i == b.i;
    }
    if (a.t != b.t) return 0;
    if (a.t == T_NONE) return 1;
    if (a.o == b.o) return 1;
    if (a.o->kind != b.o->kind) {
        return 0;
    }
    switch (a.o->kind) {
    case O_STR: return STR(a)->len == STR(b)->len && !memcmp(STR(a)->s, STR(b)->s, (size_t)STR(a)->len);
    case O_LIST: case O_TUPLE: {
        if (LIST(a)->len != LIST(b)->len) return 0;
        for (int64_t i = 0; i < LIST(a)->len; i++) if (!a_eq(LIST(a)->v[i], LIST(b)->v[i])) return 0;
        return 1;
    }
    case O_DICT: case O_SET: {
        if (DICT(a)->len != DICT(b)->len) return 0;
        for (int64_t i = 0; i < DICT(a)->nent; i++) {
            ADEnt *e = &DICT(a)->e[i]; AV w;
            if (!e->used) continue;
            if (!a_dict_get(DICT(b), e->k, &w)) return 0;
            if (a.o->kind == O_DICT && !a_eq(e->v, w)) return 0;
        }
        return 1;
    }
    case O_INST: {
        AV eq = a_getattr(a, "__eq__");
        if (eq.t != 255) { AV r = a_call(eq, 1, &b, NULL); return a_truthy(r); }
        return 0;
    }
    case O_RANGE: { ARange *x = (ARange *)a.o, *y = (ARange *)b.o; return x->start == y->start && x->stop == y->stop && x->step == y->step; }
    case O_QTY: return same_dim(QTY(a)->dim, QTY(b)->dim) && QTY(a)->v == QTY(b)->v;
    default: return 0;
    }
}
int a_truthy(AV v) {
    if (a_is_tn(v)) return a_tensor_truthy(v);
    switch (v.t) {
    case T_NONE: return 0;
    case T_BOOL: case T_INT: return v.i != 0;
    case T_FLOAT: return v.f != 0.0;
    default: break;
    }
    switch (v.o->kind) {
    case O_STR: return STR(v)->len != 0;
    case O_LIST: case O_TUPLE: return LIST(v)->len != 0;
    case O_DICT: case O_SET: return DICT(v)->len != 0;
    case O_RANGE: return a_len(v) != 0;
    case O_QTY: return ((AQty *)v.o)->v != 0.0;
    case O_INST: {
        AV f = a_getattr(v, "__bool__");
        if (f.t != 255) return a_truthy(a_call(f, 0, NULL, NULL));
        f = a_getattr(v, "__len__");
        if (f.t != 255) return a_truthy(a_call(f, 0, NULL, NULL));
        return 1;
    }
    default: return 1;
    }
}

/* =================================================================== numbers to text */
static void float_repr(double f, char *out, size_t n) {
    if (isnan(f)) { snprintf(out, n, "nan"); return; }
    if (isinf(f)) { snprintf(out, n, f > 0 ? "inf" : "-inf"); return; }
    if (f == 0.0) { snprintf(out, n, signbit(f) ? "-0.0" : "0.0"); return; }
    char e[64]; int p;
    for (p = 0; p < 17; p++) { snprintf(e, sizeof e, "%.*e", p, f); if (strtod(e, NULL) == f) break; }
    /* e = [-]d.ddddde[+-]XX : the shortest digits and the exponent, laid out the way Python's repr does */
    char digs[32]; int nd = 0, neg = e[0] == '-';
    char *q = e + neg;
    for (; *q && *q != 'e'; q++) if (*q != '.') digs[nd++] = *q;
    digs[nd] = 0;
    int ex = atoi(q + 1);
    while (nd > 1 && digs[nd - 1] == '0') digs[--nd] = 0;
    char *o = out; size_t left = n;
    #define PUT(...) do { int w = snprintf(o, left, __VA_ARGS__); o += w; left -= (size_t)w; } while (0)
    if (neg) PUT("-");
    if (ex >= 16 || ex < -4) {
        PUT("%c", digs[0]);
        if (nd > 1) PUT(".%s", digs + 1);
        PUT("e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    } else if (ex < 0) {
        PUT("0.");
        for (int i = 0; i < -ex - 1; i++) PUT("0");
        PUT("%s", digs);
    } else {
        for (int i = 0; i <= ex; i++) PUT("%c", i < nd ? digs[i] : '0');
        PUT(".");
        if (nd > ex + 1) PUT("%s", digs + ex + 1); else PUT("0");
    }
    #undef PUT
}

/* =================================================================== str() and repr() */
static void repr_into(SB *b, AV v, int repr);
static void str_repr(SB *b, AStr *s) {          /* Apsi shows a string inside a container in double quotes */
    sb_put(b, "\"", 1);
    for (int64_t i = 0; i < s->len; i++) {
        char c = s->s[i];
        if (c == '"') sb_put(b, "\\\"", 2); else sb_put(b, &c, 1);
    }
    sb_put(b, "\"", 1);
}
static void py_str_repr(SB *b, AStr *s) {       /* Python's repr, for the messages Python itself would write */
    int dq = memchr(s->s, '\'', (size_t)s->len) && !memchr(s->s, '"', (size_t)s->len);
    char q = dq ? '"' : '\'';
    sb_put(b, &q, 1);
    for (int64_t i = 0; i < s->len; i++) {
        char c = s->s[i];
        if (c == q || c == '\\') { sb_put(b, "\\", 1); sb_put(b, &c, 1); }
        else if (c == '\n') sb_puts(b, "\\n");
        else if (c == '\t') sb_puts(b, "\\t");
        else sb_put(b, &c, 1);
    }
    sb_put(b, &q, 1);
}
static void repr_into(SB *b, AV v, int repr) {
    char t[64];
    switch (v.t) {
    case T_NONE: sb_puts(b, "none"); return;
    case T_BOOL: sb_puts(b, v.i ? "true" : "false"); return;
    case T_INT: snprintf(t, sizeof t, "%lld", (long long)v.i); sb_puts(b, t); return;
    case T_FLOAT: float_repr(v.f, t, sizeof t); sb_puts(b, t); return;
    default: break;
    }
    switch (v.o->kind) {
    case O_STR: if (repr) str_repr(b, STR(v)); else sb_put(b, STR(v)->s, STR(v)->len); return;
    case O_LIST: case O_TUPLE: {
        int tup = v.o->kind == O_TUPLE;
        sb_puts(b, tup ? "(" : "[");
        for (int64_t i = 0; i < LIST(v)->len; i++) { if (i) sb_puts(b, ", "); repr_into(b, LIST(v)->v[i], 1); }
        if (tup && LIST(v)->len == 1) sb_puts(b, ",");
        sb_puts(b, tup ? ")" : "]");
        return;
    }
    case O_DICT: case O_SET: {
        int set = v.o->kind == O_SET, first = 1;
        if (set && DICT(v)->len == 0) { sb_puts(b, "set()"); return; }
        sb_puts(b, "{");
        for (int64_t i = 0; i < DICT(v)->nent; i++) {
            ADEnt *e = &DICT(v)->e[i];
            if (!e->used) continue;
            if (!first) sb_puts(b, ", ");
            first = 0;
            repr_into(b, e->k, 1);
            if (!set) { sb_puts(b, ": "); repr_into(b, e->v, 1); }
        }
        sb_puts(b, "}");
        return;
    }
    case O_RANGE: {
        ARange *r = (ARange *)v.o;
        if (r->step == 1) snprintf(t, sizeof t, "%lld..%lld", (long long)r->start, (long long)r->stop);
        else snprintf(t, sizeof t, "range(%lld, %lld, %lld)", (long long)r->start, (long long)r->stop, (long long)r->step);
        sb_puts(b, t); return;
    }
    case O_FUNC: sb_puts(b, "<fn "); sb_puts(b, ((AFunc *)v.o)->name); sb_puts(b, ">"); return;
    case O_BUILTIN: sb_puts(b, "<builtin "); sb_puts(b, ((ABuiltin *)v.o)->name); sb_puts(b, ">"); return;
    case O_CLASS:
        if (((AClass *)v.o)->is_struct) { sb_puts(b, "<struct "); sb_puts(b, ((AClass *)v.o)->name->s); sb_puts(b, ">"); return; }
        sb_puts(b, "<class '__main__."); sb_puts(b, ((AClass *)v.o)->name->s); sb_puts(b, "'>"); return;
    case O_INST: {
        AInst *s = (AInst *)v.o;
        AV f = a_getattr(v, repr ? "__repr__" : "__str__");
        if (f.t == 255 && !repr) f = a_getattr(v, "__repr__");
        if (f.t != 255) { AV r = a_call(f, 0, NULL, NULL); if (a_is(r, O_STR)) { sb_put(b, STR(r)->s, STR(r)->len); return; } }
        AClass *ec = s->cls;
        int is_exc = 0;
        for (AClass *c = ec; c; c = c->base) if ((AObj *)c == a_exc_classes[0].o) is_exc = 1;
        if (is_exc) {
            AV m = dict_get_str(s->attrs, "message");
            if (repr) { sb_puts(b, ec->name->s); sb_puts(b, "("); if (m.t != 255 && a_is(m, O_STR)) py_str_repr(b, STR(m)); sb_puts(b, ")"); }
            else if (m.t != 255) repr_into(b, m, 0);
            return;
        }
        if (ec->is_struct == 2) {          /* record(name=value, ...) */
            sb_puts(b, "record("); int first = 1;
            for (int64_t i = 0; i < s->attrs->nent; i++) if (s->attrs->e[i].used) {
                if (!first) sb_puts(b, ", "); first = 0;
                repr_into(b, s->attrs->e[i].k, 0); sb_puts(b, "="); repr_into(b, s->attrs->e[i].v, 1);
            }
            sb_puts(b, ")"); return;
        }
        if (ec->is_struct) {               /* Name(field=value, ...), or its to_str() */
            AV ts = a_getattr(v, "to_str");
            if (ts.t != 255) { AV r = a_to_str(a_call(ts, 0, NULL, NULL)); sb_put(b, STR(r)->s, STR(r)->len); return; }
            sb_puts(b, ec->name->s); sb_puts(b, "(");
            for (int i = 0; i < ec->nfields; i++) {
                AV fv;
                if (i) sb_puts(b, ", ");
                sb_puts(b, ec->fields[i]->s); sb_puts(b, "=");
                if (a_dict_get(s->attrs, a_obj(ec->fields[i]), &fv)) repr_into(b, fv, 1);
            }
            sb_puts(b, ")"); return;
        }
        if (!strncmp(ec->name->s, "module:", 7)) { sb_puts(b, "<module "); sb_puts(b, ec->name->s + 7); sb_puts(b, ">"); return; }
        snprintf(t, sizeof t, " object at %p>", (void *)v.o);
        sb_puts(b, "<__main__."); sb_puts(b, ec->name->s); sb_puts(b, t); return;
    }
    case O_BOUND: sb_puts(b, "<bound method>"); return;
    case O_TENSOR: case O_NODE: { char *s = a_tensor_text(v, !repr); sb_puts(b, s); free(s); return; }
    case O_GEN: sb_puts(b, "<generator "); sb_puts(b, gen_name(v.o)); sb_puts(b, ">"); return;
    case O_QTY: qty_text(b, (AQty *)v.o, NULL); return;
    default: sb_puts(b, "<object>"); return;
    }
}
AV a_to_str(AV v) {
    if (a_is(v, O_STR)) return v;
    SB b = {0}; repr_into(&b, v, 0); return sb_done(&b);
}
AV a_repr(AV v) { SB b = {0}; repr_into(&b, v, 1); return sb_done(&b); }

AV a_format(AV v, const char *spec) {
    if (!spec || !*spec) return a_to_str(v);
    if (a_is(v, O_QTY)) { SB b = {0}; qty_text(&b, (AQty *)v.o, spec); return sb_done(&b); }
    /* [[fill]align][sign][0][width][,][.precision][type] */
    char fill = ' ', align = 0, sign = '-', type = 0; int width = -1, prec = -1, comma = 0, zero = 0;
    const char *p = spec;
    if (p[0] && (p[1] == '<' || p[1] == '>' || p[1] == '^' || p[1] == '=')) { fill = p[0]; align = p[1]; p += 2; }
    else if (*p == '<' || *p == '>' || *p == '^' || *p == '=') align = *p++;
    if (*p == '+' || *p == '-' || *p == ' ') sign = *p++;
    if (*p == '0') { zero = 1; p++; }
    if (*p >= '0' && *p <= '9') { width = 0; while (*p >= '0' && *p <= '9') width = width * 10 + (*p++ - '0'); }
    if (*p == ',' || *p == '_') { comma = 1; p++; }
    if (*p == '.') { p++; prec = 0; while (*p >= '0' && *p <= '9') prec = prec * 10 + (*p++ - '0'); }
    if (*p) type = *p;
    char body[512];
    if (num(v) && type != 's') {
        double f = fval(v);
        int isint = (v.t == T_INT || v.t == T_BOOL) && (type == 0 || type == 'd' || type == 'x' || type == 'X' || type == 'b' || type == 'o');
        if (isint) {
            long long i = (long long)v.i;
            if (type == 'x') snprintf(body, sizeof body, "%llx", i < 0 ? -i : i);
            else if (type == 'X') snprintf(body, sizeof body, "%llX", i < 0 ? -i : i);
            else if (type == 'o') snprintf(body, sizeof body, "%llo", i < 0 ? -i : i);
            else if (type == 'b') { char t[80]; int k = 0; unsigned long long u = (unsigned long long)(i < 0 ? -i : i); do { t[k++] = (char)('0' + (u & 1)); u >>= 1; } while (u); for (int j = 0; j < k; j++) body[j] = t[k - 1 - j]; body[k] = 0; }
            else snprintf(body, sizeof body, "%lld", i < 0 ? -i : i);
            if (comma) { char t[512]; int n = (int)strlen(body), o = 0; for (int j = 0; j < n; j++) { t[o++] = body[j]; if ((n - j - 1) % 3 == 0 && j < n - 1) t[o++] = ','; } t[o] = 0; strcpy(body, t); }
            if (i < 0) { memmove(body + 1, body, strlen(body) + 1); body[0] = '-'; }
            else if (sign == '+' || sign == ' ') { memmove(body + 1, body, strlen(body) + 1); body[0] = sign; }
        } else {
            char t = type ? type : (prec >= 0 ? 'g' : 0);
            if (t == 0) { float_repr(f, body, sizeof body); }
            else if (t == '%') { snprintf(body, sizeof body, "%.*f%%", prec < 0 ? 6 : prec, f * 100.0); }
            else {
                char fmt[16]; snprintf(fmt, sizeof fmt, "%%%s.*%c", comma ? "" : "", t == 'n' ? 'g' : t);
                snprintf(body, sizeof body, fmt, prec < 0 ? 6 : prec, f);
                if (comma) {
                    char *dot = strchr(body, '.'); int ip = dot ? (int)(dot - body) : (int)strlen(body);
                    int neg = body[0] == '-'; char t2[512]; int o = 0;
                    if (neg) t2[o++] = '-';
                    for (int j = neg; j < ip; j++) { t2[o++] = body[j]; if ((ip - j - 1) % 3 == 0 && j < ip - 1) t2[o++] = ','; }
                    t2[o] = 0; strcat(t2, body + ip); strcpy(body, t2);
                }
            }
            if (f >= 0 && (sign == '+' || sign == ' ')) { memmove(body + 1, body, strlen(body) + 1); body[0] = sign; }
        }
        if (!align) align = zero ? '=' : '>';
        if (zero && fill == ' ') fill = '0';
    } else {
        AV s = a_to_str(v);
        int64_t n = STR(s)->len;
        if (prec >= 0) n = str_off(STR(s), prec < STR(s)->nchars ? prec : STR(s)->nchars);
        if (n > (int64_t)sizeof body - 1) n = sizeof body - 1;
        memcpy(body, STR(s)->s, (size_t)n); body[n] = 0;
        if (!align) align = '<';
    }
    int bl; { int a2; bl = (int)utf8_count(body, (int64_t)strlen(body), &a2); }
    if (width <= bl) return a_str(body);
    SB b = {0};
    int pad = width - bl;
    if (align == '<') { sb_puts(&b, body); for (int i = 0; i < pad; i++) sb_put(&b, &fill, 1); }
    else if (align == '^') { for (int i = 0; i < pad / 2; i++) sb_put(&b, &fill, 1); sb_puts(&b, body); for (int i = 0; i < pad - pad / 2; i++) sb_put(&b, &fill, 1); }
    else if (align == '=' && (body[0] == '-' || body[0] == '+' || body[0] == ' ')) { sb_put(&b, body, 1); for (int i = 0; i < pad; i++) sb_put(&b, &fill, 1); sb_puts(&b, body + 1); }
    else { for (int i = 0; i < pad; i++) sb_put(&b, &fill, 1); sb_puts(&b, body); }
    return sb_done(&b);
}
AV a_concat_strs(int n, AV *parts) {
    SB b = {0};
    for (int i = 0; i < n; i++) { AV s = a_to_str(parts[i]); sb_put(&b, STR(s)->s, STR(s)->len); }
    return sb_done(&b);
}


/* =================================================================== quantities with units */
static const AUnitName *unit_names; static int n_unit_names;
void a_unit_names(const AUnitName *names, int n) { unit_names = names; n_unit_names = n; }
static int dimless(const int16_t *d) { for (int i = 0; i < 7; i++) if (d[i]) return 0; return 1; }
static int same_dim(const int16_t *a, const int16_t *b) { return !memcmp(a, b, sizeof(int16_t) * 7); }
AV a_qty(double v, const int16_t *dim, const char *text, double scale) {
    if (dimless(dim)) return a_float(text ? v / scale : v);
    AQty *q = (AQty *)a_alloc(sizeof(AQty), O_QTY);
    q->v = v; memcpy(q->dim, dim, sizeof q->dim);
    q->dtext = text ? STR(a_str(text)) : NULL; q->dscale = text ? scale : 1.0;
    return a_obj(q);
}
static void exp_str(SB *b, int e) {            /* an exponent in 60ths */
    char t[32];
    if (e % 60 == 0) snprintf(t, sizeof t, "%d", e / 60);
    else { int g = 60, x = e < 0 ? -e : e; while (x) { int r = g % x; g = x; x = r; } snprintf(t, sizeof t, "(%d/%d)", e / g, 60 / g); }
    sb_puts(b, t);
}
static void unit_string(SB *b, const int16_t *d) {
    for (int i = 0; i < n_unit_names; i++) if (same_dim(unit_names[i].dim, d)) { sb_puts(b, unit_names[i].name); return; }
    static const char *base[7] = {"m", "kg", "s", "A", "K", "mol", "cd"};
    int nnum = 0, nden = 0;
    for (int i = 0; i < 7; i++) { if (d[i] > 0) nnum++; else if (d[i] < 0) nden++; }
    if (!nnum) sb_puts(b, "1");
    int k = 0;
    for (int i = 0; i < 7; i++) if (d[i] > 0) { if (k++) sb_puts(b, "\302\267"); sb_puts(b, base[i]); if (d[i] != 60) { sb_puts(b, "^"); exp_str(b, d[i]); } }
    if (nden) {
        sb_puts(b, "/"); if (nden > 1) sb_puts(b, "(");
        k = 0;
        for (int i = 0; i < 7; i++) if (d[i] < 0) { if (k++) sb_puts(b, "\302\267"); sb_puts(b, base[i]); if (d[i] != -60) { sb_puts(b, "^"); exp_str(b, -d[i]); } }
        if (nden > 1) sb_puts(b, ")");
    }
}
static const char *dim_desc(const int16_t *d, char *buf, size_t n) {
    if (dimless(d)) { snprintf(buf, n, "a dimensionless number"); return buf; }
    SB b = {0}; unit_string(&b, d); snprintf(buf, n, "[%s]", b.p); free(b.p); return buf;
}
static void qty_text(SB *b, AQty *q, const char *spec) {
    double val = q->dtext ? q->v / q->dscale : q->v;
    AV s = spec && *spec ? a_format(a_float(val), spec) : a_to_str(a_float(val));
    sb_put(b, STR(s)->s, STR(s)->len); sb_puts(b, " ");
    if (q->dtext) sb_put(b, q->dtext->s, q->dtext->len); else unit_string(b, q->dim);
}
static int is_zero_num(AV v) { return (v.t == T_INT && v.i == 0) || (v.t == T_FLOAT && v.f == 0.0); }
static double qty_same(AV a, AV b, const char *verb, int *ok) {   /* b's value when a and b share a's dimension */
    AQty *q = QTY(a); char x[96], y[96];
    *ok = 1;
    if (a_is(b, O_QTY)) {
        if (!same_dim(q->dim, QTY(b)->dim)) a_error("UnitError", "cannot %s %s and %s", verb, dim_desc(q->dim, x, sizeof x), dim_desc(QTY(b)->dim, y, sizeof y));
        return QTY(b)->v;
    }
    if (is_zero_num(b)) return 0.0;
    a_error("UnitError", "cannot %s %s and a dimensionless number", verb, dim_desc(q->dim, x, sizeof x));
}
static AV qty_binop(int op, AV a, AV b) {
    int16_t d[7]; int ok;
    int qa = a_is(a, O_QTY), qb = a_is(b, O_QTY);
    double x = qa ? QTY(a)->v : (a.t == T_FLOAT ? a.f : (a.t == T_INT || a.t == T_BOOL) ? (double)a.i : NAN);
    double y = qb ? QTY(b)->v : (b.t == T_FLOAT ? b.f : (b.t == T_INT || b.t == T_BOOL) ? (double)b.i : NAN);
    if (isnan(x) && !(a.t == T_FLOAT)) a_error("TypeError", "unsupported operand types: '%s' and a quantity", a.t == T_OBJ ? "object" : "none");
    if (isnan(y) && !(b.t == T_FLOAT)) a_error("TypeError", "unsupported operand types: a quantity and '%s'", b.t == T_OBJ ? "object" : "none");
    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MOD: {
        const char *verb = op == OP_ADD ? "add" : op == OP_SUB ? "subtract" : "take modulo of";
        const int16_t *dim;
        if (qa) { y = qty_same(a, b, verb, &ok); dim = QTY(a)->dim; } else { x = qty_same(b, a, verb, &ok); dim = QTY(b)->dim; }
        double r = op == OP_ADD ? x + y : op == OP_SUB ? x - y : fmod(x, y);
        if (op == OP_MOD && r != 0 && ((r < 0) != (y < 0))) r += y;
        return a_qty(r, dim, NULL, 1);
    }
    case OP_MUL: case OP_DIV:
        for (int i = 0; i < 7; i++) d[i] = (int16_t)((qa ? QTY(a)->dim[i] : 0) + (op == OP_MUL ? 1 : -1) * (qb ? QTY(b)->dim[i] : 0));
        if (op == OP_DIV && y == 0.0) a_error("ZeroDivisionError", "float division by zero");
        return a_qty(op == OP_MUL ? x * y : x / y, d, NULL, 1);
    case OP_FLOORDIV: if (qa) y = qty_same(a, b, "floor-divide", &ok); else x = qty_same(b, a, "floor-divide", &ok); return a_float(floor(x / y));
    case OP_POW: {
        if (qb) a_error("UnitError", "exponent must be dimensionless");
        double p = y, e = p * 60.0;
        if (fabs(e - nearbyint(e)) > 1e-9) a_error("UnitError", "a unit can only be raised to a power in 60ths");
        for (int i = 0; i < 7; i++) d[i] = (int16_t)nearbyint(QTY(a)->dim[i] * p);
        return a_qty(pow(x, p), d, NULL, 1);
    }
    default: a_error("TypeError", "unsupported operand types for %s with a quantity", opname(op));
    }
}
AV a_as_unit(AV v, const int16_t *dim, const char *text, double scale) {
    char x[96], y[96];
    if (a_is(v, O_QTY)) {
        if (!same_dim(QTY(v)->dim, dim)) a_error("UnitError", "cannot express %s in [%s] (%s)", dim_desc(QTY(v)->dim, x, sizeof x), text, dim_desc(dim, y, sizeof y));
        return a_qty(QTY(v)->v, dim, text, scale);
    }
    if (!num(v)) a_error("TypeError", "cannot convert %s to [%s]", tname(v), text);
    if (!dimless(dim)) a_error("UnitError", "cannot express a dimensionless number in [%s] (%s)", text, dim_desc(dim, y, sizeof y));
    return a_float(fval(v) / scale);
}

/* =================================================================== arithmetic */
static int64_t floordiv_i(int64_t a, int64_t b) {
    if (b == 0) a_error("ZeroDivisionError", "integer division by zero");
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}
static int64_t mod_i(int64_t a, int64_t b) {
    if (b == 0) a_error("ZeroDivisionError", "integer modulo by zero");
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static double mod_f(double a, double b) {
    if (b == 0.0) a_error("ZeroDivisionError", "float modulo");
    double r = fmod(a, b);
    if (r != 0.0 && ((r < 0) != (b < 0))) r += b;
    return r;
}
static AV seq_repeat(AV s, int64_t k) {
    if (k < 0) k = 0;
    if (a_is(s, O_STR)) {
        SB b = {0};
        for (int64_t i = 0; i < k; i++) sb_put(&b, STR(s)->s, STR(s)->len);
        return sb_done(&b);
    }
    AList *src = LIST(s);
    AList *l = new_list(0, (AKind)s.o->kind);
    for (int64_t i = 0; i < k; i++) for (int64_t j = 0; j < src->len; j++) a_list_append(l, src->v[j]);
    return a_obj(l);
}
static AV inst_op(AV a, AV b, const char *name, const char *rname, int *ok) {
    *ok = 1;
    if (a_is(a, O_INST)) { AV f = a_getattr(a, name); if (f.t != 255) return a_call(f, 1, &b, NULL); }
    if (a_is(b, O_INST) && rname) { AV f = a_getattr(b, rname); if (f.t != 255) return a_call(f, 1, &a, NULL); }
    *ok = 0; return A_NONE;
}
static const char *opname(int op) {
    static const char *n[] = {"", "+", "-", "*", "/", "//", "%", "**", "@", "&", "|", "^", "<<", ">>"};
    return op < (int)(sizeof n / sizeof *n) ? n[op] : "?";
}
static const char *tname(AV v) {
    switch (v.t) {
    case T_NONE: return "None"; case T_BOOL: return "bool"; case T_INT: return "int"; case T_FLOAT: return "float";
    default: break;
    }
    switch (v.o->kind) {
    case O_STR: return "str"; case O_LIST: return "list"; case O_TUPLE: return "tuple"; case O_DICT: return "dict";
    case O_SET: return "set"; case O_FUNC: case O_BUILTIN: case O_BOUND: return "function"; case O_CLASS: return "type";
    case O_INST: return ((AInst *)v.o)->cls->name->s; case O_RANGE: return "range";
    default: return "object";
    }
}
AV a_binop(int op, AV a, AV b) {
    if (a_is_tn(a) || a_is_tn(b)) return a_tensor_binop(op, a, b);
    if (a_is(a, O_QTY) || a_is(b, O_QTY)) return qty_binop(op, a, b);
    if ((a.t == T_INT || a.t == T_BOOL) && (b.t == T_INT || b.t == T_BOOL)) {
        int64_t x = a.i, y = b.i, r;
        switch (op) {
        case OP_ADD: if (__builtin_add_overflow(x, y, &r)) a_error("OverflowError", "integer overflow in +"); return a_int(r);
        case OP_SUB: if (__builtin_sub_overflow(x, y, &r)) a_error("OverflowError", "integer overflow in -"); return a_int(r);
        case OP_MUL: if (__builtin_mul_overflow(x, y, &r)) a_error("OverflowError", "integer overflow in *"); return a_int(r);
        case OP_DIV: if (y == 0) a_error("ZeroDivisionError", "division by zero"); return a_float((double)x / (double)y);
        case OP_FLOORDIV: return a_int(floordiv_i(x, y));
        case OP_MOD: return a_int(mod_i(x, y));
        case OP_POW: {
            if (y < 0) return a_float(pow((double)x, (double)y));
            int64_t acc = 1, base = x; int64_t e = y;
            while (e) {
                if (e & 1) { if (__builtin_mul_overflow(acc, base, &acc)) a_error("OverflowError", "integer overflow in **"); }
                e >>= 1;
                if (e && __builtin_mul_overflow(base, base, &base)) a_error("OverflowError", "integer overflow in **");
            }
            return a_int(acc);
        }
        case OP_AND: return a.t == T_BOOL && b.t == T_BOOL ? a_bool(x & y) : a_int(x & y);
        case OP_OR: return a.t == T_BOOL && b.t == T_BOOL ? a_bool(x | y) : a_int(x | y);
        case OP_XOR: return a.t == T_BOOL && b.t == T_BOOL ? a_bool(x ^ y) : a_int(x ^ y);
        case OP_SHL: if (y < 0) a_error("ValueError", "negative shift count"); return a_int(y >= 63 ? 0 : x << y);
        case OP_SHR: if (y < 0) a_error("ValueError", "negative shift count"); return a_int(y >= 63 ? (x < 0 ? -1 : 0) : x >> y);
        default: break;
        }
    }
    if (num(a) && num(b)) {
        double x = fval(a), y = fval(b);
        switch (op) {
        case OP_ADD: return a_float(x + y);
        case OP_SUB: return a_float(x - y);
        case OP_MUL: return a_float(x * y);
        case OP_DIV: if (y == 0.0) a_error("ZeroDivisionError", "float division by zero"); return a_float(x / y);
        case OP_FLOORDIV: if (y == 0.0) a_error("ZeroDivisionError", "float floor division by zero"); return a_float(floor(x / y));
        case OP_MOD: return a_float(mod_f(x, y));
        case OP_POW: return a_float(pow(x, y));
        default: break;
        }
    }
    if (op == OP_ADD && a.t == T_OBJ && b.t == T_OBJ && a.o->kind == b.o->kind) {
        if (a.o->kind == O_STR) {
            SB s = {0}; sb_put(&s, STR(a)->s, STR(a)->len); sb_put(&s, STR(b)->s, STR(b)->len); return sb_done(&s);
        }
        if (a.o->kind == O_LIST || a.o->kind == O_TUPLE) {
            AList *l = new_list(0, (AKind)a.o->kind);
            for (int64_t i = 0; i < LIST(a)->len; i++) a_list_append(l, LIST(a)->v[i]);
            for (int64_t i = 0; i < LIST(b)->len; i++) a_list_append(l, LIST(b)->v[i]);
            return a_obj(l);
        }
    }
    if (op == OP_MUL) {
        if ((a_is(a, O_STR) || a_is(a, O_LIST) || a_is(a, O_TUPLE)) && (b.t == T_INT || b.t == T_BOOL)) return seq_repeat(a, b.i);
        if ((a_is(b, O_STR) || a_is(b, O_LIST) || a_is(b, O_TUPLE)) && (a.t == T_INT || a.t == T_BOOL)) return seq_repeat(b, a.i);
    }
    if (op == OP_MOD && a_is(a, O_STR)) a_error("TypeError", "%%-formatting of strings: use an f-string");
    if ((op == OP_OR || op == OP_AND || op == OP_SUB || op == OP_XOR) && a_is(a, O_SET) && a_is(b, O_SET)) {
        AV r = a_set();
        ADict *x = DICT(a), *y = DICT(b);
        for (int64_t i = 0; i < x->nent; i++) if (x->e[i].used) {
            int inb = a_dict_get(y, x->e[i].k, NULL);
            if (op == OP_OR || (op == OP_AND && inb) || ((op == OP_SUB || op == OP_XOR) && !inb)) a_dict_set(DICT(r), x->e[i].k, A_NONE);
        }
        if (op == OP_OR || op == OP_XOR)
            for (int64_t i = 0; i < y->nent; i++) if (y->e[i].used && (op == OP_OR || !a_dict_get(x, y->e[i].k, NULL))) a_dict_set(DICT(r), y->e[i].k, A_NONE);
        return r;
    }
    if (op == OP_OR && a_is(a, O_DICT) && a_is(b, O_DICT)) {
        AV r = a_dict();
        for (int k = 0; k < 2; k++) { ADict *d = DICT(k ? b : a); for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set(DICT(r), d->e[i].k, d->e[i].v); }
        return r;
    }
    static const char *dn[] = {"", "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__", "__matmul__", "__and__", "__or__", "__xor__", "__lshift__", "__rshift__"};
    static const char *rn[] = {"", "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__", "__rmatmul__", "__rand__", "__ror__", "__rxor__", "__rlshift__", "__rrshift__"};
    if (op >= OP_ADD && op <= OP_SHR) { int ok; AV r = inst_op(a, b, dn[op], rn[op], &ok); if (ok) return r; }
    a_error("TypeError", "unsupported operand types for %s: '%s' and '%s'", opname(op), tname(a), tname(b));
}
AV a_unop(int op, AV a) {
    if (a_is_tn(a)) {
        if (op == OP_NEG) return a_tensor_neg(a);
        if (op == OP_POS) return a;
        if (op == OP_NOT) return a_bool(!a_tensor_truthy(a));
    }
    switch (op) {
    case OP_NOT: return a_bool(!a_truthy(a));
    case OP_NEG:
        if (a_is(a, O_QTY)) return a_qty(-QTY(a)->v, QTY(a)->dim, NULL, 1);
        if (a.t == T_INT || a.t == T_BOOL) { if (a.i == INT64_MIN) a_error("OverflowError", "integer overflow in -"); return a_int(-a.i); }
        if (a.t == T_FLOAT) return a_float(-a.f);
        if (a_is(a, O_INST)) { AV f = a_getattr(a, "__neg__"); if (f.t != 255) return a_call(f, 0, NULL, NULL); }
        break;
    case OP_POS: if (a_is(a, O_QTY)) return a; if (num(a)) return a.t == T_BOOL ? a_int(a.i) : a; break;
    case OP_INV: if (a.t == T_INT || a.t == T_BOOL) return a_int(~a.i); break;
    default: break;
    }
    a_error("TypeError", "bad operand type for unary operator: '%s'", tname(a));
}
static int cmp_order(AV a, AV b) {   /* <0, 0, >0 */
    if (a_is(a, O_QTY) || a_is(b, O_QTY)) {
        int ok; double x, y;
        if (a_is(a, O_QTY)) { x = QTY(a)->v; y = qty_same(a, b, "compare", &ok); } else { y = QTY(b)->v; x = qty_same(b, a, "compare", &ok); }
        return x < y ? -1 : x > y;
    }
    if (num(a) && num(b)) {
        if (a.t != T_FLOAT && b.t != T_FLOAT) return a.i < b.i ? -1 : a.i > b.i;
        double x = fval(a), y = fval(b);
        return x < y ? -1 : x > y;
    }
    if (a_is(a, O_STR) && a_is(b, O_STR)) {
        int64_t n = STR(a)->len < STR(b)->len ? STR(a)->len : STR(b)->len;
        int c = memcmp(STR(a)->s, STR(b)->s, (size_t)n);
        if (c) return c;
        return STR(a)->len < STR(b)->len ? -1 : STR(a)->len > STR(b)->len;
    }
    if ((a_is(a, O_LIST) && a_is(b, O_LIST)) || (a_is(a, O_TUPLE) && a_is(b, O_TUPLE))) {
        int64_t n = LIST(a)->len < LIST(b)->len ? LIST(a)->len : LIST(b)->len;
        for (int64_t i = 0; i < n; i++) {
            if (a_eq(LIST(a)->v[i], LIST(b)->v[i])) continue;
            return cmp_order(LIST(a)->v[i], LIST(b)->v[i]);
        }
        return LIST(a)->len < LIST(b)->len ? -1 : LIST(a)->len > LIST(b)->len;
    }
    if (a_is(a, O_INST)) {
        AV f = a_getattr(a, "__lt__");
        if (f.t != 255) { if (a_truthy(a_call(f, 1, &b, NULL))) return -1; AV g = a_getattr(b, "__lt__"); if (g.t != 255 && a_truthy(a_call(g, 1, &a, NULL))) return 1; return 0; }
    }
    a_error("TypeError", "'<' not supported between '%s' and '%s'", tname(a), tname(b));
}
static int tensor_cmp_applies(int op, AV a, AV b) {
    if (op == OP_IN || op == OP_NOTIN || op == OP_IS || op == OP_ISNOT) return 0;
    if (!(a_is_tn(a) || a_is_tn(b))) return 0;
    if ((op == OP_EQ || op == OP_NE) && (a.t == T_NONE || b.t == T_NONE || a_is(a, O_STR) || a_is(b, O_STR))) return 0;
    return 1;
}
AV a_cmp_slow(int op, AV a, AV b) {
    if (tensor_cmp_applies(op, a, b)) return a_tensor_cmp(op, a, b);
    return a_bool(a_compare(op, a, b));
}
int a_compare(int op, AV a, AV b) {
    if (tensor_cmp_applies(op, a, b)) return a_tensor_truthy(a_tensor_cmp(op, a, b));
    switch (op) {
    case OP_EQ: return a_eq(a, b);
    case OP_NE: return !a_eq(a, b);
    case OP_IS: return a.t == b.t && (a.t == T_NONE || a.i == b.i);
    case OP_ISNOT: return !(a.t == b.t && (a.t == T_NONE || a.i == b.i));
    case OP_IN: return a_contains(b, a);
    case OP_NOTIN: return !a_contains(b, a);
    default: break;
    }
    if ((a.t == T_FLOAT && isnan(a.f)) || (b.t == T_FLOAT && isnan(b.f))) return 0;
    int c = cmp_order(a, b);
    switch (op) {
    case OP_LT: return c < 0; case OP_LE: return c <= 0; case OP_GT: return c > 0; case OP_GE: return c >= 0;
    default: return 0;
    }
}
int a_contains(AV c, AV x) {
    if (c.t != T_OBJ) a_error("TypeError", "argument of type '%s' is not iterable", tname(c));
    switch (c.o->kind) {
    case O_STR: {
        if (!a_is(x, O_STR)) a_error("TypeError", "'in <string>' requires string as left operand");
        if (STR(x)->len == 0) return 1;
        for (int64_t i = 0; i + STR(x)->len <= STR(c)->len; i++) if (!memcmp(STR(c)->s + i, STR(x)->s, (size_t)STR(x)->len)) return 1;
        return 0;
    }
    case O_LIST: case O_TUPLE: for (int64_t i = 0; i < LIST(c)->len; i++) if (a_eq(LIST(c)->v[i], x)) return 1; return 0;
    case O_DICT: case O_SET: return a_dict_get(DICT(c), x, NULL);
    case O_RANGE: {
        ARange *r = (ARange *)c.o;
        if (!(x.t == T_INT || x.t == T_BOOL)) { if (x.t == T_FLOAT && x.f == floor(x.f)) x = a_int((int64_t)x.f); else return 0; }
        if (r->step > 0 ? (x.i < r->start || x.i >= r->stop) : (x.i > r->start || x.i <= r->stop)) return 0;
        return (x.i - r->start) % r->step == 0;
    }
    case O_INST: { AV f = a_getattr(c, "__contains__"); if (f.t != 255) return a_truthy(a_call(f, 1, &x, NULL)); break; }
    default: break;
    }
    int sp = a_ntmp; AV it = a_iter(c), v; A_PUSH(it);
    while (a_next(it, &v)) if (a_eq(v, x)) { a_ntmp = sp; return 1; }
    a_ntmp = sp; return 0;
}

/* =================================================================== length, indexing */
static _Noreturn void a_key_error(AV k) {
    SB b = {0};
    if (a_is(k, O_STR)) py_str_repr(&b, STR(k)); else repr_into(&b, k, 1);
    AV m = sb_done(&b);
    a_error("KeyError", "%s", STR(m)->s);
}
int64_t a_len(AV o) {
    if (a_is_tn(o)) return a_tensor_len(o);
    if (o.t == T_OBJ) switch (o.o->kind) {
    case O_STR: return STR(o)->nchars;
    case O_LIST: case O_TUPLE: return LIST(o)->len;
    case O_DICT: case O_SET: return DICT(o)->len;
    case O_RANGE: {
        ARange *r = (ARange *)o.o;
        if (r->step > 0) return r->stop > r->start ? (r->stop - r->start + r->step - 1) / r->step : 0;
        return r->start > r->stop ? (r->start - r->stop - r->step - 1) / (-r->step) : 0;
    }
    case O_INST: { AV f = a_getattr(o, "__len__"); if (f.t != 255) return a_call(f, 0, NULL, NULL).i; break; }
    default: break;
    }
    a_error("TypeError", "object of type '%s' has no len()", tname(o));
}
static int64_t norm_index(AV i, int64_t n) {
    if (!(i.t == T_INT || i.t == T_BOOL)) a_error("TypeError", "indices must be integers, not %s", tname(i));
    int64_t k = i.i < 0 ? i.i + n : i.i;
    if (k < 0 || k >= n) a_error("IndexError", "index %lld out of range (length %lld)", (long long)i.i, (long long)n);
    return k;
}
static void slice_bounds(ASlice *s, int64_t n, int64_t *lo, int64_t *hi, int64_t *st) {
    int64_t step = s->step.t == T_NONE ? 1 : s->step.i;
    if (step == 0) a_error("ValueError", "slice step cannot be zero");
    int64_t a, b;
    if (step > 0) {
        a = s->lo.t == T_NONE ? 0 : s->lo.i; b = s->hi.t == T_NONE ? n : s->hi.i;
        if (a < 0) a += n; if (a < 0) a = 0; if (a > n) a = n;
        if (b < 0) b += n; if (b < 0) b = 0; if (b > n) b = n;
    } else {
        a = s->lo.t == T_NONE ? n - 1 : s->lo.i; b = s->hi.t == T_NONE ? -1 - n : s->hi.i;
        if (a < 0) a += n; if (a < -1) a = -1; if (a >= n) a = n - 1;
        if (b < 0) b += n; if (b < -1) b = -1; if (b >= n) b = n - 1;
    }
    *lo = a; *hi = b; *st = step;
}
AV a_getitem(AV o, AV i) {
    if (a_is_tn(o)) return a_tensor_getitem(o, i);
    if (o.t != T_OBJ) a_error("TypeError", "'%s' object is not subscriptable", tname(o));
    switch (o.o->kind) {
    case O_LIST: case O_TUPLE: {
        AList *l = LIST(o);
        if (a_is(i, O_SLICE)) {
            int64_t a, b, st; slice_bounds((ASlice *)i.o, l->len, &a, &b, &st);
            AList *r = new_list(0, (AKind)o.o->kind);
            for (int64_t k = a; st > 0 ? k < b : k > b; k += st) a_list_append(r, l->v[k]);
            return a_obj(r);
        }
        return l->v[norm_index(i, l->len)];
    }
    case O_STR: {
        AStr *s = STR(o);
        if (a_is(i, O_SLICE)) {
            int64_t a, b, st; slice_bounds((ASlice *)i.o, s->nchars, &a, &b, &st);
            if (st == 1) { int64_t x = str_off(s, a), y = b > a ? str_off(s, b) : x; return a_strn(s->s + x, y - x); }
            SB sb = {0};
            for (int64_t k = a; st > 0 ? k < b : k > b; k += st) { int64_t x = str_off(s, k), y = str_off(s, k + 1); sb_put(&sb, s->s + x, y - x); }
            return sb_done(&sb);
        }
        int64_t k = norm_index(i, s->nchars), x = str_off(s, k), y = str_off(s, k + 1);
        return a_strn(s->s + x, y - x);
    }
    case O_DICT: {
        AV v;
        if (!a_dict_get(DICT(o), i, &v)) a_key_error(i);
        return v;
    }
    case O_RANGE: {
        ARange *r = (ARange *)o.o;
        int64_t k = norm_index(i, a_len(o));
        return a_int(r->start + k * r->step);
    }
    case O_INST: { AV f = a_getattr(o, "__getitem__"); if (f.t != 255) return a_call(f, 1, &i, NULL); break; }
    default: break;
    }
    a_error("TypeError", "'%s' object is not subscriptable", tname(o));
}
void a_setitem(AV o, AV i, AV v) {
    if (a_is_tn(o)) { a_tensor_setitem(o, i, v); return; }
    if (a_is(o, O_LIST)) {
        AList *l = LIST(o);
        if (a_is(i, O_SLICE)) {
            int64_t a, b, st; slice_bounds((ASlice *)i.o, l->len, &a, &b, &st);
            if (st != 1) a_error("ValueError", "extended slice assignment is not supported");
            if (b < a) b = a;
            int sp = a_ntmp; AV src = a_iter(v), x; A_PUSH(src); AList *nl = new_list(0, O_LIST); A_PUSH(a_obj(nl));
            for (int64_t k = 0; k < a; k++) a_list_append(nl, l->v[k]);
            while (a_next(src, &x)) a_list_append(nl, x);
            for (int64_t k = b; k < l->len; k++) a_list_append(nl, l->v[k]);
            AV *tmp = l->v; l->v = nl->v; l->len = nl->len; l->cap = nl->cap; nl->v = tmp; nl->len = 0;
            a_ntmp = sp; return;
        }
        l->v[norm_index(i, l->len)] = v; return;
    }
    if (a_is(o, O_DICT)) { a_dict_set(DICT(o), i, v); return; }
    if (a_is(o, O_INST)) { AV f = a_getattr(o, "__setitem__"); if (f.t != 255) { AV a2[2] = {i, v}; a_call(f, 2, a2, NULL); return; } }
    a_error("TypeError", "'%s' object does not support item assignment", tname(o));
}
void a_delitem(AV o, AV i) {
    if (a_is(o, O_LIST)) {
        AList *l = LIST(o); int64_t k = norm_index(i, l->len);
        memmove(l->v + k, l->v + k + 1, sizeof(AV) * (size_t)(l->len - k - 1)); l->len--; return;
    }
    if (a_is(o, O_DICT)) { if (!a_dict_del(DICT(o), i)) a_key_error(i); return; }
    a_error("TypeError", "'%s' object does not support item deletion", tname(o));
}


/* =================================================================== generators: each runs on a stack of its own */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <ucontext.h>
#endif
typedef struct AGen {
    AObj h;
    AFunc *fn; AFnCode body; AV args, kw;     /* what to run */
    AV value;                                 /* the value yielded (or sent in) */
    int state;                                /* 0 new, 1 suspended, 2 running, 3 finished */
    int failed; AV exc;                       /* the body raised: re-raised in the caller */
    AFrame *top, *base;                       /* the generator's frames while it is suspended */
    AFrame *caller_frame; AHandler *caller_handler, *handler; int caller_ntmp;
    struct AGen *outer;
#ifdef _WIN32
    void *fiber, *caller_fiber;
#else
    ucontext_t ctx, caller; char *stack;
#endif
} AGen;
static AGen *a_cur_gen = NULL;
static const char *gen_name(AObj *o) { return ((AGen *)o)->fn->name; }
#define GEN_STACK (512 * 1024)

static void gen_mark(AGen *g) {
    mark_obj((AObj *)g->fn); mark(g->args); mark(g->kw); mark(g->value); mark(g->exc);
    if (g->state == 1) for (AFrame *f = g->top; f; f = f->prev) { for (int i = 0; i < f->n; i++) mark(f->slots[i]); if (f == g->base) break; }
}
static void gen_free(AGen *g) {
#ifdef _WIN32
    if (g->fiber) DeleteFiber(g->fiber);
#else
    free(g->stack);
#endif
}
static void gen_switch_out(AGen *g) {         /* from the generator back to its caller */
#ifdef _WIN32
    SwitchToFiber(g->caller_fiber);
#else
    swapcontext(&g->ctx, &g->caller);
#endif
}
static void gen_run(AGen *g) {
    AHandler h;
    g->base = NULL;
    if (!A_TRY(h)) {
        AV r = g->body(g->fn, (int)LIST(g->args)->len, LIST(g->args)->v, g->kw.t == T_NONE ? NULL : DICT(g->kw));
        (void)r;
        a_handler = h.prev;
    } else { g->failed = 1; g->exc = a_exc; }
    g->state = 3;
    gen_switch_out(g);
}
#ifdef _WIN32
static void __stdcall gen_entry(void *p) { gen_run((AGen *)p); }
#else
static AGen *gen_starting;
static void gen_entry(void) { gen_run(gen_starting); }
#endif
AV a_gen_new(AFunc *fn, AFnCode body, int n, AV *args, ADict *kw) {
    AGen *g = (AGen *)a_alloc(sizeof(AGen), O_GEN);
    g->fn = fn; g->body = body; g->args = A_NONE; g->kw = A_NONE; g->value = A_NONE; g->exc = A_NONE;
    int sp = a_ntmp; A_PUSH(a_obj(g));
    g->args = a_tuple(n, args);
    if (kw) g->kw = a_obj(kw);
    a_ntmp = sp;
    return a_obj(g);
}
static int gen_resume(AGen *g, AV send, AV *out) {
    if (g->state == 3) return 0;
    if (g->state == 2) a_error("RuntimeError", "generator already running");
    g->caller_frame = a_frame; g->caller_handler = a_handler; g->caller_ntmp = a_ntmp;
    g->outer = a_cur_gen; a_cur_gen = g;
    g->value = send;
    if (g->state == 0) {
        g->state = 2;
        a_handler = NULL;
#ifdef _WIN32
        g->caller_fiber = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiber(NULL);
        g->fiber = CreateFiber(GEN_STACK, gen_entry, g);
        SwitchToFiber(g->fiber);
#else
        g->stack = (char *)xmalloc(GEN_STACK);
        getcontext(&g->ctx);
        g->ctx.uc_stack.ss_sp = g->stack; g->ctx.uc_stack.ss_size = GEN_STACK; g->ctx.uc_link = NULL;
        gen_starting = g;
        makecontext(&g->ctx, gen_entry, 0);
        swapcontext(&g->caller, &g->ctx);
#endif
    } else {
        g->state = 2;
        g->base->prev = g->caller_frame;          /* the generator's frames sit on its caller's while it runs */
        a_frame = g->top; a_handler = g->handler;
#ifdef _WIN32
        g->caller_fiber = GetCurrentFiber();
        SwitchToFiber(g->fiber);
#else
        swapcontext(&g->caller, &g->ctx);
#endif
    }
    a_cur_gen = g->outer;
    a_frame = g->caller_frame; a_handler = g->caller_handler; a_ntmp = g->caller_ntmp;
    if (g->state == 3) {
        gen_free(g);
#ifdef _WIN32
        g->fiber = NULL;
#else
        g->stack = NULL;
#endif
        if (g->failed) { g->failed = 0; AV e = g->exc; g->exc = A_NONE; a_raise(e); }
        return 0;
    }
    *out = g->value;
    return 1;
}
AV a_yield(AV v) {
    AGen *g = a_cur_gen;
    if (!g) a_error("RuntimeError", "yield outside a generator");
    /* the generator's outermost frame is the one right above its caller's */
    AFrame *f = a_frame;
    while (f && f->prev != g->caller_frame) f = f->prev;
    g->base = f; g->top = a_frame; g->handler = a_handler;
    g->value = v; g->state = 1;
    gen_switch_out(g);
    return g->value;                              /* what the caller sent (none for next()) */
}
static AV m_gen_send(int n, AV *a, ADict *kw, AV s) { (void)kw; AV v; if (!gen_resume((AGen *)s.o, n ? a[0] : A_NONE, &v)) a_error("StopIteration", ""); return v; }

/* =================================================================== iteration */
enum { IT_SEQ = 1, IT_STR, IT_DICT, IT_RANGE, IT_INST, IT_ENUM, IT_ZIP, IT_MAP, IT_FILTER, IT_REV };
static AV new_iter(int kind, AV src) {
    AIter *it = (AIter *)a_alloc(sizeof(AIter), O_ITER);
    it->kind = kind; it->src = src; it->aux = A_NONE;
    return a_obj(it);
}
AV a_iter(AV o) {
    if (a_is_tn(o)) {                 /* a tensor iterates over its first axis */
        int64_t n = a_tensor_len(o); int sp = a_ntmp; A_PUSH(o);
        AList *l = new_list(0, O_LIST); A_PUSH(a_obj(l));
        for (int64_t i = 0; i < n; i++) a_list_append(l, a_tensor_getitem(o, a_int(i)));
        a_ntmp = sp; return new_iter(IT_SEQ, a_obj(l));
    }
    if (o.t != T_OBJ) a_error("TypeError", "'%s' object is not iterable", tname(o));
    switch (o.o->kind) {
    case O_LIST: case O_TUPLE: return new_iter(IT_SEQ, o);
    case O_STR: return new_iter(IT_STR, o);
    case O_DICT: case O_SET: return new_iter(IT_DICT, o);
    case O_RANGE: {
        ARange *r = (ARange *)o.o; AV it = new_iter(IT_RANGE, o);
        ((AIter *)it.o)->pos = r->start; ((AIter *)it.o)->end = r->stop; ((AIter *)it.o)->step = r->step;
        return it;
    }
    case O_ITER: case O_GEN: return o;
    case O_INST: {
        AV f = a_getattr(o, "__iter__");
        if (f.t != 255) { AV r = a_call(f, 0, NULL, NULL); if (a_is(r, O_ITER)) return r; AV it = new_iter(IT_INST, r); return it; }
        break;
    }
    default: break;
    }
    a_error("TypeError", "'%s' object is not iterable", tname(o));
}
int a_next(AV itv, AV *out) {
    if (itv.o->kind == O_GEN) return gen_resume((AGen *)itv.o, A_NONE, out);
    AIter *it = (AIter *)itv.o;
    switch (it->kind) {
    case IT_SEQ: { AList *l = LIST(it->src); if (it->pos >= l->len) return 0; *out = l->v[it->pos++]; return 1; }
    case IT_REV: { AList *l = LIST(it->src); if (it->pos < 0 || it->pos >= l->len) return 0; *out = l->v[it->pos--]; return 1; }
    case IT_STR: {
        AStr *s = STR(it->src); if (it->pos >= s->len) return 0;
        int64_t b = it->pos + 1; while (b < s->len && (((unsigned char)s->s[b]) & 0xC0) == 0x80) b++;
        *out = a_strn(s->s + it->pos, b - it->pos); it->pos = b; return 1;
    }
    case IT_DICT: {
        ADict *d = DICT(it->src);
        while (it->pos < d->nent && !d->e[it->pos].used) it->pos++;
        if (it->pos >= d->nent) return 0;
        *out = d->e[it->pos++].k; return 1;
    }
    case IT_RANGE:
        if (it->step > 0 ? it->pos >= it->end : it->pos <= it->end) return 0;
        *out = a_int(it->pos); it->pos += it->step; return 1;
    case IT_INST: { AV f = a_getattr(it->src, "__next__"); AHandler h;
        if (!A_TRY(h)) { *out = a_call(f, 0, NULL, NULL); a_handler = h.prev; return 1; }
        if (a_exc_matches(a_exc, a_exc_classes[14])) return 0;
        a_raise(a_exc); }
    case IT_ENUM: {
        AV v; if (!a_next(it->aux, &v)) return 0;
        AV p[2] = {a_int(it->pos++), v}; *out = a_tuple(2, p); return 1;
    }
    case IT_ZIP: {
        AList *its = LIST(it->aux); AList *t = new_list(its->len, O_TUPLE); int sp = a_ntmp; A_PUSH(a_obj(t));
        for (int64_t i = 0; i < its->len; i++) t->v[i] = A_NONE;
        for (int64_t i = 0; i < its->len; i++) if (!a_next(its->v[i], &t->v[i])) { a_ntmp = sp; return 0; }
        a_ntmp = sp; *out = a_obj(t); return 1;
    }
    case IT_MAP: { AV v; if (!a_next(it->aux, &v)) return 0; *out = a_call(it->src, 1, &v, NULL); return 1; }
    case IT_FILTER: {
        AV v;
        while (a_next(it->aux, &v)) if (it->src.t == T_NONE ? a_truthy(v) : a_truthy(a_call(it->src, 1, &v, NULL))) { *out = v; return 1; }
        return 0;
    }
    default: return 0;
    }
}
void a_unpack(AV v, int n, AV *out) {
    if (a_is(v, O_LIST) || a_is(v, O_TUPLE)) {
        if (LIST(v)->len != n) a_error("ValueError", "expected %d values to unpack, got %lld", n, (long long)LIST(v)->len);
        memcpy(out, LIST(v)->v, sizeof(AV) * (size_t)n); return;
    }
    int sp = a_ntmp; AV it = a_iter(v); A_PUSH(it); int k = 0; AV x;
    while (a_next(it, &x)) { if (k >= n) a_error("ValueError", "too many values to unpack (expected %d)", n); out[k++] = x; }
    a_ntmp = sp;
    if (k != n) a_error("ValueError", "expected %d values to unpack, got %d", n, k);
}

/* =================================================================== attributes and calls */
static AV class_lookup(AClass *c, const char *name) {
    AV k = a_str(name), v;
    for (; c; c = c->base) if (a_dict_get(c->attrs, k, &v)) return v;
    v.t = 255; v.i = 0; return v;
}
static AV method_of(AV self, const char *name);
AV a_getattr(AV o, const char *name) {
    if (a_is(o, O_INST)) {
        AInst *s = (AInst *)o.o; AV v;
        if (a_dict_get(s->attrs, a_str(name), &v)) return v;
        v = class_lookup(s->cls, name);
        if (v.t != 255) {
            if (a_is(v, O_FUNC)) return a_bound(o, v);
            return v;
        }
        v.t = 255; return v;
    }
    if (a_is(o, O_CLASS)) {
        AV v = class_lookup((AClass *)o.o, name);
        if (v.t != 255) return v;
        if (!strcmp(name, "__name__")) return a_obj(((AClass *)o.o)->name);
        return v;
    }
    if (a_is_tn(o)) return a_tensor_attr(o, name);
    return method_of(o, name);
}
AV a_attr(AV o, const char *name) {
    AV v = a_getattr(o, name);
    if (v.t == 255) a_error("AttributeError", "'%s' object has no attribute '%s'", tname(o), name);
    return v;
}
void a_setattr(AV o, const char *name, AV v) {
    if (a_is(o, O_INST)) { a_dict_set(((AInst *)o.o)->attrs, a_str(name), v); return; }
    if (a_is(o, O_CLASS)) { a_dict_set(((AClass *)o.o)->attrs, a_str(name), v); return; }
    a_error("AttributeError", "cannot set attribute '%s' on '%s'", name, tname(o));
}
AV a_super_attr(AV self, AV cls, const char *name) {
    AClass *c = a_is(cls, O_CLASS) ? ((AClass *)cls.o)->base : NULL;
    AV v = c ? class_lookup(c, name) : (AV){.t = 255};
    if (v.t == 255) a_error("AttributeError", "'super' object has no attribute '%s'", name);
    return a_is(v, O_FUNC) ? a_bound(self, v) : v;
}
AV a_call(AV f, int n, AV *args, ADict *kw) {
    if (f.t != T_OBJ) a_error("TypeError", "'%s' object is not callable", tname(f));
    switch (f.o->kind) {
    case O_FUNC: { AFunc *fn = (AFunc *)f.o; return fn->code(fn, n, args, kw); }
    case O_BUILTIN: { ABuiltin *b = (ABuiltin *)f.o; return b->fn(n, args, kw, b->self); }
    case O_BOUND: {
        ABound *b = (ABound *)f.o;
        AV *a2 = (AV *)__builtin_alloca(sizeof(AV) * (size_t)(n + 1));
        a2[0] = b->self; if (n) memcpy(a2 + 1, args, sizeof(AV) * (size_t)n);
        return a_call(b->fn, n + 1, a2, kw);
    }
    case O_CLASS: {
        AClass *c = (AClass *)f.o;
        AV self = new_inst(c);
        AV init = class_lookup(c, "__init__");
        if (init.t != 255) {
            AV *a2 = (AV *)__builtin_alloca(sizeof(AV) * (size_t)(n + 1));
            a2[0] = self; if (n) memcpy(a2 + 1, args, sizeof(AV) * (size_t)n);
            a_call(init, n + 1, a2, kw);
            return self;
        }
        int is_exc = 0;
        for (AClass *x = c; x; x = x->base) if ((AObj *)x == a_exc_classes[0].o) is_exc = 1;
        if (is_exc) {
            AV m = n ? a_to_str(args[0]) : a_str("");
            AInst *s = (AInst *)self.o;
            a_dict_set(s->attrs, a_str("args"), a_tuple(n, args));
            a_dict_set(s->attrs, a_str("message"), m);
            a_dict_set(s->attrs, a_str("kind"), a_obj(exc_kind(c)));
            return self;
        }
        /* a struct: its fields in order, by position or by name, defaults from the class */
        AInst *s = (AInst *)self.o;
        int nf = 0; AClass *chain[16]; for (AClass *x = c; x && nf < 16; x = x->base) chain[nf++] = x;
        int pos = 0;
        for (int ci = nf - 1; ci >= 0; ci--) for (int i = 0; i < chain[ci]->nfields; i++) {
            AV key = a_obj(chain[ci]->fields[i]), v;
            if (pos < n) v = args[pos];
            else if (kw && a_dict_get(kw, key, &v)) {}
            else {
                AV d = class_lookup(c, chain[ci]->fields[i]->s);
                if (d.t == 255) a_error("TypeError", "%s() missing field '%s'", c->name->s, chain[ci]->fields[i]->s);
                v = d;
            }
            a_dict_set(s->attrs, key, v);
            pos++;
        }
        if (n > pos) a_error("TypeError", "%s() takes %d fields, %d given", c->name->s, pos, n);
        return self;
    }
    case O_INST: { AV c = a_getattr(f, "__call__"); if (c.t != 255) return a_call(c, n, args, kw); break; }
    default: break;
    }
    a_error("TypeError", "'%s' object is not callable", tname(f));
}
int a_param(AFunc *f, int n, AV *args, ADict *kw, int i, const char *name, int nparams, AV *out) {
    if (i < n) { *out = args[i]; return 1; }
    if (kw) { AV k = a_str(name); if (a_dict_get(kw, k, out)) return 1; }
    int d = i - (nparams - f->ndef);
    if (d >= 0 && d < f->ndef) { *out = f->defaults[d]; return 1; }
    a_error("TypeError", "%s() missing argument '%s'", f->name, name);
}
AV a_rest_args(int n, AV *args, int from) { return from < n ? a_list(n - from, args + from) : a_list(0, NULL); }
ADict *a_rest_kw(ADict *kw, int nnames, const char **names) {
    ADict *d = DICT(a_dict());
    if (!kw) return d;
    for (int64_t i = 0; i < kw->nent; i++) {
        if (!kw->e[i].used) continue;
        int named = 0;
        for (int j = 0; j < nnames; j++) if (a_is(kw->e[i].k, O_STR) && !strcmp(STR(kw->e[i].k)->s, names[j])) named = 1;
        if (!named) a_dict_set(d, kw->e[i].k, kw->e[i].v);
    }
    return d;
}

/* =================================================================== methods of built-in types */
static AV m_str_upper(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AV r = a_strn(STR(s)->s, STR(s)->len); for (int64_t i = 0; i < STR(r)->len; i++) if (STR(r)->s[i] >= 'a' && STR(r)->s[i] <= 'z') STR(r)->s[i] -= 32; return r; }
static AV m_str_lower(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AV r = a_strn(STR(s)->s, STR(s)->len); for (int64_t i = 0; i < STR(r)->len; i++) if (STR(r)->s[i] >= 'A' && STR(r)->s[i] <= 'Z') STR(r)->s[i] += 32; return r; }
static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
static AV strip_any(AV s, int n, AV *a, int left, int right) {
    const char *p = STR(s)->s; int64_t lo = 0, hi = STR(s)->len;
    const char *chars = n && a_is(a[0], O_STR) ? STR(a[0])->s : NULL;
    #define STRIPC(c) (chars ? strchr(chars, (c)) != NULL && (c) != 0 : is_ws(c))
    if (left) while (lo < hi && STRIPC(p[lo])) lo++;
    if (right) while (hi > lo && STRIPC(p[hi - 1])) hi--;
    #undef STRIPC
    return a_strn(p + lo, hi - lo);
}
static AV m_str_strip(int n, AV *a, ADict *kw, AV s) { (void)kw; return strip_any(s, n, a, 1, 1); }
static AV m_str_lstrip(int n, AV *a, ADict *kw, AV s) { (void)kw; return strip_any(s, n, a, 1, 0); }
static AV m_str_rstrip(int n, AV *a, ADict *kw, AV s) { (void)kw; return strip_any(s, n, a, 0, 1); }
static AV m_str_split(int n, AV *a, ADict *kw, AV s) {
    (void)kw;
    AList *r = new_list(0, O_LIST); const char *p = STR(s)->s; int64_t len = STR(s)->len;
    int64_t maxs = n > 1 ? a[1].i : -1;
    if (n == 0 || a[0].t == T_NONE) {
        int64_t i = 0;
        while (i < len) {
            while (i < len && is_ws(p[i])) i++;
            if (i >= len) break;
            int64_t j = i;
            if (maxs >= 0 && r->len >= maxs) { j = len; while (j > i && is_ws(p[j - 1])) j--; a_list_append(r, a_strn(p + i, j - i)); break; }
            while (j < len && !is_ws(p[j])) j++;
            a_list_append(r, a_strn(p + i, j - i)); i = j;
        }
        return a_obj(r);
    }
    AStr *sep = STR(a[0]);
    if (sep->len == 0) a_error("ValueError", "empty separator");
    int64_t i = 0, start = 0;
    while (i + sep->len <= len) {
        if ((maxs < 0 || r->len < maxs) && !memcmp(p + i, sep->s, (size_t)sep->len)) { a_list_append(r, a_strn(p + start, i - start)); i += sep->len; start = i; }
        else i++;
    }
    a_list_append(r, a_strn(p + start, len - start));
    return a_obj(r);
}
static AV m_str_join(int n, AV *a, ADict *kw, AV s) {
    (void)n; (void)kw;
    SB b = {0}; int sp = a_ntmp; AV it = a_iter(a[0]), x; A_PUSH(it); int first = 1;
    while (a_next(it, &x)) {
        if (!a_is(x, O_STR)) a_error("TypeError", "join() needs strings, found %s", tname(x));
        if (!first) sb_put(&b, STR(s)->s, STR(s)->len);
        first = 0; sb_put(&b, STR(x)->s, STR(x)->len);
    }
    a_ntmp = sp;
    return sb_done(&b);
}
static AV m_str_replace(int n, AV *a, ADict *kw, AV s) {
    (void)n; (void)kw;
    AStr *o = STR(a[0]), *w = STR(a[1]); const char *p = STR(s)->s; int64_t len = STR(s)->len;
    if (o->len == 0) return s;
    SB b = {0}; int64_t i = 0;
    while (i < len) {
        if (i + o->len <= len && !memcmp(p + i, o->s, (size_t)o->len)) { sb_put(&b, w->s, w->len); i += o->len; }
        else { sb_put(&b, p + i, 1); i++; }
    }
    return sb_done(&b);
}
static AV m_str_startswith(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; AStr *x = STR(a[0]); return a_bool(x->len <= STR(s)->len && !memcmp(STR(s)->s, x->s, (size_t)x->len)); }
static AV m_str_endswith(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; AStr *x = STR(a[0]); return a_bool(x->len <= STR(s)->len && !memcmp(STR(s)->s + STR(s)->len - x->len, x->s, (size_t)x->len)); }
static AV m_str_find(int n, AV *a, ADict *kw, AV s) {
    (void)n; (void)kw;
    AStr *x = STR(a[0]);
    for (int64_t i = 0; i + x->len <= STR(s)->len; i++) if (!memcmp(STR(s)->s + i, x->s, (size_t)x->len)) {
        int64_t c = 0; for (int64_t b = 0; b < i; b++) if ((((unsigned char)STR(s)->s[b]) & 0xC0) != 0x80) c++;
        return a_int(c);
    }
    return a_int(-1);
}
static AV m_str_count(int n, AV *a, ADict *kw, AV s) {
    (void)n; (void)kw; AStr *x = STR(a[0]); int64_t c = 0;
    if (x->len == 0) return a_int(STR(s)->nchars + 1);
    for (int64_t i = 0; i + x->len <= STR(s)->len;) { if (!memcmp(STR(s)->s + i, x->s, (size_t)x->len)) { c++; i += x->len; } else i++; }
    return a_int(c);
}
static AV m_str_isdigit(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (!STR(s)->len) return a_bool(0); for (int64_t i = 0; i < STR(s)->len; i++) if (STR(s)->s[i] < '0' || STR(s)->s[i] > '9') return a_bool(0); return a_bool(1); }
static AV m_str_isalpha(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (!STR(s)->len) return a_bool(0); for (int64_t i = 0; i < STR(s)->len; i++) { unsigned char c = (unsigned char)STR(s)->s[i]; if (!((c | 32) >= 'a' && (c | 32) <= 'z') && c < 0x80) return a_bool(0); } return a_bool(1); }
static AV m_str_isspace(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (!STR(s)->len) return a_bool(0); for (int64_t i = 0; i < STR(s)->len; i++) if (!is_ws(STR(s)->s[i])) return a_bool(0); return a_bool(1); }

static AV m_list_append(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; a_list_append(LIST(s), a[0]); return A_NONE; }
static AV m_list_extend(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; int sp = a_ntmp; AV it = a_iter(a[0]), x; A_PUSH(it); while (a_next(it, &x)) a_list_append(LIST(s), x); a_ntmp = sp; return A_NONE; }
static AV m_list_pop(int n, AV *a, ADict *kw, AV s) {
    (void)kw; AList *l = LIST(s);
    if (!l->len) a_error("IndexError", "pop from empty list");
    int64_t k = n ? norm_index(a[0], l->len) : l->len - 1;
    AV v = l->v[k]; memmove(l->v + k, l->v + k + 1, sizeof(AV) * (size_t)(l->len - k - 1)); l->len--;
    return v;
}
static AV m_list_insert(int n, AV *a, ADict *kw, AV s) {
    (void)n; (void)kw; AList *l = LIST(s); int64_t k = a[0].i;
    if (k < 0) k += l->len; if (k < 0) k = 0; if (k > l->len) k = l->len;
    a_list_append(l, A_NONE); memmove(l->v + k + 1, l->v + k, sizeof(AV) * (size_t)(l->len - k - 1)); l->v[k] = a[1];
    return A_NONE;
}
static AV m_list_index(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; for (int64_t i = 0; i < LIST(s)->len; i++) if (a_eq(LIST(s)->v[i], a[0])) return a_int(i); a_error("ValueError", "value is not in the list"); }
static AV m_list_count(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; int64_t c = 0; for (int64_t i = 0; i < LIST(s)->len; i++) if (a_eq(LIST(s)->v[i], a[0])) c++; return a_int(c); }
static AV m_list_remove(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; AList *l = LIST(s); for (int64_t i = 0; i < l->len; i++) if (a_eq(l->v[i], a[0])) { memmove(l->v + i, l->v + i + 1, sizeof(AV) * (size_t)(l->len - i - 1)); l->len--; return A_NONE; } a_error("ValueError", "value is not in the list"); }
static AV m_list_reverse(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AList *l = LIST(s); for (int64_t i = 0, j = l->len - 1; i < j; i++, j--) { AV t = l->v[i]; l->v[i] = l->v[j]; l->v[j] = t; } return A_NONE; }
static AV m_list_clear(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (a_is(s, O_LIST)) LIST(s)->len = 0; else { ADict *d = DICT(s); d->len = d->nent = 0; for (int64_t i = 0; i < d->icap; i++) d->idx[i] = -1; } return A_NONE; }
static AV m_list_copy(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; if (a_is(s, O_LIST)) return a_list(LIST(s)->len, LIST(s)->v); AV r = s.o->kind == O_SET ? a_set() : a_dict(); ADict *d = DICT(s); for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_dict_set(DICT(r), d->e[i].k, d->e[i].v); return r; }

/* sort: a stable merge sort on (key, value) */
static AV sort_keyfn; static int sort_rev;
static int sort_less(AV a, AV b) { return a_compare(OP_LT, a, b); }
static void merge_sort(AV *k, AV *v, AV *tk, AV *tv, int64_t n) {
    if (n < 2) return;
    int64_t m = n / 2;
    merge_sort(k, v, tk, tv, m); merge_sort(k + m, v + m, tk, tv, n - m);
    int64_t i = 0, j = m, o = 0;
    while (i < m && j < n) {
        int take_right = sort_rev ? sort_less(k[i], k[j]) : sort_less(k[j], k[i]);
        if (take_right) { tk[o] = k[j]; tv[o++] = v[j++]; } else { tk[o] = k[i]; tv[o++] = v[i++]; }
    }
    while (i < m) { tk[o] = k[i]; tv[o++] = v[i++]; }
    while (j < n) { tk[o] = k[j]; tv[o++] = v[j++]; }
    memcpy(k, tk, sizeof(AV) * (size_t)n); memcpy(v, tv, sizeof(AV) * (size_t)n);
}
static void sort_list(AList *l, AV key, int rev) {
    int64_t n = l->len;
    int sp = a_ntmp; A_PUSH(a_obj(l));
    AList *keys = new_list(n, O_LIST); A_PUSH(a_obj(keys));     /* keys survive the key calls */
    for (int64_t i = 0; i < n; i++) keys->v[i] = A_NONE;
    for (int64_t i = 0; i < n; i++) keys->v[i] = key.t == T_NONE ? l->v[i] : a_call(key, 1, &l->v[i], NULL);
    AV *tk = (AV *)xmalloc(sizeof(AV) * (size_t)(n ? n : 1)), *tv = (AV *)xmalloc(sizeof(AV) * (size_t)(n ? n : 1));
    AV sk = sort_keyfn; int sr = sort_rev; sort_keyfn = key; sort_rev = rev;
    merge_sort(keys->v, l->v, tk, tv, n);
    sort_keyfn = sk; sort_rev = sr;
    free(tk); free(tv);
    a_ntmp = sp;
}
static AV kwarg(ADict *kw, const char *name, AV dflt) { AV v; if (kw && a_dict_get(kw, a_str(name), &v)) return v; return dflt; }
static AV m_list_sort(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; sort_list(LIST(s), kwarg(kw, "key", A_NONE), a_truthy(kwarg(kw, "reverse", a_bool(0)))); return A_NONE; }

static AV m_dict_get(int n, AV *a, ADict *kw, AV s) { (void)kw; AV v; if (a_dict_get(DICT(s), a[0], &v)) return v; return n > 1 ? a[1] : A_NONE; }
static AV m_dict_keys(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AList *l = new_list(0, O_LIST); ADict *d = DICT(s); for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_list_append(l, d->e[i].k); return a_obj(l); }
static AV m_dict_values(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AList *l = new_list(0, O_LIST); ADict *d = DICT(s); for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) a_list_append(l, d->e[i].v); return a_obj(l); }
static AV m_dict_items(int n, AV *a, ADict *kw, AV s) { (void)n; (void)a; (void)kw; AList *l = new_list(0, O_LIST); ADict *d = DICT(s); for (int64_t i = 0; i < d->nent; i++) if (d->e[i].used) { AV p[2] = {d->e[i].k, d->e[i].v}; a_list_append(l, a_tuple(2, p)); } return a_obj(l); }
static AV m_dict_pop(int n, AV *a, ADict *kw, AV s) { (void)kw; AV v; if (a_dict_get(DICT(s), a[0], &v)) { a_dict_del(DICT(s), a[0]); return v; } if (n > 1) return a[1]; a_key_error(a[0]); }
static AV m_dict_setdefault(int n, AV *a, ADict *kw, AV s) { (void)kw; AV v; if (a_dict_get(DICT(s), a[0], &v)) return v; v = n > 1 ? a[1] : A_NONE; a_dict_set(DICT(s), a[0], v); return v; }
static AV m_dict_update(int n, AV *a, ADict *kw, AV s) {
    if (n) { ADict *o = DICT(a[0]); for (int64_t i = 0; i < o->nent; i++) if (o->e[i].used) a_dict_set(DICT(s), o->e[i].k, o->e[i].v); }
    if (kw) for (int64_t i = 0; i < kw->nent; i++) if (kw->e[i].used) a_dict_set(DICT(s), kw->e[i].k, kw->e[i].v);
    return A_NONE;
}
static AV m_set_add(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; a_dict_set(DICT(s), a[0], A_NONE); return A_NONE; }
static AV m_set_discard(int n, AV *a, ADict *kw, AV s) { (void)n; (void)kw; a_dict_del(DICT(s), a[0]); return A_NONE; }

static AV method_of(AV o, const char *name) {
    AV none; none.t = 255; none.i = 0;
    if (o.t != T_OBJ) return none;
    #define M(n_, f_) if (!strcmp(name, n_)) return a_builtin(n_, f_, o)
    switch (o.o->kind) {
    case O_STR:
        M("upper", m_str_upper); M("lower", m_str_lower); M("strip", m_str_strip); M("lstrip", m_str_lstrip);
        M("rstrip", m_str_rstrip); M("split", m_str_split); M("join", m_str_join); M("replace", m_str_replace);
        M("startswith", m_str_startswith); M("endswith", m_str_endswith); M("find", m_str_find); M("count", m_str_count);
        M("isdigit", m_str_isdigit); M("isalpha", m_str_isalpha); M("isspace", m_str_isspace);
        break;
    case O_LIST:
        M("append", m_list_append); M("extend", m_list_extend); M("pop", m_list_pop); M("insert", m_list_insert);
        M("index", m_list_index); M("count", m_list_count); M("remove", m_list_remove); M("reverse", m_list_reverse);
        M("sort", m_list_sort); M("clear", m_list_clear); M("copy", m_list_copy);
        break;
    case O_TUPLE: M("index", m_list_index); M("count", m_list_count); break;
    case O_DICT:
        M("get", m_dict_get); M("keys", m_dict_keys); M("values", m_dict_values); M("items", m_dict_items);
        M("pop", m_dict_pop); M("setdefault", m_dict_setdefault); M("update", m_dict_update); M("clear", m_list_clear);
        M("copy", m_list_copy);
        break;
    case O_GEN: M("send", m_gen_send); break;
    case O_SET: M("add", m_set_add); M("discard", m_set_discard); M("remove", m_dict_pop); M("clear", m_list_clear); M("copy", m_list_copy); break;
    default: break;
    }
    #undef M
    return none;
}

/* =================================================================== built-in functions */
static AV b_print(int n, AV *a, ADict *kw, AV self) {
    (void)self;
    AV sep = kwarg(kw, "sep", A_NONE), end = kwarg(kw, "end", A_NONE);
    for (int i = 0; i < n; i++) {
        if (i) { if (a_is(sep, O_STR)) fwrite(STR(sep)->s, 1, (size_t)STR(sep)->len, stdout); else fputc(' ', stdout); }
        AV s = a_to_str(a[i]); fwrite(STR(s)->s, 1, (size_t)STR(s)->len, stdout);
    }
    if (a_is(end, O_STR)) fwrite(STR(end)->s, 1, (size_t)STR(end)->len, stdout); else fputc('\n', stdout);
    return A_NONE;
}
static AV b_len(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_int(a_len(a[0])); }
static AV b_range(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    for (int i = 0; i < n; i++) if (!(a[i].t == T_INT || a[i].t == T_BOOL)) a_error("TypeError", "range() needs integers");
    if (n == 1) return a_range(0, a[0].i, 1);
    if (n == 2) return a_range(a[0].i, a[1].i, 1);
    return a_range(a[0].i, a[1].i, a[2].i);
}
static AV b_str(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; return n ? a_to_str(a[0]) : a_str(""); }
static AV b_repr(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_repr(a[0]); }
static AV b_int(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    if (!n) return a_int(0);
    AV v = a[0];
    if (v.t == T_INT) return v;
    if (v.t == T_BOOL) return a_int(v.i);
    if (v.t == T_FLOAT) { if (isnan(v.f) || isinf(v.f)) a_error("ValueError", "cannot convert float %f to integer", v.f); return a_int((int64_t)v.f); }
    if (a_is(v, O_STR)) {
        int base = n > 1 ? (int)a[1].i : 10; char *end; AV st = strip_any(v, 0, NULL, 1, 1);

        long long r = strtoll(STR(st)->s, &end, base);
        if (end == STR(st)->s || *end) a_error("ValueError", "invalid literal for int(): '%s'", STR(v)->s);
        return a_int(r);
    }
    a_error("TypeError", "int() of %s", tname(v));
}
static AV b_float(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    if (!n) return a_float(0.0);
    AV v = a[0];
    if (num(v)) return a_float(fval(v));
    if (a_is(v, O_STR)) {
        AV st = strip_any(v, 0, NULL, 1, 1); char *end;
        double d = strtod(STR(st)->s, &end);
        if (end == STR(st)->s || *end) a_error("ValueError", "could not convert string to float: '%s'", STR(v)->s);
        return a_float(d);
    }
    a_error("TypeError", "float() of %s", tname(v));
}
static AV b_bool(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; return a_bool(n && a_truthy(a[0])); }
static AV collect(AV it) { int sp = a_ntmp; AList *l = new_list(0, O_LIST); A_PUSH(a_obj(l)); AV i = a_iter(it), x; A_PUSH(i); while (a_next(i, &x)) a_list_append(l, x); a_ntmp = sp; return a_obj(l); }
static AV b_list(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; return n ? collect(a[0]) : a_list(0, NULL); }
static AV b_tuple(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; if (!n) return a_tuple(0, NULL); AV l = collect(a[0]); l.o->kind = O_TUPLE; return l; }
static AV b_dict(int n, AV *a, ADict *kw, AV self) {
    (void)self; AV d = a_dict();
    if (n) {
        if (a_is(a[0], O_DICT)) m_dict_update(1, a, NULL, d);
        else { int sp = a_ntmp; A_PUSH(d); AV it = a_iter(a[0]), x; A_PUSH(it); while (a_next(it, &x)) { AV kv[2]; A_PUSH(x); a_unpack(x, 2, kv); a_ntmp--; a_dict_set(DICT(d), kv[0], kv[1]); } a_ntmp = sp; }
    }
    if (kw) m_dict_update(0, NULL, kw, d);
    return d;
}
static AV b_set(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; AV s = a_set(); if (n) { int sp = a_ntmp; A_PUSH(s); AV it = a_iter(a[0]), x; A_PUSH(it); while (a_next(it, &x)) a_dict_set(DICT(s), x, A_NONE); a_ntmp = sp; } return s; }
static AV b_abs(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("abs", 1)(n, a, kw, self); (void)n; (void)kw; (void)self; if (a_is(a[0], O_QTY)) return a_qty(fabs(QTY(a[0])->v), QTY(a[0])->dim, NULL, 1); if (a[0].t == T_FLOAT) return a_float(fabs(a[0].f)); if (a[0].t == T_INT || a[0].t == T_BOOL) return a_int(a[0].i < 0 ? -a[0].i : a[0].i); a_error("TypeError", "abs() of %s", tname(a[0])); }
static AV minmax(int n, AV *a, ADict *kw, int mx) {
    AV key = kwarg(kw, "key", A_NONE), best = A_NONE, bestk = A_NONE; int have = 0;
    int sp = a_ntmp; AV src = n == 1 ? a_iter(a[0]) : A_NONE; int i = 0; AV x;
    A_PUSH(src); A_PUSH(A_NONE); A_PUSH(A_NONE); A_PUSH(A_NONE);
    for (;;) {
        if (n == 1) { if (!a_next(src, &x)) break; } else { if (i >= n) break; x = a[i++]; }
        a_tmp[sp + 3] = x;
        AV k = key.t == T_NONE ? x : a_call(key, 1, &x, NULL);
        if (!have || (mx ? a_compare(OP_GT, k, bestk) : a_compare(OP_LT, k, bestk))) { best = x; bestk = k; have = 1; a_tmp[sp + 1] = best; a_tmp[sp + 2] = bestk; }
    }
    a_ntmp = sp;
    if (!have) { AV d; if (kw && a_dict_get(kw, a_str("default"), &d)) return d; a_error("ValueError", "%s() of an empty sequence", mx ? "max" : "min"); }
    return best;
}
static AV b_min(int n, AV *a, ADict *kw, AV self) { if ((n == 1 || kw) && a_is_tn(a[0])) return a_tensor_builtin("min", 1)(n, a, kw, self); (void)self; return minmax(n, a, kw, 0); }
static AV b_max(int n, AV *a, ADict *kw, AV self) { if ((n == 1 || kw) && a_is_tn(a[0])) return a_tensor_builtin("max", 1)(n, a, kw, self); (void)self; return minmax(n, a, kw, 1); }
static AV b_sum(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("sum", 1)(n, a, kw, self);
    (void)kw; (void)self;
    int sp = a_ntmp; AV acc = n > 1 ? a[1] : a_int(0), it = a_iter(a[0]), x; A_PUSH(it); A_PUSH(acc);
    while (a_next(it, &x)) { acc = a_add(acc, x); a_tmp[sp + 1] = acc; }
    a_ntmp = sp; return acc;
}
static AV b_sorted(int n, AV *a, ADict *kw, AV self) { (void)n; (void)self; AV l = collect(a[0]); sort_list(LIST(l), kwarg(kw, "key", A_NONE), a_truthy(kwarg(kw, "reverse", a_bool(0)))); return l; }
static AV b_reversed(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; AV l = (a_is(a[0], O_LIST) || a_is(a[0], O_TUPLE)) ? a[0] : collect(a[0]); AV it = new_iter(IT_REV, l); ((AIter *)it.o)->pos = LIST(l)->len - 1; return it; }
static AV b_enumerate(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; AV src = a_iter(a[0]); AV it = new_iter(IT_ENUM, a[0]); ((AIter *)it.o)->aux = src; ((AIter *)it.o)->pos = n > 1 ? a[1].i : 0; return it; }
static AV b_zip(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; AList *its = new_list(n, O_LIST); int sp = a_ntmp; A_PUSH(a_obj(its)); for (int i = 0; i < n; i++) its->v[i] = A_NONE; for (int i = 0; i < n; i++) its->v[i] = a_iter(a[i]); a_ntmp = sp; AV it = new_iter(IT_ZIP, A_NONE); ((AIter *)it.o)->aux = a_obj(its); return it; }
static AV b_map(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; AV src = a_iter(a[1]); AV it = new_iter(IT_MAP, a[0]); ((AIter *)it.o)->aux = src; return it; }
static AV b_filter(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; AV src = a_iter(a[1]); AV it = new_iter(IT_FILTER, a[0]); ((AIter *)it.o)->aux = src; return it; }
static AV b_any(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; int sp = a_ntmp; AV it = a_iter(a[0]), x; A_PUSH(it); while (a_next(it, &x)) if (a_truthy(x)) { a_ntmp = sp; return a_bool(1); } a_ntmp = sp; return a_bool(0); }
static AV b_all(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; int sp = a_ntmp; AV it = a_iter(a[0]), x; A_PUSH(it); while (a_next(it, &x)) if (!a_truthy(x)) { a_ntmp = sp; return a_bool(0); } a_ntmp = sp; return a_bool(1); }
static AV b_round(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    if (a[0].t == T_INT || a[0].t == T_BOOL) return a_int(a[0].i);
    double x = fval(a[0]);
    if (n < 2 || a[1].t == T_NONE) { double r = nearbyint(x); return a_int((int64_t)r); }
    char buf[64]; int nd = (int)a[1].i;
    if (nd < 0) { double m = pow(10.0, (double)-nd); return a_float(nearbyint(x / m) * m); }
    snprintf(buf, sizeof buf, "%.*f", nd > 30 ? 30 : nd, x);     /* exact decimal rounding, half to even */
    return a_float(strtod(buf, NULL));
}
static AV b_isinstance(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_bool(a_isinstance(a[0], a[1])); }
static AV b_ord(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    const unsigned char *s = (const unsigned char *)STR(a[0])->s; int64_t cp;
    if (s[0] < 0x80) cp = s[0]; else if (s[0] < 0xE0) cp = ((s[0] & 31) << 6) | (s[1] & 63);
    else if (s[0] < 0xF0) cp = ((s[0] & 15) << 12) | ((s[1] & 63) << 6) | (s[2] & 63);
    else cp = ((s[0] & 7) << 18) | ((s[1] & 63) << 12) | ((s[2] & 63) << 6) | (s[3] & 63);
    return a_int(cp);
}
static AV b_chr(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    int64_t c = a[0].i; char b[5]; int k = 0;
    if (c < 0x80) b[k++] = (char)c;
    else if (c < 0x800) { b[k++] = (char)(0xC0 | (c >> 6)); b[k++] = (char)(0x80 | (c & 63)); }
    else if (c < 0x10000) { b[k++] = (char)(0xE0 | (c >> 12)); b[k++] = (char)(0x80 | ((c >> 6) & 63)); b[k++] = (char)(0x80 | (c & 63)); }
    else { b[k++] = (char)(0xF0 | (c >> 18)); b[k++] = (char)(0x80 | ((c >> 12) & 63)); b[k++] = (char)(0x80 | ((c >> 6) & 63)); b[k++] = (char)(0x80 | (c & 63)); }
    return a_strn(b, k);
}
static AV b_type(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; if (a_is(a[0], O_INST)) return a_obj(((AInst *)a[0].o)->cls); return a_str(tname(a[0])); }
static AV b_input(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    if (n) { AV s = a_to_str(a[0]); fwrite(STR(s)->s, 1, (size_t)STR(s)->len, stdout); fflush(stdout); }
    SB b = {0}; int c;
    while ((c = fgetc(stdin)) != EOF && c != '\n') { char ch = (char)c; sb_put(&b, &ch, 1); }
    return sb_done(&b);
}
static AV b_hash(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_int((int64_t)(a_hash(a[0]) >> 1)); }
static AV b_divmod(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; AV p[2] = {a_binop(OP_FLOORDIV, a[0], a[1]), a_binop(OP_MOD, a[0], a[1])}; return a_tuple(2, p); }
static AV b_pow(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_binop(OP_POW, a[0], a[1]); }
#define MATH1(nm, fn) static AV b_##nm(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; if (a_is_tn(a[0])) return a_tensor_builtin(#nm, 1)(n, a, kw, self); if (a_is(a[0], O_QTY)) a_error("UnitError", #nm "() needs a dimensionless number"); if (!num(a[0])) a_error("TypeError", #nm "() needs a number"); return a_float(fn(fval(a[0]))); }
static AV b_sqrt(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("sqrt", 1)(n, a, kw, self);
    (void)n; (void)kw; (void)self;
    if (a_is(a[0], O_QTY)) return qty_binop(OP_POW, a[0], a_float(0.5));
    if (!num(a[0])) a_error("TypeError", "sqrt() needs a number");
    double x = fval(a[0]); if (x < 0) a_error("ValueError", "math domain error: sqrt of %g", x); return a_float(sqrt(x));
}
static double mylog(double x) { if (x <= 0) a_error("ValueError", "math domain error: log of %g", x); return log(x); }
MATH1(sin, sin) MATH1(cos, cos) MATH1(tan, tan) MATH1(exp, exp) MATH1(log, mylog)
MATH1(log10, log10) MATH1(log2, log2) MATH1(atan, atan) MATH1(asin, asin) MATH1(acos, acos) MATH1(tanh, tanh)
MATH1(sinh, sinh) MATH1(cosh, cosh)

/* ---- more mathematics */
static AV b_cbrt(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; if (a_is(a[0], O_QTY)) return qty_binop(OP_POW, a[0], a_float(1.0 / 3.0)); return a_float(cbrt(fval(a[0]))); }
MATH1(erf, erf) MATH1(erfc, erfc) MATH1(expm1, expm1) MATH1(log1p, log1p) MATH1(lgamma, lgamma)
static AV b_gamma(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_float(tgamma(fval(a[0]))); }
static AV b_beta(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; double x = fval(a[0]), y = fval(a[1]); return a_float(exp(lgamma(x) + lgamma(y) - lgamma(x + y))); }
static AV b_factorial(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    if (a[0].t == T_INT) { if (a[0].i < 0) a_error("ValueError", "factorial() not defined for negative values"); int64_t r = 1; for (int64_t i = 2; i <= a[0].i; i++) if (__builtin_mul_overflow(r, i, &r)) a_error("OverflowError", "factorial(%lld) does not fit in 64 bits", (long long)a[0].i); return a_int(r); }
    return a_float(tgamma(fval(a[0]) + 1));
}
static AV b_comb(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    if (a[0].t == T_INT && a[1].t == T_INT) {
        int64_t N = a[0].i, k = a[1].i; if (k < 0 || k > N) return a_int(0); if (k > N - k) k = N - k;
        __int128 r = 1; for (int64_t i = 1; i <= k; i++) { r = r * (N - k + i) / i; if (r > INT64_MAX) a_error("OverflowError", "comb() does not fit in 64 bits"); }
        return a_int((int64_t)r);
    }
    double N = fval(a[0]), k = fval(a[1]); return a_float(exp(lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)));
}

/* ---- random numbers: xoshiro256** */
static uint64_t rng_s[4] = {0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull, 0x94D049BB133111EBull, 0x2545F4914F6CDD1Dull};
static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
uint64_t a_rand_u64(void) {
    uint64_t r = rotl(rng_s[1] * 5, 7) * 9, t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1]; rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3]; rng_s[2] ^= t; rng_s[3] = rotl(rng_s[3], 45);
    return r;
}
double a_rand_f(void) { return (double)(a_rand_u64() >> 11) * 0x1.0p-53; }
double a_rand_n(void) {
    static int have = 0; static double keep;
    if (have) { have = 0; return keep; }
    double u, v, s;
    do { u = 2 * a_rand_f() - 1; v = 2 * a_rand_f() - 1; s = u * u + v * v; } while (s >= 1 || s == 0);
    double m = sqrt(-2 * log(s) / s); keep = v * m; have = 1; return u * m;
}
void a_seed(uint64_t x) {
    for (int i = 0; i < 4; i++) { x += 0x9E3779B97F4A7C15ull; uint64_t z = x; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; rng_s[i] = z ^ (z >> 31); }
}
static uint64_t rng_saved[16][4]; static int rng_depth = 0;
static void seed_push(AV s) { if (rng_depth >= 16) a_error("RuntimeError", "seeds nested too deeply"); memcpy(rng_saved[rng_depth++], rng_s, sizeof rng_s); a_seed((uint64_t)(s.t == T_INT ? s.i : 0)); }
static void seed_pop(void) { if (rng_depth > 0) memcpy(rng_s, rng_saved[--rng_depth], sizeof rng_s); }
static AV b_seed_push(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; seed_push(n ? a[0] : a_int(0)); return A_NONE; }
static AV b_seed_pop(int n, AV *a, ADict *kw, AV self) { (void)n; (void)a; (void)kw; (void)self; seed_pop(); return A_NONE; }
static AV with_seed(AV (*f)(int, AV *, ADict *, AV), int n, AV *a, ADict *kw, AV self) {
    AV s; if (!(kw && a_dict_get(kw, a_str("seed"), &s)) || s.t == T_NONE) return f(n, a, NULL, self);
    seed_push(s); AV r = f(n, a, NULL, self); seed_pop(); return r;
}
static AV b_seed(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; a_seed((uint64_t)a[0].i); return A_NONE; }
AV (*a_tensor_random)(int kind, int n, AV *shape, AV lo, AV hi) = NULL;   /* set by the tensor runtime */
static AV rand_shaped(int kind, int n, AV *a, AV lo, AV hi) {
    if (!a_tensor_random) a_error("NotImplementedError", "tensors are not linked into this program");
    return a_tensor_random(kind, n, a, lo, hi);
}
static AV b_rand0(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; if (n) return rand_shaped(0, n, a, A_NONE, A_NONE); return a_float(a_rand_f()); }
static AV b_randn0(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; if (n) return rand_shaped(1, n, a, A_NONE, A_NONE); return a_float(a_rand_n()); }
static AV b_randint0(int n, AV *a, ADict *kw, AV self) {
    (void)kw; (void)self;
    if (n > 2) return rand_shaped(2, n - 2, a + 2, a[0], a[1]);
    int64_t lo = a[0].i, hi = a[1].i; if (hi <= lo) a_error("ValueError", "randint(): high must be above low");
    return a_int(lo + (int64_t)(a_rand_u64() % (uint64_t)(hi - lo)));
}
static AV b_permutation0(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    int64_t m = a[0].i; AList *l = new_list(m, O_LIST);
    for (int64_t i = 0; i < m; i++) l->v[i] = a_int(i);
    for (int64_t i = m - 1; i > 0; i--) { int64_t j = (int64_t)(a_rand_u64() % (uint64_t)(i + 1)); AV t = l->v[i]; l->v[i] = l->v[j]; l->v[j] = t; }
    return a_obj(l);
}

static AV b_rand(int n, AV *a, ADict *kw, AV self) { return with_seed(b_rand0, n, a, kw, self); }
static AV b_randn(int n, AV *a, ADict *kw, AV self) { return with_seed(b_randn0, n, a, kw, self); }
static AV b_randint(int n, AV *a, ADict *kw, AV self) { return with_seed(b_randint0, n, a, kw, self); }
static AV b_permutation(int n, AV *a, ADict *kw, AV self) { return with_seed(b_permutation0, n, a, kw, self); }
/* ---- files */
static AV b_exists(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; FILE *f = fopen(STR(a_to_str(a[0]))->s, "rb"); if (f) fclose(f); return a_bool(f != NULL); }
static AV b_read_text(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    const char *p = STR(a_to_str(a[0]))->s; FILE *f = fopen(p, "rb");
    if (!f) a_error("FileNotFoundError", "no such file: '%s'", p);
    SB b = {0}; char buf[65536]; size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) sb_put(&b, buf, (int64_t)k);
    fclose(f);
    if (b.n >= 3 && (unsigned char)b.p[0] == 0xEF && (unsigned char)b.p[1] == 0xBB && (unsigned char)b.p[2] == 0xBF) { memmove(b.p, b.p + 3, (size_t)b.n - 2); b.n -= 3; }
    return sb_done(&b);
}
static AV b_write_text(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)kw; (void)self;
    const char *p = STR(a_to_str(a[0]))->s; FILE *f = fopen(p, "wb");
    if (!f) a_error("OSError", "cannot write '%s'", p);
    AV s = a_to_str(a[1]); fwrite(STR(s)->s, 1, (size_t)STR(s)->len, f); fclose(f);
    return A_NONE;
}

/* ---- JSON */
static void json_out(SB *b, AV v, int indent, int depth) {
    char t[64];
    switch (v.t) {
    case T_NONE: sb_puts(b, "null"); return;
    case T_BOOL: sb_puts(b, v.i ? "true" : "false"); return;
    case T_INT: snprintf(t, sizeof t, "%lld", (long long)v.i); sb_puts(b, t); return;
    case T_FLOAT: if (isnan(v.f)) sb_puts(b, "NaN"); else if (isinf(v.f)) sb_puts(b, v.f > 0 ? "Infinity" : "-Infinity"); else { float_repr(v.f, t, sizeof t); sb_puts(b, t); } return;
    default: break;
    }
    #define NL() do { if (indent) { sb_puts(b, "\n"); for (int _i = 0; _i < indent * (depth + 1); _i++) sb_puts(b, " "); } } while (0)
    #define NLEND() do { if (indent) { sb_puts(b, "\n"); for (int _i = 0; _i < indent * depth; _i++) sb_puts(b, " "); } } while (0)
    switch (v.o->kind) {
    case O_STR: {
        AStr *s = STR(v); sb_puts(b, "\"");
        for (int64_t i = 0; i < s->len; i++) {
            unsigned char ch = (unsigned char)s->s[i];
            if (ch == '"') sb_puts(b, "\\\""); else if (ch == '\\') sb_puts(b, "\\\\"); else if (ch == '\n') sb_puts(b, "\\n");
            else if (ch == '\t') sb_puts(b, "\\t"); else if (ch == '\r') sb_puts(b, "\\r");
            else if (ch < 0x20) { snprintf(t, sizeof t, "\\u%04x", ch); sb_puts(b, t); }
            else sb_put(b, (char *)&ch, 1);
        }
        sb_puts(b, "\""); return;
    }
    case O_LIST: case O_TUPLE: {
        AList *l = LIST(v); sb_puts(b, "[");
        if (!l->len) { sb_puts(b, "]"); return; }
        for (int64_t i = 0; i < l->len; i++) { if (i) sb_puts(b, indent ? "," : ", "); NL(); json_out(b, l->v[i], indent, depth + 1); }
        NLEND(); sb_puts(b, "]"); return;
    }
    case O_DICT: {
        ADict *d = DICT(v); int first = 1; sb_puts(b, "{");
        if (!d->len) { sb_puts(b, "}"); return; }
        for (int64_t i = 0; i < d->nent; i++) {
            if (!d->e[i].used) continue;
            if (!first) sb_puts(b, indent ? "," : ", ");
            first = 0; NL();
            json_out(b, a_to_str(d->e[i].k), indent, depth + 1); sb_puts(b, ": "); json_out(b, d->e[i].v, indent, depth + 1);
        }
        NLEND(); sb_puts(b, "}"); return;
    }
    default: a_error("TypeError", "cannot write a %s as JSON", tname(v));
    }
    #undef NL
    #undef NLEND
}
typedef struct { const char *p; } JP;
static void jws(JP *j) { while (*j->p == ' ' || *j->p == '\n' || *j->p == '\t' || *j->p == '\r') j->p++; }
static AV json_in(JP *j) {
    jws(j);
    char c = *j->p;
    if (c == '{') {
        j->p++; int sp = a_ntmp; AV d = a_dict(); A_PUSH(d); jws(j);
        if (*j->p == '}') { j->p++; a_ntmp = sp; return d; }
        for (;;) {
            jws(j); AV k = json_in(j); A_PUSH(k); jws(j);
            if (*j->p++ != ':') a_error("ValueError", "JSON: expected ':'");
            AV v = json_in(j); a_dict_set(DICT(d), k, v); a_ntmp--;
            jws(j);
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == '}') { j->p++; break; }
            a_error("ValueError", "JSON: expected ',' or '}'");
        }
        a_ntmp = sp; return d;
    }
    if (c == '[') {
        j->p++; int sp = a_ntmp; AV l = a_list(0, NULL); A_PUSH(l); jws(j);
        if (*j->p == ']') { j->p++; a_ntmp = sp; return l; }
        for (;;) {
            AV v = json_in(j); a_list_append(LIST(l), v); jws(j);
            if (*j->p == ',') { j->p++; continue; }
            if (*j->p == ']') { j->p++; break; }
            a_error("ValueError", "JSON: expected ',' or ']'");
        }
        a_ntmp = sp; return l;
    }
    if (c == '"') {
        j->p++; SB b = {0};
        while (*j->p && *j->p != '"') {
            if (*j->p == '\\') {
                j->p++; char e = *j->p++;
                switch (e) {
                case 'n': sb_puts(&b, "\n"); break; case 't': sb_puts(&b, "\t"); break; case 'r': sb_puts(&b, "\r"); break;
                case 'b': sb_puts(&b, "\b"); break; case 'f': sb_puts(&b, "\f"); break;
                case 'u': {
                    unsigned cp = (unsigned)strtoul((char[]){j->p[0], j->p[1], j->p[2], j->p[3], 0}, NULL, 16); j->p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && j->p[0] == '\\' && j->p[1] == 'u') { unsigned lo = (unsigned)strtoul((char[]){j->p[2], j->p[3], j->p[4], j->p[5], 0}, NULL, 16); j->p += 6; cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                    AV ch = b_chr(1, (AV[]){a_int(cp)}, NULL, A_NONE); sb_put(&b, STR(ch)->s, STR(ch)->len); break;
                }
                default: sb_put(&b, &e, 1);
                }
            } else sb_put(&b, j->p++, 1);
        }
        if (*j->p != '"') a_error("ValueError", "JSON: unterminated string");
        j->p++; return sb_done(&b);
    }
    if (!strncmp(j->p, "true", 4)) { j->p += 4; return a_bool(1); }
    if (!strncmp(j->p, "false", 5)) { j->p += 5; return a_bool(0); }
    if (!strncmp(j->p, "null", 4)) { j->p += 4; return A_NONE; }
    if (!strncmp(j->p, "NaN", 3)) { j->p += 3; return a_float(NAN); }
    if (!strncmp(j->p, "Infinity", 8)) { j->p += 8; return a_float(INFINITY); }
    if (!strncmp(j->p, "-Infinity", 9)) { j->p += 9; return a_float(-INFINITY); }
    char *end; const char *s = j->p; int isf = 0;
    for (const char *q = s; *q && strchr("+-0123456789.eE", *q); q++) if (*q == '.' || *q == 'e' || *q == 'E') isf = 1;
    if (isf) { double d = strtod(s, &end); if (end == s) a_error("ValueError", "JSON: bad value"); j->p = end; return a_float(d); }
    long long i = strtoll(s, &end, 10); if (end == s) a_error("ValueError", "JSON: bad value near '%.10s'", s); j->p = end; return a_int(i);
}
static AV b_json_str(int n, AV *a, ADict *kw, AV self) { (void)self; AV ind = kwarg(kw, "indent", n > 1 ? a[1] : A_NONE); SB b = {0}; json_out(&b, a[0], ind.t == T_INT ? (int)ind.i : 0, 0); return sb_done(&b); }
static AV b_json_parse(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; JP j = {STR(a_to_str(a[0]))->s}; AV v = json_in(&j); jws(&j); if (*j.p) a_error("ValueError", "JSON: extra data"); return v; }
static AV b_load_json(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; AV t = b_read_text(1, a, NULL, self); int sp = a_ntmp; A_PUSH(t); AV r = b_json_parse(1, &t, NULL, self); a_ntmp = sp; return r; }
static AV b_save_json(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; SB b = {0}; json_out(&b, a[1], 2, 0); AV s = sb_done(&b); AV w[2] = {a[0], s}; return b_write_text(2, w, NULL, self); }

/* ---- the system */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
static int ncpu(void) { SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }
#else
#include <unistd.h>
static int ncpu(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1; }
#endif
#include <time.h>
static AV b_cpu_threads(int n, AV *a, ADict *kw, AV self) { (void)n; (void)a; (void)kw; (void)self; return a_int(ncpu()); }
static AV b_clock(int n, AV *a, ADict *kw, AV self) {
    (void)n; (void)a; (void)kw; (void)self;
    struct timespec ts; timespec_get(&ts, TIME_UTC); static const int16_t sec[7] = {0, 0, 60, 0, 0, 0, 0};
    return a_qty((double)ts.tv_sec + ts.tv_nsec * 1e-9, sec, NULL, 1.0);       /* seconds, as in apsi run */
}
static AV b_getenv(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; const char *v = getenv(STR(a_to_str(a[0]))->s); return v ? a_str(v) : (n > 1 ? a[1] : A_NONE); }
static AV b_write_out(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; for (int i = 0; i < n; i++) { AV s = a_to_str(a[i]); fwrite(STR(s)->s, 1, (size_t)STR(s)->len, stdout); } return A_NONE; }
static AV b_write_err(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; fflush(stdout); for (int i = 0; i < n; i++) { AV s = a_to_str(a[i]); fwrite(STR(s)->s, 1, (size_t)STR(s)->len, stderr); } return A_NONE; }
static AV b_flush_out(int n, AV *a, ADict *kw, AV self) { (void)n; (void)a; (void)kw; (void)self; fflush(stdout); return A_NONE; }
static AV b_isnan(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("isnan", 1)(n, a, kw, self); (void)n; (void)kw; (void)self; return a_bool(a[0].t == T_FLOAT && isnan(a[0].f)); }
static AV b_isfinite(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("isfinite", 1)(n, a, kw, self); (void)n; (void)kw; (void)self; return a_bool(a[0].t != T_FLOAT || isfinite(a[0].f)); }
static AV b_hex(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; char t[40]; long long v = a[0].i; snprintf(t, sizeof t, "%s0x%llx", v < 0 ? "-" : "", v < 0 ? -v : v); return a_str(t); }
static AV b_oct(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; char t[40]; long long v = a[0].i; snprintf(t, sizeof t, "%s0o%llo", v < 0 ? "-" : "", v < 0 ? -v : v); return a_str(t); }
static AV b_bin(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; AV s = a_format(a_int(a[0].i < 0 ? -a[0].i : a[0].i), "b"); SB b = {0}; sb_puts(&b, a[0].i < 0 ? "-0b" : "0b"); sb_put(&b, STR(s)->s, STR(s)->len); return sb_done(&b); }
static AV b_callable(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_bool(a[0].t == T_OBJ && (a[0].o->kind == O_FUNC || a[0].o->kind == O_BUILTIN || a[0].o->kind == O_BOUND || a[0].o->kind == O_CLASS)); }
static AV b_getattr(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; AV v = a_getattr(a[0], STR(a_to_str(a[1]))->s); if (v.t == 255) { if (n > 2) return a[2]; a_error("AttributeError", "'%s' object has no attribute '%s'", tname(a[0]), STR(a[1])->s); } return v; }
static AV b_hasattr(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_bool(a_getattr(a[0], STR(a_to_str(a[1]))->s).t != 255); }
static AV b_setattr(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; a_setattr(a[0], STR(a_to_str(a[1]))->s, a[2]); return A_NONE; }
static AV b_iter(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_iter(a[0]); }
static AV b_next(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; AV v; if (a_next(a[0], &v)) return v; if (n > 1) return a[1]; a_error("StopIteration", ""); }
static AV b_set_recursion_limit(int n, AV *a, ADict *kw, AV self) { (void)n; (void)a; (void)kw; (void)self; return A_NONE; }
static AV b_identity(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; if (n == 1 && !kw) return a[0]; return a_builtin("decorator", b_identity, A_NONE); }
static AV b_error(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; a_raise(n ? a_to_str(a[0]) : a_str("error")); }

static AV b_floor(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("floor", 1)(n, a, kw, self); (void)n; (void)kw; (void)self; if (a[0].t == T_INT) return a[0]; return a_int((int64_t)floor(fval(a[0]))); }
static AV b_ceil(int n, AV *a, ADict *kw, AV self) { if (n && a_is_tn(a[0])) return a_tensor_builtin("ceil", 1)(n, a, kw, self); (void)n; (void)kw; (void)self; if (a[0].t == T_INT) return a[0]; return a_int((int64_t)ceil(fval(a[0]))); }
static AV b_atan2(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_float(atan2(fval(a[0]), fval(a[1]))); }
static AV b_hypot(int n, AV *a, ADict *kw, AV self) { (void)n; (void)kw; (void)self; return a_float(hypot(fval(a[0]), fval(a[1]))); }
static AV b_exit(int n, AV *a, ADict *kw, AV self) { (void)kw; (void)self; fflush(stdout); exit(n ? (int)a[0].i : 0); }

typedef struct { const char *name; ABuiltinFn fn; } BEnt;
static const BEnt a_builtins[] = {
    {"print", b_print}, {"len", b_len}, {"range", b_range}, {"str", b_str}, {"repr", b_repr}, {"int", b_int},
    {"float", b_float}, {"bool", b_bool}, {"list", b_list}, {"tuple", b_tuple}, {"dict", b_dict}, {"set", b_set},
    {"abs", b_abs}, {"min", b_min}, {"max", b_max}, {"sum", b_sum}, {"sorted", b_sorted}, {"reversed", b_reversed},
    {"enumerate", b_enumerate}, {"zip", b_zip}, {"map", b_map}, {"filter", b_filter}, {"any", b_any}, {"all", b_all},
    {"round", b_round}, {"isinstance", b_isinstance}, {"ord", b_ord}, {"chr", b_chr}, {"type", b_type},
    {"input", b_input}, {"hash", b_hash}, {"divmod", b_divmod}, {"pow", b_pow}, {"sqrt", b_sqrt}, {"sin", b_sin},
    {"cos", b_cos}, {"tan", b_tan}, {"exp", b_exp}, {"log", b_log}, {"log10", b_log10}, {"log2", b_log2},
    {"atan", b_atan}, {"asin", b_asin}, {"acos", b_acos}, {"tanh", b_tanh}, {"sinh", b_sinh}, {"cosh", b_cosh},
    {"floor", b_floor}, {"ceil", b_ceil}, {"atan2", b_atan2}, {"hypot", b_hypot}, {"exit", b_exit},
    {"cbrt", b_cbrt}, {"erf", b_erf}, {"erfc", b_erfc}, {"expm1", b_expm1}, {"log1p", b_log1p}, {"lgamma", b_lgamma},
    {"gamma", b_gamma}, {"beta", b_beta}, {"factorial", b_factorial}, {"comb", b_comb}, {"seed", b_seed},
    {"rand", b_rand}, {"randn", b_randn}, {"randint", b_randint}, {"permutation", b_permutation}, {"seed_push", b_seed_push}, {"seed_pop", b_seed_pop},
    {"exists", b_exists}, {"read_text", b_read_text}, {"write_text", b_write_text}, {"json_str", b_json_str},
    {"json_parse", b_json_parse}, {"load_json", b_load_json}, {"save_json", b_save_json},
    {"cpu_threads", b_cpu_threads}, {"clock", b_clock}, {"getenv", b_getenv}, {"write_out", b_write_out},
    {"write_err", b_write_err}, {"flush_out", b_flush_out}, {"isnan", b_isnan}, {"isfinite", b_isfinite},
    {"hex", b_hex}, {"oct", b_oct}, {"bin", b_bin}, {"callable", b_callable}, {"getattr", b_getattr},
    {"hasattr", b_hasattr}, {"setattr", b_setattr}, {"iter", b_iter}, {"next", b_next},
    {"set_recursion_limit", b_set_recursion_limit}, {"error", b_error},
    {"native", b_identity}, {"gpu", b_identity}, {"memo", b_identity},
    {NULL, NULL}};

AV a_global_builtin(const char *name) {
    for (int i = 0; i < A_NEXC; i++) if (!strcmp(a_exc_names[i], name)) return a_exc_classes[i];
    if (!strcmp(name, "Error")) return a_exc_classes[31];
    if (!strcmp(name, "IOError") || !strcmp(name, "EnvironmentError")) return a_exc_classes[17];
    if (!strcmp(name, "pi")) return a_float(3.141592653589793);
    if (!strcmp(name, "tau")) return a_float(6.283185307179586);
    static const char *dtypes[][2] = {{"f64", "float64"}, {"fp64", "float64"}, {"f32", "float32"}, {"fp32", "float32"},
        {"f16", "float32"}, {"fp16", "float32"}, {"bf16", "float32"}, {"i64", "int64"}, {"i32", "int64"}, {"int8", "int64"},
        {"uint8", "int64"}, {"int4", "int64"}};
    for (size_t i = 0; i < sizeof dtypes / sizeof *dtypes; i++) if (!strcmp(name, dtypes[i][0])) return a_str(dtypes[i][1]);
    if (!strcmp(name, "e")) return a_float(2.718281828459045);
    if (!strcmp(name, "inf")) return a_float(INFINITY);
    if (!strcmp(name, "nan")) return a_float(NAN);
    for (const BEnt *b = a_builtins; b->name; b++) if (!strcmp(b->name, name)) return a_builtin(b->name, b->fn, A_NONE);
    ABuiltinFn tf = a_tensor_builtin(name, 0);
    if (tf) return a_builtin(name, tf, A_NONE);
    AV v; v.t = 255; v.i = 0; return v;
}

static AV exc_init_code(AFunc *f, int n, AV *args, ADict *kw) {
    (void)f; (void)kw;
    AInst *s = (AInst *)args[0].o;
    AV m = n > 1 ? a_to_str(args[1]) : a_str("");
    int sp = a_ntmp; A_PUSH(m);
    a_dict_set(s->attrs, a_str("args"), a_tuple(n - 1, args + 1));
    a_dict_set(s->attrs, a_str("message"), m);
    a_dict_set(s->attrs, a_str("kind"), a_obj(exc_kind(s->cls)));
    a_ntmp = sp;
    return A_NONE;
}
void a_init(int argc, char **argv) {
    (void)argc; (void)argv;
    a_exc = A_NONE;
    for (int i = 0; i < A_NEXC; i++) {
        AV base = a_exc_base[i] >= 0 ? a_exc_classes[a_exc_base[i]] : A_NONE;
        a_exc_classes[i] = a_class(a_exc_names[i], base, 0, 0, NULL);
    }
    a_tensor_init();
    a_dict_set(((AClass *)a_exc_classes[0].o)->attrs, a_str("__init__"), a_func("BaseException.__init__", exc_init_code, 0, 0));
}

/* =================================================================== helpers the compiler calls */
AV a_attrk(AV o, AV key) {
    if (a_is(o, O_INST)) {
        AInst *s = (AInst *)o.o; AV v;
        if (a_dict_get(s->attrs, key, &v)) return v;
        for (AClass *c = s->cls; c; c = c->base) if (a_dict_get(c->attrs, key, &v)) return a_is(v, O_FUNC) ? a_bound(o, v) : v;
    }
    return a_attr(o, STR(key)->s);
}
void a_setattrk(AV o, AV key, AV v) {
    if (a_is(o, O_INST)) { a_dict_set(((AInst *)o.o)->attrs, key, v); return; }
    a_setattr(o, STR(key)->s, v);
}
AV a_call_spread(AV f, AV args, AV kw) {
    int sp = a_ntmp; A_PUSH(args); A_PUSH(kw);
    AV r = a_call(f, (int)LIST(args)->len, LIST(args)->v, kw.t == T_NONE ? NULL : DICT(kw));
    a_ntmp = sp; return r;
}
AV a_inplace(int op, AV a, AV b) {
    if (op == OP_ADD && a_is(a, O_LIST)) { int sp = a_ntmp; AV it = a_iter(b), x; A_PUSH(it); while (a_next(it, &x)) a_list_append(LIST(a), x); a_ntmp = sp; return a; }
    if (op == OP_OR && a_is(a, O_DICT) && a_is(b, O_DICT)) { m_dict_update(1, &b, NULL, a); return a; }
    if (a_is(a, O_INST)) {
        static const char *in[] = {"", "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__imatmul__", "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"};
        if (op >= OP_ADD && op <= OP_SHR) { AV f = a_getattr(a, in[op]); if (f.t != 255) return a_call(f, 1, &b, NULL); }
    }
    switch (op) { case OP_ADD: return a_add(a, b); case OP_SUB: return a_sub(a, b); case OP_MUL: return a_mul(a, b); default: return a_binop(op, a, b); }
}
AV a_range_val(AV start, AV end, int inclusive) {
    if ((start.t == T_INT || start.t == T_BOOL) && (end.t == T_INT || end.t == T_BOOL)) return a_range(start.i, end.i + (inclusive ? 1 : 0), 1);
    if (!num(start) || !num(end)) a_error("TypeError", "a range needs numbers, not '%s' and '%s'", tname(start), tname(end));
    AList *l = new_list(0, O_LIST); AV x = start;
    while (a_lt(x, end) || (inclusive && a_eq(x, end))) { a_list_append(l, x); x = a_add(x, a_int(1)); }
    return a_obj(l);
}
void a_unpack_star(AV v, int before, int after, AV *out) {
    int sp = a_ntmp; AV l = collect(v); A_PUSH(l);
    int64_t n = LIST(l)->len;
    if (n < before + after) a_error("ValueError", "not enough values to unpack (expected at least %d, got %lld)", before + after, (long long)n);
    for (int i = 0; i < before; i++) out[i] = LIST(l)->v[i];
    out[before] = a_list(n - before - after, LIST(l)->v + before);
    for (int i = 0; i < after; i++) out[before + 1 + i] = LIST(l)->v[n - after + i];
    a_ntmp = sp;
}
AV a_gundef(const char *name) { a_error("NameError", "name '%s' is not defined", name); }
AV a_fmt_spec(AV v, AV spec) { return a_format(v, STR(a_to_str(spec))->s); }

AV a_module(const char *name, int n, const char **names, AV *vals) {
    char t[256]; snprintf(t, sizeof t, "module:%s", name);
    AV cls = a_class(t, A_NONE, 0, 0, NULL); int sp = a_ntmp; A_PUSH(cls);
    AV m = new_inst((AClass *)cls.o);
    for (int i = 0; i < n; i++) a_dict_set(((AInst *)m.o)->attrs, a_str(names[i]), vals[i]);
    a_ntmp = sp; return m;
}
