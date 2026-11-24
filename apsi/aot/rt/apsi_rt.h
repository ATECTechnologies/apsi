/* Apsi native runtime: the values of a compiled Apsi program, their memory (a mark-and-sweep collector that runs
 * only at safe points: function entry and loop back edges), exceptions (setjmp / longjmp), and the built-in
 * functions and methods. No Python anywhere: the program is machine code linked with this file. */
#ifndef APSI_RT_H
#define APSI_RT_H
#include <stdint.h>
#include <stddef.h>
#include <setjmp.h>

typedef enum { T_NONE = 0, T_BOOL, T_INT, T_FLOAT, T_OBJ } ATag;

struct AObj;
typedef struct AV {
    uint8_t t;
    union { int64_t i; double f; struct AObj *o; };
} AV;

typedef enum {
    O_STR = 1, O_LIST, O_TUPLE, O_DICT, O_SET, O_FUNC, O_CLASS, O_INST, O_BOUND, O_RANGE, O_ITER, O_CELL, O_BUILTIN,
    O_SLICE, O_QTY, O_GEN, O_TENSOR, O_NODE
} AKind;

typedef struct AObj { struct AObj *gcnext; uint8_t kind, mark; } AObj;

typedef struct AStr { AObj h; int64_t len; int64_t nchars; uint64_t hash; int ascii; char s[]; } AStr;
typedef struct AList { AObj h; int64_t len, cap; AV *v; } AList;           /* list and tuple */
typedef struct ADEnt { AV k, v; uint64_t hash; int used; } ADEnt;
typedef struct ADict { AObj h; int64_t len, nent, cap; ADEnt *e; int64_t *idx; int64_t icap; } ADict;  /* dict, set */
typedef struct ACell { AObj h; AV v; } ACell;
struct AFunc;
typedef AV (*AFnCode)(struct AFunc *self, int n, AV *args, ADict *kw);
typedef struct AFunc {
    AObj h; const char *name; AFnCode code;
    int ncells; ACell **cells;            /* the closure's captured variables */
    int ndef; AV *defaults;               /* defaults of the last ndef parameters */
} AFunc;
typedef struct AClass { AObj h; AStr *name; struct AClass *base; ADict *attrs; int nfields; AStr **fields; int is_struct; } AClass;
typedef struct AInst { AObj h; AClass *cls; ADict *attrs; } AInst;
typedef struct ABound { AObj h; AV self, fn; } ABound;
typedef struct ARange { AObj h; int64_t start, stop, step; } ARange;
typedef struct ASlice { AObj h; AV lo, hi, step; } ASlice;
typedef AV (*ABuiltinFn)(int n, AV *args, ADict *kw, AV self);
typedef struct ABuiltin { AObj h; const char *name; ABuiltinFn fn; AV self; } ABuiltin;
/* a quantity with a unit: the value in SI base units, the 7 SI exponents (in 60ths), the unit it shows in */
typedef struct AQty { AObj h; double v; int16_t dim[7]; struct AStr *dtext; double dscale; AV tv; } AQty;   /* tv: a tensor value (else none: v) */
typedef struct AUnitName { int16_t dim[7]; const char *name; } AUnitName;
void a_unit_names(const AUnitName *names, int n);
void a_dim_names(const AUnitName *names, int n);
AV a_qty(double v, const int16_t *dim, const char *text, double scale);   /* a dimensionless one is a plain float */
AV a_as_unit(AV v, const int16_t *dim, const char *text, double scale);
typedef struct AIter { AObj h; int kind; AV src; int64_t pos, end, step; AV aux; } AIter;

/* ------------------------------------------------------------------ values */
#define A_NONE ((AV){.t = T_NONE, .i = 0})
static inline AV a_int(int64_t i) { AV v; v.t = T_INT; v.i = i; return v; }
static inline AV a_float(double f) { AV v; v.t = T_FLOAT; v.f = f; return v; }
static inline AV a_bool(int b) { AV v; v.t = T_BOOL; v.i = b ? 1 : 0; return v; }
static inline AV a_obj(void *o) { AV v; v.t = T_OBJ; v.o = (AObj *)o; return v; }
static inline int a_is(AV v, AKind k) { return v.t == T_OBJ && v.o->kind == k; }

/* ------------------------------------------------------------------ memory */
typedef struct AFrame { struct AFrame *prev; int n; AV *slots; const char *name; int pos; } AFrame;   /* pos: line << 12 | column */
extern AFrame *a_frame;
void a_gc_safe(void);
void a_gc_full(void);                       /* collect now (the card ran out of memory) */                      /* a safe point: collects when enough was allocated */
void a_root(AV *slots, int n);             /* module globals */
void *a_alloc(size_t size, AKind kind);

/* ------------------------------------------------------------------ exceptions */
typedef struct AHandler { jmp_buf jb; struct AHandler *prev; AFrame *frame; int ntmp; } AHandler;
/* the runtime's own temporaries that must live across a call back into compiled code */
extern AV a_tmp[]; extern int a_ntmp;
#define A_PUSH(v) (a_tmp[a_ntmp] = (v), a_ntmp++ < 65535 ? (void)0 : a_overflow())
void a_overflow(void);
#define A_TRY(h) ((h).prev = a_handler, (h).frame = a_frame, (h).ntmp = a_ntmp, a_handler = &(h), setjmp((h).jb))
extern AHandler *a_handler;
extern AV a_exc;                           /* the exception being raised / handled (an instance) */
_Noreturn void a_raise(AV exc);
_Noreturn void a_error(const char *type, const char *fmt, ...);
int a_exc_matches(AV exc, AV cls);
extern const char *a_file, *a_source;

/* ------------------------------------------------------------------ objects */
AV a_str(const char *s);
AV a_strn(const char *s, int64_t n);
AV a_list(int64_t n, AV *items);
AV a_tuple(int64_t n, AV *items);
AV a_dict(void);
AV a_set(void);
AV a_range(int64_t a, int64_t b, int64_t step);
AV a_slice(AV lo, AV hi, AV step);
AV a_func(const char *name, AFnCode code, int ncells, int ndef);
ACell *a_cell(AV v);
AV a_class(const char *name, AV base, int is_struct, int nfields, const char **fields);
AV a_builtin(const char *name, ABuiltinFn fn, AV self);
void a_dict_set(ADict *d, AV k, AV v);
int a_dict_get(ADict *d, AV k, AV *out);
int a_dict_del(ADict *d, AV k);
void a_list_append(AList *l, AV v);

/* ------------------------------------------------------------------ operations */
int a_truthy(AV v);
AV a_binop(int op, AV a, AV b);
AV a_unop(int op, AV a);
int a_compare(int op, AV a, AV b);
int a_eq(AV a, AV b);
uint64_t a_hash(AV v);
int a_contains(AV container, AV item);
AV a_getitem(AV o, AV i);
void a_setitem(AV o, AV i, AV v);
void a_delitem(AV o, AV i);
AV a_getattr(AV o, const char *name);
void a_setattr(AV o, const char *name, AV v);
AV a_call(AV f, int n, AV *args, ADict *kw);
AV a_iter(AV o);
int a_next(AV it, AV *out);
int64_t a_len(AV o);
AV a_to_str(AV v);                         /* str(v) */
AV a_repr(AV v);
AV a_format(AV v, const char *spec);       /* f"{v:spec}" */
AV a_concat_strs(int n, AV *parts);
void a_unpack(AV v, int n, AV *out);
AV a_global_builtin(const char *name);     /* print, len, range, ... (NONE if unknown) */
int a_param(AFunc *f, int n, AV *args, ADict *kw, int i, const char *name, int ndefault_from, AV *out);
AV a_rest_args(int n, AV *args, int from);
ADict *a_rest_kw(ADict *kw, int nnames, const char **names);
int a_isinstance(AV v, AV cls);
AV a_super_attr(AV self, AV cls, const char *name);
void a_init(int argc, char **argv);
AV a_attr(AV o, const char *name);           /* o.name, AttributeError if missing */
AV a_attrk(AV o, AV key);                    /* o.key with a prebuilt name string */
void a_setattrk(AV o, AV key, AV v);
AV a_call_spread(AV f, AV args, AV kw);      /* f(*args, **kw): a tuple/list and a dict (or None) */
AV a_inplace(int op, AV a, AV b);            /* a op= b: lists and dicts change in place */
AV a_range_val(AV start, AV end, int inclusive);
void a_unpack_star(AV v, int before, int after, AV *out);   /* a, *b, c = v */
AV a_gundef(const char *name);
AV a_fmt_spec(AV v, AV spec);
/* tensors and differentiation (apsi_tensor.c) */
void a_note_alloc(size_t bytes);
void a_mark_value(AV v);
AV a_tensor_binop(int op, AV a, AV b);
AV a_tensor_neg(AV a);
AV a_tensor_binop_into(int op, AV a, AV b);
AV a_tensor_cmp(int op, AV a, AV b);
AV a_tensor_unary(int u, AV v);
AV a_tensor_where(AV c, AV a, AV b);
int a_tensor_truthy(AV v);
char *a_tensor_text(AV v, int top);
int64_t a_tensor_len(AV v);
AV a_tensor_getitem(AV v, AV i);
void a_tensor_setitem(AV v, AV i, AV x);
int a_tensor_eq(AV a, AV b);
double a_tensor_float(AV v);
AV a_tensor_attr(AV v, const char *name);
ABuiltinFn a_tensor_builtin(const char *name, int override_only);
void a_tensor_mark(AObj *o);
void a_tensor_free(AObj *o);
void a_tensor_init(void);
static inline int a_is_tn(AV v) { return v.t == T_OBJ && (v.o->kind == O_TENSOR || v.o->kind == O_NODE); }
AV a_cmp_slow(int op, AV a, AV b);
AV a_gen_new(AFunc *fn, AFnCode body, int n, AV *args, ADict *kw);   /* a generator function's call */
AV a_yield(AV v);                                                   /* yield v: the value sent back in */
int a_match_type(AV v, const char *t);
AV a_match_field(AV v, int i);
AV a_list_of(AV v);
AV a_cload(const char *path);
void *a_csym(AV lib, const char *name);
double a_c_f64(AV v); int64_t a_c_i64(AV v); const char *a_c_str(AV v);
void *a_tensor_ptr(AV v, int kind, const char *what);
AV a_module(const char *name, int n, const char **names, AV *vals);
uint64_t a_rand_u64(void); double a_rand_f(void); double a_rand_n(void); void a_seed(uint64_t x);
extern AV (*a_tensor_random)(int kind, int n, AV *shape, AV lo, AV hi);           /* o.name, AttributeError if missing */

enum { OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV, OP_FLOORDIV, OP_MOD, OP_POW, OP_MATMUL, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
       OP_NEG, OP_POS, OP_NOT, OP_INV,
       OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE, OP_IN, OP_NOTIN, OP_IS, OP_ISNOT };

/* ------------------------------------------------------------------ fast paths the compiler inlines */
static inline AV a_add(AV a, AV b) { int64_t r; if (a.t == T_INT && b.t == T_INT && !__builtin_add_overflow(a.i, b.i, &r)) return a_int(r); if (a.t == T_FLOAT && b.t == T_FLOAT) return a_float(a.f + b.f); return a_binop(OP_ADD, a, b); }
static inline AV a_sub(AV a, AV b) { int64_t r; if (a.t == T_INT && b.t == T_INT && !__builtin_sub_overflow(a.i, b.i, &r)) return a_int(r); if (a.t == T_FLOAT && b.t == T_FLOAT) return a_float(a.f - b.f); return a_binop(OP_SUB, a, b); }
static inline AV a_mul(AV a, AV b) { int64_t r; if (a.t == T_INT && b.t == T_INT && !__builtin_mul_overflow(a.i, b.i, &r)) return a_int(r); if (a.t == T_FLOAT && b.t == T_FLOAT) return a_float(a.f * b.f); return a_binop(OP_MUL, a, b); }
/* a op b where a is a temporary nobody else holds: a tensor's buffer is reused */
static inline AV a_binop_tmp(int op, AV a, AV b) {
    if (a.t == T_OBJ && a.o->kind == O_TENSOR) return a_tensor_binop_into(op, a, b);
    switch (op) { case OP_ADD: return a_add(a, b); case OP_SUB: return a_sub(a, b); case OP_MUL: return a_mul(a, b); default: return a_binop(op, a, b); }
}
static inline int a_lt(AV a, AV b) { if (a.t == T_INT && b.t == T_INT) return a.i < b.i; if (a.t == T_FLOAT && b.t == T_FLOAT) return a.f < b.f; return a_compare(OP_LT, a, b); }
static inline int a_le(AV a, AV b) { if (a.t == T_INT && b.t == T_INT) return a.i <= b.i; if (a.t == T_FLOAT && b.t == T_FLOAT) return a.f <= b.f; return a_compare(OP_LE, a, b); }
static inline int a_gt(AV a, AV b) { if (a.t == T_INT && b.t == T_INT) return a.i > b.i; if (a.t == T_FLOAT && b.t == T_FLOAT) return a.f > b.f; return a_compare(OP_GT, a, b); }
static inline int a_ge(AV a, AV b) { if (a.t == T_INT && b.t == T_INT) return a.i >= b.i; if (a.t == T_FLOAT && b.t == T_FLOAT) return a.f >= b.f; return a_compare(OP_GE, a, b); }
static inline int a_eqf(AV a, AV b) { if (a.t == T_INT && b.t == T_INT) return a.i == b.i; return a_eq(a, b); }
static inline AV a_cmpv(int op, AV a, AV b) {
    if (a.t == T_INT && b.t == T_INT) switch (op) { case OP_LT: return a_bool(a.i < b.i); case OP_LE: return a_bool(a.i <= b.i); case OP_GT: return a_bool(a.i > b.i); case OP_GE: return a_bool(a.i >= b.i); case OP_EQ: return a_bool(a.i == b.i); case OP_NE: return a_bool(a.i != b.i); default: break; }
    if (a.t == T_FLOAT && b.t == T_FLOAT) switch (op) { case OP_LT: return a_bool(a.f < b.f); case OP_LE: return a_bool(a.f <= b.f); case OP_GT: return a_bool(a.f > b.f); case OP_GE: return a_bool(a.f >= b.f); case OP_EQ: return a_bool(a.f == b.f); case OP_NE: return a_bool(a.f != b.f); default: break; }
    return a_cmp_slow(op, a, b);
}
static inline int a_test(AV v) { if (v.t == T_BOOL || v.t == T_INT) return v.i != 0; return a_truthy(v); }
static inline AV a_gload(AV v, const char *name) { return v.t == 255 ? a_gundef(name) : v; }
#endif
