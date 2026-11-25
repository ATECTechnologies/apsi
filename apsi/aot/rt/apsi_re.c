/* apsi_re.c: regular expressions with Python's syntax and matching rules (a backtracking machine over code points),
   for `re` in native programs. Supported: literals and escapes, . ^ $ \A \Z \b \B, classes [..] with ranges and
   \d \w \s \D \W \S, groups ( ) (?: ) (?P<name> ) (?P=name) \1, alternation, * + ? {m} {m,} {m,n} and their lazy
   forms, look-ahead (?= ) (?! ), fixed-width look-behind (?<= ) (?<! ), inline flags (?aimsx), the flags
   I M S X A. Positions are code-point indices, as Python's str. */
#include <wctype.h>
#include "apsi_unicode.h"
static int rx_in_ranges(const uint32_t (*r)[2], int n, uint32_t c) { int lo = 0, hi = n - 1; while (lo <= hi) { int m = (lo + hi) / 2; if (c < r[m][0]) hi = m - 1; else if (c > r[m][1]) lo = m + 1; else return 1; } return 0; }
#define RX_IN(tab, c) rx_in_ranges(tab, (int)(sizeof tab / sizeof tab[0]), c)

enum { RX_I = 2, RX_M = 8, RX_S = 16, RX_X = 64, RX_A = 256 };
enum { N_CHAR, N_ANY, N_CLASS, N_BOL, N_EOL, N_BEGIN, N_END, N_WORDB, N_NWORDB, N_GROUP, N_CAT, N_ALT, N_REP, N_BACKREF, N_LOOK };
typedef struct RxRange { uint32_t lo, hi; } RxRange;
typedef struct RxClass { RxRange *r; int n, cap; int neg; int cats; } RxClass;       /* cats: bits d w s D W S */
typedef struct RxNode {
    int type; uint32_t c; int cls; int group; int min, max, greedy; int ahead, neg;
    struct RxNode **kids; int nk; int icase;
} RxNode;
enum { I_CHAR, I_ANY, I_ANYNL, I_CLASS, I_SPLIT, I_JMP, I_SAVE, I_BOL, I_EOL, I_BEGIN, I_END, I_WORDB, I_NWORDB, I_BACKREF,
       I_LOOK, I_MARK, I_CHECK, I_MATCH, I_COUNT0, I_COUNTINC };
typedef struct RxInst { int op; uint32_t c; int x, y; int icase; } RxInst;
typedef struct RxProg {
    RxInst *code; int n, cap;
    RxClass *cls; int ncls, capcls;
    struct RxProg **subs; int nsub, capsub;               /* look-around programs */
    int *sublen; int *subahead; int *subneg;
    int ngroups, nregs;
    char **names; int *nameidx; int nnames;
    int flags;
} RxProg;

/* ------------------------------------------------------------------------------------------- parsing */
typedef struct { const uint32_t *p; int n, i; int flags; RxProg *prog; int ngroups; char err[160]; } RxParse;
static void rx_fail(RxParse *ps, const char *msg) { if (!ps->err[0]) snprintf(ps->err, sizeof ps->err, "%s at position %d", msg, ps->i); }
static RxNode *rx_node(int type) { RxNode *n = (RxNode *)calloc(1, sizeof(RxNode)); n->type = type; n->max = -1; n->greedy = 1; return n; }
static void rx_add(RxNode *p, RxNode *k) { p->kids = (RxNode **)realloc(p->kids, sizeof(RxNode *) * (size_t)(p->nk + 1)); p->kids[p->nk++] = k; }
static void rx_free_node(RxNode *n) { if (!n) return; for (int i = 0; i < n->nk; i++) rx_free_node(n->kids[i]); free(n->kids); free(n); }
static int rx_newclass(RxProg *pr) {
    if (pr->ncls == pr->capcls) { pr->capcls = pr->capcls ? pr->capcls * 2 : 8; pr->cls = (RxClass *)realloc(pr->cls, sizeof(RxClass) * (size_t)pr->capcls); }
    memset(&pr->cls[pr->ncls], 0, sizeof(RxClass)); return pr->ncls++;
}
static void rx_range(RxClass *c, uint32_t lo, uint32_t hi) { if (c->n == c->cap) { c->cap = c->cap ? c->cap * 2 : 8; c->r = (RxRange *)realloc(c->r, sizeof(RxRange) * (size_t)c->cap); } c->r[c->n].lo = lo; c->r[c->n].hi = hi; c->n++; }
static int rx_peek(RxParse *ps) { return ps->i < ps->n ? (int)ps->p[ps->i] : -1; }
static void rx_skip_x(RxParse *ps) {                      /* verbose mode: whitespace and # comments */
    if (!(ps->flags & RX_X)) return;
    for (;;) {
        int c = rx_peek(ps);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') { ps->i++; continue; }
        if (c == '#') { while (ps->i < ps->n && ps->p[ps->i] != '\n') ps->i++; continue; }
        break;
    }
}
static int rx_escape_char(RxParse *ps, uint32_t e, uint32_t *out) {   /* \n \t \x41 é \0 ... -> 1 */
    switch (e) {
    case 'n': *out = '\n'; return 1; case 't': *out = '\t'; return 1; case 'r': *out = '\r'; return 1;
    case 'f': *out = '\f'; return 1; case 'v': *out = '\v'; return 1; case 'a': *out = 7; return 1;
    case 'x': case 'u': case 'U': {
        int len = e == 'x' ? 2 : e == 'u' ? 4 : 8; uint32_t v = 0;
        for (int k = 0; k < len; k++) { int c = rx_peek(ps); int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; if (d < 0) { rx_fail(ps, "incomplete escape"); return 1; } v = v * 16 + (uint32_t)d; ps->i++; }
        *out = v; return 1;
    }
    case '0': { uint32_t v = 0; int k = 0; while (k < 2 && rx_peek(ps) >= '0' && rx_peek(ps) <= '7') { v = v * 8 + (uint32_t)(rx_peek(ps) - '0'); ps->i++; k++; } *out = v; return 1; }
    default: break;
    }
    if ((e >= 'a' && e <= 'z') || (e >= 'A' && e <= 'Z')) return 0;
    *out = e; return 1;
}
static int rx_cat_bit(uint32_t e) { switch (e) { case 'd': return 1; case 'w': return 2; case 's': return 4; case 'D': return 8; case 'W': return 16; case 'S': return 32; } return 0; }
static RxNode *rx_alt(RxParse *ps);
static RxNode *rx_class(RxParse *ps) {
    RxNode *n = rx_node(N_CLASS); n->cls = rx_newclass(ps->prog); RxClass *c = &ps->prog->cls[n->cls];
    if (rx_peek(ps) == '^') { c->neg = 1; ps->i++; }
    int first = 1;
    while (ps->i < ps->n && (ps->p[ps->i] != ']' || first)) {
        first = 0; uint32_t lo = ps->p[ps->i++];
        if (lo == '\\') {
            if (ps->i >= ps->n) { rx_fail(ps, "bad escape (end of pattern)"); break; }
            uint32_t e = ps->p[ps->i++]; int cb = rx_cat_bit(e);
            if (cb) { c = &ps->prog->cls[n->cls]; c->cats |= cb; continue; }
            if (e == 'b') lo = 8; else if (!rx_escape_char(ps, e, &lo)) { rx_fail(ps, "bad escape"); break; }
        }
        uint32_t hi = lo;
        if (rx_peek(ps) == '-' && ps->i + 1 < ps->n && ps->p[ps->i + 1] != ']') {
            ps->i++; hi = ps->p[ps->i++];
            if (hi == '\\') { uint32_t e = ps->p[ps->i++]; if (!rx_escape_char(ps, e, &hi)) { rx_fail(ps, "bad character range"); break; } }
            if (hi < lo) { rx_fail(ps, "bad character range"); break; }
        }
        c = &ps->prog->cls[n->cls]; rx_range(c, lo, hi);
    }
    if (rx_peek(ps) != ']') rx_fail(ps, "unterminated character set"); else ps->i++;
    return n;
}
static RxNode *rx_atom(RxParse *ps) {
    rx_skip_x(ps);
    int c = rx_peek(ps);
    if (c < 0) return NULL;
    ps->i++;
    switch (c) {
    case '.': { RxNode *n = rx_node(N_ANY); n->icase = ps->flags & RX_S; return n; }
    case '^': return rx_node(N_BOL);
    case '$': return rx_node(N_EOL);
    case '[': return rx_class(ps);
    case '(': {
        RxNode *g = rx_node(N_GROUP); g->group = -1;
        if (rx_peek(ps) == '?') {
            ps->i++; int k = rx_peek(ps);
            if (k == ':') ps->i++;
            else if (k == 'P' && ps->i + 1 < ps->n && ps->p[ps->i + 1] == '<') {
                ps->i += 2; char name[64]; int nl = 0;
                while (ps->i < ps->n && ps->p[ps->i] != '>' && nl < 63) name[nl++] = (char)ps->p[ps->i++];
                name[nl] = 0; ps->i++;
                g->group = ++ps->ngroups; RxProg *pr = ps->prog;
                pr->names = (char **)realloc(pr->names, sizeof(char *) * (size_t)(pr->nnames + 1)); pr->nameidx = (int *)realloc(pr->nameidx, sizeof(int) * (size_t)(pr->nnames + 1));
                pr->names[pr->nnames] = strdup(name); pr->nameidx[pr->nnames++] = g->group;
            } else if (k == 'P' && ps->i + 1 < ps->n && ps->p[ps->i + 1] == '=') {
                ps->i += 2; char name[64]; int nl = 0;
                while (ps->i < ps->n && ps->p[ps->i] != ')' && nl < 63) name[nl++] = (char)ps->p[ps->i++];
                name[nl] = 0; ps->i++;
                RxNode *b = rx_node(N_BACKREF); b->group = -1;
                for (int q = 0; q < ps->prog->nnames; q++) if (!strcmp(ps->prog->names[q], name)) b->group = ps->prog->nameidx[q];
                if (b->group < 0) rx_fail(ps, "unknown group name");
                b->icase = ps->flags & RX_I; rx_free_node(g); return b;
            } else if (k == '=' || k == '!') { ps->i++; g->type = N_LOOK; g->ahead = 1; g->neg = k == '!'; }
            else if (k == '<' && ps->i + 1 < ps->n && (ps->p[ps->i + 1] == '=' || ps->p[ps->i + 1] == '!')) { g->type = N_LOOK; g->ahead = 0; g->neg = ps->p[ps->i + 1] == '!'; ps->i += 2; }
            else if (k == '#') { while (ps->i < ps->n && ps->p[ps->i] != ')') ps->i++; ps->i++; rx_free_node(g); return rx_node(N_CAT); }
            else {                                            /* inline flags (?aimsx) or (?i:...) */
                int f = 0; while (ps->i < ps->n && strchr("aiLmsux", (int)ps->p[ps->i])) { switch (ps->p[ps->i]) { case 'i': f |= RX_I; break; case 'm': f |= RX_M; break; case 's': f |= RX_S; break; case 'x': f |= RX_X; break; case 'a': f |= RX_A; break; } ps->i++; }
                if (rx_peek(ps) == ')') { ps->i++; ps->flags |= f; rx_free_node(g); return rx_node(N_CAT); }
                if (rx_peek(ps) == ':') { ps->i++; int saved = ps->flags; ps->flags |= f; RxNode *in = rx_alt(ps); ps->flags = saved; if (rx_peek(ps) != ')') rx_fail(ps, "missing ), unterminated subpattern"); else ps->i++; rx_add(g, in ? in : rx_node(N_CAT)); return g; }
                rx_fail(ps, "unknown extension"); return g;
            }
        } else g->group = ++ps->ngroups;
        RxNode *in = rx_alt(ps);
        if (rx_peek(ps) != ')') rx_fail(ps, "missing ), unterminated subpattern"); else ps->i++;
        rx_add(g, in ? in : rx_node(N_CAT));
        return g;
    }
    case '\\': {
        if (ps->i >= ps->n) { rx_fail(ps, "bad escape (end of pattern)"); return NULL; }
        uint32_t e = ps->p[ps->i++];
        int cb = rx_cat_bit(e);
        if (cb) { RxNode *n = rx_node(N_CLASS); n->cls = rx_newclass(ps->prog); ps->prog->cls[n->cls].cats = cb; return n; }
        if (e == 'A') return rx_node(N_BEGIN);
        if (e == 'Z') return rx_node(N_END);
        if (e == 'b') return rx_node(N_WORDB);
        if (e == 'B') return rx_node(N_NWORDB);
        if (e >= '1' && e <= '9') { int g = (int)(e - '0'); while (rx_peek(ps) >= '0' && rx_peek(ps) <= '9' && g * 10 + (rx_peek(ps) - '0') <= ps->ngroups) { g = g * 10 + (rx_peek(ps) - '0'); ps->i++; } RxNode *b = rx_node(N_BACKREF); b->group = g; b->icase = ps->flags & RX_I; if (g > ps->ngroups) rx_fail(ps, "invalid group reference"); return b; }
        uint32_t ch; if (!rx_escape_char(ps, e, &ch)) { rx_fail(ps, "bad escape"); return NULL; }
        RxNode *n = rx_node(N_CHAR); n->c = ch; n->icase = ps->flags & RX_I; return n;
    }
    case ')': case '|': ps->i--; return NULL;
    case '*': case '+': case '?': rx_fail(ps, "nothing to repeat"); return NULL;
    default: { RxNode *n = rx_node(N_CHAR); n->c = (uint32_t)c; n->icase = ps->flags & RX_I; return n; }
    }
}
static int rx_number(RxParse *ps, int *v) { int any = 0; *v = 0; while (rx_peek(ps) >= '0' && rx_peek(ps) <= '9') { *v = *v * 10 + (rx_peek(ps) - '0'); ps->i++; any = 1; } return any; }
static RxNode *rx_seq(RxParse *ps) {
    RxNode *cat = rx_node(N_CAT);
    for (;;) {
        rx_skip_x(ps);
        int c = rx_peek(ps);
        if (c < 0 || c == '|' || c == ')') break;
        RxNode *a = rx_atom(ps);
        if (!a) break;
        for (;;) {
            rx_skip_x(ps);
            int q = rx_peek(ps), mn = -1, mx = -1;
            if (q == '*') { mn = 0; mx = -1; ps->i++; }
            else if (q == '+') { mn = 1; mx = -1; ps->i++; }
            else if (q == '?') { mn = 0; mx = 1; ps->i++; }
            else if (q == '{') {
                int save = ps->i; ps->i++; int a1 = 0, b1 = 0;
                int h1 = rx_number(ps, &a1);
                if (rx_peek(ps) == '}') { if (!h1) { ps->i = save; break; } mn = mx = a1; ps->i++; }
                else if (rx_peek(ps) == ',') { ps->i++; int h2 = rx_number(ps, &b1); if (rx_peek(ps) != '}') { ps->i = save; break; } ps->i++; mn = h1 ? a1 : 0; mx = h2 ? b1 : -1; if (h2 && b1 < mn) { rx_fail(ps, "min repeat greater than max repeat"); } }
                else { ps->i = save; break; }
            } else break;
            if (a->type == N_BOL || a->type == N_EOL || a->type == N_BEGIN || a->type == N_END || a->type == N_WORDB || a->type == N_NWORDB) { rx_fail(ps, "nothing to repeat"); }
            RxNode *r = rx_node(N_REP); r->min = mn; r->max = mx;
            if (rx_peek(ps) == '?') { r->greedy = 0; ps->i++; } else if (rx_peek(ps) == '+') { ps->i++; }
            rx_add(r, a); a = r;
            if (mn == -1) break;
        }
        rx_add(cat, a);
    }
    return cat;
}
static RxNode *rx_alt(RxParse *ps) {
    RxNode *first = rx_seq(ps);
    if (rx_peek(ps) != '|') return first;
    RxNode *alt = rx_node(N_ALT); rx_add(alt, first);
    while (rx_peek(ps) == '|') { ps->i++; rx_add(alt, rx_seq(ps)); }
    return alt;
}

/* ------------------------------------------------------------------------------------------- code */
static int rx_emit(RxProg *p, int op, uint32_t c, int x, int y) {
    if (p->n == p->cap) { p->cap = p->cap ? p->cap * 2 : 64; p->code = (RxInst *)realloc(p->code, sizeof(RxInst) * (size_t)p->cap); }
    p->code[p->n].op = op; p->code[p->n].c = c; p->code[p->n].x = x; p->code[p->n].y = y; p->code[p->n].icase = 0; return p->n++;
}
static int rx_width(RxNode *n, int *w) {                    /* fixed width of a look-behind body: 1 if fixed */
    switch (n->type) {
    case N_CHAR: case N_ANY: case N_CLASS: *w = 1; return 1;
    case N_BOL: case N_EOL: case N_BEGIN: case N_END: case N_WORDB: case N_NWORDB: case N_LOOK: *w = 0; return 1;
    case N_GROUP: return rx_width(n->kids[0], w);
    case N_CAT: { int t = 0; for (int i = 0; i < n->nk; i++) { int k; if (!rx_width(n->kids[i], &k)) return 0; t += k; } *w = t; return 1; }
    case N_ALT: { int t = -1; for (int i = 0; i < n->nk; i++) { int k; if (!rx_width(n->kids[i], &k)) return 0; if (t >= 0 && k != t) return 0; t = k; } *w = t < 0 ? 0 : t; return 1; }
    case N_REP: { int k; if (n->min != n->max || !rx_width(n->kids[0], &k)) return 0; *w = k * n->min; return 1; }
    default: return 0;
    }
}
static RxProg *rx_new_prog(void) { RxProg *p = (RxProg *)calloc(1, sizeof(RxProg)); return p; }
static void rx_gen(RxProg *p, RxNode *n, int flags);
static void rx_gen_rep(RxProg *p, RxNode *n, int flags) {
    RxNode *k = n->kids[0]; int mn = n->min, mx = n->max;
    if (mx >= 0 && mx > 2000) mx = -1 > mx ? mx : mx;           /* very large counts still unroll */
    for (int i = 0; i < mn; i++) rx_gen(p, k, flags);
    if (mx < 0) {                                              /* k*: L1: SPLIT L2, L3; L2: MARK r; k; CHECK r -> L1 / L3 */
        int reg = p->nregs++;
        int split = rx_emit(p, I_SPLIT, 0, 0, 0);
        int body = p->n; rx_emit(p, I_MARK, 0, reg, 0);
        rx_gen(p, k, flags);
        int chk = rx_emit(p, I_CHECK, 0, reg, 0);
        p->code[chk].y = split;                                /* back to the loop when the body advanced */
        int out = p->n;
        p->code[chk].x = reg; p->code[chk].c = (uint32_t)out;
        if (n->greedy) { p->code[split].x = body; p->code[split].y = out; } else { p->code[split].x = out; p->code[split].y = body; }
    } else {
        int *fix = (int *)malloc(sizeof(int) * (size_t)(mx - mn + 1)); int nf = 0;
        for (int i = mn; i < mx; i++) {
            int split = rx_emit(p, I_SPLIT, 0, 0, 0); fix[nf++] = split;
            int body = p->n; rx_gen(p, k, flags);
            if (n->greedy) p->code[split].x = body; else p->code[split].y = body;
        }
        int out = p->n;
        for (int i = 0; i < nf; i++) { if (n->greedy) p->code[fix[i]].y = out; else p->code[fix[i]].x = out; }
        free(fix);
    }
}
static void rx_gen(RxProg *p, RxNode *n, int flags) {
    switch (n->type) {
    case N_CHAR: { int i = rx_emit(p, I_CHAR, n->c, 0, 0); p->code[i].icase = n->icase; break; }
    case N_ANY: rx_emit(p, n->icase ? I_ANY : I_ANYNL, 0, 0, 0); break;
    case N_CLASS: { int i = rx_emit(p, I_CLASS, 0, n->cls, 0); p->code[i].icase = flags & RX_I; break; }
    case N_BOL: rx_emit(p, I_BOL, 0, flags & RX_M, 0); break;
    case N_EOL: rx_emit(p, I_EOL, 0, flags & RX_M, 0); break;
    case N_BEGIN: rx_emit(p, I_BEGIN, 0, 0, 0); break;
    case N_END: rx_emit(p, I_END, 0, 0, 0); break;
    case N_WORDB: rx_emit(p, I_WORDB, 0, flags & RX_A, 0); break;
    case N_NWORDB: rx_emit(p, I_NWORDB, 0, flags & RX_A, 0); break;
    case N_GROUP:
        if (n->group > 0) rx_emit(p, I_SAVE, 0, 2 * n->group, 0);
        rx_gen(p, n->kids[0], flags);
        if (n->group > 0) rx_emit(p, I_SAVE, 0, 2 * n->group + 1, 0);
        break;
    case N_CAT: for (int i = 0; i < n->nk; i++) rx_gen(p, n->kids[i], flags); break;
    case N_ALT: {
        int *jumps = (int *)malloc(sizeof(int) * (size_t)n->nk); int nj = 0;
        for (int i = 0; i < n->nk; i++) {
            if (i < n->nk - 1) {
                int split = rx_emit(p, I_SPLIT, 0, p->n + 1, 0);
                rx_gen(p, n->kids[i], flags);
                jumps[nj++] = rx_emit(p, I_JMP, 0, 0, 0);
                p->code[split].y = p->n;
            } else rx_gen(p, n->kids[i], flags);
        }
        for (int i = 0; i < nj; i++) p->code[jumps[i]].x = p->n;
        free(jumps); break;
    }
    case N_REP: rx_gen_rep(p, n, flags); break;
    case N_BACKREF: { int i = rx_emit(p, I_BACKREF, 0, n->group, 0); p->code[i].icase = n->icase; break; }
    case N_LOOK: {
        RxProg *sub = rx_new_prog(); sub->cls = NULL;
        /* the sub-program shares the class table and the registers through its parent: generate inline into a
           separate code array that uses the same class indices */
        sub->cls = p->cls; sub->ncls = p->ncls; sub->capcls = p->capcls;
        sub->nregs = p->nregs;
        rx_gen(sub, n->kids[0], flags);
        rx_emit(sub, I_MATCH, 0, 0, 0);
        p->nregs = sub->nregs;
        int w = 0;
        if (!n->ahead && !rx_width(n->kids[0], &w)) w = -1;
        if (p->nsub == p->capsub) { p->capsub = p->capsub ? p->capsub * 2 : 4; p->subs = (RxProg **)realloc(p->subs, sizeof(RxProg *) * (size_t)p->capsub); p->sublen = (int *)realloc(p->sublen, sizeof(int) * (size_t)p->capsub); p->subahead = (int *)realloc(p->subahead, sizeof(int) * (size_t)p->capsub); p->subneg = (int *)realloc(p->subneg, sizeof(int) * (size_t)p->capsub); }
        p->subs[p->nsub] = sub; p->sublen[p->nsub] = w; p->subahead[p->nsub] = n->ahead; p->subneg[p->nsub] = n->neg;
        rx_emit(p, I_LOOK, 0, p->nsub, 0); p->nsub++;
        break;
    }
    }
}
static void rx_prog_free(RxProg *p, int owns_cls) {
    if (!p) return;
    for (int i = 0; i < p->nsub; i++) rx_prog_free(p->subs[i], 0);
    free(p->subs); free(p->sublen); free(p->subahead); free(p->subneg); free(p->code);
    if (owns_cls) { for (int i = 0; i < p->ncls; i++) free(p->cls[i].r); free(p->cls); for (int i = 0; i < p->nnames; i++) free(p->names[i]); free(p->names); free(p->nameidx); }
    free(p);
}
static void rx_fix_classes(RxProg *p, RxProg *root) {        /* the look-around programs see the final class table */
    for (int i = 0; i < p->nsub; i++) { p->subs[i]->cls = root->cls; p->subs[i]->ncls = root->ncls; rx_fix_classes(p->subs[i], root); }
}

/* ------------------------------------------------------------------------------------------- matching */
static uint32_t rx_lower(uint32_t c) { return c < 128 ? (uint32_t)((c >= 'A' && c <= 'Z') ? c + 32 : c) : (uint32_t)towlower((wint_t)c); }
static uint32_t rx_upper(uint32_t c) { return c < 128 ? (uint32_t)((c >= 'a' && c <= 'z') ? c - 32 : c) : (uint32_t)towupper((wint_t)c); }
static int rx_isword(uint32_t c, int ascii) { if (c < 128) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; return !ascii && RX_IN(rx_word_ranges, c); }
static int rx_isdigit(uint32_t c, int ascii) { if (c < 128) return c >= '0' && c <= '9'; return !ascii && RX_IN(rx_digit_ranges, c); }
static int rx_isspace(uint32_t c, int ascii) { if (c < 128) return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31); return !ascii && RX_IN(rx_space_ranges, c); }
static int rx_inclass(const RxClass *k, uint32_t c, int icase, int ascii) {
    int in = 0;
    if (k->cats) {
        if ((k->cats & 1) && rx_isdigit(c, ascii)) in = 1;
        if ((k->cats & 2) && rx_isword(c, ascii)) in = 1;
        if ((k->cats & 4) && rx_isspace(c, ascii)) in = 1;
        if ((k->cats & 8) && !rx_isdigit(c, ascii)) in = 1;
        if ((k->cats & 16) && !rx_isword(c, ascii)) in = 1;
        if ((k->cats & 32) && !rx_isspace(c, ascii)) in = 1;
    }
    for (int i = 0; !in && i < k->n; i++) {
        if (c >= k->r[i].lo && c <= k->r[i].hi) in = 1;
        else if (icase) { uint32_t l = rx_lower(c), u = rx_upper(c); if ((l >= k->r[i].lo && l <= k->r[i].hi) || (u >= k->r[i].lo && u <= k->r[i].hi)) in = 1; }
    }
    return k->neg ? !in : in;
}
typedef struct { int pc, pos, kind, slot, old; } RxBT;      /* kind 0: a branch to try, 1: restore slot to old */
typedef struct { const uint32_t *s; int n; int *slots; int nslots; int *regs; int flags; long steps; int adv, start; RxProg *root; } RxCtx;
static int rx_run(RxCtx *cx, RxProg *p, int pc, int pos, int *endpos);
static int rx_look(RxCtx *cx, RxProg *p, int k, int pos) {
    RxProg *sub = p->subs[k]; int ahead = p->subahead[k], neg = p->subneg[k], w = p->sublen[k], e;
    int *save = (int *)malloc(sizeof(int) * (size_t)cx->nslots); memcpy(save, cx->slots, sizeof(int) * (size_t)cx->nslots);
    int ok;
    if (ahead) ok = rx_run(cx, sub, 0, pos, &e);
    else {
        if (w < 0) ok = 0;                                   /* (checked at compile time) */
        else ok = pos - w >= 0 && rx_run(cx, sub, 0, pos - w, &e) && e == pos;
    }
    if (neg || !ok) memcpy(cx->slots, save, sizeof(int) * (size_t)cx->nslots);
    free(save);
    return neg ? !ok : ok;
}
static int rx_run(RxCtx *cx, RxProg *p, int pc, int pos, int *endpos) {
    int cap = 64, top = 0; RxBT *st = (RxBT *)malloc(sizeof(RxBT) * (size_t)cap);
    int ascii = cx->flags & RX_A;
#define PUSH(PC, POS, KIND, SLOT, OLD) do { if (top == cap) { cap *= 2; st = (RxBT *)realloc(st, sizeof(RxBT) * (size_t)cap); } st[top].pc = (PC); st[top].pos = (POS); st[top].kind = (KIND); st[top].slot = (SLOT); st[top].old = (OLD); top++; } while (0)
    for (;;) {
        if (++cx->steps > 200000000L) { free(st); a_error("RecursionError", "regular expression too complex (more than 200 million steps)"); }
        RxInst *in = &p->code[pc]; int fail = 0;
        switch (in->op) {
        case I_CHAR:
            if (pos < cx->n && (cx->s[pos] == in->c || (in->icase && rx_lower(cx->s[pos]) == rx_lower(in->c)))) { pos++; pc++; } else fail = 1; break;
        case I_ANY: if (pos < cx->n) { pos++; pc++; } else fail = 1; break;
        case I_ANYNL: if (pos < cx->n && cx->s[pos] != '\n') { pos++; pc++; } else fail = 1; break;
        case I_CLASS: if (pos < cx->n && rx_inclass(&p->cls[in->x], cx->s[pos], in->icase, ascii)) { pos++; pc++; } else fail = 1; break;
        case I_SPLIT: PUSH(in->y, pos, 0, 0, 0); pc = in->x; break;
        case I_JMP: pc = in->x; break;
        case I_SAVE: if (in->x < cx->nslots) { PUSH(0, 0, 1, in->x, cx->slots[in->x]); cx->slots[in->x] = pos; } pc++; break;
        case I_MARK: PUSH(0, 0, 2, in->x, cx->regs[in->x]); cx->regs[in->x] = pos; pc++; break;
        case I_CHECK: pc = cx->regs[in->x] == pos ? (int)in->c : in->y; break;   /* an empty pass ends the loop */
        case I_BOL: if (pos == 0 || (in->x && cx->s[pos - 1] == '\n')) pc++; else fail = 1; break;
        case I_EOL: if (pos == cx->n || (pos == cx->n - 1 && cx->s[pos] == '\n') || (in->x && cx->s[pos] == '\n')) pc++; else fail = 1; break;
        case I_BEGIN: if (pos == 0) pc++; else fail = 1; break;
        case I_END: if (pos == cx->n) pc++; else fail = 1; break;
        case I_WORDB: case I_NWORDB: {
            int a = pos > 0 && rx_isword(cx->s[pos - 1], in->x), b = pos < cx->n && rx_isword(cx->s[pos], in->x);
            int at = a != b; if ((in->op == I_WORDB) == at) pc++; else fail = 1; break;
        }
        case I_BACKREF: {
            int s0 = 2 * in->x < cx->nslots ? cx->slots[2 * in->x] : -1, e0 = 2 * in->x + 1 < cx->nslots ? cx->slots[2 * in->x + 1] : -1;
            if (s0 < 0 || e0 < 0) { fail = 1; break; }
            int len = e0 - s0; if (pos + len > cx->n) { fail = 1; break; }
            for (int k = 0; k < len; k++) if (cx->s[s0 + k] != cx->s[pos + k] && !(in->icase && rx_lower(cx->s[s0 + k]) == rx_lower(cx->s[pos + k]))) { fail = 1; break; }
            if (!fail) { pos += len; pc++; }
            break;
        }
        case I_LOOK: if (rx_look(cx, p, in->x, pos)) pc++; else fail = 1; break;
        case I_MATCH: if (cx->adv && p == cx->root && pos == cx->start) { fail = 1; break; } free(st); *endpos = pos; return 1;
        default: fail = 1;
        }
        if (fail) {
            for (;;) {
                if (top == 0) { free(st); return 0; }
                RxBT b = st[--top];
                if (b.kind == 1) { cx->slots[b.slot] = b.old; continue; }
                if (b.kind == 2) { cx->regs[b.slot] = b.old; continue; }
                pc = b.pc; pos = b.pos; break;
            }
        }
    }
#undef PUSH
}

/* ------------------------------------------------------------------------------------------- the cache and __re */
typedef struct { char *pat; int flags; RxProg *prog; } RxCache;
static RxCache rx_cache[64]; static int rx_ncache = 0, rx_next = 0;
static uint32_t *rx_decode(const char *s, int64_t len, int *n) {
    uint32_t *o = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(len + 1)); int k = 0; const unsigned char *p = (const unsigned char *)s, *e = p + len;
    while (p < e) {
        uint32_t c = *p++;
        if (c >= 0xF0 && p + 2 < e + 0) { c = ((c & 7) << 18) | ((uint32_t)(p[0] & 63) << 12) | ((uint32_t)(p[1] & 63) << 6) | (p[2] & 63); p += 3; }
        else if (c >= 0xE0 && p + 1 < e) { c = ((c & 15) << 12) | ((uint32_t)(p[0] & 63) << 6) | (p[1] & 63); p += 2; }
        else if (c >= 0xC0 && p < e) { c = ((c & 31) << 6) | (p[0] & 63); p += 1; }
        o[k++] = c;
    }
    *n = k; return o;
}
static RxProg *rx_compile(const char *pat, int64_t plen, int flags) {
    for (int i = 0; i < rx_ncache; i++) if (rx_cache[i].flags == flags && !strcmp(rx_cache[i].pat, pat)) return rx_cache[i].prog;
    RxParse ps; memset(&ps, 0, sizeof ps); int n; uint32_t *cp = rx_decode(pat, plen, &n);
    ps.p = cp; ps.n = n; ps.flags = flags; ps.prog = rx_new_prog();
    RxNode *root = rx_alt(&ps);
    if (!ps.err[0] && ps.i < ps.n) rx_fail(&ps, ps.p[ps.i] == ')' ? "unbalanced parenthesis" : "unexpected character");
    if (ps.err[0]) { char msg[200]; snprintf(msg, sizeof msg, "%s", ps.err); rx_free_node(root); rx_prog_free(ps.prog, 1); free(cp); a_error("ValueError", "re.error: %s", msg); }
    RxProg *p = ps.prog; p->ngroups = ps.ngroups; p->flags = ps.flags;
    rx_emit(p, I_SAVE, 0, 0, 0);
    rx_gen(p, root, ps.flags);
    rx_emit(p, I_SAVE, 0, 1, 0);
    rx_emit(p, I_MATCH, 0, 0, 0);
    rx_fix_classes(p, p);
    /* look-behind bodies must have one width */
    rx_free_node(root); free(cp);
    for (int i = 0; i < p->nsub; i++) if (!p->subahead[i] && p->sublen[i] < 0) { rx_prog_free(p, 1); a_error("ValueError", "re.error: look-behind requires fixed-width pattern"); }
    if (rx_ncache == 64) { free(rx_cache[rx_next].pat); rx_prog_free(rx_cache[rx_next].prog, 1); } else rx_ncache++;
    int slot = rx_ncache == 64 ? rx_next : rx_ncache - 1; if (rx_ncache == 64) rx_next = (rx_next + 1) % 64;
    rx_cache[slot].pat = strdup(pat); rx_cache[slot].flags = flags; rx_cache[slot].prog = p;
    return p;
}
/* __re(op, pattern, flags, text, pos, endpos)
     "info"      -> [groups, {name: index}]
     "search" / "match" / "fullmatch" -> [start0, end0, start1, end1, …] in code points, -1 for unmatched; none
     "len"       -> the text's length in code points */
BI(lb_re) {
    (void)self; (void)kw;
    const char *op = lib_str(a[0], "__re")->s; AStr *pat = lib_str(a[1], "__re"); int flags = (int)a[2].i;
    RxProg *p = rx_compile(pat->s, pat->len, flags);
    if (!strcmp(op, "info")) {
        AV d = a_dict(); int sp = a_ntmp; A_PUSH(d);
        for (int i = 0; i < p->nnames; i++) a_dict_set((ADict *)d.o, a_str(p->names[i]), a_int(p->nameidx[i]));
        AV pr[2] = {a_int(p->ngroups), d}; AV r = a_list(2, pr); a_ntmp = sp; return r;
    }
    AStr *text = lib_str(a[3], "__re"); int tn; uint32_t *s = rx_decode(text->s, text->len, &tn);
    if (!strcmp(op, "len")) { free(s); return a_int(tn); }
    int pos = n > 4 && a[4].t == T_INT ? (int)a[4].i : 0, endp = n > 5 && a[5].t == T_INT ? (int)a[5].i : tn;
    if (endp > tn) endp = tn; if (endp < 0) endp = 0; if (pos < 0) pos = 0;
    int nslots = 2 * (p->ngroups + 1);
    RxCtx cx; cx.s = s; cx.n = endp; cx.nslots = nslots; cx.flags = p->flags; cx.steps = 0; cx.root = p; cx.adv = n > 6 && a_truthy(a[6]);
    cx.slots = (int *)malloc(sizeof(int) * (size_t)nslots); cx.regs = (int *)calloc((size_t)(p->nregs + 1), sizeof(int));
    int anchored = !strcmp(op, "match") || !strcmp(op, "fullmatch"), found = 0;
    for (int start = pos; start <= endp; start++) {
        for (int i = 0; i < nslots; i++) cx.slots[i] = -1;
        int e; cx.start = start; if (start > pos) cx.adv = 0;
        if (rx_run(&cx, p, 0, start, &e)) {
            found = 1; break;
        }
        if (anchored) break;
    }
    AV r = A_NONE;
    if (found) {
        AV l = a_list(0, NULL); int sp = a_ntmp; A_PUSH(l);
        for (int i = 0; i < nslots; i++) a_list_append((AList *)l.o, a_int(cx.slots[i]));
        a_ntmp = sp; r = l;
    }
    free(cx.slots); free(cx.regs); free(s);
    return r;
}
