/* Fast CSV kernels for `use dataframe` (compiled on first use when a C compiler is available). */
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
typedef long long i64;              /* 64-bit on every platform (Windows' long is 32-bit) */

/* Split a CSV buffer into cells. starts/ends are byte offsets; quoted[i] = 1 when the cell had quotes.
   Returns the number of cells, or -1 when cap was too small. *rows receives the number of records. */
i64 csv_index(const char *b, i64 n, char sep, i64 *starts, i64 *ends, unsigned char *quoted, i64 cap,
               i64 *rows) {
    i64 i = 0, c = 0, r = 0;
    while (i < n) {
        /* one record */
        for (;;) {
            if (c >= cap) return -1;
            if (i < n && b[i] == '"') {
                i64 s = ++i;
                while (i < n) {
                    if (b[i] == '"') {
                        if (i + 1 < n && b[i + 1] == '"') { i += 2; continue; }
                        break;
                    }
                    i++;
                }
                starts[c] = s; ends[c] = i; quoted[c] = 1; c++;
                if (i < n) i++;                     /* closing quote */
                while (i < n && b[i] != sep && b[i] != '\n') i++;
            } else {
                i64 s = i;
                while (i < n && b[i] != sep && b[i] != '\n') i++;
                i64 e = i;
                if (e > s && b[e - 1] == '\r') e--;
                starts[c] = s; ends[c] = e; quoted[c] = 0; c++;
            }
            if (i >= n) { r++; break; }
            if (b[i] == sep) { i++; continue; }
            i++;                                    /* newline */
            r++;
            break;
        }
    }
    *rows = r;
    return c;
}

/* A decimal with at most 15 significant digits and a power of ten up to 22: one correctly rounded operation
   (m * 10^k or m / 10^k), the same double strtod gives. Returns 1 when it parsed the whole cell, 0 otherwise. */
static const double P10[23] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                               1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
static int quick_double(const char *b, i64 s, i64 e, double *out) {
    i64 i = s; int neg = 0, digits = 0, frac = 0, seen = 0;
    unsigned long long m = 0;
    if (i < e && (b[i] == '-' || b[i] == '+')) { neg = b[i] == '-'; i++; }
    for (; i < e && (unsigned)(b[i] - '0') <= 9; i++) {
        if (digits || b[i] != '0') digits++;
        if (digits > 15) return 0;
        m = m * 10 + (unsigned)(b[i] - '0'); seen = 1;
    }
    if (i < e && b[i] == '.') {
        for (i++; i < e && (unsigned)(b[i] - '0') <= 9; i++) {
            if (digits || b[i] != '0') digits++;
            if (digits > 15) return 0;
            m = m * 10 + (unsigned)(b[i] - '0'); frac++; seen = 1;
        }
    }
    int ex = 0;
    if (i < e && (b[i] == 'e' || b[i] == 'E')) {
        int eneg = 0, ed = 0; i++;
        if (i < e && (b[i] == '-' || b[i] == '+')) { eneg = b[i] == '-'; i++; }
        for (; i < e && (unsigned)(b[i] - '0') <= 9; i++, ed++) if (ex < 100000) ex = ex * 10 + (b[i] - '0');
        if (!ed) return 0;
        if (eneg) ex = -ex;
    }
    if (i != e || !seen) return 0;
    int p = ex - frac;
    if (p < -22 || p > 22) return 0;
    double v = (double)m;
    v = p < 0 ? v / P10[-p] : v * P10[p];
    *out = neg ? -v : v;
    return 1;
}

/* Parse column `col` of a table with `ncols` columns as doubles (empty → NaN). Returns 0 or the first bad row+1. */
i64 parse_doubles(const char *b, const i64 *starts, const i64 *ends, i64 nrows, i64 ncols, i64 col, double *out) {
    char tmp[64];
    for (i64 r = 0; r < nrows; r++) {
        i64 k = r * ncols + col, s = starts[k], e = ends[k], len = e - s;
        if (len <= 0) { out[r] = NAN; continue; }
        if (quick_double(b, s, e, &out[r])) continue;
        if (len >= 63) return r + 1;
        memcpy(tmp, b + s, len); tmp[len] = 0;
        char *end;
        out[r] = strtod(tmp, &end);
        if (*end != 0) return r + 1;
    }
    return 0;
}

/* Parse as 64-bit integers. Returns 0, or the first bad (or empty) row+1. */
i64 parse_longs(const char *b, const i64 *starts, const i64 *ends, i64 nrows, i64 ncols, i64 col, long long *out) {
    for (i64 r = 0; r < nrows; r++) {
        i64 k = r * ncols + col, s = starts[k], e = ends[k];
        if (e <= s) return r + 1;
        long long v = 0; int neg = 0; i64 i = s;
        if (b[i] == '-' || b[i] == '+') { neg = b[i] == '-'; i++; }
        if (i >= e) return r + 1;
        for (; i < e; i++) {
            unsigned d = (unsigned)(b[i] - '0');
            if (d > 9) return r + 1;
            if (v > (9223372036854775807LL - d) / 10) return r + 1;
            v = v * 10 + d;
        }
        out[r] = neg ? -v : v;
    }
    return 0;
}

/* Dictionary-encode column `col`: codes[r] = id of the cell's text; first_row[id] = a row holding that text.
   Uses an open-addressing hash table of size `tsize` (a power of two). Returns the number of ids, or -1 when the
   table became half full (call again with a larger one: small tables stay in the processor's cache). */
i64 dict_encode(const char *b, const i64 *starts, const i64 *ends, i64 nrows, i64 ncols, i64 col,
                 int *codes, i64 *first_row, i64 *table, i64 tsize) {
    i64 nuniq = 0, mask = tsize - 1;
    for (i64 i = 0; i < tsize; i++) table[i] = -1;
    for (i64 r = 0; r < nrows; r++) {
        i64 k = r * ncols + col, s = starts[k], e = ends[k];
        unsigned long long h = 1469598103934665603ULL;
        for (i64 i = s; i < e; i++) { h ^= (unsigned char)b[i]; h *= 1099511628211ULL; }
        i64 slot = (i64)(h & mask);
        for (;;) {
            i64 id = table[slot];
            if (id < 0) {
                if (2 * (nuniq + 1) > tsize) return -1;
                table[slot] = nuniq; first_row[nuniq] = r; codes[r] = (int)nuniq; nuniq++;
                break;
            }
            i64 fr = first_row[id] * ncols + col, fs = starts[fr], fe = ends[fr];
            if (fe - fs == e - s && memcmp(b + fs, b + s, e - s) == 0) { codes[r] = (int)id; break; }
            slot = (slot + 1) & mask;
        }
    }
    return nuniq;
}
