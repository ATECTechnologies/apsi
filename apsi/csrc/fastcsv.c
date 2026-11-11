/* Fast CSV kernels for `use dataframe` (compiled on first use when a C compiler is available). */
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>

/* Split a CSV buffer into cells. starts/ends are byte offsets; quoted[i] = 1 when the cell had quotes.
   Returns the number of cells, or -1 when cap was too small. *rows receives the number of records. */
long csv_index(const char *b, long n, char sep, long *starts, long *ends, unsigned char *quoted, long cap,
               long *rows) {
    long i = 0, c = 0, r = 0;
    while (i < n) {
        /* one record */
        for (;;) {
            if (c >= cap) return -1;
            if (i < n && b[i] == '"') {
                long s = ++i;
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
                long s = i;
                while (i < n && b[i] != sep && b[i] != '\n') i++;
                long e = i;
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

/* Parse column `col` of a table with `ncols` columns as doubles (empty → NaN). Returns 0 or the first bad row+1. */
long parse_doubles(const char *b, const long *starts, const long *ends, long nrows, long ncols, long col, double *out) {
    char tmp[64];
    for (long r = 0; r < nrows; r++) {
        long k = r * ncols + col, s = starts[k], e = ends[k], len = e - s;
        if (len <= 0) { out[r] = NAN; continue; }
        if (len >= 63) return r + 1;
        memcpy(tmp, b + s, len); tmp[len] = 0;
        char *end;
        out[r] = strtod(tmp, &end);
        if (*end != 0) return r + 1;
    }
    return 0;
}

/* Parse as 64-bit integers. Returns 0, or the first bad (or empty) row+1. */
long parse_longs(const char *b, const long *starts, const long *ends, long nrows, long ncols, long col, long long *out) {
    for (long r = 0; r < nrows; r++) {
        long k = r * ncols + col, s = starts[k], e = ends[k];
        if (e <= s) return r + 1;
        long long v = 0; int neg = 0; long i = s;
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
   Uses an open-addressing hash table of size `tsize` (a power of two ≥ 2 × nrows). Returns the number of ids. */
long dict_encode(const char *b, const long *starts, const long *ends, long nrows, long ncols, long col,
                 int *codes, long *first_row, long *table, long tsize) {
    long nuniq = 0, mask = tsize - 1;
    for (long i = 0; i < tsize; i++) table[i] = -1;
    for (long r = 0; r < nrows; r++) {
        long k = r * ncols + col, s = starts[k], e = ends[k];
        unsigned long long h = 1469598103934665603ULL;
        for (long i = s; i < e; i++) { h ^= (unsigned char)b[i]; h *= 1099511628211ULL; }
        long slot = (long)(h & mask);
        for (;;) {
            long id = table[slot];
            if (id < 0) {
                table[slot] = nuniq; first_row[nuniq] = r; codes[r] = (int)nuniq; nuniq++;
                break;
            }
            long fr = first_row[id] * ncols + col, fs = starts[fr], fe = ends[fr];
            if (fe - fs == e - s && memcmp(b + fs, b + s, e - s) == 0) { codes[r] = (int)id; break; }
            slot = (slot + 1) & mask;
        }
    }
    return nuniq;
}
