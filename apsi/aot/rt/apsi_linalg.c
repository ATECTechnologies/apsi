/* Apsi native runtime: linear algebra (LU, Cholesky, QR, symmetric and general eigenproblems, SVD), the FFT, special
 * functions, and the convolution and pooling of neural networks with their gradients. Plain C on float64 (the
 * convolution in the tensors' own float32 / float64). Included by apsi_tensor.c, after the tensor core. */
#include <complex.h>
#undef I

static ATensor *lf64(AV v, const char *what) {           /* a host float64 copy of a matrix or vector argument */
    if (is_n(v)) a_error("TypeError", "%s() is not differentiable; take stop_gradient(x) or x.value first", what);
    if (a_is(v, O_QTY)) a_error("UnitError", "%s() takes plain numbers; strip the unit with x / unit first", what);
    AV t = as_tensor(v);
    if (TT(t)->dev) t = to_host(t);
    return TT(tcopy(TT(t), DT_F64));
}
static ATensor *lsquare(AV v, const char *what) {
    ATensor *A = lf64(v, what);
    if (A->nd != 2 || A->sh[0] != A->sh[1]) a_error("ShapeError", "%s() needs a square matrix, got shape %s", what, A->nd == 2 ? "(m, n) with m != n" : "that is not 2-D");
    return A;
}
static ATensor *lnew(int nd, int64_t d0, int64_t d1) { int64_t sh[2] = {d0, d1}; return t_new(DT_F64, nd, sh); }
static AV ltuple2(AV x, AV y) { AV p[2] = {x, y}; return a_tuple(2, p); }

/* ---- LU with partial pivoting, in place; 0: singular */
static int lu_dec(double *A, int64_t n, int64_t *piv, int *sign) {
    *sign = 1; double big = 0; for (int64_t i = 0; i < n * n; i++) if (fabs(A[i]) > big) big = fabs(A[i]);
    double tiny = big * 1e-300;
    for (int64_t k = 0; k < n; k++) {
        int64_t p = k; double m = fabs(A[k * n + k]);
        for (int64_t i = k + 1; i < n; i++) if (fabs(A[i * n + k]) > m) { m = fabs(A[i * n + k]); p = i; }
        piv[k] = p;
        if (m <= tiny || m == 0) return 0;
        if (p != k) { for (int64_t j = 0; j < n; j++) { double t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; } *sign = -*sign; }
        double inv = 1.0 / A[k * n + k];
        for (int64_t i = k + 1; i < n; i++) {
            double f = A[i * n + k] *= inv;
            if (f != 0) { double *ri = A + i * n, *rk = A + k * n; for (int64_t j = k + 1; j < n; j++) ri[j] -= f * rk[j]; }
        }
    }
    return 1;
}
static void lu_solve(const double *LU, int64_t n, const int64_t *piv, double *B, int64_t k) {     /* B: n x k, in place */
    for (int64_t i = 0; i < n; i++) if (piv[i] != i) for (int64_t j = 0; j < k; j++) { double t = B[i * k + j]; B[i * k + j] = B[piv[i] * k + j]; B[piv[i] * k + j] = t; }
    for (int64_t i = 0; i < n; i++) for (int64_t r = 0; r < i; r++) { double f = LU[i * n + r]; if (f != 0) for (int64_t j = 0; j < k; j++) B[i * k + j] -= f * B[r * k + j]; }
    for (int64_t i = n - 1; i >= 0; i--) {
        for (int64_t r = i + 1; r < n; r++) { double f = LU[i * n + r]; if (f != 0) for (int64_t j = 0; j < k; j++) B[i * k + j] -= f * B[r * k + j]; }
        double d = 1.0 / LU[i * n + i]; for (int64_t j = 0; j < k; j++) B[i * k + j] *= d;
    }
}
static AV la_solve(AV av, AV bv) {
    ATensor *A = lsquare(av, "solve"), *B = lf64(bv, "solve"); int64_t n = A->sh[0];
    if (B->nd < 1 || B->nd > 2 || B->sh[0] != n) a_error("ShapeError", "solve(a, b): b needs %lld rows", (long long)n);
    int64_t *piv = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n ? n : 1)); int sg;
    if (!lu_dec((double *)A->p, n, piv, &sg)) { free(piv); a_error("ValueError", "solve(): the matrix is singular (no unique solution); for a least-squares answer use lstsq(a, b)"); }
    lu_solve((double *)A->p, n, piv, (double *)B->p, B->nd == 2 ? B->sh[1] : 1); free(piv);
    return a_obj(B);
}
static AV la_inv(AV av) {
    ATensor *A = lsquare(av, "inv"); int64_t n = A->sh[0]; ATensor *I = t_zeros(DT_F64, 2, A->sh);
    for (int64_t i = 0; i < n; i++) ((double *)I->p)[i * n + i] = 1;
    int64_t *piv = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n ? n : 1)); int sg;
    if (!lu_dec((double *)A->p, n, piv, &sg)) { free(piv); a_error("ValueError", "inv(): the matrix is singular (it has no inverse); pinv(a) gives the pseudo-inverse"); }
    lu_solve((double *)A->p, n, piv, (double *)I->p, n); free(piv);
    return a_obj(I);
}
static AV la_det(AV av) {
    ATensor *A = lsquare(av, "det"); int64_t n = A->sh[0]; int64_t *piv = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n ? n : 1)); int sg;
    if (!lu_dec((double *)A->p, n, piv, &sg)) { free(piv); return a_float(0.0); }
    double d = sg; for (int64_t i = 0; i < n; i++) d *= ((double *)A->p)[i * n + i]; free(piv);
    return a_float(d);
}
static AV la_cholesky(AV av) {
    ATensor *A = lsquare(av, "cholesky"); int64_t n = A->sh[0]; double *a = (double *)A->p;
    ATensor *L = t_zeros(DT_F64, 2, A->sh); double *l = (double *)L->p;
    for (int64_t j = 0; j < n; j++) {
        double s = a[j * n + j]; for (int64_t k = 0; k < j; k++) s -= l[j * n + k] * l[j * n + k];
        if (!(s > 0)) a_error("ValueError", "cholesky(): the matrix is not positive definite (pivot %lld is %g)", (long long)j, s);
        double d = sqrt(s); l[j * n + j] = d;
        for (int64_t i = j + 1; i < n; i++) { double t = a[i * n + j]; for (int64_t k = 0; k < j; k++) t -= l[i * n + k] * l[j * n + k]; l[i * n + j] = t / d; }
    }
    return a_obj(L);
}
/* Householder QR: (Q m x k, R k x n), k = min(m, n) */
static AV la_qr(AV av) {
    ATensor *A = lf64(av, "qr"); if (A->nd != 2) a_error("ShapeError", "qr() needs a matrix");
    int64_t m = A->sh[0], n = A->sh[1], k = m < n ? m : n; double *a = (double *)A->p;
    double *vs = (double *)calloc((size_t)(m * k + 1), sizeof(double));
    for (int64_t j = 0; j < k; j++) {
        double nr = 0; for (int64_t i = j; i < m; i++) nr += a[i * n + j] * a[i * n + j]; nr = sqrt(nr);
        double *v = vs + j * m; if (nr == 0) continue;
        double alpha = a[j * n + j] > 0 ? -nr : nr;
        for (int64_t i = j; i < m; i++) v[i] = a[i * n + j]; v[j] -= alpha;
        double vn = 0; for (int64_t i = j; i < m; i++) vn += v[i] * v[i]; if (vn == 0) continue;
        for (int64_t c = j; c < n; c++) { double s = 0; for (int64_t i = j; i < m; i++) s += v[i] * a[i * n + c]; s = 2 * s / vn; for (int64_t i = j; i < m; i++) a[i * n + c] -= s * v[i]; }
        for (int64_t i = j; i < m; i++) v[i] /= sqrt(vn / 2);          /* v v^T = 2 v v^T / |v|^2 */
    }
    ATensor *Q = lnew(2, m, k), *R = t_zeros(DT_F64, 2, (int64_t[]){k, n}); double *q = (double *)Q->p, *r = (double *)R->p;
    for (int64_t i = 0; i < k; i++) for (int64_t j = i; j < n; j++) r[i * n + j] = a[i * n + j];
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < k; j++) q[i * k + j] = i == j;
    for (int64_t j = k - 1; j >= 0; j--) {
        double *v = vs + j * m;
        for (int64_t c = 0; c < k; c++) { double s = 0; for (int64_t i = j; i < m; i++) s += v[i] * q[i * k + c]; for (int64_t i = j; i < m; i++) q[i * k + c] -= s * v[i]; }
    }
    free(vs);
    return ltuple2(a_obj(Q), a_obj(R));
}
/* symmetric eigenproblem: Householder tridiagonalisation and the implicit QL method (as in EISPACK tred2 / tql2) */
static void tred2(int64_t n, double *V, double *d, double *e) {
#define v_(i, j) V[(i) * n + (j)]
    for (int64_t j = 0; j < n; j++) d[j] = v_(n - 1, j);
    for (int64_t i = n - 1; i > 0; i--) {
        double scale = 0, h = 0;
        for (int64_t k = 0; k < i; k++) scale += fabs(d[k]);
        if (scale == 0) {
            e[i] = d[i - 1];
            for (int64_t j = 0; j < i; j++) { d[j] = v_(i - 1, j); v_(i, j) = 0; v_(j, i) = 0; }
        } else {
            for (int64_t k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
            double f = d[i - 1], g = sqrt(h); if (f > 0) g = -g;
            e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
            for (int64_t j = 0; j < i; j++) e[j] = 0;
            for (int64_t j = 0; j < i; j++) {
                f = d[j]; v_(j, i) = f; g = e[j] + v_(j, j) * f;
                for (int64_t k = j + 1; k <= i - 1; k++) { g += v_(k, j) * d[k]; e[k] += v_(k, j) * f; }
                e[j] = g;
            }
            f = 0;
            for (int64_t j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
            double hh = f / (h + h);
            for (int64_t j = 0; j < i; j++) e[j] -= hh * d[j];
            for (int64_t j = 0; j < i; j++) {
                f = d[j]; g = e[j];
                for (int64_t k = j; k <= i - 1; k++) v_(k, j) -= f * e[k] + g * d[k];
                d[j] = v_(i - 1, j); v_(i, j) = 0;
            }
        }
        d[i] = h;
    }
    for (int64_t i = 0; i < n - 1; i++) {
        v_(n - 1, i) = v_(i, i); v_(i, i) = 1; double h = d[i + 1];
        if (h != 0) {
            for (int64_t k = 0; k <= i; k++) d[k] = v_(k, i + 1) / h;
            for (int64_t j = 0; j <= i; j++) {
                double g = 0;
                for (int64_t k = 0; k <= i; k++) g += v_(k, i + 1) * v_(k, j);
                for (int64_t k = 0; k <= i; k++) v_(k, j) -= g * d[k];
            }
        }
        for (int64_t k = 0; k <= i; k++) v_(k, i + 1) = 0;
    }
    for (int64_t j = 0; j < n; j++) { d[j] = v_(n - 1, j); v_(n - 1, j) = 0; }
    v_(n - 1, n - 1) = 1; e[0] = 0;
}
static int tql2(int64_t n, double *V, double *d, double *e) {
    for (int64_t i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    double f = 0, tst1 = 0, eps = 0x1p-52;
    for (int64_t l = 0; l < n; l++) {
        tst1 = fmax(tst1, fabs(d[l]) + fabs(e[l]));
        int64_t m = l;
        while (m < n - 1) { if (fabs(e[m]) <= eps * tst1) break; m++; }
        if (m > l) {
            int iter = 0;
            do {
                if (++iter > 200) return 0;
                double g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = hypot(p, 1.0);
                if (p < 0) r = -r;
                d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
                double dl1 = d[l + 1], h = g - d[l];
                for (int64_t i = l + 2; i < n; i++) d[i] -= h;
                f += h;
                p = d[m];
                double c = 1, c2 = c, c3 = c, el1 = e[l + 1], s = 0, s2 = 0;
                for (int64_t i = m - 1; i >= l; i--) {
                    c3 = c2; c2 = c; s2 = s;
                    g = c * e[i]; h = c * p; r = hypot(p, e[i]);
                    e[i + 1] = s * r; s = e[i] / r; c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for (int64_t k = 0; k < n; k++) { h = v_(k, i + 1); v_(k, i + 1) = s * v_(k, i) + c * h; v_(k, i) = c * v_(k, i) - s * h; }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p; d[l] = c * p;
            } while (fabs(e[l]) > eps * tst1);
        }
        d[l] += f; e[l] = 0;
    }
    for (int64_t i = 0; i < n - 1; i++) {                /* ascending, as NumPy */
        int64_t k = i; double p = d[i];
        for (int64_t j = i + 1; j < n; j++) if (d[j] < p) { k = j; p = d[j]; }
        if (k != i) { d[k] = d[i]; d[i] = p; for (int64_t j = 0; j < n; j++) { double t = v_(j, i); v_(j, i) = v_(j, k); v_(j, k) = t; } }
    }
    return 1;
#undef v_
}
static AV la_eigh(AV av) {
    ATensor *A = lsquare(av, "eigh"); int64_t n = A->sh[0]; double *a = (double *)A->p;
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < i; j++) { double s = 0.5 * (a[i * n + j] + a[j * n + i]); a[i * n + j] = a[j * n + i] = s; }
    ATensor *d = lnew(1, n, 0); double *e = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    if (n) { tred2(n, a, (double *)d->p, e); if (!tql2(n, a, (double *)d->p, e)) { free(e); a_error("ValueError", "eigh(): no convergence"); } }
    free(e);
    return ltuple2(a_obj(d), a_obj(A));
}
/* general eigenproblem: Hessenberg form, complex shifted QR for the eigenvalues, inverse iteration for the vectors */
typedef double complex cdbl;
static int c_lu(cdbl *A, int64_t n, int64_t *piv) {
    for (int64_t k = 0; k < n; k++) {
        int64_t p = k; double m = cabs(A[k * n + k]);
        for (int64_t i = k + 1; i < n; i++) if (cabs(A[i * n + k]) > m) { m = cabs(A[i * n + k]); p = i; }
        piv[k] = p;
        if (p != k) for (int64_t j = 0; j < n; j++) { cdbl t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; }
        if (A[k * n + k] == 0) A[k * n + k] = 1e-300;
        for (int64_t i = k + 1; i < n; i++) { cdbl f = A[i * n + k] /= A[k * n + k]; for (int64_t j = k + 1; j < n; j++) A[i * n + j] -= f * A[k * n + j]; }
    }
    return 1;
}
static void c_lu_solve(const cdbl *LU, int64_t n, const int64_t *piv, cdbl *b) {
    for (int64_t i = 0; i < n; i++) if (piv[i] != i) { cdbl t = b[i]; b[i] = b[piv[i]]; b[piv[i]] = t; }
    for (int64_t i = 0; i < n; i++) for (int64_t r = 0; r < i; r++) b[i] -= LU[i * n + r] * b[r];
    for (int64_t i = n - 1; i >= 0; i--) { for (int64_t r = i + 1; r < n; r++) b[i] -= LU[i * n + r] * b[r]; b[i] /= LU[i * n + i]; }
}
static AV la_eig(AV av) {
    ATensor *A = lsquare(av, "eig"); int64_t n = A->sh[0]; double *a = (double *)A->p;
    cdbl *H = (cdbl *)malloc(sizeof(cdbl) * (size_t)(n * n + 1)), *lam = (cdbl *)malloc(sizeof(cdbl) * (size_t)(n + 1));
    for (int64_t i = 0; i < n * n; i++) H[i] = a[i];
    for (int64_t k = 0; k + 2 < n + 0 && k < n - 2; k++) {                 /* to Hessenberg form (Householder) */
        double nr = 0; for (int64_t i = k + 1; i < n; i++) nr += creal(H[i * n + k]) * creal(H[i * n + k]); nr = sqrt(nr);
        if (nr == 0) continue;
        double *v = (double *)calloc((size_t)n, sizeof(double)); double alpha = creal(H[(k + 1) * n + k]) > 0 ? -nr : nr;
        for (int64_t i = k + 1; i < n; i++) v[i] = creal(H[i * n + k]); v[k + 1] -= alpha;
        double vn = 0; for (int64_t i = k + 1; i < n; i++) vn += v[i] * v[i];
        if (vn > 0) {
            for (int64_t c = 0; c < n; c++) { cdbl s = 0; for (int64_t i = k + 1; i < n; i++) s += v[i] * H[i * n + c]; s *= 2 / vn; for (int64_t i = k + 1; i < n; i++) H[i * n + c] -= s * v[i]; }
            for (int64_t r = 0; r < n; r++) { cdbl s = 0; for (int64_t i = k + 1; i < n; i++) s += H[r * n + i] * v[i]; s *= 2 / vn; for (int64_t i = k + 1; i < n; i++) H[r * n + i] -= s * v[i]; }
        }
        free(v);
    }
    double nrm = 0; for (int64_t i = 0; i < n * n; i++) nrm = fmax(nrm, cabs(H[i]));
    int64_t hi = n - 1; int iter = 0, total = 0;
    while (hi >= 0) {
        if (hi == 0) { lam[0] = H[0]; break; }
        int64_t l = hi;
        while (l > 0) { double s = cabs(H[(l - 1) * n + l - 1]) + cabs(H[l * n + l]); if (s == 0) s = nrm; if (cabs(H[l * n + l - 1]) <= 0x1p-52 * s) break; l--; }
        if (l == hi) { lam[hi] = H[hi * n + hi]; hi--; iter = 0; continue; }
        if (++total > 100 * (n + 10)) { free(H); free(lam); a_error("ValueError", "eig(): no convergence"); }
        iter++;
        cdbl a11 = H[(hi - 1) * n + hi - 1], a12 = H[(hi - 1) * n + hi], a21 = H[hi * n + hi - 1], a22 = H[hi * n + hi], mu;
        if (iter % 11 == 10) mu = a22 + 0.75 * cabs(a21);                    /* an exceptional shift */
        else { cdbl tr = a11 + a22, dt = a11 * a22 - a12 * a21, disc = csqrt(tr * tr / 4 - dt), m1 = tr / 2 + disc, m2 = tr / 2 - disc; mu = cabs(m1 - a22) < cabs(m2 - a22) ? m1 : m2; }
        for (int64_t i = l; i <= hi; i++) H[i * n + i] -= mu;
        cdbl *cs = (cdbl *)malloc(sizeof(cdbl) * 2 * (size_t)(hi - l + 1));
        for (int64_t k = l; k < hi; k++) {                                  /* QR by Givens rotations */
            cdbl x = H[k * n + k], y = H[(k + 1) * n + k]; double r = hypot(cabs(x), cabs(y));
            cdbl c = r == 0 ? 1 : x / r, s = r == 0 ? 0 : y / r;
            cs[2 * (k - l)] = c; cs[2 * (k - l) + 1] = s;
            for (int64_t j = k; j <= hi; j++) { cdbl t1 = H[k * n + j], t2 = H[(k + 1) * n + j]; H[k * n + j] = conj(c) * t1 + conj(s) * t2; H[(k + 1) * n + j] = -s * t1 + c * t2; }
        }
        for (int64_t k = l; k < hi; k++) {                                  /* then R Q */
            cdbl c = cs[2 * (k - l)], s = cs[2 * (k - l) + 1];
            int64_t top = k + 2 < hi ? k + 2 : hi;
            for (int64_t i = l; i <= top; i++) { cdbl t1 = H[i * n + k], t2 = H[i * n + k + 1]; H[i * n + k] = t1 * c + t2 * s; H[i * n + k + 1] = -t1 * conj(s) + t2 * conj(c); }
        }
        free(cs);
        for (int64_t i = l; i <= hi; i++) H[i * n + i] += mu;
    }
    /* eigenvectors of the original matrix by inverse iteration */
    cdbl *M = (cdbl *)malloc(sizeof(cdbl) * (size_t)(n * n + 1)), *x = (cdbl *)malloc(sizeof(cdbl) * (size_t)(n + 1)), *V = (cdbl *)malloc(sizeof(cdbl) * (size_t)(n * n + 1));
    int64_t *piv = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n + 1)); int cplx = 0;
    for (int64_t j = 0; j < n; j++) {
        cdbl L = lam[j] + (nrm > 0 ? nrm : 1) * 1e-10 * (1 + j * 1e-3);
        for (int64_t i = 0; i < n * n; i++) M[i] = a[i];
        for (int64_t i = 0; i < n; i++) { M[i * n + i] -= L; x[i] = 1.0 + 0.01 * (double)i; }
        c_lu(M, n, piv);
        for (int it = 0; it < 3; it++) {
            c_lu_solve(M, n, piv, x);
            double s = 0; for (int64_t i = 0; i < n; i++) s += creal(x[i] * conj(x[i])); s = sqrt(s); if (s == 0) s = 1;
            for (int64_t i = 0; i < n; i++) x[i] /= s;
        }
        int64_t big = 0; for (int64_t i = 1; i < n; i++) if (cabs(x[i]) > cabs(x[big])) big = i;   /* a real largest part */
        cdbl ph = cabs(x[big]) > 0 ? conj(x[big]) / cabs(x[big]) : 1; for (int64_t i = 0; i < n; i++) x[i] *= ph;
        for (int64_t i = 0; i < n; i++) V[i * n + j] = x[i];
        if (fabs(cimag(lam[j])) > 1e-12 * (nrm > 0 ? nrm : 1)) cplx = 1;
    }
    ATensor *wr = lnew(1, n, 0), *wi = lnew(1, n, 0), *vr = lnew(2, n, n), *vi = lnew(2, n, n);
    for (int64_t j = 0; j < n; j++) { ((double *)wr->p)[j] = creal(lam[j]); ((double *)wi->p)[j] = cplx ? cimag(lam[j]) : 0; }
    for (int64_t i = 0; i < n * n; i++) { ((double *)vr->p)[i] = creal(V[i]); ((double *)vi->p)[i] = cplx ? cimag(V[i]) : 0; }
    free(H); free(lam); free(M); free(x); free(V); free(piv);
    AV r[5] = {a_obj(wr), a_obj(vr), a_obj(wi), a_obj(vi), a_bool(cplx)};
    return a_tuple(5, r);
}
/* SVD by one-sided Jacobi rotations: (U m x k, s k, Vt k x n), s descending */
static void svd_core(ATensor *A, ATensor **Uo, ATensor **So, ATensor **Vto) {
    int64_t m = A->sh[0], n = A->sh[1]; int tr = m < n;
    int64_t M = tr ? n : m, N = tr ? m : n;                 /* work on an M x N matrix with M >= N */
    double *a = (double *)A->p, *W = (double *)malloc(sizeof(double) * (size_t)(M * N + 1)), *V = (double *)calloc((size_t)(N * N + 1), sizeof(double));
    /* W holds the columns as rows: W[j*M + i] = B[i][j] */
    for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < N; j++) W[j * M + i] = tr ? a[j * n + i] : a[i * n + j];
    for (int64_t j = 0; j < N; j++) V[j * N + j] = 1;           /* rows of V^T as rows */
    for (int sweep = 0; sweep < 80; sweep++) {
        double off = 0;
        for (int64_t p = 0; p < N - 1; p++) for (int64_t q = p + 1; q < N; q++) {
            double *wp = W + p * M, *wq = W + q * M, al = 0, be = 0, ga = 0;
            for (int64_t i = 0; i < M; i++) { al += wp[i] * wp[i]; be += wq[i] * wq[i]; ga += wp[i] * wq[i]; }
            if (fabs(ga) <= 1e-15 * sqrt(al * be) || ga == 0) continue;
            off = fmax(off, fabs(ga) / sqrt(al * be));
            double z = (be - al) / (2 * ga), t = (z >= 0 ? 1 : -1) / (fabs(z) + sqrt(1 + z * z)), c = 1 / sqrt(1 + t * t), s = c * t;
            for (int64_t i = 0; i < M; i++) { double x = wp[i], y = wq[i]; wp[i] = c * x - s * y; wq[i] = s * x + c * y; }
            double *vp = V + p * N, *vq = V + q * N;
            for (int64_t i = 0; i < N; i++) { double x = vp[i], y = vq[i]; vp[i] = c * x - s * y; vq[i] = s * x + c * y; }
        }
        if (off < 1e-15) break;
    }
    double *sv = (double *)malloc(sizeof(double) * (size_t)(N + 1)); int64_t *ord = (int64_t *)malloc(sizeof(int64_t) * (size_t)(N + 1));
    for (int64_t j = 0; j < N; j++) { double s = 0; for (int64_t i = 0; i < M; i++) s += W[j * M + i] * W[j * M + i]; sv[j] = sqrt(s); ord[j] = j; }
    for (int64_t i = 0; i < N; i++) for (int64_t j = i + 1; j < N; j++) if (sv[ord[j]] > sv[ord[i]]) { int64_t t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    double smax = N ? sv[ord[0]] : 0;
    /* left vectors: normalised columns; those of zero singular values completed to an orthonormal set */
    double *Uc = (double *)calloc((size_t)(M * N + 1), sizeof(double));
    for (int64_t r = 0; r < N; r++) {
        int64_t j = ord[r]; double *u = Uc + r * M;
        if (sv[j] > 1e-14 * smax && sv[j] > 0) for (int64_t i = 0; i < M; i++) u[i] = W[j * M + i] / sv[j];
        else {
            for (int64_t e = 0; e < M; e++) {
                for (int64_t i = 0; i < M; i++) u[i] = i == (e + r) % M;
                for (int pass = 0; pass < 2; pass++) for (int64_t q = 0; q < r; q++) { double d = 0; for (int64_t i = 0; i < M; i++) d += Uc[q * M + i] * u[i]; for (int64_t i = 0; i < M; i++) u[i] -= d * Uc[q * M + i]; }
                double s = 0; for (int64_t i = 0; i < M; i++) s += u[i] * u[i]; s = sqrt(s);
                if (s > 1e-8) { for (int64_t i = 0; i < M; i++) u[i] /= s; break; }
            }
        }
    }
    ATensor *U = lnew(2, m, N), *S = lnew(1, N, 0), *Vt = lnew(2, N, n); double *u = (double *)U->p, *vt = (double *)Vt->p;
    for (int64_t r = 0; r < N; r++) ((double *)S->p)[r] = sv[ord[r]];
    for (int64_t r = 0; r < N; r++) {
        double *uc = Uc + r * M, *vc = V + ord[r] * N;
        if (!tr) { for (int64_t i = 0; i < m; i++) u[i * N + r] = uc[i]; for (int64_t i = 0; i < n; i++) vt[r * n + i] = vc[i]; }
        else { for (int64_t i = 0; i < m; i++) u[i * N + r] = vc[i]; for (int64_t i = 0; i < n; i++) vt[r * n + i] = uc[i]; }
    }
    free(W); free(V); free(sv); free(ord); free(Uc);
    *Uo = U; *So = S; *Vto = Vt;
}
static AV la_svd(AV av) {
    ATensor *A = lf64(av, "svd"); if (A->nd != 2) a_error("ShapeError", "svd() needs a matrix");
    ATensor *U, *S, *Vt; svd_core(A, &U, &S, &Vt);
    AV r[3] = {a_obj(U), a_obj(S), a_obj(Vt)}; return a_tuple(3, r);
}
static ATensor *pinv_core(ATensor *A, double rcond, int64_t *rank) {
    ATensor *U, *S, *Vt; svd_core(A, &U, &S, &Vt);
    int64_t m = A->sh[0], n = A->sh[1], k = S->sh[0]; double *s = (double *)S->p, cut = (k ? s[0] : 0) * rcond; int64_t rk = 0;
    ATensor *P = t_zeros(DT_F64, 2, (int64_t[]){n, m}); double *p = (double *)P->p, *u = (double *)U->p, *vt = (double *)Vt->p;
    for (int64_t r = 0; r < k; r++) {
        if (!(s[r] > cut) || s[r] == 0) continue; rk++;
        double inv = 1 / s[r];
        for (int64_t i = 0; i < n; i++) { double vi = vt[r * n + i] * inv; if (vi != 0) for (int64_t j = 0; j < m; j++) p[i * m + j] += vi * u[j * k + r]; }
    }
    if (rank) *rank = rk;
    return P;
}
static double rcond_default(ATensor *A) { int64_t m = A->sh[0], n = A->sh[1]; return (double)(m > n ? m : n) * 0x1p-52; }
static AV la_pinv(AV av, AV rc) { ATensor *A = lf64(av, "pinv"); if (A->nd != 2) a_error("ShapeError", "pinv() needs a matrix"); return a_obj(pinv_core(A, rc.t == T_NONE ? 1e-15 : a_c_f64(rc), NULL)); }
static AV la_rank(AV av, AV tol) {
    ATensor *A = lf64(av, "matrix_rank"); if (A->nd != 2) a_error("ShapeError", "matrix_rank() needs a matrix");
    ATensor *U, *S, *Vt; svd_core(A, &U, &S, &Vt); double *s = (double *)S->p; int64_t k = S->sh[0], r = 0;
    double cut = tol.t == T_NONE ? (k ? s[0] : 0) * rcond_default(A) : a_c_f64(tol);
    for (int64_t i = 0; i < k; i++) if (s[i] > cut) r++;
    return a_int(r);
}
/* least squares: x minimising |a x - b|, (x, residuals, rank, s) as NumPy's lstsq */
static AV la_lstsq(AV av, AV bv) {
    ATensor *A = lf64(av, "lstsq"), *B = lf64(bv, "lstsq");
    if (A->nd != 2 || B->nd < 1 || B->nd > 2 || B->sh[0] != A->sh[0]) a_error("ShapeError", "lstsq(a, b): a is m x n, b has m rows");
    int64_t m = A->sh[0], n = A->sh[1], k = B->nd == 2 ? B->sh[1] : 1, rank = 0;
    ATensor *A2 = TT(tcopy(A, DT_F64)); ATensor *P = pinv_core(A2, rcond_default(A), &rank);
    ATensor *X = B->nd == 2 ? lnew(2, n, k) : lnew(1, n, 0); double *x = (double *)X->p, *p = (double *)P->p, *b = (double *)B->p, *a = (double *)A->p;
    for (int64_t i = 0; i < n; i++) for (int64_t c = 0; c < k; c++) { double s = 0; for (int64_t j = 0; j < m; j++) s += p[i * m + j] * b[j * k + c]; x[i * k + c] = s; }
    ATensor *R = lnew(1, (rank == n && m > n) ? k : 0, 0);
    if (rank == n && m > n) for (int64_t c = 0; c < k; c++) { double s = 0; for (int64_t j = 0; j < m; j++) { double t = -b[j * k + c]; for (int64_t i = 0; i < n; i++) t += a[j * n + i] * x[i * k + c]; s += t * t; } ((double *)R->p)[c] = s; }
    ATensor *U, *S, *Vt; svd_core(TT(tcopy(A, DT_F64)), &U, &S, &Vt);
    AV r[4] = {a_obj(X), a_obj(R), a_int(rank), a_obj(S)}; return a_tuple(4, r);
}
/* __la(op, a, b): the linalg module's machinery */
BI(lb_la) {
    (void)self; (void)kw; const char *op = lib_str(a[0], "linalg")->s; AV x = n > 1 ? a[1] : A_NONE, y = n > 2 ? a[2] : A_NONE;
    if (!strcmp(op, "solve")) return la_solve(x, y);
    if (!strcmp(op, "inv")) return la_inv(x);
    if (!strcmp(op, "det")) return la_det(x);
    if (!strcmp(op, "cholesky")) return la_cholesky(x);
    if (!strcmp(op, "qr")) return la_qr(x);
    if (!strcmp(op, "eigh")) return la_eigh(x);
    if (!strcmp(op, "eig")) return la_eig(x);
    if (!strcmp(op, "svd")) return la_svd(x);
    if (!strcmp(op, "pinv")) return la_pinv(x, y);
    if (!strcmp(op, "rank")) return la_rank(x, y);
    if (!strcmp(op, "lstsq")) return la_lstsq(x, y);
    a_error("ValueError", "linalg: unknown operation '%s'", op);
}

/* ------------------------------------------------------------------ FFT: radix 2, other lengths by Bluestein's algorithm */
static void fft2(double *re, double *im, int64_t n, int inv) {
    for (int64_t i = 1, j = 0; i < n; i++) {
        int64_t bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
        if (i < j) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int64_t len = 2; len <= n; len <<= 1) {
        double ang = (inv ? 2 : -2) * M_PI / (double)len;
        for (int64_t k = 0; k < len / 2; k++) {
            double wr = cos(ang * (double)k), wi = sin(ang * (double)k);
            for (int64_t i = k; i < n; i += len) {
                int64_t j = i + len / 2; double xr = re[j] * wr - im[j] * wi, xi = re[j] * wi + im[j] * wr;
                re[j] = re[i] - xr; im[j] = im[i] - xi; re[i] += xr; im[i] += xi;
            }
        }
    }
}
/* mixed radix (decimation in time by the smallest prime factor), twiddles from exact angles: as accurate as radix 2 */
static void fft_mixed(double *re, double *im, int64_t n, int inv, double *tr, double *ti) {
    if (n <= 1) return;
    if (!(n & (n - 1))) { fft2(re, im, n, inv); return; }
    int64_t p = 2; while (n % p) p++;
    int64_t m = n / p;
    for (int64_t r = 0; r < p; r++) for (int64_t j = 0; j < m; j++) { tr[r * m + j] = re[j * p + r]; ti[r * m + j] = im[j * p + r]; }
    for (int64_t r = 0; r < p; r++) fft_mixed(tr + r * m, ti + r * m, m, inv, re, im);   /* re / im: scratch of the right size */
    double sg = inv ? 2 * M_PI : -2 * M_PI;
    for (int64_t k = 0; k < n; k++) {
        double sr = 0, si = 0; int64_t km = k % m;
        for (int64_t r = 0; r < p; r++) {
            int64_t e = (r * k) % n; double c = cos(sg * (double)e / (double)n), s = sin(sg * (double)e / (double)n);
            double xr = tr[r * m + km], xi = ti[r * m + km];
            sr += xr * c - xi * s; si += xr * s + xi * c;
        }
        re[k] = sr; im[k] = si;
    }
}
static void fft_any(double *re, double *im, int64_t n, int inv) {
    if (n <= 1) return;
    if (!(n & (n - 1))) { fft2(re, im, n, inv); return; }
    int64_t big = 1, q = n; for (int64_t p = 2; p * p <= q; p++) while (q % p == 0) { q /= p; if (p > big) big = p; } if (q > big) big = q;
    if (big <= 64) {
        double *tr = (double *)malloc(sizeof(double) * (size_t)n), *ti = (double *)malloc(sizeof(double) * (size_t)n);
        double *sr = (double *)malloc(sizeof(double) * (size_t)n), *si = (double *)malloc(sizeof(double) * (size_t)n);
        memcpy(sr, re, sizeof(double) * (size_t)n); memcpy(si, im, sizeof(double) * (size_t)n);
        fft_mixed(sr, si, n, inv, tr, ti);
        memcpy(re, sr, sizeof(double) * (size_t)n); memcpy(im, si, sizeof(double) * (size_t)n);
        free(tr); free(ti); free(sr); free(si); return;
    }
    int64_t m = 1; while (m < 2 * n - 1) m <<= 1;
    double *ar = (double *)calloc((size_t)m, sizeof(double)), *ai = (double *)calloc((size_t)m, sizeof(double));
    double *br = (double *)calloc((size_t)m, sizeof(double)), *bi = (double *)calloc((size_t)m, sizeof(double));
    double *wr = (double *)malloc(sizeof(double) * (size_t)n), *wi = (double *)malloc(sizeof(double) * (size_t)n);
    for (int64_t k = 0; k < n; k++) {
        int64_t k2 = (int64_t)(((unsigned __int128)k * (unsigned __int128)k) % (unsigned __int128)(2 * n));
        double ang = (inv ? 1 : -1) * M_PI * (double)k2 / (double)n; wr[k] = cos(ang); wi[k] = sin(ang);
        ar[k] = re[k] * wr[k] - im[k] * wi[k]; ai[k] = re[k] * wi[k] + im[k] * wr[k];
        br[k] = wr[k]; bi[k] = -wi[k]; if (k) { br[m - k] = wr[k]; bi[m - k] = -wi[k]; }
    }
    fft2(ar, ai, m, 0); fft2(br, bi, m, 0);
    for (int64_t i = 0; i < m; i++) { double r = ar[i] * br[i] - ai[i] * bi[i], s = ar[i] * bi[i] + ai[i] * br[i]; ar[i] = r; ai[i] = s; }
    fft2(ar, ai, m, 1);
    for (int64_t k = 0; k < n; k++) { double r = ar[k] / (double)m, s = ai[k] / (double)m; re[k] = r * wr[k] - s * wi[k]; im[k] = r * wi[k] + s * wr[k]; }
    free(ar); free(ai); free(br); free(bi); free(wr); free(wi);
}
/* __fft(re, im, inverse, n_out_real): along the last axis; n_out_real > 0: an inverse real FFT of that length */
BI(lb_fft) {
    (void)self; (void)kw;
    ATensor *R = lf64(a[0], "fft"); ATensor *I = n > 1 && a[1].t != T_NONE ? lf64(a[1], "fft") : NULL;
    int inv = n > 2 && a_truthy(a[2]); int64_t nreal = n > 3 && a[3].t == T_INT ? a[3].i : 0;
    if (R->nd == 0) a_error("ShapeError", "fft() needs at least one axis");
    int64_t L = R->sh[R->nd - 1], rows = L ? R->n / L : 0;
    if (I && I->n != R->n) a_error("ShapeError", "fft(): the real and imaginary parts differ in shape");
    if (nreal > 0) {                                    /* irfft: a half spectrum of L to nreal real samples */
        int64_t sh[MAXD]; memcpy(sh, R->sh, sizeof(int64_t) * R->nd); sh[R->nd - 1] = nreal;
        ATensor *O = t_new(DT_F64, R->nd, sh); double *br = (double *)malloc(sizeof(double) * (size_t)nreal), *bi = (double *)malloc(sizeof(double) * (size_t)nreal);
        for (int64_t r = 0; r < rows; r++) {
            const double *xr = (double *)R->p + r * L, *xi = I ? (double *)I->p + r * L : NULL;
            for (int64_t k = 0; k < nreal; k++) {
                int64_t src = k <= nreal / 2 ? k : nreal - k; int cj = k > nreal / 2;
                double vr = src < L ? xr[src] : 0, vi = src < L && xi ? xi[src] : 0;
                if (k == 0 || (nreal % 2 == 0 && k == nreal / 2)) vi = 0;
                br[k] = vr; bi[k] = cj ? -vi : vi;
            }
            fft_any(br, bi, nreal, 1);
            for (int64_t k = 0; k < nreal; k++) ((double *)O->p)[r * nreal + k] = br[k] / (double)nreal;
        }
        free(br); free(bi); return a_obj(O);
    }
    ATensor *Or = TT(tcopy(R, DT_F64)), *Oi = I ? TT(tcopy(I, DT_F64)) : t_zeros(DT_F64, R->nd, R->sh);
    for (int64_t r = 0; r < rows; r++) {
        double *xr = (double *)Or->p + r * L, *xi = (double *)Oi->p + r * L;
        fft_any(xr, xi, L, inv);
        if (inv) for (int64_t k = 0; k < L; k++) { xr[k] /= (double)L; xi[k] /= (double)L; }
    }
    return ltuple2(a_obj(Or), a_obj(Oi));
}

/* ------------------------------------------------------------------ special functions */
static double sp_gammainc_p(double a, double x) {        /* the regularised lower incomplete gamma P(a, x) */
    if (x < 0 || a <= 0) return NAN;
    if (x == 0) return 0;
    double lg = lgamma(a);
    if (x < a + 1) {                                     /* the series */
        double ap = a, sum = 1 / a, del = sum;
        for (int i = 0; i < 1000; i++) { ap += 1; del *= x / ap; sum += del; if (fabs(del) < fabs(sum) * 1e-16) break; }
        return sum * exp(-x + a * log(x) - lg);
    }
    double b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;   /* the continued fraction (Lentz) for Q */
    for (int i = 1; i < 1000; i++) {
        double an = -i * (i - a); b += 2;
        d = an * d + b; if (fabs(d) < 1e-300) d = 1e-300;
        c = b + an / c; if (fabs(c) < 1e-300) c = 1e-300;
        d = 1 / d; double del = d * c; h *= del; if (fabs(del - 1) < 1e-16) break;
    }
    return 1 - exp(-x + a * log(x) - lg) * h;
}
static double sp_betacf(double a, double b, double x) {
    double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (fabs(d) < 1e-300) d = 1e-300; d = 1 / d; double h = d;
    for (int m = 1; m <= 3000; m++) {
        int m2 = 2 * m; double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (fabs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (fabs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (fabs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (fabs(c) < 1e-300) c = 1e-300; d = 1 / d;
        double del = d * c; h *= del; if (fabs(del - 1) < 1e-16) break;
    }
    return h;
}
static double sp_betainc(double a, double b, double x) {   /* the regularised incomplete beta I_x(a, b) */
    if (a <= 0 || b <= 0 || x < 0 || x > 1) return NAN;
    if (x == 0 || x == 1) return x;
    double lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log1p(-x);
    if (x < (a + 1) / (a + b + 2)) return exp(lbt) * sp_betacf(a, b, x) / a;
    return 1 - exp(lbt) * sp_betacf(b, a, 1 - x) / b;
}
static double sp_digamma(double x) {
    if (x <= 0 && floor(x) == x) return NAN;
    double r = 0;
    if (x < 0) return sp_digamma(1 - x) - M_PI / tan(M_PI * x);
    while (x < 6) { r -= 1 / x; x += 1; }
    double f = 1 / (x * x);
    return r + log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
}
static double sp_erfinv(double y) {
    if (y <= -1) return y == -1 ? -INFINITY : NAN;
    if (y >= 1) return y == 1 ? INFINITY : NAN;
    double w = -log((1 - y) * (1 + y)), x;              /* Giles' approximation, then Newton steps */
    if (w < 5) { w -= 2.5; x = 2.81022636e-08; x = 3.43273939e-07 + x * w; x = -3.5233877e-06 + x * w; x = -4.39150654e-06 + x * w; x = 0.00021858087 + x * w;
        x = -0.00125372503 + x * w; x = -0.00417768164 + x * w; x = 0.246640727 + x * w; x = 1.50140941 + x * w; }
    else { w = sqrt(w) - 3; x = -0.000200214257; x = 0.000100950558 + x * w; x = 0.00134934322 + x * w; x = -0.00367342844 + x * w; x = 0.00573950773 + x * w;
        x = -0.0076224613 + x * w; x = 0.00943887047 + x * w; x = 1.00167406 + x * w; x = 2.83297682 + x * w; }
    x *= y;
    for (int i = 0; i < 3; i++) { double err = erf(x) - y; x -= err / (2 / sqrt(M_PI) * exp(-x * x)); }
    return x;
}
/* inverses by safeguarded Newton steps inside a shrinking bracket */
static double sp_gammaincinv(double a, double p) {
    if (!(a > 0) || p < 0 || p > 1) return NAN;
    if (p == 0) return 0; if (p == 1) return INFINITY;
    double lo = 0, hi = a + 10 * sqrt(a) + 10; while (sp_gammainc_p(a, hi) < p) { lo = hi; hi *= 2; if (hi > 1e300) return INFINITY; }
    double x = a > 1 ? a * pow(1 - 1 / (9 * a) - M_SQRT2 * sp_erfinv(1 - 2 * p) / (3 * sqrt(a)), 3) : pow(p * tgamma(a + 1), 1 / a);
    if (!(x > lo && x < hi)) x = 0.5 * (lo + hi);
    double lg = lgamma(a);
    for (int i = 0; i < 200; i++) {
        double f = sp_gammainc_p(a, x) - p;
        if (f < 0) lo = x; else hi = x;
        double d = exp((a - 1) * log(x) - x - lg), nx = d > 0 ? x - f / d : 0.5 * (lo + hi);
        if (!(nx > lo && nx < hi)) nx = 0.5 * (lo + hi);
        if (fabs(nx - x) <= 1e-15 * fmax(1e-300, fabs(nx))) return nx;
        x = nx;
    }
    return x;
}
static double sp_betaincinv(double a, double b, double p) {
    if (!(a > 0) || !(b > 0) || p < 0 || p > 1) return NAN;
    if (p == 0) return 0; if (p == 1) return 1;
    double lo = 0, hi = 1, x = a / (a + b), lb = lgamma(a) + lgamma(b) - lgamma(a + b);
    for (int i = 0; i < 300; i++) {
        double f = sp_betainc(a, b, x) - p;
        if (f < 0) lo = x; else hi = x;
        double d = exp((a - 1) * log(x) + (b - 1) * log1p(-x) - lb), nx = d > 0 && isfinite(d) ? x - f / d : 0.5 * (lo + hi);
        if (!(nx > lo && nx < hi)) nx = 0.5 * (lo + hi);
        if (fabs(nx - x) <= 1e-15 * fmax(1e-300, fabs(nx)) || hi - lo < 1e-300) return nx;
        x = nx;
    }
    return x;
}
static double sp_apply(int f, double x, double p, double q) {
    switch (f) {
    case 0: return sp_gammainc_p(p, x);
    case 1: return 1 - sp_gammainc_p(p, x);
    case 2: return sp_betainc(p, q, x);
    case 3: return sp_digamma(x);
    case 4: return sp_erfinv(x);
    case 5: return M_SQRT2 * sp_erfinv(2 * x - 1);
    case 6: return 0.5 * erfc(-x / M_SQRT2);
    case 7: return lgamma(x);
    case 8: return sp_gammaincinv(p, x);
    case 9: return sp_betaincinv(p, q, x);
    default: return NAN;
    }
}
/* __special(name, x, p, q): elementwise on a number or a tensor */
BI(lb_special) {
    (void)self; (void)kw; const char *nm = lib_str(a[0], "special")->s;
    static const char *names[] = {"gammainc", "gammaincc", "betainc", "digamma", "erfinv", "ndtri", "ndtr", "gammaln", "gammaincinv", "betaincinv", NULL};
    int f = -1; for (int i = 0; names[i]; i++) if (!strcmp(names[i], nm)) f = i;
    if (f < 0) a_error("ValueError", "special: unknown function '%s'", nm);
    double p = n > 2 && a[2].t != T_NONE ? a_c_f64(a[2]) : 0, q = n > 3 && a[3].t != T_NONE ? a_c_f64(a[3]) : 0;
    AV x = a[1];
    if (x.t == T_INT || x.t == T_FLOAT || x.t == T_BOOL) return a_float(sp_apply(f, x.t == T_FLOAT ? x.f : (double)x.i, p, q));
    ATensor *X = lf64(x, nm); double *v = (double *)X->p;
    for (int64_t i = 0; i < X->n; i++) v[i] = sp_apply(f, v[i], p, q);
    return a_obj(X);
}

/* ------------------------------------------------------------------ convolution (im2col + the packed GEMM) and pooling */
typedef struct { const void *x; void *col; int64_t C, H, W, kh, kw, sh, sw, ph, pw, OH, OW; int f32; } Im2;
static void im2col(const Im2 *g, int transposed) {      /* col: [C*kh*kw, OH*OW], or [OH*OW, C*kh*kw] transposed */
    int64_t P = g->OH * g->OW, K = g->C * g->kh * g->kw;
    for (int64_t c = 0; c < g->C; c++) for (int64_t i = 0; i < g->kh; i++) for (int64_t j = 0; j < g->kw; j++) {
        int64_t row = (c * g->kh + i) * g->kw + j;
        for (int64_t oy = 0; oy < g->OH; oy++) {
            int64_t y = oy * g->sh - g->ph + i;
            for (int64_t ox = 0; ox < g->OW; ox++) {
                int64_t xx = ox * g->sw - g->pw + j, p = oy * g->OW + ox, at = transposed ? p * K + row : row * P + p;
                int in = y >= 0 && y < g->H && xx >= 0 && xx < g->W;
                if (g->f32) ((float *)g->col)[at] = in ? ((const float *)g->x)[(c * g->H + y) * g->W + xx] : 0.f;
                else ((double *)g->col)[at] = in ? ((const double *)g->x)[(c * g->H + y) * g->W + xx] : 0.0;
            }
        }
    }
}
static void col2im(const Im2 *g, void *dx) {             /* dx += the columns, back in place */
    int64_t P = g->OH * g->OW;
    for (int64_t c = 0; c < g->C; c++) for (int64_t i = 0; i < g->kh; i++) for (int64_t j = 0; j < g->kw; j++) {
        int64_t row = (c * g->kh + i) * g->kw + j;
        for (int64_t oy = 0; oy < g->OH; oy++) {
            int64_t y = oy * g->sh - g->ph + i; if (y < 0 || y >= g->H) continue;
            for (int64_t ox = 0; ox < g->OW; ox++) {
                int64_t xx = ox * g->sw - g->pw + j; if (xx < 0 || xx >= g->W) continue;
                int64_t at = (c * g->H + y) * g->W + xx, p = oy * g->OW + ox;
                if (g->f32) ((float *)dx)[at] += ((const float *)g->col)[row * P + p]; else ((double *)dx)[at] += ((const double *)g->col)[row * P + p];
            }
        }
    }
}
void a_parallel(void (*fn)(void *, int64_t), void *ctx, int64_t ntasks); int a_threads(void);
typedef struct { ATensor *X, *Wm, *Wt, *Y, *G, *DX; Im2 geo; int64_t O, base; int mode, par_gemm; void **bufs; void **dws; } ConvJob;
/* mode 0: forward; 1: backward to x; 2: backward to w; 3: both */
static void conv_one(ConvJob *J, int64_t b, int slot) {
    Im2 g = J->geo; int es = g.f32 ? 4 : 8;
    int64_t P = g.OH * g.OW, K = g.C * g.kh * g.kw, img = g.C * g.H * g.W;
    void *col = J->bufs[slot]; g.col = col; g.x = (char *)J->X->p + (size_t)(b * img) * es;
    if (J->mode == 0) {                                  /* Y[b] = W [O, K] @ col [K, P] */
        im2col(&g, 0);
        void *y = (char *)J->Y->p + (size_t)(b * J->O * P) * es;
        if (g.f32) a_gemm_f((float *)J->Wm->p, (float *)col, (float *)y, J->O, P, K, J->par_gemm, 0);
        else a_gemm_d((double *)J->Wm->p, (double *)col, (double *)y, J->O, P, K, J->par_gemm, 0);
        return;
    }
    const void *gb = (char *)J->G->p + (size_t)(b * J->O * P) * es;
    if (J->mode & 2) {                                   /* dW += G[b] [O, P] @ col^T [P, K] */
        im2col(&g, 1);
        if (g.f32) a_gemm_f((const float *)gb, (float *)col, (float *)J->dws[slot], J->O, K, P, J->par_gemm, 1);
        else a_gemm_d((const double *)gb, (double *)col, (double *)J->dws[slot], J->O, K, P, J->par_gemm, 1);
    }
    if (J->mode & 1) {                                   /* dcol [K, P] = W^T [K, O] @ G[b] [O, P], back to the image */
        if (g.f32) a_gemm_f((float *)J->Wt->p, (const float *)gb, (float *)col, K, P, J->O, J->par_gemm, 0);
        else a_gemm_d((double *)J->Wt->p, (const double *)gb, (double *)col, K, P, J->O, J->par_gemm, 0);
        col2im(&g, (char *)J->DX->p + (size_t)(b * img) * es);
    }
}
static void conv_ptask(void *p, int64_t t) { ConvJob *J = (ConvJob *)p; conv_one(J, J->base + t, (int)t); }
static void conv_run(ConvJob *J, int64_t B) {
    Im2 g = J->geo; int es = g.f32 ? 4 : 8; size_t colbytes = (size_t)(g.OH * g.OW) * (size_t)(g.C * g.kh * g.kw) * (size_t)es;
    size_t dwbytes = (size_t)(J->O * g.C * g.kh * g.kw) * (size_t)es;
    int T = a_threads(); if (T < 1) T = 1;
    J->par_gemm = B < T || colbytes * (size_t)T > ((size_t)1 << 30);      /* big images: the threads share one image's product */
    int nb = J->par_gemm ? 1 : T;
    J->bufs = (void **)calloc((size_t)nb, sizeof(void *)); J->dws = (void **)calloc((size_t)nb, sizeof(void *));
    for (int i = 0; i < nb; i++) {
        J->bufs[i] = malloc(colbytes ? colbytes : 1);
        if ((J->mode & 2)) J->dws[i] = calloc(dwbytes ? dwbytes : 1, 1);
        if (!J->bufs[i] || ((J->mode & 2) && !J->dws[i])) a_error("MemoryError", "conv2d: no memory for the image columns");
    }
    if (J->par_gemm) for (int64_t b = 0; b < B; b++) conv_one(J, b, 0);
    else for (int64_t s = 0; s < B; s += T) { J->base = s; a_parallel(conv_ptask, J, B - s < T ? B - s : T); }
    for (int i = 0; i < nb; i++) free(J->bufs[i]);
    free(J->bufs); J->bufs = NULL;
}
static int conv_pair(AV v, int64_t *a, int64_t *b, const char *what) {
    if (v.t == T_INT) { *a = *b = v.i; return 1; }
    if ((a_is(v, O_TUPLE) || a_is(v, O_LIST)) && ((AList *)v.o)->len == 2) { *a = ((AList *)v.o)->v[0].i; *b = ((AList *)v.o)->v[1].i; return 1; }
    a_error("TypeError", "conv2d: %s is a whole number or a pair (h, w)", what);
}
static ATensor *conv_host(AV v, int f32) {
    AV t = deepval(v); if (!is_t(t)) t = as_tensor(t);
    if (TT(t)->dev) t = to_host(t);
    return TT(t)->dt == (f32 ? DT_F32 : DT_F64) ? TT(t) : TT(tcopy(TT(t), f32 ? DT_F32 : DT_F64));
}
static void conv_geo(ATensor *X, ATensor *W, int64_t sh, int64_t sw, int64_t ph, int64_t pw, Im2 *g) {
    if (X->nd != 4 || W->nd != 4) a_error("ShapeError", "conv2d(x, w): x is (batch, channels, height, width), w is (out, in, kh, kw)");
    if (W->sh[1] != X->sh[1]) a_error("ShapeError", "conv2d: x has %lld channels but w expects %lld", (long long)X->sh[1], (long long)W->sh[1]);
    if (sh < 1 || sw < 1 || ph < 0 || pw < 0) a_error("ValueError", "conv2d: stride >= 1 and padding >= 0");
    memset(g, 0, sizeof *g);
    g->C = X->sh[1]; g->H = X->sh[2]; g->W = X->sh[3]; g->kh = W->sh[2]; g->kw = W->sh[3]; g->sh = sh; g->sw = sw; g->ph = ph; g->pw = pw;
    g->OH = (g->H + 2 * ph - g->kh) / sh + 1; g->OW = (g->W + 2 * pw - g->kw) / sw + 1;
    if (g->OH < 1 || g->OW < 1) a_error("ShapeError", "conv2d: the kernel (%lldx%lld) is larger than the padded image", (long long)g->kh, (long long)g->kw);
}
static ATensor *mat_t(ATensor *W, int64_t O, int64_t K) {          /* W [O, K] -> W^T [K, O] */
    ATensor *T = t_new(W->dt, 2, (int64_t[]){K, O});
    if (W->dt == DT_F32) { const float *w = (float *)W->p; float *t = (float *)T->p; for (int64_t o = 0; o < O; o++) for (int64_t k = 0; k < K; k++) t[k * O + o] = w[o * K + k]; }
    else { const double *w = (double *)W->p; double *t = (double *)T->p; for (int64_t o = 0; o < O; o++) for (int64_t k = 0; k < K; k++) t[k * O + o] = w[o * K + k]; }
    return T;
}
static AV conv_forward(AV xv, AV wv, AV opts) {
    AV x0 = deepval(xv), w0 = deepval(wv);
    int f32 = is_t(x0) && is_t(w0) && TT(x0)->dt == DT_F32 && TT(w0)->dt == DT_F32;
    ATensor *X = conv_host(xv, f32), *W = conv_host(wv, f32);
    AList *o = (AList *)opts.o; int64_t sh, sw, ph, pw; conv_pair(o->v[0], &sh, &sw, "stride"); conv_pair(o->v[1], &ph, &pw, "padding");
    ConvJob J; memset(&J, 0, sizeof J); conv_geo(X, W, sh, sw, ph, pw, &J.geo); J.geo.f32 = f32;
    int64_t B = X->sh[0]; J.O = W->sh[0]; J.X = X; J.Wm = W; J.mode = 0;
    J.Y = t_new(f32 ? DT_F32 : DT_F64, 4, (int64_t[]){B, J.O, J.geo.OH, J.geo.OW});
    conv_run(&J, B);
    return a_obj(J.Y);
}
static void conv_backward(AV g, AV a, AV b, AV opts) {
    AV x0 = deepval(a), w0 = deepval(b);
    int f32 = TT(x0)->dt == DT_F32 && TT(w0)->dt == DT_F32;
    ATensor *X = conv_host(a, f32), *W = conv_host(b, f32), *G = conv_host(g, f32);
    AList *o = (AList *)opts.o; int64_t sh, sw, ph, pw; conv_pair(o->v[0], &sh, &sw, "stride"); conv_pair(o->v[1], &ph, &pw, "padding");
    ConvJob J; memset(&J, 0, sizeof J); conv_geo(X, W, sh, sw, ph, pw, &J.geo); J.geo.f32 = f32;
    int64_t B = X->sh[0], K = J.geo.C * J.geo.kh * J.geo.kw; J.O = W->sh[0]; J.X = X; J.Wm = W; J.G = G;
    J.mode = (is_n(a) ? 1 : 0) | (is_n(b) ? 2 : 0); if (!J.mode) return;
    int sp = a_ntmp;
    if (J.mode & 1) { J.Wt = mat_t(W, J.O, K); A_PUSH(a_obj(J.Wt)); J.DX = t_zeros(f32 ? DT_F32 : DT_F64, 4, X->sh); A_PUSH(a_obj(J.DX)); }
    conv_run(&J, B);
    if (J.mode & 2) {
        int nb = J.par_gemm ? 1 : a_threads(); ATensor *DW = t_zeros(f32 ? DT_F32 : DT_F64, 4, W->sh); A_PUSH(a_obj(DW));
        for (int i = 0; i < nb; i++) {
            if (f32) { float *d = (float *)DW->p, *s = (float *)J.dws[i]; for (int64_t k = 0; k < DW->n; k++) d[k] += s[k]; }
            else { double *d = (double *)DW->p, *s = (double *)J.dws[i]; for (int64_t k = 0; k < DW->n; k++) d[k] += s[k]; }
            free(J.dws[i]);
        }
        acc_grad(b, a_obj(DW));
    }
    free(J.dws);
    if (J.mode & 1) acc_grad(a, a_obj(J.DX));
    a_ntmp = sp;
}
/* __conv2d(x, w, stride, padding): differentiable in x and w (the bias is added outside, by the tensor sum) */
BI(lb_conv2d) {
    (void)self; (void)kw; if (n < 2) a_error("TypeError", "conv2d(x, w, stride, padding)");
    AV opts[2] = {n > 2 ? a[2] : a_int(1), n > 3 ? a[3] : a_int(0)}; AV ot = a_tuple(2, opts); int sp = a_ntmp; A_PUSH(ot);
    AV y = conv_forward(a[0], a[1], ot); A_PUSH(y);
    AV r = is_n(a[0]) || is_n(a[1]) ? node_new(y, N_CONV2D, a[0], a[1], ot) : y;
    a_ntmp = sp; return r;
}
/* max pooling: the winners' positions kept for the gradient */
static AV pool_forward(AV xv, int64_t k, int64_t s, ATensor **argo) {
    AV x0 = deepval(xv); int f32 = is_t(x0) && TT(x0)->dt == DT_F32; ATensor *X = conv_host(xv, f32);
    if (X->nd != 4) a_error("ShapeError", "max_pool2d(x): x is (batch, channels, height, width)");
    if (k < 1 || s < 1) a_error("ValueError", "max_pool2d: k and stride >= 1");
    int64_t B = X->sh[0], C = X->sh[1], H = X->sh[2], W = X->sh[3], OH = (H - k) / s + 1, OW = (W - k) / s + 1;
    if (OH < 1 || OW < 1) a_error("ShapeError", "max_pool2d: the window is larger than the image");
    ATensor *Y = t_new(f32 ? DT_F32 : DT_F64, 4, (int64_t[]){B, C, OH, OW}), *I = t_new(DT_I64, 4, Y->sh);
    int64_t *ix = (int64_t *)I->p;
    for (int64_t bc = 0; bc < B * C; bc++) for (int64_t oy = 0; oy < OH; oy++) for (int64_t ox = 0; ox < OW; ox++) {
        int64_t best = bc * H * W + oy * s * W + ox * s; double bv = -INFINITY;
        for (int64_t i = 0; i < k; i++) for (int64_t j = 0; j < k; j++) {
            int64_t at = bc * H * W + (oy * s + i) * W + ox * s + j; double v = f32 ? ((float *)X->p)[at] : ((double *)X->p)[at];
            if (v > bv || isnan(v)) { bv = v; best = at; if (isnan(v)) goto done; }
        }
        done:;
        int64_t o = (bc * OH + oy) * OW + ox; ix[o] = best;
        if (f32) ((float *)Y->p)[o] = ((float *)X->p)[best]; else ((double *)Y->p)[o] = ((double *)X->p)[best];
    }
    *argo = I; return a_obj(Y);
}
static void pool_backward(AV g, AV a, AV idx) {
    if (!is_n(a)) return;
    AV x0 = deepval(a); int f32 = TT(x0)->dt == DT_F32; ATensor *G = conv_host(g, f32), *I = TT(idx);
    ATensor *DX = t_zeros(f32 ? DT_F32 : DT_F64, TT(x0)->nd, TT(x0)->sh); int sp = a_ntmp; A_PUSH(a_obj(DX));
    const int64_t *ix = (int64_t *)I->p;
    if (f32) for (int64_t o = 0; o < G->n; o++) ((float *)DX->p)[ix[o]] += ((float *)G->p)[o];
    else for (int64_t o = 0; o < G->n; o++) ((double *)DX->p)[ix[o]] += ((double *)G->p)[o];
    acc_grad(a, a_obj(DX)); a_ntmp = sp;
}
BI(lb_maxpool2d) {
    (void)self; (void)kw; int64_t k = n > 1 ? a[1].i : 2, s = n > 2 && a[2].t == T_INT ? a[2].i : k; ATensor *I;
    AV y = pool_forward(a[0], k, s, &I); int sp = a_ntmp; A_PUSH(y); A_PUSH(a_obj(I));
    AV r = is_n(a[0]) ? node_new(y, N_MAXPOOL, a[0], A_NONE, a_obj(I)) : y;
    a_ntmp = sp; return r;
}
