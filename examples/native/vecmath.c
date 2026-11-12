/* vecmath.c — an example C/C++ library that Apsi can call
 *   Linux   : sh build.sh          -> libvecmath.so
 *   Windows : build.bat            -> vecmath.dll
 * C++ works the same way: wrap the functions in extern "C" { ... }
 */
#include <stdint.h>

#if defined(_WIN32)
#  define APSI_EXPORT __declspec(dllexport)
#else
#  define APSI_EXPORT __attribute__((visibility("default")))
#endif

APSI_EXPORT double dot(const double *a, const double *b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* y <- a*x + y  (changes the Apsi tensor in place, no copy) */
APSI_EXPORT void axpy(double a, const double *x, double *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] += a * x[i];
}
