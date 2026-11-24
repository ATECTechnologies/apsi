#include <stdint.h>
#include <string.h>
double dot(const double *a, const double *b, int64_t n) { double s = 0; for (int64_t i = 0; i < n; i++) s += a[i] * b[i]; return s; }
void scale(double *a, int64_t n, double k) { for (int64_t i = 0; i < n; i++) a[i] *= k; }
int64_t length(const char *s) { return (int64_t)strlen(s); }
float halfsum(const float *a, int32_t n) { float s = 0; for (int i = 0; i < n; i++) s += a[i]; return s / 2; }
