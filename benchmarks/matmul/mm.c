/* Naive matrix multiplication in C (ikj order, like the Apsi loops): gcc -O3 -march=native mm.c -o mm */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    int sizes[] = {256, 512, 1024, 2048};
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        double *a = malloc(sizeof(double) * n * n), *b = malloc(sizeof(double) * n * n), *c = malloc(sizeof(double) * n * n);
        unsigned x = 7;
        for (long i = 0; i < (long)n * n; i++) { x = x * 1103515245u + 12345u; a[i] = (x >> 8) / 16777216.0 - 0.5;
                                              x = x * 1103515245u + 12345u; b[i] = (x >> 8) / 16777216.0 - 0.5; }
        double best = 1e9;
        for (int r = 0; r < (n >= 2048 ? 1 : 3); r++) {
            for (long i = 0; i < (long)n * n; i++) c[i] = 0;
            double t = now();
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++) {
                    double aik = a[(long)i * n + k];
                    for (int j = 0; j < n; j++) c[(long)i * n + j] += aik * b[(long)k * n + j];
                }
            t = now() - t;
            if (t < best) best = t;
        }
        printf("n=%d  C naive -O3 %9.2f ms %6.1f GFLOP/s (check %.6f)\n", n, best * 1000, 2.0 * n * n * n / best / 1e9, c[n + 1]);
        free(a); free(b); free(c);
    }
    return 0;
}
