#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================== EXP 1 & 2 ===================== */
void vector_add(long N) {
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));

    for (long i = 0; i < N; i++)
        A[i] = B[i] = 1.0;

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++)
        C[i] = A[i] + B[i];
    double end = omp_get_wtime();

    printf("Vector Addition Time: %f seconds\n", end - start);

    free(A); free(B); free(C);
}

/* ===================== EXP 3 ===================== */
void strong_weak_scaling() {
    long long base = 100000000;
    double T1 = 0.0;

    printf("\n--- STRONG SCALING ---\n");
    for (int t = 1; t <= 8; t *= 2) {
        double sum = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(t) reduction(+:sum)
        for (long long i = 0; i < base; i++) {
            double x = (i + 0.5) / base;
            sum += 4.0 / (1.0 + x * x);
        }

        double T = omp_get_wtime() - start;
        if (t == 1) T1 = T;

        printf("Threads: %d  Time: %f  Speedup: %f\n", t, T, T1 / T);
    }

    printf("\n--- WEAK SCALING ---\n");
    for (int t = 1; t <= 8; t *= 2) {
        long long N = base * t;
        double sum = 0.0;
        double start = omp_get_wtime();

        #pragma omp parallel for num_threads(t) reduction(+:sum)
        for (long long i = 0; i < N; i++) {
            double x = (i + 0.5) / N;
            sum += 4.0 / (1.0 + x * x);
        }

        printf("Threads: %d  Time: %f\n", t, omp_get_wtime() - start);
    }
}

/* ===================== EXP 4 ===================== */
void scheduling_test() {
    int N = 1000;

    double start;

    start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i * 1000; j++) sin(j);
    printf("Static Scheduling Time: %f\n", omp_get_wtime() - start);

    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,10)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i * 1000; j++) sin(j);
    printf("Dynamic Scheduling Time: %f\n", omp_get_wtime() - start);

    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < i * 1000; j++) sin(j);
    printf("Guided Scheduling Time: %f\n", omp_get_wtime() - start);
}

/* ===================== EXP 5 ===================== */
void sync_test() {
    long long N = 10000000;
    double sum = 0.0;

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        #pragma omp critical
        sum += 1.0;
    }
    printf("Critical Time: %f\n", omp_get_wtime() - start);

    sum = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; i++)
        sum += 1.0;
    printf("Reduction Time: %f\n", omp_get_wtime() - start);
}

/* ===================== EXP 6 ===================== */
#define CACHE_LINE 64
typedef struct { long long x; } Bad;
typedef struct { long long x; char pad[CACHE_LINE - sizeof(long long)]; } Good;

void false_sharing() {
    int T = omp_get_max_threads();
    Bad *bad = calloc(T, sizeof(Bad));
    Good *good = calloc(T, sizeof(Good));

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long long i = 0; i < 100000000; i++)
            bad[id].x++;
    }
    printf("False Sharing Time: %f\n", omp_get_wtime() - start);

    start = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long long i = 0; i < 100000000; i++)
            good[id].x++;
    }
    printf("Padded Time: %f\n", omp_get_wtime() - start);

    free(bad); free(good);
}

/* ===================== EXP 7 ===================== */
void memory_bandwidth() {
    long long N = 100000000;
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));

    for (long long i = 0; i < N; i++) {
        B[i] = 1.0;
        C[i] = 2.0;
    }

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++)
        A[i] = B[i] + 3.0 * C[i];

    double T = omp_get_wtime() - start;
    double BW = (3.0 * N * sizeof(double)) / (T * 1e9);

    printf("Time: %f  Bandwidth: %f GB/s\n", T, BW);

    free(A); free(B); free(C);
}

/* ===================== MAIN ===================== */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage:\n");
        printf("./ass2 vector | scaling | schedule | sync | false | bandwidth\n");
        return 0;
    }

    if (!strcmp(argv[1], "vector")) vector_add(100000000);
    else if (!strcmp(argv[1], "scaling")) strong_weak_scaling();
    else if (!strcmp(argv[1], "schedule")) scheduling_test();
    else if (!strcmp(argv[1], "sync")) sync_test();
    else if (!strcmp(argv[1], "false")) false_sharing();
    else if (!strcmp(argv[1], "bandwidth")) memory_bandwidth();

    return 0;
}
