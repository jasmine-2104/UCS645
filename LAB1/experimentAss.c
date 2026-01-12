#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N_DAXPY (1<<16)
#define N_MATRIX 1000
#define PI_STEPS 100000000

void daxpy_test(int threads) {
    double *X = (double *)malloc(N_DAXPY * sizeof(double));
    double *Y = (double *)malloc(N_DAXPY * sizeof(double));
    double a = 2.5;

    for (int i = 0; i < N_DAXPY; i++) {
        X[i] = i * 1.0;
        Y[i] = i * 2.0;
    }

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N_DAXPY; i++)
        X[i] = a * X[i] + Y[i];

    double end = omp_get_wtime();
    printf("DAXPY | Threads = %d | Time = %f seconds\n", threads, end - start);

    free(X);
    free(Y);
}

void matrix_multiply_1D(int threads) {
    static double A[N_MATRIX][N_MATRIX];
    static double B[N_MATRIX][N_MATRIX];
    static double C[N_MATRIX][N_MATRIX];

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N_MATRIX; i++)
        for (int j = 0; j < N_MATRIX; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N_MATRIX; k++)
                C[i][j] += A[i][k] * B[k][j];
        }

    double end = omp_get_wtime();
    printf("Matrix Multiply 1D | Threads = %d | Time = %f seconds\n", threads, end - start);
}

void matrix_multiply_2D(int threads) {
    static double A[N_MATRIX][N_MATRIX];
    static double B[N_MATRIX][N_MATRIX];
    static double C[N_MATRIX][N_MATRIX];

    omp_set_num_threads(threads);
    double start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N_MATRIX; i++)
        for (int j = 0; j < N_MATRIX; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N_MATRIX; k++)
                C[i][j] += A[i][k] * B[k][j];
        }

    double end = omp_get_wtime();
    printf("Matrix Multiply 2D | Threads = %d | Time = %f seconds\n", threads, end - start);
}

void calculate_pi(int threads) {
    omp_set_num_threads(threads);
    double step = 1.0 / (double)PI_STEPS;
    double sum = 0.0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < PI_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end = omp_get_wtime();
    printf("PI Calculation | Threads = %d | PI = %.10f | Time = %f seconds\n", threads, pi, end - start);
}

int main() {
    printf("\n===== Q1: DAXPY LOOP =====\n");
    for (int t = 2; t <= 8; t *= 2)
        daxpy_test(t);

    printf("\n===== Q2: MATRIX MULTIPLICATION =====\n");
    for (int t = 2; t <= 8; t *= 2) {
        matrix_multiply_1D(t);
        matrix_multiply_2D(t);
    }

    printf("\n===== Q3: CALCULATION OF PI =====\n");
    for (int t = 2; t <= 8; t *= 2)
        calculate_pi(t);

    return 0;
}

