#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 65536   // 2^16 for DAXPY

// ---------------- Q1: DAXPY ----------------
void daxpy(int rank, int size) {
    double *X = (double*)malloc(N * sizeof(double));
    double *Y = (double*)malloc(N * sizeof(double));
    double a = 2.5;

    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    int local_n = N / size;
    int start = rank * local_n;

    double t1 = MPI_Wtime();

    for (int i = start; i < start + local_n; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double t2 = MPI_Wtime();

    if (rank == 0)
        printf("MPI DAXPY Time: %f sec\n", t2 - t1);

    free(X); free(Y);
}

// ---------------- Q2: Broadcast ----------------
void my_bcast(double *data, int n, int rank, int size) {
    if (rank == 0) {
        for (int i = 1; i < size; i++)
            MPI_Send(data, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(data, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void broadcast_test(int rank, int size) {
    int n = 10000000; // 10 million
    double *arr = (double*)malloc(n * sizeof(double));

    if (rank == 0)
        for (int i = 0; i < n; i++) arr[i] = i;

    double t1 = MPI_Wtime();
    my_bcast(arr, n, rank, size);
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    MPI_Bcast(arr, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t4 = MPI_Wtime();

    if (rank == 0) {
        printf("Manual Broadcast Time: %f sec\n", t2 - t1);
        printf("MPI_Bcast Time: %f sec\n", t4 - t3);
    }

    free(arr);
}

// ---------------- Q3: Distributed Dot Product ----------------
void dot_product(int rank, int size) {
    long long total_n = 50000000; // reduced for safety (can increase)
    long long local_n = total_n / size;

    double multiplier = 2.0;
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0.0;

    double t1 = MPI_Wtime();

    for (long long i = 0; i < local_n; i++) {
        double A = 1.0;
        double B = 2.0 * multiplier;
        local_sum += A * B;
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    if (rank == 0) {
        printf("Dot Product = %.2f\n", global_sum);
        printf("Execution Time = %f sec\n", t2 - t1);
    }
}

// ---------------- Q4: Prime Numbers ----------------
int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i <= sqrt(n); i++)
        if (n % i == 0) return 0;
    return 1;
}

void primes(int rank, int size) {
    int max = 50;

    if (rank == 0) {
        int next = 2, active = size - 1;
        MPI_Status status;

        while (active > 0) {
            int num;
            MPI_Recv(&num, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            int src = status.MPI_SOURCE;

            if (num > 0)
                printf("Prime: %d\n", num);

            if (next <= max) {
                MPI_Send(&next, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                next++;
            } else {
                int stop = 0;
                MPI_Send(&stop, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                active--;
            }
        }
    } else {
        int num = 0;

        while (1) {
            MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (num == 0) break;

            int res = is_prime(num) ? num : -num;
            MPI_Send(&res, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

// ---------------- Q5: Perfect Numbers ----------------
int is_perfect(int n) {
    int sum = 1;
    for (int i = 2; i <= n/2; i++)
        if (n % i == 0) sum += i;
    return sum == n && n != 1;
}

void perfect_numbers(int rank, int size) {
    int max = 1000;

    if (rank == 0) {
        int next = 2, active = size - 1;
        MPI_Status status;

        while (active > 0) {
            int num;
            MPI_Recv(&num, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            int src = status.MPI_SOURCE;

            if (num > 0)
                printf("Perfect Number: %d\n", num);

            if (next <= max) {
                MPI_Send(&next, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                next++;
            } else {
                int stop = 0;
                MPI_Send(&stop, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
                active--;
            }
        }
    } else {
        int num = 0;

        while (1) {
            MPI_Send(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (num == 0) break;

            int res = is_perfect(num) ? num : -num;
            MPI_Send(&res, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

// ---------------- MAIN ----------------
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            printf("Usage: ./mpi_all [daxpy | bcast | dot | prime | perfect]\n");
        MPI_Finalize();
        return 0;
    }

    if (strcmp(argv[1], "daxpy") == 0)
        daxpy(rank, size);
    else if (strcmp(argv[1], "bcast") == 0)
        broadcast_test(rank, size);
    else if (strcmp(argv[1], "dot") == 0)
        dot_product(rank, size);
    else if (strcmp(argv[1], "prime") == 0)
        primes(rank, size);
    else if (strcmp(argv[1], "perfect") == 0)
        perfect_numbers(rank, size);

    MPI_Finalize();
    return 0;
}