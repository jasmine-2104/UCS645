#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
Program Name: mpi_all_programs.c

This file contains multiple MPI examples:

1 -> Hello World
2 -> Parallel Sum of 1 to 100
3 -> Parallel Dot Product
4 -> Random Numbers + Global Max/Min
5 -> Large Computation Timing
6 -> Ring Communication

Run using:
mpirun -np <num_processes> ./mpi_all_programs <option_number>
*/

#define SUM_N 100
#define DOT_N 8
#define LARGE_N 200000000

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            printf("Please provide option number (1-6)\n");
        MPI_Finalize();
        return 0;
    }

    int option = atoi(argv[1]);

    /* ==============================
       OPTION 1: HELLO WORLD
       ============================== */
    if (option == 1) {
        printf("Hello from rank %d out of %d processes\n", rank, size);
    }

    /* ==============================
       OPTION 2: PARALLEL SUM
       ============================== */
    else if (option == 2) {

        int numbers[SUM_N];
        int local_n = SUM_N / size;
        int local_array[local_n];

        if (rank == 0) {
            for (int i = 0; i < SUM_N; i++)
                numbers[i] = i + 1;
        }

        MPI_Scatter(numbers, local_n, MPI_INT,
                    local_array, local_n, MPI_INT,
                    0, MPI_COMM_WORLD);

        int local_sum = 0;
        for (int i = 0; i < local_n; i++)
            local_sum += local_array[i];

        printf("Process %d local sum = %d\n", rank, local_sum);

        int global_sum;
        MPI_Reduce(&local_sum, &global_sum, 1,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nGlobal Sum = %d\n", global_sum);
            printf("Expected = 5050\n");
            printf("Average = %.2f\n", global_sum / (float)SUM_N);
        }
    }

    /* ==============================
       OPTION 3: DOT PRODUCT
       ============================== */
    else if (option == 3) {

        int A[DOT_N], B[DOT_N];
        int local_n = DOT_N / size;
        int local_A[local_n], local_B[local_n];

        if (rank == 0) {
            for (int i = 0; i < DOT_N; i++) {
                A[i] = i + 1;
                B[i] = DOT_N - i;
            }
        }

        MPI_Scatter(A, local_n, MPI_INT,
                    local_A, local_n, MPI_INT,
                    0, MPI_COMM_WORLD);

        MPI_Scatter(B, local_n, MPI_INT,
                    local_B, local_n, MPI_INT,
                    0, MPI_COMM_WORLD);

        int local_dot = 0;
        for (int i = 0; i < local_n; i++)
            local_dot += local_A[i] * local_B[i];

        printf("Process %d partial dot = %d\n", rank, local_dot);

        int global_dot;
        MPI_Reduce(&local_dot, &global_dot, 1,
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nFinal Dot Product = %d\n", global_dot);
            printf("Expected = 120\n");
        }
    }

    /* ==============================
       OPTION 4: RANDOM MAX/MIN
       ============================== */
    else if (option == 4) {

        srand(time(NULL) + rank);

        int numbers[10];

        printf("\nProcess %d numbers: ", rank);
        for (int i = 0; i < 10; i++) {
            numbers[i] = rand() % 1001;
            printf("%d ", numbers[i]);
        }
        printf("\n");

        int local_max = numbers[0];
        int local_min = numbers[0];

        for (int i = 1; i < 10; i++) {
            if (numbers[i] > local_max) local_max = numbers[i];
            if (numbers[i] < local_min) local_min = numbers[i];
        }

        struct {
            int value;
            int rank;
        } local_maxloc, global_maxloc,
          local_minloc, global_minloc;

        local_maxloc.value = local_max;
        local_maxloc.rank = rank;

        local_minloc.value = local_min;
        local_minloc.rank = rank;

        MPI_Reduce(&local_maxloc, &global_maxloc, 1,
                   MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        MPI_Reduce(&local_minloc, &global_minloc, 1,
                   MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\nGlobal Max = %d (Process %d)\n",
                   global_maxloc.value, global_maxloc.rank);

            printf("Global Min = %d (Process %d)\n",
                   global_minloc.value, global_minloc.rank);
        }
    }

    /* ==============================
       OPTION 5: LARGE COMPUTATION
       ============================== */
    else if (option == 5) {

        long long chunk = LARGE_N / size;
        long long start = rank * chunk;
        long long end = start + chunk;

        double local_sum = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        for (long long i = start; i < end; i++)
            local_sum += sqrt(i * 1.0);

        double global_sum;

        MPI_Reduce(&local_sum, &global_sum, 1,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double end_time = MPI_Wtime();

        if (rank == 0) {
            printf("\nProcesses: %d\n", size);
            printf("Execution Time: %f seconds\n",
                   end_time - start_time);
            printf("Global Sum: %f\n", global_sum);
        }
    }

    /* ==============================
       OPTION 6: RING COMMUNICATION
       ============================== */
    else if (option == 6) {

        int value;
        int next = (rank + 1) % size;
        int prev = (rank - 1 + size) % size;

        if (rank == 0) {
            value = 100;
            printf("Process 0 starting with %d\n", value);

            MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
            MPI_Recv(&value, 1, MPI_INT, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Process 0 received final value %d\n", value);
        } else {
            MPI_Recv(&value, 1, MPI_INT, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            value += rank;
            printf("Process %d updated value to %d\n",
                   rank, value);

            MPI_Send(&value, 1, MPI_INT, next, 0,
                     MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}