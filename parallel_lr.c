/*
 * parallel_lr.c
 * Parallel Simple Linear Regression using MPI
 *
 * Model:  y = beta0 + beta1 * x   (Ordinary Least Squares)
 *
 * Three collective MPI functions used:
 *   1. MPI_Scatter  — distribute chunk-size info to each process
 *   2. MPI_Scatterv — distribute data chunks (only chunk per process allocated)
 *   3. MPI_Reduce   — accumulate partial sums (Sx, Sy, Sxx, Sxy) on root
 *
 * Memory design:
 *   - Root allocates the full x[] and y[] arrays (for generation, seq baseline, MSE).
 *   - Worker processes allocate ONLY their local chunk → O(N/P) RAM per worker.
 *   - This replaces the original MPI_Bcast approach which required O(N) RAM on EVERY
 *     process, making large N (250M+) impractical on memory-constrained machines.
 *
 * Compile:
 *   mpicc -O2 -o parallel_lr parallel_lr.c -lm
 *
 * Run:
 *   mpirun -np <P> ./parallel_lr <N>
 *   e.g.  mpirun -np 4 ./parallel_lr 1000000
 *
 * Output:
 *   Prints sequential and parallel coefficients, MSE, timings,
 *   speedup, and efficiency to stdout.
 *   Appends one CSV line to timing_output.csv for analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
/*  Generate synthetic dataset:  y = 2.5*x + 1.0 + noise             */
/* ------------------------------------------------------------------ */
void generate_data(double *x, double *y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (double)i / (double)n;
        double noise = ((double)rand() / RAND_MAX) - 0.5;
        y[i] = 2.5 * x[i] + 1.0 + noise;
    }
}

/* ------------------------------------------------------------------ */
/*  Sequential linear regression (run on root for timing baseline)    */
/* ------------------------------------------------------------------ */
void sequential_lr(const double *x, const double *y, int n,
                   double *beta0_out, double *beta1_out)
{
    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < n; i++) {
        Sx  += x[i];
        Sy  += y[i];
        Sxx += x[i] * x[i];
        Sxy += x[i] * y[i];
    }
    double denom = (double)n * Sxx - Sx * Sx;
    *beta1_out = ((double)n * Sxy - Sx * Sy) / denom;
    *beta0_out = (Sy - (*beta1_out) * Sx) / (double)n;
}

/* ------------------------------------------------------------------ */
/*  Compute MSE on the full dataset (root only, for verification)     */
/* ------------------------------------------------------------------ */
double compute_mse(const double *x, const double *y, int n,
                   double beta0, double beta1)
{
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y[i] - (beta0 + beta1 * x[i]);
        mse += err * err;
    }
    return mse / (double)n;
}

/* ================================================================== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parse N from command line (default: 1 000 000) ---- */
    int n = 1000000;
    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            if (rank == 0) fprintf(stderr, "Error: N must be positive.\n");
            MPI_Finalize();
            return 1;
        }
    }

    /* ================================================================
       STEP 1 — COMPUTE CHUNK SIZES FOR EACH PROCESS
       ================================================================ */
    int base_chunk = n / size;
    int remainder  = n % size;

    /* Build send_counts and displs on ALL processes (cheap, O(P)) */
    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));
    if (!send_counts || !displs) {
        fprintf(stderr, "Process %d: malloc for bookkeeping arrays failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i]      = (i == 0) ? 0 : displs[i-1] + send_counts[i-1];
    }

    int my_count = send_counts[rank];

    /* ================================================================
       COLLECTIVE FUNCTION 1: MPI_Scatter
       Root scatters the per-process chunk size so every process knows
       how much data it will receive (used as a bookkeeping broadcast).
       ================================================================ */
    int confirmed_count;
    MPI_Scatter(send_counts, 1, MPI_INT,
                &confirmed_count, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    /* confirmed_count == my_count; kept for symmetry with original design */

    /* ================================================================
       MEMORY ALLOCATION
       Root: full arrays (needed for data generation, seq LR, MSE)
       Workers: only their local chunk → O(N/P) RAM per worker
       ================================================================ */
    double *x_full = NULL, *y_full = NULL;   /* root only */
    double *x_local = NULL, *y_local = NULL; /* all processes */

    if (rank == 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) {
            fprintf(stderr, "Root: malloc for full arrays failed (N=%d, ~%.1f GB needed)\n",
                    n, (double)n * 2 * sizeof(double) / 1e9);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        generate_data(x_full, y_full, n, 42);
    }

    x_local = (double *)malloc(my_count * sizeof(double));
    y_local = (double *)malloc(my_count * sizeof(double));
    if (!x_local || !y_local) {
        fprintf(stderr, "Process %d: malloc for local chunk failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ================================================================
       COLLECTIVE FUNCTION 2: MPI_Scatterv
       Root distributes x[] and y[] chunks to each process.
       Each worker receives ONLY its own slice — no full-array copy.
       ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    MPI_Scatterv(x_full, send_counts, displs, MPI_DOUBLE,
                 x_local, my_count,           MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(y_full, send_counts, displs, MPI_DOUBLE,
                 y_local, my_count,           MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* ================================================================
       STEP 2 — EACH PROCESS COMPUTES PARTIAL SUMS OVER ITS CHUNK
       ================================================================ */
    double local_Sx  = 0.0;
    double local_Sy  = 0.0;
    double local_Sxx = 0.0;
    double local_Sxy = 0.0;

    for (int i = 0; i < my_count; i++) {
        local_Sx  += x_local[i];
        local_Sy  += y_local[i];
        local_Sxx += x_local[i] * x_local[i];
        local_Sxy += x_local[i] * y_local[i];
    }

    /* ================================================================
       COLLECTIVE FUNCTION 3: MPI_Reduce
       Each process sends its partial sums; root accumulates them.
       ================================================================ */
    double global_Sx  = 0.0;
    double global_Sy  = 0.0;
    double global_Sxx = 0.0;
    double global_Sxy = 0.0;

    MPI_Reduce(&local_Sx,  &global_Sx,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sy,  &global_Sy,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxx, &global_Sxx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxy, &global_Sxy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double par_end = MPI_Wtime();

    /* ================================================================
       STEP 3 — ROOT COMPUTES FINAL COEFFICIENTS AND REPORTS
       ================================================================ */
    if (rank == 0) {
        /* Parallel OLS coefficients */
        double denom     = (double)n * global_Sxx - global_Sx * global_Sx;
        double beta1_par = ((double)n * global_Sxy - global_Sx * global_Sy) / denom;
        double beta0_par = (global_Sy - beta1_par * global_Sx) / (double)n;
        double par_time  = par_end - par_start;

        /* Sequential baseline for speedup comparison */
        double seq_start = MPI_Wtime();
        double beta0_seq, beta1_seq;
        sequential_lr(x_full, y_full, n, &beta0_seq, &beta1_seq);
        double seq_time = MPI_Wtime() - seq_start;

        double mse_seq    = compute_mse(x_full, y_full, n, beta0_seq, beta1_seq);
        double mse_par    = compute_mse(x_full, y_full, n, beta0_par, beta1_par);
        double speedup    = seq_time / par_time;
        double efficiency = speedup / (double)size;

        printf("=======================================================\n");
        printf("  Parallel Linear Regression (MPI)\n");
        printf("  N = %d   P = %d\n", n, size);
        printf("=======================================================\n");
        printf("[Sequential]\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_seq, beta1_seq);
        printf("  MSE   = %.8f\n", mse_seq);
        printf("  Time  = %.6f s\n", seq_time);
        printf("\n[Parallel]\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_par, beta1_par);
        printf("  MSE   = %.8f\n", mse_par);
        printf("  Time  = %.6f s\n", par_time);
        printf("\n[Performance]\n");
        printf("  Speedup    = %.4f\n", speedup);
        printf("  Efficiency = %.4f\n", efficiency);
        printf("=======================================================\n");

        /* Append timing row to CSV (for analysis.py) */
        FILE *fp = fopen("timing_output.csv", "a");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0)
                fprintf(fp, "N,P,seq_time,par_time\n");
            fprintf(fp, "%d,%d,%.8f,%.8f\n", n, size, seq_time, par_time);
            fclose(fp);
        }

        free(x_full);
        free(y_full);
    }

    free(x_local);
    free(y_local);
    free(send_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
