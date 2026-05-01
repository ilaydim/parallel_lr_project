/*
 * parallel_lr.c
 * Parallel Simple Linear Regression using MPI
 *
 * Model:  y = beta0 + beta1 * x   (Ordinary Least Squares)
 *
 * Three collective MPI functions used:
 *   1. MPI_Bcast    — broadcast dataset size N to all processes
 *   2. MPI_Scatterv — distribute data chunks; each process receives
 *                     only its N/P elements (not the full dataset)
 *   3. MPI_Reduce   — accumulate partial sums (Sx, Sy, Sxx, Sxy) on root
 *
 * Memory design:
 *   - Root allocates full N elements (needed for data generation +
 *     sequential baseline), then scatters chunks and frees the full arrays.
 *   - Non-root processes allocate only N/P elements.
 *   - Communication volume per process: O(N/P) instead of O(N).
 *
 * Timing design:
 *   - serial_time : measured on rank 0 only, before any MPI communication.
 *   - par_time    : wall time from MPI_Barrier (before MPI_Bcast) to
 *                   MPI_Barrier (after MPI_Reduce); covers the full
 *                   parallel pipeline including communication.
 *   - speedup     : serial_time / par_time
 *   - efficiency  : speedup / P
 *
 * Compile:
 *   mpicc -O2 -o parallel_lr parallel_lr.c -lm
 *
 * Run:
 *   mpirun -np <P> ./parallel_lr <N>
 *   e.g.  mpirun -np 8 ./parallel_lr 250000000
 *
 * Output:
 *   Prints sequential and parallel coefficients, MSE, timings,
 *   speedup, and efficiency to stdout.
 *   Appends one CSV line to timing_output.csv for analysis.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
       STEP 1 — ROOT GENERATES FULL DATASET FOR SEQUENTIAL BASELINE
       Non-root processes do not allocate anything yet.
       ================================================================ */
    double *x_full = NULL, *y_full = NULL;
    double  beta0_seq = 0.0, beta1_seq = 0.0;
    double  serial_time = 0.0;

    if (rank == 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) {
            fprintf(stderr, "Root: malloc failed for full dataset\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        generate_data(x_full, y_full, n, 42);
    }

    /* ================================================================
       SERIAL BASELINE
       Measured on rank 0 only, before any MPI communication.
       All other ranks wait at the barrier so conditions are fair.
       ================================================================ */
    if (rank == 0) {
        double seq_start = MPI_Wtime();
        sequential_lr(x_full, y_full, n, &beta0_seq, &beta1_seq);
        serial_time = MPI_Wtime() - seq_start;
    }

    /* All workers wait here while root measures serial time */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ================================================================
       PARALLEL PHASE — timer starts here
       MPI_Barrier ensures all processes start together.
       ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    /* ================================================================
       COLLECTIVE FUNCTION 1: MPI_Bcast
       Root broadcasts only the dataset size N (1 integer) so that
       every process can compute its chunk size and allocate memory.
       Communication cost: O(1) — negligible.
       ================================================================ */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute chunk sizes and displacements (all processes do this) */
    int base_chunk = n / size;
    int remainder  = n % size;

    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));
    if (!send_counts || !displs) {
        fprintf(stderr, "Process %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int offset = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i]      = offset;
        offset        += send_counts[i];
    }
    int my_count = send_counts[rank];

    /* Each process allocates only its own chunk — N/P elements */
    double *x_local = (double *)malloc((size_t)my_count * sizeof(double));
    double *y_local = (double *)malloc((size_t)my_count * sizeof(double));
    if (!x_local || !y_local) {
        fprintf(stderr, "Process %d: malloc failed for local chunk\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ================================================================
       COLLECTIVE FUNCTION 2: MPI_Scatterv
       Root distributes data chunks to each process.
       Each process receives exactly its N/P elements — not the full
       dataset. Memory per process drops from O(N) to O(N/P).
       Communication cost per process: O(N/P) instead of O(N).
       ================================================================ */
    double t_scatter_start = MPI_Wtime();
    MPI_Scatterv(x_full, send_counts, displs, MPI_DOUBLE,
                 x_local, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(y_full, send_counts, displs, MPI_DOUBLE,
                 y_local, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();

    /* Root no longer needs the full arrays — free them to save memory */
    if (rank == 0) {
        free(x_full); x_full = NULL;
        free(y_full); y_full = NULL;
    }

    /* ================================================================
       STEP 2 — EACH PROCESS COMPUTES PARTIAL SUMS OVER ITS CHUNK
       ================================================================ */
    double local_Sx  = 0.0, local_Sy  = 0.0;
    double local_Sxx = 0.0, local_Sxy = 0.0;

    double t_compute_start = MPI_Wtime();
    for (int i = 0; i < my_count; i++) {
        local_Sx  += x_local[i];
        local_Sy  += y_local[i];
        local_Sxx += x_local[i] * x_local[i];
        local_Sxy += x_local[i] * y_local[i];
    }
    double t_compute_end = MPI_Wtime();
    double compute_time = t_compute_end - t_compute_start;

    free(x_local);
    free(y_local);

    /* ================================================================
       COLLECTIVE FUNCTION 3: MPI_Reduce
       Each process sends its partial sums; root accumulates them.
       ================================================================ */
    double global_Sx  = 0.0, global_Sy  = 0.0;
    double global_Sxx = 0.0, global_Sxy = 0.0;

    double t_reduce_start = MPI_Wtime();
    MPI_Reduce(&local_Sx,  &global_Sx,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sy,  &global_Sy,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxx, &global_Sxx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxy, &global_Sxy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t_reduce_end = MPI_Wtime();

    /* Parallel phase ends here */
    MPI_Barrier(MPI_COMM_WORLD);
    double par_time_total = MPI_Wtime() - par_start;

    /* Derived timing metrics */
    double comm_time        = (t_scatter_end - t_scatter_start) + (t_reduce_end - t_reduce_start);
    /* compute_time is the max across all ranks (bottleneck) — reduce to root */
    double compute_time_max = 0.0;
    MPI_Reduce(&compute_time, &compute_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double comm_time_max    = 0.0;
    MPI_Reduce(&comm_time,    &comm_time_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ================================================================
       STEP 3 — ROOT COMPUTES FINAL COEFFICIENTS AND REPORTS
       ================================================================ */
    if (rank == 0) {
        double denom     = (double)n * global_Sxx - global_Sx * global_Sx;
        double beta1_par = ((double)n * global_Sxy - global_Sx * global_Sy) / denom;
        double beta0_par = (global_Sy - beta1_par * global_Sx) / (double)n;

        /* Re-generate data on root only for MSE verification */
        double *xv = (double *)malloc((size_t)n * sizeof(double));
        double *yv = (double *)malloc((size_t)n * sizeof(double));
        if (xv && yv) generate_data(xv, yv, n, 42);

        double mse_seq = (xv && yv) ? compute_mse(xv, yv, n, beta0_seq, beta1_seq) : -1.0;
        double mse_par = (xv && yv) ? compute_mse(xv, yv, n, beta0_par, beta1_par) : -1.0;

        free(xv); free(yv);

        double speedup_compute    = serial_time / compute_time_max;
        double speedup_total      = serial_time / par_time_total;
        double efficiency_compute = speedup_compute / (double)size;
        double efficiency_total   = speedup_total   / (double)size;

        printf("=======================================================\n");
        printf("  Parallel Linear Regression (MPI)\n");
        printf("  N = %d   P = %d\n", n, size);
        printf("=======================================================\n");
        printf("[Sequential]\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_seq, beta1_seq);
        printf("  MSE   = %.8f\n", mse_seq);
        printf("  Time  = %.6f s\n", serial_time);
        printf("\n[Parallel]\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_par, beta1_par);
        printf("  MSE   = %.8f\n", mse_par);
        printf("  Compute Time = %.6f s\n", compute_time_max);
        printf("  Comm    Time = %.6f s  (Scatterv x2 + Reduce x4)\n", comm_time_max);
        printf("  Total   Time = %.6f s\n", par_time_total);
        printf("\n[Performance]\n");
        printf("  Speedup  (compute) = %.4f   Efficiency = %.4f\n",
               speedup_compute, efficiency_compute);
        printf("  Speedup  (total)   = %.4f   Efficiency = %.4f\n",
               speedup_total,   efficiency_total);
        printf("=======================================================\n");

        FILE *fp = fopen("timing_output.csv", "a");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0)
                fprintf(fp, "N,P,serial_time,compute_time,comm_time,par_time_total,"
                            "speedup_compute,speedup_total,efficiency_compute,efficiency_total\n");
            fprintf(fp, "%d,%d,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f\n",
                    n, size,
                    serial_time, compute_time_max, comm_time_max, par_time_total,
                    speedup_compute, speedup_total,
                    efficiency_compute, efficiency_total);
            fclose(fp);
        }
    }

    free(send_counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
