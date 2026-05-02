/*
 * parallel_lr_bcast.c
 * Parallel Simple Linear Regression using MPI — Bcast Version
 *
 * Bu versiyon veriyi MPI_Bcast ile tüm processlere gönderir.
 * Her process N elemanlı tam diziyi alır → O(N) iletişim maliyeti.
 *
 * Scatterv versiyonuyla karşılaştırma için kullanılır.
 *
 * Three collective MPI functions:
 *   1. MPI_Bcast  — tüm x ve y dizilerini broadcast et (O(N))
 *   2. MPI_Bcast  — N sayısını broadcast et (O(1))
 *   3. MPI_Reduce — partial sumları root'ta topla
 *
 * Compile:
 *   mpicc -O2 -o parallel_lr_bcast parallel_lr_bcast.c -lm
 *
 * Run:
 *   mpirun -np <P> ./parallel_lr_bcast <N>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void generate_data(double *x, double *y, int n, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (double)i / (double)n;
        double noise = ((double)rand() / RAND_MAX) - 0.5;
        y[i] = 2.5 * x[i] + 1.0 + noise;
    }
}

void sequential_lr(const double *x, const double *y, int n,
                   double *beta0_out, double *beta1_out)
{
    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0;
    for (int i = 0; i < n; i++) {
        Sx  += x[i]; Sy  += y[i];
        Sxx += x[i] * x[i]; Sxy += x[i] * y[i];
    }
    double denom = (double)n * Sxx - Sx * Sx;
    *beta1_out = ((double)n * Sxy - Sx * Sy) / denom;
    *beta0_out = (Sy - (*beta1_out) * Sx) / (double)n;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;
    if (argc >= 2) { n = atoi(argv[1]); }

    /* ── Serial baseline (rank 0 only) ── */
    double *x_full = NULL, *y_full = NULL;
    double serial_time = 0.0;
    double beta0_seq = 0.0, beta1_seq = 0.0;

    if (rank == 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) { MPI_Abort(MPI_COMM_WORLD, 1); }
        generate_data(x_full, y_full, n, 42);

        double seq_start = MPI_Wtime();
        sequential_lr(x_full, y_full, n, &beta0_seq, &beta1_seq);
        serial_time = MPI_Wtime() - seq_start;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ── Parallel phase starts ── */
    MPI_Barrier(MPI_COMM_WORLD);
    double par_start = MPI_Wtime();

    /* Non-root processler için bellek ayır */
    if (rank != 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    /* COLLECTIVE 1: MPI_Bcast — tüm x dizisini broadcast et (O(N)) */
    double t_comm_start = MPI_Wtime();
    MPI_Bcast(x_full, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_full, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast_end = MPI_Wtime();

    /* Her process kendi chunk'ını hesapla */
    int base_chunk = n / size;
    int remainder  = n % size;
    int my_count   = base_chunk + (rank < remainder ? 1 : 0);
    int my_start   = rank * base_chunk + (rank < remainder ? rank : remainder);

    /* COMPUTE: lokal partial sum */
    double local_Sx = 0.0, local_Sy = 0.0;
    double local_Sxx = 0.0, local_Sxy = 0.0;

    double t_compute_start = MPI_Wtime();
    for (int i = my_start; i < my_start + my_count; i++) {
        local_Sx  += x_full[i]; local_Sy  += y_full[i];
        local_Sxx += x_full[i] * x_full[i];
        local_Sxy += x_full[i] * y_full[i];
    }
    double t_compute_end = MPI_Wtime();
    double compute_time  = t_compute_end - t_compute_start;

    free(x_full); free(y_full);

    /* COLLECTIVE 2: MPI_Reduce — partial sumları topla */
    double global_Sx = 0.0, global_Sy = 0.0;
    double global_Sxx = 0.0, global_Sxy = 0.0;

    double t_reduce_start = MPI_Wtime();
    MPI_Reduce(&local_Sx,  &global_Sx,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sy,  &global_Sy,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxx, &global_Sxx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Sxy, &global_Sxy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t_reduce_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double par_time_total = MPI_Wtime() - par_start;

    double comm_time = (t_bcast_end - t_comm_start) +
                       (t_reduce_end - t_reduce_start);

    double compute_time_max = 0.0, comm_time_max = 0.0;
    MPI_Reduce(&compute_time, &compute_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time,    &comm_time_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double denom     = (double)n * global_Sxx - global_Sx * global_Sx;
        double beta1_par = ((double)n * global_Sxy - global_Sx * global_Sy) / denom;
        double beta0_par = (global_Sy - beta1_par * global_Sx) / (double)n;

        double speedup_compute    = serial_time / compute_time_max;
        double speedup_total      = serial_time / par_time_total;
        double efficiency_compute = speedup_compute / (double)size;
        double efficiency_total   = speedup_total   / (double)size;

        printf("[Bcast] N=%d P=%d | serial=%.6f compute=%.6f comm=%.6f total=%.6f | Sc=%.4f St=%.4f\n",
               n, size, serial_time, compute_time_max, comm_time_max, par_time_total,
               speedup_compute, speedup_total);

        FILE *fp = fopen("timing_bcast.csv", "a");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0)
                fprintf(fp, "N,P,serial_time,compute_time,comm_time,par_time_total,"
                            "speedup_compute,speedup_total,efficiency_compute,efficiency_total\n");
            fprintf(fp, "%d,%d,%.8f,%.8f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f\n",
                    n, size, serial_time, compute_time_max, comm_time_max, par_time_total,
                    speedup_compute, speedup_total, efficiency_compute, efficiency_total);
            fclose(fp);
        }
    }

    MPI_Finalize();
    return 0;
}
