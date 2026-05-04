/*
 * parallel_lr_bcast.c
 * Parallel Simple Linear Regression using MPI — Bcast Version
 *
 * Bu versiyon veriyi MPI_Bcast ile tüm processlere gönderir.
 * Her process N elemanlı tam diziyi alır → O(N×P) iletişim maliyeti.
 * Scatterv versiyonuyla karşılaştırma için kullanılır.
 *
 * Collective MPI functions:
 *   1. MPI_Bcast  — tüm x ve y dizilerini broadcast et (O(N))
 *   2. MPI_Reduce — partial sumları root'ta topla
 *
 * Compile:
 *   mpicc -O2 -o parallel_lr_bcast parallel_lr_bcast.c -lm
 * Run:
 *   mpirun -np <P> ./parallel_lr_bcast <N>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* California Housing CSV'den veri oku, gerekirse N'e ulaşmak için tekrarla */
void load_data(double *x, double *y, int n, const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        exit(1);
    }

    /* Header satırını atla */
    char header[256];
    fgets(header, sizeof(header), fp);

    /* Önce kaç satır var say */
    double tx, ty;
    int file_rows = 0;
    while (fscanf(fp, "%lf,%lf", &tx, &ty) == 2) file_rows++;
    rewind(fp);
    fgets(header, sizeof(header), fp); /* header'ı tekrar atla */

    /* Tüm dosyayı geçici diziye oku */
    double *xbuf = malloc(file_rows * sizeof(double));
    double *ybuf = malloc(file_rows * sizeof(double));
    for (int i = 0; i < file_rows; i++)
        fscanf(fp, "%lf,%lf", &xbuf[i], &ybuf[i]);
    fclose(fp);

    /* N'e ulaşmak için tekrarla (wrap around) */
    for (int i = 0; i < n; i++) {
        x[i] = xbuf[i % file_rows];
        y[i] = ybuf[i % file_rows];
    }

    free(xbuf);
    free(ybuf);
}

/* ================================================================== */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000000;
    if (argc >= 2) { n = atoi(argv[1]); }

    /* STEP 1 — Root veriyi yükler, diğerleri bekler */
    double *x_full = NULL, *y_full = NULL;

    if (rank == 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) { MPI_Abort(MPI_COMM_WORLD, 1); }
        /* California Housing verisini yükle, N'e ulaşmak için tekrarla */
        load_data(x_full, y_full, n, "california_housing.csv");
    }

    MPI_Barrier(MPI_COMM_WORLD); /* Paralel timer'ı senkronize başlat */
    double par_start = MPI_Wtime();

    /* Non-root processler N elemanlı tam dizi için bellek ayırıyor.
       Scatterv'den farklı olarak herkes tam diziyi alacak — O(N) değil O(N×P) iletişim */
    if (rank != 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    /* COLLECTIVE 1: MPI_Bcast — tüm x ve y dizisini broadcast et
       FARKLI OLAN TEK YER: Scatterv değil, Bcast kullanılıyor.
       Her process N elemanlı tam diziyi alıyor → O(N×P) iletişim */
    double t_comm_start = MPI_Wtime();
    MPI_Bcast(x_full, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y_full, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double t_bcast_end = MPI_Wtime();

    /* Her process kendi chunk'ının başlangıç ve bitiş indeksini hesaplıyor */
    int base_chunk = n / size;
    int remainder  = n % size;
    int my_count   = base_chunk + (rank < remainder ? 1 : 0);
    int my_start   = rank * base_chunk + (rank < remainder ? rank : remainder);

    /* STEP 2 — Her process sadece kendi chunk'ının lokal toplamlarını hesaplıyor */
    double local_Sx = 0.0, local_Sy = 0.0;
    double local_Sxx = 0.0, local_Sxy = 0.0;

    double t_compute_start = MPI_Wtime();
    for (int i = my_start; i < my_start + my_count; i++) {
        local_Sx  += x_full[i];
        local_Sy  += y_full[i];
        local_Sxx += x_full[i] * x_full[i];
        local_Sxy += x_full[i] * y_full[i];
    }
    double t_compute_end = MPI_Wtime();
    double compute_time  = t_compute_end - t_compute_start;

    free(x_full); free(y_full);

    /* COLLECTIVE 2: MPI_Reduce — partial sumları root'ta topla */
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

    double comm_time = (t_bcast_end   - t_comm_start) +
                       (t_reduce_end  - t_reduce_start);

    /* Her processin değerlerinden en büyüğünü al (darboğaz olan process) */
    double compute_time_max = 0.0, comm_time_max = 0.0;
    MPI_Reduce(&compute_time, &compute_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time,    &comm_time_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* STEP 3 — Root sonuçları hesaplar ve raporlar */
    if (rank == 0) {
        double denom     = (double)n * global_Sxx - global_Sx * global_Sx;
        double beta1_par = ((double)n * global_Sxy - global_Sx * global_Sy) / denom;
        double beta0_par = (global_Sy - beta1_par * global_Sx) / (double)n;

        printf("=======================================================\n");
        printf("  Parallel Linear Regression (MPI) — Bcast Version\n");
        printf("  N = %d   P = %d\n", n, size);
        printf("=======================================================\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_par, beta1_par);
        printf("  Compute Time = %.6f s\n", compute_time_max);
        printf("  Comm    Time = %.6f s  (Bcast x2 + Reduce x4)\n", comm_time_max);
        printf("  Total   Time = %.6f s\n", par_time_total);
        printf("=======================================================\n");

        /* CSV'ye par_time_total, compute_time, comm_time yaz.
           Speedup ve efficiency analysis_comparison.py'de P=1 baseline alınarak hesaplanıyor. */
        FILE *fp = fopen("timing_bcast.csv", "a");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0)
                fprintf(fp, "N,P,par_time_total,compute_time,comm_time\n");
            fprintf(fp, "%d,%d,%.8f,%.8f,%.8f\n",
                    n, size, par_time_total, compute_time_max, comm_time_max);
            fclose(fp);
        }
    }

    MPI_Finalize();
    return 0;
}