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
    fgets(header, sizeof(header), fp);

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

/* Compute MSE on the full dataset (root only, for verification) */
double compute_mse(const double *x, const double *y, int n, double beta0, double beta1)
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

    int n = 1000000; /* n (process size) değeri verilmezse default 1 milyon */
    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            if (rank == 0) fprintf(stderr, "Error: N must be positive.\n");
            MPI_Finalize();
            return 1;
        }
    }

    /* STEP 1 — Root bellek ayırıp veriyi yüklüyor. Diğerleri hiçbir şey yapmıyor. */
    double *x_full = NULL, *y_full = NULL;
    double  mse_par = -1.0;

    if (rank == 0) {
        x_full = (double *)malloc((size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!x_full || !y_full) {
            fprintf(stderr, "Root: malloc failed for full dataset\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* California Housing verisini yükle, N'e ulaşmak için tekrarla */
        load_data(x_full, y_full, n, "california_housing.csv");
    }

    MPI_Barrier(MPI_COMM_WORLD); /* Paralel timer'ı senkronize başlat */
    double par_start = MPI_Wtime();

    /* ================================================================
       COLLECTIVE FUNCTION 1: MPI_Bcast
       Root, sadece N sayısını tüm processlere gönderiyor.
       N sayısını bilince herkes kendi chunk boyutunu hesaplar.
       ================================================================ */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Chunk size hesaplanıyor. N her zaman P'ye tam bölünmeyebilir
       o yüzden remainder da hesaplandı. */
    int base_chunk = n / size;
    int remainder  = n % size;

    int *send_counts = (int *)malloc(size * sizeof(int)); /* her process kaç eleman alacak */
    int *displs      = (int *)malloc(size * sizeof(int)); /* her processin dizide nereden başlayacağı */
    if (!send_counts || !displs) {
        fprintf(stderr, "Process %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1); /* malloc başarısız olursa tüm processleri durdurur */
    }

    /* Scatterv için send_counts ve displs dizilerini oluşturuyoruz */
    int offset = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
        displs[i]      = offset;
        offset        += send_counts[i];
    }
    int my_count = send_counts[rank];

    /* Her process sadece kendi chunk'ı kadar bellek ayırıyor — N/P eleman */
    double *x_local = (double *)malloc((size_t)my_count * sizeof(double));
    double *y_local = (double *)malloc((size_t)my_count * sizeof(double));
    if (!x_local || !y_local) {
        fprintf(stderr, "Process %d: malloc failed for local chunk\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ================================================================
       COLLECTIVE FUNCTION 2: MPI_Scatterv
       x dizisi için bir, y dizisi için bir kez olmak üzere iki kez çağrılıyor.
       ================================================================ */
    double t_scatter_start = MPI_Wtime();
    MPI_Scatterv(x_full, send_counts, displs, MPI_DOUBLE,
                 x_local, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(y_full, send_counts, displs, MPI_DOUBLE,
                 y_local, my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();

    /* ================================================================
       STEP 2 — Her process kendi lokal toplamlarını hesaplar.
       Timer ile sadece bu döngünün süresi ayrıca ölçülüyor
       (comm_time'dan ayrıştırmak için).
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
       Dört kez çağrılıyor — Sx, Sy, Sxx, Sxy için.
       Her çağrıda tüm processlerin o değeri toplanıp root'a geliyor.
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
    /* par_time_total: Parallel fazın başından sonuna kadar geçen toplam süre.
       İçinde her şey var: iletişim + hesaplama + barrier beklemeleri. */
    double par_time_total = MPI_Wtime() - par_start;
    double comm_time = (t_scatter_end - t_scatter_start) + (t_reduce_end - t_reduce_start);

    /* Her processin compute_time ve comm_time değerlerinden en büyüğünü al
       (darboğaz olan process sistemi bekletiyor) */
    double compute_time_max = 0.0;
    MPI_Reduce(&compute_time, &compute_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double comm_time_max = 0.0;
    MPI_Reduce(&comm_time,    &comm_time_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ================================================================
       STEP 3 — ROOT COMPUTES FINAL COEFFICIENTS AND REPORTS
       Speedup ve efficiency analysis.py'de P=1 baseline'ı kullanılarak hesaplanıyor.
       ================================================================ */
    if (rank == 0) {
        double denom     = (double)n * global_Sxx - global_Sx * global_Sx;
        double beta1_par = ((double)n * global_Sxy - global_Sx * global_Sy) / denom;
        double beta0_par = (global_Sy - beta1_par * global_Sx) / (double)n;

        /* x_full ve y_full hâlâ bellekte — MSE hesapla, sonra free et */
        mse_par = compute_mse(x_full, y_full, n, beta0_par, beta1_par);
        free(x_full); x_full = NULL;
        free(y_full); y_full = NULL;

        printf("=======================================================\n");
        printf("  Parallel Linear Regression (MPI)\n");
        printf("  N = %d   P = %d\n", n, size);
        printf("=======================================================\n");
        printf("  beta0 = %.6f   beta1 = %.6f\n", beta0_par, beta1_par);
        printf("  MSE   = %.8f\n", mse_par);
        printf("  Compute Time = %.6f s\n", compute_time_max);
        printf("  Comm    Time = %.6f s  (Scatterv x2 + Reduce x4)\n", comm_time_max);
        printf("  Total   Time = %.6f s\n", par_time_total);
        printf("=======================================================\n");

        /* CSV'ye par_time_total, compute_time, comm_time yaz.
           Speedup ve efficiency analysis.py'de P=1 baseline alınarak hesaplanıyor. */
        FILE *fp = fopen("timing_output.csv", "a");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            if (ftell(fp) == 0)
                fprintf(fp, "N,P,par_time_total,compute_time,comm_time\n");
            fprintf(fp, "%d,%d,%.8f,%.8f,%.8f\n",
                    n, size, par_time_total, compute_time_max, comm_time_max);
            fclose(fp);
        }
    }

    free(send_counts);
    free(displs);
    MPI_Finalize();
    return 0;
}