#!/bin/bash
# run_comparison.sh
# Her iki versiyonu da derler ve aynı N x P kombinasyonlarında koşturur.
# Çıktılar: timing_output.csv (Scatterv) ve timing_bcast.csv (Bcast)
#
# Usage: bash run_comparison.sh

set -e

echo "=== Compiling ==="
mpicc -O2 -o parallel_lr       parallel_lr.c       -lm
mpicc -O2 -o parallel_lr_bcast parallel_lr_bcast.c -lm
echo "Compiled OK."

rm -f timing_output.csv timing_bcast.csv

N_VALUES=(5000000 50000000 100000000)
P_VALUES=(1 2 4 8 16 32)

# N=250M icin sadece kucuk P degerlerini dene (bellek)
N_LARGE=250000000
P_LARGE=(1 2 4 8 16)

echo ""
echo "=== Running Scatterv version ==="
for N in "${N_VALUES[@]}"; do
    for P in "${P_VALUES[@]}"; do
        echo "  [Scatterv] N=${N} P=${P}"
        mpirun --oversubscribe -np $P ./parallel_lr $N
    done
done
for P in "${P_LARGE[@]}"; do
    echo "  [Scatterv] N=${N_LARGE} P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr $N_LARGE
done

echo ""
echo "=== Running Bcast version ==="
for N in "${N_VALUES[@]}"; do
    for P in "${P_VALUES[@]}"; do
        echo "  [Bcast]    N=${N} P=${P}"
        mpirun --oversubscribe -np $P ./parallel_lr_bcast $N
    done
done
for P in "${P_LARGE[@]}"; do
    echo "  [Bcast]    N=${N_LARGE} P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr_bcast $N_LARGE
done

echo ""
echo "=== Done ==="
echo "  Scatterv results : timing_output.csv"
echo "  Bcast    results : timing_bcast.csv"
