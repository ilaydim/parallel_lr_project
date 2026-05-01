#!/bin/bash
# run_experiments.sh

set -e

echo "=== Compiling ==="
mpicc -O2 -o parallel_lr parallel_lr.c -lm
echo "Compiled OK."

rm -f timing_output.csv

echo ">>> N=5000000"
for P in 1 2 4 8 16 32; do
    echo "  P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr 5000000
done

echo ">>> N=50000000"
for P in 1 2 4 8 16 32; do
    echo "  P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr 50000000
done

echo ">>> N=100000000"
for P in 1 2 4 8 16 32; do
    echo "  P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr 100000000
done

echo ">>> N=250000000"
for P in 1 2 4 8 16; do
    echo "  P=${P}"
    mpirun --oversubscribe -np $P ./parallel_lr 250000000
done

echo ""
echo "=== All experiments done. Results in timing_output.csv ==="