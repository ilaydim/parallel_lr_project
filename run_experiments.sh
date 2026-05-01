#!/bin/bash
# run_experiments.sh
# Compiles parallel_lr.c and runs all N x P combinations.
# Results are appended to timing_output.csv (used by analysis.py).
#
# Requirements: MPI installed (mpicc, mpirun)
# Usage:        bash run_experiments.sh

# set -e removed: allow individual experiment failures without aborting the whole run

echo "=== Compiling ==="
mpicc -O2 -o parallel_lr parallel_lr.c -lm
echo "Compiled OK."

# Clear old results
rm -f timing_output.csv

# N=500000000 removed — requires ~8 GB RAM per process and crashes low-memory machines
N_VALUES=(5000000 50000000 100000000 250000000)
P_VALUES=(1 2 4 8 16 32)

for N in "${N_VALUES[@]}"; do
    for P in "${P_VALUES[@]}"; do
        echo ">>> N=${N}  P=${P}"
        mpirun --oversubscribe -np $P ./parallel_lr $N
    done
done

echo ""
echo "=== All experiments done. Results in timing_output.csv ==="
