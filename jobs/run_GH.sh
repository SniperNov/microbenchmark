#!/bin/bash

OUTDIR="result/GH_Output/OpenACC"
mkdir -p "$OUTDIR"

JOB_NAME="gh_microbench_openacc"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running OpenACC microbenchmark..." | tee "$OUTFILE"
./microbenchmark_distribution Delay=1,8096 Method=11 N=16384 gang_count=64 vector_length=128 | tee -a "$OUTFILE"

echo "Run completed. Output saved to $OUTFILE"
mv overhead_distribution.txt "$OUTDIR/overhead_$TIMESTAMP.txt"
mv raw_times.csv "$OUTDIR/distribution_$TIMESTAMP.csv"

python3 ./plots/plot_raw_times.py \
    "$OUTDIR/distribution_$TIMESTAMP.csv" \
    "$OUTDIR/overhead_$TIMESTAMP.txt" \
    1,18 log \
    "$OUTDIR/resultLowest_$TIMESTAMP.png"
