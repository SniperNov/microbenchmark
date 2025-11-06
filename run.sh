#!/bin/bash
# File: run_all_benchmarks.sh

OUTDIR="MBResult/GH_Output"
mkdir -p "$OUTDIR"

JOB_NAME="gh_microbench"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running microbenchmark..." | tee $OUTFILE
./microbenchmark_distribution N=2048,4096,8192,16382,32764,65528 Delay=262144 thread_count=32 team_count=184 | tee -a $OUTFILE

echo "Run completed. Output saved to $OUTFILE"
mv overhead_distribution.txt $OUTDIR/overhead_$TIMESTAMP.txt
