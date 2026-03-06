#!/bin/bash
# File: run_all_benchmarks.sh

source /work/weiyu/microbenchmark/.venv/bin/activate

OUTDIR="ACCResult/GH_Output"
mkdir -p "$OUTDIR"

JOB_NAME="gh_microbench"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running microbenchmark..." | tee $OUTFILE
./microbenchmark_dist Delay=1,8096 Method=9 thread_count=32 team_count=132 | tee -a $OUTFILE

echo "Run completed. Output saved to $OUTFILE"
mv overhead_distribution.txt $OUTDIR/overhead_$TIMESTAMP.txt
mv raw_times.csv $OUTDIR/distribution_$TIMESTAMP.csv


python plot_raw_times.py $OUTDIR/distribution_$TIMESTAMP.csv $OUTDIR/overhead_$TIMESTAMP.txt 1,18 log ACCM9.png
