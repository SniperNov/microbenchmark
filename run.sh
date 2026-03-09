#!/bin/bash
# File: run_all_benchmarks.sh

OUTDIR="MBResult_A/Archer2_Output"
mkdir -p "$OUTDIR"

JOB_NAME="ar2_microbench"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running microbenchmark..." | tee $OUTFILE
./microbenchmark_distribution Delay=1,8096 Method=6 thread_count=64 team_count=104 | tee -a $OUTFILE

echo "Run completed. Output saved to $OUTFILE"
mv overhead_distribution.txt $OUTDIR/overhead_$TIMESTAMP.txt
mv raw_times.csv $OUTDIR/distribution_$TIMESTAMP.csv

/opt/cray/pe/python/3.9.13.1/bin/python plot_raw_times.py $OUTDIR/distribution_$TIMESTAMP.csv $OUTDIR/overhead_$TIMESTAMP.txt 1,18 lin resultLoweast.png
