#!/bin/bash

OUTDIR="result/GH_Output/OpenACC"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/openacc_${TIMESTAMP}.out"

make openacc

./bin/microbenchmark API=openacc Delay=1,8096 Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 gang_count=64 worker_count=128 | tee "$OUTFILE"

mv overhead_distribution.txt "$OUTDIR/overhead_$TIMESTAMP.txt"
mv raw_times.csv "$OUTDIR/distribution_$TIMESTAMP.csv"
