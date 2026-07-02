#!/bin/bash
# EIDF A100 CUDA run script for VDI sessions.

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"

MACHINE="A100"
API="CUDA"
COMPILER_TAG=$(compiler_tag_nvcc)
OUTDIR="result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$OUTDIR"

JOB_NAME="eidf_a100_cuda"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running EIDF A100 CUDA benchmark groups..." | tee "$OUTFILE"

BLOCKS=108
THREADS=128

N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"
N_PARREPS="1,2,4,8,16,32,64,128"

DELAY_SHORT="1,8096"
PLOT_PY="${PLOT_PY:-python3}"

echo "========== EIDF A100 CUDA configuration ==========" | tee -a "$OUTFILE"
echo "MACHINE=$MACHINE" | tee -a "$OUTFILE"
echo "API=$API" | tee -a "$OUTFILE"
echo "COMPILER_TAG=$COMPILER_TAG" | tee -a "$OUTFILE"
nvcc --version 2>&1 | tee -a "$OUTFILE"
echo "BLOCKS=$BLOCKS" | tee -a "$OUTFILE"
echo "THREADS=$THREADS" | tee -a "$OUTFILE"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee -a "$OUTFILE"
fi
echo "==================================================" | tee -a "$OUTFILE"

make clean
make cuda-distribution

run_group () {
    local TAG="$1"
    local METHODS="$2"
    local NLIST="$3"
    local DELAY_RANGE="$4"
    local SAFE_DELAY="${DELAY_RANGE//,/to}"
    local BASE="${TAG}_${SAFE_DELAY}_${TIMESTAMP}"

    echo "" | tee -a "$OUTFILE"
    echo "===== [$TAG] Method=$METHODS N=$NLIST Delay=$DELAY_RANGE =====" | tee -a "$OUTFILE"
    rm -f overhead_distribution.txt raw_times.csv

    ./bin/microbenchmark API=cuda \
        Method="$METHODS" \
        N="$NLIST" \
        Delay="$DELAY_RANGE" \
        block_count="$BLOCKS" \
        thread_count="$THREADS" \
        | tee -a "$OUTFILE"

    if [ -f overhead_distribution.txt ]; then
        mv overhead_distribution.txt "$OUTDIR/overheadEIDF_${BASE}.txt"
    fi
    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distributionEIDF_${BASE}.csv"
    fi
}

run_group "M0_M10" "0,10" "$N_FIXED" "$DELAY_SHORT"
run_group "M1to4" "1,2,3,4" "$N_DATA" "$DELAY_SHORT"
run_group "M5" "5" "$N_FIXED" "$DELAY_SHORT"
run_group "M6" "6" "$N_FIXED" "$DELAY_SHORT"
run_group "M7" "7" "$N_FIXED" "$DELAY_SHORT"
run_group "M8" "8" "$N_ATORED" "$DELAY_SHORT"
run_group "M9" "9" "$N_ATORED" "$DELAY_SHORT"
run_group "M11_parreps" "11" "$N_PARREPS" "$DELAY_SHORT"

echo "" | tee -a "$OUTFILE"
echo "All EIDF A100 CUDA benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
