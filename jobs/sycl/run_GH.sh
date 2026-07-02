#!/bin/bash
# Grace Hopper / GH200 SYCL run script.

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"

MACHINE="GH200"
API="SYCL"
COMPILER_TAG=$(compiler_tag_sycl)
OUTDIR="result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$OUTDIR"

JOB_NAME="gh_sycl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running Grace Hopper SYCL benchmark groups..." | tee "$OUTFILE"

GROUPS=132
LOCAL_SIZE=128

N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"
N_PARREPS="1,2,4,8,16,32,64,128"

DELAY_SHORT="1,8096"
PLOT_PY="${PLOT_PY:-python3}"

echo "========== Grace Hopper SYCL configuration ==========" | tee -a "$OUTFILE"
echo "MACHINE=$MACHINE" | tee -a "$OUTFILE"
echo "API=$API" | tee -a "$OUTFILE"
echo "COMPILER_TAG=$COMPILER_TAG" | tee -a "$OUTFILE"
echo "GROUPS=$GROUPS" | tee -a "$OUTFILE"
echo "LOCAL_SIZE=$LOCAL_SIZE" | tee -a "$OUTFILE"
if command -v icpx >/dev/null 2>&1; then icpx --version | tee -a "$OUTFILE"; fi
if command -v dpcpp >/dev/null 2>&1; then dpcpp --version | tee -a "$OUTFILE"; fi
if command -v acpp >/dev/null 2>&1; then acpp --version | tee -a "$OUTFILE"; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi | tee -a "$OUTFILE"; fi
echo "=====================================================" | tee -a "$OUTFILE"

echo "Compiling SYCL backend..." | tee -a "$OUTFILE"
make clean
make sycl-distribution SYCL_DEFS=Makefile.defs.sycl.dpcpp.cuda

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

    ./bin/microbenchmark API=sycl \
        Method="$METHODS" \
        N="$NLIST" \
        Delay="$DELAY_RANGE" \
        group_count="$GROUPS" \
        local_size="$LOCAL_SIZE" \
        | tee -a "$OUTFILE"

    if [ -f overhead_distribution.txt ]; then
        mv overhead_distribution.txt "$OUTDIR/overhead_${BASE}.txt"
    fi

    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distribution_${BASE}.csv"
    fi

    if [ -f "$OUTDIR/distribution_${BASE}.csv" ] && [ -f "$OUTDIR/overhead_${BASE}.txt" ] && command -v "$PLOT_PY" >/dev/null 2>&1; then
        "$PLOT_PY" ./plots/plot_raw_times.py \
            "$OUTDIR/distribution_${BASE}.csv" \
            "$OUTDIR/overhead_${BASE}.txt" \
            1,18 log \
            "$OUTDIR/resultLowest_${BASE}.png" 2>&1 | tee -a "$OUTFILE"
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
echo "All Grace Hopper SYCL benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
