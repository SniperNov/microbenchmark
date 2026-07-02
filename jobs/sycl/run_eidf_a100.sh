#!/bin/bash
# EIDF A100 SYCL run script for VDI sessions.

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/../common.sh"

MACHINE="A100"
API="SYCL"
COMPILER_TAG=$(compiler_tag_sycl)
OUTDIR="result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$OUTDIR"

JOB_NAME="eidf_a100_sycl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

GROUPS=108
LOCAL_SIZE=128

N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"
N_PARREPS="1,2,4,8,16,32,64,128"
DELAY_SHORT="1,8096"

echo "Running EIDF A100 SYCL benchmark groups..." | tee "$OUTFILE"
make clean
make sycl-distribution SYCL_DEFS=Makefile.defs.sycl.dpcpp.cuda

run_group () {
    local TAG="$1"
    local METHODS="$2"
    local NLIST="$3"
    local DELAY_RANGE="$4"
    local BASE="${TAG}_${DELAY_RANGE//,/to}_${TIMESTAMP}"
    echo "" | tee -a "$OUTFILE"
    echo "===== [$TAG] Method=$METHODS N=$NLIST Delay=$DELAY_RANGE =====" | tee -a "$OUTFILE"
    rm -f overhead_distribution.txt raw_times.csv
    ./bin/microbenchmark API=sycl Method="$METHODS" N="$NLIST" Delay="$DELAY_RANGE" group_count="$GROUPS" local_size="$LOCAL_SIZE" | tee -a "$OUTFILE"
    [ -f overhead_distribution.txt ] && mv overhead_distribution.txt "$OUTDIR/overhead_${BASE}.txt"
    [ -f raw_times.csv ] && mv raw_times.csv "$OUTDIR/distribution_${BASE}.csv"
}

run_group "M0_M10" "0,10" "$N_FIXED" "$DELAY_SHORT"
run_group "M1to4" "1,2,3,4" "$N_DATA" "$DELAY_SHORT"
run_group "M5" "5" "$N_FIXED" "$DELAY_SHORT"
run_group "M6" "6" "$N_FIXED" "$DELAY_SHORT"
run_group "M7" "7" "$N_FIXED" "$DELAY_SHORT"
run_group "M8" "8" "$N_ATORED" "$DELAY_SHORT"
run_group "M9" "9" "$N_ATORED" "$DELAY_SHORT"
run_group "M11_parreps" "11" "$N_PARREPS" "$DELAY_SHORT"
