#!/bin/bash
# File: run_eidf_nvhpc_h200.sh

set -e

export OMP_TARGET_OFFLOAD=mandatory

OUTDIR="result/EIDF_H200NVC_Output/OpenMP"
mkdir -p "$OUTDIR"

JOB_NAME="eidf_h200_nvhpc__omp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running EIDF H200 NVHPC OpenMP benchmark groups..." | tee "$OUTFILE"

THREADS=32
TEAMS=132

N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"
N_PARREPS="1,2,4,8,16,32,64,128"

DELAY_SHORT="1,8096"

echo "SET UP H200 configuration NVIDIA H200 NVHPC=====" | tee -a "$OUTFILE"
echo "THREADS=$THREADS" | tee -a "$OUTFILE"
echo "TEAMS=$TEAMS" | tee -a "$OUTFILE"
nvidia-smi | tee -a "$OUTFILE"
echo "=============================================" | tee -a "$OUTFILE"

echo "Compiling..." | tee -a "$OUTFILE"

make clean
make openmp-distribution

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

    OMP_TARGET_OFFLOAD=mandatory ./bin/microbenchmark API=openmp \
        Method="$METHODS" \
        N="$NLIST" \
        Delay="$DELAY_RANGE" \
        thread_count="$THREADS" \
        team_count="$TEAMS" \
        | tee -a "$OUTFILE"

    if [ -f overhead_distribution.txt ]; then
        mv overhead_distribution.txt "$OUTDIR/overheadEIDF_NVC${BASE}.txt"
    fi

    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distributionEIDF_NVC${BASE}.csv"
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
echo "All EIDF H200 NVHPC benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
