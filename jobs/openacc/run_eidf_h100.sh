#!/bin/bash
# EIDF H100 OpenACC run script for VDI sessions.

set -e

OUTDIR="result/EIDF_H100_Output/OpenACC"
mkdir -p "$OUTDIR"

JOB_NAME="eidf_h100_openacc"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running EIDF H100 OpenACC benchmark groups..." | tee "$OUTFILE"

GANGS=132
WORKERS=128

N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"
N_PARREPS="1,2,4,8,16,32,64,128"

DELAY_SHORT="1,8096"
PLOT_PY="${PLOT_PY:-python3}"

echo "========== EIDF H100 OpenACC configuration ==========" | tee -a "$OUTFILE"
echo "GANGS=$GANGS" | tee -a "$OUTFILE"
echo "WORKERS=$WORKERS" | tee -a "$OUTFILE"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee -a "$OUTFILE"
fi
echo "=====================================================" | tee -a "$OUTFILE"

echo "Compiling OpenACC backend with NVHPC..." | tee -a "$OUTFILE"
make clean
make openacc-distribution OPENACC_DEFS=Makefile.defs.openacc.nvc

plot_group () {
    local BASE="$1"
    local CSV="$OUTDIR/distributionEIDF_${BASE}.csv"
    local TXT="$OUTDIR/overheadEIDF_${BASE}.txt"
    local PNG="$OUTDIR/resultLowestEIDF_${BASE}.png"

    if [ -f "$CSV" ] && [ -f "$TXT" ] && command -v "$PLOT_PY" >/dev/null 2>&1; then
        "$PLOT_PY" ./plots/plot_raw_times.py "$CSV" "$TXT" 1,18 log "$PNG" 2>&1 | tee -a "$OUTFILE"
    else
        echo "Plot skipped for $BASE." | tee -a "$OUTFILE"
    fi
}

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

    ./bin/microbenchmark API=openacc \
        Method="$METHODS" \
        N="$NLIST" \
        Delay="$DELAY_RANGE" \
        gang_count="$GANGS" \
        worker_count="$WORKERS" \
        | tee -a "$OUTFILE"

    if [ -f overhead_distribution.txt ]; then
        mv overhead_distribution.txt "$OUTDIR/overheadEIDF_${BASE}.txt"
    fi

    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distributionEIDF_${BASE}.csv"
    fi

    plot_group "$BASE"
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
echo "All EIDF H100 OpenACC benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
