#!/bin/bash
# Grace Hopper / GH200 CUDA run script.

set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit the job from the repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi

. "./jobs/common.sh"

MACHINE="GH200"
API="CUDA"
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
BLOCKS="$PLATFORM_GROUPS"
THREADS="$PLATFORM_WIDTH"
COMPILER_TAG=$(compiler_tag_nvcc)
OUTDIR="result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$OUTDIR"

JOB_NAME="gh_cuda"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"
archive_run_provenance "$0" "$OUTDIR" "$JOB_NAME" "$TIMESTAMP"

echo "Running Grace Hopper CUDA benchmark groups..." | tee "$OUTFILE"

PLOT_PY="${PLOT_PY:-$PWD/.venv/bin/python}"
require_plot_python "$PLOT_PY"

echo "========== Grace Hopper CUDA configuration ==========" | tee -a "$OUTFILE"
echo "MACHINE=$MACHINE" | tee -a "$OUTFILE"
echo "API=$API" | tee -a "$OUTFILE"
echo "COMPILER_TAG=$COMPILER_TAG" | tee -a "$OUTFILE"
echo "BLOCKS=$BLOCKS" | tee -a "$OUTFILE"
echo "THREADS=$THREADS" | tee -a "$OUTFILE"
nvcc --version 2>&1 | tee -a "$OUTFILE"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee -a "$OUTFILE"
fi
echo "=====================================================" | tee -a "$OUTFILE"

echo "Compiling CUDA backend..." | tee -a "$OUTFILE"
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
echo "All Grace Hopper CUDA benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
