#!/bin/bash
# Grace Hopper / GH200 OpenACC run script.

set -euo pipefail

export ACC_DEVICE_TYPE=nvidia
unset NVCOMPILER_ACC_NOTIFY NVCOMPILER_ACC_TIME

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit the job from the repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi

. "./jobs/common.sh"

MACHINE="GH200"
API="OpenACC"
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
GANGS="$PLATFORM_GROUPS"
WORKERS="$PLATFORM_WIDTH"
COMPILER_TAG=$(compiler_tag_nvc)
OUTDIR="result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$OUTDIR"

JOB_NAME="gh_openacc"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"
archive_run_provenance "$0" "$OUTDIR" "$JOB_NAME" "$TIMESTAMP"

echo "Running Grace Hopper OpenACC benchmark groups..." | tee "$OUTFILE"

PLOT_PY="${PLOT_PY:-$PWD/.venv/bin/python}"
require_plot_python "$PLOT_PY"

log_run_configuration "$OUTFILE" "$MACHINE" "$API" "$COMPILER_TAG" "$GANGS" "$WORKERS"

echo "Compiling OpenACC backend..." | tee -a "$OUTFILE"
make -s clean
make -s openacc-distribution OPENACC_DEFS=Makefile.defs.openacc.nvc

plot_group () {
    local BASE="$1"
    local SCALE="$2"
    local CSV="$OUTDIR/distribution_${BASE}.csv"
    local TXT="$OUTDIR/overhead_${BASE}.txt"
    local PNG="$OUTDIR/resultLowest_${BASE}.png"

    if [ -f "$CSV" ] && [ -f "$TXT" ] && command -v "$PLOT_PY" >/dev/null 2>&1; then
        "$PLOT_PY" ./plots/plot_raw_times.py "$CSV" "$TXT" 1,18 "$SCALE" "$PNG" 2>&1 | tee -a "$OUTFILE"
    else
        echo "Plot skipped for $BASE." | tee -a "$OUTFILE"
    fi
}

run_group () {
    local TAG="$1"
    local METHODS="$2"
    local NLIST="$3"
    local DELAY_RANGE="$4"
    local SCALE="${5:-log}"

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
        mv overhead_distribution.txt "$OUTDIR/overhead_${BASE}.txt"
    fi

    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distribution_${BASE}.csv"
    fi

    plot_group "$BASE" "$SCALE"
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
echo "All Grace Hopper OpenACC benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"
