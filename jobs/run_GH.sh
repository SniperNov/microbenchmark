#!/bin/bash
# File: GH_run.sh

set -e

make clean
make distribution

OUTDIR="result/GH_Output/OpenMP"
mkdir -p "$OUTDIR"

JOB_NAME="gh_omp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/${JOB_NAME}_${TIMESTAMP}.out"

echo "Running Grace Hopper OpenMP benchmark groups..." | tee "$OUTFILE"

# =========================================
# Machine-specific configuration: Grace Hopper
# =========================================
THREADS=32
TEAMS=132

# =========================================
# Experiment configuration
# =========================================
N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
N_FIXED="16382"
N_ATORED="16,64,256,1024,4096,8192,16382"

DELAY_SHORT="1,8096"
DELAY_FULL="1,262144"

PLOT_PY="/work/weiyu/microbenchmark/.venv/bin/python"
if [ ! -x "$PLOT_PY" ]; then
    echo "Error: plotting Python not found or not executable: $PLOT_PY"
    exit 1
fi

run_group () {
    local TAG="$1"
    local METHODS="$2"
    local NLIST="$3"
    local DELAY_RANGE="$4"

    local SAFE_DELAY="${DELAY_RANGE//,/to}"
    local BASE="${TAG}_${SAFE_DELAY}_${TIMESTAMP}"

    echo "" | tee -a "$OUTFILE"
    echo "===== [$TAG] Method=$METHODS N=$NLIST Delay=$DELAY_RANGE =====" | tee -a "$OUTFILE"

    ./microbenchmark_distribution \
        Method=$METHODS \
        N=$NLIST \
        Delay=$DELAY_RANGE \
        thread_count=$THREADS \
        team_count=$TEAMS \
        | tee -a "$OUTFILE"


    if [ -f overhead_distribution.txt ]; then
        mv overhead_distribution.txt "$OUTDIR/overhead_${BASE}.txt"
    fi

    if [ -f raw_times.csv ]; then
        mv raw_times.csv "$OUTDIR/distribution_${BASE}.csv"
    fi

    if [ -f "$OUTDIR/distribution_${BASE}.csv" ] && [ -f "$OUTDIR/overhead_${BASE}.txt" ]; then
        echo "Plotting $BASE ..." | tee -a "$OUTFILE"

        "$PLOT_PY" ./plots/plot_raw_times.py \
            "$OUTDIR/distribution_${BASE}.csv" \
            "$OUTDIR/overhead_${BASE}.txt" \
            0,19 log \
            "$OUTDIR/resultLowest_${BASE}.png" 2>&1 | tee -a "$OUTFILE"

        if [ -f "$OUTDIR/resultLowest_${BASE}.png" ]; then
            echo "Plot generated: $OUTDIR/resultLowest_${BASE}.png" | tee -a "$OUTFILE"
        else
            echo "Plot failed for $BASE" | tee -a "$OUTFILE"
        fi
    else
        echo "Plot skipped for $BASE because csv/txt missing." | tee -a "$OUTFILE"
    fi
}

# =========================================
# Group 1: methods 1-4, data size sweep
# =========================================
run_group "M1to4" "1,2,3,4" "$N_DATA" "$DELAY_SHORT"

# =========================================
# Group 2: methods 5 / 6 / 7, fixed N
# =========================================
run_group "M5" "5" "$N_FIXED" "$DELAY_SHORT"
run_group "M5" "5" "$N_FIXED" "$DELAY_FULL"

run_group "M6" "6" "$N_FIXED" "$DELAY_SHORT"
run_group "M6" "6" "$N_FIXED" "$DELAY_FULL"

run_group "M7" "7" "$N_FIXED" "$DELAY_SHORT"
run_group "M7" "7" "$N_FIXED" "$DELAY_FULL"

# =========================================
# Group 3: methods 8 / 9, atomic + reduction
# =========================================
run_group "M8to9" "8,9" "$N_ATORED" "$DELAY_SHORT"
run_group "M8to9" "8,9" "$N_ATORED" "$DELAY_FULL"

# =========================================
# Group 4a: method 0 / 10, fixed N
# =========================================
run_group "M0_M10" "0,10" "$N_FIXED" "$DELAY_SHORT"

# =========================================
# Group 4b: method 11, N is parreps
# =========================================
N_PARREPS="1,2,4,8,16,32,64,128"

run_group "M11_parreps" "11" "$N_PARREPS" "$DELAY_SHORT"


echo "" | tee -a "$OUTFILE"
echo "All benchmark groups completed." | tee -a "$OUTFILE"
echo "Results saved under $OUTDIR" | tee -a "$OUTFILE"