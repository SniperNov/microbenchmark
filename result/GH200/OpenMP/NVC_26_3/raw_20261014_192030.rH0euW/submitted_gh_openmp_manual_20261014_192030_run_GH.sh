#!/bin/bash
# GH200 raw-data collection; benchmark groups match the OpenMP reference.
set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit the job from the repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi
repo_dir="$PWD"
. "./jobs/common.sh"

MACHINE="GH200"
API="OpenMP"
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
GROUPS_COUNT="$PLATFORM_GROUPS"
THREADS="$PLATFORM_WIDTH"
COMPILER_TAG=$(compiler_tag_nvc)
result_root="$repo_dir/result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$result_root"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=$(mktemp -d "$result_root/raw_${TIMESTAMP}.XXXXXX")
JOB_NAME="gh_openmp"
OUTFILE="$OUTDIR/run.log"
archive_run_provenance "$0" "$OUTDIR" "$JOB_NAME" "$TIMESTAMP"
cp jobs/common.sh "$OUTDIR/common.sh"

printf 'Raw-data output directory: %s\n' "$OUTDIR" | tee "$OUTFILE"
log_run_configuration "$OUTFILE" "$MACHINE" "$API" "$COMPILER_TAG" "$GROUPS_COUNT" "$THREADS"
make -B openmp-distribution OPENMP_DEFS=Makefile.defs.nvc 2>&1 | tee -a "$OUTFILE"

run_group () {
    local TAG="$1" METHODS="$2" NLIST="$3" DELAY_RANGE="$4"
    local group_dir="$OUTDIR/baseline/$TAG"
    mkdir -p "$group_dir"
    printf '\n===== [%s] Method=%s N=%s Delay=%s =====\n' \
        "$TAG" "$METHODS" "$NLIST" "$DELAY_RANGE" | tee -a "$OUTFILE"
    (
        cd "$group_dir"
        "$repo_dir/build/microbenchmark_openmp" API=openmp \
            Method="$METHODS" N="$NLIST" Delay="$DELAY_RANGE" \
            thread_count="$THREADS" team_count="$GROUPS_COUNT" \
            2>&1 | tee run.log
    ) | tee -a "$OUTFILE"
}

run_group "M0_M10" "0,10" "$N_FIXED" "$DELAY_SHORT"
run_group "M1to4" "1,2,3,4" "$N_DATA"   "$DELAY_SHORT"
run_group "M5"    "5"       "$N_FIXED"  "$DELAY_SHORT"
run_group "M6"    "6"       "$N_FIXED"  "$DELAY_SHORT"
run_group "M7"    "7"       "$N_FIXED"  "$DELAY_SHORT"
run_group "M8" "8"     "$N_ATORED" "$DELAY_SHORT"
run_group "M9" "9"     "$N_ATORED" "$DELAY_SHORT"
run_group "M11_parreps" "11" "$N_PARREPS" "$DELAY_SHORT"

printf '\nRaw-data collection completed: %s\n' "$OUTDIR" | tee -a "$OUTFILE"
