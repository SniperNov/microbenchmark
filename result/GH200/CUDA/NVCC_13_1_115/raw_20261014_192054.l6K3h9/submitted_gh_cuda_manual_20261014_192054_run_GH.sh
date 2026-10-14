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
API="CUDA"
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
GROUPS_COUNT="$PLATFORM_GROUPS"
THREADS="$PLATFORM_WIDTH"
DEVICE_TIMING="${CUDA_DEVICE_TIMING:-1}"
case "$DEVICE_TIMING" in
    0|1) ;;
    *) echo "CUDA_DEVICE_TIMING must be 0 or 1" >&2; exit 1 ;;
esac
# Resolve an optional profile before changing into each group's output directory.
if [ -n "${CUDA_OPENMP_PROFILE:-}" ]; then
    profile_dir=$(CDPATH= cd -- "$(dirname -- "$CUDA_OPENMP_PROFILE")" && pwd)
    export CUDA_OPENMP_PROFILE="$profile_dir/$(basename -- "$CUDA_OPENMP_PROFILE")"
    test -s "$CUDA_OPENMP_PROFILE"
fi
COMPILER_TAG=$(compiler_tag_nvcc)
result_root="$repo_dir/result/$MACHINE/$API/$COMPILER_TAG"
mkdir -p "$result_root"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR=$(mktemp -d "$result_root/raw_${TIMESTAMP}.XXXXXX")
JOB_NAME="gh_cuda"
OUTFILE="$OUTDIR/run.log"
archive_run_provenance "$0" "$OUTDIR" "$JOB_NAME" "$TIMESTAMP"
cp jobs/common.sh "$OUTDIR/common.sh"
if [ -n "${CUDA_OPENMP_PROFILE:-}" ]; then
    cp "$CUDA_OPENMP_PROFILE" "$OUTDIR/openmp_profile.csv"
    export CUDA_OPENMP_PROFILE="$OUTDIR/openmp_profile.csv"
fi

printf 'Raw-data output directory: %s\n' "$OUTDIR" | tee "$OUTFILE"
log_run_configuration "$OUTFILE" "$MACHINE" "$API" "$COMPILER_TAG" "$GROUPS_COUNT" "$THREADS"
make -B cuda-distribution EXTRA_CFLAGS="-arch=sm_90" 2>&1 | tee -a "$OUTFILE"

run_group () {
    local TAG="$1" METHODS="$2" NLIST="$3" DELAY_RANGE="$4"
    local group_dir="$OUTDIR/baseline/$TAG"
    mkdir -p "$group_dir"
    printf '\n===== [%s] Method=%s N=%s Delay=%s =====\n' \
        "$TAG" "$METHODS" "$NLIST" "$DELAY_RANGE" | tee -a "$OUTFILE"
    (
        cd "$group_dir"
        "$repo_dir/build/microbenchmark_cuda" API=cuda \
            Method="$METHODS" N="$NLIST" Delay="$DELAY_RANGE" \
            block_count="$GROUPS_COUNT" thread_count="$THREADS" \
            DeviceTiming="$DEVICE_TIMING" \
            2>&1 | tee run.log
    ) | tee -a "$OUTFILE"
    if [ "$DEVICE_TIMING" = "1" ]; then
        mkdir -p "$OUTDIR/kernel_execution/$TAG"
        mv "$group_dir/cuda_device_times.csv" \
            "$OUTDIR/kernel_execution/$TAG/cuda_device_times.csv"
    fi
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
