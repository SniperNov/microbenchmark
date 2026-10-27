#!/usr/bin/env bash
# Raw HIP results; benchmark groups and platform controls match OpenMP.
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi
repo_dir="$PWD"
. ./jobs/common.sh
MACHINE="${HIP_MACHINE:-MI300X}"
case "$MACHINE" in MI210|MI300X) ;; *) echo "HIP_MACHINE must be MI210 or MI300X" >&2; exit 1 ;; esac
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
DEVICE_TIMING="${HIP_DEVICE_TIMING:-0}"
case "$DEVICE_TIMING" in 0|1) ;; *) echo "HIP_DEVICE_TIMING must be 0 or 1" >&2; exit 1 ;; esac
command -v hipcc >/dev/null
if [ -n "${HIP_OPENMP_PROFILE:-}" ]; then
    profile_dir=$(CDPATH= cd -- "$(dirname -- "$HIP_OPENMP_PROFILE")" && pwd)
    export HIP_OPENMP_PROFILE="$profile_dir/$(basename -- "$HIP_OPENMP_PROFILE")"
    test -s "$HIP_OPENMP_PROFILE"
fi
version=$(hipcc --version 2>&1 | sed -n 's/.*HIP version:[[:space:]]*\([^[:space:]]*\).*/\1/p' | head -n 1)
COMPILER_TAG=$(format_compiler_tag HIPCC "$version")
result_root="$repo_dir/result/$MACHINE/HIP/$COMPILER_TAG"
mkdir -p "$result_root"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR=$(mktemp -d "$result_root/raw_${TIMESTAMP}.XXXXXX")
OUTFILE="$OUTDIR/run.log"
archive_run_provenance "$0" "$OUTDIR" hip "$TIMESTAMP"
cp jobs/common.sh "$OUTDIR/common.sh"
mkdir -p "$OUTDIR/sources/src/backends" "$OUTDIR/sources/src/core" "$OUTDIR/bin"
cp -R src/backends/hip "$OUTDIR/sources/src/backends/"
cp src/core/driver.c src/core/backend.h "$OUTDIR/sources/src/core/"
cp Makefile Makefile.backend Makefile.defs.hip.hipcc "$OUTDIR/sources/"
if [ -n "${HIP_OPENMP_PROFILE:-}" ]; then
    cp "$HIP_OPENMP_PROFILE" "$OUTDIR/openmp_profile.csv"
    export HIP_OPENMP_PROFILE="$OUTDIR/openmp_profile.csv"
fi
{
    hostname
    hipcc --version
    printf 'machine=%s blocks=%s threads=%s HIP_ARCH=%s DeviceTiming=%s\n' \
        "$MACHINE" "$PLATFORM_GROUPS" "$PLATFORM_WIDTH" "${HIP_ARCH:-auto}" "$DEVICE_TIMING"
    printf 'HIP_VISIBLE_DEVICES=%s ROCR_VISIBLE_DEVICES=%s\n' "${HIP_VISIBLE_DEVICES:-unset}" "${ROCR_VISIBLE_DEVICES:-unset}"
    if command -v rocminfo >/dev/null 2>&1; then rocminfo; fi
} > "$OUTDIR/environment.log" 2>&1
printf 'Raw HIP output directory: %s\n' "$OUTDIR" | tee "$OUTFILE"
make -B hip-distribution HIP_ARCH="${HIP_ARCH:-}" 2>&1 | tee -a "$OUTFILE"
cp build/microbenchmark_hip "$OUTDIR/bin/"
run_binary="$OUTDIR/bin/microbenchmark_hip"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$run_binary" > "$OUTDIR/binary_sha256.txt"
fi
run_group() {
    local tag="$1" methods="$2" sizes="$3" delays="$4"
    local group_dir="$OUTDIR/baseline/$tag"
    mkdir -p "$group_dir"
    printf '\n===== %s Method=%s N=%s Delay=%s =====\n' "$tag" "$methods" "$sizes" "$delays" | tee -a "$OUTFILE"
    (
        cd "$group_dir"
        "$run_binary" API=hip Method="$methods" N="$sizes" Delay="$delays" \
            block_count="$PLATFORM_GROUPS" thread_count="$PLATFORM_WIDTH" \
            DeviceTiming="$DEVICE_TIMING" 2>&1 | tee run.log
    ) | tee -a "$OUTFILE"
    if [ "$DEVICE_TIMING" = 1 ]; then
        mkdir -p "$OUTDIR/kernel_execution/$tag"
        mv "$group_dir/hip_device_times.csv" "$OUTDIR/kernel_execution/$tag/"
    fi
}
run_group M0_M10 0,10 "$N_FIXED" "$DELAY_SHORT"
run_group M1to4 1,2,3,4 "$N_DATA" "$DELAY_SHORT"
run_group M5 5 "$N_FIXED" "$DELAY_SHORT"
run_group M6 6 "$N_FIXED" "$DELAY_SHORT"
run_group M7 7 "$N_FIXED" "$DELAY_SHORT"
run_group M8 8 "$N_ATORED" "$DELAY_SHORT"
run_group M9 9 "$N_ATORED" "$DELAY_SHORT"
run_group M11_parreps 11 "$N_PARREPS" "$DELAY_SHORT"
printf '\nRaw HIP collection completed: %s\n' "$OUTDIR" | tee -a "$OUTFILE"
