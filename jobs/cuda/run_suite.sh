#!/usr/bin/env bash
# Original timing baseline + external CUDA trace + optional UM/event diagnostics.
set -euo pipefail
if [ -n "${SLURM_JOB_ID:-}" ]; then
    cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
else
    cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
fi
repo_dir="$PWD"
. ./jobs/common.sh
MACHINE="${CUDA_MACHINE:-GH200}"
case "$MACHINE" in
    A100) arch_default=sm_80 ;;
    H100|H200|GH200) arch_default=sm_90 ;;
    *) echo 'CUDA_MACHINE must be A100, H100, H200 or GH200' >&2; exit 1 ;;
esac
export CUDA_ARCH="${CUDA_ARCH:-$arch_default}"
TRACE="${CUDA_TRACE:-1}"
UM_TRACE="${CUDA_TRACE_UM:-0}"
EVENT_DIAGNOSTIC="${CUDA_DEVICE_TIMING:-0}"
for flag in "$TRACE" "$UM_TRACE" "$EVENT_DIAGNOSTIC"; do
    case "$flag" in 0|1) ;; *) echo 'CUDA_TRACE, CUDA_TRACE_UM and CUDA_DEVICE_TIMING must be 0 or 1' >&2; exit 1 ;; esac
done
command -v nvcc >/dev/null
command -v python3 >/dev/null
if [ -z "${NSYS_BIN:-}" ]; then
    NSYS_BIN=nsys
    for compiler in nvc nvcc; do
        compiler_path=$(command -v "$compiler" || true)
        if [ -n "$compiler_path" ] && [ -x "$(dirname -- "$compiler_path")/nsys" ]; then
            NSYS_BIN="$(dirname -- "$compiler_path")/nsys"; break
        fi
    done
fi
export NSYS_BIN
if [ "$TRACE" = 1 ] || [ "$UM_TRACE" = 1 ]; then command -v "$NSYS_BIN" >/dev/null; fi
set_common_benchmark_parameters
set_platform_launch_parameters "$MACHINE"
COMPILER_TAG=$(compiler_tag_nvcc)
mkdir -p "$repo_dir/result/$MACHINE/CUDA/$COMPILER_TAG"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR=$(mktemp -d "$repo_dir/result/$MACHINE/CUDA/$COMPILER_TAG/raw_${TIMESTAMP}.XXXXXX")
OUTFILE="$OUTDIR/run.log"
mkdir -p "$OUTDIR/tmp" "$OUTDIR/bin" "$OUTDIR/sources/src/core" "$OUTDIR/sources/src/backends/cuda" "$OUTDIR/sources/tests" "$OUTDIR/sources/jobs/cuda"
export TMPDIR="$OUTDIR/tmp"
export NSYS_TMPDIR="$TMPDIR"
archive_run_provenance "$0" "$OUTDIR" cuda "$TIMESTAMP"
cp jobs/common.sh "$OUTDIR/common.sh"
cp src/core/driver.c src/core/backend.h "$OUTDIR/sources/src/core/"
cp src/backends/cuda/backend.cu src/backends/cuda/openmp_profile.h "$OUTDIR/sources/src/backends/cuda/"
cp tests/check_cuda_device.cu tests/run_gh200_cuda_validation.sh tests/check_gh200_cuda_results.py tests/summarize_nsys_cuda.py "$OUTDIR/sources/tests/"
cp jobs/cuda/run_suite.sh jobs/cuda/trace_nsys.py "$OUTDIR/sources/jobs/cuda/"
cp Makefile Makefile.backend Makefile.defs.cuda.nvcc "$OUTDIR/sources/"
if [ -n "${CUDA_OPENMP_PROFILE:-}" ]; then
    cp "$CUDA_OPENMP_PROFILE" "$OUTDIR/openmp_profile.csv"
    export CUDA_OPENMP_PROFILE="$OUTDIR/openmp_profile.csv"
fi
{
    printf 'Output directory: %s\n' "$OUTDIR"
    log_run_configuration "$OUTFILE" "$MACHINE" CUDA "$COMPILER_TAG" "$PLATFORM_GROUPS" "$PLATFORM_WIDTH"
    nvcc --version
    printf 'CUDA_ARCH=%s TRACE=%s UM_TRACE=%s EVENT_DIAGNOSTIC=%s\n' "$CUDA_ARCH" "$TRACE" "$UM_TRACE" "$EVENT_DIAGNOSTIC"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    if [ "$TRACE" = 1 ] || [ "$UM_TRACE" = 1 ]; then "$NSYS_BIN" --version; fi
} 2>&1 | tee -a "$OUTFILE"
bash tests/run_gh200_cuda_validation.sh "$OUTDIR/validation" 2>&1 | tee -a "$OUTFILE"
python3 - "$OUTDIR/validation/device_check.log" "$MACHINE" <<'PY'
from pathlib import Path
import sys
names=[line for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith('CUDA device ')]
if not names or not any(sys.argv[2].upper() in line.upper() for line in names):
    raise SystemExit('GPU type does not match requested machine: '+str(names))
PY
make -B cuda-distribution EXTRA_CFLAGS="-arch=$CUDA_ARCH -DPRINT_DISTRIBUTION" 2>&1 | tee -a "$OUTFILE"
cp build/microbenchmark_cuda "$OUTDIR/bin/"
run_binary="$OUTDIR/bin/microbenchmark_cuda"
sha256sum "$run_binary" > "$OUTDIR/binary_sha256.txt"
run_group() {
    local tag="$1" methods="$2" sizes="$3" delay="$4"
    local base="$OUTDIR/baseline/$tag"
    local args=(API=cuda Method="$methods" N="$sizes" Delay="$delay" block_count="$PLATFORM_GROUPS" thread_count="$PLATFORM_WIDTH")
    mkdir -p "$base"
    printf '\n===== %s =====\n' "$tag" | tee -a "$OUTFILE"
    (cd "$base"; "$run_binary" "${args[@]}" DeviceTiming=0 > run.log 2>&1) || { cat "$base/run.log"; return 1; }
    cat "$base/run.log" | tee -a "$OUTFILE"
    if [ "$TRACE" = 1 ]; then
        python3 jobs/cuda/trace_nsys.py --binary "$run_binary" --baseline "$base" \
            --driver-source "$OUTDIR/sources/src/core/driver.c" --output "$OUTDIR/tracing/$tag" -- "${args[@]}" DeviceTiming=0 2>&1 | tee -a "$OUTFILE"
    fi
    if [ "$UM_TRACE" = 1 ]; then
        python3 jobs/cuda/trace_nsys.py --um-faults --binary "$run_binary" --baseline "$base" \
            --driver-source "$OUTDIR/sources/src/core/driver.c" --output "$OUTDIR/tracing_um/$tag" -- "${args[@]}" DeviceTiming=0 2>&1 | tee -a "$OUTFILE"
    fi
    if [ "$EVENT_DIAGNOSTIC" = 1 ]; then
        mkdir -p "$OUTDIR/event_diagnostic/$tag"
        (cd "$OUTDIR/event_diagnostic/$tag"; "$run_binary" "${args[@]}" DeviceTiming=1 > run.log 2>&1)
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
python3 tests/check_gh200_cuda_results.py "$OUTDIR" --device-timing "$EVENT_DIAGNOSTIC" \
    --blocks "$PLATFORM_GROUPS" --threads "$PLATFORM_WIDTH" --device-partition event_diagnostic 2>&1 | tee -a "$OUTFILE"
python3 - "$OUTDIR" "$TRACE" "$UM_TRACE" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);checks={}
groups=[x.name for x in (p/'baseline').iterdir() if x.is_dir()]
for kind,enabled in [('tracing',sys.argv[2]),('tracing_um',sys.argv[3])]:
    if enabled=='1':
        reports=[json.loads((p/kind/g/'TRACE_COMPLETE.json').read_text()) for g in groups]
        if len(reports)!=8 or sum(r['measured_samples'] for r in reports)!=9940:raise SystemExit('Incomplete '+kind)
        checks[kind]={'groups':8,'measured_samples':9940,'kernel_activity_rows':sum(r['kernel_activity_rows'] for r in reports)}
(p/'SUITE_COMPLETE.json').write_text(json.dumps({'baseline_complete':True,'trace_checks':checks,'profiling_outputs_are_separate':True},indent=2)+'\n')
PY
printf '\nComplete CUDA baseline/tracing suite: %s\n' "$OUTDIR" | tee -a "$OUTFILE"
