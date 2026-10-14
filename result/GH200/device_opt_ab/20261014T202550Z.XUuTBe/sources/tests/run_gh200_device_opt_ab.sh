#!/usr/bin/env bash
# Matched method0/6 sensitivity test: host -O0, explicitly varied ptxas level.
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_dir"
for command_name in nvc nvcc python3 sha256sum; do
    command -v "$command_name" >/dev/null || { printf 'Missing command: %s\n' "$command_name" >&2; exit 1; }
done
mkdir -p "$repo_dir/result/GH200/device_opt_ab"
outdir=$(mktemp -d "$repo_dir/result/GH200/device_opt_ab/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
mkdir -p "$outdir/bin" "$outdir/build_logs" "$outdir/sources" "$outdir/device_code"
printf 'Output directory: %s\n' "$outdir"
{
    hostname
    nvc -V
    nvcc --version
    printf 'N=16382 MAX_ITER=16382 MAX_ARRAY_SIZE=65536 Delay=1,8192 Method=0,6\n'
    printf 'Requested groups=132 width=32; host -O0; ptxas O3 versus O0.\n'
    printf 'Driver warmup/sample/repetition constants are unchanged.\n'
    printf 'CUDA_OPENMP_PROFILE disabled for both CUDA arms to reproduce the previous standalone schedule.\n'
    printf 'OMP_DYNAMIC=%s OMP_NUM_THREADS=%s NVCOMPILER_OMP_CUDA_GRID=%s\n' \
        "${OMP_DYNAMIC:-not-set}" "${OMP_NUM_THREADS:-not-set}" "${NVCOMPILER_OMP_CUDA_GRID:-not-set}"
    if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
} > "$outdir/environment.log" 2>&1
sources=(src/core/driver.c src/core/backend.h src/backends/openmp/backend.c
         src/backends/cuda/backend.cu src/backends/cuda/openmp_profile.h
         tests/run_gh200_device_opt_ab.sh tests/summarize_gh200_device_opt_ab.py)
for path in "${sources[@]}"; do
    mkdir -p "$outdir/sources/$(dirname -- "$path")"
    cp "$path" "$outdir/sources/$path"
done
sha256sum "${sources[@]}" > "$outdir/source_sha256.txt"
reference_sha=$(sha256sum src/backends/openmp/backend.c | cut -d ' ' -f 1)
if [ "$reference_sha" != 03669299d2e150a3cbd7d81e207d2d78ee789e18b6abba6e71d1aa4316b65119 ]; then
    printf 'Original OpenMP source hash differs; stopping before compilation.\n' >&2
    exit 1
fi

printf 'Compiling the unchanged OpenMP reference...\n'
compile=(nvc -mp=gpu -O0 -DPRINT_DISTRIBUTION -I src/core
         src/core/driver.c src/backends/openmp/backend.c
         -mp=gpu -lm -o "$outdir/bin/openmp_reference")
printf '%q ' "${compile[@]}" > "$outdir/build_logs/openmp.log"
printf '\n' >> "$outdir/build_logs/openmp.log"
"${compile[@]}" 2>&1 | tee -a "$outdir/build_logs/openmp.log"
for level in 3 0; do
    name="cuda_ptxas_O$level"
    printf 'Compiling %s...\n' "$name"
    compile=(nvcc -O0 -arch=sm_90 -DMICROBENCHMARK_CUDA -DPRINT_DISTRIBUTION
             -Xptxas="-O$level,-v" -I src/core
             src/core/driver.c src/backends/cuda/backend.cu
             -lm -o "$outdir/bin/$name")
    printf '%q ' "${compile[@]}" > "$outdir/build_logs/$name.log"
    printf '\n' >> "$outdir/build_logs/$name.log"
    "${compile[@]}" 2>&1 | tee -a "$outdir/build_logs/$name.log"
done
sha256sum "$outdir/bin/"* > "$outdir/binary_sha256.txt"

# Read-only instruction snapshots; missing dumps do not invalidate raw collection.
cuobjdump_cmd=$(command -v cuobjdump || true)
if [ -z "$cuobjdump_cmd" ]; then
    candidate="$(dirname -- "$(command -v nvcc)")/cuobjdump"
    if [ -x "$candidate" ]; then cuobjdump_cmd="$candidate"; fi
fi
if [ -n "$cuobjdump_cmd" ]; then
    for name in openmp_reference cuda_ptxas_O3 cuda_ptxas_O0; do
        for format in ptx sass; do
            if ! "$cuobjdump_cmd" "--dump-$format" "$outdir/bin/$name" \
                > "$outdir/device_code/$name.$format.txt" \
                2> "$outdir/device_code/$name.$format.stderr"; then
                printf 'Could not dump %s %s; diagnostics retained.\n' "$name" "$format"
            fi
        done
    done
else
    printf 'cuobjdump unavailable; instruction snapshots skipped.\n' > "$outdir/device_code/README.txt"
fi

common_args=(Method=0,6 N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536)
run_arm()
{
    local trial="$1" name="$2"
    local arm_dir="$outdir/measurements/trial_$trial/$name"
    mkdir -p "$arm_dir/baseline"
    printf '\nRunning trial=%s arm=%s\n' "$trial" "$name"
    printf '%s trial_%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$trial" "$name" >> "$outdir/run_order.txt"
    if command -v nvidia-smi >/dev/null; then nvidia-smi > "$arm_dir/gpu_before.log" 2>&1 || true; fi
    (
        cd "$arm_dir/baseline"
        if [ "$name" = openmp_reference ]; then
            OMP_TARGET_OFFLOAD=MANDATORY "$outdir/bin/$name" API=openmp \
                "${common_args[@]}" team_count=132 thread_count=32
        else
            env -u CUDA_OPENMP_PROFILE "$outdir/bin/$name" API=cuda \
                "${common_args[@]}" block_count=132 thread_count=32 DeviceTiming=1
        fi
    ) 2>&1 | tee "$arm_dir/run.log"
    if [ "$name" != openmp_reference ]; then
        mkdir -p "$arm_dir/kernel_execution"
        mv "$arm_dir/baseline/cuda_device_times.csv" "$arm_dir/kernel_execution/"
    fi
    if command -v nvidia-smi >/dev/null; then nvidia-smi > "$arm_dir/gpu_after.log" 2>&1 || true; fi
}
# Rotation reduces order effects; snapshots do not establish constant GPU clocks.
for name in openmp_reference cuda_ptxas_O3 cuda_ptxas_O0; do run_arm 01 "$name"; done
for name in cuda_ptxas_O0 cuda_ptxas_O3 openmp_reference; do run_arm 02 "$name"; done
for name in cuda_ptxas_O3 openmp_reference cuda_ptxas_O0; do run_arm 03 "$name"; done

python3 tests/summarize_gh200_device_opt_ab.py "$outdir" | tee "$outdir/summary.log"
printf '\nCompleted. Results: %s\n' "$outdir"
printf 'This varies ptxas only; full device optimization and OpenMP execution equivalence are not established.\n'
