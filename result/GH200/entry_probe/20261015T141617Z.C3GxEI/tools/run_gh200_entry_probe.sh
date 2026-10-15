#!/usr/bin/env bash
# Isolated method0 entry-selection diagnostic. Never edits production sources.
set -euo pipefail
repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_dir"
saved="${1:-$repo_dir/result/GH200/device_opt_ab/20261014T202550Z.XUuTBe}"
saved="$(CDPATH= cd -- "$saved" && pwd)"
for tool in nvcc python3 sha256sum cuobjdump; do command -v "$tool" >/dev/null; done
test -x "$saved/bin/openmp_reference"
test -f "$saved/device_code/cuda_ptxas_O3.sass.txt"
reference_sha=03669299d2e150a3cbd7d81e207d2d78ee789e18b6abba6e71d1aa4316b65119
test "$(sha256sum "$saved/sources/src/backends/openmp/backend.c" | cut -d ' ' -f 1)" = "$reference_sha"
(
    cd "$saved"
    sha256sum --check --ignore-missing binary_sha256.txt
)
mkdir -p "$repo_dir/result/GH200/entry_probe"
out=$(mktemp -d "$repo_dir/result/GH200/entry_probe/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
mkdir -p "$out/bin" "$out/build_logs" "$out/device_code" "$out/tools"
printf 'Output directory: %s\n' "$out"
cp tests/gh200_entry_probe.py tests/run_gh200_entry_probe.sh "$out/tools/"
sha256sum src/backends/openmp/backend.c src/backends/cuda/backend.cu src/core/driver.c > "$out/production_sources_before.sha256"
trap 'sha256sum --check "$out/production_sources_before.sha256" > "$out/production_sources_check.log" || { cat "$out/production_sources_check.log"; exit 1; }' EXIT
{
    hostname
    nvcc --version
    printf 'Reference snapshot: %s\n' "$saved"
    printf 'Only Method=0 N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536\n'
    printf 'Requested B=132 T=32; actual CUDA method0 remains B=1 T=1.\n'
    printf 'No plotting; entry-gate diagnostic is separate from formal benchmark results.\n'
    printf 'OMP_DYNAMIC=%s OMP_NUM_THREADS=%s NVCOMPILER_OMP_CUDA_GRID=%s\n' "${OMP_DYNAMIC:-not-set}" "${OMP_NUM_THREADS:-not-set}" "${NVCOMPILER_OMP_CUDA_GRID:-not-set}"
    if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
} > "$out/environment.log" 2>&1
python3 "$out/tools/gh200_entry_probe.py" prepare "$out" --saved "$saved"

for arm in cuda_control cuda_entry_gate; do
    source_dir="$out/sources/$arm"
    compile=(nvcc -O0 -arch=sm_90 -DMICROBENCHMARK_CUDA -DPRINT_DISTRIBUTION -Xptxas=-O3,-v
             -I "$source_dir/src/core" "$source_dir/src/core/driver.c" "$source_dir/src/backends/cuda/backend.cu"
             -lm -o "$out/bin/$arm")
    printf '%q ' "${compile[@]}" > "$out/build_logs/$arm.log"
    printf '\n' >> "$out/build_logs/$arm.log"
    "${compile[@]}" >> "$out/build_logs/$arm.log" 2>&1 || { cat "$out/build_logs/$arm.log"; exit 1; }
    for format in sass ptx; do
        cuobjdump "--dump-$format" "$out/bin/$arm" > "$out/device_code/$arm.$format.txt" 2> "$out/device_code/$arm.$format.stderr"
    done
    check_flags=(-DMICROBENCHMARK_CUDA)
    if [ "$arm" = cuda_entry_gate ]; then check_flags+=(-DENTRY_GATE_CHECK); fi
    nvcc -O0 -arch=sm_90 -Xptxas=-O3 -I "$source_dir/src/core" \
        "${check_flags[@]}" "$source_dir/check_method0.cu" -o "$out/bin/check_$arm" \
        > "$out/build_logs/check_$arm.log" 2>&1 || { cat "$out/build_logs/check_$arm.log"; exit 1; }
    env -u CUDA_OPENMP_PROFILE "$out/bin/check_$arm" > "$out/check_$arm.log" 2>&1 || { cat "$out/check_$arm.log"; exit 1; }
done
sha256sum "$out/bin/"* "$saved/bin/openmp_reference" > "$out/binary_sha256.txt"
# Do not spend another timing run on an entry change that fails to alter the
# target loop, or that introduces memory operations inside it.
if python3 "$out/tools/gh200_entry_probe.py" gate "$out" --saved "$saved" > "$out/codegen_gate.log" 2>&1; then
    cat "$out/codegen_gate.log"
else
    gate_status=$?
    cat "$out/codegen_gate.log"
    if [ "$gate_status" -ne 10 ]; then exit "$gate_status"; fi
    printf '\nCode generation did not isolate the intended change; no timing experiment was run.\n'
    printf 'Keep and push this diagnostic directory: %s\n' "$out"
    exit 0
fi

python3 - "$out/run_windows.csv" <<'PY'
import csv,sys
with open(sys.argv[1],'w',newline='') as f:
    csv.writer(f).writerow(['utc_begin','utc_end','trial','arm'])
PY
run_arm() {
    local trial="$1" arm="$2" binary api
    local run_dir="$out/measurements/trial_$trial/$arm"
    mkdir -p "$run_dir/baseline"
    binary="$out/bin/$arm"; api=cuda
    if [ "$arm" = openmp_reference ]; then binary="$saved/bin/openmp_reference"; api=openmp; fi
    printf 'Running trial=%s arm=%s\n' "$trial" "$arm"
    local begin end
    begin=$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds"))')
    if command -v nvidia-smi >/dev/null; then nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,TEMPERATURE > "$run_dir/gpu_before.log" 2>&1 || true; fi
    (
        cd "$run_dir/baseline"
        args=(Method=0 N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536)
        if [ "$api" = openmp ]; then
            OMP_TARGET_OFFLOAD=MANDATORY "$binary" API=openmp "${args[@]}" team_count=132 thread_count=32
        else
            env -u CUDA_OPENMP_PROFILE "$binary" API=cuda "${args[@]}" block_count=132 thread_count=32 DeviceTiming=1
        fi
    ) > "$run_dir/run.log" 2>&1
    if [ "$api" = cuda ]; then
        mkdir -p "$run_dir/kernel_execution"
        mv "$run_dir/baseline/cuda_device_times.csv" "$run_dir/kernel_execution/"
    fi
    if command -v nvidia-smi >/dev/null; then nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,TEMPERATURE > "$run_dir/gpu_after.log" 2>&1 || true; fi
    end=$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat(timespec="microseconds"))')
    printf '%s,%s,%s,%s\n' "$begin" "$end" "$trial" "$arm" >> "$out/run_windows.csv"
}
for arm in openmp_reference cuda_control cuda_entry_gate; do run_arm 01 "$arm"; done
for arm in cuda_entry_gate cuda_control openmp_reference; do run_arm 02 "$arm"; done
for arm in cuda_control openmp_reference cuda_entry_gate; do run_arm 03 "$arm"; done
python3 "$out/tools/gh200_entry_probe.py" summarize "$out" | tee "$out/summary.log"
printf '\nCompleted: %s\n' "$out"
