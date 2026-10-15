#!/usr/bin/env bash
# Shared method0 code-generation runner. Never edits production sources.
set -euo pipefail
probe_kind="${GH200_PROBE_KIND:-entry}"
case "$probe_kind" in
    entry) candidate=cuda_entry_gate; checker_macro=ENTRY_GATE_CHECK ;;
    counter) candidate=cuda_counter_start; checker_macro=COUNTER_START_CHECK ;;
    *) printf 'Invalid GH200_PROBE_KIND: %s\n' "$probe_kind" >&2; exit 1 ;;
esac
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
mkdir -p "$repo_dir/result/GH200/${probe_kind}_probe"
out=$(mktemp -d "$repo_dir/result/GH200/${probe_kind}_probe/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
mkdir -p "$out/bin" "$out/build_logs" "$out/device_code" "$out/tools"
printf 'Output directory: %s\n' "$out"
cp tests/gh200_entry_probe.py tests/run_gh200_entry_probe.sh "$out/tools/"
if [ "$probe_kind" = counter ]; then
    cp tests/run_gh200_counter_probe.sh tests/gh200_counter_probe_README.txt "$out/tools/"
fi
sha256sum src/backends/openmp/backend.c src/backends/cuda/backend.cu src/core/driver.c > "$out/production_sources_before.sha256"
trap 'sha256sum --check "$out/production_sources_before.sha256" > "$out/production_sources_check.log" || { cat "$out/production_sources_check.log"; exit 1; }' EXIT
{
    hostname
    nvcc --version
    printf 'Reference snapshot: %s\n' "$saved"
    printf 'Diagnostic kind: %s; candidate: %s\n' "$probe_kind" "$candidate"
    printf 'Only Method=0 N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536\n'
    printf 'Requested B=132 T=32; actual CUDA method0 remains B=1 T=1.\n'
    printf 'No plotting; diagnostic results are separate from formal benchmark results.\n'
    printf 'OMP_DYNAMIC=%s OMP_NUM_THREADS=%s NVCOMPILER_OMP_CUDA_GRID=%s\n' "${OMP_DYNAMIC:-not-set}" "${OMP_NUM_THREADS:-not-set}" "${NVCOMPILER_OMP_CUDA_GRID:-not-set}"
    if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
} > "$out/environment.log" 2>&1
python3 "$out/tools/gh200_entry_probe.py" prepare "$out" --saved "$saved" --kind "$probe_kind"

for arm in cuda_control "$candidate"; do
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
    if [ "$arm" = "$candidate" ]; then check_flags+=("-D$checker_macro"); fi
    nvcc -O0 -arch=sm_90 -Xptxas=-O3 -I "$source_dir/src/core" \
        "${check_flags[@]}" "$source_dir/check_method0.cu" -o "$out/bin/check_$arm" \
        > "$out/build_logs/check_$arm.log" 2>&1 || { cat "$out/build_logs/check_$arm.log"; exit 1; }
    env -u CUDA_OPENMP_PROFILE "$out/bin/check_$arm" > "$out/check_$arm.log" 2>&1 || { cat "$out/check_$arm.log"; exit 1; }
done
sha256sum "$out/bin/"* "$saved/bin/openmp_reference" > "$out/binary_sha256.txt"
# Do not spend another timing run on a source change that fails to alter the
# target loop, or that introduces memory operations inside it.
if python3 "$out/tools/gh200_entry_probe.py" gate "$out" --saved "$saved" --kind "$probe_kind" > "$out/codegen_gate.log" 2>&1; then
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
for arm in openmp_reference cuda_control "$candidate"; do run_arm 01 "$arm"; done
for arm in "$candidate" cuda_control openmp_reference; do run_arm 02 "$arm"; done
for arm in cuda_control openmp_reference "$candidate"; do run_arm 03 "$arm"; done
python3 "$out/tools/gh200_entry_probe.py" summarize "$out" --kind "$probe_kind" | tee "$out/summary.log"
printf '\nCompleted: %s\n' "$out"
