#!/usr/bin/env bash
# Read existing SASS; collect diagnostic compiler commands and GPU state.
set -euo pipefail
repo_dir="${1:-/work/weiyu/microbenchmark}"
cd "$repo_dir"
repo_dir="$PWD"
saved="$repo_dir/result/GH200/device_opt_ab/20261014T202550Z.XUuTBe"
for tool in nvc python3 nvidia-smi sha256sum; do command -v "$tool" >/dev/null; done
for path in "$saved/bin/openmp_reference" "$saved/bin/cuda_ptxas_O3"; do
    test -x "$path" || { printf 'Missing original executable: %s\n' "$path" >&2; exit 1; }
done
mkdir -p "$repo_dir/result/GH200/instruction_state_check"
out=$(mktemp -d "$repo_dir/result/GH200/instruction_state_check/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf 'Diagnostic directory: %s\n' "$out"
mkdir -p "$out/loops" "$out/openmp_compiler"
cp "$0" "$out/collector.sh"
sha256sum "$saved/bin/openmp_reference" "$saved/bin/cuda_ptxas_O3" > "$out/benchmark_binary_sha256.txt"
python3 - "$saved" "$out" <<'PY'
from pathlib import Path
import collections,json,re,sys
saved=Path(sys.argv[1]);out=Path(sys.argv[2]);summary=[]
for name,symbol in [('openmp_reference','nvkernel_backend_run_method_F1L148_12'),('cuda_ptxas_O3','_Z16delay_one_kerneliPd')]:
    text=(saved/'device_code'/(name+'.sass.txt')).read_text()
    matches=[body for sym,body in re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S) if sym==symbol]
    if len(matches)!=1:
        raise SystemExit('Expected exactly one captured method0 symbol: '+symbol)
    body=matches[0];(out/'loops'/(name+'_method0.sass.txt')).write_text('Function : '+symbol+'\n'+body)
    ins=[]
    for line in body.splitlines():
        match=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
        if match:ins.append((int(match[1],16),match[2]))
    loops=[]
    for pc,op in ins:
        branch=re.search(r'\bBRA(?:\.[A-Z]+)?\s+0x([0-9a-f]+)',op)
        if branch and int(branch[1],16)<pc:
            begin=int(branch[1],16);loop=[(p,o) for p,o in ins if begin<=p<=pc]
            dadds=sum('DADD' in o for p,o in loop)
            if not dadds:continue
            record={'start':hex(begin),'back_branch':hex(pc),'static_instruction_lines':len(loop),'static_DADD':dadds,'static_I2F':sum('I2F.' in o for p,o in loop),'static_global_memory_ops':sum(bool(re.search(r'\b(?:LDG|STG)',o)) for p,o in loop)}
            loops.append(record)
            if dadds==16:
                (out/'loops'/(name+'_main_loop.txt')).write_text('\n'.join(f'{p:04x}: {o}' for p,o in loop)+'\n')
    summary.append({'binary':name,'symbol':symbol,'loops':loops})
(out/'loops/loop_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
PY

# Keep temporary compiler files and verbose phase invocations in a new directory.
# This diagnostic executable is NOT used for the timing runs below.
(
    cd "$out/openmp_compiler"
    nvc -v -mp=gpu -O0 -DPRINT_DISTRIBUTION -gpu=keep \
        -I "$saved/sources/src/core" \
        "$saved/sources/src/core/driver.c" \
        "$saved/sources/src/backends/openmp/backend.c" \
        -mp=gpu -lm -o "$out/openmp_compiler/openmp_diagnostic"
) > "$out/openmp_compiler/compile_verbose.log" 2>&1

# Verify the local driver's query fields before starting collection.
fields=timestamp,index,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,utilization.gpu
nvidia-smi -i 0 --query-gpu="$fields" --format=csv > "$out/gpu_query_preflight.csv"
nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,TEMPERATURE > "$out/gpu_detail_before.log" 2>&1
nvidia-smi -i 0 --query-gpu="$fields" --format=csv --loop-ms=500 \
    > "$out/gpu_state.csv" 2> "$out/gpu_state.stderr" &
monitor_pid=$!
cleanup()
{
    if [ -n "${monitor_pid:-}" ]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=''
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'utc_begin,utc_end,phase,method,binary\n' > "$out/run_windows.csv"
# ABBA order; method0 is measured separately from method6.
for method in 0 6; do
    phase=0
    for binary in openmp_reference cuda_ptxas_O3 cuda_ptxas_O3 openmp_reference; do
        phase=$((phase+1))
        run_dir="$out/diagnostic_runs/method$method/phase$phase-$binary"
        mkdir -p "$run_dir/baseline"
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
            > "$run_dir/processes_before.csv" 2> "$run_dir/processes_before.stderr" || true
        begin=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        printf 'Diagnostic run: method=%s phase=%s binary=%s\n' "$method" "$phase" "$binary"
        (
            cd "$run_dir/baseline"
            args=("Method=$method" N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536)
            if [ "$binary" = openmp_reference ]; then
                OMP_TARGET_OFFLOAD=MANDATORY "$saved/bin/$binary" API=openmp \
                    "${args[@]}" team_count=132 thread_count=32
            else
                env -u CUDA_OPENMP_PROFILE "$saved/bin/$binary" API=cuda \
                    "${args[@]}" block_count=132 thread_count=32 DeviceTiming=1
            fi
        ) > "$run_dir/run.log" 2>&1
        end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        printf '%s,%s,%s,%s,%s\n' "$begin" "$end" "$phase" "$method" "$binary" >> "$out/run_windows.csv"
        if [ "$binary" = cuda_ptxas_O3 ]; then
            mkdir -p "$run_dir/kernel_execution"
            mv "$run_dir/baseline/cuda_device_times.csv" "$run_dir/kernel_execution/"
        fi
        nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv \
            > "$run_dir/processes_after.csv" 2> "$run_dir/processes_after.stderr" || true
    done
done
cleanup
nvidia-smi -q -d PERFORMANCE,POWER,CLOCK,TEMPERATURE > "$out/gpu_detail_after.log" 2>&1
printf '\nCompleted diagnostic capture: %s\n' "$out"
printf 'Inspect loops/, openmp_compiler/compile_verbose.log, gpu_state.csv and run_windows.csv.\n'
printf 'GPU monitoring is coarse and can perturb timing; diagnostic_runs do not replace formal results.\n'
