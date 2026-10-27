#!/usr/bin/env bash
# Native GPU checks; requires hipcc and an accessible HIP GPU.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
repo_dir="$PWD"
command -v hipcc >/dev/null
mkdir -p result/HIP/validation
out=$(mktemp -d "$repo_dir/result/HIP/validation/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
printf 'Validation output directory: %s\n' "$out"
flags=(-std=c++17 -O0 -DMICROBENCHMARK_HIP -I src/core)
if [ -n "${HIP_ARCH:-}" ]; then flags+=("--offload-arch=$HIP_ARCH"); fi
hipcc --version > "$out/compiler.txt" 2>&1
hipcc "${flags[@]}" tests/check_hip_device.cpp -o "$out/check_hip_device" > "$out/build.log" 2>&1 || {
    cat "$out/build.log"; exit 1;
}
env -u HIP_OPENMP_PROFILE "$out/check_hip_device" 2>&1 | tee "$out/device_check.log"
make -B hip-distribution HIP_ARCH="${HIP_ARCH:-}" 2>&1 | tee "$out/build_driver.log"
mkdir -p "$out/driver"
(
    cd "$out/driver"
    env -u HIP_OPENMP_PROFILE "$repo_dir/build/microbenchmark_hip" API=hip \
        Method=0,1,2,3,4,5,6,7,8,9,10,11 N=7 Delay=1,16 \
        block_count=3 thread_count=5 MAX_ITER=23 MAX_ARRAY_SIZE=64 DeviceTiming=1
) > "$out/driver.log" 2>&1 || { cat "$out/driver.log"; exit 1; }
python3 - "$out/driver" <<'PY'
from pathlib import Path
import csv, math, sys
p=Path(sys.argv[1])
for name,metric in [('raw_times.csv','exec_time_us'),('hip_device_times.csv','device_kernel_us')]:
    rows=list(csv.DictReader((p/name).open()))
    assert len(rows)==600,(name,len(rows))
    assert {int(r['method_id']) for r in rows}==set(range(12))
    assert all(math.isfinite(float(r[metric])) and float(r[metric])>=0 for r in rows)
assert not (p/'cuda_device_times.csv').exists()
print('PASS: all 12 HIP methods, 600 host rows and 600 independent device-event rows.')
PY
printf 'HIP validation completed: %s\n' "$out"
