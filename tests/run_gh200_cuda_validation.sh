#!/usr/bin/env bash
# Native CUDA correctness only: no OpenMP tracing or performance counters.
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
command -v nvcc >/dev/null
mkdir -p result/GH200/cuda_validation
out="${1:-$(mktemp -d "$PWD/result/GH200/cuda_validation/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")}"
mkdir -p "$out"
out="$(CDPATH= cd -- "$out" && pwd)"
printf 'CUDA validation output directory: %s\n' "$out"
nvcc --version > "$out/compiler.txt" 2>&1
nvcc -arch="${CUDA_ARCH:-sm_90}" -O0 -DMICROBENCHMARK_CUDA -I src/core \
    tests/check_cuda_device.cu -o "$out/check_cuda_device" > "$out/build.log" 2>&1 || {
    cat "$out/build.log"; exit 1;
}
env -u CUDA_OPENMP_PROFILE "$out/check_cuda_device" 2>&1 | tee "$out/device_check.log"
printf 'PASS: CUDA native correctness; full benchmark sweep may proceed.\n'
