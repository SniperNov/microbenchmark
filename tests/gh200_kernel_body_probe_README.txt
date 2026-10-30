GH200/Hopper internal GPU body-clock diagnostic

Purpose
Read %globaltimer inside copies of the original method0/method6 kernels, then
compare their GPU body intervals with Nsight GPU activity durations. Do not
subtract a CPU timestamp from an uncorrelated GPU timestamp.

This is a separate diagnostic, not a replacement benchmark suite. It never
edits production CUDA/OpenMP/OpenACC/HIP source or the archived reference.
It retains the original driver, 20 inner repetitions, warmups, shuffle, launch
dimensions, workload, existing block barriers and per-launch synchronization.

Three arms
  control:     original kernel machine instructions must match the archived bin.
  stores_only: same per-thread counter/record writes, without timer reads.
  globaltimer: reads at kernel body entry/exit, then writes records afterwards.

There are no added atomics or block/grid barriers. Records are per thread, per
launch. Every thread's recorded call count and the original array values are
checked. The runner first compiles all three arms, checks native SASS, and runs
a small native data-flow/counter smoke. Only then does it collect full timings.
No GPU performance-counter privileges are needed.

Run from the repository on GH200, with the original NVHPC/CUDA environment:

  export NSYS_BIN=/work/weiyu/software/nvidia/hpc_sdk/Linux_aarch64/26.3/compilers/bin/nsys
  python3 -u tests/gh200_kernel_body_probe.py --reference \
    /work/weiyu/microbenchmark-cuda-gh200-20261028_155520/result/GH200/CUDA/NVCC_13_1_115/raw_20261028_155520.aSIlrB

Use --smoke to stop after compilation, SASS checks and the small native checks.
The full run has Delay=1..8192, N=MAX_ITER=16382, MAX_ARRAY_SIZE=65536,
requested B=132/T=32; actual method0 is B=1/T=1 and method6 B=132/T=32.
Each arm runs separately without nsys and with ordinary CUDA tracing. Device
event timing and heavy Unified Memory fault collection are disabled here.

Result directory: result/GH200/kernel_body_clock/<fresh timestamp>/
  body_timing_means.csv        duration comparisons for each method and Delay
  captures/<arm>/              independent unprofiled and profiled executions
  body_clock_launches.csv      all 20 launch records, including warmups
  body_clock_records.bin       raw little-endian uint64 begin/end pairs
  codegen_checks.json          original control match and native timer reads
  *.sass.txt, *_build.log      instruction/register/compilation evidence
  BODY_PROBE_COMPLETE.json     only written after full validation

Interpretation
  Per-thread span: that thread's exit stamp minus its entry stamp.
  Body envelope: latest thread exit minus earliest thread entry, on ONE GPU.
  Neither equals the complete kernel activity automatically. The envelope
  excludes prologue before the first stamp, record stores after exit stamps,
  and kernel epilogue. No extra fence is added to drain outstanding stores.
  Instrumentation can affect register allocation, memory traffic and scheduling.
  Control/stores-only comparisons are diagnostic, not an exact timer-overhead
  subtraction. The summary never labels a difference as instruction latency.
  Do not call Nsight activity duration a zero-perturbation ground truth either.

Hopper limitation
NVIDIA describes a finer globaltimer update on GH100+ than on pre-GH100 GPUs
outside profiling. This initial experiment requires device major >=9 and is
intended for GH200/H100/H200; A100 requires a separate timer-resolution check.
The register's behavior is target-specific, not a universal portability promise.

Local validation
  CXX=g++-14 python3 tests/test_gh200_kernel_body_probe.py
This exercises the actual generated host control flow, original driver, original
M0/M6 computations and all per-thread records with a fake runtime/clock. It does
not validate nvcc/PTX, GPU concurrency, timer placement or native timing.

Sources
https://forums.developer.nvidia.com/t/question-about-nsight-tools-timestamping-mechanism-and-clock-sources/323887
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi
