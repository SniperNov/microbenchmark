# A100 HIP formal raw results — 9 November 2026

**API: HIP. Backend: NVIDIA.** These are independently compiled/run HIP results, kept separate from CUDA measurements. Complete uninstrumented host-timing sweep:12 methods,8 groups,71 method/N configurations,9940 rows. No performance analysis,fitting or plots.

Actual runtime GPU: **NVIDIA A100-SXM4-40GB**, 108 SMs,warp32. Requested 108 blocks×32 threads; this does not establish one block per SM or continuous exclusive occupancy.

Toolchain: official ROCm7.2.0 HIP/HIP NVIDIA headers and HIPCC sources; HIPCC invokes **NVCC12.9.86**, host GCC/G++13.3.0. HIP reported version: `7.2.53210-9999`. `HIP_PLATFORM=nvidia`, `HIP_COMPILER=nvcc`, `HIP_RUNTIME=cuda`. Production command compiles `src/backends/hip/backend.cpp` with `MICROBENCHMARK_HIP`, explicit NVIDIA HIP platform/include flags, `-O0 -std=c++17`, and target `sm_80`. Upstream commits,source/binary/collector and archive SHA256 values are in the verification summary. Container CUDA image naming does not make this a CUDA-API measurement.

Delay powers1–8192;2sets×5runs,inner20,warmup10,outer1. MAX_ITER65536 for M1–4 and3456 otherwise;MAX_ARRAY_SIZE65536. Exact method/N coverage and counts appear in group markers and the summary. CSV indices are zero-based.

Archived pure-data validation and independent coverage checks passed. No missing/duplicate rows or nonpositive/nonfinite times. All28 formal commands exited0. Actual GPU identity,hardware and NVCC version match before/after;1622 captured files and all source/generated/binary/provenance/installed HIP SDK hashes verified. Exact production binary passed396 native correctness cases and260 short rows before formal collection. Formal sampling does not add a per-sample numerical oracle.

These are host timings,not separately measured launch/kernel/wait intervals. No profiler/device-event data or fitting is included. Summary elapsed_seconds covers preflight/formal pipeline,excluding scheduling/container bootstrap.

This public package contains byte-identical formal CSVs and nonidentifying verification metadata. Full source,SDK,binaries,logs and provenance remain in local archives. `PUBLISHED_FILES_SHA256.json` covers all public files except itself.
