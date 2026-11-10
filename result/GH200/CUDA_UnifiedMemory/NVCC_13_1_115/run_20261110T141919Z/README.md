# GH200 CUDA Unified Memory — 12-method suite

Verified formal collection, 10 November 2026. **Three modes, each 12 methods / 8 groups / 71 method–N configurations / 9,940 host samples; 29,820 samples total.** These results are separate from the original explicit-memory CUDA datasets. No performance analysis is supplied.

## Modes

- `managed_demand`: managed allocation with demand access; no prefetch requests.
- `managed_prefetch`: managed allocation with directional prefetch requests.
- `managed_advice_prefetch`: preferred-location advice plus directional prefetch requests.

Prefetch can cause real page migration; advice is a hint. No page residency or migration-volume measurement is claimed.

## M1–M4 timing contract: not the explicit-memory experiment

Managed allocation, initialization, free and one-time advice are outside the measured interval, including M1–M4. The original explicit-memory M1–M4 experiment includes its method-specific allocation/copy/free operations. Therefore the two variants do not measure identical memory-management work.

For prefetch modes, M1 requests GPU input prefetch before computation and CPU output prefetch afterwards; M2 requests GPU input only; M3 requests CPU output only; M4 makes no placement request. These per-iteration requests and waits are part of the measured operation. Demand mode makes no prefetch requests. The original CPU `a[0]` increment remains inside timing and now accesses the shared allocation; it may induce migration or remote access. M2/M4 GPU writes remain CPU-visible: explicit copyin/create visibility semantics are not reproduced.

M0 and M5–M11 prepare resident buffers outside timing in prefetch modes. CSV durations are aggregate host times, not isolated kernel, launch or wait times.

## Implementation and reproducibility

`code/` contains the driver, managed backend, unchanged original CUDA kernel backend, native correctness tests, host-only tests and collector. Kernels and reduction algorithms are retained. Managed input/reduction buffers use `cudaMallocManaged`; the managed execution path does not call `cudaMemcpy`. Device-event profiling is not enabled.

The published collector differs from the executed collector only in two portable defaults: `NVCC` falls back to `nvcc` on PATH; `GH200_GPU_LOCK` falls back to a package-local lock. Set a shared `GH200_GPU_LOCK` for concurrent suites and set `NVCC` to the intended compiler. README status/instructions were updated after completion. Benchmark and native-test source bytes are unchanged. `code/EXECUTED_SOURCE_SHA256.json` records the executed package; `code/SOURCE_SHA256.json` records the portable published package. The latter is used by its collector. Full original binaries and private logs remain in the local archive.

Actual GPU: NVIDIA GH200 480GB; 132 SMs, warp size 32. Compiler: NVCC 13.1.115, `-O0 -std=c++17 -arch=sm_90`. Managed/concurrent access supported. Requested grouping 132 blocks × 32 threads does not imply one block per SM; method-specific scalar/group semantics remain.

Delay: powers of two 1–8192. Two sets × five runs, inner repetitions 20, warmup 10, outer repetitions 1. M1–4 MAX_ITER=65536; other groups MAX_ITER=4224; MAX_ARRAY_SIZE=65536. Each mode passed 33 native geometry cases across all 12 methods, 240 production smoke samples and 20 exact-iteration regression samples before formal collection.

Independent verification reconstructed all expected sample keys, rejected duplicates/missing/non-finite/non-positive values, checked runtime parameters and the archived pure-data checker, confirmed before/after GPU identity and source/binary hashes, and verified the downloaded archive and every file. This preflight does not assert a numerical oracle on every formal sample. The initial runtime manifest remains `validating`; final completion, complete raw records and hashes were independently checked. Raw CSVs here are byte-identical to the verified archive.

`PUBLISHED_FILES_SHA256.json` covers every published file except itself.
