# H200 CUDA formal raw results — 9 November 2026

Complete uninstrumented host-timing sweep: **12 methods,8 groups,71 method/N configurations,9940 rows**. No performance analysis,fitting or plots.

GPU runtime query: **NVIDIA H200**,132 SMs,warp32,compute capability9.0. Requested132 blocks×32 threads. CompilerNVCC12.9.86. Same benchmark source as the verified A100 sweep; compiled and correctness-checked on this target before formal collection. Formal collection reused that exact binary without recompilation.

Delay powers of two1–8192;2sets×5runs,inner20,warmup10,outer1. MAX_ITER65536 for M1–4 and4224 for other groups;MAX_ARRAY_SIZE65536. Original method/N coverage and row counts are recorded in verification_summary.json and baseline group markers. CSV indices are zero-based.

Archived validator and independent coverage/runtime checks passed: no missing or duplicate rows,no nonfinite or nonpositive times. All28 formal commands exited0. Actual GPU identity,hardware and toolchain agree before/after. Archive and all170 captured file hashes,source,generated code,binaries and provenance verified. Preflight passed396 native cases,240 short smoke rows and20 exact-iteration rows.

These are host timings; no new profiler or device-event capture. Correctness validation precedes formal sampling; there is no per-formal-sample numerical oracle. Requested geometry does not establish one block per SM or continuous exclusive occupancy.

Original numeric CSVs and nonidentifying verification metadata are published here. Full source,binaries,logs and provenance remain in the local archive. PUBLISHED_FILES_SHA256.json covers every published file except itself. Summary elapsed_seconds covers the preflight/formal pipeline,not Kubernetes scheduling or container bootstrap.
