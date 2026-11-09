# A100 CUDA formal raw results — 9 November 2026

Complete verified uninstrumented host-timing sweep: **12 methods, 8 groups, 71 method/N configurations, 9,940 rows**. Runtime: 152.203 seconds. No performance analysis, fitting or plots were performed.

[Raw result groups](baseline/) contain the original byte-identical `raw_times.csv` files and original group configuration markers. [Verification summary](verification_summary.json) records coverage, source/binary/collector hashes and validation scope.

- GPU: A100-SXM4-40GB,108 SMs,warp32; requested108 blocks×32 threads.
- Compiler: NVCC12.9.86. Exact previously validated binary reused without recompilation.
- Delay powers of two from1 to8192;2sets×5runs,20inner repetitions,10warmup iterations,1outer repetition. CSV set/run/outer indices are zero-based.
- MAX_ITER=65536 for M1–4;3456 for all other groups. MAX_ARRAY_SIZE=65536. Explicit iteration count is preserved even when N=16382.
- M0/10 and M5/6/7 use N16382; M1–4 use N1,4,16,64,256,1024,4096,8192,16382,32768,65536; M8/9 use powers1–64; M11 uses powers1–128.

Both archived pure-data validation and independent grid/runtime checks passed. No missing/duplicate rows or nonpositive/nonfinite times. All28 required commands exited0. Before/after hardware,toolchain,GPU identity and binary integrity agree;122captured files and all source/generated/binary/provenance checksums verified. Full captured archive SHA256 is recorded in the summary.

These are host timings, not separate launch/kernel/wait measurements. No profiler/device-event pass or new numerical oracle was added to this formal sweep. Correctness relies on the exact binary previously checked with396native cases and260short rows. Requested geometry is not a production trace or evidence of one block per SM. Previous I6656 results are not mixed in.

This public package contains all formal numeric results and nonidentifying verification metadata. Full source,binaries,logs,validation provenance and archive remain local. Machine/account paths and GPU identifiers are omitted. `PUBLISHED_FILES_SHA256.json` covers all files except itself.
