# A100 OpenMP formal results — 9 November 2026

Complete unprofiled host-timing sweep: **12 methods, 8 groups, 71 method/N configurations, 9940 rows**. No performance analysis, fitting or plots.

Actual GPU: **NVIDIA A100-SXM4-40GB**, 108 SMs, warp size 32. Runtime hardware queries and GPU identity agree before and after the run. Kubernetes Job and Pod both completed successfully. Requested geometry: 108 teams/gangs and 32 threads/workers. This is not a production launch trace or proof of one team per SM. OpenMP and OpenACC mappings must not be assumed physically identical.

Compiler: **NVIDIA NVC 26.3-0**, CUDA toolkit **12.9.86**. Production flags: `-O0 -mp=gpu -gpu=cc80,cuda12.9`. GCC 11.4.0 is only the queried container environment compiler; these benchmark results were compiled by NVC. Recorded CUDA runtime version is 12090 and driver API version is 12080. Native execution preflight passed. The pinned container image and source/generated/binary/collector hashes are recorded in the verification summary.

OpenMP uses the v3 launch package; OpenACC uses v2. OpenACC M9 uses explicit private-bucket reduction and is not identical to the OpenMP reduction implementation. Earlier failed attempts are excluded.

Delay lengths are powers of two from 1 through 8192; 2 sets × 5 runs, inner repetitions 20, warmup 10, outer repetitions 1. M1–4 use MAX_ITER=65536; other groups use MAX_ITER=3456. MAX_ARRAY_SIZE=65536. CSV set/run/outer indices are zero-based.

Archived collector validation and an independent reconstruction of every expected sampling key passed: no missing or duplicate rows, and no nonfinite or nonpositive times. All required commands exited zero. Archive and every captured file hash were checked, including source, generated files and the exact production binary. Preflight included 113 native correctness cases, 240 smoke rows and 20 exact-iteration rows. Instrumented correctness checks are separate from formal timing; there is no per-formal-sample numerical oracle.

These CSVs contain aggregate host timings, not separately measured launch, kernel or wait intervals. No profiler/event results are included.

Public CSVs are byte-identical to the downloaded formal results. Full sources, binaries, logs and provenance are retained in local archives. Public metadata omits private machine paths and GPU UUIDs. `PUBLISHED_FILES_SHA256.json` covers every public file except itself.
