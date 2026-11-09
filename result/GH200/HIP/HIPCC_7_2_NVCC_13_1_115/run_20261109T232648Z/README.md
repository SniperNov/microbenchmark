# GH200 HIP formal host timings — 9 November 2026

Complete unprofiled sweep: **12 methods, 8 groups, 71 method/N configurations, 9940 host rows**. No performance analysis or fitting. This is the original method suite with explicit MAX_ITER semantics, not the separate Unified Memory placement experiment.

GPU runtime queries identify **NVIDIA GH200 480GB**, 132 SMs, warp size 32, compute capability 9.0. GPU identity and hardware agree before/after. Requested grid is 132 teams/gangs/blocks × 32 threads/workers; scalar and team-only methods retain their respective semantics. Instrumented ownership validation is separate from production timings. This is not a production launch trace or proof of one team per SM.

Toolchain: HIPCC 7.2 NVIDIA backend, rebuilt for ARM; NVCC 13.1.115; GCC/G++ 11.4.0 host compiler. Production uses `-O0` and explicit cc90/sm_90 targeting. Runtime reports CUDA 13010 and driver API 13000. Native preflight passed before formal collection. The three APIs ran serially under a shared GPU lock.

Delay lengths: powers of two 1–8192; 2 sets × 5 runs; inner repetitions 20, warmup 10, outer repetitions 1. M1–4 MAX_ITER=65536; other groups MAX_ITER=4224. MAX_ARRAY_SIZE=65536. CSV repetition indices are zero-based. OpenACC M9 uses explicit private-bucket reduction and must not be described as identical to OpenMP's native array reduction.

Validation checked the archived collector and independently reconstructed all expected sample keys, runtime parameters, finite positive durations, native preflight records, source/generated/binary/provenance hashes, final wrapper hashes and the downloaded archive. No missing or duplicate samples. The exact production binary passed 240 smoke rows and 20 exact-iteration regression rows. Native instrumented cases: OpenMP 13 generic + 100 M9; OpenACC 112; HIP 396. These checks do not add a per-formal-sample numerical oracle.

For OpenMP/OpenACC, the collector's intermediate file manifest recorded LAUNCH_STATE while starting; the wrapper later updated it on completion. The earlier state hash was reconstructed and verified; both the final wrapper manifest and capture manifest match final bytes. No benchmark data discrepancy was found.

CSV files are byte-identical to the downloaded results. Full sources, binaries and logs remain in local archives. Public metadata excludes private filesystem paths and GPU UUIDs. PUBLISHED_FILES_SHA256.json covers every public file except itself. These aggregate host times do not separately measure launch, kernel or wait intervals.
