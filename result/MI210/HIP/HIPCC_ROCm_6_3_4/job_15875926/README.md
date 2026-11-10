# ARCHER2 MI210 HIP formal results — job 15875926

Raw, unprofiled timings; no performance analysis. Slurm completed successfully in 00:55:40.

Actual device: AMD Instinct MI210 (gfx90a), 104 CUs, wavefront width 64. Requested geometry: 104 blocks × 64 threads; this does not guarantee one block per CU.

Compiler: HIPCC 6.3.42134, AMD Clang 18, ROCm 6.3.4; -O0, C++17, gfx90a target.

Coverage: 12 methods, 8 groups, 71 method/N configurations, 9,940 host rows. Delay powers of two from 1 to 8192, two sets, five runs, 20 inner repetitions, 10 warmups, one outer repetition. MAX_ITER=65536 for M1–4 and 6656 otherwise; MAX_ARRAY_SIZE=65536.

Independent verification checked the complete sampling grid, duplicates, missing rows, finite positive times, archived validator, source/generated/binary/submission/file hashes, and logged command success. Native preflight covered 37 geometries × 12 methods, 240 smoke rows and 20 exact-iteration regression rows (N=16382, I=6656).

Correctness instrumentation is separate from production timing; no per-sample formal oracle or production launch trace is claimed. Preflight before/after hardware JSON is archived; the formal collector asserts final hardware equality but does not retain a separate final hardware JSON.

Full sources, binaries, private logs and failure diagnostics remain in the local archive. Only raw CSVs, completion records, sanitized verification and checksums are published here. OpenMP diagnostic failure is not included as formal data.
