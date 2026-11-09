# MI300X HIP formal results

Completed on 31 October 2026, job 12252459, elapsed **43m41s**. This public package preserves the complete original numeric formal results and a new verification summary. Comparative analysis is deferred until the other platform datasets are ready.

## Coverage

**12 methods, 8 groups, 71 method/N configurations and 9,940 host timing rows**. Delay lengths are powers of two from 1 to 8192. Sampling uses 2 sets × 5 runs, 20 inner repetitions, 10 warmup iterations and 1 outer repetition. CSV set/run indices start at zero; `outerreps=0` identifies the single outer repetition.

| Group | N | MAX_ITER | Rows | Raw data |
|---|---|---:|---:|---|
| M0, M10 | 16382 | 19456 | 280 | [CSV](baseline/M0_M10/raw_times.csv) |
| M1, M2, M3, M4 | 1, 4, 16, 64, 256, 1024, 4096, 8192, 16382, 32768, 65536 | 65536 | 6160 | [CSV](baseline/M1to4/raw_times.csv) |
| M5 | 16382 | 19456 | 140 | [CSV](baseline/M5/raw_times.csv) |
| M6 | 16382 | 19456 | 140 | [CSV](baseline/M6/raw_times.csv) |
| M7 | 16382 | 19456 | 140 | [CSV](baseline/M7/raw_times.csv) |
| M8 | 1, 2, 4, 8, 16, 32, 64 | 19456 | 980 | [CSV](baseline/M8/raw_times.csv) |
| M9 | 1, 2, 4, 8, 16, 32, 64 | 19456 | 980 | [CSV](baseline/M9/raw_times.csv) |
| M11 | 1, 2, 4, 8, 16, 32, 64, 128 | 19456 | 1120 | [CSV](baseline/M11_parreps/raw_times.csv) |

All groups request 304 blocks × 64 threads with MAX_ARRAY_SIZE=65536. The archived runtime logs confirm these requests and the listed MAX_ITER/N settings. The GPU reports 304 CUs; the hardware inventory records wavefront width 64 and architecture gfx942. CSV geometry records requested launch configuration, not proof of identical participant/owner behavior across HIP and OpenMP or one block per CU.

## Verification and scope

The archived build uses HIP 7.2.53211-97f5574fe2, AMD Clang 22.0.0git / ROCm compiler build roc-7.2.4, `-O0 -std=c++17 -x hip --offload-arch=gfx942`. The C driver and HIP backend are compiled in HIP C++ mode.

Independent local rechecking found the complete expected grid, no missing/duplicate rows, and only finite positive times. All eight CSV files are byte-identical to the archived originals. [Verification summary](verification_summary.json) records tested source and binary hashes; [published checksums](PUBLISHED_FILES_SHA256.json) cover this package except the checksum manifest itself.

These are **uninstrumented host timing results**, with DeviceTiming=0. They do not separately measure launch, kernel or wait intervals. The archive separately records native correctness checks for all 12 HIP methods; this formal sweep does not add a numerical oracle for every timing sample. Validation smoke/device-event rows are excluded. The later HIP rocprofv3 job 12325017 is not included, and no OpenMP measurements are mixed into this HIP package.

## Publication scope

The complete original archive, source snapshots, binaries, logs, validation results and provenance remain local with verified hashes. This public package contains only the formal CSVs, sanitized group configurations and a newly prepared verification summary; it is not the complete archival payload. Original archive/source/binary hashes identify the retained evidence without publishing machine-identifying metadata. No new fitting, comparative analysis or plots are included.
