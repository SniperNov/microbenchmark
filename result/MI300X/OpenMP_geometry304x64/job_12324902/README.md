# MI300X OpenMP 304×64 formal results

Completed and independently verified on 9 November 2026, job 12324902, elapsed **42m32s**. This public package contains the complete original numeric performance results and a verification summary. Analysis is deferred until the other platforms are complete.

## Coverage

**12 methods, 8 groups, 71 method/N configurations and 9,940 host timing rows**. Delay lengths are powers of two from 1 to 8192. Sampling uses 2 sets × 5 runs, 20 inner repetitions, 10 warmup iterations and 1 outer repetition.

| Group | N | MAX_ITER | Rows | Raw data |
|---|---|---:|---:|---|
| M0, M10 | 16382 | 19456 | 280 | [CSV](baseline/M0_M10/raw_times.csv) |
| M1–4 | 1, 4, 16, 64, 256, 1024, 4096, 8192, 16382, 32768, 65536 | 65536 | 6160 | [CSV](baseline/M1to4/raw_times.csv) |
| M5 | 16382 | 19456 | 140 | [CSV](baseline/M5/raw_times.csv) |
| M6 | 16382 | 19456 | 140 | [CSV](baseline/M6/raw_times.csv) |
| M7 | 16382 | 19456 | 140 | [CSV](baseline/M7/raw_times.csv) |
| M8 | 1, 2, 4, 8, 16, 32, 64 | 19456 | 980 | [CSV](baseline/M8/raw_times.csv) |
| M9 | 1, 2, 4, 8, 16, 32, 64 | 19456 | 980 | [CSV](baseline/M9/raw_times.csv) |
| M11 | 1, 2, 4, 8, 16, 32, 64, 128 | 19456 | 1120 | [CSV](baseline/M11_parreps/raw_times.csv) |

Every group requests 304 teams × 64 threads and MAX_ARRAY_SIZE=65536. CSV geometry columns record those requests. Method-specific logical participation was checked in prior exact-binary validation: M0–4/7 use 1×1, M5 uses 304×1, and M6/8/9/10/11 use 304×64.

## Verification and interpretation

The GPU reports 304 CUs and wavefront width 64. AMD Clang 22.0.0git / ROCm 7.2 was used. The production binary was reused without recompilation. Before/after hardware, toolchain and binary integrity checks agree; all required commands succeeded. Both the archived collector's pure-data validation and independent complete-grid validation passed, with no missing/duplicate rows or nonpositive/nonfinite durations.

[Verification summary](verification_summary.json) includes the exact tested binary/source/collector hashes. [Published file checksums](PUBLISHED_FILES_SHA256.json) cover every file in this package except the checksum manifest itself. The eight raw CSV files are byte-identical to the validated local archive.

These are uninstrumented **host timing** results, not separate launch/kernel/wait measurements. No new profiler trace or numerical oracle was added to this formal sweep. Correctness and geometry refer to the earlier exact-binary validation. That prior trace observed M6/8/9 at 304×64 hardware launch dimensions and M10/11 at 304×128, including a runtime management wave with 64 logical computation threads. No one-team-per-CU placement is claimed.

The separately submitted HIP rocprofv3 collection is not included. Old performance data has not been mixed into this dataset.

## Publication scope

The complete original archive, source snapshots, binaries, runtime logs and provenance remain preserved locally with their verified hashes. This public package omits machine-identifying metadata, including account/project paths, hostnames and GPU UUIDs/serials. It is not the complete archival payload; `verification_summary.json` is a newly prepared public summary. Its original archive SHA256 is recorded for provenance without publishing that archive.
