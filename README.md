# 🚀 OpenMP Offloading Microbenchmark

<p align="center">
  <img src="https://img.shields.io/badge/OpenMP-Target_Offloading-blue?logo=openmp&style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-AMD_MI250X-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Compiler-GCC%20%7C%20Clang%20%7C%20Cray%20%7C%20NVC-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active_Development-purple?style=for-the-badge"/>
</p>

<p align="center">
A fully configurable, architecture-aware, and compiler-portable microbenchmark suite for studying  
<b>OpenMP target offloading, kernel-launch overhead, and data-mapping behavior</b>.
</p>

## 📚 Table of Contents
- Overview
- Repository Contents
- Parameters & Execution Model
- Parameter Relationships
- Compilation
- Running the Benchmark
- Benchmark Output Formats
- Example Run Script
- Default Configuration
- Recent Modifications
- Contact

## ✨ Overview

This microbenchmark suite evaluates OpenMP target offloading performance across multiple GPU architectures and compiler toolchains. It focuses on:

- Data mapping overhead (map(to/from/tofrom/alloc))
- Kernel launch overhead (teams, parallel, distribute, atomic, reduction)
- Execution scaling under varying delaylength
- Impact of N, thread_count, team_count, MAX_ITER
- Compiler/runtime differences across GCC, Clang, Cray CCE, NVIDIA NVC

## 📂 Repository Contents

| File | Description |
|------|-------------|
| microbenchmark.c | Main benchmark implementation |
| common.c | Timing, delay kernel |
| common.h | Declarations and configuration macros |
| Makefile | Multi-compiler build system |
| Makefile.defs.gcc | GCC compiler flags |
| Makefile.defs.clang | amdclang flags |
| Makefile.defs.cray | Cray CCE compiler flags |
| Makefile.defs.nvc | NVIDIA HPC SDK flags |
| run.sh | Example batch-wrapper used by SLURM jobs |

## 🧵 Parameters & Execution Model

Supported runtime parameters:

```
| Method | OMP offloading Pragma |
| N | array size (mapping) OR tmp[] size (kernel methods) |
| Delay | size of delay kernel workload; sampled log-scale |
| thread_count | threads per team |
| team_count | number of GPU teams (≈ Compute Units) |
| MAX_ITER | loop-iteration space inside GPU kernel |
| MAX_ARRAY_SIZE | memory allocation size for a[] |
```

## 🧩 Parameter Relationships

✔ Most important constraint

```
thread_count × team_count ≤ MAX_ITER ≤ MAX_ARRAY_SIZE
```

✔ Interpretation of N depends on method

| Methods | Meaning of N | Actual allocation |
|--------|--------------|------------------|
| Case 1–4 (mapping) | a size = N | double a[MAX_ARRAY_SIZE] |
| Case 5–11 (kernel-launch) | tmp[N] | a always size = MAX_ARRAY_SIZE |

✔ Recommended usage

| Goal | N |
|------|---|
| Mapping cost measuring method | large N (e.g., 16382 or 65528) |
| Kernel launch cost measuring method | very small N (2, 4, 8) |

## ⏱ Delaylength & Offloading Example

Delaylength governs the artificial per-iteration workload in the device kernel:

```c
#pragma omp target map(tofrom: a[0:N])
for (int i = 0; i < g_max_iter; ++i) {
    delay_kernel(delay, &a[i]);
}
```

Default delay settings:

```c
#define MIN_DELAYLENGTH 512
#define MAX_DELAYLENGTH 262144
#define NUM_SAMPLES 20   // must be >= 5 for regression
```
Delaylength is sampled logarithmically, capturing kernel launch → computation transitions.
Too small → noise
Too large → dominated by compute instead of launch overhead

## 🛠 Compilation

The benchmark supports multiple compilers.
Compile benchmark:

```
make
```

Produces binary:
```
./microbenchmark
```

Compile distribution-enabled version:

```
make distribution
```
Produces:
```
./microbenchmark_distribution
overhead_distribution.txt
```

## 🧪 Running the Benchmark

### 1. Adjust parameters inside run.sh:

```
./microbenchmark_distribution Method=6,8,9 N=2 Delay=262144 thread_count=16 team_count=64
```

### 2. Run via SLURM:

Inside your SLURM script:
```
srun ./run.sh
```

## 📄 Benchmark Output Formats

### 🟦 1. Output from make 

```
Running microbenchmark...
========== Runtime Configuration ==========
Delay range   : [512, 262144]
Array size(s) : 2
Threads/Teams : 16 / 64
MAX_ITER      : 6656
MAX_ARRAY_SIZE: 65536
NUM_SAMPLES   : 20
==========================================
There are  1 available devices
Host device is 1

========== Benchmark Execution ==========
Method/N                           2
teams distribute parallel for      318.800822±447.783041
···
```
This table contained:
| Field | Meaning |
|------|---------|
| Method/N | Each offloading method (cases 1–11) |
| X ± Y | Estimated intercept ± standard deviation |

### 🟩 2. Output from make microbenchmark_distribution

One More File produced:
```
overhead_distribution.txt
```

Example content:
```
[Method=1 map(tofrom: a) N=16382]
Set=0 Run=0  Lmin=8192  Intercept=619.150872μs  Slope=0.936642  R2=0.99998  BIC=72.969
Set=0 Run=1  Lmin=8192  Intercept=646.507138μs  Slope=0.938634  R2=0.99994  BIC=80.739
...
Average Intercept=443.17 ± 244.72 μs
```

Explanation:

| Field | Meaning |
|------|---------|
| Set | Independent runs |
| Run | Inner independent repeats for statistical reliability |
| Lmin | Breakpoint selected by BIC segmented regression |
| Intercept | Estimated Kernel-launch overhead |
| Slope | Per-unit Delaylength cost |
| R2 | Regression fit quality|
| BIC | Model selection score |

We defaultly get 4 (2 Sets x 2 Runs) runs per method, and even 80 (20 innerreps x 40 outerreps) runs, so totally 320 runs improving stability.

## 📥 Example run.sh (Should be wrapped in SLURM)

```bash
#!/bin/bash
OUTDIR="MBResult_A/AR2_Output"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/run_${TIMESTAMP}.out"

./microbenchmark_distribution Method=6,8,9 N=2 Delay=262144 thread_count=16 team_count=64 | tee $OUTFILE
mv overhead_distribution.txt $OUTDIR/overhead_$TIMESTAMP.txt
```
Used with:
```
srun ./run.sh
```

## 📝 Default Configuration

```
#define N_DEF 16382
#define NUM_SAMPLES 20
#define MIN_DELAYLENGTH 512
#define MAX_DELAYLENGTH 262144
#define INNERREPS 20
#define OUTERREPS 40
#define WARMUP_ITERATIONS 10
#define BENCHMARK_SETS 2
#define BENCHMARK_RUNS 2
#define NUM_METHODS 11
#define NUM_SIZES 16
#define MAX_ITER_DEF 6656
#define MAX_ARRAY_SIZE_DEF 65536

```

## 📌 Recent Modifications

- Added parameter parsing
- Added warmup routines
- Added Cray CCE support
- Improved sampling stability
- Added multi-backend Makefile system

## 📧 Contact
For issues or suggestions, please open a GitHub Issue or email:
📬 tuweiyu7749@gmail.com
