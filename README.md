# Unified Microbenchmark Suite

This branch combines the API-specific microbenchmarks into one user-facing suite. Users no longer need to switch Git branches to choose an API backend.

Current backends:

```text
openmp
openacc
cuda
sycl
```

Future backends can be added behind the same command-line entry point.

## Layout

```text
src/core/        Shared CLI, measurement loop, CSV output, and fitting logic
src/backends/    API-specific launch/mapping implementations
bin/             Unified command-line dispatcher
build/           Compiled backend binaries
jobs/            Example job scripts
plots/           Plotting helpers
result/          Output directory placeholder
```

## Build

Build all currently available backends:

```bash
make all
```

Build only one backend:

```bash
make openmp
make openacc
make cuda
make sycl
```

Build with detailed distribution logging:

```bash
make openmp-distribution
make openacc-distribution
make cuda-distribution
make sycl-distribution
```

Override compiler definitions when needed:

```bash
make openmp OPENMP_DEFS=Makefile.defs.gcc
make openacc OPENACC_DEFS=Makefile.defs.openacc.nvc
make cuda CUDA_DEFS=Makefile.defs.cuda.nvcc
make sycl SYCL_DEFS=Makefile.defs.sycl.dpcpp
make sycl SYCL_DEFS=Makefile.defs.sycl.dpcpp.cuda
```

## Run

Use one command and select the backend with `API=`.

OpenMP:

```bash
./bin/microbenchmark API=openmp Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 thread_count=32 team_count=4
```

OpenACC:

```bash
./bin/microbenchmark API=openacc Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 gang_count=64 worker_count=128
```

CUDA:

```bash
./bin/microbenchmark API=cuda Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 block_count=108 thread_count=128
```

SYCL:

```bash
./bin/microbenchmark API=sycl Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 group_count=108 local_size=128
```

If `API=` is omitted, the dispatcher defaults to OpenMP. You can also set:

```bash
export MICROBENCHMARK_API=openacc
```

## Measurement Logic

The OpenMP and OpenACC backends use one shared benchmark driver:

```text
log-spaced delay samples
deterministic shuffled delay order
warmup iterations before measured runs
raw_times.csv output
BIC-selected linear intercept
lowest observed average timing
```

Backend code only implements API-specific launch/mapping methods and default configuration names. This keeps OpenMP, OpenACC, and future CUDA tests aligned by construction.

## Execution Flow

When you run:

```bash
./bin/microbenchmark API=openmp Method=1,2 N=16384 thread_count=32 team_count=4
```

the suite follows this path:

```text
bin/microbenchmark
  -> chooses build/microbenchmark_openmp
  -> shared driver parses Method/N/Delay/config values
  -> shared driver generates delay samples
  -> backend checks that the target device is available
  -> shared driver warms up the selected method
  -> shared driver loops over set/run/delay/outer repetition
  -> backend runs the API-specific target/parallel construct
  -> shared driver writes raw_times.csv
  -> shared driver fits BIC intercept and finds lowest timing
  -> stdout prints "BIC intercept | lowest"
```

The backend-specific launch controls are:

```text
OpenMP  : thread_count, team_count
OpenACC : gang_count, worker_count
CUDA    : block_count, thread_count
SYCL    : group_count, local_size
```

Conceptual mapping:

```text
OpenMP team_count     -> OpenACC gang_count
OpenMP thread_count   -> OpenACC worker_count
OpenMP/OpenACC teams  -> CUDA block_count
OpenMP/OpenACC worker -> CUDA thread_count
CUDA block_count      -> SYCL group_count
CUDA thread_count     -> SYCL local_size
```

## API-Neutral Method Logic

All backends implement the same research logic. The syntax is different, but the measured idea is the same:

| Method | API-neutral idea |
|--------|------------------|
| 0 | smallest device launch baseline |
| 1 | allocate + host-to-device copy + kernel + device-to-host copy + release |
| 2 | allocate + host-to-device copy + kernel + release |
| 3 | allocate + kernel + device-to-host copy + release |
| 4 | allocate + kernel + release |
| 5 | one unit of work per team/gang/block/work-group |
| 6 | parallel loop over many work-items |
| 7 | asynchronous launch followed by explicit wait |
| 8 | atomic update |
| 9 | reduction |
| 10 | explicit launch shape |
| 11 | repeated work using explicit launch shape |

OpenMP/OpenACC express data movement with directive clauses. CUDA/SYCL express the same idea with explicit allocation, copy, and free calls.

## CUDA Backend Notes

CUDA uses explicit runtime calls instead of directive-based target/data regions, so not every method is a literal translation. The CUDA backend keeps the same measurement driver and method numbering, but maps the API ideas as follows:

| Method | CUDA form | Relationship to OpenMP/OpenACC |
|--------|-----------|--------------------------------|
| 0 | one scalar kernel launch | Directly comparable launch baseline |
| 1 | `cudaMalloc + H2D + kernel + D2H + cudaFree` | Mimics `map(tofrom)` / `copy` lifecycle |
| 2 | `cudaMalloc + H2D + kernel + cudaFree` | Mimics `map(to)` / `copyin` lifecycle |
| 3 | `cudaMalloc + kernel + D2H + cudaFree` | Mimics `map(from)` / `copyout` lifecycle |
| 4 | `cudaMalloc + kernel + cudaFree` | Mimics `map(alloc)` / `create` lifecycle |
| 5 | one scalar action per CUDA block | Mimics teams/gangs scalar launch |
| 6 | grid-stride loop kernel | Direct CUDA equivalent of parallel loop worksharing |
| 7 | stream launch plus stream synchronize | Mimics async launch followed by wait |
| 8 | grid-stride loop with `atomicAdd` | Direct CUDA atomic analogue |
| 9 | shared-memory block reduction plus final atomic add | CUDA-style reduction analogue |
| 10 | explicit `block_count` and `thread_count` launch | Direct CUDA launch-shape measurement |
| 11 | repeated explicit block/thread launch work, with `N` as repetition count | Mimics repeated inner parallel/worker work |

Methods 0, 6, 7, 8, 10, and 11 are the cleanest CUDA counterparts. Methods 1-4 are comparable by intent rather than syntax because CUDA exposes allocation and copy operations explicitly. Method 9 is also comparable by intent, but the implementation must use CUDA reduction mechanics rather than a directive-level reduction clause.

## SYCL Backend Notes

The SYCL backend uses USM explicit memory management, not buffer/accessor. This keeps the data-movement methods close to CUDA:

```text
sycl::malloc_device
queue.memcpy host -> device
queue.parallel_for / queue.single_task
queue.memcpy device -> host
sycl::free
```

SYCL launch-shape terms:

```text
group_count = number of SYCL work-groups
local_size  = number of SYCL work-items inside each work-group
```

This corresponds to CUDA:

```text
CUDA block_count  -> SYCL group_count
CUDA thread_count -> SYCL local_size
```

The SYCL backend uses `queue`, `event`, and `nd_range`:

```text
queue    = command queue for copies and kernels
event    = handle for one submitted async operation
nd_range = explicit global/local kernel shape
```

Methods 1-4 use `malloc_device` and `queue.memcpy` to match CUDA-style explicit data lifecycle. Method 8 uses `atomic_ref`. Method 9 uses work-group local memory plus one atomic add per group.

## Outputs

Each backend writes:

```text
raw_times.csv
overhead_distribution.txt
```

Job scripts should move these into `result/` after each run.

The platform job scripts save results using:

```text
result/<Machine>/<API>/<CompilerVersion>/
```

Examples:

```text
result/GH200/CUDA/NVCC_13_1_115/
result/GH200/OpenACC/NVC_26_3/
result/H100/OpenMP/GCC_13_2_0/
```

The compiler-version directory is detected by the job script from commands such as `nvcc --version`, `nvc --version`, `gcc -dumpfullversion -dumpversion`, or `cc --version`.

## Job Scripts

OpenMP scripts are under `jobs/openmp/`; OpenACC scripts are under `jobs/openacc/`.
CUDA scripts for NVIDIA platforms are under `jobs/cuda/`.
SYCL scripts are under `jobs/sycl/`.

For OpenACC, the current scripts are:

```text
jobs/openacc/run_eidf_a100.job     EIDF A100, NVHPC OpenACC
jobs/openacc/run_eidf_h100.job     EIDF H100, NVHPC OpenACC
jobs/openacc/run_eidf_h200.job     EIDF H200, NVHPC OpenACC
jobs/openacc/run_GH.sh            Grace Hopper / GH200, NVHPC OpenACC
jobs/openacc/run_archer2.job      Archer2 MI210, Cray OpenACC
jobs/openacc/run_cosma5.job       COSMA5 MI300X, exploratory OpenACC
```

For CUDA, the current scripts are:

```text
jobs/cuda/run_eidf_a100.job        EIDF A100, CUDA
jobs/cuda/run_eidf_h100.job        EIDF H100, CUDA
jobs/cuda/run_eidf_h200.job        EIDF H200, CUDA
jobs/cuda/run_GH.sh               Grace Hopper / GH200, CUDA
```

For SYCL, the current scripts are:

```text
jobs/sycl/run_eidf_a100.job        EIDF A100, SYCL
jobs/sycl/run_eidf_h100.job        EIDF H100, SYCL
jobs/sycl/run_eidf_h200.job        EIDF H200, SYCL
jobs/sycl/run_GH.sh               Grace Hopper / GH200, SYCL
```

All non-GH200 runs are Slurm jobs and must be submitted from the repository
root. For example:

```bash
sbatch jobs/openacc/run_eidf_h100.job
sbatch jobs/cuda/run_eidf_h100.job
sbatch jobs/sycl/run_eidf_h100.job
sbatch jobs/openacc/run_archer2.job
sbatch jobs/openmp/run_cosma5.job
```

GH200 is the only direct-run platform. Its scripts are executable:

```bash
./jobs/openmp/run_GH.sh
./jobs/openacc/run_GH.sh
./jobs/cuda/run_GH.sh
./jobs/sycl/run_GH.sh
```

Every run writes all artifacts below
`result/<machine>/<API>/<compiler_version>/`. In addition to benchmark CSV,
summary, plot, and log files, the directory contains an immutable copy of the
executed submission/run script and a Git-state record with the commit, branch,
working-tree status, and diff. Slurm stdout is moved into the same directory
when the job exits.

Shared experiment parameters and per-platform launch dimensions are defined in
`jobs/common.sh`. Every API on the same platform consumes the same platform
group count and width; API-specific names such as teams/threads, gangs/workers,
blocks/threads, and groups/local-size are only aliases for that shared pair.

The scripts use the same benchmark grouping as the OpenMP runs:

```text
M0_M10       fixed N launch-control cases
M1to4        data-clause size sweep
M5/M6/M7     scalar, loop, and async launch cases
M8/M9        atomic and reduction cases
M11_parreps  repeated inner worker-loop case
```

For OpenACC, `gang_count` is the team-like control and `worker_count` is the thread-like control.
