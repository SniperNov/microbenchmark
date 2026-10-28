# Unified Microbenchmark Suite

This branch combines the API-specific microbenchmarks into one user-facing suite. Users no longer need to switch Git branches to choose an API backend.

Current backends:

```text
openmp
openacc
cuda
hip
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
make hip
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
power-of-two delay points from 1 through 8192
deterministic shuffled delay order
warmup iterations before measured runs
10 timing samples at each delay point
raw_times.csv output
minimum per-delay sample mean (paper result)
BIC-selected linear intercept (diagnostic only)
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
  -> shared driver averages the 10 samples at each delay
  -> shared driver selects the minimum per-delay mean
  -> shared driver also fits a diagnostic BIC intercept
  -> stdout prints "BIC intercept | min mean"
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

Plotting requires NumPy, pandas, and Matplotlib. Install the declared environment
for a machine checkout before starting a benchmark run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r plots/requirements.txt
```

Run scripts check these imports before starting the expensive benchmark and
report a concise installation command instead of a Python traceback.

## Job Scripts

OpenMP scripts are under `jobs/openmp/`; OpenACC scripts are under `jobs/openacc/`.
CUDA scripts for NVIDIA platforms are under `jobs/cuda/`.
SYCL scripts are under `jobs/sycl/`.

For directly comparable NVHPC OpenMP runs on EIDF, use:

```text
jobs/openmp/run_eidf_a100_nvhpc.job  EIDF A100, NVHPC OpenMP
jobs/openmp/run_eidf_h100_nvhpc.job  EIDF H100, NVHPC OpenMP
jobs/openmp/run_eidf_h200_nvhpc.job  EIDF H200, NVHPC OpenMP
```

For GCC OpenMP offload runs on EIDF, use:

```text
jobs/openmp/run_eidf_a100.job  EIDF A100, GCC OpenMP
jobs/openmp/run_eidf_h100.job  EIDF H100, GCC OpenMP
jobs/openmp/run_eidf_h200.job  EIDF H200, GCC OpenMP
```

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

## HIP backend (AMD ROCm)

Build with ROCm's `hipcc`. The build compiles the shared driver as HIP C++ and
uses the existing method IDs, workload parameters, warmups, shuffled delays and
sample counts. Choose an architecture explicitly when cross-compiling:

```bash
make hip-distribution HIP_ARCH=gfx942     # MI300X
# make hip-distribution HIP_ARCH=gfx90a  # MI210
./bin/microbenchmark API=hip Method=0,6 N=16382 Delay=1,8192 \
    block_count=304 thread_count=64 MAX_ITER=16382 MAX_ARRAY_SIZE=65536
```

`HIP_ARCH` may be omitted to use hipcc's detected architecture. The default flags
are `-O0 -std=c++17`; unlike NVCC, HIP compiler options can affect host and device
optimization together. Identical source/optimization labels do not establish
identical GPU instructions, costs per delay, or execution slopes across vendors.

The HIP kernels port the existing CUDA execution contract for methods 0-11:
scalar launch and mapping methods, team-primary work, cyclic thread worksharing,
stream launch/wait, FP64 atomic updates, thread-private N-element array reduction
with team/global merging, team/thread work and repeated team-local phases.
Block barriers and the requested-control index calculations are retained. No
NVIDIA warp-size intrinsics are used. OpenMP, OpenACC and CUDA backend sources
are unchanged by the HIP addition.

CLI controls are `block_count` and `thread_count`. Backend defaults follow the
reference's four teams per visible device and a thread limit of 32. AMD full-run
scripts instead use the same platform configuration as the OpenMP jobs:
MI210 = 104 blocks x 64 threads; MI300X = 304 blocks x 64 threads. Actual logical
participants still depend on method: methods 0-4/7 use one useful thread and
method5 uses one primary thread per block. A wavefront width is a hardware
property, not a replacement for the configured logical participant count.

Without an observed OpenMP profile, output is labelled
`hip_static_cyclic_unverified`. This is not a claim that AMD OpenMP chose the
same owners, participants or synchronization tree. Use `OpenMPProfile=<csv>` or
`HIP_OPENMP_PROFILE=<csv>` for observations collected on the intended target and
exact workload. The host reader validates the same profile schema as CUDA;
chunk boundaries/order remain inferred, not observed runtime region events.
A profile from a different GPU/compiler must not be assumed representative.

`DeviceTiming=0` is the default. `DeviceTiming=1` enables an independent pass
AFTER all baseline samples, using HIP events on the kernel's stream. Baseline
`raw_times.csv` and device-pass `hip_device_times.csv` are separate. The host
baseline creates no events. Allocation and copies remain outside each event
interval; event intervals can still include device idle gaps while the host
submits work, and are not instruction-only timing. Existing device/stream waits
are retained; the event pass does not turn the workload into a batched graph.

Collect all standard raw-data groups without plotting:

```bash
bash jobs/hip/run_AMD.sh
# HIP_MACHINE=MI210 HIP_ARCH=gfx90a bash jobs/hip/run_AMD.sh
# HIP_DEVICE_TIMING=1 bash jobs/hip/run_AMD.sh  # optional separate event pass
```

Results use unique directories under `result/<machine>/HIP/<compiler>/`:
`baseline/<group>/` for host samples and `kernel_execution/<group>/` for optional
HIP event samples. The launcher archives compiler/source provenance and copies
the executable. It does not clean other builds or generate figures. A COSMA5
MI300X Slurm wrapper is in `jobs/hip/run_cosma5.job`; use the site's ROCm environment.

On a ROCm GPU, run the native participant/dataflow checks and driver smoke test:

```bash
bash tests/run_hip_validation.sh
```

This checks all 12 methods, per-thread call counts/order, touched array elements,
atomic/reduction buckets and preservation of host input by the event pass. It
verifies the explicit HIP contract; complete equivalence with an observed AMD
OpenMP execution remains a separate comparison. Local CPU runtime mocks verify
host orchestration and CSV isolation; they are not GPU timing validation:

```bash
python3 tests/check_hip_driver.py
python3 tests/check_hip_timing.py
```

## CUDA baseline and Nsight Systems tracing

The current CUDA collection protocol preserves the original benchmark execution:

1. `baseline/<group>/`: unprofiled host timings, always `DeviceTiming=0`.
2. `tracing/<group>/`: external Nsight Systems CUDA tracing of the same binary,
   arguments and original launch/wait sequence, also `DeviceTiming=0`.
3. `tracing_um/<group>/`: a separate, heavier Unified Memory fault diagnostic.
4. `event_diagnostic/<group>/`: optional legacy event pass, off by default.

There are no kernel/driver changes, added NVTX ranges, profiler start/stop calls,
CUDA events, graph construction or artificial batching in the traced application.
Profiler overhead is nonzero: profiled host samples are retained separately and
must not be pooled with baseline timings. The host-ratio diagnostic compares
separate runs; it is not a pure profiler-overhead correction factor.

The original eight method/N/Delay groups, warmups and sampling are unchanged.
New results still contain 9,940 host samples. Each ordinary trace sweep captures
397,600 benchmark kernel executions (198,800 measured and 198,800 warmups). Exact
counts and deterministic shuffled order are checked before associating activity
with the 9,940 measured samples. Missing events, unexpected kernels, reordered CSV
rows, missing API correlations or application event timing cause validation to fail.

### Run on GH200

Load the same NVHPC/CUDA environment used for the benchmark. Prefer the matching
SDK's `nsys`, rather than an older system installation. For example, with NVHPC:

```bash
export NSYS_BIN="$(dirname "$(command -v nvc)")/nsys"
CUDA_TRACE=1 CUDA_TRACE_UM=1 CUDA_DEVICE_TIMING=0 bash jobs/cuda/run_GH.sh
```

The GH200 wrapper enables the separate UM diagnostic by default. Set
`CUDA_TRACE_UM=0` for ordinary CUDA tracing only. `TMPDIR` is placed under the fresh
result directory, avoiding GH200's nearly full system `/tmp`. Keep results on the
work filesystem. The optional `CUDA_OPENMP_PROFILE` must cover the entire sweep.
Absent a profile, actual OpenMP execution equivalence remains unverified.

### A100, H100 and H200

```bash
sbatch jobs/cuda/run_eidf_a100.job
sbatch jobs/cuda/run_eidf_h100.job
sbatch jobs/cuda/run_eidf_h200.job
```

These wrappers use the same collector. Site-specific allocation/partition/GPU
selection must supply the requested GPU; the native validation log checks the
actual device name and stops before collection if the model is wrong. A100 uses
`sm_80`, while H100/H200/GH200 use `sm_90`. Launch counts remain those in
`jobs/common.sh`. `nsys`, `nvcc`, Python and suitable resources must exist in the
job environment. No plotting dependencies or GPU performance counters are used.
Only GH200 has been tested natively for this tracing workflow so far.

### Trace an already archived binary

This supports the existing GH200 results without recompiling the benchmark:

```bash
python3 jobs/cuda/trace_nsys.py \
  --binary /absolute/path/bin/microbenchmark_cuda \
  --driver-source /absolute/path/sources/src/core/driver.c \
  --output /absolute/path/fresh_trace_directory \
  -- API=cuda Method=0,6 N=16382 Delay=1,8192 \
     block_count=132 thread_count=32 DeviceTiming=0
```

With no `--baseline`, the helper first runs the identical arguments without nsys.
An existing matching unprofiled group can be supplied with `--baseline`. The
original driver snapshot is needed for strict offline warmup/shuffle mapping;
unsupported scheduling changes fail rather than produce guessed labels. A reused
baseline's full environment/argument identity is not independently established by
matching sample keys alone; the suite itself builds and archives both together.

For a complete archived eight-group run, infer the original method/N/Delay,
launch controls and effective MAX_ITER/MAX_ARRAY_SIZE from its CSVs/logs:

```bash
python3 jobs/cuda/trace_existing_run.py /absolute/path/to/previous/raw_run --um-faults
```

This reuses the archived executable and host baseline, writes a fresh timestamped
`nsys_*` directory below the archived run, and never rewrites the original data.
Only the necessary profiled executions are run. Both ordinary CUDA and optional
UM passes are checked for all 9,940 measured samples.

Each capture preserves `trace.nsys-rep`, `trace.sqlite`, command/version metadata,
profile logs and the profiled process's raw host output. `derived/` contains:

- `kernel_activity.csv`: every GPU activity, with warmups explicitly marked.
- `kernel_samples.csv`: mean kernel activity duration per original sample, in us,
  with method/N/Delay/set/run/outer repetition and inner repetition count.
- `cuda_api_activity.csv`: host-side CUDA calls, with separate start/end/duration.
- `synchronization_api_activity.csv`: synchronization-call durations.
- `memory_copy_activity.csv` / `memory_set_activity.csv`, when activities exist.
- `trace_summary.json`: mapping counts, UM table visibility and interpretation.

A `TRACE_COMPLETE.json` is written only after these checks pass. The full collector
writes `SUITE_COMPLETE.json` after all eight baseline/trace partitions pass.

### Measurement interpretation

Kernel time is the GPU activity's end minus start, not a pre-launch event span.
The launch API duration, API-to-GPU start gap and synchronization API duration are
exported separately. The signed gap from API end can be negative when execution
overlaps the call; it is not an exact hardware queue residence measurement.
Synchronization APIs can include waiting for GPU work, so their duration is not
an isolated synchronization overhead. CPU and GPU intervals can overlap and must
not be blindly added.

All setup, warmup and teardown memory/API activity remains in the original trace;
only the derived measured kernel samples omit warmups. Nsight's own device-side
CUDA-event completion tracing is explicitly disabled when supported, to avoid
adding that synchronization mechanism. CPU sampling/context-switch tracing is
disabled; native GH200 runs succeeded without GPU counter privileges.

GH200 hardware-coherent/unified memory does not imply that this explicit
`cudaMalloc`/`cudaMemcpy` backend performs no transfers. UM-fault collection is a
separate diagnostic with potentially substantial overhead. No UM rows can mean
no observed activity or unavailable visibility; it does not establish zero data
movement or zero coherence cost. Original historical event data are retained as
a separate diagnostic measurement scope.

HIP on COSMA5 uses AMD/ROCm. Nsight CUDA tracing is not a HIP profiler. The HIP
backend and MI300X runner are present, but native compilation, validation and
collection on COSMA5 remain required; ROCm tracing must be selected there.

Relevant vendor guidance:
[Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[CUDA activity/launch analysis](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).
