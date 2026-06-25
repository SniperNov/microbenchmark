# Unified Microbenchmark Suite

This branch combines the API-specific microbenchmarks into one user-facing suite. Users no longer need to switch Git branches to choose an API backend.

Current backends:

```text
openmp
openacc
```

Future backends, such as CUDA, can be added behind the same command-line entry point.

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
```

Build with detailed distribution logging:

```bash
make openmp-distribution
make openacc-distribution
```

Override compiler definitions when needed:

```bash
make openmp OPENMP_DEFS=Makefile.defs.gcc
make openacc OPENACC_DEFS=Makefile.defs.openacc.nvc
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
```

Conceptual mapping:

```text
OpenMP team_count     -> OpenACC gang_count
OpenMP thread_count   -> OpenACC worker_count
```

## Outputs

Each backend writes:

```text
raw_times.csv
overhead_distribution.txt
```

Job scripts should move these into `result/` after each run.
