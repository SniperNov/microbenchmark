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
src/openmp/      OpenMP backend implementation
src/openacc/     OpenACC backend implementation
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
./bin/microbenchmark API=openacc Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 gang_count=64 vector_length=128
```

If `API=` is omitted, the dispatcher defaults to OpenMP. You can also set:

```bash
export MICROBENCHMARK_API=openacc
```

## Measurement Logic

The OpenMP and OpenACC backends use the same benchmark workflow:

```text
log-spaced delay samples
deterministic shuffled delay order
warmup iterations before measured runs
raw_times.csv output
BIC-selected linear intercept
lowest observed average timing
```

The backend-specific launch controls are:

```text
OpenMP  : thread_count, team_count
OpenACC : vector_length, gang_count
```

Conceptual mapping:

```text
OpenMP team_count     -> OpenACC gang_count
OpenMP thread_count   -> OpenACC vector_length
```

## Outputs

Each backend writes:

```text
raw_times.csv
overhead_distribution.txt
```

Job scripts should move these into `result/` after each run.
