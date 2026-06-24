# OpenACC Offloading Microbenchmark

This repository contains the OpenACC version of the microbenchmark suite. It follows the same measurement workflow and output structure as the `MBopenMP` branch while using OpenACC directives in the device kernels.

## Repository Contents

| Path | Description |
|------|-------------|
| `src/microbenchmark.c` | Main benchmark driver and OpenACC benchmark methods. |
| `src/common.c` | Common timing and delay-kernel utilities. |
| `src/common.h` | Shared declarations for the benchmark. |
| `jobs/` | Example job/run scripts. |
| `plots/plot_raw_times.py` | Helper script for plotting raw timing and fitted overhead data. |
| `result/` | Empty output directory placeholder. |
| `Makefile` | Build system for default and distribution builds. |
| `Makefile.defs*` | Compiler-specific build definitions. |

## Build

Select the appropriate `Makefile.defs*` file for the target machine/compiler, then build:

```bash
make
```

To build the distribution version with detailed overhead logging enabled:

```bash
make distribution
```

## Run

Run all OpenACC benchmark methods with the default configuration:

```bash
./microbenchmark
```

Select methods and benchmark parameters using key-value arguments:

```bash
./microbenchmark Method=0,1,2,3,4,5,6,7,8,9,10,11 N=16384 gang_count=64 vector_length=128
```

Optional parameters:

```text
Method=0,1,...,11
Delay=min,max
N=size[,size...]
gang_count=count[,count...]
vector_length=length[,length...]
MAX_ITER=value
MAX_ARRAY_SIZE=value
```

## Outputs

The benchmark writes:

```text
raw_times.csv
overhead_distribution.txt
```

The table printed to stdout reports both:

```text
BIC intercept | lowest
```

This matches the current `MBopenMP` branch measurement logic.

## Control Variables

The OpenACC branch follows the same benchmark idea as `MBopenMP`, but the launch-control names are OpenACC-native:

```text
MBopenMP team_count     -> MBopenACC gang_count
MBopenMP thread_count   -> MBopenACC vector_length
```

So the OpenACC controlled-launch methods use:

```c
num_gangs(gang_count)
vector_length(vector_length)
```

## Methods

| Method | OpenACC form |
|--------|--------------|
| 0 | Pure delay kernel in an OpenACC data region |
| 1 | `copy(a[0:N])` |
| 2 | `copyin(a[0:N])` |
| 3 | `copyout(a[0:N])` |
| 4 | `create(a[0:N])` |
| 5 | `parallel present` scalar launch |
| 6 | `parallel loop present` |
| 7 | `parallel loop async present + wait` |
| 8 | `parallel loop present` with atomic update |
| 9 | `parallel loop present` with reduction |
| 10 | `parallel num_gangs(gang_count) vector_length(vector_length)` scalar launch |
| 11 | `parallel loop gang vector num_gangs(gang_count) vector_length(vector_length)` |
