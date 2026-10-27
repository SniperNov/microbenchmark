# Unified microbenchmark suite.
#
# Build one or more API backends, then run through:
#   ./bin/microbenchmark API=openmp ...
#   ./bin/microbenchmark API=openacc ...
#   ./bin/microbenchmark API=cuda ...
#   ./bin/microbenchmark API=hip ...
#   ./bin/microbenchmark API=sycl ...

OPENMP_DEFS ?= Makefile.defs.nvc
OPENACC_DEFS ?= Makefile.defs.openacc.nvc
CUDA_DEFS ?= Makefile.defs.cuda.nvcc
HIP_DEFS ?= Makefile.defs.hip.hipcc
SYCL_DEFS ?= Makefile.defs.sycl.dpcpp

BUILD_DIR := build
BIN_DIR := bin

OPENMP_BIN := $(BUILD_DIR)/microbenchmark_openmp
OPENACC_BIN := $(BUILD_DIR)/microbenchmark_openacc
CUDA_BIN := $(BUILD_DIR)/microbenchmark_cuda
HIP_BIN := $(BUILD_DIR)/microbenchmark_hip
SYCL_BIN := $(BUILD_DIR)/microbenchmark_sycl

CORE_SRC := src/core/driver.c
OPENMP_SRC := $(CORE_SRC) src/backends/openmp/backend.c
OPENACC_SRC := $(CORE_SRC) src/backends/openacc/backend.c
CUDA_SRC := $(CORE_SRC) src/backends/cuda/backend.cu
HIP_SRC := $(CORE_SRC) src/backends/hip/backend.cpp
SYCL_SRC := $(CORE_SRC) src/backends/sycl/backend.cpp

.PHONY: all openmp openacc cuda hip sycl openmp-distribution openacc-distribution cuda-distribution hip-distribution sycl-distribution distribution wrapper clean help

all: openmp openacc cuda hip sycl wrapper

openmp: $(OPENMP_BIN) wrapper

openacc: $(OPENACC_BIN) wrapper

cuda: $(CUDA_BIN) wrapper

hip: $(HIP_BIN) wrapper

sycl: $(SYCL_BIN) wrapper

openmp-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
openmp-distribution: openmp

openacc-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
openacc-distribution: openacc

cuda-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
cuda-distribution: cuda

hip-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
hip-distribution: hip

sycl-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
sycl-distribution: sycl

distribution: openmp-distribution openacc-distribution cuda-distribution hip-distribution sycl-distribution

$(OPENMP_BIN): $(OPENMP_SRC) $(OPENMP_DEFS)
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(OPENMP_DEFS) \
		EXTRA_CFLAGS="$(EXTRA_CFLAGS)" \
		BIN=$@ \
		SRC="$(OPENMP_SRC)"

$(OPENACC_BIN): $(OPENACC_SRC) $(OPENACC_DEFS)
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(OPENACC_DEFS) \
		EXTRA_CFLAGS="$(EXTRA_CFLAGS)" \
		BIN=$@ \
		SRC="$(OPENACC_SRC)"

$(CUDA_BIN): $(CUDA_SRC) $(CUDA_DEFS) src/core/backend.h src/backends/cuda/openmp_profile.h Makefile Makefile.backend
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(CUDA_DEFS) \
		EXTRA_CFLAGS="$(EXTRA_CFLAGS) -DMICROBENCHMARK_CUDA" \
		BACKEND_HEADERS="src/backends/cuda/openmp_profile.h" \
		BIN=$@ \
		SRC="$(CUDA_SRC)"

$(HIP_BIN): $(HIP_SRC) $(HIP_DEFS) src/core/backend.h src/backends/hip/openmp_profile.h Makefile Makefile.backend
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(HIP_DEFS) \
		EXTRA_CFLAGS="$(EXTRA_CFLAGS) -DMICROBENCHMARK_HIP" \
		BACKEND_HEADERS="src/backends/hip/openmp_profile.h" \
		BIN=$@ \
		SRC="$(HIP_SRC)"

$(SYCL_BIN): $(SYCL_SRC) $(SYCL_DEFS)
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(SYCL_DEFS) \
		EXTRA_CFLAGS="$(EXTRA_CFLAGS)" \
		BIN=$@ \
		SRC="$(SYCL_SRC)"

wrapper:
	@chmod +x $(BIN_DIR)/microbenchmark

clean:
	rm -rf $(BUILD_DIR)
	rm -f raw_times.csv overhead_distribution.txt cuda_device_times.csv hip_device_times.csv

help:
	@echo "Unified microbenchmark suite"
	@echo ""
	@echo "Build targets:"
	@echo "  make openmp                 Build OpenMP backend"
	@echo "  make openacc                Build OpenACC backend"
	@echo "  make cuda                   Build CUDA backend"
	@echo "  make hip                    Build HIP backend (hipcc)"
	@echo "  make sycl                   Build SYCL backend"
	@echo "  make openmp-distribution    Build OpenMP backend with PRINT_DISTRIBUTION"
	@echo "  make openacc-distribution   Build OpenACC backend with PRINT_DISTRIBUTION"
	@echo "  make cuda-distribution      Build CUDA backend with PRINT_DISTRIBUTION"
	@echo "  make hip-distribution       Build HIP backend with PRINT_DISTRIBUTION"
	@echo "  make sycl-distribution      Build SYCL backend with PRINT_DISTRIBUTION"
	@echo "  make all                    Build all backends"
	@echo "  make clean                  Remove build outputs"
	@echo ""
	@echo "Override compiler definitions:"
	@echo "  make openmp OPENMP_DEFS=Makefile.defs.gcc"
	@echo "  make openacc OPENACC_DEFS=Makefile.defs.openacc.nvc"
	@echo "  make cuda CUDA_DEFS=Makefile.defs.cuda.nvcc"
	@echo "  make hip HIP_DEFS=Makefile.defs.hip.hipcc HIP_ARCH=gfx942"
	@echo "  make sycl SYCL_DEFS=Makefile.defs.sycl.dpcpp"
	@echo ""
	@echo "Run:"
	@echo "  ./bin/microbenchmark API=openmp Method=0,1 N=16384 thread_count=32 team_count=4"
	@echo "  ./bin/microbenchmark API=openacc Method=0,1 N=16384 gang_count=64 worker_count=128"
	@echo "  ./bin/microbenchmark API=cuda Method=0,1 N=16384 block_count=108 thread_count=128"
	@echo "  ./bin/microbenchmark API=hip Method=0,1 N=16382 block_count=304 thread_count=64"
	@echo "  ./bin/microbenchmark API=sycl Method=0,1 N=16384 group_count=108 local_size=128"
