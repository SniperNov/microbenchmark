# Unified microbenchmark suite.
#
# Build one or more API backends, then run through:
#   ./bin/microbenchmark API=openmp ...
#   ./bin/microbenchmark API=openacc ...

OPENMP_DEFS ?= Makefile.defs.nvc
OPENACC_DEFS ?= Makefile.defs.openacc.nvc

BUILD_DIR := build
BIN_DIR := bin

OPENMP_BIN := $(BUILD_DIR)/microbenchmark_openmp
OPENACC_BIN := $(BUILD_DIR)/microbenchmark_openacc

CORE_SRC := src/core/driver.c
OPENMP_SRC := $(CORE_SRC) src/backends/openmp/backend.c
OPENACC_SRC := $(CORE_SRC) src/backends/openacc/backend.c

.PHONY: all openmp openacc openmp-distribution openacc-distribution distribution wrapper clean help

all: openmp openacc wrapper

openmp: $(OPENMP_BIN) wrapper

openacc: $(OPENACC_BIN) wrapper

openmp-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
openmp-distribution: openmp

openacc-distribution: EXTRA_CFLAGS += -DPRINT_DISTRIBUTION
openacc-distribution: openacc

distribution: openmp-distribution

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

wrapper:
	@chmod +x $(BIN_DIR)/microbenchmark

clean:
	rm -rf $(BUILD_DIR)
	rm -f raw_times.csv overhead_distribution.txt

help:
	@echo "Unified microbenchmark suite"
	@echo ""
	@echo "Build targets:"
	@echo "  make openmp                 Build OpenMP backend"
	@echo "  make openacc                Build OpenACC backend"
	@echo "  make openmp-distribution    Build OpenMP backend with PRINT_DISTRIBUTION"
	@echo "  make openacc-distribution   Build OpenACC backend with PRINT_DISTRIBUTION"
	@echo "  make all                    Build all backends"
	@echo "  make clean                  Remove build outputs"
	@echo ""
	@echo "Override compiler definitions:"
	@echo "  make openmp OPENMP_DEFS=Makefile.defs.gcc"
	@echo "  make openacc OPENACC_DEFS=Makefile.defs.openacc.nvc"
	@echo ""
	@echo "Run:"
	@echo "  ./bin/microbenchmark API=openmp Method=0,1 N=16384 thread_count=32 team_count=4"
	@echo "  ./bin/microbenchmark API=openacc Method=0,1 N=16384 gang_count=64 vector_length=128"
