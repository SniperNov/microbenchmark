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

OPENMP_SRC := src/openmp/microbenchmark.c src/openmp/common.c
OPENACC_SRC := src/openacc/microbenchmark.c src/openacc/common.c

.PHONY: all openmp openacc wrapper clean help

all: openmp openacc wrapper

openmp: $(OPENMP_BIN) wrapper

openacc: $(OPENACC_BIN) wrapper

$(OPENMP_BIN): $(OPENMP_SRC) $(OPENMP_DEFS)
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(OPENMP_DEFS) \
		BIN=$@ \
		SRC="$(OPENMP_SRC)"

$(OPENACC_BIN): $(OPENACC_SRC) $(OPENACC_DEFS)
	@mkdir -p $(BUILD_DIR)
	$(MAKE) -f Makefile.backend \
		DEFS=$(OPENACC_DEFS) \
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
