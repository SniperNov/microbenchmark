include Makefile.defs.clang

TARGET = microbenchmark
SRCS = microbenchmark.c common.c
OBJS = $(SRCS:.c=.o)

# Default value for N (array size)
N ?= 1024

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

# Compile with offloading overhead distribution enabled
microbenchmark_dist: microbenchmark.c common.o
	$(CC) $(CFLAGS) -DPRINT_DISTRIBUTION -o microbenchmark_dist microbenchmark.c common.o -lm $(LDFLAGS)

# Run and generate plot
run_plot: microbenchmark_dist
	./microbenchmark_dist
	$(PYTHON) plot_overhead.py

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

help:
	@echo "Available make commands:"
	@echo "  make           - Compile microbenchmark"
	@echo "  make clean     - Remove compiled files"
	@echo "  make help      - Show this help message"
	@echo "  make run_all   - Run all benchmarks and save results"
	@echo "  make microbenchmark_distribution - Compile with PRINT_DISTRIBUTION enabled"
	@echo "  make run_plot  - Run microbenchmark and generate a plot"
	@echo ""
	@echo "Usage: ./microbenchmark [method ...] [N]"
	@echo "  No arguments - Runs all benchmarks with default N=$(N) and outputs a table"
	@echo "  1 2 3       - Runs only the specified methods and outputs results in a table"
	@echo ""
	@echo "Available Methods:"
	@echo "  1  - map(tofrom: a)"
	@echo "  2  - map(to: a)"
	@echo "  3  - map(from: a)"
	@echo "  4  - map(alloc: a)"
	@echo "  5  - target"
	@echo "  6  - target teams"
	@echo "  7  - target teams parallel"
	@echo "  8  - target teams distribute parallel for"
	@echo "  9  - target nowait"
	@echo " 10  - target map(to: a[0:N])"
	@echo " 11  - target map(tofrom: a[0:N])"
	@echo " 12  - target teams num_teams(8) thread_limit(256)"
	@echo ""
	@echo "Example:"
	@echo "  ./microbenchmark 1 2 4 12 $(N)"
	@echo "    (Runs methods 1, 2, 4, 12 with array size N=$(N))"

run_all:
	@echo "Running all benchmarks with N=$(N) and generating results table..."
	@./microbenchmark 1 2 3 4 5 6 7 8 9 10 11 12 $(N) > results.txt
	@cat results.txt

clean:
	rm -f $(TARGET) microbenchmark_dist *.o results.txt overhead_distribution.txt overhead_distribution.png
