include Makefile.defs.cray

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
	@echo "  1: #pragma acc parallel loop copy(a[0:N])"
	@echo "  2: #pragma acc parallel loop copyin(a[0:N])"
	@echo "  3: #pragma acc parallel loop copyout(a[0:N])"
	@echo "  4: #pragma acc parallel loop create(a[0:N])"
	@echo "  5: #pragma acc parallel loop"
	@echo "  6: #pragma acc parallel loop gang copy(a[0:N])"
	@echo "  7: #pragma acc parallel loop gang vector copy(a[0:N])"
	@echo "  8: #pragma acc parallel loop num_gangs(64) copy(a[0:N])"
	@echo "  9: #pragma acc parallel loop async(1) copy(a[0:N]) + wait(1)"
	@echo " 10: #pragma acc parallel loop num_gangs(8) vector_length(256) copy(a[0:N])"
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
