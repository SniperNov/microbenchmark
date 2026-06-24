#include <stdio.h>   // Standard input/output functions
#include <math.h>    // Math functions such as log2, pow, ceil, sqrt
#include <stdlib.h>  // Memory allocation and general utilities
#include <string.h>  // String handling functions
#include <strings.h> // strncasecmp
#include <openacc.h> // OpenACC runtime queries and offloading support
#include "common.h" // Project-specific shared declarations

// Output file for fitted overhead / regression summary
#define OUTPUT_FILE "overhead_distribution.txt"

// Output file for all raw timing data
#define RAW_OUTPUT_FILE "raw_times.csv"

// Default array size
#define N_DEF 16382

// Number of delay sample points
#define NUM_SAMPLES 20 // If all methods output same value, enlarge [MIN_DELAYLENGTH, MAX_DELAYLENGTH]

// Minimum tested delaylength
#define MIN_DELAYLENGTH 1 // Too small introduces noise.

// Maximum tested delaylength
#define MAX_DELAYLENGTH 8096 // Log scale sampling across magnitudes, cover launches and execution.

// Number of repetitions inside one timing region
#define INNERREPS 20

// Default total iteration space used by the kernel
#define MAX_ITER_DEF 6656 // total mapping and iteration space (equivalent to MAX_ARRAY_SIZE)

// Default maximum mapped array size
#define MAX_ARRAY_SIZE_DEF 65536

// Number of outer repetitions saved for distribution analysis
#define OUTERREPS 1

// Warm-up iterations before real measurement
#define WARMUP_ITERATIONS 10

// Number of benchmark sets
#define BENCHMARK_SETS 2

// Number of runs in each set
#define BENCHMARK_RUNS 5

// Total number of benchmark methods, indexed 0..11
#define NUM_METHODS 12

// Maximum number of values allowed in size/config arrays
#define NUM_SIZES 16

// Global maximum iteration count used by the kernel
static int g_max_iter = MAX_ITER_DEF; // fixed kernel workload

// Global maximum mapped array size
static int g_max_array_size = MAX_ARRAY_SIZE_DEF; // mapping / memory size

// Global delaylength lower bound
static int g_min_delaylength = MIN_DELAYLENGTH;

// Global delaylength upper bound
static int g_max_delaylength = MAX_DELAYLENGTH;

// Human-readable names for each benchmark method
const char *method_names[] = {
    "pure delay kernel",
    "copy(a[0:N])",
    "copyin(a[0:N])",
    "copyout(a[0:N])",
    "create(a[0:N])",
    "parallel (scalar)",
    "parallel loop",
    "async parallel loop + wait",
    "parallel loop atomic",
    "parallel loop reduction",
    "parallel num_gangs + vector_length",
    "parallel loop gang vector num_gangs + vector_length"};

// Delay sample values used in the benchmark
double delays[NUM_SAMPLES];

// Measured execution times:
// [set][run][delay sample][outer repetition]
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES][OUTERREPS];

static void shuffle_indices_local(int *perm, int n, unsigned int seed);

int main(int argc, char **argv)
{
    // Number of selected methods from command line
    int num_methods = 0;

    // Arrays for user-selected N, gang_count, vector_length, and methods
    int Ns[NUM_SIZES], gang_counts[NUM_SIZES], vector_lengths[NUM_SIZES], methods[NUM_METHODS];

    // Actual number of values stored in the arrays above
    int num_Ns = 0, num_gang_configs = 0, num_vector_configs = 0;

    // Default values
    Ns[0] = N_DEF;
    num_Ns = 1;
    gang_counts[0] = 64;
    num_gang_configs = 1;
    vector_lengths[0] = 128;
    num_vector_configs = 1;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i)
    {
        // Parse Method=
        if (strncasecmp(argv[i], "Method=", 7) == 0)
        {
            num_methods = 0;
            char *token = strtok(argv[i] + 7, ",");
            while (token != NULL && num_methods < NUM_METHODS)
            {
                methods[num_methods++] = atoi(token);
                token = strtok(NULL, ",");
            }
        }

        // Parse Delay=
        else if (strncasecmp(argv[i], "Delay=", 6) == 0)
        {
            int delay_bounds[2] = {0};
            int num_delays = 0;
            char *token = strtok(argv[i] + 6, ",");
            while (token != NULL && num_delays < 2)
            {
                delay_bounds[num_delays++] = atoi(token);
                token = strtok(NULL, ",");
            }

            if (num_delays == 0)
            {
                g_min_delaylength = MIN_DELAYLENGTH;
                g_max_delaylength = MAX_DELAYLENGTH;
            }
            else if (num_delays == 1)
            {
                g_min_delaylength = MIN_DELAYLENGTH;
                g_max_delaylength = delay_bounds[0];
            }
            else
            {
                int a = delay_bounds[0], b = delay_bounds[1];
                if (a > b)
                {
                    int tmp = a;
                    a = b;
                    b = tmp;
                }

                g_min_delaylength = a;
                g_max_delaylength = b;
            }
        }

        // Parse N=
        else if (strncmp(argv[i], "N=", 2) == 0)
        {
            num_Ns = 0;
            char *token = strtok(argv[i] + 2, ",");
            while (token != NULL && num_Ns < NUM_SIZES)
            {
                Ns[num_Ns++] = atoi(token);
                token = strtok(NULL, ",");
            }

            if (num_Ns == 0)
            {
                Ns[0] = N_DEF;
                num_Ns = 1;
                printf("No N specified. Using default N=%d\n", N_DEF);
            }
        }

        // Parse gang_count=
        else if (strncasecmp(argv[i], "gang_count=", 11) == 0)
        {
            num_gang_configs = 0;
            char *token = strtok(argv[i] + 11, ",");
            while (token != NULL && num_gang_configs < NUM_SIZES)
            {
                gang_counts[num_gang_configs++] = atoi(token);
                token = strtok(NULL, ",");
            }
        }

        // Parse vector_length=
        else if (strncasecmp(argv[i], "vector_length=", 14) == 0)
        {
            num_vector_configs = 0;
            char *token = strtok(argv[i] + 14, ",");
            while (token != NULL && num_vector_configs < NUM_SIZES)
            {
                vector_lengths[num_vector_configs++] = atoi(token);
                token = strtok(NULL, ",");
            }
        }

        // Parse MAX_ITER=
        else if (strncasecmp(argv[i], "MAX_ITER=", 9) == 0)
        {
            g_max_iter = atoi(argv[i] + 9);
            printf("MAX_ITER set to %d\n", g_max_iter);
        }

        // Parse MAX_ARRAY_SIZE=
        else if (strncasecmp(argv[i], "MAX_ARRAY_SIZE=", 15) == 0)
        {
            g_max_array_size = atoi(argv[i] + 15);
            printf("MAX_ARRAY_SIZE set to %d\n", g_max_array_size);
        }
    }

    // --- Adjust MAX_ITER and MAX_ARRAY_SIZE once (before benchmarking) ---

    // Largest N in the input list
    int maxN = 0;
    for (int ni = 0; ni < num_Ns; ++ni)
    {
        if (Ns[ni] > maxN)
            maxN = Ns[ni];
    }

    // Largest gang_count * vector_length across all configurations
    int max_prod = 0;
    for (int g = 0; g < num_gang_configs; ++g)
    {
        for (int v = 0; v < num_vector_configs; ++v)
        {
            int prod = gang_counts[g] * vector_lengths[v];
            if (prod > max_prod)
                max_prod = prod;
        }
    }

    // MAX_ITER controls kernel iteration space
    int required_iter = MAX_ITER_DEF;
    if (max_prod > required_iter)
        required_iter = max_prod;
    if (maxN > required_iter)
        required_iter = maxN;
    if (g_max_iter < required_iter)
        g_max_iter = required_iter;

    // MAX_ARRAY_SIZE controls mapped / stored array space
    int required_array_size = MAX_ARRAY_SIZE_DEF;
    if (maxN > required_array_size)
        required_array_size = maxN;
    if (max_prod > required_array_size)
        required_array_size = max_prod;
    if (g_max_array_size < required_array_size)
        g_max_array_size = required_array_size;

    printf("Adjusted limits: MAX_ITER=%d (default=%d, max gangs*vectors=%d, max N=%d), "
           "MAX_ARRAY_SIZE=%d (default=%d)\n",
           g_max_iter, MAX_ITER_DEF, max_prod, maxN,
           g_max_array_size, MAX_ARRAY_SIZE_DEF);

    // Safety check: mapped array space must be able to cover iteration/indexing needs
    if (g_max_iter > g_max_array_size)
    {
        fprintf(stderr,
                "ERROR: MAX_ITER (%d) > MAX_ARRAY_SIZE (%d). Increase MAX_ARRAY_SIZE.\n",
                g_max_iter, g_max_array_size);
        exit(EXIT_FAILURE);
    }

    // --- End adjustment ---

    // Print current runtime settings
    printf("========== Runtime Configuration ==========\n");

    printf("Methods       : ");
    if (num_methods == 0)
    {
        printf("0-11 (all)\n");
    }
    else
    {
        for (int i = 0; i < num_methods; ++i)
            printf("%d%s", methods[i], (i < num_methods - 1) ? ", " : "\n");
    }

    printf("Delay range   : [%d, %d]\n", g_min_delaylength, g_max_delaylength);

    printf("Array size(s) : ");
    for (int ni = 0; ni < num_Ns; ++ni)
        printf("%d%s", Ns[ni], (ni < num_Ns - 1) ? ", " : "\n");

    printf("gang_count(s) : ");
    for (int i = 0; i < num_gang_configs; ++i)
        printf("%d%s", gang_counts[i], (i < num_gang_configs - 1) ? ", " : "\n");

    printf("vector_length(s): ");
    for (int i = 0; i < num_vector_configs; ++i)
        printf("%d%s", vector_lengths[i], (i < num_vector_configs - 1) ? ", " : "\n");

    printf("MAX_ITER      : %d  (kernel workload)\n", g_max_iter);
    printf("MAX_ARRAY_SIZE: %d  (mapping/memory space)\n", g_max_array_size);
    printf("NUM_SAMPLES   : %d\n", NUM_SAMPLES);
    printf("OUTERREPS     : %d\n", OUTERREPS);
    printf("WARMUP_ITERS  : %d\n", WARMUP_ITERATIONS);
    printf("BENCHMARK_SETS: %d\n", BENCHMARK_SETS);
    printf("BENCHMARK_RUNS: %d\n", BENCHMARK_RUNS);
    printf("==========================================\n");

    // ---- Check target device ----
    acc_device_t devtype = acc_get_device_type();
    int numdevs = acc_get_num_devices(devtype);

    printf("OpenACC device type: %d\n", (int)devtype);
    printf("Number of available devices: %d\n", numdevs);

    if (numdevs <= 0)
    {
        fprintf(stderr, "No OpenACC device available. Terminating.\n");
        return 1;
    }

    // Project-specific initialisation
    init(argc, argv);

    // New or rewrite the csv recording all the raw execution_times
    FILE *raw = fopen(RAW_OUTPUT_FILE, "w");
    if (raw)
    {
        fprintf(raw, "method_id,method_name,N,gang_count,vector_length,set,run,delaylength,outerreps,exec_time_us\n");
        fclose(raw);
    }
    else
    {
        perror("Failed to open raw data file.");
        return 1;
    }

    // ---- Print table header ----
    printf("\n========== Benchmark Execution ==========\n");
    printf("%-40s", "Method/N");
    for (int i = 0; i < num_Ns; ++i)
    {
        printf("%30d", Ns[i]);
    }
    printf("\n");

    // Second header line: what each printed result means
    printf("%-40s", "");
    for (int i = 0; i < num_Ns; ++i)
    {
        printf("%30s", "BIC intercept | lowest");
    }
    printf("\n");
    fflush(stdout);

    // ---- Print method results ----
    for (int midx = 0; midx < (num_methods == 0 ? NUM_METHODS : num_methods); ++midx)
    {
        // Use all methods if none were explicitly selected
        int m = (num_methods == 0 ? midx : methods[midx]);
        if (m < 0 || m >= NUM_METHODS)
            continue;

        // Print method name at the start of the row
        printf("%-40s", method_names[m]);
        fflush(stdout);

        for (int g = 0; g < num_gang_configs; ++g)
        {
            for (int v = 0; v < num_vector_configs; ++v)
            {
                for (int nidx = 0; nidx < num_Ns; ++nidx)
                {
                    int N = Ns[nidx];

                    // Allocate the mapped array using the maximum mapping size
                    double *a = (double *)malloc(g_max_array_size * sizeof(double));
                    if (!a)
                    {
                        fprintf(stderr, "Allocation failed for N=%d\n", N);
                        exit(EXIT_FAILURE);
                    }

                    for (int i = 0; i < g_max_array_size; ++i)
                        a[i] = 0.0;

                    // Warm up runtime / device before real measurement
                    warmup_cache(m, N, gang_counts[g], vector_lengths[v]);

                    for (int set = 0; set < BENCHMARK_SETS; ++set)
                    {
                        for (int run = 0; run < BENCHMARK_RUNS; ++run)
                            device_target(m, set, run, a, N, gang_counts[g], vector_lengths[v]);
                    }

                    // Final averaged results for this method / N
                    double intercept, intercept_err;
                    double minval, min_err;

                    compute_offloading_time(&intercept, &intercept_err, &minval, &min_err,
                                            m, method_names[m], N);

                    // Print mean ± standard deviation for both values
                    printf("%10.3f±%-10.3f | %10.3f±%-10.3f",
                           intercept, intercept_err, minval, min_err);
                    fflush(stdout);

                    free(a);
                }
            }
        }
        printf("\n");
    }
    printf("==========================================\n");

    // Project-specific cleanup
    finalise();

    return 0;
}

// Shuffle an index array using a simple deterministic pseudo-random generator
static void shuffle_indices_local(int *perm, int n, unsigned int seed)
{
    // Fallback seed if 0 is passed in
    if (!seed)
        seed = 12345u;

    // Start with identity order: 0,1,2,...,n-1
    for (int i = 0; i < n; ++i)
        perm[i] = i;

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; --i)
    {
        seed = 1664525u * seed + 1013904223u; // LCG update for reproducible pseudo-random numbers
        int j = (int)(seed % (unsigned)(i + 1));
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

// Run one benchmark method for a given set/run/N/gang/vector configuration
void device_target(int method, int set, int run, double *a, int N,
                   int gang_count, int vector_length)
{
    // Generate log-scale sampled delaylengths once
    double log_min = log2((double)g_min_delaylength);
    double log_max = log2((double)g_max_delaylength);
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        if (i == 0)
        {
            delays[i] = g_min_delaylength;
        }
        else if (i == NUM_SAMPLES - 1)
        {
            delays[i] = g_max_delaylength;
        }
        else
        {
            double ratio = (double)i / (NUM_SAMPLES - 1);
            double log_val = log_min + (log_max - log_min) * ratio;
            delays[i] = pow(2.0, log_val);
        }
    }

    // Create a shuffled order for delay samples
    int perm[NUM_SAMPLES];
    unsigned int seed = 12345u + 97u * (unsigned int)set + 1009u * (unsigned int)(run + 1);
    shuffle_indices_local(perm, NUM_SAMPLES, seed);

    // Timing variables. get_time_usec already returns microseconds.
    double start = 0.0, end = 0.0, delay = 0.0;

    // Loop over all sampled delaylengths
    for (int sidx = 0; sidx < NUM_SAMPLES; sidx++)
    {
        int logical = perm[sidx]; // shuffled index
        delay = delays[logical];

        // Repeat each sampled point OUTERREPS times
        for (int orep = 0; orep < OUTERREPS; orep++)
        {
            // Methods 1-4: include data mapping in each measured region
            if (method >= 1 && method <= 4)
            {
                start = get_time_usec();
                for (int irep = 0; irep < INNERREPS; irep++)
                {
                    switch (method)
                    {
                    case 1:
#pragma acc parallel loop copy(a[0:N])
                        for (int i = 0; i < g_max_iter; ++i)
                            delay_kernel((int)delay, &a[i % N]);
                        break;

                    case 2:
#pragma acc parallel loop copyin(a[0:N])
                        for (int i = 0; i < g_max_iter; ++i)
                            delay_kernel((int)delay, &a[i % N]);
                        break;

                    case 3:
#pragma acc parallel loop copyout(a[0:N])
                        for (int i = 0; i < g_max_iter; ++i)
                            delay_kernel((int)delay, &a[i % N]);
                        break;

                    case 4:
#pragma acc parallel loop create(a[0:N])
                        for (int i = 0; i < g_max_iter; ++i)
                            delay_kernel((int)delay, &a[i % N]);
                        break;
                    }

                    a[0] += 1.0;
                    if (a[0] < 0.0)
                    {
                        printf("%f\n", a[0]);
                    }
                }
                end = get_time_usec();
            }

            // Method 8: use a temporary array to isolate atomic-update cost
            else if (method == 8)
            {
                double *tmp = (double *)malloc(N * sizeof(double));
                if (!tmp)
                {
                    fprintf(stderr, "Allocation failed for tmp[N=%d]\n", N);
                    exit(EXIT_FAILURE);
                }

                for (int i = 0; i < N; ++i)
                    tmp[i] = 0.0;

#pragma acc data copy(a[0:g_max_array_size], tmp[0:N])
                {
                    start = get_time_usec();

                    for (int rep = 0; rep < INNERREPS; rep++)
                    {
#pragma acc parallel loop present(a[0:g_max_array_size], tmp[0:N])
                        for (int i = 0; i < g_max_iter; ++i)
                        {
                            delay_kernel((int)delay, &a[i]);
#pragma acc atomic update
                            tmp[i % N] += 1.0;
                        }

                        a[0] += 1.0;
                        if (a[0] < 0.0)
                        {
                            printf("%f\n", a[0]);
                            fflush(stdout);
                        }
                    }

                    end = get_time_usec();
                }

                free(tmp);
            }

            // Methods 0 and 5-7, 9-11: use data region, then measure launch/execution behaviour
            else if (method == 0 || (method >= 5 && method <= 11))
            {
#pragma acc data copy(a[0:g_max_array_size])
                {
                    start = get_time_usec();

                    for (int rep = 0; rep < INNERREPS; rep++)
                    {
                        switch (method)
                        {
                        case 0:
#pragma acc parallel present(a[0:g_max_array_size])
                            {
                                delay_kernel((int)delay, a);
                            }
                            break;

                        case 5:
#pragma acc parallel present(a[0:g_max_array_size])
                            {
                                delay_kernel((int)delay, a);
                            }
                            break;

                        case 6:
#pragma acc parallel loop present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i]);
                            break;

                        case 7:
#pragma acc parallel loop async(1) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i]);
#pragma acc wait(1)
                            break;

                        case 9:
                        {
                            double reduction_sum = 0.0;
#pragma acc parallel loop reduction(+ : reduction_sum) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                            {
                                delay_kernel((int)delay, &a[i]);
                                reduction_sum += 1.0;
                            }
                            if (reduction_sum < 0.0)
                                printf("%f\n", reduction_sum);
                            break;
                        }

                        case 10:
#pragma acc parallel num_gangs(gang_count) vector_length(vector_length) present(a[0:g_max_array_size])
                            {
                                delay_kernel((int)delay, a);
                            }
                            break;

                        case 11:
#pragma acc parallel loop gang vector num_gangs(gang_count) vector_length(vector_length) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i]);
                            break;
                        }

                        a[0] += 1.0;
                        if (a[0] < 0.0)
                        {
                            printf("%f\n", a[0]);
                            fflush(stdout);
                        }
                    }

                    end = get_time_usec();
                }
            }

            // Average time per inner repetition, in microseconds
            double elapsed_us = (end - start) / INNERREPS;

            // Save result only for real runs (run >= 0), not warm-up runs
            if (run >= 0)
            {
                execution_times[set][run][logical][orep] = elapsed_us;

                // Also append raw data to CSV
                FILE *raw = fopen(RAW_OUTPUT_FILE, "a");
                if (raw)
                {
                    const char *mname = (method >= 0 && method < NUM_METHODS) ? method_names[method] : "UNKNOWN";
                    fprintf(raw, "%d,\"%s\",%d,%d,%d,%d,%d,%.0f,%d,%.9f\n",
                            method, mname, N, gang_count, vector_length, set, run,
                            delays[logical], orep, elapsed_us);
                    fclose(raw);
                }
                else
                {
                    perror("fopen raw_times.csv failed");
                }
            }
        }
    }
}

// Compute two summaries from the measured times:
// 1) BIC-selected linear intercept
// 2) lowest observed average time
void compute_offloading_time(double *intercept_avg, double *intercept_err,
                             double *min_avg, double *min_err,
                             int method_id, const char *method_name, int N)
{
    FILE *file = fopen(OUTPUT_FILE, "a");
    if (!file)
    {
        perror("open BIC summary");
        return;
    }

#ifdef PRINT_DISTRIBUTION
    // Optional detailed header in the output text file
    fprintf(file, "\n[Method=%d %s N=%d]\n", method_id, method_name, N);
#endif

    int total_runs = BENCHMARK_SETS * BENCHMARK_RUNS;

    // Store one intercept and one minimum for each set/run
    double intercepts[total_runs];
    double mins[total_runs];

    // Running sums used to compute averages and standard deviations
    double sum_intercept = 0.0, sumsq_intercept = 0.0;
    double sum_min = 0.0, sumsq_min = 0.0;
    int idx = 0;

    // Process each set/run independently
    for (int set = 0; set < BENCHMARK_SETS; ++set)
    {
        for (int run = 0; run < BENCHMARK_RUNS; ++run)
        {
            // Average the OUTERREPS results for each delay point
            double avg_y[NUM_SAMPLES];

            for (int i = 0; i < NUM_SAMPLES; ++i)
            {
                double sum = 0.0;
                for (int orep = 0; orep < OUTERREPS; ++orep)
                    sum += execution_times[set][run][i][orep];
                avg_y[i] = sum / OUTERREPS;
            }

            // Find the lowest observed average time
            double run_min = avg_y[0];
            double run_min_delay = delays[0];
            for (int i = 1; i < NUM_SAMPLES; ++i)
            {
                if (avg_y[i] < run_min)
                {
                    run_min = avg_y[i];
                    run_min_delay = delays[i];
                }
            }

            // Best linear fit selected by BIC
            double best_BIC = INFINITY;
            int best_k = 0;
            double best_a = 0.0, best_b = 0.0, best_R2 = 0.0;

            // Keep at least half of the points for fitting,
            // but at least 5 points when NUM_SAMPLES is small
            const double MIN_KEEP_RATIO = 0.5;
            int min_keep;
            if (NUM_SAMPLES <= 10)
                min_keep = 5;
            else
                min_keep = (int)ceil(NUM_SAMPLES * MIN_KEEP_RATIO);

            // Try all possible starting points k for the fitted linear region
            for (int k = 0; k <= NUM_SAMPLES - min_keep; ++k)
            {
                int n = NUM_SAMPLES - k;
                double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

                // Build sums for least-squares fitting
                for (int i = k; i < NUM_SAMPLES; ++i)
                {
                    sum_x += delays[i];
                    sum_y += avg_y[i];
                    sum_xx += delays[i] * delays[i];
                    sum_xy += delays[i] * avg_y[i];
                }

                double denom = n * sum_xx - sum_x * sum_x;
                if (denom <= 0.0)
                    continue;

                // Linear model: y = a + b*x
                double b = (n * sum_xy - sum_x * sum_y) / denom;
                double a = (sum_y - b * sum_x) / n;
                double rss = 0.0, tss = 0.0;
                double mean_y = sum_y / n;

                // Compute RSS and TSS for this fit
                for (int i = k; i < NUM_SAMPLES; ++i)
                {
                    double yhat = a + b * delays[i];
                    double e = avg_y[i] - yhat;
                    rss += e * e;

                    double dy = avg_y[i] - mean_y;
                    tss += dy * dy;
                }

                if (rss <= 0.0)
                    continue;

                double R2 = (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
                int p = 2; // a and b
                double BIC = n * log(rss / n) + p * log((double)n);

                // Keep the fit with the smallest BIC
                if (BIC < best_BIC)
                {
                    best_BIC = BIC;
                    best_k = k;
                    best_a = a;
                    best_b = b;
                    best_R2 = R2;
                }
            }

            // Save per-run results
            intercepts[idx] = best_a;
            mins[idx] = run_min;
            sum_intercept += best_a;
            sum_min += run_min;

#ifdef PRINT_DISTRIBUTION
            fprintf(file,
                    "Set=%d Run=%d  Lmin=%.0f  Intercept=%.6f us  Slope=%.6f  R2=%.5f  BIC=%.3f  Lowest=%.6f us @ delay=%.0f\n",
                    set, run, delays[best_k], best_a, best_b, best_R2, best_BIC,
                    run_min, run_min_delay);
#endif
            idx++;
        }
    }

    // Compute averages across all set/run combinations
    *intercept_avg = sum_intercept / idx;
    *min_avg = sum_min / idx;

    // Compute standard deviations
    for (int i = 0; i < idx; ++i)
    {
        sumsq_intercept += (intercepts[i] - *intercept_avg) * (intercepts[i] - *intercept_avg);
        sumsq_min += (mins[i] - *min_avg) * (mins[i] - *min_avg);
    }

    *intercept_err = (idx > 1) ? sqrt(sumsq_intercept / (idx - 1)) : 0.0;
    *min_err = (idx > 1) ? sqrt(sumsq_min / (idx - 1)) : 0.0;

#ifdef PRINT_DISTRIBUTION
    fprintf(file, "Average Intercept = %.6f ± %.6f us\n", *intercept_avg, *intercept_err);
    fprintf(file, "Average Lowest    = %.6f ± %.6f us\n", *min_avg, *min_err);
#endif

    fclose(file);
}

void warmup_cache(int method, int N, int gang_count, int vector_length)
{
    double *a = (double *)malloc(g_max_array_size * sizeof(double));
    if (!a)
    {
        perror("warmup malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < g_max_array_size; ++i)
        a[i] = 0.0;

    int dummy_set = 0;
    int dummy_run = -1;

    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        device_target(method, dummy_set, dummy_run, a, N, gang_count, vector_length);
    }

    free(a);
}
