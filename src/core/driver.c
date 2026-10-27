#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "backend.h"

#define OUTPUT_FILE "overhead_distribution.txt"
#define RAW_OUTPUT_FILE "raw_times.csv"

#ifdef MICROBENCHMARK_DEVICE_TIMING
#define DEVICE_OUTPUT_FILE BACKEND_DEVICE_OUTPUT_FILE
static int g_device_timing = 0;
#endif

#define N_DEF 16382
#define MAX_DELAY_POINTS 32
#define MIN_DELAYLENGTH 1
#define MAX_DELAYLENGTH 8192
#define INNERREPS 20
#define MAX_ITER_DEF 6656
#define MAX_ARRAY_SIZE_DEF 65536
#define OUTERREPS 1
#define WARMUP_ITERATIONS 10
#define BENCHMARK_SETS 2
#define BENCHMARK_RUNS 5
#define NUM_SAMPLES (BENCHMARK_SETS * BENCHMARK_RUNS)
#define NUM_SIZES 16
#define MAX_METHODS 32

static int g_max_iter = MAX_ITER_DEF;
static int g_max_array_size = MAX_ARRAY_SIZE_DEF;
static int g_min_delaylength = MIN_DELAYLENGTH;
static int g_max_delaylength = MAX_DELAYLENGTH;

static int g_num_delay_points = 0;
static double delays[MAX_DELAY_POINTS];
static double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][MAX_DELAY_POINTS][OUTERREPS];

// Command-line options use key=value form, for example Method=1,2 or N=16384.
static int key_matches(const char *arg, const char *key)
{
    size_t n = strlen(key);
    return strncasecmp(arg, key, n) == 0 && arg[n] == '=';
}

static char *value_for(char *arg)
{
    char *eq = strchr(arg, '=');
    return eq ? eq + 1 : arg;
}

static void parse_int_list(char *value, int *items, int *count, int max_count)
{
    *count = 0;
    char *token = strtok(value, ",");
    while (token != NULL && *count < max_count)
    {
        items[(*count)++] = atoi(token);
        token = strtok(NULL, ",");
    }
}

static void parse_delay_bounds(char *value)
{
    int bounds[2] = {0};
    int count = 0;
    parse_int_list(value, bounds, &count, 2);

    if (count == 0)
    {
        g_min_delaylength = MIN_DELAYLENGTH;
        g_max_delaylength = MAX_DELAYLENGTH;
    }
    else if (count == 1)
    {
        g_min_delaylength = MIN_DELAYLENGTH;
        g_max_delaylength = bounds[0];
    }
    else
    {
        int a = bounds[0];
        int b = bounds[1];
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

static void generate_delays(void)
{
    // The paper protocol uses non-zero powers of two. Requiring power-of-two
    // bounds prevents a nominal range such as 1..8096 from silently producing
    // a different, implementation-defined sampling grid.
    if (g_min_delaylength <= 0 || g_max_delaylength < g_min_delaylength ||
        (g_min_delaylength & (g_min_delaylength - 1)) != 0 ||
        (g_max_delaylength & (g_max_delaylength - 1)) != 0)
    {
        fprintf(stderr,
                "Delay bounds must be positive powers of two (got %d,%d).\n",
                g_min_delaylength, g_max_delaylength);
        exit(EXIT_FAILURE);
    }

    g_num_delay_points = 0;
    for (int delay = g_min_delaylength; delay <= g_max_delaylength; delay *= 2)
    {
        if (g_num_delay_points >= MAX_DELAY_POINTS)
        {
            fprintf(stderr, "Too many delay points (maximum %d).\n", MAX_DELAY_POINTS);
            exit(EXIT_FAILURE);
        }
        delays[g_num_delay_points++] = (double)delay;
        if (delay > g_max_delaylength / 2)
            break;
    }
}

static void shuffle_indices(int *perm, int n, unsigned int seed)
{
    // Randomize delay order deterministically. This reduces monotonic drift
    // effects while keeping runs reproducible.
    if (!seed)
        seed = 12345u;

    for (int i = 0; i < n; ++i)
        perm[i] = i;

    for (int i = n - 1; i > 0; --i)
    {
        seed = 1664525u * seed + 1013904223u;
        int j = (int)(seed % (unsigned)(i + 1));
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

static void run_target(int method, int set, int run, double *a, int N,
                       const backend_config_t *config)
{
    // One run visits every sampled delay. The backend measures one delay point;
    // the shared driver stores raw data and keeps the output format identical.
    int perm[MAX_DELAY_POINTS];
    unsigned int seed = 12345u + 97u * (unsigned int)set + 1009u * (unsigned int)(run + 1);
    shuffle_indices(perm, g_num_delay_points, seed);

    for (int sidx = 0; sidx < g_num_delay_points; ++sidx)
    {
        int logical = perm[sidx];
        int delay = (int)delays[logical];

        for (int orep = 0; orep < OUTERREPS; ++orep)
        {
            double elapsed_us = backend_run_method(method, a, N, delay,
                                                   g_max_iter, g_max_array_size,
                                                   config, INNERREPS);

            if (run >= 0)
            {
                execution_times[set][run][logical][orep] = elapsed_us;

                FILE *raw = fopen(RAW_OUTPUT_FILE, "a");
                if (raw)
                {
                    fprintf(raw, "%d,\"%s\",%d,%d,%d,%d,%d,%.0f,%d,%.9f\n",
                            method, backend_method_name(method), N,
                            config->control_a, config->control_b,
                            set, run, delays[logical], orep, elapsed_us);
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

#ifdef MICROBENCHMARK_DEVICE_TIMING
static void run_device_target(int method, int set, int run, const double *a, int N,
                              const backend_config_t *config)
{
    // This pass runs after every baseline sample has completed, so event
    // profiling cannot alter the GPU state seen by later baseline samples.
    int perm[MAX_DELAY_POINTS];
    unsigned int seed = 12345u + 97u * (unsigned int)set + 1009u * (unsigned int)(run + 1);
    shuffle_indices(perm, g_num_delay_points, seed);
    for (int sidx = 0; sidx < g_num_delay_points; ++sidx)
    {
        int logical = perm[sidx];
        int delay = (int)delays[logical];
        for (int orep = 0; orep < OUTERREPS; ++orep)
        {
            // The GPU backend clones a; neither raw baseline data nor the
            // baseline fitting arrays are changed by this profiling call.
            double device_us = backend_run_device_method(
                method, a, N, delay, g_max_iter, g_max_array_size, config, INNERREPS);
            int actual_blocks = 0, actual_threads = 0;
            backend_gpu_last_launch_config(&actual_blocks, &actual_threads);
            if (!isfinite(device_us) || device_us < 0.0)
            {
                fprintf(stderr, "Invalid " BACKEND_DEVICE_LABEL " device time for method %d.\n", method);
                exit(EXIT_FAILURE);
            }
            FILE *device_raw = fopen(DEVICE_OUTPUT_FILE, "a");
            if (!device_raw)
            {
                perror("fopen " DEVICE_OUTPUT_FILE " failed");
                exit(EXIT_FAILURE);
            }
            fprintf(device_raw,
                    "%d,\"%s\",%d,%d,%d,%d,%d,%.0f,%d,%d,%d,%d,%.9f,same_stream_events_separate_pass,%d,%d,%s\n",
                    method, backend_method_name(method), N,
                    config->control_a, config->control_b,
                    set, run, delays[logical], orep,
                    g_max_iter, g_max_array_size, INNERREPS, device_us,
                    actual_blocks, actual_threads, backend_gpu_alignment_mode());
            fclose(device_raw);
        }
    }
}
#endif

static void warmup_cache(int method, int N, const backend_config_t *config)
{
    // Warmup is not recorded. It gives the runtime and target device a chance
    // to initialize before the measured runs.
    double *a = (double *)malloc((size_t)g_max_array_size * sizeof(double));
    if (!a)
    {
        perror("warmup malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < g_max_array_size; ++i)
        a[i] = 0.0;

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        run_target(method, 0, -1, a, N, config);

    free(a);
}

static void compute_offloading_time(double *intercept_avg, double *intercept_err,
                                    double *min_avg, double *min_err,
                                    int method_id, const char *method_name, int N)
{
    // For each set/run, fit y = intercept + slope * delay. The BIC search
    // chooses which tail of the delay curve is most linear.
    FILE *file = fopen(OUTPUT_FILE, "a");
    if (!file)
    {
        perror("open BIC summary");
        return;
    }

#ifdef PRINT_DISTRIBUTION
    fprintf(file, "\n[Method=%d %s N=%d]\n", method_id, method_name, N);
#endif

    int total_runs = BENCHMARK_SETS * BENCHMARK_RUNS;
    double intercepts[total_runs];
    double sum_intercept = 0.0, sumsq_intercept = 0.0;
    int idx = 0;

    for (int set = 0; set < BENCHMARK_SETS; ++set)
    {
        for (int run = 0; run < BENCHMARK_RUNS; ++run)
        {
            double avg_y[MAX_DELAY_POINTS];
            for (int i = 0; i < g_num_delay_points; ++i)
            {
                double sum = 0.0;
                for (int orep = 0; orep < OUTERREPS; ++orep)
                    sum += execution_times[set][run][i][orep];
                avg_y[i] = sum / OUTERREPS;
            }

            double run_min = avg_y[0];
            double run_min_delay = delays[0];
            for (int i = 1; i < g_num_delay_points; ++i)
            {
                if (avg_y[i] < run_min)
                {
                    run_min = avg_y[i];
                    run_min_delay = delays[i];
                }
            }

            double best_BIC = INFINITY;
            int best_k = 0;
            double best_a = 0.0, best_b = 0.0, best_R2 = 0.0;

            const double MIN_KEEP_RATIO = 0.5;
            int min_keep = (g_num_delay_points <= 10) ? 5 : (int)ceil(g_num_delay_points * MIN_KEEP_RATIO);

            for (int k = 0; k <= g_num_delay_points - min_keep; ++k)
            {
                int n = g_num_delay_points - k;
                double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

                for (int i = k; i < g_num_delay_points; ++i)
                {
                    sum_x += delays[i];
                    sum_y += avg_y[i];
                    sum_xx += delays[i] * delays[i];
                    sum_xy += delays[i] * avg_y[i];
                }

                double denom = n * sum_xx - sum_x * sum_x;
                if (denom <= 0.0)
                    continue;

                double b = (n * sum_xy - sum_x * sum_y) / denom;
                double a = (sum_y - b * sum_x) / n;
                double rss = 0.0, tss = 0.0;
                double mean_y = sum_y / n;

                for (int i = k; i < g_num_delay_points; ++i)
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
                double BIC = n * log(rss / n) + 2.0 * log((double)n);

                if (BIC < best_BIC)
                {
                    best_BIC = BIC;
                    best_k = k;
                    best_a = a;
                    best_b = b;
                    best_R2 = R2;
                }
            }

            intercepts[idx] = best_a;
            sum_intercept += best_a;

#ifdef PRINT_DISTRIBUTION
            fprintf(file,
                    "Set=%d Run=%d  Lmin=%.0f  Intercept=%.6f us  Slope=%.6f  R2=%.5f  BIC=%.3f  Lowest=%.6f us @ delay=%.0f\n",
                    set, run, delays[best_k], best_a, best_b, best_R2, best_BIC,
                    run_min, run_min_delay);
#endif
            idx++;
        }
    }

    *intercept_avg = sum_intercept / idx;

    // Paper estimator: average the ten timing samples independently at each
    // delay, then choose the smallest observed per-delay mean. The reported
    // uncertainty is the sample standard deviation at that selected delay.
    int best_delay = 0;
    double best_mean = INFINITY;
    double best_sd = 0.0;
    for (int d = 0; d < g_num_delay_points; ++d)
    {
        double sum = 0.0;
        int count = 0;
        for (int set = 0; set < BENCHMARK_SETS; ++set)
            for (int run = 0; run < BENCHMARK_RUNS; ++run)
                for (int orep = 0; orep < OUTERREPS; ++orep)
                {
                    sum += execution_times[set][run][d][orep];
                    count++;
                }

        double mean = sum / count;
        double sumsq = 0.0;
        for (int set = 0; set < BENCHMARK_SETS; ++set)
            for (int run = 0; run < BENCHMARK_RUNS; ++run)
                for (int orep = 0; orep < OUTERREPS; ++orep)
                {
                    double delta = execution_times[set][run][d][orep] - mean;
                    sumsq += delta * delta;
                }
        double sd = (count > 1) ? sqrt(sumsq / (count - 1)) : 0.0;

        if (mean < best_mean)
        {
            best_delay = d;
            best_mean = mean;
            best_sd = sd;
        }
    }
    *min_avg = best_mean;
    *min_err = best_sd;

    for (int i = 0; i < idx; ++i)
    {
        sumsq_intercept += (intercepts[i] - *intercept_avg) * (intercepts[i] - *intercept_avg);
    }

    *intercept_err = (idx > 1) ? sqrt(sumsq_intercept / (idx - 1)) : 0.0;

#ifdef PRINT_DISTRIBUTION
    fprintf(file, "Average Intercept = %.6f ± %.6f us\n", *intercept_avg, *intercept_err);
    fprintf(file, "Minimum Delay Mean = %.6f ± %.6f us @ delay=%.0f (%d samples)\n",
            *min_avg, *min_err, delays[best_delay], NUM_SAMPLES);
#endif

    fclose(file);
}

int main(int argc, char **argv)
{
    // High-level flow:
    // 1. Parse shared CLI options and backend-specific launch controls.
    // 2. Generate delay samples and check the selected device.
    // 3. For each method/config/N, warm up, measure, write CSV, and summarize.
    int methods[MAX_METHODS], Ns[NUM_SIZES], control_as[NUM_SIZES], control_bs[NUM_SIZES];
    int num_methods = 0, num_Ns = 1, num_control_as = 1, num_control_bs = 1;
    backend_config_t defaults;

    backend_default_config(&defaults);
    Ns[0] = N_DEF;
    control_as[0] = defaults.control_a;
    control_bs[0] = defaults.control_b;

    for (int i = 1; i < argc; ++i)
    {
        if (key_matches(argv[i], "API"))
        {
            continue;
        }
#ifdef MICROBENCHMARK_DEVICE_TIMING
        else if (key_matches(argv[i], "DeviceTiming"))
        {
            const char *value = value_for(argv[i]);
            if (strcmp(value, "0") != 0 && strcmp(value, "1") != 0)
            {
                fprintf(stderr, "DeviceTiming must be 0 or 1.\n");
                return 1;
            }
            g_device_timing = value[0] == '1';
        }
#endif
        else if (key_matches(argv[i], "Method"))
        {
            parse_int_list(value_for(argv[i]), methods, &num_methods, MAX_METHODS);
        }
        else if (key_matches(argv[i], "Delay"))
        {
            parse_delay_bounds(value_for(argv[i]));
        }
        else if (key_matches(argv[i], "N"))
        {
            parse_int_list(value_for(argv[i]), Ns, &num_Ns, NUM_SIZES);
            if (num_Ns == 0)
            {
                Ns[0] = N_DEF;
                num_Ns = 1;
            }
        }
        else if (key_matches(argv[i], backend_control_a_name()))
        {
            parse_int_list(value_for(argv[i]), control_as, &num_control_as, NUM_SIZES);
        }
        else if (key_matches(argv[i], backend_control_b_name()))
        {
            parse_int_list(value_for(argv[i]), control_bs, &num_control_bs, NUM_SIZES);
        }
        else if (key_matches(argv[i], "MAX_ITER"))
        {
            g_max_iter = atoi(value_for(argv[i]));
            printf("MAX_ITER set to %d\n", g_max_iter);
        }
        else if (key_matches(argv[i], "MAX_ARRAY_SIZE"))
        {
            g_max_array_size = atoi(value_for(argv[i]));
            printf("MAX_ARRAY_SIZE set to %d\n", g_max_array_size);
        }
    }

    if (num_control_as == 0)
    {
        control_as[0] = defaults.control_a;
        num_control_as = 1;
    }
    if (num_control_bs == 0)
    {
        control_bs[0] = defaults.control_b;
        num_control_bs = 1;
    }

    int maxN = 0;
    for (int ni = 0; ni < num_Ns; ++ni)
        if (Ns[ni] > maxN)
            maxN = Ns[ni];

    int max_prod = 0;
    for (int a = 0; a < num_control_as; ++a)
        for (int b = 0; b < num_control_bs; ++b)
            if (control_as[a] * control_bs[b] > max_prod)
                max_prod = control_as[a] * control_bs[b];

    int required_iter = MAX_ITER_DEF;
    if (max_prod > required_iter)
        required_iter = max_prod;
    if (maxN > required_iter)
        required_iter = maxN;
    if (g_max_iter < required_iter)
        g_max_iter = required_iter;

    int required_array_size = MAX_ARRAY_SIZE_DEF;
    if (maxN > required_array_size)
        required_array_size = maxN;
    if (max_prod > required_array_size)
        required_array_size = max_prod;
    if (g_max_array_size < required_array_size)
        g_max_array_size = required_array_size;

    if (g_max_iter > g_max_array_size)
    {
        fprintf(stderr, "ERROR: MAX_ITER (%d) > MAX_ARRAY_SIZE (%d).\n",
                g_max_iter, g_max_array_size);
        return 1;
    }

    generate_delays();

    printf("========== Runtime Configuration ==========\n");
    printf("API           : %s\n", backend_name());
    printf("Methods       : ");
    if (num_methods == 0)
        printf("0-%d (all)\n", backend_num_methods() - 1);
    else
        for (int i = 0; i < num_methods; ++i)
            printf("%d%s", methods[i], (i < num_methods - 1) ? ", " : "\n");

    printf("Delay range   : [%d, %d]\n", g_min_delaylength, g_max_delaylength);
    printf("Array size(s) : ");
    for (int i = 0; i < num_Ns; ++i)
        printf("%d%s", Ns[i], (i < num_Ns - 1) ? ", " : "\n");

    printf("%-15s: ", backend_control_a_name());
    for (int i = 0; i < num_control_as; ++i)
        printf("%d%s", control_as[i], (i < num_control_as - 1) ? ", " : "\n");

    printf("%-15s: ", backend_control_b_name());
    for (int i = 0; i < num_control_bs; ++i)
        printf("%d%s", control_bs[i], (i < num_control_bs - 1) ? ", " : "\n");

    printf("MAX_ITER      : %d  (kernel workload)\n", g_max_iter);
    printf("MAX_ARRAY_SIZE: %d  (mapping/memory space)\n", g_max_array_size);
    printf("DELAY_POINTS  : %d (powers of two)\n", g_num_delay_points);
    printf("NUM_SAMPLES   : %d (per delay)\n", NUM_SAMPLES);
    printf("OUTERREPS     : %d\n", OUTERREPS);
    printf("WARMUP_ITERS  : %d\n", WARMUP_ITERATIONS);
    printf("BENCHMARK_SETS: %d\n", BENCHMARK_SETS);
    printf("BENCHMARK_RUNS: %d\n", BENCHMARK_RUNS);
#ifdef MICROBENCHMARK_DEVICE_TIMING
    printf("DEVICE_TIMING : %s\n", g_device_timing
           ? "same-stream " BACKEND_DEVICE_LABEL " events in a separate pass; host timing windows unchanged"
           : "disabled (DeviceTiming=1 enables a separate profiling pass)");
#endif
    printf("==========================================\n");

    if (!backend_device_available())
        return 1;

    backend_init(argc, argv);

    FILE *raw = fopen(RAW_OUTPUT_FILE, "w");
    if (!raw)
    {
        perror("Failed to open raw data file.");
        return 1;
    }
    fprintf(raw, "method_id,method_name,N,%s,%s,set,run,delaylength,outerreps,exec_time_us\n",
            backend_control_a_name(), backend_control_b_name());
    fclose(raw);

#ifdef MICROBENCHMARK_DEVICE_TIMING
    if (g_device_timing)
    {
        FILE *device_raw = fopen(DEVICE_OUTPUT_FILE, "w");
        if (!device_raw)
        {
            perror("Failed to open " BACKEND_DEVICE_LABEL " device data file.");
            backend_finalise();
            return 1;
        }
        fprintf(device_raw,
                "method_id,method_name,N,block_count,thread_count,set,run,delaylength,outerreps,max_iter,max_array_size,inner_reps,device_kernel_us,measurement_mode,actual_block_count,actual_thread_count,alignment_mode\n");
        fclose(device_raw);
    }
#endif

    printf("\n========== Benchmark Execution ==========\n");
    printf("%-40s", "Method/N");
    for (int i = 0; i < num_Ns; ++i)
        printf("%30d", Ns[i]);
    printf("\n");

    printf("%-40s", "");
    for (int i = 0; i < num_Ns; ++i)
        printf("%30s", "BIC intercept | min mean");
    printf("\n");
    fflush(stdout);

    for (int midx = 0; midx < (num_methods == 0 ? backend_num_methods() : num_methods); ++midx)
    {
        int method = (num_methods == 0 ? midx : methods[midx]);
        if (method < 0 || method >= backend_num_methods())
            continue;

        printf("%-40s", backend_method_name(method));
        fflush(stdout);

        for (int ca = 0; ca < num_control_as; ++ca)
        {
            for (int cb = 0; cb < num_control_bs; ++cb)
            {
                backend_config_t config = {control_as[ca], control_bs[cb]};
                for (int nidx = 0; nidx < num_Ns; ++nidx)
                {
                    int N = Ns[nidx];
                    double *a = (double *)malloc((size_t)g_max_array_size * sizeof(double));
                    if (!a)
                    {
                        fprintf(stderr, "Allocation failed for N=%d\n", N);
                        exit(EXIT_FAILURE);
                    }
                    for (int i = 0; i < g_max_array_size; ++i)
                        a[i] = 0.0;

                    warmup_cache(method, N, &config);

                    for (int set = 0; set < BENCHMARK_SETS; ++set)
                        for (int run = 0; run < BENCHMARK_RUNS; ++run)
                            run_target(method, set, run, a, N, &config);

                    double intercept, intercept_err, minval, min_err;
                    compute_offloading_time(&intercept, &intercept_err, &minval, &min_err,
                                            method, backend_method_name(method), N);

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
#ifdef MICROBENCHMARK_DEVICE_TIMING
    if (g_device_timing)
    {
        printf("\n========== " BACKEND_DEVICE_LABEL " Device Profiling (Separate Pass) ==========\n");
        printf("Baseline samples are complete. Events bracket each kernel; allocation and copies are excluded.\n");
        fflush(stdout);
        for (int midx = 0; midx < (num_methods == 0 ? backend_num_methods() : num_methods); ++midx)
        {
            int method = num_methods == 0 ? midx : methods[midx];
            if (method < 0 || method >= backend_num_methods())
                continue;
            for (int ca = 0; ca < num_control_as; ++ca)
            {
                for (int cb = 0; cb < num_control_bs; ++cb)
                {
                    backend_config_t config = {control_as[ca], control_bs[cb]};
                    for (int nidx = 0; nidx < num_Ns; ++nidx)
                    {
                        int N = Ns[nidx];
                        double *a = (double *)calloc((size_t)g_max_array_size, sizeof(double));
                        if (!a)
                        {
                            fprintf(stderr, "Device profiling allocation failed for N=%d\n", N);
                            exit(EXIT_FAILURE);
                        }
                        warmup_cache(method, N, &config);
                        for (int set = 0; set < BENCHMARK_SETS; ++set)
                            for (int run = 0; run < BENCHMARK_RUNS; ++run)
                                run_device_target(method, set, run, a, N, &config);
                        printf("Profiled method=%d N=%d requested blocks=%d threads=%d\n",
                               method, N, config.control_a, config.control_b);
                        fflush(stdout);
                        free(a);
                    }
                }
            }
        }
        printf("Device profiling samples saved to %s\n", DEVICE_OUTPUT_FILE);
    }
#endif
    backend_finalise();
    return 0;
}
