#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <openacc.h>
#include "common.h"

#define OUTPUT_FILE "overhead_distribution.txt"

#define N_DEF 16382
#define NUM_SAMPLES 16
#define MIN_DELAYLENGTH 512
#define MAX_DELAYLENGTH 262144
#define INNERREPS 20
#define MAX_ITER_DEF 6656
#define MAX_ARRAY_SIZE_DEF 65536

#define OUTERREPS 40
#define WARMUP_ITERATIONS 10
#define BENCHMARK_SETS 2
#define BENCHMARK_RUNS 2

#define NUM_METHODS 10
#define NUM_SIZES 16

static int g_max_iter = MAX_ITER_DEF;
static int g_max_array_size = MAX_ARRAY_SIZE_DEF;
static int g_min_delaylength = MIN_DELAYLENGTH;
static int g_max_delaylength = MAX_DELAYLENGTH;

const char *method_names[] = {
    "parallel loop copy(a[0:N])",
    "parallel loop copyin(a[0:N])",
    "parallel loop copyout(a[0:N])",
    "parallel loop create(a[0:N])",
    "parallel loop present(a[0:g_max_array_size])",
    "parallel loop gang present(a[0:g_max_array_size])",
    "parallel loop gang vector present(a[0:g_max_array_size])",
    "parallel loop num_gangs(G) present(a[0:g_max_array_size])",
    "parallel loop async(1) present(a[0:g_max_array_size]) + wait(1)",
    "parallel loop num_gangs(G) vector_length(V) present(a[0:g_max_array_size])"
};

double delays[NUM_SAMPLES];
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES];

void delay_kernel(int delay, double *ptr);
void compute_offloading_time(double *intercept_avg, double *error,
                             int method_id, const char *method_name, int N);
void warmup_cache(int N, int gang_count, int vector_length);
void device_target(int method, int set, int run, double *a, int N,
                   int gang_count, int vector_length);

void delay_kernel(int delay, double *ptr)
{
    array_delay(delay, ptr);
}

int main(int argc, char **argv)
{
    int num_methods = 0;
    int Ns[NUM_SIZES], gang_counts[NUM_SIZES], vector_lengths[NUM_SIZES], methods[NUM_METHODS];
    int num_Ns = 0, num_gangs = 0, num_vectors = 0;

    Ns[0] = N_DEF;
    num_Ns = 1;

    gang_counts[0] = 64;
    num_gangs = 1;

    vector_lengths[0] = 128;
    num_vectors = 1;

    for (int i = 1; i < argc; ++i)
    {
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
        else if (strncasecmp(argv[i], "Delay=", 6) == 0)
        {
            int tmp_delays[4] = {0};
            int num_delays = 0;
            char *token = strtok(argv[i] + 6, ",");
            while (token != NULL && num_delays < 4)
            {
                tmp_delays[num_delays++] = atoi(token);
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
                g_max_delaylength = tmp_delays[0];
            }
            else
            {
                int a = tmp_delays[0], b = tmp_delays[1];
                if (a > b)
                {
                    int t = a; a = b; b = t;
                }
                g_min_delaylength = a;
                g_max_delaylength = b;
            }
        }
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
            }
        }
        else if (strncasecmp(argv[i], "gang_count=", 11) == 0)
        {
            num_gangs = 0;
            char *token = strtok(argv[i] + 11, ",");
            while (token != NULL && num_gangs < NUM_SIZES)
            {
                gang_counts[num_gangs++] = atoi(token);
                token = strtok(NULL, ",");
            }
        }
        else if (strncasecmp(argv[i], "vector_length=", 14) == 0)
        {
            num_vectors = 0;
            char *token = strtok(argv[i] + 14, ",");
            while (token != NULL && num_vectors < NUM_SIZES)
            {
                vector_lengths[num_vectors++] = atoi(token);
                token = strtok(NULL, ",");
            }
        }
        else if (strncasecmp(argv[i], "MAX_ITER=", 9) == 0)
        {
            g_max_iter = atoi(argv[i] + 9);
            printf("MAX_ITER set to %d\n", g_max_iter);
        }
        else if (strncasecmp(argv[i], "MAX_ARRAY_SIZE=", 15) == 0)
        {
            g_max_array_size = atoi(argv[i] + 15);
            printf("MAX_ARRAY_SIZE set to %d\n", g_max_array_size);
        }
    }

    int maxN = 0;
    for (int ni = 0; ni < num_Ns; ++ni)
        if (Ns[ni] > maxN)
            maxN = Ns[ni];

    if (g_max_array_size < maxN)
    {
        fprintf(stderr,
                "Warning: MAX_ARRAY_SIZE (%d) < max(N) (%d). Adjusting to %d.\n",
                g_max_array_size, maxN, maxN);
        g_max_array_size = maxN;
    }

    int devtype = acc_get_device_type();
    int numdevs = acc_get_num_devices(devtype);

    printf("========== Runtime Configuration ==========\n");
    printf("OpenACC device type: %d\n", devtype);
    printf("Number of available devices: %d\n", numdevs);
    printf("Delay range   : [%d, %d]\n", g_min_delaylength, g_max_delaylength);

    printf("Array size(s) : ");
    for (int ni = 0; ni < num_Ns; ++ni)
        printf("%d%s", Ns[ni], (ni < num_Ns - 1) ? ", " : "\n");

    printf("Gangs/Vectors : %d / %d\n", gang_counts[0], vector_lengths[0]);
    printf("MAX_ITER      : %d  (kernel workload)\n", g_max_iter);
    printf("MAX_ARRAY_SIZE: %d  (mapping/memory space)\n", g_max_array_size);
    printf("NUM_SAMPLES   : %d\n", NUM_SAMPLES);
    printf("==========================================\n");

    if (numdevs <= 0)
    {
        fprintf(stderr, "No OpenACC device available. Terminating.\n");
        return 1;
    }

    init(argc, argv);

    printf("\n========== Benchmark Execution ==========\n");
    printf("%-48s", "Method/N");
    for (int i = 0; i < num_Ns; ++i)
        printf("%20d", Ns[i]);
    printf("\n");

    for (int midx = 0; midx < (num_methods == 0 ? NUM_METHODS : num_methods); ++midx)
    {
        int m = (num_methods == 0 ? midx : methods[midx] - 1);
        if (m < 0 || m >= NUM_METHODS)
            continue;

        printf("%-48s", method_names[m]);

        for (int g = 0; g < num_gangs; ++g)
        {
            for (int v = 0; v < num_vectors; ++v)
            {
                for (int nidx = 0; nidx < num_Ns; ++nidx)
                {
                    int N = Ns[nidx];

                    if (gang_counts[g] * vector_lengths[v] > g_max_iter)
                    {
                        printf("%20s", "NA");
                        continue;
                    }

                    double *a = (double *)malloc(g_max_array_size * sizeof(double));
                    if (!a)
                    {
                        fprintf(stderr, "Allocation failed for N=%d\n", N);
                        exit(EXIT_FAILURE);
                    }

                    for (int i = 0; i < g_max_array_size; ++i)
                        a[i] = 0.0;

                    warmup_cache(Ns[num_Ns - 1], gang_counts[num_gangs - 1], vector_lengths[num_vectors - 1]);

                    for (int set = 0; set < BENCHMARK_SETS; ++set)
                    {
                        device_target(m + 1, set, -1, a, N, gang_counts[g], vector_lengths[v]);
                        for (int run = 0; run < BENCHMARK_RUNS; ++run)
                            device_target(m + 1, set, run, a, N, gang_counts[g], vector_lengths[v]);
                    }

                    double intercept, err;
                    compute_offloading_time(&intercept, &err, m + 1, method_names[m], N);
                    printf("%20.6f±%-10.6f", intercept, err);

                    free(a);
                }
            }
        }
        printf("\n");
    }

    printf("==========================================\n");

    finalise();
    return 0;
}

void device_target(int method, int set, int run, double *a, int N,
                   int gang_count, int vector_length)
{
    double start = 0.0, end = 0.0;

    for (int sidx = 0; sidx < NUM_SAMPLES; ++sidx)
    {
        double delay;
        if (sidx == 0)
        {
            delay = g_min_delaylength;
        }
        else if (sidx == NUM_SAMPLES - 1)
        {
            delay = g_max_delaylength;
        }
        else
        {
            double log_min = log2((double)g_min_delaylength);
            double log_max = log2((double)g_max_delaylength);
            double ratio = (double)sidx / (NUM_SAMPLES - 1);
            double log_val = log_min + (log_max - log_min) * ratio;
            delay = pow(2.0, log_val);
        }

        delays[sidx] = delay;

        if (method >= 1 && method <= 4)
        {
            start = get_time_usec();

            for (int rep = 0; rep < INNERREPS; ++rep)
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
                    printf("%f\n", a[0]);
            }

            end = get_time_usec();
        }
        else
        {
#pragma acc data copy(a[0:g_max_array_size])
            {
                start = get_time_usec();

                for (int rep = 0; rep < INNERREPS; ++rep)
                {
                    switch (method)
                    {
                        case 5:
#pragma acc parallel loop present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
                            break;

                        case 6:
#pragma acc parallel loop gang present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
                            break;

                        case 7:
#pragma acc parallel loop gang vector present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
                            break;

                        case 8:
#pragma acc parallel loop num_gangs(gang_count) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
                            break;

                        case 9:
#pragma acc parallel loop async(1) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
#pragma acc wait(1)
                            break;

                        case 10:
#pragma acc parallel loop num_gangs(gang_count) vector_length(vector_length) present(a[0:g_max_array_size])
                            for (int i = 0; i < g_max_iter; ++i)
                                delay_kernel((int)delay, &a[i % N]);
                            break;
                    }

                    a[0] += 1.0;
                    if (a[0] < 0.0)
                        printf("%f\n", a[0]);
                }

                end = get_time_usec();
            }
        }

        if (run >= 0)
            execution_times[set][run][sidx] = (end - start) / INNERREPS;
    }
}

void compute_offloading_time(double *intercept_avg, double *error,
                             int method_id, const char *method_name, int N)
{
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
    double sum_intercept = 0.0, sum_sq = 0.0;
    int idx = 0;

    for (int set = 0; set < BENCHMARK_SETS; ++set)
    {
        for (int run = 0; run < BENCHMARK_RUNS; ++run)
        {
            double best_BIC = INFINITY;
            int best_k = 0;
            double best_a = 0.0, best_b = 0.0, best_R2 = 0.0;

            for (int k = 0; k < NUM_SAMPLES - 5; ++k)
            {
                int n = NUM_SAMPLES - k;
                double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

                for (int i = k; i < NUM_SAMPLES; ++i)
                {
                    sum_x += delays[i];
                    sum_y += execution_times[set][run][i];
                    sum_xx += delays[i] * delays[i];
                    sum_xy += delays[i] * execution_times[set][run][i];
                }

                double denom = n * sum_xx - sum_x * sum_x;
                if (denom <= 0.0)
                    continue;

                double b = (n * sum_xy - sum_x * sum_y) / denom;
                double a = (sum_y - b * sum_x) / n;

                double rss = 0.0, tss = 0.0, mean_y = sum_y / n;
                for (int i = k; i < NUM_SAMPLES; ++i)
                {
                    double yhat = a + b * delays[i];
                    double e = execution_times[set][run][i] - yhat;
                    rss += e * e;
                    tss += (execution_times[set][run][i] - mean_y) *
                           (execution_times[set][run][i] - mean_y);
                }

                if (rss <= 0.0)
                    continue;

                double R2 = 1.0 - rss / tss;
                int p = 2;
                double BIC = n * log(rss / n) + p * log((double)n);

                if (BIC < best_BIC)
                {
                    best_BIC = BIC;
                    best_k = k;
                    best_a = a;
                    best_b = b;
                    best_R2 = R2;
                }
            }

            intercepts[idx++] = best_a;
            sum_intercept += best_a;

#ifdef PRINT_DISTRIBUTION
            fprintf(file,
                    "Set=%d Run=%d  Lmin=%.0f  Intercept=%.6fμs  Slope=%.6f  R2=%.5f  BIC=%.3f\n",
                    set, run, delays[best_k], best_a, best_b, best_R2, best_BIC);
#endif
        }
    }

    *intercept_avg = sum_intercept / idx;

    for (int i = 0; i < idx; ++i)
        sum_sq += (intercepts[i] - *intercept_avg) * (intercepts[i] - *intercept_avg);

    *error = sqrt(sum_sq / (idx - 1));

#ifdef PRINT_DISTRIBUTION
    fprintf(file, "Average Intercept=%.6f ± %.6f μs\n", *intercept_avg, *error);
#endif

    fclose(file);
}

void warmup_cache(int N, int gang_count, int vector_length)
{
    double *a = (double *)malloc(g_max_array_size * sizeof(double));
    if (!a)
    {
        perror("warmup malloc");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < g_max_array_size; ++i)
        a[i] = 0.0;

    device_target(1, 0, -1, a, N, gang_count, vector_length);

    free(a);
}