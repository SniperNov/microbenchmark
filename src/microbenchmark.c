#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

#define OUTPUT_FILE "overhead_distribution.txt"
#define RAW_OUTPUT_FILE "raw_times.csv"
#define N_DEF 16382
#define NUM_SAMPLES 20         // If all methods outputs same value, we should enlarge the interval of [MIN_D,MAX_D]
#define MIN_DELAYLENGTH 1    // Too small introduces noise.
#define MAX_DELAYLENGTH 8096 // Log scale sampling across magnitudes, cover launches and execution.
#define INNERREPS 20
#define MAX_ITER_DEF 6656 // total mapping and iteration space (equivalent to MAX_ARRAY_SIZE)
#define MAX_ARRAY_SIZE_DEF 65536

#define OUTERREPS 5
#define WARMUP_ITERATIONS 10
#define BENCHMARK_SETS 2
#define BENCHMARK_RUNS 2
#define NUM_METHODS 11
#define NUM_SIZES 16

static int g_max_iter = MAX_ITER_DEF;                            // fixed kernel workload
static int g_max_array_size = MAX_ARRAY_SIZE_DEF; // mapping / memory size
static int g_min_delaylength = MIN_DELAYLENGTH;
static int g_max_delaylength = MAX_DELAYLENGTH;

const char *method_names[] = {
    "map(tofrom: a)",
    "map(to: a)",
    "map(from: a)",
    "map(alloc: a)",
    "teams (scalar)",
    "teams distribute parallel for",
    "nowait",
    "teams atomic",
    "teams reduction",
    "teams parallel",
    "teams + parallel inside"};

double delays[NUM_SAMPLES];
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES][OUTERREPS];

void delay_kernel(int delay, double *ptr)
{
    array_delay(delay, ptr);
}
// void compute_offloading_time(double *intercept_avg, double *slope_avg, double *error);
void compute_offloading_time(double *intercept_avg, double *intercept_err,
    double *min_avg, double *min_err,
    int method_id, const char *method_name, int N);
void warmup_cache(int N, int thread_count, int team_count);
void device_target(int offloading_method, int set, int run, double *a, int N, int thread_count, int team_count);
static void shuffle_indices_local(int *perm, int n, unsigned int seed);

int main(int argc, char **argv)
{
    int num_methods = 0;
    int Ns[NUM_SIZES], thread_counts[NUM_SIZES], team_counts[NUM_SIZES], methods[NUM_METHODS];
    int num_Ns = 0, num_threads = 0, num_teams = 0;
    // 默认值
    Ns[0] = N_DEF;
    num_Ns = 1;
    thread_counts[0] = 32;
    num_threads = 1;
    team_counts[0] = 4 * omp_get_num_devices();
    num_teams = 1;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i)
    {

        // 解析 Method=
        if (strncasecmp(argv[i], "Method=", 7) == 0)
        {
            num_methods = 0;
            char *token = strtok(argv[i] + 7, ",");
            while (token != NULL && num_methods < NUM_METHODS)
            {
                methods[num_methods++] = atoi(token);
                token = strtok(NULL, ",");
            }

            // 解析 Delay=
        }
        else if (strncasecmp(argv[i], "Delay=", 6) == 0)
        {
            int delays[4] = {0};
            int num_delays = 0;
            char *token = strtok(argv[i] + 6, ",");
            while (token != NULL && num_delays < 4)
            {
                delays[num_delays++] = atoi(token);
                token = strtok(NULL, ",");
            }

            if (num_delays == 0)
            {
                g_min_delaylength = MIN_DELAYLENGTH;
                g_max_delaylength = MAX_DELAYLENGTH;
                // printf("Delaylength: [%d, %d]\n", g_min_delaylength, g_max_delaylength);
            }
            else if (num_delays == 1)
            {
                g_min_delaylength = MIN_DELAYLENGTH;
                g_max_delaylength = delays[0];
                // printf("Delaylength: MIN=%d (default), MAX=%d\n", g_min_delaylength, g_max_delaylength);
            }
            else
            {
                int a = delays[0], b = delays[1];
                if (a > b)
                {
                    int tmp = a;
                    a = b;
                    b = tmp;
                }
                g_min_delaylength = a;
                g_max_delaylength = b;
                if (num_delays > 2)
                {
                    printf("Warning: extra Delay values ignored.\n");
                }
                // printf("Delaylength: MIN=%d, MAX=%d\n", g_min_delaylength, g_max_delaylength);
            }

            // 解析 N=
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
                printf("No N specified. Using default N=16384\n");
            }

            for (int ni = 0; ni < num_Ns; ++ni)
            {
                if (Ns[ni] > g_max_delaylength)
                {
                    printf("Warning: N=%d exceeds current delay upper bound (%d), resetting to 16384\n",
                           Ns[ni], g_max_delaylength);
                    Ns[ni] = N_DEF;
                }
            }

            // 解析 thread_count=
        }
        else if (strncasecmp(argv[i], "thread_count=", 13) == 0)
        {
            num_threads = 0;
            char *token = strtok(argv[i] + 13, ",");
            while (token != NULL && num_threads < NUM_SIZES)
            {
                thread_counts[num_threads++] = atoi(token);
                token = strtok(NULL, ",");
            }

            // 解析 team_count=
        }
        else if (strncasecmp(argv[i], "team_count=", 11) == 0)
        {
            num_teams = 0;
            char *token = strtok(argv[i] + 11, ",");
            while (token != NULL && num_teams < NUM_SIZES)
            {
                team_counts[num_teams++] = atoi(token);
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
        fprintf(stderr, "Warning: MAX_ARRAY_SIZE (%d) < max(N) (%d). "
                        "Adjusting to %d to avoid out-of-bounds mapping.\n",
                g_max_array_size, maxN, maxN);
        g_max_array_size = maxN;
    }

    printf("========== Runtime Configuration ==========\n");
    printf("Delay range   : [%d, %d]\n", g_min_delaylength, g_max_delaylength);
    printf("Array size(s) : ");
    for (int ni = 0; ni < num_Ns; ++ni)
        printf("%d%s", Ns[ni], (ni < num_Ns - 1) ? ", " : "\n");
    printf("Threads/Teams : %d / %d\n", thread_counts[0], team_counts[0]);
    printf("MAX_ITER      : %d  (kernel workload)\n", g_max_iter);
    printf("MAX_ARRAY_SIZE: %d  (mapping/memory space)\n", g_max_array_size);
    printf("NUM_SAMPLES   : %d\n", NUM_SAMPLES);
    printf("==========================================\n");

    // ---- Check target device ----
    int numdevs = omp_get_num_devices();
    int hostdev = omp_get_initial_device();

    int targetdev = -9999;
#pragma omp target map(from : targetdev)
    {
        targetdev = omp_is_initial_device();
    }
    printf("There are  %d available devices\n", numdevs);
    printf("Host device is %d\n", hostdev);

    if (targetdev)
    {
        printf("Target region executed on host. Terminating...\n");
        fflush(stdout);
        return 0;
    }

    init(argc, argv);

    // New or rewrite the csv recording all the raw execution_times
    FILE *raw = fopen(RAW_OUTPUT_FILE, "w");
    if (raw)
    {
        fprintf(raw, "method_id,method_name,N,thread_count,team_count,set,run,delaylength,outerreps,exec_time_us\n");
        fclose(raw);
    }
    else
    {
        perror("Failed to open raw data file.");
        return 1;
    }
    // --- Adjust MAX_ITER once (before benchmarking) ---
    int required_iter = MAX_ITER_DEF;

    // cover max threads*teams
    int max_prod = 0;
    for (int t = 0; t < num_threads; ++t)
    {
        for (int tm = 0; tm < num_teams; ++tm)
        {
            int prod = thread_counts[t] * team_counts[tm];
            if (prod > max_prod)
                max_prod = prod;
        }
    }
    if (max_prod > required_iter)
        required_iter = max_prod;

    // also cover max N (reuse maxN computed above)
    if (maxN > required_iter)
        required_iter = maxN;

    g_max_iter = required_iter;

    printf("MAX_ITER set to %d (MAX_ITER_DEF=%d, max threads*teams=%d, max N=%d)\n",
           g_max_iter, MAX_ITER_DEF, max_prod, maxN);

    // safety: ensure mapping space is large enough
    if (g_max_iter > g_max_array_size)
    {
        printf("ERROR: MAX_ITER (%d) > MAX_ARRAY_SIZE (%d). Increase MAX_ARRAY_SIZE.\n",
               g_max_iter, g_max_array_size);
        exit(1);
    }
    // --- end adjust ---

    // ---- Print table header ----
    printf("\n========== Benchmark Execution ==========\n");
    printf("%-35s", "Method/N");
    for (int i = 0; i < num_Ns; ++i)
    {
        printf("%30d", Ns[i]);
    }
    printf("\n");
    
    printf("%-35s", "");
    for (int i = 0; i < num_Ns; ++i)
    {
        printf("%30s", "BIC intercept | lowest");
    }
    printf("\n");
    fflush(stdout);

    // ---- Print method results ----
    for (int midx = 0; midx < (num_methods == 0 ? NUM_METHODS : num_methods); ++midx)
    {
        int m = (num_methods == 0 ? midx : methods[midx] - 1);
        if (m < 0 || m >= NUM_METHODS)
            continue;

        printf("%-35s", method_names[m]);
        fflush(stdout);

        for (int t = 0; t < num_threads; ++t)
        {
            for (int tm = 0; tm < num_teams; ++tm)
            {
                for (int nidx = 0; nidx < num_Ns; ++nidx)
                {
                    int N = Ns[nidx];
                    // double *a = (double *)malloc(N * sizeof(double));
                    double *a = (double *)malloc(g_max_array_size * sizeof(double));
                    if (!a)
                    {
                        fprintf(stderr, "Allocation failed for N=%d\n", N);
                        exit(EXIT_FAILURE);
                    }

                    warmup_cache(Ns[num_Ns - 1], thread_counts[num_threads - 1], team_counts[num_teams - 1]);

                    for (int set = 0; set < BENCHMARK_SETS; ++set)
                    {
                        device_target(m + 1, set, -1, a, N, thread_counts[t], team_counts[tm]);
                        for (int run = 0; run < BENCHMARK_RUNS; ++run)
                            device_target(m + 1, set, run, a, N, thread_counts[t], team_counts[tm]);
                    }

                    double intercept, intercept_err;
                    double minval, min_err;

                    compute_offloading_time(&intercept, &intercept_err, &minval, &min_err,
                            m + 1, method_names[m], N);

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

    finalise();

    return 0;
}

void shuffle_indices_local(int *perm, int n, unsigned int seed)
{   
    if(!seed) seed = 12345u;
    for (int i = 0; i < n; ++i)
        perm[i] = i;
    for (int i = n - 1; i > 0; --i)
    {
        seed = 1664525u * seed + 1013904223u; // LCG 更新 seed（不用 rand_r 也行）
        int j = (int)(seed % (unsigned)(i + 1));
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}


void device_target(int method, int set, int run, double *a, int N, int thread_count, int team_count)
{
    
    // generate log-scale sampled delaylengths once (monotonic increasing)
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
    int perm[NUM_SAMPLES];
    unsigned int seed = 12345u + 97u * (unsigned int)set + 1009u * (unsigned int)(run + 1);
    shuffle_indices_local(perm, NUM_SAMPLES, seed);    double start, end, delay;
    for (int sidx = 0; sidx < NUM_SAMPLES; sidx++)
    {
        int logical = perm[sidx]; // shuffled index
        delay = delays[logical];
        for (int orep = 0; orep < OUTERREPS; orep++)
        {
            if (method >= 1 && method <= 4)
            {
                start = omp_get_wtime();
                for (int irep = 0; irep < INNERREPS; irep++)
                {
                    switch (method)
                    {
                    case 1:
#pragma omp target map(tofrom : a[0 : N])
                        delay_kernel(delay, a);
                        break;
                    case 2:
#pragma omp target map(to : a[0 : N])
                        delay_kernel(delay, a);
                        break;
                    case 3:
#pragma omp target map(from : a[0 : N])
                        delay_kernel(delay, a);
                        break;
                    case 4:
#pragma omp target map(alloc : a[0 : N])
                        delay_kernel(delay, a);
                        break;
                        /* 对标保持一致性 */
                    }
                    a[0] += 1;
                    if (a[0] < 0)
                    {
                        printf("%f \n", a[0]);
                    }
                }
                end = omp_get_wtime();
            }
            else if (method >= 5 && method <= 11)
            {
                double *tmp = (double *)malloc(N * sizeof(double));

                if (!tmp)
                {
                    fprintf(stderr, "Allocation failed for tmp[N=%d]\n", N);
                    exit(EXIT_FAILURE);
                }
                for (int i = 0; i < N; i++)
                    tmp[i] = 0.0;

#pragma omp target data map(tofrom : a[0 : g_max_array_size], tmp[0 : N])
                {
                    start = omp_get_wtime();
                    for (int rep = 0; rep < INNERREPS; rep++)
                    {
                        switch (method)
                        {
                        case 5:
#pragma omp target teams
                    {
                        int team_id = omp_get_team_num();
                        delay_kernel(delay, &a[team_id % N]);
                    }
                    break;

                    case 6: // teams_id * threads_id
#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for num_teams(team_count) thread_limit(thread_count)
                    for (int i = 0; i < g_max_iter; i++)
                        delay_kernel(delay, &a[i % N]);
                    
                    break;
#elif defined(_CRAYC) || defined(__AMD__) || defined(__clang__)
#pragma omp target teams distribute parallel for num_teams(team_count) thread_limit(thread_count)
                        {
                            for (int i = 0; i < g_max_iter; i++)
                                delay_kernel(delay, &a[i % N]);
                        }
                        break;
#endif
                    case 7:
#pragma omp target nowait
                            delay_kernel(delay, a);
#pragma omp taskwait
                            break;
                        case 8:
#pragma omp target teams distribute parallel for
                            for (int i = 0; i < g_max_iter; i++)
                            {
                                delay_kernel(delay, &a[i]);
#pragma omp atomic
                            tmp[i % N] += 1.0;
                        }
                        break;
                    case 9:
#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for reduction(+:tmp[0:N])
                        for (int i = 0; i < g_max_iter; i++)
                        {
                            delay_kernel(delay, &a[i]);
                            tmp[i % N] += 1.0;
                        }
                        break;

#elif defined(_CRAYC) || defined(__AMD__) || defined(__clang__)

#pragma omp target
#pragma teams distribute reduction(+ : tmp[0 : N])
#pragma parallel for reduction(+ : tmp[0 : N])

                        for (int i = 0; i < g_max_iter; i++)
                        {
                            delay_kernel(delay, &a[i]);
                            tmp[i % N] += 1.0;
                        }
                        break;
#endif
                    case 10:
#pragma omp target teams
                        {
#pragma omp parallel
                            delay_kernel(delay, a);
                        }
                        break;
                        case 11:
#pragma omp target teams
                        {
                            for (int r = 0; r < INNERREPS; r++)
                            {
#pragma omp parallel
                                {
                                    delay_kernel(delay, a);
                                }
                            }
                        }
                        break;
                        }
                        /* 对标保持一致性 */
                        a[0] += 1.0;
                        if (a[0] < 0.0)
                        {
                            printf("%f \n", a[0]);
                            fflush(stdout);
                        }
                    }
                    end = omp_get_wtime();
                }
                free(tmp);
            }

            // execution_times[set][run][sidx] = (end - start) * 1.0e6 / INNERREPS;
            double elapsed_us = (end - start) * 1.0e6 / INNERREPS;
            //--
            if (run >= 0)
            {
                execution_times[set][run][logical][orep] = elapsed_us;
                FILE *raw = fopen(RAW_OUTPUT_FILE, "a");
                if (raw)
                {
                    const char *mname = (method >= 1 && method <= NUM_METHODS) ? method_names[method - 1] : "UNKNOWN";
                    fprintf(raw, "%d,\"%s\",%d,%d,%d,%d,%d,%.0f,%d,%.9f\n", method, mname, N, thread_count, team_count, set, run, delays[logical], orep, elapsed_us);
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
    fprintf(file, "\n[Method=%d %s N=%d]\n", method_id, method_name, N);
#endif

    int total_runs = BENCHMARK_SETS * BENCHMARK_RUNS;
    double intercepts[total_runs];
    double mins[total_runs];

    double sum_intercept = 0.0, sumsq_intercept = 0.0;
    double sum_min = 0.0, sumsq_min = 0.0;
    int idx = 0;

    for (int set = 0; set < BENCHMARK_SETS; ++set)
    {
        for (int run = 0; run < BENCHMARK_RUNS; ++run)
        {
            double avg_y[NUM_SAMPLES];

            for (int i = 0; i < NUM_SAMPLES; ++i)
            {
                double sum = 0.0;
                for (int orep = 0; orep < OUTERREPS; ++orep)
                    sum += execution_times[set][run][i][orep];
                avg_y[i] = sum / OUTERREPS;
            }

            double run_min = avg_y[0];
            double run_min_delay = delays[0];
            for (int i = 1; i < NUM_SAMPLES; ++i)
            {
                if (avg_y[i] < run_min)
                {
                    run_min = avg_y[i];
                    run_min_delay = delays[i];
                }
                else if (avg_y[i] >= 2.0 * run_min)
                {
                    break;
                }
            }

            double best_BIC = INFINITY;
            int best_k = 0;
            double best_a = 0.0, best_b = 0.0, best_R2 = 0.0;

            const double MIN_KEEP_RATIO = 0.5;
            int min_keep;
            if (NUM_SAMPLES <= 10)
                min_keep = 5;
            else
                min_keep = (int)ceil(NUM_SAMPLES * MIN_KEEP_RATIO);

            for (int k = 0; k <= NUM_SAMPLES - min_keep; ++k)
            {
                int n = NUM_SAMPLES - k;
                double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

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

                double b = (n * sum_xy - sum_x * sum_y) / denom;
                double a = (sum_y - b * sum_x) / n;
                double rss = 0.0, tss = 0.0;
                double mean_y = sum_y / n;

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

    *intercept_avg = sum_intercept / idx;
    *min_avg = sum_min / idx;

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


void warmup_cache(int N, int thread_count, int team_count)
{
    double *a = (double *)malloc(g_max_array_size * sizeof(double));
    if (!a)
    {
        perror("warmup malloc");
        exit(EXIT_FAILURE);
    }

    int dummy_set = 0;
    int dummy_run = -1;

    device_target(1, dummy_set, dummy_run, a, N, thread_count, team_count);

    free(a);
}
