#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "common.h"
#define OUTPUT_FILE "overhead_distribution.txt"
#define NUM_SAMPLES 50
#define MIN_DELAYLENGTH 20
#define MAX_DELAYLENGTH 1000
#define INNERREPS 20
#define IDA 27
#define OUTERREPS 40
#define WARMUP_ITERATIONS 10
#define BENCHMARK_SETS 2
#define BENCHMARK_RUNS 2

#define NUM_METHODS 12
#define NUM_SIZES 14

const char *method_names[] = {
    "target map(tofrom: a)",
    "target map(to: a)",
    "target map(from: a)",
    "target map(alloc: a)",
    "target",
    "target te ams",
    "target teams parallel",
    "target teams distribute parallel for",
    "target nowait",
    "target map(to: a[0:N])",
    "target map(tofrom: a[0:N])",
    "num_teams(8) thread_limit(256)"
};

double delays[NUM_SAMPLES];
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES];

void compute_offloading_time(double *intercept_avg, double *error);
void warmup_cache();
void device_target(int offloading_method, int set, int run, double *a, int N, int thread_count, int team_count);

int main(int argc, char **argv) {
    int num_methods = 0;
    int Ns[NUM_SIZES], thread_counts[NUM_SIZES], team_counts[NUM_SIZES], methods[NUM_METHODS];
    int num_Ns = 0, num_threads = 0, num_teams = 0;
    // 默认值
    Ns[0] = 1024;
    num_Ns = 1;
    thread_counts[0] = 32;
    num_threads = 1;
    team_counts[0] = 64 * omp_get_num_devices();
    num_teams = 1;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "Method=", 7) == 0) {
        num_methods = 0;
        char *token = strtok(argv[i] + 7, ",");
        while (token != NULL && num_methods < NUM_METHODS) {
            methods[num_methods++] = atoi(token);
            token = strtok(NULL, ",");
        }
    } else if (strncmp(argv[i], "N=", 2) == 0) {
        num_Ns = 0;
        char *token = strtok(argv[i] + 2, ",");
        while (token != NULL && num_Ns < NUM_SIZES) {
            Ns[num_Ns++] = atoi(token);
            token = strtok(NULL, ",");
        }
    } else if (strncmp(argv[i], "thread_count=", 13) == 0) {
        num_threads = 0;
        char *token = strtok(argv[i] + 13, ",");
        while (token != NULL && num_threads < NUM_SIZES) {
            thread_counts[num_threads++] = atoi(token);
            token = strtok(NULL, ",");
        }
    } else if (strncmp(argv[i], "team_count=", 11) == 0) {
        num_teams = 0;
        char *token = strtok(argv[i] + 11, ",");
        while (token != NULL && num_teams < NUM_SIZES) {
            team_counts[num_teams++] = atoi(token);
            token = strtok(NULL, ",");
        }
    }
}


    // 检查设备是否支持 target
    int targetdev = -9999;
#pragma omp target map(from : targetdev)
    targetdev = omp_is_initial_device();
    if (targetdev) {
        printf("Target region executed on host. Terminating...\n");
        return 0;
    }

    init(argc, argv);

    printf("Method/N");
    for (int i = 0; i < num_Ns; ++i) {
        printf("\t%d", Ns[i]);
    }
    printf("\n");

    for (int midx = 0; midx < (num_methods == 0 ? NUM_METHODS : num_methods); ++midx) {
        int m = (num_methods == 0 ? midx : methods[midx]) - 1;
    
        for (int t = 0; t < num_threads; ++t) {
            for (int tm = 0; tm < num_teams; ++tm) {
                printf("%s [threads=%d teams=%d]\n", method_names[m], thread_counts[t], team_counts[tm]);
                for (int nidx = 0; nidx < num_Ns; ++nidx) {
                    int N = Ns[nidx];
                    double *a = (double *)malloc(N * sizeof(double));
                    if (!a) {
                        fprintf(stderr, "Allocation failed for N=%d\n", N);
                        exit(EXIT_FAILURE);
                    }
    
                    for (int set = 0; set < BENCHMARK_SETS; ++set) {
                        //warmup
                        device_target(m + 1, set, -1, a, N, thread_counts[t], team_counts[tm]);
                        for (int run = 0; run < BENCHMARK_RUNS; ++run) {
                            device_target(m + 1, set, run, a, N, thread_counts[t], team_counts[tm]);
                        }
                    }
    
                    double avg, err;
                    compute_offloading_time(&avg, &err);
                    printf("\t%.1f±%.1f", avg, err);
                    free(a);
                }
                printf("\n"); // 每组 thread-team 输出完换行
            }
        }
    }
    

    finalise();
    return 0;
}


void device_target(int offloading_method, int set, int run, double *a, int N, int thread_count, int team_count){

    for (int i = 0; i < NUM_SAMPLES; i++) {
        int delaylength = MIN_DELAYLENGTH + i * (MAX_DELAYLENGTH - MIN_DELAYLENGTH) / (NUM_SAMPLES - 1);
        delays[i] = delaylength;

        double start = omp_get_wtime();
        for (int j = 0; j < INNERREPS; j++) {
            switch (offloading_method) {
                case 1:
#pragma omp target map(tofrom : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 2:
#pragma omp target map(to : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 3:
#pragma omp target map(from : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 4:
#pragma omp target map(alloc : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 5:
#pragma omp target map(tofrom : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 6:
#pragma omp target teams map(tofrom : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 7:
// #pragma omp target teams map(tofrom : a[0 : N])
// #pragma omp parallel num_threads(thread_count)
//                     array_delay(delaylength, a);
#pragma omp target teams distribute parallel for num_teams(team_count) thread_limit(thread_count) map(tofrom : a[0:N])
                for (int i = 0; i < N; i++) {
                    array_delay(delaylength, &a[i]);
                }          
                    
                    break;
                case 8:
#pragma omp target teams distribute parallel for num_teams(team_count) thread_limit(thread_count) map(tofrom : a[0 : N])
                    for (int k = 0; k < team_count; k++)
                    {
                        array_delay(delaylength, a);
                    }
                    break;
                case 9:
#pragma omp target map(tofrom : a[0 : N]) nowait
                    array_delay(delaylength, a);
#pragma omp taskwait
                    break;
                case 10:
#pragma omp target map(to : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 11:
#pragma omp target map(tofrom : a[0 : N])
                    array_delay(delaylength, a);
                    break;
                case 12:
#pragma omp target teams map(tofrom : a[0 : N]) num_teams(8) thread_limit(256)
                    array_delay(delaylength, a);
                    break;
                default:
                    array_delay(delaylength, a);
                    break;
            }
            a[0] += 1;
            if (a[0] < 0)
            {
                printf("%f \n", a[0]);
            }
        }
        execution_times[set][run][i] = (omp_get_wtime() - start) * 1.0e6 / INNERREPS;
    }
}

// Compute averaged intercept with error bar
void compute_offloading_time(double *intercept_avg, double *error)
{
    FILE *file = fopen(OUTPUT_FILE, "w");
    if (!file)
    {
        perror("Error opening output file");
        return;
    }

    double intercepts[BENCHMARK_SETS * BENCHMARK_RUNS];
    double sum_intercept = 0, sum_sq_intercept = 0;
    int total_runs = BENCHMARK_SETS * BENCHMARK_RUNS;

    int idx = 0;
    for (int set = 0; set < BENCHMARK_SETS; set++)
    {
        for (int run = 0; run < BENCHMARK_RUNS; run++)
        {
            double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
            for (int i = 0; i < NUM_SAMPLES; i++)
            {
                sum_x += delays[i];
                sum_y += execution_times[set][run][i];
                sum_xx += delays[i] * delays[i];
                sum_xy += delays[i] * execution_times[set][run][i];
            }
            double slope = (NUM_SAMPLES * sum_xy - sum_x * sum_y) / (NUM_SAMPLES * sum_xx - sum_x * sum_x);
            double intercept = (sum_y - slope * sum_x) / NUM_SAMPLES;
            intercepts[idx++] = intercept;
            sum_intercept += intercept;

#ifdef PRINT_DISTRIBUTION
            // printf("Set %d, Run %d: Offloading Overhead = %f μs\n", set, run, intercept);
            fprintf(file, "%d %d %f\n", set, run, intercept);
#endif
        }
    }
    *intercept_avg = sum_intercept / total_runs;

    for (int i = 0; i < total_runs; i++)
    {
        sum_sq_intercept += (intercepts[i] - *intercept_avg) * (intercepts[i] - *intercept_avg);
    }
    *error = sqrt(sum_sq_intercept / (total_runs - 1));
    fclose(file);
}
