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
#define BENCHMARK_SETS 20
#define BENCHMARK_RUNS 20

#define NUM_METHODS 12
#define NUM_SIZES 14
int sizes[NUM_SIZES] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

const char *method_names[] = {
    "target map(tofrom: a)",
    "target map(to: a)",
    "target map(from: a)",
    "target map(alloc: a)",
    "target",
    "target teams",
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
void device_target(int offloading_method, int set, int run, double *a, int N);

int main(int argc, char **argv) {
    int num_devs = omp_get_num_devices();
    int hostdev = omp_get_initial_device();
    int targetdev = -9999;

#pragma omp target map(from : targetdev)
    targetdev = omp_is_initial_device();

    if (targetdev) {
        printf("Target region executed on host. Terminating...\n");
        return 0;
    }

    init(argc, argv);

    printf("Method/N");
    for (int i = 0; i < NUM_SIZES; ++i) {
        printf("\t%d", sizes[i]);
    }
    printf("\n");

    for (int m = 0; m < NUM_METHODS; ++m) {
        printf("%s", method_names[m]);
        for (int nidx = 0; nidx < NUM_SIZES; ++nidx) {
            int N = sizes[nidx];
            double *a = (double *)malloc(N * sizeof(double));
            if (!a) {
                fprintf(stderr, "Allocation failed for N=%d\n", N);
                exit(EXIT_FAILURE);
            }

            for (int set = 0; set < BENCHMARK_SETS; ++set) {
                device_target(m + 1, WARMUP_ITERATIONS, 1, a, N);
                for (int run = 0; run < BENCHMARK_RUNS; ++run) {
                    device_target(m + 1, set, run, a, N);
                }
            }

            double avg, err;
            compute_offloading_time(&avg, &err);
            printf("\t%.1f\u00b1%.1f", avg, err);
            free(a);
        }
        printf("\n");
    }

    finalise();
    return 0;
}

void device_target(int offloading_method, int set, int run, double *a, int N) {
    int num_devs = omp_get_num_devices();
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
#pragma omp target teams map(tofrom : a[0 : N])
#pragma omp parallel num_threads(32)
                    array_delay(delaylength, a);
                    break;
                case 8:
#pragma omp target teams distribute parallel for num_teams(64 * num_devs) map(tofrom : a[0 : N])
                    for (int k = 0; k < 64 * num_devs; k++)
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
