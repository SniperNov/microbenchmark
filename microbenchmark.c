#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <openacc.h>
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

#define NUM_METHODS 10
#define NUM_SIZES 14
int sizes[NUM_SIZES] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

const char *method_names[] = {
    "parallel loop copy(a[0:N])",
    "parallel loop copyin(a[0:N])",
    "parallel loop copyout(a[0:N])",
    "parallel loop create(a[0:N])",
    "parallel loop",
    "parallel loop gang copy(a[0:N])",
    "parallel loop gang vector copy(a[0:N])",
    "parallel loop num_gangs(64) copy(a[0:N])",
    "parallel loop async(1) copy(a[0:N]) + wait(1)",
    "parallel loop num_gangs(8) vector_length(256) copy(a[0:N])"
};

double delays[NUM_SAMPLES];
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES];

void compute_offloading_time(double *intercept_avg, double *error);
void warmup_cache();
void device_target(int offloading_method, int set, int run, double *a, int N);

int main(int argc, char **argv)
{
    int num_devs = acc_get_num_devices(acc_get_device_type());
    int targetdev = -9999;

    printf("OpenACC device type: %ld\n", acc_get_device_type());
    printf("Number of available devices: %d\n", num_devs);

    #pragma acc parallel present_or_create(targetdev)
{
    if (acc_on_device(acc_get_device_type())) {
        targetdev = 0;
    } else {
        targetdev = 1;
    }
}
if (targetdev == 1) {
    fprintf(stderr, "⚠️  Target region did NOT offload. Please ensure GPU is available!\n");
    exit(EXIT_FAILURE);
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
    for (int i = 0; i < NUM_SAMPLES; i++) {
        int delaylength = MIN_DELAYLENGTH + i * (MAX_DELAYLENGTH - MIN_DELAYLENGTH) / (NUM_SAMPLES - 1);
        delays[i] = delaylength;

        double start = get_time_usec();
        for (int j = 0; j < INNERREPS; j++) {
            switch (offloading_method) {
                case 1:
#pragma acc parallel loop copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 2:
#pragma acc parallel loop copyin(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 3:
#pragma acc parallel loop copyout(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 4:
#pragma acc parallel loop create(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 5:
#pragma acc parallel loop
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 6:
#pragma acc parallel loop gang copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 7:
#pragma acc parallel loop gang vector copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 8:
#pragma acc parallel loop num_gangs(64) copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                case 9:
#pragma acc parallel loop async(1) copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
#pragma acc wait(1)
                    break;
                case 10:
#pragma acc parallel loop num_gangs(8) vector_length(256) copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
                default:
#pragma acc parallel loop copy(a[0:N])
                    for (int idx = 0; idx < delaylength; idx++) a[idx % N] += 1.0;
                    break;
            }                
            a[0] += 1;
            if (a[0] < 0)
            {
                printf("%f \n", a[0]);
            }
        }
        execution_times[set][run][i] = (get_time_usec() - start) / INNERREPS;
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
