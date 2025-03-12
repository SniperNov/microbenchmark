#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
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

double delays[NUM_SAMPLES];
double execution_times[BENCHMARK_SETS][BENCHMARK_RUNS][NUM_SAMPLES];

void compute_offloading_time(double *intercept_avg, double *error);
void warmup_cache();
void device_target(int offloading_method, int set, int run, double *a, int N);

int main(int argc, char **argv)
{
    int hostdev, targetdev, num_devs;

    num_devs = omp_get_num_devices();
    hostdev = omp_get_initial_device();
    targetdev = -9999;

#pragma omp target map(from : targetdev)
    {
        targetdev = omp_is_initial_device();
    }

    printf("There are  %d available devices\n", num_devs);
    printf("Host device is %d\n", hostdev);

    if (targetdev)
    {
        printf("Target region executed on host. Terminating...\n");
        return 0;
    }
    else
    {
        printf("Target region executed on device. Continue...\n");
    }

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
        "num_teams(8) thread_limit(256)"};

    int methods[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int num_methods = 12;
    int N = 1024; // Default size

    if (argc > 1)
    {
        num_methods = argc - 1;
        for (int i = 0; i < num_methods; i++)
        {
            methods[i] = atoi(argv[i + 1]);
            if (methods[i] < 1 || methods[i] > 12)
            {
                printf("Invalid method %d. Ignoring.\n", methods[i]);
                methods[i] = 0;
            }
        }
    }

    // Parse N if provided in the command line argument
    if (argc > num_methods + 1)
    {
        N = atoi(argv[num_methods + 1]); // Set N from runtime argument
    }

    // Dynamically allocate array a based on N
    double *a = (double *)malloc(N * sizeof(double));
    if (a == NULL)
    {
        printf("Memory allocation for array a failed.\n");
        return EXIT_FAILURE;
    }

    init(argc, argv);

    printf("Benchmark results:\n");
    printf("--------------------------------------------------------------------------------------\n");
    printf("| %-40s | Estimated Offloading Time ± Error |\n", "Method");
    printf("--------------------------------------------------------------------------------------\n");

    for (int m = 0; m < num_methods; m++)
    {
        if (methods[m] == 0)
            continue;

        for (int set = 0; set < BENCHMARK_SETS; set++)
        {
            // warmup_cache();
            device_target(methods[m], WARMUP_ITERATIONS, 1, a, N);
            for (int run = 0; run < BENCHMARK_RUNS; run++)
            {
                device_target(methods[m], set, run, a, N);
            }
        }

        double intercept_avg, error;
        compute_offloading_time(&intercept_avg, &error);
        // Print the method name using methods[m]-1 as the index into method_names
        printf("| %-40s | %10.3f ± %7.3f μs          |\n", method_names[methods[m] - 1], intercept_avg, error);
    }

    printf("--------------------------------------------------------------------------------------\n");
    finalise();

    // Free allocated memory
    free(a);

    return EXIT_SUCCESS;
}

void device_target(int offloading_method, int set, int run, double *a, int N)
{
    int j, i, num_devs;
    double start, elapsed;
    num_devs = omp_get_num_devices();
    for (i = 0; i < NUM_SAMPLES; i++)
    {
        int delaylength = MIN_DELAYLENGTH + i * (MAX_DELAYLENGTH - MIN_DELAYLENGTH) / (NUM_SAMPLES - 1);
        delays[i] = delaylength;

        start = omp_get_wtime();
        for (j = 0; j < INNERREPS; j++)
        {
            switch (offloading_method)
            {
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
        elapsed = (omp_get_wtime() - start) * 1.0e6 / INNERREPS;
        execution_times[set][run][i] = elapsed;
    }
}

void warmup_cache()
{
    double dummy_array[1000000];
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        dummy_array[i] = i * 0.01;
    }
    if (dummy_array[0] < 0)
    {
        printf("%f \n", dummy_array[0]);
    }
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        dummy_array[i] += 1.0;
    }
    if (dummy_array[0] < 0)
    {
        printf("%f \n", dummy_array[0]);
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
