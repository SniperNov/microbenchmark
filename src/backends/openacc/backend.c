#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <openacc.h>
#include "backend.h"

static const char *method_names[] = {
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
    "parallel num_gangs + num_workers",
    "parallel worker loop num_gangs + num_workers"};

// This delay body must be available inside OpenACC compute regions.
#pragma acc routine seq
static void delay_kernel(int delaylength, double *array)
{
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
        array[0] += i;
    if (array[0] < 0.0)
        printf("%f\n", array[0]);
}

static double get_time_usec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1.0e6 + tv.tv_usec;
}

const char *backend_name(void) { return "openacc"; }
int backend_num_methods(void) { return 12; }
const char *backend_method_name(int method) { return method_names[method]; }
const char *backend_control_a_name(void) { return "gang_count"; }
const char *backend_control_b_name(void) { return "worker_count"; }

void backend_default_config(backend_config_t *config)
{
    config->control_a = 64;
    config->control_b = 128;
}

int backend_device_available(void)
{
    acc_device_t devtype = acc_get_device_type();
    int numdevs = acc_get_num_devices(devtype);

    printf("OpenACC device type: %d\n", (int)devtype);
    printf("Number of available devices: %d\n", numdevs);

    if (numdevs <= 0)
    {
        fprintf(stderr, "No OpenACC device available. Terminating.\n");
        return 0;
    }
    return 1;
}

void backend_init(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    printf("Initializing benchmark runtime environment...\n");
}

void backend_finalise(void)
{
    printf("Finalizing benchmark.\n");
}

double backend_run_method(int method, double *a, int N, int delay,
                          int max_iter, int max_array_size,
                          const backend_config_t *config,
                          int inner_reps)
{
    int gang_count = config->control_a;
    int worker_count = config->control_b;
    double start = 0.0, end = 0.0;

    if (method >= 1 && method <= 4)
    {
        // Methods 1-4 measure the data clause/serial target entry itself.
        start = get_time_usec();
        for (int irep = 0; irep < inner_reps; irep++)
        {
            switch (method)
            {
            case 1:
#pragma acc serial copy(a[0:N])
                delay_kernel(delay, a);
                break;

            case 2:
#pragma acc serial copyin(a[0:N])
                delay_kernel(delay, a);
                break;

            case 3:
#pragma acc serial copyout(a[0:N])
                delay_kernel(delay, a);
                break;

            case 4:
#pragma acc serial create(a[0:N])
                delay_kernel(delay, a);
                break;
            }

            a[0] += 1.0;
            if (a[0] < 0.0)
                printf("%f\n", a[0]);
        }
        end = get_time_usec();
    }
    else if (method == 0 || (method >= 5 && method <= 11))
    {
        // Methods 0 and 5-11 keep data present while timing launch/parallel
        // shape differences such as async, atomic, reduction, gangs, and workers.
        double *tmp = (double *)malloc((size_t)N * sizeof(double));
        if (!tmp)
        {
            fprintf(stderr, "Allocation failed for tmp[N=%d]\n", N);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < N; ++i)
            tmp[i] = 0.0;

#pragma acc data copy(a[0:max_array_size], tmp[0:N])
        {
            start = get_time_usec();

            for (int rep = 0; rep < inner_reps; rep++)
            {
                switch (method)
                {
                case 0:
                    // Baseline OpenACC parallel launch with only the delay kernel.
#pragma acc parallel present(a[0:max_array_size])
                    {
                        delay_kernel(delay, a);
                    }
                    break;

                case 5:
                    // Scalar parallel launch, analogous to OpenMP teams(scalar).
#pragma acc parallel present(a[0:max_array_size])
                    {
                        delay_kernel(delay, a);
                    }
                    break;

                case 6:
                    // Default parallel loop.
#pragma acc parallel loop present(a[0:max_array_size])
                    for (int i = 0; i < max_iter; ++i)
                        delay_kernel(delay, &a[i]);
                    break;

                case 7:
                    // Async launch plus explicit wait.
#pragma acc parallel loop async(1) present(a[0:max_array_size])
                    for (int i = 0; i < max_iter; ++i)
                        delay_kernel(delay, &a[i]);
#pragma acc wait(1)
                    break;

                case 8:
                    // Atomic update path, sharing the same outer data region.
#pragma acc parallel loop present(a[0:max_array_size], tmp[0:N])
                    for (int i = 0; i < max_iter; ++i)
                    {
                        delay_kernel(delay, &a[i]);
#pragma acc atomic update
                        tmp[i % N] += 1.0;
                    }
                    break;

                case 9:
                {
                    // Reduction path using an OpenACC scalar reduction.
                    double reduction_sum = 0.0;
#pragma acc parallel loop reduction(+ : reduction_sum) present(a[0:N])
                    for (int i = 0; i < max_iter; ++i)
                    {
                        delay_kernel(delay, &a[i]);
                        reduction_sum += 1.0;
                    }
                    if (reduction_sum < 0.0)
                        printf("%f\n", reduction_sum);
                    break;
                }

                case 10:
                    // Explicit gang/worker launch with a worker-level loop.
#pragma acc parallel num_gangs(gang_count) num_workers(worker_count) present(a[0:max_array_size])
                    {
#pragma acc loop worker
                        for (int i = 0; i < worker_count; ++i)
                            delay_kernel(delay, &a[i]);
                    }
                    break;

                case 11:
                {
                    // Repeated worker loop, matching the OpenMP repeated inner parallel case.
                    int parreps = N;
#pragma acc parallel num_gangs(gang_count) num_workers(worker_count) present(a[0:max_array_size])
                    {
                        for (int r = 0; r < parreps; ++r)
                        {
#pragma acc loop worker
                            for (int i = 0; i < worker_count; ++i)
                            {
                                int idx = i % max_array_size;
                                delay_kernel(delay, &a[idx]);
                            }
                        }
                    }
                    break;
                }
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

    return (end - start) / inner_reps;
}
