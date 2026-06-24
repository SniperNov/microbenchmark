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
    "parallel num_gangs + vector_length",
    "parallel loop gang vector num_gangs + vector_length"};

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
const char *backend_control_b_name(void) { return "vector_length"; }

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
    int vector_length = config->control_b;
    double start = 0.0, end = 0.0;

    if (method >= 1 && method <= 4)
    {
        // Methods 1-4 measure a compute region together with one OpenACC
        // data movement clause.
        start = get_time_usec();
        for (int irep = 0; irep < inner_reps; irep++)
        {
            switch (method)
            {
            case 1:
#pragma acc parallel loop copy(a[0:N])
                for (int i = 0; i < max_iter; ++i)
                    delay_kernel(delay, &a[i % N]);
                break;

            case 2:
#pragma acc parallel loop copyin(a[0:N])
                for (int i = 0; i < max_iter; ++i)
                    delay_kernel(delay, &a[i % N]);
                break;

            case 3:
#pragma acc parallel loop copyout(a[0:N])
                for (int i = 0; i < max_iter; ++i)
                    delay_kernel(delay, &a[i % N]);
                break;

            case 4:
#pragma acc parallel loop create(a[0:N])
                for (int i = 0; i < max_iter; ++i)
                    delay_kernel(delay, &a[i % N]);
                break;
            }

            a[0] += 1.0;
            if (a[0] < 0.0)
                printf("%f\n", a[0]);
        }
        end = get_time_usec();
    }
    else if (method == 8)
    {
        // Atomic update uses a separate tmp array, matching the OpenMP intent.
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
#pragma acc parallel loop present(a[0:max_array_size], tmp[0:N])
                for (int i = 0; i < max_iter; ++i)
                {
                    delay_kernel(delay, &a[i]);
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
    else if (method == 0 || (method >= 5 && method <= 11))
    {
        // Methods 0 and 5-11 keep data present while timing launch/parallel
        // shape differences such as async, reduction, gangs, and vectors.
#pragma acc data copy(a[0:max_array_size])
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

                case 9:
                {
                    // Reduction path using an OpenACC scalar reduction.
                    double reduction_sum = 0.0;
#pragma acc parallel loop reduction(+ : reduction_sum) present(a[0:max_array_size])
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
                    // Explicitly control gang and vector dimensions.
#pragma acc parallel num_gangs(gang_count) vector_length(vector_length) present(a[0:max_array_size])
                    {
                        delay_kernel(delay, a);
                    }
                    break;

                case 11:
                    // Controlled gang/vector worksharing loop.
#pragma acc parallel loop gang vector num_gangs(gang_count) vector_length(vector_length) present(a[0:max_array_size])
                    for (int i = 0; i < max_iter; ++i)
                        delay_kernel(delay, &a[i]);
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

    return (end - start) / inner_reps;
}
