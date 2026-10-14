#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "backend.h"

static const char *method_names[] = {
    "pure delay kernel",
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

// This delay body must be available inside OpenMP target regions.
#pragma omp declare target
static void delay_kernel(int delaylength, double *array)
{
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
        array[0] += i;
    if (array[0] < 0)
        printf("%f\n", array[0]);
}
#pragma omp end declare target

const char *backend_name(void) { return "openmp"; }
int backend_num_methods(void) { return 12; }
const char *backend_method_name(int method) { return method_names[method]; }
const char *backend_control_a_name(void) { return "thread_count"; }
const char *backend_control_b_name(void) { return "team_count"; }

void backend_default_config(backend_config_t *config)
{
    config->control_a = 32;
    config->control_b = 4 * omp_get_num_devices();
}

int backend_device_available(void)
{
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
    int thread_count = config->control_a;
    int team_count = config->control_b;
    double start = 0.0, end = 0.0;

    if (method >= 1 && method <= 4)
    {
        // Methods 1-4 measure the cost of entering a target region together
        // with a specific OpenMP map clause.
        start = omp_get_wtime();
        for (int irep = 0; irep < inner_reps; irep++)
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
            }

            a[0] += 1.0;
            if (a[0] < 0.0)
                printf("%f\n", a[0]);
        }
        end = omp_get_wtime();
    }
    else if (method == 0 || (method >= 5 && method <= 11))
    {
        // Methods 0 and 5-11 keep data mapped outside the timed inner loop.
        // This focuses the timing on launch/teams/parallel behavior.
        double *tmp = (double *)malloc((size_t)N * sizeof(double));
        if (!tmp)
        {
            fprintf(stderr, "Allocation failed for tmp[N=%d]\n", N);
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < N; i++)
            tmp[i] = 0.0;

#pragma omp target data map(tofrom : a[0 : max_array_size], tmp[0 : N])
        {
            start = omp_get_wtime();

            for (int rep = 0; rep < inner_reps; rep++)
            {
                switch (method)
                {
                case 0:
                    // Baseline target launch with only the delay kernel.
#pragma omp target
                    delay_kernel(delay, a);
                    break;

                case 5:
                    // One delay per team, controlled by team_count.
#pragma omp target teams num_teams(team_count)
                    {
                        delay_kernel(delay, &a[omp_get_team_num() * max_array_size / team_count]);
                    }
                    break;

                case 6:
                    // Main worksharing case: teams distribute parallel for.
#pragma omp target teams distribute parallel for num_teams(team_count) thread_limit(thread_count)
                    for (int i = 0; i < max_iter; i++)
                        delay_kernel(delay, &a[i]);
                    break;

                case 7:
                    // Asynchronous target launch, followed by taskwait.
#pragma omp target nowait
                    delay_kernel(delay, a);
#pragma omp taskwait
                    break;

                case 8:
                    // Atomic update path, using tmp to isolate the update target.
#pragma omp target teams distribute parallel for
                    for (int i = 0; i < max_iter; i++)
                    {
                        delay_kernel(delay, &a[i]);
#pragma omp atomic
                        tmp[i % N] += 1.0;
                    }
                    break;

                case 9:
                    // Reduction path. Compiler families disagree on syntax,
                    // so keep the backend-specific variants here.
#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for reduction(+ : tmp[0 : N])
                    for (int i = 0; i < max_iter; i++)
                    {
                        delay_kernel(delay, &a[i]);
                        tmp[i % N] += 1.0;
                    }
                    break;
#elif defined(_CRAYC) || defined(__AMD__) || defined(__clang__)
#pragma omp target
#pragma teams distribute reduction(+ : tmp[0 : N])
#pragma parallel for reduction(+ : tmp[0 : N])
                    for (int i = 0; i < max_iter; i++)
                    {
                        delay_kernel(delay, &a[i]);
                        tmp[i % N] += 1.0;
                    }
                    break;
#else
#pragma omp target teams distribute parallel for reduction(+ : tmp[0 : N])
                    for (int i = 0; i < max_iter; i++)
                    {
                        delay_kernel(delay, &a[i]);
                        tmp[i % N] += 1.0;
                    }
                    break;
#endif

                case 10:
                    // Explicit teams + inner parallel region.
#pragma omp target teams num_teams(team_count) thread_limit(thread_count)
                    {
                        int team_id = omp_get_team_num();
                        int stride = max_array_size / team_count;
#pragma omp parallel
                        {
                            delay_kernel(delay, &a[team_id * stride + omp_get_thread_num()]);
                        }
                    }
                    break;

                case 11:
                {
                    // Repeated inner parallel region; N controls repetition count.
                    int parreps = N;
#pragma omp target teams num_teams(team_count) thread_limit(thread_count)
                    {
                        int team_id = omp_get_team_num();
                        for (int r = 0; r < parreps; r++)
                        {
#pragma omp parallel
                            {
                                int idx = team_id * thread_count + omp_get_thread_num();
                                idx = idx % max_array_size;
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

            end = omp_get_wtime();
        }

        free(tmp);
    }

    return (end - start) * 1.0e6 / inner_reps;
}
