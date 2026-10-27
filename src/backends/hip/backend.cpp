// Native HIP port of the CUDA execution contract; OpenMP remains the reference.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "backend.h"
#include "openmp_profile.h"

static microbenchmark_openmp_profile::Profile openmp_profile;
static int has_openmp_profile = 0;

static const char *method_names[] = {
    "pure delay kernel",
    "hipMalloc + H2D + kernel + D2H + hipFree",
    "hipMalloc + H2D + kernel + hipFree",
    "hipMalloc + kernel + D2H + hipFree",
    "hipMalloc + kernel + hipFree",
    "kernel launch blocks scalar",
    "teams distribute thread loop",
    "stream async launch + synchronize",
    "kernel atomicAdd",
    "teams thread loop array reduction",
    "kernel launch blocks + threads",
    "kernel repeated blocks + threads"};

static double get_time_usec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1.0e6 + ts.tv_nsec / 1.0e3;
}

static void check_hip(hipError_t err, const char *where)
{
    // Always check HIP runtime calls. Otherwise a failed kernel/memcpy can
    // hide until much later and become very hard to debug.
    if (err != hipSuccess)
    {
        fprintf(stderr, "HIP error at %s: %s\n", where, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ static void delay_kernel_device(int delaylength, double *array)
{
#ifdef MICROBENCHMARK_DELAY_VISIT
    MICROBENCHMARK_DELAY_VISIT(array);
#endif
    // __device__ means this function runs on the GPU, not on the CPU.
    // Every benchmark method uses the same tiny delay body.
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
        array[0] += i;
    if (array[0] < 0.0)
        printf("%f\n", array[0]);
}

// A supplied profile reproduces observed OpenMP ownership. Its chunk scopes
// are inferred from contiguous owner runs, not observed OpenMP region events.
struct hip_omp_schedule_t
{
    const int *chunk_counts;
    const int *offsets;
    const int *indices;
    int max_chunks;
};

__device__ static int schedule_chunks(hip_omp_schedule_t schedule, int max_iter)
{
    if (schedule.chunk_counts) return schedule.chunk_counts[blockIdx.x];
    long long first = (long long)blockIdx.x * blockDim.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    return first >= max_iter ? 0 : (int)(1 + (max_iter - 1 - first) / stride);
}

__device__ static int schedule_begin(hip_omp_schedule_t schedule, int chunk)
{
    size_t slot = ((size_t)blockIdx.x * schedule.max_chunks + chunk) * blockDim.x + threadIdx.x;
    return schedule.offsets[slot];
}

__device__ static int schedule_end(hip_omp_schedule_t schedule, int chunk)
{
    size_t slot = ((size_t)blockIdx.x * schedule.max_chunks + chunk) * blockDim.x + threadIdx.x;
    return schedule.offsets[slot + 1];
}

__global__ static void delay_one_kernel(int delay, double *a)
{
    // __global__ means "HIP kernel": CPU launches it, GPU runs it.
    // Only one GPU thread does useful work here, so this is a launch baseline.
    if (blockIdx.x == 0 && threadIdx.x == 0)
        delay_kernel_device(delay, a);
}

__global__ static void block_scalar_kernel(int delay, double *a, int max_array_size, int requested_teams = 0)
{
    // One thread per block does work. This mirrors OpenMP teams / OpenACC gangs
    // where we care about team/block launch overhead more than per-thread work.
    if (threadIdx.x == 0)
    {
        int divisor = requested_teams > 0 ? requested_teams : (int)gridDim.x;
        int idx = (int)((long long)blockIdx.x * max_array_size / divisor);
        delay_kernel_device(delay, &a[idx]);
    }
}

// Without an observed profile this is an explicit HIP cyclic schedule only.
// OpenMP's original defaults are implementation-defined; never alter them here.
__global__ static void loop_kernel(int delay, double *a, int max_iter,
                                  hip_omp_schedule_t schedule = {})
{
    int chunks = schedule_chunks(schedule, max_iter);
    for (int chunk = 0; chunk < chunks; ++chunk)
    {
        if (schedule.offsets)
        {
            for (int pos = schedule_begin(schedule, chunk); pos < schedule_end(schedule, chunk); ++pos)
                delay_kernel_device(delay, &a[schedule.indices[pos]]);
        }
        else
        {
            long long i = ((long long)chunk * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < max_iter) delay_kernel_device(delay, &a[i]);
        }
        __syncthreads();
    }
}

__global__ static void atomic_kernel(int delay, double *a, double *tmp, int max_iter, int N,
                                    hip_omp_schedule_t schedule = {})
{
    int chunks = schedule_chunks(schedule, max_iter);
    for (int chunk = 0; chunk < chunks; ++chunk)
    {
        if (schedule.offsets)
        {
            for (int pos = schedule_begin(schedule, chunk); pos < schedule_end(schedule, chunk); ++pos)
            {
                int i = schedule.indices[pos];
                delay_kernel_device(delay, &a[i]);
                atomicAdd(&tmp[i % N], 1.0);
            }
        }
        else
        {
            long long i = ((long long)chunk * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < max_iter)
            {
                delay_kernel_device(delay, &a[i]);
                atomicAdd(&tmp[i % N], 1.0);
            }
        }
        __syncthreads();
    }
}

__global__ static void reduction_kernel(int delay, double *a, double *tmp,
                                       double *private_buckets, double *team_buckets,
                                       int max_iter, int N, hip_omp_schedule_t schedule = {})
{
    size_t team_base = (size_t)blockIdx.x * blockDim.x * N;
    double *local = private_buckets + team_base + (size_t)threadIdx.x * N;
    double *team = team_buckets + (size_t)blockIdx.x * N;
    for (int j = threadIdx.x; j < N; j += blockDim.x) team[j] = 0.0;
    int chunks = schedule_chunks(schedule, max_iter);
    for (int chunk = 0; chunk < chunks; ++chunk)
    {
        for (int j = 0; j < N; ++j) local[j] = 0.0;
        if (schedule.offsets)
        {
            for (int pos = schedule_begin(schedule, chunk); pos < schedule_end(schedule, chunk); ++pos)
            {
                int i = schedule.indices[pos];
                delay_kernel_device(delay, &a[i]);
                local[i % N] += 1.0;
            }
        }
        else
        {
            long long i = ((long long)chunk * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < max_iter)
            {
                delay_kernel_device(delay, &a[i]);
                local[i % N] += 1.0;
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < N; j += blockDim.x)
        {
            double sum = 0.0;
            for (unsigned int t = 0; t < blockDim.x; ++t)
                sum += private_buckets[team_base + (size_t)t * N + j];
            team[j] += sum;
        }
        __syncthreads();
    }
    for (int j = threadIdx.x; j < N; j += blockDim.x) atomicAdd(&tmp[j], team[j]);
}

__global__ static void teams_parallel_kernel(int delay, double *a, int max_array_size, int requested_teams = 0)
{
    int stride = max_array_size / (requested_teams > 0 ? requested_teams : (int)gridDim.x);
    int idx = blockIdx.x * stride + threadIdx.x;
    delay_kernel_device(delay, &a[idx]);
}

__global__ static void repeated_teams_parallel_kernel(int delay, double *a,
                                                      int max_array_size, int parreps, int requested_threads = 0)
{
    // HIP threads persist for the kernel lifetime. Block barriers preserve
    // each OpenMP inner parallel region's fork/join ordering, not its runtime cost.
    for (int r = 0; r < parreps; r++)
    {
        int configured_threads = requested_threads > 0 ? requested_threads : (int)blockDim.x;
        int idx = (blockIdx.x * configured_threads + threadIdx.x) % max_array_size;
        delay_kernel_device(delay, &a[idx]);
        __syncthreads();
    }
}

const char *backend_name(void) { return "hip"; }
int backend_num_methods(void) { return 12; }
const char *backend_method_name(int method) { return method_names[method]; }
const char *backend_control_a_name(void) { return "block_count"; }
const char *backend_control_b_name(void) { return "thread_count"; }

void backend_default_config(backend_config_t *config)
{
    int count = 0;
    check_hip(hipGetDeviceCount(&count), "default device count");
    config->control_a = 4 * count; // OpenMP default: four teams per device.
    config->control_b = 32;       // OpenMP default thread limit.
}

int backend_device_available(void)
{
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count <= 0)
    {
        fprintf(stderr, "No HIP device available: %s\n", hipGetErrorString(err));
        return 0;
    }

    int device = 0;
    hipDeviceProp_t prop;
    check_hip(hipGetDevice(&device), "hipGetDevice");
    check_hip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");

    printf("Number of available HIP devices: %d\n", count);
    printf("HIP device %d: %s\n", device, prop.name);
    printf("HIP compute unit count: %d\n", prop.multiProcessorCount);
    return 1;
}

void backend_init(int argc, char **argv)
{
    const char *profile_path = getenv("HIP_OPENMP_PROFILE");
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], "OpenMPProfile=", 14) == 0) profile_path = argv[i] + 14;
    if (profile_path && *profile_path)
    {
        try { openmp_profile = microbenchmark_openmp_profile::read(profile_path); }
        catch (const std::exception& error) { fprintf(stderr, "%s\n", error.what()); exit(EXIT_FAILURE); }
        has_openmp_profile = 1;
        printf("HIP replays observed OpenMP owners/participants from %s; chunk scopes/order are inferred.\n", profile_path);
    }
    else
        printf("HIP launch/schedule has no OpenMP runtime profile; actual participant/owner equivalence is unverified.\n");
    check_hip(hipFree(0), "HIP runtime initialization");
    printf("Initializing HIP benchmark runtime environment...\n");
}

void backend_finalise(void)
{
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize finalise");
    printf("Finalizing HIP benchmark.\n");
}

static void copy_host_to_device(double *d_a, const double *a, int count)
{
    // H2D: host-to-device, CPU memory -> GPU memory.
    check_hip(hipMemcpy(d_a, a, (size_t)count * sizeof(double), hipMemcpyHostToDevice),
               "hipMemcpy host to device");
}

static void copy_device_to_host(double *a, const double *d_a, int count)
{
    // D2H: device-to-host, GPU memory -> CPU memory.
    check_hip(hipMemcpy(a, d_a, (size_t)count * sizeof(double), hipMemcpyDeviceToHost),
               "hipMemcpy device to host");
}

// Events are created only for a separate profiling pass. Their interval covers
// stream execution between the two records, excluding allocation/copies.
// The interval can include device idle gaps while the host submits commands.
struct device_events_t
{
    int enabled;
    hipEvent_t start, stop;
    double total_us;
};

static void event_begin(device_events_t *events, hipStream_t stream = 0)
{
    if (events->enabled) check_hip(hipEventRecord(events->start, stream), "kernel start event");
}

static void event_stop(device_events_t *events, hipStream_t stream = 0)
{
    if (events->enabled) check_hip(hipEventRecord(events->stop, stream), "kernel stop event");
}

static void event_collect(device_events_t *events)
{
    if (events->enabled)
    {
        float elapsed_ms = 0.0f;
        check_hip(hipEventElapsedTime(&elapsed_ms, events->start, events->stop), "kernel event elapsed time");
        events->total_us += (double)elapsed_ms * 1000.0;
    }
}

static int last_launch_blocks = 0, last_launch_threads = 0;

const char *backend_hip_alignment_mode(void)
{
    return has_openmp_profile ? "observed_owners_inferred_chunks" : "hip_static_cyclic_unverified";
}

void backend_hip_last_launch_config(int *blocks, int *threads)
{
    *blocks = last_launch_blocks;
    *threads = last_launch_threads;
}

static int *upload_profile_vector(const std::vector<int>& values)
{
    if (values.empty()) return NULL;
    int *device = NULL;
    check_hip(hipMalloc((void **)&device, values.size() * sizeof(int)), "allocate OpenMP profile schedule");
    check_hip(hipMemcpy(device, values.data(), values.size() * sizeof(int), hipMemcpyHostToDevice), "upload OpenMP profile schedule");
    return device;
}

template <bool measure_device>
static double run_hip_method(int method, double *a, int N, int delay,
                          int max_iter, int max_array_size,
                          const backend_config_t *config,
                          int inner_reps)
{
    int block_count = config->control_a;
    int thread_count = config->control_b;
    const int requested_teams = block_count, requested_threads = thread_count;
    microbenchmark_openmp_profile::Plan plan;
    if (has_openmp_profile)
    {
        try { plan = openmp_profile.select(method, N, max_iter, max_array_size, requested_teams, requested_threads); }
        catch (const std::exception& error) { fprintf(stderr, "%s\n", error.what()); exit(EXIT_FAILURE); }
        block_count = plan.effective_team_count;
        thread_count = plan.effective_thread_count;
    }
    int device;
    hipDeviceProp_t prop;
    check_hip(hipGetDevice(&device), "configuration device");
    check_hip(hipGetDeviceProperties(&prop, device), "configuration properties");
    int uses_teams = method == 5 || method == 6 || method >= 8;
    int uses_threads = method == 6 || method >= 8;
    int uses_worksharing_loop = method == 6 || method == 8 || method == 9;
    if (method < 0 || method >= backend_num_methods() ||
        N <= 0 || inner_reps <= 0 || max_array_size <= 0 ||
        (method >= 1 && method <= 4 && N > max_array_size) ||
        (uses_teams && (block_count <= 0 || block_count > prop.maxGridSize[0])) ||
        (uses_threads && (thread_count <= 0 || thread_count > prop.maxThreadsPerBlock ||
                          thread_count > prop.maxThreadsDim[0])) ||
        (uses_worksharing_loop && (max_iter < 0 || max_iter > max_array_size)) ||
        (method == 5 && (requested_teams <= 0 || block_count > requested_teams || requested_teams > max_array_size)) ||
        (method == 10 && (requested_teams <= 0 ||
         (long long)(block_count - 1) * (max_array_size / requested_teams) + thread_count > max_array_size)) ||
        (method == 11 && (requested_threads <= 0 ||
         (long long)(block_count - 1) * requested_threads + thread_count > max_array_size)))
    {
        fprintf(stderr, "Invalid HIP workload or launch dimensions\n");
        exit(EXIT_FAILURE);
    }
    hip_omp_schedule_t schedule = {};
    if (has_openmp_profile && uses_worksharing_loop)
    {
        schedule.chunk_counts = upload_profile_vector(plan.chunk_counts);
        schedule.offsets = upload_profile_vector(plan.offsets);
        schedule.indices = upload_profile_vector(plan.indices);
        schedule.max_chunks = plan.max_chunks;
    }
    last_launch_blocks = (method <= 4 || method == 7) ? 1 : block_count;
    last_launch_threads = (method <= 5 || method == 7) ? 1 : thread_count;
    device_events_t events = {measure_device, 0, 0, 0.0};
    if (measure_device)
    {
        check_hip(hipEventCreate(&events.start), "create kernel start event");
        check_hip(hipEventCreate(&events.stop), "create kernel stop event");
    }
    double start = 0.0, end = 0.0;

    if (method >= 1 && method <= 4)
    {
        // Methods 1-4 include allocation/copy/free inside the timed region.
        // This is the explicit HIP version of OpenMP/OpenACC data clauses.
        start = get_time_usec();
        for (int rep = 0; rep < inner_reps; rep++)
        {
            double *d_a = NULL;
            // d_a is device memory. CPU cannot dereference it directly.
            check_hip(hipMalloc((void **)&d_a, (size_t)N * sizeof(double)), "hipMalloc method 1-4");

            switch (method)
            {
            case 1:
                // tofrom/copy idea: H2D, kernel, D2H.
                copy_host_to_device(d_a, a, N);
                if (measure_device) event_begin(&events, 0);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 1 launch");
                check_hip(hipDeviceSynchronize(), "method 1 synchronize");
                if (measure_device) event_collect(&events);
                copy_device_to_host(a, d_a, N);
                break;

            case 2:
                // to/copyin idea: H2D, kernel, no D2H.
                copy_host_to_device(d_a, a, N);
                if (measure_device) event_begin(&events, 0);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 2 launch");
                check_hip(hipDeviceSynchronize(), "method 2 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 3:
                // from/copyout idea: no H2D, kernel, D2H.
                if (measure_device) event_begin(&events, 0);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 3 launch");
                check_hip(hipDeviceSynchronize(), "method 3 synchronize");
                if (measure_device) event_collect(&events);
                copy_device_to_host(a, d_a, N);
                break;

            case 4:
                // alloc/create idea: allocate device memory, kernel, no copy.
                if (measure_device) event_begin(&events, 0);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 4 launch");
                check_hip(hipDeviceSynchronize(), "method 4 synchronize");
                if (measure_device) event_collect(&events);
                break;
            }

            check_hip(hipFree(d_a), "hipFree method 1-4");
            a[0] += 1.0;
            if (a[0] < 0.0)
                printf("%f\n", a[0]);
        }
        end = get_time_usec();
    }
    else if (method == 0 || (method >= 5 && method <= 11))
    {
        // Methods 0 and 5-11 allocate device memory once outside timing.
        // Then the timed part focuses on launch/parallel behavior.
        double *d_a = NULL;
        double *tmp = (double *)calloc((size_t)N, sizeof(double));
        if (!tmp)
        {
            fprintf(stderr, "Allocation failed for tmp[N=%d]\n", N);
            exit(EXIT_FAILURE);
        }
        double *d_tmp = NULL;
        double *d_private = NULL;
        double *d_team = NULL;
        hipStream_t stream = NULL;
        check_hip(hipMalloc((void **)&d_a, (size_t)max_array_size * sizeof(double)), "hipMalloc d_a");
        check_hip(hipMalloc((void **)&d_tmp, (size_t)N * sizeof(double)), "hipMalloc d_tmp");
        copy_host_to_device(d_a, a, max_array_size);
        copy_host_to_device(d_tmp, tmp, N);
        if (method == 9)
        {
            size_t participants = (size_t)block_count * thread_count;
            if (participants > SIZE_MAX / sizeof(double) / (size_t)N)
            {
                fprintf(stderr, "HIP array reduction workspace size overflow\n");
                exit(EXIT_FAILURE);
            }
            check_hip(hipMalloc((void **)&d_private, participants * N * sizeof(double)),
                       "hipMalloc array reduction private buckets");
            check_hip(hipMalloc((void **)&d_team, (size_t)block_count * N * sizeof(double)),
                       "hipMalloc array reduction team buckets");
        }
        if (method == 7)
            check_hip(hipStreamCreate(&stream), "hipStreamCreate");
        check_hip(hipDeviceSynchronize(), "resident data setup synchronize");

#ifdef MICROBENCHMARK_RESIDENT_DATA
        MICROBENCHMARK_RESIDENT_DATA(d_a, d_tmp);
#endif
        start = get_time_usec();
        for (int rep = 0; rep < inner_reps; rep++)
        {
            switch (method)
            {
            case 0:
                if (measure_device) event_begin(&events, 0);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 0 launch");
                check_hip(hipDeviceSynchronize(), "method 0 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 5:
                if (measure_device) event_begin(&events, 0);
                block_scalar_kernel<<<block_count, 1>>>(delay, d_a, max_array_size, requested_teams);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 5 launch");
                check_hip(hipDeviceSynchronize(), "method 5 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 6:
                if (measure_device) event_begin(&events, 0);
                loop_kernel<<<block_count, thread_count>>>(delay, d_a, max_iter, schedule);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 6 launch");
                check_hip(hipDeviceSynchronize(), "method 6 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 7:
                // Async case: launch into a HIP stream, then explicitly wait.
                if (measure_device) event_begin(&events, stream);
                delay_one_kernel<<<1, 1, 0, stream>>>(delay, d_a);
                if (measure_device) event_stop(&events, stream);
                check_hip(hipGetLastError(), "method 7 launch");
                check_hip(hipStreamSynchronize(stream), "method 7 stream synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 8:
                if (measure_device) event_begin(&events, 0);
                atomic_kernel<<<block_count, thread_count>>>(delay, d_a, d_tmp, max_iter, N, schedule);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 8 launch");
                check_hip(hipDeviceSynchronize(), "method 8 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 9:
                if (measure_device) event_begin(&events, 0);
                reduction_kernel<<<block_count, thread_count>>>(delay, d_a, d_tmp, d_private, d_team, max_iter, N, schedule);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 9 launch");
                check_hip(hipDeviceSynchronize(), "method 9 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 10:
                if (measure_device) event_begin(&events, 0);
                teams_parallel_kernel<<<block_count, thread_count>>>(delay, d_a, max_array_size, requested_teams);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 10 launch");
                check_hip(hipDeviceSynchronize(), "method 10 synchronize");
                if (measure_device) event_collect(&events);
                break;

            case 11:
                if (measure_device) event_begin(&events, 0);
                repeated_teams_parallel_kernel<<<block_count, thread_count>>>(delay, d_a, max_array_size, N, requested_threads);
                if (measure_device) event_stop(&events, 0);
                check_hip(hipGetLastError(), "method 11 launch");
                check_hip(hipDeviceSynchronize(), "method 11 synchronize");
                if (measure_device) event_collect(&events);
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

        copy_device_to_host(a, d_a, max_array_size);
        copy_device_to_host(tmp, d_tmp, N);
#ifdef MICROBENCHMARK_RESULT
        MICROBENCHMARK_RESULT(a, tmp);
#endif
        if (method == 7)
            check_hip(hipStreamDestroy(stream), "hipStreamDestroy");
        if (d_team)
            check_hip(hipFree(d_team), "hipFree reduction team buckets");
        if (d_private)
            check_hip(hipFree(d_private), "hipFree reduction private buckets");
        check_hip(hipFree(d_tmp), "hipFree d_tmp");
        check_hip(hipFree(d_a), "hipFree d_a");
        free(tmp);
    }

    if (schedule.indices) check_hip(hipFree((void *)schedule.indices), "free profile indices");
    if (schedule.offsets) check_hip(hipFree((void *)schedule.offsets), "free profile offsets");
    if (schedule.chunk_counts) check_hip(hipFree((void *)schedule.chunk_counts), "free profile chunk counts");
    if (measure_device)
    {
        check_hip(hipEventDestroy(events.stop), "destroy kernel stop event");
        check_hip(hipEventDestroy(events.start), "destroy kernel start event");
        return events.total_us / inner_reps;
    }
    return (end - start) / inner_reps;
}

double backend_run_method(int method, double *a, int N, int delay,
                          int max_iter, int max_array_size,
                          const backend_config_t *config, int inner_reps)
{
    return run_hip_method<false>(method, a, N, delay, max_iter, max_array_size, config, inner_reps);
}

double backend_run_device_method(int method, const double *a, int N, int delay,
                                 int max_iter, int max_array_size,
                                 const backend_config_t *config, int inner_reps)
{
    // Preserve the original sample's host state. The profile executes exactly
    // the same method on a clone and never feeds its increments into the driver.
    double *profile_a = (double *)malloc((size_t)max_array_size * sizeof(double));
    if (!profile_a) { fprintf(stderr, "HIP device profile host allocation failed\n"); exit(EXIT_FAILURE); }
    memcpy(profile_a, a, (size_t)max_array_size * sizeof(double));
    double result = run_hip_method<true>(method, profile_a, N, delay, max_iter, max_array_size, config, inner_reps);
    free(profile_a);
    return result;
}
