#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "backend.h"

static const char *method_names[] = {
    "pure delay kernel",
    "cudaMalloc + H2D + kernel + D2H + cudaFree",
    "cudaMalloc + H2D + kernel + cudaFree",
    "cudaMalloc + kernel + D2H + cudaFree",
    "cudaMalloc + kernel + cudaFree",
    "kernel launch blocks scalar",
    "kernel launch grid-stride loop",
    "stream async launch + synchronize",
    "kernel atomicAdd",
    "kernel block reduction",
    "kernel launch blocks + threads",
    "kernel repeated blocks + threads"};

static double get_time_usec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1.0e6 + tv.tv_usec;
}

static void check_cuda(cudaError_t err, const char *where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ static void delay_kernel_device(int delaylength, double *array)
{
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
        array[0] += i;
}

__global__ static void delay_one_kernel(int delay, double *a)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        delay_kernel_device(delay, a);
}

__global__ static void block_scalar_kernel(int delay, double *a, int max_array_size)
{
    if (threadIdx.x == 0)
    {
        int idx = blockIdx.x * max_array_size / gridDim.x;
        delay_kernel_device(delay, &a[idx]);
    }
}

__global__ static void loop_kernel(int delay, double *a, int max_iter)
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < max_iter; i += stride)
        delay_kernel_device(delay, &a[i]);
}

__global__ static void atomic_kernel(int delay, double *a, double *tmp, int max_iter, int N)
{
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < max_iter; i += stride)
    {
        delay_kernel_device(delay, &a[i]);
        atomicAdd(&tmp[i % N], 1.0);
    }
}

__global__ static void reduction_kernel(int delay, double *a, double *tmp, int max_iter)
{
    extern __shared__ double scratch[];
    double local_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < max_iter; i += stride)
    {
        delay_kernel_device(delay, &a[i]);
        local_sum += 1.0;
    }

    scratch[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(&tmp[0], scratch[0]);
}

__global__ static void launch_shape_kernel(int delay, double *a, int max_array_size)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) % max_array_size;
    delay_kernel_device(delay, &a[idx]);
}

__global__ static void repeated_launch_shape_kernel(int delay, double *a, int max_array_size, int parreps)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) % max_array_size;
    for (int r = 0; r < parreps; r++)
        delay_kernel_device(delay, &a[idx]);
}

const char *backend_name(void) { return "cuda"; }
int backend_num_methods(void) { return 12; }
const char *backend_method_name(int method) { return method_names[method]; }
const char *backend_control_a_name(void) { return "block_count"; }
const char *backend_control_b_name(void) { return "thread_count"; }

void backend_default_config(backend_config_t *config)
{
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, device) == cudaSuccess)
        config->control_a = prop.multiProcessorCount;
    else
        config->control_a = 64;
    config->control_b = 128;
}

int backend_device_available(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0)
    {
        fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(err));
        return 0;
    }

    int device = 0;
    cudaDeviceProp prop;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    printf("Number of available CUDA devices: %d\n", count);
    printf("CUDA device %d: %s\n", device, prop.name);
    printf("CUDA SM count: %d\n", prop.multiProcessorCount);
    return 1;
}

void backend_init(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    check_cuda(cudaFree(0), "CUDA runtime initialization");
    printf("Initializing CUDA benchmark runtime environment...\n");
}

void backend_finalise(void)
{
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize finalise");
    printf("Finalizing CUDA benchmark.\n");
}

static void copy_host_to_device(double *d_a, const double *a, int count)
{
    check_cuda(cudaMemcpy(d_a, a, (size_t)count * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
}

static void copy_device_to_host(double *a, const double *d_a, int count)
{
    check_cuda(cudaMemcpy(a, d_a, (size_t)count * sizeof(double), cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");
}

double backend_run_method(int method, double *a, int N, int delay,
                          int max_iter, int max_array_size,
                          const backend_config_t *config,
                          int inner_reps)
{
    int block_count = config->control_a > 0 ? config->control_a : 1;
    int thread_count = config->control_b > 0 ? config->control_b : 1;
    double start = 0.0, end = 0.0;

    if (method >= 1 && method <= 4)
    {
        start = get_time_usec();
        for (int rep = 0; rep < inner_reps; rep++)
        {
            double *d_a = NULL;
            check_cuda(cudaMalloc((void **)&d_a, (size_t)N * sizeof(double)), "cudaMalloc method 1-4");

            switch (method)
            {
            case 1:
                copy_host_to_device(d_a, a, N);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                check_cuda(cudaGetLastError(), "method 1 launch");
                check_cuda(cudaDeviceSynchronize(), "method 1 synchronize");
                copy_device_to_host(a, d_a, N);
                break;

            case 2:
                copy_host_to_device(d_a, a, N);
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                check_cuda(cudaGetLastError(), "method 2 launch");
                check_cuda(cudaDeviceSynchronize(), "method 2 synchronize");
                break;

            case 3:
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                check_cuda(cudaGetLastError(), "method 3 launch");
                check_cuda(cudaDeviceSynchronize(), "method 3 synchronize");
                copy_device_to_host(a, d_a, N);
                break;

            case 4:
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                check_cuda(cudaGetLastError(), "method 4 launch");
                check_cuda(cudaDeviceSynchronize(), "method 4 synchronize");
                break;
            }

            check_cuda(cudaFree(d_a), "cudaFree method 1-4");
            a[0] += 1.0;
            if (a[0] < 0.0)
                printf("%f\n", a[0]);
        }
        end = get_time_usec();
    }
    else if (method == 0 || (method >= 5 && method <= 11))
    {
        double *d_a = NULL;
        double *d_tmp = NULL;
        cudaStream_t stream = NULL;
        check_cuda(cudaMalloc((void **)&d_a, (size_t)max_array_size * sizeof(double)), "cudaMalloc d_a");
        check_cuda(cudaMalloc((void **)&d_tmp, (size_t)N * sizeof(double)), "cudaMalloc d_tmp");
        copy_host_to_device(d_a, a, max_array_size);
        check_cuda(cudaMemset(d_tmp, 0, (size_t)N * sizeof(double)), "cudaMemset d_tmp");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        start = get_time_usec();
        for (int rep = 0; rep < inner_reps; rep++)
        {
            switch (method)
            {
            case 0:
                delay_one_kernel<<<1, 1>>>(delay, d_a);
                check_cuda(cudaGetLastError(), "method 0 launch");
                check_cuda(cudaDeviceSynchronize(), "method 0 synchronize");
                break;

            case 5:
                block_scalar_kernel<<<block_count, 1>>>(delay, d_a, max_array_size);
                check_cuda(cudaGetLastError(), "method 5 launch");
                check_cuda(cudaDeviceSynchronize(), "method 5 synchronize");
                break;

            case 6:
                loop_kernel<<<block_count, thread_count>>>(delay, d_a, max_iter);
                check_cuda(cudaGetLastError(), "method 6 launch");
                check_cuda(cudaDeviceSynchronize(), "method 6 synchronize");
                break;

            case 7:
                loop_kernel<<<block_count, thread_count, 0, stream>>>(delay, d_a, max_iter);
                check_cuda(cudaGetLastError(), "method 7 launch");
                check_cuda(cudaStreamSynchronize(stream), "method 7 stream synchronize");
                break;

            case 8:
                atomic_kernel<<<block_count, thread_count>>>(delay, d_a, d_tmp, max_iter, N);
                check_cuda(cudaGetLastError(), "method 8 launch");
                check_cuda(cudaDeviceSynchronize(), "method 8 synchronize");
                break;

            case 9:
                check_cuda(cudaMemset(d_tmp, 0, sizeof(double)), "method 9 reset reduction output");
                reduction_kernel<<<block_count, thread_count, (size_t)thread_count * sizeof(double)>>>(delay, d_a, d_tmp, max_iter);
                check_cuda(cudaGetLastError(), "method 9 launch");
                check_cuda(cudaDeviceSynchronize(), "method 9 synchronize");
                break;

            case 10:
                launch_shape_kernel<<<block_count, thread_count>>>(delay, d_a, max_array_size);
                check_cuda(cudaGetLastError(), "method 10 launch");
                check_cuda(cudaDeviceSynchronize(), "method 10 synchronize");
                break;

            case 11:
                repeated_launch_shape_kernel<<<block_count, thread_count>>>(delay, d_a, max_array_size, N);
                check_cuda(cudaGetLastError(), "method 11 launch");
                check_cuda(cudaDeviceSynchronize(), "method 11 synchronize");
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
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        check_cuda(cudaFree(d_tmp), "cudaFree d_tmp");
        check_cuda(cudaFree(d_a), "cudaFree d_a");
    }

    return (end - start) / inner_reps;
}
