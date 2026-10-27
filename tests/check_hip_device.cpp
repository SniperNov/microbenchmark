// Run tests/run_hip_validation.sh on an accessible ROCm GPU.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

enum { trace_blocks = 304, trace_threads = 64, trace_calls = 128,
       trace_participants = trace_blocks * trace_threads };
__device__ unsigned long long delay_visits[trace_participants];
__device__ unsigned long long delay_addresses[trace_participants * trace_calls];
__device__ int trace_overflow;

__device__ static void record_delay_visit(double *array)
{
    if (blockIdx.x >= trace_blocks || threadIdx.x >= trace_threads)
    {
        atomicExch(&trace_overflow, 1);
        return;
    }
    unsigned int participant = blockIdx.x * trace_threads + threadIdx.x;
    unsigned long long ordinal = atomicAdd(&delay_visits[participant], 1ULL);
    if (ordinal >= trace_calls)
    {
        atomicExch(&trace_overflow, 1);
        return;
    }
    delay_addresses[(size_t)participant * trace_calls + ordinal] =
        reinterpret_cast<unsigned long long>(array);
}
#define MICROBENCHMARK_DELAY_VISIT(array) record_delay_visit(array)
static void check_backend_result(double *a, double *tmp);
#define MICROBENCHMARK_RESULT(a, tmp) check_backend_result(a, tmp)
#include "../src/backends/hip/backend.cpp"

static int checked_method, checked_n, checked_iterations, checked_reps, result_callbacks;
static void check_backend_result(double *a, double *tmp)
{
    (void)a;
    ++result_callbacks;
    for (int j = 0; j < checked_n; ++j)
    {
        double expected = checked_method == 8 || checked_method == 9 ?
            checked_reps * (checked_iterations / checked_n + (j < checked_iterations % checked_n)) : 0.0;
        if (tmp[j] != expected)
        {
            fprintf(stderr, "Method %d backend tmp[%d]=%g expected=%g\n", checked_method, j, tmp[j], expected);
            exit(3);
        }
    }
}

static void reset_trace(void)
{
    static const unsigned long long zero_visits[trace_participants] = {};
    int zero = 0;
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(delay_visits), zero_visits, sizeof(delay_visits), 0, hipMemcpyHostToDevice), "reset visits");
    check_hip(hipMemcpyToSymbol(HIP_SYMBOL(trace_overflow), &zero, sizeof(zero), 0, hipMemcpyHostToDevice), "reset trace overflow");
}

// Contract oracle: chunk k belongs to team k % B, and its T iterations belong
// to threads 0..T-1. Preserve each thread's order across chunks and repetitions.
static std::vector<int> expected_indices(int method, int b, int t,
                                         int blocks, int threads, int iterations,
                                         int size, int n, int reps)
{
    std::vector<int> indices;
    for (int r = 0; r < reps; ++r)
    {
        if (method <= 4 || method == 7)
        {
            if (b == 0 && t == 0) indices.push_back(0);
        }
        else if (b < blocks && method == 5)
        {
            if (t == 0) indices.push_back((int)((long long)b * size / blocks));
        }
        else if (b < blocks && t < threads)
        {
            if (method == 10) indices.push_back(b * (size / blocks) + t);
            else if (method == 11)
                for (int inner = 0; inner < n; ++inner)
                    indices.push_back((b * threads + t) % size);
            else
                for (int i = b * threads + t; i < iterations; i += blocks * threads)
                    indices.push_back(i);
        }
    }
    return indices;
}

static void verify_trace(int method, int blocks, int threads, int iterations,
                         int size, int n, int reps)
{
    std::vector<unsigned long long> visits(trace_participants);
    std::vector<unsigned long long> addresses(trace_participants * trace_calls);
    int overflow = 0;
    check_hip(hipMemcpyFromSymbol(visits.data(), HIP_SYMBOL(delay_visits), sizeof(delay_visits), 0, hipMemcpyDeviceToHost), "read visits");
    check_hip(hipMemcpyFromSymbol(addresses.data(), HIP_SYMBOL(delay_addresses), sizeof(delay_addresses), 0, hipMemcpyDeviceToHost), "read addresses");
    check_hip(hipMemcpyFromSymbol(&overflow, HIP_SYMBOL(trace_overflow), sizeof(overflow), 0, hipMemcpyDeviceToHost), "read trace overflow");
    if (overflow)
    {
        fprintf(stderr, "Device trace capacity exceeded\n");
        exit(4);
    }
    unsigned long long base = std::numeric_limits<unsigned long long>::max();
    // Each nonempty resident-data workload visits index zero. Methods 1-4
    // allocate separately on each repetition, so their pointer order is opaque.
    bool check_addresses = method == 0 || method >= 5;
    if (check_addresses)
        for (int p = 0; p < trace_participants; ++p)
            for (unsigned long long k = 0; k < visits[p]; ++k)
                base = std::min(base, addresses[(size_t)p * trace_calls + k]);
    for (int b = 0; b < trace_blocks; ++b)
    for (int t = 0; t < trace_threads; ++t)
    {
        int p = b * trace_threads + t;
        std::vector<int> expected = expected_indices(method, b, t, blocks, threads,
                                                     iterations, size, n, reps);
        if (visits[p] != expected.size())
        {
            fprintf(stderr, "Method %d B=%d T=%d I=%d N=%d team=%d thread=%d: calls=%llu expected=%zu\n",
                    method, blocks, threads, iterations, n, b, t, visits[p], expected.size());
            exit(4);
        }
        if (check_addresses)
            for (size_t k = 0; k < expected.size(); ++k)
            {
                unsigned long long actual = addresses[(size_t)p * trace_calls + k];
                unsigned long long wanted = base + (unsigned long long)expected[k] * sizeof(double);
                if (actual != wanted)
                {
                    fprintf(stderr, "Method %d B=%d T=%d I=%d N=%d team=%d thread=%d call=%zu: index=%llu expected=%d\n",
                            method, blocks, threads, iterations, n, b, t, k,
                            (actual - base) / sizeof(double), expected[k]);
                    exit(4);
                }
            }
    }
}

static void verify_array(const std::vector<double>& actual,
                         const std::vector<double>& expected,
                         int method, int blocks, int threads, int iterations, int n)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        // Copyout leaves unwritten a[0:N] values undefined; values outside N
        // must still retain their original host data.
        if (method == 3 && i > 0 && i < (size_t)n) continue;
        if (actual[i] != expected[i])
        {
            fprintf(stderr, "Method %d B=%d T=%d I=%d N=%d index=%zu: value=%g expected=%g\n",
                    method, blocks, threads, iterations, n, i, actual[i], expected[i]);
            exit(2);
        }
    }
}

static void verify_reduction(int method, int blocks, int threads, int iterations,
                             int size, int n, int reps)
{
    std::vector<double> a(size, -99.0), buckets(n), initial(n);
    for (int j = 0; j < n; ++j) initial[j] = buckets[j] = j + 0.25;
    double *d_a = NULL, *d_tmp = NULL, *d_private = NULL, *d_team = NULL;
    check_hip(hipMalloc((void **)&d_a, size * sizeof(double)), "test array");
    check_hip(hipMalloc((void **)&d_tmp, n * sizeof(double)), "test buckets");
    check_hip(hipMalloc((void **)&d_private, (size_t)blocks * threads * n * sizeof(double)), "test thread buckets");
    check_hip(hipMalloc((void **)&d_team, (size_t)blocks * n * sizeof(double)), "test team buckets");
    copy_host_to_device(d_a, a.data(), size);
    copy_host_to_device(d_tmp, buckets.data(), n);
    check_hip(hipMemset(d_private, 0xff, (size_t)blocks * threads * n * sizeof(double)), "poison thread buckets");
    check_hip(hipMemset(d_team, 0xff, (size_t)blocks * n * sizeof(double)), "poison team buckets");
    reset_trace();
    for (int r = 0; r < reps; ++r)
    {
        if (method == 8)
            atomic_kernel<<<blocks, threads>>>(4, d_a, d_tmp, iterations, n);
        else
            reduction_kernel<<<blocks, threads>>>(4, d_a, d_tmp, d_private, d_team, iterations, n);
        check_hip(hipGetLastError(), "test bucket launch");
        check_hip(hipDeviceSynchronize(), "test bucket synchronize");
    }
    verify_trace(method, blocks, threads, iterations, size, n, reps);
    copy_device_to_host(buckets.data(), d_tmp, n);
    for (int j = 0; j < n; ++j)
    {
        double expected = initial[j] + reps * (iterations / n + (j < iterations % n));
        if (buckets[j] != expected)
        {
            fprintf(stderr, "Method %d B=%d T=%d I=%d N=%d bucket=%d: value=%g expected=%g\n",
                    method, blocks, threads, iterations, n, j, buckets[j], expected);
            exit(3);
        }
    }
    copy_device_to_host(a.data(), d_a, size);
    std::vector<double> expected_a(size, -99.0);
    for (int i = 0; i < iterations; ++i) expected_a[i] = 7.0;
    verify_array(a, expected_a, method, blocks, threads, iterations, n);
    if (method == 9)
    {
        std::vector<double> team((size_t)blocks * n), local((size_t)blocks * threads * n);
        copy_device_to_host(team.data(), d_team, blocks * n);
        copy_device_to_host(local.data(), d_private, blocks * threads * n);
        for (int b = 0; b < blocks; ++b)
        {
            std::vector<double> expected_team(n, 0.0);
            int last_chunk = -1;
            for (int base = b * threads; base < iterations; base += blocks * threads)
            {
                last_chunk = base;
                for (int t = 0; t < threads && base + t < iterations; ++t)
                    expected_team[(base + t) % n] += 1.0;
            }
            for (int j = 0; j < n; ++j)
                if (team[(size_t)b * n + j] != expected_team[j])
                {
                    fprintf(stderr, "Method 9 B=%d T=%d I=%d N=%d team=%d bucket=%d: team value=%g expected=%g\n",
                            blocks, threads, iterations, n, b, j, team[(size_t)b * n + j], expected_team[j]);
                    exit(3);
                }
            // Teams with no chunk do not use or initialise thread scratch.
            if (last_chunk < 0) continue;
            for (int t = 0; t < threads; ++t)
            for (int j = 0; j < n; ++j)
            {
                double expected = last_chunk + t < iterations && (last_chunk + t) % n == j ? 1.0 : 0.0;
                size_t offset = ((size_t)b * threads + t) * n + j;
                if (local[offset] != expected)
                {
                    fprintf(stderr, "Method 9 B=%d T=%d I=%d N=%d team=%d thread=%d bucket=%d: last chunk value=%g expected=%g\n",
                            blocks, threads, iterations, n, b, t, j, local[offset], expected);
                    exit(3);
                }
            }
        }
    }
    check_hip(hipFree(d_team), "test free team buckets");
    check_hip(hipFree(d_private), "test free thread buckets");
    check_hip(hipFree(d_tmp), "test free buckets");
    check_hip(hipFree(d_a), "test free array");
}

static void run_case(int blocks, int threads, int iterations, int n, int requested_size = 0)
{
    const int reps = 2;
    const int size = requested_size > 0 ? requested_size :
        std::max(120, std::max(iterations, blocks * threads + threads));
    backend_config_t config = {blocks, threads};
    for (int method = 0; method < 12; ++method)
    {
        std::vector<double> a(size), expected(size);
        for (int i = 0; i < size; ++i) a[i] = expected[i] = 1000 + i;
        checked_method = method; checked_n = n;
        checked_iterations = iterations; checked_reps = reps; result_callbacks = 0;
        reset_trace();
        backend_run_method(method, a.data(), n, 4, iterations, size, &config, reps);
        if (result_callbacks != (method >= 1 && method <= 4 ? 0 : 1))
        {
            fprintf(stderr, "Method %d backend result callback count=%d\n", method, result_callbacks);
            exit(3);
        }
        verify_trace(method, blocks, threads, iterations, size, n, reps);
        if (method == 2 || method == 4) expected[0] += reps;
        else if (method == 1 || method == 3) expected[0] = 8.0;
        else
            for (int b = 0; b < blocks; ++b)
            for (int t = 0; t < threads; ++t)
            {
                std::vector<int> indices = expected_indices(method, b, t, blocks, threads,
                                                            iterations, size, n, reps);
                for (size_t k = 0; k < indices.size(); ++k) expected[indices[k]] = 7.0;
            }
        verify_array(a, expected, method, blocks, threads, iterations, n);
        std::vector<unsigned char> before(a.size() * sizeof(double));
        memcpy(before.data(), a.data(), before.size());
        result_callbacks = 0;
        reset_trace();
        double device_us = backend_run_device_method(method, a.data(), n, 4, iterations, size, &config, reps);
        if (!std::isfinite(device_us) || device_us < 0.0 ||
            memcmp(before.data(), a.data(), before.size()) ||
            result_callbacks != (method >= 1 && method <= 4 ? 0 : 1))
        {
            fprintf(stderr, "Method %d invalid event time, profile clone or profile dataflow\n", method);
            exit(5);
        }
        verify_trace(method, blocks, threads, iterations, size, n, reps);
        if (method == 8 || method == 9)
            verify_reduction(method, blocks, threads, iterations, size, n, reps);
    }
    printf("PASS: B=%d T=%d I=%d N=%d, methods 0-11, participant/iteration traces and data flow\n",
           blocks, threads, iterations, n);
}

int main(void)
{
    backend_init(0, NULL);
    if (!backend_device_available()) return 1;
    for (int blocks = 1; blocks <= 3; ++blocks)
    for (int threads = 1; threads <= 5; threads += 2)
    for (int n = 2; n <= 7; n += 5)
        run_case(blocks, threads, 23, n);
    const int small_iterations[] = {0, 1, 4, 5, 6, 14, 15, 16, 31};
    for (size_t c = 0; c < sizeof(small_iterations) / sizeof(small_iterations[0]); ++c)
        run_case(3, 5, small_iterations[c], 7);
    run_case(304, 64, 6656, 7, 65536);
    // MI300X dimensions: empty teams, a partial final chunk, exact full-grid
    // coverage and more than two complete rounds of team chunks.
    const int amd_iterations[] = {13, 19455, 19456, 19461, 38941};
    for (size_t c = 0; c < sizeof(amd_iterations) / sizeof(amd_iterations[0]); ++c)
        run_case(304, 64, amd_iterations[c], 17);
    backend_finalise();
    puts("PASS: all 12 HIP methods; exact owner, per-thread order/count, array values and thread/team/global reduction buckets");
    return 0;
}
