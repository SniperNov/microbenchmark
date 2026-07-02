#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <exception>
#include <sycl/sycl.hpp>
#include "backend.h"

namespace
{

static const char *method_names[] = {
    "pure delay kernel",
    "USM malloc + H2D + kernel + D2H + free",
    "USM malloc + H2D + kernel + free",
    "USM malloc + kernel + D2H + free",
    "USM malloc + kernel + free",
    "nd_range groups scalar",
    "nd_range parallel_for loop",
    "event async submit + wait",
    "nd_range atomic_ref",
    "nd_range local reduction",
    "nd_range explicit groups + local_size",
    "nd_range repeated groups + local_size"};

static double get_time_usec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1.0e6 + tv.tv_usec;
}

static void die_sycl(const char *where, const std::exception &ex)
{
    fprintf(stderr, "SYCL error at %s: %s\n", where, ex.what());
    exit(EXIT_FAILURE);
}

static sycl::queue &get_queue()
{
    // One queue means one command line to the selected accelerator.
    // Think of it like a CUDA stream plus device selector.
    static sycl::queue q{sycl::default_selector_v};
    return q;
}

static inline void delay_body(int delaylength, double *array)
{
    // Tiny artificial workload used by every backend.
    // The writes keep the compiler from deleting the loop.
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
        array[0] += i;
}

static sycl::nd_range<1> make_launch_shape(int group_count, int local_size)
{
    // SYCL global range = all work-items.
    // SYCL local range  = work-items inside one work-group.
    // So this is the SYCL version of CUDA <<<group_count, local_size>>>.
    return sycl::nd_range<1>(
        sycl::range<1>((size_t)group_count * (size_t)local_size),
        sycl::range<1>((size_t)local_size));
}

static void copy_host_to_device(sycl::queue &q, double *d_a, const double *a, int count)
{
    // H2D means host-to-device: CPU memory -> GPU/device memory.
    q.memcpy(d_a, a, (size_t)count * sizeof(double)).wait();
}

static void copy_device_to_host(sycl::queue &q, double *a, const double *d_a, int count)
{
    // D2H means device-to-host: GPU/device memory -> CPU memory.
    q.memcpy(a, d_a, (size_t)count * sizeof(double)).wait();
}

static void submit_one_task(sycl::queue &q, int delay, double *d_a)
{
    // single_task launches one device task. This is our smallest launch.
    q.single_task([=]() {
         delay_body(delay, d_a);
     }).wait();
}

static void submit_group_scalar(sycl::queue &q, int delay, double *d_a,
                                int group_count, int max_array_size)
{
    // One work-item per work-group does useful work.
    // This matches the "teams/gangs scalar" idea.
    q.parallel_for(make_launch_shape(group_count, 1), [=](sycl::nd_item<1> item) {
         if (item.get_local_id(0) == 0)
         {
             int group_id = (int)item.get_group_linear_id();
             int idx = group_id * max_array_size / group_count;
             delay_body(delay, &d_a[idx]);
         }
     }).wait();
}

static sycl::event submit_loop(sycl::queue &q, int delay, double *d_a,
                               int max_iter, int group_count, int local_size)
{
    // nd_item is SYCL's "who am I?" object inside an nd_range kernel.
    // global_id is like CUDA's blockIdx.x * blockDim.x + threadIdx.x.
    // global_range is total work-items, used as the grid-stride.
    return q.parallel_for(make_launch_shape(group_count, local_size), [=](sycl::nd_item<1> item) {
        int gid = (int)item.get_global_id(0);
        int stride = (int)item.get_global_range(0);
        for (int i = gid; i < max_iter; i += stride)
            delay_body(delay, &d_a[i]);
    });
}

static void submit_atomic(sycl::queue &q, int delay, double *d_a, double *d_tmp,
                          int max_iter, int N, int group_count, int local_size)
{
    q.parallel_for(make_launch_shape(group_count, local_size), [=](sycl::nd_item<1> item) {
         int gid = (int)item.get_global_id(0);
         int stride = (int)item.get_global_range(0);
         for (int i = gid; i < max_iter; i += stride)
         {
             delay_body(delay, &d_a[i]);

             // atomic_ref means "many work-items may update this same value;
             // make the update safe instead of racing".
             sycl::atomic_ref<double,
                              sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                 ref(d_tmp[i % N]);
             ref.fetch_add(1.0);
         }
     }).wait();
}

static void submit_reduction(sycl::queue &q, int delay, double *d_a, double *d_tmp,
                             int max_iter, int group_count, int local_size)
{
    q.submit([&](sycl::handler &h) {
         // local_accessor is scratch memory shared inside one work-group.
         // CUDA calls a similar thing "__shared__ memory".
         sycl::local_accessor<double, 1> scratch(sycl::range<1>((size_t)local_size), h);

         h.parallel_for(make_launch_shape(group_count, local_size), [=](sycl::nd_item<1> item) {
             int gid = (int)item.get_global_id(0);
             int stride = (int)item.get_global_range(0);
             int lid = (int)item.get_local_id(0);
             double local_sum = 0.0;

             for (int i = gid; i < max_iter; i += stride)
             {
                 delay_body(delay, &d_a[i]);
                 local_sum += 1.0;
             }

             scratch[lid] = local_sum;
             item.barrier(sycl::access::fence_space::local_space);

             // Tree reduction inside one work-group.
             // Best used with local_size as a power of two, e.g. 128.
             for (int offset = local_size / 2; offset > 0; offset >>= 1)
             {
                 if (lid < offset)
                     scratch[lid] += scratch[lid + offset];
                 item.barrier(sycl::access::fence_space::local_space);
             }

             if (lid == 0)
             {
                 sycl::atomic_ref<double,
                                  sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                     ref(d_tmp[0]);
                 ref.fetch_add(scratch[0]);
             }
         });
     }).wait();
}

static void submit_shape(sycl::queue &q, int delay, double *d_a,
                         int max_array_size, int group_count, int local_size)
{
    q.parallel_for(make_launch_shape(group_count, local_size), [=](sycl::nd_item<1> item) {
         int idx = (int)(item.get_global_id(0) % (size_t)max_array_size);
         delay_body(delay, &d_a[idx]);
     }).wait();
}

static void submit_repeated_shape(sycl::queue &q, int delay, double *d_a,
                                  int max_array_size, int parreps,
                                  int group_count, int local_size)
{
    q.parallel_for(make_launch_shape(group_count, local_size), [=](sycl::nd_item<1> item) {
         int idx = (int)(item.get_global_id(0) % (size_t)max_array_size);
         for (int r = 0; r < parreps; r++)
             delay_body(delay, &d_a[idx]);
     }).wait();
}

} // namespace

extern "C" const char *backend_name(void) { return "sycl"; }
extern "C" int backend_num_methods(void) { return 12; }
extern "C" const char *backend_method_name(int method) { return method_names[method]; }
extern "C" const char *backend_control_a_name(void) { return "group_count"; }
extern "C" const char *backend_control_b_name(void) { return "local_size"; }

extern "C" void backend_default_config(backend_config_t *config)
{
    try
    {
        sycl::queue &q = get_queue();
        config->control_a = (int)q.get_device().get_info<sycl::info::device::max_compute_units>();
    }
    catch (const std::exception &)
    {
        config->control_a = 64;
    }
    config->control_b = 128;
}

extern "C" int backend_device_available(void)
{
    try
    {
        sycl::queue &q = get_queue();
        sycl::device dev = q.get_device();
        printf("SYCL device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        printf("SYCL vendor: %s\n", dev.get_info<sycl::info::device::vendor>().c_str());
        printf("SYCL max compute units: %d\n",
               (int)dev.get_info<sycl::info::device::max_compute_units>());
        return 1;
    }
    catch (const std::exception &ex)
    {
        fprintf(stderr, "No SYCL device available: %s\n", ex.what());
        return 0;
    }
}

extern "C" void backend_init(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    try
    {
        get_queue().wait();
        printf("Initializing SYCL benchmark runtime environment...\n");
    }
    catch (const std::exception &ex)
    {
        die_sycl("backend_init", ex);
    }
}

extern "C" void backend_finalise(void)
{
    try
    {
        get_queue().wait();
        printf("Finalizing SYCL benchmark.\n");
    }
    catch (const std::exception &ex)
    {
        die_sycl("backend_finalise", ex);
    }
}

extern "C" double backend_run_method(int method, double *a, int N, int delay,
                                     int max_iter, int max_array_size,
                                     const backend_config_t *config,
                                     int inner_reps)
{
    sycl::queue &q = get_queue();
    int group_count = config->control_a > 0 ? config->control_a : 1;
    int local_size = config->control_b > 0 ? config->control_b : 1;
    double start = 0.0, end = 0.0;

    try
    {
        if (method >= 1 && method <= 4)
        {
            // Methods 1-4 intentionally allocate/copy/free inside timing.
            // This mirrors OpenMP/OpenACC data clauses and CUDA malloc/copy/free.
            start = get_time_usec();
            for (int rep = 0; rep < inner_reps; rep++)
            {
                double *d_a = sycl::malloc_device<double>((size_t)N, q);
                if (!d_a)
                {
                    fprintf(stderr, "SYCL malloc_device failed for N=%d\n", N);
                    exit(EXIT_FAILURE);
                }

                switch (method)
                {
                case 1:
                    copy_host_to_device(q, d_a, a, N);
                    submit_one_task(q, delay, d_a);
                    copy_device_to_host(q, a, d_a, N);
                    break;

                case 2:
                    copy_host_to_device(q, d_a, a, N);
                    submit_one_task(q, delay, d_a);
                    break;

                case 3:
                    submit_one_task(q, delay, d_a);
                    copy_device_to_host(q, a, d_a, N);
                    break;

                case 4:
                    submit_one_task(q, delay, d_a);
                    break;
                }

                sycl::free(d_a, q);
                a[0] += 1.0;
                if (a[0] < 0.0)
                    printf("%f\n", a[0]);
            }
            end = get_time_usec();
        }
        else if (method == 0 || (method >= 5 && method <= 11))
        {
            // Methods 0 and 5-11 keep memory allocated outside timing.
            // This focuses timing on launch, async, atomic, reduction, and shape.
            double *d_a = sycl::malloc_device<double>((size_t)max_array_size, q);
            double *d_tmp = sycl::malloc_device<double>((size_t)N, q);
            if (!d_a || !d_tmp)
            {
                fprintf(stderr, "SYCL malloc_device failed for main data region\n");
                exit(EXIT_FAILURE);
            }

            copy_host_to_device(q, d_a, a, max_array_size);
            q.memset(d_tmp, 0, (size_t)N * sizeof(double)).wait();

            start = get_time_usec();
            for (int rep = 0; rep < inner_reps; rep++)
            {
                switch (method)
                {
                case 0:
                    submit_one_task(q, delay, d_a);
                    break;

                case 5:
                    submit_group_scalar(q, delay, d_a, group_count, max_array_size);
                    break;

                case 6:
                    submit_loop(q, delay, d_a, max_iter, group_count, local_size).wait();
                    break;

                case 7:
                {
                    // The kernel is submitted first. The returned event is waited on explicitly.
                    sycl::event e = submit_loop(q, delay, d_a, max_iter, group_count, local_size);
                    e.wait();
                    break;
                }

                case 8:
                    submit_atomic(q, delay, d_a, d_tmp, max_iter, N, group_count, local_size);
                    break;

                case 9:
                    q.memset(d_tmp, 0, sizeof(double)).wait();
                    submit_reduction(q, delay, d_a, d_tmp, max_iter, group_count, local_size);
                    break;

                case 10:
                    submit_shape(q, delay, d_a, max_array_size, group_count, local_size);
                    break;

                case 11:
                    submit_repeated_shape(q, delay, d_a, max_array_size, N, group_count, local_size);
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

            copy_device_to_host(q, a, d_a, max_array_size);
            sycl::free(d_tmp, q);
            sycl::free(d_a, q);
        }
    }
    catch (const std::exception &ex)
    {
        die_sycl("backend_run_method", ex);
    }

    return (end - start) / inner_reps;
}
