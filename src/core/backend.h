#ifndef MICROBENCHMARK_BACKEND_H
#define MICROBENCHMARK_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    // Backend-specific launch controls.
    // OpenMP:  control_a=thread_count, control_b=team_count
    // OpenACC: control_a=gang_count,   control_b=worker_count
    // CUDA:    control_a=block_count,  control_b=thread_count
    // HIP:     control_a=block_count,  control_b=thread_count
    // SYCL:    control_a=group_count,  control_b=local_size
    int control_a;
    int control_b;
} backend_config_t;

// Small metadata hooks used by the shared driver for printing, parsing, and CSV headers.
const char *backend_name(void);
int backend_num_methods(void);
const char *backend_method_name(int method);

const char *backend_control_a_name(void);
const char *backend_control_b_name(void);
void backend_default_config(backend_config_t *config);

int backend_device_available(void);
void backend_init(int argc, char **argv);
void backend_finalise(void);

// Run one method for one delay value. The backend owns the API-specific
// target/data/launch construct and returns average time per inner repetition.
double backend_run_method(int method, double *a, int N, int delay,
                          int max_iter, int max_array_size,
                          const backend_config_t *config,
                          int inner_reps);

#ifdef MICROBENCHMARK_CUDA
const char *backend_cuda_alignment_mode(void);
void backend_cuda_last_launch_config(int *blocks, int *threads);
#define BACKEND_DEVICE_LABEL "CUDA"
#define BACKEND_DEVICE_OUTPUT_FILE "cuda_device_times.csv"
#define backend_gpu_alignment_mode backend_cuda_alignment_mode
#define backend_gpu_last_launch_config backend_cuda_last_launch_config
#endif

#ifdef MICROBENCHMARK_HIP
const char *backend_hip_alignment_mode(void);
void backend_hip_last_launch_config(int *blocks, int *threads);
#define BACKEND_DEVICE_LABEL "HIP"
#define BACKEND_DEVICE_OUTPUT_FILE "hip_device_times.csv"
#define backend_gpu_alignment_mode backend_hip_alignment_mode
#define backend_gpu_last_launch_config backend_hip_last_launch_config
#endif

#if defined(MICROBENCHMARK_CUDA) && defined(MICROBENCHMARK_HIP)
#error "Build CUDA and HIP as separate backend executables."
#endif

#if defined(MICROBENCHMARK_CUDA) || defined(MICROBENCHMARK_HIP)
#define MICROBENCHMARK_DEVICE_TIMING 1

// Separate GPU profiling pass. The backend clones a and returns the mean
// same-stream event interval around each kernel, excluding allocation/copies.
// Profiling must leave the baseline host array and timing window unchanged.
double backend_run_device_method(int method, const double *a, int N, int delay,
                                 int max_iter, int max_array_size,
                                 const backend_config_t *config,
                                 int inner_reps);
#endif

#ifdef __cplusplus
}
#endif

#endif
