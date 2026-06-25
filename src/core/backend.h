#ifndef MICROBENCHMARK_BACKEND_H
#define MICROBENCHMARK_BACKEND_H

typedef struct
{
    // Backend-specific launch controls.
    // OpenMP:  control_a=thread_count, control_b=team_count
    // OpenACC: control_a=gang_count,   control_b=worker_count
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

#endif
