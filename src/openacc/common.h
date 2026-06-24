#ifndef COMMON_H
#define COMMON_H

void init(int argc, char **argv);
void finalise(void);
double get_time_usec(void);

#pragma acc routine seq
void delay_kernel(int delaylength, double *array);

void compute_offloading_time(double *intercept_avg, double *intercept_err,
                             double *min_avg, double *min_err,
                             int method_id, const char *method_name, int N);
void warmup_cache(int method, int N, int gang_count, int vector_length);
void device_target(int method, int set, int run, double *a, int N,
                   int gang_count, int vector_length);

#endif
