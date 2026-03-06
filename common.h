#ifndef COMMON_H
#define COMMON_H

void init(int argc, char **argv);
void finalise(void);
double get_time_usec(void);

#pragma acc routine seq
void array_delay(int delaylength, double *array);

#endif