#ifndef COMMON_H
#define COMMON_H

void init(int argc, char **argv);
void delay(int delaylength);
void array_delay(int delaylength, double a[1]);
double get_time_usec();
void finalise(void);

#endif