void init(int argc, char **argv);
#pragma omp declare target
void delay(int delaylength);

void array_delay(int delaylength, double a[1]);
#pragma omp end declare target
void finalise(void);