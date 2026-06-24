#include <stdio.h>
#include <sys/time.h>
#include "common.h"

void init(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    printf("Initializing benchmark runtime environment...\n");
}

#pragma acc routine seq
void delay_kernel(int delaylength, double *array)
{
    array[0] = 1.0;
    for (int i = 0; i < delaylength; i++)
    {
        array[0] += i;
    }
    if (array[0] < 0.0)
        printf("%f\n", array[0]);
}

double get_time_usec(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1.0e6 + tv.tv_usec;
}

void finalise(void)
{
    printf("Finalizing benchmark.\n");
}
