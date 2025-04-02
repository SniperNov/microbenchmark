#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "common.h"

#define MAX_ARRAY_SIZE 1024 // Ensure it doesn't exceed the allowed memory section

double a[MAX_ARRAY_SIZE]; // Define an array with a proper size

void init(int argc, char **argv)
{
    for (int i = 0; i < MAX_ARRAY_SIZE; i++)
    {
        a[i] = 0.0;
    }
}

void array_delay(int delaylength, double *array)
{
    // write initial values here
    for (int i = 0; i < delaylength; i++)
    {
        array[i % MAX_ARRAY_SIZE] += 1.0; // Ensure access stays within array bounds
    }
}


double get_time_usec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}


void finalise()
{
    printf("Finalizing benchmark.\n");
}
