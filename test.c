#include <stdio.h>
int main() { int x = -1;
    #pragma omp target map(from:x)
    { x = 42;
    }
    printf("x = %d\n", x); return 0;
}

