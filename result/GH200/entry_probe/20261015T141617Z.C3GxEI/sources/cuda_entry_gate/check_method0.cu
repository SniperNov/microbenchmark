
#include <cmath>
#include <cstring>
#include <vector>
#include "src/backends/cuda/backend.cu"
int main(int argc, char **argv) {
    backend_init(argc, argv);
    backend_config_t config = {132, 32};
    const int delays[] = {0,1,2,3,4,15,16,17,1024,8192};
    for (int delay : delays) {
        for (int reps : {1,3}) {
            std::vector<double> a(32, -19.0);
            backend_run_method(0, a.data(), 7, delay, 23, 32, &config, reps);
            double expected = 1.0 + (double)delay*(delay-1)/2;
            if (a[0] != expected) return 3;
            for (int i=1; i<32; ++i) if (a[i] != -19.0) return 4;
            std::vector<double> before = a;
            double us = backend_run_device_method(0, a.data(), 7, delay, 23, 32, &config, reps);
            if (!(std::isfinite(us) && us>0) || a != before) return 5;
        }
    }
#ifdef ENTRY_GATE_CHECK
    int zero = 0;
    check_cuda(cudaMemcpyToSymbol(diagnostic_entry_gate, &zero, sizeof(zero)), "close diagnostic gate");
    std::vector<double> a(32, -19.0);
    backend_run_method(0, a.data(), 7, 17, 23, 32, &config, 1);
    for (double value : a) if (value != -19.0) return 6;
#endif
    backend_finalise();
    puts("PASS: method0 values, untouched array elements, repeated calls, profiling host-state preservation");
    return 0;
}
