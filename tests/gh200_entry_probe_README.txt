GH200 method0 entry-selection diagnostic, 2026-10-15

Purpose
Test whether a small entry-selection change changes the CUDA delay loop's
integer instructions / uniform-register use and its measured slope.
This is a diagnostic, not a modification to the production benchmark.

The script uses the saved device_opt_ab/20261014T202550Z.XUuTBe source snapshot
and original OpenMP executable. An optional first argument selects another
snapshot with the same directory layout. Two new CUDA executables use the
same host -O0 and ptxas -O3 options.

The control is the unmodified saved CUDA source. The candidate adds a volatile
device integer initialized to 1 and reads it once in method0's entry condition.
Both launch one block with one thread. The delay function, iteration count,
repetitions, launch sites, synchronization and host/device timing windows are
unchanged. The candidate includes an extra device read before the delay loop;
this can change the intercept and can influence register allocation/code layout.
It is not an isolated measurement of one individual SASS opcode.

Before timing, both CUDA variants run separate numerical checks for delays
0,1,2,3,4,15,16,17,1024,8192, repetitions 1 and 3, untouched array elements and
profiling host-state preservation. The candidate also checks that a zero gate
suppresses the computation. These validation executables are not timed.

The script dumps PTX and SASS. It runs the timing experiment only when:
- The rebuilt control's main-loop feature signature matches the saved control.
- The candidate changes the main-loop feature signature.
- Both main loops have matching conversion/add counts and no LDG/STG/LDL/STL/
  LDS/STS instructions.
This feature check is not a proof of equal cycle counts or semantic equivalence;
the source-scoped patch and numerical checks provide separate evidence.

If this gate fails, codegen_gate.json says CODEGEN_NOT_ISOLATED and no timing
runs are performed. Push the diagnostic records anyway. This outcome means the
chosen intervention did not establish the intended comparison; it does not
reject the instruction-generation hypothesis.

Otherwise three rotating trials measure original OpenMP, CUDA control and CUDA
entry gate. Only Method=0 is run. Configuration:
N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536
Requested team/block count=132, thread_count=32
Actual CUDA method0 launch=1 block x 1 thread
Driver warmups, inner repetitions and 2 sets x 5 runs are unchanged.
CUDA event data are produced in a separate pass and moved to kernel_execution/.
Baseline host timings remain in baseline/. No plots are generated.

Results: result/GH200/entry_probe/<new timestamp>/
Primary files: codegen_gate.json, source_change.diff, device_code/*_main_loop.txt,
check_*.log, slope_per_trial.csv, slope_summary.csv, summary.log.
An unchanged-loop outcome does not contain slope CSVs or measurements/.

Interpretation
- A reproducible device-slope change establishes sensitivity to this entry/code
  generation intervention, not a unique cause for the OpenMP/CUDA difference.
- If only the intercept changes, this intervention has not explained the slope
  difference.
- Numerical mismatch or unexpected hot-loop memory instructions stops the run.
- No OpenACC or method6 experiment is added here.

Local validation
Python syntax, Bash syntax, exact source patch scope and captured SASS parsing
were checked. Synthetic full 2100-row data verified slope units, sample coverage
and duplicate rejection. A mocked shell workflow checked both pre-timing stop
and complete three-trial rotation, data partition and production source hashes.
The mock tests do not compile CUDA or establish any GPU result. Actual CUDA
compilation and numerical checks run on GH200 before any timing experiment.
