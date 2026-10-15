GH200 method0 counter-initial-value diagnostic, 2026-10-15

The preceding entry-gate diagnostic (commit 18d90d7) did not alter the
instruction/control encodings in any of the three delay loops. It stopped
before timing, so it provides no slope comparison.

This diagnostic changes the source of the loop counter's initial value.
The unmodified saved CUDA source is the control. A generated candidate copy
initializes i from a volatile device integer that remains zero in the timed
executable, instead of the literal zero. The selected thread, loop bound,
increment, floating-point operations, output array, launches, synchronization,
driver configuration and timing windows retain their original source code.
The new read occurs once per delay call, before its loop; only method0 runs.
No production backend or driver is edited. OpenMP reuses the saved executable.

This directly targets the uniform-register induction hypothesis. CUDA may use
ordinary per-thread integer instructions when it cannot prove that the initial
value is uniform across threads. This is a hypothesis to check in the captured
SASS, not a guarantee about the compiler.

Validation executables check the default-zero computation, untouched array
elements, repeated calls and preservation of the host array by device timing.
The candidate validator additionally sets the counter initial value to 2 and
checks that a delay of 17 produces 136, to verify the variable is actually used.
That test occurs in a separate executable. Timed executables never write the
variable; they retain its device initializer zero.

Before timing, the script requires the control's main-loop features to match
the previous saved CUDA O3 binary, the candidate to change those features,
matching conversion/add counts, no loop-body global/local/shared memory
instructions, and no UIADD3 instructions in the candidate main loop together
with more than one ordinary integer-add instruction. Otherwise it records
CODEGEN_NOT_ISOLATED and skips timing. This outcome does not reject the
instruction-generation hypothesis.

READY_FOR_TIMING runs three rotating trials of original OpenMP, CUDA control
and CUDA counter candidate. The original driver constants remain unchanged:
Method=0 N=16382 Delay=1,8192 MAX_ITER=16382 MAX_ARRAY_SIZE=65536
Requested team/block_count=132 thread_count=32; actual CUDA method0 launch=1x1.
CUDA event measurement is a separate pass. Host raw data stay in baseline/;
CUDA event data stay in kernel_execution/. No plotting occurs.

Output: result/GH200/counter_probe/<new timestamp>/
Read codegen_gate.json and device_code/*_main_loop.txt first. Slope CSVs exist
only if the code-generation gate passed. SD in slope_summary.csv describes
three trial slopes; it is not a confidence interval.

A slope change would establish sensitivity to the intervention. It would not
prove that a single instruction or register class explains the whole OpenMP
gap: unrolling, register allocation, instruction scheduling and layout can
also change. A fixed-cost change alone would not explain a slope difference.

Run from GH200:
bash tests/run_gh200_counter_probe.sh

Local checks cover script syntax, patch scope, captured SASS parsing, synthetic
data validation and a mocked shell flow. CUDA compilation, GPU numerical
correctness and actual code generation must pass on GH200 before timing.
