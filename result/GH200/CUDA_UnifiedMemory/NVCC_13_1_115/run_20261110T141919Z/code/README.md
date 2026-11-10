# GH200 CUDA Unified Memory: all twelve methods

Status: native compilation, correctness and formal collection completed on GH200; all three modes independently verified. This is distinct from the November9 independent full-array probe.

All twelve original CUDA kernel bodies are included unchanged from backend.cu. A separate backend_managed.cu supplies memory policy and launch dispatch. Shared input arrays and reduction buffers use cudaMallocManaged. Three modes: managed_demand, managed_prefetch, managed_advice_prefetch. No cudaMemcpy is called by the managed execution path. The included explicit backend remains unchanged and is not the managed dispatch path.

Important experimental contract:
- Managed arrays are allocated and CPU-initialized outside the timed region. Allocation/free and one-time preferred-location advice are not measured, including M1–M4. This differs from the explicit-copy M1–M4 allocation/copy/free experiment and must be labeled in any later comparison.
- M1 requests GPU input and CPU output prefetch; M2 requests GPU input only; M3 requests CPU output only; M4 makes no placement requests. Demand mode makes none for any method. Advice is a hint; prefetch can physically move pages.
- For M0 and M5–M11, prefetch mode prepares managed input, reduction buffers and workspaces before timing. Demand mode allows demand access during execution.
- Each iteration waits for kernel completion then performs the original host a[0] increment. That increment now changes the same shared array and can induce page migration or remote access. It remains inside timing. No artificial removal of this access is made.
- Shared memory is coherent; M2/M4 do not hide GPU writes from the CPU. All touched elements are checked according to shared-memory semantics, not explicit copyin/create semantics. Thus these are UM variants of the computations, not claims of identical map-direction semantics.
- N remains array/reduction extent (M11 repetition count); original kernels, exact iteration count, reduction algorithm and requested grouping are retained. No padding of M9, no dropped methods.
- No claims about physical migration bytes, page residency, pure GPU kernel time or one-block-per-SM placement. No device event profiling or performance analysis in this task.

Build/execute after obtaining an authorized idle GH200 session:
`python3 -u run_suite.py`
Set NVCC to the CUDA13.1.115 compiler (default: nvcc from PATH); -O0 -arch=sm_90. All three modes must pass33geometry cases x12methods,240smoke rows and20exact-I regression rows before any formal mode runs. Native instrumentation tests calls, owners, order and full arrays; reduction checks include buckets. Instrumentation is separate from the formal binary.

Formal grid: per mode12methods,8groups,71method/Nconfigurations,9940rows;14delays1..8192,2sets5runs,inner20,warmup10,outer1. Requested132blocks32threads; MAX_ITER4224 except M1–4=65536, MAX_ARRAY_SIZE65536. Three modes yield29820rows. Results and manifests label memory mode. Failure stops collection. Hardware identity, source and binary hashes are checked before/after.

The CPU-only test replaces launch syntax and simulates a single thread; it checks CUDA12/13 API signatures, managed allocation policy, absence of explicit copy calls and directional prefetch calls. It does not prove native compilation or GPU parallel correctness.
