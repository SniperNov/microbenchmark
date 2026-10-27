#!/usr/bin/env python3
"""Compile the actual CUDA-enabled driver against deterministic CPU test hooks.
Verify device samples are a second pass, warmups are excluded, and the original
raw CSV is byte-identical with device profiling switched off/on.
"""
from pathlib import Path
import csv
import os
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
STUB=r'''
#include <stdio.h>
#include <stdlib.h>
#include "backend.h"
static int host_calls, device_calls;
const char *backend_name(void) { return "cuda"; }
int backend_num_methods(void) { return 12; }
const char *backend_method_name(int method) { (void)method; return "test kernel"; }
const char *backend_control_a_name(void) { return "block_count"; }
const char *backend_control_b_name(void) { return "thread_count"; }
void backend_default_config(backend_config_t *config) { config->control_a=2; config->control_b=3; }
int backend_device_available(void) { return 1; }
void backend_init(int argc,char **argv) { (void)argc; (void)argv; }
void backend_finalise(void) { FILE *f=fopen("calls.log","w"); fprintf(f,"%d %d\n",host_calls,device_calls); fclose(f); }
double backend_run_method(int method,double *a,int n,int delay,int iterations,int size,
                          const backend_config_t *config,int reps) {
    (void)method; (void)iterations; (void)size; (void)config; (void)reps;
    ++host_calls; a[0]+=1; return n+2.0*delay;
}
const char *backend_cuda_alignment_mode(void) { return "cuda_static_cyclic_unverified"; }
void backend_cuda_last_launch_config(int *blocks,int *threads) { *blocks=1; *threads=1; }
double backend_run_device_method(int method,const double *a,int n,int delay,int iterations,int size,
                                 const backend_config_t *config,int reps) {
    (void)method; (void)a; (void)n; (void)iterations; (void)size; (void)config; (void)reps;
    // All sixty measured baseline rows must exist before ANY device sample.
    FILE *f=fopen("raw_times.csv","r"); if (!f) abort();
    int lines=0,ch; while ((ch=fgetc(f))!=EOF) if(ch=='\n') ++lines; fclose(f);
    if(lines!=61) abort();
    ++device_calls; return 0.25*delay;
}
'''
with tempfile.TemporaryDirectory(prefix='cuda-driver-') as directory:
    d=Path(directory); stub=d/'stub.c'; stub.write_text(STUB)
    compiler=os.environ.get('CC',shutil.which('gcc-14') or 'gcc')
    subprocess.run([compiler,'-O0','-DMICROBENCHMARK_CUDA','-I',str(ROOT/'src/core'),
                    str(ROOT/'src/core/driver.c'),str(stub),'-lm','-o',str(d/'driver')],check=True)
    raws=[]
    for enabled in (0,1):
        run=d/str(enabled); run.mkdir()
        result=subprocess.run([str(d/'driver'),'Method=0','N=7','Delay=1,32',f'DeviceTiming={enabled}'],
                              cwd=run,text=True,capture_output=True)
        if result.returncode: raise RuntimeError(result.stdout+result.stderr)
        raws.append((run/'raw_times.csv').read_bytes())
        with (run/'raw_times.csv').open() as f: rows=list(csv.DictReader(f))
        assert len(rows)==60 and len(rows[0])==10
        host,device=map(int,(run/'calls.log').read_text().split())
        assert host==(180 if enabled else 120) and device==(60 if enabled else 0)
        if enabled:
            with (run/'cuda_device_times.csv').open() as f: samples=list(csv.DictReader(f))
            assert len(samples)==60 and len(samples[0])==17
            assert {float(row['device_kernel_us']) for row in samples}=={0.25,0.5,1,2,4,8}
            assert all(row['measurement_mode']=='same_stream_events_separate_pass' for row in samples)
            assert all(row['inner_reps']=='20' for row in samples)
        else: assert not (run/'cuda_device_times.csv').exists()
    assert raws[0]==raws[1]
print('PASS: actual driver preserves raw CSV; complete baseline precedes optional device pass; warmups excluded; workload/sample metadata correct')
