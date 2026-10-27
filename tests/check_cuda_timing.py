#!/usr/bin/env python3
"""Test actual CUDA host orchestration with a fake runtime (no GPU timing claim).
Ensure optional events surround only the kernel, use its stream, precede the
original wait, and the independent profiling API leaves caller data untouched.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADER = r'''
#pragma once
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#define __device__
#define __global__
struct dim { unsigned int x; };
static dim blockIdx={0},threadIdx={0},blockDim={1},gridDim={1};
static void __syncthreads() {}
static double atomicAdd(double *p,double v) { double old=*p; *p+=v; return old; }
typedef int cudaError_t;
static const int cudaSuccess=0;
typedef void *cudaStream_t;
struct FakeEvent { size_t position; cudaStream_t stream; };
typedef FakeEvent *cudaEvent_t;
struct cudaDeviceProp { char name[64]; int multiProcessorCount,maxGridSize[3],maxThreadsDim[3],maxThreadsPerBlock; };
static std::vector<std::string> calls;
static cudaStream_t kernel_stream;
static size_t wait_position;
static int event_records, event_creates, elapsed_calls;
static const int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
static const char *cudaGetErrorString(cudaError_t) { return "fake runtime failure"; }
static cudaError_t cudaGetDeviceCount(int *p) { *p=1; return 0; }
static cudaError_t cudaGetDevice(int *p) { *p=0; return 0; }
static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p,int) {
    strcpy(p->name,"fake"); p->multiProcessorCount=132;
    p->maxGridSize[0]=100000; p->maxThreadsPerBlock=1024; p->maxThreadsDim[0]=1024; return 0;
}
static cudaError_t cudaMalloc(void **p,size_t n) { *p=calloc(n,1); calls.push_back("allocate"); return *p?0:1; }
static cudaError_t cudaFree(void *p) { free(p); calls.push_back("free"); return 0; }
static cudaError_t cudaMemcpy(void *to,const void *from,size_t n,int kind) {
    memcpy(to,from,n); calls.push_back(kind==1?"H2D":"D2H"); return 0;
}
static cudaError_t cudaStreamCreate(cudaStream_t *p) { *p=(void*)1; calls.push_back("create stream"); return 0; }
static cudaError_t cudaStreamDestroy(cudaStream_t) { calls.push_back("destroy stream"); return 0; }
static cudaError_t cudaGetLastError() { calls.push_back("launch check"); return 0; }
static cudaError_t cudaDeviceSynchronize() { wait_position=calls.size(); calls.push_back("wait device"); return 0; }
static cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (stream!=kernel_stream) abort(); wait_position=calls.size(); calls.push_back("wait stream"); return 0;
}
static cudaError_t cudaEventCreate(cudaEvent_t *p) { *p=new FakeEvent; ++event_creates; return 0; }
static cudaError_t cudaEventDestroy(cudaEvent_t p) { delete p; return 0; }
static cudaError_t cudaEventRecord(cudaEvent_t p,cudaStream_t stream) {
    p->position=calls.size(); p->stream=stream; ++event_records; calls.push_back("event"); return 0;
}
static void fake_launch(const char *,unsigned int,unsigned int,cudaStream_t stream) {
    kernel_stream=stream; calls.push_back("kernel");
}
static cudaError_t cudaEventElapsedTime(float *ms,cudaEvent_t start,cudaEvent_t stop) {
    if (stop->position!=start->position+2 || calls[start->position+1]!="kernel" ||
        start->stream!=kernel_stream || stop->stream!=kernel_stream || wait_position<=stop->position) abort();
    // Fake elapsed values only verify unit conversion and averaging.
    *ms=0.002f; ++elapsed_calls; return 0;
}
'''
MAIN = r'''
int main(int argc, char **argv) {
    unsetenv("CUDA_OPENMP_PROFILE"); backend_init(0,NULL);
    backend_config_t cfg={2,3};
    for (int method=0;method<12;++method) {
        double a[64], original[64]; for(int i=0;i<64;++i) a[i]=original[i]=1000+i;
        calls.clear(); event_records=event_creates=elapsed_calls=0;
        backend_run_method(method,a,7,4,23,64,&cfg,3);
        if (event_records || event_creates || elapsed_calls) abort();
        memcpy(a,original,sizeof(a)); calls.clear();
        double device_us=backend_run_device_method(method,a,7,4,23,64,&cfg,3);
        if (event_records!=6 || event_creates!=2 || elapsed_calls!=3 || fabs(device_us-2.0)>0.0001 ||
            memcmp(a,original,sizeof(a))) abort();
    }
    if (argc != 2) abort();
    char option[4096]; snprintf(option,sizeof(option),"OpenMPProfile=%s",argv[1]);
    char *options[]={(char*)"test",option}; backend_init(2,options);
    backend_config_t requested={3,5};
    const int profiled_methods[]={5,6,10,11};
    for (int method:profiled_methods) {
        double a[64]; for(int i=0;i<64;++i) a[i]=1000+i;
        calls.clear(); event_records=event_creates=elapsed_calls=0;
        backend_run_device_method(method,a,7,4,23,64,&requested,3);
        int blocks,threads; backend_cuda_last_launch_config(&blocks,&threads);
        if (blocks!=2 || threads!=(method==5?1:2) || event_records!=6 || elapsed_calls!=3 ||
            strcmp(backend_cuda_alignment_mode(),"observed_owners_inferred_chunks")) abort();
    }
    puts("PASS: all 12 actual host paths; events contain only kernel; same stream; stop before wait; baseline has no events; clone preserved; averaging/unit conversion");
}
'''
source=(ROOT/'src/backends/cuda/backend.cu').read_text()
# Retain original host statements and arguments, replacing only launch syntax.
def launch(match):
    dims=[part.strip() for part in match.group(2).split(',')]
    stream=dims[3] if len(dims)==4 else '0'
    return f'fake_launch("{match.group(1)}", {dims[0]}, {dims[1]}, {stream});'
source=re.sub(r'(\w+)<<<(.*?)>>>\((.*?)\);',launch,source,flags=re.S)
with tempfile.TemporaryDirectory(prefix='cuda-timing-') as directory:
    d=Path(directory); (d/'cuda_runtime.h').write_text(HEADER)
    (d/'test.cpp').write_text('#include <cmath>\n'+source+MAIN)
    compiler=os.environ.get('CXX',shutil.which('g++-14') or 'c++')
    subprocess.run([compiler,'-std=c++11','-O0','-I',str(d),'-I',str(ROOT/'src/core'),
                    '-I',str(ROOT/'src/backends/cuda'),str(d/'test.cpp'),'-o',str(d/'test')],check=True)
    rows=['method_id,N,max_iter,max_array_size,requested_team_count,requested_thread_count,actual_team_count,actual_thread_count,team_id,thread_id,index,call_count']
    for method in (5,6,10,11):
        teams,threads=2,1 if method==5 else 2
        owners=[]
        if method==5: owners=[(b,0,b*64//3,1) for b in range(2)]
        elif method==6: owners=[(0 if i<12 else 1, i%2, i, 1) for i in range(23)]
        else: owners=[(b,t,b*(64//3 if method==10 else 5)+t,7 if method==11 else 1) for b in range(2) for t in range(2)]
        rows += [','.join(map(str,(method,7,23,64,3,5,teams,threads,b,t,i,count))) for b,t,i,count in owners]
    profile=d/'profile.csv'; profile.write_text('\n'.join(rows)+'\n')
    subprocess.run([str(d/'test'),str(profile)],check=True)
