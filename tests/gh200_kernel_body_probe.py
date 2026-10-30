#!/usr/bin/env python3
"""Diagnostic M0/M6 GPU body timestamps, using a copy of the archived source.

Run on GH200/Hopper. Control, stores-only and globaltimer arms preserve the
driver's original sampling, launch dimensions and launch/synchronize schedule.
No production source is edited; no new kernel barriers/atomics/events are added.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
ARMS = {'control': 0, 'stores_only': 1, 'globaltimer': 2}
TARGETS = ('delay_one_kernel', 'loop_kernel')

HEADER = r'''
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <limits>
#include "backend.h"
struct ProbeStamp { unsigned long long begin, end; };
static_assert(sizeof(ProbeStamp) == 16, "Unexpected timestamp record layout");
__device__ __constant__ ProbeStamp *probe_records;
__device__ __constant__ unsigned int *probe_counts;
__device__ __constant__ unsigned int probe_stride;
__device__ __constant__ unsigned int probe_capacity;
__device__ __forceinline__ unsigned long long probe_now() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}
__device__ __forceinline__ void probe_save(unsigned long long begin, unsigned long long end) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int slot = probe_counts[tid]++;
    if (slot < probe_capacity) {
        probe_records[(size_t)slot * probe_stride + tid].begin = begin;
        probe_records[(size_t)slot * probe_stride + tid].end = end;
    }
}
static void body_probe_begin(int, int, int, int, int, const backend_config_t *, int);
static void body_probe_end(const double *, int, int, int, int, int, const backend_config_t *, int);
#define MICROBENCHMARK_RESIDENT_DATA(a,tmp) body_probe_begin(method,N,delay,max_iter,max_array_size,config,inner_reps)
#define MICROBENCHMARK_RESULT(a,tmp) body_probe_end(a,method,N,delay,max_iter,max_array_size,config,inner_reps)
'''

FOOTER = r'''
static ProbeStamp *host_probe_device_records = nullptr;
static unsigned int *host_probe_device_counts = nullptr;
static unsigned int host_probe_stride = 0, host_probe_capacity = 0;
static FILE *host_probe_csv = nullptr, *host_probe_raw = nullptr;
static unsigned long long host_probe_call = 0, host_probe_raw_offset = 0;
static void probe_require(bool condition, const char *reason) {
    if (!condition) { fprintf(stderr,"BODY PROBE FAILED: %s\n",reason); exit(EXIT_FAILURE); }
}
static void body_probe_begin(int method, int, int, int, int,
                             const backend_config_t *config, int inner_reps) {
    probe_require(method == 0 || method == 6,"Only methods 0 and 6 are supported");
    if (!host_probe_csv) {
        int device; cudaDeviceProp prop;
        check_cuda(cudaGetDevice(&device),"probe device");
        check_cuda(cudaGetDeviceProperties(&prop,device),"probe device properties");
        probe_require(prop.major >= 9,"Use Hopper: do not assume A100 timer resolution");
        host_probe_csv = fopen("body_clock_launches.csv","w");
        probe_require(host_probe_csv != nullptr,"open body CSV");
        fprintf(host_probe_csv,"call_id,method_id,N,delaylength,inner_rep,participants,mode,body_first_start_ns,body_last_end_ns,body_envelope_us,max_thread_span_us,mean_thread_span_us,raw_offset,raw_bytes\n");
        if (BODY_PROBE_MODE == 2) {
            host_probe_raw = fopen("body_clock_records.bin","wb");
            probe_require(host_probe_raw != nullptr,"open body raw records");
        }
    }
    const unsigned int participants = method == 0 ? 1u : (unsigned int)config->control_a * config->control_b;
    probe_require(inner_reps == 20,"Preserve the archived driver INNERREPS=20");
    if (participants != host_probe_stride || (unsigned int)inner_reps != host_probe_capacity) {
        if (host_probe_device_records) check_cuda(cudaFree(host_probe_device_records),"free old probe records");
        if (host_probe_device_counts) check_cuda(cudaFree(host_probe_device_counts),"free old probe counts");
        host_probe_stride = participants; host_probe_capacity = inner_reps;
        check_cuda(cudaMalloc((void **)&host_probe_device_records,(size_t)participants * inner_reps * sizeof(ProbeStamp)),"allocate probe records");
        check_cuda(cudaMalloc((void **)&host_probe_device_counts,(size_t)participants * sizeof(unsigned int)),"allocate probe counters");
        check_cuda(cudaMemset(host_probe_device_records,0,(size_t)participants * inner_reps * sizeof(ProbeStamp)),"clear probe records");
        check_cuda(cudaMemcpyToSymbol(probe_records,&host_probe_device_records,sizeof(host_probe_device_records)),"bind probe records");
        check_cuda(cudaMemcpyToSymbol(probe_counts,&host_probe_device_counts,sizeof(host_probe_device_counts)),"bind probe counters");
        check_cuda(cudaMemcpyToSymbol(probe_stride,&host_probe_stride,sizeof(host_probe_stride)),"bind probe stride");
        check_cuda(cudaMemcpyToSymbol(probe_capacity,&host_probe_capacity,sizeof(host_probe_capacity)),"bind probe capacity");
    }
    // This hook is before the original backend timer and before all 20 launches.
    check_cuda(cudaMemset(host_probe_device_counts,0,(size_t)participants * sizeof(unsigned int)),"reset probe counters");
}
static void body_probe_end(const double *a, int method, int N, int delay, int max_iter,
                           int, const backend_config_t *, int inner_reps) {
    // This hook is after the original backend timer and its result copies.
    const double expected = 1.0 + (double)delay * (delay - 1) / 2.0;
    for (int i=0;i<(method == 0 ? 1 : max_iter);++i)
        probe_require(a[i] == expected,"GPU result changed");
    std::vector<unsigned int> counts(host_probe_stride);
    std::vector<ProbeStamp> stamps((size_t)host_probe_stride * inner_reps);
    check_cuda(cudaMemcpy(counts.data(),host_probe_device_counts,counts.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost),"copy probe counters");
    check_cuda(cudaMemcpy(stamps.data(),host_probe_device_records,stamps.size()*sizeof(ProbeStamp),cudaMemcpyDeviceToHost),"copy probe records");
    for (unsigned int count: counts)
        probe_require(count == (BODY_PROBE_MODE == 0 ? 0u : (unsigned int)inner_reps),"per-thread launch count mismatch");
    for (int rep=0;rep<inner_reps;++rep) {
        unsigned long long first=std::numeric_limits<unsigned long long>::max(),last=0,max_span=0;
        long double total_span=0;
        for (unsigned int tid=0;tid<host_probe_stride;++tid) {
            const ProbeStamp &s=stamps[(size_t)rep*host_probe_stride+tid];
            if (BODY_PROBE_MODE == 2) {
                probe_require(s.end >= s.begin && s.begin != 0,"invalid GPU timestamp order");
                first=std::min(first,s.begin);last=std::max(last,s.end);
                max_span=std::max(max_span,s.end-s.begin);total_span+=s.end-s.begin;
            } else probe_require(s.begin == 0 && s.end == 0,"unexpected timestamps in control");
        }
        const unsigned long long bytes=BODY_PROBE_MODE == 2 ? (unsigned long long)host_probe_stride*sizeof(ProbeStamp) : 0;
        if (BODY_PROBE_MODE != 2) first=last=0;
        fprintf(host_probe_csv,"%llu,%d,%d,%d,%d,%u,%d,%llu,%llu,%.9f,%.9f,%.9Lf,%llu,%llu\n",
                host_probe_call,method,N,delay,rep,host_probe_stride,BODY_PROBE_MODE,first,last,
                (last-first)/1000.0,max_span/1000.0,total_span/host_probe_stride/1000.0L,
                host_probe_raw_offset+(unsigned long long)rep*bytes,bytes);
    }
    if (BODY_PROBE_MODE == 2) {
        probe_require(fwrite(stamps.data(),sizeof(ProbeStamp),stamps.size(),host_probe_raw)==stamps.size(),"write raw stamps");
        host_probe_raw_offset+=(unsigned long long)stamps.size()*sizeof(ProbeStamp);
        probe_require(fflush(host_probe_raw)==0,"flush raw stamps");
    }
    probe_require(fflush(host_probe_csv)==0,"flush body CSV");
    ++host_probe_call;
}
'''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def instrument(source, mode):
    if mode == 0:
        return source
    for name in TARGETS:
        match=re.search(r'__global__ static void '+name+r'\s*\(',source)
        if not match:raise ValueError('Missing original kernel '+name)
        # Skip the entire parameter list, including schedule = {} defaults.
        position=match.end();paren_depth=1
        while paren_depth:
            paren_depth+=(source[position]=='(')-(source[position]==')');position+=1
        begin=source.index('{',position);depth=1;end=begin+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}');end+=1
        body=source[begin+1:end-1]
        if re.search(r'\breturn\b',body):raise ValueError('Early return needs a separate instrumentation design')
        start='const unsigned long long probe_begin = probe_now();' if mode==2 else 'const unsigned long long probe_begin = 0;'
        stop='const unsigned long long probe_end = probe_now();' if mode==2 else 'const unsigned long long probe_end = 0;'
        source=source[:begin+1]+'\n    '+start+'\n'+body+'\n    '+stop+'\n    probe_save(probe_begin,probe_end);\n'+source[end-1:]
    return source


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def sass_kernel(text, name):
    sections=re.split(r'Function\s*:\s*',text)
    matches=[part for part in sections[1:] if name in part.splitlines()[0]]
    if len(matches)!=1:raise ValueError('Cannot select SASS function '+name)
    return [' '.join(m.group(1).split()) for m in re.finditer(r'/\*[0-9a-fA-F]+\*/\s+([^\n]*?;)',matches[0])]


def check_codegen(out):
    original=(out/'original.sass.txt').read_text()
    checks={}
    for name in TARGETS:
        ref=sass_kernel(original,name)
        checks[name]={}
        for arm in ARMS:
            instructions=sass_kernel((out/(arm+'.sass.txt')).read_text(),name)
            timer_reads=sum('SR_GLOBALTIMER' in line for line in instructions)
            if arm=='control' and instructions!=ref:
                raise ValueError('Control kernel differs from archived SASS; inspect original/control before timing')
            if (arm=='globaltimer' and timer_reads<2) or (arm!='globaltimer' and timer_reads):
                raise ValueError('Unexpected native globaltimer instruction count in '+arm+'/'+name)
            checks[name][arm]={'instructions':len(instructions),'globaltimer_instructions':timer_reads}
    (out/'codegen_checks.json').write_text(json.dumps(checks,indent=2)+'\n')


def summarize(out):
    points={};checks=[]
    for arm,mode in ARMS.items():
        capture=out/'captures'/arm
        trace=read(capture/'derived/kernel_activity.csv')
        for run_kind in ('unprofiled','profiled_process'):
            process=capture/run_kind;body=read(process/'body_clock_launches.csv')
            if len(body)!=len(trace):raise ValueError('Body/kernel count mismatch')
            raw_size=(process/'body_clock_records.bin').stat().st_size if mode==2 else 0
            for b,k in zip(body,trace):
                for field in ('method_id','N','delaylength','inner_rep'):
                    if b[field]!=k[field]:raise ValueError('Body/native launch order mismatch: '+field)
                if mode==2 and int(b['raw_offset'])+int(b['raw_bytes'])>raw_size:raise ValueError('Truncated raw timestamps')
                if k['is_warmup']=='1':continue
                key=(int(k['method_id']),int(k['N']),int(k['delaylength']))
                p=points.setdefault(key,{'method_id':key[0],'N':key[1],'delaylength':key[2]})
                if mode==2:
                    p.setdefault(run_kind+'_GPU_body_envelope_us',[]).append(float(b['body_envelope_us']))
                    p.setdefault(run_kind+'_max_thread_body_us',[]).append(float(b['max_thread_span_us']))
                if run_kind=='profiled_process':
                    p.setdefault(arm+'_nsys_kernel_us',[]).append(float(k['kernel_us']))
            host=read(process/'raw_times.csv')
            for h in host:
                key=(int(h['method_id']),int(h['N']),int(h['delaylength']))
                points[key].setdefault(arm+'_'+run_kind+'_host_us',[]).append(float(h['exec_time_us']))
            checks.append({'arm':arm,'process':run_kind,'kernel_rows':len(body),'raw_bytes':raw_size})
    result=[]
    for key,p in sorted(points.items()):
        row={k:v for k,v in p.items() if not isinstance(v,list)}
        for k,v in p.items():
            if isinstance(v,list):row[k]=statistics.mean(v);row[k+'_SD']=statistics.stdev(v)
        result.append(row)
    with (out/'body_timing_means.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    notes=['Diagnostic copies only: production sources and original data remain unchanged.',
           'globaltimer body envelope = last instrumented thread end minus first instrumented thread start, on the same GPU.',
           'It excludes work before entry timestamps and after exit timestamps, including timestamp-record stores and kernel epilogue.',
           'A thread span is not whole-kernel elapsed time; both are reported with separate names.',
           'No new kernel barriers, global atomics, CPU/GPU clock subtraction, event timing or graph batching.',
           'Instrumentation changes registers, stores and scheduling. Control/stores-only comparisons diagnose its impact, not an exact instruction-latency correction.',
           'Memory clobbers constrain compiler motion, but no extra memory drain/fence is inserted; the end stamp does not guarantee completion of all outstanding stores.',
           'Raw records are little-endian uint64 begin/end pairs; offsets and lengths are recorded in body_clock_launches.csv.',
           'First/last timestamps must not be directly subtracted from CPU or nsys timestamps. Compare durations only.',
           'Hopper only for this experiment; do not assume the same unprofiled timer resolution on A100.']
    (out/'BODY_PROBE_COMPLETE.json').write_text(json.dumps({'checks':checks,'mean_points':len(result),'notes':notes},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,type=Path,help='Archived original CUDA raw_run directory')
    p.add_argument('--output',type=Path)
    p.add_argument('--nsys',default=os.environ.get('NSYS_BIN','nsys'))
    p.add_argument('--nvcc',default='nvcc')
    p.add_argument('--smoke',action='store_true',help='Native generation/counter/data-flow smoke; no nsys, Delay=1 only')
    a=p.parse_args();reference=a.reference.resolve()
    source=reference/'sources';driver=source/'src/core/driver.c';backend=source/'src/backends/cuda/backend.cu'
    for required in (driver,backend,source/'src/backends/cuda/openmp_profile.h',source/'src/core/backend.h',reference/'bin/microbenchmark_cuda'):
        if not required.is_file():p.error('Missing archived source '+str(required))
    for tool in (a.nvcc,'cuobjdump'):
        if not shutil.which(tool):p.error('Missing tool '+tool)
    if not a.smoke and not shutil.which(a.nsys):p.error('Missing nsys; set NSYS_BIN')
    if a.output:
        out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    else:
        parent=REPO/'result/GH200/kernel_body_clock';parent.mkdir(parents=True,exist_ok=True)
        out=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%dT%H%M%SZ_',time.gmtime()),dir=parent))
    print('Output directory:',out,flush=True)
    (out/'tmp').mkdir();(out/'bin').mkdir();(out/'sources').mkdir()
    shutil.copytree(source/'src',out/'sources/src')
    shutil.copy2(__file__,out/'probe_script.py')
    archived_text=backend.read_text()
    manifest={'reference':str(reference),'original_binary_sha256':(reference/'binary_sha256.txt').read_text().split()[0],
              'archived_backend_sha256':sha(backend),'archived_driver_sha256':sha(driver),'modes':ARMS,'smoke_only':a.smoke,
              'methods':[0,6],'N':16382,'MAX_ITER':16382,'MAX_ARRAY_SIZE':65536,'requested_blocks':132,'requested_threads':32}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ);env.pop('CUDA_OPENMP_PROFILE',None)
    env['TMPDIR']=str(out/'tmp');env['NSYS_TMPDIR']=env['TMPDIR'];env['NSYS_BIN']=a.nsys
    if sha(reference/'bin/microbenchmark_cuda')!=manifest['original_binary_sha256']:
        raise ValueError('Archived original binary hash mismatch')
    with (out/'original.sass.txt').open('w') as log:
        subprocess.run(['cuobjdump','--dump-sass',str(reference/'bin/microbenchmark_cuda')],env=env,stdout=log,check=True)
    for arm,mode in ARMS.items():
        generated=out/'sources'/('backend_'+arm+'.cu')
        generated.write_text('#define BODY_PROBE_MODE '+str(mode)+'\n'+HEADER+'\n'+instrument(archived_text,mode)+'\n'+FOOTER)
        binary=out/'bin'/arm
        command=[a.nvcc,'-std=c++17','-O0','-arch=sm_90','-DMICROBENCHMARK_CUDA','-DPRINT_DISTRIBUTION',
                 '-Xptxas=-v','-I',str(out/'sources/src/core'),'-I',str(out/'sources/src/backends/cuda'),
                 str(out/'sources/src/core/driver.c'),str(generated),'-lm','-o',str(binary)]
        (out/(arm+'_build_command.json')).write_text(json.dumps(command,indent=2)+'\n')
        with (out/(arm+'_build.log')).open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (out/(arm+'.sass.txt')).open('w') as log:subprocess.run(['cuobjdump','--dump-sass',str(binary)],env=env,stdout=log,check=True)
    check_codegen(out)
    arguments=['API=cuda','Method=0,6','N=16382','Delay=1,1','block_count=132','thread_count=32',
               'MAX_ITER=16382','MAX_ARRAY_SIZE=65536','DeviceTiming=0']
    for arm in ARMS:
        binary=out/'bin'/arm;process=out/'smoke'/arm;process.mkdir(parents=True)
        with (process/'run.log').open('w') as log:subprocess.run([str(binary),*arguments],cwd=process,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=read(process/'body_clock_launches.csv')
        if len(report)!=800:raise ValueError('Unexpected smoke kernel count')
        print('PASS native smoke:',arm,flush=True)
    if not a.smoke:
        arguments[3]='Delay=1,8192'
        for arm in ARMS:
            binary=out/'bin'/arm
            subprocess.run([sys.executable,str(REPO/'jobs/cuda/trace_nsys.py'),'--binary',str(binary),
                            '--driver-source',str(out/'sources/src/core/driver.c'),'--output',str(out/'captures'/arm),
                            '--',*arguments],env=env,check=True)
    if sha(backend)!=manifest['archived_backend_sha256'] or sha(driver)!=manifest['archived_driver_sha256']:
        raise ValueError('Archived reference sources changed')
    if not a.smoke:summarize(out)
    else:(out/'SMOKE_COMPLETE.json').write_text(json.dumps({'modes':list(ARMS),'kernels_per_mode':800},indent=2)+'\n')
    print('Completed body-clock diagnostic:',out,flush=True)


if __name__=='__main__':
    main()
