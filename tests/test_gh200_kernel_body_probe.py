#!/usr/bin/env python3
"""Host-only integration of original driver/backend with clock-record hooks.

Fake clocks verify recording/data-flow only, never GPU timing or PTX validity.
"""
import ast
import csv
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('body_probe',ROOT/'tests/gh200_kernel_body_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
tree=ast.parse((ROOT/'tests/check_cuda_timing.py').read_text())
header=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HEADER' for t in n.targets))
header=header.replace('#define __device__','#define __constant__\n#define __forceinline__ inline\n#define __device__')
header=header.replace('int multiProcessorCount,','int major,multiProcessorCount,').replace('p->multiProcessorCount=132;','p->major=9; p->multiProcessorCount=132;')
header+='''
static unsigned long long fake_tick=32;
static cudaError_t cudaMemset(void *p,int value,size_t n) { memset(p,value,n); return 0; }
template<class T> static cudaError_t cudaMemcpyToSymbol(T &symbol,const void *p,size_t n) { memcpy(&symbol,p,n);return 0; }
template<class F> static void fake_grid(unsigned b,unsigned t,F body) {
  gridDim.x=b;blockDim.x=t;
  for(unsigned bi=0;bi<b;++bi)for(unsigned ti=0;ti<t;++ti){blockIdx.x=bi;threadIdx.x=ti;body();}
}
'''
compiler=os.environ.get('CXX',shutil.which('g++-14') or 'c++')
original=(ROOT/'src/backends/cuda/backend.cu').read_text()
assert probe.instrument(original,0)==original
with tempfile.TemporaryDirectory(prefix='body-clock-host-check-') as temp:
    d=Path(temp);(d/'cuda_runtime.h').write_text(header)
    for arm,mode in probe.ARMS.items():
        source='#define BODY_PROBE_MODE '+str(mode)+'\n'+probe.HEADER+probe.instrument(original,mode)+probe.FOOTER
        source=source.replace('asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");','value = (fake_tick += 32);')
        def launch(m):
            shape=[v.strip() for v in m.group(2).split(',')]
            return 'fake_grid('+shape[0]+','+shape[1]+',[&](){'+m.group(1)+'('+m.group(3)+');});'
        source=re.sub(r'(\w+)<<<(.*?)>>>\((.*?)\);',launch,source,flags=re.S)
        source+='\n'+(ROOT/'src/core/driver.c').read_text()
        cpp=d/(arm+'.cpp');cpp.write_text(source)
        binary=d/arm
        subprocess.run([compiler,'-std=c++17','-O0','-DMICROBENCHMARK_CUDA','-I',str(d),'-I',str(ROOT/'src/core'),
                        '-I',str(ROOT/'src/backends/cuda'),str(cpp),'-lm','-o',str(binary)],check=True)
        run=d/(arm+'_run');run.mkdir()
        args=['API=cuda','Method=0,6','N=7','Delay=1,16','block_count=2','thread_count=3','MAX_ITER=23','MAX_ARRAY_SIZE=64','DeviceTiming=0']
        with (run/'run.log').open('w') as log:subprocess.run([str(binary),*args],cwd=run,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (run/'body_clock_launches.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==4000
        assert {int(r['participants']) for r in rows}=={1,6}
        assert all(int(r['mode'])==mode for r in rows)
        if mode==2:
            raw=run/'body_clock_records.bin'
            assert raw.stat().st_size==sum(int(r['raw_bytes']) for r in rows)
            assert all(int(r['body_last_end_ns'])>int(r['body_first_start_ns'])>0 for r in rows)
            assert all(int(r['raw_offset'])+int(r['raw_bytes'])<=raw.stat().st_size for r in rows)
        else:assert all(float(r['body_envelope_us'])==0 for r in rows)
        print('PASS host-only:',arm,'original M0/M6 results, 20 launches per call, per-thread counters and record layout')
print('Native nvcc/globaltimer/SASS validation remains required on Hopper.')
