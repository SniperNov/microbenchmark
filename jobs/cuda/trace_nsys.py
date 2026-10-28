#!/usr/bin/env python3
"""External CUDA tracing of the original benchmark, with no application edits."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--baseline',type=Path,help='Existing unprofiled group directory; otherwise run the same command once without nsys')
    p.add_argument('--driver-source',type=Path,default=ROOT/'src/core/driver.c')
    p.add_argument('--nsys',default=os.environ.get('NSYS_BIN','nsys'))
    p.add_argument('--um-faults',action='store_true',help='Separate high-overhead UM diagnostic; not normal timing')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args()
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if not args:p.error('Pass original benchmark arguments after --')
    for arg in args:
        if arg.startswith('DeviceTiming=') and arg!='DeviceTiming=0':p.error('Trace the original schedule with DeviceTiming=0')
    if not any(arg.startswith('DeviceTiming=') for arg in args):args.append('DeviceTiming=0')
    binary=a.binary.resolve();driver=a.driver_source.resolve();out=a.output.resolve()
    if not binary.is_file():p.error('Benchmark binary does not exist')
    nsys=shutil.which(a.nsys)
    if not nsys:p.error('Nsight Systems unavailable; set NSYS_BIN to the tool matching your CUDA version')
    if out.exists():p.error('Output directory already exists; choose a fresh path')
    out.mkdir(parents=True)
    temp=Path(os.environ.get('NSYS_TMPDIR',str(out/'tmp'))).resolve();temp.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'TMPDIR':str(temp)}
    version=subprocess.run([nsys,'--version'],env=env,text=True,capture_output=True,check=True)
    help_result=subprocess.run([nsys,'profile','--help'],env=env,text=True,capture_output=True,check=True)
    help_text=help_result.stdout+help_result.stderr
    options=['--trace=cuda','--sample=none','--cpuctxsw=none','--stats=false','--export=sqlite','--force-overwrite=false']
    event_control='unavailable in this version'
    if '--cuda-event-trace' in help_text:options.append('--cuda-event-trace=false');event_control='explicitly false'
    if a.um_faults:
        for opt in ('--cuda-um-cpu-page-faults','--cuda-um-gpu-page-faults'):
            if opt not in help_text:p.error('Requested UM tracing is unsupported by this nsys version: '+opt)
            options.append(opt+'=true')
    initial_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
    metadata={'binary':str(binary),'binary_sha256':initial_sha,'arguments':args,
      'driver_source':str(driver),'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),
      'nsys':nsys,'nsys_version':version.stdout.strip(),'profile_options':options,
      'nsys_event_completion_tracing':event_control,'um_fault_diagnostic':a.um_faults,
      'tmpdir':str(temp),'external_baseline':a.baseline is not None,
      'scope':'Original application schedule under external profiler; no inserted application events, NVTX, graph replay or added synchronization.',
      'limitations':['Profiler overhead is nonzero. Profiled host times are not the primary benchmark baseline.',
       'UM diagnostics may be highly intrusive; absence of UM records does not establish zero movement or coherent-memory cost.']}
    (out/'capture_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'nsys_version.txt').write_text(version.stdout+version.stderr)
    def run(command,directory,log):
        directory.mkdir(parents=True,exist_ok=True)
        with log.open('w') as f:subprocess.run(command,cwd=directory,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    baseline=a.baseline.resolve() if a.baseline else out/'unprofiled'
    if a.baseline is None:run([str(binary),*args],baseline,out/'unprofiled.log')
    if not (baseline/'raw_times.csv').is_file():p.error('Missing baseline raw CSV')
    target=out/'profiled_process';target.mkdir()
    command=[nsys,'profile',*options,'--output='+str(out/'trace'),str(binary),*args]
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    run(command,target,out/'profile.log')
    baseline_log=baseline/'run.log'
    if baseline_log.is_file():
        old_names=re.findall(r'^CUDA device \d+:\s*(.+)$',baseline_log.read_text(),re.M)
        new_names=re.findall(r'^CUDA device \d+:\s*(.+)$',(out/'profile.log').read_text(),re.M)
        if old_names and new_names and old_names!=new_names:raise ValueError('Baseline and trace GPU models differ')
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=initial_sha:raise RuntimeError('Benchmark binary changed during capture')
    if not (out/'trace.nsys-rep').is_file() or not (out/'trace.sqlite').is_file():raise RuntimeError('Missing native report/SQLite export')
    spec=importlib.util.spec_from_file_location('nsys_reader',ROOT/'tests/summarize_nsys_cuda.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    result=reader.summarize(out/'trace.sqlite',target/'raw_times.csv',driver,out/'derived')
    result['capture_messages']=[line for line in (out/'profile.log').read_text(errors='replace').splitlines()
                                if any(word in line.lower() for word in ('warning:','error:','not supported','permission denied'))][-100:]
    result['um_visibility']=('activity_rows_observed' if any(result['um_tables'].values()) else
                             'no_activity_rows_observed_or_exposed') if a.um_faults else 'fault_diagnostic_not_requested'
    def load(path):
        with path.open() as stream:rows=list(csv.DictReader(stream))
        keys=('method_id','N','block_count','thread_count','delaylength','set','run','outerreps')
        values={tuple(r[k] for k in keys):float(r['exec_time_us']) for r in rows}
        if len(values)!=len(rows) or any(not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError('Invalid/duplicate host samples')
        return values
    plain=load(baseline/'raw_times.csv');traced=load(target/'raw_times.csv')
    if plain.keys()!=traced.keys():raise RuntimeError('Profiled and unprofiled workloads differ')
    pairs=[{'method_id':k[0],'N':k[1],'block_count':k[2],'thread_count':k[3],
            'delaylength':k[4],'set':k[5],'run':k[6],'outerreps':k[7],
            'unprofiled_host_us':plain[k],'profiled_host_us':traced[k],
            'observed_profiled_over_unprofiled_ratio':traced[k]/plain[k]} for k in plain]
    reader.write_csv(out/'host_perturbation_samples.csv',pairs)
    result['baseline_directory']=str(baseline)
    result['um_fault_diagnostic']=a.um_faults
    result['note']='Host ratios compare separate runs; they do not isolate pure tracer overhead or establish a correction factor.'
    (out/'TRACE_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Nsight capture complete:',out)
    print('Measured kernel samples:',result['measured_samples'],'warmup kernels separately labelled:',result['warmup_kernel_rows'])


if __name__=='__main__':
    try:main()
    except (subprocess.CalledProcessError,RuntimeError,ValueError,OSError) as error:
        print('TRACE INCOMPLETE:',error,file=sys.stderr);raise SystemExit(1)
