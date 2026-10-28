#!/usr/bin/env python3
"""Read Nsight Systems SQLite without executing exported code.

Map the unchanged driver's deterministic launch order, including warmups, back
to method/N/Delay. Reject incomplete or unexpected captures instead of guessing.
Kernel time is activity end-start; API/synchronization durations remain separate.
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

KERNELS={0:'delay_one_kernel',1:'delay_one_kernel',2:'delay_one_kernel',3:'delay_one_kernel',
         4:'delay_one_kernel',5:'block_scalar_kernel',6:'loop_kernel',7:'delay_one_kernel',
         8:'atomic_kernel',9:'reduction_kernel',10:'teams_parallel_kernel',11:'repeated_teams_parallel_kernel'}
CONFIG=('method_id','N','block_count','thread_count')


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def permutation(count,seed):
    result=list(range(count))
    for i in range(count-1,0,-1):
        seed=(1664525*seed+1013904223)&0xffffffff
        j=seed%(i+1);result[i],result[j]=result[j],result[i]
    return result


def descriptors(rows,inner,warmups,sets=2,runs=5,outer=1):
    groups=collections.OrderedDict()
    seen=set()
    for r in rows:
        key=tuple(int(r[k]) for k in CONFIG)
        sample=key+tuple(int(r[k]) for k in ('delaylength','set','run','outerreps'))
        if sample in seen:raise ValueError('Duplicate raw sample')
        seen.add(sample);groups.setdefault(key,[]).append(r)
    for key,rr in groups.items():
        delays=sorted({int(r['delaylength']) for r in rr})
        observed=[tuple(int(r[k]) for k in ('delaylength','set','run','outerreps')) for r in rr]
        measured_order=[(delays[i],se,ru,o) for se in range(sets) for ru in range(runs)
                        for i in permutation(len(delays),12345+97*se+1009*(ru+1)) for o in range(outer)]
        if observed!=measured_order:raise ValueError('Raw sample order/coverage does not match the unchanged driver')
        for w in range(warmups):
            for index in permutation(len(delays),12345):
                for rep in range(inner):
                    yield key,{'delaylength':delays[index],'set':0,'run':-1,'outerreps':0},True,w,rep
        for r in rr:
            for rep in range(inner):yield key,r,False,-1,rep


def summarize(sqlite_path,raw_path,driver_path,out):
    text=driver_path.read_text()
    constants={k:int(re.search(r'^#define\s+'+k+r'\s+(\d+)\s*$',text,re.M).group(1))
               for k in ('INNERREPS','WARMUP_ITERATIONS','BENCHMARK_SETS','BENCHMARK_RUNS','OUTERREPS')}
    # The mapping contract is explicitly tied to this unchanged driver algorithm.
    for required in ('1664525u * seed + 1013904223u','1009u * (unsigned int)(run + 1)',
                     'run_target(method, 0, -1, a, N, config)'):
        if required not in text:raise ValueError('Unsupported driver scheduling contract')
    with raw_path.open() as stream:rows=list(csv.DictReader(stream))
    if not rows:raise ValueError('Empty traced-process raw CSV')
    out.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect('file:'+str(sqlite_path.resolve())+'?mode=ro',uri=True)
    con.row_factory=sqlite3.Row
    tables={r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    strings={r['id']:r['value'] for r in con.execute('select id,value from StringIds')}
    table=next((t for t in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL') if t in tables),None)
    if table is None:raise ValueError('No CUDA kernel activity: capture/tool compatibility failed')
    def named(row):
        keys=row.keys()
        value=row['shortName'] if 'shortName' in keys else row['demangledName'] if 'demangledName' in keys else row['name']
        return strings.get(value,str(value))
    runtime_count=sync_count=0;apis={}
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' not in tables:raise ValueError('No CUDA API trace')
    fields=['name','start_ns','end_ns','duration_us','correlation_id','global_tid']
    with (out/'cuda_api_activity.csv').open('w',newline='') as af, (out/'synchronization_api_activity.csv').open('w',newline='') as sf:
        aw=csv.DictWriter(af,fieldnames=fields);sw=csv.DictWriter(sf,fieldnames=fields)
        aw.writeheader();sw.writeheader()
        for r in con.execute('select start,end,correlationId,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
            name=strings.get(r['nameId'],str(r['nameId']))
            if name.startswith(('cudaEventCreate','cudaEventRecord','cudaEventElapsedTime','cudaEventSynchronize')):
                raise ValueError('Event profiling detected: require DeviceTiming=0')
            record={'name':name,'start_ns':r['start'],'end_ns':r['end'],'duration_us':(r['end']-r['start'])/1000,
                    'correlation_id':r['correlationId'],'global_tid':r['globalTid']}
            aw.writerow(record);runtime_count+=1
            if 'Synchronize' in name:sw.writerow(record);sync_count+=1
            if 'Launch' in name:apis[r['globalTid']&~0xffffff,r['correlationId']]={'name':name,'start':r['start'],'end':r['end']}
    kernels=[]
    cols={r[1] for r in con.execute('pragma table_info("'+table+'")')}
    name_col=next(c for c in ('shortName','demangledName','name') if c in cols)
    query='select start,end,correlationId,globalPid,'+name_col+',streamId,gridX,blockX from "'+table+'" order by start,end'
    for r in con.execute(query):
        name=named(r)
        matches=[k for k in set(KERNELS.values()) if re.search(r'\b'+k+r'\b',name)]
        if matches:kernels.append((r,matches[0]))
    if len({r['globalPid'] for r,name in kernels})!=1:raise ValueError('Expected exactly one benchmark process')
    expected=list(descriptors(rows,constants['INNERREPS'],constants['WARMUP_ITERATIONS'],
                              constants['BENCHMARK_SETS'],constants['BENCHMARK_RUNS'],constants['OUTERREPS']))
    if len(kernels)!=len(expected):
        raise ValueError(f'Incomplete/unexpected kernel count: observed {len(kernels)}, expected {len(expected)}')
    all_kernels=[];samples=collections.OrderedDict()
    for (r,name),(config,source,warmup,round_id,rep) in zip(kernels,expected):
        method,n,blocks,threads=config
        if name!=KERNELS[method]:raise ValueError('Kernel order does not match unchanged driver schedule')
        api=apis.get((r['globalPid'],r['correlationId']))
        if api is None:raise ValueError('Kernel launch has no matching CUDA API record')
        if r['end']<r['start'] or api['end']<api['start']:raise ValueError('Invalid timeline timestamps')
        record={'method_id':method,'N':n,'block_count':blocks,'thread_count':threads,
          'delaylength':int(source['delaylength']),'set':int(source['set']),'run':int(source['run']),
          'outerreps':int(source['outerreps']),'is_warmup':int(warmup),'warmup_round':round_id,
          'inner_rep':rep,'kernel_name':name,'kernel_start_ns':r['start'],'kernel_end_ns':r['end'],
          'kernel_us':(r['end']-r['start'])/1000,'launch_api':api['name'],
          'launch_api_us':(api['end']-api['start'])/1000,
          'gpu_start_minus_api_start_us':(r['start']-api['start'])/1000,
          'gpu_start_minus_api_end_us':(r['start']-api['end'])/1000,
          'stream_id':r['streamId'],'correlation_id':r['correlationId'],
          'actual_grid_x':r['gridX'],'actual_block_x':r['blockX']}
        all_kernels.append(record)
        if not warmup:
            key=config+tuple(int(source[k]) for k in ('delaylength','set','run','outerreps'))
            samples.setdefault(key,[]).append(record)
    output=[]
    for key,kk in samples.items():
        if len(kk)!=constants['INNERREPS']:raise ValueError('Incomplete inner repetition sample')
        row=dict(zip(CONFIG+('delaylength','set','run','outerreps'),key))
        row.update(inner_reps=len(kk),kernel_us=statistics.mean(k['kernel_us'] for k in kk),
          launch_api_us=statistics.mean(k['launch_api_us'] for k in kk),
          gpu_start_minus_api_start_us=statistics.mean(k['gpu_start_minus_api_start_us'] for k in kk),
          gpu_start_minus_api_end_us=statistics.mean(k['gpu_start_minus_api_end_us'] for k in kk),
          measurement_mode='nsys_kernel_activity_original_schedule')
        output.append(row)
    if len(output)!=len(rows):raise ValueError('Kernel samples and traced host samples do not match')
    write_csv(out/'kernel_activity.csv',all_kernels)
    write_csv(out/'kernel_samples.csv',output)
    for name,filename in [('CUPTI_ACTIVITY_KIND_MEMCPY','memory_copy_activity.csv'),('CUPTI_ACTIVITY_KIND_MEMSET','memory_set_activity.csv')]:
        if name in tables:
            items=[dict(r) for r in con.execute('select * from '+name+' order by start')]
            if items:write_csv(out/filename,items)
    um_tables=[t for t in sorted(tables) if not t.startswith('ENUM_') and
               any(x in t.upper() for x in ('UNIFIED','_UM_','_UNIF_MEM_'))]
    report={'kernel_activity_rows':len(kernels),'measured_kernel_rows':sum(not k['is_warmup'] for k in all_kernels),
      'warmup_kernel_rows':sum(k['is_warmup'] for k in all_kernels),'measured_samples':len(output),
      'cuda_api_rows':runtime_count,'synchronization_api_rows':sync_count,'inner_reps':constants['INNERREPS'],
      'um_tables':{t:con.execute('select count(*) from "'+t+'"').fetchone()[0] for t in um_tables},
      'event_api_rows':0,'sqlite_timestamp_unit':'ns','derived_duration_unit':'us',
      'mapping':'Offline deterministic driver order; exact kernel counts and names checked; no NVTX or added application timing.',
      'timing_notes':[
       'Kernel duration is Nsight GPU activity end-start, measured under profiler overhead.',
       'GPU-start minus API-end is signed; negative values mean overlap, not invalid data. It is not an exact hardware queue-residency time.',
       'Synchronization API duration includes waiting for pending work; it is not isolated synchronization overhead.',
       'Memory/API records include runtime setup, warmups and teardown, not only baseline timing windows.',
       'No UM table/row does not prove zero memory movement or zero GH200 coherent-memory cost.']}
    (out/'trace_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    con.close()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path);p.add_argument('raw',type=Path);p.add_argument('driver',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    print(json.dumps(summarize(a.sqlite,a.raw,a.driver,a.output),indent=2))
