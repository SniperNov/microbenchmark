#!/usr/bin/env python3
"""Trace an archived CUDA sweep without recompilation or repeating its baseline."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--um-faults',action='store_true',help='Add a distinct heavy UM diagnostic pass')
    a=p.parse_args();run=a.run.resolve()
    binary=run/'bin/microbenchmark_cuda';driver=run/'sources/src/core/driver.c'
    if not binary.is_file() or not driver.is_file():p.error('Archived executable and matching driver source are required')
    expected=(run/'binary_sha256.txt').read_text().split()[0]
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected:p.error('Archived binary hash mismatch')
    groups=sorted(x for x in (run/'baseline').iterdir() if x.is_dir())
    if len(groups)!=8:p.error('Expected the complete eight-group CUDA sweep')
    out=Path(tempfile.mkdtemp(prefix=time.strftime('nsys_%Y%m%dT%H%M%SZ_',time.gmtime()),dir=run))
    env=dict(os.environ);env['TMPDIR']=str(out/'tmp');env['NSYS_TMPDIR']=env['TMPDIR'];Path(env['TMPDIR']).mkdir()
    profile=run/'openmp_profile.csv'
    if profile.is_file():env['CUDA_OPENMP_PROFILE']=str(profile)
    else:env.pop('CUDA_OPENMP_PROFILE',None)
    meta={'original_run':str(run),'binary_sha256':expected,'recompiled':False,'baseline_rerun':False,'groups':[]}
    helper=Path(__file__).with_name('trace_nsys.py')
    for group in groups:
        with (group/'raw_times.csv').open() as stream:rows=list(csv.DictReader(stream))
        methods=list(dict.fromkeys(r['method_id'] for r in rows));sizes=list(dict.fromkeys(r['N'] for r in rows))
        delays=sorted({int(r['delaylength']) for r in rows})
        dimensions={(r['block_count'],r['thread_count']) for r in rows}
        if len(dimensions)!=1:raise ValueError('Expected one launch configuration per archived group')
        blocks,threads=dimensions.pop()
        log=(group/'run.log').read_text()
        workload={key:re.search(r'^'+key+r'\s*:\s*(\d+)',log,re.M).group(1) for key in ('MAX_ITER','MAX_ARRAY_SIZE')}
        args=['API=cuda','Method='+','.join(methods),'N='+','.join(sizes),f'Delay={delays[0]},{delays[-1]}',
              'block_count='+blocks,'thread_count='+threads,*[k+'='+v for k,v in workload.items()],'DeviceTiming=0']
        for kind,um in [('tracing',False)]+([('tracing_um',True)] if a.um_faults else []):
            command=[sys.executable,str(helper),'--binary',str(binary),'--driver-source',str(driver),
                     '--baseline',str(group),'--output',str(out/kind/group.name)]
            if um:command.append('--um-faults')
            command+=['--',*args]
            subprocess.run(command,env=env,check=True)
        meta['groups'].append({'group':group.name,'arguments':args,'baseline_samples':len(rows)})
    if sum(g['baseline_samples'] for g in meta['groups'])!=9940:raise ValueError('Unexpected baseline coverage')
    for kind in ['tracing']+(['tracing_um'] if a.um_faults else []):
        summaries=[json.loads((out/kind/g.name/'TRACE_COMPLETE.json').read_text()) for g in groups]
        if sum(s['measured_samples'] for s in summaries)!=9940:raise ValueError('Incomplete '+kind)
    (out/'REPLAY_COMPLETE.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Complete tracing of archived run:',out)


if __name__=='__main__':main()
