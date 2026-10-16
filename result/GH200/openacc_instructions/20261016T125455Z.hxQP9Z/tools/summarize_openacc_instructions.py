#!/usr/bin/env python3
"""Index OpenACC SASS, retaining explicit existing/rebuilt provenance."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def index_file(path, kind, root):
    text=path.read_text()
    functions=[]
    for name,body in re.findall(r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)',text,re.S):
        if not re.search(r'backend_run_method|delay_kernel',name):
            continue
        ins=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
            if m:ins.append((int(m[1],16),m[2]))
        loops=[]
        for pc,op in ins:
            m=re.search(r'\bBRA(?:\.[A-Z]+)?\s+0x([0-9a-f]+)',op)
            if not m or int(m[1],16)>=pc:continue
            rows=[(p,o) for p,o in ins if int(m[1],16)<=p<=pc]
            if not any('DADD' in o for p,o in rows):continue
            loops.append(dict(start=hex(rows[0][0]),back_branch=hex(pc),instruction_lines=len(rows),DADD=sum('DADD' in o for p,o in rows),I2F=sum('I2F.' in o for p,o in rows),UIADD3=sum('UIADD3' in o for p,o in rows),IADD3=sum(bool(re.search(r'\bIADD3\b',o)) for p,o in rows),VIADD=sum(bool(re.search(r'\bVIADD\b',o)) for p,o in rows)))
        functions.append(dict(symbol=name.strip(),static_instruction_lines=len(ins),delay_loops=loops))
    return dict(path=str(path.relative_to(root)),provenance=kind,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),functions=functions)


def main(out):
    rebuilt=out/'historical_rebuild/openacc.sass.txt'
    text=rebuilt.read_text()
    if not re.search(r'\bsm_90a?\b',text):
        raise ValueError('Historical rebuild does not contain an sm_90/sm_90a SASS image; inspect compile/environment logs.')
    records=[index_file(rebuilt,'HISTORICAL_SOURCE_REBUILD_NOT_ORIGINAL_TIMED_BINARY',out)]
    if not records[0]['functions']:
        raise ValueError('No benchmark GPU functions found in rebuilt OpenACC SASS.')
    current=out/'current_rebuild/openacc.sass.txt'
    if current.exists():
        record=index_file(current,'CURRENT_WORKING_SOURCE_REBUILD_NOT_HISTORICAL_TIMED_BINARY',out)
        if not record['functions']:
            raise ValueError('No benchmark GPU functions found in the current-source SASS.')
        records.append(record)
    for path in sorted((out/'existing_candidates').glob('*.sass.txt')):
        records.append(index_file(path,'EXISTING_CANDIDATE_TIMING_ASSOCIATION_UNVERIFIED',out))
    (out/'instruction_index.json').write_text(json.dumps(dict(no_benchmark_executed=True,records=records),indent=2)+'\n')
    for label in ('historical','current'):
        source=(out/(label+'_rebuild/source/src/backends/openacc/backend.c')).read_text().splitlines()
        cases=[]
        for i,line in enumerate(source):
            m=re.match(r'\s*case (\d+):',line)
            if not m:continue
            end=next((j for j in range(i+1,len(source)) if 'break;' in source[j]),len(source)-1)
            cases.append(dict(method=int(m[1]),case_line=i+1,openacc_directive_lines=','.join(str(j+1) for j in range(i+1,end+1) if '#pragma acc' in source[j])))
        with (out/(label+'_source_method_lines.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['method','case_line','openacc_directive_lines'])
            writer.writeheader();writer.writerows(sorted(cases,key=lambda r:r['method']))
    print('PASS: OpenACC sm_90 SASS collected; benchmark GPU functions found.')
    print('Rebuilt benchmark GPU functions:',len(records[0]['functions']))
    print('Existing binary candidates:',sum(r['provenance'].startswith('EXISTING_') for r in records))
    print('Current source:',(out/'current_rebuild/status.txt').read_text().strip())
    print('Historical rebuild is explicitly labelled; it is not asserted to be the original timed binary.')
    print('No benchmark was run; no timing data were changed.')


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
