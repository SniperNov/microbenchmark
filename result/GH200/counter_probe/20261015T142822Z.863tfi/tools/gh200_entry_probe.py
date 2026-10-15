#!/usr/bin/env python3
"""Generate and inspect isolated method0 code-generation diagnostics."""
import argparse
from collections import Counter, defaultdict
import csv
import difflib
import json
import math
from pathlib import Path
import re
import statistics

SYMBOL = '_Z16delay_one_kerneliPd'


def candidate_name(kind):
    return 'cuda_entry_gate' if kind == 'entry' else 'cuda_counter_start'


def prepare(saved, out, kind='entry'):
    candidate = candidate_name(kind)
    source = saved / 'sources'
    paths = ['src/core/driver.c', 'src/core/backend.h',
             'src/backends/cuda/backend.cu', 'src/backends/cuda/openmp_profile.h']
    for arm in ('cuda_control', candidate):
        for relative in paths:
            dest = out / 'sources' / arm / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((source / relative).read_bytes())
    dest = out / 'sources' / candidate / 'src/backends/cuda/backend.cu'
    original = dest.read_text()
    if kind == 'entry':
        begin = original.index('__global__ static void delay_one_kernel(')
        end = original.index('__global__ static void block_scalar_kernel(', begin)
        region = original[begin:end]
        condition = 'if (blockIdx.x == 0 && threadIdx.x == 0)'
        if region.count(condition) != 1:
            raise ValueError('Expected the original method0 entry condition exactly once')
        changed = region.replace(condition, condition[:-1] + ' && diagnostic_entry_gate != 0)')
        declaration = '// Diagnostic only: one volatile device read per kernel, before the unchanged delay.\n__device__ volatile int diagnostic_entry_gate = 1;\n\n'
    else:
        begin = original.index('__device__ static void delay_kernel_device(')
        end = original.index('// A supplied profile', begin)
        region = original[begin:end]
        initial = 'for (int i = 0; i < delaylength; i++)'
        if region.count(initial) != 1:
            raise ValueError('Expected the original delay counter initialization exactly once')
        changed = region.replace(initial, 'for (int i = diagnostic_counter_start; i < delaylength; i++)')
        declaration = '// Diagnostic only: read the counter initial value once per delay call.\n// This device variable stays zero in the timed executable.\n__device__ volatile int diagnostic_counter_start = 0;\n\n'
    updated = original[:begin] + declaration + changed + original[end:]
    dest.write_text(updated)
    (out / 'source_change.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), updated.splitlines(True),
        fromfile='cuda_control/backend.cu', tofile=candidate+'/backend.cu')))
    (out/'experiment_config.json').write_text(json.dumps(dict(kind=kind,candidate=candidate),indent=2)+'\n')
    checker = r'''
#include <cmath>
#include <cstring>
#include <vector>
#include "src/backends/cuda/backend.cu"
int main(int argc, char **argv) {
    backend_init(argc, argv);
    backend_config_t config = {132, 32};
    const int delays[] = {0,1,2,3,4,15,16,17,1024,8192};
    for (int delay : delays) {
        for (int reps : {1,3}) {
            std::vector<double> a(32, -19.0);
            backend_run_method(0, a.data(), 7, delay, 23, 32, &config, reps);
            double expected = 1.0 + (double)delay*(delay-1)/2;
            if (a[0] != expected) return 3;
            for (int i=1; i<32; ++i) if (a[i] != -19.0) return 4;
            std::vector<double> before = a;
            double us = backend_run_device_method(0, a.data(), 7, delay, 23, 32, &config, reps);
            if (!(std::isfinite(us) && us>0) || a != before) return 5;
        }
    }
#ifdef ENTRY_GATE_CHECK
    int zero = 0;
    check_cuda(cudaMemcpyToSymbol(diagnostic_entry_gate, &zero, sizeof(zero)), "close diagnostic gate");
    std::vector<double> a(32, -19.0);
    backend_run_method(0, a.data(), 7, 17, 23, 32, &config, 1);
    for (double value : a) if (value != -19.0) return 6;
#endif
#ifdef COUNTER_START_CHECK
    // Validation executable only: changing the initial value to 2 must omit i=1.
    // Timed executables never write the variable and retain the initializer 0.
    int two = 2;
    check_cuda(cudaMemcpyToSymbol(diagnostic_counter_start, &two, sizeof(two)), "validate counter initial value");
    std::vector<double> a(32, -19.0);
    backend_run_method(0, a.data(), 7, 17, 23, 32, &config, 1);
    if (a[0] != 136.0) return 7;
    for (int i=1; i<32; ++i) if (a[i] != -19.0) return 8;
#endif
    backend_finalise();
    puts("PASS: method0 values, untouched array elements, repeated calls, profiling host-state preservation");
    return 0;
}
'''
    for arm in ('cuda_control', candidate):
        (out / 'sources' / arm / 'check_method0.cu').write_text(checker)


def inspect(path):
    text = path.read_text()
    matches = [body for name, body in re.findall(
        r'Function : ([^\n]+)\n(.*?)(?=\n\s*Function :|\Z)', text, re.S)
        if name.strip() == SYMBOL]
    if len(matches) != 1:
        raise ValueError('Expected one method0 kernel in '+str(path))
    instructions = []
    for line in matches[0].splitlines():
        match = re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;', line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    loops = []
    for pc, op in instructions:
        branch = re.search(r'\bBRA(?:\.[A-Z]+)?\s+0x([0-9a-f]+)', op)
        if not branch or int(branch[1], 16) >= pc:
            continue
        body = [(p, o) for p, o in instructions if int(branch[1], 16) <= p <= pc]
        ops = [re.sub(r'^@\S+\s+', '', o).split()[0] for p, o in body]
        histogram = dict(sorted(Counter(ops).items()))
        if 'DADD' not in histogram:
            continue
        loops.append(dict(start=hex(body[0][0]), back_branch=hex(pc),
                          histogram=histogram, instructions=len(body),
                          uniform_register_lines=sum(bool(re.search(r'\bUR\d+', o)) for p,o in body),
                          memory_operations=sum(bool(re.match(r'(LDG|STG|LDL|STL|LDS|STS)(\.|$)', op)) for op in ops),
                          assembly=[f'{p:04x}: {o}' for p,o in body]))
    if not loops:
        raise ValueError('No floating-point delay loop found in '+str(path))
    hot = max(loops, key=lambda loop: loop['histogram']['DADD'])
    return dict(symbol=SYMBOL, main_loop=hot, all_delay_loops=loops)


def signature(record):
    loop = record['main_loop']
    return {key:loop[key] for key in ('histogram','instructions','uniform_register_lines','memory_operations')}


def gate(saved, out, kind='entry'):
    candidate = candidate_name(kind)
    records = {arm:inspect(out / 'device_code' / (arm+'.sass.txt'))
               for arm in ('cuda_control', candidate)}
    old = inspect(saved / 'device_code/cuda_ptxas_O3.sass.txt')
    control_matches = signature(old) == signature(records['cuda_control'])
    changed = signature(records['cuda_control']) != signature(records[candidate])
    memory_free = all(r['main_loop']['memory_operations'] == 0 for r in records.values())
    arithmetic_ok = all(r['main_loop']['histogram'].get('I2F.F64',0) == r['main_loop']['histogram']['DADD'] for r in records.values())
    # Counter intervention is informative only if it removes the uniform
    # integer additions from the main loop, rather than merely rearranging it.
    counter_changed = kind != 'counter' or (
        records[candidate]['main_loop']['histogram'].get('UIADD3',0) == 0 and
        sum(records[candidate]['main_loop']['histogram'].get(op,0) for op in ('IADD3','VIADD')) > 1)
    approved = control_matches and changed and memory_free and arithmetic_ok and counter_changed
    result = dict(status='READY_FOR_TIMING' if approved else 'CODEGEN_NOT_ISOLATED',
                  control_main_loop_matches_previous_features=control_matches,
                  candidate_changes_main_loop_features=changed,
                  counter_target_met=counter_changed if kind == 'counter' else None,
                  main_loops_have_no_memory_operations=memory_free,
                  conversion_add_counts_match=arithmetic_ok,
                  limitation='Static features do not prove identical cycle counts or isolate a single scheduling effect.',
                  records=records)
    (out/'codegen_gate.json').write_text(json.dumps(result, indent=2)+'\n')
    for arm, record in records.items():
        (out/'device_code'/(arm+'_main_loop.txt')).write_text('\n'.join(record['main_loop']['assembly'])+'\n')
    print(result['status'])
    print('control reproduces previous loop features:', control_matches)
    print('candidate changes loop features:', changed)
    if kind == 'counter': print('counter uses ordinary integer adds:', counter_changed)
    print('loop has no memory operations:', memory_free)
    return 0 if approved else 10


def fit(points):
    xm = statistics.mean(x for x,y in points)
    ym = statistics.mean(y for x,y in points)
    b = sum((x-xm)*(y-ym) for x,y in points)/sum((x-xm)**2 for x,y in points)
    a = ym-b*xm
    return b*1000, a, 1-sum((y-a-b*x)**2 for x,y in points)/sum((y-ym)**2 for x,y in points)


def summarize(out, kind='entry'):
    candidate = candidate_name(kind)
    expected={(0,16382,2**d,s,r,0) for d in range(14) for s in range(2) for r in range(5)}
    slopes=[]
    for trial in ('01','02','03'):
        for arm in ('openmp_reference','cuda_control',candidate):
            folder=out/'measurements'/('trial_'+trial)/arm
            log=(folder/'run.log').read_text()
            for required in ('MAX_ITER      : 16382','MAX_ARRAY_SIZE: 65536'):
                if required not in log:
                    raise ValueError('Runtime workload mismatch: '+str(folder))
            phases=[('baseline','raw_times.csv','exec_time_us')]
            if arm!='openmp_reference':
                phases.append(('kernel_execution','cuda_device_times.csv','device_kernel_us'))
            for phase, filename, metric in phases:
                path=folder/phase/filename
                rows=list(csv.DictReader(path.open()))
                keys=[tuple(int(r[k]) for k in ('method_id','N','delaylength','set','run','outerreps')) for r in rows]
                if len(rows)!=140 or len(set(keys))!=140 or set(keys)!=expected:
                    raise ValueError('Incomplete, duplicate or unexpected coverage: '+str(path))
                groups=defaultdict(list)
                for row in rows:
                    value=float(row[metric])
                    if not math.isfinite(value) or value<=0:
                        raise ValueError('Invalid timing: '+str(path))
                    control='team_count' if arm=='openmp_reference' else 'block_count'
                    if (int(row[control]),int(row['thread_count']))!=(132,32):
                        raise ValueError('Requested configuration differs: '+str(path))
                    if phase=='kernel_execution':
                        if tuple(int(row[k]) for k in ('actual_block_count','actual_thread_count','max_iter','max_array_size','inner_reps'))!=(1,1,16382,65536,20):
                            raise ValueError('Device launch or workload differs: '+str(path))
                        if row['measurement_mode']!='same_stream_events_separate_pass' or row['alignment_mode']!='cuda_static_cyclic_unverified':
                            raise ValueError('Unexpected device timing/alignment mode: '+str(path))
                    groups[int(row['delaylength'])].append(value)
                b,a,r2=fit([(d,statistics.mean(groups[d])) for d in (1024,2048,4096,8192)])
                slopes.append(dict(trial=trial,arm=arm,phase=phase,slope_ns_per_delay=b,intercept_us=a,r_squared=r2))
    with (out/'slope_per_trial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(slopes[0]));writer.writeheader();writer.writerows(slopes)
    summary=[]
    for arm,phase in [('openmp_reference','baseline'),('cuda_control','baseline'),('cuda_control','kernel_execution'),(candidate,'baseline'),(candidate,'kernel_execution')]:
        values=[r['slope_ns_per_delay'] for r in slopes if (r['arm'],r['phase'])==(arm,phase)]
        summary.append(dict(arm=arm,phase=phase,mean_slope_ns_per_delay=statistics.mean(values),sd_slope_ns_per_delay=statistics.stdev(values)))
    with (out/'slope_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print('PASS: 2100 samples, exact coverage and positive finite times')
    print('Method0; fit Delay=1024,2048,4096,8192; ns/Delay; SD across three trials')
    for row in summary:
        print('{arm:18s} {phase:16s} {mean_slope_ns_per_delay:.6f} +/- {sd_slope_ns_per_delay:.6f}'.format(**row))
    print('Candidate adds a device read outside the delay loop. This is a code-generation sensitivity test, not proof of the cause of the OpenMP gap.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','gate','summarize'])
    parser.add_argument('out',type=Path)
    parser.add_argument('--saved',type=Path)
    parser.add_argument('--kind',choices=['entry','counter'],default='entry')
    args=parser.parse_args()
    if args.action in ('prepare','gate') and args.saved is None:
        parser.error('--saved is required')
    if args.action=='prepare':
        prepare(args.saved,args.out,args.kind)
    elif args.action=='gate':
        raise SystemExit(gate(args.saved,args.out,args.kind))
    else:
        summarize(args.out,args.kind)
