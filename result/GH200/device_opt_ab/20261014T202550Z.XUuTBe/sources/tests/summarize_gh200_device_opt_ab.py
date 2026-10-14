#!/usr/bin/env python3
"""Validate matched A/B raw samples and report per-trial tail slopes, without plots."""
from pathlib import Path
import collections
import csv
import math
import statistics
import sys


def fit(points):
    xm = statistics.mean(x for x, y in points)
    ym = statistics.mean(y for x, y in points)
    slope = sum((x-xm)*(y-ym) for x, y in points) / sum((x-xm)**2 for x, y in points)
    intercept = ym - slope*xm
    rss = sum((y-intercept-slope*x)**2 for x, y in points)
    tss = sum((y-ym)**2 for x, y in points)
    return slope*1000, intercept, 1-rss/tss


def main(directory):
    out = Path(directory).resolve()
    expected = {(m, 16382, 2**d, s, r, 0) for m in (0, 6)
                for d in range(14) for s in range(2) for r in range(5)}
    slopes = []
    total = 0
    for trial in ('01', '02', '03'):
        for arm in ('openmp_reference', 'cuda_ptxas_O3', 'cuda_ptxas_O0'):
            arm_dir = out/'measurements'/('trial_'+trial)/arm
            phases = [('baseline', 'raw_times.csv', 'exec_time_us')]
            if arm != 'openmp_reference':
                phases.append(('kernel_execution', 'cuda_device_times.csv', 'device_kernel_us'))
            log = (arm_dir/'run.log').read_text()
            for required in ('MAX_ITER      : 16382', 'MAX_ARRAY_SIZE: 65536'):
                if required not in log:
                    raise ValueError('Runtime workload mismatch: '+str(arm_dir))
            for phase, filename, metric in phases:
                path = arm_dir/phase/filename
                with path.open(newline='') as source:
                    rows = list(csv.DictReader(source))
                actual = set()
                groups = collections.defaultdict(list)
                for row in rows:
                    key = tuple(int(row[k]) for k in ('method_id', 'N', 'delaylength', 'set', 'run', 'outerreps'))
                    if key in actual:
                        raise ValueError('Duplicate sample: '+str(path))
                    actual.add(key)
                    value = float(row[metric])
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError('Invalid time: '+str(path))
                    control = 'team_count' if arm == 'openmp_reference' else 'block_count'
                    if (int(row[control]), int(row['thread_count'])) != (132, 32):
                        raise ValueError('Requested launch controls differ: '+str(path))
                    if phase == 'kernel_execution':
                        if (int(row['max_iter']), int(row['max_array_size']), int(row['inner_reps'])) != (16382, 65536, 20):
                            raise ValueError('Device workload differs: '+str(path))
                        if row['measurement_mode'] != 'same_stream_events_separate_pass' or row['alignment_mode'] != 'cuda_static_cyclic_unverified':
                            raise ValueError('Unexpected timing/alignment mode: '+str(path))
                        geometry = (1, 1) if key[0] == 0 else (132, 32)
                        if (int(row['actual_block_count']), int(row['actual_thread_count'])) != geometry:
                            raise ValueError('CUDA launch geometry differs: '+str(path))
                    groups[key[:3]].append(value)
                if actual != expected or len(rows) != 280:
                    raise ValueError('Incomplete/unexpected coverage: '+str(path))
                total += len(rows)
                for method in (0, 6):
                    points = [(d, statistics.mean(groups[method, 16382, d])) for d in (1024, 2048, 4096, 8192)]
                    b, a, r2 = fit(points)
                    slopes.append(dict(trial=trial, arm=arm, phase=phase, method=method,
                                       slope_ns_per_delay=b, intercept_us=a, r_squared=r2))
    with (out/'slope_per_trial.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(slopes[0]))
        writer.writeheader(); writer.writerows(slopes)
    summary = []
    for method in (0, 6):
        for arm, phase in [('openmp_reference', 'baseline'), ('cuda_ptxas_O3', 'baseline'),
                          ('cuda_ptxas_O3', 'kernel_execution'), ('cuda_ptxas_O0', 'baseline'),
                          ('cuda_ptxas_O0', 'kernel_execution')]:
            values = [r['slope_ns_per_delay'] for r in slopes if (r['method'], r['arm'], r['phase']) == (method, arm, phase)]
            summary.append(dict(method=method, arm=arm, phase=phase, trials=len(values),
                                mean_slope_ns_per_delay=statistics.mean(values),
                                sd_slope_ns_per_delay=statistics.stdev(values)))
    with (out/'slope_summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print('PASS: 4200 raw samples; exact coverage, unique keys, finite positive times, matched workload and requested controls')
    if total != 4200:
        raise ValueError('Unexpected total sample count')
    print('Tail fit: Delay=1024,2048,4096,8192; slope units ns/Delay')
    for r in summary:
        print('m{method} {arm:18s} {phase:16s} b={mean_slope_ns_per_delay:.6f} +/- {sd_slope_ns_per_delay:.6f} (SD across 3 trials)'.format(**r))
    print('No plots generated. Original raw samples remain in separate baseline/kernel_execution directories.')
    print('ptxas O0 is an assembler-stage sensitivity test, not proof that all CUDA device optimizations match NVC -O0.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: summarize_gh200_device_opt_ab.py <experiment directory>')
    main(sys.argv[1])
