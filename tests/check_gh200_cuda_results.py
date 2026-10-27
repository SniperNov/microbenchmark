#!/usr/bin/env python3
"""Check completeness of the standard GH200 CUDA raw sweep, without plotting.

This validates configuration/sample coverage and finite values, not GPU
correctness, OpenMP runtime equivalence, or timing accuracy.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import math

GROUPS = {
    'M0_M10': ([0, 10], [16382]),
    'M1to4': ([1, 2, 3, 4], [1, 4, 16, 64, 256, 1024, 4096, 8192, 16382, 32768, 65536]),
    'M5': ([5], [16382]),
    'M6': ([6], [16382]),
    'M7': ([7], [16382]),
    'M8': ([8], [1, 2, 4, 8, 16, 32, 64]),
    'M9': ([9], [1, 2, 4, 8, 16, 32, 64]),
    'M11_parreps': ([11], [1, 2, 4, 8, 16, 32, 64, 128]),
}
KEYS = ('method_id', 'N', 'delaylength', 'set', 'run', 'outerreps')


def check_csv(path, methods, sizes, metric, device=False):
    expected = set(itertools.product(methods, sizes, [2**i for i in range(14)], range(2), range(5), [0]))
    seen = set()
    fields = set(KEYS) | {'block_count', 'thread_count', metric}
    if device:
        fields |= {'max_iter', 'max_array_size', 'inner_reps', 'measurement_mode',
                   'actual_block_count', 'actual_thread_count', 'alignment_mode'}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not fields <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: missing CSV columns')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}: malformed CSV row')
            key = tuple(int(row[field]) for field in KEYS)
            if key not in expected or key in seen:
                raise ValueError(f'{path}: unexpected or duplicate sample {key}')
            if (int(row['block_count']), int(row['thread_count'])) != (132, 32):
                raise ValueError(f'{path}: wrong requested launch controls')
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{path}: invalid {metric}')
            if device:
                workload = (max(6656, max(sizes)), 65536, 20)
                actual = tuple(int(row[f]) for f in ('max_iter', 'max_array_size', 'inner_reps'))
                if actual != workload or row['measurement_mode'] != 'same_stream_events_separate_pass':
                    raise ValueError(f'{path}: wrong workload or measurement mode')
                mode = row['alignment_mode']
                launch = (int(row['actual_block_count']), int(row['actual_thread_count']))
                if min(launch) <= 0:
                    raise ValueError(f'{path}: invalid actual launch controls')
                if mode == 'cuda_static_cyclic_unverified':
                    m = key[0]
                    wanted = (1, 1) if m <= 4 or m == 7 else (132, 1 if m == 5 else 32)
                    if launch != wanted:
                        raise ValueError(f'{path}: unexpected default launch dimensions')
                elif mode != 'observed_owners_inferred_chunks':
                    raise ValueError(f'{path}: unknown ownership alignment mode')
            seen.add(key)
    if seen != expected:
        raise ValueError(f'{path}: missing {len(expected - seen)} samples; got {len(seen)}, expected {len(expected)}')
    return len(seen)


def validate(directory, device_timing=True):
    counts = {}
    for group, (methods, sizes) in GROUPS.items():
        base = directory / 'baseline' / group
        count = check_csv(base / 'raw_times.csv', methods, sizes, 'exec_time_us')
        for name in ('overhead_distribution.txt', 'run.log'):
            if not (base / name).is_file() or (base / name).stat().st_size == 0:
                raise ValueError(f'{base / name}: missing or empty')
        device_file = directory / 'kernel_execution' / group / 'cuda_device_times.csv'
        if device_timing:
            device_count = check_csv(device_file, methods, sizes, 'device_kernel_us', device=True)
        else:
            if device_file.exists():
                raise ValueError(f'{device_file}: unexpected device-pass output')
            device_count = 0
        if (base / 'cuda_device_times.csv').exists():
            raise ValueError(f'{base}: device CSV not separated from baseline')
        counts[group] = {'host_rows': count, 'device_rows': device_count}
    return {'groups': counts, 'host_rows': sum(c['host_rows'] for c in counts.values()),
            'device_rows': sum(c['device_rows'] for c in counts.values()),
            'methods': list(range(12)), 'requested_blocks': 132, 'requested_threads': 32,
            'delaylengths': [2**i for i in range(14)], 'samples_per_delay': 10,
            'device_timing': bool(device_timing),
            'validation_scope': 'raw-data coverage and metadata; not GPU correctness or timing accuracy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--device-timing', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    marker = args.directory / 'COMPLETE.json'
    if marker.exists():
        marker.unlink()
    try:
        report = validate(args.directory, args.device_timing)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f'INCOMPLETE: {error}\n')
    marker.write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: methods 0-11, 8 groups; host rows={report['host_rows']}, device rows={report['device_rows']}")
    print(f'Completeness report: {marker}')


if __name__ == '__main__':
    main()
