#!/usr/bin/env python3
"""Synthetic CSV tests for full-run coverage checks; no GPU measurements."""
import csv
import itertools
from pathlib import Path
import tempfile
import unittest

import check_gh200_cuda_results as check


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for group, (methods, sizes) in check.GROUPS.items():
            rows = []
            for values in itertools.product(methods, sizes, [2**i for i in range(14)], range(2), range(5), [0]):
                row = dict(zip(check.KEYS, values))
                row.update(block_count=132, thread_count=32, exec_time_us=3.0,
                           device_kernel_us=2.0, max_iter=max(6656, max(sizes)),
                           max_array_size=65536, inner_reps=20,
                           measurement_mode='same_stream_events_separate_pass',
                           alignment_mode='cuda_static_cyclic_unverified')
                method = values[0]
                row['actual_block_count'] = 1 if method <= 4 or method == 7 else 132
                row['actual_thread_count'] = 1 if method <= 5 or method == 7 else 32
                rows.append(row)
            base = self.root / 'baseline' / group
            self.write(base / 'raw_times.csv', rows)
            (base / 'run.log').write_text('benchmark configuration and output\n')
            (base / 'overhead_distribution.txt').write_text('benchmark summary\n')
            self.write(self.root / 'kernel_execution' / group / 'cuda_device_times.csv', rows)

    @staticmethod
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_sweep(self):
        report = check.validate(self.root)
        self.assertEqual((report['host_rows'], report['device_rows']), (9940, 9940))

    def test_missing_and_duplicate_samples(self):
        path = self.root / 'baseline/M5/raw_times.csv'
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        for bad in (rows[:-1], rows + [rows[0]]):
            self.write(path, bad)
            with self.assertRaises(ValueError):
                check.validate(self.root)

    def test_device_metadata_and_nan_rejected(self):
        path = self.root / 'kernel_execution/M5/cuda_device_times.csv'
        with path.open() as stream:
            original = list(csv.DictReader(stream))
        for field, value in [('device_kernel_us', 'nan'), ('actual_thread_count', 32),
                             ('max_iter', 6656), ('measurement_mode', 'host_wall')]:
            rows = [row.copy() for row in original]
            rows[0][field] = value
            self.write(path, rows)
            with self.assertRaises(ValueError):
                check.validate(self.root)

    def test_wrong_output_partition_rejected(self):
        (self.root / 'baseline/M5/cuda_device_times.csv').write_text('wrong location\n')
        with self.assertRaises(ValueError):
            check.validate(self.root)

    def test_host_only_collection(self):
        for path in self.root.glob('kernel_execution/*/cuda_device_times.csv'):
            path.unlink()
        report = check.validate(self.root, device_timing=False)
        self.assertEqual((report['host_rows'], report['device_rows']), (9940, 0))


if __name__ == '__main__':
    unittest.main()
