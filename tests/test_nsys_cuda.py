#!/usr/bin/env python3
"""Synthetic SQLite tests: strict mapping and honest timing boundaries, no GPU."""
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest
import summarize_nsys_cuda as ns

ROOT=Path(__file__).resolve().parents[1]


class TraceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.p=Path(self.tmp.name);self.raw=self.p/'raw.csv';self.db=self.p/'trace.sqlite'
        rows=[]
        for m in (0,6):
            for se in range(2):
                for ru in range(5):
                    for i in ns.permutation(2,12345+97*se+1009*(ru+1)):
                        rows.append(dict(method_id=m,N=7,block_count=3,thread_count=5,delaylength=(1,16)[i],set=se,run=ru,outerreps=0,exec_time_us=8))
        ns.write_csv(self.raw,rows)
        con=sqlite3.connect(self.db)
        con.executescript('''create table StringIds(id integer,value text);
create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,correlationId integer,globalTid integer,nameId integer);
create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,correlationId integer,globalPid integer,shortName integer,streamId integer,gridX integer,blockX integer);
create table ENUM_CUDA_UNIF_MEM_MIGRATION(id integer);
insert into ENUM_CUDA_UNIF_MEM_MIGRATION values(1);
insert into StringIds values(1,'cudaLaunchKernel_v7000'),(2,'delay_one_kernel'),(3,'loop_kernel'),(4,'cudaEventRecord_v3020');''')
        for index,(key,r,w,round_id,rep) in enumerate(ns.descriptors(rows,20,10),1):
            # Deliberate negative GPU-start minus API-end: execution overlaps the API.
            con.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(index*10000,index*10000+4000,index,(123<<24)+77,1))
            con.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?)',(index*10000+1000,index*10000+3000,index,123<<24,2 if key[0]==0 else 3,7,1 if key[0]==0 else 3,1 if key[0]==0 else 5))
        con.commit();con.close()

    def run_check(self):return ns.summarize(self.db,self.raw,ROOT/'src/core/driver.c',self.p/'derived')

    def test_mapping_warmups_and_overlap(self):
        result=self.run_check()
        self.assertEqual(result['measured_samples'],40)
        self.assertEqual(result['warmup_kernel_rows'],800)
        self.assertEqual(result['kernel_activity_rows'],1600)
        self.assertEqual(result['um_tables'],{})
        with (self.p/'derived/kernel_samples.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertTrue(all(float(r['kernel_us'])==2 for r in rows))
        self.assertTrue(all(float(r['gpu_start_minus_api_end_us'])==-3 for r in rows))

    def test_missing_kernel_rejected(self):
        with sqlite3.connect(self.db) as c:c.execute('delete from CUPTI_ACTIVITY_KIND_KERNEL where correlationId=1')
        with self.assertRaisesRegex(ValueError,'kernel count'):self.run_check()

    def test_added_events_rejected(self):
        with sqlite3.connect(self.db) as c:c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(0,1,99999,0,4)')
        with self.assertRaisesRegex(ValueError,'Event profiling'):self.run_check()

    def test_missing_api_correlation_rejected(self):
        with sqlite3.connect(self.db) as c:c.execute('delete from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=1')
        with self.assertRaisesRegex(ValueError,'matching CUDA API'):self.run_check()

    def test_shuffled_csv_rejected(self):
        with self.raw.open() as stream:rows=list(csv.DictReader(stream))
        rows[0],rows[1]=rows[1],rows[0]
        ns.write_csv(self.raw,rows)
        with self.assertRaisesRegex(ValueError,'sample order'):self.run_check()

    def test_trace_helper_rejects_event_pass(self):
        import subprocess,sys
        done=subprocess.run([sys.executable,str(ROOT/'jobs/cuda/trace_nsys.py'),'--binary','unused','--output',str(self.p/'capture'),'--','DeviceTiming=1'],capture_output=True,text=True)
        self.assertNotEqual(done.returncode,0)
        self.assertIn('DeviceTiming=0',done.stderr)


if __name__=='__main__':unittest.main()
