#!/usr/bin/env python3
"""GH200 CUDA UM 12-method correctness-gated plain collection. No analysis."""
import csv,json,hashlib,itertools,math,os,subprocess,sys,traceback,fcntl,re,time
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=False)
MODES=['managed_demand','managed_prefetch','managed_advice_prefetch']
GROUPS={'M0_M10':([0,10],[16382]),'M1to4':([1,2,3,4],[1,4,16,64,256,1024,4096,8192,16382,32768,65536]),'M5':([5],[16382]),'M6':([6],[16382]),'M7':([7],[16382]),'M8':([8],[1,2,4,8,16,32,64]),'M9':([9],[1,2,4,8,16,32,64]),'M11_parreps':([11],[1,2,4,8,16,32,64,128])}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
cmds=[]
def run(cmd,cwd,env=None,timeout=10800,label='run'):
 cwd.mkdir(parents=True,exist_ok=True);cmd=list(map(str,cmd));start=time.time()
 with (cwd/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 cmds.append({'argv':cmd,'cwd':str(cwd),'exit':p.returncode,'seconds':time.time()-start});dump(O/'commands.json',cmds)
 if p.returncode:raise RuntimeError('Command failed: '+str(cwd/label))
def check(path,mode,methods,sizes,delays,iterations):
 rows=list(csv.DictReader((path/'raw_times.csv').open()));want=set(itertools.product(methods,sizes,delays,range(2),range(5),(0,)));seen=set()
 for x in rows:
  k=tuple(int(x[c]) for c in ['method_id','N','delaylength','set','run','outerreps']);v=float(x['exec_time_us'])
  assert k in want and k not in seen and math.isfinite(v) and v>0
  assert (int(x['block_count']),int(x['thread_count']))==(132,32) and x['memory_mode']==mode;seen.add(k)
 assert seen==want
 log=(path/'run.log').read_text();assert 'Finalizing CUDA benchmark.' in log
 for key,value in [('MAX_ITER',iterations),('MAX_ARRAY_SIZE',65536),('WARMUP_ITERS',10),('BENCHMARK_SETS',2),('BENCHMARK_RUNS',5),('OUTERREPS',1)]:
  assert re.findall(r'^\s*'+key+r'\s*:\s*(\d+)',log,re.M)==[str(value)],key
 hw=json.loads((path/'um_hardware.json').read_text());assert 'GH200' in hw['name'] and (hw['sm_count'],hw['warp_size'])==(132,32) and hw['memory_mode']==mode
 return {'rows':len(rows),'methods':methods,'N':sizes,'delays':delays,'memory_mode':mode,'MAX_ITER':iterations}
def main():
 lock=open(os.environ.get('GH200_GPU_LOCK',str(R/'gpu.lock')),'a');fcntl.flock(lock,fcntl.LOCK_EX)
 run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],O,label='gpu_before')
 run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],O,label='processes_before')
 assert not (O/'processes_before.log').read_text().strip(),'GPU compute processes present'
 expected=json.loads((R/'SOURCE_SHA256.json').read_text())
 for n,h in expected.items():assert sha(R/n)==h
 nvcc=os.environ.get('NVCC','nvcc')
 run([nvcc,'--version'],O,label='compiler')
 flags=['-O0','-std=c++17','-arch=sm_90','-DMICROBENCHMARK_CUDA','-I',R/'src/core']
 run([nvcc]+flags+['-x','cu',R/'src/core/driver.c',R/'src/backends/cuda/backend_managed.cu','-DPRINT_DISTRIBUTION','-o',O/'production'],O,label='build_production')
 run([nvcc]+flags+[R/'tests/check_cuda_device.cu','-o',O/'native_check'],O,label='build_check')
 binary=sha(O/'production');dump(O/'manifest.json',{'status':'validating','binary_sha256':binary,'modes':MODES,'scope':'Managed12methods; distinct memory/timing semantics from explicit copy'})
 for mode in MODES:
  env=dict(os.environ,CUDA_MEMORY_MODE=mode);env.pop('CUDA_OPENMP_PROFILE',None)
  p=O/'validation'/mode;run([O/'native_check'],p,env,label='native')
  log=(p/'native.log').read_text();assert log.count('PASS: B=')==33 and 'PASS: all 12 CUDA methods;' in log
  args=['API=cuda','MemoryMode='+mode,'block_count=132','thread_count=32','MAX_ARRAY_SIZE=65536','DeviceTiming=0']
  sm=p/'smoke';run([O/'production']+args+['Method='+','.join(map(str,range(12))),'N=7','Delay=1,2','MAX_ITER=4224'],sm,env)
  check(sm,mode,list(range(12)),[7],[1,2],4224)
  ex=p/'exact_iter';run([O/'production']+args+['Method=6','N=16382','Delay=1,2','MAX_ITER=4224'],ex,env);check(ex,mode,[6],[16382],[1,2],4224)
 dump(O/'VALIDATION_COMPLETE.json',{'modes':MODES,'native_geometry_cases_per_mode':33,'smoke_rows_per_mode':240,'exact_iter_rows_per_mode':20,'all12methods':True})
 for mode in MODES:
  env=dict(os.environ,CUDA_MEMORY_MODE=mode);env.pop('CUDA_OPENMP_PROFILE',None);total=0
  for tag,(methods,sizes) in GROUPS.items():
   assert sha(O/'production')==binary
   I=65536 if tag=='M1to4' else 4224;p=O/'formal'/mode/tag
   run([O/'production','API=cuda','MemoryMode='+mode,'Method='+','.join(map(str,methods)),'N='+','.join(map(str,sizes)),'Delay=1,8192','block_count=132','thread_count=32','MAX_ITER='+str(I),'MAX_ARRAY_SIZE=65536','DeviceTiming=0'],p,env)
   checked=check(p,mode,methods,sizes,[2**i for i in range(14)],I);total+=checked['rows'];dump(p/'GROUP_COMPLETE.json',checked)
  assert total==9940;dump(O/'formal'/mode/'FORMAL_COMPLETE.json',{'rows':total,'methods':12,'groups':8,'configurations':71,'memory_mode':mode})
 run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],O,label='gpu_after');assert (O/'gpu_before.log').read_bytes()==(O/'gpu_after.log').read_bytes()
 for n,h in expected.items():assert sha(R/n)==h
 assert sha(O/'production')==binary
 dump(O/'COMPLETE.json',{'status':'COMPLETE','measured_rows':29820,'rows_per_mode':9940,'binary_sha256':binary,'performance_analysis':False})
 dump(O/'ARTIFACT_SHA256.json',{str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
try:main()
except BaseException as e:dump(O/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});raise
