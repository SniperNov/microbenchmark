import ast,re,tempfile,subprocess,shutil,os
from pathlib import Path
D=Path(__file__).resolve().parents[1];T=Path(tempfile.mkdtemp(prefix='um12_host_'))
h=(D/'tests/fake_cuda_runtime.h').read_text()
(T/'cuda_runtime.h').write_text(h)
for n in ['backend.cu','backend_managed.cu','openmp_profile.h']:
 s=(D/'src/backends/cuda'/n).read_text();s=re.sub(r'<<<.*?>>>','',s);(T/n).write_text(s)
(T/'backend.h').write_bytes((D/'src/core/backend.h').read_bytes())
(T/'test.cpp').write_text('''#include "backend_managed.cu"
int main(int argc,char **argv){backend_init(argc,argv);backend_config_t c={1,1};double *a=(double*)backend_managed_allocate(16*sizeof(double));for(int m=0;m<12;m++){for(int i=0;i<16;i++)a[i]=1000+i;calls.clear();backend_run_method(m,a,3,4,5,16,&c,2);if(a[0]!=8.0)return 10+m;for(auto &x:calls)if(x=="COPY")return 30;int expected=(m==1?4:(m==2||m==3?2:0));int got=0;for(auto &x:calls)if(x.find("prefetch")!=std::string::npos)got++;if(m>=1&&m<=4&&um_mode>0&&got!=expected){fprintf(stderr,"prefetch count %d expected%d\\n",got,expected);return 40;} }backend_managed_free(a);return 0;}
''')
for version in [12090,13010]:
 cmd=['/usr/bin/clang++','-std=c++17','-DMICROBENCHMARK_CUDA','-DCUDART_VERSION='+str(version),'-I',str(T),str(T/'test.cpp'),'-o',str(T/'test')];subprocess.run(cmd,check=True)
 for mode in ['managed_demand','managed_prefetch','managed_advice_prefetch']:
  subprocess.run([str(T/'test'),'MemoryMode='+mode],cwd=T,check=True)
print('PASS host-only API12/13 syntax + single-thread kernel smoke + no memcpy and M1-M4 placement policy; NOT native CUDA validation')
print(T)
