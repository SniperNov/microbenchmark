
#pragma once
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#define __device__
#define __global__
struct Dim { unsigned int x; };
static Dim blockIdx={0},blockDim={1},threadIdx={0},gridDim={1};
typedef int cudaError_t;
static const int cudaSuccess=0, cudaCpuDeviceId=-1, cudaStreamNonBlocking=1, cudaMemAttachGlobal=1;
typedef void *cudaStream_t;
struct Event { int position; cudaStream_t stream; };
typedef Event *cudaEvent_t;
struct cudaDeviceProp { char name[128]; int multiProcessorCount,warpSize,major,minor,maxThreadsPerBlock,maxThreadsDim[3],maxGridSize[3]; };
enum cudaDeviceAttr { cudaDevAttrManagedMemory, cudaDevAttrConcurrentManagedAccess, cudaDevAttrPageableMemoryAccess,
                     cudaDevAttrPageableMemoryAccessUsesHostPageTables, cudaDevAttrDirectManagedMemAccessFromHost };
enum cudaMemoryAdvise { cudaMemAdviseSetPreferredLocation };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum cudaMemLocationType { cudaMemLocationTypeDevice, cudaMemLocationTypeHost };
struct cudaMemLocation { cudaMemLocationType type; int id; };
static std::map<void *,bool> managed;
static std::vector<std::string> calls;
struct Log { ~Log() { const char *p=std::getenv("FAKE_RUNTIME_LOG"); if(p) { FILE *f=std::fopen(p,"w"); if(f) { for(auto &x:calls) std::fprintf(f,"%s\n",x.c_str()); std::fclose(f); } } } } log_at_exit;
static const char *cudaGetErrorString(int) { return "fake error"; }
static int cudaGetDeviceCount(int *p) { *p=1; return 0; }
static int cudaGetDevice(int *p) { *p=0; return 0; }
static int cudaGetDeviceProperties(cudaDeviceProp *p,int) { std::strcpy(p->name,"GH200 FAKE HOST TEST"); p->multiProcessorCount=132;p->warpSize=32;p->major=9;p->minor=0;p->maxThreadsPerBlock=p->maxThreadsDim[0]=1024;p->maxGridSize[0]=2147483647;return 0; }
static int cudaDeviceGetAttribute(int *p,cudaDeviceAttr,int) { *p=std::getenv("FAKE_UNSUPPORTED")?0:1; return 0; }
static int cudaDriverGetVersion(int *p) { *p=CUDART_VERSION;return 0; }
static int cudaRuntimeGetVersion(int *p) { *p=CUDART_VERSION;return 0; }
static int cudaMalloc(void **p,size_t bytes) { *p=std::malloc(bytes);managed[*p]=false;calls.push_back("allocate explicit");return *p?0:1; }
static int cudaMallocManaged(void **p,size_t bytes,unsigned int flags=cudaMemAttachGlobal) { if(flags!=cudaMemAttachGlobal) std::abort(); *p=std::malloc(bytes);managed[*p]=true;calls.push_back("allocate managed");return *p?0:1; }
static int cudaFree(void *p) { if(p) { if(managed.erase(p)!=1) std::abort(); std::free(p);calls.push_back("free"); } return 0; }
static int cudaStreamCreateWithFlags(cudaStream_t *p,unsigned int flags=cudaMemAttachGlobal) { if(flags!=cudaStreamNonBlocking) std::abort();*p=(void*)42;return 0; }
static int cudaStreamDestroy(cudaStream_t p) { if(p!=(void*)42 && p!=0) std::abort();return 0; }
static int cudaStreamSynchronize(cudaStream_t p) { if(p!=(void*)42 && p!=0) std::abort();calls.push_back("wait");return 0; }
static int cudaGetLastError() { calls.push_back("launch check");return 0; }
static int cudaMemcpyAsync(void *to,const void *from,size_t n,cudaMemcpyKind kind,cudaStream_t stream) {
 if((stream!=(void*)42 && stream!=0)) std::abort();void *device=kind==cudaMemcpyHostToDevice?to:(void*)from;
 if(!managed.count(device)||managed[device]) std::abort();
 std::memcpy(to,from,n);calls.push_back(kind==cudaMemcpyHostToDevice?"copy GPU":"copy CPU");return 0;
}
static int do_prefetch(const void *p,int destination,cudaStream_t stream) {
 if(!managed.count((void*)p)||!managed[(void*)p]||(stream!=(void*)42 && stream!=0)) std::abort();
 if(destination!=0 && destination!=cudaCpuDeviceId) std::abort();
 calls.push_back(destination==0?"prefetch GPU":"prefetch CPU");return 0;
}
#if CUDART_VERSION >= 13000
static int cudaMemPrefetchAsync(const void *p,size_t,cudaMemLocation location,unsigned int flags,cudaStream_t stream) {
 if(flags!=0 || location.id!=0) std::abort();return do_prefetch(p,location.type==cudaMemLocationTypeHost?cudaCpuDeviceId:location.id,stream);
}
static int cudaMemAdvise(const void *p,size_t,cudaMemoryAdvise advice,cudaMemLocation location) {
 if(advice!=cudaMemAdviseSetPreferredLocation||location.type!=cudaMemLocationTypeDevice||location.id!=0||!managed[(void*)p]) std::abort();calls.push_back("advice GPU");return 0;
}
#else
static int cudaMemPrefetchAsync(const void *p,size_t,int destination,cudaStream_t stream) { return do_prefetch(p,destination,stream); }
static int cudaMemAdvise(const void *p,size_t,cudaMemoryAdvise advice,int device) {
 if(advice!=cudaMemAdviseSetPreferredLocation||device!=0||!managed[(void*)p]) std::abort();calls.push_back("advice GPU");return 0;
}
#endif
static int cudaEventCreate(cudaEvent_t *p) { *p=new Event;calls.push_back("create event");return 0; }
static int cudaEventDestroy(cudaEvent_t p) { delete p;calls.push_back("destroy event");return 0; }
static int cudaEventRecord(cudaEvent_t p,cudaStream_t stream) { p->position=(int)calls.size();p->stream=stream;calls.push_back("event");return 0; }
static int cudaEventElapsedTime(float *ms,cudaEvent_t start,cudaEvent_t stop) {
 if(start->stream!=(void*)42 || stop->stream!=start->stream ||
    calls[start->position+1]!="kernel" || calls[start->position+2]!="launch check" ||
    stop->position!=start->position+3 || calls[stop->position+1]!="wait") std::abort();
 *ms=0.002f;calls.push_back("event elapsed");return 0;
}

inline void __syncthreads(){}
static int cudaDeviceSynchronize(){return 0;}
static int cudaStreamCreate(cudaStream_t *s){return cudaStreamCreateWithFlags(s,cudaStreamNonBlocking);}
template<class T> T atomicAdd(T *p,T x){T old=*p;*p+=x;return old;}
const int cudaMemoryTypeManaged=3;
struct cudaPointerAttributes { int type; };
static int cudaPointerGetAttributes(cudaPointerAttributes *a,void *p){a->type=managed[p]?3:0;return 0;}
static int cudaMemcpy(void *a,const void *b,size_t n,cudaMemcpyKind){std::memcpy(a,b,n);calls.push_back("COPY");return 0;}
