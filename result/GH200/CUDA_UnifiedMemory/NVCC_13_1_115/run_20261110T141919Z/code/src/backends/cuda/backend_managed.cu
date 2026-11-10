// UM variant reuses the exact CUDA kernels; explicit-copy backend stays intact.
#define backend_init explicit_backend_init
#define backend_run_method explicit_backend_run_method
#define backend_run_device_method explicit_backend_run_device_method
#include "backend.cu"
#undef backend_init
#undef backend_run_method
#undef backend_run_device_method
#include <limits>
static int um_mode = 1; // demand=0, prefetch=1, advice_prefetch=2
static int um_device = 0;
static const char *um_names[] = {"managed_demand","managed_prefetch","managed_advice_prefetch"};
extern "C" const char *backend_memory_mode(void) { return um_names[um_mode]; }
static void um_prefetch(void *p, size_t bytes, int destination) {
#if CUDART_VERSION >= 13000
    cudaMemLocation loc = {}; loc.type = destination == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    loc.id = destination == cudaCpuDeviceId ? 0 : destination;
    check_cuda(cudaMemPrefetchAsync(p, bytes, loc, 0, 0), "UM prefetch");
#else
    check_cuda(cudaMemPrefetchAsync(p, bytes, destination, 0), "UM prefetch");
#endif
    check_cuda(cudaStreamSynchronize(0), "UM prefetch wait");
}
extern "C" void *backend_managed_allocate(size_t bytes) {
    if (!bytes) { fprintf(stderr,"Zero UM allocation\n"); exit(1); }
    void *p = NULL; check_cuda(cudaMallocManaged(&p, bytes), "UM allocation");
    if (um_mode == 2) {
#if CUDART_VERSION >= 13000
        cudaMemLocation loc = {}; loc.type=cudaMemLocationTypeDevice; loc.id=um_device;
        check_cuda(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, loc), "UM advice");
#else
        check_cuda(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, um_device), "UM advice");
#endif
    }
    return p;
}
extern "C" void backend_managed_free(void *p) { check_cuda(cudaFree(p), "UM free"); }
extern "C" void backend_init(int argc,char **argv) {
    const char *mode=getenv("CUDA_MEMORY_MODE");
    for(int i=1;i<argc;i++) {
        if(!strncmp(argv[i],"MemoryMode=",11)) mode=argv[i]+11;
        if(!strcmp(argv[i],"DeviceTiming=1")) {fprintf(stderr,"UM suite supports plain collection only\n");exit(1);}
    }
    if(mode) {int found=0;for(int i=0;i<3;i++)if(!strcmp(mode,um_names[i])){um_mode=i;found=1;}if(!found){fprintf(stderr,"Invalid MemoryMode\n");exit(1);}}
    explicit_backend_init(argc,argv);
    if(has_openmp_profile){fprintf(stderr,"UM suite does not accept owner replay profiles\n");exit(1);}
    check_cuda(cudaGetDevice(&um_device),"UM device");cudaDeviceProp p;check_cuda(cudaGetDeviceProperties(&p,um_device),"UM properties");
    int count=0,managed=0,concurrent=0;check_cuda(cudaGetDeviceCount(&count),"UM count");
    check_cuda(cudaDeviceGetAttribute(&managed,cudaDevAttrManagedMemory,um_device),"UM support");
    check_cuda(cudaDeviceGetAttribute(&concurrent,cudaDevAttrConcurrentManagedAccess,um_device),"UM concurrent support");
    if(count!=1||!strstr(p.name,"GH200")||!managed||!concurrent){fprintf(stderr,"Requires one GH200 with managed/concurrent access\n");exit(1);}
    method_names[1]="UM input prefetch + kernel + output prefetch";method_names[2]="UM input prefetch + kernel";
    method_names[3]="UM kernel + output prefetch";method_names[4]="UM kernel without placement requests";
    printf("MEMORY_MODE : %s\nUM shared allocation; allocation/advice outside timing; demand mode makes no prefetch requests.\n",backend_memory_mode());
    FILE *f=fopen("um_hardware.json","w");if(!f){perror("um_hardware.json");exit(1);}
    fprintf(f,"{\"name\":\"%s\",\"sm_count\":%d,\"warp_size\":%d,\"managed\":%d,\"concurrent\":%d,\"compiled_cuda\":%d,\"memory_mode\":\"%s\"}\n",p.name,p.multiProcessorCount,p.warpSize,managed,concurrent,CUDART_VERSION,backend_memory_mode());fclose(f);
}
extern "C" double backend_run_method(int method,double *a,int N,int delay,int max_iter,int size,const backend_config_t *cfg,int reps) {
    int block_count=cfg->control_a,thread_count=cfg->control_b,requested_teams=block_count,requested_threads=thread_count;
    cudaDeviceProp prop;check_cuda(cudaGetDeviceProperties(&prop,um_device),"UM launch properties");
    if(method<0||method>11||N<=0||size<=0||N>size||reps<=0||max_iter<0||max_iter>size||block_count<1||block_count>prop.maxGridSize[0]||thread_count<1||thread_count>prop.maxThreadsPerBlock||
       ((method==5||method==10)&&block_count>size)||(method==10&&(long long)(block_count-1)*(size/block_count)+thread_count>size)||(method==11&&(long long)block_count*thread_count>size)) {fprintf(stderr,"Invalid UM workload\n");exit(1);}
    cudaPointerAttributes attr;check_cuda(cudaPointerGetAttributes(&attr,a),"UM pointer");if(attr.type!=cudaMemoryTypeManaged){fprintf(stderr,"Driver must supply shared managed array\n");exit(1);}
    last_launch_blocks=(method<=4||method==7)?1:block_count;last_launch_threads=(method<=5||method==7)?1:thread_count;
    double *d_a=a,*d_tmp=NULL,*d_private=NULL,*d_team=NULL;
    cuda_omp_schedule_t schedule={};cudaStream_t stream=0;
    const bool resident=method==0||method>=5;
    if(resident){d_tmp=(double*)backend_managed_allocate((size_t)N*sizeof(double));memset(d_tmp,0,(size_t)N*sizeof(double));}
    if(method==9){size_t part=(size_t)block_count*thread_count;if(part>SIZE_MAX/(size_t)N/sizeof(double)){fprintf(stderr,"UM reduction size overflow\n");exit(1);}
        d_private=(double*)backend_managed_allocate(part*N*sizeof(double));d_team=(double*)backend_managed_allocate((size_t)block_count*N*sizeof(double));
        if(um_mode){um_prefetch(d_private,part*N*sizeof(double),um_device);um_prefetch(d_team,(size_t)block_count*N*sizeof(double),um_device);}}
    if(method==7)check_cuda(cudaStreamCreate(&stream),"UM stream");
    if(resident&&um_mode){um_prefetch(a,(size_t)size*sizeof(double),um_device);um_prefetch(d_tmp,(size_t)N*sizeof(double),um_device);}
    check_cuda(cudaDeviceSynchronize(),"UM setup");
    double start=get_time_usec();
    for(int rep=0;rep<reps;rep++) {
        if(um_mode&&(method==1||method==2))um_prefetch(a,(size_t)N*sizeof(double),um_device);
        switch(method){
        case 0:case 1:case 2:case 3:case 4: delay_one_kernel<<<1,1>>>(delay,d_a);break;
        case 5:block_scalar_kernel<<<block_count,1>>>(delay,d_a,size,requested_teams);break;
        case 6:loop_kernel<<<block_count,thread_count>>>(delay,d_a,max_iter,schedule);break;
        case 7:delay_one_kernel<<<1,1,0,stream>>>(delay,d_a);break;
        case 8:atomic_kernel<<<block_count,thread_count>>>(delay,d_a,d_tmp,max_iter,N,schedule);break;
        case 9:reduction_kernel<<<block_count,thread_count>>>(delay,d_a,d_tmp,d_private,d_team,max_iter,N,schedule);break;
        case 10:teams_parallel_kernel<<<block_count,thread_count>>>(delay,d_a,size,requested_teams);break;
        case 11:repeated_teams_parallel_kernel<<<block_count,thread_count>>>(delay,d_a,size,N,requested_threads);break;
        }
        check_cuda(cudaGetLastError(),"UM kernel launch");
        if(method==7)check_cuda(cudaStreamSynchronize(stream),"UM stream wait");else check_cuda(cudaDeviceSynchronize(),"UM kernel wait");
        if(um_mode&&(method==1||method==3))um_prefetch(a,(size_t)N*sizeof(double),cudaCpuDeviceId);
        // Same driver-side increment, now visible in the shared allocation.
        a[0]+=1.0;if(a[0]<0.0)printf("%f\n",a[0]);
    }
    double end=get_time_usec();
#ifdef MICROBENCHMARK_RESULT
    if(resident)MICROBENCHMARK_RESULT(a,d_tmp);
#endif
    if(method==7)check_cuda(cudaStreamDestroy(stream),"UM stream free");
    if(d_team)backend_managed_free(d_team);if(d_private)backend_managed_free(d_private);if(d_tmp)backend_managed_free(d_tmp);
    return (end-start)/reps;
}
extern "C" double backend_run_device_method(int,const double*,int,int,int,int,const backend_config_t*,int){fprintf(stderr,"No device-event mode in UM suite\n");exit(1);}
