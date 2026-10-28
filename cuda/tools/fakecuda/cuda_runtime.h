#ifndef CUDA_RUNTIME_STUB_H
#define CUDA_RUNTIME_STUB_H
// Host-side stub for syntax checking .cu files without nvcc/GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
typedef int cudaError_t;
typedef void* cudaEvent_t;
typedef void* cudaStream_t;
struct cudaDeviceProp { char name[256]; int multiProcessorCount, major, minor, clockRate; size_t totalGlobalMem; };
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define __global__
#define __device__
#define __host__
#define __shared__ static
#define blockDim  (dim3_blockDim)
#define blockIdx  (dim3_blockIdx)
#define threadIdx (dim3_threadIdx)
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
static dim3 dim3_blockDim, dim3_blockIdx, dim3_threadIdx;
// kernel launch replacement: CP_LAUNCH(grid,block,args...) -> just call fn(args...)
#define CP_LAUNCH(...) /* swallow launch; function called separately */
inline const char* cudaGetErrorString(cudaError_t e){return e?"stub-err":"stub-ok";}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp*p,int d){(void)d;p->name[0]='s';p->multiProcessorCount=1;p->major=8;p->minor=6;p->clockRate=1000;p->totalGlobalMem=1ULL<<30;return 0;}
inline cudaError_t cudaEventCreate(cudaEvent_t*e){*e=(void*)1;return 0;}
inline cudaError_t cudaEventDestroy(cudaEvent_t e){(void)e;return 0;}
inline cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t s=0){(void)e;(void)s;return 0;}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e){(void)e;return 0;}
inline cudaError_t cudaEventElapsedTime(float*m,cudaEvent_t a,cudaEvent_t b){(void)a;(void)b;*m=0;return 0;}
inline cudaError_t cudaDeviceSynchronize(){return 0;}
template<typename T> inline cudaError_t cudaMalloc(T**p,size_t n){*p=(T*)std::calloc(n?n:1,1);return 0;}
inline cudaError_t cudaFree(void*p){std::free(p);return 0;}
inline cudaError_t cudaMemcpy(void*d,const void*s,size_t n,int k){(void)k;std::memcpy(d,s,n);return 0;}
inline cudaError_t cudaGetLastError(){return 0;}
// warp shuffle stubs (sequential semantics on host)
inline float __shfl_down_sync(unsigned m,float v,unsigned d,unsigned w=32){(void)m;(void)d;(void)w;return v;}
inline float __shfl_xor_sync(unsigned m,float v,unsigned l,unsigned w=32){(void)m;(void)l;(void)w;return v;}
inline float __shfl_sync(unsigned m,float v,unsigned s,unsigned w=32){(void)m;(void)s;(void)w;return v;}
inline unsigned __activemask(){return 0xffffffffu;}
inline int __popc(unsigned v){int c=0;while(v){c+=v&1;v>>=1;}return c;}
inline float __fmul_rn(float a,float b){return a*b;}
inline float __fadd_rn(float a,float b){return a+b;}
#define __ldcs(p) (*(p))
#define __stcs(p,v) (*(p)=(v))
#define warpSize 32
inline cudaError_t cudaMemset(void*p,int v,size_t n){std::memset(p,v,n);return 0;}
inline float atomicAdd(float*addr,float v){float o=*addr;*addr=o+v;return o;}
#define __forceinline__ inline
#ifndef __CUDACC__
struct float4 { float x,y,z,w; };
struct double4 { double x,y,z,w; };
struct int4 { int x,y,z,w; };
inline void __syncthreads(){}

#ifndef __CUDACC__
inline float rsqrtf(float v){return 1.0f/std::sqrt(v);}
inline float expf_(float v){return std::exp(v);}
#endif
#endif

#endif
