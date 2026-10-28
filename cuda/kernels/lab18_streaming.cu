// lab18_streaming.cu — Tier4: __ldcs/__stcs streaming cache hints + copy ladder.
// Problem: bulk copies evict L2-resident data a later kernel needs. Streaming
// loads/stores bypass/evict-first policy keep hot tensors in cache.
// Source map: rough/llm.c layernorm_forward_kernel uses __ldcs on residual reads
// and __stcs on outputs (docs/audit/llm-c-05-train_gpt2_fp32_cu.md §memory).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

__global__ void copy_plain(const float4* __restrict__ src,float4* __restrict__ dst,size_t n4){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n4) dst[i]=src[i];
}
__global__ void copy_stream(const float4* __restrict__ src,float4* __restrict__ dst,size_t n4){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n4) __stcs(&dst[i],__ldcs(&src[i]));
}
int main(){
    size_t n=1u<<24; // 64 MB floats
    float *a=(float*)malloc(n*4),*b=(float*)malloc(n*4);
    fill_rand(a,n,61);
    float *ga,*gb; CK(cudaMalloc(&ga,n*4));CK(cudaMalloc(&gb,n*4));
    CK(cudaMemcpy(ga,a,n*4,cudaMemcpyHostToDevice));
    size_t n4=n/4; unsigned grd=(unsigned)((n4+255)/256);
    float t1=bench([&]{copy_plain<<<grd,256>>>((const float4*)ga,(float4*)gb,n4);},2,20);
    float t2=bench([&]{copy_stream<<<grd,256>>>((const float4*)ga,(float4*)gb,n4);},2,20);
    CK(cudaMemcpy(b,gb,n*4,cudaMemcpyDeviceToHost));
    double *ra=(double*)malloc(n*8); for(size_t i=0;i<n;i++)ra[i]=a[i];
    check_close("stream copy correctness",ra,b,n,0.f);
    printf("plain  %.3f ms -> %.1f GB/s\nstream %.3f ms -> %.1f GB/s (hints can be neutral or help when L2 reuse matters)\n",
           t1,(2.0*n*4)/t1/1e6,t2,(2.0*n*4)/t2/1e6);
    free(a);free(b);free(ra);cudaFree(ga);cudaFree(gb);
    return report();
}
