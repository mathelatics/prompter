// lab17_swiglu.cu — Tier3: SwiGLU MLP forward + backward (llama3 FFN shape).
// Math: h = swish(x@W1) * (x@W3);  out = h@W2   (rough/llm.c train_llama3.py
// uses dim_ffn=14336, hidden=4096; we use small dims for the oracle loop.)
// swish(z) = z*sigmoid(z); gelu-approx not used here — SwiGLU is exact.
// Backward chain (three matmuls fused elementwise in one kernel):
//   dh    = dy @ W2^T
//   d1    = dh * swish'(g) * s     where g=x@W1, s=x@W3
//   d3    = dh * g * sigmoid(s)... -> standard: u=swish(g), v=silu? We use:
//   y = swish(g)*s ; dg = ds*s'... keep it simple & verified vs double oracle.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

static __device__ float swish(float z){ return z/(1.f+expf(-z)); }
static __device__ float swish_d(float z){ float s=1.f/(1.f+expf(-z)); return s*(1.f+z*(1.f-s)); }

// forward elementwise: given g,h pre-activated by cuBLAS-style matmuls elsewhere,
// fuse: y[i] = swish(g[i]) * s[i]
__global__ void swiglu_fwd(const float* __restrict__ g,const float* __restrict__ s,
                           float* __restrict__ y,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=swish(g[i])*s[i];
}
// backward: dg = dy*swish(g)*... no: d/dg[swish(g)*s]=dy*s*swish'(g);
//           ds = dy*swish(g)
__global__ void swiglu_bwd(const float* __restrict__ dy,const float* __restrict__ g,
                           const float* __restrict__ s,float* __restrict__ dg,
                           float* __restrict__ ds,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float gi=g[i],si=s[i];
    dg[i]=dy[i]*si*swish_d(gi);
    ds[i]=dy[i]*swish(gi);
}
int main(){
    size_t n=1u<<20;
    float *g=(float*)malloc(n*4),*s=(float*)malloc(n*4),*y=(float*)malloc(n*4);
    float *dy=(float*)malloc(n*4),*dg=(float*)malloc(n*4),*ds=(float*)malloc(n*4);
    fill_rand(g,n,51);fill_rand(s,n,52);fill_rand(dy,n,53);
    float *gg,*gs,*gy,*gdy,*gdg,*gds;
    CK(cudaMalloc(&gg,n*4));CK(cudaMalloc(&gs,n*4));CK(cudaMalloc(&gy,n*4));
    CK(cudaMalloc(&gdy,n*4));CK(cudaMalloc(&gdg,n*4));CK(cudaMalloc(&gds,n*4));
    CK(cudaMemcpy(gg,g,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gs,s,n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gdy,dy,n*4,cudaMemcpyHostToDevice));
    unsigned grd=(unsigned)((n+255)/256);
    swiglu_fwd<<<grd,256>>>(gg,gs,gy,n);
    swiglu_bwd<<<grd,256>>>(gdy,gg,gs,gdg,gds,n);
    CK(cudaDeviceSynchronize());CK(cudaGetLastError());
    CK(cudaMemcpy(y,gy,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dg,gdg,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ds,gds,n*4,cudaMemcpyDeviceToHost));
    double *oy=(double*)malloc(n*8),*odg=(double*)malloc(n*8),*ods=(double*)malloc(n*8);
    for(size_t i=0;i<n;i++){
        double gi=g[i],si=s[i],d=dy[i];
        double sg=1.0/(1.0+exp(-gi)), sw=gi*sg;
        oy[i]=sw*si; odg[i]=d*si*sg*(1.0+gi*(1.0-sg)); ods[i]=d*sw;
    }
    check_close("swiglu fwd",oy,y,n,1e-3f);
    check_close("swiglu bwd dg",odg,dg,n,1e-3f);
    check_close("swiglu bwd ds",ods,ds,n,1e-3f);
    float ms=bench([&]{swiglu_fwd<<<grd,256>>>(gg,gs,gy,n);},2,50);
    printf("swiglu fwd %.3f ms (%.1f GB/s)\n",ms,(3.0*n*4)/ms/1e6);
    free(g);free(s);free(y);free(dy);free(dg);free(ds);free(oy);free(odg);free(ods);
    cudaFree(gg);cudaFree(gs);cudaFree(gy);cudaFree(gdy);cudaFree(gdg);cudaFree(gds);
    return report();
}
