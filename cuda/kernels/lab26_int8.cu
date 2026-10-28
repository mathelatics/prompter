// lab26_int8.cu — per-channel symmetric int8 quantization + __dp4a dot product.
// Problem: W8A8 inference cuts memory traffic 4x; dequant needs per-output-channel
// scales. __dp4a computes 4 x (int8*int8 -> int32 accumulate) per instruction on
// any CUDA core (Turing+). Source map: llm.c bf16 weight path motivation + our
// tier-3 project idea "FP8/INT8 mixed precision" (docs/audit cuda-kernels plan §Tier6).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "harness.h"

#define K 256   // reduction dim, multiple of 4 for dp4a
// On real GPUs (nvcc) __dp4a is a hardware intrinsic (sm_61+). For the host
// typecheck path we provide an identical scalar implementation so the code and
// semantics are the same either way.
#ifdef CP_HOST_TYPECHECK
static inline int __dp4a(int acc,unsigned a,unsigned b){
    int s=acc; for(int i=0;i<4;i++){ signed char x=(signed char)((a>>(8*i))&0xFF),y=(signed char)((b>>(8*i))&0xFF); s+=(int)x*(int)y; } return s; }
#endif

__global__ void w8_dot(const unsigned* __restrict__ A,const unsigned* __restrict__ B,
                       const float* __restrict__ scaleA,const float* __restrict__ scaleB,
                       float* __restrict__ C,int M,int N){
    int m=blockIdx.x*blockDim.x+threadIdx.x; if(m>=M) return;
    int n=blockIdx.y;
    int acc=0;
    for(int kk=0;kk<K/4;kk++) acc=__dp4a(acc,A[m*(K/4)+kk],B[n*(K/4)+kk]);
    C[m*N+n]=(float)acc*scaleA[m]*scaleB[n];
}
int main(){
    const int M=64,N=32; size_t na=(size_t)M*K, nb=(size_t)N*K;
    float *Af=(float*)malloc(na*4),*Bf=(float*)malloc(nb*4),*Cf=(float*)malloc((size_t)M*N*4);
    fill_rand(Af,na,130);fill_rand(Bf,nb,131);
    // quantize per-row (A) and per-col-of-B (output channel): scale = max|.|/127
    float *sA=(float*)malloc(M*4),*sB=(float*)malloc(N*4);
    unsigned char *qA=(unsigned char*)malloc(na),*qB=(unsigned char*)malloc(nb);
    for(int m=0;m<M;m++){ float mx=0; for(int k=0;k<K;k++)mx=fmaxf(mx,fabsf(Af[m*K+k])); sA[m]=mx/127.f;
        for(int k=0;k<K;k++){ int v=(int)lroundf(Af[m*K+k]/sA[m]); if(v>127)v=127;if(v<-127)v=-127; qA[m*K+k]=(unsigned char)(signed char)v; } }
    for(int n=0;n<N;n++){ float mx=0; for(int k=0;k<K;k++)mx=fmaxf(mx,fabsf(Bf[n*K+k])); sB[n]=mx/127.f;
        for(int k=0;k<K;k++){ int v=(int)lroundf(Bf[n*K+k]/sB[n]); if(v>127)v=127;if(v<-127)v=-127; qB[n*K+k]=(unsigned char)(signed char)v; } }
    // fp32 reference matmul
    double* ref=(double*)malloc((size_t)M*N*8);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)Af[m*K+k]*Bf[n*K+k]; ref[m*N+n]=s; }
    // GPU int8 path
    unsigned *gA,*gB; float *gsA,*gsB,*gC;
    CK(cudaMalloc(&gA,na));CK(cudaMalloc(&gB,nb));CK(cudaMalloc(&gsA,M*4));CK(cudaMalloc(&gsB,N*4));CK(cudaMalloc(&gC,(size_t)M*N*4));
    CK(cudaMemcpy(gA,qA,na,cudaMemcpyHostToDevice));CK(cudaMemcpy(gB,qB,nb,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gsA,sA,M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gsB,sB,N*4,cudaMemcpyHostToDevice));
    dim3 grd((M+31)/32,N); w8_dot<<<grd,32>>>((const unsigned*)gA,(const unsigned*)gB,gsA,gsB,gC,M,N);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(Cf,gC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    // error budget: quant noise ~ relative L2 should be < 3% for random data
    double num=0,den=0; for(int i=0;i<M*N;i++){ double d=Cf[i]-ref[i]; num+=d*d; den+=ref[i]*ref[i]; }
    double rel=sqrt(num/den);
    char msg[128]; snprintf(msg,sizeof msg,"rel_l2=%.4f (budget<0.03)",rel);
    check("int8 W8A8 within error budget",rel<0.03,msg);
    free(Af);free(Bf);free(Cf);free(ref);free(sA);free(sB);free(qA);free(qB);
    cudaFree(gA);cudaFree(gB);cudaFree(gsA);cudaFree(gsB);cudaFree(gC);
    return report();
}
