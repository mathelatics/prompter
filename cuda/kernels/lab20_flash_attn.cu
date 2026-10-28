// lab20_flash_attn.cu — Tier4: online-softmax causal attention (flash-style, fp32).
// Problem: materializing the T x T score matrix blows memory at long context.
// FlashAttention computes softmax(QK^T)V in tiles with an online (running) max
// and rescaling accumulator. Source map: llm.c cudnn_att_forward_kernel_f32_cu
// and our audit docs/audit/llm-c-08-kernels-bf16.md; math from lab07 softmax.
// Layout: single head, Q,K,V [T,C], causal mask, output O [T,C].
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define BLOCK 64

__global__ void flash_attn_fwd(const float* __restrict__ Q,const float* __restrict__ K,
                               const float* __restrict__ V,float* __restrict__ O,int T,int C){
    int q0=blockIdx.x*BLOCK;                 // query tile start
    extern __shared__ float sm[];            // reused for k/v tiles
    float* Ks=sm; float* Vs=sm+BLOCK*C;
    float m_i=-INFINITY, l_i=0.f;
    int tid=threadIdx.x;                      // one thread per query row (C loop serial)
    if(q0+tid>=T) return;
    const float* q=Q+(size_t)(q0+tid)*C;
    double acc[512]; // C<=512 assumed for this teaching kernel
    for(int c=0;c<C;c++) acc[c]=0.0;
    for(int kv0=0;kv0<=q0+tid;kv0+=BLOCK){
        int nk=(BLOCK<T-kv0)?BLOCK:T-kv0;
        for(int i=tid;i<nk*C;i+=blockDim.x) { Ks[i]=K[(size_t)(kv0+i/C)*C+i%C]; }
        for(int i=tid;i<nk*C;i+=blockDim.x) { Vs[i]=V[(size_t)(kv0+i/C)*C+i%C]; }
        __syncthreads();
        // scores for my query vs all keys in tile
        float s[64]; int ns=0; float mt=m_i;
        for(int j=0;j<nk && kv0+j<=q0+tid;j++){
            float d=0; for(int c=0;c<C;c++) d+=q[c]*Ks[j*C+c];
            s[ns++]=d/sqrtf((float)C); mt=fmaxf(mt,s[ns-1]);
        }
        float corr = (m_i==-INFINITY)?1.f:expf(m_i-mt);
        for(int c=0;c<C;c++){ acc[c]*=corr; }
        l_i*=corr;
        for(int j=0;j<ns;j++){ float p=expf(s[j]-mt); l_i+=p;
            for(int c=0;c<C;c++) acc[c]+=p*Vs[j*C+c]; }
        m_i=mt;
        __syncthreads();
    }
    float inv=1.f/l_i;
    for(int c=0;c<C;c++) O[(size_t)(q0+tid)*C+c]=(float)acc[c]*inv;
}

int main(){
    int T=256,C=64; size_t n=(size_t)T*C;
    float *q=(float*)malloc(n*4),*k=(float*)malloc(n*4),*v=(float*)malloc(n*4),*o=(float*)malloc(n*4);
    fill_rand(q,n,80);fill_rand(k,n,81);fill_rand(v,n,82);
    float *gq,*gk,*gv,*go; CK(cudaMalloc(&gq,n*4));CK(cudaMalloc(&gk,n*4));CK(cudaMalloc(&gv,n*4));CK(cudaMalloc(&go,n*4));
    CK(cudaMemcpy(gq,q,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gk,k,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gv,v,n*4,cudaMemcpyHostToDevice));
    size_t shm=2*BLOCK*C*4;
    cudaFuncSetAttribute(flash_attn_fwd,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm);
    flash_attn_fwd<<<(T+BLOCK-1)/BLOCK,BLOCK,shm>>>(gq,gk,gv,go,T,C);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o,go,n*4,cudaMemcpyDeviceToHost));
    // CPU oracle: causal attention in double
    double* ref=(double*)malloc(n*8);
    for(int i=0;i<T;i++){
        double m=-INFINITY,l=0; double a[512]={0};
        for(int j=0;j<=i;j++){ double d=0; for(int c=0;c<C;c++) d+=(double)q[i*C+c]*k[j*C+c]; d/=sqrt((double)C);
            m=fmax(m,d); }
        // recompute properly
        l=0; for(int c=0;c<C;c++)a[c]=0;
        for(int j=0;j<=i;j++){ double d=0; for(int c=0;c<C;c++) d+=(double)q[i*C+c]*k[j*C+c]; d/=sqrt((double)C);
            double p=exp(d-m); l+=p; for(int c=0;c<C;c++) a[c]+=p*(double)v[j*C+c]; }
        for(int c=0;c<C;c++) ref[i*C+c]=a[c]/l;
    }
    check_close("flash attn fwd vs CPU",ref,o,n,2e-3f);
    free(q);free(k);free(v);free(o);free(ref);
    cudaFree(gq);cudaFree(gk);cudaFree(gv);cudaFree(go);
    return report();
}
