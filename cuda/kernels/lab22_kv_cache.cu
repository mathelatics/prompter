// lab22_kv_cache.cu — Tier6: paged KV-cache decode step (inference).
// Problem: autoregressive decode recomputes attention over all past tokens.
// Cache K,V rows per sequence slot; a decode step appends one row and does a
// single-query causal dot against the cache. Source map: llm.c train_gpt2_fp32
// cuda forward attention + our docs/audit/llm-c-05 §attention; gpt.c inference path.
// Layout: cacheK/cacheV [B, MAXT, C]; lens[B] current length; new token k,v per batch.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define B 4
#define MAXT 512
#define C 64

__global__ void kv_append(const float* __restrict__ newk,const float* __restrict__ newv,
                          float* K,float* V,const int* lens){
    int b=blockIdx.x; int len=lens[b];
    for(int c=threadIdx.x;c<C;c+=blockDim.x){
        K[((size_t)b*MAXT+len)*C+c]=newk[b*C+c];
        V[((size_t)b*MAXT+len)*C+c]=newv[b*C+c];
    }
}
// one block per batch: q . K[0..len] -> softmax -> .V
__global__ void kv_decode(const float* __restrict__ Q,const float* K,const float* V,
                          const int* lens,float* O){
    int b=blockIdx.x,len=lens[b]+1; // after append
    extern __shared__ float smem[];
    float* s=(float*)smem;
    const float* q=Q+(size_t)b*C;
    float sc[256]; if(len>256) return; // teaching cap
    if(threadIdx.x==0){
        double m=-INFINITY,l=0,a[512]={0};
        for(int j=0;j<len;j++){ double d=0; const float* k=K+((size_t)b*MAXT+j)*C;
            for(int c=0;c<C;c++) d+=(double)q[c]*k[c]; d/=sqrt((double)C); sc[j]=(float)d; m=fmax(m,d);}
        for(int j=0;j<len;j++){ double p=exp((double)sc[j]-m); l+=p; const float* v=V+((size_t)b*MAXT+j)*C;
            for(int c=0;c<C;c++) a[c]+=p*(double)v[c]; }
        for(int c=0;c<C;c++) s[c]=(float)(a[c]/l);
    }
    __syncthreads();
    if(threadIdx.x<C) O[b*C+threadIdx.x]=s[threadIdx.x];
}
int main(){
    size_t nk=(size_t)B*C;
    float *q=(float*)malloc(nk*4),*newk=(float*)malloc(nk*4),*newv=(float*)malloc(nk*4),*o=(float*)malloc(nk*4);
    fill_rand(q,nk,100);fill_rand(newk,nk,101);fill_rand(newv,nk,102);
    int lens[B]; for(int b=0;b<B;b++) lens[b]=17+b*33;
    size_t kc=(size_t)B*MAXT*C;
    float *K=(float*)calloc(kc,4),*V=(float*)calloc(kc,4);
    // pre-fill cache with random history
    for(size_t i=0;i<(size_t)B;lens[0],i++); // noop keep compiler calm
    { unsigned long long sd=123; for(int b=0;b<B;b++) for(int t=0;t<lens[b];t++) for(int c=0;c<C;c++){
        sd=sd*6364136223846793005ull+1442695040888963407ull;
        K[((size_t)b*MAXT+t)*C+c]=((sd>>40)&0xFFFFF)/524288.f-1.f;
        sd=sd*6364136223846793005ull+1442695040888963407ull;
        V[((size_t)b*MAXT+t)*C+c]=((sd>>40)&0xFFFFF)/524288.f-1.f; } }
    float *gq,*gk,*gv,*go,*gnk,*gnv; int *gl;
    CK(cudaMalloc(&gq,nk*4));CK(cudaMalloc(&gk,kc*4));CK(cudaMalloc(&gv,kc*4));CK(cudaMalloc(&go,nk*4));
    CK(cudaMalloc(&gnk,nk*4));CK(cudaMalloc(&gnv,nk*4));CK(cudaMalloc(&gl,B*4));
    CK(cudaMemcpy(gq,q,nk*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gk,K,kc*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gv,V,kc*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gnk,newk,nk*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gnv,newv,nk*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gl,lens,B*4,cudaMemcpyHostToDevice));
    kv_append<<<B,C>>>(gnk,gnv,gk,gv,gl);
    size_t shm=C*4;
    cudaFuncSetAttribute(kv_decode,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shm);
    kv_decode<<<B,C,shm>>>(gq,gk,gv,gl,go);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o,go,nk*4,cudaMemcpyDeviceToHost));
    // CPU oracle incl appended row
    double* ref=(double*)malloc(nk*8);
    for(int b=0;b<B;b++){ int len=lens[b]+1;
        K[((size_t)b*MAXT+lens[b])*C+0]=newk[b*C+0]; // ensure host copy has appended too (all c below)
        for(int c=0;c<C;c++){ K[((size_t)b*MAXT+lens[b])*C+c]=newk[b*C+c]; V[((size_t)b*MAXT+lens[b])*C+c]=newv[b*C+c]; }
        double m=-INFINITY,l=0; double sc[256],a[512]={0};
        for(int j=0;j<len;j++){ double d=0; for(int c=0;c<C;c++) d+=(double)q[b*C+c]*K[((size_t)b*MAXT+j)*C+c]; d/=sqrt((double)C); sc[j]=d; m=fmax(m,d);}
        for(int j=0;j<len;j++){ double p=exp(sc[j]-m); l+=p; for(int c=0;c<C;c++) a[c]+=p*(double)V[((size_t)b*MAXT+j)*C+c]; }
        for(int c=0;c<C;c++) ref[b*C+c]=a[c]/l; }
    check_close("KV-cache decode == full recompute",ref,o,nk,2e-3f);
    free(q);free(newk);free(newv);free(o);free(K);free(V);free(ref);
    cudaFree(gq);cudaFree(gk);cudaFree(gv);cudaFree(go);cudaFree(gnk);cudaFree(gnv);cudaFree(gl);
    return report();
}
