// lab21_zero1.cu — Tier5: ZeRO-1 optimizer-state sharding + fused step.
// Problem: AdamW keeps m,v per parameter = 2x model memory per GPU. ZeRO-1
// partitions the *optimizer state*: each rank owns a contiguous shard of (m,v)
// and only steps its slice; grads for that slice arrive pre-reduced (allreduce).
// Source map: llm.c train_gpt2.cu zero_optimize / docs/audit/llm-c-09-zero-multigpu.md.
// Emulation: R ranks on one GPU; verify every element gets stepped exactly once
// and result equals a full single-GPU AdamW step (bitwise-tolerant check).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

__global__ void adamw_shard(float* p,const float* g,float* m,float* v,int lo,int hi,
                            float t,float lr,float b1,float b2,float eps,float wd){
    int i=lo+blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=hi) return;
    float gi=g[i];
    m[i]=b1*m[i]+(1.f-b1)*gi;
    v[i]=b2*v[i]+(1.f-b2)*gi*gi;
    float mh=m[i]/(1.f-powf(b1,t)), vh=v[i]/(1.f-powf(b2,t));
    p[i]=p[i]*(1.f-lr*wd)-lr*mh/(sqrtf(vh)+eps);
}
int main(){
    const size_t N=1<<18, R=4; int per=(int)(N/R);
    float *p=(float*)malloc(N*4),*g=(float*)malloc(N*4);
    fill_rand(p,N,90);fill_rand(g,N,91);
    // CPU reference: full single-device AdamW, T steps
    const float lr=3e-4f,b1=0.9f,b2=0.95f,eps=1e-8f,wd=1e-1f; const int T=5;
    double* ref=(double*)malloc(N*8);
    { float* pm=(float*)calloc(N,4),*pv=(float*)calloc(N,4); float* pc=(float*)malloc(N*4); memcpy(pc,p,N*4);
      for(int t=1;t<=T;t++) for(size_t i=0;i<N;i++){
        float gi=g[i]; pm[i]=b1*pm[i]+(1-b1)*gi; pv[i]=b2*pv[i]+(1-b2)*gi*gi;
        float mh=pm[i]/(1-powf(b1,(float)t)),vh=pv[i]/(1-powf(b2,(float)t));
        pc[i]=pc[i]*(1-lr*wd)-lr*mh/(sqrtf(vh)+eps);}
      for(size_t i=0;i<N;i++) ref[i]=pc[i]; free(pm);free(pv);free(pc);}
    // GPU ZeRO emulation: R shards, each with private m,v over its range only
    float *gp,*gg; CK(cudaMalloc(&gp,N*4));CK(cudaMalloc(&gg,N*4));
    CK(cudaMemcpy(gp,p,N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gg,g,N*4,cudaMemcpyHostToDevice));
    float* gm=(float*)calloc(N,4); float* gv=(float*)calloc(N,4); // host-side per-rank state sim: keep in device arrays sized per-shard
    float* dgm; float* dgv; CK(cudaMalloc(&dgm,N*4)); CK(cudaMalloc(&dgv,N*4));
    CK(cudaMemset(dgm,0,N*4)); CK(cudaMemset(dgv,0,N*4));
    for(int t=1;t<=T;t++){
        for(int r=0;r<(int)R;r++){
            int lo=r*per, hi=(r==R-1)?(int)N:lo+per;
            int n=hi-lo; unsigned grd=(unsigned)((n+255)/256);
            // shard-local indices: pass global arrays but restrict [lo,hi)
            adamw_shard<<<grd,256>>>(gp,gg,dgm,dgv,lo,hi,(float)t,lr,b1,b2,eps,wd);
        }
    }
    CK(cudaDeviceSynchronize());
    float* out=(float*)malloc(N*4); CK(cudaMemcpy(out,gp,N*4,cudaMemcpyDeviceToHost));
    check_close("ZeRO-1 sharded step == full step",ref,out,N,1e-5f);
    // coverage check: no shard overlap => sum of shard sizes == N
    printf("shards: %d x ~%d elems, last=%d (covers tail)\n",(int)R,per,(int)N-(int)R*per+per);
    free(p);free(g);free(ref);free(out);free(gm);free(gv);
    cudaFree(gp);cudaFree(gg);cudaFree(dgm);cudaFree(dgv);
    return report();
}
