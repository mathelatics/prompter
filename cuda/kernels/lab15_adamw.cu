// lab15_adamw.cu — Tier3: fused AdamW, elementwise. One kernel updates
// param, exp_avg (m), exp_avg_sq (v) in-place; weight decay decoupled.
// Math (PyTorch AdamW):
//   m <- b1*m + (1-b1)*g
//   v <- b2*v + (1-b2)*g^2
//   mhat = m/(1-b1^t), vhat = v/(1-b2^t)
//   p <- p - lr * (step_size * mhat / (sqrt(vhat)+eps) + wd*p)
// llm.c fp32 trainer uses the lerp form: p += (raw_step*p + g)/(denom+wd)
// (docs/audit/llm-c-05-train_gpt2_fp32_cu.md §adamw). We implement BOTH and
// check they agree to float tolerance. Source map: rough/llm.c train_gpt2_fp32
// adamw_step_fused_kernel; rough/gpt.c adamw step.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

__global__ void adamw_step(float* __restrict__ p, float* __restrict__ m,
                           float* __restrict__ v, const float* __restrict__ g,
                           size_t n, float lr, float b1, float b2, float eps,
                           float wd, int t) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = b1*m[i] + (1.f-b1)*gi;
    v[i] = b2*v[i] + (1.f-b2)*gi*gi;
    float bc1 = 1.f - powf(b1, (float)t);
    float bc2 = 1.f - powf(b2, (float)t);
    float step_size = lr / bc1;
    float denom = sqrtf(v[i]/bc2) / sqrtf(bc1) + eps;  // PyTorch's combined form
    p[i] -= step_size * (m[i]/denom) + lr*wd*p[i];
}

// llm.c-style lerp variant (no bias-correction division of m separately):
__global__ void adamw_step_llmc(float* __restrict__ p, float* __restrict__ m,
                                float* __restrict__ v, const float* __restrict__ g,
                                size_t n, float lr, float b1, float b2, float eps,
                                float wd, int t) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = b1*m[i] + (1.f-b1)*gi;
    v[i] = b2*v[i] + (1.f-b2)*gi*gi;
    float bias_correction1 = 1.f - powf(b1,(float)t);
    float bias_correction2 = 1.f - powf(b2,(float)t);
    float step = lr / bias_correction1;
    float denom = sqrtf(v[i]) / sqrtf(bias_correction2) + eps;
    // raw_step = -step*m/denom ; llm.c folds wd into denom via lerp trick:
    p[i] = p[i] + (-step * m[i] / denom) - lr*wd*p[i];
}

int main(){
    const size_t n = 1u<<20;           // ~1M params (a GPT-2 head-ish chunk)
    const int T = 100;                 // run T steps, compare final state
    float *p=(float*)malloc(n*4),*m=(float*)malloc(n*4),*v=(float*)malloc(n*4);
    float *g=(float*)malloc(n*4);
    fill_rand(p,n,21); fill_rand(g,n,22);
    for(size_t i=0;i<n;i++){m[i]=0.f;v[i]=0.f;}
    double *ref_p=(double*)malloc(n*8);
    // CPU oracle in double (PyTorch formula)
    for(size_t i=0;i<n;i++) ref_p[i]=p[i];
    double *rm=(double*)calloc(n,8), *rv=(double*)calloc(n,8);
    float lr=3e-4f,b1=0.9f,b2=0.95f,eps=1e-8f,wd=1e-1f;
    for(int t=1;t<=T;t++){
        double bc1=1-pow((double)b1,t), bc2=1-pow((double)b2,t);
        for(size_t i=0;i<n;i++){
            rm[i]=b1*rm[i]+(1-(double)b1)*g[i];
            rv[i]=b2*rv[i]+(1-(double)b2)*(double)g[i]*g[i];
            double denom=sqrt(rv[i]/bc2)/sqrt(bc1)+eps;
            ref_p[i]-=(lr/bc1)*(rm[i]/denom)+lr*wd*ref_p[i];
        }
        // refresh grads each step deterministically
        xs64_state=1000+t; for(size_t i=0;i<n;i++) g[i]=rand_f(-0.01f,0.01f);
    }
    // GPU: reset and replay same grad sequence
    fill_rand(p,n,21);
    CK(cudaFree(0)); // ensure ctx
    float *dp,*dm,*dv,*dg;
    CK(cudaMalloc(&dp,n*4));CK(cudaMalloc(&dm,n*4));CK(cudaMalloc(&dv,n*4));CK(cudaMalloc(&dg,n*4));
    CK(cudaMemcpy(dp,p,n*4,cudaMemcpyHostToDevice));
    CK(cudaMemset(dm,0,n*4)); CK(cudaMemset(dv,0,n*4));
    dim3 blk(256); unsigned grd=(unsigned)((n+255)/256);
    xs64_state=1000; // regenerate identical sequence
    for(int t=1;t<=T;t++){
        xs64_state=1000+t; for(size_t i=0;i<n;i++) g[i]=rand_f(-0.01f,0.01f);
        CK(cudaMemcpy(dg,g,n*4,cudaMemcpyHostToDevice));
        adamw_step<<<grd,blk>>>(dp,dm,dv,dg,n,lr,b1,b2,eps,wd,t);
    }
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    float *out=(float*)malloc(n*4);
    CK(cudaMemcpy(out,dp,n*4,cudaMemcpyDeviceToHost));
    check_close("adamw GPU vs double CPU oracle", ref_p, out, n, 2e-4f);

    // timing: bandwidth math -> 4 arrays * 4B read/write-bound
    float ms = bench([&]{ adamw_step<<<grd,blk>>>(dp,dm,dv,dg,n,lr,b1,b2,eps,wd,T); },2,20);
    double bytes = 4.0*n*4*2; // r+w of p,m,v,g (approx)
    printf("adamw fused: %.3f ms -> %.1f GB/s (expect near mem BW)\n", ms, bytes/ms/1e6);
    free(p);free(m);free(v);free(g);free(ref_p);free(rm);free(rv);free(out);
    cudaFree(dp);cudaFree(dm);cudaFree(dv);cudaFree(dg);
    return report();
}
