// lab13_fused_add_ln.cu — Tier3: fused residual-add + LayerNorm forward.
// Unfused chain (llm.c fp32 trainer): x2 = x + r   (kernel A)
//                                     y = LN(x2)  (kernel B, layernorm_forward)
// Fused: ONE pass per row -> write x2 once, compute mean/var from registers/
// shared, write y. Halves HBM round-trips for the intermediate.
// Math: mu = mean_d(x); s2 = var_d(x); y = gamma*(x-mu)/sqrt(s2+eps) + beta
// eps=1e-5 (docs/audit/llm-c-03 §layernorm). Backward note: llm.c stores x2 and
// recomputes stats in backward — we emit x2 as an output too so that pattern
// ports directly. Source map: rough/llm.c train_gpt2_fp32 layer kernels;
// docs/audit/llm-c-05-train_gpt2_fp32_cu.md (layer_norm_forward_kernel).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define C 768            // GPT-2 small hidden size
#define EPS 1e-5f

// warp-per-row variant: block = 256 threads = 8 warps, each warp does one row.
__global__ void fused_add_ln(const float* __restrict__ x, const float* __restrict__ r,
                             const float* __restrict__ w, const float* __restrict__ b,
                             float* __restrict__ x2, float* __restrict__ y, int rows){
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x/32) + warp;
    if (row >= rows) return;
    const float* xr = x + (size_t)row*C;
    const float* rr = r + (size_t)row*C;
    float* x2r = x2 + (size_t)row*C;
    float* yr  = y  + (size_t)row*C;
    float sums[24]; // C/32 = 24 elements per lane
    float s=0.f, ss=0.f;
    #pragma unroll
    for (int i=0;i<24;i++){
        int d = lane + 32*i;
        float v = xr[d] + rr[d];      // residual add happens HERE, once
        x2r[d] = v;                    // store for backward recompute
        sums[i]=v; s+=v; ss+=v*v;
    }
    // butterfly reduce sum & sumsq across the warp (__shfl_down_sync)
    #pragma unroll
    for (int off=16;off>0;off>>=1){
        s  += __shfl_down_sync(0xffffffffu, s,  off);
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    }
    s  = __shfl_sync(0xffffffffu, s,  0);
    ss = __shfl_sync(0xffffffffu, ss, 0);
    float mu = s / C;
    float var = ss / C - mu*mu;                 // E[x^2]-E[x]^2 (one-pass)
    float rstd = rsqrtf(var + EPS);
    #pragma unroll
    for (int i=0;i<24;i++){
        int d = lane + 32*i;
        yr[d] = w[d]*(sums[i]-mu)*rstd + b[d];
    }
}

int main(){
    const int BT = 1024;                        // tokens
    size_t n=(size_t)BT*C;
    float *x=(float*)malloc(n*4),*r=(float*)malloc(n*4);
    float *w=(float*)malloc(C*4),*b=(float*)malloc(C*4);
    fill_rand(x,n,41); fill_rand(r,n,42);
    for(int i=0;i<C;i++){ w[i]=1.f+0.01f*((i%7)-3); b[i]=0.001f*(i%5); }
    float *y=(float*)malloc(n*4),*x2=(float*)malloc(n*4);
    float *gx,*gr,*gw,*gb,*gy,*gx2;
    CK(cudaMalloc(&gx,n*4));CK(cudaMalloc(&gr,n*4));CK(cudaMalloc(&gw,C*4));CK(cudaMalloc(&gb,C*4));
    CK(cudaMalloc(&gy,n*4));CK(cudaMalloc(&gx2,n*4));
    CK(cudaMemcpy(gx,x,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gr,r,n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gw,w,C*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(gb,b,C*4,cudaMemcpyHostToDevice));
    dim3 blk(256); unsigned grd=(unsigned)((BT+7)/8);   // 8 warps/block
    fused_add_ln<<<grd,blk>>>(gx,gr,gw,gb,gx2,gy,BT);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(y,gy,n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(x2,gx2,n*4,cudaMemcpyDeviceToHost));
    // CPU oracle in double
    double *oy=(double*)malloc(n*8),*ox2=(double*)malloc(n*8);
    for(int t=0;t<BT;t++){
        double s=0,ss=0;
        for(int d=0;d<C;d++){ double v=(double)x[(size_t)t*C+d]+r[(size_t)t*C+d];
            ox2[(size_t)t*C+d]=v; s+=v; ss+=v*v; }
        double mu=s/C, var=ss/C-mu*mu, rstd=1.0/sqrt(var+EPS);
        for(int d=0;d<C;d++) oy[(size_t)t*C+d]=w[d]*(ox2[(size_t)t*C+d]-mu)*rstd+b[d];
    }
    check_close("fused_add_ln x2 == x+r", ox2, x2, n, 1e-4f);
    check_close("fused_add_ln y == LN(x+r)", oy, y, n, 1e-3f);
    float ms=bench([&]{ fused_add_ln<<<grd,blk>>>(gx,gr,gw,gb,gx2,gy,BT); },2,50);
    double bytes=3.0*n*4; // read x,r write x2,y ~ 3 passes worth
    printf("fused add+LN: %.3f ms -> %.1f GB/s effective\n", ms, bytes/ms/1e6);
    free(x);free(r);free(w);free(b);free(y);free(x2);free(oy);free(ox2);
    cudaFree(gx);cudaFree(gr);cudaFree(gw);cudaFree(gb);cudaFree(gy);cudaFree(gx2);
    return report();
}
