// lab16_rope.cu — Tier3: Rotary Position Embedding (RoPE) fwd + bwd.
// Math: for pair (x_{2i}, x_{2i+1}) with freq theta_i = base^(-2i/d):
//   q'_{2i}   = x_{2i} cos(pos*theta_i) - x_{2i+1} sin(pos*theta_i)
//   q'_{2i+1} = x_{2i} sin(pos*theta_i) + x_{2i+1} cos(pos*theta_i)
// Backward is rotation by -angle (exact transpose). No learnable params.
// llama3 config (rough/llm.c train_llama3.py: base=500000, head_dim=128,
// 8 kv heads GQA) — see docs/audit/llm-c-10-dev-and-tests.md.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define HD 128
__constant__ float c_theta[HD/2];

__global__ void rope_fwd(const float* __restrict__ x, float* __restrict__ y,
                         int seq, float pos_scale){
    int tok = blockIdx.x;                       // one block per token
    int d2  = threadIdx.x;                      // pair index 0..HD/2-1
    if (tok >= seq || d2 >= HD/2) return;
    float a = (float)tok * pos_scale * c_theta[d2];
    float c = cosf(a), s = sinf(a);
    const float* xr = x + (size_t)tok*HD;
    float* yr = y + (size_t)tok*HD;
    float x0 = xr[2*d2], x1 = xr[2*d2+1];
    yr[2*d2]   = x0*c - x1*s;
    yr[2*d2+1] = x0*s + x1*c;
}
__global__ void rope_bwd(const float* __restrict__ dy, float* __restrict__ dx,
                         int seq, float pos_scale){
    int tok = blockIdx.x; int d2 = threadIdx.x;
    if (tok >= seq || d2 >= HD/2) return;
    float a = (float)tok * pos_scale * c_theta[d2];
    float c = cosf(a), s = -sinf(a);            // rotate by -angle
    const float* dr = dy + (size_t)tok*HD;
    float* wr = dx + (size_t)tok*HD;
    float y0 = dr[2*d2], y1 = dr[2*d2+1];
    wr[2*d2]   = y0*c - y1*s;
    wr[2*d2+1] = y0*s + y1*c;
}

int main(){
    const int seq = 2048; const float base = 500000.f;
    float theta[HD/2]; for(int i=0;i<HD/2;i++) theta[i]=1.f/powf(base, 2.f*i/HD);
    CK(cudaMemcpyToSymbol(c_theta, theta, sizeof(theta)));
    size_t n=(size_t)seq*HD;
    float *x=(float*)malloc(n*4); fill_rand(x,n,31);
    float *y=(float*)malloc(n*4), *dy=(float*)malloc(n*4), *dx=(float*)malloc(n*4);
    fill_rand(dy,n,32);
    float *gx,*gy,*gdy,*gdx;
    CK(cudaMalloc(&gx,n*4));CK(cudaMalloc(&gy,n*4));CK(cudaMalloc(&gdy,n*4));CK(cudaMalloc(&gdx,n*4));
    CK(cudaMemcpy(gx,x,n*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(gdy,dy,n*4,cudaMemcpyHostToDevice));
    rope_fwd<<<seq, dim3(HD/2)>>>(gx,gy,seq,1.f);
    rope_bwd<<<seq, dim3(HD/2)>>>(gdy,gdx,seq,1.f);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    CK(cudaMemcpy(y,gy,n*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(dx,gdx,n*4,cudaMemcpyDeviceToHost));
    // CPU oracle
    double *ry=(double*)malloc(n*8), *rdx=(double*)malloc(n*8);
    for(int t=0;t<seq;t++) for(int i=0;i<HD/2;i++){
        double a=(double)t*theta[i], c=cos(a), s=sin(a);
        double x0=x[(size_t)t*HD+2*i], x1=x[(size_t)t*HD+2*i+1];
        ry[(size_t)t*HD+2*i]=x0*c-x1*s; ry[(size_t)t*HD+2*i+1]=x0*s+x1*c;
        double y0=dy[(size_t)t*HD+2*i], y1=dy[(size_t)t*HD+2*i+1];
        rdx[(size_t)t*HD+2*i]=y0*c+y1*s; rdx[(size_t)t*HD+2*i+1]=-y0*s+y1*c;
    }
    check_close("rope fwd vs oracle", ry, y, n, 1e-3f);
    check_close("rope bwd vs oracle", rdx, dx, n, 1e-3f);
    // property: fwd then bwd-of-same-angle = identity on x
    float *z; CK(cudaMalloc(&z,n*4));
    rope_bwd<<<seq,dim3(HD/2)>>>(gy,z,seq,1.f);
    CK(cudaDeviceSynchronize());
    float *hz=(float*)malloc(n*4); CK(cudaMemcpy(hz,z,n*4,cudaMemcpyDeviceToHost));
    double *rx=(double*)malloc(n*8); for(size_t i=0;i<n;i++) rx[i]=x[i];
    check_close("rope bwd(fwd(x)) == x (rotation inverse)", rx, hz, n, 1e-3f);
    float ms=bench([&]{ rope_fwd<<<seq,dim3(HD/2)>>>(gx,gy,seq,1.f); },2,50);
    printf("rope fwd %d tokens: %.4f ms (%.1f ns/token)\n", seq, ms, ms*1e6/seq);
    free(x);free(y);free(dy);free(dx);free(hz);free(ry);free(rdx);free(rx);
    cudaFree(gx);cudaFree(gy);cudaFree(gdy);cudaFree(gdx);cudaFree(z);
    return report();
}
