// lab11 — Attention backward: dV = P^T dO, dP = dO V^T, causal-mask dS, dQ/dK via matmuls
// Source map: rough/llm.c/train_gpt2.c attention_backward (the "four passes" doc in
//             docs/audit/llm-c-03 §attention). GPU chain mirrors llm.c fp32 cuBLAS order.
// Math (per b,h):  dV = P^T · dO
//                  dP = dO · V^T
//                  dS = P ∘ (dP - rowsum(P ∘ dP)) / sqrt(C)   [softmax bwd, causal rows only]
//                  dQ = dS · K ;  dK = dS^T · Q
#include "harness.h"

#define B 4
#define H 12
#define T 64
#define C 64
#define SCALE (1.0f/sqrtf((float)C))

// We reuse FORWARD outputs p (probs, causal-lower-triangular) and recompute s not needed.
// kernel D1: dV[j,c] = sum_i>=j P[i,j] * dO[i,c]      (one thread per (b,h,j,c))
__global__ void attn_bwd_dv(const float* __restrict__ p, const float* __restrict__ do_,
                            float* __restrict__ dv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B*H*T*C;
    if (idx >= total) return;
    int c = idx % C; int j = (idx / C) % T; int h = (idx / (C*T)) % H; int b = idx / (H*C*T);
    const float* pb = p + ((size_t)b*H + h)*T*T;
    const float* dob = do_ + ((size_t)b*H + h)*T*C;
    float acc = 0.f;
    for (int i = j; i < T; i++) acc += pb[(size_t)i*T + j] * dob[(size_t)i*C + c];
    dv[idx] = acc;
}

// kernel D2a: dP[b,h,i,j] = dO[i,:]·V[j,:] (only causal entries; upper triangle left 0)
__global__ void attn_dp(const float* __restrict__ do_, const float* __restrict__ v,
                        float* __restrict__ dp_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per (b,h,i,j)
    int total = B*H*T*T;
    if (idx >= total) return;
    int j = idx % T; int i = (idx / T) % T; int h = (idx / (T*T)) % H; int b = idx / (H*T*T);
    if (j > i) { dp_out[idx] = 0.f; return; }
    const float* doi = do_ + (((size_t)b*H + h)*T + i) * C;
    const float* vv  = v  + (((size_t)b*H + h)*T + j) * C;
    float acc = 0.f;
    for (int c = 0; c < C; c++) acc += doi[c] * vv[c];
    dp_out[idx] = acc;
}

// kernel D2b: rowdot[i] = sum_j P[i,j]*dP[i,j] — warp-per-row over the FULL T-wide rows
__global__ void attn_rowdot(const float* __restrict__ p, const float* __restrict__ dp_in,
                            float* __restrict__ rd) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; // (b,h,i)
    if (row >= B*H*T) return;
    const float* pr = p + (size_t)row * T;
    const float* dr = dp_in + (size_t)row * T;
    float a = 0.f;
    for (int j = lane; j < T; j += 32) a += pr[j] * dr[j]; // upper triangle is 0 in both
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
    if (lane == 0) rd[row] = a;
}

// kernel D3: dS = P ∘ (dP - rowdot) * SCALE (elementwise)
__global__ void attn_ds(const float* __restrict__ p, const float* __restrict__ dp_in,
                        const float* __restrict__ rd, float* __restrict__ ds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B*H*T*T;
    if (idx >= total) return;
    int i = (idx / T) % T; int h = (idx / (T*T)) % H; int b = idx / (H*T*T);
    int row = (b*H + h)*T + i;
    ds[idx] = p[idx] * (dp_in[idx] - rd[row]) * SCALE;
}

// kernel D4: dQ = dS·K, dK = dS^T·Q (two loops in one kernel is fine at this size)
__global__ void attn_dq_dk(const float* __restrict__ ds, const float* __restrict__ q,
                           const float* __restrict__ k, float* __restrict__ dq,
                           float* __restrict__ dk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B*H*T*C;
    if (idx >= total) return;
    int c = idx % C; int i = (idx / C) % T; int h = (idx / (C*T)) % H; int b = idx / (H*C*T);
    const float* dsb = ds + ((size_t)b*H + h)*T*T;
    const float* qb  = q  + ((size_t)b*H + h)*T*C;
    const float* kb  = k  + ((size_t)b*H + h)*T*C;
    float a1 = 0.f, a2 = 0.f;
    for (int j = 0; j <= i; j++) a1 += dsb[(size_t)i*T + j] * kb[(size_t)j*C + c];
    for (int r = i; r < T; r++)  a2 += dsb[(size_t)r*T + i] * qb[(size_t)r*C + c];
    dq[idx] = a1;
    dk[idx] = a2;
}

int main() {
    print_gpu_info();
    size_t n = (size_t)B*H*T*C, ns = (size_t)B*H*T*T;
    float *q=(float*)malloc(n*4),*k=(float*)malloc(n*4),*v=(float*)malloc(n*4);
    float *p=(float*)malloc(ns*4),*do_=(float*)malloc(n*4);
    float *dq=(float*)malloc(n*4),*dk=(float*)malloc(n*4),*dv=(float*)malloc(n*4);
    fill_rand(q,n,11); fill_rand(k,n,22); fill_rand(v,n,33); fill_rand(do_,n,44);
    // build forward probs p causally (CPU double then cast) so backward oracle is consistent
    double *pd=(double*)malloc(ns*8);
    for (int b=0;b<B;b++)for(int h=0;h<H;h++)for(int i=0;i<T;i++){
        double sc[T],m=-1e300,s=0;
        for(int j=0;j<=i;j++){double a=0;for(int c=0;c<C;c++)a+=q[(((size_t)b*H+h)*T+i)*C+c]*k[(((size_t)b*H+h)*T+j)*C+c];sc[j]=a*SCALE;m=fmax(m,sc[j]);}
        for(int j=0;j<=i;j++){sc[j]=exp(sc[j]-m);s+=sc[j];}
        for(int j=0;j<T;j++)pd[((size_t)b*H+h)*T*T+(size_t)i*T+j]=(j<=i)?sc[j]/s:0.0;
    }
    for(size_t i=0;i<ns;i++)p[i]=(float)pd[i];

    // CPU double oracle for all grads using pd
    double *rdq=(double*)calloc(n,8),*rdk=(double*)calloc(n,8),*rdv=(double*)calloc(n,8);
    double *dsd=(double*)malloc(ns*8);
    for (int b=0;b<B;b++)for(int h=0;h<H;h++)for(int i=0;i<T;i++){
        double dp[T],rd=0;
        for(int j=0;j<=i;j++){double a=0;for(int c=0;c<C;c++)a+=(double)do_[(((size_t)b*H+h)*T+i)*C+c]*v[(((size_t)b*H+h)*T+j)*C+c];dp[j]=a;}
        for(int j=0;j<=i;j++) rd+=pd[((size_t)b*H+h)*T*T+(size_t)i*T+j]*dp[j];
        for(int j=0;j<=i;j++) dsd[((size_t)b*H+h)*T*T+(size_t)i*T+j]=pd[((size_t)b*H+h)*T*T+(size_t)i*T+j]*(dp[j]-rd)*SCALE;
        for(int j=i+1;j<T;j++) dsd[((size_t)b*H+h)*T*T+(size_t)i*T+j]=0.0;
    }
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int i=0;i<T;i++)for(int c=0;c<C;c++){
        double a1=0,a2=0;
        for(int j=0;j<=i;j++){a1+=dsd[((size_t)b*H+h)*T*T+(size_t)i*T+j]*k[(((size_t)b*H+h)*T+j)*C+c];}
        for(int r=i;r<T;r++){a2+=dsd[((size_t)b*H+h)*T*T+(size_t)r*T+i]*q[(((size_t)b*H+h)*T+r)*C+c];}
        rdq[(((size_t)b*H+h)*T+i)*C+c]=a1; rdk[(((size_t)b*H+h)*T+i)*C+c]=a2;
    }
    for(int b=0;b<B;b++)for(int h=0;h<H;h++)for(int j=0;j<T;j++)for(int c=0;c<C;c++){
        double a=0;for(int i=j;i<T;i++)a+=pd[((size_t)b*H+h)*T*T+(size_t)i*T+j]*do_[(((size_t)b*H+h)*T+i)*C+c];
        rdv[(((size_t)b*H+h)*T+j)*C+c]=a;
    }

    float *d_q,*d_k,*d_v,*d_p,*d_do,*d_rd,*d_ds,*d_dp,*d_dq,*d_dk,*d_dv;
    CK(cudaMalloc(&d_q,n*4));CK(cudaMalloc(&d_k,n*4));CK(cudaMalloc(&d_v,n*4));
    CK(cudaMalloc(&d_p,ns*4));CK(cudaMalloc(&d_do,n*4));CK(cudaMalloc(&d_rd,(size_t)B*H*T*4));
    CK(cudaMalloc(&d_ds,ns*4));CK(cudaMalloc(&d_dp,ns*4));CK(cudaMalloc(&d_dq,n*4));CK(cudaMalloc(&d_dk,n*4));CK(cudaMalloc(&d_dv,n*4));
    CK(cudaMemcpy(d_q,q,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_k,k,n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v,v,n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(d_p,p,ns*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_do,do_,n*4,cudaMemcpyHostToDevice));
    dim3 blk(256);
    auto chain=[&]{
        attn_dp<<<(unsigned)((ns+255)/256),blk>>>(d_do,d_v,d_dp);
        attn_rowdot<<<(unsigned)((B*H*T+7)/8),dim3(256)>>>(d_p,d_dp,d_rd);
        attn_ds<<<(unsigned)((ns+255)/256),blk>>>(d_p,d_dp,d_rd,d_ds);
        attn_dq_dk<<<(unsigned)((n+255)/256),blk>>>(d_ds,d_q,d_k,d_dq,d_dk);
        attn_bwd_dv<<<(unsigned)((n+255)/256),blk>>>(d_p,d_do,d_dv);
    };
    float ms=bench(chain);
    CK(cudaMemcpy(dq,d_dq,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dk,d_dk,n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dv,d_dv,n*4,cudaMemcpyDeviceToHost));
    printf("attention bwd (4-kernel chain): %.3f ms\n", ms);
    check_close("attn bwd dQ", rdq, dq, n, 1e-3f);
    check_close("attn bwd dK", rdk, dk, n, 1e-3f);
    check_close("attn bwd dV", rdv, dv, n, 1e-3f);
    return report();
}
