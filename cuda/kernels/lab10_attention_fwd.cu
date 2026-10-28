// lab10 — Causal self-attention forward as four composable kernels (llm.c fp32 chain)
// Source map: rough/llm.c/train_gpt2_fp32.cu matmul_forward(strided-batched)+kernel5 softmax+
//             matmul + causal mask pattern; CPU ref attention_forward in train_gpt2.c
//             (docs/audit/llm-c-03 §attention "four passes"). gpt.c attention_kernel = fused naive.
// Math per (b,h): S = Q K^T / sqrt(C); P = softmax_causal(S); O = P V.
// Causal: j>i masked to -inf BEFORE softmax (row-max trick handles it).
#include "harness.h"

#define B 4
#define H 12
#define T 64
#define C 64           // head dim = 768/12
#define SCALE (1.0f/sqrtf((float)C))

// kernel A: scores S[b,h,i,j] = dot(Q[i,:],K[j,:])*scale, only j<=i computed (others left 0)
__global__ void attn_scores(const float* __restrict__ q, const float* __restrict__ k,
                            float* __restrict__ s) {
    int ij = blockIdx.x * blockDim.x + threadIdx.x; // one thread per (b,h,i,j)
    int total = B*H*T*T;
    if (ij >= total) return;
    int j = ij % T; int i = (ij / T) % T; int h = (ij / (T*T)) % H; int b = ij / (H*T*T);
    if (j > i) { s[ij] = 0.f; return; }            // masked-out region unused
    const float* qi = q + (((size_t)b*H + h)*T + i) * C;
    const float* kj = k + (((size_t)b*H + h)*T + j) * C;
    float acc = 0.f;
    for (int c = 0; c < C; c++) acc += qi[c] * kj[c];
    s[ij] = acc * SCALE;
}

// kernel B: row-wise causal softmax over [0..i] — warp-per-row
__global__ void attn_softmax(float* __restrict__ s) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; // row in [0, B*H*T)
    if (row >= B*H*T) return;
    int i = row % T;
    float* r = s + (size_t)row * T;
    float mx = -1e30f;
    for (int j = lane; j <= i; j += 32) mx = fmaxf(mx, r[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float sum = 0.f;
    for (int j = lane; j <= i; j += 32) { float e = expf(r[j]-mx); r[j]=e; sum+=e; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    for (int j = lane; j <= i; j += 32) r[j] /= sum;
}

// kernel C: O = P V
__global__ void attn_pv(const float* __restrict__ p, const float* __restrict__ v,
                        float* __restrict__ o) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // one thread per (b,h,i,c)
    int total = B*H*T*C;
    if (idx >= total) return;
    int c = idx % C; int i = (idx / C) % T; int h = (idx / (C*T)) % H; int b = idx / (H*C*T);
    const float* pr = p + (((size_t)b*H + h)*T + i) * T;
    const float* vv = v + (((size_t)b*H + h)*T ) * C;
    float acc = 0.f;
    for (int j = 0; j <= i; j++) acc += pr[j] * vv[(size_t)j*C + c];
    o[idx] = acc;
}

int main() {
    print_gpu_info();
    size_t n = (size_t)B*H*T*C, ns = (size_t)B*H*T*T;
    float *q = (float*)malloc(n*4), *k = (float*)malloc(n*4), *v = (float*)malloc(n*4);
    float *o = (float*)malloc(n*4), *s = (float*)malloc(ns*4);
    fill_rand(q, n, 101); fill_rand(k, n, 202); fill_rand(v, n, 303);
    double *ref = (double*)malloc(n*8);
    // CPU oracle (double)
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        double sc[T]; // T=64 fine on stack for the lab
        double m = -1e300, sm = 0;
        for (int j = 0; j <= i; j++) {
            double a = 0;
            for (int c = 0; c < C; c++)
                a += q[(((size_t)b*H+h)*T+i)*C+c] * k[(((size_t)b*H+h)*T+j)*C+c];
            sc[j] = a * SCALE; m = fmax(m, sc[j]);
        }
        for (int j = 0; j <= i; j++) { sc[j] = exp(sc[j]-m); sm += sc[j]; }
        for (int c = 0; c < C; c++) {
            double a = 0;
            for (int j = 0; j <= i; j++)
                a += (sc[j]/sm) * v[(((size_t)b*H+h)*T+j)*C+c];
            ref[(((size_t)b*H+h)*T+i)*C+c] = a;
        }
    }
    float *d_q,*d_k,*d_v,*d_o,*d_s;
    CK(cudaMalloc(&d_q,n*4)); CK(cudaMalloc(&d_k,n*4)); CK(cudaMalloc(&d_v,n*4));
    CK(cudaMalloc(&d_o,n*4)); CK(cudaMalloc(&d_s,ns*4));
    CK(cudaMemcpy(d_q,q,n*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_k,k,n*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v,v,n*4,cudaMemcpyHostToDevice));
    dim3 blk(256);
    auto chain = [&]{
        attn_scores<<<(unsigned)((ns+255)/256), blk>>>(d_q,d_k,d_s);
        attn_softmax<<<(unsigned)((B*H*T+7)/8), dim3(256)>>>(d_s);
        attn_pv<<<(unsigned)((n+255)/256), blk>>>(d_s,d_v,d_o);
    };
    float ms = bench(chain);
    CK(cudaMemcpy(o,d_o,n*4,cudaMemcpyDeviceToHost));
    printf("causal attn (3-kernel chain): %.3f ms\n", ms);
    check_close("attention fwd vs oracle", ref, o, n, 1e-3f);
    return report();
}
