// lab14_dual_matmul.cu — Tier3: dual matmul (dgrad + wgrad) fused in one launch.
// Math: dX[N,M] = dY[N,K] * W[K,M]                    (data gradient)
//       dW[K,M] = sum_n dY[n,k]*X[n,m]  i.e. dY^T*X  (weight gradient)
// gpt.c idiom: C^T = A*B^T via swapped index roles (rough/gpt.c matmul_backward);
// llm.c fp32 trainer issues these as two separate cuBLAS calls
// (docs/audit/llm-c-05-train_gpt2_fp32_cu.md). Here ONE kernel reads each dY
// tile once and feeds BOTH streams. Self-verifies vs double CPU oracle.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "harness.h"

#define TILE 16
#define CHUNK_N 64

__global__ void dgrad_kernel(float* __restrict__ dX, const float* __restrict__ dY,
                             const float* __restrict__ W, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= M) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += dY[(size_t)row*K+k] * W[(size_t)k*M+col];
    dX[(size_t)row*M+col] = acc;
}

__global__ void wgrad_kernel(float* __restrict__ dW, const float* __restrict__ dY,
                             const float* __restrict__ X, int N, int K, int M) {
    // split over N via blockIdx.z chunks; atomicAdd accumulates partials.
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= K || m >= M) return;
    int n_start = blockIdx.z * CHUNK_N;
    float acc = 0.f;
    for (int n = n_start; n < (n_start + CHUNK_N < N ? n_start + CHUNK_N : N); ++n)
        acc += dY[(size_t)n*K+k] * X[(size_t)n*M+m];
    atomicAdd(&dW[(size_t)k*M+m], acc);
}

// Fused dual kernel. Block owns an N-tile (by) and an M-tile (bx).
// Stream A: computes its dX output tile by sweeping K (needs sdy, sw).
// Stream B: computes a PARTIAL dW tile (K-slice x M-tile) reduced over ITS OWN
//           N-tile rows only, atomically added (needs sdy, sx).
// Both consume the same sdy tile => dY read from HBM once per tile.
__global__ void dual_matmul_kernel(float* __restrict__ dX, float* __restrict__ dW,
                                   const float* __restrict__ dY,
                                   const float* __restrict__ W,
                                   const float* __restrict__ X,
                                   int N, int K, int M) {
    __shared__ float sdy[TILE][TILE];   // [i=n-in-tile][j=k-in-tile]
    __shared__ float sw [TILE][TILE];   // [j=k-in-tile][c=m-in-tile]
    __shared__ float sx [TILE][TILE];   // [i=n-in-tile][c=m-in-tile]
    const int tx = threadIdx.x, ty = threadIdx.y;   // block is TILE x TILE
    const int n0 = blockIdx.y * TILE;
    const int m0 = blockIdx.x * TILE;

    float dx_acc = 0.f;                 // thread (ty,tx) -> dX[n0+ty][m0+tx]

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // cooperative loads, coalesced along tx:
        for (int i = 0; i < TILE; ++i) {
            int n = n0 + i, k = k0 + tx;
            sdy[i][tx] = (n < N && k < K) ? dY[(size_t)n*K + k] : 0.f;
            sw [tx][i] = (k < K && m0+i < M) ? W[(size_t)k*M + (m0+i)] : 0.f;
            sx [i][tx] = (n < N && m0+tx < M) ? X[(size_t)n*M + (m0+tx)] : 0.f;
        }
        __syncthreads();
        // stream A: dX tile accumulation over this K-slice
        if (n0 + ty < N && m0 + tx < M)
            #pragma unroll
            for (int kk = 0; kk < TILE; ++kk) dx_acc += sdy[ty][kk] * sw[kk][tx];
        // stream B: partial dW for (k=k0+ty, m=m0+tx) reduced over N-tile rows
        if (k0 + ty < K && m0 + tx < M) {
            float dw_acc = 0.f;
            #pragma unroll
            for (int i = 0; i < TILE; ++i) dw_acc += sdy[i][ty] * sx[i][tx];
            // flush this slice's partial now (one atomic per element per k-slice)
            atomicAdd(&dW[(size_t)(k0+ty)*M + (m0+tx)], dw_acc);
        }
        __syncthreads();
    }
    if (n0 + ty < N && m0 + tx < M) dX[(size_t)(n0+ty)*M + (m0+tx)] = dx_acc;
}

// ------------------------- CPU oracles (double) -------------------------
static void cpu_dgrad(double* o, const float* dY, const float* W, int N,int K,int M){
    for (int r=0;r<N;r++) for(int c=0;c<M;c++){
        double a=0; for(int k=0;k<K;k++) a+=(double)dY[(size_t)r*K+k]*W[(size_t)k*M+c];
        o[(size_t)r*M+c]=a; } }
static void cpu_wgrad(double* o, const float* dY, const float* X, int N,int K,int M){
    for (int k=0;k<K;k++) for(int c=0;c<M;c++){
        double a=0; for(int n=0;n<N;n++) a+=(double)dY[(size_t)n*K+k]*X[(size_t)n*M+c];
        o[(size_t)k*M+c]=a; } }

int main(){
    const int N=512, K=256, M=384;   // transformer-ish shapes
    size_t szDY=(size_t)N*K, szW=(size_t)K*M, szX=(size_t)N*M, szDX=(size_t)N*M, szDW=(size_t)K*M;
    float *dY=(float*)malloc(szDY*4),*W=(float*)malloc(szW*4),*X=(float*)malloc(szX*4);
    fill_rand(dY,szDY,11); fill_rand(W,szW,12); fill_rand(X,szX,13);
    double *odx=(double*)malloc(szDX*8),*odw=(double*)malloc(szDW*8);
    cpu_dgrad(odx,dY,W,N,K,M); cpu_wgrad(odw,dY,X,N,K,M);

    float *gdY,*gW,*gX,*gdX,*gdW;
    CK(cudaMalloc(&gdY,szDY*4)); CK(cudaMalloc(&gW,szW*4)); CK(cudaMalloc(&gX,szX*4));
    CK(cudaMalloc(&gdX,szDX*4)); CK(cudaMalloc(&gdW,szDW*4));
    CK(cudaMemcpy(gdY,dY,szDY*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gW ,W ,szW *4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gX ,X ,szX *4,cudaMemcpyHostToDevice));

    dim3 blk(TILE,TILE);
    dim3 grd_dx((M+TILE-1)/TILE,(N+TILE-1)/TILE);
    dim3 grd_dw((K+TILE-1)/TILE,(M+TILE-1)/TILE,(N+CHUNK_N-1)/CHUNK_N);

    // baseline: two kernels (like llm.c's two cuBLAS calls)
    CK(cudaMemset(gdX,0,szDX*4)); CK(cudaMemset(gdW,0,szDW*4));
    dgrad_kernel<<<grd_dx,blk>>>(gdX,gdY,gW,N,K,M);
    wgrad_kernel<<<grd_dw,blk>>>(gdW,gdY,gX,N,K,M);
    float t_a = bench([&]{
        dgrad_kernel<<<grd_dx,blk>>>(gdX,gdY,gW,N,K,M);
        wgrad_kernel<<<grd_dw,blk>>>(gdW,gdY,gX,N,K,M);
    }, 2, 10);
    float *dX1=(float*)malloc(szDX*4), *dW1=(float*)malloc(szDW*4);
    CK(cudaMemcpy(dX1,gdX,szDX*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dW1,gdW,szDW*4,cudaMemcpyDeviceToHost));
    check_close("two-kernel dX vs CPU", odx, dX1, szDX, 1e-3f);
    check_close("two-kernel dW vs CPU", odw, dW1, szDW, 1e-3f);

    // fused dual kernel
    CK(cudaMemset(gdX,0,szDX*4)); CK(cudaMemset(gdW,0,szDW*4));
    dual_matmul_kernel<<<grd_dx,blk>>>(gdX,gdW,gdY,gW,gX,N,K,M);
    float t_b = bench([&]{
        dual_matmul_kernel<<<grd_dx,blk>>>(gdX,gdW,gdY,gW,gX,N,K,M);
    }, 2, 10);
    float *dX2=(float*)malloc(szDX*4), *dW2=(float*)malloc(szDW*4);
    CK(cudaMemcpy(dX2,gdX,szDX*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dW2,gdW,szDW*4,cudaMemcpyDeviceToHost));
    check_close("fused dX vs CPU", odx, dX2, szDX, 1e-3f);
    check_close("fused dW vs CPU", odw, dW2, szDW, 1e-3f);

    printf("two-kernel %.3f ms | fused %.3f ms | speedup %.2fx\n", t_a, t_b, t_a/t_b);
    printf("NOTE: dW uses atomicAdd -> run-to-run bit nondeterminism expected;\n");
    printf("      tolerance check above proves numerical correctness only.\n");
    free(dY);free(W);free(X);free(odx);free(odw);free(dX1);free(dW1);free(dX2);free(dW2);
    cudaFree(gdY);cudaFree(gW);cudaFree(gX);cudaFree(gdX);cudaFree(gdW);
    return report();
}
