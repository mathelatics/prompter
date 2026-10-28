// lab05_matmul.cu — Lab 05: tiled SGEMM, v0 -> v3 (the llm.c journey).
// C = A @ B with A(MxK) B(KxN) C(MxN), row-major. First principles:
//   - Each output element needs K MACs; naive global loads => 2K loads/element
//     from DRAM/L2. Arithmetic intensity of the whole GEMM is O(N) so it's
//     COMPUTE-bound once you stop re-fetching tiles from global memory.
//   - Tiling in shared memory amortizes each global load TILE times.
//   - Register blocking (thread computes a small tile) turns memory-bound
//     inner loops into FMA throughput problems — which is what we want.
// Source map: gpt.c matmul_kernel (naive), llm.c train_gpt2_fp32_cu
// matmul_forward_kernel4 (tiled, docs/audit/llm-c-05).
#include "harness.h"

#define TS 16   // tile side

__global__ void matmul_v0(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void matmul_v1_tiled(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    __shared__ float As[TS][TS], Bs[TS][TS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
    float acc = 0.f;
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int akc = t * TS + tx;              // A column to load
        int bkr = t * TS + ty;              // B row to load
        As[ty][tx] = (row < M && akc < K) ? A[row * K + akc] : 0.f;
        Bs[ty][tx] = (bkr < K && col < N) ? B[bkr * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TS; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// v2: thread computes 2x2 outputs => 4x fewer shared->reg loads per FMA,
// better FMA:load ratio. This is the smallest register-blocking step.
__global__ void matmul_v2_reg2x2(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    __shared__ float As[2 * TS][TS], Bs[TS][2 * TS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * (TS * 2) + ty * 2;   // 2 rows per thread
    int col = blockIdx.x * (TS * 2) + tx * 2;   // 2 cols per thread
    float c00 = 0, c01 = 0, c10 = 0, c11 = 0;
    for (int t = 0; t < (K + TS - 1) / TS; t++) {
        int akc = t * TS + tx, bkr = t * TS + ty;
        As[ty * 2][tx]     = (row < M && akc < K)     ? A[row * K + akc] : 0.f;
        As[ty * 2 + 1][tx] = (row + 1 < M && akc < K) ? A[(row + 1) * K + akc] : 0.f;
        Bs[ty][tx * 2]     = (bkr < K && col < N)     ? B[bkr * N + col] : 0.f;
        Bs[ty][tx * 2 + 1] = (bkr < K && col + 1 < N) ? B[bkr * N + col + 1] : 0.f;
        __syncthreads();
        for (int k = 0; k < TS; k++) {
            float a0 = As[ty * 2][k], a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2], b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0; c01 += a0 * b1;
            c10 += a1 * b0; c11 += a1 * b1;
        }
        __syncthreads();
    }
    if (row < M && col < N)     C[row * N + col] = c00;
    if (row < M && col + 1 < N) C[row * N + col + 1] = c01;
    if (row + 1 < M && col < N) C[(row + 1) * N + col] = c10;
    if (row + 1 < M && col + 1 < N) C[(row + 1) * N + col + 1] = c11;
}

int main(void) {
    print_gpu_info();
    const int M = 1024, N = 1024, K = 1024;
    size_t ab = (size_t)M * K * sizeof(float), bb = (size_t)K * N * sizeof(float), cb = (size_t)M * N * sizeof(float);
    float *hA = (float*)malloc(ab), *hB = (float*)malloc(bb);
    float *hC = (float*)malloc(cb);
    double *ref = (double*)malloc(cb);
    float *dA, *dB, *dC;
    CK(cudaMalloc(&dA, ab)); CK(cudaMalloc(&dB, bb)); CK(cudaMalloc(&dC, cb));
    for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = rand_f(-0.5f, 0.5f);
    for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = rand_f(-0.5f, 0.5f);
    CK(cudaMemcpy(dA, hA, ab, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bb, cudaMemcpyHostToDevice));

    // CPU oracle in double (single-threaded; 1G MACs ~ seconds)
    double t0 = cpu_time_begin();
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)hA[r * K + k] * hB[k * N + c];
            ref[r * N + c] = s;
        }
    double cpu_ms = cpu_time_ms_since(t0);

    auto check_c = [&](const char *name) {
        CK(cudaMemcpy(hC, dC, cb, cudaMemcpyDeviceToHost));
        float e = rel_l2(hC, ref, (size_t)M * N);
        CHECK(name, e < 1e-3f, "rel-L2 %.3e", e);
    };

    dim3 b16(TS, TS);
    dim3 g0((N + TS - 1) / TS, (M + TS - 1) / TS);
    matmul_v0<<<g0, b16>>>(dA, dB, dC, M, N, K);
    CK(cudaGetLastError()); check_c("v0 naive");
    float t0_ = bench([&]{ matmul_v0<<<g0, b16>>>(dA, dB, dC, M, N, K); }, 2, 10);

    dim3 g1((N + TS - 1) / TS, (M + TS - 1) / TS);
    matmul_v1_tiled<<<g1, b16>>>(dA, dB, dC, M, N, K);
    CK(cudaGetLastError()); check_c("v1 tiled");
    float t1_ = bench([&]{ matmul_v1_tiled<<<g1, b16>>>(dA, dB, dC, M, N, K); }, 2, 10);

    dim3 g2((N + TS * 2 - 1) / (TS * 2), (M + TS * 2 - 1) / (TS * 2));
    matmul_v2_reg2x2<<<g2, b16>>>(dA, dB, dC, M, N, K);
    CK(cudaGetLastError()); check_c("v2 reg 2x2");
    float t2_ = bench([&]{ matmul_v2_reg2x2<<<g2, b16>>>(dA, dB, dC, M, N, K); }, 2, 10);

    double gflop = 2.0 * M * N * K / 1e9;
    printf("cpu double 1T : %8.1f ms | %7.1f GFLOP/s\n", cpu_ms, gflop / (cpu_ms * 1e-3));
    printf("v0 naive      : %8.4f ms | %7.1f GFLOP/s\n", t0_, gflop / (t0_ * 1e-3));
    printf("v1 tiled      : %8.4f ms | %7.1f GFLOP/s\n", t1_, gflop / (t1_ * 1e-3));
    printf("v2 reg 2x2    : %8.4f ms | %7.1f GFLOP/s\n", t2_, gflop / (t2_ * 1e-3));
    printf("# lesson: tiling fixes bandwidth; register blocking feeds FMA units.\n");
    printf("#         next real step is tensor cores (mma.sync) or cuBLAS — see lab16.\n");

    free(hA); free(hB); free(hC); free(ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return harness_summary();
}
