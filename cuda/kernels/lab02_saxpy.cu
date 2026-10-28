// lab02_saxpy.cu — Lab 02: memory-bound kernels & effective bandwidth.
// y = a*x + y is THE canonical memory-bandwidth benchmark (STREAM's cousin).
// First principles: this kernel does 2 flops and moves 3 floats (12 bytes)
// per element -> arithmetic intensity 2/12 = 0.167 flop/byte. A T4 can do
// ~8 TFLOP fp32 but only ~300 GB/s DRAM, so ANY speedup here comes from
// better memory access, never from math. We measure GB/s, not GFLOP/s.
#include "harness.h"

__global__ void saxpy_naive(float a, const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// float4 vectorization: one thread handles 4 contiguous floats => 16-byte
// transactions, fewer address computations, wider memory pipes. This is the
// exact trick llm.c uses in its encoder kernel (docs/audit/llm-c-05 §float4).
__global__ void saxpy_f4(float a, const float4 *x, float4 *y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 xv = x[i], yv = y[i];
        xv.x = a * xv.x + yv.x;  xv.y = a * xv.y + yv.y;
        xv.z = a * xv.z + yv.z;  xv.w = a * xv.w + yv.w;
        y[i] = xv;
    }
}

int main(void) {
    print_gpu_info();
    const int N = 1 << 24;            // 16M floats = 64 MB per array (>> L2)
    const int N4 = N / 4;
    float a = 2.5f;
    float *hx = (float*)malloc(N * sizeof(float));
    float *hy0 = (float*)malloc(N * sizeof(float));
    float *hy1 = (float*)malloc(N * sizeof(float));
    double *ref = (double*)malloc(N * sizeof(double));
    float *dx, *dy;
    CK(cudaMalloc(&dx, N * sizeof(float)));
    CK(cudaMalloc(&dy, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        hx[i] = rand_f(-1.f, 1.f); hy0[i] = rand_f(-1.f, 1.f);
        ref[i] = (double)a * hx[i] + hy0[i];
    }

    int block = 256;
    // --- naive ---
    CK(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hy0, N * sizeof(float), cudaMemcpyHostToDevice));
    saxpy_naive<<<(N + block - 1) / block, block>>>(a, dx, dy, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hy1, dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK("saxpy naive", max_abs_diff(hy1, ref, N) < 1e-4f,
          "max abs err %.3e", max_abs_diff(hy1, ref, N));

    // --- float4 ---
    CK(cudaMemcpy(dy, hy0, N * sizeof(float), cudaMemcpyHostToDevice));
    saxpy_f4<<<(N4 + block - 1) / block, block>>>(a, (const float4*)dx, (float4*)dy, N4);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hy1, dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK("saxpy float4", max_abs_diff(hy1, ref, N) < 1e-4f,
          "max abs err %.3e", max_abs_diff(hy1, ref, N));

    // --- bandwidth measurement: 3 arrays moved per element ---
    float t1 = bench([&]{ saxpy_naive<<<(N+block-1)/block, block>>>(a, dx, dy, N); }, 3, 20);
    float t2 = bench([&]{ saxpy_f4<<<(N4+block-1)/block, block>>>(a, (const float4*)dx, (float4*)dy, N4); }, 3, 20);
    double gb = 3.0 * N * sizeof(float) / 1e9;
    printf("naive  : %8.4f ms | %7.1f GB/s\n", t1, gb / (t1 * 1e-3));
    printf("float4 : %8.4f ms | %7.1f GB/s\n", t2, gb / (t2 * 1e-3));
    printf("# lesson: when GB/s approaches device peak, you are bandwidth-bound;\n");
    printf("#         stop optimizing compute and start fusing kernels instead.\n");

    cudaFree(dx); cudaFree(dy); free(hx); free(hy0); free(hy1); free(ref);
    return harness_summary();
}
