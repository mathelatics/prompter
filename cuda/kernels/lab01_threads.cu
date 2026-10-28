// lab01_threads.cu — Lab 01: thread mapping, global indices, bounds checks.
// Build: nvcc -O3 lab01_threads.cu -o bin/lab01_threads   (or `make check`)
// Run:   ./bin/lab01_threads  -> self-verifies vs CPU oracle; exit 0 on PASS.
//
// First principles: a GPU kernel is ONE program run by millions of threads in
// parallel with NO knowledge of each other. The entire art of data-parallel
// CUDA starts here: (1) compute your global index, (2) guard the bounds,
// (3) optionally loop with grid-stride so a fixed grid handles any input size.
#include "harness.h"

// Idiom #1: one element per thread. ceil-division grid + bounds guard.
__global__ void add_one_naive(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // global linear index
    if (i < n) a[i] += 1.0f;                        // guard: last block partial
}

// Idiom #2: grid-stride loop. Works for ANY n with a FIXED grid — the pattern
// used by cuBLAS/PyTorch and llm.c's train_gpt2_fp32 elementwise kernels.
__global__ void add_one_stride(float *a, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        a[i] += 1.0f;
}

int main(void) {
    print_gpu_info();
    const int N = 1000003;            // deliberately NOT divisible by 256
    const int block = 256;
    const int grid = (N + block - 1) / block;   // ceiling division idiom

    float *orig = (float*)malloc(N * sizeof(float));
    float *got  = (float*)malloc(N * sizeof(float));
    double *ref = (double*)malloc(N * sizeof(double));
    float *da;
    CK(cudaMalloc(&da, N * sizeof(float)));

    for (int i = 0; i < N; i++) { orig[i] = rand_f(-1.f, 1.f); ref[i] = (double)orig[i] + 1.0; }

    // --- Test 1: naive launch, verify every element moved exactly +1.0 ---
    CK(cudaMemcpy(da, orig, N * sizeof(float), cudaMemcpyHostToDevice));
    add_one_naive<<<grid, block>>>(da, N);        // async launch
    CK(cudaGetLastError());                        // catches BAD launch config
    CK(cudaDeviceSynchronize());                   // blocks until done
    CK(cudaMemcpy(got, da, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK("naive add-one (unaligned N)", max_abs_diff(got, ref, N) == 0.0,
          "max abs err %.3e over %d elems", max_abs_diff(got, ref, N), N);

    // --- Test 2: grid-stride with a deliberately undersized fixed grid ---
    CK(cudaMemcpy(da, orig, N * sizeof(float), cudaMemcpyHostToDevice));
    add_one_stride<<<1024, block>>>(da, N);        // 1024*256 < N! loop covers rest
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got, da, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK("grid-stride (fixed small grid)", max_abs_diff(got, ref, N) == 0.0,
          "max abs err %.3e", max_abs_diff(got, ref, N));

    // --- Timing both variants (same math, different launch strategy) ---
    float t_naive  = bench([&]{ add_one_naive<<<grid,  block>>>(da, N); });
    float t_stride = bench([&]{ add_one_stride<<<1024, block>>>(da, N); });
    printf("naive(%d blocks)=%.4f ms | stride(1024 blocks)=%.4f ms\n",
           grid, t_naive, t_stride);
    printf("# lesson: correctness identical; stride decouples grid size from N.\n");

    cudaFree(da); free(orig); free(got); free(ref);
    return harness_summary();
}
