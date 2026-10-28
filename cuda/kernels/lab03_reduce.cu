// lab03_reduce.cu — Lab 03: parallel reduction, the 6-stage evolution.
// Sum of N floats is THE teaching kernel: every optimization concept
// (divergence, bank conflicts, occupancy, ILP, atomics, determinism) shows up.
// Stages:
//   v1 tree-in-shared with `if` guard (divergent warps in later levels)
//   v2 first warp unrolled after __syncthreads (removes half the syncs)
//   v3 blockDim-strided pre-add (fewer blocks, each thread does more work = ILP)
//   v4 warp shuffles + shared only across warps (no shared for intra-warp)
//   v5 atomicAdd final merge (one launch, but NON-deterministic order!)
// Determinism note (from llm.c docs/audit/llm-c-08): training grads need
// reproducible sums; atomic float adds reorder -> different low bits per run.
#include "harness.h"

#define TPB 256

__global__ void reduce_v1(const float *in, float *out, int n) {
    __shared__ float s[TPB];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];   // divergent tail warps idle
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduce_v2(const float *in, float *out, int n) {
    __shared__ float s[TPB];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
    s[tid] = (i < n) ? in[i] : 0.f;
    __syncthreads();
    // Interleaved addressing FIRST so threads add adjacent elements (sets up v4).
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {                     // ONE warp handles last 5 levels
        volatile float *vs = s;         // force real memory ops on old archs
        vs[tid] += vs[tid + 32]; vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];  vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];  vs[tid] += vs[tid + 1];
        if (tid == 0) out[blockIdx.x] = s[0];
    }
}

__global__ void reduce_v3(const float *in, float *out, int n) {
    __shared__ float s[TPB / 32];       // one slot PER WARP, not per thread
    int tid = threadIdx.x;
    float sum = 0.f;
    // Grid-stride over data: each thread accumulates many elements (ILP!).
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    s[tid / 32] = sum;                  // warp 0 writes slot 0, warp 1 slot 1...
    __syncthreads();
    if (tid < TPB / 32 && tid > 0) s[0] += s[tid];  // merge warp slots
    if (tid == 0) out[blockIdx.x] = s[0];
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // Butterfly via shuffle: log2(32)=5 steps, no shared memory, no syncthreads.
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;                            // valid in lane 0 of each warp
}

__global__ void reduce_v4(const float *in, float *out, int n) {
    __shared__ float s[TPB / 32];
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    sum = warp_reduce_sum(sum);          // intra-warp free via shuffles
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = sum;  // lane0 per warp
    __syncthreads();
    if (threadIdx.x < 32) {              // warp 0 merges the 8 warp-sums
        float t = (threadIdx.x < TPB / 32) ? s[threadIdx.x] : 0.f;
        t = warp_reduce_sum(t);
        if (threadIdx.x == 0) out[blockIdx.x] = t;
    }
}

__global__ void reduce_v5_atomic(const float *in, float *out, int n) {
    // Same block reduction as v4, then a single atomicAdd into one scalar.
    __shared__ float s[TPB / 32];
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        sum += in[i];
    sum = warp_reduce_sum(sum);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < TPB / 32) ? s[threadIdx.x] : 0.f;
        t = warp_reduce_sum(t);
        if (threadIdx.x == 0) atomicAdd(out, t);   // ORDER NOT GUARANTEED
    }
}

int main(void) {
    print_gpu_info();
    const int N = 1 << 24;
    float *h = (float*)malloc(N * sizeof(float));
    double ref_seq = 0.0;
    for (int i = 0; i < N; i++) { h[i] = rand_f(-1.f, 1.f); ref_seq += h[i]; }
    float *d, *dp;
    CK(cudaMalloc(&d, N * sizeof(float)));
    CK(cudaMalloc(&dp, TPB * sizeof(float)));      // per-block partials
    CK(cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice));

    auto run_partial = [&](int variant, int grid) -> float {
        float *part = (float*)malloc(grid * sizeof(float));
        CK(cudaMemset(dp, 0, grid * sizeof(float)));
        switch (variant) {
            case 1: reduce_v1<<<grid, TPB>>>(d, dp, N); break;
            case 2: reduce_v2<<<grid, TPB>>>(d, dp, N); break;
            case 3: reduce_v3<<<grid, TPB>>>(d, dp, N); break;
            case 4: reduce_v4<<<grid, TPB>>>(d, dp, N); break;
        }
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(part, dp, grid * sizeof(float), cudaMemcpyDeviceToHost));
        double s = 0; for (int i = 0; i < grid; i++) s += part[i];
        free(part);
        return (float)s;
    };

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    int grid_full = (N + TPB - 1) / TPB;
    int grid_small = sms * 8;   // persistent-ish grid for v3/v4

    double tol = 1e-2;  // fp32 sum of 16M random values: rel err ~sqrt(N)*eps
    float got;
    got = run_partial(1, grid_full);
    CHECK("v1 guarded tree", fabs(got - ref_seq) < tol * fabs(ref_seq) + 1.0,
          "got %.4f ref %.4f", got, ref_seq);
    got = run_partial(2, grid_full);
    CHECK("v2 warp-unrolled", fabs(got - ref_seq) < tol * fabs(ref_seq) + 1.0,
          "got %.4f ref %.4f", got, ref_seq);
    got = run_partial(3, grid_small);
    CHECK("v3 strided ILP", fabs(got - ref_seq) < tol * fabs(ref_seq) + 1.0,
          "got %.4f ref %.4f", got, ref_seq);
    got = run_partial(4, grid_small);
    CHECK("v4 warp-shuffle", fabs(got - ref_seq) < tol * fabs(ref_seq) + 1.0,
          "got %.4f ref %.4f", got, ref_seq);

    // v5: atomic single-scalar version (also verify correctness once)
    float *dsum; CK(cudaMalloc(&dsum, sizeof(float)));
    CK(cudaMemset(dsum, 0, sizeof(float)));
    reduce_v5_atomic<<<grid_small, TPB>>>(d, dsum, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float asum; CK(cudaMemcpy(&asum, dsum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK("v5 atomicAdd merge", fabs(asum - ref_seq) < tol * fabs(ref_seq) + 1.0,
          "got %.4f ref %.4f", asum, ref_seq);

    // --- timing table ---
    printf("%-22s %10s\n", "variant", "ms");
    float t;
    t = bench([&]{ reduce_v1<<<grid_full, TPB>>>(d, dp, N); }, 5, 50);
    printf("%-22s %10.4f\n", "v1 guarded tree", t);
    t = bench([&]{ reduce_v2<<<grid_full, TPB>>>(d, dp, N); }, 5, 50);
    printf("%-22s %10.4f\n", "v2 warp-unrolled", t);
    t = bench([&]{ reduce_v3<<<grid_small, TPB>>>(d, dp, N); }, 5, 50);
    printf("%-22s %10.4f\n", "v3 strided ILP", t);
    t = bench([&]{ reduce_v4<<<grid_small, TPB>>>(d, dp, N); }, 5, 50);
    printf("%-22s %10.4f\n", "v4 warp-shuffle", t);
    t = bench([&]{ CK(cudaMemset(dsum,0,sizeof(float))); reduce_v5_atomic<<<grid_small, TPB>>>(d, dsum, N); }, 5, 50);
    printf("%-22s %10.4f\n", "v5 atomic(+memset)", t);

    // --- determinism demo: run v5 twice, compare bit patterns ---
    float r1, r2;
    CK(cudaMemset(dsum, 0, sizeof(float))); reduce_v5_atomic<<<grid_small, TPB>>>(d, dsum, N);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&r1, dsum, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemset(dsum, 0, sizeof(float))); reduce_v5_atomic<<<grid_small, TPB>>>(d, dsum, N);
    CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&r2, dsum, 4, cudaMemcpyDeviceToHost));
    printf("atomic run1 bits=%08x run2 bits=%08x identical=%s\n",
           *(unsigned*)&r1, *(unsigned*)&r2, memcmp(&r1,&r2,4)==0 ? "yes" : "NO (expected!)");
    printf("# lesson: atomics are fast but nondeterministic; training kernels\n");
    printf("#         (llm.c backward) avoid them or accept bitwise drift.\n");

    cudaFree(d); cudaFree(dp); cudaFree(dsum); free(h);
    return harness_summary();
}
