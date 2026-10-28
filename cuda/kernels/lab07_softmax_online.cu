// lab07 — Online softmax (the FlashAttention math) + stable softmax baseline
// Source map: rough/llm.c/train_gpt2_fp32.cu kernel5_time_maps (online softmax over time),
//             docs/audit/llm-c-05 §softmax. Also gpt.c softmax_kernel (two-pass naive).
// Math: softmax(x)_i = exp(x_i - max) / sum(exp(x_j - max)). Naive needs the max
//       BEFORE exponentiating (2 passes). Online algorithm keeps running (m, l):
//         for each new block x_b: m' = max(m, max_b); l = l*exp(m-m') + sum(exp(x_b-m'))
//       One pass, and it's exactly why flash-attention never materializes P.
#include "harness.h"

// v1: two-pass per-row softmax, one warp per row (baseline, llm.c CPU-style)
__global__ void softmax_twopass(const float* __restrict__ x, float* __restrict__ y,
                                int T, int N) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= T) return;
    const float* xr = x + (size_t)row * N;
    float mx = -1e30f;
    for (int i = lane; i < N; i += 32) mx = fmaxf(mx, xr[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    float s = 0.f;
    for (int i = lane; i < N; i += 32) s += expf(xr[i] - mx);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    for (int i = lane; i < N; i += 32)
        y[(size_t)row * N + i] = expf(xr[i] - mx) / s;
}

// v2: ONLINE single-pass softmax: keep running max m and running sum l, rescale l
//     whenever m grows. Processes the row in chunks of 32 (one per lane).
__global__ void softmax_online(const float* __restrict__ x, float* __restrict__ y,
                               int T, int N) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= T) return;
    const float* xr = x + (size_t)row * N;
    float m = -1e30f, l = 0.f;
    // forward pass: streaming (m,l) update per chunk
    for (int base = 0; base < N; base += 32) {
        int i = base + lane;
        float v = (i < N) ? xr[i] : -1e30f;
        float mc = v;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, off));
        float mn = fmaxf(m, mc);
        float e = (i < N) ? expf(v - mn) : 0.f;
        float sc = 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sc += __shfl_xor_sync(0xffffffffu, e, off);
        l = l * expf(m - mn) + sc;
        m = mn;
    }
    // write pass: recompute exp cheaply (or store; here recompute = 2 reads total)
    for (int i = lane; i < N; i += 32)
        y[(size_t)row * N + i] = expf(xr[i] - m) / l;
}

static void softmax_cpu(const double* x, double* y, int T, int N) {
    for (int t = 0; t < T; t++) {
        const double* xr = x + (size_t)t * N;
        double m = -1e300;
        for (int i = 0; i < N; i++) m = fmax(m, xr[i]);
        double s = 0;
        for (int i = 0; i < N; i++) s += exp(xr[i] - m);
        for (int i = 0; i < N; i++) y[(size_t)t*N+i] = exp(xr[i]-m)/s;
    }
}

int main() {
    print_gpu_info();
    int T = 4096, N = 1024;   // attention-like row width
    size_t sz = (size_t)T * N * sizeof(float);
    float *x = (float*)malloc(sz), *y1 = (float*)malloc(sz), *y2 = (float*)malloc(sz);
    double *xc = (double*)malloc((size_t)T*N*8), *yc = (double*)malloc((size_t)T*N*8);
    fill_rand(x, (size_t)T*N, 2026);
    // inject a huge value to prove overflow-safety of both variants
    for (int t = 0; t < T; t++) x[(size_t)t*N + (t % N)] = 90.f; // exp(90) would overflow raw
    for (size_t i = 0; i < (size_t)T*N; i++) xc[i] = x[i];
    softmax_cpu(xc, yc, T, N);

    float *d_x, *d_y;
    CK(cudaMalloc(&d_x, sz)); CK(cudaMalloc(&d_y, sz));
    CK(cudaMemcpy(d_x, x, sz, cudaMemcpyHostToDevice));
    dim3 blk(256), grd((T + 7) / 8);

    float ms1 = bench([&]{ softmax_twopass<<<grd, blk>>>(d_x, d_y, T, N); });
    CK(cudaMemcpy(y1, d_y, sz, cudaMemcpyDeviceToHost));
    printf("softmax twopass: %.3f ms\n", ms1);
    check_close("softmax twopass vs oracle", yc, y1, (size_t)T*N, 1e-4f);

    CK(cudaMemset(d_y, 0, sz));
    float ms2 = bench([&]{ softmax_online<<<grd, blk>>>(d_x, d_y, T, N); });
    CK(cudaMemcpy(y2, d_y, sz, cudaMemcpyDeviceToHost));
    printf("softmax online: %.3f ms (%.2fx)\n", ms2, ms1/ms2);
    check_close("softmax online vs oracle", yc, y2, (size_t)T*N, 1e-4f);
    // online must not produce NaN even though exp(90)=inf in fp32 raw
    int nan = 0; for (size_t i = 0; i < (size_t)T*N; i++) if (!(y2[i]==y2[i])) nan++;
    CHECK("no NaN from exp(90) rows", nan == 0, "%d NaNs", nan);
    return report();
}
