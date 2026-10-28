// lab06 — LayerNorm forward+backward, warp-per-row (llm.c style)
// Source map: rough/llm.c/train_gpt2_fp32.cu kernel1_map_ln_not_res; CPU ref
//             layer_norm_forward/backward in train_gpt2.c (docs/audit/llm-c-03).
// Math: y = (x-mean)/sqrt(var+eps)*w + b, eps=1e-5. Backward uses c1=sum(dy*w),
//       c2=sum(dy*w*xhat): dx = w*(dy - c1/N - xhat*c2/N)*rstd.
// GPU idiom: ONE WARP PER ROW, butterfly __shfl_xor reductions (no shared mem).
#include "harness.h"

#define EPS 1e-5f

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__global__ void ln_fwd_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ b, float* __restrict__ y,
                              float* __restrict__ rstd, int T, int N) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= T) return;
    const float* xr = x + (size_t)row * N;
    float* yr = y + (size_t)row * N;

    float s = 0.f;
    for (int i = lane; i < N; i += 32) s += xr[i];
    float mean = warp_sum(s) / N;

    float vs = 0.f;
    for (int i = lane; i < N; i += 32) { float d = xr[i] - mean; vs += d * d; }
    float r = rsqrtf(warp_sum(vs) / N + EPS);
    if (lane == 0) rstd[row] = r;
    for (int i = lane; i < N; i += 32)
        yr[i] = (xr[i] - mean) * r * w[i] + b[i];
}

__global__ void ln_bwd_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                              const float* __restrict__ w, const float* __restrict__ rstd,
                              float* __restrict__ dx, int T, int N) {
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= T) return;
    const float* dyr = dy + (size_t)row * N;
    const float* xr = x + (size_t)row * N;
    float r = rstd[row];

    float s = 0.f;
    for (int i = lane; i < N; i += 32) s += xr[i];
    float mean = warp_sum(s) / N;

    float c1 = 0.f, c2 = 0.f;
    for (int i = lane; i < N; i += 32) {
        float dwv = dyr[i] * w[i];
        float xhat = (xr[i] - mean) * r;
        c1 += dwv;
        c2 += dwv * xhat;
    }
    c1 = warp_sum(c1) / N;
    c2 = warp_sum(c2) / N;
    for (int i = lane; i < N; i += 32) {
        float xhat = (xr[i] - mean) * r;
        dx[(size_t)row * N + i] = w[i] * (dyr[i] - c1 - xhat * c2) * r;
    }
}

static void ln_cpu_fwd(const double* x, const double* w, const double* b,
                       double* y, int T, int N) {
    for (int t = 0; t < T; t++) {
        const double* xr = x + (size_t)t * N;
        double m = 0; for (int i = 0; i < N; i++) m += xr[i]; m /= N;
        double v = 0; for (int i = 0; i < N; i++) { double d = xr[i]-m; v += d*d; } v /= N;
        double r = 1.0 / sqrt(v + EPS);
        for (int i = 0; i < N; i++)
            y[(size_t)t*N+i] = (xr[i]-m) * r * w[i] + b[i];
    }
}

int main() {
    print_gpu_info();
    int T = 4096, N = 768; // GPT-2 124M hidden size
    size_t sz = (size_t)T * N * sizeof(float);
    float *x = (float*)malloc(sz), *w = (float*)malloc(N*4), *b = (float*)malloc(N*4);
    float *y_gpu = (float*)malloc(sz), *rstd_h = (float*)malloc(T*4);
    float *dy = (float*)malloc(sz), *dx = (float*)malloc(sz);
    double *xc = (double*)malloc((size_t)T*N*8), *wc = (double*)malloc(N*8);
    double *bc = (double*)malloc(N*8), *yc = (double*)malloc((size_t)T*N*8);
    double *dc = (double*)malloc((size_t)T*N*8);
    fill_rand(x, (size_t)T*N, 1234); fill_rand(w, N, 999); fill_rand(b, N, 777);
    fill_rand(dy, (size_t)T*N, 4242);
    for (size_t i = 0; i < (size_t)T*N; i++) xc[i] = x[i];
    for (int i = 0; i < N; i++) { wc[i] = w[i]; bc[i] = b[i]; }

    ln_cpu_fwd(xc, wc, bc, yc, T, N);

    float *d_x, *d_y, *d_rstd, *d_w, *d_b, *d_dy, *d_dx;
    CK(cudaMalloc(&d_x, sz)); CK(cudaMalloc(&d_y, sz)); CK(cudaMalloc(&d_rstd, T*4));
    CK(cudaMalloc(&d_w, N*4)); CK(cudaMalloc(&d_b, N*4));
    CK(cudaMalloc(&d_dy, sz)); CK(cudaMalloc(&d_dx, sz));
    CK(cudaMemcpy(d_x, x, sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_w, w, N*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, b, N*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dy, dy, sz, cudaMemcpyHostToDevice));

    dim3 blk(256); dim3 grd((T + 7) / 8);
    float ms_f = bench([&]{ ln_fwd_kernel<<<grd, blk>>>(d_x, d_w, d_b, d_y, d_rstd, T, N); });
    CK(cudaMemcpy(y_gpu, d_y, sz, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rstd_h, d_rstd, T*4, cudaMemcpyDeviceToHost));
    double gbytes = (double)(T*N*sizeof(float)) * 2 / ms_f / 1e6; // rd+wr approx
    printf("ln_fwd: %.3f ms (~%.1f GB/s)\n", ms_f, gbytes);
    check_close("layernorm fwd", yc, y_gpu, (size_t)T*N, 1e-3f);

    float ms_b = bench([&]{ ln_bwd_kernel<<<grd, blk>>>(d_dy, d_x, d_w, d_rstd, d_dx, T, N); });
    CK(cudaMemcpy(dx, d_dx, sz, cudaMemcpyDeviceToHost));
    printf("ln_bwd: %.3f ms\n", ms_b);
    // exact double oracle for backward (same algebra as kernel)
    for (int t = 0; t < T; t++) {
        const double* xr = xc + (size_t)t*N; const float* dyr = dy + (size_t)t*N;
        double m=0,v=0; for(int i=0;i<N;i++) m+=xr[i]; m/=N;
        for(int i=0;i<N;i++){double d=xr[i]-m; v+=d*d;} v/=N;
        double r=1.0/sqrt(v+EPS);
        double c1=0,c2=0;
        for(int i=0;i<N;i++){double dwv=dyr[i]*wc[i]; double xh=(xr[i]-m)*r; c1+=dwv; c2+=dwv*xh;}
        c1/=N; c2/=N;
        for(int i=0;i<N;i++){double xh=(xr[i]-m)*r; dc[(size_t)t*N+i]=wc[i]*((double)dyr[i]-c1-xh*c2)*r;}
    }
    check_close("layernorm bwd", dc, dx, (size_t)T*N, 1e-3f);
    return report();
}
