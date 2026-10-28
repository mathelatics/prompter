// lab08 — GELU (tanh approximation) forward + backward, with the llm.c float-control trick
// Source map: rough/llm.c/train_gpt2.c gelu_forward/gelu_backward; train_gpt2_fp32.cu
//             kernel4_map_gelu_not_res. docs/audit/llm-c-03 §gelu.
// Math: gelu_tanh(x) = 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
//       d/dx via chain rule; llm.c computes in fp32 and pins x87-style precision on CPU
//       with #pragma STDC FP_CONTRACT OFF / fenv tricks; on GPU we disable FMA contraction
//       for the oracle-matching variant using __fmul_rn/__fadd_rn intrinsics.
#include "harness.h"

#define SQRT_2_PI 0.7978845608028654f
#define C0        0.044715f

__device__ __forceinline__ float gelu_f(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.f + tanhf(SQRT_2_PI * (x + C0 * x3)));
}

// non-contracted variant: force rounded mul/add so bit-patterns match a
// no-FMA reference (this is the GPU analogue of llm.c's STCF pragma story).
__device__ __forceinline__ float gelu_f_nc(float x) {
    float x3 = __fmul_rn(__fmul_rn(x, x), x);
    float inner = __fadd_rn(x, __fmul_rn(C0, x3));
    float t = tanhf(__fmul_rn(SQRT_2_PI, inner));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.f, t));
}

__global__ void gelu_fwd(const float* __restrict__ x, float* __restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gelu_f(x[i]);
}

// backward: g'(x) = 0.5(1+t) + 0.5x(1-t^2)*u', u=sqrt(2/pi)(x+c x^3), u'=sqrt(2/pi)(1+3c x^2)
__global__ void gelu_bwd(const float* __restrict__ dy, const float* __restrict__ x,
                         float* __restrict__ dx, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float u = SQRT_2_PI * (v + C0 * v * v * v);
    float t = tanhf(u);
    float du = SQRT_2_PI * (1.f + 3.f * C0 * v * v);
    dx[i] = dy[i] * (0.5f * (1.f + t) + 0.5f * v * (1.f - t * t) * du);
}

static double gelu_cpu(double x) {
    double u = sqrt(2.0/M_PI) * (x + 0.044715 * x*x*x);
    return 0.5 * x * (1.0 + tanh(u));
}

int main() {
    print_gpu_info();
    size_t n = 1 << 22; // 4M elements
    float *x = (float*)malloc(n*4), *y = (float*)malloc(n*4), *dy = (float*)malloc(n*4);
    double *ref = (double*)malloc(n*8);
    fill_rand(x, n, 555); fill_rand(dy, n, 666);
    for (size_t i = 0; i < n; i++) ref[i] = gelu_cpu(x[i]);

    float *d_x, *d_y, *d_dy, *d_dx;
    CK(cudaMalloc(&d_x, n*4)); CK(cudaMalloc(&d_y, n*4));
    CK(cudaMalloc(&d_dy, n*4)); CK(cudaMalloc(&d_dx, n*4));
    CK(cudaMemcpy(d_x, x, n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dy, dy, n*4, cudaMemcpyHostToDevice));
    dim3 blk(256), grd((unsigned)((n + 255) / 256));
    float *dxh = (float*)malloc(n*4);

    float ms = bench([&]{ gelu_fwd<<<grd, blk>>>(d_x, d_y, n); });
    CK(cudaMemcpy(y, d_y, n*4, cudaMemcpyDeviceToHost));
    printf("gelu fwd: %.3f ms (%.1f GB/s)\n", ms, (double)n*8/ms/1e6);
    check_close("gelu fwd vs double oracle", ref, y, n, 1e-5f);

    // backward kernel vs double oracle (analytic derivative)
    float ms_b = bench([&]{ gelu_bwd<<<grd, blk>>>(d_dy, d_x, d_dx, n); });
    CK(cudaMemcpy(dxh, d_dx, n*4, cudaMemcpyDeviceToHost));
    printf("gelu bwd: %.3f ms\n", ms_b);
    for (size_t i = 0; i < n; i++) {
        double v = x[i];
        double u = sqrt(2.0/M_PI) * (v + 0.044715*v*v*v);
        double t = tanh(u), du = sqrt(2.0/M_PI)*(1.0+3.0*0.044715*v*v);
        ref[i] = dy[i] * (0.5*(1.0+t) + 0.5*v*(1.0-t*t)*du);
    }
    check_close("gelu bwd vs double oracle", ref, dxh, n, 1e-4f);

    // gradient sanity: central finite difference of the FORWARD at 1000 samples
    int bad_fd = 0;
    for (int s = 0; s < 1000; s++) {
        size_t i = (size_t)(xs64_next() % n);
        double h = 1e-3, xp = x[i]+h, xm = x[i]-h;
        double fd = (gelu_cpu(xp) - gelu_cpu(xm)) / (2*h);
        if (fabs(fd - dxh[i]) > 5e-3 * (1.0 + fabs(fd))) bad_fd++;
    }
    CHECK("gelu grad vs finite-diff", bad_fd == 0, "%d/1000 mismatches", bad_fd);
    return report();
}
