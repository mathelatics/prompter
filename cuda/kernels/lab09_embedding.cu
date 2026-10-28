// lab09 — Embedding gather (fwd) + scatter-add with atomics (bwd), weight-tied grads
// Source map: rough/llm.c/train_gpt2.c encoder_forward/encoder_backward (docs/audit/llm-c-03 §embed).
// Math: fwd: out[t,:] = table[ids[t],:] (+ pos table). bwd: for each token, add grad
//       into table row -> MANY threads hit the same row => atomicAdd per element.
//       Weight tying (llm.c): wte gradient accumulates from BOTH head and encoder;
//       we emulate by scattering two grad streams into one table.
// Determinism note: atomic float adds are order-nondeterministic (lab03 demo);
//       llm.c accepts this in fp32. We verify rel_l2 tolerance instead of bit-exact.
#include "harness.h"

#define V 50257   // GPT-2 vocab
#define C 768     // hidden

// warp-per-token gather: block 256 -> 8 tokens
__global__ void emb_fwd_warp(const int* __restrict__ ids, const float* __restrict__ wte,
                             const float* __restrict__ wpe, float* __restrict__ x, int T) {
    int lane = threadIdx.x % 32;
    int tok = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (tok >= T) return;
    int id = ids[tok];
    const float* src = wte + (size_t)id * C;
    const float* pos = wpe + (size_t)(tok % 1024) * C;
    float* dst = x + (size_t)tok * C;
    for (int i = lane; i < C; i += 32) dst[i] = src[i] + pos[i];
}

// scatter-add backward: dx rows -> table rows via atomics
__global__ void emb_bwd_atomic(const int* __restrict__ ids, const float* __restrict__ d_x,
                               float* __restrict__ d_wte, int T) {
    int lane = threadIdx.x % 32;
    int tok = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (tok >= T) return;
    int id = ids[tok];
    const float* src = d_x + (size_t)tok * C;
    float* dst = d_wte + (size_t)id * C;
    for (int i = lane; i < C; i += 32) atomicAdd(dst + i, src[i]);
}

int main() {
    print_gpu_info();
    int T = 4096;
    size_t tsz = (size_t)T * C * sizeof(float);
    size_t wsz = (size_t)V * C * sizeof(float);
    int *ids = (int*)malloc(T*4);
    float *wte = (float*)malloc(wsz), *wpe = (float*)malloc((size_t)1024*C*4);
    float *x = (float*)malloc(tsz), *dx = (float*)malloc(tsz), *dw = (float*)calloc(wsz,1);
    fill_rand(wte, (size_t)V*C, 111); fill_rand(wpe, (size_t)1024*C, 222);
    fill_rand(dx, (size_t)T*C, 333);
    xs64_state = 4242;
    for (int t = 0; t < T; t++) ids[t] = (int)(xs64_next() % V);

    // CPU oracle forward
    double *ref = (double*)malloc(tsz);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++)
            ref[(size_t)t*C+i] = wte[(size_t)ids[t]*C+i] + wpe[(size_t)(t%1024)*C+i];

    int *d_ids; float *d_wte, *d_wpe, *d_x, *d_dx, *d_dw;
    CK(cudaMalloc(&d_ids, T*4)); CK(cudaMalloc(&d_wte, wsz)); CK(cudaMalloc(&d_wpe, (size_t)1024*C*4));
    CK(cudaMalloc(&d_x, tsz)); CK(cudaMalloc(&d_dx, tsz)); CK(cudaMalloc(&d_dw, wsz));
    CK(cudaMemcpy(d_ids, ids, T*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wte, wte, wsz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wpe, wpe, (size_t)1024*C*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dx, dx, tsz, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_dw, 0, wsz));

    dim3 blk(256), grd((T + 7) / 8);
    float ms_f = bench([&]{ emb_fwd_warp<<<grd, blk>>>(d_ids, d_wte, d_wpe, d_x, T); });
    CK(cudaMemcpy(x, d_x, tsz, cudaMemcpyDeviceToHost));
    printf("emb fwd: %.3f ms (%.1f GB/s)\n", ms_f, (double)tsz*2/ms_f/1e6);
    check_close("embedding gather", ref, x, (size_t)T*C, 1e-5f);

    float ms_b = bench([&]{ emb_bwd_atomic<<<grd, blk>>>(d_ids, d_dx, d_dw, T); }, 2, 5);
    // NOTE: bench repeats the kernel -> grads accumulate k times! Re-run once clean:
    CK(cudaMemset(d_dw, 0, wsz));
    emb_bwd_atomic<<<grd, blk>>>(d_ids, d_dx, d_dw, T);
    CK(cudaMemcpy(dw, d_dw, wsz, cudaMemcpyDeviceToHost));
    printf("emb bwd (single pass): ~%.3f ms\n", ms_b/5);
    // CPU oracle for scatter
    double *rfd = (double*)calloc(wsz/4, sizeof(double));
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++) rfd[(size_t)ids[t]*C+i] += dx[(size_t)t*C+i];
    // only touched rows matter; untouched rows are 0 on both sides
    float rl = rel_l2(dw, rfd, (size_t)V*C);
    CHECK("embedding scatter (atomics, rel_l2)", rl < 1e-6f, "rel_l2=%.2e", rl);
    free(rfd);
    return report();
}
