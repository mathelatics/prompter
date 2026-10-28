// lab04_transpose.cu — Lab 04: shared memory & the bank-conflict saga.
// Matrix transpose is THE canonical example of "same math, 10x speed gap"
// depending purely on memory access patterns.
// Why banks matter: shared memory = 32 banks x 4 bytes. Threads in a warp
// hitting DIFFERENT banks run in one cycle; hitting the SAME bank serialize.
// Reading tile ROW-WISE (stride 1) -> conflict-free. Writing OUT row-wise
// from a column-major view -> every lane hits same bank -> 32-way conflict.
// The +1 padding trick shifts each row by one float so columns land on
// different banks. This exact pattern appears in llm.c matmul_forward_kernel
// (docs/audit/llm-c-05 §tiled matmul).
#include "harness.h"

#define TILE 32   // blockDim.x == blockDim.y == 32 => exactly one warp per row

__global__ void transpose_naive(const float *in, float *out, int n) {
    // Coalesced READ of in[row][col], but SCATTERED write to out[col][row].
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) out[col * n + row] = in[row * n + col];
}

__global__ void transpose_tile_nopad(const float *in, float *out, int n) {
    __shared__ float tile[TILE][TILE];          // no padding
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    if (row < n && col < n) tile[ty][tx] = in[row * n + col];
    __syncthreads();
    // After swap, this warp writes consecutive `out` addresses (coalesced)...
    int orow = blockIdx.x * TILE + tx, ocol = blockIdx.y * TILE + ty;
    if (orow < n && ocol < n) out[ocol * n + orow] = tile[tx][ty];
    // ...but reading tile[tx][ty] with tx varying strides by TILE floats =
    // all 32 lanes read the SAME bank (bank = (tx*TILE+ty)%32 = ty const!)
    // => 32-way conflict here even though global stores are coalesced.
}

__global__ void transpose_tile_pad(const float *in, float *out, int n) {
    __shared__ float tile[TILE][TILE + 1];      // +1 pad kills the conflict
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    if (row < n && col < n) tile[ty][tx] = in[row * n + col];
    __syncthreads();
    int orow = blockIdx.x * TILE + tx, ocol = blockIdx.y * TILE + ty;
    if (orow < n && ocol < n) out[ocol * n + orow] = tile[tx][ty];
    // bank = (tx*(TILE+1)+ty) % 32 varies with tx => conflict-free.
}

int main(void) {
    print_gpu_info();
    const int N = 4096;                       // 4096^2 floats = 64 MB
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hin = (float*)malloc(bytes);
    float *hout = (float*)malloc(bytes);
    float *d_in, *d_out;
    CK(cudaMalloc(&d_in, bytes)); CK(cudaMalloc(&d_out, bytes));
    for (size_t i = 0; i < (size_t)N * N; i++) hin[i] = rand_f(-1.f, 1.f);
    CK(cudaMemcpy(d_in, hin, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // correctness for all three: verify a few known positions after transpose
    auto verify = [&](const char *name) {
        CK(cudaMemcpy(hout, d_out, bytes, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int k = 0; k < 200; k++) {
            int r = rand_f(0, (float)N - 0.5f), c = rand_f(0, (float)N - 0.5f);
            if (hout[(size_t)c * N + r] != hin[(size_t)r * N + c]) bad++;
        }
        CHECK(name, bad == 0, "%d mismatches on sampled positions", bad);
    };

    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    CK(cudaGetLastError()); verify("naive scatter");
    float t_naive = bench([&]{ transpose_naive<<<grid, block>>>(d_in, d_out, N); }, 3, 20);

    CK(cudaMemset(d_out, 0, bytes));
    transpose_tile_nopad<<<grid, block>>>(d_in, d_out, N);
    CK(cudaGetLastError()); verify("tile no-pad");
    float t_nopad = bench([&]{ transpose_tile_nopad<<<grid, block>>>(d_in, d_out, N); }, 3, 20);

    CK(cudaMemset(d_out, 0, bytes));
    transpose_tile_pad<<<grid, block>>>(d_in, d_out, N);
    CK(cudaGetLastError()); verify("tile padded +1");
    float t_pad = bench([&]{ transpose_tile_pad<<<grid, block>>>(d_in, d_out, N); }, 3, 20);

    double gb = 2.0 * bytes / 1e9;  // read + write whole matrix
    printf("naive     : %.4f ms | %.1f GB/s\n", t_naive, gb / (t_naive * 1e-3));
    printf("tile nopad: %.4f ms | %.1f GB/s  <- shared-load bank conflicts\n", t_nopad, gb / (t_nopad * 1e-3));
    printf("tile pad+1: %.4f ms | %.1f GB/s  <- fix costs 1 extra column\n", t_pad, gb / (t_pad * 1e-3));
    printf("# lesson: identical algorithm; layout alone moves performance.\n");
    printf("#         profile with: ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./lab04\n");

    cudaFree(d_in); cudaFree(d_out); free(hin); free(hout);
    return harness_summary();
}
