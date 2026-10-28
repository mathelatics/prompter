# Lab 04 — Shared Memory & the Bank-Conflict Saga (Transpose)

**Tier 1** · Source analogue: llm.c `matmul_forward_kernel4` tile staging (docs/audit/llm-c-05 §tiled matmul).
**Runnable:** `cuda/kernels/lab04_transpose.cu`

## 1. Problem
Transposed writes break coalescing; transposed reads from tiled shared memory break banks. Transpose is "same algorithm, 3 implementations, up to 10× speed spread" — the perfect demonstration that **layout is performance** on GPUs.

## 2. Math / hardware model
- Global DRAM serves a warp in 32-B sectors: consecutive lanes → consecutive floats = 1 transaction per 8 lanes ("coalesced"). Scattered = up to 32 transactions.
- Shared memory = **32 banks × 4 B**. A warp accessing bank `addr%32`: all-different banks → 1 cycle; k lanes same bank → k-way conflict, serialized k×.
- Row-major array `S[R][C]`: row stride C floats. If C ≡ 0 (mod 32), column reads (`S[tx][ty]`, tx varying) hit bank `(tx*C+ty)%32 = ty` for every lane → **32-way conflict**. Padding to C=TILE+1 makes it `(tx*(TILE+1)+ty)%32`, distinct per tx ✔.

## 3. Naive version
`out[col*n + row] = in[row*n + col]` — coalesced global READ, scattered global WRITE (~N²/8 extra sectors). On 4096² fp32 lab measures this directly (GB/s printed).

## 4. Optimization journey
1. **Tile through shared memory**: block loads a 32×32 tile coalesced, writes it out coalesced *in transposed orientation*. Both global passes now coalesced.
2. **But** the shared read-back `tile[tx][ty]` reintroduces the bank conflict above → measure v_nopad vs v_pad in the lab (typically 1.3–2× gap on transpose alone; larger on older archs).
3. `tile[TILE][TILE+1]` fixes it for one float of extra smem. Alternatives you'll see in production: XOR-swizzle layouts (no padding waste — used by CUTLASS/cuBLASLt), or transpose-free algorithms.

## 5. Source map
| Concept | Where |
|---|---|
| Tiled staging with syncthreads between load/compute | `rough/llm.c/train_gpt2_fp32_cu.cu` matmul_forward_kernel4 (docs/audit/llm-c-05) |
| bf16 tiles where conflicts are counted in 4-B pairs | `rough/llm.c/llmc/kernels/matmul_forward.cuh` (docs/audit/llm-c-08) |

## 6. Copy-paste checklist
- [ ] Any kernel reading a column (or stride-C slice) of shared arrays → pad `+1` or swizzle.
- [ ] Verify with `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./lab04` — expect ~0 only in padded variant.
- [ ] Keep both global patterns coalesced by swapping which index is `threadIdx.x` at store time.
- [ ] For bf16/fp16 packed access, conflicts are computed on 4-byte words: two halfs share a bank — pad in *words*, not elements.

## 7. Pitfalls
- `__shared__ float s[TILE][TILE+1]` indexed as `[ty][tx]` for stores but `[tx][ty]` for loads — mixing these up silently corrupts results (correctness check in lab catches it via sampled positions).
- Padded dimension breaks `float4` vectorized shared access alignment — choose one trick per kernel.
- Occupancy: TILE×TILE smem per block limits blocks/SM; 32×32 fp32 = 4 KB (+pad ≈ 4.1 KB) is safe everywhere.
