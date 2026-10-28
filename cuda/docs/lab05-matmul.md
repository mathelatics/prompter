# Lab 05 — Tiled SGEMM: v0 → v2 (the path llm.c actually walks)

**Tier 1 capstone** · Source analogues: `rough/gpt.c` naive `matmul_kernel`; `rough/llm.c/train_gpt2_fp32_cu.cu` `matmul_forward_kernel4` (docs/audit/llm-c-05); `llmc/kernels/matmul_forward.cuh` bf16/cuBLASLt endpoint (docs/audit/llm-c-08).
**Runnable:** `cuda/kernels/lab05_matmul.cu` (1024³ fp32, double-precision CPU oracle)

## 1. Problem
C = A·B, row-major, M=N=K=1024. GEMM is 70–90% of transformer runtime; understanding why hand kernels lag cuBLAS by ~5× and what each optimization buys is the core competency of this curriculum.

## 2. Math on paper
- Work: 2·M·N·K FLOPs (≈2.1 GFLOP here). Naive traffic per output element: 2K loads from DRAM/L2 ⇒ total ≈ 2·M·N·K·4 B — traffic scales like work × K, i.e. **memory-bound by re-fetching**.
- Tiling with tile side T: each global element loaded once per tile-use ⇒ traffic ÷T. With T=16, intensity rises to ~T/2 flop/B = 8 — approaching T4's balance point (26), which is why tiling gets you most of the way.
- Register blocking R×R per thread cuts shared→register traffic by R² vs R for same compute (load R+R values, do R² FMAs).

## 3. Naive version (v0)
One output per thread, inner loop straight over global A row & B column. Lab times it against a single-threaded double CPU reference — expect only modest speedup; that's the lesson: *naive GEMM wastes the GPU*.

## 4. Optimization journey (all in lab source)
1. **v1 tiled**: stage 16×16 sub-tiles of A and B into shared memory, sweep K in tiles, two `__syncthreads()` per sweep. Classic textbook kernel; typically 3–6× v0.
2. **v2 register 2×2**: each thread owns 4 outputs; shared arrays sized `[2T][T]` / `[T][2T]` so both A rows and B cols are staged. FMA:shared-load ratio doubles → often another 1.5–2×.
3. What the labs deliberately stop short of (covered later): double-buffering (`cp.async`/prefetch to hide sync latency), 64-B swizzled smem layouts, tensor-core `mma.sync.m16n8k16` (Lab 16), and finally calling cuBLASLt with autotuned algos (Lab 17) — mirroring exactly how llm.c went from custom fp32 kernels to cuBLAS(+Lt) bf16.

## 5. Source map
| Concept | Where |
|---|---|
| naive triple-loop kernel | `rough/gpt.c` matmul_kernel |
| 16×16-style tiled fwd + fused bias add | `train_gpt2_fp32_cu.cu` matmul_forward_kernel4 |
| weight-stationary variants & backward dW/dX pair | docs/audit/llm-c-03 §matmul, llm-c-08 §matmul_forward/backward |

## 6. Copy-paste checklist
- [ ] Verify against double CPU oracle with rel-L2 < 1e-3 (fp32 accumulation over K=1024 drifts more than elementwise ops).
- [ ] Print GFLOP/s = 2MNK/t; compare to device peak fp32 FMA rate %.
- [ ] Odd shapes: guard every load/store; never assume K % TILE == 0 (training seq-len varies!).
- [ ] For transformer use, fuse the bias-add and (optionally) gelu epilogue while accumulators live in registers — see Lab 12.

## 7. Pitfalls
- Two `__syncthreads()` per K-sweep are mandatory; removing the second lets a fast warp overwrite tiles still being read (intermittent wrong answers — the hardest bug class in CUDA).
- Shared array dims must match the *staged* region (`[2T][T]` in v2), not the block dims — mismatch = silent OOB reads within smem.
- Bank conflicts inside `As[ty][k]` broadcast (all lanes same address) are free; conflicts appear in `Bs[k][tx]` if padded layout wrong (Lab 04 math applies).
- TF32: on Ampere+, plain `float` GEMMs via cuBLAS silently use 10-bit-mantissa TC path unless opted out (llm.c toggles this explicitly — docs/audit/llm-c-05 CLI section). Hand kernels don't; your numbers differ from cuBLAS partly because of this.
