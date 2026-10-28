# Lab 01 — Thread Mapping, Global Indices & Bounds Guards

**Tier 1 (Foundations)** · Source analogues: every kernel in `rough/gpt.c` (`matmul_kernel`, `softmax_*`) and `rough/llm.c/train_gpt2_fp32_cu.cu`.
**Runnable:** `cuda/kernels/lab01_threads.cu` → `nvcc -O3 lab01_threads.cu -o bin/lab01 && ./bin/lab01`

## 1. Problem
A GPU executes one scalar program on thousands–millions of *independent* threads simultaneously. There is no shared control flow between threads; each must figure out **which slice of the data it owns**. Getting this mapping right is prerequisite to everything else.

## 2. Math / model on paper
- Execution hierarchy: **grid → blocks → warps (32 threads) → lanes**.
- Coordinates per thread: `blockIdx.{x,y,z}` (block id), `threadIdx.{x,y,z}` (lane-in-block), `blockDim` (threads/block), `gridDim` (blocks/grid).
- Global linear index for 1-D problems:
  `i = blockIdx.x * blockDim.x + threadIdx.x`
- Grid size for N elements with B threads/block: `grid = ceil(N/B) = (N + B - 1) / B`.

## 3. Naive version and its failure modes
Lab 01 runs `add_one_naive` with N = 1,000,003 (not divisible by 256):
- **Missing bounds guard** → last block writes past array end → silent corruption or crash. The guard `if (i < n)` is not optional.
- **Floor instead of ceiling division** → tail elements never processed. Classic off-by-one.
- **Forgetting `cudaGetLastError()` after launch** → a bad launch config (e.g., >1024 threads/block) fails *silently*; you only find out at the next unrelated CUDA call. Always check after async launches.

## 4. Optimization journey
Not speed here — *robustness*:
1. One element per thread + guard (works, grid scales with N).
2. **Grid-stride loop**: fixed small grid; each thread loops `i += blockDim.x*gridDim.x`. Handles any N with the same launch config; fewer block-scheduling overheads for huge N; required style for persistent-kernel designs. This is what cuBLAS/PyTorch use internally and what llm.c's fp32 trainer uses for its elementwise ops.

Measured (lab prints real numbers on your GPU): both variants produce bit-identical results; timing difference is noise-level for this bandwidth-bound toy, but stride wins when N changes per iteration (training shapes change!) because host code never recomputes grid dims.

## 5. Source map (real repos)
| Concept | Where |
|---|---|
| `i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n)` | `rough/gpt.c` all kernels; `llm.c/train_gpt2_fp32_cu.cu` encoder kernel |
| ceil-division launch `(N+B-1)/B` | every `<<<...>>>` site in gpt.c main |
| Bounds-guarded float4 tail handling | `llm.c/train_gpt2.cu` forward pass "encoder" section (see docs/audit/llm-c-05 §float4) |

## 6. Copy-paste checklist for your project
- [ ] `CK()` macro around every runtime call (harness.h provides it).
- [ ] `cudaGetLastError()` immediately after each launch, `cudaDeviceSynchronize()` before reading results.
- [ ] Ceiling division helper: `#define GRID(n,b) (((n)+(b)-1)/(b))`.
- [ ] Prefer grid-stride when N varies between calls.
- [ ] Never assume `threadIdx.x` fits in `int` math overflow: for >2^31 elements cast to `long long` / split into chunks.

## 7. Pitfalls
- **Divergence inside a warp**: `if (i<n)` costs nothing when only the last partial warp takes the false branch; *data-dependent* branches cost real time.
- blockDim 256 is a safe default; multiples of 32 always; max 1024.
- Launching `<<<0, ...>>>` (empty grid when N=0) is illegal — guard host-side.
- Async semantics: memcpy D2H synchronizes implicitly; benchmarking without warmup measures first-launch module load (~ms), not kernel time. Harness `bench()` does both correctly.
