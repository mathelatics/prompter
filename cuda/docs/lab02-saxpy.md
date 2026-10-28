# Lab 02 — Memory-Bound Kernels & Effective Bandwidth (SAXPY)

**Tier 1** · Source analogue: llm.c `float4` encoder kernel (`docs/audit/llm-c-05 §kernel1`).
**Runnable:** `cuda/kernels/lab02_saxpy.cu`

## 1. Problem
`y = a*x + y` over 16M floats. It looks trivial; it is the single most important mental model in CUDA: **most kernels are limited by how fast bytes move, not how fast numbers compute.**

## 2. Math on paper
Per element: 2 FLOPs (mul+add), 3 float transfers (read x, read-modify-write y ⇒ counted as 2 reads + 1 write = 12 B).
Arithmetic intensity = 2 / 12 ≈ **0.17 flop/byte**.
Roofline: T4 ≈ 8,000 GFLOP/s fp32 but ≈ 300 GB/s DRAM → machine balance = 26 flop/byte. Since 0.17 ≪ 26, SAXPY sits flat on the bandwidth roof. **Ceiling time for N=2²⁴:** 3·N·4 B / 300 GB/s ≈ **0.67 ms**. If your measurement is near that, you're done — no math trick can help.

## 3. Naive version
One thread per element. On modern GPUs this already hits ~90% of peak *if* accesses are coalesced (consecutive lanes → consecutive addresses). Failure mode to demonstrate: strided access (e.g., `i += 32`) collapses bandwidth ~32× because each warp touches 32 separate 32-byte sectors.

## 4. Optimization journey
1. **Coalescing audit** (index math above) — free.
2. **float4 vectorization**: one thread loads/stores 16 bytes at once. Fewer instructions, wider transactions, better MLP (memory-level parallelism). This is exactly llm.c's trick: cast `float*` → `float4*`, process `n/4`, handle tail with scalar guard. Requires 16-byte alignment (cudaMalloc guarantees 256-B alignment ✔).
3. Diminishing returns beyond ~95% of STREAM copy bandwidth; remaining gains come from **fusion** (see Lab 12–14): don't write y at all, feed it into the next op while it's in registers. Each saved round-trip through DRAM is a direct win.

Lab prints measured GB/s for both variants on your hardware — record them in your notes; they calibrate every future estimate.

## 5. Source map
| Concept | Where |
|---|---|
| float4 elementwise loop + tail | `rough/llm.c/train_gpt2_fp32_cu.cu` forward-pass encoder (~"kernel 1"); digested in `docs/audit/llm-c-05-train_gpt2_fp32_cu.md` |
| Same idea bf16-packed | `rough/llm.c/llmc/kernels/elementwise.cuh` (docs/audit/llm-c-08) |

## 6. Copy-paste checklist
- [ ] Compute arithmetic intensity BEFORE optimizing; if < machine balance, only memory tricks matter.
- [ ] Measure GB/s = bytes_moved / time and compare against device spec sheet % — target ≥85%.
- [ ] Vectorize with `float4`/`__nv_bfloat162`; keep a scalar tail path.
- [ ] Prefer fusing adjacent elementwise ops into one kernel (each fusion saves 2 passes of DRAM).

## 7. Pitfalls
- Read-modify-write on `y` means 3 streams; some "saxpy" benchmarks count 2 — know which convention you use before comparing to published STREAM numbers.
- `--use_fast_math` does nothing here; don't blame the compiler when bandwidth-bound code won't speed up.
- Small N fits in L2 (T4: 4 MB? check `p.l2CacheSize`): timings then reflect cache, not DRAM — size inputs > L2 for honest bandwidth tests (lab uses 64 MB/array).
