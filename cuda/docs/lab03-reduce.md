# Lab 03 — Parallel Reduction: the 6-Stage Evolution (+ Determinism)

**Tier 1** · Source analogues: llm.c `layernorm_forward` warp reduction (`__shfl_xor` butterfly, docs/audit/llm-c-05), `softmax_backward_data` atomics, gpt.c sum kernels.
**Runnable:** `cuda/kernels/lab03_reduce.cu`

## 1. Problem
Sum N floats (here 2²⁴). Reduction is the template for every "aggregate across threads" op in transformers: layernorm mean/var, softmax sums, gradient accumulation, loss computation. Every CUDA optimization concept has its smallest non-trivial example here.

## 2. Math on paper
Sequential sum = N adds, latency O(N). Tree reduction over P processors = O(N/P + log P): each thread locally sums N/P elements (the *sequential phase*), then log P steps combine partials. Key insight: **log₂(32)=5 shuffle steps per warp are nearly free**, so you want each thread to do many local adds before crossing threads.

Floating-point caveat: addition isn't associative. Different trees ⇒ different roundings ⇒ bit-different results vs CPU. Verify with tolerance (rel-L2 < 1e-4 typical), never `==`.

## 3–4. Naive → optimized journey (all five variants live in the lab)
| Stage | Change | What it teaches |
|---|---|---|
| v1 | shared-memory tree, `if (tid<stride)` guard every level | correctness; later levels leave whole warps idle (divergence waste) |
| v2 | unroll last warp after `stride>32` loop | removes 5 of 8 `__syncthreads()`; interleaved addressing sets up conflict-free reads |
| v3 | grid-stride **pre-add**: each thread serially accumulates many elements into one register, block has only TPB/32 slots | ILP beats synchronization; fewer blocks to merge |
| v4 | `__shfl_down_sync` butterfly inside warps, shared only between warps | no shared mem / no syncthreads for intra-warp step — this is exactly llm.c's layernorm pattern |
| v5 | final merge via `atomicAdd` into one scalar | single-launch convenience — and the determinism trap (below) |

Lab prints a timing table for all variants on your GPU; typical T4 result: v1→v4 ≈ 2–3× faster, approaching DRAM roofline (reduction is bandwidth-bound: 4 B/elem read).

## 5. Source map
| Concept | Where |
|---|---|
| warp butterfly reduce (`__shfl_xor`) | `rough/llm.c/train_gpt2_fp32_cu.cu` layernorm kernel; `llmc/kernels/elementwise.cuh` bf16 version (docs/audit/llm-c-05, -08) |
| atomic grad accumulation & its nondeterminism | `llm.c/train_gpt2.cu` backward scatter sections; discussed in docs/audit/llm-c-03 |
| two-pass softmax sums | gpt.c `softmax_*`; online variant → Lab 07 |

## 6. Copy-paste checklist
- [ ] Steal `warp_reduce_sum` verbatim (add `_prod/_max` twins by swapping `+`/`*`/`fmaxf`).
- [ ] Block reduce = warp-reduce + 1 small shared array of size TPB/32.
- [ ] Grid reduce options: (a) second tiny kernel over partials, (b) `atomicAdd`, (c) cooperative-groups grid sync. For training grads use (a) if you need reproducibility.
- [ ] Always verify against double-precision CPU oracle with rel-tolerance.

## 7. Pitfalls
- **Determinism demo in the lab runs v5 twice and byte-compares**: identical bits are luck, not guarantee. PyTorch refuses default atomics in reproducible mode for the same reason. If your CI diffs losses run-to-run, atomics are suspect #1.
- `volatile` in v2 is legacy (pre-Volta); modern code uses `__syncwarp()` or the `_sync` shuffle intrinsics instead.
- Summing 16M values in fp32 loses ~log₂√N bits: error grows like √N·ε. Use Kahan or double accumulators when precision matters (loss scaling!).
- Template the whole thing once for sum/max/min/argmax — they share 95% of structure.
