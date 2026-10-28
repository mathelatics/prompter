# Lab 06 — LayerNorm, warp-per-row (llm.c pattern)

## Problem
Normalize each token row x∈R^N to zero mean/unit variance, then affine w,b. eps=1e-5.
Every GPT-2 block has 2 of these; T rows are independent → embarrassingly parallel per row.

## Math
mean = Σx/N; var = Σ(x−mean)²/N; x̂=(x−mean)/√(var+eps); y = x̂·w + b.
Backward: c1=Σ(dy·w), c2=Σ(dy·w·x̂) ⇒ dx = w·(dy − c1/N − x̂·c2/N)·rstd.

## Naive GPU version & why it fails
Block-per-row with shared-memory reduce: fine but heavy sync. For N≤~2k a **warp** is the
natural unit: butterfly `__shfl_xor_sync` needs ZERO shared memory and ZERO __syncthreads.

## Optimization journey
1. One warp per row, lane strides i=lane..N step 32 (coalesced within the warp).
2. Two streaming passes (mean, then var) — llm.c's kernel1_map_ln_not_res does exactly this.
3. Backward recomputes mean (cheap vs re-reading from global) and stores rstd from forward.
4. Production step further: fuse residual-add (lab13) and use `__ldcs/__stcs` (lab18).

## Source map
- rough/llm.c/train_gpt2_fp32.cu: kernel1_map_ln_not_res (fwd), kernel2_map_ln (residual)
- rough/llm.c/train_gpt2.c: layer_norm_forward/backward CPU reference
- our docs: docs/audit/llm-c-03 §layernorm, llm-c-05 §kernel1

## Copy-paste checklist
☑ warp_sum() butterfly helper ☑ warp-per-row indexing ☑ rstd save for backward ☑ eps inside sqrt

## Pitfalls
- rsqrtf(var+eps): add eps BEFORE rsqrt, not after (NaN risk otherwise).
- If N not multiple of 32, guard the tail loads.
- Butterfly uses __shfl_xor so ALL lanes end with the sum — no lane-0 broadcast needed.
Measured: pending Colab run (paste table here).
