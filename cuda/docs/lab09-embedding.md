# Lab 09 — Embedding gather/scatter (+ weight tying note)

## Problem
Fwd: x[t] = wte[ids[t]] + wpe[t]. Bwd: gradients scatter INTO table rows; many tokens share
an id ⇒ concurrent writes to the same row.

## Math
Gather is indexed copy (random read, coalesced write). Scatter-add is Σ over tokens of dx
into d_wte rows — associative but order-dependent in fp32 ⇒ atomics.

## Naive GPU version & measured problem
One thread per (token,dim) doing atomicAdd works but serializes hot rows (BPE 'the' appears
~10% of positions). Measured fix path: sort-by-id segmented reduce (production), or accept
atomics like llm.c does.

## Optimization journey
1. warp-per-token gather (coalesced 32-float lanes across the hidden dim).
2. atomicAdd scatter; verify with rel_l2 vs double oracle (NOT bit-exact — see lab03 determinism demo).
3. Weight tying (llm.c): d_wte accumulates from BOTH the LM head and the encoder input; two
   scatter launches into the same buffer is the correct pattern (docs/audit/llm-c-03 §tying).

## Source map
- rough/llm.c/train_gpt2.c encoder_forward/encoder_backward; train_gpt2_fp32.cu kernel6_time_maps_cross_entropy ties grads
- gpt.c embed_kernel

## Copy-paste checklist
☑ ids as int32 ☑ position modulo max_seq ☑ zero-init d_wte before scatter ☑ rel_l2 check

## Pitfalls
- bench() RE-RUNS kernels — atomics accumulate! Re-run once clean before verifying (done here).
- Don't compare atomic results bit-exactly across runs/orders.
Measured: pending Colab run.
