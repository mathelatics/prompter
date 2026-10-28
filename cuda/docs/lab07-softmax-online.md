# Lab 07 — Online softmax (the FlashAttention trick in miniature)

## Problem
Row-wise softmax over up-to-context-length scores; must never overflow exp() and ideally
read the row ONCE (attention tiles can't be materialized twice at large T).

## Math
Naive stable: m=max(x); p=exp(x−m); s=Σp; y=p/s (2 passes).
Online invariant after processing prefix: l ≈ Σ_{prefix} exp(x−m). New chunk max mc>m:
l ← l·exp(m−mc) + Σexp(chunk−mc); m←mc. Proves FlashAttention's rescaling.

## Naive GPU version & measured problem
Two-pass warp-per-row: correct but reads x twice; at T=1024 that's 2× bandwidth on the
hottest inner loop of attention.

## Optimization journey
1. twopass baseline (correctness anchor).
2. online single-streaming-pass variant; write pass recomputes exp (or keep chunk probs in regs).
3. In flash-attention you fuse this INTO the matmul loop (lab21) so x is never stored at all.

## Source map
- rough/llm.c/train_gpt2_fp32.cu kernel5_time_maps_softmax (per-time-step online over j)
- docs/audit/llm-c-05 §kernel5; gpt.c softmax_kernel = two-pass naive for comparison

## Copy-paste checklist
☑ running (m,l) with rescale factor exp(m_old−m_new) ☑ -1e30 fill for masked lanes ☑ NaN check test

## Pitfalls
- Rescale l, NOT the individual e values, when max grows mid-chunk.
- Chunk max must be warp-reduced before computing mn=fmax(m,mc) or lanes diverge.
Measured: pending Colab run.
