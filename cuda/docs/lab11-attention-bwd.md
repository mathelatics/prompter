# Lab 11 — Attention backward (dQ/dK/dV) via 5-kernel chain

## Problem
Given dO and cached forward probs P, compute dQ,dK,dV — the exact graph llm.c builds from
cuBLAS calls in attention_backward (CPU) / matmul chains (fp32 CUDA).

## Math
dV = PᵀdO ; dP = dO·Vᵀ ; rowdot_i = Σ_j P_ij dP_ij ;
dS = P∘(dP−rowdot)·SCALE ; dQ = dS·K ; dK = dSᵀ·Q.
(Softmax bwd identity: Σ_j P_ij dP_ij is subtracted because rows of P sum to 1.)

## Naive GPU version & measured problem
First draft computed dp inline inside two different kernels (D2 read dp before D3 produced
it — caught by review BEFORE running; fixed by materializing dP once: attn_dp → attn_rowdot
→ attn_ds). Lesson: data-flow order between kernels is part of correctness.

## Optimization journey
1. Five small kernels mirroring the cuBLAS call sequence (easy to diff against CPU oracle).
2. Upper triangle kept zero through the whole chain (lab10 invariant) so no masking branches.
3. Production: fuse all five into flash-attn bwd (one pass, recomputes P from Q,K — see lab21).

## Source map
- rough/llm.c/train_gpt2.c attention_backward; train_gpt2_fp32.cu bwd matmul chain
- docs/audit/llm-c-03 §attention-backward, llm-c-05 §matmul_backward

## Copy-paste checklist
☑ rowdot reduction ☑ dS formula with SCALE factor ☑ dK uses dSᵀ column-stride loop ☑ triple check_close(dQ,dK,dV)

## Pitfalls
- Forgetting −rowdot makes dQ/dK wrong but dV still passes — test ALL THREE grads.
- bench() re-runs the chain: pure functions, safe here (unlike atomics in lab09).
Measured: pending Colab run.
