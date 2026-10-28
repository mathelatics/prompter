# Lab 10 — Causal attention forward as a composable kernel chain

## Problem
O = softmax_causal(QKᵀ/√C)·V for B=4,H=12,T=C=64 (GPT-2 head shape scaled down).

## Math
Per (b,h): S_ij=(q_i·k_j)/√C for j≤i; row-softmax over the causal prefix; O_i=Σ_j P_ij v_j.

## Naive GPU version & measured problem
gpt.c's single fused attention_kernel: one thread per (b,h,i,c), inner loop re-does the
whole softmax per output dim ⇒ C× redundant work. Our chain splits it: scores → softmax → PV.

## Optimization journey
1. attn_scores: thread per (b,h,i,j); skip upper triangle (write 0, never read).
2. attn_softmax: warp-per-row over causal prefix only (j≤i) — lab07 machinery reused.
3. attn_pv: thread per (b,h,i,c). Chain is what llm.c fp32 does with cuBLAS strided-batched
   + kernel5 softmax (docs/audit/llm-c-05). Fusion into flash-attention = lab21.

## Source map
- rough/llm.c/train_gpt2_fp32.cu: cublas strided-batched matmul_forward ×2 + kernel5_time_maps_softmax
- rough/llm.c/train_gpt2.c attention_forward (4-pass CPU ref, docs/audit/llm-c-03 §attention)
- rough/gpt.c attention_kernel (fused naive contrast)

## Copy-paste checklist
☑ SCALE=1/√C applied at scores not after softmax ☑ causal prefix loops ☑ -1e30-safe max trick

## Pitfalls
- Softmax rows of length 1 (i=0) must give exactly 1.0 — covered by oracle.
- Masked entries: write zeros so later kernels can read full rows without branches (lab11 relies on this).
Measured: pending Colab run.
