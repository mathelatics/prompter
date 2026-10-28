# Lab 08 — GELU (tanh approx) fwd/bwd + the float-control story

## Problem
MLP activation in GPT-2. Must match PyTorch's F.gelu(approximate='tanh') to load/verify ckpts.

## Math
g(x)=½x(1+tanh(u)), u=√(2/π)(x+0.044715x³).
g'(x)=½(1+t)+½x(1−t²)·u′, u′=√(2/π)(1+3·0.044715x²).

## Naive GPU version & why it matters
Plain fp32 expression: compiler contracts a*b+c into FMA → different rounding than a CPU
reference that computes mul then add separately. llm.c pins this on the CPU side with
`#pragma STDC FP_CONTRACT OFF` + fenv float-control tricks (docs/audit/llm-c-03 §gelu).
GPU analogue: __fmul_rn/__fadd_rn intrinsics force per-op rounding (gelu_f_nc variant).

## Optimization journey
1. fwd kernel vs double oracle (tol 1e-5) — elementwise, bandwidth-bound (~8 B/elem).
2. bwd kernel vs analytic double oracle.
3. FINITE-DIFFERENTIAL gradient test (sampled 1000 points): catches derivative algebra bugs
   that "oracle matches kernel because both were written wrong the same way" would miss.

## Source map
- rough/llm.c/train_gpt2.c gelu_forward/gelu_backward; train_gpt2_fp32.cu kernel4_map_gelu_not_res
- gpt.c gelu_kernel (single fused fwd)

## Copy-paste checklist
☑ SQRT_2_PI/C0 constants ☑ non-contracted variant for bit-matching ☑ finite-diff grad check

## Pitfalls
- tanhf is MUFU-backed (~2 ulp); don't demand bit-exactness vs CPU tanh — use tolerance.
- x³ overflow guard not needed for activations but IS if reused elsewhere.
Measured: pending Colab run.
