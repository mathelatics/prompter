# Lab 12 — Fused classifier (matmul + cross-entropy), in-place logits trick

## Problem
GPT-2 head: logits = x·Wᵗᵉ (weight-tied!), loss = CE(logits, targets). llm.c fuses the
softmax+log into one reduction pass and even OVERWRITES the logits buffer to save memory.

## Math
loss_i = −log softmax(logits_i)[t_i] = −(logits[i,t] − logsumexp_j logits[i,j]).
Grad wrt logits: p − onehot(t) (scaled by 1/BT in llm.c).

## Optimization journey
1. Compute row logsumexp with online max/sum (lab07 machinery).
2. Write grad in place over the logits array (llm.c's kernel6 does exactly this — memory win).
3. Weight tying: same scatter as lab09 adds d_logits rows into d_wte.

## Source map
- rough/llm.c/train_gpt2_fp32.cu kernel6_time_maps_cross_entropy; train_gpt2.c fused_classifier
- docs/audit/llm-c-05 §kernel6

## Copy-paste checklist
☑ logsumexp stable form ☑ in-place overwrite documented ☑ p−onehot grad

## Pitfalls
- Label smoothing off matches PyTorch default; add flag if needed.
Measured: pending Colab run. (Implementation file lab12_fused_classifier.cu next session.)
