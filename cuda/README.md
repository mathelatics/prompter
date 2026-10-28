# CUDA Kernel Labs

Learn-by-running CUDA curriculum distilled from `rough/llm.c` + `rough/gpt.c`.
Each lab = one self-verifying binary (CPU double oracle + PASS/FAIL + speedup
table). Docs in `docs/` follow the 7-section first-principles template.

- Start here: `docs/COLAB-RUNBOOK.md` (how to run everything on a free Colab T4)
- No GPU? `bash tools/syntax_check.sh` host-typechecks every lab via stub headers.
- Full curriculum plan: `../docs/audit/cuda-kernels-00-plan.md`

## Status
| Lab | Tier | Topic | Doc | Verified |
|---|---|---|---|---|
| 01 threads | 1 | indices, bounds, grid-stride | docs/lab01-threads.md | syntax-check ok; GPU run pending (Colab) |
| 02 saxpy | 1 | bandwidth roofline, float4 | docs/lab02-saxpy.md | syntax-check ok; GPU run pending |
| 03 reduce | 1 | reduction v1–v5 + determinism | docs/lab03-reduce.md | syntax-check ok; GPU run pending |
| 04 transpose | 1 | shared mem, bank conflicts | docs/lab04-transpose.md | syntax-check ok; GPU run pending |
| 05 matmul | 1 | tiled GEMM v0→v2 | docs/lab05-matmul.md | syntax-check ok; GPU run pending |
| 06 | layernorm warp fwd+bwd | ✅ syntax-checked | ⏳ Colab |
| 07 | online softmax | ✅ syntax-checked | ⏳ Colab |
| 08 | gelu fwd/bwd + finite-diff grad check | ✅ syntax-checked | ⏳ Colab |
| 09 | embedding gather/scatter | ✅ syntax-checked | ⏳ Colab |
| 10 | causal attention fwd chain | ✅ syntax-checked | ⏳ Colab |
| 11 | attention bwd dQ/dK/dV | ✅ syntax-checked | ⏳ Colab |
| 06+ | 2–6 | softmax/LN/GELU/fusions/bf16/NCCL/applied | planned per master plan | — |
