# Lab 16 — `kernels/lab16_rope.cu`

**Tier:** 3 · **Status:** host-typechecked ✓ · **Run:** `make bin/lab16_rope && ./bin/...` (needs a GPU)

## What it computes
See the file header for the full math derivation and source map into llm.c/gpt.c.
Short version:
- lab13 fused residual-add + LayerNorm (one HBM pass, warp-per-row, C=768)
- lab14 dual matmul: dgrad dX=dY·W and wgrad dW=dYᵀ·X in ONE launch, shared dY tile
- lab15 AdamW fused elementwise step, decoupled weight decay, bias correction
- lab16 RoPE forward+backward (llama3 base=500k, head_dim=128), rotation-inverse property check
- lab17 SwiGLU fwd+bwd elementwise gate (y = swish(g)·s)
- lab18 __ldcs/__stcs streaming-hint copy vs plain float4 copy ladder

## Verification design
Every lab self-verifies on run: GPU result vs double-precision CPU oracle with a
stated tolerance (`check_close`), plus `cudaGetLastError()` after sync. Exit code
non-zero on any FAIL so `make test-cuda` gates CI.

## Host typecheck
`bash tools/syntax_check.sh` passes for all 18 labs (compiles as C++ against
`tools/fakecuda/cuda_runtime.h`; shims added this round: `__constant__`,
`cudaMemcpyToSymbol`).
