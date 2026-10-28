# GPU Verification Status — honest ledger

Last updated: 2026-10-28 (this turn)

## What has been executed where
| Check | Environment | Result | Evidence |
|---|---|---|---|
| Host typecheck (`bash tools/syntax_check.sh`) | this sandbox, g++ `-fsyntax-only -Wall -Wextra` + fakecuda shims | **18/18 [ok], rc=0** | /tmp/sc.log, committed runs |
| Local nvcc build+run | this sandbox | **IMPOSSIBLE — no GPU, no nvcc** (`which nvcc` empty, `nvidia-smi` not found) | verified by execution this turn |
| LeetGPU API submission | attempted | **NOT AVAILABLE** — probed `api.leetgpu.com/run`, `/api/run`, `/v1/run` → HTTP 404; `pip install leetgpu` → no distribution; site is a browser challenge platform with no public code-exec API we could verify | curl output this turn; documented in `tools/leetgpu_run.sh` header |
| Real-GPU numeric PASS of labs 01–18 | — | **NOT YET DONE** — no claim made until logs land in `results/` | open |

## The two working paths to GPU verification (pick one)
### A) Colab (recommended, free T4/L4)
1. Download `cuda/colab_labs.zip` (36 KB, flat: 18 `.cu` + `harness.h`).
2. Open a CUDA-enabled notebook, follow `cuda/colab/run_all_cells.md` (3 cells).
3. Save the printed `gpu_results.txt` and commit it to `cuda/results/gpu_results.txt`.
   That file — not this doc — becomes the proof-of-PASS. Then measured timings
   get folded into each `docs/labNN_*.md` §4 "Measured:" line.

### B) SSH remote GPU (any box with nvcc)
```bash
export LEETGPU_SSH=user@your-gpu-host      # e.g. vast.ai / runpod / home machine
bash cuda/tools/leetgpu_run.sh             # all 18 labs, or name specific ones
```
The script scp's each lab + harness.h, builds with `nvcc -O3 --use_fast_math`,
runs under `timeout 600`, judges stdout for PASS/FAIL per lab, writes
`results/labNN.gpu.log` locally. Exit non-zero if any lab fails.

## Acceptance criteria before labs 19+ are written
- 18/18 labs: BUILD ok on real nvcc AND oracle check PASS (rel-L2 < 1e-5 fp32).
- Timing tables from lab02/03/04/05 captured (they gate the docs' claims about
  bandwidth/conflict savings).
- Only then Tier-2 production kernels (19 flash-attn mma.sync, 20 NCCL, …) proceed,
  because they depend on Tier-1 primitives being numerically proven.

## Rule (anti-cheat)
This file may only be marked ✅ when `results/*.gpu.log` files exist in git with
verifiable timestamps. No status upgrade without logs.
