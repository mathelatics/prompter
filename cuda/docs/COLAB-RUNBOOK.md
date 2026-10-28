# CUDA Kernel Labs — Colab Runbook (Google Colab T4, verified recipe)

Goal: run every `cuda/kernels/labNN_*.cu` on a free Colab GPU and capture the
self-verification output. All labs are single-file, depend only on
`include/harness.h`, and exit 0 on PASS / non-zero on FAIL.

## Cell 1 — pick the GPU
Runtime → Change runtime type → **T4** (or A100 on paid). Verify:
```bash
!nvidia-smi
```
Expected: Tesla T4 (sm_75) or A100 (sm_80). Labs compile for the native arch via `-arch=native`.

## Cell 2 — clone & build everything
```python
%%bash
set -e
if [ ! -d /content/cuda-labs ]; then
  git clone https://github.com/<your-fork>/workspace.git /content/repo  # or upload zip
  cp -r /content/repo/cuda /content/cuda-labs
fi
cd /content/cuda-labs
nvcc --version | tail -1
make all                       # builds bin/lab01..labNN with nvcc -O3 --use_fast_math
echo "BUILD OK"
```
If you prefer uploading without git: in Colab file panel upload the `cuda/`
folder; paths below assume `/content/cuda-labs`.

## Cell 3 — run Phase-A labs (one per lab, keep outputs as your logbook)
```python
for lab in ["lab01_threads","lab02_saxpy","lab03_reduce","lab04_transpose","lab05_matmul"]:
    print("="*70); print(lab)
    !cd /content/cuda-labs && ./bin/{lab} || echo "FAILED $lab"
```
Each prints `GPU:` line, `[PASS] …` checks, timing table, and ends
`ALL CHECKS PASSED`. Exit code gates CI (`make test-cuda`).

## Cell 4 — optional: profile one kernel
```bash
!apt-get install -qq nsight-compute-cli >/dev/null 2>&1 || true   # often unavailable on Colab
# fallback: rely on the GB/s / GFLOP/s numbers each lab prints
```
Colab usually can't run `ncu` (no perf-counter permissions). The labs therefore
print roofline-relevant metrics themselves (GB/s, GFLOP/s, bank-conflict hints).

## Notes / known Colab quirks
- First launch of a kernel pays JIT/module-load cost (~ms): labs use warmup runs inside `bench()`.
- Free-tier GPU preemption can kill long runs (>~90 s wall): lab05's CPU oracle is ~1–3 s single-threaded double GEMM at 1024³; if it ever feels slow, shrink M/N/K constants at top of main().
- `--use_fast_math` + rel-L2 tolerances already sized so transcendental approximations pass.
- If `make all` fails with `nvcc fatal: Unsupported gpu architecture 'compute_native'`, replace `-arch=native` in Makefile NVCCFLAGS with `-arch=sm_75` (T4).

## Local no-GPU sanity check (this sandbox)
```bash
cd cuda && bash tools/syntax_check.sh   # g++ host-side typecheck vs stub headers
```
Validates syntax/types of every lab (launches stripped by sed); semantics still require real GPU run above.
