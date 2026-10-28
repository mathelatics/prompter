# llm.c Audit 01 — Repo Overview, README & Build System

Repo: `rough/llm.c` (cloned from https://github.com/karpathy/llm.c.git; mirror ref 10605/llm.c)
HEAD at audit time: `f1e2ace` ("Merge pull request #801 from ngc92/ngc92/fix-test"), branch `master`.
License: MIT. Tagline: "Trains GPT-2 (or LLaMA 3) in the most massive, simple, portable, easily-hackable, reasonably efficient C you can possibly write." No PyTorch/Python needed for training (Python only to *create* the initial checkpoint).

## Top-level layout
```
LICENSE Makefile README.md requirements.txt
train_gpt2.c        # CPU reference trainer (GPT-2), clean/minimal, OpenMP only
train_gpt2_fp32.cu  # GPU fp32 trainer (reference on GPU)
train_gpt2.cu       # GPU bf16/fp16 + cuDNN/cutlass fused trainer (the fast one)
test_gpt2.c / test_gpt2.cu / test_gpt2_fp32.cu  # gradcheck vs PyTorch (TESTING define skips main)
profile_gpt2.cu + profile_gpt2cu.py             # Nsight compute analysis scripts
train_gpt2.py       # exports gpt2_124M.bin/.hf.bin checkpoint + gpt2_tokenizer.bin
train_llama3.py     # exports llama3_1B.bin checkpoint
llmc/               # shared header-only library (kernels .cuh, utils .h)
dev/                # experiments: dev/cuda/* standalone kernel benches, dev/cpu/matmul_forward.c, dev/data/* downloaders, dev/train_llama3.sh etc.
doc/                # notes e.g. doc/layernorm/layernorm.c
scripts/            # shell helpers (run scripts)
```

## README key claims / usage
- Quick start: `python train_gpt2.py` (creates `gpt2_124M.bin`, `gpt2_124M_debug.bin`, `gpt2_tokenizer.bin`; needs `pip install datasets transformers tqdm`), then `make train_gpt2` style or:
  - CPU: `gcc -O3 -Ofast -funsafe-math-optimizations -std=c11 -march=native -DOMP train_gpt2.c -o train_gpt2 -fopenmp -lm` (Makefile does it)
  - GPU: `nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_80:code=sm_80 ...`
- Results: single-GPU GPT-2 124M fine-tuning reaches ~2.85 val loss on TinyStories; multi-node via Zero1; LLaMA-3 1B supported in train_gpt2.cu path.
- Model checkpoints format is custom `.bin` (see file-format doc 02).

## Makefile (verified against actual file at HEAD f1e2ace)
- Default `CC ?= clang`. `CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes`; `LDLIBS = -lm`; conditional flag `-march=native` added only if compiler accepts it (`check_and_add_flag` test-compiles a trivial main).
- NVCC flags: `NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)` (FORCE_NVCC_O default 3). `NVCC_LDFLAGS = -lcublas -lcublasLt -lnvidia-ml` (nvidia-ml for GPU power/clock queries in cuda_utils).
- GPU arch auto-detect: unless `CI=true` or `GPU_COMPUTE_CAPABILITY=` given, runs `nvidia-smi --query-gpu=compute_cap` (strips dot, sorts, takes lowest) and appends `--generate-code arch=compute_XX,code=[compute_XX,sm_XX]`.
- Build objects go to `build/` dir (auto-created). Windows supported via separate `cl` flag block (`/fp:fast /openmp:llvm` etc.).
- Targets (PHONY): `all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu profile_gpt2cu`.
  - `train_gpt2`: `$(CC) $(CFLAGS) ... train_gpt2.c -o train_gpt2`
  - `test_gpt2`: same from test_gpt2.c (note: test_gpt2.c itself `#define TESTING` at top rather than via -D)
  - `train_gpt2cu`: from train_gpt2.cu + optional `build/cudnn_att.o`; uses `$(PFLAGS)` precision define
  - `train_gpt2fp32cu`: from train_gpt2_fp32.cu (no PFLAGS — always fp32)
  - `profile_gpt2cu`: adds `-lineinfo` for ncu profiling
  - `clean`: removes targets + build/*.o
- Feature autodetection (printed at make time):
  - cuDNN flash-attention: opt-in `USE_CUDNN=1` → requires cudnn-frontend headers at `$HOME/cudnn-frontend/include` or `./cudnn-frontend/include`; sets `-DENABLE_CUDNN`, `-lcudnn`, compiles llmc/cudnn_att.cpp separately (compile time blows up ~1 minute, hence off by default).
  - OpenMP: auto unless `NO_OMP=1`; macOS Homebrew libomp paths hardcoded (/opt/homebrew/opt/libomp, /usr/local/opt/libomp); Linux `-fopenmp -DOMP -lgomp`. Runtime: `OMP_NUM_THREADS=N ./train_gpt2`.
  - NCCL multi-GPU: auto unless `NO_MULTI_GPU=1`; detected via `dpkg -l | grep nccl`; sets `-DMULTI_GPU -lnccl`.
  - OpenMPI: auto unless `NO_USE_MPI=1`; looks in `/usr/lib/x86_64-linux-gnu/openmpi` (override `OPENMPI_DIR`); sets `-DUSE_MPI -lmpi`.
  - Precision: `PRECISION ?= BF16`, valid {FP32, FP16, BF16} → `-DENABLE_FP32/-DENABLE_FP16/-DENABLE_BF16`.
- If `nvcc` not found, GPU targets silently skipped.

## Data pipeline (README + dev/data)
- Supported datasets w/ downloader scripts in `dev/data/`: tinystories, fineweb, tinyshakespeare, openwebtext, llama2, smol; output = raw uint16 token `.bin` files (tokens stored as `uint16_t`, vocab < 65536 for GPT-2; dataloader casts to int).
- Tokens bin format: flat array of `uint16` token ids, no header. Dataloader maps them to inputs/targets shifted by 1.

## README details (verified digest — full text is long; these are the load-bearing facts)
- Goal statement: "simple, portable, easily-hackable, reasonably efficient C"; currently trains GPT-2 (train_gpt2.c/.cu) and LLaMA-3 1B (train_gpt2.cu path with llama3_1B.bin from train_llama3.py).
- Quick start CPU: install nothing but a C compiler; `python train_gpt2.py` once to make checkpoint; then `make train_gpt2 && ./train_gpt2`. OpenMP strongly advised (`brew install libomp` / `apt install libomp-dev`).
- GPU notes: needs CUDA toolkit; TF32 enabled by default on Ampere+ via cublas math mode; cuDNN flash attention optional (`git clone https://github.com/NVIDIA/cudnn-frontend.git` into $HOME, `USE_CUDNN=1 make`).
- Multi-GPU single-node: `torchrun --nproc_per_node=4 ./train_gpt2cu` style launch; env vars MASTER_ADDR/MASTER_PORT/RANK/LOCAL_WORLD_SIZE read under USE_MPI or direct torchrun-style env; multi-node via MPI (`mpirun`) supported.
- Dev results table: fine-tune GPT-2 124M on TinyStories to val loss 2.85 in ~30 min on one A100 (train_gpt2_fp32.cu slower reference; train_gpt2.cu bf16 fastest); CPU reference ~loss 3.29 after 40 steps demo.
- Experiments listed in README: fp32 vs bf16/fp16, fused layers, ZeRO-1 data parallelism, cudnn SDPA, manual grad accumulation, MFU measurement (~30-40% H100), etc.
- The repo intentionally keeps three tiers: (1) train_gpt2.c readable CPU reference, (2) train_gpt2_fp32.cu straightforward CUDA translation, (3) train_gpt2.cu heavily optimized (fused kernels, bf16, cublasLt, cudnn, nccl, zero1).

## Notable design decisions (README)
- Everything in C/CUDA, minimal deps: only libcublas(-Lt), optional libcudnn, optional NCCL/MPI.
- Reference implementation first (train_gpt2.c), optimized variants later (train_gpt2_fp32.cu, train_gpt2.cu).
- Uses "hand-written" kernels for layernorm/attention fwd+bwd; matmuls delegated to cuBLASLt.
- Mixed precision master weights kept in fp32, activations/grads in bf16 (or fp16 on V100 via `-DFP16`? — see zero/matmul docs).
- Multi-GPU: NCCL all-reduce + gradient accumulation; ZeRO-1 optimizer sharding (llmc/zero.cuh).
- TF32 note: `cudaSetFlags(cudaDefaultDevice)`, `-DLAUNCH_INDIVIDUALLY` debug flag exists in code.
- MFU tracking via llmc/mfu.h.
- Logging via llmc/logger.h (wandb-ish jsonl) and outlier detector.

## Cross-reference map (other audit docs)
- 02: Checkpoint/tokenizer file formats + train_gpt2.py export details
- 03: train_gpt2.c CPU reference walkthrough (every function)
- 04: llmc/ shared headers (dataloader, tokenizer, rand, sampler, schedulers, utils, logger, mfu, outlier)
- 05: train_gpt2_fp32.cu GPU fp32 trainer
- 06: llmc/*.cuh kernel library (fp32 kernels: layernorm, attention, gelu, softmax, matmul, adamw, global_norm, fused_classifier)
- 07: train_gpt2.cu fast bf16 trainer + main loop state machine
- 08: llmc bf16/fp16 kernels + cublas wrapper + cudnn att + cutlass-free fused ops
- 09: llmc/zero.cuh ZeRO-1 + nccl_all_reduce + cuda_utils.cuh
- 10: dev/ directory (kernel benches, data prep, llama3 path) + tests
- 11: gpt.c repo full audit (separate file series)
