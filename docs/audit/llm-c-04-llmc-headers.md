# llm.c Audit 04 — Shared C/C++ Headers in `llmc/` (non-.cuh)

Files audited: `llmc/dataloader.h` (519 L), `llmc/tokenizer.h` (105 L), `llmc/rand.h` (239 L),
`llmc/utils.h` (222 L), `llmc/sampler.h` (40 L), `llmc/schedulers.h` (100 L), `llmc/logger.h` (57 L),
`llmc/outlier_detector.h` (70 L), `llmc/mfu.h` (244 L). All line references against HEAD `f1e2ace`.

These headers are shared by `train_gpt2.cu` (and the fp32 `.cu` trainer uses several too).
The CPU reference trainer `train_gpt2.c` has its own inline copies instead (see doc 03).

---

## 1. `llmc/dataloader.h` — DataLoader + EvalLoader

Header comment: "DataLoader for model training. Reads and serves data shards. EvalLoader for
multiple-choice evaluation datasets, e.g. HellaSwag." Uses POSIX `glob()` (Windows impl in `dev/unistd.h`).

### 1.1 Constants
- `#define HEADER_SIZE 256` → header is 256 × int32 = **1024 bytes** on disk.
- `#define ASSUMED_NUM_COMPLETIONS 4` (hard-coded; eval loader asserts num_completions == 4).
- `#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))`

### 1.2 `DataLoader` struct fields
```
process_rank, num_processes            // distributed sharding of each batch across ranks
size_t B, T, num_tokens                // num_tokens = total tokens across ALL shards
size_t shard_num_samples               // samples per current shard PER PROCESS
glob_t glob_result                     // matched shard files
size_t current_shard_idx, current_sample_idx
FILE* tokens_file                      // only one shard open at a time
uint16_t* buffer                       // fread staging buffer, B*T+1 tokens
int* inputs, *targets                  // B*T each, cast from uint16
mt19937_state shuffle_rng; int should_shuffle
int* shard_indices, *intra_shard_indices
size_t total_batch_size_bytes          // num_processes*B*T*sizeof(uint16)
size_t local_batch_offset_bytes        // process_rank*B*T*sizeof(uint16)
size_t header_bytes (=1024); int64_t file_size_bytes
```

### 1.3 Shard file format validation (`dataloader_load_shard_`)
- If shuffling: `shard_index = loader->shard_indices[shard_index]` (indirection).
- Read 256-int header. `header[0]` must be magic **20240520**, `header[1]` version **1**.
  Bad magic prints hint about re-running data prepro (May 20 2024 move to dev/data).
- `ntok = header[2]`; assert ntok > 0.
- File size check: `expected = 1024 + ntok*2` bytes must equal actual ftell.
- `shard_num_samples = (ntok*2 - 2) / total_batch_size_bytes`
  (the `-2`: batches read B*T+1 tokens but advance by B*T).
- Returns ntok.

### 1.4 Epoch/shuffle mechanics
- `prepare_intra_shard_indices_`: reallocs `intra_shard_indices` to shard_num_samples, identity-init then `random_permutation(..., &shuffle_rng)` (Fisher-Yates via mt19937). Called per shard load when shuffling.
- `dataloader_reset`: shard_idx=0, sample_idx=0, permutes shard_indices (whole epoch order), loads shard 0, prepares intra-shard permutation.
- `dataloader_advance_`: if at LAST shard → full reset (new epoch, reshuffles everything); else advance `(idx+1) % gl_pathc`, sample_idx=0, load shard (+ new intra permutation).
- `dataloader_init(pattern, B, T, process_rank, num_processes, should_shuffle)`:
  - glob pattern; error if 0 matches.
  - if shuffle: seed mt19937 with **`42 + process_rank`**; init identity shard_indices.
  - **Validates EVERY shard** upfront (loads each, asserts `shard_ntok >= num_processes*B*T + 1`).
  - Allocates buffer (B*T+1 uint16), inputs/targets (B*T int). `num_tokens` = sum over shards.
  - Calls `dataloader_reset`.
- `dataloader_load_batch`:
  - `idx = should_shuffle ? intra_shard_indices[current_sample_idx] : current_sample_idx`
  - `offset = header_bytes + idx*total_batch_size_bytes + local_batch_offset_bytes`
    → i.e., the global stream is cut into (num_processes × B×T) token slabs; this rank reads its slab.
  - fseek + fread **B*T+1** uint16 tokens; `inputs[i]=buffer[i]`, `targets[i]=buffer[i+1]` (next-token prediction).
- `dataloader_next_batch`: if `current_sample_idx >= shard_num_samples` → `dataloader_advance_`; then load_batch; `current_sample_idx += 1`.
- `dataloader_resume(loader, current_shard_idx, current_sample_idx)`: used by `-y 1` resume; sets indices and reloads that shard (NOTE: does not restore RNG state of shuffle_rng — epoch order continues from existing permutations set up at init).
- `dataloader_free`: frees buffers, closes file, globfree.

### 1.5 EvalLoader (HellaSwag-style multiple choice)
Eval file stream format (copied verbatim from data_common.py in header comment):
```
256-int32 header, then examples as uint16 stream:
<START_EXAMPLE>=65535, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>, <LABEL>, <NUM_COMPLETIONS>,
<CONTEXT_LEN><context tokens>, then per completion <COMP_LEN><comp tokens>
```
Struct fields: process_rank/num_processes, B, T, `FILE* eval_file`, `uint16_t* buffer`,
public: `num_examples`, `num_batches`, `start/end/current_example_index`, `int* inputs,*targets`,
`char* mask`, `int* label`, `int num_completions`.

- `evalloader_init(filename,B,T,rank,nproc)`: header magic **20240522**, version 1,
  `num_examples=header[2]` (assert ≥ num_processes), `longest_example_bytes=header[3]`
  (sanity assert: `0 < longest < (1+4)*T*2`). Allocates buffer(longest_example_bytes),
  calloc'd inputs/targets (B*T ints), mask (B*T chars), label (B/4 ints). Then reset.
- `evalloader_reset`: work partitioning: `examples_per_process = CEIL_DIV(num_examples, num_processes)`;
  `can_fit_examples = B/4` (errors if 0 → hint "-h 0 or increase -b");
  `num_batches = CEIL_DIV(examples_per_process, can_fit_examples)`;
  `start = examples_per_process*rank`, `end = min(examples_per_process*(rank+1), num_examples)`;
  fast-skips to start example using stored `<EXAMPLE_BYTES>` (reads 3 uint16 per example: START, BYTES, INDEX; asserts START==65535 and INDEX==i; skips BYTES-6 more bytes).
- `evalloader_next_example_(loader, example_batch_index)`: reads 3-uint16 header (validates START, INDEX within [start,end)), reads remaining bytes into buffer.
  - `label = buffer[0]` (assert 0≤label<4) → `loader->label[example_batch_index]`.
  - `num_completions = buffer[1]` (assert ==4, fits batch row range).
  - `context_length = buffer[2]`, tokens at buffer[3..]. Assert 0<len<T. Copies same context into all 4 rows: `inputs[(batch_dim_offset+b)*T + i]`.
  - Completions iterate pointer `loader->buffer + 3 + context_length`; per completion: len=`iter[0]`, tokens at iter+1; assert len>0 and ctx+len<T.
    - `inputs[coff*T + context_length + i] = tok`
    - `targets[coff*T + context_length + i - 1] = tok` (last context token predicts first comp token)
    - `mask[coff*T + context_length + i - 1] = 1` ← loss averaged ONLY over these positions.
- `evalloader_next_batch`: memset mask=0; loop `can_fit_examples` (B/4) packing examples into batch rows of 4; break when current_example_index >= end (rest of batch stays zero/padded).
- `evalloader_stat_losses(loader, losses /*B*T float*/) -> int correct`:
  For each example slot i, each of 4 completions b: average_loss = mean of losses where mask==1; pick argmin; if any masked position existed (active) and argmin == label → correct++.
- `evalloader_free`: frees all + fclose.

---

## 2. `llmc/tokenizer.h` — GPT-2 decoder-only tokenizer

- Decoding ONLY (tokens→strings); no encoder (regex BPE noted as tricky in C).
- `Tokenizer {uint32_t vocab_size; char **token_table; int init_ok; int eot_token;}`
- `safe_printf(piece)`: skips NULL/empty; for single-byte tokens skips non-printable/non-space bytes.
- `tokenizer_init(filename)`:
  - Missing file → warning ("new feature added April 14 2024. Re-run python train_gpt2.py"), `init_ok=0`, no crash.
  - Header: 256 × **uint32** (note: NOT int32 like other formats). magic `header[0]==20240328`.
  - `version=header[1]`, `vocab_size=header[2]`.
  - v1: no EOT stored → assume `eot_token=50256` (asserts vocab_size==50257 defensively).
  - v2: `eot_token = header[3]`. Other versions → exit.
  - Tokens: for i in [0,vocab): read 1 unsigned char length (assert >0), malloc length+1, fread bytes, null-terminate.
- `tokenizer_decode(token_id)`: NULL if !init_ok; prints "invalid token id" if ≥ vocab_size.
- `tokenizer_free`: frees each string + table.

---

## 3. `llmc/rand.h` — mt19937 numerically identical to PyTorch

Purpose: bit-exact reproduction of `torch.manual_seed` streams so weight init matches train_gpt2.py exactly.
Header docstring gives full cross-check example: seed 137 → randint32 sequence
`4053805790, 2173880614, 380293709, 1237255315, 2986595568`, then normal_() of 8 zeros →
`0.7947664260864258, 1.4369317293167114, -0.2292192131280899, ...` etc. (documented values match torch).

Constants: `MERSENNE_STATE_M 397`, `MERSENNE_STATE_N 624`, `LMASK 0x7fffffff`, `UMASK 0x80000000`.

State struct: `{unsigned long long seed_; int left_; unsigned int next_; unsigned int state_[624]; unsigned int MATRIX_A[2];}`

- `manual_seed(state, seed)`: MATRIX_A={0, 0x9908b0df}; `state_[0]=seed&0xffffffff`;
  recurrence `state_[j] = (1812433253*(state_[j-1] ^ (state_[j-1]>>30)) + j) & 0xffffffff`; left_=1, next_=0.
- `next_state`: standard MT twist split in two loops (j<624-397 uses +M; rest uses +(M-N)); last element uses state_[0] for low bits.
- `randint32`: auto-reseeds to 5489 if MATRIX_A corrupted; decrements left_, twists when exhausted; tempering y ^= y>>11; y ^= (y<<7)&0x9d2c5680; y ^= (y<<15)&0xefc60000; y ^= y>>18.
- `randint64` = two randint32 concatenated. `randfloat32` = low 24 bits × 2^-24. `randfloat64` = low 53 bits × 2^-53.
- `uniform_(data,numel,from,to,state)`: u*(to-from)+from.
- **Normal generation mirrors torch's two paths:**
  - `numel >= 16` → `normal_fill`: fill all numel with randfloat32, Box-Muller in blocks of 16
    (`normal_fill_16`: pairs t∈[0,8): u1=1-data[t], u2=data[t+8], radius=sqrt(-2*log(u1+1e-12)), theta=2π·u2; outputs cos@t, sin@t+8 scaled). If numel%16≠0: REGENERATE uniform values for the LAST 16 elements (overwriting already-consumed ones!) and transform again — exact torch quirk.
  - `numel < 16` → `normal_` inline path uses **float64** draws: u1=(float)randfloat64, u2=(float)randfloat64; radius=sqrtf(-2*logf(1-u2+1e-12)); theta=2π·u1; emits cos now, saves sin for next draw (alternating).
- `init_identity_permutation(data,numel)`, `random_permutation(data,numel,state)`: Fisher–Yates descending i, `j = randint32 % (i+1)`, swap.

---

## 4. `llmc/utils.h` — checked I/O macros & misc

All functions are `extern inline` (C99/gnu inline, single TU definition expected).

| Macro | Behavior on failure |
|---|---|
| `fopenCheck(path,mode)` | prints path/mode/file/line + HINT1 (data moved to dev/data May 20 2024) + HINT2 (re-run python train_gpt2.py); exit |
| `freadCheck(ptr,size,nmemb,stream)` | distinguishes feof / ferror / partial read; prints expected vs read; exit |
| `fcloseCheck(fp)` | exit on nonzero |
| `scloseCheck(sockfd)` | close() wrapper for sockets |
| `closesocketCheck` | Windows-only variant |
| `fseekCheck(fp,off,whence)` | prints offset/whence; exit |
| `fwriteCheck(ptr,size,nmemb,stream)` | same as freadCheck for writes |
| `mallocCheck(size)` | prints size in bytes; exit |
| `tokenCheck(tokens,count,vocab)` | asserts every token in [0,vocab); prints offending token/position/vocab; exit |

Misc helpers:
- `create_dir_if_not_exists(dir)`: mkdir mode **0700**, prints "created directory".
- `find_max_step(output_log_dir)`: scans dir for files named `DONE_<step>`, returns max step (or -1). Used for resume.
- `ends_with_bin(str)`: suffix check ".bin".
- Includes `<arpa/inet.h>` (non-Windows) — used by other code (utils for socket-based logging? actually referenced by train_gpt2.cu's W&B-less logger... note: it's here for consumers).

---

## 5. `llmc/sampler.h` — inference-time sampler (CPU reference style)

- xorshift64\*: `random_u32(state)`: s^=s>>12; s^=s<<25; s^=s>>27; return (s*0x2545F4914F6CDD1D)>>32.
  `random_f32`: (u32>>8)/2^24 ∈ [0,1). Identical to the one embedded in train_gpt2.c (doc 03 §Sampler).
- `sample_softmax(logits, n, coin)`: norm=Σexp; coin*=norm; walk CDF of expf(logits[i]); return first i with coin<cdf; fallback n-1. No temperature/top-k here (callers scale logits beforehand).

---

## 6. `llmc/schedulers.h` — LR schedulers

`LearningRateScheduler {const char* type; float learning_rate; int warmup_iterations; int train_num_batches; float final_learning_rate_frac;}` + `lr_scheduler_init(...)`.

All warmup formulas identical: `lr * (step+1)/warmup` for step<warmup (first step gets lr/warmup, never 0).

- **cosine**: after warmup, `decay_ratio=(step-warmup)/(train_num_batches-warmup)`; `coeff=0.5*(1+cos(π·ratio))`; `lr=min_lr+coeff*(max_lr-min_lr)`, min_lr = lr*final_frac.
- **linear**: `lr = max_lr - decay_ratio*(max_lr-min_lr)`.
- **constant**: ignores step.
- **wsd** (WSD paper arXiv:2405.18392): warmup linear; constant until `decay_point = 0.8*train_num_batches`; then `min_lr + (1-sqrt(ratio))*(max_lr-min_lr)` (ratio over last 20%). NOTE: returns directly inside branch.
- Dispatcher `get_learning_rate(scheduler, step)` strcmp's type; unknown → exit.

---

## 7. `llmc/logger.h` — plain-text run logger

- `Logger {int active; char output_log_file[512];}` — stateless appends; only rank 0 logs.
- `logger_init(log_dir, process_rank, resume)`: file = `<log_dir>/main.log`; if resume==0 truncates it (open "w", close).
- Formats (append, reopen per line):
  - train: `s:%d trl:%.4f lr:%.6f norm:%.2f\n`
  - val:   `s:%d tel:%.4f\n`
  - eval:  `s:%d eval:%.4f\n`
- Matches the `DONE_<step>` convention used by find_max_step (trainer writes DONE files separately).

---

## 8. `llmc/outlier_detector.h` — rolling z-score monitor

- Compile-time window `OUTLIER_DETECTOR_WINDOW_SIZE 128`; struct holds fixed `double buffer[128]`, count, ring index, running `sum`, `sum_sq` (O(1) updates; doubles to avoid drift).
- `init_detector`: zero everything.
- `update_detector(detector, new_value) -> double z`:
  - While filling (<128 seen): push, update sums, return `nan("")` (not enough data).
  - Once full: pop oldest (subtract from sums), overwrite at ring index, push new, advance index mod 128;
    `mean=sum/128; variance=sum_sq/128 - mean²; std=sqrt(variance); z=(new-mean)/std`; std==0 → return 0.
- Used in train_gpt2.cu: skip optimizer step if grad-norm z-score exceeds threshold (loss/grad spike guard).

---

## 9. `llmc/mfu.h` — promised-FLOPS DB + NVML GPU telemetry

- Optional NVML: `#if __has_include(<nvml.h>) → USE_NVML 1`; `nvmlCheck` macro exits on error.
- Precision ids: `MFUH_PRECISION_FP32 0 / FP16 1 / BF16 2` (comment: tied to enum PrecisionMode, unify later).
- `PerfData {TF_32, BF_16_32, FP_16_32, FP_16_16, FP_8_32, FP_8_16, CLOCK, CORES}` (TFLOPS @ spec clock; -1 = unsupported).
- Arch baselines:
  - VOLTA: TF32=125, FP16=125, 1530MHz, 640 TCs
  - AMPERE_DATACENTER: 156/312/312/312, 1410MHz, 432
  - AMPERE_CONSUMER: 40/80/80/160, 1860MHz, 336
  - HOPPER: 378/756/756/756/1513/1513, 1620MHz, 456
  - ADA: 82.6/165.2/165.2/330.3/330.3/660.6, 2520MHz, 512
- `gpu_db[]`: ~37 exact-name entries (V100 SXM2/PCIE, A100 ×4, RTX A2000–A6000, 3090Ti–3060, "RTX Axxxx ADA" variants, 4090→4060 (4070 duplicated entry), H100 PCIe, "NVIDIA H100 80GB HBM3" = SXM5 with 528 cores @1830MHz). Each overrides cores+MHz vs arch baseline.
- `get_flops_promised(device, precision_mode) -> float in units of 1e12 flops/s`:
  linear search by exact name; value = arch TFLOPS for mode (FP32→TF_32 i.e. TF32 tensorcore number!, FP16→FP_16_32, BF16→BF_16_32);
  scale: `value * (new_cores/arch_cores) * (new_mhz/arch_clock)`; negative lookup → warn + -1; unknown GPU → -1.
  Validation example in comment: 4080: 165.2/512*304 /2520*2505 ≈ 97.5 TFLOPS.
- `GPUUtilInfo {clock,max_clock,power,power_limit,fan,temperature,temp_slowdown,gpu_utilization,mem_utilization,throttle_reason}`.
- `nvml_get_device()`: lazy init, handle for device index 0 (`nvmlDeviceGetHandleByIndex_v2`).
- `get_throttle_reason(bits)`: "power cap" / "thermal cap" / "other cap" / "no cap".
- `get_gpu_utilization_info()`: queries SM clocks, power limit/usage, temp + slowdown threshold, throttle reasons, fan; utilization = MEAN over ≤128 hard-coded-size NVML sample buffer (`NVML_GPU_UTILIZATION_SAMPLES`, then MEMORY samples; sample_count reset between). Without NVML → error exit.

Used by train_gpt2.cu to print MFU % = actual fwd+bwd flops / promised flops, plus optional `[gpu|util X% ...]` status lines.

---

Cross-references: file-format magics also in doc 02; how trainers wire these headers together in docs 05 & 07; CUDA kernel-level .cuh headers (matmul/layernorm/etc.) covered in doc 08; ZeRO/multi-GPU in doc 09.
