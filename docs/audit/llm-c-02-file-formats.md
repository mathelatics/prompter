# llm.c Audit 02 — File Formats (checkpoint, debug state, tokenizer, data shards) + train_gpt2.py

All formats verified against `train_gpt2.py` (writer side) and `llmc/*.h`, `train_gpt2.c` (reader side).

## 1. Model checkpoint `gpt2_124M.bin` (fp32) / `gpt2_124M_bf16.bin` (bf16)
Header: 256 × int32 = 1024 bytes.
- header[0] magic = **20240326**
- header[1] version: **3** = fp32 tensors; **5** = bf16 tensors (float16 not implemented)
- header[2] block_size (maxT, 1024)
- header[3] vocab_size V (50257)
- header[4] n_layer L (12)
- header[5] n_head NH (12)
- header[6] n_embd C (768)
- header[7] padded_vocab_size Vp (50304 = ceil(50257/128)*128; pad multiple=128, value=0)
Then raw little-endian float32 (or bfloat16 reinterpreted as int16) parameter tensors in this exact order (the "write_tensors" order, grouped tensor-by-tensor across layers, NOT interleaved):
1. wte (Vp, C)          # token embeddings, vocab-padded rows are zeros
2. wpe (maxT, C)        # position embeddings
3. ln1w (L, C)          # for i in range(L): h.i.ln_1.weight
4. ln1b (L, C)
5. qkvw (L, 3C, C)      # attn.c_attn.weight — stored as nn.Linear weight [out,in], i.e. row-major (OC, C); HF Conv1D weights transposed at load
6. qkvb (L, 3C)
7. attprojw (L, C, C)   # attn.c_proj.weight
8. attprojb (L, C)
9. ln2w (L, C)
10. ln2b (L, C)
11. fcw (L, 4C, C)      # mlp.c_fc.weight
12. fcb (L, 4C)
13. fcprojw (L, C, 4C)  # mlp.c_proj.weight
14. fcprojb (L, C)
15. lnfw (C)            # final layernorm weight
16. lnfb (C)
Total params GPT-2 124M ≈ 123,864,320 (with padded vocab). lm_head tied to wte (same storage), so no separate head tensor.
Reader: `gpt2_build_from_checkpoint` in train_gpt2.c reads header[0..7] identically; errors on bad magic/version.

## 2. Debug state `gpt2_124M_debug_state.bin` (for gradcheck tests)
Header 256×int32:
- header[0] magic = **20240327**, header[1] version = **2** (v1→v2 padded vocab change)
- header[2] B, header[3] T
Body order: x (B,T int32), y (B,T int32), logits (B,T,Vp fp32 — note: padded? logits written from model output which is (B,T,V)... careful: write_state writes `logits.cpu()` directly → shape (B,T,V=50257) fp32; grads written with pad_vocab on wte grad (Vp,C)), loss (1 fp32 scalar), then all gradient tensors via write_tensors(grads,...,"float32") in same 16-tensor order as checkpoint. wte grad padded to Vp.
Used by test_gpt2.c / test_gpt2.cu to compare forward/backward numerics vs PyTorch ground truth.

## 3. Tokenizer `gpt2_tokenizer.bin`
Header 256×uint32 (read as uint32 in tokenizer.h but written as int32 — same bits):
- header[0] magic = **20240328**
- header[1] version: 1 (no EOT field; assume 50256 if vocab==50257) or **2** (current)
- header[2] n tokens (50257)
- header[3] eot_token id (50256, `<|endoftext|>`)
Body: for each token i in [0,n): 1 byte length L (<256, assert ≥1), then L raw bytes of `enc.decode_bytes([i])` (tiktoken gpt2 encoding). Decoder-only (int→bytes). Reader builds `char* token_table[]` with null terminators. `safe_printf` filters non-printable single-byte tokens.

## 4. Training data shard `.bin` (e.g. TinyStories_train.bin, tiny_shakespeare_train.bin, fineweb shards)
Header 256×int32:
- header[0] magic = **20240520**
- header[1] version = 1
- header[2] ntok (number of uint16 tokens following)
Body: flat array of `uint16_t` token ids (little endian). No other padding. File size must equal 256*4 + ntok*2 (checked by dataloader).
Semantics: one contiguous token stream; a "sample/batch slot" = B*T+1 consecutive tokens; inputs = buf[0..BT-1], targets = buf[1..BT]. Distributed: process r reads its B*T slice at offset rank*B*T within each global B*T*world batch (stride-based sharding across processes).
Loader details (llmc/dataloader.h `DataLoader`):
- glob pattern → shards list; validates every shard header upfront; asserts each shard ntok ≥ num_processes*B*T+1.
- shard_num_samples = (ntok*2 - 2) / (world*B*T*2) (drops last partial batch; -1 uint16 because window slides by BT though reads BT+1).
- shuffle mode (`should_shuffle=1`, train loader): mt19937 seeded 42+rank; permutes shard order on reset AND intra-shard sample permutation per loaded shard (Fisher–Yates via random_permutation).
- next_batch(): if current_sample_idx >= shard_num_samples → advance to next shard (wraparound resets epoch). fseek to header_bytes + idx*total_batch_bytes + local_offset, fread B*T+1 uint16.
- `dataloader_resume(loader, shard_idx, sample_idx)` used for checkpointed restarts (-y flag in cu trainer).
- EvalLoader (HellaSwag-style MC evals): magic **20240522**, version 1, header[2]=num_examples, header[3]=longest_example_bytes. Example stream of uint16: START_EXAMPLE delimiter 65535, EXAMPLE_BYTES, EXAMPLE_INDEX, LABEL, NUM_COMPLETIONS(=4 assumed), then context: NUM + tokens, then per completion NUM + tokens. Packs 4 completions per example into 4 consecutive batch rows sharing context; targets shifted so last context token predicts first completion token; mask=1 only on completion positions; evalloader_stat_losses picks argmin average masked loss per example, counts correct vs label.

## 5. train_gpt2.py (PyTorch reference + exporter) key facts
- Model: standard GPT-2; NewGELU exact tanh approximation; residual projections init std 0.02/sqrt(2*n_layer) via LLMC_RESIDUAL_SCALE_FLAG on attn.c_proj & mlp.c_proj; biases zero; other weights N(0, 0.02) with dedicated torch.Generator seed 42; lm_head tied to wte (LLMC_SKIP_INIT).
- from_pretrained supports gpt2/-medium/-large/-xl (transposes Conv1D weights); `d12|d24|d36|d48` = from-scratch equivalents.
- Default main run: overfit_single_batch=1, B=4, T=64, total_batch_size=256 → grad_accum=..., lr 1e-4, betas (0.9, 0.95), grad_clip 1.0, dtype float32; writes tensors on first iter before training.
- Exports produced when `--write_tensors=1`: gpt2_tokenizer.bin, gpt2_{size}.bin (fp32), gpt2_{size}_bf16.bin, gpt2_{size}_debug_state.bin.
- DDP via torchrun; ZeroRedundancyOptimizer stage support; TF32 via --tensorcores=1; flash via --flash=1 (SDPA path).
- Cosine LR with warmup down to lr*decay_frac; log file lines `s:%d tel:%f` / `trl`.
- requirements.txt: numpy, torch, transformers, datasets, tqdm (tiktoken imported inside __main__).

## 6. Other magics seen elsewhere
- dev/data scripts use same 20240520 format; llama3 tokenizer bin has its own format (see doc 10).
- Resume DONE files: utils.h find_max_step scans dir for `DONE_<step>` files (used by cu trainer's `-y 1` resume logic).
