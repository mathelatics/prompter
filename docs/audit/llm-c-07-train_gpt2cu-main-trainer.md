# llm.c Audit 07 — `train_gpt2.cu` (main bf16 fast trainer, 1904 lines @ HEAD f1e2ace)

The flagship CUDA trainer. Uses all `llmc/*.cuh` kernels (docs 08/09) and shared headers (doc 04).
Companion fp32 single-file version audited separately in doc 05. CPU reference in doc 03.

## 1. Includes & globals
- CPU utils: `llmc/utils.h`, `tokenizer.h`, `dataloader.h`, `rand.h`, `schedulers.h`, `sampler.h`, `logger.h`, `mfu.h`, `outlier_detector.h`.
- GPU: `cuda_common.h` (WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE, NVTX_RANGE_FN),
  `cuda_utils.cuh` (Packed128/f128/x128, warpReduceSum/Max, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged),
  `cublas_common.h` (CUBLAS_LOWP, cublasCheck, cublaslt_workspace(+size), cublas_compute, handles),
  layer kernels: `encoder.cuh`, `layernorm.cuh` (incl. `fused_residual_forward5`), `matmul.cuh` (`matmul_cublaslt`, `matmul_forward_cublaslt`, gelu fwd/bwd),
  attention: cuDNN path (`cudnn_att.h`) if `ENABLE_CUDNN` else custom `attention.cuh`;
  `fused_classifier.cuh`, `adamw.cuh` (`adamw_kernel3`), `global_norm.cuh`, `zero.cuh` (MultiGpuConfig etc.).
- Globals: `char filename_buffer[512]`; `cudaDeviceProp deviceProp` (filled in common_start); `cudaStream_t main_stream`;
  `IO_BUF_SIZE = 32 MiB` for device↔disk streaming io (`device_to_file`/`file_to_device`).

## 2. Config & parameters
`GPT2Config {max_seq_len, vocab_size, padded_vocab_size, num_layers, num_heads, channels}`.
`NUM_PARAMETER_TENSORS = 16` — same order/shapes as checkpoint format (doc 02): wte(Vp,C), wpe(maxT,C), ln1w/b, qkvw(L,3C,C), qkvb, attprojw(L,C,C), attprojb, ln2w/b, fcw(L,4C,C), fcb, fcprojw(L,C,4C), fcprojb, lnfw/b. All `floatX*`. static_assert struct size == 16 pointers.
`fill_in_parameter_sizes`: element counts per tensor; `param_sizeof[i]=sizeof(floatX)` uniformly (kept per-tensor for future mixed precision).
`malloc_and_point_parameters`: ONE big cudaMalloc for all params; pointers carved sequentially by byte offsets.

## 3. Activations: 21 tensors + recompute levels
`NUM_ACTIVATION_TENSORS = 21` (`ActivationTensors` fields, index order matters):
```
0 encoded(B,T,C) 1 ln1(L,B,T,C)* 2 ln1_mean 3 ln1_rstd (L,B,T float)
4 atty(L,B,T,C) 5 att: cuDNN→float (L,B,NH,T stats); else floatX (L,B,NH,T,T)
6 residual2 7 ln2* 8 ln2_mean 9 ln2_rstd 10 fch(L,B,T,4C) 11 fch_gelu(L,B,T,4C)**
12 residual3 13 lnf(B,T,C) 14 lnf_mean 15 lnf_rstd 16 losses(B,T float)
17 qkvr(L,B,T,3C) 18 output = B*T*max(3C, NH*T, Vp)   // logits OR dlogits OR scratchpad
19 scratch_bt4c(B,T,4C) 20 scratch_btc(B,T,C)
```
`*` zeroed when `recompute >= 2` (LN forward recomputed in backward; single `lnf` buffer reused for ALL layernorms).
`**` when `recompute >= 1`: only `B*T*4C` kept (single-layer GELU scratch, recomputed in backward).
Mechanism: `TensorSpec{void** ptr; size_t size; DType type}` + `TENSOR_SPEC(ptr,size)` macro; `fill_in_activation_sizes` sets sizes to 0 per recompute level; `malloc_and_point_activations` sums bytes → one cudaMalloc → **cudaMemset(acts_memory, 0, bytes)** (needed: non-cuDNN attention assumes zeroed att buffer; ~100ms cost acknowledged) → sequential pointer assignment; size==0 ⇒ pointer set NULL ("extra protection"). Prints MiB allocated.

## 4. `GPT2` struct
config; params + param_elements/sizeof + params_memory + num_parameters(+bytes); grads (same layout, separate memory);
`m_memory,v_memory,master_weights` (all fp32, sized `shard_num_parameters`); acts + specs + acts_memory;
batch_size/seq_len (fixed on first allocate_state call); inputs/targets (GPU int buffers); mean_loss (host; -1 = none);
accumulated_mean_loss (GPU float); cpu_losses (pinned host B*T); `rng_state` (xorshift seed for stochastic rounding) and
`rng_state_last_update`; use_master_weights; init_state; gelu_fusion; recompute; workload_indices/bucket_info (CPU scratch for encoder_backward bucketing, sized B*T*num_c_groups with `num_c_groups = CEIL_DIV(C, WARP_SIZE*x128::size)`).

`gpt2_init_common`: lazy NULLs; defaults: rng_state = **13371337 + process_rank**; use_master_weights=1; recompute=1; gelu_fusion=0 (comment: would be 2 for sm≥90; default off for now).

`gpt2_allocate_state(model,B,T)`: allocates grads (same big-block trick), activations, inputs/targets/accumulated_mean_loss, pinned cpu_losses, CPU scratch for encoder_backward, then AdamW m/v (+master if enabled) via `cudaMallocConditionallyManaged` (falls back to cudaMallocManaged on OOM; status all-reduced via `multi_gpu_cpu_float_sum`; prints WARNING about slow host-mem movement). Then prints `cudaMemGetInfo` usage, bytes-per-sequence and **estimated max batch size** `B + free/bytes_per_sequence`.

## 5. Checkpoint IO
`gpt2_write_to_checkpoint`: header magic 20240326; version = 3 (fp32) or 5 (bf16) (asserts mode ∈ {FP32,BF16}); fields [2..7]=maxT,V,L,NH,C,Vp; weights streamed via `device_to_file(..., IO_BUF_SIZE, main_stream)`.
`gpt2_build_from_checkpoint(path, weight_init=true)`: FP16 unsupported (exit, needs online conversion); validates magic/version (3|5, hint re-run python train_gpt2.py); cross-checks precision vs version (BF16 wants v5 `_bf16.bin`; FP32 wants v3 plain `.bin`, hints `make train_gpt2cu PRECISION=FP32`); reads config from header; allocates; `file_to_device` unless weight_init==false (resume-with-master-weights case: only hyperparams read); ends with cudaDeviceSynchronize.

### Model descriptors (`-e` can be a .bin OR descriptor string)
`gpt2_set_hyperparameters(depth)`: 6→(384,6h) gpt2-tiny 30M; 12→(768,12) 124M; 24→(1024,16) 350M; 36→(1280,20) 774M; 48→(1600,25) 1558M; 60→(1920,30) 2.7B; 72→(2880,30) 7.3B; 84→(3456,36) 12.2B. max_seq_len=1024.
`gpt3_set_hyperparameters(channels)` (note: dense attention unlike real banded GPT-3): 384→d6/h64 tiny31M; 768→d12 small125M; 1024→d24 medium; 1536→d24 hs96 large760M; 2048→d24 hs128 xl1.3B "[heads fixed]"; 2560→d32 hs80 2.7B; 4096→d32 hs128 6.7B; 5140→d40 13B; 12288→d96 175B. max_seq_len=**2048**.
`gpt_build_from_descriptor("dX" | "gpt2:dX" | "gpt3:cX")`: V=50257, Vp=50304; allocate; init on CPU with mt19937 **seed 42**, exactly matching torch's init ORDER: loops l=0..L-1 outer, tensor i inner; LN weights (i=2,8,14) set to 1 once at l=0; weight tensors i∈{0,1,4,6,10,12}: for i=0 only first V*C elems (not padded rows); per-layer tensors n/=L with layer_offset=l*n; scale = 0.02f, except i==6||i==12 (attproj, fcproj) scaled by `1/sqrt(2L)`; biases stay 0 (memset); `normal_()` fp32 then cast floatX; single cudaMemcpy to GPU.

## 6. Forward (`gpt2_forward`)
Fully synchronous (ends cudaDeviceSynchronize). Guards: params initialized; B≤model.batch_size && T≤model.seq_len (smaller allowed for inference); memcpy inputs→GPU; tokenCheck while copy in flight.
Graph per layer (identical math to doc 03 but kernel-fused):
1. `encoder_forward(encoded, inputs, wte, wpe, ...)` (residual[0]).
2. `layernorm_forward(ln1[0] | lnf, ...)` — first LN NOT fused.
3. For each l: `residual = l==0 ? encoded : residual3[l-1]`.
   - cuDNN: `matmul_forward_cublaslt(qkvr, ln1, qkvw,qkvb)` → `attention_forward_cudnn(atty, att_stats, qkvr)` (q,k,v split inside cuDNN from packed qkvr).
   - custom: memset att if T<seq_len (stale rows must be 0); `matmul_forward_cublaslt(scratch=outbuf, ...)` → `attention_forward(atty, qkvr, att, scratch)` (uses scratch for pre-softmax scores; att keeps probs).
   - `matmul_forward_cublaslt(scratch, atty, attprojw,b)`;
   - `fused_residual_forward5(residual2, ln2, ln2_mean/rstd, residual, scratch, ln2w,b, B*T,C)` ← residual+LN fused;
   - `matmul_forward_cublaslt(fch_gelu, ln2, fcw,fcb, ..., epilogue_out=fch, gelu_fusion)` ← optional cuBLASLt GELU epilogue fusion (writes both pre-GELU fch AND post-GELU result depending on flag);
   - `matmul_forward_cublaslt(scratch, fch_gelu, fcprojw,b)`;
   - **cross-block fusion**: next layer's `fused_residual_forward5(residual3[l], ln1[l+1]...)`, or final `lnf` at last layer.
4. `matmul_forward_cublaslt(output, lnf, wte /*weight tying*/, NULL bias, B,T,C,Vp)` → logits in `acts.output`.

`gpt2_validate(inputs,targets,B,T)`: forward + `fused_classifier(output, losses, dloss=1/(B*T), targets, ..., generate_d_logits=False)`; copies losses to pinned cpu_losses; returns mean (also leaves per-token losses for HellaSwag).

## 7. Backward (`gpt2_backward_and_reduce(inputs,targets,grad_accum_steps,micro_step)`)
- micro_step==0: memsetAsync losses AND entire grads_memory (both accumulate with += across micro-steps).
- `dloss = 1/(B*T*grad_accum_steps)`; memcpy targets + tokenCheck; `fused_classifier(..., True)` → fills losses (+=) AND dlogits into acts.output.
- `dresidual = scratch_btc` memset 0. `scratchF/X = acts.output` reused as backward scratch.
- Classifier matmul backward: `matmul_backward(scratch_bt4c=d_lnfx, grads.wte, NULL, acts.output(dlogits), acts.lnf, params.wte)` (weight grad accumulates into grads.wte — weight tying handled here).
- `layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, scratch_bt4c, residual3[L-1], lnfw, lnf_mean/rstd)`; after this, `residual3[L-1]` memory is repurposed as `dl_btc` scratch (no longer needed).
- Layer loop l=L-1..0 mirrors forward reversed:
  - if recompute≥1: `gelu_forward(l_fch_gelu_single_buf, l_fch_pre_gelu)` recompute;
  - `matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, ..., pre-gelu buf + gelu_fusion)` (fused gelu_backward in matmul when fusion==2);
  - if recompute≥2: recompute ln2 forward;
  - `matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw)`;
  - `layernorm_backward(dresidual +=, dl_ln2w,b, ...)` (LN backward ADDs to dresidual → accumulates MLP+Attn branches);
  - `matmul_backward(dl_btc, dl_attprojw,b, dresidual, l_atty, ...)`;
  - attention backward: cudnn `attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, stats)` OR custom `attention_backward(dl_bt4c, buffer_b=fch_pre_gelu, scratchX, buffer_a=atty, dl_btc, l_qkvr, l_att)` (reuses atty & fch buffers as scratch);
  - if recompute≥2: recompute ln1; `matmul_backward(dl_btc, dl_qkvw,b, dl_bt4c, l_ln1, ...)`; `layernorm_backward(dresidual +=, dl_ln1w,b ...)`.
  - On last micro-step: `multi_gpu_async_reduce_gradient(pointers[12 per-layer grads], nelem[], ..., main_stream)` — background-stream NCCL reduce overlap.
- `encoder_backward(grads.wte, grads.wpe, scratchX, workload_indices, bucket_info, dresidual, model->inputs(gpu), inputs(cpu), B,T,C, seed=random_u32(&rng_state))` — bucketed sparse scatter-add (see doc 08).
- If last_step: `global_sum_deterministic(accumulated_mean_loss, acts.losses, B*T)`; NCCL AllReduce avg across GPUs; async copy to host mean_loss; also async-reduce wte/wpe/lnfw/lnfb grads (sizes Vp*C, T*C, C, C).
- Sync; host `mean_loss /= B*T*grad_accum_steps` (else -1).

## 8. Grad norm (`gpt2_calculate_grad_norm`)
Writes squared norm into acts.output (repurposed). `get_max_num_block_sums({1,L},2)`. ZeRO-1: iterate 16 tensors, shard offset via `multi_gpu_get_shard_offset(size, cfg, stage=1)`, per-tensor `global_norm_squared` (strided over L layers for block tensors i∈[2,13]), `global_sum_deterministic` over block-sums, then NCCL AllReduce SUM across GPUs. DDP: whole grads vector locally, no comms. Returns sqrtf(host copy).

## 9. Optimizer step (`gpt2_update(lr,beta1,beta2,eps,wd,grad_scale,t,...,init_from_master_only=false)`)
- First call (init_state): memset m,v to 0 over shard_num_parameters; init master_weights via `copy_and_cast_kernel<<<grid=CEIL_DIV(shard.size,512), blocks=(,num_layers), 512>>>`.
- Saves `rng_state_last_update = rng_state` BEFORE consuming random numbers (for bit-exact resume re-rounding).
- Per tensor i (0..15): fresh `seed = random_u32(&model->rng_state)`; num_layers=1 for i<2||i>13;
  `wd = (i ∈ {0,1,4,6,10,12}) ? weight_decay : 0` — decays wte,wpe,qkvw,attprojw,fcw,fcprojw (all 2D matrices incl. embeddings); biases & LN untouched.
  Param/grad pointers use FULL-tensor local shard offset (`local_offset_full = tensor.offset + shard.offset`);
  optimizer-state offset: DDP → same full offset; ZeRO-1 → `local_offset_partial = tensor.offset / num_processes` (contiguous own slice of m/v/master).
  `init_from_master_only` path: `init_from_master(param_ptr, master_ptr, ...)` restores low-precision weights from fp32 masters using saved RNG (identical stochastic rounding).
  Else `adamw_update(param, master, grad, m, v, shard.size, tensor.size, tensor.size, shard.size, num_layers, lr, beta1, beta2, t, eps, wd, grad_scale, seed)` (kernel details doc 08).
- ZeRO-1: after update, grouped `ncclAllGather` per layer on `nccl_stream` broadcasts updated shard to all ranks' params_memory.
- Ends with device sync.

## 10. MFU estimate (`gpt2_estimate_mfu(num_tokens, dt)`)
Paper arXiv:2001.08361 §2.1. `flops_per_token = 6N + 6*L*C*T` (N = ALL params incl. LN/bias/wpe — acknowledged slight overestimate; wte included legitimately due to tying in classifier matmul). flops_achieved = flops_per_step/dt; promised = `get_flops_promised(deviceProp.name, PRECISION_MODE)*1e12`; unknown GPU → -1.

## 11. State save/resume (`save_state`/`load_state`, magic **20240527** v1)
Header ints: [2]=num_processes [3]=rank [4]=use_master_weights [5]=should_shuffle; [10]=step; [20..21](u64)=rng_state; [22..23]=rng_state_last_update; [30..31](size_t)=current_shard_idx; [32..33]=current_sample_idx.
Body: m (shard float), v, master (if used) via device_to_file; then if shuffle: gl_pathc, shard_indices[], shard_num_samples, intra_shard_indices[], raw `mt19937_state` struct.
`load_state`: asserts procs/rank match; warns if state has masters but run doesn't (ok), EXITS if run wants masters missing from state; sets init_state=false; loads m,v,master; **bit-perfect weight restoration trick**: rng_state = rng_state_last_update → `gpt2_update(..., init_from_master_only=true)` re-applies identical stochastic rounding from master weights → then restore final rng_state from header; dataloader arrays validated against current glob (assert same shard/sample counts); `dataloader_resume(shard_idx, sample_idx)`.

Checkpoint bundle (`write_checkpoint`): rank0 writes `model_%08d.bin`; ALL ranks write `state_%08d_%05d.bin`; barrier; rank0 touches `DONE_%08d`. `delete_checkpoint` mirrors with remove(). Retention policy in main: keep `-nk` most recent; `-nm` marks "major" checkpoints never deleted (`step_delete = step - checkpoints_keep*checkpoint_every`, skipped if multiple of major_checkpoint_every).

## 12. CLI (error_usage documents all; parser requires `-x val` pairs, flags are 1-2 letters)
Defaults: -i dev/data/tinyshakespeare/tiny_shakespeare_train.bin; -j ...val.bin; -e gpt2_124M_bf16.bin; -o NULL; -lg -1; -n 0; -nk 0; -nm 0; -y 0; -b 4; -t 1024; -d auto (=B*T*procs); -x -1 (one epoch); -k cosine; -l 3e-4; -u 0 warmup; -q 1.0 (no decay); -c 0.0 wd; -sl/-sg 0.0 (off); -v 20; -m 20; -s 20; -g 64; -h 0; -a 0; -f 1 (tf32 override); -w 1 (master weights); -ge -1(auto→0 currently); -z 0; -r 1; multi-node: -pn 1 -pr 0 -pg 8 -pm mpi (-pi method, -ps server ip, -pf fs path).
Mapping notes: `-l` vs `-lg` disambiguated by argv[i][2]; `-n` vs `-nk`/`-nm`; `-s` vs `-sl`/`-sg`; `-g` vs `-ge`.

## 13. main() flow
1. `multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method)`; `common_start(override_enable_tf32, false)`.
2. tokens_per_fwdbwd = B*T*num_procs; total_batch_size default = that (grad_accum 1); assert divisible; grad_accum_steps = total/tokens_per_fwdbwd. gelu_fusion==-1 → 0. overfit_single_batch==1 → train pattern := val pattern.
3. Big parameter table printf0 (includes device name, peak TFlops, precision string: FP32 shows "TF32" if cublas_compute==CUBLAS_COMPUTE_32F_FAST_TF32 else "FP32"; BF16/FP16 literal).
4. Resume detection: `find_max_step(output_log_dir)` scans DONE_* files; if resuming: build from `model_%08d.bin` with weight_init = !use_master_weights. Else `.bin` extension → checkpoint load; else descriptor init.
5. Model meta printed (V, Vp, L, NH, C, num_parameters).
6. Loaders: train with permute=1 (0 if overfit-single-batch), val permute=0. train_num_batches = max_steps or `ntok / total_batch_size` (one epoch, outer-loop steps); val_num_batches = val_max_steps or `ntok_val / tokens_per_fwdbwd`.
7. HellaSwag: path `dev/data/hellaswag/hellaswag_val.bin`; runs only if file exists (`access F_OK`) and -h 1; hint to run python dev/data/hellaswag.py.
8. `set_zero_configs(&multi_gpu_config, zero_stage, num_params)` (doc 09; may downgrade stage if not divisible).
9. `gpt2_allocate_state`; if resuming → load_state(`state_%08d_<rank>.bin`). OutlierDetectors for loss & gradnorm init.
10. Sanity: warn loudly if T < model max_seq_len (unused wpe rows; hint -t 1024 GPT-2 / 2048 GPT-3); assert T <= max_seq_len.
11. Training loop `for (; step <= train_num_batches; step++)` with `last_step = step==train_num_batches` and break AFTER eval/checkpoint section on last_step (so eval+ckpt happen one extra time at end):
    - Val every `val_loss_every` (default 20) & last: reset val_loader, loop val_num_batches `gpt2_validate`, sum; average across GPUs via `multi_gpu_cpu_float_sum / num_processes`; log `tel:`.
    - HellaSwag every val_loss_every (step>0) & last: evalloader_reset; per batch: next_batch, gpt2_validate, `evalloader_stat_losses(model.cpu_losses)`; acc = Σcorrect (all-reduced) / num_examples; logs `eval:`.
    - Sampling (rank0, sample_every>0): rng_state=**1337**; prompt filled with eot_token; loop t=1..genT-1: `gpt2_forward(gen_tokens, 1, CEIL_DIV(t, min(T,256))*min(T,256))` (inference batching trick; note comment: smaller B always bit-exact, T≥256 bit-exact on cuDNN 9.2.1/FE 1.5.2); logits = output + (t-1)*Vp; copy first V entries to CPU, cast to float if needed; `coin=random_f32(&state)`; `sample_softmax`; append token; safe_printf decode; fflush.
    - Checkpoints (only when resuming==0!): write every checkpoint_every & last; retention delete logic; `resuming = 0` after first iteration.
    - Train: cudaEventRecord(start); micro-step loop {next_batch; forward; backward_and_reduce}; z-loss via detector; LR from scheduler at `step`; grad_norm; z-grad; skip update if zloss/zgrad exceed thresholds (-sl/-sg) with printf; else grad_clip=**1.0** hardcoded, `grad_scale = min(1, 1.0/grad_norm)`; `gpt2_update(lr, β1=0.9, β2=0.95, ε=1e-8, wd, grad_scale, t=step+1)`; event end + sync.
    - Timing: tokens_processed = num_procs*B*T*grad_accum; tok/s; EMA 0.95 with bias correction `ema/(1-0.95^step)`; first step excluded (warmup for cuBLAS/cuDNN init); MFU %; print line: `step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| lr %.2e | %.2f ms | %.1f%% bf16 MFU | %.0f tok/s` (literal "bf16" even for other precisions); optional GPUUtilInfo line every -lg steps; logger `trl/lr/norm`; `cudaProfilerStop()` after step 3.
    - Final: total average iteration time = sum/(train_num_batches-1) (excludes warmup).
12. Cleanup: events, eval/train/val loaders, tokenizer, gen buffers, multi_gpu_config_free, gpt2_free, common_free.

`common_start(override_enable_tf32=true, print_device_info=true)`: cudaGetDeviceProperties(local_device_idx); create+name main_stream; cublasLtCreate + cudaMalloc workspace; TF32 iff FP32-mode && cc≥8 && not overridden; create_cudnn if ENABLE_CUDNN.
`common_free`: destroy stream/workspace/handles/cudnn.

`#ifndef TESTING` wraps everything from error_usage/main down — `test_gpt2cu.cu` compiles this file with -DTESTING and provides its own main.

## 14. Notable design decisions / gotchas
- One giant cudaMalloc per group (params, grads, activations) → deterministic layout, easy save/load, no allocator overhead.
- Loss scaling: forward-only validate uses dloss=1/(B*T); training divides additionally by grad_accum_steps; host mean_loss again divided by B*T*grad_accum after global avg (fused_classifier accumulates partial sums into acts.losses across micro-steps).
- `acts.output` triple-duty: logits (fwd) / dlogits (bwd start) / generic scratch (grad_norm buffer!) — hence oversized max(3C, NH*T, Vp).
- Weight tying means grads.wte receives contributions from BOTH encoder (via encoder_backward) and classifier matmul backward; encoder_backward uses stochastic rounding seed drawn from model rng.
- Resume correctness depends on saving rng_state before last update (re-rounding master→low-precision identically).
- Stochastic rounding (when floatX=bf16 without master weights) consumes rng per tensor per update.
- No fp16 support anywhere in checkpoint loading (PRECISION=FP16 exits early in build_from_checkpoint).

Cross-ref: kernels in doc 08; multi-GPU/ZeRO internals doc 09; formats doc 02; shared headers doc 04.
