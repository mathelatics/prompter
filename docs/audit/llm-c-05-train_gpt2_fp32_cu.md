# llm.c Audit 05 — `train_gpt2_fp32.cu` GPU fp32 Reference Trainer (1764 lines)

Direct CUDA translation of train_gpt2.c with smarter memory reuse. Header comment: params grads use `+=` (accumulation), activation grads use `=` except residual-stream ones which `+=` (layernorm backward adds into dresidual).

## Includes & globals
cublas_v2, cuda_runtime, cooperative_groups(+reduce); llmc utils/tokenizer/dataloader. `namespace cg = cooperative_groups`. Global `cublasHandle_t cublas_handle; cublasComputeType_t cublas_compute_type;`. cudaCheck/cublasCheck error macros. CEIL_DIV macro.

## Kernels (all in this file, NOT llmc/ — this trainer is self-contained)
- **encoder_forward_kernel3**: float4-vectorized (C/4 threads per row): out.bt,c4 = wte[ix]+wpe as float4 add. 128-bit LDG/STG motivation.
- **encoder_backward_kernel**: naive atomicAdd per element into dwte,dwpe ("really bad" by design—reference).
- **layernorm_forward_kernel3**: one WARP per row (idx = blockIdx*warp.meta_group_size()+meta_group_rank); warp-shuffle reduce for mean then var; rsqrtf(sum/C+1e-5f); uses __ldcs/__stcs streaming cache hints so weight/bias stay cached.
- **permute_kernel / permute_kernel_backward**: qkv fused (B,T,3,NH,d) ↔ separate q,k,v each (B,NH,T,d).
- **unpermute_kernel / _backward**: (B,NH,T,d) → (B,T,NH*d).
- **softmax_forward_kernel5**: autoregressive online softmax on preatt rows (N=B*NH, T×T); assert T%4==0; grid iterated BACKWARD (cache hint for following matmul); float4 vector scan over pos_by_4 full vecs + scalar tail up to own_pos; warp reduce max(greater) then sum(plus); renormalize; writes only i≤own_pos (upper triangle untouched garbage—but masked semantics since causal). inv_temperature param multiplies inside exp (scale folded in). NOTE: it does NOT zero the upper part; consumers must not read t>own_pos.
- **residual_forward_kernel**, **gelu_forward_kernel/backward** (same tanh formula as CPU).
- **matmul_backward_bias_kernel4**: column reduction of dout (B*T,OC): grid OC/32 blocks × 1024 threads; each warp handles 32 consecutive columns (lane→column), strided rows; smem[128] cross-warp accumulate; warp0 writes dbias[col] += sum.
- **layernorm_backward_kernel2**: warp-per-row; shared mem 2*C halves for dbias/dweight accumulators (atomicAdd within block); after loop whole block atomicAdds smem sums to global dweight/dbias. dinp += (as CPU formula).
- **softmax_autoregressive_backward_kernel**: dim3(T/4, B*NH) grid, 256 threads; processes 4 t-rows per block from the END backwards; computes local_sum Σ att*datt via two-stage block reduce (warp shuffle → smem block_acc[32] → warp shuffle again reading smem at lane index trick); then dpreatt[t][t3] = scale*att*(datt−local_sum) for t3≤t using __ldcs/__stcs.
- **adamw_kernel2**: flat elementwise AdamW with lerp trick (fma-based interpolation) and precomputed bias-correction denominators (beta1_correction=1−β1^t passed in). Same update math as CPU.
- **prepare_softmax_blockwide_nofloat4** + **fused_classifier_kernel3**: block-wide (1024 thr) online softmax over vocab row V (loop starts offset by threadIdx for L2-hit reverse iteration; two-stage warp/smem reduce); thread0 computes loss=-log(prob_target); then second pass recomputes probs writing optional probs buffer AND OVERWRITES logits IN-PLACE with (prob−indicator)*dloss (dloss default 1/(B*T)). This fuses softmax fwd + CE fwd + CE bwd; logits buffer becomes dlogits afterwards.
- **matmul_forward_kernel4**: hand-tiled SGEMM: each thread 8×8 outputs, block 16×16 threads → 128×128 tile; smem lhs_s[128][32], rhs_s[128][32] double-buffered style with syncthreads around 32-wide k-slices; float4 loads/stores; bias preloaded into vals; `__launch_bounds__(16*16, 2)`. Assumes dims divisible by 128? launch grid CEIL_DIV(BT,128)×CEIL_DIV(OC,128) — no bounds check inside kernel → requires multiples of 128 (works because callers pass B*T and OC that are multiples... actually C=768, 3C, 4C, Vp=50304 all %128==0; B*T must be multiple of 128! main doesn't enforce but README examples use big B*T; small test B=4,T=64=256 ok.)

## Launchers (block sizes)
encoder_forward 512thr; encoder_backward 256; layernorm_forward grid=ceil(N*32/512), 512; matmul_forward kernel4 dim3(16,16) grid ceil(BT/128)×ceil(OC/128); attention_forward = permute → cublasSgemmStridedBatched(kᵀq → preatt INTO inp scratch) → softmax_kernel5(scale) → SgemmStridedBatched(att·v → vaccum=inp scratch) → unpermute. Note `inp` (qkv scratch) reused as preatt & vaccum buffers (memory cleverness; qkv itself saved separately in qkvr buffer).
matmul_backward: dinp = dout·W (Sgemm NN, beta=0 set); dweight = inpᵀ·dout (NT, beta=1 accumulate); bias via kernel4.
layernorm_backward grid ceil(32N/512) bs512 smem 2C floats.
attention_backward chain: unpermute_bwd(scratch←dout) → Sgemm TT? (datt = v·dsplitᵀ? exact ops: datt ← vᵀ? see code: SgemmStridedBatched(OP_T,OP_N, T,T,HS, v, scratch → datt), dv = OP_N,OP_T(scratch, att), softmax_autoregressive_backward → dq (k·dpreatt), dk (qᵀ side), permute_backward. Buffers: dl_bt4c holds dqkv later etc. Heavy aliasing documented inline.
fused_classifier3: grid=N=B*T, block 1024.

## Model structs
Same ParameterTensors(16)/sizes as CPU. ActivationTensors has **21** entries (vs CPU 23): drops preatt (computed on the fly in bwd), keeps atty, att, adds qkvr (L,B,T,3C) and generic `output` scratch sized B*T*max(3C, NH*T, Vp) used for: forward QKV temp, preatt, vaccum, logits/dlogits. GradActTensors NUM_BACKWARD_TENSORS=3: bt4c (B,T,4C), preatt (B,NH,T,T), residual3 (B,T,C) — single-layer-sized buffers REUSED across all L layers in backward (huge memory saving vs CPU version). All allocations via cudaMalloc contiguous + pointer walking (malloc_and_point).

## gpt2_forward differences vs CPU
inputs/targets copied to device buffers (cudaMemcpy); lazy alloc prints MiB; per layer: ln1 → matmul into scratch → attention_forward(atty, qkvr, att, scratch) → attproj → residual2 → ln2 → fch → gelu → fcproj → residual3; final lnf → matmul_forward(output, lnf, wte, NULL, C→Vp) → if targets: fused_classifier3(output, losses, NULL, targets...) which leaves dlogits in output; mean_loss computed by copying (B,T) losses to pinned CPU buffer and averaging on host (sync point).
Backward reuses acts.output as scratch, aliases dl_btc onto acts.lnf (no longer needed), buffer_a=l_atty, buffer_b=l_fch reused inside attention_backward. First backward op: matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output /*dlogits*/, acts.lnf, params.wte, C,Vp) — dlogits already seeded uniform 1/(B*T) by fused classifier. Ends encoder_backward(dwte,dwpe,dresidual,inputs).

## gpt2_update
block 512; corrections computed host-side powf(β,t); lazily cudaMalloc+memset m,v.

## main() CLI defaults (differs from CPU!)
-i dev/data/tinyshakespeare/tiny_shakespeare_train.bin, -j ..._val.bin, -o logfile, -b 4, **-t 1024**, **-l 3e-4**, -v 20 (val every), -m 20 val batches, -s 20 sample every, -g 64 genT. Prints ASCII table of params. TF32 auto-enabled for sm≥80 (cublasSetMathMode CUBLAS_TF32_TENSOR_OP_MATH). One epoch = train_num_batches = num_tokens/(B*T). AdamW betas (0.9, 0.999), eps 1e-8, wd 0. Sampling reads model.acts.output logits row directly (after a forward WITHOUT targets output holds logits), copies first V to cpu_logits, sample_softmax. Logger lite: flush every 10 steps, fields s/trl tel only. Total avg time printed at end.

## test_gpt2_fp32.cu
Same gradcheck pattern as CPU test against gpt2_124M_debug_state.bin.
