# llm.c Audit 08 — CUDA Kernel Headers in `llmc/*.cuh`

Files audited (line counts @ HEAD f1e2ace): `cuda_common.h` 208, `cuda_utils.cuh` 285, `cublas_common.h` ~50,
`matmul.cuh` 290, `gelu.cuh` 66, `encoder.cuh` 234, `layernorm.cuh` 505, `fused_classifier.cuh` 149,
`global_norm.cuh` 89, `adamw.cuh` 98, `attention.cuh` 276. Consumed by train_gpt2.cu (doc 07).

---

## 1. `cuda_common.h` — global CUDA scaffolding
Include guard CUDA_COMMON_H. Includes cuda_runtime, NVTX3 (+cudaRt), cuda_profiler_api, bf16/fp16 headers, utils.h.
- `extern cudaDeviceProp deviceProp;` — instantiated only in the main .cu file; kernels read it for occupancy sizing.
- `WARP_SIZE 32U`; `MAX_1024_THREADS_BLOCKS`: 2 on sm_80/sm_90+ (fit two 1024-thread blocks per SM), else 1. Used in __launch_bounds__.
- `CEIL_DIV(M,N)`; `constexpr std::bool_constant<true> True; False;` (compile-time bool function args; note: **False is actually true-valued** — used positionally as tag type param, template picks value from its own bool param, so harmless but confusing).
- `cudaCheck(err)` macro → prints file:line + cudaGetErrorString, exit. `cudaFreeCheck(ptr)` frees + resets pointer to nullptr + checks.
- Precision: `enum PrecisionMode {PRECISION_FP32, PRECISION_FP16, PRECISION_BF16}`; compile-time select: ENABLE_FP32→float, ENABLE_FP16→half ("gradient scaler not implemented" warning), default **__nv_bfloat16 floatX**.
- Back-compat `__ldcs/__stcs` overloads for bfloat16 (as unsigned short) when CUDA <12 and arch <800.
- NVTX: RAII class `NvtxRange(name)` / `NvtxRange(base, number)`; macro `NVTX_RANGE_FN()` auto-names with __FUNCTION__.
- **device_to_file(dest, src, num_bytes, buffer_size, stream)**: pinned 2×buffer double-buffer pipeline GPU→disk; primes first async D2H read, syncs, then loop: start next read async, fwrite previous buffer, streamSync, swap; final flush write buffer. Frees host buf.
- **file_to_device(dest, src, num_bytes, buffer_size, stream)**: mirror H2D; uses `cudaMallocHost(..., cudaHostAllocWriteCombined)` (CPU writes/GPU reads optimization); fread CPU-side overlaps cudaMemcpyAsync.

## 2. `cuda_utils.cuh` — device primitives
- `Packed128<ElementType>` alignas(16): forces LDG.128/STS.128 vectorized access for any element width; ctor from int4 (memcpy), `constant/zeros/ones`, operator[], get_bits(); `static constexpr size = 16/sizeof(ElementType)` (bf16→8, fp32→4, fp8→16).
  Typedefs: `f128 = Packed128<float>`, `x128 = Packed128<floatX>`.
- Helpers: load128, load128cs (streaming), store128, store128cs, store128cg (L2-only, bypass L1). All require 16B-aligned addresses.
- DType enum class {FP32,FP16,BF16} + `sizeof_dtype()` + overloaded `dtype_of(ptr)` (used by TensorSpec machinery in doc 07).
- `cast_value<Td,Ts>` specializations float←{float,half,bfloat16}; `copy_and_cast_kernel<Td,Ts>(dst, src, n, stride_dst, stride_src)` grid (ceil(n/block), num_slices) — used e.g. master-weight init.
- Reductions: `warpReduceSum/Max` via __shfl_xor_sync butterfly; `blockReduce<warp_reduction>(val, final_sync=false, out_of_bounds=0)` — two-level shuffle/shared/shuffle; each call adds 128B static shared; final_sync needed if reused in loops.
- Deterministic sum: `global_sum_single_block_kernel` asserts gridDim.x==1, 1024 threads grid-stride + blockReduce → single guaranteed-order result (`global_sum_deterministic(result, values, count, stream)`).
- `cudaMallocConditionallyManaged(out, bytes)`: try cudaMalloc; on OOM reset error, fall back cudaMallocManaged + MemAdvise PreferredLocation=CPU; returns 1 if fell back (callers aggregate & warn).
- RNG-for-rounding: **SquirrelNoise5(positionX, seed)** constexpr hash (5 mix constants 0xd2a80a3f, 0xa884f197, 0x6C736F4B, 0xB79F3ABB, 0x1b56c4f5); `Get2dNoiseUint(x,y,seed)=SquirrelNoise5(x + 198491317*y, seed)`.
- `stochastic_rounding(float in, bf16* out, seed)`: random = Get2dNoiseUint(threadIdx.x, blockIdx.x*blockDim.x+blockIdx.y, seed); threshold = random & 0xFFFF; take low 16 bits of float; if > threshold round up (set low 16 bits) else truncate; convert via __float2bfloat16_rn. half version: **TODO unimplemented (plain cast)**; float version: no-op.

## 3. `cublas_common.h`
CUBLAS_LOWP = CUDA_R_32F/16F/**16BF** per precision flag. Globals: workspace 32 MiB (comment: only Hopper needs 32, others fine with 4), `cublas_compute` default CUBLAS_COMPUTE_32F (overridden for TF32 in common_start), cublaslt_handle. cublasCheck macro prints status code + location, exits.

## 4. `matmul.cuh` — cuBLASLt wrapper + bias backward
### `matmul_cublaslt(d, a, b, bias, m,n,k, stream, transA=true, transB=false, batch_count, strideA, strideB, strideOut, accumulate, pre_gelu, backward)`
Column-major cuBLASLt conventions: computes D = op(A)·op(B) (+bias epilogue). All pointers must be 16B aligned (hard exit).
Layouts: A (k×m ld=k) if transA else (m×k ld=m); B similar; C/D (m×n ld=m). FP8 accommodation: C forced BF16 when sizeof(floatX)==1; bias dtype forced BF16 for FP8. Scale type always FP32.
Strided-batched path sets BATCH_COUNT + STRIDED_BATCH_OFFSET on all layouts (batch used by non-flash attention; C and D share strideOut).
Epilogues: has_gelu → GELU_AUX_BIAS / GELU_AUX (fwd; AUX_LD=m) or **DGELU** (backward; asserts no bias with bwd-gelu); has_bias → BIAS (fwd) or **BGRADB** (unused here since dbias handled separately); else DEFAULT.
Heuristic algo lookup (max 1 result) with preference capped at cublaslt_workspace_size; alpha=1, beta=accumulate?1:0; all descriptors created/destroyed per call (heuristic caching internal to cuBLASLt noted).
### `matmul_forward_cublaslt(out, inp, weight, bias, B,T,C,OC, stream, pre_gelu, gelu_fusion)`
maps row-major Y[B*T,OC] = X[B*T,C] · W[OC,C]^T + bias → calls cublaslt(weight^T view as A, m=OC, n=B*T, k=C). If gelu_fusion<1 and pre_gelu given: unfused matmul into pre_gelu then separate `gelu_forward`; else fused GELU_AUX epilogue writing both out and pre_gelu.
### `matmul_backward(dinp, dweight, dbias, dout, inp, weight, dbias_buffer, B,T,C,OC, stream, pre_gelu, gelu_fusion)`
1. dbias != NULL: custom kernel9 (below). block_dim=(4,8,blockSize/32); OC_per_warp=64@bf16; grid_y sized to fill whole GPU; if grid_y==1 write directly into dbias (with += old value), else write partials to dbias_buffer (aux mode, strided by blockIdx.y*OC) then `reduce_add_sum_kernel` folds (dst[i] += Σ_l src[i + n*l], f128-vectorized) — kernel9 overwrites buffer so no memset needed. Sets dbias=NULL after so fused BGRADB isn't used later.
2. dinp = dout · weight (no accumulate) with optional fused DGELU epilogue when gelu_fusion>=2 (pre_gelu passed as aux); else separate `gelu_backward_inplace(dinp, pre_gelu)` when pre_gelu set.
3. dweight += inpᵀ... : matmul_cublaslt(dweight, inp, dout, accumulate=true) — **+= semantics** required for grad accumulation across micro-steps.
### `matmul_backward_bias_kernel9<OutFloat, UseAuxBuffer>`
Each warp covers bdy(8) × x128::size(8) = 64 output channels; threadIdx.x=warp_d(4 lanes split BT), y=warp_c, z=block_d. Grid-strides over B*T rows loading x128 of dout column-slice; fp32 register accumulators; warp-shuffle reduce (width 4) → shared sub_results[x128.size][block_d][warp_c]; second pass cross-block_d reduce; writes either atomic-free direct (needs unique owner per OC — guaranteed when grid_y==1) or aux slot per blockIdx.y.

## 5. `gelu.cuh`
Tanh approximation: `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`. forward kernel2: x128 grid, load128cs input (single-use), normal store128 (next op may reuse cache). backward inplace kernel: needs pre-gelu `inp` + dout co-located in d_in_out; derivative via sech² = 1/cosh²; block 128. Launchers assert N divisible by block*x128.size.

## 6. `encoder.cuh`
- forward kernel3: one x128 per thread; out[b,t,c] = fl(wte[ix,c]) + fl(wpe[t,c]); block 256.
- **wte_backward_kernel<BLOCK=256>** (deterministic embedding-grad scatter): CPU pre-buckets all (bt, c_group) pairs by token id (see launcher); bucket key = c_group + num_c_groups*ix; bucket_info int4 {start_idx, size, ix, c_group}; workload_indices list bt's. Warp handles WARP_SIZE*x128.size channels (256@bf16; c>=C guards partial warps for C=768). Each warp of the block processes items strided by BLOCK/WARP_SIZE; non-zero warps dump fp32 accum to shared; warp 0 aggregates shared + loads existing dwte row, then **stochastic_rounding(accum + old, &out, seed + bucket*WARP_SIZE + threadIdx.x + k)** — unique per-element noise ⇒ bit-reproducible RMW (each bucket written exactly once).
- wpe_backward_kernel: thread owns x128 slice of (t,c); sums over batch b (load128cs), stochastic-rounds into dwpe with seed+idx+k. Fully deterministic because each (t,c) updated once.
- Launcher `encoder_backward(dwte, dwpe, scratch, workload_indices, bucket_info, dout, inp(gpu), inputs_cpu, B,T,C, seed, stream)`: launches wpe kernel FIRST (overlaps CPU bucketing); builds std::unordered_map buckets (data packs bt | c_group<<32 | ix<<42), sorts descending by size (longest-first scheduling), fills int4/int arrays, two async H2D copies into `scratch` (acts.output re-used; assert capacity), launches wte kernel with grid=num_buckets.

## 7. `layernorm.cuh`
Header comment policy: params use += (grad accumulation), activations use = except residual-stream ones which += deliberately.
- forward kernel3 (fallback, no smem): one warp per row; strided scalar loops; warpReduceSum twice; rsqrtf(sum/C+1e-5); __ldcs/__stcs streaming to keep w/b in cache; mean/rstd optional (NULL allowed).
- forward kernel6 (preferred): blockDim=(32, block_y=8); dynamic smem holds weight+bias copies + per-warp-row input cache s_in ((2+threadIdx.y)*C elements); three passes (sum, var, normalize) reading from smem; lane0 stores mean/rstd. Launcher tries cudaFuncSetAttribute(MaxDynamicSharedMemorySize, smem); on failure falls back to kernel3.
- **fused_residual_forward_kernel5**: same skeleton as kernel6 but first pass computes res=out=inp1+inp2 (fp32 add, stored streaming AND cached in smem s_res), then LN on res → normed; writes residual global too. Fallback path = residual_forward + layernorm_forward sequentially.
- residual_forward_kernel: trivial x128 add.
- **layernorm_backward_kernel10** (the monster; __launch_bounds__(512,2)): grid = 2 blocks/SM × SMs (persistent-ish). Each warp strides over bt rows; two warp reductions compute Σ(w·dout) and Σ(w·dout·x̂)-style terms: dnorm_mean, dnorm_norm_mean = (Σ dnorm·x)/C·rstd − dnorm_mean·mean·rstd; inner loop over C in x128 chunks: dinp += rstd·(w·dout − dnorm_mean − norm_bti·dnorm_norm_mean) (classic LN backward, accumulated into existing dinp via load128 → supports += chaining across MLP/attn branches); store128cg (L2 cache for next kernel, bypass L1).
  dweight/dbias per-block accumulation in shared (dbias_shared/dweight_shared, rounded_C layout); cross-warp combine trick: warps≠0 write tmp slots, warp0 loads+sums (avoids extra syncthreads cost; comment about 64-bit store tradeoff). At block end: each block appends partials to global scratch layout [flag | pad 32 floats | dbias|dweight interleaved per block offset 2*C*blockIdx.x]; `atomicInc(scratchFlag, gridDim.x)` last-block pattern; last block reduces all partials in shared, then converts & **adds into existing dbias/dweight (RMW with load128)** casting float→floatX. Scratch flag memset (1 float) before launch. Shared mem sized `(2*rounded_C + 2*(block_size-32)*f128.size)*4`.
- Launcher signatures: layernorm_forward(out, mean, rstd, inp, weight, bias, B,T,C, stream); layernorm_backward(dinp, dweight, dbias, scratch(float*), dout, inp, weight, mean, rstd, B,T,C, stream).

## 8. `fused_classifier.cuh`
Kernel5 (grid=B*T, block=1024, launch_bounds(1024, MAX_1024_THREADS_BLOCKS)): processes ONE (b,t) row per block; **blocks run in reverse index order** (`idx = gridDim.x - (blockIdx.x+1)`) for L2 cache hits against freshly-written logits from the classifier matmul.
- prepare_softmax_blockwide3: online max/sum over row (padded P stride), backwards i starting past end; special remainder loop for tail beyond V (bounds-checked), main loop unbound-checked using load128 packed; blockReduce max then adjust thread sums, blockReduce sum → SoftmaxParams{1/sum, max}.
- loss: thread0 computes prob=target logit softmax, `losses[idx] -= logf(prob)` (note `-=`: accumulates across micro-steps; caller zeroes losses).
- __syncthreads() REQUIRED before gradient overwrite (race documented: loss could see exp(+90)→inf otherwise).
- dlogits loop: recompute prob from logits (kept in cache thanks to cs hints ordering), `logits ← (prob − 1{ix})·dloss` in-place (template WriteDLogits; store128cs streaming); optional probs buffer (unused in training path, NULL passed). Tail loop handles V % x128.size.
- Launcher: fused_classifier<Type, WriteDLogits>(logits, losses, dloss, targets, B,T,V,P, bool_constant, stream).

## 9. `global_norm.cuh`
- global_norm_squared_kernel<T>: 2D grid (gx data slices, gy=num_slices layers/segments); grid-stride fp32 accumulator of x²; blockReduce; thread0 writes partial to out[blockIdx.y*gridDim.x + blockIdx.x] with plain += (no atomics; regions disjoint).
- global_norm_aggregate_kernel: single block, 1024 threads? (uses threadIdx.x directly, expects ≤blockDim entries) reduces partials → out[0].
- `get_max_num_block_sums(num_slices_all, numel)` must stay in sync with launcher grid math: grid_size = maxThreadsPerSM*SMs/512; gx=CEIL_DIV(grid,slices), gy=slices; assert gx*gy<1024.
- launcher(reset=...) memsets max_num_block_sums floats first (used for first tensor pass).
Used by gpt2_calculate_grad_norm (doc 07 §8). Note: final aggregation happens via separate `global_sum_deterministic` there, not this aggregate kernel (both exist).

## 10. `adamw.cuh`
- `lerp(start,end,w) = fma(w,end, fma(-w,start,start))` (2-op NVIDIA lerp trick).
- adamw_update device fn (one thread per param): grad *= grad_scale; m = lerp(grad, m, β1); v = lerp(grad², v, β2); m̂=m/(1−β1ᵗ), v̂=v/(1−β2ᵗ) (corrections computed host-side); old = master ? master[idx] : (float)param; param = old − lr·(m̂/(√v̂+ε) + wd·old); **stochastic_rounding(param, &params_memory[idx], seed)** writes low-precision copy; update master copy if present. Decoupled AdamW ✓ (matches PyTorch semantics modulo SR).
- adamw_kernel3: adds layer-dimension offsets: params += blockIdx.y*w_stride, grads += blockIdx.y*g_stride, m/v/master += blockIdx.y*s_stride (lets one launch cover all L layers of a stacked tensor; grid=(num_blocks, num_slices)).
- init_from_master_kernel: just SR-copy master→param (resume path). Block size 512 MUST match adamw_update so SquirrelNoise indices (which include blockIdx) reproduce identical rounding.
- Host launchers compute beta corrections via powf(t) and CEIL_DIV grid.

## 11. `attention.cuh` (non-cuDNN fallback; dense, T² memory)
Pipeline fwd: qkvr buffer holds permuted Q,K,V each (B,NH,T,HS) carved from packed (B,T,3,NH,HS) by permute_kernel (scalar copies, __ldcs).
preatt = KᵀQ via strided-batched matmul_cublaslt (transA=true, batch B*NH, strides T*HS/T*HS/T*T) into `inp` scratch reuse.
softmax_forward_kernel5: warp per row (N=B*NH rows × T), ONLINE softmax over causal prefix only (i < pos_by_4 vectorized-by-4 aligned loop + single-lane diagonal term at 4*pos_by_4+lane ≤ own_pos), inv_temperature = 1/√HS folded in; warpReduceMax/Sum; renormalize loop recomputes exp (faster than reload) and stores att probs. Backwards block order for cache affinity with following matmul. Zero upper triangle implicitly never written — requires zero-initialized att buffer (why train_gpt2.cu memsets when T shrinks).
att·V batched matmul (transB=false pair) → vaccum (scratch again) → unpermute_kernel → out (B,T,C).
Backward: unpermute_kernel_backward (dout→(B,NH,T,HS) scratch); datt = V·dOᵀ-style batched matmul (transA); dv = attᵀ... wait actual: `matmul_cublaslt(datt, v, dout_perm, ... transA=true)` and `matmul_cublaslt(dv, dout_perm, att, ..., transB=true)`; softmax_autoregressive_backward_inplace_kernel: block=(256), grid=(T/4, B*NH), each block handles 4 consecutive t rows reversed (slowest first); local_sum = Σ_{t'≤t} att·datt (blockReduce); dpreatt = att·(datt − local_sum)·scale in-place overwriting datt, zeros above diagonal (needed because fwd never wrote them). Then dq = dpreatt·K (transB=false... as coded: `matmul_cublaslt(dq, k, dpreatt, false,false)`), dk = dpreattᵀ·q via transB variant, permute_kernel_backward packs dq/dk/dv → dinp (B,T,3C).
Known inefficiencies vs Flash Attention: materializes T² probs; multiple scratch reuses documented in comments.

---

Cross-references: usage order/orchestration in doc 07 §6-9; cuDNN flash-attention alternative in `llmc/cudnn_att*.h/.cpp` (not detailed here); multi-GPU reduction wrappers in doc 09.
